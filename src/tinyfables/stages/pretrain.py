"""Pretrain stage: train the hand-written GPT on the packed shards with
completion-only next-token loss (labels -100 on prompt spans), then checkpoint
model + optimizer + step so a dead session costs at most a few steps.

Determinism: batches are a pure function of the step (seed + step), dropout is 0,
and the checkpoint round-trips model (safetensors, bit-exact) + optimizer state +
step. So train-N equals train-k then resume then train-(N-k) exactly."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from tinyfables.checkpoint import find_latest_checkpoint, load_checkpoint, prune_checkpoints, save_checkpoint
from tinyfables.config import PretrainConfig
from tinyfables.metrics import MetricsLogger, plot_loss_curve
from tinyfables.model import GPT, GPTConfig
from tinyfables.stage import write_manifest


def _mirror_checkpoint(ckpt_dir, repo_id) -> None:
    """Best-effort push of the latest checkpoint to the Hub. A network failure
    must never kill a multi-hour run, so all errors are swallowed with a note."""
    try:
        from tinyfables import hub

        latest = find_latest_checkpoint(ckpt_dir)
        if latest is not None:
            hub.upload_checkpoint_dir(latest, repo_id, private=True)
    except Exception as e:  # noqa: BLE001 — resilience over correctness here
        print(f"[tinyfables] checkpoint hub mirror failed (continuing): {e}")


def build_model(cfg: PretrainConfig, vocab_size: int) -> GPT:
    return GPT(
        GPTConfig(
            vocab_size=vocab_size,
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            d_model=cfg.d_model,
            n_ctx=cfg.n_ctx,
        )
    )


def lr_at(step: int, cfg: PretrainConfig) -> float:
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    return cfg.lr


def _resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_batch(tokens, mask, n_windows, window, cfg, step, device):
    g = torch.Generator().manual_seed(cfg.seed + step)
    idx = torch.randint(0, n_windows, (cfg.batch_size,), generator=g).numpy()
    tok_rows = tokens.reshape(n_windows, window)[idx].astype(np.int64)
    mask_rows = mask.reshape(n_windows, window)[idx].astype(np.int64)
    input_ids = torch.from_numpy(tok_rows).to(device)
    labels = input_ids.clone()
    labels[torch.from_numpy(mask_rows).to(device) == 0] = -100
    return input_ids, labels


def run(cfg: PretrainConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(cfg.device)

    tok = Tokenizer.from_file(str(Path(cfg.tokenizer_dir) / "tokenizer.json"))
    vocab_size = tok.get_vocab_size()

    prep_dir = Path(cfg.prep_dir)
    summary = json.loads((prep_dir / "prep_summary.json").read_text())
    window = summary["window"]
    if window > cfg.n_ctx:
        raise ValueError(f"prep window {window} exceeds model n_ctx {cfg.n_ctx}")
    tokens = np.memmap(prep_dir / "tokens.bin", dtype="<u2", mode="r")
    mask = np.memmap(prep_dir / "mask.bin", dtype=np.uint8, mode="r")
    n_windows = tokens.shape[0] // window
    if n_windows < cfg.batch_size:
        raise ValueError(f"only {n_windows} windows for batch_size {cfg.batch_size}")

    use_amp = cfg.amp and device == "cuda"
    autocast_device = "cuda" if device == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(device, enabled=use_amp)

    resume_src = None
    if cfg.resume_from:
        resume_src = Path(cfg.resume_from)
    elif cfg.ckpt_dir:
        resume_src = find_latest_checkpoint(cfg.ckpt_dir)

    if resume_src is not None:
        model, state = load_checkpoint(resume_src, GPT, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        optimizer.load_state_dict(state["optimizer"])
        if state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
        start_step = state["step"]
    else:
        torch.manual_seed(cfg.seed)
        model = build_model(cfg, vocab_size).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        start_step = 0

    csv_path = Path(cfg.ckpt_dir or out_dir) / "loss_log.csv"
    logger = MetricsLogger(
        csv_path,
        project=cfg.trackio_project,
        run_name=cfg.run_name,
        space_id=cfg.trackio_space_id,
        config={"n_layer": cfg.n_layer, "d_model": cfg.d_model, "n_head": cfg.n_head,
                "n_ctx": cfg.n_ctx, "batch_size": cfg.batch_size, "steps": cfg.steps,
                "lr": cfg.lr, "amp": use_amp},
    )
    tokens_per_step = cfg.batch_size * window
    t_log = time.perf_counter()
    steps_since_log = 0

    model.train()
    last_loss = float("nan")
    for step in range(start_step, cfg.steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, cfg)
        input_ids, labels = get_batch(tokens, mask, n_windows, window, cfg, step, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=autocast_device, dtype=torch.float16, enabled=use_amp):
            out = model(input_ids=input_ids, labels=labels)
        scaler.scale(out.loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        last_loss = out.loss.item()

        completed = step + 1
        if cfg.ckpt_dir and cfg.ckpt_every and completed % cfg.ckpt_every == 0 and completed < cfg.steps:
            save_checkpoint(model, optimizer, scaler, completed, cfg.ckpt_dir)
            prune_checkpoints(cfg.ckpt_dir, cfg.keep_last_k)
            if cfg.ckpt_hub_repo:
                _mirror_checkpoint(cfg.ckpt_dir, cfg.ckpt_hub_repo)

        steps_since_log += 1
        if cfg.log_every and completed % cfg.log_every == 0:
            now = time.perf_counter()
            tps = (steps_since_log * tokens_per_step) / (now - t_log) if now > t_log else 0.0
            logger.log(step=completed, loss=last_loss, lr=lr_at(step, cfg), tokens_per_sec=tps)
            t_log = now
            steps_since_log = 0

    model.save_pretrained(out_dir)
    torch.save(
        {"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict() if use_amp else None, "step": cfg.steps},
        out_dir / "optimizer.pt",
    )

    logger.log(step=cfg.steps, loss=last_loss, lr=lr_at(max(cfg.steps - 1, 0), cfg), tokens_per_sec=0.0)
    logger.finish()
    out_csv = out_dir / "loss_log.csv"
    if csv_path != out_csv:
        shutil.copyfile(csv_path, out_csv)
    png = plot_loss_curve(out_csv, out_dir / "loss_curve.png")

    n_params = sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())
    summary_out = {
        "steps": cfg.steps,
        "start_step": start_step,
        "final_loss": round(last_loss, 4),
        "n_params": n_params,
        "vocab_size": vocab_size,
        "window": window,
        "n_windows": n_windows,
        "device": device,
        "amp": use_amp,
    }
    (out_dir / "pretrain_summary.json").write_text(
        json.dumps(summary_out, indent=2, sort_keys=True) + "\n"
    )

    artifacts = [
        out_dir / "config.json",
        out_dir / "generation_config.json",
        out_dir / "model.safetensors",
        out_dir / "optimizer.pt",
        out_dir / "pretrain_summary.json",
        out_dir / "loss_log.csv",
    ]
    if png is not None:
        artifacts.append(out_dir / "loss_curve.png")
    write_manifest(
        out_dir,
        "pretrain",
        cfg,
        artifacts,
        inputs={
            "tokens.bin": prep_dir / "tokens.bin",
            "mask.bin": prep_dir / "mask.bin",
            "tokenizer.json": Path(cfg.tokenizer_dir) / "tokenizer.json",
        },
    )
