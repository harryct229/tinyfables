"""Pretrain stage: train the hand-written GPT on the packed shards with
completion-only next-token loss (labels -100 on prompt spans), then checkpoint
model + optimizer + step so a dead session costs at most a few steps.

Determinism: batches are a pure function of the step (seed + step), dropout is 0,
and the checkpoint round-trips model (safetensors, bit-exact) + optimizer state +
step. So train-N equals train-k then resume then train-(N-k) exactly."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from tinyfables.config import PretrainConfig
from tinyfables.model import GPT, GPTConfig
from tinyfables.stage import write_manifest


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

    if cfg.resume_from:
        model = GPT.from_pretrained(cfg.resume_from).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        state = torch.load(Path(cfg.resume_from) / "optimizer.pt", map_location=device)
        optimizer.load_state_dict(state["optimizer"])
        start_step = state["step"]
    else:
        torch.manual_seed(cfg.seed)
        model = build_model(cfg, vocab_size).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        start_step = 0

    model.train()
    last_loss = float("nan")
    for step in range(start_step, cfg.steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, cfg)
        input_ids, labels = get_batch(tokens, mask, n_windows, window, cfg, step, device)
        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        last_loss = out.loss.item()

    model.save_pretrained(out_dir)
    torch.save({"optimizer": optimizer.state_dict(), "step": cfg.steps}, out_dir / "optimizer.pt")

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
    }
    (out_dir / "pretrain_summary.json").write_text(
        json.dumps(summary_out, indent=2, sort_keys=True) + "\n"
    )

    write_manifest(
        out_dir,
        "pretrain",
        cfg,
        [
            out_dir / "config.json",
            out_dir / "model.safetensors",
            out_dir / "optimizer.pt",
            out_dir / "pretrain_summary.json",
        ],
        inputs={
            "tokens.bin": prep_dir / "tokens.bin",
            "mask.bin": prep_dir / "mask.bin",
            "tokenizer.json": Path(cfg.tokenizer_dir) / "tokenizer.json",
        },
    )
