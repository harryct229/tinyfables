"""Reward-model training stage.

Trains TinyFables Base Model + scalar head with Bradley-Terry loss, evaluates on
the deterministic held-out preference split, and reruns smaller train subsets for
the committed data-curve ablation.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import torch
from tokenizers import Tokenizer

from tinyfables.config import RewardTrainConfig
from tinyfables.reward_data import (
    PreferenceExample,
    collate_preference_batch,
    load_preferences,
    select_train_subset,
)
from tinyfables.reward_model import (
    RewardModel,
    bradley_terry_loss,
    preference_accuracy,
    save_reward_model,
)
from tinyfables.stage import write_manifest


def _resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _lr_at(step: int, cfg: RewardTrainConfig) -> float:
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    return cfg.lr


def _batch(
    examples: list[PreferenceExample],
    batch_size: int,
    seed: int,
    step: int,
) -> list[PreferenceExample]:
    rng = random.Random(seed + step)
    if len(examples) <= batch_size:
        return list(examples)
    idx = rng.sample(range(len(examples)), batch_size)
    return [examples[i] for i in idx]


@torch.no_grad()
def _evaluate(
    model: RewardModel,
    examples: list[PreferenceExample],
    tokenizer,
    cfg: RewardTrainConfig,
    device: str,
) -> float:
    if not examples:
        return 0.0
    model.eval()
    accs = []
    for start in range(0, len(examples), cfg.batch_size):
        rows = examples[start : start + cfg.batch_size]
        batch = collate_preference_batch(rows, tokenizer, cfg.n_ctx, device)
        chosen = model(batch["chosen_input_ids"], batch["chosen_attention_mask"])
        rejected = model(batch["rejected_input_ids"], batch["rejected_attention_mask"])
        accs.append(preference_accuracy(chosen, rejected) * len(rows))
    return sum(accs) / len(examples)


def _train_model(
    cfg: RewardTrainConfig,
    train: list[PreferenceExample],
    held_out: list[PreferenceExample],
    tokenizer,
    device: str,
    steps: int,
    csv_path: Path | None,
) -> tuple[RewardModel, dict]:
    torch.manual_seed(cfg.seed)
    model = RewardModel.from_base_checkpoint(cfg.base_checkpoint, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    use_amp = cfg.amp and device == "cuda"
    autocast_device = "cuda" if device == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(device, enabled=use_amp)

    writer = None
    fh = None
    if csv_path is not None:
        fh = csv_path.open("w", newline="")
        writer = csv.DictWriter(fh, fieldnames=["step", "loss", "held_out_accuracy"])
        writer.writeheader()

    last_loss = 0.0
    for step in range(steps):
        model.train()
        for group in optimizer.param_groups:
            group["lr"] = _lr_at(step, cfg)
        rows = _batch(train, cfg.batch_size, cfg.seed, step)
        batch = collate_preference_batch(rows, tokenizer, cfg.n_ctx, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=autocast_device, dtype=torch.float16, enabled=use_amp):
            chosen = model(batch["chosen_input_ids"], batch["chosen_attention_mask"])
            rejected = model(batch["rejected_input_ids"], batch["rejected_attention_mask"])
            loss = bradley_terry_loss(chosen, rejected)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        last_loss = float(loss.item())

        completed = step + 1
        if writer is not None and cfg.log_every and completed % cfg.log_every == 0:
            writer.writerow(
                {
                    "step": completed,
                    "loss": f"{last_loss:.6f}",
                    "held_out_accuracy": f"{_evaluate(model, held_out, tokenizer, cfg, device):.6f}",
                }
            )

    held_out_accuracy = _evaluate(model, held_out, tokenizer, cfg, device)
    if writer is not None and (not cfg.log_every or steps % cfg.log_every != 0):
        writer.writerow(
            {
                "step": steps,
                "loss": f"{last_loss:.6f}",
                "held_out_accuracy": f"{held_out_accuracy:.6f}",
            }
        )
    if fh is not None:
        fh.close()

    return model, {"final_loss": last_loss, "held_out_accuracy": held_out_accuracy}


def run(cfg: RewardTrainConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(cfg.device)
    tokenizer_path = Path(cfg.tokenizer_dir) / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    train = load_preferences(cfg.preferences, split="train")
    held_out = load_preferences(cfg.preferences, split="held_out")
    if not train:
        raise ValueError("reward training requires at least one train preference")
    if not held_out:
        raise ValueError("reward training requires at least one held_out preference")

    model, metrics = _train_model(
        cfg, train, held_out, tokenizer, device, cfg.steps, out_dir / "loss_log.csv"
    )
    save_reward_model(model, out_dir)

    points = []
    for requested in cfg.curve_sizes:
        subset = select_train_subset(train, requested, cfg.seed)
        _curve_model, curve_metrics = _train_model(
            cfg, subset, held_out, tokenizer, device, cfg.curve_steps, None
        )
        points.append(
            {
                "requested_train_size": requested,
                "train_size": len(subset),
                "steps": cfg.curve_steps,
                "held_out_accuracy": round(curve_metrics["held_out_accuracy"], 6),
                "final_loss": round(curve_metrics["final_loss"], 6),
            }
        )

    curve_path = out_dir / "data_curve.json"
    curve_path.write_text(json.dumps({"points": points}, indent=2, sort_keys=True) + "\n")

    summary = {
        "n_train": len(train),
        "n_held_out": len(held_out),
        "steps": cfg.steps,
        "device": device,
        "amp": cfg.amp and device == "cuda",
        "final_loss": round(metrics["final_loss"], 6),
        "held_out_accuracy": round(metrics["held_out_accuracy"], 6),
        "accuracy_gate": cfg.accuracy_gate,
        "rm_gate_pass": metrics["held_out_accuracy"] >= cfg.accuracy_gate,
    }
    summary_path = out_dir / "reward_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    write_manifest(
        out_dir,
        "reward",
        cfg,
        [
            out_dir / "backbone" / "config.json",
            out_dir / "backbone" / "generation_config.json",
            out_dir / "backbone" / "model.safetensors",
            out_dir / "reward_head.pt",
            out_dir / "reward_model_config.json",
            out_dir / "reward_summary.json",
            out_dir / "data_curve.json",
            out_dir / "loss_log.csv",
        ],
        inputs={
            "preferences.jsonl": Path(cfg.preferences),
            "tokenizer.json": tokenizer_path,
            "base_config.json": Path(cfg.base_checkpoint) / "config.json",
            "base_model.safetensors": Path(cfg.base_checkpoint) / "model.safetensors",
        },
    )
