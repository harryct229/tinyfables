"""RM-accuracy-by-margin diagnostic (issue 08, STEP 1).

Stratifies held-out preferences by aggregate-score margin and reports RM
accuracy per bucket. High accuracy on wide margins + ~chance on near-ties
means the label signal is real but tie-diluted (margin-filtering/relabeling
helps); flat across margins means the labels carry little usable signal (PPO
on this RM would optimize noise). Feeds the ADR-0005 fork decision."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from tinyfables.config import MarginsConfig
from tinyfables.reward_data import collate_preference_batch, load_preferences
from tinyfables.reward_model import load_reward_model
from tinyfables.stage import write_manifest


def _resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _margin(rec: dict) -> float:
    return rec["aggregate_chosen"] - rec["aggregate_rejected"]


def _bucket_index(margin: float, edges: list[float]) -> int:
    idx = 0
    for i, edge in enumerate(edges):
        if margin >= edge:
            idx = i
    return idx


@torch.no_grad()
def run(cfg: MarginsConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(cfg.device)
    tokenizer_path = Path(cfg.tokenizer_dir) / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    model = load_reward_model(cfg.reward_model_dir, device=device).eval()

    held_out = load_preferences(cfg.preferences, split="held_out")
    if not held_out:
        raise ValueError("margins diagnostic requires at least one held_out preference")
    raw = {
        rec["pair_id"]: rec
        for rec in (json.loads(line) for line in Path(cfg.preferences).read_text().splitlines() if line.strip())
        if rec["split"] == "held_out"
    }

    correct_by_pair: dict[str, bool] = {}
    for start in range(0, len(held_out), cfg.batch_size):
        rows = held_out[start : start + cfg.batch_size]
        batch = collate_preference_batch(rows, tokenizer, cfg.n_ctx, device)
        chosen = model(batch["chosen_input_ids"], batch["chosen_attention_mask"])
        rejected = model(batch["rejected_input_ids"], batch["rejected_attention_mask"])
        for row, c, r in zip(rows, chosen.tolist(), rejected.tolist()):
            correct_by_pair[row.pair_id] = c > r

    edges = list(cfg.bucket_edges)
    buckets = [{"lo": edge, "hi": (edges[i + 1] if i + 1 < len(edges) else None), "n": 0, "n_correct": 0} for i, edge in enumerate(edges)]
    for pair_id, correct in sorted(correct_by_pair.items()):
        b = buckets[_bucket_index(_margin(raw[pair_id]), edges)]
        b["n"] += 1
        b["n_correct"] += int(correct)
    for b in buckets:
        b["accuracy"] = round(b["n_correct"] / b["n"], 6) if b["n"] else None

    overall = sum(correct_by_pair.values()) / len(correct_by_pair)
    report = {
        "overall_accuracy": round(overall, 6),
        "n_held_out": len(correct_by_pair),
        "bucket_edges": edges,
        "buckets": buckets,
    }
    (out_dir / "margins.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    lines = [
        "# RM accuracy by aggregate-score margin",
        "",
        f"Held-out preferences: {report['n_held_out']}; overall accuracy {report['overall_accuracy']:.3f}",
        "",
        "| margin | n | accuracy |",
        "|---|---|---|",
    ]
    for b in buckets:
        hi = f"{b['hi']:.1f}" if b["hi"] is not None else "inf"
        acc = f"{b['accuracy']:.3f}" if b["accuracy"] is not None else "-"
        lines.append(f"| [{b['lo']:.1f}, {hi}) | {b['n']} | {acc} |")
    (out_dir / "margins_report.md").write_text("\n".join(lines) + "\n")

    write_manifest(
        out_dir,
        "margins",
        cfg,
        [out_dir / "margins.json", out_dir / "margins_report.md"],
        inputs={
            "preferences.jsonl": Path(cfg.preferences),
            "tokenizer.json": tokenizer_path,
            "reward_head.pt": Path(cfg.reward_model_dir) / "reward_head.pt",
            "backbone_model.safetensors": Path(cfg.reward_model_dir) / "backbone" / "model.safetensors",
        },
    )
