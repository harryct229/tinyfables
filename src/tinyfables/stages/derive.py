"""Preference derivation (issue 06): turn AI labeler ratings into stable
preferences plus a weight-sensitivity summary. Torch-free."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tinyfables.config import DeriveConfig
from tinyfables.feedback import WEIGHTS, aggregate_score, derive_preference, weight_sensitivity
from tinyfables.labeler import load_label_cohort
from tinyfables.stage import write_manifest


def _split_of(pair_id: str, held_out_fraction: float) -> str:
    bucket = int(hashlib.sha256(pair_id.encode()).hexdigest(), 16) % 1000
    return "held_out" if bucket < round(held_out_fraction * 1000) else "train"


def _read_pairs(path: str | Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return {row["pair_id"]: row for row in rows}


def _canonical_main_labels(instances: list[dict]) -> list[dict]:
    canonical: dict[str, dict] = {}
    for rec in instances:
        if rec["phase"] != "main" or rec["order"] != "ab":
            continue
        pair_id = rec["pair_id"]
        if pair_id in canonical:
            raise ValueError(f"duplicate canonical main label for pair {pair_id!r}")
        canonical[pair_id] = rec
    return [canonical[pair_id] for pair_id in sorted(canonical)]


def run(cfg: DeriveConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    instances, _summary = load_label_cohort(cfg.labels)
    if not instances:
        raise ValueError(f"labels cache is empty: {cfg.labels}")
    pairs = _read_pairs(cfg.pairs)

    mains = _canonical_main_labels(instances)

    preferences = []
    pair_ratings = []
    for rec in mains:
        ratings_0 = rec["ratings_0"]
        ratings_1 = rec["ratings_1"]
        pair_ratings.append((ratings_0, ratings_1))
        preferred = derive_preference(ratings_0, ratings_1)
        if preferred is None:
            continue
        pair = pairs[rec["pair_id"]]
        chosen_idx, rejected_idx = ((0, 1) if preferred == 0 else (1, 0))
        chosen_ratings = (ratings_0, ratings_1)[chosen_idx]
        rejected_ratings = (ratings_0, ratings_1)[rejected_idx]
        preferences.append(
            {
                "pair_id": rec["pair_id"],
                "prompt": pair["prompt"],
                "chosen": pair["fables"][chosen_idx],
                "rejected": pair["fables"][rejected_idx],
                "ratings_chosen": chosen_ratings,
                "ratings_rejected": rejected_ratings,
                "aggregate_chosen": round(aggregate_score(chosen_ratings), 4),
                "aggregate_rejected": round(aggregate_score(rejected_ratings), 4),
                "split": _split_of(rec["pair_id"], cfg.held_out_fraction),
            }
        )

    prefs_path = out_dir / "preferences.jsonl"
    prefs_path.write_text("".join(json.dumps(row) + "\n" for row in preferences))

    sensitivity = weight_sensitivity(pair_ratings, WEIGHTS, cfg.weight_delta)
    sens_path = out_dir / "sensitivity.json"
    sens_path.write_text(json.dumps(sensitivity, indent=2, sort_keys=True) + "\n")

    n_train = sum(1 for row in preferences if row["split"] == "train")
    summary = {
        "n_pairs_labeled": len(mains),
        "n_preferences": len(preferences),
        "n_ties_skipped": len(mains) - len(preferences),
        "n_train": n_train,
        "n_held_out": len(preferences) - n_train,
        "weights": WEIGHTS,
    }
    summary_path = out_dir / "derive_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    write_manifest(
        out_dir,
        "derive",
        cfg,
        [prefs_path, sens_path, summary_path],
        inputs={"labels.jsonl": Path(cfg.labels), "pairs.jsonl": Path(cfg.pairs)},
    )
