"""Preference derivation (issue 06): turn AI labeler ratings into stable
preferences plus a weight-sensitivity summary. Torch-free.

ADR-0005 adds an optional majority-vote mode (`extra_labels`): when set, the
stage loads the primary cache plus one or more additional independently
labeled caches for the same pairs and derives preferences from the *voted*
outcome across caches instead of a single cache. `extra_labels` unset (the
default) keeps the original single-cache behavior byte-identical."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import mean

from tinyfables.config import DeriveConfig
from tinyfables.feedback import (
    AXES,
    WEIGHTS,
    aggregate_score,
    derive_preference,
    voted_preference,
    voted_weight_sensitivity,
    weight_sensitivity,
)
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


def _load_canonical_main(path: str | Path) -> list[dict]:
    instances, _summary = load_label_cohort(path)
    if not instances:
        raise ValueError(f"labels cache is empty: {path}")
    return _canonical_main_labels(instances)


def run(cfg: DeriveConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    instances, _summary = load_label_cohort(cfg.labels)
    if not instances:
        raise ValueError(f"labels cache is empty: {cfg.labels}")
    pairs = _read_pairs(cfg.pairs)

    mains = _canonical_main_labels(instances)

    inputs = {"labels.jsonl": Path(cfg.labels), "pairs.jsonl": Path(cfg.pairs)}

    if cfg.extra_labels is None:
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

        sensitivity = weight_sensitivity(pair_ratings, WEIGHTS, cfg.weight_delta)
        n_ties_skipped = len(mains) - len(preferences)
        summary_extra: dict = {}
    else:
        main_by_id = {rec["pair_id"]: rec for rec in mains}
        extra_by_id_list = []
        for i, path in enumerate(cfg.extra_labels):
            extra_mains = _load_canonical_main(path)
            extra_by_id = {rec["pair_id"]: rec for rec in extra_mains}
            if set(extra_by_id) != set(main_by_id):
                raise ValueError(
                    "label caches disagree on their canonical pair-id set: "
                    f"{cfg.labels} vs {path}"
                )
            extra_by_id_list.append(extra_by_id)
            inputs[f"labels_extra_{i}.jsonl"] = Path(path)

        preferences = []
        cohort_ratings = []
        n_no_majority = 0
        for pair_id in sorted(main_by_id):
            caches_ratings = [
                (main_by_id[pair_id]["ratings_0"], main_by_id[pair_id]["ratings_1"])
            ]
            for extra_by_id in extra_by_id_list:
                rec = extra_by_id[pair_id]
                caches_ratings.append((rec["ratings_0"], rec["ratings_1"]))
            cohort_ratings.append(caches_ratings)

            winner = voted_preference(caches_ratings, WEIGHTS)
            if winner is None:
                n_no_majority += 1
                continue

            agreeing = [
                cr for cr in caches_ratings if derive_preference(cr[0], cr[1], WEIGHTS) == winner
            ]
            chosen_ratings = {axis: round(mean(cr[winner][axis] for cr in agreeing), 4) for axis in AXES}
            rejected_ratings = {
                axis: round(mean(cr[1 - winner][axis] for cr in agreeing), 4) for axis in AXES
            }

            pair = pairs[pair_id]
            chosen_idx, rejected_idx = ((0, 1) if winner == 0 else (1, 0))
            preferences.append(
                {
                    "pair_id": pair_id,
                    "prompt": pair["prompt"],
                    "chosen": pair["fables"][chosen_idx],
                    "rejected": pair["fables"][rejected_idx],
                    "ratings_chosen": chosen_ratings,
                    "ratings_rejected": rejected_ratings,
                    "aggregate_chosen": round(aggregate_score(chosen_ratings), 4),
                    "aggregate_rejected": round(aggregate_score(rejected_ratings), 4),
                    "split": _split_of(pair_id, cfg.held_out_fraction),
                }
            )

        sensitivity = voted_weight_sensitivity(cohort_ratings, WEIGHTS, cfg.weight_delta)
        # Per-cache ties are absorbed into n_no_majority in voting mode; every
        # skipped-in-voting pair (no >=2-cache majority, including the case
        # where the vote itself is the tiebreak) is counted there instead of
        # split across two overlapping counters.
        n_ties_skipped = 0
        summary_extra = {"n_no_majority": n_no_majority}

    prefs_path = out_dir / "preferences.jsonl"
    prefs_path.write_text("".join(json.dumps(row) + "\n" for row in preferences))

    sens_path = out_dir / "sensitivity.json"
    sens_path.write_text(json.dumps(sensitivity, indent=2, sort_keys=True) + "\n")

    n_train = sum(1 for row in preferences if row["split"] == "train")
    summary = {
        "n_pairs_labeled": len(mains),
        "n_preferences": len(preferences),
        "n_ties_skipped": n_ties_skipped,
        "n_train": n_train,
        "n_held_out": len(preferences) - n_train,
        "weights": WEIGHTS,
        **summary_extra,
    }
    summary_path = out_dir / "derive_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    write_manifest(
        out_dir,
        "derive",
        cfg,
        [prefs_path, sens_path, summary_path],
        inputs=inputs,
    )
