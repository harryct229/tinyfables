"""AI Labeler stage with deterministic scheduling and append-only resume."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from tinyfables.config import LabelConfig
from tinyfables.labeler import (
    append_cache,
    build_batch_prompt,
    cache_key,
    claude_runner,
    load_cache,
    load_prompt,
    parse_labeler_response,
)
from tinyfables.stage import write_manifest


def _read_pairs(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def plan_work(pairs: list[dict], cfg: LabelConfig) -> list[dict]:
    def item(pair: dict, phase: str, order: str) -> dict:
        return {
            "pair_id": pair["pair_id"],
            "phase": phase,
            "order": order,
            "fables": pair["fables"],
        }

    rng = random.Random(cfg.seed)
    calib = pairs[: cfg.calibration_size]
    n_swap = int(round(len(pairs) * cfg.swap_fraction))
    pair_ids = [pair["pair_id"] for pair in pairs]
    swap_ids = set(rng.sample(pair_ids, n_swap)) if n_swap else set()
    swap = [pair for pair in pairs if pair["pair_id"] in swap_ids]

    main = [item(pair, "main", "ab") for pair in pairs]
    third = max(1, len(main) // 3) if main else 1
    work: list[dict] = []
    work += [item(pair, "calib-0", "ab") for pair in calib]
    work += main[:third]
    work += [item(pair, "calib-1", "ab") for pair in calib]
    work += main[third : 2 * third]
    work += [item(pair, "swap", "ba") for pair in swap]
    work += main[2 * third :]
    work += [item(pair, "calib-2", "ab") for pair in calib]
    return work


def _store_ratings(item: dict, label) -> tuple[dict, dict]:
    if item["order"] == "ab":
        return label.ratings_a, label.ratings_b
    return label.ratings_b, label.ratings_a


def _batched(items: list[dict], n: int):
    for i in range(0, len(items), n):
        yield items[i : i + n]


def run(cfg: LabelConfig, out_dir: Path, runner=None) -> None:
    runner = runner or claude_runner
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "labels.jsonl"

    rubric_text = Path(cfg.rubric).read_text()
    prompt_version, template = load_prompt(cfg.labeler_prompt)
    rubric_sha = hashlib.sha256(rubric_text.encode()).hexdigest()
    prompt_sha = hashlib.sha256(template.encode()).hexdigest()

    pairs = _read_pairs(cfg.pairs)
    work = plan_work(pairs, cfg)
    cache = load_cache(cache_path)
    pending = [
        item
        for item in work
        if cache_key(item["pair_id"], item["phase"], item["order"], cfg.model, prompt_version)
        not in cache
    ]

    for rec in cache.values():
        if rec["prompt_version"] == prompt_version and (
            rec.get("rubric_sha") != rubric_sha or rec.get("prompt_sha") != prompt_sha
        ):
            raise ValueError(
                "cache has labels at this prompt_version with a different rubric/prompt hash; "
                "bump `version` in labeler_prompt.yaml before re-labeling (RUBRIC/prompt changed)."
            )

    n_batches = 0
    for batch in _batched(pending, cfg.batch_size):
        if cfg.max_batches is not None and n_batches >= cfg.max_batches:
            break
        prompt_batch = [
            {
                "pair_id": item["pair_id"],
                "fable_a": item["fables"][0] if item["order"] == "ab" else item["fables"][1],
                "fable_b": item["fables"][1] if item["order"] == "ab" else item["fables"][0],
            }
            for item in batch
        ]
        prompt = build_batch_prompt(rubric_text, template, prompt_batch)
        response = runner(prompt, cfg.model)
        labels = {
            label.pair_id: label
            for label in parse_labeler_response(response, [item["pair_id"] for item in batch])
        }
        for item in batch:
            label = labels[item["pair_id"]]
            ratings_0, ratings_1 = _store_ratings(item, label)
            append_cache(
                cache_path,
                {
                    "pair_id": item["pair_id"],
                    "phase": item["phase"],
                    "order": item["order"],
                    "model_version": cfg.model,
                    "prompt_version": prompt_version,
                    "rubric_sha": rubric_sha,
                    "prompt_sha": prompt_sha,
                    "ratings_0": ratings_0,
                    "ratings_1": ratings_1,
                    "justification": label.justification,
                },
            )
        n_batches += 1

    final = load_cache(cache_path)
    summary = {
        "n_pairs": len(pairs),
        "n_label_instances": len(final),
        "n_batches_this_run": n_batches,
        "model_version": cfg.model,
        "prompt_version": prompt_version,
        "rubric_sha": rubric_sha,
        "prompt_sha": prompt_sha,
        "complete": len(final) >= len(work),
    }
    summary_path = out_dir / "label_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_manifest(
        out_dir,
        "label",
        cfg,
        [cache_path, summary_path],
        inputs={"pairs.jsonl": Path(cfg.pairs)},
    )
