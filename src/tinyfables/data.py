"""Source reading: local JSONL (fixtures, toy runs) or HF streaming (real runs).
Rows are reduced to the two fields the pipeline uses: prompt and fable."""

from __future__ import annotations

import json
import random
from typing import Iterator

from tinyfables.config import SourceSpec


def iter_rows(src: SourceSpec, seed: int) -> Iterator[dict]:
    if src.jsonl_path is not None:
        with open(src.jsonl_path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        rng = random.Random(seed)
        rng.shuffle(rows)
        if src.max_rows is not None:
            rows = rows[: src.max_rows]
        yield from ({"prompt": r["prompt"], "fable": r["fable"]} for r in rows)
        return

    from datasets import load_dataset  # imported lazily: network dependency

    ds = load_dataset(src.hf_dataset, split=src.hf_split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    n = 0
    for r in ds:
        yield {"prompt": r["prompt"], "fable": r["fable"]}
        n += 1
        if src.max_rows is not None and n >= src.max_rows:
            break


def read_rows(src: SourceSpec, seed: int) -> list[dict]:
    return list(iter_rows(src, seed))
