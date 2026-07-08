"""Pair generation (issue 06): sample Preference Pairs from the Base Model."""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict
from pathlib import Path

import torch
from tokenizers import Tokenizer

from tinyfables.config import PairgenConfig
from tinyfables.data import iter_rows
from tinyfables.generate import generate_fable
from tinyfables.model import GPT
from tinyfables.paraphrases import parse_canonical_prompt
from tinyfables.prompts import FableSpec, render_canonical_prompt
from tinyfables.stage import _versions, sha256_file, write_manifest

_ELEMENTS = ("character", "setting", "challenge", "outcome", "moral")


def _resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _spec_from_prompt(prompt: str) -> FableSpec | None:
    parsed = parse_canonical_prompt(prompt)
    if parsed is None:
        return None
    return FableSpec(
        **{field: getattr(parsed, field) for field in _ELEMENTS},
        age_range=parsed.age_range,
        word_count=parsed.word_count,
    )


def run(cfg: PairgenConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = itertools.islice(iter_rows(cfg.source, cfg.seed), cfg.n_pairs * 4)
    specs: list[FableSpec] = []
    for row in rows:
        spec = _spec_from_prompt(row["prompt"])
        if spec is not None:
            specs.append(spec)
        if len(specs) >= cfg.n_pairs:
            break
    if len(specs) < cfg.n_pairs:
        raise ValueError(
            f"pairgen requested {cfg.n_pairs} pairs but only found {len(specs)} canonical prompts "
            f"in {cfg.source}"
        )

    device = _resolve_device(cfg.device)
    tok = Tokenizer.from_file(str(Path(cfg.tokenizer_dir) / "tokenizer.json"))
    model = GPT.from_pretrained(cfg.checkpoint).to(device).eval()

    def _gen(prompt_text: str, seed: int) -> str:
        return generate_fable(
            model,
            tok,
            prompt_text,
            max_new_tokens=cfg.max_new_tokens,
            min_new_tokens=cfg.min_new_tokens,
            do_sample=True,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            seed=seed,
            device=device,
        )

    pairs = []
    for i, spec in enumerate(specs):
        prompt = render_canonical_prompt(spec)
        s0, s1 = cfg.seed + 2 * i, cfg.seed + 2 * i + 1
        pairs.append(
            {
                "pair_id": f"pair-{i:06d}",
                "spec": asdict(spec),
                "prompt": prompt,
                "fables": [_gen(prompt, s0), _gen(prompt, s1)],
                "seeds": [s0, s1],
            }
        )

    pairs_path = out_dir / "pairs.jsonl"
    pairs_path.write_text("".join(json.dumps(pair) + "\n" for pair in pairs))

    summary = {
        "checkpoint_sha": sha256_file(Path(cfg.checkpoint) / "model.safetensors"),
        "tokenizer_sha": sha256_file(Path(cfg.tokenizer_dir) / "tokenizer.json"),
        "n_pairs": len(pairs),
        "seed": cfg.seed,
        "temperature": cfg.temperature,
        "versions": _versions(),
    }
    summary_path = out_dir / "pairgen_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    write_manifest(
        out_dir,
        "pairgen",
        cfg,
        [pairs_path, summary_path],
        inputs={
            "model.safetensors": Path(cfg.checkpoint) / "model.safetensors",
            "tokenizer.json": Path(cfg.tokenizer_dir) / "tokenizer.json",
        },
    )
