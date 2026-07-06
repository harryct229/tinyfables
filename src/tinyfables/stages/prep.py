"""Prep stage: tokenize prompt+fable rows, mark loss spans (fable + EOT only,
prompts are masked out), pack into a contiguous stream, trim to a multiple of
the window, and write uint16/uint8 shards.

Contract note: prompt and fable are encoded SEPARATELY and concatenated — no
separator token, no merged encoding across the boundary. Generation (issue 02)
must encode prompts the same way so train and inference token streams match."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from tinyfables.config import PrepConfig
from tinyfables.constants import EOT
from tinyfables.data import read_rows
from tinyfables.stage import write_manifest


def run(cfg: PrepConfig, out_dir: Path) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(Path(cfg.tokenizer_dir) / "tokenizer.json"))
    if tok.get_vocab_size() > 65535:
        raise ValueError("vocab too large for uint16 shards")
    eot_id = tok.token_to_id(EOT)

    rows = read_rows(cfg.source, cfg.seed)
    toks: list[int] = []
    mask: list[int] = []
    n_prompt, n_fable = 0, 0
    for r in rows:
        p = tok.encode(r["prompt"]).ids
        f = tok.encode(r["fable"]).ids
        toks.extend(p)
        mask.extend([0] * len(p))
        toks.extend(f)
        toks.append(eot_id)
        mask.extend([1] * (len(f) + 1))
        n_prompt += len(p)
        n_fable += len(f) + 1

    n_windows = len(toks) // cfg.window
    n_keep = n_windows * cfg.window
    tokens_path = out_dir / "tokens.bin"
    mask_path = out_dir / "mask.bin"
    tokens_path.write_bytes(np.asarray(toks[:n_keep], dtype=np.uint16).tobytes())
    mask_path.write_bytes(np.asarray(mask[:n_keep], dtype=np.uint8).tobytes())

    summary = {
        "n_rows": len(rows),
        "window": cfg.window,
        "n_windows": n_windows,
        "n_tokens_written": n_keep,
        "n_prompt_tokens_total": n_prompt,
        "n_fable_tokens_total": n_fable,
        "loss_token_fraction": round(sum(mask[:n_keep]) / n_keep, 4) if n_keep else 0.0,
        "vocab_size": tok.get_vocab_size(),
    }
    summary_path = out_dir / "prep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    write_manifest(out_dir, "prep", cfg, [tokens_path, mask_path, summary_path])
