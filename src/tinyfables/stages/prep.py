"""Prep stage: tokenize prompt+fable rows, mark loss spans (fable + EOT only,
prompts are masked out), pack into a contiguous stream, trim to a multiple of
the window, and write uint16/uint8 shards.

Contract note: prompt and fable are encoded SEPARATELY and concatenated — no
separator token, no merged encoding across the boundary. Generation (issue 02)
must encode prompts the same way so train and inference token streams match.

Rows are consumed via `iter_rows` and the token/mask buffers are flushed well
before they reach corpus size, so peak RAM stays O(buffer) end to end (the
jsonl source may still buffer its rows internally to support seeded shuffling)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from tinyfables.config import PrepConfig
from tinyfables.constants import EOT
from tinyfables.data import iter_rows
from tinyfables.stage import write_manifest


# Flush the row-buffer to disk once it holds roughly this many tokens, so peak
# RAM is O(buffer) rather than O(corpus) at the real ~250M-token scale.
_FLUSH_AT_TOKENS = 1_000_000


def run(cfg: PrepConfig, out_dir: Path) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = Path(cfg.tokenizer_dir) / "tokenizer.json"
    tok = Tokenizer.from_file(str(tokenizer_path))
    if tok.get_vocab_size() > 65535:
        raise ValueError("vocab too large for uint16 shards")
    eot_id = tok.token_to_id(EOT)
    if eot_id is None:
        raise ValueError(f"tokenizer at {cfg.tokenizer_dir} lacks the <|endoftext|> special")

    tokens_path = out_dir / "tokens.bin"
    mask_path = out_dir / "mask.bin"

    toks_buf: list[int] = []
    mask_buf: list[int] = []
    n_written = 0
    n_rows = 0
    n_prompt, n_fable = 0, 0

    with open(tokens_path, "wb") as tf, open(mask_path, "wb") as mf:

        def flush() -> None:
            nonlocal n_written
            if not toks_buf:
                return
            tf.write(np.asarray(toks_buf, dtype="<u2").tobytes())
            mf.write(np.asarray(mask_buf, dtype=np.uint8).tobytes())
            n_written += len(toks_buf)
            toks_buf.clear()
            mask_buf.clear()

        for r in iter_rows(cfg.source, cfg.seed):
            n_rows += 1
            p = tok.encode(r["prompt"]).ids
            f = tok.encode(r["fable"]).ids
            toks_buf.extend(p)
            mask_buf.extend([0] * len(p))
            toks_buf.extend(f)
            toks_buf.append(eot_id)
            mask_buf.extend([1] * (len(f) + 1))
            n_prompt += len(p)
            n_fable += len(f) + 1
            if len(toks_buf) >= _FLUSH_AT_TOKENS:
                flush()
        flush()

    n_windows = n_written // cfg.window
    n_keep = n_windows * cfg.window
    n_tokens_dropped = n_written - n_keep
    with open(tokens_path, "r+b") as tf:
        tf.truncate(n_keep * 2)
    with open(mask_path, "r+b") as mf:
        mf.truncate(n_keep)

    if n_keep:
        kept_mask = np.memmap(mask_path, dtype=np.uint8, mode="r", shape=(n_keep,))
        loss_token_total = int(kept_mask.sum())
        del kept_mask
    else:
        loss_token_total = 0

    summary = {
        "n_rows": n_rows,
        "window": cfg.window,
        "n_windows": n_windows,
        "n_tokens_written": n_keep,
        "n_tokens_dropped": n_tokens_dropped,
        "n_prompt_tokens_total": n_prompt,
        "n_fable_tokens_total": n_fable,
        "loss_token_fraction": round(loss_token_total / n_keep, 4) if n_keep else 0.0,
        "vocab_size": tok.get_vocab_size(),
    }
    summary_path = out_dir / "prep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    write_manifest(
        out_dir,
        "prep",
        cfg,
        [tokens_path, mask_path, summary_path],
        inputs={"tokenizer.json": tokenizer_path},
    )
