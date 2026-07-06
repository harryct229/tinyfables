"""Tokenizer stage: train the byte-level BPE on the fable corpus and emit the
compression report that justifies the vocab choice (ADR-0002)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

from tinyfables.config import TokenizerConfig
from tinyfables.constants import SPECIALS
from tinyfables.data import read_rows
from tinyfables.stage import write_manifest


def run(cfg: TokenizerConfig, out_dir: Path) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # determinism over speed
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(cfg.source, cfg.seed)
    texts = [r["prompt"] + "\n" + r["fable"] for r in rows]

    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator(
        texts, vocab_size=cfg.vocab_size, min_frequency=2, special_tokens=SPECIALS
    )
    tok_path = out_dir / "tokenizer.json"
    tok.save(str(tok_path))

    report_path = out_dir / "compression_report.json"
    report = _compression_report(tok, [r["fable"] for r in rows], cfg)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    write_manifest(out_dir, "tokenizer", cfg, [tok_path, report_path])


def _compression_report(tok, fables: list[str], cfg: TokenizerConfig) -> dict:
    ours = [len(tok.encode(t).ids) for t in fables]
    report = {
        "n_fables": len(fables),
        "vocab_size": tok.get_vocab_size(),
        "ours_avg_tokens_per_fable": round(sum(ours) / len(ours), 2),
        "gpt2_avg_tokens_per_fable": None,
    }
    if cfg.compare_gpt2:
        from transformers import GPT2TokenizerFast  # optional dep (network on first use)

        gpt2 = GPT2TokenizerFast.from_pretrained("gpt2")
        counts = [len(gpt2(t)["input_ids"]) for t in fables]
        report["gpt2_avg_tokens_per_fable"] = round(sum(counts) / len(counts), 2)
    return report
