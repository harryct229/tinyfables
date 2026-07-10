"""Wrap the project tokenizer.json as a transformers PreTrainedTokenizerFast.

TRL trainers (issue 08) require a PreTrainedTokenizerBase `processing_class`;
the raw `tokenizers.Tokenizer` is not one. The wrapper preserves ids exactly
(no post-processor is added) and exposes the reserved specials. transformers
is imported lazily to keep the light import path light."""

from __future__ import annotations

from pathlib import Path

from tinyfables.constants import EOT, PAD


def load_hf_tokenizer(tokenizer_dir: str | Path, padding_side: str = "right"):
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer.from_file(str(Path(tokenizer_dir) / "tokenizer.json"))
    hf = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token=PAD,
        eos_token=EOT,
        padding_side=padding_side,
    )
    return hf
