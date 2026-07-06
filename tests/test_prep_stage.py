import json
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer

from tinyfables.config import PrepConfig, SourceSpec, TokenizerConfig
from tinyfables.data import read_rows
from tinyfables.stages import prep as prep_stage
from tinyfables.stages import tokenizer as tokenizer_stage

FIXTURE = str(Path(__file__).parent / "fixtures" / "tiny_corpus.jsonl")
SRC = SourceSpec(jsonl_path=FIXTURE)


@pytest.fixture(scope="module")
def tok_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("tok")
    tokenizer_stage.run(TokenizerConfig(source=SRC, vocab_size=512, seed=0), out)
    return out


def prep_cfg(tok_dir):
    return PrepConfig(source=SRC, tokenizer_dir=str(tok_dir), window=256, seed=0)


def run_prep(tmp_path, tok_dir, name="prep"):
    out = tmp_path / name
    prep_stage.run(prep_cfg(tok_dir), out)
    return out


def test_shards_are_window_aligned_uint16(tmp_path, tok_dir):
    out = run_prep(tmp_path, tok_dir)
    summary = json.loads((out / "prep_summary.json").read_text())
    tokens = np.frombuffer((out / "tokens.bin").read_bytes(), dtype=np.uint16)
    mask = np.frombuffer((out / "mask.bin").read_bytes(), dtype=np.uint8)
    assert len(tokens) == len(mask) == summary["n_tokens_written"]
    assert summary["n_tokens_written"] == summary["n_windows"] * summary["window"]
    assert summary["n_windows"] > 0
    assert set(np.unique(mask)) <= {0, 1}


def test_mask_marks_prompt_zero_then_fable_one(tmp_path, tok_dir):
    out = run_prep(tmp_path, tok_dir)
    tok = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
    first = read_rows(SRC, seed=0)[0]  # same seed => same order as the stage
    n_p = len(tok.encode(first["prompt"]).ids)
    n_f = len(tok.encode(first["fable"]).ids)
    mask = np.frombuffer((out / "mask.bin").read_bytes(), dtype=np.uint8)
    tokens = np.frombuffer((out / "tokens.bin").read_bytes(), dtype=np.uint16)
    assert not mask[:n_p].any()                       # prompt tokens carry no loss
    assert mask[n_p : n_p + n_f + 1].all()            # fable + EOT carry loss
    assert tokens[n_p + n_f] == tok.token_to_id("<|endoftext|>")


def test_summary_accounting(tmp_path, tok_dir):
    out = run_prep(tmp_path, tok_dir)
    s = json.loads((out / "prep_summary.json").read_text())
    assert s["n_rows"] == 24
    assert 0.3 < s["loss_token_fraction"] < 0.9
    assert s["vocab_size"] <= 512


def test_prep_is_hash_deterministic(tmp_path, tok_dir):
    a = json.loads((run_prep(tmp_path, tok_dir, "a") / "manifest.json").read_text())
    b = json.loads((run_prep(tmp_path, tok_dir, "b") / "manifest.json").read_text())
    assert a["artifacts"] == b["artifacts"]
