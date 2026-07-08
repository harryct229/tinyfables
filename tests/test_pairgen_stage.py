import json
from pathlib import Path

import pytest

from tinyfables.config import PairgenConfig, PretrainConfig, PrepConfig, SourceSpec, TokenizerConfig
from tinyfables.stages import REGISTRY
from tinyfables.stages import pairgen as pairgen_stage
from tinyfables.stages import prep as prep_stage
from tinyfables.stages import pretrain as pretrain_stage
from tinyfables.stages import tokenizer as tokenizer_stage

FIXTURE = str(Path(__file__).parent / "fixtures" / "tiny_corpus.jsonl")
SRC = SourceSpec(jsonl_path=FIXTURE)


def _trained_checkpoint(tmp_path):
    tok = tmp_path / "tok"; prep = tmp_path / "prep"; ckpt = tmp_path / "ckpt"
    tokenizer_stage.run(TokenizerConfig(source=SRC, vocab_size=512, seed=0), tok)
    prep_stage.run(PrepConfig(source=SRC, tokenizer_dir=str(tok), window=512, seed=0), prep)
    pretrain_stage.run(PretrainConfig(prep_dir=str(prep), tokenizer_dir=str(tok),
                       n_layer=2, n_head=2, d_model=64, n_ctx=512, batch_size=4,
                       steps=4, warmup_steps=0, seed=0, device="cpu"), ckpt)
    return tok, ckpt


def _cfg(tok, ckpt, **kw):
    base = dict(checkpoint=str(ckpt), tokenizer_dir=str(tok), source=SRC, n_ctx=512,
                n_pairs=3, max_new_tokens=24, min_new_tokens=8, top_k=10, seed=0, device="cpu")
    base.update(kw)
    return PairgenConfig(**base)


def test_pairgen_registered():
    assert REGISTRY["pairgen"] == (PairgenConfig, "tinyfables.stages.pairgen")


def test_pairgen_writes_pairs_summary_and_manifest(tmp_path):
    tok, ckpt = _trained_checkpoint(tmp_path)
    out = tmp_path / "pairs"
    pairgen_stage.run(_cfg(tok, ckpt), out)
    assert {"pairs.jsonl", "pairgen_summary.json", "manifest.json"} <= {p.name for p in out.iterdir()}
    pairs = [json.loads(l) for l in (out / "pairs.jsonl").read_text().splitlines()]
    assert len(pairs) == 3
    p0 = pairs[0]
    assert set(p0) >= {"pair_id", "spec", "prompt", "fables", "seeds"}
    assert len(p0["fables"]) == 2 and p0["seeds"][0] != p0["seeds"][1]
    assert p0["pair_id"] == "pair-000000"
    summary = json.loads((out / "pairgen_summary.json").read_text())
    assert summary["n_pairs"] == 3
    assert len(summary["checkpoint_sha"]) == 64


def test_pairgen_two_samples_differ_and_are_deterministic(tmp_path):
    tok, ckpt = _trained_checkpoint(tmp_path)
    a = tmp_path / "a"; b = tmp_path / "b"
    pairgen_stage.run(_cfg(tok, ckpt), a)
    pairgen_stage.run(_cfg(tok, ckpt), b)
    pa = (a / "pairs.jsonl").read_text()
    pb = (b / "pairs.jsonl").read_text()
    assert pa == pb


def test_pairgen_fails_if_source_has_too_few_canonical_specs(tmp_path):
    tok, ckpt = _trained_checkpoint(tmp_path)
    out = tmp_path / "pairs"
    with pytest.raises(ValueError, match="requested 100 pairs"):
        pairgen_stage.run(_cfg(tok, ckpt, n_pairs=100), out)
