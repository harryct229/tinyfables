import csv
import json
from pathlib import Path

import pytest

pytest.importorskip("trl")

import torch

from tinyfables.config import DPOStageConfig, SourceSpec, TokenizerConfig
from tinyfables.model import GPT, GPTConfig
from tinyfables.stages import dpo as dpo_stage
from tinyfables.stages import tokenizer as tokenizer_stage

FIXTURES = Path(__file__).parent / "fixtures"


def _toy_world(tmp_path):
    tok_dir = tmp_path / "tok"
    tokenizer_stage.run(
        TokenizerConfig(source=SourceSpec(jsonl_path=str(FIXTURES / "tiny_corpus.jsonl")), vocab_size=512, seed=0),
        tok_dir,
    )
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
    torch.manual_seed(0)
    base = tmp_path / "base"
    GPT(GPTConfig(vocab_size=tok.get_vocab_size(), n_layer=1, n_head=2, d_model=32, n_ctx=128)).save_pretrained(base)
    return tok_dir, base


def _cfg(tok_dir, base, **overrides):
    kwargs = dict(
        preferences=str(FIXTURES / "preferences_replay.jsonl"),
        base_checkpoint=str(base),
        tokenizer_dir=str(tok_dir),
        adr_decision="ADR-0005-test",
        n_ctx=128,
        lr=1e-4,
        num_train_epochs=1.0,
        batch_size=2,
        gradient_accumulation_steps=1,
        logging_steps=1,
        n_probe_prompts=2,
        probe_max_new_tokens=12,
        seed=0,
        device="cpu",
    )
    kwargs.update(overrides)
    return DPOStageConfig(**kwargs)


def test_dpo_toy_smoke_trains_and_writes_artifacts(tmp_path):
    tok_dir, base = _toy_world(tmp_path)
    out = tmp_path / "aligned"
    dpo_stage.run(_cfg(tok_dir, base), out)

    assert (out / "model.safetensors").exists()
    summary = json.loads((out / "dpo_summary.json").read_text())
    assert summary["adr_decision"] == "ADR-0005-test"
    assert summary["n_train_pairs"] >= 1
    assert isinstance(summary["length_alarm_triggered"], bool)
    assert "preferences_sha" in summary and "base_checkpoint_sha" in summary
    assert list(csv.DictReader((out / "dpo_curves.csv").open()))

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["stage"] == "dpo"
    assert manifest["config"]["adr_decision"] == "ADR-0005-test"
    for key in ("preferences.jsonl", "base_model.safetensors", "tokenizer.json"):
        assert key in manifest["inputs"]

    # aligned weights differ from the base (training happened)
    aligned = GPT.from_pretrained(out)
    base_model = GPT.from_pretrained(base)
    assert not torch.equal(aligned.tok.weight, base_model.tok.weight)


def test_dpo_min_margin_filters_pairs(tmp_path):
    tok_dir, base = _toy_world(tmp_path)
    out = tmp_path / "aligned_filtered"
    # tests/fixtures/preferences_replay.jsonl train-split margins (aggregate_chosen -
    # aggregate_rejected) are [2.3, 1.6, 1.6, 2.3]; min_margin=2.0 keeps the two 2.3
    # pairs and filters the two 1.6 pairs -- "some but not all" of the 4 train rows.
    dpo_stage.run(_cfg(tok_dir, base, min_margin=2.0), out)
    summary = json.loads((out / "dpo_summary.json").read_text())
    assert summary["min_margin"] == 2.0
    assert summary["n_filtered_out"] >= 1
    assert summary["n_train_pairs"] + summary["n_filtered_out"] == summary["n_train_pairs_available"]


def test_dpo_all_pairs_filtered_is_loud(tmp_path):
    tok_dir, base = _toy_world(tmp_path)
    with pytest.raises(ValueError, match="min_margin"):
        dpo_stage.run(_cfg(tok_dir, base, min_margin=99.0), tmp_path / "out")
