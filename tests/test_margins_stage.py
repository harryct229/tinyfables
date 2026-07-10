import json
from pathlib import Path

import torch

from tinyfables.config import MarginsConfig, SourceSpec, TokenizerConfig
from tinyfables.model import GPT, GPTConfig
from tinyfables.reward_model import RewardModel, save_reward_model
from tinyfables.stages import margins as margins_stage
from tinyfables.stages import tokenizer as tokenizer_stage

FIXTURES = Path(__file__).parent / "fixtures"


def test_margins_stage_buckets_held_out_accuracy(tmp_path):
    tok_dir = tmp_path / "tok"
    tokenizer_stage.run(
        TokenizerConfig(source=SourceSpec(jsonl_path=str(FIXTURES / "tiny_corpus.jsonl")), vocab_size=512, seed=0),
        tok_dir,
    )
    torch.manual_seed(0)
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
    rm = RewardModel(GPT(GPTConfig(vocab_size=tok.get_vocab_size(), n_layer=1, n_head=2, d_model=32, n_ctx=128)))
    save_reward_model(rm, tmp_path / "rm")

    out = tmp_path / "margins"
    margins_stage.run(
        MarginsConfig(
            preferences=str(FIXTURES / "preferences_replay.jsonl"),
            reward_model_dir=str(tmp_path / "rm"),
            tokenizer_dir=str(tok_dir),
            n_ctx=128,
            batch_size=2,
            device="cpu",
        ),
        out,
    )

    report = json.loads((out / "margins.json").read_text())
    n_held_out = report["n_held_out"]
    assert n_held_out >= 1
    assert 0.0 <= report["overall_accuracy"] <= 1.0
    assert sum(b["n"] for b in report["buckets"]) == n_held_out
    for b in report["buckets"]:
        assert b["accuracy"] is None if b["n"] == 0 else 0.0 <= b["accuracy"] <= 1.0
    assert (out / "margins_report.md").exists()
    assert (out / "manifest.json").exists()


def test_margins_config_rejects_bad_edges():
    import pytest

    with pytest.raises(ValueError):
        MarginsConfig(preferences="p", reward_model_dir="r", tokenizer_dir="t", bucket_edges=[0.5, 0.3])
