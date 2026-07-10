import csv
import json
from pathlib import Path

import pytest

pytest.importorskip("trl")

import torch

from tinyfables.config import PPOStageConfig, SourceSpec, TokenizerConfig
from tinyfables.gates import GateError
from tinyfables.model import GPT, GPTConfig
from tinyfables.reward_model import RewardModel, save_reward_model
from tinyfables.stages import ppo as ppo_stage
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
    torch.manual_seed(1)
    rm = RewardModel(GPT(GPTConfig(vocab_size=tok.get_vocab_size(), n_layer=1, n_head=2, d_model=32, n_ctx=128)))
    save_reward_model(rm, tmp_path / "rm")
    return tok_dir, base, tmp_path / "rm"


def _cfg(tmp_path, tok_dir, base, rm_dir, gate):
    return PPOStageConfig(
        gate=str(gate),
        preferences=str(FIXTURES / "preferences_replay.jsonl"),
        base_checkpoint=str(base),
        reward_model_dir=str(rm_dir),
        tokenizer_dir=str(tok_dir),
        adr_decision="ADR-0005-test",
        n_ctx=128,
        response_length=12,
        total_episodes=4,
        batch_size=2,
        gradient_accumulation_steps=1,
        local_rollout_forward_batch_size=2,
        num_ppo_epochs=1,
        num_mini_batches=1,
        lr=1e-4,
        n_probe_prompts=2,
        seed=0,
        device="cpu",
    )


def test_ppo_refuses_without_gate_pass(tmp_path):
    tok_dir, base, rm_dir = _toy_world(tmp_path)
    with pytest.raises(GateError):
        ppo_stage.run(_cfg(tmp_path, tok_dir, base, rm_dir, FIXTURES / "gate_nogo.json"), tmp_path / "out")
    with pytest.raises((GateError, FileNotFoundError)):
        ppo_stage.run(_cfg(tmp_path, tok_dir, base, rm_dir, tmp_path / "missing.json"), tmp_path / "out2")
    assert not (tmp_path / "out" / "manifest.json").exists()


def test_ppo_toy_smoke_trains_and_writes_artifacts(tmp_path):
    tok_dir, base, rm_dir = _toy_world(tmp_path)
    out = tmp_path / "aligned"
    ppo_stage.run(_cfg(tmp_path, tok_dir, base, rm_dir, FIXTURES / "gate_pass.json"), out)

    assert (out / "model.safetensors").exists()
    summary = json.loads((out / "ppo_summary.json").read_text())
    assert summary["adr_decision"] == "ADR-0005-test"
    assert summary["gate"] == "pass"
    assert "preferences_sha" in summary and "base_checkpoint_sha" in summary
    assert isinstance(summary["length_alarm_triggered"], bool)

    rows = list(csv.DictReader((out / "ppo_curves.csv").open()))
    assert rows, "at least one logged iteration"
    for col in ("episode", "objective_kl", "objective_scores", "probe_mean_words", "length_alarm"):
        assert col in rows[0]

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["stage"] == "ppo"
    assert manifest["config"]["adr_decision"] == "ADR-0005-test"
    for key in ("preferences.jsonl", "gate.json", "base_model.safetensors", "tokenizer.json", "reward_head.pt"):
        assert key in manifest["inputs"]

    # the aligned model reloads and generates
    policy = GPT.from_pretrained(out)
    assert policy.config.n_ctx == 128
