import csv
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from tinyfables.config import RewardTrainConfig, SourceSpec, TokenizerConfig
from tinyfables.model import GPT, GPTConfig
from tinyfables.reward_model import load_reward_model
from tinyfables.stages import REGISTRY
from tinyfables.stages import reward as reward_stage
from tinyfables.stages import tokenizer as tokenizer_stage

FIX = Path(__file__).parent / "fixtures"
PREFS = FIX / "preferences_replay.jsonl"
CORPUS = FIX / "tiny_corpus.jsonl"


def make_toy_inputs(tmp_path):
    tok_dir = tmp_path / "tok"
    tokenizer_stage.run(
        TokenizerConfig(source=SourceSpec(jsonl_path=str(CORPUS)), vocab_size=512, seed=0),
        tok_dir,
    )
    tok = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
    base = tmp_path / "base"
    torch.manual_seed(0)
    GPT(
        GPTConfig(
            vocab_size=tok.get_vocab_size(),
            n_layer=1,
            n_head=2,
            d_model=32,
            n_ctx=128,
        )
    ).save_pretrained(base)
    return tok_dir, base


def cfg(tmp_path, **overrides):
    tok_dir, base = make_toy_inputs(tmp_path)
    data = dict(
        preferences=str(PREFS),
        base_checkpoint=str(base),
        tokenizer_dir=str(tok_dir),
        n_ctx=128,
        batch_size=2,
        steps=8,
        curve_steps=4,
        lr=1e-3,
        weight_decay=0.0,
        warmup_steps=0,
        grad_clip=1.0,
        seed=0,
        device="cpu",
        amp=False,
        log_every=2,
        curve_sizes=[2, 4],
        accuracy_gate=0.65,
    )
    data.update(overrides)
    return RewardTrainConfig(**data)


def test_reward_stage_registered():
    from tinyfables.config import RewardTrainConfig

    assert REGISTRY["reward"] == (RewardTrainConfig, "tinyfables.stages.reward")


def test_reward_stage_trains_and_writes_artifacts(tmp_path):
    out = tmp_path / "reward"
    reward_stage.run(cfg(tmp_path), out)

    assert (out / "backbone" / "model.safetensors").exists()
    assert (out / "reward_head.pt").exists()
    assert (out / "reward_model_config.json").exists()
    assert (out / "reward_summary.json").exists()
    assert (out / "data_curve.json").exists()
    assert (out / "loss_log.csv").exists()
    assert (out / "manifest.json").exists()

    summary = json.loads((out / "reward_summary.json").read_text())
    assert summary["n_train"] == 4
    assert summary["n_held_out"] == 2
    assert 0.0 <= summary["held_out_accuracy"] <= 1.0
    assert summary["accuracy_gate"] == 0.65
    assert summary["rm_gate_pass"] == (summary["held_out_accuracy"] >= 0.65)

    curve = json.loads((out / "data_curve.json").read_text())
    assert [row["requested_train_size"] for row in curve["points"]] == [2, 4]
    assert all(0.0 <= row["held_out_accuracy"] <= 1.0 for row in curve["points"])

    rows = list(csv.DictReader(open(out / "loss_log.csv")))
    assert rows
    assert set(rows[0]) == {"step", "loss", "held_out_accuracy"}
    assert [row["step"] for row in rows].count("8") == 1


def test_saved_reward_model_can_score_after_stage(tmp_path):
    out = tmp_path / "reward"
    reward_stage.run(cfg(tmp_path), out)
    model = load_reward_model(out, device="cpu").eval()
    ids = torch.randint(0, model.backbone.config.vocab_size, (2, 8))
    mask = torch.ones_like(ids)
    rewards = model(ids, mask)
    assert rewards.shape == (2,)


def test_data_curve_clamps_oversized_train_request(tmp_path):
    out = tmp_path / "reward"
    reward_stage.run(cfg(tmp_path, curve_sizes=[99]), out)

    curve = json.loads((out / "data_curve.json").read_text())
    point = curve["points"][0]
    assert point["requested_train_size"] == 99
    assert point["train_size"] == 4
    assert point["steps"] == 4
    assert 0.0 <= point["held_out_accuracy"] <= 1.0


def test_reward_stage_manifest_records_inputs(tmp_path):
    out = tmp_path / "reward"
    reward_stage.run(cfg(tmp_path), out)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["stage"] == "reward"
    assert "preferences.jsonl" in manifest["inputs"]
    assert "tokenizer.json" in manifest["inputs"]
    assert "reward_summary.json" in manifest["artifacts"]
