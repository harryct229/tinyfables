import json
from pathlib import Path

import pytest
import torch

from tinyfables.config import PrepConfig, PretrainConfig, SourceSpec, TokenizerConfig
from tinyfables.model import GPT
from tinyfables.stages import pretrain as pretrain_stage
from tinyfables.stages import prep as prep_stage
from tinyfables.stages import tokenizer as tokenizer_stage

FIXTURE = str(Path(__file__).parent / "fixtures" / "tiny_corpus.jsonl")
SRC = SourceSpec(jsonl_path=FIXTURE)


@pytest.fixture(scope="module")
def toy_shards(tmp_path_factory):
    root = tmp_path_factory.mktemp("shards")
    tok_dir = root / "tok"
    prep_dir = root / "prep"
    tokenizer_stage.run(TokenizerConfig(source=SRC, vocab_size=512, seed=0), tok_dir)
    prep_stage.run(PrepConfig(source=SRC, tokenizer_dir=str(tok_dir), window=256, seed=0), prep_dir)
    return tok_dir, prep_dir


def toy_cfg(toy_shards, **overrides):
    tok_dir, prep_dir = toy_shards
    base = dict(
        prep_dir=str(prep_dir),
        tokenizer_dir=str(tok_dir),
        n_layer=2, n_head=2, d_model=64, n_ctx=256,
        batch_size=4, steps=6, lr=1e-3, warmup_steps=0, seed=0, device="cpu",
    )
    base.update(overrides)
    return PretrainConfig(**base)


def test_overfit_one_batch_drives_loss_down():
    # A tiny model must memorize a single fixed batch quickly.
    torch.manual_seed(0)
    cfg = PretrainConfig(prep_dir="unused", tokenizer_dir="unused",
                         n_layer=2, n_head=2, d_model=64, n_ctx=64, batch_size=4, steps=0, seed=0)
    model = pretrain_stage.build_model(cfg, vocab_size=48)
    ids = torch.randint(0, 48, (4, 32))
    labels = ids.clone()
    labels[:, :16] = -100  # mask the "prompt" half; loss only on the rest
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(200):
        opt.zero_grad(set_to_none=True)
        out = model(input_ids=ids, labels=labels)
        out.loss.backward()
        opt.step()
        losses.append(out.loss.item())
    assert losses[-1] < 0.5
    assert losses[-1] < losses[0] * 0.2


def test_stage_writes_checkpoint_and_manifest(toy_shards, tmp_path):
    out = tmp_path / "pretrain"
    pretrain_stage.run(toy_cfg(toy_shards, steps=4), out)
    names = {p.name for p in out.iterdir()}
    assert {"config.json", "model.safetensors", "generation_config.json",
            "optimizer.pt", "pretrain_summary.json", "manifest.json"} <= names
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["stage"] == "pretrain"
    assert manifest["artifacts"]
    assert "tokens.bin" in manifest["inputs"] and "tokenizer.json" in manifest["inputs"]
    summary = json.loads((out / "pretrain_summary.json").read_text())
    assert summary["steps"] == 4
    assert 13_000 < summary["n_params"] < 10_000_000  # tiny model, sanity band


def test_amp_flag_is_safe_noop_on_cpu(toy_shards, tmp_path):
    # cfg.amp=True on CPU must not crash (GradScaler is disabled off-CUDA) and
    # must still produce a usable checkpoint with finite loss.
    out = tmp_path / "amp_cpu"
    pretrain_stage.run(toy_cfg(toy_shards, steps=4, amp=True), out)
    summary = json.loads((out / "pretrain_summary.json").read_text())
    assert summary["steps"] == 4
    import math
    assert not math.isnan(summary["final_loss"])


def test_final_optimizer_pt_has_scaler_key(toy_shards, tmp_path):
    out = tmp_path / "sc"
    pretrain_stage.run(toy_cfg(toy_shards, steps=2), out)
    state = torch.load(out / "optimizer.pt", map_location="cpu")
    assert "scaler" in state  # None on CPU (AMP off), but the key is always present
    assert state["step"] == 2


def test_resume_equality_within_tolerance(toy_shards, tmp_path):
    full = tmp_path / "full"
    part = tmp_path / "part"
    resumed = tmp_path / "resumed"
    pretrain_stage.run(toy_cfg(toy_shards, steps=6), full)
    pretrain_stage.run(toy_cfg(toy_shards, steps=3), part)
    pretrain_stage.run(toy_cfg(toy_shards, steps=6, resume_from=str(part)), resumed)

    a = GPT.from_pretrained(full).state_dict()
    b = GPT.from_pretrained(resumed).state_dict()
    assert a.keys() == b.keys()
    for k in a:
        assert torch.allclose(a[k], b[k], atol=1e-5), f"param {k} diverged on resume"
