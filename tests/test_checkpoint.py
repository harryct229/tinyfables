import torch

from tinyfables.checkpoint import (
    find_latest_checkpoint,
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
)
from tinyfables.model import GPT, GPTConfig


def _model_opt_scaler():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=48, n_layer=2, n_head=2, d_model=64, n_ctx=64))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    return model, opt, scaler


def test_save_then_find_latest(tmp_path):
    model, opt, scaler = _model_opt_scaler()
    save_checkpoint(model, opt, scaler, step=2, ckpt_dir=tmp_path)
    save_checkpoint(model, opt, scaler, step=4, ckpt_dir=tmp_path)
    latest = find_latest_checkpoint(tmp_path)
    assert latest is not None and latest.name == "step_4"
    assert (latest / "checkpoint_state.json").exists()
    assert (latest / "optimizer.pt").exists()
    assert (latest / "model.safetensors").exists()


def test_incomplete_checkpoint_ignored(tmp_path):
    model, opt, scaler = _model_opt_scaler()
    save_checkpoint(model, opt, scaler, step=2, ckpt_dir=tmp_path)
    d = save_checkpoint(model, opt, scaler, step=4, ckpt_dir=tmp_path)
    (d / "checkpoint_state.json").unlink()  # simulate death mid-write (marker last)
    latest = find_latest_checkpoint(tmp_path)
    assert latest.name == "step_2"  # the incomplete step_4 is not selected


def test_find_returns_none_when_empty(tmp_path):
    assert find_latest_checkpoint(tmp_path) is None
    assert find_latest_checkpoint(tmp_path / "missing") is None


def test_load_restores_step_and_optimizer(tmp_path):
    model, opt, scaler = _model_opt_scaler()
    save_checkpoint(model, opt, scaler, step=7, ckpt_dir=tmp_path)
    loaded, state = load_checkpoint(tmp_path / "step_7", GPT, device="cpu")
    assert state["step"] == 7
    assert "optimizer" in state and state["scaler"] is None
    assert isinstance(loaded, GPT)


def test_prune_keeps_last_k(tmp_path):
    model, opt, scaler = _model_opt_scaler()
    for s in (2, 4, 6):
        save_checkpoint(model, opt, scaler, step=s, ckpt_dir=tmp_path)
    prune_checkpoints(tmp_path, keep_last_k=1)
    remaining = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())
    assert remaining == ["step_6"]


def test_load_restores_enabled_scaler_state(tmp_path):
    # AMP-on: unlike the disabled scaler (state_dict() == {} -> normalized to
    # None), an enabled scaler's state_dict is non-empty and must round-trip
    # through save/load intact.
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=48, n_layer=2, n_head=2, d_model=64, n_ctx=64))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cpu", enabled=True)

    save_checkpoint(model, opt, scaler, step=5, ckpt_dir=tmp_path)
    _loaded, state = load_checkpoint(tmp_path / "step_5", GPT, device="cpu")

    assert state["scaler"] is not None
    assert isinstance(state["scaler"], dict)
    assert "scale" in state["scaler"]

    fresh_scaler = torch.amp.GradScaler("cpu", enabled=True)
    fresh_scaler.load_state_dict(state["scaler"])  # must not raise
