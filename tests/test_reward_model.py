from pathlib import Path

import torch

from tinyfables.model import GPT, GPTConfig
from tinyfables.reward_model import (
    RewardModel,
    bradley_terry_loss,
    load_reward_model,
    preference_accuracy,
    save_reward_model,
)


def tiny_backbone(vocab_size=32):
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=vocab_size, n_layer=1, n_head=2, d_model=16, n_ctx=32))


def test_reward_model_returns_one_scalar_per_sequence():
    rm = RewardModel(tiny_backbone()).eval()
    ids = torch.randint(0, 32, (3, 7))
    mask = torch.ones_like(ids)

    rewards = rm(ids, mask)

    assert rewards.shape == (3,)
    assert rewards.dtype == torch.float32


def test_reward_model_pools_last_non_pad_token():
    rm = RewardModel(tiny_backbone()).eval()
    ids = torch.randint(0, 32, (2, 8))
    mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]])

    rewards = rm(ids, mask)

    assert rewards.shape == (2,)
    assert torch.isfinite(rewards).all()


def test_bradley_terry_loss_and_accuracy():
    chosen = torch.tensor([3.0, 2.0, 1.0])
    rejected = torch.tensor([1.0, 4.0, 1.0])

    loss = bradley_terry_loss(chosen, rejected)
    acc = preference_accuracy(chosen, rejected)

    assert loss.item() > 0
    assert acc == 1 / 3


def test_save_and_load_reward_model_round_trips(tmp_path):
    rm = RewardModel(tiny_backbone()).eval()
    ids = torch.randint(0, 32, (2, 6))
    mask = torch.ones_like(ids)
    before = rm(ids, mask).detach()

    save_reward_model(rm, tmp_path)
    loaded = load_reward_model(tmp_path, device="cpu").eval()
    after = loaded(ids, mask).detach()

    assert (tmp_path / "backbone" / "model.safetensors").exists()
    assert (tmp_path / "reward_head.pt").exists()
    assert (tmp_path / "reward_model_config.json").exists()
    assert torch.allclose(before, after)


def test_from_base_checkpoint_loads_backbone(tmp_path):
    base = tmp_path / "base"
    tiny_backbone().save_pretrained(base)

    rm = RewardModel.from_base_checkpoint(base, device="cpu")

    assert isinstance(rm, RewardModel)
