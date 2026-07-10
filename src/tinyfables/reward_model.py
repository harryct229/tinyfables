"""Reward model: TinyFables GPT backbone plus scalar reward head."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from tinyfables.model import GPT


class RewardModel(nn.Module):
    def __init__(self, backbone: GPT) -> None:
        super().__init__()
        self.backbone = backbone
        self.reward_head = nn.Linear(backbone.config.d_model, 1)

    @classmethod
    def from_base_checkpoint(
        cls, path: str | Path, device: str | torch.device = "cpu"
    ) -> "RewardModel":
        backbone = GPT.from_pretrained(path).to(device)
        return cls(backbone).to(device)

    def hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        if T > self.backbone.config.n_ctx:
            raise ValueError(f"sequence length {T} exceeds n_ctx {self.backbone.config.n_ctx}")
        pos = torch.arange(T, device=input_ids.device)
        x = self.backbone.tok(input_ids) + self.backbone.pos(pos)[None, :, :]
        for block in self.backbone.blocks:
            x = block(x)
        return self.backbone.lnf(x)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = self.hidden_states(input_ids)
        if attention_mask is None:
            idx = torch.full((input_ids.size(0),), input_ids.size(1) - 1, device=input_ids.device)
        else:
            idx = attention_mask.long().sum(dim=1).clamp(min=1) - 1
        batch = torch.arange(input_ids.size(0), device=input_ids.device)
        pooled = h[batch, idx]
        return self.reward_head(pooled).squeeze(-1)


def bradley_terry_loss(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()


def preference_accuracy(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> float:
    correct = int((chosen_rewards > rejected_rewards).sum().item())
    return correct / chosen_rewards.numel()


def save_reward_model(model: RewardModel, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(out / "backbone")
    torch.save(model.reward_head.state_dict(), out / "reward_head.pt")
    cfg = {
        "backbone_model_type": model.backbone.config.model_type,
        "d_model": model.backbone.config.d_model,
        "pooling": "last_non_pad",
    }
    (out / "reward_model_config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")


def load_reward_model(path: str | Path, device: str | torch.device = "cpu") -> RewardModel:
    root = Path(path)
    backbone = GPT.from_pretrained(root / "backbone").to(device)
    model = RewardModel(backbone).to(device)
    state = torch.load(root / "reward_head.pt", map_location=device)
    model.reward_head.load_state_dict(state)
    return model
