"""Adapters satisfying TRL's reward/value-model calling convention.

TRL's experimental PPOTrainer computes rewards as:
    lm_backbone = getattr(model, model.base_model_prefix)
    out = lm_backbone(input_ids=…, attention_mask=…, position_ids=…,
                      return_dict=True, output_hidden_states=True, use_cache=False)
    model.score(out.hidden_states[-1])   # then pools last non-pad itself
so an adapter needs exactly: `base_model_prefix`, the backbone attribute, and a
`.score` head. The TinyFables RewardModel (backbone + reward_head, last-non-pad
pooling) maps onto this 1:1."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from tinyfables.model import GPT


class ScoredModelAdapter(nn.Module):
    base_model_prefix = "backbone"

    def __init__(self, backbone: GPT) -> None:
        super().__init__()
        self.backbone = backbone
        self.score = nn.Linear(backbone.config.d_model, 1)

    @classmethod
    def from_reward_model_dir(
        cls, rm_dir: str | Path, device: str | torch.device = "cpu"
    ) -> "ScoredModelAdapter":
        root = Path(rm_dir)
        backbone = GPT.from_pretrained(root / "backbone").to(device)
        adapter = cls(backbone).to(device)
        state = torch.load(root / "reward_head.pt", map_location=device)
        adapter.score.load_state_dict(state)
        return adapter

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        out = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
        return self.score(out.hidden_states[-1])
