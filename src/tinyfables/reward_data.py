"""Reward-model preference data utilities.

The reward model scores prompt+fable sequences. Derived preference rows from issue
06 already contain chosen/rejected text and a deterministic train/held_out split.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import random
from pathlib import Path

import torch

from tinyfables.constants import EOT, PAD


@dataclass(frozen=True)
class PreferenceExample:
    pair_id: str
    prompt: str
    chosen: str
    rejected: str
    split: str


def load_preferences(path: str | Path, *, split: str | None = None) -> list[PreferenceExample]:
    rows: list[PreferenceExample] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if split is not None and rec["split"] != split:
            continue
        rows.append(
            PreferenceExample(
                pair_id=rec["pair_id"],
                prompt=rec["prompt"],
                chosen=rec["chosen"],
                rejected=rec["rejected"],
                split=rec["split"],
            )
        )
    return rows


def select_train_subset(
    examples: list[PreferenceExample], requested_size: int, seed: int
) -> list[PreferenceExample]:
    if requested_size <= 0:
        raise ValueError("requested_size must be positive")
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    return shuffled[: min(requested_size, len(shuffled))]


def encode_reward_text(tokenizer, prompt: str, fable: str, n_ctx: int) -> list[int]:
    if n_ctx <= 0:
        raise ValueError("n_ctx must be positive")
    eot_id = tokenizer.token_to_id(EOT)
    if eot_id is None:
        raise ValueError("tokenizer lacks <|endoftext|>")
    content_ids = tokenizer.encode(prompt).ids + tokenizer.encode(fable).ids
    if len(content_ids) + 1 <= n_ctx:
        return content_ids + [eot_id]
    return content_ids[: n_ctx - 1] + [eot_id]


def _pad(
    rows: list[list[int]],
    pad_id: int,
    device: str | torch.device,
    width: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(row) for row in rows) if width is None else width
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    for i, row in enumerate(rows):
        ids = torch.tensor(row, dtype=torch.long, device=device)
        input_ids[i, : len(row)] = ids
        attention_mask[i, : len(row)] = 1
    return input_ids, attention_mask


def collate_preference_batch(
    examples: list[PreferenceExample],
    tokenizer,
    n_ctx: int,
    device: str | torch.device,
) -> dict[str, torch.Tensor]:
    if not examples:
        raise ValueError("cannot collate an empty preference batch")
    pad_id = tokenizer.token_to_id(PAD)
    if pad_id is None:
        raise ValueError("tokenizer lacks <|pad|>")
    chosen = [encode_reward_text(tokenizer, ex.prompt, ex.chosen, n_ctx) for ex in examples]
    rejected = [encode_reward_text(tokenizer, ex.prompt, ex.rejected, n_ctx) for ex in examples]
    width = max(max(map(len, chosen)), max(map(len, rejected)))
    chosen_ids, chosen_mask = _pad(chosen, pad_id, device, width)
    rejected_ids, rejected_mask = _pad(rejected, pad_id, device, width)
    return {
        "chosen_input_ids": chosen_ids,
        "chosen_attention_mask": chosen_mask,
        "rejected_input_ids": rejected_ids,
        "rejected_attention_mask": rejected_mask,
    }
