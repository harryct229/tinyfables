"""Completion-only perplexity over fable tokens.

Each validation row is scored as `prompt + fable + EOT`, but prompt tokens are
masked so the cross-entropy matches training: only completion tokens contribute
to loss. Rows longer than `n_ctx` are skipped. The result is the mean loss over
all scored completion tokens, its perplexity, and the number of rows actually
scored.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from tinyfables.constants import EOT


def fable_token_perplexity(model, tokenizer, rows, n_ctx, device, max_rows) -> tuple[float, float, int]:
    eot_id = tokenizer.token_to_id(EOT)
    if eot_id is None:
        raise ValueError(f"tokenizer is missing required special token {EOT!r}")

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_used = 0

    with torch.no_grad():
        for row in rows:
            if n_used >= max_rows:
                break

            prompt_ids = tokenizer.encode(row["prompt"]).ids
            fable_ids = tokenizer.encode(row["fable"]).ids + [eot_id]
            ids = prompt_ids + fable_ids
            if len(ids) > n_ctx or len(ids) < 2:
                continue

            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            labels = input_ids.clone()
            labels[0, : len(prompt_ids)] = -100

            logits = model(input_ids=input_ids).logits[0, :-1, :]
            target = labels[0, 1:]
            keep = target != -100
            if not torch.any(keep):
                continue

            total_loss += F.cross_entropy(logits[keep], target[keep], reduction="sum").item()
            total_tokens += int(keep.sum().item())
            n_used += 1

    if total_tokens == 0:
        return float("nan"), float("nan"), 0

    mean_loss = total_loss / total_tokens
    return mean_loss, math.exp(mean_loss), n_used
