"""The generation contract: one entrypoint that accepts either a FableSpec
(rendered to the Canonical Prompt) or raw instruction text and returns fable
text. The prompt is encoded exactly as the prep stage encodes it (prompt alone,
via `tokenizer.encode(prompt).ids`, no separator) so train and inference token
streams match. The walking-skeleton model has no KV cache, so generation runs
with use_cache=False (the full window is recomputed each step — fine at toy and
ctx-1024 demo scale)."""

from __future__ import annotations

import torch

from tinyfables.constants import EOT
from tinyfables.prompts import FableSpec, render_canonical_prompt


def encode_prompt(tokenizer, request) -> list[int]:
    text = render_canonical_prompt(request) if isinstance(request, FableSpec) else request
    return tokenizer.encode(text).ids


def generate_fable(
    model,
    tokenizer,
    request,
    *,
    max_new_tokens: int = 256,
    min_new_tokens: int = 0,
    temperature: float = 1.0,
    top_k: int | None = None,
    do_sample: bool = False,
    seed: int | None = None,
    device=None,
) -> str:
    device = device or next(model.parameters()).device
    prompt_ids = encode_prompt(tokenizer, request)
    eot_id = tokenizer.token_to_id(EOT)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    if seed is not None:
        torch.manual_seed(seed)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        use_cache=False,
        pad_token_id=eot_id,
        eos_token_id=eot_id,
    )
    if min_new_tokens:
        gen_kwargs["min_new_tokens"] = min_new_tokens
    if do_sample:
        gen_kwargs["temperature"] = temperature
        if top_k is not None:
            gen_kwargs["top_k"] = top_k

    with torch.no_grad():
        out = model.generate(input_ids, **gen_kwargs)

    new_ids = out[0, len(prompt_ids):].tolist()
    if eot_id in new_ids:
        new_ids = new_ids[: new_ids.index(eot_id)]
    return tokenizer.decode(new_ids)
