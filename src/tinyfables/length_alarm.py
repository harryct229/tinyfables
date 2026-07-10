"""Length-drift alarm (issue 08): length bias is the classic reward hack, and
the ~250-word fable format makes it detectable. Both alignment stages probe a
fixed prompt set with the current policy and compare mean generated length to
the frozen Base Model's baseline."""

from __future__ import annotations

from statistics import mean


def probe_lengths(
    model,
    tokenizer,
    prompts: list[str],
    *,
    max_new_tokens: int,
    temperature: float,
    seed: int,
    device=None,
) -> dict:
    import torch
    from tinyfables.generate import encode_prompt  # torch imported lazily via caller
    from tinyfables.constants import EOT

    device = device or next(model.parameters()).device
    words: list[int] = []
    new_tokens: list[int] = []

    eot_id = tokenizer.token_to_id(EOT)

    for i, prompt in enumerate(prompts):
        prompt_ids = encode_prompt(tokenizer, prompt)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        if seed is not None:
            torch.manual_seed(seed + i)

        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                use_cache=False,
                pad_token_id=eot_id,
                eos_token_id=eot_id,
                temperature=temperature,
            )

        new_ids = out[0, len(prompt_ids):].tolist()
        if eot_id in new_ids:
            new_ids = new_ids[: new_ids.index(eot_id)]

        fable = tokenizer.decode(new_ids)
        words.append(len(fable.split()))
        new_tokens.append(len(new_ids))

    return {
        "mean_words": mean(words) if words else 0.0,
        "mean_new_tokens": mean(new_tokens) if new_tokens else 0.0,
        "n_prompts": len(prompts),
    }


def length_drift(baseline_mean_words: float, current_mean_words: float, threshold: float) -> dict:
    drift = (
        abs(current_mean_words - baseline_mean_words) / baseline_mean_words
        if baseline_mean_words > 0
        else 0.0
    )
    return {
        "baseline_mean_words": baseline_mean_words,
        "current_mean_words": current_mean_words,
        "drift_pct": drift,
        "alarm": drift > threshold,
    }
