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
    from tinyfables.generate import generate_fable_ids

    words: list[int] = []
    new_tokens: list[int] = []

    for i, prompt in enumerate(prompts):
        fable, ids = generate_fable_ids(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            seed=seed + i,
            device=device,
        )
        words.append(len(fable.split()))
        new_tokens.append(len(ids))

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
