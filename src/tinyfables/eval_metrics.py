"""Deterministic, torch-free text metrics for the eval suite.

These helpers are intentionally mechanical: they measure substring element
adherence, distinct-n / repetition, and simple length statistics without any
model or tokenizer dependencies.
"""

from __future__ import annotations

import re
from statistics import mean, median

from tinyfables.prompts import FableSpec

_ADHERENCE_ELEMENTS = ("character", "setting", "challenge", "outcome")
_WORD_RE = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def element_adherence(fable: str, spec: FableSpec) -> dict[str, bool]:
    """Return case-insensitive substring presence for populated non-moral elements."""

    lowered = fable.lower()
    adherence: dict[str, bool] = {}
    for name in _ADHERENCE_ELEMENTS:
        value = getattr(spec, name)
        if value is not None:
            adherence[name] = value.lower() in lowered
    return adherence


def distinct_n(text: str, n: int) -> float:
    """Return unique n-grams / total n-grams, or 0.0 when too short."""

    tokens = _tokens(text)
    grams = _ngrams(tokens, n)
    if not grams:
        return 0.0
    return len(set(grams)) / len(grams)


def repetition_rate(text: str, n: int = 4) -> float:
    """Return 1 - distinct_n(text, n), or 0.0 when too short."""

    tokens = _tokens(text)
    grams = _ngrams(tokens, n)
    if not grams:
        return 0.0
    return 1.0 - (len(set(grams)) / len(grams))


def length_stats(texts: list[str]) -> dict[str, float | int]:
    """Return word-count summary statistics for a batch of texts."""

    counts = [len(_tokens(text)) for text in texts]
    if not counts:
        return {"mean": 0.0, "median": 0.0, "min": 0, "max": 0}
    return {
        "mean": mean(counts),
        "median": median(counts),
        "min": min(counts),
        "max": max(counts),
    }
