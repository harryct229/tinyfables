"""Preference logic for Aggregate Score and derived pairwise preference."""

from __future__ import annotations

from math import isclose

AXES: tuple[str, ...] = ("moral", "adherence", "coherence", "prose")
WEIGHTS: dict[str, float] = {
    "moral": 0.4,
    "adherence": 0.3,
    "coherence": 0.2,
    "prose": 0.1,
}


def aggregate_score(ratings: dict[str, float], weights: dict[str, float] = WEIGHTS) -> float:
    """Return the weighted sum across the four rating axes."""

    return sum(weights[axis] * ratings[axis] for axis in AXES)


def derive_preference(
    ratings_0: dict[str, float],
    ratings_1: dict[str, float],
    weights: dict[str, float] = WEIGHTS,
) -> int | None:
    """Return 0 or 1 for the preferred fable, or None for an exact tie."""

    score_0 = aggregate_score(ratings_0, weights)
    score_1 = aggregate_score(ratings_1, weights)
    if isclose(score_0, score_1, rel_tol=0.0, abs_tol=1e-12):
        return None
    return 0 if score_0 > score_1 else 1
