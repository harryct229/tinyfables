"""Preference logic for Aggregate Score and derived pairwise preference."""

from __future__ import annotations

from decimal import Decimal

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

    score_0 = sum(
        Decimal(str(weights[axis])) * Decimal(str(ratings_0[axis])) for axis in AXES
    )
    score_1 = sum(
        Decimal(str(weights[axis])) * Decimal(str(ratings_1[axis])) for axis in AXES
    )
    if score_0 == score_1:
        return None
    return 0 if score_0 > score_1 else 1
