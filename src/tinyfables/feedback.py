"""Preference logic for Aggregate Score and derived pairwise preference."""

from __future__ import annotations

from itertools import combinations
from statistics import mean
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


def perturb_weights(weights: dict[str, float], axis: str, delta: float) -> dict[str, float]:
    """Bump one axis by `delta` and renormalize the others proportionally."""

    bumped = weights[axis] + delta
    others = [a for a in AXES if a != axis]
    base_rest = sum(weights[a] for a in others)
    new_rest = 1.0 - bumped
    out = {axis: bumped}
    for other in others:
        out[other] = weights[other] / base_rest * new_rest
    return out


def weight_sensitivity(
    pair_ratings: list[tuple[dict[str, float], dict[str, float]]],
    weights: dict[str, float] = WEIGHTS,
    delta: float = 0.1,
) -> dict:
    """Summarize how many pairwise preferences flip under +/- weight perturbations."""

    base_preferences = [derive_preference(r0, r1, weights) for r0, r1 in pair_ratings]
    non_tie_indices = [i for i, pref in enumerate(base_preferences) if pref is not None]
    rows = []

    for axis in AXES:
        for direction, signed_delta in (("+", delta), ("-", -delta)):
            perturbed = perturb_weights(weights, axis, signed_delta)
            n_flipped = sum(
                1
                for i in non_tie_indices
                if derive_preference(*pair_ratings[i], perturbed) != base_preferences[i]
            )
            n_base_preferences = len(non_tie_indices)
            rows.append(
                {
                    "axis": axis,
                    "direction": direction,
                    "n_flipped": n_flipped,
                    "pct_flipped": (n_flipped / n_base_preferences) if n_base_preferences else 0.0,
                }
            )

    return {"delta": delta, "n_base_preferences": len(non_tie_indices), "rows": rows}


def _pref(inst: dict) -> int | None:
    return derive_preference(inst["ratings_0"], inst["ratings_1"])


def position_flip_rate(instances: list[dict]) -> dict:
    """Position-swap audit over main/swap labels for the same pair."""

    main = {inst["pair_id"]: inst for inst in instances if inst["phase"] == "main"}
    swap = {inst["pair_id"]: inst for inst in instances if inst["phase"] == "swap"}
    pair_ids = sorted(set(main) & set(swap))
    n_flipped = sum(1 for pair_id in pair_ids if _pref(main[pair_id]) != _pref(swap[pair_id]))
    n_pairs = len(pair_ids)
    return {
        "n_pairs": n_pairs,
        "n_flipped": n_flipped,
        "flip_rate": (n_flipped / n_pairs) if n_pairs else 0.0,
    }


def self_consistency(instances: list[dict]) -> dict:
    """Calibration self-consistency over repeated labels for each pair."""

    by_pair: dict[str, list[int | None]] = {}
    for inst in instances:
        if str(inst["phase"]).startswith("calib"):
            by_pair.setdefault(inst["pair_id"], []).append(_pref(inst))

    agreements = []
    n_unanimous = 0
    for prefs in by_pair.values():
        if len(prefs) < 2:
            continue
        total_pairs = 0
        n_agree = 0
        for a, b in combinations(prefs, 2):
            total_pairs += 1
            if a == b:
                n_agree += 1
        agreements.append(n_agree / total_pairs)
        if len(set(prefs)) == 1:
            n_unanimous += 1

    return {
        "n_pairs": len(agreements),
        "mean_agreement": mean(agreements) if agreements else 0.0,
        "n_unanimous": n_unanimous,
    }
