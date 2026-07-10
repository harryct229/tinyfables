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


# --- ADR-0005 majority-vote primitives --------------------------------------
#
# One round of signal improvement runs 3 independent label caches per pair.
# `derive`/`audit` grow a voting mode over these primitives instead of
# operating on a single cache's preference.


def voted_preference(
    ratings_pairs: list[tuple[dict[str, float], dict[str, float]]],
    weights: dict[str, float] = WEIGHTS,
) -> int | None:
    """Majority-vote preference across multiple independent label caches for
    the same fable pair. Each element of `ratings_pairs` is one cache's
    (ratings_0, ratings_1) for the pair.

    A per-cache tie (`derive_preference` returning None) never wins a vote.
    The winner is whichever of {0, 1} holds a strict majority of the votes
    (>=2 of 3 caches in production); anything short of that -- including a
    1-1 split with the remaining cache(s) tied -- returns None. A list with
    only one cache can never reach the >=2 quorum, so it always returns
    None; this generalizes cleanly to the production 3-cache case without
    hardcoding the list length.
    """

    votes = [derive_preference(r0, r1, weights) for r0, r1 in ratings_pairs]
    n0 = votes.count(0)
    n1 = votes.count(1)
    if n0 >= 2 and n0 > n1:
        return 0
    if n1 >= 2 and n1 > n0:
        return 1
    return None


def voted_weight_sensitivity(
    cohort_ratings: list[list[tuple[dict[str, float], dict[str, float]]]],
    weights: dict[str, float] = WEIGHTS,
    delta: float = 0.1,
) -> dict:
    """Like `weight_sensitivity`, but over the voted preference across
    multiple caches. `cohort_ratings[i]` is the list of per-cache
    (ratings_0, ratings_1) tuples for pair i. Re-derives per-cache
    preferences under each perturbed weight set and re-votes; counts
    voted-outcome flips over the pairs that had a non-None voted preference
    at the base weights."""

    base_votes = [voted_preference(caches, weights) for caches in cohort_ratings]
    non_none_indices = [i for i, vote in enumerate(base_votes) if vote is not None]
    rows = []

    for axis in AXES:
        for direction, signed_delta in (("+", delta), ("-", -delta)):
            perturbed = perturb_weights(weights, axis, signed_delta)
            n_flipped = sum(
                1
                for i in non_none_indices
                if voted_preference(cohort_ratings[i], perturbed) != base_votes[i]
            )
            n_base_preferences = len(non_none_indices)
            rows.append(
                {
                    "axis": axis,
                    "direction": direction,
                    "n_flipped": n_flipped,
                    "pct_flipped": (n_flipped / n_base_preferences) if n_base_preferences else 0.0,
                }
            )

    return {"delta": delta, "n_base_preferences": len(non_none_indices), "rows": rows}


def voted_position_flip_rate(instances_by_cache: list[list[dict]]) -> dict:
    """Position-swap audit over main/swap labels for the same pair, voting
    across multiple label caches. Each element of `instances_by_cache` is
    one cache's full instance list (same shape `position_flip_rate` takes
    for a single cache).

    A pair is scored only when it has a main AND a swap instance in EVERY
    cache, so both sides of the comparison see the same number of ballots.
    None (no majority) counts as a value, matching the single-cache
    `_pref`-comparison semantics."""

    mains = [
        {inst["pair_id"]: inst for inst in instances if inst["phase"] == "main"}
        for instances in instances_by_cache
    ]
    swaps = [
        {inst["pair_id"]: inst for inst in instances if inst["phase"] == "swap"}
        for instances in instances_by_cache
    ]
    pair_ids = sorted(
        set.intersection(*(set(m) for m in mains)) & set.intersection(*(set(s) for s in swaps))
    )

    n_flipped = 0
    for pair_id in pair_ids:
        main_vote = voted_preference([(m[pair_id]["ratings_0"], m[pair_id]["ratings_1"]) for m in mains])
        swap_vote = voted_preference([(s[pair_id]["ratings_0"], s[pair_id]["ratings_1"]) for s in swaps])
        if main_vote != swap_vote:
            n_flipped += 1

    n_pairs = len(pair_ids)
    return {
        "n_pairs": n_pairs,
        "n_flipped": n_flipped,
        "flip_rate": (n_flipped / n_pairs) if n_pairs else 0.0,
    }


def voted_self_consistency(instances_by_cache: list[list[dict]]) -> dict:
    """Calibration self-consistency over repeated labels for each pair,
    voting across multiple label caches. Each element of
    `instances_by_cache` is one cache's full instance list.

    Instances are grouped by (pair_id, phase); a single preference is voted
    across the caches for each group present in EVERY cache, and the
    existing pairwise-agreement math then runs over those per-phase voted
    preferences, exactly like the single-cache `self_consistency`."""

    by_key_per_cache: list[dict[tuple[str, str], dict]] = []
    for instances in instances_by_cache:
        by_key_per_cache.append(
            {
                (inst["pair_id"], inst["phase"]): inst
                for inst in instances
                if str(inst["phase"]).startswith("calib")
            }
        )

    common_keys = set.intersection(*(set(d) for d in by_key_per_cache))

    by_pair: dict[str, list[int | None]] = {}
    for pair_id, phase in sorted(common_keys):
        ratings_pairs = [
            (d[(pair_id, phase)]["ratings_0"], d[(pair_id, phase)]["ratings_1"])
            for d in by_key_per_cache
        ]
        by_pair.setdefault(pair_id, []).append(voted_preference(ratings_pairs))

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
