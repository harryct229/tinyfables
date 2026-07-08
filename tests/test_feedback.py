import re
from pathlib import Path

import pytest

from tinyfables.feedback import (
    AXES,
    WEIGHTS,
    aggregate_score,
    derive_preference,
    position_flip_rate,
    perturb_weights,
    self_consistency,
    weight_sensitivity,
)


def test_axes_and_weights_are_the_adr_0003_spec():
    assert AXES == ("moral", "adherence", "coherence", "prose")
    assert WEIGHTS == {"moral": 0.4, "adherence": 0.3, "coherence": 0.2, "prose": 0.1}
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


def test_feedback_weights_have_one_production_source_of_truth():
    repo = Path(__file__).resolve().parents[1]
    mapping_hits = []
    formula_hits = []
    formula_re = re.compile(
        r"0\.4\s*\*.*moral.*0\.3\s*\*.*adherence.*0\.2\s*\*.*coherence.*0\.1\s*\*.*prose",
        re.DOTALL,
    )

    for path in (repo / "src" / "tinyfables").rglob("*.py"):
        if path.name == "feedback.py":
            continue
        text = path.read_text()
        if all(
            fragment in text
            for fragment in ('"moral": 0.4', '"adherence": 0.3', '"coherence": 0.2', '"prose": 0.1')
        ):
            mapping_hits.append(path.relative_to(repo).as_posix())
        if formula_re.search(text):
            formula_hits.append(path.relative_to(repo).as_posix())

    assert not mapping_hits
    assert not formula_hits


def test_aggregate_score_is_the_weighted_sum():
    r = {"moral": 5, "adherence": 4, "coherence": 3, "prose": 2}
    # 0.4*5 + 0.3*4 + 0.2*3 + 0.1*2 = 2.0 + 1.2 + 0.6 + 0.2 = 4.0
    assert aggregate_score(r) == pytest.approx(4.0)


def test_derive_preference_prefers_higher_aggregate():
    hi = {"moral": 5, "adherence": 5, "coherence": 5, "prose": 5}
    lo = {"moral": 1, "adherence": 1, "coherence": 1, "prose": 1}
    assert derive_preference(hi, lo) == 0
    assert derive_preference(lo, hi) == 1


def test_derive_preference_skips_ties():
    a = {"moral": 1, "adherence": 1, "coherence": 1, "prose": 3}
    b = {"moral": 1, "adherence": 1, "coherence": 2, "prose": 1}
    assert derive_preference(a, b) is None


def test_derive_preference_prefers_close_but_non_equal_scores():
    a = {"moral": 1.0000000000005, "adherence": 1, "coherence": 1, "prose": 1}
    b = {"moral": 1.0, "adherence": 1, "coherence": 1, "prose": 1}
    assert derive_preference(a, b) == 0


def test_moral_weight_dominates_a_single_axis_swing():
    # moral 5 vs 1 (Δagg 0.4*4=1.6) beats prose 1 vs 5 (Δagg 0.1*4=0.4)
    a = {"moral": 5, "adherence": 3, "coherence": 3, "prose": 1}
    b = {"moral": 1, "adherence": 3, "coherence": 3, "prose": 5}
    assert derive_preference(a, b) == 0


def test_perturb_weights_keeps_sum_one_and_bumps_target():
    w = perturb_weights(WEIGHTS, "moral", 0.1)
    assert w["moral"] == pytest.approx(0.5)
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["adherence"] == pytest.approx(0.25)
    assert w["coherence"] == pytest.approx(0.1666667, abs=1e-4)
    assert w["prose"] == pytest.approx(0.0833333, abs=1e-4)


def test_weight_sensitivity_counts_flips_and_excludes_base_ties():
    robust = (
        {"moral": 5, "adherence": 5, "coherence": 5, "prose": 5},
        {"moral": 1, "adherence": 1, "coherence": 1, "prose": 1},
    )
    tie = (
        {"moral": 3, "adherence": 3, "coherence": 3, "prose": 3},
        {"moral": 3, "adherence": 3, "coherence": 3, "prose": 3},
    )
    knife = (
        {"moral": 3, "adherence": 3, "coherence": 3, "prose": 5},
        {"moral": 3, "adherence": 3, "coherence": 3, "prose": 4},
    )
    out = weight_sensitivity([robust, tie, knife], delta=0.1)
    assert out["delta"] == 0.1
    assert out["n_base_preferences"] == 2
    rows = {(r["axis"], r["direction"]): r for r in out["rows"]}
    assert len(out["rows"]) == 8
    prose_down = rows[("prose", "-")]
    assert prose_down["n_flipped"] == 1
    assert prose_down["pct_flipped"] == pytest.approx(0.5)
    assert all(r["n_flipped"] <= 1 for r in out["rows"])


HI = {"moral": 5, "adherence": 5, "coherence": 5, "prose": 5}
LO = {"moral": 1, "adherence": 1, "coherence": 1, "prose": 1}


def _inst(pair_id, phase, order, r0, r1):
    return {
        "pair_id": pair_id,
        "phase": phase,
        "order": order,
        "ratings_0": r0,
        "ratings_1": r1,
    }


def test_position_flip_rate_counts_disagreement_between_orders():
    instances = [
        _inst("p1", "main", "ab", HI, LO),
        _inst("p1", "swap", "ba", HI, LO),
        _inst("p2", "main", "ab", HI, LO),
        _inst("p2", "swap", "ba", LO, HI),
        _inst("p3", "main", "ab", HI, LO),
    ]
    out = position_flip_rate(instances)
    assert out["n_pairs"] == 2
    assert out["n_flipped"] == 1
    assert out["flip_rate"] == pytest.approx(0.5)


def test_self_consistency_is_mean_pairwise_agreement():
    instances = [
        _inst("c1", "calib-0", "ab", HI, LO),
        _inst("c1", "calib-1", "ab", HI, LO),
        _inst("c1", "calib-2", "ab", HI, LO),
        _inst("c2", "calib-0", "ab", HI, LO),
        _inst("c2", "calib-1", "ab", HI, LO),
        _inst("c2", "calib-2", "ab", LO, HI),
    ]
    out = self_consistency(instances)
    assert out["n_pairs"] == 2
    assert out["n_unanimous"] == 1
    assert out["mean_agreement"] == pytest.approx((1.0 + 1 / 3) / 2)
