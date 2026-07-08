import pytest

from tinyfables.feedback import AXES, WEIGHTS, aggregate_score, derive_preference


def test_axes_and_weights_are_the_adr_0003_spec():
    assert AXES == ("moral", "adherence", "coherence", "prose")
    assert WEIGHTS == {"moral": 0.4, "adherence": 0.3, "coherence": 0.2, "prose": 0.1}
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


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
