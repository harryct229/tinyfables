from tinyfables.eval_metrics import distinct_n, element_adherence, length_stats, repetition_rate
from tinyfables.prompts import FableSpec

SPEC = FableSpec(
    character="a shy hedgehog",
    setting="a moonlit garden",
    challenge="fear of the dark",
    outcome="finds a friend",
    moral="courage grows when shared",
)


def test_element_adherence_case_insensitive_substring():
    fable = "Once a Shy Hedgehog in a bright cave faced fear of the dark and won."
    a = element_adherence(fable, SPEC)
    assert a == {"character": True, "setting": False, "challenge": True, "outcome": False}
    # moral is NOT in the adherence set (scored separately as moral delivery)
    assert "moral" not in a


def test_element_adherence_skips_missing_elements():
    a = element_adherence("a mouse ran", FableSpec(character="a mouse"))
    assert a == {"character": True}


def test_distinct_n_and_repetition():
    assert distinct_n("a a a a", 1) == 0.25
    assert distinct_n("the cat sat", 2) == 1.0
    assert distinct_n("hi", 3) == 0.0
    assert repetition_rate("a a a a", 1) == 0.75
    assert repetition_rate("hi", 4) == 0.0


def test_length_stats():
    s = length_stats(["one two three", "one two three four five", "one"])
    assert s["min"] == 1 and s["max"] == 5 and s["mean"] == 3.0 and s["median"] == 3
    assert length_stats([]) == {"mean": 0.0, "median": 0.0, "min": 0, "max": 0}
