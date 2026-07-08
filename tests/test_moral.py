from tinyfables.moral import (
    extract_moral,
    extraction_precision,
    moral_delivered,
    moral_similarity,
    naive_regex_moral,
)


def test_extract_prefers_last_bold_segment():
    fable = "A tale. **Be kind.** More story. **Courage grows when shared**"
    assert extract_moral(fable) == "Courage grows when shared"


def test_extract_falls_back_to_last_sentence_when_no_bold():
    # The issue-04 finding: generations rarely have bold markers.
    fable = "The mouse shared its food. From that day, kindness returned to the meadow."
    assert extract_moral(fable) == "From that day, kindness returned to the meadow"
    assert extract_moral("") is None


def test_naive_regex_baseline_matches_keyword_forms_only():
    assert naive_regex_moral("The moral is: patience pays.") == "patience pays"
    assert naive_regex_moral("Lesson: share and thrive") == "share and thrive"
    # No keyword marker -> the naive baseline MISSES it (extract_moral would not).
    assert naive_regex_moral("And so kindness returned to the meadow.") is None


def test_moral_similarity_and_delivered():
    assert moral_similarity("courage grows when shared", "courage grows when shared") == 1.0
    assert moral_similarity(None, "x") == 0.0
    fable = "The friends learned that courage grows when it is shared."
    assert moral_delivered(fable, "courage grows when shared", threshold=0.5)
    assert not moral_delivered(fable, "greed leads to ruin", threshold=0.5)


def test_extraction_precision():
    assert extraction_precision([True, True, False, True]) == 0.75
    assert extraction_precision([]) == 0.0
