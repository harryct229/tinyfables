import json
from pathlib import Path

from tinyfables.prompts import FableSpec, render_canonical_prompt

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_corpus.jsonl"


def test_full_spec_matches_fixture_canonical_prompt():
    # tiny_corpus.jsonl is written in itertools.product order; row 0 is
    # (a shy octopus, a quiet tide pool, courage grows by small steps).
    row0 = json.loads(FIXTURE.read_text().splitlines()[0])
    spec = FableSpec(
        character="a shy octopus",
        setting="a quiet tide pool",
        challenge="doubting oneself",
        outcome="a friend helps just in time",
        moral="courage grows by small steps",
        age_range="4-7",
        word_count=60,
    )
    assert render_canonical_prompt(spec) == row0["prompt"]


def test_partial_spec_omits_missing_elements():
    text = render_canonical_prompt(FableSpec(character="a brave mouse"))
    assert text.startswith("Create a fable based on the following elements")
    assert "- Main Character: a brave mouse" in text
    assert "- Setting:" not in text
    assert "- Teaching:" not in text
    assert text.endswith("and about 60 words.")


def test_age_word_line_matches_renderer_output():
    from tinyfables.prompts import AGE_WORD_LINE, FableSpec, render_canonical_prompt

    spec = FableSpec(character="a mouse", age_range="4-7", word_count=60)
    assert render_canonical_prompt(spec).endswith(
        AGE_WORD_LINE.format(age_range="4-7", word_count=60)
    )
