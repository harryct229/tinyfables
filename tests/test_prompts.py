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
        word_count=250,
    )
    assert render_canonical_prompt(spec) == row0["prompt"]


def test_partial_spec_omits_missing_elements():
    text = render_canonical_prompt(FableSpec(character="a brave mouse"))
    assert text.startswith("Create a fable based on the following elements")
    assert "  - Main Character: a brave mouse" in text
    assert "  - Setting:" not in text
    assert "  - Teaching:" not in text
    assert "The fable should:" in text  # the fixed style block is always emitted
    assert text.endswith("around 250 words.")


def test_word_line_and_age_bullet_match_parser_regexes():
    # Guard against drift between the renderer's output and the parser's regexes
    # (which are derived from prompts.WORD_LINE). If either the word line or the
    # age-group bullet changes shape, this fails before parse silently returns None.
    from tinyfables.paraphrases import _AGE_GROUP_RE, _WORD_LINE_RE
    from tinyfables.prompts import WORD_LINE, FableSpec, render_canonical_prompt

    lines = render_canonical_prompt(FableSpec(character="a mouse", word_count=250)).split("\n")
    assert _WORD_LINE_RE.match(WORD_LINE.format(word_count=250))
    assert any(_WORD_LINE_RE.match(line) for line in lines)
    assert any(_AGE_GROUP_RE.match(line) for line in lines)
