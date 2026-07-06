import json
from pathlib import Path

import pytest

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


def test_render_output_matches_what_the_parser_requires():
    # Guard against drift between the renderer and the parser: the parser requires
    # the word line (regex derived from prompts.WORD_LINE) and the STYLE_BULLETS
    # block verbatim. If render stops emitting either, parse would silently start
    # returning None; this fails first.
    from tinyfables.paraphrases import _WORD_LINE_RE
    from tinyfables.prompts import STYLE_BULLETS, WORD_LINE, FableSpec, render_canonical_prompt

    lines = render_canonical_prompt(FableSpec(character="a mouse", word_count=250)).split("\n")
    assert _WORD_LINE_RE.match(WORD_LINE.format(word_count=250))
    assert any(_WORD_LINE_RE.match(line) for line in lines)
    for bullet in STYLE_BULLETS:
        assert bullet in lines


def test_render_rejects_non_single_band_age_range():
    # ds-tf1-en-3m is single-band; a non-"4-7" age_range must fail loudly rather
    # than render a prompt whose age bullet silently disagrees with the request.
    with pytest.raises(ValueError, match="single-band"):
        render_canonical_prompt(FableSpec(character="a mouse", age_range="8-10"))
