from pathlib import Path

import pytest

from tinyfables.paraphrases import (
    FAMILIES,
    ParaphraseBank,
    Template,
    load_bank,
)

BANK_PATH = Path(__file__).parents[1] / "configs" / "paraphrases.yaml"


def write_bank(tmp_path, templates_yaml, version=1):
    p = tmp_path / "bank.yaml"
    p.write_text(f"version: {version}\ntemplates:\n{templates_yaml}")
    return p


GOOD_TEMPLATE = (
    "  - id: t1\n"
    "    held_out: false\n"
    "    text: |-\n"
    "      {character} in {setting} faces {challenge}; then {outcome}. Moral: {moral}.\n"
)


def test_family_codes_are_stable_order():
    assert FAMILIES == ["canonical", "seen-template", "held-out-template"]


def test_load_bank_parses_templates(tmp_path):
    bank = load_bank(write_bank(tmp_path, GOOD_TEMPLATE))
    assert isinstance(bank, ParaphraseBank)
    assert bank.version == 1
    assert bank.templates[0] == Template(
        id="t1",
        held_out=False,
        text="{character} in {setting} faces {challenge}; then {outcome}. Moral: {moral}.",
    )


def test_seen_and_held_out_partition(tmp_path):
    yaml = GOOD_TEMPLATE + (
        "  - id: h1\n"
        "    held_out: true\n"
        "    text: |-\n"
        "      A fable: {character}/{setting}/{challenge}/{outcome} -> {moral}.\n"
    )
    bank = load_bank(write_bank(tmp_path, yaml))
    assert [t.id for t in bank.seen_templates] == ["t1"]
    assert [t.id for t in bank.held_out_templates] == ["h1"]


def test_choose_seen_never_returns_held_out(tmp_path):
    import random

    yaml = GOOD_TEMPLATE + (
        "  - id: h1\n"
        "    held_out: true\n"
        "    text: |-\n"
        "      A fable: {character}/{setting}/{challenge}/{outcome} -> {moral}.\n"
    )
    bank = load_bank(write_bank(tmp_path, yaml))
    rng = random.Random(0)
    assert all(not bank.choose_seen(rng).held_out for _ in range(200))


def test_rejects_missing_element_slot(tmp_path):
    bad = (
        "  - id: t1\n"
        "    held_out: false\n"
        "    text: |-\n"
        "      {character} in {setting} learns {moral}.\n"  # no challenge/outcome
    )
    with pytest.raises(ValueError):
        load_bank(write_bank(tmp_path, bad))


def test_rejects_unknown_slot(tmp_path):
    bad = (
        "  - id: t1\n"
        "    held_out: false\n"
        "    text: |-\n"
        "      {character}/{setting}/{challenge}/{outcome}/{moral} for {audience}.\n"
    )
    with pytest.raises(ValueError):
        load_bank(write_bank(tmp_path, bad))


def test_rejects_format_spec_that_could_alter_value(tmp_path):
    bad = (
        "  - id: t1\n"
        "    held_out: false\n"
        "    text: |-\n"
        "      {character:.3}/{setting}/{challenge}/{outcome}/{moral}.\n"  # truncation
    )
    with pytest.raises(ValueError):
        load_bank(write_bank(tmp_path, bad))


def test_rejects_duplicate_ids(tmp_path):
    with pytest.raises(ValueError):
        load_bank(write_bank(tmp_path, GOOD_TEMPLATE + GOOD_TEMPLATE))


def test_committed_bank_loads(tmp_path):
    bank = load_bank(BANK_PATH)
    assert len(bank.templates) == 30
    assert len(bank.held_out_templates) == 5


def test_committed_bank_ids_are_unique():
    bank = load_bank(BANK_PATH)
    ids = [t.id for t in bank.templates]
    assert len(ids) == len(set(ids))


from tinyfables.paraphrases import ParsedPrompt, parse_canonical_prompt
from tinyfables.prompts import FableSpec, render_canonical_prompt

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_corpus.jsonl"


def test_parse_recovers_fixture_elements():
    import json

    row0 = json.loads(FIXTURE.read_text().splitlines()[0])
    parsed = parse_canonical_prompt(row0["prompt"])
    assert parsed == ParsedPrompt(
        character="a shy octopus",
        setting="a quiet tide pool",
        challenge="doubting oneself",
        outcome="a friend helps just in time",
        moral="courage grows by small steps",
        age_range="4-7",
        word_count=60,
    )


def test_parse_round_trips_renderer():
    spec = FableSpec(
        character="a bold hare",
        setting="a windy hill",
        challenge="a river to cross",
        outcome="the bridge holds",
        moral="patience pays",
        age_range="8-10",
        word_count=120,
    )
    parsed = parse_canonical_prompt(render_canonical_prompt(spec))
    assert parsed is not None
    for field in ("character", "setting", "challenge", "outcome", "moral", "age_range", "word_count"):
        assert getattr(parsed, field) == getattr(spec, field)


def test_parse_preserves_value_containing_colon():
    spec = FableSpec(
        character="a mole",
        setting="a burrow",
        challenge="a riddle: what walks at dawn?",
        outcome="the riddle is solved",
        moral="think before you dig",
        age_range="4-7",
        word_count=60,
    )
    parsed = parse_canonical_prompt(render_canonical_prompt(spec))
    assert parsed.challenge == "a riddle: what walks at dawn?"


def test_parse_returns_none_on_non_canonical():
    assert parse_canonical_prompt("Just write me a fable about a cat, please.") is None


def test_parse_returns_none_when_an_element_is_missing():
    # A canonical-looking prompt with only four bullets must not parse.
    text = render_canonical_prompt(
        FableSpec(character="a", setting="b", challenge="c", outcome="d", moral="e")
    ).replace("- Teaching: e\n", "")
    assert parse_canonical_prompt(text) is None
