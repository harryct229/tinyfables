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


from tinyfables.paraphrases import render_paraphrase

# Adversarial Element values: braces (must NOT be re-substituted), a literal
# "{character}" (must survive as-is), colons, and quotes (the JSON template).
ADVERSARIAL_SPECS = [
    FableSpec(
        character="a shy octopus",
        setting="a quiet tide pool",
        challenge="doubting oneself",
        outcome="a friend helps just in time",
        moral="courage grows by small steps",
        age_range="4-7",
        word_count=60,
    ),
    FableSpec(
        character="a cat with {setting} posters",   # literal braces + slot name
        setting="a room labeled {character}",
        challenge="a riddle: colons, {braces}, and \"quotes\"",
        outcome="the {outcome} resolves itself",
        moral="be kind: always {moral}",
        age_range="11-13",
        word_count=200,
    ),
]


def test_render_preserves_every_element_value_verbatim():
    bank = load_bank(BANK_PATH)
    for template in bank.templates:  # includes held-out templates
        for spec in ADVERSARIAL_SPECS:
            out = render_paraphrase(template, spec)
            for field in ("character", "setting", "challenge", "outcome", "moral"):
                assert getattr(spec, field) in out, (template.id, field)


def test_render_does_not_reinterpret_injected_braces():
    bank = load_bank(BANK_PATH)
    spec = FableSpec(
        character="{setting}",  # if reprocessed this would become the setting value
        setting="a barn",
        challenge="c",
        outcome="d",
        moral="e",
    )
    for template in bank.templates:
        out = render_paraphrase(template, spec)
        assert "{setting}" in out  # the literal survives; not replaced by "a barn"


def test_render_rejects_missing_element():
    bank = load_bank(BANK_PATH)
    with pytest.raises(ValueError):
        render_paraphrase(bank.seen_templates[0], FableSpec(character="only a character"))


from tinyfables.paraphrases import CANONICAL, HELD_OUT_TEMPLATE, SEEN_TEMPLATE, RowResult, select_row


def _fixture_prompt():
    import json

    return json.loads(FIXTURE.read_text().splitlines()[0])["prompt"]


def test_select_coverage_zero_is_canonical_unchanged():
    bank = load_bank(BANK_PATH)
    text = _fixture_prompt()
    r = select_row(text, 0, coverage=0.0, seed=0, bank=bank)
    assert r == RowResult(text, CANONICAL, False)


def test_select_none_bank_is_canonical():
    text = _fixture_prompt()
    assert select_row(text, 0, coverage=0.5, seed=0, bank=None).family == CANONICAL


def test_select_coverage_one_always_paraphrases_with_seen():
    bank = load_bank(BANK_PATH)
    text = _fixture_prompt()
    held_out_texts = {render_paraphrase(t, parse_canonical_prompt(text)) for t in bank.held_out_templates}
    for i in range(200):
        r = select_row(text, i, coverage=1.0, seed=0, bank=bank)
        assert r.family == SEEN_TEMPLATE
        assert r.prompt != text
        assert r.prompt not in held_out_texts  # held-out phrasing never produced


def test_select_is_deterministic():
    bank = load_bank(BANK_PATH)
    text = _fixture_prompt()
    a = [select_row(text, i, 0.5, 0, bank) for i in range(50)]
    b = [select_row(text, i, 0.5, 0, bank) for i in range(50)]
    assert a == b


def test_select_never_reports_held_out_family():
    bank = load_bank(BANK_PATH)
    text = _fixture_prompt()
    fams = {select_row(text, i, 0.5, s, bank).family for s in range(5) for i in range(200)}
    assert HELD_OUT_TEMPLATE not in fams
    assert fams <= {CANONICAL, SEEN_TEMPLATE}


def test_select_flags_parse_failure_when_bank_active():
    bank = load_bank(BANK_PATH)
    r = select_row("not a canonical prompt", 0, coverage=1.0, seed=0, bank=bank)
    assert r == RowResult("not a canonical prompt", CANONICAL, True)


def test_load_families_round_trip(tmp_path):
    import numpy as np

    from tinyfables.paraphrases import FAMILIES, load_families

    p = tmp_path / "families.bin"
    p.write_bytes(np.array([0, 1, 0, 1, 2], dtype=np.uint8).tobytes())
    arr = load_families(p)
    assert arr.dtype == np.uint8
    assert [FAMILIES[c] for c in arr] == [
        "canonical", "seen-template", "canonical", "seen-template", "held-out-template",
    ]
