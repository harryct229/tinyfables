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
