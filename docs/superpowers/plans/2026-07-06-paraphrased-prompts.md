# Paraphrased Prompts (Issue 03) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an offline paraphrase mechanism — a versioned template bank, a verbatim slot-filling renderer, and prep-stage integration — that rewrites a configurable fraction of training prompts under alternative phrasings while preserving Element values and tagging every row's prompt-family, with five templates provably held out of training.

**Architecture:** A new torch-free module `paraphrases.py` owns the bank (`configs/paraphrases.yaml`), parsing Element values back out of a rendered Canonical Prompt, verbatim rendering, and deterministic per-row selection. `stages/prep.py` calls it once per row: it decides canonical vs. paraphrase from a coverage knob seeded by `(seed, row_index)`, writes a per-row `families.bin` tag stream alongside the token shards, and records family counts in `prep_summary.json`. Held-out templates are never drawn by the training selector, so they never enter the shards or the family tags — a fact checked directly in the artifacts.

**Tech Stack:** Python 3.10+, stdlib (`random`, `re`, `string`, `dataclasses`), `pyyaml`, `numpy`, `tokenizers`. No new dependencies.

## Global Constraints

- No new dependencies (numpy, pyyaml, tokenizers, torch, transformers are already core; do not add others).
- Tests stay offline: run with `.venv/bin/pytest` (the `network` marker is deselected by default via `addopts = "-m 'not network'"`). No test may hit the network.
- Shell state does not persist between Bash calls — always invoke tools as `.venv/bin/pytest` / `.venv/bin/python` with absolute-or-repo-root-relative paths; do not rely on a previous `cd` or activated venv.
- Everything runs at toy scale against the committed fixture `tests/fixtures/tiny_corpus.jsonl` (24 rows, canonical bullet template).
- Element values are preserved **verbatim**: rendering substitutes values with `str.format`, which never re-interprets substituted text. Templates may not use format-specs or conversions on any slot.
- The five Elements are `character, setting, challenge, outcome, moral` (the `moral` slot is labelled "Teaching" in the canonical text — the dataset's field name). `age_range` and `word_count` are audience knobs, not Elements.
- The bank is a **human-curated, versioned artifact**. The 30 template phrasings and the 5 held-out choices in Task 1 are a *draft for curation* — do not commit them until the human has approved/edited them.
- Preserve the stage contract: `manifest.json` is written **last** (marks completion); all other artifacts first.
- Backward compatibility: existing prep configs/tests set no paraphrase fields → coverage defaults to 0.0, bank defaults to None → prompt tokens are byte-identical to today. `families.bin` is still written (all-canonical) so downstream can always read it.

---

## File Structure

**New files:**
- `src/tinyfables/paraphrases.py` — bank loading/validation, canonical-prompt parsing, verbatim rendering, deterministic per-row selection, family constants, `load_families` reader. Torch-free.
- `configs/paraphrases.yaml` — the versioned template bank (30 templates, 5 held out). Human-curated.
- `tests/test_paraphrases.py` — unit + property tests for the module and the committed bank.
- `tests/test_paraphrase_prep.py` — prep-integration tests (proportions, held-out-absence, determinism, parse-failure counting, backward-compat).

**Modified files:**
- `src/tinyfables/prompts.py` — expose `CANONICAL_HEADER`, `ELEMENT_FIELDS`, `AGE_WORD_LINE`; render via `AGE_WORD_LINE` (output unchanged).
- `src/tinyfables/config.py` — `PrepConfig` gains `paraphrase_bank` + `paraphrase_coverage` with validation.
- `src/tinyfables/stages/prep.py` — per-row selection, `families.bin`, summary counts, bank as a hashed manifest input.
- `tests/test_config.py` — cover the new PrepConfig knobs.
- `configs/prep_toy.yaml` — enable paraphrasing so a real toy prep run demonstrates the knob.
- `docs/design.md` — record the issue-03 decisions.
- `docs/issues/README.md` — tick issue 03 (final, post-merge).

**Module boundaries.** `prompts.py` stays the single source of truth for the canonical template (header, element labels, age/word line); `paraphrases.py` imports those constants so the parser can never drift from the renderer. `paraphrases.py` is pure/torch-free like `prompts.py`; the only prep-stage change is a thin per-row call plus one extra output artifact.

---

### Task 1: Paraphrase bank artifact + loader/validation

**Files:**
- Create: `src/tinyfables/paraphrases.py` (bank types + `load_bank` + slot validation only; parsing/rendering/selection land in Tasks 2–4)
- Create: `configs/paraphrases.yaml` (draft — curate before commit)
- Test: `tests/test_paraphrases.py`

**Interfaces:**
- Produces:
  - `CANONICAL = "canonical"`, `SEEN_TEMPLATE = "seen-template"`, `HELD_OUT_TEMPLATE = "held-out-template"`, `FAMILIES = [CANONICAL, SEEN_TEMPLATE, HELD_OUT_TEMPLATE]` (list index == the uint8 code written to `families.bin`).
  - `@dataclass(frozen=True) Template(id: str, held_out: bool, text: str)`
  - `@dataclass(frozen=True) ParaphraseBank(version: int, templates: tuple[Template, ...])` with properties `seen_templates -> tuple[Template, ...]`, `held_out_templates -> tuple[Template, ...]`, and method `choose_seen(rng: random.Random) -> Template`.
  - `load_bank(path: str | Path) -> ParaphraseBank` — validates: unique ids; only the allowed slots `{character, setting, challenge, outcome, moral, age_range, word_count}`; every template references all five element slots; no format-spec/conversion on any slot; at least one seen template.
- Consumes: `ELEMENT_FIELDS` from `prompts.py` (added in Task 2). To keep Task 1 self-contained and testable first, define the element-slot list locally in this task and switch it to import `ELEMENT_FIELDS` in Task 2. (Concretely: in Task 1 write `_ELEMENT_SLOTS = ("character", "setting", "challenge", "outcome", "moral")`; Task 2 replaces the literal with a derivation from the imported `ELEMENT_FIELDS`.)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_paraphrases.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_paraphrases.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tinyfables.paraphrases'` (and the committed-bank tests fail on the missing `configs/paraphrases.yaml`).

- [ ] **Step 3: Write the module (bank + loader only)**

Create `src/tinyfables/paraphrases.py`:

```python
"""Paraphrased Prompts (issue 03): a versioned bank of alternative instruction
phrasings, slot-filled mechanically so Element values are preserved verbatim.

Torch-free (imports anywhere), like prompts.py. Responsibilities added across
issue 03: load_bank (this task), parse_canonical_prompt (Task 2),
render_paraphrase (Task 3), select_row + load_families (Task 4).

Held-out templates (design.md Interface) never enter training: select_row draws
only from the seen templates, and prep records a per-row prompt-family tag so
held-out leakage is checkable directly in the artifacts."""

from __future__ import annotations

import random
import string
from dataclasses import dataclass
from pathlib import Path

import yaml

# Prompt-family tags. The list index is the uint8 code written to families.bin.
CANONICAL = "canonical"
SEEN_TEMPLATE = "seen-template"
HELD_OUT_TEMPLATE = "held-out-template"
FAMILIES = [CANONICAL, SEEN_TEMPLATE, HELD_OUT_TEMPLATE]

# The five story Elements every template must slot in verbatim. In Task 2 this
# is re-derived from prompts.ELEMENT_FIELDS so it cannot drift from the renderer.
_ELEMENT_SLOTS = ("character", "setting", "challenge", "outcome", "moral")
_OPTIONAL_SLOTS = ("age_range", "word_count")
_ALLOWED_SLOTS = frozenset(_ELEMENT_SLOTS) | frozenset(_OPTIONAL_SLOTS)


@dataclass(frozen=True)
class Template:
    id: str
    held_out: bool
    text: str


@dataclass(frozen=True)
class ParaphraseBank:
    version: int
    templates: tuple[Template, ...]

    @property
    def seen_templates(self) -> tuple[Template, ...]:
        return tuple(t for t in self.templates if not t.held_out)

    @property
    def held_out_templates(self) -> tuple[Template, ...]:
        return tuple(t for t in self.templates if t.held_out)

    def choose_seen(self, rng: random.Random) -> Template:
        return rng.choice(self.seen_templates)


def _template_slots(text: str) -> set[str]:
    """Field names referenced by `text`. Rejects positional `{}` and any
    format-spec/conversion, which could silently alter a substituted value."""
    slots: set[str] = set()
    for _literal, field, spec, conv in string.Formatter().parse(text):
        if field is None:
            continue  # literal text, including escaped {{ }}
        if field == "":
            raise ValueError(f"positional {{}} placeholder not allowed: {text!r}")
        if spec or conv:
            raise ValueError(f"format spec/conversion not allowed on {{{field}}}: {text!r}")
        slots.add(field)
    return slots


def load_bank(path: str | Path) -> ParaphraseBank:
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "version" not in data or "templates" not in data:
        raise ValueError("paraphrase bank must be a mapping with 'version' and 'templates'")
    templates: list[Template] = []
    seen_ids: set[str] = set()
    for entry in data["templates"]:
        tid = entry["id"]
        if tid in seen_ids:
            raise ValueError(f"duplicate template id: {tid}")
        seen_ids.add(tid)
        text = entry["text"]
        slots = _template_slots(text)
        unknown = slots - _ALLOWED_SLOTS
        if unknown:
            raise ValueError(f"template {tid} uses unknown slots {sorted(unknown)}")
        missing = frozenset(_ELEMENT_SLOTS) - slots
        if missing:
            raise ValueError(f"template {tid} missing required element slots {sorted(missing)}")
        templates.append(Template(id=tid, held_out=bool(entry["held_out"]), text=text))
    if not any(not t.held_out for t in templates):
        raise ValueError("bank has no seen (training) templates")
    return ParaphraseBank(version=int(data["version"]), templates=tuple(templates))
```

- [ ] **Step 4: Author the draft bank (curate before commit)**

Create `configs/paraphrases.yaml` with the draft below. **Every template contains all five element slots** (`{character} {setting} {challenge} {outcome} {moral}`); `{age_range}`/`{word_count}` are optional per template. The five `held_out: true` templates are deliberately structurally distinct (question / JSON-ish / letter / terse) so they test robustness to unseen phrasing. **Present this to the human for curation and record their edits before committing.**

```yaml
# Paraphrased Prompt bank (issue 03). Versioned, human-curated.
# Every template slot-fills all five Elements verbatim; {age_range}/{word_count}
# are optional. Exactly five templates are held_out: true — they are NEVER used
# in training (select_row draws only from held_out: false), and are the
# measuring stick for the augmentation robustness grid (issues 05/10).
version: 1
templates:
  - id: seen-prose-01
    held_out: false
    text: |-
      Write a short moral fable for children ages {age_range}, around {word_count} words. The hero is {character}, the setting is {setting}, and the central challenge is {challenge}. It should end when {outcome}, teaching the lesson: {moral}.
  - id: seen-fields-02
    held_out: false
    text: |-
      Tell me a fable (about {word_count} words, suitable for ages {age_range}). Main character: {character}. Setting: {setting}. Challenge: {challenge}. Outcome: {outcome}. Moral: {moral}.
  - id: seen-gentle-03
    held_out: false
    text: |-
      Compose a gentle fable for ages {age_range}. It stars {character} in {setting}, who must face {challenge}. In the end, {outcome}, and the story teaches that {moral}. Keep it to roughly {word_count} words.
  - id: seen-young-04
    held_out: false
    text: |-
      Please write a ~{word_count}-word fable for young readers ({age_range}) about {character}. Set it in {setting}. The problem to overcome is {challenge}; by the close, {outcome}. Leave the reader with this moral: {moral}.
  - id: seen-request-05
    held_out: false
    text: |-
      I'd like a fable starring {character}, set in {setting}. The story turns on {challenge} and resolves as {outcome}. Its moral: {moral}. Aim for about {word_count} words, appropriate for ages {age_range}.
  - id: seen-craft-06
    held_out: false
    text: |-
      Craft a fable in which {character} finds themselves in {setting} and must deal with {challenge}. Ultimately, {outcome}. The lesson readers should take away is: {moral}. Length: about {word_count} words, for ages {age_range}.
  - id: seen-equals-07
    held_out: false
    text: |-
      A writing task: produce a short fable (ages {age_range}, ~{word_count} words). Character = {character}; place = {setting}; challenge = {challenge}; ending = {outcome}; moral = {moral}.
  - id: seen-continue-08
    held_out: false
    text: |-
      Once upon a time there was {character} living in {setting}. Continue this into a full fable about {challenge}, ending with {outcome} and the moral that {moral}. Write it for ages {age_range} in about {word_count} words.
  - id: seen-bedtime-09
    held_out: false
    text: |-
      Write a bedtime fable for a {age_range} year-old, about {word_count} words long. Feature {character} in {setting}, dealing with {challenge}. Make sure {outcome}, and that the tale clearly teaches: {moral}.
  - id: seen-labels-10
    held_out: false
    text: |-
      Your task: a moral fable. Protagonist: {character}. Location: {setting}. Conflict: {challenge}. Resolution: {outcome}. Takeaway: {moral}. Write it for ages {age_range}, ~{word_count} words.
  - id: seen-explore-11
    held_out: false
    text: |-
      Create a fable for young children (ages {age_range}) about {character} who lives in {setting}. The fable should explore {challenge}, arrive at {outcome}, and end on the moral: {moral}. Keep it near {word_count} words.
  - id: seen-spin-12
    held_out: false
    text: |-
      Spin a short tale-with-a-lesson: {character}, in {setting}, wrestles with {challenge}. It should end when {outcome}. The moral of the story is {moral}. Target length ~{word_count} words for ages {age_range}.
  - id: seen-needs-13
    held_out: false
    text: |-
      Write a fable. Here's what it needs: a main character ({character}), a setting ({setting}), a challenge ({challenge}), an outcome ({outcome}), and a clear moral ({moral}). Keep it about {word_count} words and suitable for ages {age_range}.
  - id: seen-book-14
    held_out: false
    text: |-
      For a children's book (ages {age_range}), draft a ~{word_count}-word fable. The story follows {character} through {setting}, centers on {challenge}, and finishes with {outcome}. The moral to convey: {moral}.
  - id: seen-give-15
    held_out: false
    text: |-
      Give me a fable about {character}. Where it happens: {setting}. The challenge: {challenge}. How it ends: {outcome}. The moral: {moral}. Please keep it short (about {word_count} words) and age-appropriate for {age_range}.
  - id: seen-imagine-16
    held_out: false
    text: |-
      Imagine {character} in {setting}. Write their fable: they confront {challenge}, and in the end {outcome}. Close with the moral that {moral}. Aim for {word_count} words, readable by ages {age_range}.
  - id: seen-terse-17
    held_out: false
    text: |-
      Short fable request — hero: {character}; world: {setting}; trouble: {challenge}; how it wraps up: {outcome}; lesson: {moral}. Roughly {word_count} words, for ages {age_range}.
  - id: seen-must-18
    held_out: false
    text: |-
      Write a moral story (a fable) of about {word_count} words for ages {age_range}. It must include {character} as the main character, take place in {setting}, revolve around {challenge}, conclude with {outcome}, and end by teaching {moral}.
  - id: seen-plain-19
    held_out: false
    text: |-
      Draft a fable where the main character is {character} and the setting is {setting}. The challenge they face is {challenge}. The outcome is {outcome}. The moral is {moral}. Write about {word_count} words for ages {age_range}.
  - id: seen-canyou-20
    held_out: false
    text: |-
      Can you write a little fable? It's about {character} in {setting}, facing {challenge}. It ends with {outcome}, and the moral is {moral}. Keep it around {word_count} words, for kids ages {age_range}.
  - id: seen-brief-21
    held_out: false
    text: |-
      Fable brief — Audience: ages {age_range}. Length: ~{word_count} words. Star: {character}. Place: {setting}. Central challenge: {challenge}. Ending: {outcome}. Moral: {moral}.
  - id: seen-suited-22
    held_out: false
    text: |-
      Please compose a fable suited to ages {age_range}. Center it on {character} in {setting}. Build the plot around {challenge}, resolve it so that {outcome}, and finish with the moral: {moral}. About {word_count} words.
  - id: seen-animal-23
    held_out: false
    text: |-
      Write me a short animal-fable-style story (~{word_count} words, ages {age_range}) featuring {character}, set in {setting}. The heart of it is {challenge}; it ends when {outcome}; and it teaches: {moral}.
  - id: seen-new-24
    held_out: false
    text: |-
      New fable. Main character: {character}. Setting: {setting}. The challenge at its core: {challenge}. By the end: {outcome}. Moral to land: {moral}. Keep to about {word_count} words, ages {age_range}.
  - id: seen-lesson-25
    held_out: false
    text: |-
      Tell a story that teaches a lesson. {character} is in {setting} and must handle {challenge}. Things resolve when {outcome}. The lesson: {moral}. Make it about {word_count} words and fit for ages {age_range}.
  - id: heldout-minimal-01
    held_out: true
    text: |-
      A fable about {character} in {setting}: they face {challenge}, then {outcome}. Moral — {moral}.
  - id: heldout-question-02
    held_out: true
    text: |-
      What happens when {character} in {setting} must confront {challenge}? Write the fable; make it end with {outcome} and teach that {moral}. Keep it around {word_count} words for ages {age_range}.
  - id: heldout-json-03
    held_out: true
    text: |-
      Write a fable from this spec: {{"character": "{character}", "setting": "{setting}", "challenge": "{challenge}", "outcome": "{outcome}", "moral": "{moral}"}}. Suitable for ages {age_range}, about {word_count} words.
  - id: heldout-letter-04
    held_out: true
    text: |-
      Dear storyteller, please write us a fable for our ages-{age_range} class (about {word_count} words). We'd love it to be about {character} in {setting}, to deal with {challenge}, to end with {outcome}, and to teach {moral}. Thank you!
  - id: heldout-headline-05
    held_out: true
    text: |-
      Fable, ~{word_count} words, ages {age_range}. {character} + {setting} + {challenge} -> {outcome}. Lesson: {moral}.
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_paraphrases.py -q`
Expected: PASS (10 tests).

- [ ] **Step 6: Commit**

```bash
git add src/tinyfables/paraphrases.py configs/paraphrases.yaml tests/test_paraphrases.py
git commit -m "feat: paraphrase template bank + loader/validation (issue 03)"
```

---

### Task 2: Canonical-prompt parsing (recover Element values)

**Files:**
- Modify: `src/tinyfables/prompts.py` (add public constants; render via `AGE_WORD_LINE`)
- Modify: `src/tinyfables/paraphrases.py` (add `ParsedPrompt`, `parse_canonical_prompt`; derive `_ELEMENT_SLOTS` from the import)
- Test: `tests/test_paraphrases.py` (append), `tests/test_prompts.py` (append round-trip)

**Interfaces:**
- Consumes: `CANONICAL_HEADER: str`, `ELEMENT_FIELDS: tuple[tuple[str, str], ...]` (each `(field, label)`), `AGE_WORD_LINE: str` — added to `prompts.py`.
- Produces:
  - `@dataclass(frozen=True) ParsedPrompt(character, setting, challenge, outcome, moral: str | None = None, age_range: str = "4-7", word_count: int = 60)` — same field set as `FableSpec`.
  - `parse_canonical_prompt(text: str) -> ParsedPrompt | None` — returns `None` unless `text` is exactly `CANONICAL_HEADER` + 5 element bullets + the age/word line, with all five element labels present. Values are taken verbatim (split on the first `": "`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_paraphrases.py`:

```python
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
```

Append to `tests/test_prompts.py`:

```python
def test_age_word_line_matches_renderer_output():
    from tinyfables.prompts import AGE_WORD_LINE, FableSpec, render_canonical_prompt

    spec = FableSpec(character="a mouse", age_range="4-7", word_count=60)
    assert render_canonical_prompt(spec).endswith(
        AGE_WORD_LINE.format(age_range="4-7", word_count=60)
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_paraphrases.py tests/test_prompts.py -q`
Expected: FAIL — `ImportError`/`AttributeError` for `ParsedPrompt`, `parse_canonical_prompt`, `AGE_WORD_LINE`.

- [ ] **Step 3: Add public constants to `prompts.py`**

In `src/tinyfables/prompts.py`, after the `_ELEMENTS` list (around line 22) add:

```python
# Public views for parsers/renderers (paraphrases.py) so the parser cannot drift
# from this module's canonical template.
CANONICAL_HEADER = _HEADER
ELEMENT_FIELDS = tuple(_ELEMENTS)
AGE_WORD_LINE = "Keep it age-appropriate for ages {age_range} and about {word_count} words."
```

Then change the trailing-line append inside `render_canonical_prompt` from the f-string to `AGE_WORD_LINE` (output byte-identical):

```python
    lines.append(AGE_WORD_LINE.format(age_range=spec.age_range, word_count=spec.word_count))
```

- [ ] **Step 4: Wire the import and add the parser in `paraphrases.py`**

In `src/tinyfables/paraphrases.py`, add `import re` at the top, and replace the temporary `_ELEMENT_SLOTS` literal + add the label map and age/word regex:

```python
from tinyfables.prompts import AGE_WORD_LINE, CANONICAL_HEADER, ELEMENT_FIELDS

_ELEMENT_SLOTS = tuple(field for field, _label in ELEMENT_FIELDS)
_OPTIONAL_SLOTS = ("age_range", "word_count")
_ALLOWED_SLOTS = frozenset(_ELEMENT_SLOTS) | frozenset(_OPTIONAL_SLOTS)

_LABEL_TO_FIELD = {label: field for field, label in ELEMENT_FIELDS}
# Mirrors prompts.AGE_WORD_LINE; test_age_word_line_matches_renderer_output guards drift.
_AGE_WORD_RE = re.compile(
    r"^Keep it age-appropriate for ages (?P<age_range>.+?) and about (?P<word_count>\d+) words\.$"
)
```

Add the dataclass + parser (place `ParsedPrompt` near `Template`, and `parse_canonical_prompt` after `load_bank`):

```python
@dataclass(frozen=True)
class ParsedPrompt:
    character: str | None = None
    setting: str | None = None
    challenge: str | None = None
    outcome: str | None = None
    moral: str | None = None
    age_range: str = "4-7"
    word_count: int = 60


def parse_canonical_prompt(text: str) -> ParsedPrompt | None:
    """Recover Element values from a rendered Canonical Prompt, or None if `text`
    is not the canonical structure (header + 5 element bullets + age/word line).
    Values are verbatim (split on the first ': ')."""
    lines = text.split("\n")
    if len(lines) != len(ELEMENT_FIELDS) + 2 or lines[0] != CANONICAL_HEADER:
        return None
    values: dict[str, object] = {}
    for line in lines[1:-1]:
        if not line.startswith("- ") or ": " not in line:
            return None
        label, value = line[2:].split(": ", 1)
        field = _LABEL_TO_FIELD.get(label)
        if field is None:
            return None
        values[field] = value
    if set(values) != set(_ELEMENT_SLOTS):
        return None
    m = _AGE_WORD_RE.match(lines[-1])
    if m is None:
        return None
    return ParsedPrompt(age_range=m["age_range"], word_count=int(m["word_count"]), **values)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_paraphrases.py tests/test_prompts.py -q`
Expected: PASS.

- [ ] **Step 6: Run the full suite (guard the `prompts.py` refactor)**

Run: `.venv/bin/pytest -q`
Expected: PASS — the canonical output is unchanged, so `test_prep_stage.py` / `test_toy_chain.py` still pass.

- [ ] **Step 7: Commit**

```bash
git add src/tinyfables/prompts.py src/tinyfables/paraphrases.py tests/test_paraphrases.py tests/test_prompts.py
git commit -m "feat: parse Element values back out of a Canonical Prompt (issue 03)"
```

---

### Task 3: Verbatim rendering + property test

**Files:**
- Modify: `src/tinyfables/paraphrases.py` (add `render_paraphrase` + `_element_values`)
- Test: `tests/test_paraphrases.py` (append the property test)

**Interfaces:**
- Consumes: `Template`, `ParsedPrompt` (Task 2), `FableSpec` (from `prompts.py`).
- Produces:
  - `render_paraphrase(template: Template, obj) -> str` — `obj` is any object with the seven attributes (`ParsedPrompt` or `FableSpec`). Substitutes verbatim via `str.format`. Raises `ValueError` if any of the five element values is `None`.
  - `_element_values(obj) -> dict` (internal helper).

- [ ] **Step 1: Write the failing property test**

Append to `tests/test_paraphrases.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_paraphrases.py -k render -q`
Expected: FAIL — `render_paraphrase` is undefined.

- [ ] **Step 3: Implement the renderer**

Add to `src/tinyfables/paraphrases.py`:

```python
def _element_values(obj) -> dict:
    values = {slot: getattr(obj, slot) for slot in _ALLOWED_SLOTS}
    missing = [s for s in _ELEMENT_SLOTS if values[s] is None]
    if missing:
        raise ValueError(f"cannot paraphrase: missing element values {missing}")
    return values


def render_paraphrase(template: Template, obj) -> str:
    """Render `template` from a ParsedPrompt/FableSpec. str.format substitutes
    Element values verbatim and never re-interprets the substituted text, so
    values containing braces/colons/quotes are preserved exactly."""
    return template.text.format(**_element_values(obj))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_paraphrases.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tinyfables/paraphrases.py tests/test_paraphrases.py
git commit -m "feat: verbatim paraphrase rendering + property test (issue 03)"
```

---

### Task 4: Deterministic per-row selection + families reader

**Files:**
- Modify: `src/tinyfables/paraphrases.py` (add `RowResult`, `select_row`, `load_families`)
- Test: `tests/test_paraphrases.py` (append selection tests)

**Interfaces:**
- Consumes: `ParaphraseBank`, `parse_canonical_prompt`, `render_paraphrase`, `CANONICAL`, `SEEN_TEMPLATE`.
- Produces:
  - `@dataclass(frozen=True) RowResult(prompt: str, family: str, parse_failed: bool)`.
  - `select_row(text: str, row_index: int, coverage: float, seed: int, bank: ParaphraseBank | None) -> RowResult` — deterministic in `(seed, row_index)`. Returns canonical unchanged when `bank is None` or `coverage <= 0`; returns canonical with `parse_failed=True` when the bank is active but `text` is not canonical; otherwise paraphrases with probability `coverage` using a **seen** template. `family` is only ever `canonical` or `seen-template` (never `held-out-template`).
  - `load_families(path: str | Path) -> numpy.ndarray` — reads `families.bin` as `uint8`; `FAMILIES[code]` gives the tag name.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_paraphrases.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_paraphrases.py -k "select or load_families" -q`
Expected: FAIL — `select_row` / `load_families` undefined.

- [ ] **Step 3: Implement selection + reader**

Add to `src/tinyfables/paraphrases.py`:

```python
@dataclass(frozen=True)
class RowResult:
    prompt: str
    family: str
    parse_failed: bool


def select_row(
    text: str, row_index: int, coverage: float, seed: int, bank: ParaphraseBank | None
) -> RowResult:
    """Decide a row's prompt + prompt-family. Deterministic in (seed, row_index)
    so two prep runs with the same config produce identical families. Training
    never uses held-out templates, so family is canonical or seen-template."""
    if bank is None or coverage <= 0.0:
        return RowResult(text, CANONICAL, False)
    parsed = parse_canonical_prompt(text)
    if parsed is None:
        return RowResult(text, CANONICAL, True)
    rng = random.Random(f"{seed}-{row_index}")
    if rng.random() >= coverage:
        return RowResult(text, CANONICAL, False)
    template = bank.choose_seen(rng)
    return RowResult(render_paraphrase(template, parsed), SEEN_TEMPLATE, False)


def load_families(path: str | Path):
    """Read families.bin as a uint8 array; FAMILIES[code] gives the tag name."""
    import numpy as np

    return np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_paraphrases.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tinyfables/paraphrases.py tests/test_paraphrases.py
git commit -m "feat: deterministic per-row paraphrase selection + families reader (issue 03)"
```

---

### Task 5: PrepConfig knobs

**Files:**
- Modify: `src/tinyfables/config.py` (`PrepConfig` fields + `__post_init__`)
- Test: `tests/test_config.py` (append)

**Interfaces:**
- Produces: `PrepConfig` gains `paraphrase_bank: str | None = None`, `paraphrase_coverage: float = 0.0`. Validation: coverage in `[0, 1]`; `coverage > 0` requires a non-None `paraphrase_bank`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`:

```python
def test_prep_paraphrase_defaults():
    cfg = PrepConfig(source=SourceSpec(jsonl_path="c.jsonl"), tokenizer_dir="t")
    assert cfg.paraphrase_bank is None
    assert cfg.paraphrase_coverage == 0.0


def test_prep_loads_paraphrase_knobs(tmp_path):
    p = write_yaml(
        tmp_path,
        "source:\n  jsonl_path: corpus.jsonl\ntokenizer_dir: runs/tok\n"
        "paraphrase_bank: configs/paraphrases.yaml\nparaphrase_coverage: 0.15\n",
    )
    cfg = load_config(p, PrepConfig)
    assert cfg.paraphrase_bank == "configs/paraphrases.yaml"
    assert cfg.paraphrase_coverage == 0.15


def test_prep_rejects_coverage_out_of_range():
    with pytest.raises(ValueError):
        PrepConfig(
            source=SourceSpec(jsonl_path="c.jsonl"),
            tokenizer_dir="t",
            paraphrase_bank="b.yaml",
            paraphrase_coverage=1.5,
        )


def test_prep_coverage_requires_bank():
    with pytest.raises(ValueError):
        PrepConfig(
            source=SourceSpec(jsonl_path="c.jsonl"),
            tokenizer_dir="t",
            paraphrase_coverage=0.2,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_config.py -q`
Expected: FAIL — `PrepConfig` has no `paraphrase_bank`/`paraphrase_coverage` (and no validation).

- [ ] **Step 3: Add the fields + validation**

In `src/tinyfables/config.py`, replace the `PrepConfig` dataclass body with:

```python
@dataclass(frozen=True)
class PrepConfig:
    source: SourceSpec
    tokenizer_dir: str
    window: int = 1024
    seed: int = 0
    paraphrase_bank: str | None = None
    paraphrase_coverage: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.paraphrase_coverage <= 1.0:
            raise ValueError("paraphrase_coverage must be in [0, 1]")
        if self.paraphrase_coverage > 0.0 and self.paraphrase_bank is None:
            raise ValueError("paraphrase_coverage > 0 requires paraphrase_bank")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_config.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tinyfables/config.py tests/test_config.py
git commit -m "feat: PrepConfig paraphrase_bank + paraphrase_coverage knobs (issue 03)"
```

---

### Task 6: Prep integration — families.bin + summary counts + bank provenance

**Files:**
- Modify: `src/tinyfables/stages/prep.py`
- Test: `tests/test_paraphrase_prep.py` (new)

**Interfaces:**
- Consumes: `select_row`, `load_bank`, `load_families`, `FAMILIES`, `CANONICAL`, `SEEN_TEMPLATE`, `HELD_OUT_TEMPLATE` (paraphrases.py); `PrepConfig` (Task 5).
- Produces (new prep artifacts/behavior):
  - `families.bin` — one `uint8` per input row, in prep-consumption (post-shuffle) order; code = `FAMILIES.index(family)`. Listed in `manifest.json` artifacts.
  - `prep_summary.json` gains `paraphrase_coverage: float`, `n_parse_failures: int`, `family_counts: {canonical, seen-template, held-out-template}`.
  - When `paraphrase_bank` is set, the bank file is added to `manifest.json` `inputs` under key `paraphrases.yaml` (SHA-256 tracked).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_paraphrase_prep.py`:

```python
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from tinyfables.config import PrepConfig, SourceSpec, TokenizerConfig
from tinyfables.paraphrases import FAMILIES, load_families
from tinyfables.stages import prep as prep_stage
from tinyfables.stages import tokenizer as tokenizer_stage

FIXTURE = str(Path(__file__).parent / "fixtures" / "tiny_corpus.jsonl")
BANK = str(Path(__file__).parents[1] / "configs" / "paraphrases.yaml")
SRC = SourceSpec(jsonl_path=FIXTURE)


def _tok_dir(tmp_path):
    out = tmp_path / "tok"
    tokenizer_stage.run(TokenizerConfig(source=SRC, vocab_size=512, seed=0), out)
    return out


def _run(tmp_path, coverage, name, bank=BANK):
    out = tmp_path / name
    cfg = PrepConfig(
        source=SRC,
        tokenizer_dir=str(_tok_dir(tmp_path)),
        window=256,
        seed=0,
        paraphrase_bank=bank if coverage > 0 else bank,
        paraphrase_coverage=coverage,
    )
    prep_stage.run(cfg, out)
    return out


def test_families_written_one_per_row(tmp_path):
    out = _run(tmp_path, 0.5, "half")
    fam = load_families(out / "families.bin")
    summary = json.loads((out / "prep_summary.json").read_text())
    assert len(fam) == summary["n_rows"] == 24
    assert set(np.unique(fam)) <= {0, 1}  # canonical / seen only in training


def test_summary_counts_show_expected_proportions(tmp_path):
    out = _run(tmp_path, 0.5, "half")
    s = json.loads((out / "prep_summary.json").read_text())
    fc = s["family_counts"]
    assert fc["canonical"] + fc["seen-template"] == 24
    assert fc["held-out-template"] == 0
    assert 6 <= fc["seen-template"] <= 18  # ~half of 24, deterministic-but-banded
    assert s["paraphrase_coverage"] == 0.5


def test_coverage_zero_is_all_canonical_and_tokens_match_no_bank(tmp_path):
    # Prompt tokens must be byte-identical whether paraphrasing is off via
    # coverage=0 or absent entirely (backward compatibility).
    with_bank = _run(tmp_path, 0.0, "cov0")
    no_bank = tmp_path / "nobank"
    prep_stage.run(
        PrepConfig(source=SRC, tokenizer_dir=str(_tok_dir(tmp_path)), window=256, seed=0),
        no_bank,
    )
    fam = load_families(with_bank / "families.bin")
    assert set(np.unique(fam)) == {0}  # all canonical
    a = (with_bank / "tokens.bin").read_bytes()
    b = (no_bank / "tokens.bin").read_bytes()
    assert a == b


def test_held_out_never_in_training_artifacts(tmp_path):
    # At full coverage every parseable row is paraphrased; held-out must be 0.
    out = _run(tmp_path, 1.0, "full")
    s = json.loads((out / "prep_summary.json").read_text())
    assert s["family_counts"]["held-out-template"] == 0
    assert s["family_counts"]["seen-template"] == 24
    fam = load_families(out / "families.bin")
    assert 2 not in set(np.unique(fam))  # code 2 == held-out-template


def test_prep_families_are_hash_deterministic(tmp_path):
    a = json.loads((_run(tmp_path, 0.5, "a") / "manifest.json").read_text())
    b = json.loads((_run(tmp_path, 0.5, "b") / "manifest.json").read_text())
    assert a["artifacts"] == b["artifacts"]  # includes families.bin + summary


def test_manifest_tracks_bank_input_hash(tmp_path):
    out = _run(tmp_path, 0.5, "prov")
    manifest = json.loads((out / "manifest.json").read_text())
    assert "paraphrases.yaml" in manifest["inputs"]
    assert "families.bin" in manifest["artifacts"]


def test_parse_failures_counted_and_left_canonical(tmp_path):
    corpus = tmp_path / "mixed.jsonl"
    canonical = json.loads(Path(FIXTURE).read_text().splitlines()[0])
    bad = {"prompt": "Write a fable about a fox.", "fable": "A fox learned to share.\n\n**The Moral:** share"}
    corpus.write_text(json.dumps(canonical) + "\n" + json.dumps(bad) + "\n")
    src = SourceSpec(jsonl_path=str(corpus))
    tok = tmp_path / "tok2"
    tokenizer_stage.run(TokenizerConfig(source=src, vocab_size=512, seed=0), tok)
    out = tmp_path / "mixedprep"
    prep_stage.run(
        PrepConfig(source=src, tokenizer_dir=str(tok), window=256, seed=0,
                   paraphrase_bank=BANK, paraphrase_coverage=1.0),
        out,
    )
    s = json.loads((out / "prep_summary.json").read_text())
    assert s["n_parse_failures"] == 1  # the non-canonical row fell back to canonical
    fam = load_families(out / "families.bin")
    assert (fam == FAMILIES.index("canonical")).sum() >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_paraphrase_prep.py -q`
Expected: FAIL — prep does not yet write `families.bin` / the new summary keys / bank input.

- [ ] **Step 3: Integrate paraphrasing into `prep.py`**

In `src/tinyfables/stages/prep.py`, add the import near the others:

```python
from tinyfables.paraphrases import FAMILIES, load_bank, select_row
```

Load the bank once and initialize per-row accumulators, before the `with open(...)` block (after `mask_path = out_dir / "mask.bin"`):

```python
    families_path = out_dir / "families.bin"
    bank = load_bank(cfg.paraphrase_bank) if cfg.paraphrase_bank else None
    families: list[int] = []
    fam_counts = {name: 0 for name in FAMILIES}
    n_parse_failures = 0
```

Replace the per-row body of the `for` loop. Change the loop header to enumerate, and select the prompt before tokenizing:

```python
        for i, r in enumerate(iter_rows(cfg.source, cfg.seed)):
            n_rows += 1
            row = select_row(r["prompt"], i, cfg.paraphrase_coverage, cfg.seed, bank)
            fam_counts[row.family] += 1
            families.append(FAMILIES.index(row.family))
            if row.parse_failed:
                n_parse_failures += 1
            p = tok.encode(row.prompt).ids
            f = tok.encode(r["fable"]).ids
            toks_buf.extend(p)
            mask_buf.extend([0] * len(p))
            toks_buf.extend(f)
            toks_buf.append(eot_id)
            mask_buf.extend([1] * (len(f) + 1))
            n_prompt += len(p)
            n_fable += len(f) + 1
            if len(toks_buf) >= _FLUSH_AT_TOKENS:
                flush()
        flush()
```

After the token/mask truncation block and before building `summary`, write the families artifact (per-row and tiny — O(n_rows) bytes, no incremental flush needed):

```python
    families_path.write_bytes(np.asarray(families, dtype=np.uint8).tobytes())
```

Add the three new keys to the `summary` dict:

```python
        "paraphrase_coverage": cfg.paraphrase_coverage,
        "n_parse_failures": n_parse_failures,
        "family_counts": fam_counts,
```

Finally, extend the manifest call to list `families.bin` and (when present) hash the bank as an input:

```python
    inputs = {"tokenizer.json": tokenizer_path}
    if cfg.paraphrase_bank:
        inputs["paraphrases.yaml"] = Path(cfg.paraphrase_bank)

    write_manifest(
        out_dir,
        "prep",
        cfg,
        [tokens_path, mask_path, families_path, summary_path],
        inputs=inputs,
    )
```

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `.venv/bin/pytest tests/test_paraphrase_prep.py -q`
Expected: PASS.

- [ ] **Step 5: Run the full suite (guard backward compatibility)**

Run: `.venv/bin/pytest -q`
Expected: PASS — `test_prep_stage.py` (no bank → prompts unchanged, `families.bin` all-canonical) and `test_toy_chain.py` still pass.

- [ ] **Step 6: Commit**

```bash
git add src/tinyfables/stages/prep.py tests/test_paraphrase_prep.py
git commit -m "feat: prep writes prompt-family tags + paraphrases via the bank (issue 03)"
```

---

### Task 7: Toy config demo + design record

**Files:**
- Modify: `configs/prep_toy.yaml` (enable paraphrasing so a real toy prep run demonstrates the knob)
- Modify: `docs/design.md` (record issue-03 decisions)

**Interfaces:** none (config/docs only). No new code.

- [ ] **Step 1: Enable paraphrasing in the toy prep config**

Replace `configs/prep_toy.yaml` with (adds bank + a mid-design-range 15% coverage):

```yaml
source:
  jsonl_path: tests/fixtures/tiny_corpus.jsonl
tokenizer_dir: runs/tokenizer_toy
window: 256
seed: 0
paraphrase_bank: configs/paraphrases.yaml
paraphrase_coverage: 0.15
```

- [ ] **Step 2: Demonstrate the knob end-to-end (a "toy prep run shows the expected proportions")**

Run from the repo root:

```bash
.venv/bin/python -m tinyfables run tokenizer --config configs/tokenizer_toy.yaml --out runs/tokenizer_toy
.venv/bin/python -m tinyfables run prep --config configs/prep_toy.yaml --out runs/prep_toy
.venv/bin/python -c "import json; print(json.load(open('runs/prep_toy/prep_summary.json'))['family_counts'])"
```

Expected: prints a `family_counts` dict summing to 24 with `held-out-template == 0` and a small non-zero `seen-template` count (≈2–5 at 15% × 24). Then clean up the scratch run so it is not committed:

```bash
rm -rf runs/prep_toy runs/tokenizer_toy
```

- [ ] **Step 3: Record the decisions in `docs/design.md`**

Under the **Interface** section (after the Paraphrased Prompt paragraph), append a short subsection:

```markdown
### Paraphrase augmentation (issue 03)

The bank lives at `configs/paraphrases.yaml` (versioned, human-curated): 30
templates, each slot-filling all five Elements verbatim; **5 are `held_out:
true`** (`heldout-minimal-01`, `heldout-question-02`, `heldout-json-03`,
`heldout-letter-04`, `heldout-headline-05`) and are never used in training —
they are the robustness grid's unseen-phrasing column (issues 05/10). Prep
(`stages/prep.py`) rewrites a `paraphrase_coverage` fraction of rows (10–20%
design range; the toy config uses 0.15): it parses Element values back out of
each Canonical Prompt, re-renders under a **seen** template chosen by an RNG
seeded on `(seed, row_index)` (so runs stay hash-deterministic), and writes a
per-row `families.bin` (uint8: 0 canonical / 1 seen-template / 2
held-out-template) plus `family_counts` in `prep_summary.json`. Rows that do not
match the canonical structure fall back to canonical unchanged and are counted
in `n_parse_failures`. Held-out leakage is provable from the artifacts:
`families.bin` never contains code 2 and `family_counts["held-out-template"]`
is 0 for any training prep. Verbatim preservation is a property test over every
template × spec; `str.format` never re-interprets substituted values, so
Element values with braces/colons survive exactly, keeping spec adherence
mechanically measurable.
```

- [ ] **Step 4: Run the full suite once more**

Run: `.venv/bin/pytest -q`
Expected: PASS (whole suite green).

- [ ] **Step 5: Commit**

```bash
git add configs/prep_toy.yaml docs/design.md
git commit -m "docs: enable toy paraphrase demo + record issue-03 decisions"
```

---

## Post-plan workflow (after all tasks pass, per the task brief)

1. Merge the feature branch into `main` locally (no PR): `git checkout main && git merge --no-ff <branch>`.
2. Tick issue 03 in `docs/issues/README.md` (change its `☐` to `☑` on the issue-03 row), commit: `git commit -am "chore: mark issue 03 done"`.
3. Confirm `docs/design.md` records the new decisions (done in Task 7) — no separate ADR is needed (this issue introduces no hard-to-reverse binding decision beyond what design.md now documents; the bank format and held-out set are versioned in-repo).

---

## Self-Review

**1. Spec coverage** (issue 03 acceptance criteria):
- "Template bank is a versioned, human-editable artifact in the repository" → `configs/paraphrases.yaml` with `version:` (Task 1); curated by the human before commit.
- "Property test: rendering never alters Element values, for any template × FableSpec combination" → `test_render_preserves_every_element_value_verbatim` iterates all 30 templates × adversarial specs (Task 3), backed by the loader rejecting format-specs.
- "Paraphrase coverage is a config knob; a toy prep run shows the expected proportions" → `PrepConfig.paraphrase_coverage` (Task 5); `test_summary_counts_show_expected_proportions` + the `configs/prep_toy.yaml` demo run (Task 7).
- "Held-out templates provably never appear in training artifacts (test)" → selector-level (`test_select_coverage_one_always_paraphrases_with_seen`, `test_select_never_reports_held_out_family`, Task 4) + artifact-level (`test_held_out_never_in_training_artifacts`, Task 6: `families.bin` has no code 2 and `family_counts["held-out-template"] == 0` at full coverage).
- "Prompt-family tags are present in prep output and readable by later stages" → per-row `families.bin` + `family_counts` in the summary (Task 6), read via `load_families` (`test_load_families_round_trip`).

**2. Placeholder scan:** every code step contains complete, runnable code; no TBD/TODO/"handle edge cases". The one deferred item — the bank *content* — is explicitly a human-curation gate, with a full draft provided.

**3. Type consistency:** `select_row(text, row_index, coverage, seed, bank)` signature and `RowResult(prompt, family, parse_failed)` are used identically in Task 4 (definition) and Task 6 (prep call). `FAMILIES` index↔name mapping is consistent (`0 canonical / 1 seen-template / 2 held-out-template`) across the module, `families.bin` writer, and every test. `parse_canonical_prompt -> ParsedPrompt | None` and `render_paraphrase(template, obj)` line up between Tasks 2, 3, and 4. `ELEMENT_FIELDS`/`AGE_WORD_LINE`/`CANONICAL_HEADER` are defined in Task 2's `prompts.py` edit and imported in the same task.

**Note on the Task-1→Task-2 `_ELEMENT_SLOTS` transition:** Task 1 defines it as a literal tuple; Task 2 rederives it from the imported `ELEMENT_FIELDS`. Both evaluate to `("character","setting","challenge","outcome","moral")`, so no behavior changes and Task 1's tests keep passing after Task 2.
