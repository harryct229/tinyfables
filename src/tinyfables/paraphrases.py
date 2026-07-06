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
import re
import string
from dataclasses import dataclass
from pathlib import Path

import yaml

from tinyfables.prompts import (
    CANONICAL_AGE_RANGE,
    CANONICAL_HEADER,
    ELEMENT_FIELDS,
    FABLE_SHOULD_HEADER,
    STYLE_BULLETS,
    WORD_LINE,
)

# Prompt-family tags. The list index is the uint8 code written to families.bin.
CANONICAL = "canonical"
SEEN_TEMPLATE = "seen-template"
HELD_OUT_TEMPLATE = "held-out-template"
FAMILIES = [CANONICAL, SEEN_TEMPLATE, HELD_OUT_TEMPLATE]

# The five story Elements every template must slot in verbatim. Derived from
# prompts.ELEMENT_FIELDS so it cannot drift from the renderer.
_ELEMENT_SLOTS = tuple(field for field, _label in ELEMENT_FIELDS)
_OPTIONAL_SLOTS = ("age_range", "word_count")
_ALLOWED_SLOTS = frozenset(_ELEMENT_SLOTS) | frozenset(_OPTIONAL_SLOTS)

# Derived from prompts.WORD_LINE so it cannot drift from the renderer; a guard
# test pins the two together. The age band is not regex-matched — the style block
# is required verbatim (see parse_canonical_prompt), so the band is exactly the
# single one the dataset uses and age_range is prompts.CANONICAL_AGE_RANGE.
_WORD_LINE_RE = re.compile(
    "^" + re.escape(WORD_LINE).replace(r"\{word_count\}", r"(?P<word_count>\d+)") + "$"
)


@dataclass(frozen=True)
class Template:
    id: str
    held_out: bool
    text: str


@dataclass(frozen=True)
class ParsedPrompt:
    character: str | None = None
    setting: str | None = None
    challenge: str | None = None
    outcome: str | None = None
    moral: str | None = None
    age_range: str = "4-7"
    word_count: int = 250


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
        if not isinstance(entry["held_out"], bool):
            raise ValueError(f"template {tid} field 'held_out' must be a bool, got {entry['held_out']!r}")
        templates.append(Template(id=tid, held_out=entry["held_out"], text=text))
    if not any(not t.held_out for t in templates):
        raise ValueError("bank has no seen (training) templates")
    return ParaphraseBank(version=int(data["version"]), templates=tuple(templates))


def parse_canonical_prompt(text: str) -> ParsedPrompt | None:
    """Recover Element values from a rendered Canonical Prompt, or None if `text`
    is not EXACTLY the canonical structure. ds-tf1-en-3m is single-band, so the
    template is fixed except for the five Element values and the word count: we
    require the header, five two-space Element bullets in order, the verbatim
    "The fable should:" style block, then the word line — nothing more, nothing
    less. A row that deviates (extra trailing lines, a mangled/duplicated block,
    a different band) returns None, so it is counted as a parse failure and left
    canonical rather than silently mis-parsed. Element values are verbatim
    (everything after the "  - {Label}: " prefix, so a colon in a value survives)."""
    lines = text.split("\n")
    if len(lines) != 1 + len(ELEMENT_FIELDS) + 1 + len(STYLE_BULLETS) + 1:
        return None
    if lines[0] != CANONICAL_HEADER:
        return None
    values: dict[str, object] = {}
    i = 1
    for field, label in ELEMENT_FIELDS:
        prefix = f"  - {label}: "
        if not lines[i].startswith(prefix):
            return None
        values[field] = lines[i][len(prefix):]
        i += 1
    if lines[i] != FABLE_SHOULD_HEADER:
        return None
    i += 1
    if tuple(lines[i:i + len(STYLE_BULLETS)]) != STYLE_BULLETS:
        return None
    i += len(STYLE_BULLETS)
    word = _WORD_LINE_RE.match(lines[i])
    if word is None:
        return None
    # The style block is verified verbatim above, so the band — hence age_range —
    # is exactly the single band the dataset uses.
    return ParsedPrompt(age_range=CANONICAL_AGE_RANGE, word_count=int(word["word_count"]), **values)


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
    """Read families.bin as a uint8 array (a read-only view over the file bytes); FAMILIES[code] gives the tag name."""
    import numpy as np

    return np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
