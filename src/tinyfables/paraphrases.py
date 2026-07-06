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
