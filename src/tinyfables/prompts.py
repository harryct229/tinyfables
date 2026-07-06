"""Rendering a FableSpec to the dataset's Canonical Prompt. Kept torch-free so it
can be imported anywhere. For a fully-populated spec, `render_canonical_prompt`
reproduces the dataset/fixture prompt byte-for-byte, so a model trained on that
distribution generates in-distribution. Issue 03 extends this with Paraphrased
Prompts; element values are always preserved verbatim so spec adherence stays
mechanically measurable."""

from __future__ import annotations

from dataclasses import dataclass

_HEADER = "Create a fable based on the following elements. Weave them naturally into a story:"

# (FableSpec field, rendered label). The moral is labelled "Teaching" — the
# dataset's own field name, kept verbatim in the prompt text.
_ELEMENTS = [
    ("character", "Main Character"),
    ("setting", "Setting"),
    ("challenge", "Challenge"),
    ("outcome", "Outcome"),
    ("moral", "Teaching"),
]


@dataclass(frozen=True)
class FableSpec:
    character: str | None = None
    setting: str | None = None
    challenge: str | None = None
    outcome: str | None = None
    moral: str | None = None
    age_range: str = "4-7"
    word_count: int = 60


def render_canonical_prompt(spec: FableSpec) -> str:
    lines = [_HEADER]
    for field, label in _ELEMENTS:
        value = getattr(spec, field)
        if value is not None:
            lines.append(f"- {label}: {value}")
    lines.append(
        f"Keep it age-appropriate for ages {spec.age_range} and about {spec.word_count} words."
    )
    return "\n".join(lines)
