"""Rendering a FableSpec to the dataset's Canonical Prompt. Kept torch-free so it
can be imported anywhere. `render_canonical_prompt` reproduces the actual
klusai/ds-tf1-en-3m prompt byte-for-byte for a fully-populated spec, so a model
trained on that distribution generates in-distribution.

Dataset fact (issue 04, verified over 11k rows): ds-tf1-en-3m is **single-band** —
every prompt is age group B (4-7 years), "around 250 words", with an identical
"The fable should:" style block. So the canonical template is fixed except for the
five Element values and the word count; `render_canonical_prompt` reproduces that
fixed template. `age_range` stays on FableSpec (paraphrase templates may slot it
and the parser recovers it), but the canonical render always emits the band-B block.

Issue 03 extends this with Paraphrased Prompts; Element values are always preserved
verbatim so spec adherence stays mechanically measurable."""

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

# The "The fable should:" style block, verbatim from ds-tf1-en-3m (band B). The
# whole block is constant in the dataset; the age band lives in its first bullet.
_FABLE_SHOULD_HEADER = "The fable should:"
_STYLE_BULLETS = (
    "  - Be appropriate for age group B (4-7 years)",
    "  - Use simple vocabulary that 4-7 year olds can understand",
    "  - Use concrete rather than abstract language",
    "  - Begin with vivid scene-setting",
    "  - Not use names for the characters, instead use the trait and character",
    "  - Include meaningful but simple dialogue",
    "  - Show (don't tell) the character's growth",
    "  - End with a clear connection to the moral",
)
_WORD_LINE = "Keep the story concise but engaging, around {word_count} words."

# The single age band ds-tf1-en-3m uses; the parser recovers this from the block.
CANONICAL_AGE_RANGE = "4-7"

# Public views for parsers/renderers (paraphrases.py) so the parser cannot drift
# from this module's canonical template. Guard tests pin them to render output.
CANONICAL_HEADER = _HEADER
ELEMENT_FIELDS = tuple(_ELEMENTS)
FABLE_SHOULD_HEADER = _FABLE_SHOULD_HEADER
STYLE_BULLETS = _STYLE_BULLETS
WORD_LINE = _WORD_LINE


@dataclass(frozen=True)
class FableSpec:
    character: str | None = None
    setting: str | None = None
    challenge: str | None = None
    outcome: str | None = None
    moral: str | None = None
    age_range: str = CANONICAL_AGE_RANGE
    word_count: int = 250


def render_canonical_prompt(spec: FableSpec) -> str:
    # ds-tf1-en-3m is single-band, so the Base Model only ever saw band B. Honor
    # that instead of silently mislabelling: a non-canonical age_range is a loud
    # error, not a prompt whose age bullet quietly disagrees with the request.
    if spec.age_range != CANONICAL_AGE_RANGE:
        raise ValueError(
            f"ds-tf1-en-3m is single-band (age group B, {CANONICAL_AGE_RANGE}); the Base "
            f"Model was trained only on that band, so render_canonical_prompt does not "
            f"support age_range={spec.age_range!r} (use {CANONICAL_AGE_RANGE!r})."
        )
    lines = [_HEADER]
    for field, label in _ELEMENTS:
        value = getattr(spec, field)
        if value is not None:
            lines.append(f"  - {label}: {value}")
    lines.append(_FABLE_SHOULD_HEADER)
    lines.extend(_STYLE_BULLETS)
    lines.append(_WORD_LINE.format(word_count=spec.word_count))
    return "\n".join(lines)
