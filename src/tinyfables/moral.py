"""Moral extraction and matching for the eval suite (torch-free).

`extract_moral` recovers the fable's own stated moral without peeking at the
requested moral (so moral delivery is not circular): it prefers the last bold
segment, but falls back to the trailing sentence because the Base Model rarely
emits the dataset's markdown marker.

`naive_regex_moral` is the keyword-regex baseline the design flags as missing
morals; our extraction must beat it during Track B hand validation.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_NAIVE_RE = re.compile(
    r"(?:the\s+)?(?:moral|lesson|teaching)(?:\s+of\s+the\s+story)?\s*(?:is\s*:|is|:)\s*(.+)",
    re.IGNORECASE,
)


def extract_moral(fable: str) -> str | None:
    bolds = _BOLD_RE.findall(fable)
    if bolds:
        return bolds[-1].strip().strip(".")
    sentences = [s.strip() for s in _SENT_SPLIT.split(fable.strip()) if s.strip()]
    return sentences[-1].strip().strip(".") if sentences else None


def naive_regex_moral(fable: str) -> str | None:
    found = None
    for line in fable.splitlines():
        m = _NAIVE_RE.search(line)
        if m:
            found = m.group(1).strip().strip(".")
    return found


def moral_similarity(a: str | None, b: str | None) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def moral_delivered(fable: str, requested: str, threshold: float) -> bool:
    return moral_similarity(extract_moral(fable), requested) >= threshold


def extraction_precision(labels: list[bool]) -> float:
    return sum(labels) / len(labels) if labels else 0.0
