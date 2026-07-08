"""The AI Labeler plumbing (issue 06, ADR-0004): build the batched Claude prompt,
parse its strict-JSON per-Axis ratings, and cache labels append-only. Torch-free.

The ONLY code that invokes Claude is `claude_runner` (Task 6); every test injects a
fake runner or reads a recorded fixture, so tests never call Claude live. One schema
contract test validates `parse_labeler_response` against a recorded live sample.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from tinyfables.feedback import AXES


class LabelerError(ValueError):
    """A labeler response did not match the strict-JSON per-Axis 1-5 contract."""


@dataclass(frozen=True)
class PairLabel:
    pair_id: str
    ratings_a: dict[str, int]
    ratings_b: dict[str, int]
    justification: str | None = None


def load_prompt(path: str | Path) -> tuple[int, str]:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict) or "version" not in data or "template" not in data:
        raise ValueError("labeler prompt must be a mapping with 'version' and 'template'")
    return int(data["version"]), str(data["template"])


def build_batch_prompt(rubric_text: str, template_text: str, batch: list[dict]) -> str:
    blocks = []
    for i, row in enumerate(batch, 1):
        blocks.append(
            f"[{i}] pair_id: {row['pair_id']}\n"
            f"--- Fable A ---\n{row['fable_a']}\n"
            f"--- Fable B ---\n{row['fable_b']}\n"
        )
    return template_text.format(rubric=rubric_text, pairs="\n".join(blocks))


def _extract_json_object(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise LabelerError(f"no JSON object found in labeler response: {text[:200]!r}")
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        raise LabelerError(f"labeler response is not valid JSON: {e}") from e
    if not isinstance(payload, dict):
        raise LabelerError("labeler response JSON must be an object")
    return payload


def _validate_ratings(obj, pair_id: str, side: str) -> dict[str, int]:
    if not isinstance(obj, dict):
        raise LabelerError(f"{pair_id} {side}: ratings must be an object, got {type(obj).__name__}")
    out: dict[str, int] = {}
    for axis in AXES:
        if axis not in obj:
            raise LabelerError(f"{pair_id} {side}: missing axis {axis!r}")
        v = obj[axis]
        if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 5:
            raise LabelerError(f"{pair_id} {side}: axis {axis!r} must be int 1-5, got {v!r}")
        out[axis] = v
    return out


def parse_labeler_response(text: str, expected_pair_ids: list[str]) -> list[PairLabel]:
    payload = _extract_json_object(text)
    rows = payload.get("labels")
    if not isinstance(rows, list):
        raise LabelerError("labeler response missing a 'labels' list")

    by_id: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict) or "pair_id" not in row:
            raise LabelerError(f"malformed label row: {row!r}")
        pair_id = row["pair_id"]
        if not isinstance(pair_id, str):
            raise LabelerError(f"malformed label row pair_id: {pair_id!r}")
        by_id[pair_id] = row

    labels: list[PairLabel] = []
    for pid in expected_pair_ids:
        if pid not in by_id:
            raise LabelerError(f"labeler response missing expected pair {pid!r}")
        row = by_id[pid]
        labels.append(
            PairLabel(
                pair_id=pid,
                ratings_a=_validate_ratings(row.get("fable_a"), pid, "fable_a"),
                ratings_b=_validate_ratings(row.get("fable_b"), pid, "fable_b"),
                justification=row.get("justification") if isinstance(row.get("justification"), str) else None,
            )
        )
    return labels
