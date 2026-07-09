"""The AI Labeler plumbing (issue 06, ADR-0004): build the batched Claude prompt,
parse its strict-JSON per-Axis ratings, and cache labels append-only. Torch-free.

The ONLY code that invokes Claude is `claude_runner` (Task 6); every test injects a
fake runner or reads a recorded fixture, so tests never call Claude live. One schema
contract test validates `parse_labeler_response` against a recorded live sample.
"""

from __future__ import annotations

import json
import subprocess
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
            f"--- Requested Fable ---\n{row['requested_prompt']}\n"
            f"--- Fable A ---\n{row['fable_a']}\n"
            f"--- Fable B ---\n{row['fable_b']}\n"
        )
    return template_text.format(rubric=rubric_text, pairs="\n".join(blocks))


def cache_key(
    pair_id: str,
    phase: str,
    order: str,
    model_version: str,
    prompt_version: int,
) -> tuple:
    return (pair_id, phase, order, model_version, prompt_version)


def load_cache(path: str | Path) -> dict[tuple, dict]:
    p = Path(path)
    if not p.exists():
        return {}

    out: dict[tuple, dict] = {}
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[
            cache_key(
                rec["pair_id"],
                rec["phase"],
                rec["order"],
                rec["model_version"],
                rec["prompt_version"],
            )
        ] = rec
    return out


def append_cache(path: str | Path, record: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(record) + "\n")


def load_label_cohort(path: str | Path) -> tuple[list[dict], dict | None]:
    """Load one coherent label cohort, using label stage metadata when present."""

    labels_path = Path(path)
    cache = load_cache(labels_path)
    records = list(cache.values())
    summary_path = labels_path.with_name("label_summary.json")
    manifest_path = labels_path.with_name("manifest.json")

    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if not summary.get("complete", False):
            raise ValueError(f"label run is incomplete per {summary_path}")
        if not manifest_path.exists():
            raise ValueError(f"complete label run is missing manifest: {manifest_path}")

        model_version = summary.get("model_version")
        prompt_version = summary.get("prompt_version")
        filtered = [
            rec
            for rec in records
            if rec.get("model_version") == model_version and rec.get("prompt_version") == prompt_version
        ]
        expected = int(summary.get("n_cached_current", len(filtered)))
        if len(filtered) != expected:
            raise ValueError(
                "labels cache does not match the complete label summary cohort: "
                f"expected {expected} records for ({model_version!r}, {prompt_version!r}), "
                f"found {len(filtered)}"
            )
        return filtered, summary

    cohorts = sorted({(rec["model_version"], rec["prompt_version"]) for rec in records})
    if len(cohorts) > 1:
        rendered = ", ".join(f"{model}@v{prompt}" for model, prompt in cohorts)
        raise ValueError(
            "labels cache mixes multiple label cohorts without label_summary.json; "
            f"found: {rendered}"
        )
    return records, None


def claude_runner(prompt: str, model: str) -> str:
    proc = subprocess.run(
        ["claude", "-p", "--model", model, "--output-format", "text"],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        if len(stderr) > 500:
            stderr = stderr[:500]
        raise LabelerError(f"claude -p failed (exit {proc.returncode}): {stderr}")
    return proc.stdout


def _extract_json_object(text: str) -> dict:
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            payload, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        if isinstance(payload, dict):
            return payload
        start = text.find("{", start + 1)
    raise LabelerError(f"no JSON object found in labeler response: {text[:200]!r}")


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

    expected = set(expected_pair_ids)
    by_id: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict) or "pair_id" not in row:
            raise LabelerError(f"malformed label row: {row!r}")
        pair_id = row["pair_id"]
        if not isinstance(pair_id, str):
            raise LabelerError(f"malformed label row pair_id: {pair_id!r}")
        if pair_id in by_id:
            raise LabelerError(f"duplicate label row for pair {pair_id!r}")
        by_id[pair_id] = row

    actual = set(by_id)
    if actual != expected:
        missing = sorted(expected - actual)
        extras = sorted(actual - expected)
        problems = []
        if missing:
            problems.append(f"missing={missing}")
        if extras:
            problems.append(f"extra={extras}")
        raise LabelerError(
            "labeler response pair ids did not match the requested batch: " + ", ".join(problems)
        )

    labels: list[PairLabel] = []
    for pid in expected_pair_ids:
        row = by_id[pid]
        justification = row.get("justification")
        if "justification" in row and justification is not None and not isinstance(justification, str):
            raise LabelerError(f"{pid}: justification must be a string or null, got {justification!r}")
        labels.append(
            PairLabel(
                pair_id=pid,
                ratings_a=_validate_ratings(row.get("fable_a"), pid, "fable_a"),
                ratings_b=_validate_ratings(row.get("fable_b"), pid, "fable_b"),
                justification=justification,
            )
        )
    return labels
