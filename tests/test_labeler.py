import json
import subprocess
from pathlib import Path

import pytest
import yaml

from tinyfables.labeler import (
    append_cache,
    cache_key,
    LabelerError,
    PairLabel,
    claude_runner,
    build_batch_prompt,
    load_cache,
    load_label_cohort,
    load_prompt,
    parse_labeler_response,
)

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "tests" / "fixtures" / "labeler_response_sample.json"


def test_rubric_exists_and_anchors_all_four_axes():
    text = (REPO / "RUBRIC.md").read_text()
    for axis in ("Moral delivery", "Spec adherence", "Coherence", "Prose"):
        assert axis in text, f"RUBRIC.md missing axis: {axis}"
    # every 1-5 anchor is present (each axis defines all five scale points)
    for point in ("1", "2", "3", "4", "5"):
        assert f"- {point}:" in text or f"**{point}**" in text
    # the fixed weights appear so a labeler/report reader sees them
    for w in ("0.4", "0.3", "0.2", "0.1"):
        assert w in text


def test_labeler_prompt_is_versioned_and_slotted():
    data = yaml.safe_load((REPO / "configs" / "labeler_prompt.yaml").read_text())
    assert isinstance(data["version"], int) and data["version"] >= 1
    tmpl = data["template"]
    assert "{rubric}" in tmpl and "{pairs}" in tmpl
    # instructs strict JSON with the four axes named
    for axis in ("moral", "adherence", "coherence", "prose"):
        assert axis in tmpl
    # literal JSON braces are escaped for str.format (no stray single braces
    # besides the two named slots) -> format with dummy values must not raise
    tmpl.format(rubric="R", pairs="P")


def test_load_prompt_returns_version_and_template():
    version, template = load_prompt(REPO / "configs" / "labeler_prompt.yaml")
    assert version == 2
    assert "{rubric}" not in build_batch_prompt(
        "RUBRIC",
        template,
        [
            {
                "pair_id": "pair-0",
                "requested_prompt": "REQUEST",
                "fable_a": "A",
                "fable_b": "B",
            }
        ],
    )


def test_build_batch_prompt_embeds_rubric_and_every_pair():
    _, template = load_prompt(REPO / "configs" / "labeler_prompt.yaml")
    batch = [
        {
            "pair_id": "pair-000001",
            "requested_prompt": "Write about a fox",
            "fable_a": "the fox ran",
            "fable_b": "the owl slept",
        },
        {
            "pair_id": "pair-000002",
            "requested_prompt": "Write about an owl",
            "fable_a": "a {brace} value",
            "fable_b": "b",
        },
    ]
    prompt = build_batch_prompt("MY RUBRIC BODY", template, batch)
    assert "MY RUBRIC BODY" in prompt
    for row in batch:
        assert row["pair_id"] in prompt
        assert row["requested_prompt"] in prompt
        assert row["fable_a"] in prompt and row["fable_b"] in prompt
    assert "a {brace} value" in prompt


def test_parse_valid_response_returns_labels_in_expected_order():
    payload = {
        "labels": [
            {
                "pair_id": "pair-000001",
                "fable_a": {"moral": 4, "adherence": 3, "coherence": 4, "prose": 3},
                "fable_b": {"moral": 2, "adherence": 2, "coherence": 3, "prose": 3},
                "justification": "A is closer to the requested lesson.",
            },
            {
                "pair_id": "pair-000002",
                "fable_a": {"moral": 3, "adherence": 4, "coherence": 4, "prose": 4},
                "fable_b": {"moral": 5, "adherence": 4, "coherence": 5, "prose": 4},
            },
        ]
    }
    text = json.dumps(payload)
    labels = parse_labeler_response(text, ["pair-000001", "pair-000002"])
    assert [l.pair_id for l in labels] == ["pair-000001", "pair-000002"]
    for label in labels:
        assert isinstance(label, PairLabel)
        for side in (label.ratings_a, label.ratings_b):
            assert set(side) == {"moral", "adherence", "coherence", "prose"}
            assert all(isinstance(v, int) and 1 <= v <= 5 for v in side.values())
    assert labels[1].justification is None


def test_parse_extracts_json_from_surrounding_prose():
    body = json.dumps(
        {
            "labels": [
                {
                    "pair_id": "p0",
                    "fable_a": {"moral": 4, "adherence": 3, "coherence": 4, "prose": 3},
                    "fable_b": {"moral": 2, "adherence": 2, "coherence": 3, "prose": 3},
                }
            ]
        }
    )
    text = "Intro with braces {not json}\n```json\n" + body + "\n```\nDone."
    labels = parse_labeler_response(text, ["p0"])
    assert labels[0].ratings_a["moral"] == 4


def test_parse_rejects_missing_pair():
    body = json.dumps(
        {
            "labels": [
                {
                    "pair_id": "p0",
                    "fable_a": {"moral": 4, "adherence": 3, "coherence": 4, "prose": 3},
                    "fable_b": {"moral": 2, "adherence": 2, "coherence": 3, "prose": 3},
                }
            ]
        }
    )
    with pytest.raises(LabelerError):
        parse_labeler_response(body, ["p0", "p1"])


def test_parse_rejects_duplicate_pair_id():
    body = json.dumps(
        {
            "labels": [
                {
                    "pair_id": "p0",
                    "fable_a": {"moral": 4, "adherence": 3, "coherence": 4, "prose": 3},
                    "fable_b": {"moral": 2, "adherence": 2, "coherence": 3, "prose": 3},
                },
                {
                    "pair_id": "p0",
                    "fable_a": {"moral": 5, "adherence": 5, "coherence": 5, "prose": 5},
                    "fable_b": {"moral": 1, "adherence": 1, "coherence": 1, "prose": 1},
                },
            ]
        }
    )
    with pytest.raises(LabelerError, match="duplicate"):
        parse_labeler_response(body, ["p0"])


def test_parse_rejects_extra_pair():
    body = json.dumps(
        {
            "labels": [
                {
                    "pair_id": "p0",
                    "fable_a": {"moral": 4, "adherence": 3, "coherence": 4, "prose": 3},
                    "fable_b": {"moral": 2, "adherence": 2, "coherence": 3, "prose": 3},
                },
                {
                    "pair_id": "p1",
                    "fable_a": {"moral": 4, "adherence": 3, "coherence": 4, "prose": 3},
                    "fable_b": {"moral": 2, "adherence": 2, "coherence": 3, "prose": 3},
                },
            ]
        }
    )
    with pytest.raises(LabelerError, match="extra"):
        parse_labeler_response(body, ["p0"])


def test_parse_rejects_out_of_range_and_non_int():
    for bad in ("6", "0", "3.5", '"4"'):
        body = (
            '{"labels":[{"pair_id":"p0",'
            f'"fable_a":{{"moral":{bad},"adherence":3,"coherence":4,"prose":3}},'
            '"fable_b":{"moral":2,"adherence":2,"coherence":3,"prose":3}}]}'
        )
        with pytest.raises(LabelerError):
            parse_labeler_response(body, ["p0"])


def test_parse_rejects_missing_axis():
    body = json.dumps(
        {
            "labels": [
                {
                    "pair_id": "p0",
                    "fable_a": {"moral": 4, "adherence": 3, "coherence": 4},
                    "fable_b": {"moral": 2, "adherence": 2, "coherence": 3, "prose": 3},
                }
            ]
        }
    )
    with pytest.raises(LabelerError):
        parse_labeler_response(body, ["p0"])


def test_parse_rejects_non_string_justification_but_allows_null():
    body = json.dumps(
        {
            "labels": [
                {
                    "pair_id": "p0",
                    "fable_a": {"moral": 4, "adherence": 3, "coherence": 4, "prose": 3},
                    "fable_b": {"moral": 2, "adherence": 2, "coherence": 3, "prose": 3},
                    "justification": None,
                }
            ]
        }
    )
    labels = parse_labeler_response(body, ["p0"])
    assert labels[0].justification is None

    bad = json.dumps(
        {
            "labels": [
                {
                    "pair_id": "p0",
                    "fable_a": {"moral": 4, "adherence": 3, "coherence": 4, "prose": 3},
                    "fable_b": {"moral": 2, "adherence": 2, "coherence": 3, "prose": 3},
                    "justification": 123,
                }
            ]
        }
    )
    with pytest.raises(LabelerError):
        parse_labeler_response(bad, ["p0"])


def test_recorded_sample_is_the_schema_contract():
    # ONE schema contract test against a recorded live sample (ADR-0004). Track B
    # replaces this fixture with a verbatim real `claude -p` capture; this test must
    # stay green, which is exactly what pins our schema to Claude's real output.
    text = FIX.read_text()
    payload = json.loads(text)
    assert "labels" in payload and payload["labels"]
    ids = [row["pair_id"] for row in payload["labels"]]
    parse_labeler_response(text, ids)


def test_cache_key_is_a_stable_tuple():
    k = cache_key("pair-0", "main", "ab", "claude-opus-4-8", 1)
    assert k == ("pair-0", "main", "ab", "claude-opus-4-8", 1)


def test_append_then_load_roundtrips_by_key(tmp_path):
    path = tmp_path / "labels.jsonl"
    rec = {
        "pair_id": "pair-0",
        "phase": "main",
        "order": "ab",
        "model_version": "claude-opus-4-8",
        "prompt_version": 1,
        "ratings_0": {"moral": 5, "adherence": 4, "coherence": 4, "prose": 3},
        "ratings_1": {"moral": 2, "adherence": 3, "coherence": 3, "prose": 3},
    }
    append_cache(path, rec)
    cache = load_cache(path)
    key = cache_key("pair-0", "main", "ab", "claude-opus-4-8", 1)
    assert key in cache
    assert cache[key]["ratings_0"]["moral"] == 5


def test_load_cache_missing_file_is_empty(tmp_path):
    assert load_cache(tmp_path / "nope.jsonl") == {}


def test_claude_runner_raises_labeler_error_on_nonzero_exit(monkeypatch):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=2, stdout="", stderr="boom")

    monkeypatch.setattr("tinyfables.labeler.subprocess.run", fake_run)

    with pytest.raises(LabelerError, match="claude -p failed"):
        claude_runner("prompt", "fake-model")


def test_load_label_cohort_rejects_mixed_cache_without_summary(tmp_path):
    labels = tmp_path / "labels.jsonl"
    append_cache(
        labels,
        {
            "pair_id": "p0",
            "phase": "main",
            "order": "ab",
            "model_version": "m0",
            "prompt_version": 1,
            "ratings_0": {"moral": 5, "adherence": 5, "coherence": 5, "prose": 5},
            "ratings_1": {"moral": 1, "adherence": 1, "coherence": 1, "prose": 1},
        },
    )
    append_cache(
        labels,
        {
            "pair_id": "p1",
            "phase": "main",
            "order": "ab",
            "model_version": "m1",
            "prompt_version": 2,
            "ratings_0": {"moral": 5, "adherence": 5, "coherence": 5, "prose": 5},
            "ratings_1": {"moral": 1, "adherence": 1, "coherence": 1, "prose": 1},
        },
    )

    with pytest.raises(ValueError, match="multiple label cohorts"):
        load_label_cohort(labels)
