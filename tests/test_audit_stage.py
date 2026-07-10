import json
from pathlib import Path

import pytest

from tinyfables.config import AuditConfig
from tinyfables.stages import REGISTRY
from tinyfables.stages import audit as audit_stage

LABELS = str(Path(__file__).parent / "fixtures" / "labels_replay.jsonl")


def test_audit_registered():
    assert REGISTRY["audit"] == (AuditConfig, "tinyfables.stages.audit")


def test_audit_reports_both_numbers_and_gate(tmp_path):
    out = tmp_path / "audit"
    audit_stage.run(AuditConfig(labels=LABELS), out)
    assert {"audit.json", "audit_report.md", "manifest.json"} <= {p.name for p in out.iterdir()}
    a = json.loads((out / "audit.json").read_text())
    assert a["position_swap"]["n_pairs"] == 2
    assert a["position_swap"]["flip_rate"] == 0.5
    assert a["gate"]["position_swap_review_threshold"] == 0.5
    assert a["gate"]["position_swap_review_flag"] is True
    assert a["self_consistency"]["n_pairs"] == 2
    assert 0.0 <= a["self_consistency"]["mean_agreement"] <= 1.0
    assert isinstance(a["gate"]["self_consistency_pass"], bool)
    report = (out / "audit_report.md").read_text()
    lowered = report.lower()
    assert "flip rate" in lowered and "self-consistency" in lowered
    assert "position-swap review: review (threshold 0.50)" in lowered


def test_audit_gate_flags_below_threshold(tmp_path):
    out = tmp_path / "audit"
    audit_stage.run(AuditConfig(labels=LABELS, self_consistency_gate=0.999), out)
    a = json.loads((out / "audit.json").read_text())
    assert a["gate"]["self_consistency_pass"] is False


def test_audit_rejects_missing_labels_path(tmp_path):
    out = tmp_path / "audit"
    with pytest.raises(FileNotFoundError):
        audit_stage.run(AuditConfig(labels=str(tmp_path / "missing.jsonl")), out)


def test_audit_rejects_incomplete_label_summary(tmp_path):
    labels = tmp_path / "labels.jsonl"
    labels.write_text(Path(LABELS).read_text())
    (tmp_path / "label_summary.json").write_text(json.dumps({"complete": False}))

    out = tmp_path / "audit"
    with pytest.raises(ValueError, match="incomplete"):
        audit_stage.run(AuditConfig(labels=str(labels)), out)


# --- ADR-0005 majority-vote audit -------------------------------------------

HI = {"moral": 5, "adherence": 5, "coherence": 5, "prose": 5}
LO = {"moral": 1, "adherence": 1, "coherence": 1, "prose": 1}


def _rec(pair_id, phase, order, ratings_0, ratings_1):
    return {
        "pair_id": pair_id,
        "phase": phase,
        "order": order,
        "model_version": "recorded",
        "prompt_version": 3,
        "rubric_sha": "fixturesha",
        "prompt_sha": "fixturesha",
        "ratings_0": ratings_0,
        "ratings_1": ratings_1,
    }


def _write_cache(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def test_voted_audit_position_swap_flip_rate_differs_from_any_single_cache(tmp_path):
    # main is always (HI, LO) -> preference 0 in every cache/pair. The swap
    # side flips per (cache, pair): p1 F,T,T; p2 T,F,T; p3 T,T,F -- every
    # pair has a 2-of-3 majority voting swap=1, so the VOTED result flips all
    # 3 pairs even though every single cache only flips 2 of its own 3.
    def swap_ratings(flip: bool):
        return (LO, HI) if flip else (HI, LO)

    flips = {"p1": (False, True, True), "p2": (True, False, True), "p3": (True, True, False)}

    caches: list[list[dict]] = [[], [], []]
    for pair_id, per_cache_flip in flips.items():
        for i, flip in enumerate(per_cache_flip):
            caches[i].append(_rec(pair_id, "main", "ab", HI, LO))
            caches[i].append(_rec(pair_id, "swap", "ba", *swap_ratings(flip)))

    labels_paths = [tmp_path / f"labels{i}.jsonl" for i in range(3)]
    for path, records in zip(labels_paths, caches):
        _write_cache(path, records)

    out = tmp_path / "audit"
    audit_stage.run(
        AuditConfig(labels=str(labels_paths[0]), extra_labels=[str(labels_paths[1]), str(labels_paths[2])]),
        out,
    )
    a = json.loads((out / "audit.json").read_text())
    assert set(a.keys()) == {"position_swap", "self_consistency", "gate"}
    assert a["position_swap"]["n_pairs"] == 3
    assert a["position_swap"]["n_flipped"] == 3
    assert a["position_swap"]["flip_rate"] == pytest.approx(1.0)

    for path in labels_paths:
        single_out = tmp_path / f"single_{path.stem}"
        audit_stage.run(AuditConfig(labels=str(path)), single_out)
        single_a = json.loads((single_out / "audit.json").read_text())
        assert single_a["position_swap"]["flip_rate"] == pytest.approx(2 / 3)
        assert single_a["position_swap"]["flip_rate"] != a["position_swap"]["flip_rate"]


def test_voted_audit_self_consistency_differs_from_any_single_cache(tmp_path):
    # Each phase's per-cache preference dissents from the other two, but a
    # 2-of-3 majority always lands on preference 1, so the voted phase
    # sequence is unanimous [1, 1, 1] even though every single cache's own
    # phase sequence disagrees with itself internally.
    pref_by_phase = {"calib-0": (0, 1, 1), "calib-1": (1, 0, 1), "calib-2": (1, 1, 0)}

    def ratings(pref: int):
        return (HI, LO) if pref == 0 else (LO, HI)

    caches: list[list[dict]] = [[], [], []]
    for phase, per_cache_pref in pref_by_phase.items():
        for i, pref in enumerate(per_cache_pref):
            r0, r1 = ratings(pref)
            caches[i].append(_rec("c1", phase, "ab", r0, r1))

    labels_paths = [tmp_path / f"labels{i}.jsonl" for i in range(3)]
    for path, records in zip(labels_paths, caches):
        _write_cache(path, records)

    out = tmp_path / "audit"
    audit_stage.run(
        AuditConfig(labels=str(labels_paths[0]), extra_labels=[str(labels_paths[1]), str(labels_paths[2])]),
        out,
    )
    a = json.loads((out / "audit.json").read_text())
    assert a["self_consistency"]["n_pairs"] == 1
    assert a["self_consistency"]["n_unanimous"] == 1
    assert a["self_consistency"]["mean_agreement"] == pytest.approx(1.0)

    for path in labels_paths:
        single_out = tmp_path / f"single_{path.stem}"
        audit_stage.run(AuditConfig(labels=str(path)), single_out)
        single_a = json.loads((single_out / "audit.json").read_text())
        assert single_a["self_consistency"]["mean_agreement"] == pytest.approx(1 / 3)
        assert single_a["self_consistency"]["n_unanimous"] == 0


def test_voted_audit_manifest_lists_extra_cache_inputs(tmp_path):
    labels_paths = [tmp_path / f"labels{i}.jsonl" for i in range(3)]
    for path in labels_paths:
        _write_cache(path, [_rec("p1", "main", "ab", HI, LO)])

    out = tmp_path / "audit"
    audit_stage.run(
        AuditConfig(labels=str(labels_paths[0]), extra_labels=[str(labels_paths[1]), str(labels_paths[2])]),
        out,
    )
    manifest = json.loads((out / "manifest.json").read_text())
    assert set(manifest["inputs"]) == {"labels.jsonl", "labels_extra_0.jsonl", "labels_extra_1.jsonl"}
