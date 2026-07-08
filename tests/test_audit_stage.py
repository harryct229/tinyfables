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
