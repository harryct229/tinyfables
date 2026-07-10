import json

import pytest

from tinyfables.config import GateConfig
from tinyfables.gates import GateError, assert_gate_passed
from tinyfables.stages import REGISTRY
from tinyfables.stages import gate as gate_stage


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return path


def audit(pass_self=True, position_flag=False):
    return {
        "gate": {
            "self_consistency_gate": 0.85,
            "self_consistency_pass": pass_self,
            "position_swap_review_flag": position_flag,
            "position_swap_review_threshold": 0.5,
        },
        "self_consistency": {
            "mean_agreement": 0.9 if pass_self else 0.7777777777777778,
            "n_pairs": 30,
            "n_unanimous": 27 if pass_self else 21,
        },
        "position_swap": {
            "flip_rate": 0.2 if not position_flag else 0.6,
            "n_flipped": 40 if not position_flag else 120,
            "n_pairs": 200,
        },
    }


def reward(acc=0.72):
    return {
        "held_out_accuracy": acc,
        "accuracy_gate": 0.65,
        "rm_gate_pass": acc >= 0.65,
        "n_train": 1761,
        "n_held_out": 188,
    }


def test_gate_stage_registered():
    from tinyfables.config import GateConfig

    assert REGISTRY["gate"] == (GateConfig, "tinyfables.stages.gate")


def test_gate_passes_when_labeler_and_rm_pass(tmp_path):
    audit_path = write_json(tmp_path / "audit.json", audit(pass_self=True))
    reward_path = write_json(tmp_path / "reward_summary.json", reward(acc=0.72))
    out = tmp_path / "gate"

    gate_stage.run(GateConfig(audit=str(audit_path), reward_summary=str(reward_path)), out)

    gate = json.loads((out / "gate.json").read_text())
    assert gate["pass"] is True
    assert gate["reasons"] == []
    assert gate["inputs"]["reward"]["held_out_accuracy"] == 0.72
    assert (out / "gate_report.md").exists()
    assert (out / "manifest.json").exists()
    assert assert_gate_passed(out / "gate.json")["pass"] is True


def test_gate_fails_on_issue06_self_consistency_failure(tmp_path):
    audit_path = write_json(tmp_path / "audit.json", audit(pass_self=False))
    reward_path = write_json(tmp_path / "reward_summary.json", reward(acc=0.72))
    out = tmp_path / "gate"

    gate_stage.run(GateConfig(audit=str(audit_path), reward_summary=str(reward_path)), out)

    gate = json.loads((out / "gate.json").read_text())
    assert gate["pass"] is False
    assert "labeler self-consistency below gate" in gate["reasons"]
    with pytest.raises(GateError, match="alignment gate failed"):
        assert_gate_passed(out / "gate.json")


def test_gate_fails_on_rm_accuracy_below_threshold(tmp_path):
    audit_path = write_json(tmp_path / "audit.json", audit(pass_self=True))
    reward_path = write_json(tmp_path / "reward_summary.json", reward(acc=0.61))
    out = tmp_path / "gate"

    gate_stage.run(GateConfig(audit=str(audit_path), reward_summary=str(reward_path)), out)

    gate = json.loads((out / "gate.json").read_text())
    assert gate["pass"] is False
    assert "reward model held-out accuracy below gate" in gate["reasons"]


def test_position_swap_review_flag_is_warning_not_hard_fail(tmp_path):
    audit_path = write_json(tmp_path / "audit.json", audit(pass_self=True, position_flag=True))
    reward_path = write_json(tmp_path / "reward_summary.json", reward(acc=0.72))
    out = tmp_path / "gate"

    gate_stage.run(GateConfig(audit=str(audit_path), reward_summary=str(reward_path)), out)

    gate = json.loads((out / "gate.json").read_text())
    assert gate["pass"] is True
    assert "position-swap review flag set" in gate["warnings"]
