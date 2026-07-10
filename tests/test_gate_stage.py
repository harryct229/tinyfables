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


@pytest.mark.parametrize(
    ("gate_patch", "message"),
    [
        ({"thresholds": {"rm_accuracy_gate": 0.5}}, "reward model held-out accuracy"),
        ({"thresholds": {"labeler_self_consistency_required": False}}, "policy disables"),
        ({"pass": "true"}, "invalid pass"),
        ({"inputs": {"audit": {"self_consistency_pass": "true"}}}, "invalid labeler"),
        ({"inputs": {"reward": {"held_out_accuracy": float("nan")}}}, "invalid reward accuracy"),
        ({"inputs": {"reward": {"held_out_accuracy": 1.1}}}, "invalid reward accuracy"),
    ],
)
def test_assert_gate_passed_fails_closed_on_weakened_or_malformed_gate(tmp_path, gate_patch, message):
    gate = {
        "pass": True,
        "reasons": [],
        "thresholds": {
            "rm_accuracy_gate": 0.65,
            "labeler_self_consistency_required": True,
        },
        "inputs": {
            "audit": {"self_consistency_pass": True},
            "reward": {"held_out_accuracy": 0.72},
        },
    }
    for key, value in gate_patch.items():
        gate[key].update(value) if isinstance(value, dict) else gate.update({key: value})
    gate_path = write_json(tmp_path / "gate.json", gate)

    with pytest.raises(GateError, match=message):
        assert_gate_passed(gate_path)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rm_accuracy_gate": 0.64}, "rm_accuracy_gate"),
        ({"require_labeler_self_consistency": False}, "labeler self-consistency"),
        ({"require_labeler_self_consistency": "false"}, "must be a boolean"),
    ],
)
def test_gate_config_rejects_weakened_or_malformed_policy(kwargs, message):
    with pytest.raises(ValueError, match=message):
        GateConfig(audit="audit.json", reward_summary="reward.json", **kwargs)


@pytest.mark.parametrize(
    ("audit_patch", "reward_patch", "message"),
    [
        ({"self_consistency_pass": "false"}, {}, "self_consistency_pass"),
        ({}, {"held_out_accuracy": float("nan")}, "held_out_accuracy"),
        ({}, {"held_out_accuracy": -0.1}, "held_out_accuracy"),
    ],
)
def test_gate_stage_rejects_malformed_input_values(tmp_path, audit_patch, reward_patch, message):
    audit_data = audit()
    audit_data["gate"].update(audit_patch)
    reward_data = reward()
    reward_data.update(reward_patch)
    audit_path = write_json(tmp_path / "audit.json", audit_data)
    reward_path = write_json(tmp_path / "reward_summary.json", reward_data)

    with pytest.raises(ValueError, match=message):
        gate_stage.run(GateConfig(audit=str(audit_path), reward_summary=str(reward_path)), tmp_path / "gate")
