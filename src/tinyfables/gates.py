"""Machine-readable alignment gate helpers.

Future PPO code must call assert_gate_passed before optimizing against a reward
model. This keeps "do not optimize noisy/self-inconsistent labels" executable.
"""

from __future__ import annotations

import json
import math
from numbers import Real
from pathlib import Path


class GateError(RuntimeError):
    pass


def load_gate(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def _require_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise GateError(f"alignment gate has invalid {name}")
    return value


def _require_accuracy(value: object) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise GateError("alignment gate has invalid reward accuracy")
    accuracy = float(value)
    if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
        raise GateError("alignment gate has invalid reward accuracy")
    return accuracy


def assert_gate_passed(path: str | Path) -> dict:
    gate = load_gate(path)
    if not isinstance(gate, dict):
        raise GateError("alignment gate has invalid structure")

    if not _require_bool(gate.get("pass"), "pass"):
        reasons = ", ".join(gate.get("reasons", [])) or "unknown reason"
        raise GateError(f"alignment gate failed: {reasons}")

    thresholds = gate.get("thresholds")
    inputs = gate.get("inputs")
    if not isinstance(thresholds, dict) or not isinstance(inputs, dict):
        raise GateError("alignment gate has invalid structure")
    configured_accuracy_gate = _require_accuracy(thresholds.get("rm_accuracy_gate"))
    if configured_accuracy_gate < 0.65:
        raise GateError("alignment gate policy weakens the reward model held-out accuracy gate")
    if not _require_bool(
        thresholds.get("labeler_self_consistency_required"),
        "labeler self-consistency requirement",
    ):
        raise GateError("alignment gate policy disables labeler self-consistency")

    audit = inputs.get("audit")
    reward = inputs.get("reward")
    if not isinstance(audit, dict) or not isinstance(reward, dict):
        raise GateError("alignment gate has invalid structure")
    if not _require_bool(audit.get("self_consistency_pass"), "labeler self-consistency result"):
        raise GateError("alignment gate failed: labeler self-consistency below gate")

    accuracy = _require_accuracy(reward.get("held_out_accuracy"))
    if accuracy < 0.65:
        raise GateError("alignment gate failed: reward model held-out accuracy below production gate")
    return gate
