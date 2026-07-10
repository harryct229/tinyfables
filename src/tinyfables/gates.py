"""Machine-readable alignment gate helpers.

Future PPO code must call assert_gate_passed before optimizing against a reward
model. This keeps "do not optimize noisy/self-inconsistent labels" executable.
"""

from __future__ import annotations

import json
from pathlib import Path


class GateError(RuntimeError):
    pass


def load_gate(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def assert_gate_passed(path: str | Path) -> dict:
    gate = load_gate(path)
    if not gate.get("pass", False):
        reasons = ", ".join(gate.get("reasons", [])) or "unknown reason"
        raise GateError(f"alignment gate failed: {reasons}")
    return gate
