"""Alignment gate stage for issue 07."""

from __future__ import annotations

import json
from pathlib import Path

from tinyfables.config import GateConfig
from tinyfables.stage import write_manifest


def run(cfg: GateConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = Path(cfg.audit)
    reward_path = Path(cfg.reward_summary)
    audit = json.loads(audit_path.read_text())
    reward = json.loads(reward_path.read_text())

    reasons: list[str] = []
    warnings: list[str] = []

    labeler_pass = bool(audit["gate"]["self_consistency_pass"])
    if cfg.require_labeler_self_consistency and not labeler_pass:
        reasons.append("labeler self-consistency below gate")

    rm_accuracy = float(reward["held_out_accuracy"])
    if rm_accuracy < cfg.rm_accuracy_gate:
        reasons.append("reward model held-out accuracy below gate")

    if audit["gate"].get("position_swap_review_flag", False):
        warnings.append("position-swap review flag set")

    gate = {
        "pass": not reasons,
        "reasons": reasons,
        "warnings": warnings,
        "thresholds": {
            "rm_accuracy_gate": cfg.rm_accuracy_gate,
            "labeler_self_consistency_required": cfg.require_labeler_self_consistency,
            "labeler_self_consistency_gate": audit["gate"]["self_consistency_gate"],
            "position_swap_review_threshold": audit["gate"]["position_swap_review_threshold"],
        },
        "inputs": {
            "audit": {
                "self_consistency_pass": labeler_pass,
                "self_consistency": audit["self_consistency"],
                "position_swap": audit["position_swap"],
                "position_swap_review_flag": audit["gate"].get("position_swap_review_flag", False),
            },
            "reward": {
                "held_out_accuracy": rm_accuracy,
                "n_train": reward["n_train"],
                "n_held_out": reward["n_held_out"],
                "rm_gate_pass": reward["rm_gate_pass"],
            },
        },
    }

    gate_path = out_dir / "gate.json"
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")

    verdict = "PASS - PPO may run" if gate["pass"] else "NO-GO - PPO must not run"
    report_lines = [
        "# Alignment gate report",
        "",
        f"Verdict: {verdict}",
        "",
        "| input | value |",
        "|---|---|",
        f"| RM held-out accuracy | {rm_accuracy:.3f} (gate {cfg.rm_accuracy_gate:.2f}) |",
        f"| Labeler self-consistency | {audit['self_consistency']['mean_agreement']:.3f} (gate {audit['gate']['self_consistency_gate']:.2f}) |",
        f"| Position-swap flip rate | {audit['position_swap']['flip_rate']:.3f} |",
        "",
    ]
    if reasons:
        report_lines.extend(["Reasons:", *[f"- {reason}" for reason in reasons], ""])
    if warnings:
        report_lines.extend(["Warnings:", *[f"- {warning}" for warning in warnings], ""])
    report_path = out_dir / "gate_report.md"
    report_path.write_text("\n".join(report_lines))

    write_manifest(
        out_dir,
        "gate",
        cfg,
        [gate_path, report_path],
        inputs={"audit.json": audit_path, "reward_summary.json": reward_path},
    )
