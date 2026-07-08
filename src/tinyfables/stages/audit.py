"""Labeler audit stage.

Computes the position-swap flip rate and calibration self-consistency over the
cached labels, then writes the audit JSON, a short markdown report, and the
manifest last. Torch-free.
"""

from __future__ import annotations

import json
from pathlib import Path

from tinyfables.config import AuditConfig
from tinyfables.feedback import position_flip_rate, self_consistency
from tinyfables.labeler import load_label_cohort
from tinyfables.stage import write_manifest


def run(cfg: AuditConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_path = Path(cfg.labels)
    if not labels_path.exists():
        raise FileNotFoundError(f"labels cache not found: {labels_path}")

    instances, _summary = load_label_cohort(labels_path)
    if not instances:
        raise ValueError(f"labels cache is empty: {labels_path}")

    swap = position_flip_rate(instances)
    consistency = self_consistency(instances)
    passed = consistency["mean_agreement"] >= cfg.self_consistency_gate
    review_flag = swap["n_pairs"] > 0 and swap["flip_rate"] >= cfg.position_swap_review_threshold

    audit = {
        "position_swap": swap,
        "self_consistency": consistency,
        "gate": {
            "self_consistency_gate": cfg.self_consistency_gate,
            "self_consistency_pass": passed,
            "position_swap_review_flag": review_flag,
            "position_swap_review_threshold": cfg.position_swap_review_threshold,
        },
    }

    audit_path = out_dir / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    verdict = "PASS" if passed else "BELOW GATE - flag for issue 07"
    swap_verdict = (
        f"REVIEW (threshold {cfg.position_swap_review_threshold:.2f})"
        if review_flag
        else f"no high-flip flag (threshold {cfg.position_swap_review_threshold:.2f})"
    )
    report_lines = [
        "# Labeler audit report",
        "",
        "| audit | value |",
        "|---|---|",
        f"| position-swap flip rate | {swap['flip_rate']:.3f} ({swap['n_flipped']}/{swap['n_pairs']} pairs) |",
        f"| Calibration self-consistency | {consistency['mean_agreement']:.3f} ({consistency['n_unanimous']}/{consistency['n_pairs']} unanimous) |",
        "",
        f"Verdict: {verdict} (gate {cfg.self_consistency_gate:.2f})",
        f"Position-swap review: {swap_verdict}",
        "",
    ]
    report_path = out_dir / "audit_report.md"
    report_path.write_text("\n".join(report_lines))

    write_manifest(
        out_dir,
        "audit",
        cfg,
        [audit_path, report_path],
        inputs={"labels.jsonl": Path(cfg.labels)},
    )
