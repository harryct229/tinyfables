"""Labeler audit stage.

Computes the position-swap flip rate and calibration self-consistency over the
cached labels, then writes the audit JSON, a short markdown report, and the
manifest last. Torch-free.

ADR-0005 adds an optional majority-vote mode (`extra_labels`): when set, the
stage loads the primary cache plus additional independently labeled caches
and computes both audits over the *voted* preference across caches. The
`audit.json` / `audit_report.md` schema is unchanged either way -- the gate
stage does not change. `extra_labels` unset (the default) keeps the original
single-cache behavior byte-identical.
"""

from __future__ import annotations

import json
from pathlib import Path

from tinyfables.config import AuditConfig
from tinyfables.feedback import (
    position_flip_rate,
    self_consistency,
    voted_position_flip_rate,
    voted_self_consistency,
)
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

    inputs = {"labels.jsonl": labels_path}

    if cfg.extra_labels is None:
        swap = position_flip_rate(instances)
        consistency = self_consistency(instances)
    else:
        instances_by_cache = [instances]
        for i, path in enumerate(cfg.extra_labels):
            extra_path = Path(path)
            if not extra_path.exists():
                raise FileNotFoundError(f"labels cache not found: {extra_path}")
            extra_instances, _extra_summary = load_label_cohort(extra_path)
            if not extra_instances:
                raise ValueError(f"labels cache is empty: {extra_path}")
            instances_by_cache.append(extra_instances)
            inputs[f"labels_extra_{i}.jsonl"] = extra_path

        swap = voted_position_flip_rate(instances_by_cache)
        consistency = voted_self_consistency(instances_by_cache)

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
        inputs=inputs,
    )
