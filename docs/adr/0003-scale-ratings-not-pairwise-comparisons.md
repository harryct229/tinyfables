# Feedback as per-Axis 1–5 ratings with an aggregation rule, not pairwise comparisons

**Status: amended by ADR-0004** — the labeler is now an AI (pure RLAIF). The Axes, scale format, aggregation weights, and mitigations carry over, reinterpreted for an AI labeler (drift controls become self-consistency and position-bias audits). The labeling-hours trade-off below no longer binds.

Each fable in a Preference Pair is rated on four Axes (Moral delivery, spec adherence, coherence, prose & age-fit); the preference is derived from a fixed Aggregate Score (weights 0.4/0.3/0.2/0.1, Moral delivery highest) and the reward model trains on derived preferences with Bradley–Terry loss. This deliberately deviates from InstructGPT-style direct comparisons — which are more reliable for a single labeler and ~2× faster — because absolute per-axis score distributions and per-axis analysis are report deliverables in their own right. The ~15–20h labeling investment makes this hard to reverse once begun.

## Consequences

Mitigations are part of the design, not optional: an anchored Rubric written before labeling (each scale point defined with examples), side-by-side rating within each pair (preferences depend on within-pair score differences, recovering much of comparison reliability), a Calibration Set re-rated every session (drift becomes a measured number), and a weight-sensitivity table showing how many derived preferences flip under ±0.1 weight perturbation. Fallback floor: 150 pairs if labeling overruns. If someone proposes "just do A/B comparisons" mid-project: that was considered and rejected for the reasons above — switching formats mid-labeling would waste all labels collected so far.
