# Preference labels come entirely from Claude (pure RLAIF), not from human labeling

All preference labels are produced by Claude (headless `claude -p`, pinned model version, versioned prompt) rating fable pairs against the human-authored Rubric — no human labels in the training loop. The 15–20h of solo human labeling was both the schedule bottleneck and the signal bottleneck (200 pairs is thin RM training data); AI labeling produces ~2,000 pairs in an unattended afternoon at zero marginal cost. The human authors the rubric and the aggregation weights — the *definition* of good — and the AI applies it at scale.

## Consequences

- The project is **RLAIF**: the report and title must say so, framed as "the full RLHF pipeline with AI preference feedback." Professor sign-off that this satisfies the course's RLHF requirement is a week-1 action item.
- **Circularity rule**: the eval LLM-judge must be a different model family than the labeler (Gemini, never Claude) — a model optimized toward Claude's preferences and then judged by Claude would show inflated win rates by construction. The blind human eval remains the headline, and now doubles as the test of the RLAIF premise itself: does alignment to Claude's ratings transfer to human preference?
- The labeler is treated as a measurement instrument: position-swap audit (10% slice labeled in both orders) and a re-labeled Calibration Set (self-consistency) are mandatory, reported numbers.
- Fallback if literal human feedback is required: a ~60-pair human Gold Set (~5h, blind, same rubric) bolts on as a validation layer measuring human–AI agreement; nothing upstream is retrained.

## Considered Options

- Human-anchored hybrid (60-pair human Gold Set gating Claude bulk labels) — safest framing, rejected for the extra ~5h and because the gold-set validation can still be added later if needed
- Claude pre-fills + human review — anchoring bias makes "human-approved" soft while still costing 8–10h and capping data at 200 pairs
- Pure human labeling (ADR-0003 original) — 15–20h, 200 pairs; the bottleneck this decision removes
