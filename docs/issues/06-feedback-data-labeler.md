# 06 — Feedback data: Rubric, pairs, AI Labeler, audits

**Triage**: `ready-for-agent`

## Parent

[TinyFables PRD](../prd.md)

## What to build

The full feedback-data path (ADR-0003 format, ADR-0004 source). First, the one artifact only
the human can produce: the anchored Rubric — per-Axis 1–5 definitions with concrete examples —
committed before any real labeling. Then three stages: pair generation (~2,000 Preference
Pairs: two independent samples at temperature ≈0.9 per held-out FableSpec); the AI Labeler —
resumable batched invocation of headless Claude, strict-JSON per-Axis ratings into an
append-only cache where every label carries model and prompt versions, with the cache doubling
as a replay fixture so tests never call Claude; and preference derivation — Aggregate Scores
(0.4/0.3/0.2/0.1), ties skipped, plus the ±0.1 weight-sensitivity table. Labeler audits are
built in: a 10% slice labeled in both A/B orders (flip rate) and a 30-pair Calibration Set
re-labeled at start/middle/end (self-consistency).

## Acceptance criteria

- [ ] Rubric committed before the real labeling run
- [ ] Pair-generation stage produces a pairs artifact from any checkpoint
- [ ] Labeler resumes mid-run from its cache; replay fixtures let the stage run offline in tests
- [ ] Schema contract test validates strict-JSON ratings against a recorded live sample
- [ ] Every label carries model version and prompt version
- [ ] Audit report artifact: position-swap flip rate and Calibration Set self-consistency
- [ ] Derivation emits preferences and the weight-sensitivity table
- [ ] Real run complete: ~2,000 pairs labeled; audit numbers within gate thresholds or explicitly flagged

## Blocked by

- [02 — Walking skeleton](./02-walking-skeleton-model-generate.md) (toy path)
- [04 — Real Base Model](./04-real-base-model-colab.md) (real labeling run)
