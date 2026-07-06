# 07 — Reward Model + gates

**Triage**: `enhancement` · `ready-for-agent`

## Parent

[TinyFables PRD](../prd.md)

## What to build

The reward model stage — Base Model plus scalar head, Bradley–Terry loss over derived
preferences (~1,800 train / 200 held-out) — its held-out accuracy report, and the committed
data-curve ablation (accuracy at 100/500/1,000/2,000 pairs: is the signal real, and where does
it saturate?). Also the gate stage: an explicit machine-readable go/no-go record combining the
labeler audits (issue 06) and RM held-out accuracy ≥ 65%, which the PPO stage will refuse to
run without. Optimizing a noisy signal is worse than stopping — the gate is the design's
central safety property.

## Acceptance criteria

- [ ] RM trains at toy scale inside the test suite; real run reports held-out accuracy
- [ ] Data-curve artifact with all four points, report-ready
- [ ] Gate stage emits an explicit pass/fail record with its inputs (audit numbers, RM accuracy)
- [ ] Reward Model pushed to the Hub

## Blocked by

- [06 — Feedback data](./06-feedback-data-labeler.md)
