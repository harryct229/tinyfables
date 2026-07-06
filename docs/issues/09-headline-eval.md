# 09 — Headline eval: judge + blind human A/B

**Triage**: `ready-for-agent`

## Parent

[TinyFables PRD](../prd.md)

## What to build

The two evaluations that decide the project's headline claim. The LLM judge: a different model
family from the AI Labeler (per ADR-0004's circularity rule — the labeler's family must never
grade a model optimized toward its own taste), grading ~100 Base-vs-Aligned generations on
grammar / creativity / consistency / Moral with versioned prompts and cached responses. The
blind human A/B tool: ~50 held-out FableSpecs, randomized sides, fast keyboard flow, verdicts
to an append-only record, model identities revealed only after the session. The result
artifact combines human win rate (the money chart), judge scores, and their agreement — under
RLAIF the human eval doubles as the test of the premise itself: does alignment to Claude's
ratings transfer to a human's preference?

## Acceptance criteria

- [ ] Judge responses cached; a rerun against the cache is deterministic and free
- [ ] Config-level test enforces judge family ≠ labeler family
- [ ] Blind tool cannot reveal model identity until the session ends
- [ ] Headline artifact: human win rate, judge scores, human–judge agreement, report-ready

## Blocked by

- [05 — Mechanical eval suite](./05-mechanical-eval-suite.md)
- [08 — Aligned Model](./08-aligned-model-ppo.md)
