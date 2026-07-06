# 03 — Paraphrased Prompts: template bank + rendering

**Triage**: `enhancement` · `ready-for-agent`

## Parent

[TinyFables PRD](../prd.md)

## What to build

The offline paraphrase mechanism: a versioned bank of ~30 alternative instruction phrasings
(LLM-brainstormed once, human-curated), slot-filled mechanically so Element values are
preserved verbatim, applied to a configurable 10–20% of training examples at data-prep time.
Five templates are held out of training entirely, and every rendered example carries a
prompt-family tag (canonical / seen-template / held-out-template) that flows through the
shards into downstream evaluation — the held-out column is the measuring stick for the
augmentation ablation (issue 10).

## Acceptance criteria

- [ ] Template bank is a versioned, human-editable artifact in the repository
- [ ] Property test: rendering never alters Element values, for any template × FableSpec combination
- [ ] Paraphrase coverage is a config knob; a toy prep run shows the expected proportions
- [ ] Held-out templates provably never appear in training artifacts (test)
- [ ] Prompt-family tags are present in prep output and readable by later stages

## Blocked by

- [01 — Stage-contract skeleton](./01-stage-contract-skeleton.md)
