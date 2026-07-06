# 10 — Committed ablation + report figures

**Triage**: `ready-for-agent`

## Parent

[TinyFables PRD](../prd.md)

## What to build

The paraphrase-augmentation ablation and the report's figure assembly. Train the no-aug
sibling: an identical pretraining run whose only config difference is paraphrase coverage
zero. Evaluate both models on the robustness grid — spec adherence and Moral delivery across
canonical / seen-template / held-out-template prompt families — where the held-out column is
the augmentation story: did training on paraphrases buy robustness to phrasings the model
never saw? Assemble the grid, the RM data curve (issue 07), and the headline results into
report-ready figures with run provenance. (The PPO-vs-DPO comparison and ~3M scaling sibling
remain stretch, not part of this issue.)

## Acceptance criteria

- [ ] No-aug sibling trained from a config differing only in paraphrase coverage
- [ ] Robustness grid artifact: 2 models × 3 prompt families, adherence + Moral delivery
- [ ] Figures exported report-ready, each traceable to run identifiers
- [ ] Honest outcome recorded either way — "augmentation didn't help" is a valid, reportable result

## Blocked by

- [04 — Real Base Model](./04-real-base-model-colab.md)
- [05 — Mechanical eval suite](./05-mechanical-eval-suite.md)
