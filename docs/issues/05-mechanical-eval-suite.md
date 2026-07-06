# 05 — Mechanical eval suite

**Triage**: `enhancement` · `ready-for-agent`

## Parent

[TinyFables PRD](../prd.md)

## What to build

An evaluation stage runnable against any checkpoint, producing a report-ready metrics
document: fable-token perplexity on the validation slice; spec adherence (verbatim Element
presence); Moral delivery (extract the markdown-bolded moral, fuzzy-match against the
requested Moral — the dataset audit showed keyword regexes miss ~10% of morals, so extraction
must be hand-validated on a sample and beat that); repetition/distinct-n; and length
statistics. Once prompt-family tags exist (issue 03), adherence results are split by
canonical / seen-template / held-out-template. This suite is how any two checkpoints are
compared for the rest of the project.

## Acceptance criteria

- [ ] One command evaluates any checkpoint into a metrics document (deterministic given seed and eval set)
- [ ] Moral-extraction precision hand-validated on ~50 samples and recorded; beats the naive-regex baseline
- [ ] Adherence grid splits by prompt family when tags are present
- [ ] Metrics tables export in a report-usable form with run provenance
- [ ] Suite runs at toy scale inside the test suite

## Blocked by

- [02 — Walking skeleton](./02-walking-skeleton-model-generate.md) (prompt-family grid activates after [03](./03-paraphrased-prompts.md))
