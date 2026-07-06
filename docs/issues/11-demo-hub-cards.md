# 11 — Demo Space + Hub cards

**Triage**: `ready-for-agent`

## Parent

[TinyFables PRD](../prd.md)

## What to build

The public face. A Gradio app on a free CPU Space: FableSpec form (Elements + Age Group) and a
free-text instruction box feeding the same generation contract, producing side-by-side Base
Model and Aligned Model outputs so the effect of the feedback stage is visible in ten seconds.
And the reproducibility surface: all five Hub repos (tokenizer, Base, Reward Model, Aligned,
preferences) public and carded — the preference set carded explicitly as RLAIF, linking the
Rubric and the labeler prompt version — so every claim in the report resolves to a clickable
artifact.

## Acceptance criteria

- [ ] Space runs on the free CPU tier with acceptable generation latency for a 13M model
- [ ] Both input modes work; the two outputs are clearly labeled Base vs Aligned
- [ ] Age Group control changes the rendered prompt and visibly affects output register
- [ ] Five Hub artifacts public with cards; the preference card states RLAIF and links Rubric + labeler prompt version
- [ ] Demo and artifact links recorded for the report and slides

## Blocked by

- [08 — Aligned Model](./08-aligned-model-ppo.md)
