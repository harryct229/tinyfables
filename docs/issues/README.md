# Issue Queue

Local stand-in for an issue tracker (none exists yet — if a GitHub repo is created, file these
as issues, apply `ready-for-agent`, and link back to the PRD as parent). All issues are triaged
`enhancement` · `ready-for-agent`. Parent: [PRD](../prd.md). Tick the box when a slice lands.

| # | Slice | Blocked by | Done |
|---|-------|-----------|------|
| 01 | [Stage-contract skeleton: data prep + tokenizer](./01-stage-contract-skeleton.md) | — | ☑ |
| 02 | [Walking skeleton: hand-written GPT writes its first fable](./02-walking-skeleton-model-generate.md) | 01 | ☑ |
| 03 | [Paraphrased Prompts: template bank + rendering](./03-paraphrased-prompts.md) | 01 | ☑ |
| 04 | [Real Base Model on free Colab](./04-real-base-model-colab.md) | 02, 03 | ☐ |
| 05 | [Mechanical eval suite](./05-mechanical-eval-suite.md) | 02 | ☐ |
| 06 | [Feedback data: Rubric, pairs, AI Labeler, audits](./06-feedback-data-labeler.md) | 02 (toy), 04 (real) | ☐ |
| 07 | [Reward Model + gates](./07-reward-model-gates.md) | 06 | ☐ |
| 08 | [Aligned Model: PPO with DPO fallback](./08-aligned-model-ppo.md) | 07 | ☐ |
| 09 | [Headline eval: judge + blind human A/B](./09-headline-eval.md) | 05, 08 | ☐ |
| 10 | [Committed ablation + report figures](./10-committed-ablation-figures.md) | 04, 05 | ☐ |
| 11 | [Demo Space + Hub cards](./11-demo-hub-cards.md) | 08 | ☐ |

```
01 ─┬─▶ 02 ─┬─▶ 04 ─┬─▶ 06 ─▶ 07 ─▶ 08 ─┬─▶ 09 ◀─ 05
    └─▶ 03 ─┘       ├─▶ 10 ◀─ 05        └─▶ 11
                    └────────────────────────▲
Parallel tracks while the GPU trains (04): 03, 05, 06-toy.
```
