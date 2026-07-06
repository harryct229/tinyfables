# 04 — Real Base Model on free Colab

**Triage**: `ready-for-agent`

## Parent

[TinyFables PRD](../prd.md)

## What to build

Scale the walking skeleton to the real thing on free-tier Colab (T4): first the benchmark
stage — measured tokens/sec that confirms or revises the 13M size before any long run (the
week-1 gate: measure, then commit) — then fp16 pretraining over the full ~250M-token shard
set (rendered with Paraphrased Prompts per issue 03), with checkpoints including optimizer
state pushed to Drive/Hub every N steps, exact resume across session deaths, and metrics
streamed to the experiment tracker. Sessions are driven from Claude Code via the Colab MCP
connection. Output: the Base Model, pushed to the Hub with its loss curve.

## Acceptance criteria

- [ ] Benchmark stage reports measured T4 tokens/sec; the size go/no-go decision is recorded
- [ ] A real session death is survived: training resumes losing at most N steps of progress
- [ ] Metrics are visible in the tracker while a Colab run is in flight
- [ ] Base Model and tokenizer are pushed to the Hub; the loss-curve artifact is saved
- [ ] Spot check (~20 varied FableSpecs): generations are fluent fables that broadly honor the requested Elements

## Blocked by

- [02 — Walking skeleton](./02-walking-skeleton-model-generate.md)
- [03 — Paraphrased Prompts](./03-paraphrased-prompts.md)
