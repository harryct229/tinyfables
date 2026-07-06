# 01 — Stage-contract skeleton: data prep + tokenizer

**Triage**: `enhancement` · `ready-for-agent`

## Parent

[TinyFables PRD](../prd.md)

## What to build

Establish the repository's architectural spine — the pipeline-stage contract (config in →
artifact files out, deterministic given a seed, resumable) — and prove it with the first two
real stages. Data prep samples rows from the source dataset, renders Canonical Prompts, packs
prompt+fable+EOS into context windows with prompt spans marked for loss masking, and writes
token shards plus a prep summary (counts of examples, tokens, coverage). Tokenizer training
produces the 8k byte-level BPE (end-of-text and pad specials reserved at creation, per
ADR-0002) plus a compression report comparing tokens-per-fable against the GPT-2 vocabulary.
A committed tiny fixture corpus makes both stages runnable at toy scale on CPU in minutes.
This slice is the prefactor for everything after it: each later slice should only need to add
a stage, never invent new conventions.

## Acceptance criteria

- [ ] A stage is invoked uniformly: config in → artifacts out; the same config and seed reproduce hash-stable artifacts
- [ ] Data-prep emits packed shards with prompt spans masked, plus a prep summary artifact
- [ ] Tokenizer stage trains the 8k BPE with specials reserved; round-trip encode/decode test passes
- [ ] Compression report artifact compares tokens/fable for the 8k BPE vs GPT-2's vocabulary (the ADR-0002 report figure)
- [ ] Contract tests run both stages end-to-end at toy scale on CPU in under ~2 minutes, asserting on artifact properties only
- [ ] No pipeline logic lives in notebooks

## Blocked by

None — can start immediately.
