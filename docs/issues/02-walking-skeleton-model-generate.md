# 02 — Walking skeleton: hand-written GPT writes its first fable

**Triage**: `enhancement` · `ready-for-agent`

## Parent

[TinyFables PRD](../prd.md)

## What to build

The hand-written decoder-only GPT (design-record config: 6 layers, d_model 384, 6 heads,
context 1024, pre-norm, GELU, learned positions, tied embeddings) wrapped in the Hugging Face
model interface, a toy pretraining stage over the slice-01 shards, and the generation contract:
one entrypoint that accepts either a FableSpec (rendered to a Canonical Prompt) or raw
instruction text and returns fable text. This completes the walking skeleton — after this
slice, the system generates (terrible) fables end-to-end on CPU/MPS, and every later slice
improves an existing path instead of opening a new one. Includes the model-forward contract
(seam 2) and the training-loop trust tests.

## Acceptance criteria

- [ ] Causality test: perturbing future tokens does not change current-position logits
- [ ] fp16 stability test: forward pass shape/dtype-stable in half precision
- [ ] Loss-mask test: masked prompt tokens contribute exactly zero loss
- [ ] Overfit-one-batch drives loss near zero within minutes on CPU/MPS
- [ ] Resume-equality: training N steps matches train-k/resume/train-(N−k) within tolerance
- [ ] Generation contract accepts both a FableSpec and free instruction text
- [ ] The toy chain (prep → tokenizer → pretrain → generate) runs as a single test
- [ ] A documented command produces a fable from a toy-trained checkpoint (the demo)

## Blocked by

- [01 — Stage-contract skeleton](./01-stage-contract-skeleton.md)
