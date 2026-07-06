# TinyFables: Design Record

A ~13M-parameter GPT trained **from random weights** on 3M synthetic moral fables
([klusai/ds-tf1-en-3m](https://huggingface.co/datasets/klusai/ds-tf1-en-3m)), then aligned via
the full **RLHF pipeline** (reward model + PPO) with **AI preference feedback — RLAIF**
(Claude rates pairs under a human-authored rubric; ADR-0004). University course project: graded on
design reasoning, so every decision below carries its justification. Vocabulary: see
[CONTEXT.md](../CONTEXT.md). Hard-to-reverse decisions: see [docs/adr/](./adr/).

## Constraints (fixed)

| Constraint | Value | Consequence |
|---|---|---|
| Team | Solo | Ruthless scoping; no labeling pool → AI feedback (RLAIF) with instrument audits; the human authors the rubric and runs the final blind eval |
| Time | 4–8 weeks | Two committed ablations only; stretch goals gated on schedule |
| Compute | Free Colab (T4 16GB), sessions die | Checkpoint-resume is a first-class requirement, fp16, ~250M-token data budget |
| Grading | Design explanation over accuracy | Decision record + ADRs + measured justifications are deliverables, not overhead |

## Pipeline (ADR-0001)

```
2.8M-row train split ──sample──▶ ~450k fables ≈ 250M tokens
                                      │ tokenize once → uint16 shards (~500MB, on Drive/HF)
                                      ▼
        PRETRAIN from scratch (next-token, completion-only loss)
                                      ▼
                      BASE MODEL  ← already instruction-following
                (pretraining data is 100% prompt→fable — no SFT stage)
                                      ▼
        RLAIF: Claude-labeled preference pairs → reward model → PPO (KL anchor = Base)
                                      ▼
                      ALIGNED MODEL  → demo, eval, report
```

Why no SFT and why no pretrained base: ADR-0001.

## Interface

- A **FableSpec** (character, setting, challenge, outcome, Moral, Age Group — possibly partial)
  is rendered to text the model completes.
- **Canonical Prompt**: the dataset's own fixed template ("Create a fable based on the following
  elements…"). **Paraphrased Prompts**: ~30 alternative phrasings in a versioned `paraphrases.yaml`
  (LLM-brainstormed once, human-curated), slot-filled mechanically over 10–20% of training
  examples. Element values are preserved **verbatim** so spec adherence stays mechanically
  measurable.
- **5 of ~30 templates are held out of training** — robustness to unseen phrasing is measured,
  not asserted.
- Demo accepts either a structured form or free-text instruction.

### Paraphrase augmentation (issue 03)

The bank lives at `configs/paraphrases.yaml` (versioned, human-curated): 30
templates, each slot-filling all five Elements verbatim; **5 are `held_out:
true`** (`heldout-minimal-01`, `heldout-question-02`, `heldout-json-03`,
`heldout-letter-04`, `heldout-headline-05`) and are never used in training —
they are the robustness grid's unseen-phrasing column (issues 05/10). Prep
(`stages/prep.py`) rewrites a `paraphrase_coverage` fraction of rows (10–20%
design range; the toy config uses 0.15): it parses Element values back out of
each Canonical Prompt, re-renders under a **seen** template chosen by an RNG
seeded on `(seed, row_index)` (so runs stay hash-deterministic), and writes a
per-row `families.bin` (uint8: 0 canonical / 1 seen-template / 2
held-out-template) plus `family_counts` in `prep_summary.json`. Rows that do not
match the canonical structure fall back to canonical unchanged and are counted
in `n_parse_failures`. Held-out leakage is provable from the artifacts:
`families.bin` never contains code 2 and `family_counts["held-out-template"]`
is 0 for any training prep. Verbatim preservation is a property test over every
template × spec; `str.format` never re-interprets substituted values, so
Element values with braces/colons survive exactly, keeping spec adherence
mechanically measurable.

## Data

Dataset facts: 3M fables, Llama-3.1-8B-Instruct-generated, 2.8M/100k/100k splits, avg ~521
tokens (Llama tokenizer), fixed prompt template + identical system message, five structured
elements per fable, markdown formatting (bolded morals) present, MIT license.

**Audit before filtering** (600 rows sampled across six regions of the train split):

| Check | Result |
|---|---|
| Truncated (no terminal punctuation) | 2/600 raw; 1 was a false positive → **~0.2% real** |
| Duplicate `prompt_hash` in sample | 0 |
| Duplicate 80-char openings | 2/600 (formulaic, not true dupes) |
| Fable length (words) min/p10/p50/p90/max | 223 / 247 / 266 / 286 / 332 |

**Decision: no filtering, uniform random sample** (~450k fables). The audit shows the failure
modes filtering would target barely exist; "we measured, then chose not to filter" is the
justification. Markdown is kept raw — the bolded moral makes the Moral mechanically extractable
at eval time.

Prep: tokenize once → uint16 memmap shards (~250M tokens ≈ 500MB) pushed to Drive/HF; training
sessions start in seconds. Provided val/test splits reused (small slices) — val for perplexity
and FableSpec sampling, test untouched until final eval. Sequences packed (prompt+fable+EOS)
into 1024-token windows, GPT-2 style. **Loss on fable tokens only** (labels −100 on prompt
spans): the template repeats 3M times, and completion-only loss makes "val perplexity" mean
perplexity *of fables*.

## Tokenizer (ADR-0002)

Custom byte-level BPE, vocab 8192, trained on a corpus sample; specials reserved now
(end-of-text, pad). Embedding cost at d=384 tied: 3.1M vs 19.3M for GPT-2's vocab. Report
figure: avg tokens/fable under GPT-2-50k vs ours-8k vs ours-4k.

## Model

Hand-written decoder-only GPT (course-lecture-aligned), wrapped as a HF `PreTrainedModel`
subclass for ecosystem compat (generate, push_to_hub, TRL).

| Knob | Value | Why |
|---|---|---|
| Layers / d_model / heads | 6 / 384 / 6 | ~10.6M transformer + 3.1M emb = **13.7M**, TinyStories fluency regime |
| Context | 1024 | whole prompt+fable fits; Moral (at the end!) never truncated |
| Positions | learned absolute | simplest correct implementation; RoPE rejected: risk without benefit at ctx 1024 |
| Norm / act | pre-norm LayerNorm / GELU | fp16-stable, GPT-2 standard |
| Dropout | 0 | ~1–2 epochs over 250M tokens = underfitting regime |
| Weight tying | yes | saves 3.1M params |

Sizing argument: Chinchilla ~20 tokens/param → 13.7M wants ~210–270M tokens — exactly the free-
Colab budget. **Week-1 gate: measured T4 tokens/sec benchmark before committing** (est. 1–2h per
200M-token epoch; if reality is far worse, shrink data budget or model — measure, then commit).

### Implementation (issue 02)

- **HF wrapper (transformers 5.x).** `GPT` subclasses `PreTrainedModel` + `GenerationMixin`
  so `.generate()`, `save_pretrained`/`from_pretrained`, TRL (issue 08) and `push_to_hub`
  (issue 11) come for free. Weight tying uses the 5.x contract: `_tied_weights_keys =
  {"head.weight": "tok.weight"}` (a dict, not a list), `tie_word_embeddings=True` passed
  explicitly through the config (5.x does not default it), and tying performed by
  `post_init()` — hand-aliasing the tensors does not survive `from_pretrained`.
- **Attention written out explicitly** (QKV projection → scaled dot-product → causal-mask
  softmax → output projection), not a fused SDPA/`MultiheadAttention` call — the model is
  the pedagogical core, so the causal mask and softmax are visible and directly tested. The
  causal mask is built inline in `forward`, not cached as a registered buffer: transformers
  5.x constructs the model under a meta device in `from_pretrained`, and a non-persistent
  buffer would be materialized as uninitialized garbage there, corrupting the mask after a
  save/load round-trip.
- **Generation** uses HF `.generate()` with `use_cache=False` (no KV cache in the walking
  skeleton — the full window is recomputed each step; acceptable at ctx 1024 demo scale, a
  later optimization if needed). The generation *contract* (FableSpec/free-text → fable) is
  our own code and encodes the prompt exactly as prep does.
- **Measured parameter count** at the real config: **14,186,496** (≈14.19M) — 13.8M
  transformer+token-embedding + 0.39M learned positions. The 13.7M headline counts
  transformer+token-embedding.
- **Walking-skeleton training is fp32** on CPU/MPS/CUDA (device-selectable). Batches are a
  pure function of the step and dropout is 0, so checkpoint-resume is bit-exact (train-N ==
  train-k → resume → train-(N−k)). Mixed precision (fp16 + GradScaler) and the real
  data/checkpoint budget are deferred to issue 04.
- `torch` and `transformers` are core dependencies but kept out of the light import path
  (stage registry is lazy by module-path string; the CLI's `generate` branch imports torch
  lazily). The Canonical Prompt template lives in `prompts.py` (torch-free); Paraphrased
  Prompts extend it in issue 03.

## Build vs buy

**Build** (line by line, pedagogical core): model, pretraining loop with checkpoint-resume,
generation. **Buy** (justified per-choice in report): `tokenizers` (BPE training), `datasets`
(streaming/prep), TRL (PPO + value head), trackio (metrics; W&B fallback). Repo is a real
package (`src/tinyfables/`); Colab notebooks are thin drivers that `pip install` the repo — no
logic in notebooks; every run reproducible from a commit + config. Frozen dataclass configs per
experiment, fixed seeds. Sanity suite before any long session: tokenizer round-trip, forward
shape/dtype, **overfit-one-batch** (13M params must memorize 32 examples in minutes, else stop).

Checkpoints (with optimizer state) pushed to Drive/HF every N steps — a dead session costs ≤ N
steps.

## Feedback stage (RLAIF — ADR-0004; label format — ADR-0003)

- **Pairs**: 2 independent samples (temp ≈0.9) per FableSpec, specs drawn from val split;
  target ~2,000 pairs (≈4,000 generations from the Base Model — under an hour on T4).
- **AI Labeler**: Claude via headless `claude -p` — a resumable script feeds `RUBRIC.md` +
  ~5 pairs per call, receives strict-JSON per-Axis 1–5 ratings, caches to JSONL. Model version
  and prompt version logged with every label; short justifications sampled for hand audit.
  Runs unattended in an afternoon; zero marginal cost on the Claude subscription. The anchored
  `RUBRIC.md` is still written first, by the human — the rubric is where the human defines
  "good"; the AI applies that definition at scale.
- **Labeler audits** (the ADR-0003 drift controls, reinterpreted for an AI instrument):
  position-swap audit — a 10% slice labeled in both A/B orders, flip rate reported;
  Calibration Set — 30 fixed pairs re-labeled at start/middle/end of the run → self-consistency
  reported. Framing for the report: "we treat the labeler as a measurement instrument."
- **Preference derivation**: Aggregate Score = 0.4·moral + 0.3·adherence + 0.2·coherence +
  0.1·prose; ties skipped; ±0.1 weight-sensitivity table in report.
- **Reward model**: Base Model + scalar head, Bradley–Terry loss on derived preferences,
  ~1,800 train / 200 held-out.
- **GATES**: labeler audits pass (self-consistency ≥ ~85%; flip rate reported and reasoned
  about) **and RM held-out accuracy ≥ 65%** (chance 50%). Below gate → do not run PPO on a
  noisy or self-inconsistent signal; diagnose or report the negative result with the RM curve.
- **PPO** via TRL: KL penalty anchored to the frozen Base Model; policy+ref+RM fit one T4 at
  13M. **Length-drift alarm**: mean generation length tracked every PPO iteration (length bias
  is the classic reward hack; the ~250-word fable format makes it detectable).
- **Documented fallbacks**: (a) if the course requires literal human feedback, a ~60-pair
  human **Gold Set** (~5h, blind, same rubric) bolts on as a validation layer — human–AI
  agreement measured, nothing upstream retrained; (b) if PPO won't stabilize by end of week 5,
  DPO on the same derived preferences ships as the Aligned Model, and the PPO attempt becomes
  report material.

## Evaluation

| Layer | What | Cost |
|---|---|---|
| Mechanical (every checkpoint) | fable-token perplexity; spec adherence (verbatim Element presence); Moral delivery proxy (extract bolded moral, fuzzy-match to requested); distinct-n / repetition; length stats | free |
| LLM-as-judge (Base vs Aligned, ~100 gens) | grammar / creativity / consistency / moral, TinyStories-style; **must be a different model family than the AI Labeler** (Gemini free tier, never Claude — the labeler judging a model optimized toward its own taste inflates win rates by construction); prompts versioned, responses cached | ~0 |
| Human, headline | blind A/B on ~50 held-out FableSpecs, Base vs Aligned (+2–3 classmates if possible) → **win rate = the money chart**, and under RLAIF it also tests the premise itself: does alignment to Claude's ratings transfer to human preference? | hours |

Paraphrase robustness grid: adherence on canonical / seen-template / **held-out-template**
prompts. Known limitation from the audit: keyword regexes miss ~10% of morals → moral detection
uses the markdown marker + fuzzy matching, validated by hand on a sample.

## Ablations

**Committed**: (1) paraphrase augmentation on/off — second pretrain without Paraphrased
Prompts, compared on the robustness grid; (2) RM data curve — accuracy at 100/500/1,000/2,000
pairs (AI labels are cheap, so the curve extends until it saturates; powers the PPO gate).
**Stretch**: PPO vs DPO from the same preferences; ~3M scaling sibling.

## Deliverables

Report (design-reasoning-first) · Gradio demo on HF Spaces free CPU (FableSpec form + free-text
box, **side-by-side Base vs Aligned** — the RLHF effect made visible) · public Hub artifacts
(`tinyfables-tokenizer`, `-13m-base`, `-13m-rm`, `-13m-aligned`, `-preferences` — AI-labeled,
carded explicitly as RLAIF, incl. rubric + labeler prompt) each with a card · ~10 slides (pipeline, sizing math, audit table, rubric, RM curve, ablation
grid, win-rate chart, retrospective).

## Schedule (8 weeks, gates in bold)

| Week | Work |
|---|---|
| 1 | Scaffold, tokenizer + compression figure, prep shards, sanity suite, **T4 benchmark gate → confirm size**; **ask professor: does AI feedback satisfy the RLHF requirement?** |
| 2 | Pretrain main run (resumable); paraphrase-free sibling queued in spare sessions |
| 3 | Base eval pass; RUBRIC.md; AI-labeler script + audits; generate ~2,000 pairs; bulk labeling runs unattended |
| 4 | Derive preferences + sensitivity table; train RM + extended data curve; **labeler audits + RM ≥65% gate**; PPO prep with the reclaimed labeling time |
| 5 | PPO: tune KL coef, watch length alarm; **fallback decision: DPO if unstable** |
| 6 | Finish no-aug sibling → ablation grid; LLM-judge pass; stretch: DPO comparison |
| 7 | Blind human eval; freeze results; build Spaces demo |
| 8 | Report + slides + buffer |

## Risk register

| Risk | Mitigation |
|---|---|
| Colab session death | resume from step-N checkpoint w/ optimizer state; shards not raw data |
| T4 slower than estimate | week-1 benchmark gate; shrink data budget before shrinking model |
| RM learns noise (few labels) | 65% accuracy gate; RM data curve; floor 150 pairs |
| PPO instability / reward hacking | KL anchor, length-drift alarm, DPO fallback documented in advance |
| AI labeler bias / inconsistency | position-swap audit, calibration re-labeling, sampled justifications hand-reviewed |
| Judge circularity (labeler grading its own taste) | eval judge is a different model family (Gemini); blind human eval is the headline |
| ~~Course requires literal human feedback~~ | **Resolved 2026-07-06: professor signed off on RLAIF.** Gold Set bolt-on stays documented but inactive |
