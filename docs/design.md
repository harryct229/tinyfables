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
- **Template reconciled with real data (issue 04).** ds-tf1-en-3m's actual prompt is a rich fixed
  template — 2-space-indented Element bullets, a `The fable should:` block of 8 style bullets, and
  `Keep the story concise but engaging, around 250 words.` The dataset is **single-band**: every
  row is age group B (4-7 years), ~250 words (verified over 11k rows). Issues 02–03 shipped a
  *simplified* reconstruction that only matched hand-built fixtures; issue-04 Track B caught it when
  the fail-loud "all rows failed to parse" guard aborted real prep. `prompts.render_canonical_prompt`
  now reproduces the real band-B template byte-for-byte and `paraphrases.parse_canonical_prompt`
  parses it; the toy fixtures were rebuilt from the real format so tests guard reality. `age_range`
  stays on FableSpec (paraphrase templates slot it; the parser recovers it) but the canonical render
  always emits the band-B block. Toy `window`/`n_ctx` went 256→512 to fit the longer real prompt.
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

**Measured (issue 04, 50k-fable sample):** ours-8k = **337.8** tokens/fable vs GPT-2-50k =
**349.79** — a ~3.4% compression gain. The honest read: compression is a minor bonus; the
real ADR-0002 win is the embedding-parameter savings (3.1M vs 19.3M), not sequence length.
(ours-4k not trained — a stretch variant.)

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

### Implementation (issue 04)

- **Mixed precision (T4).** Training runs under `torch.amp.autocast(fp16)` with a
  `GradScaler`; AMP is enabled only on CUDA (`amp and device=="cuda"`), so CPU/MPS
  toy runs stay fp32 and the walking-skeleton bit-exact resume guarantee holds. The
  scaler's dynamic scale is checkpointed (in `optimizer.pt`) so resume continues the
  loss-scaling schedule.
- **Checkpoint-resume for session death.** `checkpoint.py` writes
  `ckpt_dir/step_{n}/` (model safetensors + optimizer + scaler + step), with
  `checkpoint_state.json` written LAST as the completion marker (an interrupted
  write is never mistaken for complete). `pretrain` checkpoints every `ckpt_every`
  steps, prunes to `keep_last_k`, and on startup AUTO-RESUMES from the latest
  complete checkpoint in `ckpt_dir` — so a dead Colab session is recovered by
  re-running the *same command*. On Colab `ckpt_dir` is Drive-mounted, so it
  outlives the runtime; a session death costs ≤ `ckpt_every` steps. `out_dir` still
  holds the FINAL model + manifest-last.
- **Benchmark gate.** The `benchmark` stage times the target geometry with fp16 on
  synthetic batches (real vocab/shape), excludes warmup, and records
  `tokens_per_second`, `est_seconds_per_epoch`, and a `go`/`revise` decision vs
  `target_tokens_per_sec`. Run it BEFORE `pretrain_full`; record the measured number
  and the decision here (Track B) — this is the week-1 "measure, then commit" gate.
- **Metrics + loss curve.** `metrics.MetricsLogger` mirrors loss/lr/tokens-per-sec
  to a durable `loss_log.csv` (in `ckpt_dir`, so it survives session death) and, when
  a `trackio_project` is configured and trackio is installed, to a trackio Space for
  a live dashboard. On finish the CSV is copied into `out_dir` and `loss_curve.png`
  is rendered (matplotlib, best-effort). trackio + matplotlib are the optional
  `.[train]` extra; absent, logging degrades to CSV-only (tests run offline).
- **Hub push.** `hub.py` (+ `python -m tinyfables push`) pushes the tokenizer
  (`tinyfables-tokenizer`) and the Base Model (`tinyfables-13m-base`); the model push
  ignores `optimizer.pt`/checkpoint markers so a checkpoint dir can be the source.
- **Operational record (Track B, real T4 run 2026-07-06):** measured T4 throughput =
  **50,971 tokens/sec** (fp16 AMP, batch 16, ctx 1024, target geometry); benchmark
  decision = **go** (35k floor cleared; est. epoch over 250M tokens ≈ 82 min, full
  20k-step run ≈ 1.8–2 h). Chosen `ckpt_every` = **500** (interval ≈ 161 s → a session
  death costs ≤ ~2.7 min of training); `keep_last_k` = 2. No size/data-budget change —
  geometry stayed 6/384/6, ctx 1024, batch 16, 20k steps.
- **Base Model result (Track B complete, 2026-07-06).** Prep: 450k rows → **219.9M
  tokens**, paraphrase coverage 0.15 (67,629 seen-template rows), **0 parse failures,
  0 held-out leakage**. Pretrain: 20,000 steps at ~49k tok/s (~1.8 h), fp16 AMP, final
  **fable-token loss 1.4466** (from 7.03); measured params **14,186,496**. Artifacts on
  the Hub: **[`congthanh991/tinyfables-13m-base`]** (model + `loss_curve.png` + logs) and
  **[`congthanh991/tinyfables-tokenizer`]**. Compression figure (ADR-0002): ours-8k 337.8
  vs GPT-2 349.79 tok/fable. **Spot check (20 varied FableSpecs, single-band 4-7):**
  fluent ~250-word fables with proper fable structure; character honored 16/18 (one drift,
  lamb→wolf), settings broadly present, morals delivered in prose. **Note for issue 05:**
  the model rarely emits the `**bold**` moral marker (2/20), so moral extraction must lean
  on the fuzzy fallback, not the markdown marker. **Checkpoint-resume (AC-2):** implemented,
  unit-tested (bit-exact auto-resume), and checkpoints were written to Drive every 500
  steps throughout — but this run completed in a single stable session, so a *real* death
  was not triggered; the survival capability is proven by tests + the on-Drive checkpoints,
  not by an actual mid-run death this session.
- **Dataset is single-band** (issue 04): ds-tf1-en-3m is entirely age group B (4-7 years),
  ~250 words (verified 11k rows; 0/450k parse failures at prep confirms it). The canonical
  prompt template was reconciled with the real dataset here (issues 02–03 shipped a
  simplified reconstruction masked by hand-built fixtures; the fail-loud parse guard caught
  it). `render_canonical_prompt` reproduces the real band-B template byte-for-byte and
  raises on any non-4-7 age; `parse_canonical_prompt` requires the exact structure.

### Implementation (issue 05)

- **`evaluate` stage.** One command (`python -m tinyfables run evaluate --config … --out …`)
  scores any checkpoint into `eval_metrics.json` (machine), `eval_report.md` (report tables),
  and `moral_calibration.jsonl` (hand-labeling worksheet), manifest last. Deterministic given
  seed + eval set: perplexity has no sampling; generation is seeded per spec.
- **Fable-token perplexity** (`perplexity.py`): completion-only next-token loss over the val
  slice (prompt masked), so it means perplexity *of fables*; exp of the mean.
- **Adherence robustness grid** (`eval_metrics.element_adherence`): verbatim (case-insensitive)
  presence of each requested Element (character/setting/challenge/outcome) in the generated
  fable — a mechanical, comparable proxy. Each spec is rendered under canonical + a seen
  template + a held-out template and adherence is grouped by family; held-out templates are
  eval-only (used here, never trained on).
- **Moral delivery** (`moral.py`): `extract_moral` prefers the last `**bold**` segment but
  falls back to the trailing sentence (the issue-04 Base Model rarely emits the marker), then
  fuzzy-matches (SequenceMatcher) to the requested moral above a threshold. `naive_regex_moral`
  is the keyword-regex baseline the extraction must beat; precision is hand-validated in Track B.
- **Distinct-n / repetition / length** round out the generation-quality view. No new deps
  (stdlib `re`/`difflib`/`statistics`).
- **Track B real Base Model eval (Colab T4, 2026-07-08).** Ran
  `python -m tinyfables run evaluate --config configs/eval_full.yaml --out runs/eval_base`
  against `congthanh991/tinyfables-13m-base` + `congthanh991/tinyfables-tokenizer`;
  artifacts stayed at `/content/tinyfables/runs/eval_base` because Drive was not mounted.
  Eval finished in 4.9 min. Fable-token loss/perplexity: `1.402` / `4.063` over 500 rows.
  Generation: 50 specs, distinct-1 `0.133`, distinct-2 `0.565`, repetition-4 `0.051`,
  length mean/median `252.1/253`, min/max `233/270`, moral delivery `0.380` at threshold
  `0.300`. Adherence grid: canonical overall `0.145` (character `0.580`, moral `0.380`),
  seen-template overall `0.130` (character `0.500`, challenge `0.020`, moral `0.320`),
  held-out-template overall `0.090` (character `0.360`, moral `0.280`). Provenance:
  checkpoint sha `edbd8f125a9bf12578e4b5246eebbbab51b93c9fa169f785373bb4db95065e0a`,
  tokenizer sha `9d53402d4c4dd94d0d7bf133ea8073cf966413e510b8b3a8c813ae0ecc625dfb`,
  seed `0`, versions `{tinyfables: 0.1.0, tokenizers: 0.22.2, numpy: 2.0.2,
  torch: 2.11.0+cu128, transformers: 5.12.1}`. Moral extraction hand validation over all
  50 `moral_calibration.jsonl` rows: ours `27/50 = 0.54`, naive regex `0/50 = 0.00`
  (`naive` was null on all 50 rows), so the extraction beats the baseline; residual risk is
  that many misses are outcome/title captures after an explicit moral sentence.

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

### Implementation (issue 06, Track A)

- **Rubric + labeler prompt (human artifacts).** `RUBRIC.md` (four Axes, anchored 1-5
  definitions with example snippets) and `configs/labeler_prompt.yaml` (versioned
  instruction template, `{rubric}`/`{pairs}` slots) are the human-authored definition
  of "good". `prompt_version` is logged on every label; changing the rubric/template
  without bumping `version` is a loud error at label time.
- **`feedback.py` (torch-free).** Single source of truth for the ADR-0003 weights
  (0.4/0.3/0.2/0.1); `aggregate_score`, `derive_preference` (ties skipped),
  `weight_sensitivity` (±0.1 table), and the two audit metrics
  (`position_flip_rate`, `self_consistency`).
- **Single-source boundary.** The production weight mapping and aggregate formula
  live only in `src/tinyfables/feedback.py`; human-facing mirrors in `RUBRIC.md`
  and this design doc exist for review and approval, not as a second logic source.
- **`pairgen` stage.** Mirrors `evaluate`: 2 independent samples (temp 0.9, seeded
  per sample) per val FableSpec under the Canonical Prompt → `pairs.jsonl`;
  `pairgen_summary.json` records the Base Model `checkpoint_sha` so preferences trace
  back to the exact model.
- **`label` stage (AI Labeler, ADR-0004).** Resumable headless `claude -p` behind an
  injected `runner`; append-only `labels.jsonl` cache keyed by
  `(pair_id, phase, order, model_version, prompt_version)` doubles as the offline
  replay fixture. Work list interleaves the Calibration passes (start/middle/end) and
  the 10% position-swap slice. Tests use a deterministic fake runner; one schema
  contract test validates parsing against a recorded sample. `AuditConfig.
  position_swap_review_threshold` documents the high-flip review threshold for the
  qualitative position-swap flag, so the audit contract records both the measured flip
  rate and when it should trigger manual review.
- **`derive` + `audit` stages.** `derive` → `preferences.jsonl` (chosen/rejected +
  aggregates + deterministic train/held-out split) + `sensitivity.json`; `audit` →
  `audit_report.md`/`audit.json` (flip rate + self-consistency vs the ≥0.85 gate).
- **Track B real feedback run (2026-07-10).** Pairgen ran on Colab T4 from
  `congthanh991/tinyfables-13m-base` + `congthanh991/tinyfables-tokenizer`, producing
  **2,000** non-empty, non-identical single-band pairs. Labeling used `claude-sonnet-5`
  with prompt version **2** and completed **2,290** label instances: 2,000 primary pairs,
  a 10% position-swap slice, and the 30-pair calibration set re-labeled across the run.
  `derive` emitted **1,949** preferences and skipped **51** aggregate-score ties
  (train/held-out split: 1,761/188). Weight-sensitivity at +/-0.1 was low: worst case
  moral -0.1 flipped **64/1,949 = 3.28%** of non-tie preferences; all other perturbations
  were lower. Audit result: position-swap flip rate **0.295** (59/200), below the 0.50
  review flag; Calibration Set self-consistency **0.778** (21/30 unanimous), **below**
  the 0.85 gate, so issue 07 must treat the reward-model input as flagged rather than
  silently passed. Artifacts: `runs/labels_base`, `runs/audit_base`, `runs/derive_base`
  (local, gitignored); stable downstream input is `runs/derive_base/preferences.jsonl`.

### Implementation (issue 07)

- **Reward model stage.** `reward` trains the Base Model plus a scalar head using Bradley-Terry
  loss over `runs/derive_base/preferences.jsonl`. The reward score is pooled from the last
  non-pad token of `prompt + fable + <|endoftext|>`, matching the prep/generation tokenization
  contract. The stage writes the RM folder (`backbone/`, `reward_head.pt`,
  `reward_model_config.json`), `reward_summary.json`, `data_curve.json`, `loss_log.csv`, and
  manifest last.
- **Track B real RM run (2026-07-10).** Trained on `1761` train preferences and
  evaluated on `188` held-out preferences. Held-out accuracy was
  `0.590`, so the RM accuracy gate (`>=0.65`) was `fail`.
  Data curve points were: `100 -> 0.617`, `500 -> 0.612`,
  `1000 -> 0.580`, `2000 requested / 1761 available -> 0.628`.
- **Alignment gate.** `gate` combines issue 06 audit numbers with RM accuracy into an explicit
  PPO go/no-go record. The real gate verdict was `NO-GO`: `labeler self-consistency below gate,
  reward model held-out accuracy below gate`. Because issue 06 labeler self-consistency was
  `0.778 < 0.85`, the gate is expected to remain no-go unless the project intentionally accepts
  noisy-label risk in a later ADR. The Reward Model artifact was pushed to
  `congthanh991/tinyfables-13m-rm`.

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
