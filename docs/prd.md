# PRD: TinyFables — a from-scratch moral-fable generator aligned with AI feedback

> **Triage**: `ready-for-agent` (recorded here as text — no issue tracker exists yet; when the
> GitHub repo is created, file this as an issue and apply the label there).
> **Companions**: the decision record (`docs/design.md`), the glossary (`CONTEXT.md`), and
> ADRs 0001–0004 (`docs/adr/`). Where this PRD and the design record overlap, they must agree;
> where they conflict, the ADRs win.

## Problem Statement

I am a solo student in an ML course that grades **design reasoning over benchmark accuracy**. I
must build a generative model for English moral fables **trained from the beginning** — not a
fine-tune of someone else's weights — and aligned with an RLHF-style feedback stage, and I must
be able to defend every design decision to a grader. My constraints are hard: I work alone, on
free Colab (sessions die without warning), with 4–8 weeks of calendar time. Hand-labeling
preference data would cost me 15–20 hours I don't have, and 200 hand-labeled pairs would still
be too thin a signal to train a trustworthy reward model.

## Solution

TinyFables: a ~13M-parameter GPT, hand-written but Hugging Face-compatible, pretrained from
random weights on ~450k synthetic fables (klusai/ds-tf1-en-3m) and then aligned via the full
RLHF pipeline — reward model plus PPO — with **AI preference feedback (RLAIF)**: Claude, as the
AI Labeler, rates ~2,000 Preference Pairs against a human-authored Rubric, unattended, in an
afternoon. The user-facing product accepts a FableSpec (Elements + Age Group) or a free-text
instruction and returns a Fable that honors it, demonstrated in a side-by-side Base-vs-Aligned
demo. Every pipeline stage is a resumable, deterministic config-in/artifacts-out step, so a
dying Colab session, a skeptical grader, or a reproducing researcher can all be answered the
same way: rerun the stage, check the artifact. Evaluation is triangulated — mechanical metrics,
a cross-family LLM judge (never the labeler's family), and a blind human A/B eval as the
headline — and the whole design is documented in ADRs the report is built from.

## User Stories

**Data & tokenizer**

1. As the student, I want a one-command data-prep stage that samples ~450k fables and writes tokenized shards, so that every Colab session starts training in seconds instead of re-processing 13.4 GB.
2. As the student, I want 10–20% of training examples rendered as Paraphrased Prompts from a versioned template bank with 5 templates held out of training, so that robustness to unseen instruction phrasing is trainable and measurable.
3. As the student, I want Element values preserved verbatim in every prompt rendering, so that spec adherence stays mechanically checkable.
4. As the student, I want a tokenizer-training stage that outputs the 8k BPE together with a compression report across candidate vocabularies, so that the vocabulary decision is a measured figure in my report, not an assertion.
5. As the student, I want prompt spans marked in the packed shards, so that training loss lands on fable tokens only and validation perplexity means perplexity of Fables.

**Model & pretraining**

6. As the student, I want the transformer hand-written but wrapped in the Hugging Face model interface, so that I understand every line while keeping generation, Hub pushes, and TRL interop for free.
7. As the student, I want a throughput-benchmark stage for the T4, so that I confirm the model size against measured tokens/sec before spending real sessions (measure, then commit).
8. As the student, I want pretraining to checkpoint with optimizer state and resume exactly, so that a dead Colab session costs minutes, not hours.
9. As the student, I want an overfit-one-batch sanity mode, so that broken training code is exposed in minutes rather than after a wasted session.
10. As the student, I want every run driven by a frozen, committed config with a fixed seed, so that any artifact is reproducible from a commit hash.
11. As the student, I want training metrics streamed to an experiment tracker, so that I can watch a Colab run's loss curve from anywhere.

**Generation & demo**

12. As the student, I want one generation contract that accepts either a FableSpec or free instruction text, so that the demo, the evaluation suite, and pair generation all share a single code path.
13. As a demo visitor, I want to fill in Elements — or just type an instruction — and receive a Fable that honors them, so that I can steer the story being told.
14. As a demo visitor, I want an Age Group control, so that the same FableSpec can produce a toddler-simple or a teen-appropriate fable.
15. As a demo visitor, I want side-by-side Base Model and Aligned Model outputs for the same request, so that the effect of the feedback stage is visible in ten seconds.
16. As a grader, I want the demo hosted on a free CPU Space, so that I can try the system with zero setup.

**Feedback stage (RLAIF)**

17. As the student, I want a pair-generation stage that samples ~2,000 Preference Pairs from held-out FableSpecs, so that the reward model gets training signal at a scale hand-labeling could never reach.
18. As the student, I want the Rubric authored and committed before any labeling, so that the definition of a good fable is mine, fixed, and citable.
19. As the student, I want the AI Labeler to run unattended and resumably, caching per-Axis ratings to an append-only record with model and prompt versions on every label, so that labeling costs one afternoon and every label is auditable.
20. As the student, I want position-swap and Calibration Set audits built into the labeling run, so that I can report the AI Labeler as a validated measurement instrument rather than an article of faith.
21. As the student, I want preferences derived from Aggregate Scores with a weight-sensitivity table, so that the aggregation weights are defended before a grader asks.
22. As the student, I want reward-model training with held-out accuracy and a data-size curve, so that I know whether the preference signal is real and where it saturates.
23. As the student, I want hard gates enforced before PPO — labeler audits pass and reward-model accuracy ≥ 65% — so that I never spend GPU sessions optimizing noise.
24. As the student, I want PPO to run with a KL anchor to the frozen Base Model and a length-drift alarm, so that reward hacking is detected the day it starts, not in the report's post-mortem.
25. As the student, I want a DPO stage runnable on the same derived preferences, so that a PPO failure still ships an Aligned Model and the project cannot die in week 5.
26. As the professor, I want the feedback stage explicitly framed as RLAIF with a documented human Gold Set fallback, so that I can judge honestly whether the course's RLHF requirement is met.

**Evaluation & ablations**

27. As the student, I want a mechanical evaluation suite runnable on any checkpoint — fable perplexity, spec adherence, Moral delivery, repetition, length stats — so that any two checkpoints are comparable with one command.
28. As the student, I want the adherence results split by Canonical, seen-template, and held-out-template prompts, so that the paraphrase augmentation is judged on phrasing the model never saw.
29. As the student, I want the LLM judge to be a different model family from the AI Labeler, with versioned prompts and cached responses, so that judge scores are repeatable and free of self-grading circularity.
30. As the student, I want a blind A/B tool for the final human evaluation over ~50 held-out FableSpecs, so that the headline win rate is credible — and so that I can test whether alignment to Claude's ratings transfers to a human's preference.
31. As the student, I want the paraphrase on/off sibling to be a config-only variant of the pretraining stage, so that the committed ablation is one more run, not new code.
32. As the student, I want the full pipeline runnable at toy scale on CPU in minutes, so that the entire chain is testable before any GPU session is spent.
33. As the student, I want evaluation outputs saved as report-ready tables and figures, so that week 8 is assembly, not archaeology.

**Reproducibility & operations**

34. As a reproducing researcher, I want the tokenizer, Base Model, reward model, Aligned Model, and the AI-labeled preference set published to the Hub with cards (the preference set carded explicitly as RLAIF), so that every claim in the report links to a checkable artifact.
35. As the student, I want Colab sessions drivable from Claude Code via the Colab MCP connection, so that training operations stay scripted and observable instead of manual notebook fiddling.
36. As a grader, I want every non-obvious decision traceable to an ADR or a measurement in the decision record, so that the design reasoning — the thing being graded — is assessable without asking the author.

## Implementation Decisions

- **Pipeline shape (ADR-0001)**: two stages — pretrain from random weights on the
  instruction-formatted corpus, then the feedback stage directly on the Base Model. No SFT
  stage (pretraining data is already 100% prompt→fable); no pretrained base weights anywhere.
- **Stage contract (the primary architectural rule)**: every pipeline stage is invoked as
  config-in → artifact-files-out (shards, checkpoints, ratings records, metrics documents),
  deterministic given a seed, and resumable from its own partial output. Colab notebooks are
  thin drivers over these stages; no logic lives in notebooks.
- **Interface**: a FableSpec (Elements + Age Group) renders to either the Canonical Prompt (the
  dataset's own template) or a Paraphrased Prompt from an offline, versioned template bank
  (~30 phrasings, 5 held out of training). Element values are never rewritten.
- **Tokenizer (ADR-0002)**: custom byte-level BPE, vocab 8192, trained on the fable corpus;
  end-of-text and pad specials reserved at creation. Embedding cost math and compression
  comparison are report artifacts.
- **Model**: decoder-only GPT, ~13.7M params (6 layers, d_model 384, 6 heads, context 1024),
  pre-norm LayerNorm, GELU, learned absolute positions, tied embeddings, dropout 0. Hand-written
  modules inside a Hugging Face `PreTrainedModel` wrapper.
- **Data policy**: uniform random ~450k-fable sample, no filtering — backed by a 600-row audit
  (real truncation ≈ 0.2%, zero duplicate hashes). Markdown kept raw (the bolded moral is the
  eval-time extraction hook). Provided val/test splits reused; sequences packed with EOS;
  loss masked to fable tokens.
- **Feedback stage (ADR-0004, format per ADR-0003)**: pure RLAIF. The AI Labeler is Claude via
  headless CLI invocation, batched ~5 pairs per call, emitting strict-JSON per-Axis 1–5 ratings
  into an append-only cache keyed with model and prompt versions. Audits: position-swap on a 10%
  slice; a 30-pair Calibration Set re-labeled at start/middle/end. Preferences derive from the
  Aggregate Score (0.4 moral / 0.3 adherence / 0.2 coherence / 0.1 prose), ties skipped.
- **Reward model**: Base Model + scalar head, Bradley–Terry loss, ~1,800 train / 200 held-out.
- **Gates**: labeler self-consistency ≥ ~85% and reported flip rate; reward-model held-out
  accuracy ≥ 65%. PPO does not run unless both pass.
- **PPO**: via TRL, KL-anchored to the frozen Base Model, length-drift alarm each iteration.
  Documented fallbacks: DPO on the same preferences if PPO won't stabilize; a ~60-pair human
  Gold Set bolts on as a validation layer if literal human feedback is required.
- **Judge-circularity rule**: the eval LLM judge is a different model family from the AI
  Labeler (Gemini-class free tier, never Claude). The blind human eval is the headline metric.
- **Build vs buy**: hand-write the model and training loop; buy tokenizer training, dataset
  streaming, PPO machinery (TRL), and experiment tracking (trackio, W&B fallback).
- **Artifact layout**: five Hub repos (tokenizer, base, reward model, aligned, preferences)
  plus Drive/Hub-hosted training shards and checkpoints pushed every N steps with optimizer
  state.

## Testing Decisions

- **Philosophy**: a good test exercises external behavior at a seam and asserts on observable
  artifacts or outputs — never on internals (no attention-weight inspection, no mocking of the
  model's own modules). If a refactor that preserves behavior breaks a test, the test was wrong.
- **Seam 1 — the pipeline-stage contract (primary)**: every stage is tested by running it for
  real at toy scale (2-layer model, ~100 fables, ~10 pairs, CPU, minutes) and asserting on the
  artifacts: shard token counts and mask spans, checkpoint resume-equality (train N steps ≈
  train k then resume N−k), ratings-record schema and cache idempotency, metrics-document
  completeness. The highest test in the suite is the full toy chain: pretrain → generate pairs
  → label (replayed) → derive preferences → reward model → PPO → evaluate.
- **Seam 2 — the model-forward contract**: black-box checks on the hand-written transformer:
  perturbing future tokens must not change current-position logits (causality); shapes and
  dtypes stable under fp16; masked prompt tokens contribute zero loss.
- **Labeler tests never call Claude**: the append-only ratings cache doubles as a replay
  fixture; tests run the labeling stage against recorded responses, and one contract test
  validates the strict-JSON schema against a live-recorded sample.
- **Sanity suite as pre-session gate**: tokenizer round-trip, forward contract, and
  overfit-one-batch run before any long Colab session — they are tests first, operational
  guards second.
- **Prior art**: none — the repository is greenfield; these seams establish the pattern, and
  the ideal-seam-count rule (as few as possible) is the standing bar for future additions.

## Out of Scope

- Fine-tuning any pretrained model, at any stage — the from-scratch claim is the project.
- Guaranteed robustness to arbitrary free-text instructions; the contract is the Canonical
  Prompt plus the paraphrase family, with held-out-template performance measured, not promised.
- Human preference labeling at training scale (crowdsourcing, classmates-as-labelers); the
  human Gold Set exists only as a documented fallback, built only if the professor requires it.
- The stretch ablations as commitments: PPO-vs-DPO comparison and the ~3M scaling sibling
  happen only if the schedule holds; the committed ablations are paraphrase on/off and the
  reward-model data curve.
- Safety alignment beyond the four Axes (toxicity filtering, jailbreak resistance) — the
  corpus is children's fables; the Rubric's age-fit Axis is the extent of it.
- Multi-language generation; regenerating or extending the source dataset.
- Production concerns: serving latency, autoscaling, model quantization for deployment beyond
  what the free CPU Space needs.
- Hyperparameter search beyond the named gates and the week-1 throughput benchmark.
- Issue-tracker workflow — no tracker exists yet; this PRD is filed as a document and should
  become issue #1 with the `ready-for-agent` label when the GitHub repo is created.

## Further Notes

- The schedule and its three go/no-go gates (week-1 T4 benchmark, week-4 labeler-audit +
  reward-model gate, week-5 PPO-or-DPO decision) live in the decision record and bind this PRD.
- **The one action item only the human can do**: ask the professor in week 1 whether AI
  preference feedback satisfies the course's RLHF requirement. The Gold Set fallback is cheap
  (~5 hours) but only if triggered early.
- The report and title must say RLAIF, framed as "the full RLHF pipeline with AI preference
  feedback" — honesty about the H is a graded strength, not a confession.
- Vocabulary in this PRD is normative and defined in the glossary: Fable, Moral, Element,
  FableSpec, Age Group, Canonical Prompt, Paraphrased Prompt, Base Model, Aligned Model, Axis,
  Rubric, Preference Pair, Aggregate Score, Calibration Set, AI Labeler, Gold Set.
