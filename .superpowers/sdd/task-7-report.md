# Task 7 report: `ppo` stage

## Summary

Implemented the gate-guarded PPO alignment stage (`src/tinyfables/stages/ppo.py`)
integrating TRL 1.8.0's experimental `PPOTrainer` with the hand-written `GPT`,
`PPOStageConfig` in `src/tinyfables/config.py`, registration in
`src/tinyfables/stages/__init__.py`, `configs/ppo_toy.yaml` /
`configs/ppo_full.yaml`, `tests/test_ppo_stage.py` (verbatim from the brief),
and `test_ppo_full_config_loads` in `tests/test_full_configs.py`. All tests
pass, including the real (non-mocked) TRL PPO training loop on CPU with a
toy model.

## What I implemented

- `PPOStageConfig` — copied verbatim from the brief (frozen dataclass, same
  fields, same `__post_init__` validation).
- Registry entry `"ppo": (PPOStageConfig, "tinyfables.stages.ppo")`.
- `src/tinyfables/stages/ppo.py` — gate-first (`assert_gate_passed` before any
  torch/trl import or `out_dir` write), builds train/eval `datasets.Dataset`s
  from `preferences.jsonl` (train split for rollout prompts, held-out split
  for length-drift probes), constructs two independent `ScoredModelAdapter`
  instances (reward_model, value_model), trains via `trl.experimental.ppo.PPOTrainer`,
  logs per-iteration KL/reward/loss + a length-drift probe into `ppo_curves.csv`
  via a `TrainerCallback.on_log` hook, saves the aligned policy with
  `save_pretrained(out_dir)`, writes `ppo_summary.json`, then `manifest.json` last.
- `configs/ppo_toy.yaml` — mirrors the test `_cfg`, pointing at
  `tests/fixtures/gate_pass.json` / `tests/fixtures/preferences_replay.jsonl`,
  with `runs/pretrain_toy` / `runs/reward_toy` / `runs/tokenizer_toy` as the
  toy-chain checkpoint dirs (consistent with `configs/pairgen_toy.yaml`'s
  naming convention; no test loads this YAML directly, only `_cfg` in
  `test_ppo_stage.py` is exercised).
- `configs/ppo_full.yaml` — copied verbatim from the brief, including the
  deliberately-honest `gate: runs/gate_base/gate.json` (today's real verdict
  is NO-GO) and `adr_decision: SET-ME-AT-FORK`.
- `tests/test_ppo_stage.py` — copied verbatim from the brief.
- `tests/test_full_configs.py::test_ppo_full_config_loads` — new test in the
  established style (see `test_margins_full_config_loads`), asserting the
  config loads and every field matches `ppo_full.yaml`, without asserting
  anything about `adr_decision` beyond "still SET-ME-AT-FORK" and the gate
  path being the honest `runs/gate_base/gate.json`.

## Deviations from the brief's stage code (with TRL-source justification)

Before touching TRL, I read the actually-installed source
(`.venv/lib/python3.14/site-packages/trl/experimental/ppo/ppo_trainer.py`,
`ppo_config.py`, `trl/experimental/utils.py`, `trl/trainer/base_config.py`)
per the task's Step 4 note to trust real source over memory. Two adaptations
to the brief's stage code came out of that reading, applied *before* first
running the test (so they never showed up as failures in the RED/GREEN log
below — the smoke test passed on the first real attempt):

1. **`eval_dataset` is now passed to `PPOTrainer`, built from the held-out
   probe prompts.** `PPOTrainer.__init__` (`ppo_trainer.py` lines ~546-552)
   unconditionally does:
   ```python
   self.eval_dataloader = DataLoader(
       self.eval_dataset, batch_size=args.per_device_eval_batch_size,
       collate_fn=self.data_collator, drop_last=True,
   )
   self.eval_dataloader = accelerator.prepare(self.eval_dataloader)
   ```
   with no `if self.eval_dataset is not None` guard. Passing `eval_dataset=None`
   (the brief's stage code omits the kwarg, so it defaults to `None`) crashes
   `torch.utils.data.DataLoader(None, ...)` at trainer construction, before
   any training happens. I built `eval_dataset` from the same held-out
   `probe_prompts` already computed for the length-drift baseline (same
   `_tokenize` mapping as the train set) — no new data invented, and since
   `num_sample_generations=0` the eval dataloader is never actually iterated
   (`generate_completions` is only called when `num_sample_generations > 0`),
   so this is purely a construction-time requirement, not a real eval loop.
2. **`gradient_checkpointing=False` explicit in `PPOConfig`.**
   `trl.trainer.base_config._BaseConfig` (which `PPOConfig` subclasses)
   defaults `gradient_checkpointing=True`. Our hand-written `GPT` does not set
   `supports_gradient_checkpointing = True`, so `PreTrainedModel.gradient_checkpointing_enable()`
   would raise `ValueError` if ever invoked. I confirmed by reading
   `PPOTrainer.__init__`/`.train()` that this custom trainer never calls
   `model.gradient_checkpointing_enable()` on the exercised path (it only
   touches `args.gradient_checkpointing_kwargs` for the reentrant-flag
   workaround), so this wasn't causing an actual crash — but leaving the
   default `True` is a live landmine against any future trl code path (or a
   trl upgrade) that does call it, and gradient checkpointing buys nothing at
   this model/batch scale. Disabling it removes the risk for zero cost.

No other changes were needed. The rest of the brief's stage code — the gate
check ordering, `ScoredModelAdapter` usage, `PPOConfig` field mapping, the
`LengthAlarmCallback` shape, artifact names, manifest inputs — worked exactly
as written against the real TRL 1.8.0 API.

## TDD evidence

RED (Step 2 — before `stages/ppo.py` existed, but after `PPOStageConfig` and
the registry entry were already added):

```
$ .venv/bin/pytest tests/test_ppo_stage.py -v
...
ERRORS
___________________ ERROR collecting tests/test_ppo_stage.py ___________________
ImportError while importing test module '.../tests/test_ppo_stage.py'.
tests/test_ppo_stage.py:15: in <module>
    from tinyfables.stages import ppo as ppo_stage
E   ImportError: cannot import name 'ppo' from 'tinyfables.stages'
=========================== short test summary info ============================
ERROR tests/test_ppo_stage.py
1 error in 1.81s
```

GREEN (Step 4 — after `stages/ppo.py` implemented, both TRL-source-informed
deviations applied up front):

```
$ .venv/bin/pytest tests/test_ppo_stage.py -v
tests/test_ppo_stage.py::test_ppo_refuses_without_gate_pass PASSED       [ 50%]
tests/test_ppo_stage.py::test_ppo_toy_smoke_trains_and_writes_artifacts PASSED [100%]
========================= 2 passed, 1 warning in 2.00s ===========================
```
(Sole warning: `torch.var()` degrees-of-freedom UserWarning from TRL's own
`val/ratio_var` metric computation on a 1-element stats tensor — internal to
trl, harmless, unrelated to our code.)

Full-suite regression (Step 5):

```
$ .venv/bin/pytest -q
258 passed, 1 deselected, 1 warning in 7.25s
```

(1 deselected = the pre-existing `network` marker, unrelated to this task.
258 = the prior 257 + the new `test_ppo_full_config_loads`.)

## Manual artifact inspection

Ran the toy scenario standalone (outside pytest) to inspect real output on
disk. `ppo_curves.csv` (2 rows, matching `num_total_batches=2` for
`total_episodes=4 / batch_size=2`):

```
episode,objective_kl,objective_scores,objective_rlhf_reward,objective_non_score_reward,loss_policy_avg,loss_value_avg,probe_mean_words,probe_mean_new_tokens,length_drift_pct,length_alarm
2,0.0,-0.9821195602416992,-0.9821195602416992,0.0,0.0,0.5549945831298828,4.5,12,0.0,False
4,-0.0019185543060302734,-1.0981613397598267,-1.0977776050567627,0.0003837108379229903,5.21540641784668e-08,0.5589196681976318,4.5,12,0.0,False
```

`ppo_summary.json` has `adr_decision`, `gate: "pass"`, `preferences_sha`,
`base_checkpoint_sha`, `reward_model_sha`, `length_alarm_triggered` (bool),
`baseline_probe`, `final_length_drift`. `manifest.json` has
`stage: "ppo"`, `config.adr_decision` echoed, and all 5 required `inputs`
keys (`preferences.jsonl`, `gate.json`, `base_model.safetensors`,
`tokenizer.json`, `reward_head.pt`). `out_dir` root has `config.json`,
`model.safetensors`, `generation_config.json` at the root (matches the
base-model run layout), plus an empty `trainer/` directory left by
`PPOConfig(output_dir=str(out_dir / "trainer"))` — TRL's own `os.makedirs`
side effect (harmless; `save_strategy="no"` means it's never populated; not
part of the manifest's artifact list).

The aligned model reloads and reports the correct `n_ctx` via
`GPT.from_pretrained(out)`, confirmed by the test.

## Files changed

- `src/tinyfables/config.py` — added `PPOStageConfig`.
- `src/tinyfables/stages/__init__.py` — registered `"ppo"`.
- `src/tinyfables/stages/ppo.py` — new stage implementation.
- `configs/ppo_toy.yaml`, `configs/ppo_full.yaml` — new.
- `tests/test_ppo_stage.py` — new (verbatim from brief).
- `tests/test_full_configs.py` — added `test_ppo_full_config_loads`.

## Self-review findings

- Verified the light import path stays light: `import tinyfables.cli,
  tinyfables.stages` pulls in neither `torch` nor `trl` nor `transformers`
  (the registry maps to a module-path string, imported lazily at dispatch).
- Verified `test_ppo_refuses_without_gate_pass`'s hard requirement — gate
  check happens before `out_dir.mkdir()` — by confirming `out_dir` is never
  created on the no-go path (test asserts `not (tmp_path / "out" /
  "manifest.json").exists()`; also manually confirmed no `out`/`out2`
  directories are created at all when the gate fails).
- Confirmed `trainer.policy_model` (saved via `save_pretrained`) is the same
  underlying `nn.Module` object whose parameters are updated during
  `PolicyAndValueWrapper`'s forward/backward passes — matches how TRL's own
  `PPOTrainer.save_model()` does it (`self.model = self.model.policy` before
  calling `super().save_model()`), so the saved checkpoint is genuinely the
  post-training policy, not the frozen initial one.
- No linter is configured in this repo (no ruff/flake8 config); skipped.
- Did not touch `.superpowers/sdd/task-1-report.md`, `task-3-report.md`,
  `task-5-report.md`, `task-6-report.md`, or the untracked
  `docs/superpowers/plans/2026-07-10-issue-08-aligned-model.md` — these were
  already modified/untracked before this task started (unrelated to Task 7)
  and are excluded from this commit.
- Overwrote a stale `task-7-report.md` that had been left over from an
  earlier (unrelated) task numbering, documenting a `pairgen` stage — that
  content did not belong to this task and is superseded by this report.

## Concerns

- None blocking. The two TRL-source deviations are narrow, load-bearing, and
  documented above with file/line justification from the installed package.
  The `trainer/` leftover empty directory is cosmetic.
- `configs/ppo_full.yaml`'s `gate: runs/gate_base/gate.json` intentionally
  points at the real, currently-failing gate record — running this stage
  today (`python -m tinyfables run ppo --config configs/ppo_full.yaml --out
  ...`) will correctly raise `GateError` and refuse to train, per design.

## Fix report

Fixed 1 Important + 3 Minor review findings on this task's landed code.

### 1. Important — length probe perturbed the training RNG stream

`probe_lengths` (`src/tinyfables/length_alarm.py`) calls
`generate_fable_ids` per prompt, which reseeds the GLOBAL `torch` RNG
(`torch.manual_seed(seed + i)`) for its own per-prompt determinism. Since
the PPO stage's `LengthAlarmCallback.on_log` runs the probe between
training updates, every post-probe PPO rollout was silently starting from
probe-derived RNG state instead of continuing the training stream.

Fixed centrally inside `probe_lengths` (covers current and future
callers, no public signature change): snapshot `torch.get_rng_state()`
(and, when `torch.cuda.is_available()`, `torch.cuda.get_rng_state_all()`)
on entry, restore both in a `finally` block so the probe is invisible to
the caller's RNG stream regardless of how it exits. The probe's own
internal determinism (`seed + i` per prompt) happens entirely inside the
snapshot/restore window and is unaffected.

Added `test_probe_lengths_does_not_perturb_caller_rng_stream` to
`tests/test_length_alarm.py`: seeds the global RNG to simulate a caller
mid-stream, captures `torch.get_rng_state()`, calls `probe_lengths`,
asserts `torch.equal(before, after)`, and separately re-asserts the
probe's own result is identical across two fresh calls (determinism
intact).

### 2. Minor — CSV handle leak on exception

`curves_fh` in `src/tinyfables/stages/ppo.py` was opened before
`trainer.train()` with no enclosing `try`/`finally`, so a training
exception would leak the open file handle. Wrapped `PPOTrainer(...)`
construction + `trainer.train()` in `try: ... finally: curves_fh.close()`.
Manifest-last semantics are preserved: `summary`/`manifest.json` writes
still happen only after the `try/finally` block returns normally, so an
exception during training still yields no summary/manifest — unchanged
behavior, just no more leaked file descriptor.

### 3. Minor — dead assignment

`gate = assert_gate_passed(cfg.gate)` in `src/tinyfables/stages/ppo.py`
bound a return value that was never read. Changed to a bare
`assert_gate_passed(cfg.gate)` call.

### 4. Minor — missing config guard

Added to `PPOStageConfig.__post_init__` in `src/tinyfables/config.py`:
`if not self.response_length < self.n_ctx: raise ValueError("response_length must be < n_ctx")`.
Verified both real configs already satisfy this (`ppo_toy.yaml`:
12 < 128; `ppo_full.yaml`: 320 < 1024) and the test fixture's `_cfg` in
`tests/test_ppo_stage.py` (12 < 128) is unaffected.

Added `test_ppo_stage_config_rejects_response_length_ge_n_ctx` to
`tests/test_config.py`, following the file's existing
`pytest.raises(ValueError, match=...)` style used for the other
`__post_init__` guards (e.g. `test_reward_train_config_rejects_invalid_values`).

### Constraints honored

- No public signature changes to `probe_lengths`/`length_drift`.
- No artifact schema / summary field changes (`ppo_summary.json`,
  `ppo_curves.csv`, `manifest.json` field sets are untouched).
- No existing test weakened — all pre-existing assertions in
  `tests/test_length_alarm.py`, `tests/test_ppo_stage.py`,
  `tests/test_config.py` are unchanged; only new tests were added.

### Covering-test evidence

```
$ .venv/bin/pytest tests/test_length_alarm.py tests/test_ppo_stage.py tests/test_config.py -q
................................                                         [100%]
=============================== warnings summary ===============================
tests/test_ppo_stage.py::test_ppo_toy_smoke_trains_and_writes_artifacts
  .../trl/experimental/ppo/ppo_trainer.py:903: UserWarning: var(): degrees of freedom is <= 0. ...
-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
32 passed, 1 warning in 2.17s
```

### Full-suite regression

```
$ .venv/bin/pytest -q
........................................................................ [ 27%]
........................................................................ [ 55%]
........................................................................ [ 83%]
............................................                             [100%]
=============================== warnings summary ===============================
tests/test_ppo_stage.py::test_ppo_toy_smoke_trains_and_writes_artifacts
  .../trl/experimental/ppo/ppo_trainer.py:903: UserWarning: var(): degrees of freedom is <= 0. ...
-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
260 passed, 1 deselected, 1 warning in 7.43s
```

(1 deselected = pre-existing `network` marker, unrelated. 260 = prior 258
+ the 2 new tests added by this fix pass.)

### Files changed in this fix pass

- `src/tinyfables/length_alarm.py` — RNG snapshot/restore in `probe_lengths`.
- `src/tinyfables/stages/ppo.py` — `try/finally` around trainer
  construction+train to close `curves_fh`; dropped dead `gate =` binding.
- `src/tinyfables/config.py` — new `response_length < n_ctx` guard in
  `PPOStageConfig.__post_init__`.
- `tests/test_length_alarm.py` — new RNG-isolation test.
- `tests/test_config.py` — new guard-rejection test.
