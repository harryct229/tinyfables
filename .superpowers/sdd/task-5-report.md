# Task 5 Report: `margins` stage — RM-accuracy-by-margin diagnostic

(Note: this file previously held a report for an unrelated, earlier "Task 5" — labeler-parser
work from a different project phase. That content is superseded below; the labeler work itself
is untouched in the codebase, only this report file was overwritten per task instructions.)

## What I implemented

Exactly per `.superpowers/sdd/task-5-brief.md`:

1. **`src/tinyfables/config.py`** — added `MarginsConfig` (frozen dataclass) with fields
   `preferences`, `reward_model_dir`, `tokenizer_dir`, `n_ctx=1024`, `batch_size=16`,
   `bucket_edges=None` (defaults to `[0.1, 0.3, 0.6, 1.0]` in `__post_init__`), `device="auto"`,
   `seed=0`. Validation: `n_ctx > 0`, `batch_size > 0`, `bucket_edges` positive and strictly
   ascending (via `sorted(edges) != list(edges)` check).

2. **`src/tinyfables/stages/__init__.py`** — registered `"margins": (MarginsConfig,
   "tinyfables.stages.margins")` in `REGISTRY` (alongside the `MarginsConfig` import), so the
   stage is automatically available via `python -m tinyfables run margins ...` (CLI derives
   `choices` from `sorted(REGISTRY)`).

3. **`src/tinyfables/stages/margins.py`** (new) — the diagnostic stage. Loads the held-out
   preference split via `load_preferences(..., split="held_out")`, batches through
   `collate_preference_batch` + the loaded `RewardModel`, computes `chosen_score > rejected_score`
   per pair, buckets by `aggregate_chosen - aggregate_rejected` margin using the half-open
   `[edges[i], edges[i+1])` / `[edges[-1], inf)` scheme (margins below `edges[0]` fall into the
   first bucket, so nothing is dropped), and writes `margins.json`, `margins_report.md`, and
   `manifest.json` (via `write_manifest`, written last per the stage contract).

4. **`configs/margins_full.yaml`** (new) — production config pointing at
   `runs/derive_base/preferences.jsonl`, `runs/rm_hub` (snapshot of the Hub RM), and
   `runs/tokenizer_full`, with `bucket_edges: [0.1, 0.3, 0.6, 1.0]`.

5. **`tests/test_margins_stage.py`** (new) — the brief's two tests verbatim:
   - `test_margins_stage_buckets_held_out_accuracy`: builds a tiny tokenizer + untrained
     `RewardModel` from fixtures, runs the stage, asserts `n_held_out >= 1`, accuracy bounds,
     bucket `n` sums to `n_held_out`, per-bucket accuracy is `None` iff `n == 0`, and that
     `margins_report.md` / `manifest.json` exist.
   - `test_margins_config_rejects_bad_edges`: non-ascending `bucket_edges` raises `ValueError`.

6. **`tests/test_full_configs.py`** — added `test_margins_full_config_loads`, mirroring this
   file's existing per-stage convention (one hardcoded test per `configs/*_full.yaml`, e.g.
   `test_reward_and_gate_full_configs_load`). See "Step 4 note" below for why this was added
   rather than modifying an enumeration.

## Step 4 note (repo convention check)

I read `tests/test_full_configs.py` before finishing, as instructed. It does **not** glob
`configs/*_full.yaml`; it has one dedicated test function per stage's full config
(`test_prep_full_trains_with_augmentation`, `test_reward_and_gate_full_configs_load`, etc.), each
hardcoding the expected field values. Since there's no enumeration table to add
`margins_full.yaml` to, I followed the established convention instead and added
`test_margins_full_config_loads`, which loads `configs/margins_full.yaml` via `load_config` and
asserts every field matches the YAML (`preferences`, `reward_model_dir`, `tokenizer_dir`, `n_ctx`,
`batch_size`, `bucket_edges`, `device`, `seed`). This keeps `margins_full.yaml` under the same
regression coverage every other full config gets.

## TDD evidence

**RED** — `MarginsConfig` didn't exist yet:

```
$ .venv/bin/pytest tests/test_margins_stage.py -v
...
ImportError: cannot import name 'MarginsConfig' from 'tinyfables.config'
Interrupted: 1 error during collection
```

**GREEN** — after implementing `MarginsConfig`, `stages/__init__.py` registration, and
`stages/margins.py`:

```
$ .venv/bin/pytest tests/test_margins_stage.py tests/test_config.py tests/test_cli_dispatch.py -v
...
tests/test_margins_stage.py::test_margins_stage_buckets_held_out_accuracy PASSED
tests/test_margins_stage.py::test_margins_config_rejects_bad_edges PASSED
[... 27 more PASSED ...]
============================== 29 passed in 1.60s ==============================
```

`tests/test_full_configs.py` (including the new `test_margins_full_config_loads`):

```
$ .venv/bin/pytest tests/test_full_configs.py -v
...
tests/test_full_configs.py::test_margins_full_config_loads PASSED
============================== 8 passed in 0.03s ===============================
```

Full suite:

```
$ .venv/bin/pytest -q
........................................................................ [ 28%]
........................................................................ [ 56%]
........................................................................ [ 84%]
.......................................                                  [100%]
255 passed, 1 deselected in 6.34s
```

(Same "1 deselected" as pre-existing on this branch — an unrelated slow/marked test, not
introduced by this change.)

## Files changed (this commit)

- `src/tinyfables/config.py` — added `MarginsConfig`
- `src/tinyfables/stages/__init__.py` — registered `margins` stage
- `src/tinyfables/stages/margins.py` — new stage implementation
- `configs/margins_full.yaml` — new production config
- `tests/test_margins_stage.py` — new stage tests
- `tests/test_full_configs.py` — added `test_margins_full_config_loads`

Commit: `cd640cc feat(margins): RM-accuracy-by-margin diagnostic stage`

Note: three items were already modified/untracked in the working tree before this task started
(`.superpowers/sdd/task-1-report.md`, `.superpowers/sdd/task-3-report.md`,
`docs/superpowers/plans/2026-07-10-issue-08-aligned-model.md` — pre-existing from other
tasks/agents on this branch, per `git status` at session start). I left them untouched and out of
this commit; they remain unstaged in the working tree.

## Self-review findings

- **Bucket boundary logic correct**: `_bucket_index` scans all edges and keeps the last one
  `<= margin`, giving `[edges[0], edges[1])`, …, `[edges[-1], inf)`, with `idx` initialized to 0
  so margins `< edges[0]` still land in bucket 0 (verified by the `sum(b.n) == n_held_out`
  assertion in the test, which would fail if any row were dropped).
- **`raw` dict keyed by `pair_id`** filters the same file for `split == "held_out"` independently
  of `load_preferences`; both come from the same source file and field, so they stay consistent
  as long as `pair_id` is unique per row — true of the fixture and of `derive`'s output contract
  (not re-verified here beyond what the brief specifies).
- **Manifest input paths** (`backbone/model.safetensors`, `reward_head.pt`) match exactly what
  `save_reward_model`/`load_reward_model` in `reward_model.py` and the existing `reward` stage's
  `write_manifest` call already use — no drift risk there.
- **`MarginsConfig` placement** in `config.py`: I inserted it directly before `_build` (i.e.,
  after `GateConfig`), following the brief's snippet placement rather than reordering to sit next
  to `RewardTrainConfig`. Purely cosmetic — dataclass position doesn't affect behavior — but
  flagging in case the team prefers grouping RM-related configs together.
- No dead code, no TODOs, no unused imports introduced. Ran `git diff` on every touched file
  before committing to confirm nothing accidental was staged (the pre-existing unrelated changes
  were explicitly excluded from `git add`).

## Concerns

None blocking. The implementation is copied verbatim from the brief (code, test, and YAML), and
all specified commands pass. The one judgment call — adding a test to `tests/test_full_configs.py`
rather than modifying a nonexistent enumeration — follows the file's own established pattern and
was explicitly anticipated by the brief's phrasing ("read that test and make sure
`configs/margins_full.yaml` passes it").
