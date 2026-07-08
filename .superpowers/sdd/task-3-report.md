# Task 3 Report: Periodic Checkpoint Save/Find/Load/Prune (issue 04)

## Summary

Implemented a checkpoint module for training resilience on Colab. The module provides four functions to save, find, load, and prune training checkpoints with a completion-marker pattern that ensures interrupted writes are never mistaken for complete state.

## What Was Built

**Module**: `src/tinyfables/checkpoint.py` (78 lines)

- `save_checkpoint(model, optimizer, scaler, step: int, ckpt_dir) -> Path`
  - Saves model via `save_pretrained()` (safetensors + config)
  - Saves optimizer + scaler + step to `optimizer.pt`
  - Writes `checkpoint_state.json` LAST as completion marker
  - Returns path to checkpoint directory `step_{step}/`

- `find_latest_checkpoint(ckpt_dir) -> Path | None`
  - Scans for complete checkpoints (have all 3 files + marker)
  - Returns highest-step checkpoint path, or None if none exist

- `load_checkpoint(ckpt_path, model_cls, device) -> tuple[model, dict]`
  - Restores model from pretrained state to device
  - Loads optimizer state dict with step
  - Returns (model, state_dict)

- `prune_checkpoints(ckpt_dir, keep_last_k: int) -> None`
  - Deletes all but newest k complete checkpoints
  - Gracefully handles missing/corrupted directories

**Test Suite**: `tests/test_checkpoint.py` (62 lines)

- 5 test cases covering save→find, incomplete detection, empty dirs, load with state restoration, and pruning

## TDD Evidence

**RED** — confirmed module did not exist:

```
$ .venv/bin/pytest tests/test_checkpoint.py -v

ERROR collecting tests/test_checkpoint.py
E   ModuleNotFoundError: No module named 'tinyfables.checkpoint'
```

**GREEN** — after implementation:

```
$ .venv/bin/pytest tests/test_checkpoint.py -v

test_save_then_find_latest PASSED              [ 20%]
test_incomplete_checkpoint_ignored PASSED      [ 40%]
test_find_returns_none_when_empty PASSED       [ 60%]
test_load_restores_step_and_optimizer PASSED   [ 80%]
test_prune_keeps_last_k PASSED                 [100%]

5 passed in 1.14s
```

**Full suite**:

```
$ .venv/bin/pytest -q
93 passed, 1 deselected in 2.83s
```

## Implementation Note

The brief's code pattern for saving disabled GradScaler state was refined:

- A disabled `torch.amp.GradScaler("cpu", enabled=False)` returns an empty dict `{}` from `state_dict()`
- Modified `save_checkpoint` to convert empty scaler dicts to `None` for cleaner restoration
- This ensures the loaded state has `scaler=None` when the original was disabled, matching test expectations:

```python
scaler_state = None
if scaler is not None:
    scaler_dict = scaler.state_dict()
    scaler_state = scaler_dict if scaler_dict else None  # Empty dict → None
```

## Files Changed

- `src/tinyfables/checkpoint.py` — +78 lines (4 public functions + 2 internal helpers)
- `tests/test_checkpoint.py` — +62 lines (5 test cases)

Both files created from scratch (no existing code modified).

## Commit

`9b58659` — "feat: periodic checkpoint save/find/load/prune with completion marker (issue 04)"
(2 files changed, 140 insertions(+))

## Concerns

None. The implementation:
- Follows TDD strictly (RED → GREEN)
- Passes all 5 targeted tests
- Passes full suite (93 tests, no regressions)
- Integrates cleanly with existing `tinyfables.model.GPT` and torch APIs
- Ready for Task 8 (pretrain stage wiring)
# Task 3 Report — Fable-token perplexity

## Implemented
- Added `src/tinyfables/perplexity.py` with `fable_token_perplexity(model, tokenizer, rows, n_ctx, device, max_rows)`.
- The helper encodes `prompt + fable + EOT`, masks prompt tokens with `-100`, scores only completion tokens, skips rows longer than `n_ctx`, and returns `(mean_loss, perplexity)`.
- Added `tests/test_perplexity.py` covering:
  - perplexity equals `exp(mean_loss)` and is finite on the fixture corpus,
  - completion-only loss matches an independent manual cross-entropy calculation.

## TDD Evidence

### RED
Command: `.venv/bin/pytest tests/test_perplexity.py -v`

Expected failure occurred before implementation:
`ModuleNotFoundError: No module named 'tinyfables.perplexity'`.

### GREEN
Focused command: `.venv/bin/pytest tests/test_perplexity.py -v`
Result: `2 passed`.

Full command: `.venv/bin/pytest`
Result: `128 passed, 1 deselected`.

## Files Changed
- `src/tinyfables/perplexity.py`
- `tests/test_perplexity.py`

## Self-Review
- The helper mirrors training-time masking by ignoring prompt tokens and scoring only `fable + EOT`.
- The implementation is deterministic, uses `torch.no_grad()`, and returns `nan, nan` when no rows are usable.
- One small divergence from the brief sample: I keep the `len(ids) < 2` guard to avoid empty cross-entropy when an input row produces no next-token target.
