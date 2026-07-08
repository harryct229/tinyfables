# Task 4 Report: EvalConfig for TinyFables Eval Suite

## RED
- Added the new eval-config tests to `tests/test_config.py` first.
- Ran the focused test command before implementing the dataclass:
  - `./.venv/bin/pytest tests/test_config.py -k eval_config -v`
- Expected failure observed:
  - `ImportError: cannot import name 'EvalConfig' from 'tinyfables.config'`

## GREEN
- Implemented `EvalConfig` in `src/tinyfables/config.py` as a frozen dataclass with the brief's exact fields and defaults.
- No loader changes were needed because existing `_build` already handles nested `source` mappings and unknown-key rejection.
- Re-ran the focused test command:
  - `./.venv/bin/pytest tests/test_config.py -k eval_config -v`
  - Result: `2 passed`
- Re-ran the full suite once:
  - `./.venv/bin/pytest`
  - Result: `130 passed, 1 deselected`

## Files Changed
- `src/tinyfables/config.py`
- `tests/test_config.py`
- `.superpowers/sdd/task-4-report.md`

## Tests
- Focused: `./.venv/bin/pytest tests/test_config.py -k eval_config -v`
- Full suite: `./.venv/bin/pytest`

## Self-Review
- The change is narrowly scoped to config dataclasses and tests, matching the task boundary.
- Unknown-key rejection still flows through the existing `load_config` / `_build` path.
- No evaluation-stage registry work was added here; that remains for Task 6.
