# Task 1 Report: Text Metrics for TinyFables Eval Suite

## RED
- Added `tests/test_eval_metrics.py` first and ran `./.venv/bin/pytest tests/test_eval_metrics.py -v`.
- Expected failure observed:
  - `ModuleNotFoundError: No module named 'tinyfables.eval_metrics'`

## GREEN
- Implemented `src/tinyfables/eval_metrics.py` with stdlib-only helpers:
  - `element_adherence(fable, spec)`
  - `distinct_n(text, n)`
  - `repetition_rate(text, n=4)`
  - `length_stats(texts)`
- Re-ran the focused test suite:
  - `./.venv/bin/pytest tests/test_eval_metrics.py -v`
  - Result: `4 passed`
- Re-ran the full suite:
  - `./.venv/bin/pytest`
  - Result: `121 passed, 1 deselected`

## Files Changed
- `src/tinyfables/eval_metrics.py`
- `tests/test_eval_metrics.py`

## Notes
- The helper stays torch-free and uses only standard library imports plus `tinyfables.prompts.FableSpec`.
- `element_adherence` intentionally excludes `moral` from scoring, matching the brief.
- Length statistics are based on word-token counts derived from the same lightweight tokenizer used by the helper.

## Self-Review
- The implementation is small, deterministic, and offline.
- The behavior matches the brief and the tests pin the exact expected outputs.
- Residual risk is low; the only intentional simplification is substring-based adherence scoring, which is exactly what the task asked for.
