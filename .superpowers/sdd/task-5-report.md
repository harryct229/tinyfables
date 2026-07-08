# Task 5 Report: Eval report Markdown export

## RED
- Added `tests/test_report.py` with the brief's exact metrics fixture and assertions.
- Ran `pytest tests/test_report.py -v`.
- Result: `ModuleNotFoundError: No module named 'tinyfables.report'` during collection, as expected before implementation.

## GREEN
- Implemented `src/tinyfables/report.py` with `write_eval_report(out_dir, metrics) -> Path`.
- The writer is torch-free and writes `eval_report.md` under the provided output directory.
- It renders:
  - a perplexity summary line,
  - a generation-quality table,
  - an adherence-grid table in canonical / seen-template / held-out-template order,
  - a provenance block with truncated SHAs, seed, and versions.
- Re-ran `pytest tests/test_report.py -v`.
- Result: `1 passed`.
- Ran the full suite with `pytest`.
- Result: `131 passed, 1 deselected`.

## Files Changed
- `src/tinyfables/report.py`
- `tests/test_report.py`

## Tests
- `pytest tests/test_report.py -v`
- `pytest`

## Self-review
- The implementation stays minimal and offline, with no torch or model dependencies.
- Ordering of adherence rows is fixed and matches the brief.
- Float formatting is stable enough for the current report assertions, while still readable in Markdown.
- I did not modify unrelated tracked or untracked files.
