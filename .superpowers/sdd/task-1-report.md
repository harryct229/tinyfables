# Task 1 Report: Aggregate Score + preference derivation

## Outcome
Implemented `tinyfables.feedback` as a torch-free preference module with the ADR-0003 axis order and weights, plus weighted score and tie-skipping preference derivation.

## RED Evidence
Initial focused test run failed as expected before the module existed:

```text
ModuleNotFoundError: No module named 'tinyfables.feedback'
```

Command:

```bash
/Users/thanh/code/tinystories/.venv/bin/pytest tests/test_feedback.py -q
```

## GREEN Evidence
Focused test:

```text
5 passed in 0.01s
```

Full suite:

```text
141 passed, 1 deselected in 4.84s
```

## Notes
- `AXES` is `("moral", "adherence", "coherence", "prose")`.
- `WEIGHTS` is `{"moral": 0.4, "adherence": 0.3, "coherence": 0.2, "prose": 0.1}`.
- `derive_preference` returns `0`, `1`, or `None` for exact ties.

## Review Fix
- Added a regression test for the mathematical-tie example:
  `{"moral": 1, "adherence": 1, "coherence": 1, "prose": 3}` vs
  `{"moral": 1, "adherence": 1, "coherence": 2, "prose": 1}` now returns `None`.
- Updated `derive_preference` to use `math.isclose(..., abs_tol=1e-12)` so float-rounding ties are skipped reliably.

## Verification
Required command attempted from the worktree:

```text
/Users/thanh/code/tinystories/.venv/bin/pytest tests/test_feedback.py -q
```

Result:

```text
ImportError: No module named 'tinyfables.feedback'
```

Worktree-local rerun:

```text
PYTHONPATH=src /Users/thanh/code/tinystories/.venv/bin/pytest tests/test_feedback.py -q
```

Result:

```text
6 passed in 0.00s
```

## Review Fix Follow-Up
- Replaced the tolerance-based tie check in `derive_preference` with exact `Decimal(str(...))` score comparison built from the existing `WEIGHTS` values.
- Kept exact mathematical ties skipped, while allowing close-but-distinct scores to produce a winner.
- Updated tests to cover both the mathematical tie case and a near-equal non-tie.

## Verification Rerun
Command:

```text
PYTHONPATH=src /Users/thanh/code/tinystories/.venv/bin/pytest tests/test_feedback.py -q
```

Result:

```text
......                                                                   [100%]
6 passed in 0.00s
```
