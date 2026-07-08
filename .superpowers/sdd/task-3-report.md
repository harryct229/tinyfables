# Task 3 Report: Labeler audit metrics

## Outcome
Implemented `position_flip_rate` and `self_consistency` in `tinyfables.feedback` using the cache-record shape from the brief.

## Verification
Focused worktree test:

```text
10 passed in 0.01s
```

Command:

```text
rtk env PYTHONPATH=src /Users/thanh/code/tinystories/.venv/bin/pytest tests/test_feedback.py -q
```

## Notes
- `position_flip_rate` pairs `phase=="main"` with `phase=="swap"` for the same `pair_id` and counts derived-preference disagreements.
- `self_consistency` uses calibration instances with `phase` starting with `calib`, computes pairwise agreement across repeats, and counts unanimous calibration pairs.
