# Task 10 Report

Implemented `AuditConfig`, the lazy `audit` stage registry entry, and `src/tinyfables/stages/audit.py` as a torch-free stage that:
- reads the cached labels from `labels.jsonl`
- computes `position_flip_rate` over `main` vs `swap`
- computes `self_consistency` over `calib*` repeats
- writes `audit.json` with the two metrics plus the gate result
- writes `audit_report.md` with a compact table and verdict
- writes `manifest.json` last

Added `configs/audit_toy.yaml`, `configs/audit_full.yaml`, `tests/test_audit_stage.py`, and the `AuditConfig` default test in `tests/test_config.py`.

Focused verification:

```bash
rtk env PYTHONPATH=src /Users/thanh/code/tinystories/.venv/bin/pytest tests/test_audit_stage.py tests/test_config.py -q
```

Result: `25 passed`.

Append for review fix:
- `audit.json` now includes `gate.position_swap_review_flag`, set from `position_swap.n_flipped > 0`.
- `audit_report.md` now includes a position-swap verdict line that reads `REVIEW` when flips are observed, otherwise `no flips observed`.

Verification rerun:

```bash
rtk env PYTHONPATH=src /Users/thanh/code/tinystories/.venv/bin/pytest tests/test_audit_stage.py tests/test_config.py -q
```

Result: `25 passed in 0.06s`

Review-fix append:
- `audit.run()` now fails fast if `labels` is missing on disk or if `load_cache()` returns an empty cache.
- `gate.position_swap_review_flag` now uses `position_swap_review_threshold` and only evaluates when `n_pairs > 0`.
- `AuditConfig` now validates both `self_consistency_gate` and `position_swap_review_threshold` in `[0, 1]`.

Verification rerun:

```bash
rtk env PYTHONPATH=src /Users/thanh/code/tinystories/.venv/bin/pytest tests/test_audit_stage.py tests/test_config.py -q
```

Result: `27 passed in 0.06s`
