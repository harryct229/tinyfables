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
