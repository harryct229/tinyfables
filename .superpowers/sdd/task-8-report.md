Task 8 report

- Added `LabelConfig` with the specified defaults and validation for `swap_fraction` and `batch_size`.
- Implemented `tinyfables.stages.label` with deterministic `plan_work`, injected `runner`, append-only cache resume, stable `ratings_0`/`ratings_1` storage, and manifest-last completion.
- Preserved lazy stage registration by adding only the registry entry in `tinyfables.stages.__init__`.
- Added `configs/label_toy.yaml` and `configs/label_full.yaml`.
- Added focused tests for config defaults, registry wiring, cache/summary/manifest outputs, swap/calibration scheduling, full resume, and stable-order rating storage.
- Verified with `rtk env PYTHONPATH=src /Users/thanh/code/tinystories/.venv/bin/pytest tests/test_label_stage.py tests/test_config.py -q` -> `24 passed`.
