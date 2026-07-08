# Task 9 Report

Implemented `DeriveConfig` and the lazy `derive` stage registry entry, then added `src/tinyfables/stages/derive.py` as a torch-free stage that:
- reads canonical `main`/`ab` labels from the label cache
- skips exact ties
- emits deterministic `preferences.jsonl` rows with chosen/rejected texts, ratings, aggregates, and stable train/held-out split
- emits `sensitivity.json` from `weight_sensitivity`
- emits `derive_summary.json`
- writes `manifest.json` last

Added the replay fixtures `tests/fixtures/labels_replay.jsonl` and `tests/fixtures/pairs_replay.jsonl`, plus focused coverage in `tests/test_derive_stage.py` and `tests/test_config.py`.

Focused verification:

```bash
rtk env PYTHONPATH=src /Users/thanh/code/tinystories/.venv/bin/pytest tests/test_derive_stage.py tests/test_config.py -q
```

Result: `24 passed`.
