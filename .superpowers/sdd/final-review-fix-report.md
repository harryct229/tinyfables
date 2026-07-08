# Final Review Fix Report

## Scope

Fixed the final whole-branch review findings for issue 06 Track A in the feedback-data path.

## Fixes

1. **`label` stage completion and manifest semantics**
   - Built the current expected key set from `(pair_id, phase, order, model_version, prompt_version)`.
   - Summary now reports:
     - `n_expected_current`
     - `n_cached_current`
     - `n_pending_current`
     - `complete`
   - `manifest.json` is written only when the current cohort is complete.
   - Partial bounded runs keep `labels.jsonl` and `label_summary.json` but remove any stale manifest.
   - Tightened batch formation so a single Claude batch never contains the same `pair_id` twice, which is required by the stricter parser.

2. **Cohort validation for `derive` and `audit`**
   - Added `load_label_cohort()` in `tinyfables.labeler`.
   - If `label_summary.json` exists next to `labels.jsonl`, `derive`/`audit` now require:
     - `complete: true`
     - sibling `manifest.json`
     - records filtered to the summary cohort `(model_version, prompt_version)`
     - filtered-record count matching the summary cohort count
   - If no summary exists, replay fixtures still work, but mixed cohorts now fail clearly instead of being silently merged.

3. **Strict labeler parsing**
   - `parse_labeler_response()` now rejects:
     - duplicate `pair_id` rows
     - extra rows
     - any pair-id mismatch between request and response

4. **Single-source production weights**
   - Added a production-code test that scans `src/tinyfables` outside `feedback.py` for a duplicated literal weight mapping or aggregate formula.
   - Clarified in `docs/design.md` that `feedback.py` is the only production logic source; rubric/design mirrors are human-facing only.

5. **Low-risk follow-ups**
   - `pairgen` now fails fast when the source does not contain enough canonical prompts to satisfy `n_pairs`.
   - Added a direct `claude_runner` nonzero-exit test.
   - Added a CLI lazy-dispatch regression for `pairgen`, `label`, `derive`, and `audit`.
   - Removed the redundant `hf_split` assertion in `tests/test_full_configs.py`.

## Focused Verification

Command:

```bash
rtk env PYTHONPATH=src /Users/thanh/code/tinystories/.venv/bin/pytest tests/test_feedback.py tests/test_labeler.py tests/test_label_stage.py tests/test_derive_stage.py tests/test_audit_stage.py tests/test_pairgen_stage.py tests/test_full_configs.py tests/test_cli_dispatch.py -q
```

Output:

```text
..........................................................               [100%]
58 passed in 2.06s
```

## Full Verification

Command:

```bash
rtk env PYTHONPATH=src /Users/thanh/code/tinystories/.venv/bin/pytest -q
```

Output:

```text
........................................................................ [ 36%]
........................................................................ [ 72%]
......................................................                   [100%]
198 passed, 1 deselected in 5.96s
```

## Notes

- The stricter parser exposed a real scheduling bug: calibration/main interleaving could put the same `pair_id` into one model batch. `_batched()` now enforces per-request uniqueness.
- `derive` no longer selects a “latest” label across mixed cohorts; it either filters to the complete summarized cohort or raises.
