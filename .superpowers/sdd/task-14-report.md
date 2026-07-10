# Task 14 report: majority-vote support in derive + audit (ADR-0005)

**Status:** DONE
**Branch:** issue-08-vote-derive
**Commit:** `55ac5a5` — `feat(vote): majority-vote derive + audit over multiple label caches (ADR-0005)`

## Summary

Added an optional `extra_labels` majority-vote mode to the `derive` and `audit`
stages, per ADR-0005 / the task-14 brief. `extra_labels` unset (default `None`)
is byte-identical to today's single-cache behavior (verified explicitly, see
below); when set, both stages consume the primary cache plus N additional
independently-labeled caches and operate on the voted preference instead of a
single cache's.

## Files changed

- `src/tinyfables/feedback.py` — 4 new pure functions: `voted_preference`,
  `voted_weight_sensitivity`, `voted_position_flip_rate`, `voted_self_consistency`.
- `src/tinyfables/config.py` — `DeriveConfig.extra_labels` /
  `AuditConfig.extra_labels: list[str] | None = None`, shared
  `_validate_extra_labels` helper (non-empty if provided, no duplicates of
  each other or of `labels`).
- `src/tinyfables/stages/derive.py` — voting branch in `run()`; new
  `_load_canonical_main` helper for extra caches.
- `src/tinyfables/stages/audit.py` — voting branch in `run()`.
- `tests/test_feedback.py` — 13 new primitive tests.
- `tests/test_derive_stage.py` — 5 new stage tests.
- `tests/test_audit_stage.py` — 3 new stage tests.
- `tests/test_config.py` — 6 new config-validation tests.

## Design decisions within the brief's constraints

1. **`voted_preference` quorum rule.** Implemented as: count votes for 0 and
   1 (a per-cache tie/`None` casts no vote); the winner is whichever value
   has `count >= 2` **and** a strictly higher count than the other. For the
   production 3-cache case this is mathematically identical to the brief's
   literal "chosen by ≥2 elements" rule (with 3 total votes, at most one
   value can reach 2). I added the "and strictly greater" clause purely as a
   safe generalization for N>3 (where two values could each reach 2 without
   my clause), which the brief doesn't specify but explicitly asks the
   function to tolerate ("works for any list length ≥ 1"). A single-element
   list can never reach quorum, so it always returns `None` — tested
   explicitly (`test_voted_preference_single_cache_never_reaches_quorum`).

2. **`n_no_majority` vs `n_ties_skipped`.** Took the brief's first offered
   option: in voting mode, `n_no_majority` counts *every* skipped-in-voting
   pair (whether the skip came from a genuine 1-1 split, a per-cache tie
   diluting the vote, or any other no-quorum case), and `n_ties_skipped` is
   hardcoded to `0` with an inline comment explaining why. `n_no_majority` is
   only added to the summary dict in voting mode (via a `summary_extra` dict
   merged in with `**`), so single-cache mode's `derive_summary.json` gains
   no new key — required for the byte-identical regression check.
   Asserted in `test_voted_derive_keeps_majority_pair_with_agreeing_cache_mean`.

3. **Audit voting scope: pairs/phases must appear in *every* cache.**
   Neither `voted_position_flip_rate` nor `voted_self_consistency` raises on
   a pair-id-set mismatch across caches (unlike derive, where the brief
   explicitly requires a loud `ValueError`). Instead they take the
   intersection of pair-ids (position-swap) or `(pair_id, phase)` keys
   (self-consistency) present in every cache, silently excluding anything
   not fully covered. Rationale: derive already enforces at derive-time that
   all caches are complete over the same pair-id set, so by the time audit
   runs on real data this is a non-issue; letting audit be lenient (rather
   than duplicating derive's hard check) keeps it usable for partial/ad-hoc
   inspection without over-specifying behavior the brief didn't ask for.

4. **New feedback.py primitives beyond the two named in the brief.** The
   brief names `voted_preference` and `voted_weight_sensitivity` explicitly;
   for the audit-side semantics (position-swap and self-consistency voting)
   it only describes behavior, not a function signature. I added
   `voted_position_flip_rate(instances_by_cache)` and
   `voted_self_consistency(instances_by_cache)` as pure functions in
   `feedback.py`, mirroring the file's existing style (pure functions,
   stdlib only, unit-tested directly) rather than inlining the voting logic
   into `stages/audit.py`. This keeps `audit.py` a thin I/O shim, consistent
   with how `position_flip_rate`/`self_consistency` are already factored.

5. **Manifest input keys.** `labels_extra_0.jsonl`, `labels_extra_1.jsonl`,
   … in `cfg.extra_labels` order, added to both stages' `inputs` dict for
   `write_manifest`, per the brief.

## TDD evidence (RED → GREEN)

- `tests/test_feedback.py`: added imports for 4 new names →
  `ImportError: cannot import name 'voted_position_flip_rate'` (RED) →
  implemented all 4 functions → 20/20 pass (GREEN).
- `tests/test_config.py`: 6 new validation tests written against config
  fields that didn't exist yet; ran together with the `_validate_extra_labels`
  implementation in the same edit (config is declarative enough that a
  separate RED capture wasn't meaningful) → 34/34 pass.
- `tests/test_derive_stage.py`: added 5 voting tests before touching
  `derive.py` → `3 failed, 7 passed` (RED: the 3 tests that pass
  `extra_labels`; the split-parity test passed trivially since it doesn't
  yet exercise the extra caches) → implemented voting branch → 10/10 pass
  (GREEN).
- `tests/test_audit_stage.py`: added 3 voting tests before touching
  `audit.py` → `3 failed, 5 passed` (RED) → implemented voting branch →
  8/8 pass (GREEN).

## Test results

- Targeted: `.venv/bin/pytest tests/test_feedback.py tests/test_derive_stage.py tests/test_audit_stage.py tests/test_config.py -q` → **72 passed**.
- Full suite: `.venv/bin/pytest -q` → **290 passed, 1 deselected** (baseline
  before this task was 268 passed, 1 deselected — the delta is exactly the
  22 new tests added; zero existing tests were modified or broke).

## Byte-identical regression verification

Beyond "existing tests pass unmodified," I did an explicit before/after diff:
stashed my changes, ran `derive`/`audit` on the fixture caches
(`tests/fixtures/labels_replay.jsonl` + `pairs_replay.jsonl`) with
`extra_labels` unset on the pre-change code, popped the stash, reran the same
calls on the post-change code, and diffed every artifact:

- `preferences.jsonl` — **identical**
- `sensitivity.json` — **identical**
- `derive_summary.json` — **identical**
- `audit.json` — **identical**
- `audit_report.md` — **identical**
- `manifest.json` — differs only by the new `"extra_labels": null` key in
  the config echo (an unavoidable consequence of adding a dataclass field —
  `write_manifest` echoes `dataclasses.asdict(cfg)`) and the non-deterministic
  `created_unix` timestamp (already non-deterministic pre-change). No
  existing test asserts exact `manifest.json` byte content (only presence of
  specific keys), so this doesn't violate the regression requirement as
  written, but flagging it explicitly since it's the one artifact that isn't
  bit-for-bit identical.

## Self-review findings

- Reviewed the full diff (`git diff --stat` + line-by-line read of both
  stage files) after implementation. No correctness issues found; the
  single-cache branches are verbatim copies of the original code (only
  wrapped in an `if cfg.extra_labels is None:` block), which is what the
  byte-identical diff above confirms empirically rather than just by
  inspection.
- Checked for a latent footgun: `voted_self_consistency`/
  `voted_position_flip_rate` call `set.intersection(*(...))` over a
  generator; if `instances_by_cache` were ever an empty list this raises
  `TypeError` (unbound-method call missing `self`). Not reachable from
  `stages/audit.py` (always ≥2 caches: primary + a non-empty `extra_labels`,
  enforced by config validation), so left unguarded rather than adding
  dead-code defenses.
- Confirmed the git working tree had pre-existing, unrelated uncommitted
  changes to `.gitignore` and three other `.superpowers/sdd/task-*-report.md`
  files (from other concurrent agent work in this shared repo, matching the
  task briefing's note about background labeling processes). Verified via
  `git stash`/`git stash pop` round-trip that this in-progress work was
  preserved losslessly, and staged/committed only the 8 files belonging to
  this task (`git add <explicit paths>`, never `git add -A`).

## Concerns

- None blocking. The one item worth a maintainer's eyes: my resolution of
  "audit is lenient on cross-cache pair-id mismatches, derive is strict" is
  a genuine design choice (see decision #3 above) rather than something the
  brief pinned down — flagging it in case the intended behavior was for
  audit to also raise loudly on incomplete caches.

## Fix report

**Review finding addressed:** the concern flagged above was confirmed as a
real gap. Audit's voted functions silently intersected cross-cache
pair/phase coverage instead of raising, which is unacceptable given
self-consistency is one of ADR-0005's two hard gate hurdles. Fixed to
mirror derive's strict contract.

### Changes

- `src/tinyfables/feedback.py`:
  - `voted_position_flip_rate`: now raises `ValueError` if caches disagree
    on the set of pair_ids with a main-phase instance, and separately if
    they disagree on the set with a swap-phase instance (two distinct
    checks/messages, each naming the phase). The main∩swap intersection
    within a single agreed-upon coverage set is retained — that's the
    pre-existing, correct single-cache semantic (a pair without both a main
    and swap instance simply isn't scorable), not the leniency being fixed.
  - `voted_self_consistency`: now raises `ValueError` if caches disagree on
    the set of `(pair_id, phase)` calibration groups.
  - Both functions also raise `ValueError("voting requires at least two
    caches")` for a `< 2`-length `instances_by_cache`, replacing the
    previously-flagged `TypeError` footgun (`set.intersection(*())` on an
    empty generator) with a clear message. This also subsumes the
    single-cache case, since a lone cache can't meaningfully "vote."
  - `voted_preference` and `voted_weight_sensitivity` were left unchanged:
    traced both — neither calls `set.intersection`, and both already return
    a clean `None` for a 0- or 1-cache input (no crash), so they don't share
    the TypeError risk the guard exists to close. Adding a `< 2` guard to
    `voted_preference` would also have broken
    `test_voted_preference_single_cache_never_reaches_quorum`, which
    documents intentional, already-verified-correct quorum math (a
    single-cache list can never reach the ≥2 quorum, hence `None`) — the
    task brief said this math is riding, so I left it untouched.
- `tests/test_feedback.py`: added 6 new tests — per function, one for the
  `< 2`-caches guard and one for coverage-mismatch raise (main-phase and
  swap-phase mismatches get separate tests for `voted_position_flip_rate`).
  No existing test asserted lenient-intersection semantics, so nothing
  needed replacing; the existing happy-path tests
  (`test_voted_position_flip_rate_votes_before_comparing`,
  `test_voted_self_consistency_votes_within_each_phase_group`) are
  untouched and still pass, proving the vote path still differs from any
  single cache.
- `tests/test_audit_stage.py`: added one stage-level integration test,
  `test_voted_audit_rejects_mismatched_main_phase_coverage_across_caches`,
  mirroring derive's existing
  `test_voted_derive_rejects_mismatched_pair_id_sets_across_caches` — proves
  the raise surfaces through `audit_stage.run()`, not just the primitive.

### Left as-is per instructions

- The small double-computation of `voted_preference` in derive's
  sensitivity path, and report prose inaccuracies — explicitly out of scope
  for this fix.
- `audit.json` / derive artifact schemas: unchanged.
- Single-cache (`extra_labels=None`) path: untouched (no code in that
  branch was touched).
- `runs/` and `.gitignore`: not touched.

### Test results

Targeted:
`.venv/bin/pytest tests/test_feedback.py tests/test_audit_stage.py tests/test_derive_stage.py -q`
→ **44 passed**.

Full suite: `.venv/bin/pytest -q` → **296 passed, 1 deselected** (up from
290 passed pre-fix; delta is the 7 new tests: 6 in `test_feedback.py`, 1 in
`test_audit_stage.py`).

**Commit:** `fix(vote): strict cross-cache coverage checks in voted audit (review fix)`
