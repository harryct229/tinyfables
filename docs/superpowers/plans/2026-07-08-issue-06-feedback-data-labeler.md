# Feedback Data: Rubric, Pairs, AI Labeler, Audits (Issue 06) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the full feedback-data path — the human-authored anchored `RUBRIC.md`, a `pairgen` stage that samples ~2,000 Preference Pairs from the Base Model, a resumable AI Labeler (`claude -p`) whose append-only JSONL cache doubles as the offline replay fixture, and `derive`/`audit` stages that turn per-Axis 1–5 ratings into a stable preferences artifact plus the weight-sensitivity table, position-swap flip rate, and Calibration-Set self-consistency.

**Architecture:** One torch-free logic module (`feedback.py`: fixed Aggregate-Score weights, preference derivation, weight-sensitivity, audit metrics) and one torch-free labeler module (`labeler.py`: prompt building, strict-JSON parsing, append-only cache, and the *single* `claude -p` subprocess boundary — injected as a `runner` callable so tests never touch the network). Four new stages mirror the existing `evaluate` pattern: `pairgen` (torch, generation) and torch-free `label`/`derive`/`audit`, all registered lazily. Track A is code, toy-tested offline via a fake runner + recorded fixtures; Track B runs real generation, the real labeling run on the Mac, then audits + derivation, recording the gate numbers in `docs/design.md`.

**Tech Stack:** Python 3.10+, PyTorch (pairgen generation only), tokenizers, transformers (`GPT.from_pretrained`), stdlib only for logic/labeler (`json`, `subprocess`, `hashlib`, `random`, `statistics`), pyyaml (prompt/config). No new dependencies. pytest. Local `.venv` (`.venv/bin/pytest`, `.venv/bin/python`).

## Global Constraints

- **ADR-0003 label format is fixed** — four Axes `("moral", "adherence", "coherence", "prose")`, each rated an integer **1–5**; the Aggregate Score is the fixed weighted sum **0.4·moral + 0.3·adherence + 0.2·coherence + 0.1·prose**; **ties are skipped** (no preference emitted). These weights live in exactly one place (`feedback.WEIGHTS`) and are never re-typed elsewhere.
- **ADR-0004 source is Claude (pure RLAIF)** — all preference labels come from `claude -p`, pinned to one model version and one prompt version, rating against the human-authored Rubric. Every label record MUST carry `model_version` and `prompt_version`. No human labels enter the training loop.
- **Tests NEVER call Claude live** — the `claude -p` subprocess lives behind a single injected `runner` callable; every test passes a fake runner or reads a recorded fixture. Exactly **one schema contract test** validates strict-JSON parsing against a recorded live sample (`tests/fixtures/labeler_response_sample.json`). `pyproject.toml` already deselects `network`-marked tests; the labeling stage's tests are plain (offline) tests.
- **Resume is free** — the labeler's cache is append-only JSONL at `<out>/labels.jsonl`, keyed by `(pair_id, phase, order, model_version, prompt_version)`; a resumed run rebuilds the deterministic work list and skips keys already in the cache. An interruption (rate limit, session death) costs nothing. `manifest.json` is written LAST as the completion marker, matching `prep`/`pretrain`/`evaluate`.
- **Rubric before labeling** — `RUBRIC.md` and `configs/labeler_prompt.yaml` are human-authored artifacts; they MUST be committed and human-approved before the Track B labeling run. They are drafted in Track A (guard-tested for well-formedness) and approved at the Track A→B boundary.
- **Lazy stage registry intact** — the four new stages register by module-path STRING in `REGISTRY`; `pairgen` imports torch, but `label`/`derive`/`audit` and both logic modules (`feedback.py`, `labeler.py`) MUST NOT import torch at module top level.
- **Provenance chain** — `pairgen` records the Base Model `checkpoint_sha` in `pairgen_summary.json` + `manifest.json` inputs; `label` records `model_version`/`prompt_version`/`rubric_sha`/`prompt_sha` per row; `derive`/`audit` record the `labels.jsonl` + `pairs.jsonl` SHAs in their manifests. The `preferences.jsonl` artifact is the stable input to issue 07.
- **Single-band dataset (issue 04)** — FableSpecs parsed from val prompts carry `age_range="4-7"`; `render_canonical_prompt` raises on any other band. Pairs use the Canonical Prompt only (held-out templates are an eval-robustness concern, not a preference-pair concern).
- **Gates (record, reason, do not silently pass)** — Calibration-Set self-consistency **≥ ~0.85**; position-swap flip rate is **reported and reasoned about** (no hard numeric bar, but a high flip rate is flagged). These gate issue 07's reward model; the `audit` stage emits pass/flag against the configured threshold.
- **Determinism where it can exist** — `pairgen` is deterministic given seed + checkpoint (seeded per-sample generation). `label` with a *deterministic fake runner* produces a byte-identical cache (used in tests); with real Claude it is not reproducible (Claude is stochastic) and that is expected. `derive`/`audit` are pure functions of their input cache.

---

## File Structure

**New modules:**
- `src/tinyfables/feedback.py` — torch-free logic: `AXES`, `WEIGHTS`, `aggregate_score`, `derive_preference`, `perturb_weights`, `weight_sensitivity`, `position_flip_rate`, `self_consistency`.
- `src/tinyfables/labeler.py` — torch-free labeler plumbing: `LabelerError`, `PairLabel`, `load_prompt`, `build_batch_prompt`, `parse_labeler_response`, `cache_key`, `load_cache`, `append_cache`, `claude_runner`.
- `src/tinyfables/stages/pairgen.py` — `pairgen` stage (torch; seeded 2-samples-per-spec generation).
- `src/tinyfables/stages/label.py` — `label` stage (resumable orchestrator; injected runner).
- `src/tinyfables/stages/derive.py` — `derive` stage (preferences.jsonl + sensitivity.json).
- `src/tinyfables/stages/audit.py` — `audit` stage (audit_report.md + audit.json vs gates).

**New human artifacts:**
- `RUBRIC.md` (repo root) — anchored per-Axis 1–5 definitions with example snippets.
- `configs/labeler_prompt.yaml` — versioned labeler instruction template (`version`, `template`).

**New configs:**
- `configs/pairgen_toy.yaml`, `configs/pairgen_full.yaml`
- `configs/label_toy.yaml`, `configs/label_full.yaml`
- `configs/derive_toy.yaml`, `configs/derive_full.yaml`
- `configs/audit_toy.yaml`, `configs/audit_full.yaml`

**Modified:**
- `src/tinyfables/config.py` — add `PairgenConfig`, `LabelConfig`, `DeriveConfig`, `AuditConfig`.
- `src/tinyfables/stages/__init__.py` — register `pairgen`, `label`, `derive`, `audit`.
- `docs/design.md` — "Implementation (issue 06)" note (Track A) + real-run gate numbers (Track B).
- `docs/issues/06-feedback-data-labeler.md` — tick acceptance criteria (end of Track B).

**New tests:** `tests/test_feedback.py`, `tests/test_labeler.py`, `tests/test_pairgen_stage.py`, `tests/test_label_stage.py`, `tests/test_derive_stage.py`, `tests/test_audit_stage.py`; additions to `tests/test_config.py` and `tests/test_full_configs.py`.

**New fixtures:**
- `tests/fixtures/labeler_response_sample.json` — one recorded batched Claude response (schema contract).
- `tests/fixtures/labels_replay.jsonl` — a small hand-built label cache (real schema) for `derive`/`audit` tests.

**Branch:** `feat/issue-06-feedback-data` off `main`.

---

# TRACK A — Local, subagent-driven, offline-tested

---

### Task 1: Aggregate Score + preference derivation (`feedback.py`)

**Files:**
- Create: `src/tinyfables/feedback.py`
- Test: `tests/test_feedback.py`

**Interfaces:**
- Consumes: nothing (pure stdlib).
- Produces:
  - `AXES: tuple[str, ...]` = `("moral", "adherence", "coherence", "prose")`.
  - `WEIGHTS: dict[str, float]` = `{"moral": 0.4, "adherence": 0.3, "coherence": 0.2, "prose": 0.1}`.
  - `aggregate_score(ratings: dict[str, float], weights: dict[str, float] = WEIGHTS) -> float`.
  - `derive_preference(ratings_0, ratings_1, weights=WEIGHTS) -> int | None` — returns `0` (fable 0 preferred), `1` (fable 1 preferred), or `None` (tie → skipped).
  - Consumed by Tasks 3, 4, 9, 10.

- [ ] **Step 1: Write the failing test**

Create `tests/test_feedback.py`:

```python
import pytest

from tinyfables.feedback import AXES, WEIGHTS, aggregate_score, derive_preference


def test_axes_and_weights_are_the_adr_0003_spec():
    assert AXES == ("moral", "adherence", "coherence", "prose")
    assert WEIGHTS == {"moral": 0.4, "adherence": 0.3, "coherence": 0.2, "prose": 0.1}
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


def test_aggregate_score_is_the_weighted_sum():
    r = {"moral": 5, "adherence": 4, "coherence": 3, "prose": 2}
    # 0.4*5 + 0.3*4 + 0.2*3 + 0.1*2 = 2.0 + 1.2 + 0.6 + 0.2 = 4.0
    assert aggregate_score(r) == pytest.approx(4.0)


def test_derive_preference_prefers_higher_aggregate():
    hi = {"moral": 5, "adherence": 5, "coherence": 5, "prose": 5}
    lo = {"moral": 1, "adherence": 1, "coherence": 1, "prose": 1}
    assert derive_preference(hi, lo) == 0
    assert derive_preference(lo, hi) == 1


def test_derive_preference_skips_ties():
    r = {"moral": 3, "adherence": 3, "coherence": 3, "prose": 3}
    assert derive_preference(r, dict(r)) is None


def test_moral_weight_dominates_a_single_axis_swing():
    # moral 5 vs 1 (Δagg 0.4*4=1.6) beats prose 1 vs 5 (Δagg 0.1*4=0.4)
    a = {"moral": 5, "adherence": 3, "coherence": 3, "prose": 1}
    b = {"moral": 1, "adherence": 3, "coherence": 3, "prose": 5}
    assert derive_preference(a, b) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_feedback.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'tinyfables.feedback'`

- [ ] **Step 3: Write minimal implementation**

Create `src/tinyfables/feedback.py`:

```python
"""Preference logic (issue 06), torch-free so it imports anywhere and is trivially
unit-tested. The Aggregate Score weights are the ADR-0003 spec and live ONLY here:
0.4 moral / 0.3 adherence / 0.2 coherence / 0.1 prose. Preferences are derived from
within-pair Aggregate-Score differences; exact ties are skipped (no label). Also
holds the report deliverables computed from the label cache — the weight-sensitivity
table and the two labeler-audit numbers (position-swap flip rate, Calibration-Set
self-consistency)."""

from __future__ import annotations

from statistics import mean

# ADR-0003: four Axes, moral delivery weighted highest. Single source of truth.
AXES: tuple[str, ...] = ("moral", "adherence", "coherence", "prose")
WEIGHTS: dict[str, float] = {"moral": 0.4, "adherence": 0.3, "coherence": 0.2, "prose": 0.1}


def aggregate_score(ratings: dict[str, float], weights: dict[str, float] = WEIGHTS) -> float:
    """Fixed weighted combination of a fable's four Axis ratings."""
    return sum(weights[axis] * ratings[axis] for axis in AXES)


def derive_preference(
    ratings_0: dict[str, float],
    ratings_1: dict[str, float],
    weights: dict[str, float] = WEIGHTS,
) -> int | None:
    """0 if fable 0 preferred, 1 if fable 1 preferred, None on an exact tie (skipped)."""
    s0 = aggregate_score(ratings_0, weights)
    s1 = aggregate_score(ratings_1, weights)
    if s0 == s1:
        return None
    return 0 if s0 > s1 else 1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_feedback.py -q`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/tinyfables/feedback.py tests/test_feedback.py
git commit -m "feat(feedback): Aggregate Score + tie-skipping preference derivation (issue 06)"
```

---

### Task 2: Weight-sensitivity table (`feedback.py`)

**Files:**
- Modify: `src/tinyfables/feedback.py`
- Test: `tests/test_feedback.py`

**Interfaces:**
- Consumes: `AXES`, `WEIGHTS`, `derive_preference` (Task 1).
- Produces:
  - `perturb_weights(weights, axis, delta) -> dict[str, float]` — bump `axis` by `delta`, renormalize the other three proportionally so the four weights still sum to 1.0.
  - `weight_sensitivity(pair_ratings, weights=WEIGHTS, delta=0.1) -> dict` — `pair_ratings` is a list of `(ratings_0, ratings_1)` tuples (one per pair). Returns `{"delta": delta, "n_base_preferences": int, "rows": [{"axis","direction","n_flipped","pct_flipped"}...]}` where a *flip* is any pair whose perturbed preference differs from its base preference (a base non-tie becoming a tie counts as a flip; base ties are excluded from the denominator).
  - Consumed by Task 9 (`derive` stage).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_feedback.py`:

```python
from tinyfables.feedback import perturb_weights, weight_sensitivity


def test_perturb_weights_keeps_sum_one_and_bumps_target():
    w = perturb_weights(WEIGHTS, "moral", 0.1)
    assert w["moral"] == pytest.approx(0.5)
    assert sum(w.values()) == pytest.approx(1.0)
    # the other three keep their relative proportions (0.3:0.2:0.1)
    assert w["adherence"] == pytest.approx(0.25)
    assert w["coherence"] == pytest.approx(0.1666667, abs=1e-4)
    assert w["prose"] == pytest.approx(0.0833333, abs=1e-4)


def test_weight_sensitivity_counts_flips_and_excludes_base_ties():
    # pair 0: robust — fable 0 wins on every axis, no perturbation flips it
    robust = ({"moral": 5, "adherence": 5, "coherence": 5, "prose": 5},
              {"moral": 1, "adherence": 1, "coherence": 1, "prose": 1})
    # pair 1: base tie — excluded from every denominator
    tie = ({"moral": 3, "adherence": 3, "coherence": 3, "prose": 3},
           {"moral": 3, "adherence": 3, "coherence": 3, "prose": 3})
    # pair 2: knife-edge on the prose axis only
    #   base agg: f0 = 0.4*3+0.3*3+0.2*3+0.1*5 = 3.2 ; f1 = ...+0.1*4 = 3.1 -> pref 0
    #   prose weight -0.1 (->0): f0 loses its prose edge -> agg 3.0 vs 3.0 tie -> flip
    knife = ({"moral": 3, "adherence": 3, "coherence": 3, "prose": 5},
             {"moral": 3, "adherence": 3, "coherence": 3, "prose": 4})
    out = weight_sensitivity([robust, tie, knife], delta=0.1)
    assert out["delta"] == 0.1
    assert out["n_base_preferences"] == 2  # tie excluded
    rows = {(r["axis"], r["direction"]): r for r in out["rows"]}
    # 8 rows: each of 4 axes, +delta and -delta
    assert len(out["rows"]) == 8
    prose_down = rows[("prose", "-")]
    assert prose_down["n_flipped"] == 1
    assert prose_down["pct_flipped"] == pytest.approx(0.5)  # 1 of 2 non-tie pairs
    # the robust pair never flips on any perturbation
    assert all(r["n_flipped"] <= 1 for r in out["rows"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_feedback.py -q`
Expected: FAIL with `ImportError: cannot import name 'perturb_weights'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/tinyfables/feedback.py`:

```python
def perturb_weights(weights: dict[str, float], axis: str, delta: float) -> dict[str, float]:
    """Bump `axis` by `delta` and renormalize the other axes proportionally so the
    four weights still sum to 1.0 (holding their relative shares). Used for the
    ±0.1 weight-sensitivity table."""
    bumped = weights[axis] + delta
    others = [a for a in AXES if a != axis]
    base_rest = sum(weights[a] for a in others)  # = 1 - weights[axis]
    new_rest = 1.0 - bumped
    out = {axis: bumped}
    for a in others:
        out[a] = weights[a] / base_rest * new_rest
    return out


def weight_sensitivity(
    pair_ratings: list[tuple[dict, dict]],
    weights: dict[str, float] = WEIGHTS,
    delta: float = 0.1,
) -> dict:
    """How many derived preferences flip when each axis weight is perturbed by ±delta.
    A flip = the perturbed preference differs from the base preference; base ties are
    excluded from the denominator (they had no preference to flip)."""
    base = [derive_preference(r0, r1, weights) for r0, r1 in pair_ratings]
    non_tie = [i for i, p in enumerate(base) if p is not None]
    rows = []
    for axis in AXES:
        for direction, signed in (("+", delta), ("-", -delta)):
            w = perturb_weights(weights, axis, signed)
            flipped = sum(
                1 for i in non_tie
                if derive_preference(*pair_ratings[i], w) != base[i]
            )
            n = len(non_tie)
            rows.append({
                "axis": axis,
                "direction": direction,
                "n_flipped": flipped,
                "pct_flipped": round(flipped / n, 4) if n else 0.0,
            })
    return {"delta": delta, "n_base_preferences": len(non_tie), "rows": rows}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_feedback.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tinyfables/feedback.py tests/test_feedback.py
git commit -m "feat(feedback): +/-0.1 weight-sensitivity table (issue 06)"
```

---

### Task 3: Labeler audit metrics (`feedback.py`)

**Files:**
- Modify: `src/tinyfables/feedback.py`
- Test: `tests/test_feedback.py`

**Interfaces:**
- Consumes: `derive_preference` (Task 1).
- Produces:
  - `position_flip_rate(instances) -> dict` — `instances` is a list of cache-record dicts each with `pair_id`, `phase`, `order` (`"ab"`/`"ba"`), `ratings_0`, `ratings_1`. Pairs each pair's `phase=="main"` (`order=="ab"`) instance with its `phase=="swap"` (`order=="ba"`) instance; a *position flip* is when the two derived preferences disagree. Returns `{"n_pairs","n_flipped","flip_rate"}`.
  - `self_consistency(instances) -> dict` — over the calibration instances (`phase` starting with `"calib"`), groups by `pair_id`, derives a preference per repeat, and returns `{"n_pairs","mean_agreement","n_unanimous"}` where `mean_agreement` is the mean over calibration pairs of pairwise-agreement of the derived preference across repeats.
  - Consumed by Task 10 (`audit` stage).

**Cache-record shape (defined here, produced by Task 8):** each label instance stores ratings mapped back to the *stable* fable order — `ratings_0` = ratings for `fables[0]`, `ratings_1` = ratings for `fables[1]` — regardless of the presentation `order`. So `derive_preference(ratings_0, ratings_1)` is comparable across `order`/`phase`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_feedback.py`:

```python
from tinyfables.feedback import position_flip_rate, self_consistency

HI = {"moral": 5, "adherence": 5, "coherence": 5, "prose": 5}
LO = {"moral": 1, "adherence": 1, "coherence": 1, "prose": 1}


def _inst(pair_id, phase, order, r0, r1):
    return {"pair_id": pair_id, "phase": phase, "order": order,
            "ratings_0": r0, "ratings_1": r1}


def test_position_flip_rate_counts_disagreement_between_orders():
    instances = [
        # p1: main prefers 0, swap still prefers 0 -> no flip
        _inst("p1", "main", "ab", HI, LO),
        _inst("p1", "swap", "ba", HI, LO),
        # p2: main prefers 0, swap prefers 1 -> flip (position bias)
        _inst("p2", "main", "ab", HI, LO),
        _inst("p2", "swap", "ba", LO, HI),
        # p3: only a main label -> not part of the swap slice, ignored
        _inst("p3", "main", "ab", HI, LO),
    ]
    out = position_flip_rate(instances)
    assert out["n_pairs"] == 2
    assert out["n_flipped"] == 1
    assert out["flip_rate"] == pytest.approx(0.5)


def test_self_consistency_is_mean_pairwise_agreement():
    instances = [
        # c1: all three repeats prefer 0 -> unanimous, agreement 1.0
        _inst("c1", "calib-0", "ab", HI, LO),
        _inst("c1", "calib-1", "ab", HI, LO),
        _inst("c1", "calib-2", "ab", HI, LO),
        # c2: two prefer 0, one prefers 1 -> pairwise agreement 1/3
        _inst("c2", "calib-0", "ab", HI, LO),
        _inst("c2", "calib-1", "ab", HI, LO),
        _inst("c2", "calib-2", "ab", LO, HI),
    ]
    out = self_consistency(instances)
    assert out["n_pairs"] == 2
    assert out["n_unanimous"] == 1
    # mean of (1.0, 1/3) = 2/3
    assert out["mean_agreement"] == pytest.approx((1.0 + 1 / 3) / 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_feedback.py -q`
Expected: FAIL with `ImportError: cannot import name 'position_flip_rate'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/tinyfables/feedback.py`:

```python
from itertools import combinations


def _pref(inst: dict) -> int | None:
    return derive_preference(inst["ratings_0"], inst["ratings_1"])


def position_flip_rate(instances: list[dict]) -> dict:
    """Position-swap audit: over pairs labeled in BOTH orders (a `main`/`ab` instance
    and a `swap`/`ba` instance), the fraction whose derived preference disagrees
    between the two presentation orders. A flip means presentation order moved the
    ratings — i.e. position bias."""
    main = {i["pair_id"]: i for i in instances if i["phase"] == "main"}
    swap = {i["pair_id"]: i for i in instances if i["phase"] == "swap"}
    both = sorted(set(main) & set(swap))
    flipped = sum(1 for pid in both if _pref(main[pid]) != _pref(swap[pid]))
    n = len(both)
    return {"n_pairs": n, "n_flipped": flipped, "flip_rate": round(flipped / n, 4) if n else 0.0}


def self_consistency(instances: list[dict]) -> dict:
    """Calibration-Set self-consistency: for each calibration pair re-labeled several
    times across the run, the mean pairwise agreement of the derived preference.
    Reported against the >= ~0.85 gate."""
    by_pair: dict[str, list] = {}
    for i in instances:
        if str(i["phase"]).startswith("calib"):
            by_pair.setdefault(i["pair_id"], []).append(_pref(i))
    agreements = []
    unanimous = 0
    for prefs in by_pair.values():
        if len(prefs) < 2:
            continue
        pairs = list(combinations(prefs, 2))
        agree = sum(1 for a, b in pairs if a == b) / len(pairs)
        agreements.append(agree)
        if len(set(prefs)) == 1:
            unanimous += 1
    return {
        "n_pairs": len(agreements),
        "mean_agreement": round(mean(agreements), 4) if agreements else 0.0,
        "n_unanimous": unanimous,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_feedback.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tinyfables/feedback.py tests/test_feedback.py
git commit -m "feat(feedback): position-swap flip rate + calibration self-consistency (issue 06)"
```

---

### Task 4: The anchored Rubric + labeler prompt (human artifacts)

**Files:**
- Create: `RUBRIC.md`
- Create: `configs/labeler_prompt.yaml`
- Test: `tests/test_labeler.py` (well-formedness guard only; parsing logic comes in Task 5)

> **This is a draft for the human to approve.** The content below is a complete, committable starting point. The human refines and approves `RUBRIC.md` + `configs/labeler_prompt.yaml` at the Track A→B boundary (Task 12) — they MUST be committed and approved before the real labeling run. Track A code does not depend on the rubric *content* (the fake runner ignores it), so Track A proceeds with this draft.

**Interfaces:**
- Produces: `RUBRIC.md` (markdown, four `## Axis` sections with weight + 1–5 anchors), `configs/labeler_prompt.yaml` (`version: int`, `template: str` with `{rubric}` and `{pairs}` slots and literal `{{ }}`-escaped JSON braces).
- Consumed by Task 5 (`load_prompt`, `build_batch_prompt`) and Task 8 (`label` stage).

- [ ] **Step 1: Write the failing test**

Create `tests/test_labeler.py`:

```python
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def test_rubric_exists_and_anchors_all_four_axes():
    text = (REPO / "RUBRIC.md").read_text()
    for axis in ("Moral delivery", "Spec adherence", "Coherence", "Prose"):
        assert axis in text, f"RUBRIC.md missing axis: {axis}"
    # every 1-5 anchor is present (each axis defines all five scale points)
    for point in ("1", "2", "3", "4", "5"):
        assert f"- {point}:" in text or f"**{point}**" in text
    # the fixed weights appear so a labeler/report reader sees them
    for w in ("0.4", "0.3", "0.2", "0.1"):
        assert w in text


def test_labeler_prompt_is_versioned_and_slotted():
    data = yaml.safe_load((REPO / "configs" / "labeler_prompt.yaml").read_text())
    assert isinstance(data["version"], int) and data["version"] >= 1
    tmpl = data["template"]
    assert "{rubric}" in tmpl and "{pairs}" in tmpl
    # instructs strict JSON with the four axes named
    for axis in ("moral", "adherence", "coherence", "prose"):
        assert axis in tmpl
    # literal JSON braces are escaped for str.format (no stray single braces
    # besides the two named slots) -> format with dummy values must not raise
    tmpl.format(rubric="R", pairs="P")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_labeler.py -q`
Expected: FAIL with `FileNotFoundError: .../RUBRIC.md`

- [ ] **Step 3: Write the artifacts**

Create `RUBRIC.md`:

```markdown
# TinyFables Rubric

The anchored definition of "good" for the feedback stage. A human authored this;
the AI Labeler (Claude, ADR-0004) applies it at scale. Every fable in a Preference
Pair is rated on all four Axes below, each an integer **1–5**. The preference is
derived from the fixed **Aggregate Score = 0.4·moral + 0.3·adherence + 0.2·coherence
+ 0.1·prose** (ADR-0003); exact ties are skipped. Rate each fable on its own merits
against these anchors — not relative to the other fable in the pair.

Every fable was requested for children **ages 4–7** (dataset band B), around 250 words.

---

## Axis 1 — Moral delivery (weight 0.4)

Does the fable teach the **requested** Moral, clearly and inside the story?

- 1: No discernible moral, or the moral contradicts the requested one.
- 2: A moral is gestured at but muddled, tacked on, or a different lesson than requested.
- 3: A moral is present and on-topic but generic, or only loosely the requested one
  (e.g. "be brave" when "courage grows by small steps" was asked).
- 4: The requested moral is clearly delivered — either strongly dramatized by the plot
  OR stated plainly at the end, but not both; wording may differ from the request.
- 5: The requested moral is unmistakable — dramatized by the plot AND echoed in a clear
  closing line a 4–7-year-old would grasp.
  _Example (requested "courage grows by small steps"):_ "...and each small brave step
  made her braver. **Courage grows one small step at a time.**"

## Axis 2 — Spec adherence (weight 0.3)

Are the requested Elements (main character, setting, challenge, outcome) actually used
and faithful to the request? Rate faithful **use**, not literal string matching
(verbatim presence is measured separately and mechanically).

- 1: The fable ignores most of the spec, or is about something else entirely.
- 2: Two or more Elements are missing/changed, or the outcome contradicts the request.
- 3: One Element is missing or clearly changed (e.g. the requested lamb becomes a wolf),
  the rest honored.
- 4: All four Elements present; one is only lightly touched or slightly altered.
- 5: All four requested Elements appear and drive the story, faithful to the request.

## Axis 3 — Coherence (weight 0.2)

Is it a single, sensible story that holds together from beginning to end?

- 1: Incoherent word-salad, or it abandons the narrative.
- 2: Disjointed — events don't connect, or the story stalls or loops.
- 3: Followable but loose — some repetition, a dropped thread, or a soft ending.
- 4: Coherent with a minor bump (a small logic gap or an abrupt transition).
- 5: Clear beginning-middle-end; cause and effect track; no contradictions or non-sequiturs.

## Axis 4 — Prose & age-fit (weight 0.1)

Is the language fluent and right for ages 4–7?

- 1: Broken grammar, garbled, or wholly age-inappropriate.
- 2: Frequent grammar errors, or vocabulary too abstract/adult for 4–7.
- 3: Readable but plain or occasionally clunky/repetitive; a few too-hard words.
- 4: Fluent and age-appropriate with a rare awkward phrase or slightly advanced word.
- 5: Smooth, grammatical, simple concrete vocabulary; vivid but easy — a delight read aloud.
```

Create `configs/labeler_prompt.yaml`:

```yaml
# Labeler prompt (issue 06). Versioned + human-approved alongside RUBRIC.md.
# `version` is logged as prompt_version on EVERY label (ADR-0004). Bump it whenever
# this template OR RUBRIC.md changes — the label cache is keyed on prompt_version,
# so a bump correctly forces re-labeling. {rubric} and {pairs} are str.format slots;
# all other braces (the JSON schema) are escaped as {{ }}.
version: 1
template: |
  You are a careful, consistent judge of children's moral fables (ages 4-7). Rate
  each fable strictly against this rubric. Rate each fable on its own merits, not
  relative to the other fable in its pair.

  RUBRIC:
  {rubric}

  Below are numbered pairs. For EACH pair, rate BOTH "fable_a" and "fable_b" on all
  four axes with an integer 1-5: moral, adherence, coherence, prose. Add one short
  justification sentence per pair.

  Output ONLY a single JSON object, no prose before or after, in exactly this shape:
  {{"labels": [
    {{"pair_id": "<id>",
      "fable_a": {{"moral": 0, "adherence": 0, "coherence": 0, "prose": 0}},
      "fable_b": {{"moral": 0, "adherence": 0, "coherence": 0, "prose": 0}},
      "justification": "<one sentence>"}}
  ]}}

  PAIRS:
  {pairs}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_labeler.py -q`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add RUBRIC.md configs/labeler_prompt.yaml tests/test_labeler.py
git commit -m "feat(rubric): draft anchored 1-5 Rubric + versioned labeler prompt (issue 06)"
```

---

### Task 5: Labeler prompt building + strict-JSON parsing + schema contract (`labeler.py`)

**Files:**
- Create: `src/tinyfables/labeler.py`
- Create: `tests/fixtures/labeler_response_sample.json`
- Test: `tests/test_labeler.py`

**Interfaces:**
- Consumes: `feedback.AXES`, `configs/labeler_prompt.yaml`, `RUBRIC.md`.
- Produces:
  - `LabelerError(ValueError)` — raised on any malformed labeler response.
  - `PairLabel` — frozen dataclass `(pair_id: str, ratings_a: dict[str,int], ratings_b: dict[str,int], justification: str | None)`; `ratings_a`/`ratings_b` are the fables **as presented** (A = shown first).
  - `load_prompt(path) -> tuple[int, str]` — returns `(version, template)` from a labeler-prompt YAML.
  - `build_batch_prompt(rubric_text, template_text, batch) -> str` — `batch` is a list of `{"pair_id","fable_a","fable_b"}`; fills `{rubric}` and `{pairs}`.
  - `parse_labeler_response(text, expected_pair_ids) -> list[PairLabel]` — extracts the JSON object from `text`, validates every expected pair is present with all four axes as ints in `[1,5]` for both fables, returns labels ordered by `expected_pair_ids`.
  - Consumed by Task 8 (`label` stage).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_labeler.py`:

```python
import json

import pytest

from tinyfables.labeler import (
    LabelerError,
    PairLabel,
    build_batch_prompt,
    load_prompt,
    parse_labeler_response,
)

FIX = REPO / "tests" / "fixtures" / "labeler_response_sample.json"


def test_load_prompt_returns_version_and_template():
    version, template = load_prompt(REPO / "configs" / "labeler_prompt.yaml")
    assert version == 1
    assert "{rubric}" not in build_batch_prompt("RUBRIC", template,
        [{"pair_id": "pair-0", "fable_a": "A", "fable_b": "B"}])


def test_build_batch_prompt_embeds_rubric_and_every_pair():
    _, template = load_prompt(REPO / "configs" / "labeler_prompt.yaml")
    batch = [
        {"pair_id": "pair-000001", "fable_a": "the fox ran", "fable_b": "the owl slept"},
        {"pair_id": "pair-000002", "fable_a": "a {brace} value", "fable_b": "b"},
    ]
    prompt = build_batch_prompt("MY RUBRIC BODY", template, batch)
    assert "MY RUBRIC BODY" in prompt
    for row in batch:
        assert row["pair_id"] in prompt
        assert row["fable_a"] in prompt and row["fable_b"] in prompt
    # braces inside fable text survive (they are values, never re-formatted)
    assert "a {brace} value" in prompt


def test_parse_valid_response_returns_labels_in_expected_order():
    text = FIX.read_text()
    ids = [row["pair_id"] for row in json.loads(text)["labels"]]
    labels = parse_labeler_response(text, ids)
    assert [l.pair_id for l in labels] == ids
    for l in labels:
        assert isinstance(l, PairLabel)
        for side in (l.ratings_a, l.ratings_b):
            assert set(side) == {"moral", "adherence", "coherence", "prose"}
            assert all(isinstance(v, int) and 1 <= v <= 5 for v in side.values())


def test_parse_extracts_json_from_surrounding_prose():
    body = '{"labels":[{"pair_id":"p0",' \
           '"fable_a":{"moral":4,"adherence":3,"coherence":4,"prose":3},' \
           '"fable_b":{"moral":2,"adherence":2,"coherence":3,"prose":3}}]}'
    labels = parse_labeler_response("Sure! Here it is:\n" + body + "\nDone.", ["p0"])
    assert labels[0].ratings_a["moral"] == 4


def test_parse_rejects_missing_pair():
    body = '{"labels":[{"pair_id":"p0",' \
           '"fable_a":{"moral":4,"adherence":3,"coherence":4,"prose":3},' \
           '"fable_b":{"moral":2,"adherence":2,"coherence":3,"prose":3}}]}'
    with pytest.raises(LabelerError):
        parse_labeler_response(body, ["p0", "p1"])  # p1 absent


def test_parse_rejects_out_of_range_and_non_int():
    for bad in ("6", "0", "3.5", '"4"'):
        body = ('{"labels":[{"pair_id":"p0",'
                f'"fable_a":{{"moral":{bad},"adherence":3,"coherence":4,"prose":3}},'
                '"fable_b":{"moral":2,"adherence":2,"coherence":3,"prose":3}}]}')
        with pytest.raises(LabelerError):
            parse_labeler_response(body, ["p0"])


def test_parse_rejects_missing_axis():
    body = '{"labels":[{"pair_id":"p0",' \
           '"fable_a":{"moral":4,"adherence":3,"coherence":4},' \
           '"fable_b":{"moral":2,"adherence":2,"coherence":3,"prose":3}}]}'
    with pytest.raises(LabelerError):
        parse_labeler_response(body, ["p0"])


def test_recorded_sample_is_the_schema_contract():
    # ONE schema contract test against a recorded live sample (ADR-0004). Track B
    # replaces this fixture with a verbatim real `claude -p` capture; this test must
    # stay green, which is exactly what pins our schema to Claude's real output.
    text = FIX.read_text()
    payload = json.loads(text)
    assert "labels" in payload and payload["labels"]
    ids = [row["pair_id"] for row in payload["labels"]]
    parse_labeler_response(text, ids)  # must not raise
```

- [ ] **Step 2: Create the recorded-sample fixture**

Create `tests/fixtures/labeler_response_sample.json` (provisional Track-A capture matching the designed schema; **Task 14 replaces it with a real `claude -p` capture**):

```json
{
  "labels": [
    {
      "pair_id": "pair-000000",
      "fable_a": {"moral": 5, "adherence": 4, "coherence": 4, "prose": 4},
      "fable_b": {"moral": 3, "adherence": 4, "coherence": 3, "prose": 4},
      "justification": "A states and dramatizes the requested moral; B only implies it."
    },
    {
      "pair_id": "pair-000001",
      "fable_a": {"moral": 2, "adherence": 3, "coherence": 3, "prose": 3},
      "fable_b": {"moral": 4, "adherence": 4, "coherence": 4, "prose": 4},
      "justification": "B follows the spec and ends on a clear lesson; A drifts from the outcome."
    }
  ]
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_labeler.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'tinyfables.labeler'`

- [ ] **Step 4: Write minimal implementation**

Create `src/tinyfables/labeler.py` (parsing + prompt building only; cache + runner in Task 6):

```python
"""The AI Labeler plumbing (issue 06, ADR-0004): build the batched Claude prompt,
parse its strict-JSON per-Axis ratings, and cache labels append-only. Torch-free.

The ONLY code that invokes Claude is `claude_runner` (Task 6); every test injects a
fake runner or reads a recorded fixture, so tests never call Claude live. One schema
contract test validates `parse_labeler_response` against a recorded live sample."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from tinyfables.feedback import AXES


class LabelerError(ValueError):
    """A labeler response did not match the strict-JSON per-Axis 1-5 contract."""


@dataclass(frozen=True)
class PairLabel:
    pair_id: str
    ratings_a: dict[str, int]   # fable shown FIRST in this presentation
    ratings_b: dict[str, int]   # fable shown SECOND
    justification: str | None = None


def load_prompt(path: str | Path) -> tuple[int, str]:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict) or "version" not in data or "template" not in data:
        raise ValueError("labeler prompt must be a mapping with 'version' and 'template'")
    return int(data["version"]), str(data["template"])


def build_batch_prompt(rubric_text: str, template_text: str, batch: list[dict]) -> str:
    """Render the labeler prompt for one batch. rubric_text and the pairs block are
    substituted as VALUES (str.format only interprets the template's own braces), so
    braces/colons inside fable text survive verbatim — same guarantee as paraphrases."""
    blocks = []
    for i, row in enumerate(batch, 1):
        blocks.append(
            f"[{i}] pair_id: {row['pair_id']}\n"
            f"--- Fable A ---\n{row['fable_a']}\n"
            f"--- Fable B ---\n{row['fable_b']}\n"
        )
    return template_text.format(rubric=rubric_text, pairs="\n".join(blocks))


def _extract_json_object(text: str) -> dict:
    """Take the outermost {...} span from `text` and json.loads it. Tolerates prose
    or ```json fences around the object (Claude sometimes adds a sentence)."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise LabelerError(f"no JSON object found in labeler response: {text[:200]!r}")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        raise LabelerError(f"labeler response is not valid JSON: {e}") from e


def _validate_ratings(obj, pair_id: str, side: str) -> dict[str, int]:
    if not isinstance(obj, dict):
        raise LabelerError(f"{pair_id} {side}: ratings must be an object, got {type(obj).__name__}")
    out = {}
    for axis in AXES:
        if axis not in obj:
            raise LabelerError(f"{pair_id} {side}: missing axis {axis!r}")
        v = obj[axis]
        # bool is an int subclass; reject it explicitly. floats/strings rejected too.
        if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 5:
            raise LabelerError(f"{pair_id} {side}: axis {axis!r} must be int 1-5, got {v!r}")
        out[axis] = v
    return out


def parse_labeler_response(text: str, expected_pair_ids: list[str]) -> list[PairLabel]:
    payload = _extract_json_object(text)
    rows = payload.get("labels")
    if not isinstance(rows, list):
        raise LabelerError("labeler response missing a 'labels' list")
    by_id = {}
    for row in rows:
        if not isinstance(row, dict) or "pair_id" not in row:
            raise LabelerError(f"malformed label row: {row!r}")
        by_id[row["pair_id"]] = row
    labels = []
    for pid in expected_pair_ids:
        if pid not in by_id:
            raise LabelerError(f"labeler response missing expected pair {pid!r}")
        row = by_id[pid]
        labels.append(PairLabel(
            pair_id=pid,
            ratings_a=_validate_ratings(row.get("fable_a"), pid, "fable_a"),
            ratings_b=_validate_ratings(row.get("fable_b"), pid, "fable_b"),
            justification=row.get("justification"),
        ))
    return labels
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_labeler.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/tinyfables/labeler.py tests/test_labeler.py tests/fixtures/labeler_response_sample.json
git commit -m "feat(labeler): batch-prompt builder + strict-JSON parser + schema contract (issue 06)"
```

---

### Task 6: Append-only cache + the `claude -p` runner boundary (`labeler.py`)

**Files:**
- Modify: `src/tinyfables/labeler.py`
- Test: `tests/test_labeler.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `cache_key(pair_id, phase, order, model_version, prompt_version) -> tuple` — the resume/dedupe key.
  - `load_cache(path) -> dict[tuple, dict]` — reads a JSONL cache into `{cache_key(...): record}`; returns `{}` if the file is absent.
  - `append_cache(path, record) -> None` — appends one JSON line (creates the file/parent if needed).
  - `claude_runner(prompt, model) -> str` — the single subprocess boundary; runs `claude -p --model <model> --output-format text` with `prompt` on stdin, returns stdout, raises `LabelerError` on non-zero exit. Never called by tests.
  - Consumed by Task 8 (`label` stage).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_labeler.py`:

```python
from tinyfables.labeler import append_cache, cache_key, load_cache


def test_cache_key_is_a_stable_tuple():
    k = cache_key("pair-0", "main", "ab", "claude-opus-4-8", 1)
    assert k == ("pair-0", "main", "ab", "claude-opus-4-8", 1)


def test_append_then_load_roundtrips_by_key(tmp_path):
    path = tmp_path / "labels.jsonl"
    rec = {"pair_id": "pair-0", "phase": "main", "order": "ab",
           "model_version": "claude-opus-4-8", "prompt_version": 1,
           "ratings_0": {"moral": 5, "adherence": 4, "coherence": 4, "prose": 3},
           "ratings_1": {"moral": 2, "adherence": 3, "coherence": 3, "prose": 3}}
    append_cache(path, rec)
    cache = load_cache(path)
    key = cache_key("pair-0", "main", "ab", "claude-opus-4-8", 1)
    assert key in cache
    assert cache[key]["ratings_0"]["moral"] == 5


def test_load_cache_missing_file_is_empty(tmp_path):
    assert load_cache(tmp_path / "nope.jsonl") == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_labeler.py -q`
Expected: FAIL with `ImportError: cannot import name 'cache_key'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/tinyfables/labeler.py`:

```python
import subprocess


def cache_key(pair_id: str, phase: str, order: str, model_version: str, prompt_version: int) -> tuple:
    """Resume/dedupe key for one label instance. A run skips any key already cached,
    so rate limits or interruptions cost nothing."""
    return (pair_id, phase, order, model_version, prompt_version)


def load_cache(path: str | Path) -> dict[tuple, dict]:
    p = Path(path)
    if not p.exists():
        return {}
    out: dict[tuple, dict] = {}
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[cache_key(rec["pair_id"], rec["phase"], rec["order"],
                      rec["model_version"], rec["prompt_version"])] = rec
    return out


def append_cache(path: str | Path, record: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(record) + "\n")


def claude_runner(prompt: str, model: str) -> str:
    """The ONE place Claude is invoked: headless `claude -p`, prompt on stdin (safe
    for large prompts), text output. Raises LabelerError on failure so the resumable
    stage can stop cleanly and be re-run. Tests never call this — they inject a fake."""
    proc = subprocess.run(
        ["claude", "-p", "--model", model, "--output-format", "text"],
        input=prompt, capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        raise LabelerError(f"claude -p failed (exit {proc.returncode}): {proc.stderr[:500]}")
    return proc.stdout
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_labeler.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tinyfables/labeler.py tests/test_labeler.py
git commit -m "feat(labeler): append-only JSONL cache + claude -p runner boundary (issue 06)"
```

---

### Task 7: `pairgen` stage — Preference Pairs from the Base Model

**Files:**
- Modify: `src/tinyfables/config.py` (add `PairgenConfig`)
- Create: `src/tinyfables/stages/pairgen.py`
- Modify: `src/tinyfables/stages/__init__.py` (register `pairgen`)
- Create: `configs/pairgen_toy.yaml`, `configs/pairgen_full.yaml`
- Test: `tests/test_pairgen_stage.py`, additions to `tests/test_config.py`

**Interfaces:**
- Consumes: `EvalConfig` pattern, `data.iter_rows`, `paraphrases.parse_canonical_prompt`, `prompts.FableSpec`/`render_canonical_prompt`, `generate.generate_fable`, `model.GPT`, `stage.write_manifest`/`sha256_file`/`_versions`.
- Produces (`stages/pairgen.py`):
  - `run(cfg: PairgenConfig, out_dir: Path) -> None` — writes `pairs.jsonl` (one pair per line: `{"pair_id","spec","prompt","fables":[f0,f1],"seeds":[s0,s1]}`), `pairgen_summary.json` (provenance: `checkpoint_sha`, `tokenizer_sha`, `n_pairs`, `seed`, `temperature`, `versions`), and `manifest.json` LAST.
  - `PairgenConfig` fields: `checkpoint`, `tokenizer_dir`, `source: SourceSpec`, `n_ctx=1024`, `n_pairs=2000`, `max_new_tokens=320`, `min_new_tokens=80`, `temperature=0.9`, `top_k=50`, `seed=0`, `device="auto"`.
- Consumed by Task 8 (`label` reads `pairs.jsonl`) and Task 9 (`derive` reads it for provenance).

- [ ] **Step 1: Add the config (write the failing config test first)**

Append to `tests/test_config.py`:

```python
def test_pairgen_config_defaults_match_the_feedback_spec():
    from tinyfables.config import PairgenConfig, SourceSpec

    cfg = PairgenConfig(checkpoint="runs/base_model", tokenizer_dir="runs/tokenizer_full",
                        source=SourceSpec(hf_dataset="klusai/ds-tf1-en-3m", hf_split="validation"))
    assert cfg.n_pairs == 2000
    assert cfg.temperature == 0.9  # design: 2 independent samples at temp ~0.9
    assert cfg.max_new_tokens == 320
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/pytest tests/test_config.py::test_pairgen_config_defaults_match_the_feedback_spec -q`
Expected: FAIL with `ImportError: cannot import name 'PairgenConfig'`

- [ ] **Step 3: Add `PairgenConfig` to `config.py`**

Insert after `EvalConfig` in `src/tinyfables/config.py`:

```python
@dataclass(frozen=True)
class PairgenConfig:
    checkpoint: str
    tokenizer_dir: str
    source: SourceSpec
    n_ctx: int = 1024
    n_pairs: int = 2000
    max_new_tokens: int = 320
    min_new_tokens: int = 80
    temperature: float = 0.9
    top_k: int = 50
    seed: int = 0
    device: str = "auto"
```

- [ ] **Step 4: Write the failing stage test**

Create `tests/test_pairgen_stage.py`:

```python
import json
from pathlib import Path

from tinyfables.config import PairgenConfig, PretrainConfig, PrepConfig, SourceSpec, TokenizerConfig
from tinyfables.stages import REGISTRY
from tinyfables.stages import pairgen as pairgen_stage
from tinyfables.stages import prep as prep_stage
from tinyfables.stages import pretrain as pretrain_stage
from tinyfables.stages import tokenizer as tokenizer_stage

FIXTURE = str(Path(__file__).parent / "fixtures" / "tiny_corpus.jsonl")
SRC = SourceSpec(jsonl_path=FIXTURE)


def _trained_checkpoint(tmp_path):
    tok = tmp_path / "tok"; prep = tmp_path / "prep"; ckpt = tmp_path / "ckpt"
    tokenizer_stage.run(TokenizerConfig(source=SRC, vocab_size=512, seed=0), tok)
    prep_stage.run(PrepConfig(source=SRC, tokenizer_dir=str(tok), window=512, seed=0), prep)
    pretrain_stage.run(PretrainConfig(prep_dir=str(prep), tokenizer_dir=str(tok),
                       n_layer=2, n_head=2, d_model=64, n_ctx=512, batch_size=4,
                       steps=4, warmup_steps=0, seed=0, device="cpu"), ckpt)
    return tok, ckpt


def _cfg(tok, ckpt, **kw):
    base = dict(checkpoint=str(ckpt), tokenizer_dir=str(tok), source=SRC, n_ctx=512,
                n_pairs=3, max_new_tokens=24, min_new_tokens=8, top_k=10, seed=0, device="cpu")
    base.update(kw)
    return PairgenConfig(**base)


def test_pairgen_registered():
    assert REGISTRY["pairgen"] == (PairgenConfig, "tinyfables.stages.pairgen")


def test_pairgen_writes_pairs_summary_and_manifest(tmp_path):
    tok, ckpt = _trained_checkpoint(tmp_path)
    out = tmp_path / "pairs"
    pairgen_stage.run(_cfg(tok, ckpt), out)
    assert {"pairs.jsonl", "pairgen_summary.json", "manifest.json"} <= {p.name for p in out.iterdir()}
    pairs = [json.loads(l) for l in (out / "pairs.jsonl").read_text().splitlines()]
    assert len(pairs) == 3
    p0 = pairs[0]
    assert set(p0) >= {"pair_id", "spec", "prompt", "fables", "seeds"}
    assert len(p0["fables"]) == 2 and p0["seeds"][0] != p0["seeds"][1]
    assert p0["pair_id"] == "pair-000000"
    summary = json.loads((out / "pairgen_summary.json").read_text())
    assert summary["n_pairs"] == 3
    assert len(summary["checkpoint_sha"]) == 64  # provenance: Base Model hash recorded


def test_pairgen_two_samples_differ_and_are_deterministic(tmp_path):
    tok, ckpt = _trained_checkpoint(tmp_path)
    a = tmp_path / "a"; b = tmp_path / "b"
    pairgen_stage.run(_cfg(tok, ckpt), a)
    pairgen_stage.run(_cfg(tok, ckpt), b)
    pa = (a / "pairs.jsonl").read_text()
    pb = (b / "pairs.jsonl").read_text()
    assert pa == pb  # deterministic given seed + checkpoint
```

- [ ] **Step 5: Run it to verify it fails**

Run: `.venv/bin/pytest tests/test_pairgen_stage.py -q`
Expected: FAIL — `pairgen` not in `REGISTRY` / no module `tinyfables.stages.pairgen`.

- [ ] **Step 6: Implement the stage + register it**

Create `src/tinyfables/stages/pairgen.py`:

```python
"""Pair generation (issue 06): sample Preference Pairs from the Base Model — two
independent samples (temperature ~0.9) per held-out FableSpec drawn from the val
split, rendered under the Canonical Prompt. Mirrors stages/evaluate.py (checkpoint
load, val-split spec sampling, seeded generation). Deterministic given seed +
checkpoint; the Base Model checkpoint hash is recorded as provenance so downstream
preferences (issue 07) trace back to the exact model they came from."""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict
from pathlib import Path

import torch
from tokenizers import Tokenizer

from tinyfables.config import PairgenConfig
from tinyfables.data import iter_rows
from tinyfables.generate import generate_fable
from tinyfables.model import GPT
from tinyfables.paraphrases import parse_canonical_prompt
from tinyfables.prompts import FableSpec, render_canonical_prompt
from tinyfables.stage import _versions, sha256_file, write_manifest

_ELEMENTS = ("character", "setting", "challenge", "outcome", "moral")


def _resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _spec_from_prompt(prompt: str) -> FableSpec | None:
    parsed = parse_canonical_prompt(prompt)
    if parsed is None:
        return None
    return FableSpec(**{f: getattr(parsed, f) for f in _ELEMENTS},
                     age_range=parsed.age_range, word_count=parsed.word_count)


def run(cfg: PairgenConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(cfg.device)
    tok = Tokenizer.from_file(str(Path(cfg.tokenizer_dir) / "tokenizer.json"))
    model = GPT.from_pretrained(cfg.checkpoint).to(device).eval()

    rows = itertools.islice(iter_rows(cfg.source, cfg.seed), cfg.n_pairs * 4)
    specs: list[FableSpec] = []
    for r in rows:
        spec = _spec_from_prompt(r["prompt"])
        if spec is not None:
            specs.append(spec)
        if len(specs) >= cfg.n_pairs:
            break

    def _gen(prompt_text: str, seed: int) -> str:
        return generate_fable(model, tok, prompt_text, max_new_tokens=cfg.max_new_tokens,
                              min_new_tokens=cfg.min_new_tokens, do_sample=True,
                              temperature=cfg.temperature, top_k=cfg.top_k,
                              seed=seed, device=device)

    pairs = []
    for i, spec in enumerate(specs):
        prompt = render_canonical_prompt(spec)
        # distinct, unique-across-pairs seeds -> two INDEPENDENT samples at temp ~0.9
        s0, s1 = cfg.seed + 2 * i, cfg.seed + 2 * i + 1
        pairs.append({
            "pair_id": f"pair-{i:06d}",
            "spec": asdict(spec),
            "prompt": prompt,
            "fables": [_gen(prompt, s0), _gen(prompt, s1)],
            "seeds": [s0, s1],
        })

    pairs_path = out_dir / "pairs.jsonl"
    pairs_path.write_text("".join(json.dumps(p) + "\n" for p in pairs))

    summary = {
        "n_pairs": len(pairs),
        "checkpoint_sha": sha256_file(Path(cfg.checkpoint) / "model.safetensors"),
        "tokenizer_sha": sha256_file(Path(cfg.tokenizer_dir) / "tokenizer.json"),
        "seed": cfg.seed,
        "temperature": cfg.temperature,
        "versions": _versions(),
    }
    summary_path = out_dir / "pairgen_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    write_manifest(
        out_dir, "pairgen", cfg, [pairs_path, summary_path],
        inputs={
            "model.safetensors": Path(cfg.checkpoint) / "model.safetensors",
            "tokenizer.json": Path(cfg.tokenizer_dir) / "tokenizer.json",
        },
    )
```

Register in `src/tinyfables/stages/__init__.py` — update the imports and `REGISTRY`:

```python
from tinyfables.config import (
    BenchmarkConfig,
    EvalConfig,
    PairgenConfig,
    PrepConfig,
    PretrainConfig,
    TokenizerConfig,
)

REGISTRY: dict[str, tuple[type, str]] = {
    "tokenizer": (TokenizerConfig, "tinyfables.stages.tokenizer"),
    "prep": (PrepConfig, "tinyfables.stages.prep"),
    "benchmark": (BenchmarkConfig, "tinyfables.stages.benchmark"),
    "pretrain": (PretrainConfig, "tinyfables.stages.pretrain"),
    "evaluate": (EvalConfig, "tinyfables.stages.evaluate"),
    "pairgen": (PairgenConfig, "tinyfables.stages.pairgen"),
}
```

- [ ] **Step 7: Create the configs**

Create `configs/pairgen_toy.yaml`:

```yaml
# Toy pair generation over the fixture (CPU, seconds).
checkpoint: runs/pretrain_toy
tokenizer_dir: runs/tokenizer_toy
source:
  jsonl_path: tests/fixtures/tiny_corpus.jsonl
n_ctx: 512
n_pairs: 4
max_new_tokens: 48
min_new_tokens: 8
temperature: 0.9
top_k: 20
seed: 0
device: cpu
```

Create `configs/pairgen_full.yaml`:

```yaml
# Real pair generation (issue 06, Track B): ~2,000 pairs = 2 samples per val
# FableSpec from the Base Model (~4,000 generations, under an hour on T4).
# checkpoint set at run time to the Base Model dir (runs/base_model or a snapshot
# of congthanh991/tinyfables-13m-base).
checkpoint: runs/base_model
tokenizer_dir: runs/tokenizer_full
source:
  hf_dataset: klusai/ds-tf1-en-3m
  hf_split: validation
  max_rows: 8000
n_ctx: 1024
n_pairs: 2000
max_new_tokens: 320
min_new_tokens: 80
temperature: 0.9
top_k: 50
seed: 0
device: auto
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_pairgen_stage.py tests/test_config.py -q`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add src/tinyfables/config.py src/tinyfables/stages/pairgen.py \
  src/tinyfables/stages/__init__.py configs/pairgen_toy.yaml configs/pairgen_full.yaml \
  tests/test_pairgen_stage.py tests/test_config.py
git commit -m "feat(pairgen): Preference Pairs from the Base Model with checkpoint provenance (issue 06)"
```

---

### Task 8: `label` stage — resumable AI Labeler with injected runner

**Files:**
- Modify: `src/tinyfables/config.py` (add `LabelConfig`)
- Create: `src/tinyfables/stages/label.py`
- Modify: `src/tinyfables/stages/__init__.py` (register `label`)
- Create: `configs/label_toy.yaml`, `configs/label_full.yaml`
- Test: `tests/test_label_stage.py`, additions to `tests/test_config.py`

**Interfaces:**
- Consumes: `labeler.*` (Tasks 5–6), `feedback.AXES`, `stage.write_manifest`/`sha256_file`, `pairs.jsonl` (Task 7), `RUBRIC.md`, `configs/labeler_prompt.yaml`.
- Produces (`stages/label.py`):
  - `run(cfg: LabelConfig, out_dir: Path, runner=None) -> None` — `runner` defaults to `labeler.claude_runner`; tests inject a fake `runner(prompt, model) -> str`. Builds the deterministic work list, resumes from `<out>/labels.jsonl`, batches `batch_size` pairs/call, parses, maps ratings back to stable `(ratings_0, ratings_1)`, appends to the cache, and writes `label_summary.json` + `manifest.json` LAST.
  - `plan_work(pairs, cfg) -> list[dict]` — the deterministic schedule: calibration pass 0 (start), first third of main, calibration pass 1 (middle), second third, swap slice, last third, calibration pass 2 (end). Each item is `{"pair_id","phase","order","fables":[...]}`.
  - `LabelConfig` fields: `pairs` (path to `pairs.jsonl`), `rubric` (`RUBRIC.md`), `labeler_prompt` (`configs/labeler_prompt.yaml`), `model` (logged as `model_version`), `batch_size=5`, `swap_fraction=0.10`, `calibration_size=30`, `seed=0`, `max_batches: int | None = None`.
- Consumed by Task 9 (`derive`) and Task 10 (`audit`), which read `labels.jsonl`.

**Presentation → storage mapping (critical):** an item with `order=="ab"` shows `fables[0]` as Fable A, `fables[1]` as Fable B, so `ratings_0 = label.ratings_a`, `ratings_1 = label.ratings_b`. An item with `order=="ba"` shows `fables[1]` as A, `fables[0]` as B, so `ratings_0 = label.ratings_b`, `ratings_1 = label.ratings_a`. Store `ratings_0`/`ratings_1` (stable order) so `derive`/`audit` are order-agnostic.

- [ ] **Step 1: Add the config test**

Append to `tests/test_config.py`:

```python
def test_label_config_defaults():
    from tinyfables.config import LabelConfig

    cfg = LabelConfig(pairs="runs/pairs/pairs.jsonl", rubric="RUBRIC.md",
                      labeler_prompt="configs/labeler_prompt.yaml", model="claude-opus-4-8")
    assert cfg.batch_size == 5
    assert cfg.swap_fraction == 0.10  # design: 10% position-swap slice
    assert cfg.calibration_size == 30  # design: 30-pair Calibration Set
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/pytest tests/test_config.py::test_label_config_defaults -q`
Expected: FAIL with `ImportError: cannot import name 'LabelConfig'`

- [ ] **Step 3: Add `LabelConfig` to `config.py`**

Insert after `PairgenConfig`:

```python
@dataclass(frozen=True)
class LabelConfig:
    pairs: str
    rubric: str
    labeler_prompt: str
    model: str
    batch_size: int = 5
    swap_fraction: float = 0.10
    calibration_size: int = 30
    seed: int = 0
    max_batches: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.swap_fraction <= 1.0:
            raise ValueError("swap_fraction must be in [0, 1]")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
```

- [ ] **Step 4: Write the failing stage test**

Create `tests/test_label_stage.py`:

```python
import json
from pathlib import Path

from tinyfables.config import LabelConfig
from tinyfables.feedback import AXES
from tinyfables.labeler import cache_key, load_cache
from tinyfables.stages import REGISTRY
from tinyfables.stages import label as label_stage

REPO = Path(__file__).resolve().parents[1]
RUBRIC = str(REPO / "RUBRIC.md")
PROMPT = str(REPO / "configs" / "labeler_prompt.yaml")


def _write_pairs(path, n):
    rows = []
    for i in range(n):
        rows.append({"pair_id": f"pair-{i:06d}", "spec": {}, "prompt": "p",
                     "fables": [f"fable {i} zero", f"fable {i} one"], "seeds": [2 * i, 2 * i + 1]})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _fake_runner_factory():
    """Deterministic fake Claude: rates every fable by a hash of its text so ratings
    vary but are reproducible; echoes back exactly the requested pair_ids. Records the
    number of calls so resume can be asserted."""
    calls = []

    def runner(prompt, model):
        calls.append(prompt)
        # recover requested pair_ids from the rendered prompt (order preserved)
        ids = [ln.split("pair_id: ")[1].strip()
               for ln in prompt.splitlines() if "pair_id: " in ln]

        def rate(seedtext):
            h = abs(hash(seedtext))
            return {a: 1 + (h >> (3 * j)) % 5 for j, a in enumerate(AXES)}

        labels = [{"pair_id": pid,
                   "fable_a": rate(pid + "a"), "fable_b": rate(pid + "b"),
                   "justification": "ok"} for pid in ids]
        return json.dumps({"labels": labels})

    return runner, calls


def _cfg(pairs_path, **kw):
    base = dict(pairs=str(pairs_path), rubric=RUBRIC, labeler_prompt=PROMPT,
                model="fake-model", batch_size=2, swap_fraction=0.5,
                calibration_size=2, seed=0)
    base.update(kw)
    return LabelConfig(**base)


def test_label_registered():
    assert REGISTRY["label"] == (LabelConfig, "tinyfables.stages.label")


def test_label_writes_cache_summary_and_manifest(tmp_path):
    pairs = tmp_path / "pairs.jsonl"; _write_pairs(pairs, 4)
    out = tmp_path / "labels"
    runner, calls = _fake_runner_factory()
    label_stage.run(_cfg(pairs), out, runner=runner)
    assert {"labels.jsonl", "label_summary.json", "manifest.json"} <= {p.name for p in out.iterdir()}
    cache = load_cache(out / "labels.jsonl")
    # every record carries model_version + prompt_version (ADR-0004)
    for rec in cache.values():
        assert rec["model_version"] == "fake-model"
        assert rec["prompt_version"] == 1
        assert set(rec["ratings_0"]) == set(AXES)
    # main labels for all 4 pairs are present
    for i in range(4):
        assert cache_key(f"pair-{i:06d}", "main", "ab", "fake-model", 1) in cache


def test_label_has_swap_slice_and_calibration_passes(tmp_path):
    pairs = tmp_path / "pairs.jsonl"; _write_pairs(pairs, 4)
    out = tmp_path / "labels"
    runner, _ = _fake_runner_factory()
    label_stage.run(_cfg(pairs), out, runner=runner)
    cache = load_cache(out / "labels.jsonl")
    phases = {rec["phase"] for rec in cache.values()}
    orders = {rec["order"] for rec in cache.values()}
    assert "swap" in phases and "ba" in orders          # 10%+ position-swap slice
    assert {"calib-0", "calib-1", "calib-2"} <= phases  # start/middle/end passes
    # swap slice is swap_fraction of pairs (0.5 * 4 = 2)
    assert sum(1 for r in cache.values() if r["phase"] == "swap") == 2


def test_label_resumes_and_skips_cached_work(tmp_path):
    pairs = tmp_path / "pairs.jsonl"; _write_pairs(pairs, 4)
    out = tmp_path / "labels"
    r1, calls1 = _fake_runner_factory()
    label_stage.run(_cfg(pairs), out, runner=r1)
    n_rows = len(load_cache(out / "labels.jsonl"))
    # re-run: everything is cached, so the runner is never called again
    r2, calls2 = _fake_runner_factory()
    label_stage.run(_cfg(pairs), out, runner=r2)
    assert calls2 == []  # full resume, zero new Claude calls
    assert len(load_cache(out / "labels.jsonl")) == n_rows


def test_label_swap_stores_ratings_in_stable_fable_order(tmp_path):
    # a runner that gives fable-A a fixed high rating and fable-B a fixed low one,
    # so we can prove the ba-order row swaps them back into stable (0,1) storage.
    pairs = tmp_path / "pairs.jsonl"; _write_pairs(pairs, 2)
    out = tmp_path / "labels"

    def runner(prompt, model):
        ids = [ln.split("pair_id: ")[1].strip()
               for ln in prompt.splitlines() if "pair_id: " in ln]
        hi = {a: 5 for a in AXES}; lo = {a: 1 for a in AXES}
        return json.dumps({"labels": [
            {"pair_id": pid, "fable_a": hi, "fable_b": lo} for pid in ids]})

    label_stage.run(_cfg(pairs, swap_fraction=1.0, calibration_size=0), out, runner=runner)
    cache = load_cache(out / "labels.jsonl")
    main = cache[cache_key("pair-000000", "main", "ab", "fake-model", 1)]
    swap = cache[cache_key("pair-000000", "swap", "ba", "fake-model", 1)]
    # main/ab: A=fable0 -> ratings_0 high ; swap/ba: A=fable1 -> ratings_0 low
    assert main["ratings_0"]["moral"] == 5 and main["ratings_1"]["moral"] == 1
    assert swap["ratings_0"]["moral"] == 1 and swap["ratings_1"]["moral"] == 5
```

- [ ] **Step 5: Run it to verify it fails**

Run: `.venv/bin/pytest tests/test_label_stage.py -q`
Expected: FAIL — `label` not registered / no module.

- [ ] **Step 6: Implement the stage + register it**

Create `src/tinyfables/stages/label.py`:

```python
"""AI Labeler stage (issue 06, ADR-0004): headless `claude -p` rates Preference Pairs
against RUBRIC.md, strict-JSON per-Axis 1-5, into an append-only JSONL cache that
doubles as the offline replay fixture. Resumable — the deterministic work list is
rebuilt each run and cached keys are skipped, so rate limits or interruptions cost
nothing. The `claude -p` call is injected as `runner` so tests never call Claude.

Torch-free (reads/writes JSONL + subprocess only). The work list places the
Calibration-Set passes at the start/middle/end of the run and a swap slice mid-run
so the audits (Task: audit) measure drift and position bias across the whole run."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from tinyfables.config import LabelConfig
from tinyfables.labeler import (
    append_cache,
    build_batch_prompt,
    cache_key,
    claude_runner,
    load_cache,
    load_prompt,
    parse_labeler_response,
)
from tinyfables.stage import sha256_file, write_manifest


def _read_pairs(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def plan_work(pairs: list[dict], cfg: LabelConfig) -> list[dict]:
    """Deterministic label schedule. Main pass (every pair, order ab) is split into
    thirds with the three Calibration passes interleaved at start/middle/end and the
    position-swap slice (order ba) in the middle."""
    def item(p, phase, order):
        return {"pair_id": p["pair_id"], "phase": phase, "order": order, "fables": p["fables"]}

    rng = random.Random(cfg.seed)
    calib = pairs[: cfg.calibration_size]
    n_swap = int(round(len(pairs) * cfg.swap_fraction))
    swap_ids = set(rng.sample([p["pair_id"] for p in pairs], n_swap)) if n_swap else set()
    swap = [p for p in pairs if p["pair_id"] in swap_ids]

    main = [item(p, "main", "ab") for p in pairs]
    third = max(1, len(main) // 3)
    work: list[dict] = []
    work += [item(p, "calib-0", "ab") for p in calib]
    work += main[:third]
    work += [item(p, "calib-1", "ab") for p in calib]
    work += main[third: 2 * third]
    work += [item(p, "swap", "ba") for p in swap]
    work += main[2 * third:]
    work += [item(p, "calib-2", "ab") for p in calib]
    return work


def _store_ratings(item: dict, label) -> tuple[dict, dict]:
    """Map presentation-order ratings back to stable fable order (ratings_0=fables[0])."""
    if item["order"] == "ab":
        return label.ratings_a, label.ratings_b
    return label.ratings_b, label.ratings_a  # ba: A shown = fables[1]


def _batched(items: list, n: int):
    for i in range(0, len(items), n):
        yield items[i: i + n]


def run(cfg: LabelConfig, out_dir: Path, runner=None) -> None:
    runner = runner or claude_runner
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "labels.jsonl"

    rubric_text = Path(cfg.rubric).read_text()
    prompt_version, template = load_prompt(cfg.labeler_prompt)
    rubric_sha = hashlib.sha256(rubric_text.encode()).hexdigest()
    prompt_sha = hashlib.sha256(template.encode()).hexdigest()

    pairs = _read_pairs(cfg.pairs)
    work = plan_work(pairs, cfg)
    cache = load_cache(cache_path)

    pending = [
        it for it in work
        if cache_key(it["pair_id"], it["phase"], it["order"], cfg.model, prompt_version) not in cache
    ]
    # guard: a cached row whose prompt_version matches but whose rubric/template hash
    # differs means RUBRIC.md changed without a version bump — fail loud, do not mix.
    for rec in cache.values():
        if rec["prompt_version"] == prompt_version and (
            rec.get("rubric_sha") != rubric_sha or rec.get("prompt_sha") != prompt_sha
        ):
            raise ValueError(
                "cache has labels at this prompt_version with a different rubric/prompt hash; "
                "bump `version` in labeler_prompt.yaml before re-labeling (RUBRIC/prompt changed)."
            )

    n_batches = 0
    for batch in _batched(pending, cfg.batch_size):
        if cfg.max_batches is not None and n_batches >= cfg.max_batches:
            break
        prompt_batch = [
            {"pair_id": it["pair_id"],
             "fable_a": it["fables"][0] if it["order"] == "ab" else it["fables"][1],
             "fable_b": it["fables"][1] if it["order"] == "ab" else it["fables"][0]}
            for it in batch
        ]
        prompt = build_batch_prompt(rubric_text, template, prompt_batch)
        response = runner(prompt, cfg.model)
        labels = {l.pair_id: l for l in parse_labeler_response(response, [it["pair_id"] for it in batch])}
        for it in batch:
            r0, r1 = _store_ratings(it, labels[it["pair_id"]])
            append_cache(cache_path, {
                "pair_id": it["pair_id"], "phase": it["phase"], "order": it["order"],
                "model_version": cfg.model, "prompt_version": prompt_version,
                "rubric_sha": rubric_sha, "prompt_sha": prompt_sha,
                "ratings_0": r0, "ratings_1": r1,
                "justification": labels[it["pair_id"]].justification,
            })
        n_batches += 1

    final = load_cache(cache_path)
    summary = {
        "n_pairs": len(pairs),
        "n_label_instances": len(final),
        "n_batches_this_run": n_batches,
        "model_version": cfg.model,
        "prompt_version": prompt_version,
        "rubric_sha": rubric_sha,
        "prompt_sha": prompt_sha,
        "complete": len(final) >= len(work),
    }
    summary_path = out_dir / "label_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_manifest(out_dir, "label", cfg, [cache_path, summary_path],
                   inputs={"pairs.jsonl": Path(cfg.pairs)})
```

Register in `src/tinyfables/stages/__init__.py`: add `LabelConfig` to the import and `"label": (LabelConfig, "tinyfables.stages.label"),` to `REGISTRY`.

- [ ] **Step 7: Create the configs**

Create `configs/label_toy.yaml`:

```yaml
# Toy labeling over a tiny pairs file. Tests inject a fake runner; this config exists
# for CLI-shape validation and the toy chain. `model` is a placeholder here.
pairs: runs/pairgen_toy/pairs.jsonl
rubric: RUBRIC.md
labeler_prompt: configs/labeler_prompt.yaml
model: claude-opus-4-8
batch_size: 2
swap_fraction: 0.5
calibration_size: 2
seed: 0
```

Create `configs/label_full.yaml`:

```yaml
# Real labeling run (issue 06, Track B) on THIS Mac via the Claude subscription.
# Resumable: re-run the same command after any interruption. `model` is pinned and
# logged as model_version on every label.
pairs: runs/pairgen_base/pairs.jsonl
rubric: RUBRIC.md
labeler_prompt: configs/labeler_prompt.yaml
model: claude-opus-4-8
batch_size: 5
swap_fraction: 0.10
calibration_size: 30
seed: 0
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_label_stage.py tests/test_config.py -q`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add src/tinyfables/config.py src/tinyfables/stages/label.py \
  src/tinyfables/stages/__init__.py configs/label_toy.yaml configs/label_full.yaml \
  tests/test_label_stage.py tests/test_config.py
git commit -m "feat(label): resumable AI Labeler stage with injected runner + audits schedule (issue 06)"
```

---

### Task 9: `derive` stage — preferences + weight-sensitivity table

**Files:**
- Modify: `src/tinyfables/config.py` (add `DeriveConfig`)
- Create: `src/tinyfables/stages/derive.py`
- Modify: `src/tinyfables/stages/__init__.py` (register `derive`)
- Create: `tests/fixtures/labels_replay.jsonl`
- Create: `configs/derive_toy.yaml`, `configs/derive_full.yaml`
- Test: `tests/test_derive_stage.py`, additions to `tests/test_config.py`

**Interfaces:**
- Consumes: `feedback.derive_preference`/`aggregate_score`/`weight_sensitivity`/`WEIGHTS`, `labeler.load_cache`, `stage.write_manifest`/`sha256_file`, `labels.jsonl` (Task 8), `pairs.jsonl` (Task 7).
- Produces (`stages/derive.py`):
  - `run(cfg: DeriveConfig, out_dir: Path) -> None` — from the `main`/`ab` label of each pair, derives a preference (ties skipped), writes `preferences.jsonl` (one row per non-tie pair with `chosen`/`rejected` fable text, ratings, aggregates, and a deterministic `split` of `"train"`/`"held_out"`), `sensitivity.json` (the ±`weight_delta` table), `derive_summary.json`, and `manifest.json` LAST.
  - `DeriveConfig` fields: `labels` (path to `labels.jsonl`), `pairs` (path to `pairs.jsonl`), `weight_delta=0.1`, `held_out_fraction=0.10`, `seed=0`.
- Consumed by issue 07 (`preferences.jsonl`).

**Deterministic split:** `split = "held_out" if (int(sha256(pair_id).hexdigest(), 16) % 1000) < round(held_out_fraction*1000) else "train"` — stable across runs, ~`held_out_fraction` of pairs held out.

- [ ] **Step 1: Add the config test**

Append to `tests/test_config.py`:

```python
def test_derive_config_defaults():
    from tinyfables.config import DeriveConfig

    cfg = DeriveConfig(labels="runs/labels/labels.jsonl", pairs="runs/pairs/pairs.jsonl")
    assert cfg.weight_delta == 0.1     # ADR-0003 +/-0.1 sensitivity
    assert cfg.held_out_fraction == 0.10  # ~200/2000 held out for the RM
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/pytest tests/test_config.py::test_derive_config_defaults -q`
Expected: FAIL with `ImportError: cannot import name 'DeriveConfig'`

- [ ] **Step 3: Add `DeriveConfig` to `config.py`**

Insert after `LabelConfig`:

```python
@dataclass(frozen=True)
class DeriveConfig:
    labels: str
    pairs: str
    weight_delta: float = 0.1
    held_out_fraction: float = 0.10
    seed: int = 0
```

- [ ] **Step 4: Create the replay-cache fixture**

Create `tests/fixtures/labels_replay.jsonl` (hand-built to the exact schema Task 8 emits; six pairs, one swap, two calibration pairs re-labeled three times — used by both `derive` and `audit` tests). Every row omits `justification` for brevity, which the parser allows:

```jsonl
{"pair_id": "pair-000000", "phase": "main", "order": "ab", "model_version": "recorded", "prompt_version": 1, "rubric_sha": "fixturesha", "prompt_sha": "fixturesha", "ratings_0": {"moral": 5, "adherence": 4, "coherence": 4, "prose": 4}, "ratings_1": {"moral": 2, "adherence": 3, "coherence": 3, "prose": 3}}
{"pair_id": "pair-000001", "phase": "main", "order": "ab", "model_version": "recorded", "prompt_version": 1, "rubric_sha": "fixturesha", "prompt_sha": "fixturesha", "ratings_0": {"moral": 2, "adherence": 2, "coherence": 3, "prose": 3}, "ratings_1": {"moral": 5, "adherence": 4, "coherence": 4, "prose": 5}}
{"pair_id": "pair-000002", "phase": "main", "order": "ab", "model_version": "recorded", "prompt_version": 1, "rubric_sha": "fixturesha", "prompt_sha": "fixturesha", "ratings_0": {"moral": 3, "adherence": 3, "coherence": 3, "prose": 3}, "ratings_1": {"moral": 3, "adherence": 3, "coherence": 3, "prose": 3}}
{"pair_id": "pair-000003", "phase": "main", "order": "ab", "model_version": "recorded", "prompt_version": 1, "rubric_sha": "fixturesha", "prompt_sha": "fixturesha", "ratings_0": {"moral": 4, "adherence": 5, "coherence": 4, "prose": 4}, "ratings_1": {"moral": 3, "adherence": 2, "coherence": 3, "prose": 4}}
{"pair_id": "pair-000004", "phase": "main", "order": "ab", "model_version": "recorded", "prompt_version": 1, "rubric_sha": "fixturesha", "prompt_sha": "fixturesha", "ratings_0": {"moral": 2, "adherence": 3, "coherence": 2, "prose": 3}, "ratings_1": {"moral": 4, "adherence": 4, "coherence": 5, "prose": 4}}
{"pair_id": "pair-000005", "phase": "main", "order": "ab", "model_version": "recorded", "prompt_version": 1, "rubric_sha": "fixturesha", "prompt_sha": "fixturesha", "ratings_0": {"moral": 5, "adherence": 5, "coherence": 5, "prose": 5}, "ratings_1": {"moral": 1, "adherence": 2, "coherence": 2, "prose": 2}}
{"pair_id": "pair-000000", "phase": "swap", "order": "ba", "model_version": "recorded", "prompt_version": 1, "rubric_sha": "fixturesha", "prompt_sha": "fixturesha", "ratings_0": {"moral": 5, "adherence": 4, "coherence": 4, "prose": 4}, "ratings_1": {"moral": 2, "adherence": 3, "coherence": 3, "prose": 3}}
{"pair_id": "pair-000001", "phase": "swap", "order": "ba", "model_version": "recorded", "prompt_version": 1, "rubric_sha": "fixturesha", "prompt_sha": "fixturesha", "ratings_0": {"moral": 4, "adherence": 4, "coherence": 4, "prose": 4}, "ratings_1": {"moral": 2, "adherence": 2, "coherence": 3, "prose": 3}}
{"pair_id": "pair-000000", "phase": "calib-0", "order": "ab", "model_version": "recorded", "prompt_version": 1, "rubric_sha": "fixturesha", "prompt_sha": "fixturesha", "ratings_0": {"moral": 5, "adherence": 4, "coherence": 4, "prose": 4}, "ratings_1": {"moral": 2, "adherence": 3, "coherence": 3, "prose": 3}}
{"pair_id": "pair-000000", "phase": "calib-1", "order": "ab", "model_version": "recorded", "prompt_version": 1, "rubric_sha": "fixturesha", "prompt_sha": "fixturesha", "ratings_0": {"moral": 5, "adherence": 5, "coherence": 4, "prose": 4}, "ratings_1": {"moral": 2, "adherence": 2, "coherence": 3, "prose": 3}}
{"pair_id": "pair-000000", "phase": "calib-2", "order": "ab", "model_version": "recorded", "prompt_version": 1, "rubric_sha": "fixturesha", "prompt_sha": "fixturesha", "ratings_0": {"moral": 4, "adherence": 4, "coherence": 4, "prose": 4}, "ratings_1": {"moral": 3, "adherence": 3, "coherence": 3, "prose": 3}}
{"pair_id": "pair-000001", "phase": "calib-0", "order": "ab", "model_version": "recorded", "prompt_version": 1, "rubric_sha": "fixturesha", "prompt_sha": "fixturesha", "ratings_0": {"moral": 2, "adherence": 2, "coherence": 3, "prose": 3}, "ratings_1": {"moral": 5, "adherence": 4, "coherence": 4, "prose": 5}}
{"pair_id": "pair-000001", "phase": "calib-1", "order": "ab", "model_version": "recorded", "prompt_version": 1, "rubric_sha": "fixturesha", "prompt_sha": "fixturesha", "ratings_0": {"moral": 5, "adherence": 5, "coherence": 4, "prose": 4}, "ratings_1": {"moral": 2, "adherence": 2, "coherence": 3, "prose": 3}}
{"pair_id": "pair-000001", "phase": "calib-2", "order": "ab", "model_version": "recorded", "prompt_version": 1, "rubric_sha": "fixturesha", "prompt_sha": "fixturesha", "ratings_0": {"moral": 2, "adherence": 2, "coherence": 3, "prose": 3}, "ratings_1": {"moral": 5, "adherence": 5, "coherence": 4, "prose": 4}}
```

Also create a matching minimal `tests/fixtures/pairs_replay.jsonl` (six pairs, so `derive` can attach fable text):

```jsonl
{"pair_id": "pair-000000", "spec": {"character": "a shy octopus"}, "prompt": "p0", "fables": ["fable zero A", "fable zero B"], "seeds": [0, 1]}
{"pair_id": "pair-000001", "spec": {"character": "a bold sparrow"}, "prompt": "p1", "fables": ["fable one A", "fable one B"], "seeds": [2, 3]}
{"pair_id": "pair-000002", "spec": {"character": "a tie pair"}, "prompt": "p2", "fables": ["fable two A", "fable two B"], "seeds": [4, 5]}
{"pair_id": "pair-000003", "spec": {"character": "a clever mole"}, "prompt": "p3", "fables": ["fable three A", "fable three B"], "seeds": [6, 7]}
{"pair_id": "pair-000004", "spec": {"character": "a kind bear"}, "prompt": "p4", "fables": ["fable four A", "fable four B"], "seeds": [8, 9]}
{"pair_id": "pair-000005", "spec": {"character": "a quick hare"}, "prompt": "p5", "fables": ["fable five A", "fable five B"], "seeds": [10, 11]}
```

- [ ] **Step 5: Write the failing stage test**

Create `tests/test_derive_stage.py`:

```python
import json
from pathlib import Path

from tinyfables.config import DeriveConfig
from tinyfables.stages import REGISTRY
from tinyfables.stages import derive as derive_stage

FIX = Path(__file__).parent / "fixtures"
LABELS = str(FIX / "labels_replay.jsonl")
PAIRS = str(FIX / "pairs_replay.jsonl")


def test_derive_registered():
    assert REGISTRY["derive"] == (DeriveConfig, "tinyfables.stages.derive")


def test_derive_emits_preferences_skipping_ties(tmp_path):
    out = tmp_path / "derive"
    derive_stage.run(DeriveConfig(labels=LABELS, pairs=PAIRS), out)
    prefs = [json.loads(l) for l in (out / "preferences.jsonl").read_text().splitlines()]
    # 6 pairs, pair-000002 is an exact tie -> skipped -> 5 preferences
    assert len(prefs) == 5
    ids = {p["pair_id"] for p in prefs}
    assert "pair-000002" not in ids
    row = next(p for p in prefs if p["pair_id"] == "pair-000000")
    # fable 0 wins pair 0 (higher aggregate); chosen text is fables[0]
    assert row["chosen"] == "fable zero A" and row["rejected"] == "fable zero B"
    assert row["aggregate_chosen"] > row["aggregate_rejected"]
    assert row["split"] in {"train", "held_out"}


def test_derive_split_is_deterministic_and_stable(tmp_path):
    a = tmp_path / "a"; b = tmp_path / "b"
    derive_stage.run(DeriveConfig(labels=LABELS, pairs=PAIRS), a)
    derive_stage.run(DeriveConfig(labels=LABELS, pairs=PAIRS), b)
    assert (a / "preferences.jsonl").read_text() == (b / "preferences.jsonl").read_text()


def test_derive_writes_sensitivity_table_and_manifest(tmp_path):
    out = tmp_path / "derive"
    derive_stage.run(DeriveConfig(labels=LABELS, pairs=PAIRS), out)
    sens = json.loads((out / "sensitivity.json").read_text())
    assert sens["delta"] == 0.1
    assert len(sens["rows"]) == 8  # 4 axes x {+,-}
    assert all({"axis", "direction", "n_flipped", "pct_flipped"} <= set(r) for r in sens["rows"])
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["stage"] == "derive"
    assert "labels.jsonl" in manifest["inputs"] and "pairs.jsonl" in manifest["inputs"]
```

- [ ] **Step 6: Run it to verify it fails**

Run: `.venv/bin/pytest tests/test_derive_stage.py -q`
Expected: FAIL — `derive` not registered.

- [ ] **Step 7: Implement the stage + register it**

Create `src/tinyfables/stages/derive.py`:

```python
"""Preference derivation (issue 06): turn the AI Labeler's per-Axis 1-5 ratings into
the stable preferences artifact issue 07 trains on. Uses each pair's canonical
`main`/`ab` label; the Aggregate Score (ADR-0003 fixed weights) picks the preferred
fable, exact ties are skipped, and the +/-0.1 weight-sensitivity table reports how
brittle the preferences are to the weighting. Torch-free."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tinyfables.config import DeriveConfig
from tinyfables.feedback import WEIGHTS, aggregate_score, derive_preference, weight_sensitivity
from tinyfables.labeler import cache_key, load_cache
from tinyfables.stage import sha256_file, write_manifest


def _split_of(pair_id: str, held_out_fraction: float) -> str:
    bucket = int(hashlib.sha256(pair_id.encode()).hexdigest(), 16) % 1000
    return "held_out" if bucket < round(held_out_fraction * 1000) else "train"


def run(cfg: DeriveConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = load_cache(cfg.labels)
    pairs = {json.loads(l)["pair_id"]: json.loads(l)
             for l in Path(cfg.pairs).read_text().splitlines() if l.strip()}

    # canonical labels = phase main, order ab (in pair-id order for stable output)
    mains = sorted(
        (rec for rec in cache.values() if rec["phase"] == "main" and rec["order"] == "ab"),
        key=lambda r: r["pair_id"],
    )

    preferences = []
    pair_ratings = []  # for the sensitivity table
    for rec in mains:
        r0, r1 = rec["ratings_0"], rec["ratings_1"]
        pair_ratings.append((r0, r1))
        pref = derive_preference(r0, r1)
        if pref is None:
            continue  # tie skipped
        pair = pairs[rec["pair_id"]]
        chosen_idx, rejected_idx = (0, 1) if pref == 0 else (1, 0)
        preferences.append({
            "pair_id": rec["pair_id"],
            "prompt": pair["prompt"],
            "chosen": pair["fables"][chosen_idx],
            "rejected": pair["fables"][rejected_idx],
            "ratings_chosen": (r0, r1)[chosen_idx],
            "ratings_rejected": (r0, r1)[rejected_idx],
            "aggregate_chosen": round(aggregate_score((r0, r1)[chosen_idx]), 4),
            "aggregate_rejected": round(aggregate_score((r0, r1)[rejected_idx]), 4),
            "split": _split_of(rec["pair_id"], cfg.held_out_fraction),
        })

    prefs_path = out_dir / "preferences.jsonl"
    prefs_path.write_text("".join(json.dumps(p) + "\n" for p in preferences))

    sensitivity = weight_sensitivity(pair_ratings, WEIGHTS, cfg.weight_delta)
    sens_path = out_dir / "sensitivity.json"
    sens_path.write_text(json.dumps(sensitivity, indent=2, sort_keys=True) + "\n")

    n_train = sum(1 for p in preferences if p["split"] == "train")
    summary = {
        "n_pairs_labeled": len(mains),
        "n_preferences": len(preferences),
        "n_ties_skipped": len(mains) - len(preferences),
        "n_train": n_train,
        "n_held_out": len(preferences) - n_train,
        "weights": WEIGHTS,
    }
    summary_path = out_dir / "derive_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    write_manifest(out_dir, "derive", cfg, [prefs_path, sens_path, summary_path],
                   inputs={"labels.jsonl": Path(cfg.labels), "pairs.jsonl": Path(cfg.pairs)})
```

Register in `stages/__init__.py`: add `DeriveConfig` import + `"derive": (DeriveConfig, "tinyfables.stages.derive"),`.

- [ ] **Step 8: Create the configs**

Create `configs/derive_toy.yaml`:

```yaml
labels: tests/fixtures/labels_replay.jsonl
pairs: tests/fixtures/pairs_replay.jsonl
weight_delta: 0.1
held_out_fraction: 0.10
seed: 0
```

Create `configs/derive_full.yaml`:

```yaml
# Real derivation (issue 06, Track B): over the real labeling cache + pairs.
labels: runs/labels_base/labels.jsonl
pairs: runs/pairgen_base/pairs.jsonl
weight_delta: 0.1
held_out_fraction: 0.10
seed: 0
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_derive_stage.py tests/test_config.py -q`
Expected: PASS

- [ ] **Step 10: Commit**

```bash
git add src/tinyfables/config.py src/tinyfables/stages/derive.py \
  src/tinyfables/stages/__init__.py tests/fixtures/labels_replay.jsonl \
  tests/fixtures/pairs_replay.jsonl configs/derive_toy.yaml configs/derive_full.yaml \
  tests/test_derive_stage.py tests/test_config.py
git commit -m "feat(derive): preferences.jsonl + weight-sensitivity table with train/held-out split (issue 06)"
```

---

### Task 10: `audit` stage — position-swap flip rate + Calibration self-consistency

**Files:**
- Modify: `src/tinyfables/config.py` (add `AuditConfig`)
- Create: `src/tinyfables/stages/audit.py`
- Modify: `src/tinyfables/stages/__init__.py` (register `audit`)
- Create: `configs/audit_toy.yaml`, `configs/audit_full.yaml`
- Test: `tests/test_audit_stage.py`, additions to `tests/test_config.py`

**Interfaces:**
- Consumes: `feedback.position_flip_rate`/`self_consistency`, `labeler.load_cache`, `stage.write_manifest`/`sha256_file`, `labels.jsonl` (Task 8).
- Produces (`stages/audit.py`):
  - `run(cfg: AuditConfig, out_dir: Path) -> None` — computes the two audit numbers over the label cache and writes `audit.json` (`{"position_swap":{...}, "self_consistency":{...}, "gate":{...}}` with a `self_consistency_pass` boolean vs `self_consistency_gate`), `audit_report.md` (report-ready table + a one-line verdict), and `manifest.json` LAST.
  - `AuditConfig` fields: `labels` (path to `labels.jsonl`), `self_consistency_gate=0.85`.
- Consumed by the issue-07 gate (design.md records the numbers).

- [ ] **Step 1: Add the config test**

Append to `tests/test_config.py`:

```python
def test_audit_config_defaults():
    from tinyfables.config import AuditConfig

    cfg = AuditConfig(labels="runs/labels/labels.jsonl")
    assert cfg.self_consistency_gate == 0.85  # design: >= ~85%
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/pytest tests/test_config.py::test_audit_config_defaults -q`
Expected: FAIL with `ImportError: cannot import name 'AuditConfig'`

- [ ] **Step 3: Add `AuditConfig` to `config.py`**

Insert after `DeriveConfig`:

```python
@dataclass(frozen=True)
class AuditConfig:
    labels: str
    self_consistency_gate: float = 0.85
```

- [ ] **Step 4: Write the failing stage test**

Create `tests/test_audit_stage.py`:

```python
import json
from pathlib import Path

from tinyfables.config import AuditConfig
from tinyfables.stages import REGISTRY
from tinyfables.stages import audit as audit_stage

LABELS = str(Path(__file__).parent / "fixtures" / "labels_replay.jsonl")


def test_audit_registered():
    assert REGISTRY["audit"] == (AuditConfig, "tinyfables.stages.audit")


def test_audit_reports_both_numbers_and_gate(tmp_path):
    out = tmp_path / "audit"
    audit_stage.run(AuditConfig(labels=LABELS), out)
    assert {"audit.json", "audit_report.md", "manifest.json"} <= {p.name for p in out.iterdir()}
    a = json.loads((out / "audit.json").read_text())
    # fixture: pair-000000 swap agrees (no flip), pair-000001 swap flips -> 1/2
    assert a["position_swap"]["n_pairs"] == 2
    assert a["position_swap"]["flip_rate"] == 0.5
    # fixture: pair-000000 calib unanimous; pair-000001 calib 2-vs-1 -> mean (1.0, 1/3)
    assert a["self_consistency"]["n_pairs"] == 2
    assert 0.0 <= a["self_consistency"]["mean_agreement"] <= 1.0
    assert isinstance(a["gate"]["self_consistency_pass"], bool)
    report = (out / "audit_report.md").read_text()
    assert "flip rate" in report.lower() and "self-consistency" in report.lower()


def test_audit_gate_flags_below_threshold(tmp_path):
    out = tmp_path / "audit"
    # gate impossibly high so the fixture is guaranteed below it
    audit_stage.run(AuditConfig(labels=LABELS, self_consistency_gate=0.999), out)
    a = json.loads((out / "audit.json").read_text())
    assert a["gate"]["self_consistency_pass"] is False
```

- [ ] **Step 5: Run it to verify it fails**

Run: `.venv/bin/pytest tests/test_audit_stage.py -q`
Expected: FAIL — `audit` not registered.

- [ ] **Step 6: Implement the stage + register it**

Create `src/tinyfables/stages/audit.py`:

```python
"""Labeler audits (issue 06, ADR-0004 drift controls for an AI instrument): the
position-swap flip rate (10% slice labeled in both A/B orders) and the Calibration-Set
self-consistency (30 pairs re-labeled at start/middle/end). We treat the labeler as a
measurement instrument; these are reported numbers that gate issue 07's reward model
(self-consistency >= ~0.85; flip rate reported and reasoned about). Torch-free."""

from __future__ import annotations

import json
from pathlib import Path

from tinyfables.config import AuditConfig
from tinyfables.feedback import position_flip_rate, self_consistency
from tinyfables.labeler import load_cache
from tinyfables.stage import write_manifest


def run(cfg: AuditConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    instances = list(load_cache(cfg.labels).values())

    swap = position_flip_rate(instances)
    consistency = self_consistency(instances)
    passed = consistency["mean_agreement"] >= cfg.self_consistency_gate

    audit = {
        "position_swap": swap,
        "self_consistency": consistency,
        "gate": {
            "self_consistency_gate": cfg.self_consistency_gate,
            "self_consistency_pass": passed,
        },
    }
    audit_path = out_dir / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    verdict = "PASS" if passed else "BELOW GATE — flag for issue 07"
    report_lines = [
        "# Labeler audit report",
        "",
        "The AI Labeler is treated as a measurement instrument (ADR-0004).",
        "",
        "| audit | number |",
        "|---|---|",
        f"| position-swap flip rate | {swap['flip_rate']:.3f} "
        f"({swap['n_flipped']}/{swap['n_pairs']} pairs) |",
        f"| Calibration self-consistency (mean pairwise agreement) | "
        f"{consistency['mean_agreement']:.3f} "
        f"({consistency['n_unanimous']}/{consistency['n_pairs']} unanimous) |",
        "",
        f"**Self-consistency gate ({cfg.self_consistency_gate:.2f}):** {verdict}",
        "",
        "_Flip rate is reported and reasoned about — a high flip rate means presentation "
        "order moved ratings (position bias), not that the labeler is unusable._",
        "",
    ]
    report_path = out_dir / "audit_report.md"
    report_path.write_text("\n".join(report_lines))

    write_manifest(out_dir, "audit", cfg, [audit_path, report_path],
                   inputs={"labels.jsonl": Path(cfg.labels)})
```

Register in `stages/__init__.py`: add `AuditConfig` import + `"audit": (AuditConfig, "tinyfables.stages.audit"),`.

- [ ] **Step 7: Create the configs**

Create `configs/audit_toy.yaml`:

```yaml
labels: tests/fixtures/labels_replay.jsonl
self_consistency_gate: 0.85
```

Create `configs/audit_full.yaml`:

```yaml
# Real labeler audit (issue 06, Track B) over the real labeling cache.
labels: runs/labels_base/labels.jsonl
self_consistency_gate: 0.85
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_audit_stage.py tests/test_config.py -q`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add src/tinyfables/config.py src/tinyfables/stages/audit.py \
  src/tinyfables/stages/__init__.py configs/audit_toy.yaml configs/audit_full.yaml \
  tests/test_audit_stage.py tests/test_config.py
git commit -m "feat(audit): position-swap flip rate + calibration self-consistency vs gate (issue 06)"
```

---

### Task 11: Full-config validation, CLI toy chain, design/issue docs

**Files:**
- Modify: `tests/test_full_configs.py`
- Create: `tests/test_feedback_chain.py` (end-to-end pairgen→label(fake)→derive→audit over the fixture)
- Modify: `docs/design.md` ("Implementation (issue 06)" Track A note)
- Test run: full suite

**Interfaces:**
- Consumes: every stage above.
- Produces: a smoke test proving the four stages compose, and validation that the `_full` configs load with the right shapes.

- [ ] **Step 1: Write the full-config validation test**

Append to `tests/test_full_configs.py`:

```python
def test_pairgen_full_targets_2000_pairs_at_temp_0_9():
    from tinyfables.config import PairgenConfig

    cfg = load_config(REPO / "configs" / "pairgen_full.yaml", PairgenConfig)
    assert cfg.n_pairs == 2000
    assert cfg.temperature == 0.9
    assert cfg.source.hf_split == "validation"  # specs from the val split


def test_label_full_uses_rubric_and_versioned_prompt():
    from tinyfables.config import LabelConfig

    cfg = load_config(REPO / "configs" / "label_full.yaml", LabelConfig)
    assert cfg.rubric == "RUBRIC.md"
    assert cfg.labeler_prompt == "configs/labeler_prompt.yaml"
    assert cfg.swap_fraction == 0.10 and cfg.calibration_size == 30
    assert cfg.model  # a pinned model id is logged as model_version


def test_derive_and_audit_full_configs_load():
    from tinyfables.config import AuditConfig, DeriveConfig

    d = load_config(REPO / "configs" / "derive_full.yaml", DeriveConfig)
    assert d.weight_delta == 0.1
    a = load_config(REPO / "configs" / "audit_full.yaml", AuditConfig)
    assert a.self_consistency_gate == 0.85
```

- [ ] **Step 2: Write the end-to-end chain test**

Create `tests/test_feedback_chain.py`:

```python
import json
from pathlib import Path

from tinyfables.config import (
    AuditConfig, DeriveConfig, LabelConfig, PairgenConfig,
    PretrainConfig, PrepConfig, SourceSpec, TokenizerConfig,
)
from tinyfables.feedback import AXES
from tinyfables.stages import audit as audit_stage
from tinyfables.stages import derive as derive_stage
from tinyfables.stages import label as label_stage
from tinyfables.stages import pairgen as pairgen_stage
from tinyfables.stages import prep as prep_stage
from tinyfables.stages import pretrain as pretrain_stage
from tinyfables.stages import tokenizer as tokenizer_stage

REPO = Path(__file__).resolve().parents[1]
FIXTURE = str(Path(__file__).parent / "fixtures" / "tiny_corpus.jsonl")
SRC = SourceSpec(jsonl_path=FIXTURE)


def _fake_runner(prompt, model):
    ids = [ln.split("pair_id: ")[1].strip()
           for ln in prompt.splitlines() if "pair_id: " in ln]

    def rate(seedtext):
        h = abs(hash(seedtext))
        return {a: 1 + (h >> (3 * j)) % 5 for j, a in enumerate(AXES)}

    return json.dumps({"labels": [
        {"pair_id": pid, "fable_a": rate(pid + "a"), "fable_b": rate(pid + "b")}
        for pid in ids]})


def test_feedback_path_composes_offline(tmp_path):
    tok = tmp_path / "tok"; prep = tmp_path / "prep"; ckpt = tmp_path / "ckpt"
    tokenizer_stage.run(TokenizerConfig(source=SRC, vocab_size=512, seed=0), tok)
    prep_stage.run(PrepConfig(source=SRC, tokenizer_dir=str(tok), window=512, seed=0), prep)
    pretrain_stage.run(PretrainConfig(prep_dir=str(prep), tokenizer_dir=str(tok),
                       n_layer=2, n_head=2, d_model=64, n_ctx=512, batch_size=4,
                       steps=4, warmup_steps=0, seed=0, device="cpu"), ckpt)

    pairs_dir = tmp_path / "pairs"
    pairgen_stage.run(PairgenConfig(checkpoint=str(ckpt), tokenizer_dir=str(tok), source=SRC,
                      n_ctx=512, n_pairs=6, max_new_tokens=24, min_new_tokens=8,
                      top_k=10, seed=0, device="cpu"), pairs_dir)

    labels_dir = tmp_path / "labels"
    label_stage.run(LabelConfig(pairs=str(pairs_dir / "pairs.jsonl"), rubric=str(REPO / "RUBRIC.md"),
                    labeler_prompt=str(REPO / "configs" / "labeler_prompt.yaml"),
                    model="fake-model", batch_size=2, swap_fraction=0.5, calibration_size=2, seed=0),
                    labels_dir, runner=_fake_runner)

    derive_dir = tmp_path / "derive"
    derive_stage.run(DeriveConfig(labels=str(labels_dir / "labels.jsonl"),
                     pairs=str(pairs_dir / "pairs.jsonl")), derive_dir)

    audit_dir = tmp_path / "audit"
    audit_stage.run(AuditConfig(labels=str(labels_dir / "labels.jsonl")), audit_dir)

    # each stage produced its manifest-marked artifact set
    assert (pairs_dir / "manifest.json").exists()
    assert (labels_dir / "labels.jsonl").exists()
    assert (derive_dir / "preferences.jsonl").exists()
    assert json.loads((audit_dir / "audit.json").read_text())["self_consistency"]["n_pairs"] == 2
```

- [ ] **Step 3: Run both new tests to verify they pass**

Run: `.venv/bin/pytest tests/test_full_configs.py tests/test_feedback_chain.py -q`
Expected: PASS

- [ ] **Step 4: Run the whole suite offline**

Run: `.venv/bin/pytest -q`
Expected: PASS (all prior tests + the new feedback tests; `network`-marked tests deselected).

- [ ] **Step 5: Add the Track-A design note**

In `docs/design.md`, under the "Feedback stage" section, add an `### Implementation (issue 06, Track A)` subsection recording the built artifacts (mirror the issue-05 note style):

```markdown
### Implementation (issue 06, Track A)

- **Rubric + labeler prompt (human artifacts).** `RUBRIC.md` (four Axes, anchored 1-5
  definitions with example snippets) and `configs/labeler_prompt.yaml` (versioned
  instruction template, `{rubric}`/`{pairs}` slots) are the human-authored definition
  of "good". `prompt_version` is logged on every label; changing the rubric/template
  without bumping `version` is a loud error at label time.
- **`feedback.py` (torch-free).** Single source of truth for the ADR-0003 weights
  (0.4/0.3/0.2/0.1); `aggregate_score`, `derive_preference` (ties skipped),
  `weight_sensitivity` (±0.1 table), and the two audit metrics
  (`position_flip_rate`, `self_consistency`).
- **`pairgen` stage.** Mirrors `evaluate`: 2 independent samples (temp 0.9, seeded
  per sample) per val FableSpec under the Canonical Prompt → `pairs.jsonl`;
  `pairgen_summary.json` records the Base Model `checkpoint_sha` so preferences trace
  back to the exact model.
- **`label` stage (AI Labeler, ADR-0004).** Resumable headless `claude -p` behind an
  injected `runner`; append-only `labels.jsonl` cache keyed by
  `(pair_id, phase, order, model_version, prompt_version)` doubles as the offline
  replay fixture. Work list interleaves the Calibration passes (start/middle/end) and
  the 10% position-swap slice. Tests use a deterministic fake runner; one schema
  contract test validates parsing against a recorded sample.
- **`derive` + `audit` stages.** `derive` → `preferences.jsonl` (chosen/rejected +
  aggregates + deterministic train/held-out split) + `sensitivity.json`; `audit` →
  `audit_report.md`/`audit.json` (flip rate + self-consistency vs the ≥0.85 gate).
- **Track B (real run): _pending_** — gate numbers recorded below after the labeling run.
```

- [ ] **Step 6: Commit**

```bash
git add tests/test_full_configs.py tests/test_feedback_chain.py docs/design.md
git commit -m "test(feedback): full-config validation + offline pairgen->label->derive->audit chain; design note (issue 06 Track A)"
```

- [ ] **Step 7: Merge Track A to main**

Use `superpowers:finishing-a-development-branch` to merge `feat/issue-06-feedback-data` into `main` after review. **Do not tick the issue-06 acceptance boxes yet** — those close at the end of Track B.

---

# TRACK B — Operations (gated on human approval)

> Track B runs on real hardware + the real Claude subscription. It is **not** subagent-driven; it is executed with the human, who approves the Rubric first. Each step records a real number into `docs/design.md`.

---

### Task 12: Rubric approval gate (human)

- [ ] **Step 1:** Present `RUBRIC.md` + `configs/labeler_prompt.yaml` to the human for review. Refine the anchors/examples per their feedback in-session.
- [ ] **Step 2:** On approval, if any wording changed, bump `version` in `configs/labeler_prompt.yaml` and commit both files:

```bash
git add RUBRIC.md configs/labeler_prompt.yaml
git commit -m "docs(rubric): human-approved Rubric + labeler prompt (issue 06)"
```

**Do not proceed to Task 14 (labeling) until this commit exists** — the rubric is the definition of "good" for the whole project and MUST predate any real label.

---

### Task 13: Real pair generation (~4,000 generations)

- [ ] **Step 1:** Ensure the Base Model + tokenizer are local (`runs/base_model`, `runs/tokenizer_full`) — download a snapshot of `congthanh991/tinyfables-13m-base` if needed. `pairgen_summary.json` records the `checkpoint_sha` of exactly the `model.safetensors` used; note it in the design record so the preferences trace back to this specific Base Model snapshot.
- [ ] **Step 2:** Run pairgen. On Colab T4 (via colab-mcp) or local MPS overnight:

```bash
python -m tinyfables run pairgen --config configs/pairgen_full.yaml --out runs/pairgen_base
```

Expected: `runs/pairgen_base/pairs.jsonl` with ~2,000 pairs (~4,000 generations, under ~1h on T4), plus `pairgen_summary.json` + `manifest.json`.

- [ ] **Step 3:** Spot-check `pairs.jsonl` — pairs are non-empty, the two fables per pair differ, prompts are canonical band-B. Record `n_pairs` + `checkpoint_sha` for the design note.

---

### Task 14: Capture a real labeler sample → replace the schema fixture

- [ ] **Step 1:** With the approved rubric/prompt, run the `label` stage bounded to a single batch so exactly one real `claude -p` call happens (uses the default real runner):

```bash
python -m tinyfables run label --config configs/label_full.yaml --out runs/labels_base
# (interrupt after the first batch, or temporarily set max_batches: 1 in a copy of the config)
```

- [ ] **Step 2:** Capture the verbatim strict-JSON of that first real response into `tests/fixtures/labeler_response_sample.json` (overwrite the provisional Track-A fixture with real Claude output). If the real output includes surrounding prose, keep the JSON object only (the parser tolerates prose, but the committed sample should be the clean object matching what `parse_labeler_response` accepts).
- [ ] **Step 3:** Re-run the schema contract test — it MUST stay green against the real capture (this is what pins our schema to Claude's real output, per the derive-fixtures-from-real-data lesson):

```bash
.venv/bin/pytest tests/test_labeler.py::test_recorded_sample_is_the_schema_contract -q
```

If it fails, the real output deviates from the designed schema → adjust `configs/labeler_prompt.yaml` (bump `version`), re-capture, and commit. Then:

```bash
git add tests/fixtures/labeler_response_sample.json configs/labeler_prompt.yaml
git commit -m "test(labeler): pin schema contract to a real claude -p capture (issue 06 Track B)"
```

---

### Task 15: Real labeling run (resumable, on the Mac)

- [ ] **Step 1:** Run the full labeling to completion on this Mac; re-run the same command after any rate-limit/interruption (it resumes from `runs/labels_base/labels.jsonl` for free):

```bash
python -m tinyfables run label --config configs/label_full.yaml --out runs/labels_base
```

Expected: `labels.jsonl` grows to ~2,000 main + ~200 swap + ~90 calibration instances (batched ~5/call, an afternoon of unattended calls). `label_summary.json` reports `"complete": true` when the whole work list is cached. `manifest.json` is written on completion.

- [ ] **Step 2:** Confirm every record carries `model_version` + `prompt_version` (acceptance criterion) — spot-check a few lines of `labels.jsonl`.

- [ ] **Step 3 (operational note):** run from a clean working dir or accept that the repo `CLAUDE.md`/hooks load into the labeler's context; the labeler prompt is fully self-contained, so this does not change ratings, but note it in the run log. Do NOT use `--bare` (it disables subscription auth).

---

### Task 16: Real audits + derivation; record gate numbers

- [ ] **Step 1:** Run the audit and derivation over the real cache:

```bash
python -m tinyfables run audit  --config configs/audit_full.yaml  --out runs/audit_base
python -m tinyfables run derive --config configs/derive_full.yaml --out runs/derive_base
```

- [ ] **Step 2:** Read `runs/audit_base/audit.json` (flip rate + self-consistency) and `runs/derive_base/{derive_summary.json,sensitivity.json}` (n_preferences, n_train/n_held_out, ties skipped, sensitivity flips).

- [ ] **Step 3:** Record the real numbers in `docs/design.md` (replace the "Track B: _pending_" line under the issue-06 note) — an audit table with the flip rate + self-consistency and the pass/flag verdict vs the ≥0.85 gate, plus the preferences count and sensitivity summary. If self-consistency is below gate, flag it explicitly and reason about it (do not silently pass to issue 07).

```bash
git add docs/design.md
git commit -m "docs(feedback): record real labeler audit + preference numbers (issue 06 Track B)"
```

---

### Task 17: Publish the preferences artifact + close the issue

- [ ] **Step 1:** Make the preferences artifact available to issue 07 — push `preferences.jsonl` (+ `RUBRIC.md`, `configs/labeler_prompt.yaml`, `audit.json`) as the `tinyfables-preferences` Hub dataset, carded explicitly as **RLAIF** (AI-labeled), per the design's deliverables list. Reuse the `hub.py` push pattern; keep the registry lazy.
- [ ] **Step 2:** Tick the issue-06 acceptance criteria in `docs/issues/06-feedback-data-labeler.md`:
  - Rubric committed before the real labeling run ✓ (Task 12 precedes Task 15)
  - Pair-generation stage produces a pairs artifact from any checkpoint ✓
  - Labeler resumes mid-run from its cache; replay fixtures let the stage run offline ✓
  - Schema contract test validates strict-JSON ratings against a recorded live sample ✓ (Task 14)
  - Every label carries model version and prompt version ✓
  - Audit report artifact: position-swap flip rate + Calibration self-consistency ✓
  - Derivation emits preferences + the weight-sensitivity table ✓
  - Real run complete: ~2,000 pairs labeled; audit numbers within gates or explicitly flagged ✓
- [ ] **Step 3:** Commit the ticked issue + final `docs/design.md`:

```bash
git add docs/issues/06-feedback-data-labeler.md docs/design.md
git commit -m "docs: close issue 06 — feedback data path complete (rubric, pairs, labeler, audits)"
```

---

## Self-Review

**1. Spec coverage** (issue acceptance criteria + design "Feedback stage" bullets):

| Requirement | Task |
|---|---|
| Anchored Rubric, committed before labeling | Task 4 (draft) + Task 12 (approve, gates Task 15) |
| Labeler prompt versioned in the repo | Task 4 (`configs/labeler_prompt.yaml`, `version`) |
| Pair-generation stage from any checkpoint (2 samples, temp 0.9, val specs) | Task 7 |
| Pairs record Base Model checkpoint hash | Task 7 (`pairgen_summary.json.checkpoint_sha` + manifest inputs) |
| AI Labeler: headless `claude -p`, batched ~5, strict-JSON per-Axis 1–5 both fables | Tasks 5, 6, 8 |
| Append-only JSONL cache keyed by pair id + model version + prompt version | Task 6 (`cache_key`) + Task 8 |
| Resumable; interruptions cost nothing | Task 8 (`plan_work` + skip cached) + Task 15 |
| Cache doubles as replay fixture; tests never call Claude | Tasks 8, 9, 10 (fake runner + `labels_replay.jsonl`) |
| One schema contract test vs recorded live sample | Task 5 (`test_recorded_sample_is_the_schema_contract`) + Task 14 (real capture) |
| Every label carries model + prompt version | Task 8 (record fields) + Task 15 Step 2 |
| Aggregate Score 0.4/0.3/0.2/0.1, ties skipped | Task 1 (`WEIGHTS`, `derive_preference`) + Task 9 |
| ±0.1 weight-sensitivity table | Task 2 + Task 9 (`sensitivity.json`) |
| Position-swap audit (10% slice, both orders) → flip rate | Task 3 + Task 8 (`swap` phase) + Task 10 |
| Calibration Set (30 pairs, start/middle/end) → self-consistency ≥ ~85% gate | Task 3 + Task 8 (`plan_work` interleave) + Task 10 (`gate`) |
| Audit report artifact | Task 10 (`audit_report.md`/`audit.json`) |
| Preferences artifact stable + documented for issue 07 (~1,800/200) | Task 9 (`preferences.jsonl` + split) + Task 17 |
| Real run + audit numbers in design.md | Tasks 13–16 |

**2. Placeholder scan:** No "TBD"/"handle edge cases"/"similar to Task N". Every code step shows complete code; every test step shows real assertions. The only intentionally-deferred content is the Track-B real capture (Task 14) and real gate numbers (Task 16) — those are operational outputs, not plan placeholders, and the provisional fixture (Task 5) is complete and valid on its own.

**3. Type consistency:** Checked across tasks — `WEIGHTS`/`AXES` defined once (Task 1), imported everywhere; `derive_preference` returns `0|1|None` used identically in Tasks 2/3/9/10; cache-record shape (`pair_id/phase/order/model_version/prompt_version/ratings_0/ratings_1/rubric_sha/prompt_sha/justification`) is written in Task 8 and read by `cache_key`/`load_cache` (Task 6), `derive` (Task 9), `audit` (Task 10), and the `labels_replay.jsonl` fixture (Task 9) — all consistent; `PairLabel.ratings_a/ratings_b` (Task 5) mapped to stable `ratings_0/ratings_1` by `_store_ratings` (Task 8); stage `run(cfg, out_dir)` signature matches the CLI dispatch (`run_fn(cfg, Path(args.out))`), with `label.run` adding an optional `runner=None` that defaults to real (CLI-safe). Config classes registered in `REGISTRY` match `load_config` usage.

---

## Execution notes

- **Offline default:** all Track A tests run under `.venv/bin/pytest -q` with no network (the `label` tests inject a fake runner; `pairgen`/chain tests train a tiny CPU checkpoint over the fixture). `network`-marked tests remain deselected by `pyproject.toml`.
- **Lazy registry preserved:** `feedback.py`, `labeler.py`, `label.py`, `derive.py`, `audit.py` never import torch; only `pairgen.py` does, and only at dispatch.
- **Determinism:** `pairgen` (seed+checkpoint), `derive`, `audit`, and `label` with the deterministic fake runner all produce stable outputs; real `label` is stochastic by nature (Claude) and that is expected and documented.
