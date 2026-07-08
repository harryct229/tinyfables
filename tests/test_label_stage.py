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
        rows.append(
            {
                "pair_id": f"pair-{i:06d}",
                "spec": {},
                "prompt": "p",
                "fables": [f"fable {i} zero", f"fable {i} one"],
                "seeds": [2 * i, 2 * i + 1],
            }
        )
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _fake_runner_factory():
    """Deterministic fake Claude for resume/caching tests."""
    calls = []

    def runner(prompt, model):
        calls.append(prompt)
        ids = [ln.split("pair_id: ")[1].strip() for ln in prompt.splitlines() if "pair_id: " in ln]

        def rate(seedtext):
            h = abs(hash(seedtext))
            return {a: 1 + (h >> (3 * j)) % 5 for j, a in enumerate(AXES)}

        labels = [
            {
                "pair_id": pid,
                "fable_a": rate(pid + "a"),
                "fable_b": rate(pid + "b"),
                "justification": "ok",
            }
            for pid in ids
        ]
        return json.dumps({"labels": labels})

    return runner, calls


def _cfg(pairs_path, **kw):
    base = dict(
        pairs=str(pairs_path),
        rubric=RUBRIC,
        labeler_prompt=PROMPT,
        model="fake-model",
        batch_size=2,
        swap_fraction=0.5,
        calibration_size=2,
        seed=0,
    )
    base.update(kw)
    return LabelConfig(**base)


def test_label_registered():
    assert REGISTRY["label"] == (LabelConfig, "tinyfables.stages.label")


def test_label_writes_cache_summary_and_manifest(tmp_path):
    pairs = tmp_path / "pairs.jsonl"
    _write_pairs(pairs, 4)
    out = tmp_path / "labels"
    runner, calls = _fake_runner_factory()
    label_stage.run(_cfg(pairs), out, runner=runner)
    assert calls
    assert {"labels.jsonl", "label_summary.json", "manifest.json"} <= {p.name for p in out.iterdir()}
    cache = load_cache(out / "labels.jsonl")
    for rec in cache.values():
        assert rec["model_version"] == "fake-model"
        assert rec["prompt_version"] == 1
        assert set(rec["ratings_0"]) == set(AXES)
    for i in range(4):
        assert cache_key(f"pair-{i:06d}", "main", "ab", "fake-model", 1) in cache


def test_label_has_swap_slice_and_calibration_passes(tmp_path):
    pairs = tmp_path / "pairs.jsonl"
    _write_pairs(pairs, 4)
    out = tmp_path / "labels"
    runner, _ = _fake_runner_factory()
    label_stage.run(_cfg(pairs), out, runner=runner)
    cache = load_cache(out / "labels.jsonl")
    phases = {rec["phase"] for rec in cache.values()}
    orders = {rec["order"] for rec in cache.values()}
    assert "swap" in phases and "ba" in orders
    assert {"calib-0", "calib-1", "calib-2"} <= phases
    assert sum(1 for r in cache.values() if r["phase"] == "swap") == 2


def test_label_resumes_and_skips_cached_work(tmp_path):
    pairs = tmp_path / "pairs.jsonl"
    _write_pairs(pairs, 4)
    out = tmp_path / "labels"
    r1, calls1 = _fake_runner_factory()
    label_stage.run(_cfg(pairs), out, runner=r1)
    assert calls1
    n_rows = len(load_cache(out / "labels.jsonl"))
    r2, calls2 = _fake_runner_factory()
    label_stage.run(_cfg(pairs), out, runner=r2)
    assert calls2 == []
    assert len(load_cache(out / "labels.jsonl")) == n_rows


def test_label_partial_bounded_run_writes_summary_without_manifest(tmp_path):
    pairs = tmp_path / "pairs.jsonl"
    _write_pairs(pairs, 4)
    out = tmp_path / "labels"
    runner, calls = _fake_runner_factory()

    label_stage.run(_cfg(pairs, max_batches=1), out, runner=runner)

    assert calls
    assert (out / "labels.jsonl").exists()
    assert (out / "label_summary.json").exists()
    assert not (out / "manifest.json").exists()

    summary = json.loads((out / "label_summary.json").read_text())
    assert summary["complete"] is False
    assert summary["n_batches_this_run"] == 1
    assert summary["n_expected_current"] == len(label_stage.plan_work(label_stage._read_pairs(str(pairs)), _cfg(pairs)))
    assert summary["n_cached_current"] < summary["n_expected_current"]
    assert summary["n_pending_current"] > 0


def test_label_manifest_ignores_other_cohorts_and_is_removed_when_incomplete(tmp_path):
    pairs = tmp_path / "pairs.jsonl"
    _write_pairs(pairs, 4)
    out = tmp_path / "labels"
    runner, _ = _fake_runner_factory()

    label_stage.run(_cfg(pairs), out, runner=runner)
    assert (out / "manifest.json").exists()

    label_stage.run(_cfg(pairs, model="other-model", max_batches=0), out, runner=runner)

    summary = json.loads((out / "label_summary.json").read_text())
    assert summary["model_version"] == "other-model"
    assert summary["complete"] is False
    assert summary["n_cached_current"] == 0
    assert summary["n_pending_current"] == summary["n_expected_current"]
    assert summary["n_label_instances"] > 0
    assert not (out / "manifest.json").exists()


def test_label_swap_stores_ratings_in_stable_fable_order(tmp_path):
    pairs = tmp_path / "pairs.jsonl"
    _write_pairs(pairs, 2)
    out = tmp_path / "labels"

    def runner(prompt, model):
        ids = [ln.split("pair_id: ")[1].strip() for ln in prompt.splitlines() if "pair_id: " in ln]
        hi = {a: 5 for a in AXES}
        lo = {a: 1 for a in AXES}
        return json.dumps(
            {"labels": [{"pair_id": pid, "fable_a": hi, "fable_b": lo} for pid in ids]}
        )

    label_stage.run(_cfg(pairs, swap_fraction=1.0, calibration_size=0), out, runner=runner)
    cache = load_cache(out / "labels.jsonl")
    main = cache[cache_key("pair-000000", "main", "ab", "fake-model", 1)]
    swap = cache[cache_key("pair-000000", "swap", "ba", "fake-model", 1)]
    assert main["ratings_0"]["moral"] == 5 and main["ratings_1"]["moral"] == 1
    assert swap["ratings_0"]["moral"] == 1 and swap["ratings_1"]["moral"] == 5
