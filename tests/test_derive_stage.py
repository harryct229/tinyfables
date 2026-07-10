import json
from pathlib import Path

import pytest

from tinyfables.config import DeriveConfig
from tinyfables.feedback import WEIGHTS, aggregate_score
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
    prefs = [json.loads(line) for line in (out / "preferences.jsonl").read_text().splitlines()]
    assert len(prefs) == 5
    ids = {pref["pair_id"] for pref in prefs}
    assert "pair-000002" not in ids
    row = next(pref for pref in prefs if pref["pair_id"] == "pair-000000")
    assert row["chosen"] == "fable zero A" and row["rejected"] == "fable zero B"
    assert row["aggregate_chosen"] > row["aggregate_rejected"]
    assert row["split"] in {"train", "held_out"}


def test_derive_split_is_deterministic_and_stable(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    derive_stage.run(DeriveConfig(labels=LABELS, pairs=PAIRS), a)
    derive_stage.run(DeriveConfig(labels=LABELS, pairs=PAIRS), b)
    assert (a / "preferences.jsonl").read_text() == (b / "preferences.jsonl").read_text()


def test_derive_writes_sensitivity_table_and_manifest(tmp_path):
    out = tmp_path / "derive"
    derive_stage.run(DeriveConfig(labels=LABELS, pairs=PAIRS), out)
    sens = json.loads((out / "sensitivity.json").read_text())
    assert sens["delta"] == 0.1
    assert len(sens["rows"]) == 8
    assert all({"axis", "direction", "n_flipped", "pct_flipped"} <= set(row) for row in sens["rows"])
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["stage"] == "derive"
    assert "labels.jsonl" in manifest["inputs"] and "pairs.jsonl" in manifest["inputs"]


def test_derive_rejects_mixed_cohorts_without_summary(tmp_path):
    labels_path = tmp_path / "labels.jsonl"
    labels_path.write_text((FIX / "labels_replay.jsonl").read_text())
    duplicate = {
        "pair_id": "pair-000000",
        "phase": "main",
        "order": "ab",
        "model_version": "zz-latest",
        "prompt_version": 2,
        "rubric_sha": "fixturesha",
        "prompt_sha": "fixturesha",
        "ratings_0": {"moral": 1, "adherence": 1, "coherence": 1, "prose": 1},
        "ratings_1": {"moral": 5, "adherence": 5, "coherence": 5, "prose": 5},
    }
    with labels_path.open("a") as fh:
        fh.write(json.dumps(duplicate) + "\n")

    out = tmp_path / "derive"
    with pytest.raises(ValueError, match="multiple label cohorts"):
        derive_stage.run(DeriveConfig(labels=str(labels_path), pairs=PAIRS), out)


def test_derive_filters_to_complete_summary_cohort(tmp_path):
    labels_path = tmp_path / "labels.jsonl"
    labels_path.write_text((FIX / "labels_replay.jsonl").read_text())
    duplicate = {
        "pair_id": "pair-000000",
        "phase": "main",
        "order": "ab",
        "model_version": "zz-latest",
        "prompt_version": 2,
        "rubric_sha": "fixturesha",
        "prompt_sha": "fixturesha",
        "ratings_0": {"moral": 1, "adherence": 1, "coherence": 1, "prose": 1},
        "ratings_1": {"moral": 5, "adherence": 5, "coherence": 5, "prose": 5},
    }
    with labels_path.open("a") as fh:
        fh.write(json.dumps(duplicate) + "\n")
    (tmp_path / "label_summary.json").write_text(
        json.dumps(
            {
                "complete": True,
                "model_version": "recorded",
                "prompt_version": 1,
                "n_cached_current": 14,
            }
        )
    )
    (tmp_path / "manifest.json").write_text("{}\n")

    out = tmp_path / "derive"
    derive_stage.run(DeriveConfig(labels=str(labels_path), pairs=PAIRS), out)

    prefs = [json.loads(line) for line in (out / "preferences.jsonl").read_text().splitlines()]
    row = next(pref for pref in prefs if pref["pair_id"] == "pair-000000")
    assert row["chosen"] == "fable zero A"
    assert row["rejected"] == "fable zero B"
    assert row["aggregate_chosen"] == round(aggregate_score({"moral": 5, "adherence": 4, "coherence": 4, "prose": 4}, WEIGHTS), 4)
    assert row["aggregate_rejected"] == round(aggregate_score({"moral": 2, "adherence": 3, "coherence": 3, "prose": 3}, WEIGHTS), 4)

    summary = json.loads((out / "derive_summary.json").read_text())
    assert summary["n_pairs_labeled"] == 6
    assert summary["n_preferences"] == 5
    assert summary["n_ties_skipped"] == 1

    sens = json.loads((out / "sensitivity.json").read_text())
    assert sens["n_base_preferences"] == 5


# --- ADR-0005 majority-vote derive ------------------------------------------

HI = {"moral": 5, "adherence": 5, "coherence": 5, "prose": 5}
LO = {"moral": 1, "adherence": 1, "coherence": 1, "prose": 1}
TIE = ({"moral": 3, "adherence": 3, "coherence": 3, "prose": 3}, {"moral": 3, "adherence": 3, "coherence": 3, "prose": 3})


def _label_rec(pair_id, ratings_0, ratings_1, phase="main", order="ab"):
    return {
        "pair_id": pair_id,
        "phase": phase,
        "order": order,
        "model_version": "recorded",
        "prompt_version": 3,
        "rubric_sha": "fixturesha",
        "prompt_sha": "fixturesha",
        "ratings_0": ratings_0,
        "ratings_1": ratings_1,
    }


def _write_cache(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def test_voted_derive_keeps_majority_pair_with_agreeing_cache_mean(tmp_path):
    labels0 = tmp_path / "labels0.jsonl"
    labels1 = tmp_path / "labels1.jsonl"
    labels2 = tmp_path / "labels2.jsonl"

    # pair-000000: caches 0 and 1 agree fable 0 is preferred, cache 2 dissents.
    r0_0 = (
        {"moral": 5, "adherence": 4, "coherence": 4, "prose": 4},
        {"moral": 2, "adherence": 3, "coherence": 3, "prose": 3},
    )
    r0_1 = (
        {"moral": 4, "adherence": 5, "coherence": 4, "prose": 4},
        {"moral": 1, "adherence": 2, "coherence": 3, "prose": 3},
    )
    r0_2 = (
        {"moral": 1, "adherence": 1, "coherence": 1, "prose": 1},
        {"moral": 5, "adherence": 5, "coherence": 5, "prose": 5},
    )

    # pair-000001: no majority -- one vote each way plus a per-cache tie.
    r1_0 = (HI, LO)
    r1_1 = (LO, HI)
    r1_2 = TIE

    _write_cache(labels0, [_label_rec("pair-000000", *r0_0), _label_rec("pair-000001", *r1_0)])
    _write_cache(labels1, [_label_rec("pair-000000", *r0_1), _label_rec("pair-000001", *r1_1)])
    _write_cache(labels2, [_label_rec("pair-000000", *r0_2), _label_rec("pair-000001", *r1_2)])

    out = tmp_path / "derive"
    derive_stage.run(
        DeriveConfig(labels=str(labels0), pairs=PAIRS, extra_labels=[str(labels1), str(labels2)]),
        out,
    )

    prefs = [json.loads(line) for line in (out / "preferences.jsonl").read_text().splitlines()]
    assert len(prefs) == 1
    row = prefs[0]
    assert row["pair_id"] == "pair-000000"
    assert row["chosen"] == "fable zero A"
    assert row["rejected"] == "fable zero B"

    expected_chosen = {
        "moral": round((5 + 4) / 2, 4),
        "adherence": round((4 + 5) / 2, 4),
        "coherence": round((4 + 4) / 2, 4),
        "prose": round((4 + 4) / 2, 4),
    }
    expected_rejected = {
        "moral": round((2 + 1) / 2, 4),
        "adherence": round((3 + 2) / 2, 4),
        "coherence": round((3 + 3) / 2, 4),
        "prose": round((3 + 3) / 2, 4),
    }
    assert row["ratings_chosen"] == expected_chosen
    assert row["ratings_rejected"] == expected_rejected
    assert row["aggregate_chosen"] == round(aggregate_score(expected_chosen, WEIGHTS), 4)
    assert row["aggregate_rejected"] == round(aggregate_score(expected_rejected, WEIGHTS), 4)

    summary = json.loads((out / "derive_summary.json").read_text())
    assert summary["n_pairs_labeled"] == 2
    assert summary["n_preferences"] == 1
    assert summary["n_ties_skipped"] == 0
    assert summary["n_no_majority"] == 1
    assert summary["n_train"] + summary["n_held_out"] == 1

    sens = json.loads((out / "sensitivity.json").read_text())
    assert sens["n_base_preferences"] == 1  # only pair-000000 has a base voted preference


def test_voted_derive_split_matches_single_cache_split_for_same_pair_id(tmp_path):
    baseline_out = tmp_path / "baseline"
    derive_stage.run(DeriveConfig(labels=LABELS, pairs=PAIRS), baseline_out)
    baseline_prefs = [json.loads(line) for line in (baseline_out / "preferences.jsonl").read_text().splitlines()]
    baseline_split = next(p["split"] for p in baseline_prefs if p["pair_id"] == "pair-000000")

    labels0 = tmp_path / "v0.jsonl"
    labels1 = tmp_path / "v1.jsonl"
    labels2 = tmp_path / "v2.jsonl"
    rec = (
        {"moral": 5, "adherence": 4, "coherence": 4, "prose": 4},
        {"moral": 2, "adherence": 3, "coherence": 3, "prose": 3},
    )
    for path in (labels0, labels1, labels2):
        _write_cache(path, [_label_rec("pair-000000", *rec)])

    vote_out = tmp_path / "vote"
    derive_stage.run(
        DeriveConfig(labels=str(labels0), pairs=PAIRS, extra_labels=[str(labels1), str(labels2)]),
        vote_out,
    )
    vote_prefs = [json.loads(line) for line in (vote_out / "preferences.jsonl").read_text().splitlines()]
    assert vote_prefs[0]["pair_id"] == "pair-000000"
    assert vote_prefs[0]["split"] == baseline_split


def test_voted_derive_rejects_mismatched_pair_id_sets_across_caches(tmp_path):
    labels0 = tmp_path / "v0.jsonl"
    labels1 = tmp_path / "v1.jsonl"
    rec = (
        {"moral": 5, "adherence": 4, "coherence": 4, "prose": 4},
        {"moral": 2, "adherence": 3, "coherence": 3, "prose": 3},
    )
    _write_cache(labels0, [_label_rec("pair-000000", *rec), _label_rec("pair-000001", *rec)])
    _write_cache(labels1, [_label_rec("pair-000000", *rec)])  # missing pair-000001

    out = tmp_path / "derive"
    with pytest.raises(ValueError, match="disagree"):
        derive_stage.run(
            DeriveConfig(labels=str(labels0), pairs=PAIRS, extra_labels=[str(labels1)]),
            out,
        )


def test_voted_derive_manifest_lists_extra_cache_inputs(tmp_path):
    labels0 = tmp_path / "v0.jsonl"
    labels1 = tmp_path / "v1.jsonl"
    labels2 = tmp_path / "v2.jsonl"
    rec = (
        {"moral": 5, "adherence": 4, "coherence": 4, "prose": 4},
        {"moral": 2, "adherence": 3, "coherence": 3, "prose": 3},
    )
    for path in (labels0, labels1, labels2):
        _write_cache(path, [_label_rec("pair-000000", *rec)])

    out = tmp_path / "derive"
    derive_stage.run(
        DeriveConfig(labels=str(labels0), pairs=PAIRS, extra_labels=[str(labels1), str(labels2)]),
        out,
    )
    manifest = json.loads((out / "manifest.json").read_text())
    assert set(manifest["inputs"]) == {
        "labels.jsonl",
        "pairs.jsonl",
        "labels_extra_0.jsonl",
        "labels_extra_1.jsonl",
    }
