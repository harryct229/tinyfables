import json
from pathlib import Path

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


def test_derive_uses_latest_canonical_label_version_once(tmp_path):
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
    derive_stage.run(DeriveConfig(labels=str(labels_path), pairs=PAIRS), out)

    prefs = [json.loads(line) for line in (out / "preferences.jsonl").read_text().splitlines()]
    pair_rows = [row for row in prefs if row["pair_id"] == "pair-000000"]
    assert len(pair_rows) == 1
    row = pair_rows[0]
    assert row["chosen"] == "fable zero B"
    assert row["rejected"] == "fable zero A"
    assert row["aggregate_chosen"] == round(aggregate_score(duplicate["ratings_1"], WEIGHTS), 4)
    assert row["aggregate_rejected"] == round(aggregate_score(duplicate["ratings_0"], WEIGHTS), 4)

    summary = json.loads((out / "derive_summary.json").read_text())
    assert summary["n_pairs_labeled"] == 6
    assert summary["n_preferences"] == 5
    assert summary["n_ties_skipped"] == 1

    sens = json.loads((out / "sensitivity.json").read_text())
    assert sens["n_base_preferences"] == 5
