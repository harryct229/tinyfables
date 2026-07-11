import json
from dataclasses import replace
from pathlib import Path

import pytest

from tinyfables.config import EvalConfig, FiguresConfig, RewardTrainConfig, SourceSpec
from tinyfables.figure_data import assemble_figure_data
from tinyfables.stage import sha256_file, write_manifest

REPO = Path(__file__).resolve().parents[1]
FIX = Path(__file__).parent / "fixtures"


def _write_eval_run(tmp_path, name, fixture_name, model_bytes, tokenizer_path, bank_path):
    checkpoint = tmp_path / f"model_{name}"
    checkpoint.mkdir()
    model_path = checkpoint / "model.safetensors"
    model_path.write_bytes(model_bytes)

    out = tmp_path / name
    out.mkdir()
    metrics = json.loads((FIX / fixture_name).read_text())
    metrics["provenance"]["checkpoint_sha"] = sha256_file(model_path)
    metrics["provenance"]["tokenizer_sha"] = sha256_file(tokenizer_path)
    metrics_path = out / "eval_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    cfg = EvalConfig(
        checkpoint=str(checkpoint),
        tokenizer_dir=str(tokenizer_path.parent),
        source=SourceSpec(jsonl_path="shared-validation.jsonl", max_rows=2000),
        paraphrase_bank=str(bank_path),
        n_ctx=1024,
        n_perplexity_rows=500,
        n_generations=50,
        max_new_tokens=320,
        min_new_tokens=80,
        temperature=0.9,
        top_k=50,
        moral_threshold=0.3,
        seed=0,
        device="cpu",
    )
    write_manifest(
        out,
        "evaluate",
        cfg,
        [metrics_path],
        inputs={
            "model.safetensors": model_path,
            "tokenizer.json": tokenizer_path,
            "paraphrases.yaml": bank_path,
        },
    )
    return out


def _write_reward_run(tmp_path, tokenizer_path):
    out = tmp_path / "reward"
    out.mkdir()
    curve_path = out / "data_curve.json"
    curve_path.write_text((FIX / "issue10_data_curve.json").read_text())
    base_dir = tmp_path / "reward_base_model"
    base_dir.mkdir()
    base_model = base_dir / "model.safetensors"
    base_model.write_bytes(b"base")
    prefs = tmp_path / "preferences.jsonl"
    prefs.write_text('{"pair_id": "p0"}\n')
    cfg = RewardTrainConfig(
        preferences=str(prefs),
        base_checkpoint=str(base_dir),
        tokenizer_dir=str(tokenizer_path.parent),
        n_ctx=1024,
        batch_size=16,
        steps=1000,
        curve_steps=400,
        lr=1e-5,
        weight_decay=0.0,
        warmup_steps=50,
        grad_clip=1.0,
        seed=0,
        device="cpu",
        amp=False,
        log_every=20,
        curve_sizes=[100, 500, 1000, 2000],
        accuracy_gate=0.65,
    )
    write_manifest(
        out,
        "reward",
        cfg,
        [curve_path],
        inputs={
            "preferences.jsonl": prefs,
            "base_model.safetensors": base_model,
            "tokenizer.json": tokenizer_path,
        },
    )
    return out


def make_inputs(tmp_path):
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    tokenizer_path.write_text('{"version": "fixture"}\n')
    bank_path = tmp_path / "paraphrases.yaml"
    bank_path.write_text("version: 1\ntemplates: []\n")
    aug = _write_eval_run(
        tmp_path,
        "eval_aug",
        "issue10_eval_aug.json",
        b"aug-model",
        tokenizer_path,
        bank_path,
    )
    noaug = _write_eval_run(
        tmp_path,
        "eval_noaug",
        "issue10_eval_noaug.json",
        b"noaug-model",
        tokenizer_path,
        bank_path,
    )
    reward = _write_reward_run(tmp_path, tokenizer_path)
    cfg = FiguresConfig(
        augmented_eval_dir=str(aug),
        noaug_eval_dir=str(noaug),
        reward_dir=str(reward),
        augmented_run_id="issue10-eval-with-aug-v1",
        noaug_run_id="issue10-eval-noaug-v1",
        reward_run_id="issue07-reward-base-v1",
        augmented_prep_config=str(REPO / "configs" / "prep_full.yaml"),
        noaug_prep_config=str(REPO / "configs" / "prep_noaug_full.yaml"),
        augmented_pretrain_config=str(REPO / "configs" / "pretrain_full.yaml"),
        noaug_pretrain_config=str(REPO / "configs" / "pretrain_noaug_full.yaml"),
    )
    return cfg, aug, noaug, reward


def _rewrite_eval_metrics(run_dir, mutate):
    metrics_path = run_dir / "eval_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    mutate(metrics)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["eval_metrics.json"] = sha256_file(metrics_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _set_all_eval_counts(metrics, value):
    for cell in metrics["adherence_grid"].values():
        cell["n"] = value


def _config_copy_with_replacement(tmp_path, source_name, old, new):
    source = REPO / "configs" / source_name
    destination = tmp_path / source_name
    original = source.read_text()
    changed = original.replace(old, new)
    assert changed != original
    destination.write_text(changed)
    return destination


def test_assemble_complete_tables_and_provenance(tmp_path):
    cfg, aug, noaug, reward = make_inputs(tmp_path)
    data = assemble_figure_data(cfg)

    assert len(data.robustness_rows) == 6
    assert {(row["model"], row["family"]) for row in data.robustness_rows} == {
        (model, family)
        for model in ("with-aug", "no-aug")
        for family in ("canonical", "seen-template", "held-out-template")
    }
    held = {
        row["model"]: row
        for row in data.robustness_rows
        if row["family"] == "held-out-template"
    }
    assert held["with-aug"]["adherence"] == 0.4
    assert held["no-aug"]["adherence"] == 0.3
    assert held["with-aug"]["moral_delivery"] == 0.5
    assert held["no-aug"]["moral_delivery"] == 0.35
    assert "held-out templates" in data.held_out_statement
    assert "+10.0 percentage points" in data.held_out_statement
    assert "+15.0 percentage points" in data.held_out_statement

    assert [row["requested_train_size"] for row in data.rm_rows] == [100, 500, 1000, 2000]
    assert data.rm_rows[-1]["train_size"] == 1761
    assert data.rm_rows[-1]["accuracy_gate"] == 0.65
    robust_prov = data.provenance["figures"]["robustness_grid"]
    assert robust_prov["run_ids"] == [
        "issue10-eval-with-aug-v1",
        "issue10-eval-noaug-v1",
    ]
    assert len(robust_prov["eval_manifest_sha256"]) == 2
    assert robust_prov["checkpoint_sha256"][0] != robust_prov["checkpoint_sha256"][1]
    assert data.provenance["ablation_controls"]["prep_only_difference"] == {
        "paraphrase_coverage": [0.15, 0.0]
    }
    assert set(data.inputs) == {
        "aug_eval_metrics.json",
        "aug_eval_manifest.json",
        "noaug_eval_metrics.json",
        "noaug_eval_manifest.json",
        "data_curve.json",
        "reward_manifest.json",
        "prep_full.yaml",
        "prep_noaug_full.yaml",
        "pretrain_full.yaml",
        "pretrain_noaug_full.yaml",
    }


def test_rejects_eval_difference_beyond_checkpoint(tmp_path):
    cfg, _, noaug, _ = make_inputs(tmp_path)
    manifest_path = noaug / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config"]["temperature"] = 0.8
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="eval configs must differ only in checkpoint"):
        assemble_figure_data(cfg)


def test_rejects_tampered_upstream_artifact(tmp_path):
    cfg, _, _, reward = make_inputs(tmp_path)
    with open(reward / "data_curve.json", "a") as f:
        f.write(" ")

    with pytest.raises(ValueError, match="data_curve.json hash mismatch"):
        assemble_figure_data(cfg)


def test_rejects_uniformly_underfilled_eval_cells(tmp_path):
    cfg, aug, noaug, _ = make_inputs(tmp_path)
    for run_dir in (aug, noaug):
        _rewrite_eval_metrics(
            run_dir, lambda metrics: _set_all_eval_counts(metrics, 49)
        )

    with pytest.raises(ValueError, match="must equal requested n_generations=50"):
        assemble_figure_data(cfg)


@pytest.mark.parametrize("bad_count", [49.5, True])
def test_rejects_non_integer_eval_cell_counts(tmp_path, bad_count):
    cfg, aug, noaug, _ = make_inputs(tmp_path)
    for run_dir in (aug, noaug):
        _rewrite_eval_metrics(
            run_dir, lambda metrics: _set_all_eval_counts(metrics, bad_count)
        )

    with pytest.raises(ValueError, match="n must be an exact integer"):
        assemble_figure_data(cfg)


def test_rejects_pretrain_configs_that_share_one_prep_dir(tmp_path):
    cfg, _, _, _ = make_inputs(tmp_path)
    contaminated = _config_copy_with_replacement(
        tmp_path,
        "pretrain_noaug_full.yaml",
        "prep_dir: runs/prep_noaug",
        "prep_dir: runs/prep_full",
    )
    cfg = replace(cfg, noaug_pretrain_config=str(contaminated))

    with pytest.raises(ValueError, match="pretrain prep dirs must differ"):
        assemble_figure_data(cfg)


def test_rejects_different_tokenizer_between_prep_and_pretrain(tmp_path):
    cfg, _, _, _ = make_inputs(tmp_path)
    pretrain_aug = _config_copy_with_replacement(
        tmp_path,
        "pretrain_full.yaml",
        "tokenizer_dir: runs/tokenizer_full",
        "tokenizer_dir: runs/other_tokenizer",
    )
    pretrain_noaug = _config_copy_with_replacement(
        tmp_path,
        "pretrain_noaug_full.yaml",
        "tokenizer_dir: runs/tokenizer_full",
        "tokenizer_dir: runs/other_tokenizer",
    )
    cfg = replace(
        cfg,
        augmented_pretrain_config=str(pretrain_aug),
        noaug_pretrain_config=str(pretrain_noaug),
    )

    with pytest.raises(
        ValueError, match="all prep and pretrain configs must reuse one tokenizer"
    ):
        assemble_figure_data(cfg)


@pytest.mark.parametrize(
    ("aug_adherence", "aug_moral", "expected"),
    [
        (0.2, 0.3, "augmentation didn't help"),
        (0.4, 0.3, "augmentation had a mixed measured result"),
    ],
)
def test_reports_non_positive_held_out_outcomes_honestly(
    tmp_path, aug_adherence, aug_moral, expected
):
    cfg, aug, _, _ = make_inputs(tmp_path)

    def change_held_out(metrics):
        held_out = metrics["adherence_grid"]["held-out-template"]
        held_out["overall"] = aug_adherence
        held_out["moral_delivery"] = aug_moral

    _rewrite_eval_metrics(aug, change_held_out)

    assert expected in assemble_figure_data(cfg).held_out_statement
