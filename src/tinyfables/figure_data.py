"""Validated, renderer-independent data for Issue 10 report figures.

This module is deliberately torch- and matplotlib-free. It verifies upstream
artifact hashes, the one-variable prep contract, the scientific pretrain
control, and checkpoint-only eval differences before exposing any plot rows.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from tinyfables.config import FiguresConfig, PrepConfig, PretrainConfig, load_config
from tinyfables.stage import sha256_file

FAMILY_ORDER = ("canonical", "seen-template", "held-out-template")
MODEL_ORDER = ("with-aug", "no-aug")
_PRETRAIN_OPERATIONAL_FIELDS = {"prep_dir", "ckpt_dir", "run_name", "ckpt_hub_repo"}
_ADHERENCE_FIELDS = ("character", "setting", "challenge", "outcome")


@dataclass(frozen=True)
class FigureData:
    robustness_rows: list[dict]
    rm_rows: list[dict]
    provenance: dict
    held_out_statement: str
    inputs: dict[str, Path]


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _manifest(run_dir: Path, expected_stage: str) -> tuple[dict, Path, str]:
    path = run_dir / "manifest.json"
    manifest = _read_json(path)
    if manifest.get("stage") != expected_stage:
        raise ValueError(
            f"{path} stage must be {expected_stage!r}, got {manifest.get('stage')!r}"
        )
    return manifest, path, sha256_file(path)


def _verify_artifact(run_dir: Path, manifest: dict, name: str) -> Path:
    path = run_dir / name
    expected = manifest.get("artifacts", {}).get(name)
    if expected is None:
        raise ValueError(f"{run_dir / 'manifest.json'} does not record {name}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{name} hash mismatch: manifest={expected}, actual={actual}")
    return path


def _diffs(left: dict, right: dict) -> dict[str, list]:
    return {
        key: [left.get(key), right.get(key)]
        for key in sorted(set(left) | set(right))
        if left.get(key) != right.get(key)
    }


def _control_provenance(cfg: FiguresConfig) -> tuple[dict, dict[str, Path]]:
    prep_aug_path = Path(cfg.augmented_prep_config)
    prep_noaug_path = Path(cfg.noaug_prep_config)
    pre_aug_path = Path(cfg.augmented_pretrain_config)
    pre_noaug_path = Path(cfg.noaug_pretrain_config)
    prep_aug = asdict(load_config(prep_aug_path, PrepConfig))
    prep_noaug = asdict(load_config(prep_noaug_path, PrepConfig))
    prep_diffs = _diffs(prep_aug, prep_noaug)
    expected_prep_diff = {"paraphrase_coverage": [0.15, 0.0]}
    if prep_diffs != expected_prep_diff:
        raise ValueError(
            f"prep configs must differ only 0.15 -> 0.0 coverage; got {prep_diffs}"
        )

    pre_aug = asdict(load_config(pre_aug_path, PretrainConfig))
    pre_noaug = asdict(load_config(pre_noaug_path, PretrainConfig))
    pre_diffs = _diffs(pre_aug, pre_noaug)
    scientific_diffs = {
        key: value
        for key, value in pre_diffs.items()
        if key not in _PRETRAIN_OPERATIONAL_FIELDS
    }
    if scientific_diffs:
        raise ValueError(f"pretrain scientific configs differ: {scientific_diffs}")
    if pre_aug["prep_dir"] == pre_noaug["prep_dir"]:
        raise ValueError("augmented and no-aug pretrain prep dirs must differ")
    tokenizer_dirs = {
        prep_aug["tokenizer_dir"],
        prep_noaug["tokenizer_dir"],
        pre_aug["tokenizer_dir"],
        pre_noaug["tokenizer_dir"],
    }
    if len(tokenizer_dirs) != 1:
        raise ValueError("all prep and pretrain configs must reuse one tokenizer")

    provenance = {
        "prep_only_difference": expected_prep_diff,
        "pretrain_scientific_fields_equal": True,
        "pretrain_operational_differences": pre_diffs,
        "config_sha256": {
            "prep_full.yaml": sha256_file(prep_aug_path),
            "prep_noaug_full.yaml": sha256_file(prep_noaug_path),
            "pretrain_full.yaml": sha256_file(pre_aug_path),
            "pretrain_noaug_full.yaml": sha256_file(pre_noaug_path),
        },
    }
    inputs = {
        "prep_full.yaml": prep_aug_path,
        "prep_noaug_full.yaml": prep_noaug_path,
        "pretrain_full.yaml": pre_aug_path,
        "pretrain_noaug_full.yaml": pre_noaug_path,
    }
    return provenance, inputs


def _load_eval(run_dir: Path) -> tuple[dict, dict, Path, Path, str]:
    manifest, manifest_path, manifest_sha = _manifest(run_dir, "evaluate")
    metrics_path = _verify_artifact(run_dir, manifest, "eval_metrics.json")
    metrics = _read_json(metrics_path)
    grid = metrics.get("adherence_grid", {})
    if set(grid) != set(FAMILY_ORDER):
        raise ValueError(f"eval grid must contain exactly {FAMILY_ORDER}; got {sorted(grid)}")
    inputs = manifest.get("inputs", {})
    for name in ("model.safetensors", "tokenizer.json", "paraphrases.yaml"):
        if name not in inputs:
            raise ValueError(f"eval manifest is missing input hash {name}")
    metrics_prov = metrics.get("provenance", {})
    if metrics_prov.get("checkpoint_sha") != inputs["model.safetensors"]:
        raise ValueError("eval checkpoint hash disagrees with its manifest")
    if metrics_prov.get("tokenizer_sha") != inputs["tokenizer.json"]:
        raise ValueError("eval tokenizer hash disagrees with its manifest")
    if metrics_prov.get("seed") != manifest.get("config", {}).get("seed"):
        raise ValueError("eval seed disagrees with its manifest")
    return metrics, manifest, metrics_path, manifest_path, manifest_sha


def _rate(value, label: str) -> float:
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be in [0, 1], got {result}")
    return result


def _robustness_rows(
    cfg: FiguresConfig,
    aug_metrics: dict,
    aug_manifest: dict,
    aug_manifest_sha: str,
    noaug_metrics: dict,
    noaug_manifest: dict,
    noaug_manifest_sha: str,
) -> list[dict]:
    aug_cfg = aug_manifest.get("config", {})
    noaug_cfg = noaug_manifest.get("config", {})
    eval_diffs = _diffs(aug_cfg, noaug_cfg)
    if set(eval_diffs) != {"checkpoint"}:
        raise ValueError(f"eval configs must differ only in checkpoint; got {eval_diffs}")
    aug_inputs = aug_manifest["inputs"]
    noaug_inputs = noaug_manifest["inputs"]
    if aug_inputs["model.safetensors"] == noaug_inputs["model.safetensors"]:
        raise ValueError("augmented and no-aug evals used the same checkpoint hash")
    if aug_inputs["tokenizer.json"] != noaug_inputs["tokenizer.json"]:
        raise ValueError("augmented and no-aug evals used different tokenizers")
    if aug_inputs["paraphrases.yaml"] != noaug_inputs["paraphrases.yaml"]:
        raise ValueError("augmented and no-aug evals used different template banks")

    rows = []
    runs = (
        (
            "with-aug",
            0.15,
            cfg.augmented_run_id,
            aug_metrics,
            aug_inputs,
            aug_manifest_sha,
            aug_cfg,
        ),
        (
            "no-aug",
            0.0,
            cfg.noaug_run_id,
            noaug_metrics,
            noaug_inputs,
            noaug_manifest_sha,
            noaug_cfg,
        ),
    )
    for model, coverage, run_id, metrics, inputs, manifest_sha, run_cfg in runs:
        requested_n = run_cfg.get("n_generations")
        if type(requested_n) is not int or requested_n <= 0:
            raise ValueError(
                f"{model} manifest n_generations must be a positive exact integer"
            )
        perplexity = float(metrics["perplexity"]["perplexity"])
        for family in FAMILY_ORDER:
            source = metrics["adherence_grid"][family]
            cell_n = source.get("n")
            if type(cell_n) is not int:
                raise ValueError(
                    f"{model}/{family} n must be an exact integer, got {cell_n!r}"
                )
            if cell_n != requested_n:
                raise ValueError(
                    f"{model}/{family} n must equal requested "
                    f"n_generations={requested_n}, got {cell_n}"
                )
            row = {
                "model": model,
                "paraphrase_coverage": coverage,
                "run_id": run_id,
                "family": family,
                **{
                    field: _rate(source[field], f"{model}/{family}/{field}")
                    for field in _ADHERENCE_FIELDS
                },
                "adherence": _rate(source["overall"], f"{model}/{family}/overall"),
                "moral_delivery": _rate(
                    source["moral_delivery"], f"{model}/{family}/moral_delivery"
                ),
                "n": cell_n,
                "perplexity": perplexity,
                "checkpoint_sha256": inputs["model.safetensors"],
                "eval_manifest_sha256": manifest_sha,
            }
            rows.append(row)
    if len({row["n"] for row in rows}) != 1:
        raise ValueError("all robustness cells must use the same number of FableSpecs")
    return rows


def _rm_rows(cfg: FiguresConfig, reward_dir: Path) -> tuple[list[dict], dict, Path, Path, str]:
    manifest, manifest_path, manifest_sha = _manifest(reward_dir, "reward")
    curve_path = _verify_artifact(reward_dir, manifest, "data_curve.json")
    curve = _read_json(curve_path)
    points = curve.get("points")
    if not isinstance(points, list):
        raise ValueError("data_curve.json points must be a list")
    requested = [int(point["requested_train_size"]) for point in points]
    if requested != [100, 500, 1000, 2000]:
        raise ValueError(f"RM curve must contain requested sizes 100/500/1000/2000; got {requested}")
    gate = _rate(manifest.get("config", {}).get("accuracy_gate"), "accuracy_gate")
    rows = []
    for point in points:
        rows.append(
            {
                "run_id": cfg.reward_run_id,
                "requested_train_size": int(point["requested_train_size"]),
                "train_size": int(point["train_size"]),
                "held_out_accuracy": _rate(
                    point["held_out_accuracy"], "held_out_accuracy"
                ),
                "final_loss": float(point["final_loss"]),
                "steps": int(point["steps"]),
                "accuracy_gate": gate,
                "reward_manifest_sha256": manifest_sha,
            }
        )
    return rows, manifest, curve_path, manifest_path, manifest_sha


def _held_out_statement(rows: list[dict]) -> str:
    held = {row["model"]: row for row in rows if row["family"] == "held-out-template"}
    adherence_pp = 100.0 * (held["with-aug"]["adherence"] - held["no-aug"]["adherence"])
    moral_pp = 100.0 * (
        held["with-aug"]["moral_delivery"] - held["no-aug"]["moral_delivery"]
    )
    deltas = (
        f"{adherence_pp:+.1f} percentage points in adherence and "
        f"{moral_pp:+.1f} percentage points in Moral delivery"
    )
    if adherence_pp > 0.0 and moral_pp > 0.0:
        return (
            f"On held-out templates, the measured augmentation effect was {deltas}; "
            "this is the unseen-phrasing result, without a significance claim."
        )
    if adherence_pp <= 0.0 and moral_pp <= 0.0:
        return (
            f"On held-out templates, augmentation didn't help: the measured effect was {deltas}; "
            "the negative result is published without cherry-picking."
        )
    return (
        f"On held-out templates, augmentation had a mixed measured result: {deltas}; "
        "both directions are published without cherry-picking."
    )


def assemble_figure_data(cfg: FiguresConfig) -> FigureData:
    controls, control_inputs = _control_provenance(cfg)
    aug_dir = Path(cfg.augmented_eval_dir)
    noaug_dir = Path(cfg.noaug_eval_dir)
    reward_dir = Path(cfg.reward_dir)
    aug_metrics, aug_manifest, aug_metrics_path, aug_manifest_path, aug_manifest_sha = (
        _load_eval(aug_dir)
    )
    noaug_metrics, noaug_manifest, noaug_metrics_path, noaug_manifest_path, noaug_manifest_sha = (
        _load_eval(noaug_dir)
    )
    robustness = _robustness_rows(
        cfg,
        aug_metrics,
        aug_manifest,
        aug_manifest_sha,
        noaug_metrics,
        noaug_manifest,
        noaug_manifest_sha,
    )
    rm_rows, reward_manifest, curve_path, reward_manifest_path, reward_manifest_sha = _rm_rows(
        cfg, reward_dir
    )
    statement = _held_out_statement(robustness)
    provenance = {
        "schema_version": 1,
        "ablation_controls": controls,
        "figures": {
            "robustness_grid": {
                "run_ids": [cfg.augmented_run_id, cfg.noaug_run_id],
                "eval_manifest_sha256": [aug_manifest_sha, noaug_manifest_sha],
                "checkpoint_sha256": [
                    aug_manifest["inputs"]["model.safetensors"],
                    noaug_manifest["inputs"]["model.safetensors"],
                ],
                "tokenizer_sha256": aug_manifest["inputs"]["tokenizer.json"],
                "paraphrase_bank_sha256": aug_manifest["inputs"]["paraphrases.yaml"],
                "held_out_column": "held-out-template (five eval-only templates)",
            },
            "rm_data_curve": {
                "run_id": cfg.reward_run_id,
                "reward_manifest_sha256": reward_manifest_sha,
                "data_curve_sha256": reward_manifest["artifacts"]["data_curve.json"],
                "base_checkpoint_sha256": reward_manifest.get("inputs", {}).get(
                    "base_model.safetensors"
                ),
                "preferences_sha256": reward_manifest.get("inputs", {}).get(
                    "preferences.jsonl"
                ),
            },
        },
        "held_out_interpretation": statement,
    }
    inputs = {
        "aug_eval_metrics.json": aug_metrics_path,
        "aug_eval_manifest.json": aug_manifest_path,
        "noaug_eval_metrics.json": noaug_metrics_path,
        "noaug_eval_manifest.json": noaug_manifest_path,
        "data_curve.json": curve_path,
        "reward_manifest.json": reward_manifest_path,
        **control_inputs,
    }
    return FigureData(robustness, rm_rows, provenance, statement, inputs)
