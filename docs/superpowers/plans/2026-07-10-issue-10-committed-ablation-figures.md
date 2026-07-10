# Issue 10 — Committed Ablation + Report Figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate the controlled no-augmentation Base Model sibling, then publish the complete 2-model × 3-family robustness grid and the Issue 07 reward-model data curve as report-ready, provenance-bearing artifacts.

**Architecture:** Track A is local, offline-tested work on an isolated `issue-10-committed-ablation-figures` feature branch: exact config siblings, stronger evaluate provenance, torch-free figure-data validation, a thin matplotlib stage, and toy/fixture tests. Track B is main-session-only Colab work: pull immutable inputs from the Hub, run `prep_noaug` and resumable pretraining, evaluate both checkpoints under configs differing only in `checkpoint`, assemble both figures, and push every artifact before the runtime ends. The figure stage refuses contaminated ablations or mismatched eval configs and records upstream run IDs plus manifest hashes in both tables and PNG footers.

**Tech Stack:** Python 3.10+, frozen dataclass/YAML configs, numpy, torch, transformers, tokenizers, matplotlib from the existing `.[train]` extra, Hugging Face Hub, Google Colab T4, pytest. Local commands use `.venv/bin/...`; local tests are offline.

## Global Constraints

- All compute runs happen on **Colab via colab-mcp**. Colab inputs are pulled from the Hub; every stage artifact is pushed to the Hub before the session ends. An unpushed Colab run did not happen.
- The no-augmentation treatment is exactly `paraphrase_coverage: 0.0`; the augmented control remains exactly `0.15`. `configs/prep_noaug_full.yaml` must otherwise equal `configs/prep_full.yaml` byte-for-field after YAML parsing.
- Reuse `congthanh991/tinyfables-tokenizer` at `runs/tokenizer_full`; do not run the tokenizer stage for the sibling. Coverage does not affect tokenizer training because the tokenizer consumes raw rows.
- The sampled corpus stays `klusai/ds-tf1-en-3m`, split `train`, `max_rows: 450000`, `seed: 0`. The no-aug prep must report 450,000 canonical rows, zero seen-template rows, and zero held-out-template rows.
- Pretraining scientific fields stay identical to the real Base Model: 6 layers, d_model 384, 6 heads, context 1024, batch 16, 20,000 steps, warmup 200, lr 0.0003, weight decay 0.1, grad clip 1.0, seed 0, fp16 AMP. Only artifact/logging paths and run identifiers may differ.
- Checkpoint-resume reuses Issue 04 verbatim: Drive-backed `ckpt_dir`, `ckpt_every: 500`, `keep_last_k: 2`, auto-resume by rerunning the same command, and Hub checkpoint mirroring. Do not restart from random weights after a session interruption.
- Evaluate both models with configs identical after deleting only `checkpoint`: same validation source, tokenizer, paraphrase bank, 500 perplexity rows, 50 FableSpecs, generation settings, moral threshold, and seed. The five `held_out: true` templates are the held-out column; they remain eval-only.
- Publish every family and both metrics regardless of direction. There is no winner filtering. If augmentation does not improve the held-out measurements, the generated statement must literally report that augmentation did not help.
- Each figure exports one PNG and one long-form CSV. `figure_provenance.json` records source run IDs, upstream manifest SHA-256 values, checkpoint/tokenizer/template-bank hashes, the controlled-config checks, and the Issue 07 reward manifest hash.
- Perplexity is context, not the ablation headline: show both model perplexities in the robustness PNG footer and CSV, but interpret the held-out adherence and Moral-delivery cells as the augmentation result.
- The Issue 07 curve is already mirrored at Hub model revision `58865e69ccbe1e90b3816068cabca410f28e43aa`: `data_curve.json` has manifest-recorded SHA-256 `721e321ecde5758702210d3e3de5ca1c9cf6fdc62d826f47360426455da9a6c0`. Pull that immutable revision; Issue 08's not-yet-created preferences mirror is not a dependency.
- Hub destinations: final sibling model and the Issue 10 artifact bundle live in `congthanh991/tinyfables-13m-base-noaug`; resumable checkpoints live in private `congthanh991/tinyfables-13m-base-noaug-ckpts`. Bundle paths are `prep_noaug/`, `eval_with_aug/`, `eval_noaug/`, and `figures/`.
- Local development uses `.venv/bin/pytest`; shell state does not persist, so every command names the executable explicitly. Tests do not contact datasets, Claude, Colab, or the Hub.
- Track A is subagent-driven on a feature branch. Hub credentials, colab-mcp, long runs, measured-result interpretation, Hub uploads, and final documentation remain main-session-only.
- Before any worker writes `src/tinyfables/stages/figures.py`, the main session must load the dataviz skill and pass its guidance to that worker. If the skill is still unavailable, pause that task and install/enable it or ask the maintainer; do not silently skip the requested skill gate.

---

## Scope Check

This remains one plan, not separate sub-projects. The configs establish the scientific control, the dual eval produces the only valid figure inputs, and the figure stage verifies those exact controls before rendering. Splitting them into independent plans would weaken the provenance chain that Issue 10 exists to provide.

## File Map

| File | Responsibility |
|---|---|
| `configs/prep_noaug_full.yaml` | Full 450k prep sibling; differs from `prep_full.yaml` only in coverage 0.15 → 0.0 |
| `configs/pretrain_noaug_full.yaml` | Full resumable sibling pretrain with identical scientific hyperparameters and distinct artifact paths |
| `configs/eval_noaug_full.yaml` | Eval sibling; differs from `eval_full.yaml` only in checkpoint path |
| `configs/figures_full.yaml` | Names the three upstream run directories, stable run IDs, and four control config files |
| `src/tinyfables/stages/evaluate.py` | Add the paraphrase-bank hash to every eval manifest |
| `src/tinyfables/config.py` | Add `FiguresConfig` |
| `src/tinyfables/figure_data.py` | Torch/matplotlib-free source validation, table assembly, and honest held-out interpretation |
| `src/tinyfables/stages/figures.py` | Write CSV/JSON/Markdown, render two PNGs, and write manifest last |
| `src/tinyfables/stages/__init__.py` | Register `figures` lazily |
| `src/tinyfables/stage.py` | Include matplotlib in version provenance when installed |
| `tests/fixtures/issue10_eval_aug.json` | Fixture augmented eval metrics |
| `tests/fixtures/issue10_eval_noaug.json` | Fixture no-aug eval metrics |
| `tests/fixtures/issue10_data_curve.json` | Fixture copy of the measured Issue 07 curve |
| `tests/test_full_configs.py` | Prove prep treatment-only, pretrain scientific equivalence, and eval checkpoint-only differences |
| `tests/test_evaluate_stage.py` | Prove eval manifests hash the paraphrase bank |
| `tests/test_figure_data.py` | Fixture-driven validation/table/provenance tests |
| `tests/test_figures_stage.py` | Offline figure-stage assembly with injected PNG renderers; optional real matplotlib smoke test |
| `tests/test_issue10_toy_chain.py` | One-tokenizer toy no-aug prep → pretrain → three-family eval chain |
| `docs/design.md` | Track A contract, measured 2×3 grid, held-out interpretation, curve and Hub provenance |
| `docs/issues/10-committed-ablation-figures.md` | Check the four acceptance criteria only after Hub round-trip verification |
| `docs/issues/README.md` | Tick Issue 10 only at final closeout |

## Main-Session Setup — Isolated Worktree and Skill Gate

These are orchestration steps, not subagent tasks.

- [ ] **Step S.1: Protect the dirty Issue 08 worktree**

The planning worktree currently contains unrelated Issue 08 changes. Load `superpowers:using-git-worktrees`, fetch the latest local `main`, and create an isolated worktree/branch named `issue-10-committed-ablation-figures`. Do not stash, reset, clean, or commit the Issue 08 files as part of Issue 10.

- [ ] **Step S.2: Select subagent-driven execution**

Load `superpowers:subagent-driven-development`. Dispatch one fresh implementation subagent per Track A task below, followed by spec-compliance review and code-quality review before moving to the next task.

- [ ] **Step S.3: Load dataviz before Task 4**

Load the dataviz skill before dispatching Task 4. The selected AntV
`chart-visualization` skill maps categorical comparisons to grouped
bar/column charts, so the visual contract is: two grouped-column panels
(adherence and Moral delivery), common 0–1 rate range, paired with-aug/no-aug
bars across canonical/seen/held-out, held-out category highlighted, exact
annotations, neutral titles, both perplexities in context, and no
outcome-dependent selection. This decision supersedes the earlier heatmap
sample later in this plan.

---

## Track A — Local Feature Branch, Subagent-Driven, Offline

### Task 1: Controlled Full-Run Config Siblings

**Files:**
- Create: `configs/prep_noaug_full.yaml`
- Create: `configs/pretrain_noaug_full.yaml`
- Create: `configs/eval_noaug_full.yaml`
- Modify: `tests/test_full_configs.py`

**Interfaces:**
- Consumes: existing `PrepConfig`, `PretrainConfig`, `EvalConfig`, `configs/{prep,pretrain,eval}_full.yaml`.
- Produces: committed configs used verbatim in Colab; invariants consumed by `figure_data.assemble_figure_data` in Task 3.

- [ ] **Step 1: Write the failing config-invariant test**

Append to `tests/test_full_configs.py`:

```python
def test_issue10_noaug_configs_preserve_the_controlled_experiment():
    from dataclasses import asdict

    from tinyfables.config import EvalConfig, PretrainConfig

    prep_aug = asdict(load_config(REPO / "configs" / "prep_full.yaml", PrepConfig))
    prep_noaug = asdict(load_config(REPO / "configs" / "prep_noaug_full.yaml", PrepConfig))
    prep_diffs = {
        key: (prep_aug[key], prep_noaug[key])
        for key in prep_aug
        if prep_aug[key] != prep_noaug[key]
    }
    assert prep_diffs == {"paraphrase_coverage": (0.15, 0.0)}
    assert prep_noaug["tokenizer_dir"] == "runs/tokenizer_full"
    assert prep_noaug["source"] == {
        "jsonl_path": None,
        "hf_dataset": "klusai/ds-tf1-en-3m",
        "hf_split": "train",
        "max_rows": 450000,
    }

    pre_aug = asdict(load_config(REPO / "configs" / "pretrain_full.yaml", PretrainConfig))
    pre_noaug = asdict(
        load_config(REPO / "configs" / "pretrain_noaug_full.yaml", PretrainConfig)
    )
    operational = {"prep_dir", "ckpt_dir", "run_name", "ckpt_hub_repo"}
    scientific_aug = {key: value for key, value in pre_aug.items() if key not in operational}
    scientific_noaug = {key: value for key, value in pre_noaug.items() if key not in operational}
    assert scientific_noaug == scientific_aug
    assert {
        key: (pre_aug[key], pre_noaug[key])
        for key in operational
        if pre_aug[key] != pre_noaug[key]
    } == {
        "prep_dir": ("runs/prep_full", "runs/prep_noaug"),
        "ckpt_dir": (
            "/content/drive/MyDrive/tinyfables/ckpt_base",
            "/content/drive/MyDrive/tinyfables/ckpt_base_noaug",
        ),
        "run_name": ("base-v1", "base-noaug-v1"),
        "ckpt_hub_repo": (
            None,
            "congthanh991/tinyfables-13m-base-noaug-ckpts",
        ),
    }

    eval_aug = asdict(load_config(REPO / "configs" / "eval_full.yaml", EvalConfig))
    eval_noaug = asdict(load_config(REPO / "configs" / "eval_noaug_full.yaml", EvalConfig))
    eval_diffs = {
        key: (eval_aug[key], eval_noaug[key])
        for key in eval_aug
        if eval_aug[key] != eval_noaug[key]
    }
    assert eval_diffs == {"checkpoint": ("runs/base_model", "runs/base_noaug")}
```

- [ ] **Step 2: Run the test to verify failure**

Run from the Issue 10 worktree:

```bash
rtk .venv/bin/pytest tests/test_full_configs.py::test_issue10_noaug_configs_preserve_the_controlled_experiment -v
```

Expected: FAIL with `FileNotFoundError` for `configs/prep_noaug_full.yaml`.

- [ ] **Step 3: Create the treatment-only prep config**

Create `configs/prep_noaug_full.yaml` exactly as:

```yaml
# Issue 10 controlled sibling: byte-for-field equal to prep_full.yaml after
# YAML parsing except paraphrase_coverage 0.15 -> 0.0. The bank remains named
# so the single treatment difference is explicit and machine-checkable.
source:
  hf_dataset: klusai/ds-tf1-en-3m
  hf_split: train
  max_rows: 450000
tokenizer_dir: runs/tokenizer_full
window: 1024
seed: 0
paraphrase_bank: configs/paraphrases.yaml
paraphrase_coverage: 0.0
```

- [ ] **Step 4: Create the resumable sibling pretrain config**

Create `configs/pretrain_noaug_full.yaml` exactly as:

```yaml
# Issue 10 no-augmentation Base Model sibling. Scientific hyperparameters are
# identical to pretrain_full.yaml; only prep/artifact/run identifiers differ.
prep_dir: runs/prep_noaug
tokenizer_dir: runs/tokenizer_full
n_layer: 6
n_head: 6
d_model: 384
n_ctx: 1024
batch_size: 16
steps: 20000
warmup_steps: 200
lr: 0.0003
weight_decay: 0.1
grad_clip: 1.0
device: auto
seed: 0
amp: true
ckpt_dir: /content/drive/MyDrive/tinyfables/ckpt_base_noaug
ckpt_every: 500
keep_last_k: 2
log_every: 50
trackio_project: tinyfables-base
run_name: base-noaug-v1
ckpt_hub_repo: congthanh991/tinyfables-13m-base-noaug-ckpts
```

- [ ] **Step 5: Create the checkpoint-only eval sibling**

Create `configs/eval_noaug_full.yaml` exactly as:

```yaml
# Issue 10 no-aug evaluation. Parsed config differs from eval_full.yaml only
# in checkpoint; all val rows, templates, seeds, and generation knobs match.
checkpoint: runs/base_noaug
tokenizer_dir: runs/tokenizer_full
source:
  hf_dataset: klusai/ds-tf1-en-3m
  hf_split: validation
  max_rows: 2000
paraphrase_bank: configs/paraphrases.yaml
n_ctx: 1024
n_perplexity_rows: 500
n_generations: 50
max_new_tokens: 320
min_new_tokens: 80
temperature: 0.9
top_k: 50
moral_threshold: 0.3
seed: 0
device: auto
```

- [ ] **Step 6: Run the focused and existing full-config tests**

```bash
rtk .venv/bin/pytest tests/test_full_configs.py -v
```

Expected: every test in `tests/test_full_configs.py` PASS.

- [ ] **Step 7: Commit**

```bash
rtk git add configs/prep_noaug_full.yaml configs/pretrain_noaug_full.yaml configs/eval_noaug_full.yaml tests/test_full_configs.py
rtk git commit -m "config(ablation): add controlled no-augmentation siblings"
```

### Task 2: Hash the Eval Template Bank in Run Provenance

**Files:**
- Modify: `src/tinyfables/stages/evaluate.py:155-163`
- Modify: `tests/test_evaluate_stage.py:39-58`

**Interfaces:**
- Consumes: `EvalConfig.paraphrase_bank`, `stage.write_manifest`.
- Produces: every Issue 10 eval manifest has `inputs["paraphrases.yaml"]`; Task 3 requires both evals to have the same bank hash.

- [ ] **Step 1: Strengthen the existing artifact test**

Add the import and assertion to `tests/test_evaluate_stage.py`:

```python
from tinyfables.stage import sha256_file
```

Inside `test_evaluate_writes_metrics_report_and_manifest`, immediately after `assert manifest["stage"] == "evaluate"`, add:

```python
    assert manifest["inputs"]["paraphrases.yaml"] == sha256_file(Path(BANK))
```

- [ ] **Step 2: Run the focused test to verify failure**

```bash
rtk .venv/bin/pytest tests/test_evaluate_stage.py::test_evaluate_writes_metrics_report_and_manifest -v
```

Expected: FAIL with `KeyError: 'paraphrases.yaml'`.

- [ ] **Step 3: Add the bank to manifest inputs**

Replace the final `write_manifest` block in `src/tinyfables/stages/evaluate.py` with:

```python
    inputs = {
        "model.safetensors": Path(cfg.checkpoint) / "model.safetensors",
        "tokenizer.json": Path(cfg.tokenizer_dir) / "tokenizer.json",
    }
    if cfg.paraphrase_bank:
        inputs["paraphrases.yaml"] = Path(cfg.paraphrase_bank)

    write_manifest(
        out_dir,
        "evaluate",
        cfg,
        [metrics_path, report_path, cal_path],
        inputs=inputs,
    )
```

- [ ] **Step 4: Run all evaluate tests**

```bash
rtk .venv/bin/pytest tests/test_evaluate_stage.py tests/test_report.py -v
```

Expected: all tests PASS; deterministic metrics remain unchanged because only the manifest gains one input hash.

- [ ] **Step 5: Commit**

```bash
rtk git add src/tinyfables/stages/evaluate.py tests/test_evaluate_stage.py
rtk git commit -m "fix(eval): record paraphrase bank in provenance"
```

### Task 3: Torch-Free Figure Data, Control Validation, and Provenance

**Files:**
- Modify: `src/tinyfables/config.py:100-118`
- Create: `src/tinyfables/figure_data.py`
- Create: `tests/fixtures/issue10_eval_aug.json`
- Create: `tests/fixtures/issue10_eval_noaug.json`
- Create: `tests/fixtures/issue10_data_curve.json`
- Create: `tests/test_figure_data.py`

**Interfaces:**
- Consumes: `FiguresConfig`; augmented/no-aug `eval_metrics.json` + `manifest.json`; Issue 07 `data_curve.json` + reward `manifest.json`; the four committed prep/pretrain configs.
- Produces: `FigureData(robustness_rows: list[dict], rm_rows: list[dict], provenance: dict, held_out_statement: str, inputs: dict[str, Path])` from `assemble_figure_data(cfg: FiguresConfig) -> FigureData`. Task 4 writes and renders these values without reinterpreting them.

- [ ] **Step 1: Add fixture augmented metrics**

Create `tests/fixtures/issue10_eval_aug.json`:

```json
{
  "perplexity": {
    "fable_token_loss": 1.4019,
    "perplexity": 4.063,
    "n_rows": 500
  },
  "adherence_grid": {
    "canonical": {
      "character": 0.8,
      "setting": 0.4,
      "challenge": 0.3,
      "outcome": 0.5,
      "overall": 0.5,
      "moral_delivery": 0.6,
      "n": 50
    },
    "seen-template": {
      "character": 0.7,
      "setting": 0.4,
      "challenge": 0.3,
      "outcome": 0.4,
      "overall": 0.45,
      "moral_delivery": 0.55,
      "n": 50
    },
    "held-out-template": {
      "character": 0.6,
      "setting": 0.4,
      "challenge": 0.2,
      "outcome": 0.4,
      "overall": 0.4,
      "moral_delivery": 0.5,
      "n": 50
    }
  },
  "provenance": {
    "checkpoint_sha": "0000000000000000000000000000000000000000000000000000000000000000",
    "tokenizer_sha": "1111111111111111111111111111111111111111111111111111111111111111",
    "seed": 0,
    "versions": {}
  }
}
```

- [ ] **Step 2: Add fixture no-aug metrics**

Create `tests/fixtures/issue10_eval_noaug.json`:

```json
{
  "perplexity": {
    "fable_token_loss": 1.4702,
    "perplexity": 4.35,
    "n_rows": 500
  },
  "adherence_grid": {
    "canonical": {
      "character": 0.76,
      "setting": 0.4,
      "challenge": 0.28,
      "outcome": 0.48,
      "overall": 0.48,
      "moral_delivery": 0.58,
      "n": 50
    },
    "seen-template": {
      "character": 0.64,
      "setting": 0.34,
      "challenge": 0.24,
      "outcome": 0.38,
      "overall": 0.4,
      "moral_delivery": 0.45,
      "n": 50
    },
    "held-out-template": {
      "character": 0.5,
      "setting": 0.3,
      "challenge": 0.1,
      "outcome": 0.3,
      "overall": 0.3,
      "moral_delivery": 0.35,
      "n": 50
    }
  },
  "provenance": {
    "checkpoint_sha": "0000000000000000000000000000000000000000000000000000000000000000",
    "tokenizer_sha": "1111111111111111111111111111111111111111111111111111111111111111",
    "seed": 0,
    "versions": {}
  }
}
```

- [ ] **Step 3: Add the measured Issue 07 curve fixture**

Create `tests/fixtures/issue10_data_curve.json`:

```json
{
  "points": [
    {
      "final_loss": 0.000061,
      "held_out_accuracy": 0.617021,
      "requested_train_size": 100,
      "steps": 400,
      "train_size": 100
    },
    {
      "final_loss": 0.001634,
      "held_out_accuracy": 0.611702,
      "requested_train_size": 500,
      "steps": 400,
      "train_size": 500
    },
    {
      "final_loss": 0.085327,
      "held_out_accuracy": 0.579787,
      "requested_train_size": 1000,
      "steps": 400,
      "train_size": 1000
    },
    {
      "final_loss": 0.248901,
      "held_out_accuracy": 0.62766,
      "requested_train_size": 2000,
      "steps": 400,
      "train_size": 1761
    }
  ]
}
```

- [ ] **Step 4: Write the failing data/provenance tests**

Create `tests/test_figure_data.py`:

```python
import json
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
```

- [ ] **Step 5: Run the tests to verify failure**

```bash
rtk .venv/bin/pytest tests/test_figure_data.py -v
```

Expected: collection FAILS because `FiguresConfig` and `tinyfables.figure_data` do not exist.

- [ ] **Step 6: Add `FiguresConfig`**

Insert after `EvalConfig` in `src/tinyfables/config.py`:

```python
@dataclass(frozen=True)
class FiguresConfig:
    augmented_eval_dir: str
    noaug_eval_dir: str
    reward_dir: str
    augmented_run_id: str
    noaug_run_id: str
    reward_run_id: str
    augmented_prep_config: str
    noaug_prep_config: str
    augmented_pretrain_config: str
    noaug_pretrain_config: str

    def __post_init__(self) -> None:
        values = (
            self.augmented_eval_dir,
            self.noaug_eval_dir,
            self.reward_dir,
            self.augmented_run_id,
            self.noaug_run_id,
            self.reward_run_id,
            self.augmented_prep_config,
            self.noaug_prep_config,
            self.augmented_pretrain_config,
            self.noaug_pretrain_config,
        )
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError("figure paths and run ids must be non-empty strings")
        if len({self.augmented_run_id, self.noaug_run_id, self.reward_run_id}) != 3:
            raise ValueError("figure run ids must be distinct")
```

- [ ] **Step 7: Implement the torch-free assembler**

Create `src/tinyfables/figure_data.py`:

```python
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
    if prep_aug["tokenizer_dir"] != prep_noaug["tokenizer_dir"]:
        raise ValueError("both prep configs must reuse one tokenizer")
    if pre_aug["tokenizer_dir"] != pre_noaug["tokenizer_dir"]:
        raise ValueError("both pretrain configs must reuse one tokenizer")

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
        ("with-aug", 0.15, cfg.augmented_run_id, aug_metrics, aug_inputs, aug_manifest_sha),
        ("no-aug", 0.0, cfg.noaug_run_id, noaug_metrics, noaug_inputs, noaug_manifest_sha),
    )
    for model, coverage, run_id, metrics, inputs, manifest_sha in runs:
        perplexity = float(metrics["perplexity"]["perplexity"])
        for family in FAMILY_ORDER:
            source = metrics["adherence_grid"][family]
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
                "n": int(source["n"]),
                "perplexity": perplexity,
                "checkpoint_sha256": inputs["model.safetensors"],
                "eval_manifest_sha256": manifest_sha,
            }
            if row["n"] <= 0:
                raise ValueError(f"{model}/{family} has no generations")
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
```

- [ ] **Step 8: Run the data tests**

```bash
rtk .venv/bin/pytest tests/test_figure_data.py -v
```

Expected: 3 PASS. The test proves all six robustness cells are retained, the 2,000-request curve point reports 1,761 actual rows, config contamination fails loudly, and artifact tampering fails loudly.

- [ ] **Step 9: Commit**

```bash
rtk git add src/tinyfables/config.py src/tinyfables/figure_data.py tests/fixtures/issue10_eval_aug.json tests/fixtures/issue10_eval_noaug.json tests/fixtures/issue10_data_curve.json tests/test_figure_data.py
rtk git commit -m "feat(figures): validate ablation inputs and assemble tables"
```

### Task 4: Report-Ready Figure Stage

**Skill gate:** The main session must load the dataviz skill before dispatching this task and must tell the worker that the skill is active. This task is the first one that writes figure code.

**Files:**
- Create: `src/tinyfables/stages/figures.py`
- Modify: `src/tinyfables/stages/__init__.py:3-33`
- Modify: `src/tinyfables/stage.py:18`
- Create: `configs/figures_full.yaml`
- Create: `tests/test_figures_stage.py`

**Interfaces:**
- Consumes: `FiguresConfig` and `assemble_figure_data(cfg) -> FigureData` from Task 3.
- Produces: stage `figures`; `robustness_grid.png`, `robustness_grid.csv`, `rm_data_curve.png`, `rm_data_curve.csv`, `figure_provenance.json`, `figure_summary.md`, then `manifest.json` last. `run` accepts injected renderers for dependency-free tests; CLI dispatch uses the real matplotlib renderers.

- [ ] **Step 1: Write the failing stage tests**

Create `tests/test_figures_stage.py`:

```python
import csv
import json
from pathlib import Path

import pytest

from tests.test_figure_data import make_inputs
from tinyfables.config import FiguresConfig, load_config
from tinyfables.figure_data import assemble_figure_data
from tinyfables.stages import REGISTRY
from tinyfables.stages import figures as figures_stage

REPO = Path(__file__).resolve().parents[1]
PNG_HEADER = b"\x89PNG\r\n\x1a\n"


def fake_renderer(rows, provenance, path):
    assert rows
    assert provenance["schema_version"] == 1
    Path(path).write_bytes(PNG_HEADER)


def test_figures_registered_and_full_config_loads():
    assert REGISTRY["figures"] == (FiguresConfig, "tinyfables.stages.figures")
    cfg = load_config(REPO / "configs" / "figures_full.yaml", FiguresConfig)
    assert cfg.augmented_eval_dir == "runs/eval_aug"
    assert cfg.noaug_eval_dir == "runs/eval_noaug"
    assert cfg.reward_dir == "runs/reward_base"
    assert cfg.augmented_run_id == "issue10-eval-with-aug-v1"
    assert cfg.noaug_run_id == "issue10-eval-noaug-v1"
    assert cfg.reward_run_id == "issue07-reward-base-v1"


def test_stage_exports_png_table_provenance_summary_and_manifest(tmp_path):
    cfg, _, _, _ = make_inputs(tmp_path)
    out = tmp_path / "figures"
    figures_stage.run(
        cfg,
        out,
        robustness_renderer=fake_renderer,
        rm_renderer=fake_renderer,
    )

    expected = {
        "robustness_grid.png",
        "robustness_grid.csv",
        "rm_data_curve.png",
        "rm_data_curve.csv",
        "figure_provenance.json",
        "figure_summary.md",
        "manifest.json",
    }
    assert expected <= {path.name for path in out.iterdir()}
    assert (out / "robustness_grid.png").read_bytes() == PNG_HEADER
    assert (out / "rm_data_curve.png").read_bytes() == PNG_HEADER
    robustness = list(csv.DictReader(open(out / "robustness_grid.csv")))
    curve = list(csv.DictReader(open(out / "rm_data_curve.csv")))
    assert len(robustness) == 6
    assert len(curve) == 4
    assert [row["family"] for row in robustness[:3]] == [
        "canonical",
        "seen-template",
        "held-out-template",
    ]
    assert robustness[2]["eval_manifest_sha256"]
    assert curve[-1]["requested_train_size"] == "2000"
    assert curve[-1]["train_size"] == "1761"
    provenance = json.loads((out / "figure_provenance.json").read_text())
    assert provenance["figures"]["robustness_grid"]["held_out_column"].startswith(
        "held-out-template"
    )
    summary = (out / "figure_summary.md").read_text()
    assert "## Robustness grid" in summary
    assert "## RM data curve" in summary
    assert "held-out templates" in summary
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["stage"] == "figures"
    assert set(manifest["artifacts"]) == expected - {"manifest.json"}
    assert set(manifest["inputs"]) == {
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


def test_real_matplotlib_renderers_when_dependency_is_installed(tmp_path):
    pytest.importorskip("matplotlib")
    cfg, _, _, _ = make_inputs(tmp_path)
    data = assemble_figure_data(cfg)
    robustness_png = tmp_path / "robustness.png"
    curve_png = tmp_path / "curve.png"
    figures_stage.render_robustness_grid(
        data.robustness_rows, data.provenance, robustness_png
    )
    figures_stage.render_rm_data_curve(data.rm_rows, data.provenance, curve_png)
    assert robustness_png.read_bytes().startswith(PNG_HEADER)
    assert curve_png.read_bytes().startswith(PNG_HEADER)
    assert robustness_png.stat().st_size > 1000
    assert curve_png.stat().st_size > 1000
```

- [ ] **Step 2: Run the tests to verify failure**

```bash
rtk .venv/bin/pytest tests/test_figures_stage.py -v
```

Expected: collection FAILS because `tinyfables.stages.figures` is absent and `figures` is not registered.

- [ ] **Step 3: Add the committed full figure config**

Create `configs/figures_full.yaml`:

```yaml
# Issue 10 report figures. On Colab, these paths are populated only from
# completed local stage outputs or immutable Hub snapshots.
augmented_eval_dir: runs/eval_aug
noaug_eval_dir: runs/eval_noaug
reward_dir: runs/reward_base
augmented_run_id: issue10-eval-with-aug-v1
noaug_run_id: issue10-eval-noaug-v1
reward_run_id: issue07-reward-base-v1
augmented_prep_config: configs/prep_full.yaml
noaug_prep_config: configs/prep_noaug_full.yaml
augmented_pretrain_config: configs/pretrain_full.yaml
noaug_pretrain_config: configs/pretrain_noaug_full.yaml
```

- [ ] **Step 4: Implement the stage and renderers**

Create `src/tinyfables/stages/figures.py`:

```python
"""Issue 10 report figures: controlled robustness grid and RM data curve."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from tinyfables.config import FiguresConfig
from tinyfables.figure_data import FAMILY_ORDER, MODEL_ORDER, assemble_figure_data
from tinyfables.stage import write_manifest

ROBUSTNESS_FIELDS = (
    "model",
    "paraphrase_coverage",
    "run_id",
    "family",
    "character",
    "setting",
    "challenge",
    "outcome",
    "adherence",
    "moral_delivery",
    "n",
    "perplexity",
    "checkpoint_sha256",
    "eval_manifest_sha256",
)
RM_FIELDS = (
    "run_id",
    "requested_train_size",
    "train_size",
    "held_out_accuracy",
    "final_loss",
    "steps",
    "accuracy_gate",
    "reward_manifest_sha256",
)


def _matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as exc:
        raise RuntimeError(
            "the figures stage requires matplotlib; install tinyfables[train]"
        ) from exc
    return plt, Rectangle


def render_robustness_grid(rows, provenance, png_path) -> None:
    """Render two literal 2-model × 3-family heatmaps on a common 0-1 scale."""
    import numpy as np

    plt, Rectangle = _matplotlib()
    lookup = {(row["model"], row["family"]): row for row in rows}
    metrics = (("adherence", "Element adherence"), ("moral_delivery", "Moral delivery"))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    image = None
    for ax, (field, title) in zip(axes, metrics):
        matrix = np.asarray(
            [[lookup[(model, family)][field] for family in FAMILY_ORDER] for model in MODEL_ORDER],
            dtype=float,
        )
        image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues", aspect="auto")
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(3), ("Canonical", "Seen", "Held-out"))
        ax.set_yticks(range(2), ("With aug", "No aug"))
        ax.tick_params(axis="both", length=0)
        for row_idx in range(2):
            for col_idx in range(3):
                value = matrix[row_idx, col_idx]
                color = "white" if value >= 0.55 else "#111111"
                ax.text(
                    col_idx,
                    row_idx,
                    f"{100.0 * value:.1f}%",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=10,
                    fontweight="semibold",
                )
        ax.add_patch(
            Rectangle(
                (1.5, -0.5),
                1.0,
                2.0,
                fill=False,
                edgecolor="#d55e00",
                linewidth=2.2,
                clip_on=False,
            )
        )
    fig.colorbar(image, ax=axes, label="rate", fraction=0.03, pad=0.04)
    fig.suptitle("Prompt robustness: paraphrase-augmentation ablation", fontsize=13)
    perplexity = {
        model: next(row["perplexity"] for row in rows if row["model"] == model)
        for model in MODEL_ORDER
    }
    manifests = provenance["figures"]["robustness_grid"]["eval_manifest_sha256"]
    run_ids = provenance["figures"]["robustness_grid"]["run_ids"]
    fig.text(
        0.01,
        0.055,
        (
            f"Fable-token perplexity — with aug: {perplexity['with-aug']:.3f}; "
            f"no aug: {perplexity['no-aug']:.3f}. Held-out = five eval-only templates."
        ),
        fontsize=7.5,
    )
    fig.text(
        0.01,
        0.018,
        (
            f"Runs: {run_ids[0]} / {run_ids[1]} · "
            f"manifests: {manifests[0][:12]} / {manifests[1][:12]}"
        ),
        fontsize=6.5,
        color="#444444",
    )
    fig.subplots_adjust(left=0.10, right=0.88, bottom=0.23, top=0.78, wspace=0.36)
    fig.savefig(
        png_path,
        dpi=200,
        bbox_inches="tight",
        metadata={"Title": "Issue 10 robustness grid", "Creator": "tinyfables figures stage"},
    )
    plt.close(fig)


def render_rm_data_curve(rows, provenance, png_path) -> None:
    """Render all four committed RM curve points with chance and gate references."""
    plt, _ = _matplotlib()
    x = [row["train_size"] for row in rows]
    y = [row["held_out_accuracy"] for row in rows]
    gate = rows[0]["accuracy_gate"]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(x, y, marker="o", linewidth=2.0, color="#0072b2")
    ax.axhline(0.5, linestyle=":", linewidth=1.2, color="#666666", label="chance (0.50)")
    ax.axhline(gate, linestyle="--", linewidth=1.4, color="#d55e00", label=f"gate ({gate:.2f})")
    for point_x, point_y in zip(x, y):
        ax.annotate(
            f"{100.0 * point_y:.1f}%",
            (point_x, point_y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    labels = [
        (
            f"{row['train_size']:,}\n({row['requested_train_size']:,} requested)"
            if row["train_size"] != row["requested_train_size"]
            else f"{row['train_size']:,}"
        )
        for row in rows
    ]
    lower = max(0.0, min(0.45, min(y) - 0.05))
    upper = min(1.0, max(0.70, max(y) + 0.05, gate + 0.03))
    ax.set_ylim(lower, upper)
    ax.set_xticks(x, labels)
    ax.set_xlabel("Preference pairs used for training")
    ax.set_ylabel("Held-out accuracy")
    ax.set_title("Reward-model data curve")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    rm_prov = provenance["figures"]["rm_data_curve"]
    fig.text(
        0.01,
        0.015,
        (
            f"Run: {rm_prov['run_id']} · reward manifest: "
            f"{rm_prov['reward_manifest_sha256'][:12]} · all committed points shown"
        ),
        fontsize=6.5,
        color="#444444",
    )
    fig.subplots_adjust(bottom=0.22, top=0.86)
    fig.savefig(
        png_path,
        dpi=200,
        bbox_inches="tight",
        metadata={"Title": "Issue 07 RM data curve", "Creator": "tinyfables figures stage"},
    )
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, robustness_rows, rm_rows, statement, provenance) -> None:
    lookup = {(row["model"], row["family"]): row for row in robustness_rows}
    lines = [
        "# Issue 10 figure summary",
        "",
        statement,
        "",
        "## Robustness grid",
        "",
        "| model | perplexity | canonical adherence / Moral | seen adherence / Moral | held-out adherence / Moral |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in MODEL_ORDER:
        row_values = [lookup[(model, family)] for family in FAMILY_ORDER]
        cells = [f"{row['adherence']:.3f} / {row['moral_delivery']:.3f}" for row in row_values]
        lines.append(
            f"| {model} | {row_values[0]['perplexity']:.3f} | {cells[0]} | {cells[1]} | {cells[2]} |"
        )
    lines.extend(
        [
            "",
            "## RM data curve",
            "",
            "| requested pairs | actual pairs | held-out accuracy | final loss | steps |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rm_rows:
        lines.append(
            f"| {row['requested_train_size']} | {row['train_size']} | "
            f"{row['held_out_accuracy']:.6f} | {row['final_loss']:.6f} | {row['steps']} |"
        )
    robust_prov = provenance["figures"]["robustness_grid"]
    rm_prov = provenance["figures"]["rm_data_curve"]
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- eval run ids: `{robust_prov['run_ids'][0]}`, `{robust_prov['run_ids'][1]}`",
            f"- eval manifests: `{robust_prov['eval_manifest_sha256'][0]}`, `{robust_prov['eval_manifest_sha256'][1]}`",
            f"- checkpoints: `{robust_prov['checkpoint_sha256'][0]}`, `{robust_prov['checkpoint_sha256'][1]}`",
            f"- tokenizer: `{robust_prov['tokenizer_sha256']}`",
            f"- paraphrase bank: `{robust_prov['paraphrase_bank_sha256']}`",
            f"- RM run id: `{rm_prov['run_id']}`",
            f"- RM manifest: `{rm_prov['reward_manifest_sha256']}`",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def run(
    cfg: FiguresConfig,
    out_dir: Path,
    *,
    robustness_renderer=None,
    rm_renderer=None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = assemble_figure_data(cfg)
    robustness_csv = out_dir / "robustness_grid.csv"
    rm_csv = out_dir / "rm_data_curve.csv"
    robustness_png = out_dir / "robustness_grid.png"
    rm_png = out_dir / "rm_data_curve.png"
    provenance_path = out_dir / "figure_provenance.json"
    summary_path = out_dir / "figure_summary.md"

    _write_csv(robustness_csv, data.robustness_rows, ROBUSTNESS_FIELDS)
    _write_csv(rm_csv, data.rm_rows, RM_FIELDS)
    provenance_path.write_text(json.dumps(data.provenance, indent=2, sort_keys=True) + "\n")
    _write_summary(
        summary_path,
        data.robustness_rows,
        data.rm_rows,
        data.held_out_statement,
        data.provenance,
    )
    (robustness_renderer or render_robustness_grid)(
        data.robustness_rows, data.provenance, robustness_png
    )
    (rm_renderer or render_rm_data_curve)(data.rm_rows, data.provenance, rm_png)

    write_manifest(
        out_dir,
        "figures",
        cfg,
        [
            robustness_png,
            robustness_csv,
            rm_png,
            rm_csv,
            provenance_path,
            summary_path,
        ],
        inputs=data.inputs,
    )
```

- [ ] **Step 5: Register the stage lazily**

Add `FiguresConfig` to the import list in `src/tinyfables/stages/__init__.py`, then add:

```python
    "figures": (FiguresConfig, "tinyfables.stages.figures"),
```

to `REGISTRY` immediately after `evaluate`.

- [ ] **Step 6: Record matplotlib versions when present**

Replace `_VERSIONED_PACKAGES` in `src/tinyfables/stage.py` with:

```python
_VERSIONED_PACKAGES = [
    "tinyfables",
    "tokenizers",
    "numpy",
    "torch",
    "transformers",
    "matplotlib",
]
```

The existing `PackageNotFoundError` path keeps local environments without matplotlib valid.

- [ ] **Step 7: Run the figure-stage tests**

```bash
rtk .venv/bin/pytest tests/test_figure_data.py tests/test_figures_stage.py -v
```

Expected in the current local `.venv`: 5 PASS and 1 SKIP (`matplotlib` is not installed). In a `.venv` with `.[train]`, expect 6 PASS. No test performs network I/O.

- [ ] **Step 8: Prove the light import path stayed light**

```bash
rtk .venv/bin/python -c "import sys, tinyfables.cli; assert 'matplotlib' not in sys.modules and 'torch' not in sys.modules; print('light')"
```

Expected: `light`.

- [ ] **Step 9: Commit**

```bash
rtk git add src/tinyfables/stages/figures.py src/tinyfables/stages/__init__.py src/tinyfables/stage.py configs/figures_full.yaml tests/test_figures_stage.py
rtk git commit -m "feat(figures): render robustness grid and RM data curve"
```

### Task 5: Toy No-Augmentation Chain

**Files:**
- Create: `tests/test_issue10_toy_chain.py`

**Interfaces:**
- Consumes: existing tokenizer, prep, pretrain, and evaluate stage APIs plus the real `configs/paraphrases.yaml` bank.
- Produces: an offline proof that one tokenizer feeds explicit coverage-zero prep, sibling pretraining, and a complete canonical/seen/held-out eval grid.

- [ ] **Step 1: Write the toy-chain test**

Create `tests/test_issue10_toy_chain.py`:

```python
import json
from pathlib import Path

from tinyfables.config import EvalConfig, PretrainConfig, PrepConfig, SourceSpec, TokenizerConfig
from tinyfables.stage import sha256_file
from tinyfables.stages import evaluate as evaluate_stage
from tinyfables.stages import prep as prep_stage
from tinyfables.stages import pretrain as pretrain_stage
from tinyfables.stages import tokenizer as tokenizer_stage

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_corpus.jsonl"
BANK = Path(__file__).resolve().parents[1] / "configs" / "paraphrases.yaml"


def test_toy_noaug_chain_reuses_tokenizer_and_reaches_three_family_eval(tmp_path):
    source = SourceSpec(jsonl_path=str(FIXTURE))
    tokenizer_dir = tmp_path / "tokenizer"
    prep_dir = tmp_path / "prep_noaug"
    model_dir = tmp_path / "base_noaug"
    eval_dir = tmp_path / "eval_noaug"

    tokenizer_stage.run(
        TokenizerConfig(source=source, vocab_size=512, seed=0),
        tokenizer_dir,
    )
    tokenizer_sha = sha256_file(tokenizer_dir / "tokenizer.json")
    prep_stage.run(
        PrepConfig(
            source=source,
            tokenizer_dir=str(tokenizer_dir),
            window=512,
            seed=0,
            paraphrase_bank=str(BANK),
            paraphrase_coverage=0.0,
        ),
        prep_dir,
    )
    prep_summary = json.loads((prep_dir / "prep_summary.json").read_text())
    assert prep_summary["n_rows"] == 24
    assert prep_summary["paraphrase_coverage"] == 0.0
    assert prep_summary["family_counts"] == {
        "canonical": 24,
        "seen-template": 0,
        "held-out-template": 0,
    }
    prep_manifest = json.loads((prep_dir / "manifest.json").read_text())
    assert prep_manifest["inputs"]["tokenizer.json"] == tokenizer_sha

    pretrain_stage.run(
        PretrainConfig(
            prep_dir=str(prep_dir),
            tokenizer_dir=str(tokenizer_dir),
            n_layer=1,
            n_head=2,
            d_model=32,
            n_ctx=512,
            batch_size=2,
            steps=2,
            warmup_steps=0,
            lr=1e-3,
            seed=0,
            device="cpu",
        ),
        model_dir,
    )
    pretrain_manifest = json.loads((model_dir / "manifest.json").read_text())
    assert pretrain_manifest["inputs"]["tokenizer.json"] == tokenizer_sha

    evaluate_stage.run(
        EvalConfig(
            checkpoint=str(model_dir),
            tokenizer_dir=str(tokenizer_dir),
            source=source,
            paraphrase_bank=str(BANK),
            n_ctx=512,
            n_perplexity_rows=2,
            n_generations=1,
            max_new_tokens=8,
            min_new_tokens=2,
            temperature=0.9,
            top_k=10,
            moral_threshold=0.3,
            seed=0,
            device="cpu",
        ),
        eval_dir,
    )
    metrics = json.loads((eval_dir / "eval_metrics.json").read_text())
    assert set(metrics["adherence_grid"]) == {
        "canonical",
        "seen-template",
        "held-out-template",
    }
    assert {row["n"] for row in metrics["adherence_grid"].values()} == {1}
    eval_manifest = json.loads((eval_dir / "manifest.json").read_text())
    assert eval_manifest["inputs"]["tokenizer.json"] == tokenizer_sha
    assert eval_manifest["inputs"]["paraphrases.yaml"] == sha256_file(BANK)
```

- [ ] **Step 2: Run the new chain**

```bash
rtk .venv/bin/pytest tests/test_issue10_toy_chain.py -v
```

Expected: 1 PASS. The test runs no network calls and creates the tokenizer exactly once.

- [ ] **Step 3: Run the Issue 10 focused suite**

```bash
rtk .venv/bin/pytest tests/test_full_configs.py tests/test_evaluate_stage.py tests/test_figure_data.py tests/test_figures_stage.py tests/test_issue10_toy_chain.py -v
```

Expected in the current `.venv`: all non-matplotlib tests PASS and the one real-renderer smoke test SKIPS.

- [ ] **Step 4: Commit**

```bash
rtk git add tests/test_issue10_toy_chain.py
rtk git commit -m "test(ablation): add toy no-augmentation chain"
```

### Task 6: Track A Design Record and Whole-Suite Gate

**Files:**
- Modify: `docs/design.md:350-370`

**Interfaces:**
- Consumes: Tasks 1–5 and their passing tests.
- Produces: committed Track A contract for the main-session Colab operator; no empirical claim is added before Track B runs.

- [ ] **Step 1: Add the Track A implementation record**

Insert after the Issue 07 implementation section and before the Operations section in `docs/design.md`:

```markdown
### Implementation (issue 10, Track A)

- **Controlled sibling configs.** `prep_noaug_full.yaml` differs from `prep_full.yaml`
  only in `paraphrase_coverage` (`0.15 -> 0.0`). It retains the same 450k-row HF
  source, source seed, 1024-token window, paraphrase bank, and the existing
  `runs/tokenizer_full`; the tokenizer is reused, not retrained. The sibling pretrain
  keeps every scientific field from `pretrain_full.yaml`; only prep/checkpoint/run
  paths identify the sibling. The two full eval configs differ only in `checkpoint`.
- **Report figure stage.** `figures` validates upstream artifact hashes, treatment-only
  prep, scientific pretrain equivalence, and checkpoint-only eval configs before it
  writes the literal 2-model x 3-family adherence/Moral grid and the complete Issue 07
  RM curve. Each figure has a PNG, long-form CSV, run ids and manifest hashes; the
  held-out column is explicitly the five eval-only templates.
- **Offline evidence.** The toy chain reuses one tokenizer for coverage-zero prep,
  sibling pretraining, and three-family eval. Fixture tests retain all six grid cells
  and all four RM-curve points, reject config contamination, and reject tampered
  artifacts. Measured values enter this section only after their Hub round trip.
```

- [ ] **Step 2: Run the complete offline suite**

```bash
rtk .venv/bin/pytest -q
```

Expected: every non-network test PASS; the optional real matplotlib renderer test may SKIP when matplotlib is absent.

- [ ] **Step 3: Inspect branch scope**

```bash
rtk git status --short
rtk git diff --stat main...HEAD
rtk git log --oneline --decorate main..HEAD
```

Expected: only the Issue 10 files in the File Map are changed; the unrelated Issue 08 worktree files are absent; Tasks 1–5 each have their own commit.

- [ ] **Step 4: Commit the Track A record**

```bash
rtk git add docs/design.md
rtk git commit -m "docs(design): record issue-10 figure contract"
```

## CHECKPOINT A — Review and Merge Track A

This checkpoint is main-session-only.

- [ ] **Step A.1: Complete two-stage review**

Use the subagent-driven-development workflow's spec-compliance reviewer first and code-quality reviewer second. Fix findings in focused commits and rerun `rtk .venv/bin/pytest -q`.

- [ ] **Step A.2: Integrate current `main` safely**

If Issue 08 landed while Track A was running, merge the latest `main` into the Issue 10 branch, resolve only genuine overlaps in `config.py`, `stages/__init__.py`, `tests/test_full_configs.py`, and `docs/design.md`, then rerun the full suite. Do not overwrite Issue 08 additions.

- [ ] **Step A.3: Finish the branch**

Load `superpowers:finishing-a-development-branch`, select the merge-to-main path, merge `issue-10-committed-ablation-figures`, and verify on `main`:

```bash
rtk .venv/bin/pytest -q
```

Expected: every non-network test PASS. Push `main` before opening Colab so the runtime clones the exact code under test.

---

## Track B — Main Session on Colab via colab-mcp

Track B is not delegated. Every time colab-mcp requires a browser connection, call `open_colab_browser_connection` and reopen it before the next code cell if the connection expires. Use `/content/repo` consistently.

## CHECKPOINT B — GPU-Quota Scheduling

- [ ] **Step B.1: Give Issue 08's short run first claim on a ready T4**

If Issue 08 Track B has a merged stage/config and a recorded ADR fork, run its short DPO/PPO job and push its artifacts before starting the Issue 10 20k-step pretrain. This avoids idling a nearly finished no-aug run while waiting for a second GPU quota.

- [ ] **Step B.2: Use checkpoint boundaries if the long run has already started**

If Issue 08 becomes runnable after the no-aug pretrain has started, yield the quota only after a completed `step_N/checkpoint_state.json` exists both on Drive and in `congthanh991/tinyfables-13m-base-noaug-ckpts`. End that runtime, complete and push Issue 08 Track B, then rerun the exact no-aug pretrain command; auto-resume must report `start_step: N`. Never change `steps`, seed, optimizer, learning rate, or data to make interleaving faster.

## CHECKPOINT C — Colab Setup and Immutable Hub Inputs

- [ ] **Step C.1: Open a live Colab T4 runtime**

Through colab-mcp, connect to a notebook with a live kernel and select a T4 GPU. Reopen the browser connection immediately before each `run_code_cell` call.

- [ ] **Step C.2: Clone the merged repository and install train/dev extras**

Run one Colab code cell:

```python
import os
import subprocess
import sys

subprocess.run(
    ["git", "clone", "https://github.com/harryct229/tinyfables.git", "/content/repo"],
    check=True,
)
os.chdir("/content/repo")
subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".[train,dev]", "-q"], check=True)
subprocess.run(
    [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_full_configs.py",
        "tests/test_figure_data.py",
        "tests/test_figures_stage.py",
        "-q",
    ],
    check=True,
)
```

Expected: the focused tests PASS and the real matplotlib smoke test also PASSES because `.[train]` is installed.

- [ ] **Step C.3: Mount Drive and authenticate without printing the token**

Run one Colab code cell:

```python
import os
from google.colab import drive, userdata
from huggingface_hub import HfApi, login

drive.mount("/content/drive")
token = userdata.get("HF_TOKEN")
assert token, "Add HF_TOKEN to Colab Secrets before running Track B"
login(token=token, add_to_git_credential=False)
api = HfApi(token=token)
api.create_repo(
    repo_id="congthanh991/tinyfables-13m-base-noaug",
    repo_type="model",
    private=True,
    exist_ok=True,
)
print("Drive mounted; Hub authenticated; no-aug repo ready")
```

Expected: the final line prints; the token value never appears in cell output.

- [ ] **Step C.4: Pull the Base Model, tokenizer, and pinned RM artifacts from the Hub**

Run one Colab code cell:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "congthanh991/tinyfables-13m-base",
    local_dir="/content/repo/runs/base_model",
)
snapshot_download(
    "congthanh991/tinyfables-tokenizer",
    local_dir="/content/repo/runs/tokenizer_full",
)
snapshot_download(
    "congthanh991/tinyfables-13m-rm",
    revision="58865e69ccbe1e90b3816068cabca410f28e43aa",
    allow_patterns=["data_curve.json", "manifest.json"],
    local_dir="/content/repo/runs/reward_base",
)
```

Expected: no input is read from the maintainer's local `runs/` directory.

- [ ] **Step C.5: Verify immutable input hashes and the mirrored curve**

Run one Colab code cell:

```python
import json
from pathlib import Path

from tinyfables.stage import sha256_file

root = Path("/content/repo")
base_manifest = json.loads((root / "runs/base_model/manifest.json").read_text())
rm_manifest = json.loads((root / "runs/reward_base/manifest.json").read_text())
base_model = root / "runs/base_model/model.safetensors"
tokenizer = root / "runs/tokenizer_full/tokenizer.json"
curve = root / "runs/reward_base/data_curve.json"

assert sha256_file(base_model) == "0dbd8f125a9bf12578e4b5246eebb0bab51b93c9fa169f78537bb4db95065e0a"
assert base_manifest["artifacts"]["model.safetensors"] == sha256_file(base_model)
assert sha256_file(tokenizer) == "9d53402d4c4dd94d0d7bf133ea8073cf9664103e51b08b3a8c813a00ec625dfb"
assert sha256_file(curve) == "721e321ecde5758702210d3e3de5ca1c9cf6fdc62d826f47360426455da9a6c0"
assert rm_manifest["artifacts"]["data_curve.json"] == sha256_file(curve)
assert [point["requested_train_size"] for point in json.loads(curve.read_text())["points"]] == [
    100,
    500,
    1000,
    2000,
]
print("immutable inputs verified")
```

Expected: `immutable inputs verified`. If any assertion fails, stop before prep; do not substitute a local artifact or another revision.

- [ ] **Step C.6: On every resumed pretrain runtime, pull prep from its Hub mirror**

Skip this only in the first runtime while Step D's local prep output is still present. After any Colab session death or planned handoff, repeat Steps C.1–C.5, then run:

```python
snapshot_download(
    "congthanh991/tinyfables-13m-base-noaug",
    allow_patterns=["prep_noaug/*"],
    local_dir="/content/repo/runs",
)
assert Path("/content/repo/runs/prep_noaug/tokens.bin").exists()
assert Path("/content/repo/runs/prep_noaug/mask.bin").exists()
assert Path("/content/repo/runs/prep_noaug/manifest.json").exists()
print("resumed runtime pulled prep from Hub")
```

Expected: `resumed runtime pulled prep from Hub`. Do not rerun the dataset prep merely to reconstruct an ephemeral input; the pushed stage artifact is the canonical resume input.

## CHECKPOINT D — No-Aug Prep, Immediate Hub Push

- [ ] **Step D.1: Run the full coverage-zero prep**

Run one Colab code cell:

```python
import subprocess
import sys

subprocess.run(
    [
        sys.executable,
        "-m",
        "tinyfables",
        "run",
        "prep",
        "--config",
        "configs/prep_noaug_full.yaml",
        "--out",
        "runs/prep_noaug",
    ],
    check=True,
)
```

Expected: `[tinyfables] stage 'prep' complete -> runs/prep_noaug`.

- [ ] **Step D.2: Enforce the full-run prep assertions**

Run one Colab code cell:

```python
import json
from pathlib import Path

summary = json.loads(Path("runs/prep_noaug/prep_summary.json").read_text())
manifest = json.loads(Path("runs/prep_noaug/manifest.json").read_text())
assert manifest["stage"] == "prep"
assert manifest["config"]["paraphrase_coverage"] == 0.0
assert manifest["config"]["source"] == {
    "jsonl_path": None,
    "hf_dataset": "klusai/ds-tf1-en-3m",
    "hf_split": "train",
    "max_rows": 450000,
}
assert summary["n_rows"] == 450000
assert summary["paraphrase_coverage"] == 0.0
assert summary["n_parse_failures"] == 0
assert summary["family_counts"] == {
    "canonical": 450000,
    "seen-template": 0,
    "held-out-template": 0,
}
assert manifest["inputs"]["tokenizer.json"] == "9d53402d4c4dd94d0d7bf133ea8073cf9664103e51b08b3a8c813a00ec625dfb"
print(json.dumps(summary, indent=2, sort_keys=True))
```

Expected: all assertions pass. This is the actual-run evidence for the same source/seed/450k sample and zero augmentation.

- [ ] **Step D.3: Push every prep artifact before starting pretrain**

Run one Colab code cell using the authenticated `api` from Step C.3 (recreate `api = HfApi(token=userdata.get("HF_TOKEN"))` if the kernel was restarted):

```python
prep_commit = api.upload_folder(
    folder_path="runs/prep_noaug",
    path_in_repo="prep_noaug",
    repo_id="congthanh991/tinyfables-13m-base-noaug",
    repo_type="model",
    commit_message="issue10: upload no-aug prep artifacts",
)
print("prep commit", prep_commit.oid)
```

Expected: a Hub commit SHA prints. The upload includes `tokens.bin`, `mask.bin`, `families.bin`, `prep_summary.json`, and `manifest.json`.

- [ ] **Step D.4: Verify the prep manifest round trip**

Run one Colab code cell:

```python
from huggingface_hub import hf_hub_download
from tinyfables.stage import sha256_file

remote_prep_manifest = hf_hub_download(
    "congthanh991/tinyfables-13m-base-noaug",
    filename="prep_noaug/manifest.json",
    force_download=True,
)
assert sha256_file(Path(remote_prep_manifest)) == sha256_file(Path("runs/prep_noaug/manifest.json"))
remote_files = set(api.list_repo_files("congthanh991/tinyfables-13m-base-noaug"))
assert {
    "prep_noaug/tokens.bin",
    "prep_noaug/mask.bin",
    "prep_noaug/families.bin",
    "prep_noaug/prep_summary.json",
    "prep_noaug/manifest.json",
} <= remote_files
print("prep round trip verified")
```

Expected: `prep round trip verified` before GPU training begins.

## CHECKPOINT E — Resumable Sibling Pretrain

- [ ] **Step E.1: Start or resume with the same command**

Run one Colab code cell:

```python
import subprocess
import sys

subprocess.run(
    [
        sys.executable,
        "-m",
        "tinyfables",
        "run",
        "pretrain",
        "--config",
        "configs/pretrain_noaug_full.yaml",
        "--out",
        "runs/base_noaug",
    ],
    check=True,
)
```

Expected on a new run: training starts at step 0. Expected after a Colab interruption: it discovers `/content/drive/MyDrive/tinyfables/ckpt_base_noaug/step_N` and resumes at `N`. The command, config, and `--out` path do not change across sessions.

- [ ] **Step E.2: Before any planned session handoff, verify both durable checkpoint copies**

Only needed when yielding the GPU before step 20,000. After the most recent 500-step boundary, run:

```python
from pathlib import Path
from tinyfables.checkpoint import find_latest_checkpoint

latest = find_latest_checkpoint("/content/drive/MyDrive/tinyfables/ckpt_base_noaug")
assert latest is not None
assert (latest / "checkpoint_state.json").exists()
checkpoint_files = set(
    api.list_repo_files("congthanh991/tinyfables-13m-base-noaug-ckpts", repo_type="model")
)
assert {"checkpoint_state.json", "model.safetensors", "optimizer.pt"} <= checkpoint_files
print("safe handoff checkpoint", latest.name)
```

Expected: `safe handoff checkpoint step_N`. If the Hub mirror is missing because the best-effort upload failed, run this in the same cell before ending the runtime:

```python
api.upload_folder(
    folder_path=str(latest),
    repo_id="congthanh991/tinyfables-13m-base-noaug-ckpts",
    repo_type="model",
    commit_message=f"issue10: recover {latest.name} checkpoint mirror",
)
```

- [ ] **Step E.3: Validate the completed sibling and scientific control**

After step 20,000, run:

```python
import json
import math
from pathlib import Path

summary = json.loads(Path("runs/base_noaug/pretrain_summary.json").read_text())
noaug_manifest = json.loads(Path("runs/base_noaug/manifest.json").read_text())
aug_manifest = json.loads(Path("runs/base_model/manifest.json").read_text())
assert summary["steps"] == 20000
assert summary["n_params"] == 14186496
assert summary["window"] == 1024
assert summary["amp"] is True
assert math.isfinite(summary["final_loss"])
assert noaug_manifest["stage"] == "pretrain"
assert noaug_manifest["inputs"]["tokenizer.json"] == aug_manifest["inputs"]["tokenizer.json"]

operational = {
    "prep_dir",
    "ckpt_dir",
    "run_name",
    "ckpt_hub_repo",
    "trackio_space_id",
}
aug_scientific = {
    key: value for key, value in aug_manifest["config"].items() if key not in operational
}
noaug_scientific = {
    key: value for key, value in noaug_manifest["config"].items() if key not in operational
}
assert noaug_scientific == aug_scientific
print(json.dumps(summary, indent=2, sort_keys=True))
```

Expected: the summary prints, and scientific config equality holds against the actual augmented Base Model manifest, not only the committed YAML.

- [ ] **Step E.4: Push the loadable model and the normally ignored optimizer artifact**

Run one Colab code cell:

```python
import subprocess
import sys

subprocess.run(
    [
        sys.executable,
        "-m",
        "tinyfables",
        "push",
        "--kind",
        "model",
        "--path",
        "runs/base_noaug",
        "--repo",
        "congthanh991/tinyfables-13m-base-noaug",
    ],
    check=True,
)
optimizer_commit = api.upload_file(
    path_or_fileobj="runs/base_noaug/optimizer.pt",
    path_in_repo="optimizer.pt",
    repo_id="congthanh991/tinyfables-13m-base-noaug",
    repo_type="model",
    commit_message="issue10: preserve final optimizer artifact",
)
print("optimizer commit", optimizer_commit.oid)
```

Expected: the model push succeeds, then the final optimizer state is restored at the manifest-recorded root path so every final pretrain artifact is present.

- [ ] **Step E.5: Round-trip load the sibling before evaluation**

Run one Colab code cell:

```python
import json
from pathlib import Path
from huggingface_hub import snapshot_download
from tinyfables.model import GPT
from tinyfables.stage import sha256_file

roundtrip = Path(
    snapshot_download(
        "congthanh991/tinyfables-13m-base-noaug",
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "optimizer.pt",
            "pretrain_summary.json",
            "loss_log.csv",
            "loss_curve.png",
            "manifest.json",
        ],
        local_dir="/content/roundtrip_noaug_model",
        force_download=True,
    )
)
manifest = json.loads((roundtrip / "manifest.json").read_text())
for relative, expected_hash in manifest["artifacts"].items():
    path = roundtrip / relative
    assert path.exists(), relative
    assert sha256_file(path) == expected_hash, relative
model = GPT.from_pretrained(roundtrip).eval()
assert sum(parameter.numel() for parameter in model.parameters()) == 14186496
print("sibling model round trip verified")
```

Expected: `sibling model round trip verified`. Do not evaluate a local model that failed this Hub round trip.

## CHECKPOINT F — Identical Dual Eval, Figures, and Final Hub Bundle

- [ ] **Step F.1: Prove the eval configs differ only in checkpoint immediately before running**

Run one Colab code cell:

```python
from dataclasses import asdict

from tinyfables.config import EvalConfig, load_config

aug_eval_cfg = asdict(load_config("configs/eval_full.yaml", EvalConfig))
noaug_eval_cfg = asdict(load_config("configs/eval_noaug_full.yaml", EvalConfig))
eval_diffs = {
    key: [aug_eval_cfg[key], noaug_eval_cfg[key]]
    for key in aug_eval_cfg
    if aug_eval_cfg[key] != noaug_eval_cfg[key]
}
assert eval_diffs == {"checkpoint": ["runs/base_model", "runs/base_noaug"]}
print("eval configs controlled")
```

Expected: `eval configs controlled`.

- [ ] **Step F.2: Evaluate the augmented Base Model and push immediately**

Run one Colab code cell:

```python
import subprocess
import sys

subprocess.run(
    [
        sys.executable,
        "-m",
        "tinyfables",
        "run",
        "evaluate",
        "--config",
        "configs/eval_full.yaml",
        "--out",
        "runs/eval_aug",
    ],
    check=True,
)
aug_eval_commit = api.upload_folder(
    folder_path="runs/eval_aug",
    path_in_repo="eval_with_aug",
    repo_id="congthanh991/tinyfables-13m-base-noaug",
    repo_type="model",
    commit_message="issue10: upload augmented-model eval",
)
print("aug eval commit", aug_eval_commit.oid)
```

Expected: eval completes and the Hub commit SHA prints before the no-aug eval begins.

- [ ] **Step F.3: Evaluate the no-aug sibling and push immediately**

Run one Colab code cell:

```python
subprocess.run(
    [
        sys.executable,
        "-m",
        "tinyfables",
        "run",
        "evaluate",
        "--config",
        "configs/eval_noaug_full.yaml",
        "--out",
        "runs/eval_noaug",
    ],
    check=True,
)
noaug_eval_commit = api.upload_folder(
    folder_path="runs/eval_noaug",
    path_in_repo="eval_noaug",
    repo_id="congthanh991/tinyfables-13m-base-noaug",
    repo_type="model",
    commit_message="issue10: upload no-aug-model eval",
)
print("no-aug eval commit", noaug_eval_commit.oid)
```

Expected: eval completes and the second Hub commit SHA prints.

- [ ] **Step F.4: Validate the actual dual-eval artifacts before plotting**

Run one Colab code cell:

```python
import json
from pathlib import Path

aug_metrics = json.loads(Path("runs/eval_aug/eval_metrics.json").read_text())
noaug_metrics = json.loads(Path("runs/eval_noaug/eval_metrics.json").read_text())
aug_manifest = json.loads(Path("runs/eval_aug/manifest.json").read_text())
noaug_manifest = json.loads(Path("runs/eval_noaug/manifest.json").read_text())
families = {"canonical", "seen-template", "held-out-template"}
assert set(aug_metrics["adherence_grid"]) == families
assert set(noaug_metrics["adherence_grid"]) == families
assert {row["n"] for row in aug_metrics["adherence_grid"].values()} == {50}
assert {row["n"] for row in noaug_metrics["adherence_grid"].values()} == {50}
assert aug_manifest["inputs"]["tokenizer.json"] == noaug_manifest["inputs"]["tokenizer.json"]
assert aug_manifest["inputs"]["paraphrases.yaml"] == noaug_manifest["inputs"]["paraphrases.yaml"]
assert aug_manifest["inputs"]["model.safetensors"] != noaug_manifest["inputs"]["model.safetensors"]
assert {
    key: [aug_manifest["config"][key], noaug_manifest["config"][key]]
    for key in aug_manifest["config"]
    if aug_manifest["config"][key] != noaug_manifest["config"][key]
} == {"checkpoint": ["runs/base_model", "runs/base_noaug"]}
print("dual eval artifacts controlled")
```

Expected: `dual eval artifacts controlled`. Any failure invalidates the ablation; do not edit metrics or continue to plotting.

- [ ] **Step F.5: Assemble both figures from the validated runs**

Run one Colab code cell:

```python
subprocess.run(
    [
        sys.executable,
        "-m",
        "tinyfables",
        "run",
        "figures",
        "--config",
        "configs/figures_full.yaml",
        "--out",
        "runs/figures_issue10",
    ],
    check=True,
)
```

Expected: `[tinyfables] stage 'figures' complete -> runs/figures_issue10` and both PNGs exist.

- [ ] **Step F.6: Inspect the complete tables and honest statement**

Run one Colab code cell:

```python
import csv
import json
from pathlib import Path

robustness = list(csv.DictReader(open("runs/figures_issue10/robustness_grid.csv")))
curve = list(csv.DictReader(open("runs/figures_issue10/rm_data_curve.csv")))
provenance = json.loads(Path("runs/figures_issue10/figure_provenance.json").read_text())
assert len(robustness) == 6
assert len(curve) == 4
assert {(row["model"], row["family"]) for row in robustness} == {
    (model, family)
    for model in ("with-aug", "no-aug")
    for family in ("canonical", "seen-template", "held-out-template")
}
assert provenance["figures"]["robustness_grid"]["run_ids"] == [
    "issue10-eval-with-aug-v1",
    "issue10-eval-noaug-v1",
]
assert provenance["figures"]["rm_data_curve"]["data_curve_sha256"] == (
    "721e321ecde5758702210d3e3de5ca1c9cf6fdc62d826f47360426455da9a6c0"
)
print(Path("runs/figures_issue10/figure_summary.md").read_text())
```

Expected: the printed Markdown contains the measured 2×3 grid, both perplexities, all four curve points, and exactly one positive, mixed, or “augmentation didn't help” held-out statement. Do not revise the wording based on which model won.

- [ ] **Step F.7: Display both PNGs in Colab for visual QA**

Run one Colab code cell:

```python
from IPython.display import Image, display

display(Image(filename="runs/figures_issue10/robustness_grid.png"))
display(Image(filename="runs/figures_issue10/rm_data_curve.png"))
```

Verify visually: titles and annotations are legible; both grouped-column panels
share the same 0–1 rate range; each family has paired with-augmentation and
no-augmentation bars; the held-out category is highlighted; no label or bar is
cropped; the RM chart contains 100, 500, 1,000, and actual 1,761 with “2,000
requested”; chance and 0.65 gate lines are labeled; neither PNG contains a
hand-entered number.

- [ ] **Step F.8: Push the complete figure folder**

Run one Colab code cell:

```python
figures_commit = api.upload_folder(
    folder_path="runs/figures_issue10",
    path_in_repo="figures",
    repo_id="congthanh991/tinyfables-13m-base-noaug",
    repo_type="model",
    commit_message="issue10: upload report figures and provenance",
)
print("figures commit", figures_commit.oid)
```

Expected: a Hub commit SHA prints before the Colab runtime is released.

- [ ] **Step F.9: Round-trip every small Issue 10 artifact and list every large prep artifact**

Run one Colab code cell:

```python
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from tinyfables.stage import sha256_file

bundle = Path(
    snapshot_download(
        "congthanh991/tinyfables-13m-base-noaug",
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "optimizer.pt",
            "pretrain_summary.json",
            "loss_log.csv",
            "loss_curve.png",
            "manifest.json",
            "prep_noaug/prep_summary.json",
            "prep_noaug/manifest.json",
            "eval_with_aug/*",
            "eval_noaug/*",
            "figures/*",
        ],
        local_dir="/content/roundtrip_issue10_bundle",
        force_download=True,
    )
)
for subdir in (bundle, bundle / "eval_with_aug", bundle / "eval_noaug", bundle / "figures"):
    manifest = json.loads((subdir / "manifest.json").read_text())
    for relative, expected_hash in manifest["artifacts"].items():
        artifact = subdir / relative
        assert artifact.exists(), artifact
        assert sha256_file(artifact) == expected_hash, artifact

remote_files = set(api.list_repo_files("congthanh991/tinyfables-13m-base-noaug"))
assert {
    "prep_noaug/tokens.bin",
    "prep_noaug/mask.bin",
    "prep_noaug/families.bin",
    "prep_noaug/prep_summary.json",
    "prep_noaug/manifest.json",
    "eval_with_aug/eval_metrics.json",
    "eval_with_aug/manifest.json",
    "eval_noaug/eval_metrics.json",
    "eval_noaug/manifest.json",
    "figures/robustness_grid.png",
    "figures/robustness_grid.csv",
    "figures/rm_data_curve.png",
    "figures/rm_data_curve.csv",
    "figures/figure_provenance.json",
    "figures/manifest.json",
} <= remote_files
repo_revision = api.model_info("congthanh991/tinyfables-13m-base-noaug").sha
print("final issue10 Hub revision", repo_revision)
```

Expected: every downloaded hash matches and a final immutable repo revision prints. Record that SHA for the design document.

## CHECKPOINT G — Local Visual Review and Measured Documentation

This checkpoint is main-session-only and happens after the Colab runtime has pushed everything.

- [ ] **Step G.1: Pull the final bundle from the Hub, never from a local run cache**

From `/Users/thanh/code/tinystories` on updated `main`, run:

```bash
rtk .venv/bin/python -c "from huggingface_hub import snapshot_download; print(snapshot_download('congthanh991/tinyfables-13m-base-noaug', allow_patterns=['pretrain_summary.json','manifest.json','prep_noaug/prep_summary.json','prep_noaug/manifest.json','eval_with_aug/*','eval_noaug/*','figures/*'], local_dir='/tmp/issue10_closeout', force_download=True))"
```

Expected: `/tmp/issue10_closeout` prints.

- [ ] **Step G.2: Inspect the round-tripped PNGs with the image viewer**

Use `view_image` on:

- `/tmp/issue10_closeout/figures/robustness_grid.png`
- `/tmp/issue10_closeout/figures/rm_data_curve.png`

Confirm the same visual QA checklist from Step F.7. If a rendering defect exists, fix renderer code through a focused Track A-style branch/test/review cycle, rerun only the deterministic figure stage on Colab, push the replacement bundle, and repeat the round trip. Never hand-edit a PNG.

- [ ] **Step G.3: Generate the exact measured design block from Hub artifacts**

Run this read-only command; it prints ready-to-apply Markdown with no hand-transcribed metrics:

```bash
rtk .venv/bin/python - <<'PY'
import csv
import json
from pathlib import Path
from huggingface_hub import HfApi
from tinyfables.stage import sha256_file

root = Path("/tmp/issue10_closeout")
prep = json.loads((root / "prep_noaug/prep_summary.json").read_text())
pretrain = json.loads((root / "pretrain_summary.json").read_text())
rows = list(csv.DictReader(open(root / "figures/robustness_grid.csv")))
curve = list(csv.DictReader(open(root / "figures/rm_data_curve.csv")))
prov = json.loads((root / "figures/figure_provenance.json").read_text())
statement = prov["held_out_interpretation"]
revision = HfApi().model_info("congthanh991/tinyfables-13m-base-noaug").sha
lookup = {(row["model"], row["family"]): row for row in rows}
families = ("canonical", "seen-template", "held-out-template")

print("### Implementation (issue 10, Track B)")
print()
print(
    f"- **No-aug sibling.** Prep sampled {prep['n_rows']:,} rows with coverage "
    f"{prep['paraphrase_coverage']:.2f}: {prep['family_counts']['canonical']:,} canonical, "
    f"{prep['family_counts']['seen-template']:,} seen-template, "
    f"{prep['family_counts']['held-out-template']:,} held-out-template; "
    f"pretrain completed {pretrain['steps']:,} controlled steps with final fable-token "
    f"loss {pretrain['final_loss']:.4f} and {pretrain['n_params']:,} parameters."
)
print("- **Controlled robustness grid.** Both checkpoints used the identical 50-spec eval config; only `checkpoint` differed.")
print()
print("| model | fable-token perplexity | canonical adherence / Moral | seen adherence / Moral | held-out adherence / Moral |")
print("|---|---:|---:|---:|---:|")
for model in ("with-aug", "no-aug"):
    values = [lookup[(model, family)] for family in families]
    cells = [f"{float(row['adherence']):.3f} / {float(row['moral_delivery']):.3f}" for row in values]
    print(f"| {model} | {float(values[0]['perplexity']):.3f} | {cells[0]} | {cells[1]} | {cells[2]} |")
print()
print(f"- **Held-out result.** {statement}")
print("- **RM data curve (Issue 07).** " + "; ".join(
    f"{row['requested_train_size']} requested / {row['train_size']} actual -> {float(row['held_out_accuracy']):.6f}"
    for row in curve
) + ".")
robust = prov["figures"]["robustness_grid"]
rm = prov["figures"]["rm_data_curve"]
print(
    f"- **Provenance.** Eval run ids `{robust['run_ids'][0]}` and `{robust['run_ids'][1]}`; "
    f"eval manifest hashes `{robust['eval_manifest_sha256'][0]}` and "
    f"`{robust['eval_manifest_sha256'][1]}`; RM manifest `{rm['reward_manifest_sha256']}`; "
    f"figure manifest `{sha256_file(root / 'figures/manifest.json')}`; Hub repo "
    f"`congthanh991/tinyfables-13m-base-noaug` revision `{revision}`."
)
PY
```

Expected: one complete `### Implementation (issue 10, Track B)` Markdown block containing the measured 2×3 grid, perplexities, held-out statement, four curve points, and immutable provenance.

- [ ] **Step G.4: Apply the measured block and correct the old prose SHA transcription**

Use `apply_patch` to insert the exact Step G.3 stdout immediately after the Track A subsection in `docs/design.md`. In the existing Issue 05 Track B paragraph, replace the two hand-transcribed hashes with the actual Hub-manifest values:

```text
checkpoint sha 0dbd8f125a9bf12578e4b5246eebb0bab51b93c9fa169f78537bb4db95065e0a
tokenizer sha 9d53402d4c4dd94d0d7bf133ea8073cf9664103e51b08b3a8c813a00ec625dfb
```

Do not alter the historical Issue 05 metrics; this correction only makes their provenance agree with the immutable Base Model manifest.

- [ ] **Step G.5: Check Issue 10 acceptance criteria and queue status**

After the Hub round trip and design update are true, use `apply_patch` to change all four Issue 10 acceptance boxes in `docs/issues/10-committed-ablation-figures.md` from `- [ ]` to `- [x]`, and change only the Issue 10 row in `docs/issues/README.md` from `☐` to `☑`.

- [ ] **Step G.6: Run final tests and inspect documentation diff**

```bash
rtk .venv/bin/pytest -q
rtk git diff --check
rtk git diff -- docs/design.md docs/issues/10-committed-ablation-figures.md docs/issues/README.md
```

Expected: all non-network tests PASS; `git diff --check` is silent; the design table exactly matches `robustness_grid.csv`; the held-out sentence exactly matches `figure_provenance.json`; Issue 10 alone is newly ticked.

- [ ] **Step G.7: Commit and push closeout**

```bash
rtk git add docs/design.md docs/issues/10-committed-ablation-figures.md docs/issues/README.md
rtk git commit -m "docs(results): record issue-10 ablation and figures"
rtk git push
```

## Done Means

- `congthanh991/tinyfables-13m-base-noaug` round-trips into a loadable 14,186,496-parameter model and contains every final pretrain artifact, full no-aug prep artifacts, both complete eval runs, both PNG/CSV figure pairs, figure provenance, summaries, and manifests.
- The checkpoint repo contains a complete resumable checkpoint whenever a multi-session handoff occurred.
- The no-aug prep config differs from the augmented prep config only in coverage; actual pretrain scientific fields match the augmented Base Model manifest; actual eval configs differ only in checkpoint.
- The robustness table has exactly six rows and publishes adherence plus Moral delivery for canonical, seen-template, and held-out-template prompts for both models, with perplexity as context.
- The held-out column is explicitly interpreted as the five-template unseen-phrasing result, including a negative or mixed result without cherry-picking.
- The RM figure contains all four Issue 07 points pulled from immutable revision `58865e69ccbe1e90b3816068cabca410f28e43aa`, and its CSV preserves requested and actual pair counts.
- Every figure is traceable to run IDs and full upstream manifest hashes in the CSV/provenance/summary; the PNG carries abbreviated trace IDs in its footer.
- `docs/design.md` contains the measured 2×3 grid, explicit held-out statement, RM curve, and Hub provenance; Issue 10's acceptance boxes and queue row are checked; the offline suite passes.
