# Real Base Model on free Colab (Issue 04) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scale the walking-skeleton pretrain to the real ~14M-param Base Model on free-tier Colab (T4): a throughput-benchmark gate, fp16 mixed precision, periodic checkpoints (optimizer + scaler state) that survive session death via exact resume, streamed metrics, and Hub push of the model + tokenizer + loss curve.

**Architecture:** Two tracks. **Track A** (Tasks 1–10, subagent-driven, all testable offline at toy scale) adds four small library modules (`metrics.py`, `checkpoint.py`, `hub.py`, and a new `benchmark` stage) and threads AMP + periodic checkpointing + metrics into the existing `pretrain` stage — every addition is behind a config flag that defaults to the current behavior, so the toy suite stays green. **Track B** (operations, NOT delegatable) drives the real Colab session over the colab-mcp browser connection: install the repo, authenticate, run `tokenizer_full → prep_full → benchmark_full → pretrain_full`, watch the benchmark gate, launch the multi-hour run, and survive session deaths by re-running the same command (auto-resume from the Drive-mounted checkpoint dir).

**Tech Stack:** Python 3.10+, PyTorch 2.12 (`torch.amp` autocast + GradScaler), transformers 5.x (`PreTrainedModel` wrapper), tokenizers, huggingface_hub 1.x (push), trackio (optional live dashboard), matplotlib (optional loss-curve PNG), pytest. Local dev in `.venv` (`.venv/bin/pytest`, `.venv/bin/python`).

## Global Constraints

- **Free Colab (T4 16GB), sessions die** — checkpoint-resume is a first-class requirement; fp16; ~250M-token data budget. Resume must lose at most `ckpt_every` steps.
- **Measure, then commit** — the benchmark stage records measured T4 tokens/sec and a go/no-go decision *before* the multi-hour pretrain (the week-1 gate). Every operational number (measured tokens/sec, chosen `ckpt_every`, any size change) is recorded in `docs/design.md` immediately.
- **Lazy import registry stays intact** — the stage `REGISTRY` resolves module paths by string at dispatch time; new library modules (`metrics.py`, `hub.py`) MUST NOT import torch/trackio/huggingface_hub at module top level (import them lazily inside functions). Heavy stage modules (`benchmark.py`, `pretrain.py`) may import torch at top level since they load only when dispatched.
- **Tests stay offline by default** — no test imports trackio, and no test touches the network. `pyproject.toml` marks `network` tests deselected (`addopts = "-m 'not network'"`); do not add network tests. trackio and matplotlib are OPTIONAL (`.[train]` extra) and absent locally — code must degrade to a no-op / CSV-only path when they are missing.
- **Manifest-written-last is the completion marker** — `stage.write_manifest` writes `manifest.json` LAST; new stages (`benchmark`) preserve this. Periodic training checkpoints use the SAME convention with their own marker file `checkpoint_state.json` written last.
- **Augmentation ON for the main Base Model** — `prep_full.yaml` must set `paraphrase_coverage` in the design range 10–20% with `paraphrase_bank: configs/paraphrases.yaml`, or the issue-10 ablation collapses (the no-aug sibling would equal the main model). Chosen value: **0.15** (mid-range, matches the toy).
- **Model geometry is fixed** (design.md Model table): 6 layers / d_model 384 / 6 heads, ctx 1024, batch 16, 20k steps. `pretrain_full.yaml` already fixes geometry; this issue adds AMP/checkpoint/metrics knobs only — do not change geometry.
- **Element values preserved verbatim** — no change to prompt rendering; this issue does not touch `prompts.py`/`paraphrases.py` logic.

---

## File Structure

**New library modules (import-light, offline-testable):**
- `src/tinyfables/metrics.py` — `MetricsLogger` (lazy trackio, CSV mirror, no-op fallback) + `plot_loss_curve` (matplotlib optional).
- `src/tinyfables/checkpoint.py` — periodic-checkpoint save/find/load/prune with a completion-marker convention (model safetensors + optimizer + scaler + step).
- `src/tinyfables/hub.py` — Hub push utilities (model / tokenizer / checkpoint dir), injectable `api` so tests never touch the network.

**New stage:**
- `src/tinyfables/stages/benchmark.py` — throughput benchmark (synthetic batches at real geometry, tokens/sec, epoch estimate, go/no-go).

**Modified:**
- `src/tinyfables/config.py` — add `BenchmarkConfig`; extend `PretrainConfig` with AMP/checkpoint/metrics knobs (all defaulted to current behavior).
- `src/tinyfables/stages/__init__.py` — register `benchmark`.
- `src/tinyfables/stages/pretrain.py` — AMP (Task 7), periodic checkpoint + auto-resume (Task 8), metrics + loss curve (Task 9).
- `src/tinyfables/cli.py` — `push` subcommand (Task 4).
- `configs/prep_full.yaml` — augmentation knobs (Task 1).
- `configs/benchmark_full.yaml` (new) — real benchmark config (Task 6).
- `configs/pretrain_full.yaml` — AMP/checkpoint/metrics knobs (Task 10).
- `pyproject.toml` — `huggingface_hub` core dep (Task 4); `train` optional extra `trackio`+`matplotlib` (Task 10).
- `docs/design.md` — "Implementation (issue 04)" notes (Task 10) + operational numbers (Track B).
- `docs/issues/README.md` — tick issue 04 (end of Track B).

**New tests:** `tests/test_full_configs.py`, `tests/test_metrics.py`, `tests/test_checkpoint.py`, `tests/test_hub.py`, `tests/test_benchmark_stage.py`; additions to `tests/test_config.py`, `tests/test_pretrain_stage.py`.

**Branch:** create `feat/issue-04-real-base-model` off `main` (via `superpowers:using-git-worktrees` if executing in a worktree). All Track A tasks commit to this branch; merge `--no-ff` to `main` before Track B.

---

### Task 1: Lock paraphrase augmentation into the real prep config

**Files:**
- Modify: `configs/prep_full.yaml`
- Test: `tests/test_full_configs.py` (create)

**Interfaces:**
- Consumes: `tinyfables.config.load_config`, `PrepConfig`; `tinyfables.paraphrases.load_bank`.
- Produces: `configs/prep_full.yaml` parses to a `PrepConfig` with `paraphrase_bank="configs/paraphrases.yaml"` and `0.10 <= paraphrase_coverage <= 0.20`. Later config tasks add cases to this same test file.

- [ ] **Step 1: Write the failing test**

Create `tests/test_full_configs.py`:

```python
from pathlib import Path

from tinyfables.config import PrepConfig, load_config
from tinyfables.paraphrases import load_bank

REPO = Path(__file__).resolve().parents[1]


def test_prep_full_trains_with_augmentation():
    cfg = load_config(REPO / "configs" / "prep_full.yaml", PrepConfig)
    # The main Base Model MUST train with augmentation, or issue-10's no-aug
    # ablation sibling would be identical to the main model.
    assert cfg.paraphrase_bank == "configs/paraphrases.yaml"
    assert 0.10 <= cfg.paraphrase_coverage <= 0.20  # design range
    bank = load_bank(REPO / cfg.paraphrase_bank)
    assert bank.seen_templates  # bank actually usable for training
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_full_configs.py -v`
Expected: FAIL — `assert cfg.paraphrase_bank == "configs/paraphrases.yaml"` (currently `None`).

- [ ] **Step 3: Add the augmentation knobs to `configs/prep_full.yaml`**

The file becomes:

```yaml
# Real run (issue 04, Colab): ~450k fables ≈ 250M tokens (design.md data budget).
# Augmentation ON (design range 10–20%; 0.15 matches the toy) — the main Base
# Model trains WITH Paraphrased Prompts so issue-10's no-aug sibling is a real
# ablation, not a duplicate. families.bin/family_counts prove held-out leakage = 0.
source:
  hf_dataset: klusai/ds-tf1-en-3m
  hf_split: train
  max_rows: 450000
tokenizer_dir: runs/tokenizer_full
window: 1024
seed: 0
paraphrase_bank: configs/paraphrases.yaml
paraphrase_coverage: 0.15
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_full_configs.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add configs/prep_full.yaml tests/test_full_configs.py
git commit -m "config: train real Base Model with 0.15 paraphrase coverage (issue 04)"
```

---

### Task 2: `MetricsLogger` + loss-curve plotting

**Files:**
- Create: `src/tinyfables/metrics.py`
- Test: `tests/test_metrics.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks. trackio and matplotlib are OPTIONAL and imported lazily inside functions.
- Produces:
  - `class MetricsLogger` with `__init__(self, csv_path, project=None, run_name=None, space_id=None, config=None)`, `log(self, step: int, loss: float, lr: float, tokens_per_sec: float) -> None`, `alert(self, title: str, text: str, level: str = "WARN") -> None`, `finish(self) -> None`. Always writes/append to `csv_path` (header `step,loss,lr,tokens_per_sec`); mirrors to trackio only when `project` is set AND trackio importable.
  - `plot_loss_curve(csv_path, png_path) -> Path | None` — renders a PNG if matplotlib is importable, else returns `None`.
  - Used by `pretrain.run` in Task 9.

- [ ] **Step 1: Write the failing test**

Create `tests/test_metrics.py`:

```python
import csv
from pathlib import Path

from tinyfables.metrics import MetricsLogger, plot_loss_curve


def test_csv_mirror_written_without_trackio(tmp_path):
    csv_path = tmp_path / "loss_log.csv"
    logger = MetricsLogger(csv_path)  # project=None -> CSV-only, no trackio import
    logger.log(step=0, loss=2.5, lr=1e-4, tokens_per_sec=1000.0)
    logger.log(step=1, loss=2.0, lr=2e-4, tokens_per_sec=1100.0)
    logger.finish()
    rows = list(csv.DictReader(open(csv_path)))
    assert [r["step"] for r in rows] == ["0", "1"]
    assert rows[0]["loss"].startswith("2.5")
    assert set(rows[0]) == {"step", "loss", "lr", "tokens_per_sec"}


def test_project_set_but_trackio_absent_degrades_to_csv(tmp_path):
    # trackio is NOT installed locally; a configured project must not crash.
    logger = MetricsLogger(tmp_path / "l.csv", project="ignored", run_name="r")
    logger.log(step=0, loss=1.0, lr=1e-4, tokens_per_sec=500.0)
    logger.alert("t", "x", level="ERROR")  # no trackio -> prints, no raise
    logger.finish()
    assert (tmp_path / "l.csv").exists()


def test_csv_appends_across_sessions(tmp_path):
    csv_path = tmp_path / "loss_log.csv"
    MetricsLogger(csv_path).log(step=0, loss=3.0, lr=1e-4, tokens_per_sec=1.0)
    MetricsLogger(csv_path).log(step=1, loss=2.9, lr=1e-4, tokens_per_sec=1.0)  # reopen
    rows = list(csv.DictReader(open(csv_path)))
    assert [r["step"] for r in rows] == ["0", "1"]  # header written once, rows appended


def test_plot_loss_curve_optional(tmp_path):
    csv_path = tmp_path / "loss_log.csv"
    logger = MetricsLogger(csv_path)
    for s in range(3):
        logger.log(step=s, loss=3.0 - s, lr=1e-4, tokens_per_sec=1.0)
    png = tmp_path / "loss_curve.png"
    res = plot_loss_curve(csv_path, png)
    # matplotlib may be absent (returns None) or present (PNG exists) — both OK.
    assert res is None or Path(res).exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tinyfables.metrics'`.

- [ ] **Step 3: Write `src/tinyfables/metrics.py`**

```python
"""Experiment metrics: a thin tracking wrapper. trackio is OPTIONAL and imported
lazily, so the package imports and the test suite run without it. When no project
is configured (or trackio is not installed / init fails), logging is a no-op that
still mirrors every metric to a durable local CSV — the CSV is the artifact the
report needs; trackio, when present, is the live cross-session dashboard.

The CSV is append-safe across sessions (header written once): a Colab session
death and re-run keeps extending the same loss log."""

from __future__ import annotations

import csv
from pathlib import Path

_FIELDS = ["step", "loss", "lr", "tokens_per_sec"]


class MetricsLogger:
    def __init__(self, csv_path, project=None, run_name=None, space_id=None, config=None):
        self.csv_path = Path(csv_path)
        self._trackio = None
        if project:
            try:
                import trackio  # lazy: optional dependency

                trackio.init(project=project, name=run_name, space_id=space_id, config=config or {})
                self._trackio = trackio
            except Exception as e:  # trackio absent or init failed -> CSV-only
                print(f"[tinyfables] trackio unavailable ({e}); logging to CSV only")
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(_FIELDS)

    def log(self, step: int, loss: float, lr: float, tokens_per_sec: float) -> None:
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([step, f"{loss:.6f}", f"{lr:.8f}", f"{tokens_per_sec:.2f}"])
        if self._trackio is not None:
            self._trackio.log({"loss": loss, "lr": lr, "tokens_per_sec": tokens_per_sec, "step": step})

    def alert(self, title: str, text: str, level: str = "WARN") -> None:
        if self._trackio is not None:
            lvl = getattr(self._trackio.AlertLevel, level, None)
            self._trackio.alert(title=title, text=text, level=lvl)
        else:
            print(f"[alert:{level}] {title}: {text}")

    def finish(self) -> None:
        if self._trackio is not None:
            self._trackio.finish()


def plot_loss_curve(csv_path, png_path):
    """Render a loss-vs-step PNG from the CSV. Returns the PNG path, or None if
    matplotlib is unavailable (the CSV remains the durable artifact)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    steps, losses = [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["loss"] and row["loss"] != "nan":
                steps.append(int(row["step"]))
                losses.append(float(row["loss"]))
    fig, ax = plt.subplots()
    ax.plot(steps, losses)
    ax.set_xlabel("step")
    ax.set_ylabel("fable-token loss")
    ax.set_title("Pretrain loss")
    Path(png_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return Path(png_path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_metrics.py -v`
Expected: PASS (4 passed). `test_plot_loss_curve_optional` passes with `res is None` (matplotlib absent locally).

- [ ] **Step 5: Commit**

```bash
git add src/tinyfables/metrics.py tests/test_metrics.py
git commit -m "feat: MetricsLogger (lazy trackio + CSV mirror) and loss-curve plot (issue 04)"
```

---

### Task 3: Periodic checkpoint save / find / load / prune

**Files:**
- Create: `src/tinyfables/checkpoint.py`
- Test: `tests/test_checkpoint.py` (create)

**Interfaces:**
- Consumes: `tinyfables.model.GPT`, `GPTConfig`; torch (`torch.amp.GradScaler`, `torch.optim.AdamW`).
- Produces:
  - `save_checkpoint(model, optimizer, scaler, step: int, ckpt_dir) -> Path` — writes `ckpt_dir/step_{step}/` containing `model.safetensors` (+ `config.json`, `generation_config.json`) via `model.save_pretrained`, `optimizer.pt` (`{"optimizer", "scaler", "step"}`, `scaler` may be `None`), and `checkpoint_state.json` written LAST.
  - `find_latest_checkpoint(ckpt_dir) -> Path | None` — highest-`step` COMPLETE checkpoint (has the marker + optimizer.pt + model.safetensors), else `None`.
  - `load_checkpoint(ckpt_path, model_cls, device) -> tuple[model, dict]` — returns `(model_on_device, state)` where `state = {"optimizer", "scaler", "step"}`.
  - `prune_checkpoints(ckpt_dir, keep_last_k: int) -> None` — delete all but the newest `keep_last_k` complete checkpoints.
  - Used by `pretrain.run` in Task 8.

- [ ] **Step 1: Write the failing test**

Create `tests/test_checkpoint.py`:

```python
import torch

from tinyfables.checkpoint import (
    find_latest_checkpoint,
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
)
from tinyfables.model import GPT, GPTConfig


def _model_opt_scaler():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=48, n_layer=2, n_head=2, d_model=64, n_ctx=64))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    return model, opt, scaler


def test_save_then_find_latest(tmp_path):
    model, opt, scaler = _model_opt_scaler()
    save_checkpoint(model, opt, scaler, step=2, ckpt_dir=tmp_path)
    save_checkpoint(model, opt, scaler, step=4, ckpt_dir=tmp_path)
    latest = find_latest_checkpoint(tmp_path)
    assert latest is not None and latest.name == "step_4"
    assert (latest / "checkpoint_state.json").exists()
    assert (latest / "optimizer.pt").exists()
    assert (latest / "model.safetensors").exists()


def test_incomplete_checkpoint_ignored(tmp_path):
    model, opt, scaler = _model_opt_scaler()
    save_checkpoint(model, opt, scaler, step=2, ckpt_dir=tmp_path)
    d = save_checkpoint(model, opt, scaler, step=4, ckpt_dir=tmp_path)
    (d / "checkpoint_state.json").unlink()  # simulate death mid-write (marker last)
    latest = find_latest_checkpoint(tmp_path)
    assert latest.name == "step_2"  # the incomplete step_4 is not selected


def test_find_returns_none_when_empty(tmp_path):
    assert find_latest_checkpoint(tmp_path) is None
    assert find_latest_checkpoint(tmp_path / "missing") is None


def test_load_restores_step_and_optimizer(tmp_path):
    model, opt, scaler = _model_opt_scaler()
    save_checkpoint(model, opt, scaler, step=7, ckpt_dir=tmp_path)
    loaded, state = load_checkpoint(tmp_path / "step_7", GPT, device="cpu")
    assert state["step"] == 7
    assert "optimizer" in state and state["scaler"] is None
    assert isinstance(loaded, GPT)


def test_prune_keeps_last_k(tmp_path):
    model, opt, scaler = _model_opt_scaler()
    for s in (2, 4, 6):
        save_checkpoint(model, opt, scaler, step=s, ckpt_dir=tmp_path)
    prune_checkpoints(tmp_path, keep_last_k=1)
    remaining = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())
    assert remaining == ["step_6"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_checkpoint.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tinyfables.checkpoint'`.

- [ ] **Step 3: Write `src/tinyfables/checkpoint.py`**

```python
"""Periodic training checkpoints for session-death resume. A checkpoint dir holds
the model (safetensors via save_pretrained), optimizer + AMP scaler + step
(optimizer.pt), and a completion marker (checkpoint_state.json) written LAST —
mirroring the stage-manifest convention, so a checkpoint interrupted mid-write is
never mistaken for complete. On Colab the parent ckpt_dir is Drive-mounted, so
checkpoints outlive the runtime and the same command resumes after a session death."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch

_MARKER = "checkpoint_state.json"
_PREFIX = "step_"


def save_checkpoint(model, optimizer, scaler, step: int, ckpt_dir) -> Path:
    d = Path(ckpt_dir) / f"{_PREFIX}{step}"
    d.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(d)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "step": step,
        },
        d / "optimizer.pt",
    )
    (d / _MARKER).write_text(json.dumps({"step": step}) + "\n")  # LAST = completion marker
    return d


def _complete(d: Path) -> bool:
    return (
        (d / _MARKER).exists()
        and (d / "optimizer.pt").exists()
        and (d / "model.safetensors").exists()
    )


def _complete_checkpoints(ckpt_dir) -> list[tuple[int, Path]]:
    root = Path(ckpt_dir)
    if not root.exists():
        return []
    out: list[tuple[int, Path]] = []
    for d in root.iterdir():
        if d.is_dir() and d.name.startswith(_PREFIX) and _complete(d):
            try:
                out.append((int(d.name[len(_PREFIX) :]), d))
            except ValueError:
                pass
    return sorted(out, key=lambda t: t[0])


def find_latest_checkpoint(ckpt_dir) -> Path | None:
    ckpts = _complete_checkpoints(ckpt_dir)
    return ckpts[-1][1] if ckpts else None


def load_checkpoint(ckpt_path, model_cls, device):
    d = Path(ckpt_path)
    model = model_cls.from_pretrained(d).to(device)
    state = torch.load(d / "optimizer.pt", map_location=device)
    return model, state


def prune_checkpoints(ckpt_dir, keep_last_k: int) -> None:
    if keep_last_k <= 0:
        return
    ckpts = _complete_checkpoints(ckpt_dir)
    for _step, d in ckpts[:-keep_last_k]:
        shutil.rmtree(d, ignore_errors=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_checkpoint.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/tinyfables/checkpoint.py tests/test_checkpoint.py
git commit -m "feat: periodic checkpoint save/find/load/prune with completion marker (issue 04)"
```

---

### Task 4: Hub push utilities + `push` CLI subcommand

**Files:**
- Create: `src/tinyfables/hub.py`
- Modify: `src/tinyfables/cli.py`
- Modify: `pyproject.toml` (add `huggingface_hub` to core deps)
- Test: `tests/test_hub.py` (create)

**Interfaces:**
- Consumes: `huggingface_hub.HfApi` (imported lazily inside `_default_api`).
- Produces:
  - `push_tokenizer_to_hub(tokenizer_dir, repo_id, *, private=True, api=None) -> str`
  - `push_model_to_hub(model_dir, repo_id, *, private=True, api=None) -> str` (uploads the folder, ignoring `optimizer.pt`/`checkpoint_state.json` so a checkpoint dir can double as the source)
  - `upload_checkpoint_dir(ckpt_dir, repo_id, *, private=True, api=None) -> str`
  - All accept an injectable `api` (default lazy `HfApi()`) so tests exercise the call sequence without network.
  - CLI: `python -m tinyfables push --kind {model,tokenizer,checkpoint} --path DIR --repo REPO [--public]`.
  - Used by `pretrain.run` (Task 8, optional checkpoint mirror) and by Track B (final push).

- [ ] **Step 1: Write the failing test**

Create `tests/test_hub.py`:

```python
from tinyfables import hub
from tinyfables.cli import main


class FakeApi:
    def __init__(self):
        self.calls = []

    def create_repo(self, **kw):
        self.calls.append(("create_repo", kw))

    def upload_file(self, **kw):
        self.calls.append(("upload_file", kw))

    def upload_folder(self, **kw):
        self.calls.append(("upload_folder", kw))


def test_push_model_creates_repo_and_uploads_folder(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"x")
    api = FakeApi()
    repo = hub.push_model_to_hub(tmp_path, "user/tinyfables-13m-base", api=api)
    assert repo == "user/tinyfables-13m-base"
    names = [c[0] for c in api.calls]
    assert names == ["create_repo", "upload_folder"]
    create_kw = api.calls[0][1]
    assert create_kw["repo_id"] == "user/tinyfables-13m-base" and create_kw["private"] is True
    folder_kw = api.calls[1][1]
    assert "optimizer.pt" in folder_kw["ignore_patterns"]


def test_push_tokenizer_uploads_tokenizer_json(tmp_path):
    (tmp_path / "tokenizer.json").write_text("{}")
    api = FakeApi()
    hub.push_tokenizer_to_hub(tmp_path, "user/tinyfables-tokenizer", private=False, api=api)
    up = [c for c in api.calls if c[0] == "upload_file"][0][1]
    assert up["path_in_repo"] == "tokenizer.json"
    assert api.calls[0][1]["private"] is False


def test_cli_push_dispatches(tmp_path, monkeypatch, capsys):
    recorded = {}

    def fake_push_model(path, repo, *, private=True, api=None):
        recorded["args"] = (str(path), repo, private)
        return repo

    monkeypatch.setattr(hub, "push_model_to_hub", fake_push_model)
    rc = main(["push", "--kind", "model", "--path", str(tmp_path), "--repo", "user/m"])
    assert rc == 0
    assert recorded["args"] == (str(tmp_path), "user/m", True)
    assert "pushed model" in capsys.readouterr().out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_hub.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tinyfables.hub'`.

- [ ] **Step 3: Write `src/tinyfables/hub.py`**

```python
"""Hub push utilities. huggingface_hub is imported lazily (keeps package import
light and offline). Every function takes an injectable `api` so tests exercise the
call sequence without touching the network. Repos default to private (course
artifacts go public deliberately, via --public / private=False)."""

from __future__ import annotations

from pathlib import Path


def _default_api():
    from huggingface_hub import HfApi  # lazy: heavy + network-adjacent

    return HfApi()


def push_tokenizer_to_hub(tokenizer_dir, repo_id, *, private=True, api=None) -> str:
    api = api or _default_api()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(Path(tokenizer_dir) / "tokenizer.json"),
        path_in_repo="tokenizer.json",
        repo_id=repo_id,
        repo_type="model",
    )
    return repo_id


def push_model_to_hub(model_dir, repo_id, *, private=True, api=None) -> str:
    api = api or _default_api()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=["optimizer.pt", "checkpoint_state.json"],
    )
    return repo_id


def upload_checkpoint_dir(ckpt_dir, repo_id, *, private=True, api=None) -> str:
    api = api or _default_api()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(ckpt_dir), repo_id=repo_id, repo_type="model")
    return repo_id
```

- [ ] **Step 4: Add the `push` subcommand to `src/tinyfables/cli.py`**

Add the docstring line, a parser builder, a dispatcher, and wire them into `main`. Insert after `_add_generate_parser`:

```python
def _add_push_parser(sub) -> None:
    push_p = sub.add_parser("push", help="push artifacts to the Hugging Face Hub")
    push_p.add_argument("--kind", choices=["model", "tokenizer", "checkpoint"], required=True)
    push_p.add_argument("--path", required=True, help="local dir to push")
    push_p.add_argument("--repo", required=True, help="target repo id, e.g. user/tinyfables-13m-base")
    push_p.add_argument("--public", action="store_true", help="create a public repo (default private)")
```

Add the dispatcher after `_generate`:

```python
def _push(args) -> int:
    # hub imported lazily so `import tinyfables.cli` stays light.
    from tinyfables import hub

    fn = {
        "model": hub.push_model_to_hub,
        "tokenizer": hub.push_tokenizer_to_hub,
        "checkpoint": hub.upload_checkpoint_dir,
    }[args.kind]
    repo = fn(args.path, args.repo, private=not args.public)
    print(f"[tinyfables] pushed {args.kind} -> {repo}")
    return 0
```

In `main`, register the parser and route the command. Change the sub-parser registrations and dispatch tail:

```python
    _add_run_parser(sub)
    _add_generate_parser(sub)
    _add_push_parser(sub)
    args = parser.parse_args(argv)

    if args.cmd == "run":
        return _run_stage(args)
    if args.cmd == "push":
        return _push(args)
    return _generate(args)
```

- [ ] **Step 5: Add `huggingface_hub` to core deps in `pyproject.toml`**

In the `dependencies` list, add the line (we now import it directly):

```toml
    "huggingface_hub>=0.24",
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_hub.py -v`
Expected: PASS (3 passed).

- [ ] **Step 7: Commit**

```bash
git add src/tinyfables/hub.py src/tinyfables/cli.py pyproject.toml tests/test_hub.py
git commit -m "feat: Hub push utilities + 'push' CLI subcommand (issue 04)"
```

---

### Task 5: Config extensions — `BenchmarkConfig` + `PretrainConfig` knobs

**Files:**
- Modify: `src/tinyfables/config.py`
- Test: `tests/test_config.py` (add cases)

**Interfaces:**
- Consumes: existing `load_config`, `_build`, frozen-dataclass + unknown-key-rejection machinery.
- Produces:
  - `BenchmarkConfig` (frozen) with fields: `tokenizer_dir: str`, `n_layer=6`, `n_head=6`, `d_model=384`, `n_ctx=1024`, `batch_size=16`, `window=1024`, `warmup_steps=5`, `measure_steps=20`, `token_budget=250_000_000`, `target_tokens_per_sec=0.0`, `amp=True`, `device="auto"`, `seed=0`.
  - `PretrainConfig` gains (all defaulted to current behavior): `amp: bool = False`, `ckpt_dir: str | None = None`, `ckpt_every: int = 0`, `keep_last_k: int = 2`, `log_every: int = 0`, `trackio_project: str | None = None`, `trackio_space_id: str | None = None`, `run_name: str | None = None`, `ckpt_hub_repo: str | None = None`.
  - The stage `REGISTRY` is NOT touched here (Task 6 registers `benchmark` alongside creating the module, so no dangling entry).
  - Consumed by Tasks 6–10.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py` (append; keep existing imports, add `BenchmarkConfig` to the import line):

```python
def test_benchmark_config_defaults_and_overrides(tmp_path):
    from tinyfables.config import BenchmarkConfig, load_config

    p = tmp_path / "b.yaml"
    p.write_text("tokenizer_dir: runs/tok\nmeasure_steps: 30\ntarget_tokens_per_sec: 8000\n")
    cfg = load_config(p, BenchmarkConfig)
    assert cfg.tokenizer_dir == "runs/tok"
    assert cfg.measure_steps == 30 and cfg.target_tokens_per_sec == 8000
    assert cfg.n_layer == 6 and cfg.window == 1024 and cfg.amp is True  # design defaults


def test_pretrain_config_new_knobs_default_to_current_behavior(tmp_path):
    from tinyfables.config import PretrainConfig, load_config

    p = tmp_path / "p.yaml"
    p.write_text("prep_dir: runs/prep\ntokenizer_dir: runs/tok\n")
    cfg = load_config(p, PretrainConfig)
    assert cfg.amp is False and cfg.ckpt_dir is None and cfg.ckpt_every == 0
    assert cfg.log_every == 0 and cfg.keep_last_k == 2
    assert cfg.trackio_project is None and cfg.run_name is None and cfg.ckpt_hub_repo is None


def test_pretrain_config_accepts_issue04_knobs(tmp_path):
    from tinyfables.config import PretrainConfig, load_config

    p = tmp_path / "p.yaml"
    p.write_text(
        "prep_dir: runs/prep\ntokenizer_dir: runs/tok\n"
        "amp: true\nckpt_dir: /content/drive/MyDrive/ckpt\nckpt_every: 500\n"
        "log_every: 50\ntrackio_project: tinyfables\nrun_name: base-v1\n"
    )
    cfg = load_config(p, PretrainConfig)
    assert cfg.amp is True and cfg.ckpt_every == 500 and cfg.log_every == 50
    assert cfg.ckpt_dir.endswith("/ckpt") and cfg.run_name == "base-v1"


def test_benchmark_config_rejects_unknown_key(tmp_path):
    import pytest

    from tinyfables.config import BenchmarkConfig, load_config

    p = tmp_path / "b.yaml"
    p.write_text("tokenizer_dir: runs/tok\nbogus: 1\n")
    with pytest.raises(KeyError):
        load_config(p, BenchmarkConfig)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_config.py -k "benchmark or new_knobs or issue04" -v`
Expected: FAIL — `ImportError: cannot import name 'BenchmarkConfig'` / missing attributes.

- [ ] **Step 3: Add `BenchmarkConfig` and extend `PretrainConfig` in `src/tinyfables/config.py`**

Add after `PretrainConfig` the new dataclass:

```python
@dataclass(frozen=True)
class BenchmarkConfig:
    tokenizer_dir: str
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    n_ctx: int = 1024
    batch_size: int = 16
    window: int = 1024
    warmup_steps: int = 5
    measure_steps: int = 20
    token_budget: int = 250_000_000
    target_tokens_per_sec: float = 0.0
    amp: bool = True
    device: str = "auto"
    seed: int = 0
```

Extend `PretrainConfig` — append these fields after `seed: int = 0`:

```python
    amp: bool = False
    ckpt_dir: str | None = None
    ckpt_every: int = 0
    keep_last_k: int = 2
    log_every: int = 0
    trackio_project: str | None = None
    trackio_space_id: str | None = None
    run_name: str | None = None
    ckpt_hub_repo: str | None = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: PASS (all existing + 4 new).

- [ ] **Step 5: Commit**

```bash
git add src/tinyfables/config.py tests/test_config.py
git commit -m "feat: BenchmarkConfig + PretrainConfig AMP/checkpoint/metrics knobs (issue 04)"
```

---

### Task 6: Benchmark stage (throughput gate)

**Files:**
- Create: `src/tinyfables/stages/benchmark.py`
- Modify: `src/tinyfables/stages/__init__.py` (register `benchmark`)
- Create: `configs/benchmark_full.yaml`
- Test: `tests/test_benchmark_stage.py` (create)

**Interfaces:**
- Consumes: `BenchmarkConfig`; `GPT`/`GPTConfig`; `tokenizers.Tokenizer` (for vocab_size); `stage.write_manifest`.
- Produces:
  - `run(cfg: BenchmarkConfig, out_dir: Path) -> None` — builds the model at target geometry, times `measure_steps` forward+backward+step iterations on synthetic batches (excluding `warmup_steps`), writes `benchmark_summary.json` (keys: `device`, `amp`, `vocab_size`, `batch_size`, `window`, `measure_steps`, `elapsed_sec`, `tokens_per_second`, `est_seconds_per_epoch`, `token_budget`, `target_tokens_per_sec`, `decision`) then `manifest.json` LAST.
  - `decision` is `"go"` when `target_tokens_per_sec <= 0` OR `tokens_per_second >= target_tokens_per_sec`, else `"revise"`.
  - `REGISTRY["benchmark"] = (BenchmarkConfig, "tinyfables.stages.benchmark")`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_benchmark_stage.py`:

```python
import json
from pathlib import Path

from tinyfables.config import BenchmarkConfig, SourceSpec, TokenizerConfig
from tinyfables.stages import REGISTRY
from tinyfables.stages import benchmark as benchmark_stage
from tinyfables.stages import tokenizer as tokenizer_stage

FIXTURE = str(Path(__file__).parent / "fixtures" / "tiny_corpus.jsonl")


def _tok_dir(tmp_path):
    out = tmp_path / "tok"
    tokenizer_stage.run(TokenizerConfig(source=SourceSpec(jsonl_path=FIXTURE), vocab_size=512, seed=0), out)
    return out


def _cfg(tok_dir, **kw):
    base = dict(
        tokenizer_dir=str(tok_dir), n_layer=2, n_head=2, d_model=64, n_ctx=64,
        batch_size=2, window=64, warmup_steps=1, measure_steps=2, amp=False, device="cpu",
    )
    base.update(kw)
    return BenchmarkConfig(**base)


def test_benchmark_is_registered():
    assert REGISTRY["benchmark"] == (BenchmarkConfig, "tinyfables.stages.benchmark")


def test_benchmark_writes_summary_and_manifest(tmp_path):
    tok_dir = _tok_dir(tmp_path)
    out = tmp_path / "bench"
    benchmark_stage.run(_cfg(tok_dir), out)
    summary = json.loads((out / "benchmark_summary.json").read_text())
    assert summary["tokens_per_second"] > 0
    assert summary["decision"] in {"go", "revise"}
    assert summary["est_seconds_per_epoch"] is not None
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["stage"] == "benchmark"
    assert "tokenizer.json" in manifest["inputs"]


def test_decision_go_when_no_target(tmp_path):
    tok_dir = _tok_dir(tmp_path)
    out = tmp_path / "b2"
    benchmark_stage.run(_cfg(tok_dir, target_tokens_per_sec=0.0), out)
    assert json.loads((out / "benchmark_summary.json").read_text())["decision"] == "go"


def test_decision_revise_when_target_unreachable(tmp_path):
    tok_dir = _tok_dir(tmp_path)
    out = tmp_path / "b3"
    benchmark_stage.run(_cfg(tok_dir, target_tokens_per_sec=1e12), out)
    assert json.loads((out / "benchmark_summary.json").read_text())["decision"] == "revise"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_benchmark_stage.py -v`
Expected: FAIL — `KeyError: 'benchmark'` (registry) / `ModuleNotFoundError: ...stages.benchmark`.

- [ ] **Step 3: Write `src/tinyfables/stages/benchmark.py`**

```python
"""Benchmark stage: measure real T4 throughput (tokens/sec) at the TARGET model
geometry BEFORE committing to the multi-hour pretrain — the week-1 size gate
(design.md: measure, then commit). Runs synthetic batches (real vocab from the
tokenizer, real batch×window shape, real AMP path), excludes warmup, and records
tokens/sec, an epoch-time estimate for the token budget, and a go/no-go decision.

Synthetic batches (not prep shards) keep this stage a pure compute probe: GPU
matmul throughput dominates; the memmap read is negligible. It depends only on the
tokenizer (vocab size drives the embedding/head cost), so it can run the moment
the tokenizer exists."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from tinyfables.config import BenchmarkConfig
from tinyfables.model import GPT, GPTConfig
from tinyfables.stage import write_manifest


def _resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run(cfg: BenchmarkConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(cfg.device)
    tok = Tokenizer.from_file(str(Path(cfg.tokenizer_dir) / "tokenizer.json"))
    vocab_size = tok.get_vocab_size()

    model = GPT(
        GPTConfig(
            vocab_size=vocab_size,
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            d_model=cfg.d_model,
            n_ctx=cfg.n_ctx,
        )
    ).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    use_amp = cfg.amp and device == "cuda"
    autocast_device = "cuda" if device == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(device, enabled=use_amp)
    gen = torch.Generator().manual_seed(cfg.seed)

    def make_batch() -> torch.Tensor:
        ids = torch.randint(0, vocab_size, (cfg.batch_size, cfg.window), generator=gen)
        return ids.to(device)

    def one_step(ids: torch.Tensor) -> None:
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=autocast_device, dtype=torch.float16, enabled=use_amp):
            out = model(input_ids=ids, labels=ids)
        scaler.scale(out.loss).backward()
        scaler.step(optimizer)
        scaler.update()

    for _ in range(cfg.warmup_steps):
        one_step(make_batch())
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(cfg.measure_steps):
        one_step(make_batch())
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tokens = cfg.measure_steps * cfg.batch_size * cfg.window
    tps = tokens / elapsed if elapsed > 0 else 0.0
    decision = "go" if (cfg.target_tokens_per_sec <= 0 or tps >= cfg.target_tokens_per_sec) else "revise"

    summary = {
        "device": device,
        "amp": use_amp,
        "vocab_size": vocab_size,
        "batch_size": cfg.batch_size,
        "window": cfg.window,
        "measure_steps": cfg.measure_steps,
        "elapsed_sec": round(elapsed, 4),
        "tokens_per_second": round(tps, 2),
        "est_seconds_per_epoch": round(cfg.token_budget / tps, 1) if tps > 0 else None,
        "token_budget": cfg.token_budget,
        "target_tokens_per_sec": cfg.target_tokens_per_sec,
        "decision": decision,
    }
    summary_path = out_dir / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    write_manifest(
        out_dir,
        "benchmark",
        cfg,
        [summary_path],
        inputs={"tokenizer.json": Path(cfg.tokenizer_dir) / "tokenizer.json"},
    )
```

- [ ] **Step 4: Register the stage in `src/tinyfables/stages/__init__.py`**

Replace the file body's import + registry with:

```python
from tinyfables.config import BenchmarkConfig, PrepConfig, PretrainConfig, TokenizerConfig

REGISTRY: dict[str, tuple[type, str]] = {
    "tokenizer": (TokenizerConfig, "tinyfables.stages.tokenizer"),
    "prep": (PrepConfig, "tinyfables.stages.prep"),
    "benchmark": (BenchmarkConfig, "tinyfables.stages.benchmark"),
    "pretrain": (PretrainConfig, "tinyfables.stages.pretrain"),
}
```

- [ ] **Step 5: Create `configs/benchmark_full.yaml`**

```yaml
# Real T4 benchmark (issue 04, Colab): the week-1 size gate. Times the target
# geometry (6/384/6, ctx 1024, batch 16) with fp16 AMP on synthetic batches.
# target_tokens_per_sec is a floor for the "go" decision: at ~250M tokens/epoch
# and an est. 1–2h/epoch budget, ~35k tok/s clears a 2h epoch. Set from the design
# estimate; revise after the first real reading (record the number in design.md).
tokenizer_dir: runs/tokenizer_full
n_layer: 6
n_head: 6
d_model: 384
n_ctx: 1024
batch_size: 16
window: 1024
warmup_steps: 5
measure_steps: 30
token_budget: 250000000
target_tokens_per_sec: 35000
amp: true
device: auto
seed: 0
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_benchmark_stage.py -v`
Expected: PASS (4 passed).

- [ ] **Step 7: Commit**

```bash
git add src/tinyfables/stages/benchmark.py src/tinyfables/stages/__init__.py configs/benchmark_full.yaml tests/test_benchmark_stage.py
git commit -m "feat: benchmark stage — measured tokens/sec go/no-go gate (issue 04)"
```

---

### Task 7: Pretrain — fp16 mixed precision (AMP)

**Files:**
- Modify: `src/tinyfables/stages/pretrain.py`
- Test: `tests/test_pretrain_stage.py` (add cases)

**Interfaces:**
- Consumes: existing `run`, `build_model`, `lr_at`, `get_batch`, `_resolve_device`; `torch.amp.GradScaler`/`autocast`.
- Produces: `run` now trains through a `GradScaler` (`use_amp = cfg.amp and device == "cuda"`); autocast wraps the forward; `clip_grad_norm_` runs after `scaler.unscale_`; the final `optimizer.pt` dict gains a `"scaler"` key (`None` when AMP off). Resume-from loads the scaler if present.
- Guarantees preserved: CPU fp32 path (AMP off) is bit-identical to before — the existing `test_resume_equality_within_tolerance` and `test_overfit_one_batch_drives_loss_down` still pass. Determinism/checkpoint additions land in Task 8.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pretrain_stage.py`:

```python
def test_amp_flag_is_safe_noop_on_cpu(toy_shards, tmp_path):
    # cfg.amp=True on CPU must not crash (GradScaler is disabled off-CUDA) and
    # must still produce a usable checkpoint with finite loss.
    out = tmp_path / "amp_cpu"
    pretrain_stage.run(toy_cfg(toy_shards, steps=4, amp=True), out)
    summary = json.loads((out / "pretrain_summary.json").read_text())
    assert summary["steps"] == 4
    import math
    assert not math.isnan(summary["final_loss"])


def test_final_optimizer_pt_has_scaler_key(toy_shards, tmp_path):
    out = tmp_path / "sc"
    pretrain_stage.run(toy_cfg(toy_shards, steps=2), out)
    state = torch.load(out / "optimizer.pt", map_location="cpu")
    assert "scaler" in state  # None on CPU (AMP off), but the key is always present
    assert state["step"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_pretrain_stage.py -k "amp or scaler" -v`
Expected: FAIL — `KeyError: 'amp'` (config override unknown) is NOT the failure (Task 5 added `amp`); it fails on `assert "scaler" in state` (current save writes only `optimizer`+`step`).

- [ ] **Step 3: Add AMP to `src/tinyfables/stages/pretrain.py`**

Insert scaler construction before the resume/init block (after `if n_windows < cfg.batch_size: ...`):

```python
    use_amp = cfg.amp and device == "cuda"
    autocast_device = "cuda" if device == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(device, enabled=use_amp)
```

In the `if cfg.resume_from:` branch, after `optimizer.load_state_dict(state["optimizer"])`, add:

```python
        if state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
```

Replace the training-step body inside the loop:

```python
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=autocast_device, dtype=torch.float16, enabled=use_amp):
            out = model(input_ids=input_ids, labels=labels)
        scaler.scale(out.loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        last_loss = out.loss.item()
```

Replace the final `torch.save(...)`:

```python
    torch.save(
        {"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict() if use_amp else None, "step": cfg.steps},
        out_dir / "optimizer.pt",
    )
```

Add `"amp": use_amp` to the `summary_out` dict (after `"device": device,`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_pretrain_stage.py -v`
Expected: PASS — new AMP tests pass AND `test_resume_equality_within_tolerance` (bit-exact CPU resume) still passes (disabled scaler is a math no-op).

- [ ] **Step 5: Run the toy end-to-end chain to confirm no regression**

Run: `.venv/bin/pytest tests/test_toy_chain.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/tinyfables/stages/pretrain.py tests/test_pretrain_stage.py
git commit -m "feat: fp16 AMP (autocast + GradScaler) in pretrain, scaler in checkpoint (issue 04)"
```

---

### Task 8: Pretrain — periodic checkpoint + auto-resume across session death

**Files:**
- Modify: `src/tinyfables/stages/pretrain.py`
- Test: `tests/test_pretrain_stage.py` (add cases)

**Interfaces:**
- Consumes: `checkpoint.save_checkpoint`, `find_latest_checkpoint`, `load_checkpoint`, `prune_checkpoints`; `hub.upload_checkpoint_dir` (lazy, best-effort mirror); `PretrainConfig` knobs `ckpt_dir`, `ckpt_every`, `keep_last_k`, `ckpt_hub_repo`.
- Produces: unified resume — if `resume_from` is set use it, else if `ckpt_dir` has a complete checkpoint auto-resume from the latest. During training, every `ckpt_every` completed steps (when `ckpt_dir` set and the count is below `steps`) write `ckpt_dir/step_{n}/`, prune to `keep_last_k`, and (if `ckpt_hub_repo` set) mirror to the Hub best-effort. `out_dir` still holds the FINAL model + manifest-last (unchanged). Auto-resume loses at most `ckpt_every` steps → satisfies AC "session death survived."

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pretrain_stage.py` (add `from tinyfables.checkpoint import find_latest_checkpoint` at the top with the other imports):

```python
def test_session_death_resume_via_ckpt_dir(toy_shards, tmp_path):
    ckpt = tmp_path / "ckpt"
    # "Session 1" dies after 3 steps; a checkpoint was taken at completed step 2.
    pretrain_stage.run(toy_cfg(toy_shards, steps=3, ckpt_dir=str(ckpt), ckpt_every=2), tmp_path / "s1")
    latest = find_latest_checkpoint(ckpt)
    assert latest is not None and latest.name == "step_2"

    # "Session 2" re-runs the SAME command (target steps=6) -> auto-resumes from step_2.
    pretrain_stage.run(toy_cfg(toy_shards, steps=6, ckpt_dir=str(ckpt), ckpt_every=2), tmp_path / "s2")

    # A straight-through 6-step run is the ground truth; CPU determinism -> bit-exact.
    pretrain_stage.run(toy_cfg(toy_shards, steps=6), tmp_path / "full")
    a = GPT.from_pretrained(tmp_path / "full").state_dict()
    b = GPT.from_pretrained(tmp_path / "s2").state_dict()
    assert a.keys() == b.keys()
    for k in a:
        assert torch.allclose(a[k], b[k], atol=1e-5), f"param {k} diverged after resume"


def test_prune_keeps_last_k_during_training(toy_shards, tmp_path):
    ckpt = tmp_path / "ckpt"
    pretrain_stage.run(toy_cfg(toy_shards, steps=6, ckpt_dir=str(ckpt), ckpt_every=2, keep_last_k=1), tmp_path / "o")
    dirs = sorted(p.name for p in ckpt.iterdir() if p.is_dir())
    assert dirs == ["step_4"]  # checkpoints at 2 and 4 (6==steps skipped); keep last 1


def test_ckpt_hub_mirror_invoked_when_configured(toy_shards, tmp_path, monkeypatch):
    from tinyfables import hub

    seen = []
    monkeypatch.setattr(hub, "upload_checkpoint_dir", lambda d, repo, **kw: seen.append((str(d), repo)) or repo)
    pretrain_stage.run(
        toy_cfg(toy_shards, steps=4, ckpt_dir=str(tmp_path / "ckpt"), ckpt_every=2, ckpt_hub_repo="user/ckpts"),
        tmp_path / "o",
    )
    assert seen and seen[-1][1] == "user/ckpts"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_pretrain_stage.py -k "session_death or prune or hub_mirror" -v`
Expected: FAIL — `ckpt_dir` is not honored yet (no checkpoints written; `find_latest_checkpoint` returns `None`).

- [ ] **Step 3: Wire checkpointing + auto-resume into `src/tinyfables/stages/pretrain.py`**

Add imports at the top (with the existing imports):

```python
from tinyfables.checkpoint import find_latest_checkpoint, load_checkpoint, prune_checkpoints, save_checkpoint
```

Add a best-effort mirror helper near the top of the module (after the imports, before `build_model`):

```python
def _mirror_checkpoint(ckpt_dir, repo_id) -> None:
    """Best-effort push of the latest checkpoint to the Hub. A network failure
    must never kill a multi-hour run, so all errors are swallowed with a note."""
    try:
        from tinyfables import hub

        latest = find_latest_checkpoint(ckpt_dir)
        if latest is not None:
            hub.upload_checkpoint_dir(latest, repo_id, private=True)
    except Exception as e:  # noqa: BLE001 — resilience over correctness here
        print(f"[tinyfables] checkpoint hub mirror failed (continuing): {e}")
```

Replace the whole resume/init block (`if cfg.resume_from: ... else: ...`) with the unified version:

```python
    resume_src = None
    if cfg.resume_from:
        resume_src = Path(cfg.resume_from)
    elif cfg.ckpt_dir:
        resume_src = find_latest_checkpoint(cfg.ckpt_dir)

    if resume_src is not None:
        model, state = load_checkpoint(resume_src, GPT, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        optimizer.load_state_dict(state["optimizer"])
        if state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
        start_step = state["step"]
    else:
        torch.manual_seed(cfg.seed)
        model = build_model(cfg, vocab_size).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        start_step = 0
```

(Note: the `scaler` was created in Task 7 before this block — keep that ordering.)

At the end of the training-loop body (after `last_loss = out.loss.item()`), add periodic checkpointing:

```python
        completed = step + 1
        if cfg.ckpt_dir and cfg.ckpt_every and completed % cfg.ckpt_every == 0 and completed < cfg.steps:
            save_checkpoint(model, optimizer, scaler, completed, cfg.ckpt_dir)
            prune_checkpoints(cfg.ckpt_dir, cfg.keep_last_k)
            if cfg.ckpt_hub_repo:
                _mirror_checkpoint(cfg.ckpt_dir, cfg.ckpt_hub_repo)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_pretrain_stage.py -v`
Expected: PASS — new checkpoint/resume/prune/mirror tests pass; the original bit-exact `resume_from` test still passes (unified path uses `load_checkpoint`, same optimizer.pt shape).

- [ ] **Step 5: Commit**

```bash
git add src/tinyfables/stages/pretrain.py tests/test_pretrain_stage.py
git commit -m "feat: periodic checkpoint + auto-resume across session death in pretrain (issue 04)"
```

---

### Task 9: Pretrain — metrics streaming + loss-curve artifact

**Files:**
- Modify: `src/tinyfables/stages/pretrain.py`
- Test: `tests/test_pretrain_stage.py` (add cases)

**Interfaces:**
- Consumes: `metrics.MetricsLogger`, `metrics.plot_loss_curve`; `PretrainConfig` knobs `log_every`, `trackio_project`, `trackio_space_id`, `run_name`.
- Produces: a `MetricsLogger` writing `loss_log.csv` (in `ckpt_dir` if set — survives session death — else `out_dir`); logs every `log_every` steps (and a final row); on finish copies `loss_log.csv` into `out_dir` (always) and renders `out_dir/loss_curve.png` (best-effort). `loss_log.csv` is added to the manifest artifacts (and `loss_curve.png` when it exists). trackio is used only when `trackio_project` is set AND trackio is importable (never in tests).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pretrain_stage.py`:

```python
def test_metrics_csv_and_manifest_artifact(toy_shards, tmp_path):
    out = tmp_path / "m"
    pretrain_stage.run(toy_cfg(toy_shards, steps=6, log_every=2), out)
    csv_path = out / "loss_log.csv"
    assert csv_path.exists()
    import csv as _csv
    rows = list(_csv.DictReader(open(csv_path)))
    assert len(rows) >= 3  # steps 2, 4, and the final row
    assert set(rows[0]) == {"step", "loss", "lr", "tokens_per_sec"}
    manifest = json.loads((out / "manifest.json").read_text())
    assert "loss_log.csv" in manifest["artifacts"]


def test_metrics_csv_lives_in_ckpt_dir_then_copied(toy_shards, tmp_path):
    ckpt = tmp_path / "ckpt"
    out = tmp_path / "o"
    pretrain_stage.run(toy_cfg(toy_shards, steps=4, log_every=1, ckpt_dir=str(ckpt)), out)
    assert (ckpt / "loss_log.csv").exists()  # durable across session death (Drive)
    assert (out / "loss_log.csv").exists()   # copied into the final stage dir


def test_trackio_project_set_but_absent_does_not_crash(toy_shards, tmp_path):
    # trackio is not installed; a configured project must degrade to CSV-only.
    out = tmp_path / "t"
    pretrain_stage.run(toy_cfg(toy_shards, steps=2, log_every=1, trackio_project="ignored", run_name="r"), out)
    assert (out / "loss_log.csv").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_pretrain_stage.py -k "metrics or trackio" -v`
Expected: FAIL — no `loss_log.csv` is written (metrics not wired yet).

- [ ] **Step 3: Wire metrics into `src/tinyfables/stages/pretrain.py`**

Add imports (with the existing ones):

```python
import shutil
import time

from tinyfables.metrics import MetricsLogger, plot_loss_curve
```

Before the training loop (after the resume/init block, before `model.train()`), construct the logger and timing anchors:

```python
    csv_path = Path(cfg.ckpt_dir or out_dir) / "loss_log.csv"
    logger = MetricsLogger(
        csv_path,
        project=cfg.trackio_project,
        run_name=cfg.run_name,
        space_id=cfg.trackio_space_id,
        config={"n_layer": cfg.n_layer, "d_model": cfg.d_model, "n_head": cfg.n_head,
                "n_ctx": cfg.n_ctx, "batch_size": cfg.batch_size, "steps": cfg.steps,
                "lr": cfg.lr, "amp": use_amp},
    )
    tokens_per_step = cfg.batch_size * window
    t_log = time.perf_counter()
    steps_since_log = 0
```

Inside the loop, after the periodic-checkpoint block, add metrics logging:

```python
        steps_since_log += 1
        if cfg.log_every and completed % cfg.log_every == 0:
            now = time.perf_counter()
            tps = (steps_since_log * tokens_per_step) / (now - t_log) if now > t_log else 0.0
            logger.log(step=completed, loss=last_loss, lr=lr_at(step, cfg), tokens_per_sec=tps)
            t_log = now
            steps_since_log = 0
```

After the loop and the final `model.save_pretrained` / `torch.save` (before writing `pretrain_summary.json`), finalize metrics:

```python
    logger.log(step=cfg.steps, loss=last_loss, lr=lr_at(max(cfg.steps - 1, 0), cfg), tokens_per_sec=0.0)
    logger.finish()
    out_csv = out_dir / "loss_log.csv"
    if csv_path != out_csv:
        shutil.copyfile(csv_path, out_csv)
    png = plot_loss_curve(out_csv, out_dir / "loss_curve.png")
```

Add `loss_log.csv` (and the PNG when present) to the manifest artifact list. Change the `write_manifest` artifacts argument:

```python
    artifacts = [
        out_dir / "config.json",
        out_dir / "generation_config.json",
        out_dir / "model.safetensors",
        out_dir / "optimizer.pt",
        out_dir / "pretrain_summary.json",
        out_dir / "loss_log.csv",
    ]
    if png is not None:
        artifacts.append(out_dir / "loss_curve.png")
    write_manifest(
        out_dir,
        "pretrain",
        cfg,
        artifacts,
        inputs={
            "tokens.bin": prep_dir / "tokens.bin",
            "mask.bin": prep_dir / "mask.bin",
            "tokenizer.json": Path(cfg.tokenizer_dir) / "tokenizer.json",
        },
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_pretrain_stage.py -v`
Expected: PASS (all pretrain tests, old + new).

- [ ] **Step 5: Run the whole suite to confirm no regressions**

Run: `.venv/bin/pytest -q`
Expected: PASS — all tests green (the toy chain now also emits `loss_log.csv`, which is additive; `test_toy_chain.py` and `test_cli_e2e.py` still pass since they check pretrain artifacts by subset / existence).

- [ ] **Step 6: Commit**

```bash
git add src/tinyfables/stages/pretrain.py tests/test_pretrain_stage.py
git commit -m "feat: stream metrics + save loss-curve artifact in pretrain (issue 04)"
```

---

### Task 10: Real-run configs, optional deps, design.md implementation notes

**Files:**
- Modify: `configs/pretrain_full.yaml`
- Modify: `pyproject.toml` (add `train` optional extra)
- Modify: `docs/design.md` (add "Implementation (issue 04)" subsection)
- Test: `tests/test_full_configs.py` (add cases)

**Interfaces:**
- Consumes: `load_config`, `PretrainConfig`, `BenchmarkConfig`.
- Produces: `configs/pretrain_full.yaml` parses with `amp=True`, `ckpt_every>0`, `log_every>0`, `trackio_project` set, and a `ckpt_dir` under a Drive path. `configs/benchmark_full.yaml` parses to a valid `BenchmarkConfig`. Documentation records the contract; operational numbers are filled in during Track B.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_full_configs.py`:

```python
def test_pretrain_full_enables_amp_and_checkpointing():
    from tinyfables.config import PretrainConfig

    cfg = load_config(REPO / "configs" / "pretrain_full.yaml", PretrainConfig)
    assert cfg.amp is True
    assert cfg.ckpt_every > 0 and cfg.log_every > 0
    assert cfg.ckpt_dir is not None
    assert cfg.trackio_project  # non-empty -> live dashboard on Colab
    # geometry is unchanged from the design table
    assert (cfg.n_layer, cfg.d_model, cfg.n_head, cfg.n_ctx) == (6, 384, 6, 1024)
    assert cfg.batch_size == 16 and cfg.steps == 20000


def test_benchmark_full_is_valid():
    from tinyfables.config import BenchmarkConfig

    cfg = load_config(REPO / "configs" / "benchmark_full.yaml", BenchmarkConfig)
    assert cfg.measure_steps > 0 and cfg.token_budget >= 200_000_000
    assert (cfg.n_layer, cfg.d_model, cfg.n_ctx) == (6, 384, 1024)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_full_configs.py -k "pretrain_full or benchmark_full" -v`
Expected: FAIL — `pretrain_full.yaml` has no `amp`/`ckpt_every`/`log_every`/`trackio_project` yet (defaults make the asserts fail).

- [ ] **Step 3: Update `configs/pretrain_full.yaml`**

```yaml
# Real Base Model run (issue 04, Colab T4): design geometry (6/384/6, ctx 1024,
# batch 16, 20k steps) + fp16 AMP + Drive checkpointing + trackio metrics.
# ckpt_dir is Drive-mounted so checkpoints survive session death; re-running this
# same command auto-resumes from the latest checkpoint. ckpt_every/keep_last_k
# are operational — TUNE after the benchmark reading and RECORD the choice in
# docs/design.md. run_name is stable so a resumed run continues the same trackio run.
prep_dir: runs/prep_full
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
ckpt_dir: /content/drive/MyDrive/tinyfables/ckpt_base
ckpt_every: 500
keep_last_k: 2
log_every: 50
trackio_project: tinyfables-base
run_name: base-v1
# trackio_space_id: congthanh991/tinyfables-trackio   # set on Colab for the live dashboard
# ckpt_hub_repo: congthanh991/tinyfables-13m-base-ckpts  # optional Hub mirror of checkpoints
```

- [ ] **Step 4: Add the `train` optional extra to `pyproject.toml`**

In `[project.optional-dependencies]`, add:

```toml
train = ["trackio", "matplotlib>=3.7"]
```

- [ ] **Step 5: Add the "Implementation (issue 04)" subsection to `docs/design.md`**

Insert directly after the `### Implementation (issue 02)` block (before `## Build vs buy`):

```markdown
### Implementation (issue 04)

- **Mixed precision (T4).** Training runs under `torch.amp.autocast(fp16)` with a
  `GradScaler`; AMP is enabled only on CUDA (`amp and device=="cuda"`), so CPU/MPS
  toy runs stay fp32 and the walking-skeleton bit-exact resume guarantee holds. The
  scaler's dynamic scale is checkpointed (in `optimizer.pt`) so resume continues the
  loss-scaling schedule.
- **Checkpoint-resume for session death.** `checkpoint.py` writes
  `ckpt_dir/step_{n}/` (model safetensors + optimizer + scaler + step), with
  `checkpoint_state.json` written LAST as the completion marker (an interrupted
  write is never mistaken for complete). `pretrain` checkpoints every `ckpt_every`
  steps, prunes to `keep_last_k`, and on startup AUTO-RESUMES from the latest
  complete checkpoint in `ckpt_dir` — so a dead Colab session is recovered by
  re-running the *same command*. On Colab `ckpt_dir` is Drive-mounted, so it
  outlives the runtime; a session death costs ≤ `ckpt_every` steps. `out_dir` still
  holds the FINAL model + manifest-last.
- **Benchmark gate.** The `benchmark` stage times the target geometry with fp16 on
  synthetic batches (real vocab/shape), excludes warmup, and records
  `tokens_per_second`, `est_seconds_per_epoch`, and a `go`/`revise` decision vs
  `target_tokens_per_sec`. Run it BEFORE `pretrain_full`; record the measured number
  and the decision here (Track B) — this is the week-1 "measure, then commit" gate.
- **Metrics + loss curve.** `metrics.MetricsLogger` mirrors loss/lr/tokens-per-sec
  to a durable `loss_log.csv` (in `ckpt_dir`, so it survives session death) and, when
  a `trackio_project` is configured and trackio is installed, to a trackio Space for
  a live dashboard. On finish the CSV is copied into `out_dir` and `loss_curve.png`
  is rendered (matplotlib, best-effort). trackio + matplotlib are the optional
  `.[train]` extra; absent, logging degrades to CSV-only (tests run offline).
- **Hub push.** `hub.py` (+ `python -m tinyfables push`) pushes the tokenizer
  (`tinyfables-tokenizer`) and the Base Model (`tinyfables-13m-base`); the model push
  ignores `optimizer.pt`/checkpoint markers so a checkpoint dir can be the source.
- **Operational record (Track B — fill in during the real run):** measured T4
  tokens/sec = TBD; benchmark decision = TBD; chosen `ckpt_every` = TBD (± steps lost
  per death); any size/data-budget change = TBD.
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_full_configs.py -v`
Expected: PASS (all 3).

- [ ] **Step 7: Run the whole suite one more time**

Run: `.venv/bin/pytest -q`
Expected: PASS — everything green.

- [ ] **Step 8: Commit**

```bash
git add configs/pretrain_full.yaml pyproject.toml docs/design.md tests/test_full_configs.py
git commit -m "config+docs: real-run knobs, train extra, issue-04 implementation notes (issue 04)"
```

---

## Track A completion: merge to main

- [ ] Run the full suite once more on the branch: `.venv/bin/pytest -q` → all pass.
- [ ] Request code review (superpowers:requesting-code-review) over the branch diff vs `main`; address Critical/Important findings.
- [ ] Merge with history preserved:

```bash
git checkout main
git merge --no-ff feat/issue-04-real-base-model -m "Merge feat/issue-04-real-base-model: real Base Model scaffolding (issue 04 Track A)"
```

Do NOT tick issue 04 in `docs/issues/README.md` yet — that happens after Track B produces the Hub artifacts and the benchmark decision.

---

## Track B: Colab operations (NOT delegatable to subagents)

> Drive the real Colab session interactively via the **colab-mcp** browser connection. Training runs for hours and Colab sessions die, so this spans multiple chat sessions. The continuity mechanisms are: (1) Drive-mounted `ckpt_dir` + auto-resume, and (2) `.superpowers/sdd/progress.md`. **Record every operational number in `docs/design.md` immediately** (measured tokens/sec, benchmark decision, chosen `ckpt_every`, any size change).

**Preconditions:** Track A merged to `main`; HF account `congthanh991`; `HF_TOKEN` (write scope) available to paste into the Colab session; a Colab notebook with a T4 runtime.

- [ ] **B1 — Open the Colab connection.** Use `mcp__colab-mcp__open_colab_browser_connection` and connect to a GPU (T4) runtime. Confirm the GPU with a cell: `!nvidia-smi` (expect a Tesla T4, ~15–16GB).

- [ ] **B2 — Install the repo + train extra.** In a cell:
  `!pip install -q "git+https://github.com/<owner>/<repo>.git@main#egg=tinyfables[train]"` (or clone + `pip install -e ".[train]"` if the repo isn't on GitHub yet). Verify: `python -c "import tinyfables, torch, trackio, matplotlib; print(torch.cuda.is_available())"` → `True`.

- [ ] **B3 — Auth + Drive.** `from huggingface_hub import login; login(token=...)` with the write token (user `congthanh991`); `from google.colab import drive; drive.mount('/content/drive')`; `!mkdir -p /content/drive/MyDrive/tinyfables`. Confirm `hf_whoami` → `congthanh991`.

- [ ] **B4 — Tokenizer (real).** `!python -m tinyfables run tokenizer --config configs/tokenizer_full.yaml --out runs/tokenizer_full` (streams 50k fables; needs network). Check `runs/tokenizer_full/compression_report.json` — record the GPT-2-50k vs ours-8k avg-tokens/fable figure into `docs/design.md`/ADR-0002 (this is the ADR-0002 report figure). Push the tokenizer: `!python -m tinyfables push --kind tokenizer --path runs/tokenizer_full --repo congthanh991/tinyfables-tokenizer --public`.

- [ ] **B5 — Prep (real, WITH augmentation).** `!python -m tinyfables run prep --config configs/prep_full.yaml --out runs/prep_full` (≈450k fables → ~250M tokens; minutes–low-tens-of-minutes). Verify `runs/prep_full/prep_summary.json`: `family_counts["held-out-template"] == 0` (leakage guard), `paraphrase_coverage == 0.15`, seen-template count > 0. **Copy the shards to Drive** so a session death doesn't force a re-prep: `!cp -r runs/prep_full runs/tokenizer_full /content/drive/MyDrive/tinyfables/`.

- [ ] **B6 — Benchmark gate (measure, then commit).** `!python -m tinyfables run benchmark --config configs/benchmark_full.yaml --out runs/benchmark_full`. Read `runs/benchmark_full/benchmark_summary.json`:
  - Record `tokens_per_second`, `est_seconds_per_epoch`, and `decision` in `docs/design.md` (the "Operational record" line from Task 10 Step 5).
  - **Decision:** if `decision == "go"` (throughput clears the ~1–2h/epoch budget), proceed to B7 with the geometry unchanged. If `"revise"`, shrink the **data budget first, then the model** (design.md risk register) — adjust `max_rows` in `prep_full.yaml` (re-prep) or, only if necessary, the geometry; re-run the benchmark; record the change and rationale in `docs/design.md`. Do NOT start the long run until the decision is `go`.

- [ ] **B7 — Set the live dashboard + launch the long run.** Uncomment/point `trackio_space_id: congthanh991/tinyfables-trackio` in `configs/pretrain_full.yaml` (edit the file in the Colab checkout). Tune `ckpt_every` from the benchmark reading (steps ≈ a few minutes of training, so a death costs minutes not hours) and record it in `docs/design.md`. Launch:
  `!python -m tinyfables run pretrain --config configs/pretrain_full.yaml --out /content/drive/MyDrive/tinyfables/base_out`
  Point `out_dir` at Drive so the final artifact also survives. Confirm metrics appear in the trackio Space (AC 3) while the run is in flight.

- [ ] **B8 — Survive session death (expected, possibly several times).** When a session dies: reconnect (B1), re-install (B2), re-auth + re-mount Drive (B3), and **re-run the exact B7 command**. Auto-resume picks up the latest `ckpt_dir` checkpoint (≤ `ckpt_every` steps lost). This validates AC 2 — note in `.superpowers/sdd/progress.md` that a real death was survived and how many steps were lost. Repeat until `steps` (20k) is reached.

- [ ] **B9 — Push the Base Model + loss curve.** After completion, `base_out/` holds `model.safetensors`, `loss_log.csv`, `loss_curve.png`, `pretrain_summary.json`, `manifest.json`. Push: `!python -m tinyfables push --kind model --path /content/drive/MyDrive/tinyfables/base_out --repo congthanh991/tinyfables-13m-base --public`. Save the loss-curve artifact (the PNG + CSV) alongside the report; confirm the Hub repo shows the model card + files (AC 4).

- [ ] **B10 — Spot check (~20 varied FableSpecs).** Run `python -m tinyfables generate --checkpoint /content/drive/MyDrive/tinyfables/base_out --tokenizer runs/tokenizer_full ...` for ~20 diverse specs (vary character/setting/challenge/outcome/moral/age; include a few free-text `--text` instructions and at least one held-out-style phrasing). Confirm generations are fluent fables that broadly honor the requested Elements (AC 5). Record a couple of representative samples in `.superpowers/sdd/progress.md` and note any adherence gaps for issue 05.

- [ ] **B11 — Close out.** Record final numbers in `docs/design.md` (measured tokens/sec, benchmark decision, chosen `ckpt_every`, deaths survived + steps lost, any size change). Tick issue 04 in `docs/issues/README.md` (`| 04 | ... | 02, 03 | ☑ |`). Commit the docs on `main`:

```bash
git add docs/design.md docs/issues/README.md
git commit -m "docs: record issue-04 operational results; tick issue 04 done"
```

**Done when:** Base Model + tokenizer on the Hub; loss curve saved; benchmark decision recorded; a real session death survived via resume; metrics were visible during the run; ~20-FableSpec spot check passed; issue 04 ticked; `docs/design.md` updated.

---

## Self-Review

**1. Spec coverage** (issue 04 acceptance criteria + spec gotchas):
- AC "Benchmark stage reports tokens/sec; size go/no-go recorded" → Task 6 (stage + decision) + B6 (run + record). ✓
- AC "session death survived, resume ≤ N steps" → Task 8 (auto-resume + periodic checkpoint) + B8. ✓
- AC "metrics visible in tracker during run" → Task 9 (MetricsLogger + trackio) + B7. ✓
- AC "Base Model + tokenizer pushed to Hub; loss-curve saved" → Task 4 (hub) + Task 9 (loss curve) + B4/B9. ✓
- AC "spot check ~20 FableSpecs fluent" → B10 (uses existing `generate.py`). ✓
- Gotcha "prep_full has no paraphrase_coverage" → Task 1 (set 0.15 + regression lock) + B5 (verify held-out=0). ✓
- Gotcha "real-run order tokenizer→prep→benchmark→pretrain" → Track B B4–B7 in that order. ✓
- Spec "AMP fp16 for T4; checkpoint+optimizer-state push; resume; trackio; Hub push utilities" → Tasks 7/8/9/4. ✓
- Spec "manifest-written-last completion marker preserved" → benchmark uses `write_manifest`; checkpoints use their own last-written marker. ✓
- Spec "lazy-registry / light-import property intact" → `metrics.py`/`hub.py` import torch/trackio/hf lazily; benchmark registered as a lazy module path string; verified by the suite running with trackio/matplotlib absent. ✓
- ADR-0002 compression figure (tokenizer_full has `compare_gpt2: true`) → B4 records it. ✓

**2. Placeholder scan:** No "TBD/handle edge cases/similar to Task N" in the *code* steps — every code step shows complete code. The only "TBD"s are in the design.md *operational record* line, which is intentional: those values are produced by Track B and the plan says so explicitly. Track B steps are operations (a runbook), not TDD code, so they carry commands, not test cycles — as specified for the non-delegatable track.

**3. Type consistency:** Cross-task interfaces checked — `save_checkpoint(model, optimizer, scaler, step, ckpt_dir)`, `find_latest_checkpoint`, `load_checkpoint(path, model_cls, device) -> (model, state)`, `prune_checkpoints(dir, keep_last_k)` (Task 3) are used with those exact signatures in Task 8. `MetricsLogger(csv_path, project, run_name, space_id, config)` + `.log(step, loss, lr, tokens_per_sec)` + `.finish()` and `plot_loss_curve(csv, png)` (Task 2) are used identically in Task 9. `hub.push_model_to_hub/push_tokenizer_to_hub/upload_checkpoint_dir(dir, repo_id, *, private, api)` (Task 4) are used by the CLI (Task 4) and the pretrain mirror (Task 8). `BenchmarkConfig` fields (Task 5) match `benchmark.run` reads (Task 6) and `benchmark_full.yaml` (Task 6). `PretrainConfig` new fields (Task 5) match every read in Tasks 7–9 and the keys in `pretrain_full.yaml` (Task 10). The `optimizer.pt` dict shape `{"optimizer","scaler","step"}` is written in Task 7 (final) + Task 3 (checkpoints) and read by `load_checkpoint` (Task 3) used in Task 8 — consistent. No naming drift found.
