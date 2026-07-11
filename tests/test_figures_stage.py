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


class _FailingAxes:
    def axvspan(self, *args, **kwargs):
        raise RuntimeError("robustness render failed")

    def plot(self, *args, **kwargs):
        raise RuntimeError("RM render failed")


class _FakePyplot:
    def __init__(self, axes):
        self.figure = object()
        self.axes = axes
        self.closed = []

    def subplots(self, *args, **kwargs):
        return self.figure, self.axes

    def close(self, figure):
        self.closed.append(figure)


def test_robustness_renderer_closes_figure_when_rendering_fails(monkeypatch, tmp_path):
    pyplot = _FakePyplot([_FailingAxes(), _FailingAxes()])
    monkeypatch.setattr(figures_stage, "_matplotlib", lambda: pyplot)

    with pytest.raises(RuntimeError, match="robustness render failed"):
        figures_stage.render_robustness_grid([], {}, tmp_path / "robustness.png")

    assert pyplot.closed == [pyplot.figure]


def test_rm_renderer_closes_figure_when_rendering_fails(monkeypatch, tmp_path):
    pyplot = _FakePyplot(_FailingAxes())
    monkeypatch.setattr(figures_stage, "_matplotlib", lambda: pyplot)
    rows = [{"train_size": 100, "held_out_accuracy": 0.5, "accuracy_gate": 0.65}]

    with pytest.raises(RuntimeError, match="RM render failed"):
        figures_stage.render_rm_data_curve(rows, {}, tmp_path / "curve.png")

    assert pyplot.closed == [pyplot.figure]
