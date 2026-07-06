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
