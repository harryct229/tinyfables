import csv
from pathlib import Path

from tinyfables.metrics import MetricsLogger, _loss_series, plot_loss_curve


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


def test_loss_series_dedups_and_sorts_across_resume(tmp_path):
    # Simulate a session-death resume: session 1 logs steps 50, 100, 150, ..., up
    # through 3050, 3100, then dies at step 3200 (mid-session, past the last
    # checkpoint). Session 2 resumes from the last CHECKPOINT (step_3000) and
    # re-runs steps 3050 and 3100 with DIFFERENT loss values before continuing —
    # so the CSV holds duplicate rows for those two steps, out of step order.
    csv_path = tmp_path / "loss_log.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "loss", "lr", "tokens_per_sec"])
        # session 1
        w.writerow([50, "3.000000", "1e-4", "100.0"])
        w.writerow([100, "2.900000", "1e-4", "100.0"])
        w.writerow([150, "2.800000", "1e-4", "100.0"])
        w.writerow([3050, "2.000000", "1e-4", "100.0"])  # session 1's original value
        w.writerow([3100, "1.900000", "1e-4", "100.0"])  # session 1's original value
        w.writerow([3200, "1.500000", "1e-4", "100.0"])  # last row before death
        # session 2 (resumed from step_3000 checkpoint) re-runs 3050/3100
        w.writerow([3050, "1.999999", "1e-4", "100.0"])  # re-run, different value
        w.writerow([3100, "1.888888", "1e-4", "100.0"])  # re-run, different value
        # rows with an empty/"nan" loss must be skipped
        w.writerow([3150, "nan", "1e-4", "100.0"])
        w.writerow([3160, "", "1e-4", "100.0"])

    steps, losses = _loss_series(csv_path)

    assert steps == sorted(steps)
    assert len(steps) == len(set(steps))  # strictly unique
    assert steps == [50, 100, 150, 3050, 3100, 3200]
    # the LAST-written loss wins for the duplicated steps (session 2's re-run)
    assert losses[steps.index(3050)] == 1.999999
    assert losses[steps.index(3100)] == 1.888888
    # session-1-only steps keep their original values
    assert losses[steps.index(3200)] == 1.5
