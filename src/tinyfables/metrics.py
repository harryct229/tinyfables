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
