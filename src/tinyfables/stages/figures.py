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
    except ImportError as exc:
        raise RuntimeError(
            "the figures stage requires matplotlib; install tinyfables[train]"
        ) from exc
    return plt


def render_robustness_grid(rows, provenance, png_path) -> None:
    """Render paired model columns for all three prompt-template families."""
    plt = _matplotlib()
    lookup = {(row["model"], row["family"]): row for row in rows}
    metrics = (
        ("adherence", "Element adherence"),
        ("moral_delivery", "Moral delivery"),
    )
    family_labels = ("Canonical", "Seen", "Held-out")
    model_labels = {"with-aug": "With augmentation", "no-aug": "No augmentation"}
    model_colors = {"with-aug": "#0072b2", "no-aug": "#999999"}
    positions = list(range(len(FAMILY_ORDER)))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), sharey=True)

    for panel_idx, (ax, (field, title)) in enumerate(zip(axes, metrics)):
        ax.axvspan(
            1.5,
            2.5,
            color="#f4a261",
            alpha=0.14,
            linewidth=0,
            zorder=0,
        )
        for model_idx, model in enumerate(MODEL_ORDER):
            offset = (model_idx - 0.5) * width
            values = [lookup[(model, family)][field] for family in FAMILY_ORDER]
            bars = ax.bar(
                [position + offset for position in positions],
                values,
                width=width,
                label=model_labels[model],
                color=model_colors[model],
                zorder=2,
            )
            ax.bar_label(
                bars,
                labels=[f"{100.0 * value:.1f}%" for value in values],
                padding=3,
                fontsize=8,
            )
        ax.set_title(title, fontsize=11)
        ax.set_xticks(positions, family_labels)
        ax.get_xticklabels()[-1].set_color("#b45309")
        ax.get_xticklabels()[-1].set_fontweight("bold")
        ax.set_ylim(0.0, 1.08)
        ax.set_ylabel("Rate" if panel_idx == 0 else "")
        ax.grid(axis="y", alpha=0.22, zorder=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle("Prompt robustness: paraphrase-augmentation ablation", fontsize=13)
    perplexity = {
        model: next(row["perplexity"] for row in rows if row["model"] == model)
        for model in MODEL_ORDER
    }
    robust_prov = provenance["figures"]["robustness_grid"]
    manifests = robust_prov["eval_manifest_sha256"]
    run_ids = robust_prov["run_ids"]
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
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.80, wspace=0.20)
    fig.savefig(
        png_path,
        dpi=200,
        bbox_inches="tight",
        metadata={
            "Title": "Issue 10 robustness grid",
            "Creator": "tinyfables figures stage",
        },
    )
    plt.close(fig)


def render_rm_data_curve(rows, provenance, png_path) -> None:
    """Render all four committed RM curve points with chance and gate references."""
    plt = _matplotlib()
    x = [row["train_size"] for row in rows]
    y = [row["held_out_accuracy"] for row in rows]
    gate = rows[0]["accuracy_gate"]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(x, y, marker="o", linewidth=2.0, color="#0072b2")
    ax.axhline(
        0.5,
        linestyle=":",
        linewidth=1.2,
        color="#666666",
        label="chance (0.50)",
    )
    ax.axhline(
        gate,
        linestyle="--",
        linewidth=1.4,
        color="#d55e00",
        label=f"gate ({gate:.2f})",
    )
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
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
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
        metadata={
            "Title": "Issue 07 RM data curve",
            "Creator": "tinyfables figures stage",
        },
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
        cells = [
            f"{row['adherence']:.3f} / {row['moral_delivery']:.3f}"
            for row in row_values
        ]
        lines.append(
            f"| {model} | {row_values[0]['perplexity']:.3f} | "
            f"{cells[0]} | {cells[1]} | {cells[2]} |"
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
            f"{row['held_out_accuracy']:.6f} | {row['final_loss']:.6f} | "
            f"{row['steps']} |"
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
    provenance_path.write_text(
        json.dumps(data.provenance, indent=2, sort_keys=True) + "\n"
    )
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
