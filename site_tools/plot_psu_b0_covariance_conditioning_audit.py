#!/usr/bin/env python3
"""Plot covariance retuning, fair baselines, and stopping-rule failure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "component": "#64777d",
    "graph": "#147d70",
    "selection": "#3578a8",
    "diagnostic": "#b14f61",
    "gold": "#d1902f",
    "ink": "#52636b",
    "grid": "#dfe7e4",
}


def _diagnosis_lookup(report: dict) -> dict[tuple[str, str], dict]:
    return {
        (str(row["split"]), str(row["candidate_id"])): row
        for row in report["summaries"]
    }


def build_figure(
    diagnosis: dict,
    stopping: dict,
    output: Path,
) -> None:
    lookup = _diagnosis_lookup(diagnosis)
    figure, axes = plt.subplots(2, 2, figsize=(13.6, 9.2))
    figure.patch.set_facecolor("#f7f9f8")
    for axis in axes.flat:
        axis.set_facecolor("#ffffff")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(color=COLORS["grid"], linewidth=0.8, alpha=0.85)
        axis.set_axisbelow(True)

    methods = (
        ("spatial_a0_s3_k4", "Retuned component\ns3-k4"),
        ("full_graph_s3_k4", "Full graph\ns3-k4"),
    )
    splits = ("selection", "opened_diagnostic_check")
    split_labels = ("Opened selection", "Opened diagnostic check")
    x = np.arange(len(methods))
    width = 0.34
    for index, (split, label) in enumerate(
        zip(splits, split_labels, strict=True)
    ):
        values = [
            lookup[(split, candidate)]["mean_field_gain_percent"]
            for candidate, _ in methods
        ]
        axes[0, 0].bar(
            x + (index - 0.5) * width,
            values,
            width,
            color=(
                COLORS["selection"]
                if split == "selection"
                else COLORS["diagnostic"]
            ),
            alpha=0.86,
            label=label,
        )
    axes[0, 0].set(
        title="A  Most of the apparent gain is solver retuning",
        ylabel="Mean field gain vs legacy component s5-k4 (%)",
        xticks=x,
        xticklabels=[label for _, label in methods],
    )
    axes[0, 0].legend(frameon=False, fontsize=8.5)

    constant = stopping["constant_stage_context"]
    stages = (3, 4, 5)
    means = [
        constant[str(stage)]["opened_diagnostic_check"][
            "mean_field_gain_percent"
        ]
        for stage in stages
    ]
    p10 = [
        constant[str(stage)]["opened_diagnostic_check"][
            "field_gain_p10_percent"
        ]
        for stage in stages
    ]
    worst = [
        constant[str(stage)]["opened_diagnostic_check"][
            "worst_field_gain_percent"
        ]
        for stage in stages
    ]
    axes[0, 1].bar(
        stages,
        means,
        color=[COLORS["component"], COLORS["graph"], COLORS["gold"]],
        alpha=0.82,
        label="mean",
    )
    axes[0, 1].plot(
        stages,
        p10,
        marker="o",
        color=COLORS["selection"],
        linewidth=1.8,
        label="p10",
    )
    axes[0, 1].plot(
        stages,
        worst,
        marker="x",
        color=COLORS["diagnostic"],
        linewidth=1.8,
        label="worst field",
    )
    axes[0, 1].axhline(0.0, color=COLORS["ink"], linewidth=1)
    axes[0, 1].set(
        title="B  Mean gain and tail risk move in opposite directions",
        xlabel="Graph-whitened PCGLS stage",
        ylabel="Gain vs component s3-k4 (%)",
        xticks=stages,
    )
    axes[0, 1].legend(frameon=False, fontsize=8.5)

    selected = stopping["decision"]["selected_rule"]["selection"]
    diagnostic = stopping["decision"]["opened_diagnostic_check"]
    metrics = (
        ("mean_field_gain_percent", "Mean"),
        ("field_gain_p10_percent", "p10"),
        ("worst_field_gain_percent", "Worst"),
    )
    metric_x = np.arange(len(metrics))
    axes[1, 0].bar(
        metric_x - width / 2,
        [selected[key] for key, _ in metrics],
        width,
        color=COLORS["selection"],
        alpha=0.86,
        label="Opened selection",
    )
    axes[1, 0].bar(
        metric_x + width / 2,
        [diagnostic[key] for key, _ in metrics],
        width,
        color=COLORS["diagnostic"],
        alpha=0.86,
        label="Opened diagnostic check",
    )
    axes[1, 0].axhline(0.0, color=COLORS["ink"], linewidth=1)
    axes[1, 0].axhline(
        -2.0,
        color=COLORS["gold"],
        linestyle="--",
        linewidth=1,
        label="worst-field floor",
    )
    axes[1, 0].set(
        title="C  A selection-safe rule collapses on the second half",
        ylabel="Gain vs component s3-k4 (%)",
        xticks=metric_x,
        xticklabels=[label for _, label in metrics],
    )
    axes[1, 0].legend(frameon=False, fontsize=8.0)

    stage_labels = ("3", "4", "5")
    selection_counts = selected["stage_counts"]
    diagnostic_counts = diagnostic["stage_counts"]
    bottom = np.zeros(2)
    for stage, color in zip(
        stage_labels,
        (COLORS["component"], COLORS["graph"], COLORS["gold"]),
        strict=True,
    ):
        values = np.asarray(
            [selection_counts[stage], diagnostic_counts[stage]]
        )
        axes[1, 1].bar(
            (0, 1),
            values,
            bottom=bottom,
            color=color,
            alpha=0.84,
            label=f"stage {stage}",
        )
        bottom += values
    axes[1, 1].set(
        title="D  Similar action counts do not imply risk transfer",
        ylabel="Fields",
        xticks=(0, 1),
        xticklabels=("Selection", "Diagnostic check"),
        ylim=(0, 73),
    )
    axes[1, 1].legend(frameon=False, fontsize=8.2, ncol=3)
    axes[1, 1].text(
        0.5,
        68,
        "NO-GO: p10 -1.75%, harm 12.5%, worst -17.53%",
        ha="center",
        va="center",
        color=COLORS["diagnostic"],
        fontsize=9,
        fontweight="bold",
    )

    figure.suptitle(
        "Covariance-conditioned PCGLS: fair-baseline signal, unsafe pooled stopping",
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.012,
        "Real PSU detector geometry; analytic reaction morphologies and "
        "synthetic graph noise only. All rule selection is post-open.",
        ha="center",
        fontsize=8.5,
        color=COLORS["ink"],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=190, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnosis-report", type=Path, required=True)
    parser.add_argument("--stopping-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_figure(
        json.loads(args.diagnosis_report.read_text(encoding="utf-8")),
        json.loads(args.stopping_report.read_text(encoding="utf-8")),
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
