#!/usr/bin/env python3
"""Plot the detector-graph covariance acquisition-planning experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "component_iid": "#3578a8",
    "graph_heat": "#147d70",
    "nonstationary_drift": "#b14f61",
    "gold": "#d1902f",
    "ink": "#52636b",
    "grid": "#dfe7e4",
}
LABELS = {
    "component_iid": "IID truth",
    "graph_heat": "Graph-heat truth",
    "nonstationary_drift": "Nonstationary + drift",
}


def _summary_lookup(report: dict) -> dict[tuple[str, int, str], dict]:
    return {
        (
            str(row["truth_family"]),
            int(row["repeat_count"]),
            str(row["model"]),
        ): row
        for row in report["summary"]
    }


def build_figure(report: dict, output: Path) -> None:
    repeats = [
        int(value)
        for value in report["configuration"]["simulation"]["repeat_counts"]
    ]
    lookup = _summary_lookup(report)
    families = (
        "component_iid",
        "graph_heat",
        "nonstationary_drift",
    )
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.8))
    figure.patch.set_facecolor("#f7f9f8")
    for axis in axes.flat:
        axis.set_facecolor("#ffffff")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(color=COLORS["grid"], linewidth=0.8, alpha=0.85)
        axis.set_axisbelow(True)

    for family in families:
        rows = [lookup[(family, count, "dg_covgate")] for count in repeats]
        median = [
            row["nll_gain_vs_component_iid"]["median"] for row in rows
        ]
        p10 = [row["nll_gain_vs_component_iid"]["p10"] for row in rows]
        p90 = [row["nll_gain_vs_component_iid"]["p90"] for row in rows]
        axes[0, 0].plot(
            repeats,
            median,
            marker="o",
            linewidth=2,
            color=COLORS[family],
            label=LABELS[family],
        )
        axes[0, 0].fill_between(
            repeats,
            p10,
            p90,
            color=COLORS[family],
            alpha=0.12,
        )
    axes[0, 0].axhline(0.0, color=COLORS["ink"], linewidth=1)
    axes[0, 0].set(
        title="A  Sealed likelihood gain of DG-CovGate",
        xlabel="Available flow-off repeats per camera",
        ylabel="NLL gain vs component-IID (nat / dimension)",
        xticks=repeats,
    )
    axes[0, 0].legend(frameon=False, fontsize=8.5)

    for family in families:
        activation = [
            lookup[(family, count, "dg_covgate")][
                "graph_activation_rate"
            ]
            for count in repeats
        ]
        axes[0, 1].plot(
            repeats,
            100.0 * np.asarray(activation),
            marker="o",
            linewidth=2,
            color=COLORS[family],
            label=LABELS[family],
        )
    axes[0, 1].axhline(
        75.0,
        color=COLORS["gold"],
        linewidth=1,
        linestyle="--",
        label="in-class target",
    )
    axes[0, 1].axhline(
        25.0,
        color=COLORS["ink"],
        linewidth=1,
        linestyle=":",
        label="IID false-activation ceiling",
    )
    axes[0, 1].set(
        title="B  Held-out repeats decide whether graph structure is used",
        xlabel="Available flow-off repeats per camera",
        ylabel="Graph activation rate (%)",
        xticks=repeats,
        ylim=(-4, 104),
    )
    axes[0, 1].legend(frameon=False, fontsize=8.0)

    models = (
        "diagonal_shrinkage",
        "always_graph",
        "always_low_rank_drift",
        "dg_covgate",
    )
    model_labels = (
        "Diagonal",
        "Always graph",
        "Always graph + amplitude + rank-1 drift",
        "DG-CovGate",
    )
    width = 0.19
    x = np.arange(len(repeats), dtype=np.float64)
    for index, (model, label) in enumerate(zip(models, model_labels, strict=True)):
        error = []
        for count in repeats:
            error.append(
                max(
                    lookup[(family, count, model)][
                        "absolute_coverage_error"
                    ]["p90"]
                    for family in families
                )
            )
        axes[1, 0].bar(
            x + (index - 1.5) * width,
            100.0 * np.asarray(error),
            width=width,
            label=label,
            color=(
                "#9aa9ad",
                COLORS["gold"],
                "#7656a8",
                COLORS["graph_heat"],
            )[index],
        )
    axes[1, 0].axhline(
        8.0,
        color=COLORS["nonstationary_drift"],
        linestyle="--",
        linewidth=1,
    )
    axes[1, 0].set(
        title="C  Worst-family 95% whitening-coverage error",
        xlabel="Available flow-off repeats per camera",
        ylabel="p90 absolute coverage error (percentage points)",
        xticks=x,
        xticklabels=[str(value) for value in repeats],
    )
    axes[1, 0].legend(frameon=False, fontsize=8.2)

    rank_rows = report["empirical_covariance_rank_ceiling"]
    rank_fraction = [
        100.0 * float(row["rank_fraction"]) for row in rank_rows
    ]
    axes[1, 1].bar(
        np.arange(len(rank_rows)),
        rank_fraction,
        color=COLORS["component_iid"],
        alpha=0.82,
    )
    axes[1, 1].set(
        title="D  Dense empirical covariance remains rank deficient",
        xlabel="Available flow-off repeats per camera",
        ylabel="Maximum empirical rank / 512 dimensions (%)",
        xticks=np.arange(len(rank_rows)),
        xticklabels=[str(row["repeat_count"]) for row in rank_rows],
    )
    decision = report["acquisition_gate"]
    minimum = decision["minimum_repeat_count_passing_synthetic_gate"]
    message = (
        f"Synthetic gate first passes at R={minimum}"
        if minimum is not None
        else "No tested repeat count passes every synthetic gate"
    )
    axes[1, 1].text(
        0.02,
        0.96,
        message
        + "\nDense covariance is never authorized; fit a low-parameter model\n"
        + "and reserve 25% of real repeats for validation.",
        transform=axes[1, 1].transAxes,
        va="top",
        fontsize=8.8,
        color=COLORS["ink"],
        bbox={
            "facecolor": "#ffffff",
            "edgecolor": COLORS["grid"],
            "boxstyle": "round,pad=0.35",
        },
    )

    figure.suptitle(
        "DG-CovGate: flow-off acquisition planning on the real PSU detector graph",
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.01,
        "Synthetic covariance families only. No real temporal repeats, "
        "3D reconstruction, or superiority claim.",
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
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_figure(
        json.loads(args.report.read_text(encoding="utf-8")),
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
