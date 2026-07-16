#!/usr/bin/env python3
"""Plot the fresh multi-seed detector-whitened PCGLS development pilot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "dg_covgate": "#147d70",
    "oracle_covariance": "#3578a8",
    "wrong_graph_assumption": "#b14f61",
    "diagonal_shrinkage": "#d1902f",
    "component_iid": "#66777e",
    "grid": "#dfe7e4",
    "ink": "#52636b",
}
LABELS = {
    "dg_covgate": "DG-CovGate",
    "oracle_covariance": "Oracle covariance",
    "wrong_graph_assumption": "Wrong graph",
    "diagonal_shrinkage": "Diagonal shrinkage",
    "component_iid": "Component IID",
}


def _summary_lookup(report: dict) -> dict[tuple[str, str], dict]:
    return {
        (str(row["noise_family"]), str(row["method"])): row
        for row in report["summaries"]
    }


def build_figure(
    report: dict,
    replicate_rows: list[dict[str, str]],
    output: Path,
) -> None:
    lookup = _summary_lookup(report)
    figure, axes = plt.subplots(2, 2, figsize=(13.6, 9.0))
    figure.patch.set_facecolor("#f7f9f8")
    for axis in axes.flat:
        axis.set_facecolor("#ffffff")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(color=COLORS["grid"], linewidth=0.8, alpha=0.85)
        axis.set_axisbelow(True)

    methods = (
        "dg_covgate",
        "oracle_covariance",
        "diagonal_shrinkage",
        "wrong_graph_assumption",
    )
    x = np.arange(len(methods))
    means = [
        lookup[("graph_heat", method)][
            "replicate_mean_field_gain_percent"
        ]
        for method in methods
    ]
    lower = [
        lookup[("graph_heat", method)][
            "replicate_mean_field_gain_ci95_lower"
        ]
        for method in methods
    ]
    upper = [
        lookup[("graph_heat", method)][
            "replicate_mean_field_gain_ci95_upper"
        ]
        for method in methods
    ]
    axes[0, 0].bar(
        x,
        means,
        color=[COLORS[method] for method in methods],
        alpha=0.88,
    )
    axes[0, 0].errorbar(
        x,
        means,
        yerr=[
            np.asarray(means) - np.asarray(lower),
            np.asarray(upper) - np.asarray(means),
        ],
        fmt="none",
        ecolor=COLORS["ink"],
        capsize=4,
        linewidth=1.2,
    )
    axes[0, 0].axhline(0.0, color=COLORS["ink"], linewidth=1)
    axes[0, 0].set(
        title="A  Mean 3-D field gain under graph-correlated noise",
        ylabel="Gain vs component-IID PCGLS-4 (%)",
        xticks=x,
        xticklabels=[LABELS[method] for method in methods],
    )
    axes[0, 0].tick_params(axis="x", rotation=15)

    family_rows = [
        row
        for row in report["family_summaries"]
        if row["noise_family"] == "graph_heat"
        and row["method"] in {
            "dg_covgate",
            "oracle_covariance",
        }
    ]
    families = sorted(
        {str(row["reaction_family"]) for row in family_rows}
    )
    y = np.arange(len(families))
    for offset, method in zip(
        (-0.13, 0.13),
        ("dg_covgate", "oracle_covariance"),
        strict=True,
    ):
        selected = {
            str(row["reaction_family"]): row
            for row in family_rows
            if row["method"] == method
        }
        axes[0, 1].plot(
            [selected[family]["field_gain_percent_mean"] for family in families],
            y + offset,
            marker="o",
            linewidth=1.8,
            color=COLORS[method],
            label=LABELS[method],
        )
    axes[0, 1].axvline(0.0, color=COLORS["ink"], linewidth=1)
    axes[0, 1].set(
        title="B  Morphology dependence exposes the unsafe tail",
        xlabel="Mean field gain vs component IID (%)",
        yticks=y,
        yticklabels=[value.replace("_", " ") for value in families],
    )
    axes[0, 1].legend(frameon=False, fontsize=8.5)

    scatter_methods = (
        "dg_covgate",
        "oracle_covariance",
        "diagonal_shrinkage",
        "wrong_graph_assumption",
        "component_iid",
    )
    for method in scatter_methods:
        row = lookup[("graph_heat", method)]
        axes[1, 0].scatter(
            row["field_gain_percent_p10"],
            100.0 * row["field_harm_over_one_percent_rate"],
            s=75,
            color=COLORS[method],
            label=LABELS[method],
            zorder=3,
        )
    axes[1, 0].axvline(
        -0.5,
        color=COLORS["wrong_graph_assumption"],
        linestyle="--",
        linewidth=1,
        label="preregistered p10 floor",
    )
    axes[1, 0].axhline(
        10.0,
        color=COLORS["diagonal_shrinkage"],
        linestyle=":",
        linewidth=1,
        label="preregistered harm ceiling",
    )
    axes[1, 0].set(
        title="C  The mean improves, but the safety gate fails",
        xlabel="p10 paired field gain (%)",
        ylabel="Fields harmed by more than 1% (%)",
    )
    axes[1, 0].legend(frameon=False, fontsize=7.7, ncol=2)

    replicate_methods = (
        "dg_covgate",
        "oracle_covariance",
        "wrong_graph_assumption",
    )
    distributions = []
    for method in replicate_methods:
        distributions.append(
            [
                float(row["field_gain_percent_mean"])
                for row in replicate_rows
                if row["noise_family"] == "graph_heat"
                and row["method"] == method
            ]
        )
    boxes = axes[1, 1].boxplot(
        distributions,
        tick_labels=[LABELS[method] for method in replicate_methods],
        patch_artist=True,
        showmeans=True,
        meanline=True,
    )
    for patch, method in zip(
        boxes["boxes"],
        replicate_methods,
        strict=True,
    ):
        patch.set_facecolor(COLORS[method])
        patch.set_alpha(0.55)
    axes[1, 1].axhline(0.0, color=COLORS["ink"], linewidth=1)
    axes[1, 1].set(
        title="D  Replicate-clustered gain across 16 fresh seeds",
        ylabel="Mean gain across eight reaction morphologies (%)",
    )
    axes[1, 1].tick_params(axis="x", rotation=10)

    decision = report["decision"]
    figure.suptitle(
        "DG-WPCGLS: correct covariance helps on average, but fails the tail-risk gate",
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.012,
        "Real PSU detector geometry; analytic reaction morphologies and "
        "synthetic noise only. "
        f"Verdict: {decision['verdict']}. No superiority claim.",
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
    parser.add_argument("--replicate-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.replicate_csv.open(newline="", encoding="utf-8") as handle:
        replicate_rows = list(csv.DictReader(handle))
    build_figure(
        json.loads(args.report.read_text(encoding="utf-8")),
        replicate_rows,
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
