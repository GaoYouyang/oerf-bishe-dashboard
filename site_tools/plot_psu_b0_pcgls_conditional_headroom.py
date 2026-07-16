#!/usr/bin/env python3
"""Plot the finite-family PCGLS conditional-headroom audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "validation": "#147d70",
    "calibration": "#b14f61",
    "oracle": "#d1902f",
    "neutral": "#52636b",
}


LABELS = {
    "train_global": "Global",
    "train_view_count": "View count",
    "train_view_count_plus_noise": "View + noise",
    "train_family_label": "Family label",
    "split_family_truth_oracle": "Split-family oracle",
    "sample_combined_truth_oracle": "Sample combined oracle",
    "sample_field_truth_oracle": "Sample field oracle",
}


def build_figure(report: dict, output: Path) -> None:
    summaries = report["paired_gain_summary"]
    lookup = {
        (str(row["candidate_method"]), str(row["split"])): row
        for row in summaries
    }
    methods = [
        "train_global",
        "train_view_count",
        "train_view_count_plus_noise",
        "train_family_label",
        "split_family_truth_oracle",
        "sample_field_truth_oracle",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.4))
    fig.patch.set_facecolor("#f7f9f8")
    for axis in axes.flat:
        axis.set_facecolor("#ffffff")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#dfe7e4", linewidth=0.8, alpha=0.8)
        axis.set_axisbelow(True)

    x = np.arange(len(methods), dtype=np.float64)
    for offset, split, color, label in (
        (-0.12, "risk_validation", COLORS["validation"], "Validation"),
        (0.12, "risk_calibration", COLORS["calibration"], "Calibration"),
    ):
        rows = [lookup[(method, split)] for method in methods]
        means = np.asarray([row["mean_field_gain_percent"] for row in rows])
        lows = np.asarray(
            [row["bootstrap_mean_95_interval_percent"][0] for row in rows]
        )
        highs = np.asarray(
            [row["bootstrap_mean_95_interval_percent"][1] for row in rows]
        )
        axes[0, 0].errorbar(
            x + offset,
            means,
            yerr=np.vstack((means - lows, highs - means)),
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            linewidth=1.6,
            markersize=6,
            label=label,
        )
    axes[0, 0].axhline(0.0, color="#88969b", linewidth=1.0)
    axes[0, 0].axhline(
        2.0,
        color="#88969b",
        linewidth=1.0,
        linestyle="--",
    )
    axes[0, 0].set_xticks(
        x,
        [LABELS[method] for method in methods],
        rotation=23,
        ha="right",
    )
    axes[0, 0].set_ylabel("Field error reduction vs static PCGLS-4 (%)")
    axes[0, 0].set_title(
        "A  Headroom exists, but simple observable strata do not transfer",
        loc="left",
        fontsize=11.5,
        fontweight="bold",
    )
    axes[0, 0].legend(frameon=False)

    tail_methods = [
        "train_view_count",
        "train_view_count_plus_noise",
        "train_family_label",
        "sample_field_truth_oracle",
    ]
    tail_x = np.arange(len(tail_methods), dtype=np.float64)
    width = 0.34
    for offset, split, color, label in (
        (-width / 2, "risk_validation", COLORS["validation"], "Validation p10"),
        (width / 2, "risk_calibration", COLORS["calibration"], "Calibration p10"),
    ):
        values = [
            lookup[(method, split)]["p10_field_gain_percent"]
            for method in tail_methods
        ]
        axes[0, 1].bar(
            tail_x + offset,
            values,
            width,
            color=color,
            label=label,
        )
    axes[0, 1].axhline(0.0, color="#88969b", linewidth=1.0)
    axes[0, 1].set_xticks(
        tail_x,
        [LABELS[method] for method in tail_methods],
        rotation=18,
        ha="right",
    )
    axes[0, 1].set_ylabel("Paired field gain p10 (%)")
    axes[0, 1].set_title(
        "B  Only the per-sample oracle removes the negative field tail",
        loc="left",
        fontsize=11.5,
        fontweight="bold",
    )
    axes[0, 1].legend(frameon=False)

    scatter_methods = [
        "train_global",
        "train_view_count",
        "train_view_count_plus_noise",
        "train_family_label",
        "sample_field_truth_oracle",
    ]
    for method in scatter_methods:
        for split, marker, color in (
            ("risk_validation", "o", COLORS["validation"]),
            ("risk_calibration", "s", COLORS["calibration"]),
        ):
            row = lookup[(method, split)]
            measurement = row["secondary_metric_gain"][
                "measurement_relative_l2"
            ]["mean_gain_percent"]
            field = row["mean_field_gain_percent"]
            axes[1, 0].scatter(
                measurement,
                field,
                marker=marker,
                color=color,
                s=56,
            )
            axes[1, 0].annotate(
                f"{LABELS[method]} {'V' if split.endswith('validation') else 'C'}",
                (measurement, field),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7.7,
            )
    axes[1, 0].axhline(0.0, color="#88969b", linewidth=1.0)
    axes[1, 0].axvline(0.0, color="#88969b", linewidth=1.0)
    axes[1, 0].set_xlabel("Measurement residual reduction (%)")
    axes[1, 0].set_ylabel("Field error reduction (%)")
    axes[1, 0].set_title(
        "C  Projection fit is not a reliable selector for the 3-D field",
        loc="left",
        fontsize=11.5,
        fontweight="bold",
    )

    family_rows = report["train_selection"]["family_label_non_deployable"]
    candidates = {
        str(row["candidate_id"]): row
        for row in report["candidate_grid"]["candidates"]
    }
    family_names = [str(row["stratum"][0]) for row in family_rows]
    family_labels = [
        name.replace("_", "\n") for name in family_names
    ]
    strengths = [
        float(candidates[str(row["candidate_id"])]["strength"])
        for row in family_rows
    ]
    pattern_colors = {
        "isotropic": "#147d70",
        "x_high": "#d1902f",
    }
    colors = [
        pattern_colors.get(
            str(candidates[str(row["candidate_id"])]["axis_pattern"]),
            "#52636b",
        )
        for row in family_rows
    ]
    axes[1, 1].bar(
        np.arange(len(family_rows)),
        strengths,
        color=colors,
        width=0.68,
    )
    for index, row in enumerate(family_rows):
        candidate = candidates[str(row["candidate_id"])]
        axes[1, 1].text(
            index,
            strengths[index] + 0.08,
            (
                f"{candidate['axis_pattern']}\n"
                f"eps={float(candidate['epsilon']):g}"
            ),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[1, 1].set_xticks(
        np.arange(len(family_rows)),
        family_labels,
        fontsize=8.5,
    )
    axes[1, 1].set_ylim(0.0, max(strengths) + 1.25)
    axes[1, 1].set_ylabel("Selected Sobolev strength")
    axes[1, 1].set_title(
        "D  Synthetic morphology labels select distinct spectral experts",
        loc="left",
        fontsize=11.5,
        fontweight="bold",
    )

    fig.suptitle(
        "PCGLS conditional headroom: learn morphology, not view count",
        x=0.055,
        y=0.99,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color="#1f3438",
    )
    fig.text(
        0.055,
        0.952,
        (
            "105 fixed-SPD candidates, identical 4F+4AT budget; train-only "
            "selection transfers to validation/calibration, fresh remains excluded."
        ),
        ha="left",
        fontsize=10,
        color="#53666b",
    )
    fig.tight_layout(rect=(0.04, 0.04, 0.99, 0.93), h_pad=2.3, w_pad=1.8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    build_figure(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
