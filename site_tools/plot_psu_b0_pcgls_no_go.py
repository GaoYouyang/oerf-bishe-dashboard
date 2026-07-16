#!/usr/bin/env python3
"""Plot the same-budget Sobolev-PCGLS no-go result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "pcgls": "#117a65",
    "learned": "#b04a5a",
    "neutral": "#52616b",
    "light": "#d8e2e6",
}


def build_figure(report: dict, output: Path) -> None:
    summaries = report["paired_split_summary"]
    development = [
        row for row in summaries if row["split"].startswith("risk_")
    ]
    fresh = [
        row for row in summaries if row["split"].startswith("fresh_")
    ]
    frontier = report["pooled_fresh_frontier"]
    labels = {
        "risk_validation": "Validation",
        "risk_calibration": "Calibration",
        "fresh_iid_support": "IID",
        "fresh_family_ood": "Family",
        "fresh_correlated_noise_ood": "Corr. noise",
        "fresh_family_noise_ood": "Family+noise",
        "fresh_geometry_ood": "Geometry",
        "fresh_joint_ood": "Joint",
        "fresh_exact_operator_control": "Exact op.",
    }
    method_labels = {
        "sobolev_selected": "Sobolev SD",
        "raw_learned_seed_mean": "Learned SD",
        "gated_learned_seed_mean": "Gated SD",
        "pcgls_3_selected": "PCGLS-3",
        "pcgls_4_selected": "PCGLS-4",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2))
    fig.patch.set_facecolor("#f7f9f8")
    for axis in axes.flat:
        axis.set_facecolor("#ffffff")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#dfe7e4", linewidth=0.8, alpha=0.8)
        axis.set_axisbelow(True)

    def interval_panel(axis, values, title):
        x = np.arange(len(values))
        mean = np.asarray(
            [row["pcgls_relative_error_reduction_mean_percent"] for row in values]
        )
        low = np.asarray(
            [row["bootstrap_mean_95_interval_percent"][0] for row in values]
        )
        high = np.asarray(
            [row["bootstrap_mean_95_interval_percent"][1] for row in values]
        )
        axis.axhline(0.0, color="#88969b", linewidth=1.0)
        axis.errorbar(
            x,
            mean,
            yerr=np.vstack((mean - low, high - mean)),
            fmt="o",
            markersize=7,
            color=COLORS["pcgls"],
            ecolor=COLORS["pcgls"],
            capsize=4,
            linewidth=1.8,
        )
        axis.set_xticks(x, [labels[row["split"]] for row in values])
        axis.set_ylabel("PCGLS error reduction vs learned (%)")
        axis.set_title(title, loc="left", fontsize=12, fontweight="bold")
        for index, value in enumerate(mean):
            axis.annotate(
                f"{value:.1f}%",
                (index, high[index]),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color="#24534a",
            )

    interval_panel(
        axes[0, 0],
        development,
        "A  Validation-selected PCGLS persists on calibration",
    )
    interval_panel(
        axes[0, 1],
        fresh,
        "B  Opened stress splits: descriptive, not fresh evidence",
    )
    axes[0, 1].tick_params(axis="x", rotation=24)

    x = np.arange(len(summaries))
    wins = np.asarray([row["pcgls_win_rate"] for row in summaries])
    axes[1, 0].bar(
        x,
        100.0 * wins,
        color=[
            COLORS["neutral"] if row in development else COLORS["pcgls"]
            for row in summaries
        ],
        width=0.68,
    )
    axes[1, 0].axhline(50.0, color="#88969b", linewidth=1.0, linestyle="--")
    axes[1, 0].set_xticks(
        x,
        [labels[row["split"]] for row in summaries],
        rotation=28,
        ha="right",
    )
    axes[1, 0].set_ylim(0.0, 108.0)
    axes[1, 0].set_ylabel("Fields won by PCGLS-4 (%)")
    axes[1, 0].set_title(
        "C  Paired field wins against the three-seed learned mean",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )
    for index, row in enumerate(summaries):
        axes[1, 0].text(
            index,
            100.0 * wins[index] + 2.0,
            f"{row['pcgls_win_count']}/{row['sample_count']}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    for row in frontier:
        marker = "D" if row["method"] == "pcgls_4_selected" else "o"
        color = (
            COLORS["pcgls"]
            if row["method"].startswith("pcgls")
            else (
                COLORS["learned"]
                if "learned" in row["method"]
                else COLORS["neutral"]
            )
        )
        axes[1, 1].scatter(
            row["total_operator_calls"],
            row["field_relative_l2_mean"],
            s=90 if marker == "D" else 65,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        axes[1, 1].annotate(
            method_labels[row["method"]],
            (
                row["total_operator_calls"],
                row["field_relative_l2_mean"],
            ),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=9,
        )
    axes[1, 1].set_xticks([6, 8])
    axes[1, 1].set_xlim(5.5, 8.8)
    axes[1, 1].set_xlabel("Forward + adjoint calls")
    axes[1, 1].set_ylabel("Pooled opened-fresh field relative L2")
    axes[1, 1].set_title(
        "D  Same-call frontier (168 opened diagnostic fields)",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )

    fig.suptitle(
        "Strong-baseline audit: learned steepest direction is not competitive",
        x=0.06,
        y=0.99,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color="#1f3438",
    )
    fig.text(
        0.06,
        0.952,
        (
            "Sobolev-PCGLS-4 uses 4 forward + 4 adjoint calls and zero trainable "
            "parameters; intervals are paired field bootstrap intervals."
        ),
        ha="left",
        fontsize=10,
        color="#53666b",
    )
    fig.tight_layout(rect=(0.04, 0.04, 0.99, 0.93), h_pad=2.2, w_pad=1.8)
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
