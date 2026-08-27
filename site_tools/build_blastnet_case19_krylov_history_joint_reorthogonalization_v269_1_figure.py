from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_krylov_history_joint_reorthogonalization_v269_1_public_summary.json"
OUTPUT = ROOT / "assets/figures/blastnet_case19_krylov_history_joint_reorthogonalization_v269_1.png"


def _label_counts(axis: plt.Axes, bars: plt.Container) -> None:
    for bar in bars:
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.28,
            f"{int(round(bar.get_height()))}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    diagnostic = data["diagnostic_results"]
    independent = data["independent_validation"]

    fig = plt.figure(figsize=(18.5, 7.2), facecolor="#f3f5f4")
    grid = fig.add_gridspec(1, 3, width_ratios=(0.9, 1.05, 1.25), wspace=0.3)
    validity_ax = fig.add_subplot(grid[0, 0])
    count_ax = fig.add_subplot(grid[0, 1])
    ratio_ax = fig.add_subplot(grid[0, 2])

    validity_bars = validity_ax.bar(
        np.arange(2),
        (data["formal_validation"]["checks_passed"], independent["checks_passed"]),
        color=("#3d8a61", "#d08a2f"),
        width=0.62,
    )
    _label_counts(validity_ax, validity_bars)
    validity_ax.scatter(
        np.arange(2),
        (data["formal_validation"]["checks_total"], independent["checks_total"]),
        color="#202629",
        marker="_",
        s=550,
        linewidths=2.2,
        label="required",
    )
    validity_ax.set_xticks(np.arange(2), ("Formal", "Independent"))
    validity_ax.set_ylim(0, 32)
    validity_ax.set_ylabel("Validity checks")
    validity_ax.set_title("One exact-equality check blocks validation", loc="left", fontweight="bold")
    validity_ax.text(
        0.5,
        0.08,
        "observation difference\n5.20e-16",
        transform=validity_ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        color="#9b5a14",
        fontweight="bold",
    )
    validity_ax.grid(axis="y", alpha=0.2)

    count_values = (
        diagnostic["primary_absolute_pass_cells"],
        diagnostic["primary_matched_pass_cells"],
        diagnostic["sealed_v265_single_half_matched_pass_cells"],
        diagnostic["sealed_v267_full_row_matched_pass_cells"],
    )
    count_bars = count_ax.bar(
        np.arange(4),
        count_values,
        color=("#3d8a61", "#c94f4f", "#d08a2f", "#3279a8"),
        width=0.68,
    )
    _label_counts(count_ax, count_bars)
    count_ax.set_xticks(
        np.arange(4),
        ("Candidate\nabsolute", "Candidate\nmatched", "Single-half\ncontrol", "Full-row\ncontrol"),
    )
    count_ax.set_ylim(0, 15)
    count_ax.set_ylabel("Frame-zero cells out of 13")
    count_ax.set_title("Performance counts are diagnostic only", loc="left", fontweight="bold")
    count_ax.grid(axis="y", alpha=0.2)

    metric_labels = ("Field", "Gradient", "Interior\ngradient", "Observation")
    p90 = np.asarray(diagnostic["matched_ratio_p90_higher"], dtype=np.float64)
    worst = np.asarray(diagnostic["matched_ratio_worst"], dtype=np.float64)
    positions = np.arange(4)
    ratio_ax.bar(positions - 0.18, p90, width=0.36, color="#3279a8", label="p90-higher")
    ratio_ax.bar(positions + 0.18, worst, width=0.36, color="#c94f4f", label="worst")
    ratio_ax.axhline(1.05, color="#202629", linewidth=1.4, linestyle="--", label="matched limit 1.05")
    ratio_ax.set_xticks(positions, metric_labels)
    ratio_ax.set_ylim(0, 1.48)
    ratio_ax.set_ylabel("Candidate / K16 error ratio")
    ratio_ax.set_title("Diagnostic observation tail misses matching", loc="left", fontweight="bold")
    ratio_ax.legend(frameon=False, fontsize=9)
    ratio_ax.grid(axis="y", alpha=0.2)

    fig.suptitle(
        "v269.1 | Cached-history joint reorthogonalization remains inconclusive",
        x=0.045,
        y=0.985,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.045,
        0.018,
        "Formal 21/21 | independent 28/29 | diagnostic candidate 13/13 absolute, 0/13 matched | no rerun, training, GPU, or resource claim",
        fontsize=10,
        color="#40474b",
    )
    for axis in (validity_ax, count_ax, ratio_ax):
        axis.set_facecolor("#ffffff")
        axis.spines[["top", "right"]].set_visible(False)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
