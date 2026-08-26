from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_two_color_additive_schwarz_v267_public_summary.json"
OUTPUT = ROOT / "assets/figures/blastnet_case19_two_color_additive_schwarz_v267.png"


def label_bars(axis: plt.Axes, bars: plt.Container, *, suffix: str = "") -> None:
    for bar in bars:
        value = int(round(bar.get_height()))
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 8,
            f"{value}{suffix}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    results = data["results"]
    interference = data["cross_color_interference"]

    fig = plt.figure(figsize=(18.5, 7.2), facecolor="#f3f5f4")
    grid = fig.add_gridspec(1, 3, width_ratios=(1.28, 0.82, 1.05), wspace=0.29)
    matched_ax = fig.add_subplot(grid[0, 0])
    failure_ax = fig.add_subplot(grid[0, 1])
    interference_ax = fig.add_subplot(grid[0, 2])

    labels = ("Two-color", "Full-row\ncontrol", "One-half", "v258 parent", "K15", "K16 ref.")
    values = (
        results["primary_matched_pass_cells"],
        results["same_work_full_row_control_matched_pass_cells"],
        results["sealed_one_half_control_matched_pass_cells"],
        results["sealed_parent_matched_pass_cells"],
        results["k15_control_matched_pass_cells"],
        results["k16_reference_matched_pass_cells"],
    )
    colors = ("#c94f4f", "#3279a8", "#7c8a96", "#9aa4ac", "#c1c7cc", "#3d8a61")
    matched_bars = matched_ax.bar(np.arange(len(labels)), values, color=colors, width=0.72)
    label_bars(matched_ax, matched_bars)
    matched_ax.set_xticks(np.arange(len(labels)), labels)
    matched_ax.set_ylim(0, 480)
    matched_ax.set_ylabel("K16-matched cells out of 429")
    matched_ax.set_title("Matched accuracy does not pass", loc="left", fontweight="bold")
    matched_ax.grid(axis="y", alpha=0.2)

    failure_values = (
        results["matched_failure_counts"]["observation"],
        results["primary_matched_pass_cells"],
    )
    failure_bars = failure_ax.bar(
        np.arange(2),
        failure_values,
        color=("#c94f4f", "#3d8a61"),
        width=0.64,
    )
    label_bars(failure_ax, failure_bars)
    failure_ax.set_xticks(np.arange(2), ("Observation-\nonly failures", "Matched"))
    failure_ax.set_ylim(0, 480)
    failure_ax.set_ylabel("Cell count")
    failure_ax.set_title("All 427 misses are observation-only", loc="left", fontweight="bold")
    failure_ax.grid(axis="y", alpha=0.2)

    interference_values = (
        interference["each_color_own_local_step_improved_cells"],
        interference["combined_state_worse_than_each_color_own_diagonal_cells"]["color_0"],
        interference["global_residual_vs_parent"]["worsened_cells"],
    )
    interference_labels = ("Each local step\nimproves", "Combined worse\nthan local", "Full residual worse\nthan parent")
    interference_bars = interference_ax.bar(
        np.arange(3),
        interference_values,
        color=("#3d8a61", "#d08a2f", "#c94f4f"),
        width=0.66,
    )
    label_bars(interference_ax, interference_bars)
    interference_ax.set_xticks(np.arange(3), interference_labels)
    interference_ax.set_ylim(0, 480)
    interference_ax.set_ylabel("Cells out of 429")
    interference_ax.set_title("Synchronous blocks interfere", loc="left", fontweight="bold")
    interference_ax.grid(axis="y", alpha=0.2)

    fig.suptitle(
        "v267 | Exact two-color local steps fail under complete-observation coupling",
        x=0.045,
        y=0.985,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.045,
        0.018,
        "Independent 24/24 | candidate 16A+15AT vs K16 16A+16AT | post-open negative evidence; no wall/RSS or external claim",
        fontsize=10,
        color="#40474b",
    )
    for axis in (matched_ax, failure_ax, interference_ax):
        axis.set_facecolor("#ffffff")
        axis.spines[["top", "right"]].set_visible(False)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
