from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_coarse_residual_galerkin_v268_public_summary.json"
OUTPUT = ROOT / "assets/figures/blastnet_case19_coarse_residual_galerkin_v268.png"


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
    results = data["results"]

    fig = plt.figure(figsize=(18.5, 7.2), facecolor="#f3f5f4")
    grid = fig.add_gridspec(1, 3, width_ratios=(1.05, 1.22, 1.08), wspace=0.3)
    count_ax = fig.add_subplot(grid[0, 0])
    ratio_ax = fig.add_subplot(grid[0, 1])
    failure_ax = fig.add_subplot(grid[0, 2])

    count_labels = ("Candidate\nabsolute", "Candidate\nmatched", "Full-row\ncontrol", "Single-half\ncontrol")
    count_values = (
        results["primary_absolute_pass_cells"],
        results["primary_matched_pass_cells"],
        results["same_work_full_row_control_pass_cells"],
        results["cheaper_single_half_control_pass_cells"],
    )
    count_colors = ("#3d8a61", "#c94f4f", "#3279a8", "#d08a2f")
    count_bars = count_ax.bar(np.arange(4), count_values, color=count_colors, width=0.68)
    _label_counts(count_ax, count_bars)
    count_ax.set_xticks(np.arange(4), count_labels)
    count_ax.set_ylim(0, 15)
    count_ax.set_ylabel("Passing frame-zero cells out of 13")
    count_ax.set_title("Absolute accuracy passes; matching does not", loc="left", fontweight="bold")
    count_ax.grid(axis="y", alpha=0.2)

    metric_labels = ("Field", "Gradient", "Interior\ngradient", "Observation")
    p90 = np.asarray(results["matched_ratio_p90_higher"], dtype=np.float64)
    worst = np.asarray(results["matched_ratio_worst"], dtype=np.float64)
    positions = np.arange(4)
    ratio_ax.bar(positions - 0.18, p90, width=0.36, color="#3279a8", label="p90-higher")
    ratio_ax.bar(positions + 0.18, worst, width=0.36, color="#c94f4f", label="worst")
    ratio_ax.axhline(1.05, color="#202629", linewidth=1.4, linestyle="--", label="matched limit 1.05")
    ratio_ax.set_xticks(positions, metric_labels)
    ratio_ax.set_ylim(0, 1.32)
    ratio_ax.set_ylabel("Candidate / K16 error ratio")
    ratio_ax.set_title("Only observation crosses the matched limit", loc="left", fontweight="bold")
    ratio_ax.legend(frameon=False, fontsize=9)
    ratio_ax.grid(axis="y", alpha=0.2)

    failed = np.asarray(results["failing_observation_ratios"], dtype=np.float64)
    failure_bars = failure_ax.bar(
        np.arange(failed.size),
        failed,
        color=("#d08a2f", "#c94f4f", "#d08a2f", "#d08a2f", "#d08a2f", "#d08a2f"),
        width=0.68,
    )
    failure_ax.axhline(1.05, color="#202629", linewidth=1.4, linestyle="--")
    for bar, value in zip(failure_bars, failed, strict=True):
        failure_ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.012,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=90,
        )
    failure_ax.set_xticks(np.arange(failed.size), tuple(f"Fail {index + 1}" for index in range(failed.size)))
    failure_ax.set_ylim(0.98, 1.25)
    failure_ax.set_ylabel("Observation error ratio")
    failure_ax.set_title("Six rigs fail observation matching", loc="left", fontweight="bold")
    failure_ax.grid(axis="y", alpha=0.2)

    fig.suptitle(
        "v268 | One global coarse residual step is not K16-matched",
        x=0.045,
        y=0.985,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.045,
        0.018,
        "Independent 25/25 | candidate 16A+15AT vs K16 16A+16AT | post-open frame-zero negative evidence; no wall/RSS or external claim",
        fontsize=10,
        color="#40474b",
    )
    for axis in (count_ax, ratio_ax, failure_ax):
        axis.set_facecolor("#ffffff")
        axis.spines[["top", "right"]].set_visible(False)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
