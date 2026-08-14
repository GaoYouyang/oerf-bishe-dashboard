from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_k1_dual_local_identifiability_v144_public_summary.json"
OUTPUT = ROOT / "assets/figures/poolfire_k1_dual_local_identifiability_v144.png"


def build_figure() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    methods = data["methods"]
    gates = data["frozen_gates"]
    cross = methods["cross_trajectory_knn"]
    within = methods["same_trajectory_diagnostic_knn"]
    mean = methods["structural_mean_control"]

    fig = plt.figure(figsize=(16.2, 9.2), facecolor="#f4f7f6")
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.82, 1.55], wspace=0.32)
    colors = ["#2e7180", "#73954f", "#c26954"]

    error_axis = fig.add_subplot(grid[0, 0])
    names = ["Cross\ntrajectory", "Within\ntrajectory*", "Structural\nmean"]
    medians = [
        cross["scale_invariant_relative_l2_median"],
        within["scale_invariant_relative_l2_median"],
        mean["scale_invariant_relative_l2_median"],
    ]
    bars = error_axis.bar(np.arange(3), medians, color=colors, width=0.62)
    error_axis.axhline(
        gates["scale_invariant_relative_l2_each_sentinel_maximum"],
        color="#8f3d35",
        linestyle="--",
        linewidth=2,
        label="Frozen per-sentinel gate",
    )
    error_axis.set_xticks(np.arange(3), names, fontsize=9.1)
    error_axis.set_ylim(0, 1.02)
    error_axis.set_ylabel("Median scale-invariant target error")
    error_axis.set_title("A. Local targets remain inconsistent", loc="left", fontsize=12.2, weight="bold")
    error_axis.grid(axis="y", alpha=0.2)
    error_axis.legend(frameon=False, fontsize=8.8, loc="upper left")
    for bar, value in zip(bars, medians, strict=True):
        error_axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.022,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9.8,
            weight="bold",
        )

    pass_axis = fig.add_subplot(grid[0, 1])
    pass_counts = [
        cross["sentinel_pass_count"],
        within["sentinel_pass_count"],
        mean["sentinel_pass_count"],
    ]
    bars = pass_axis.bar(np.arange(3), pass_counts, color=colors, width=0.62)
    pass_axis.axhline(20, color="#8f3d35", linestyle="--", linewidth=2, label="Required 20/20")
    pass_axis.set_xticks(np.arange(3), ["Cross", "Within*", "Mean"])
    pass_axis.set_ylim(0, 21.5)
    pass_axis.set_ylabel("Sentinels passing")
    pass_axis.set_title("B. Strict gates", loc="left", fontsize=12.2, weight="bold")
    pass_axis.grid(axis="y", alpha=0.2)
    pass_axis.legend(frameon=False, fontsize=8.8, loc="upper right")
    for bar, value in zip(bars, pass_counts, strict=True):
        pass_axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.55,
            f"{value}/20",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )

    tail_axis = fig.add_subplot(grid[0, 2])
    trajectory_names = list(cross["trajectory_p90_higher"])
    labels = [name.replace("kw_size", " kW / size ") for name in trajectory_names]
    cross_values = [cross["trajectory_p90_higher"][name] for name in trajectory_names]
    within_values = [within["trajectory_p90_higher"][name] for name in trajectory_names]
    x = np.arange(len(labels))
    width = 0.36
    tail_axis.bar(x - width / 2, cross_values, width, color=colors[0], label="Cross-trajectory")
    tail_axis.bar(x + width / 2, within_values, width, color=colors[1], label="Same-trajectory diagnostic*")
    tail_axis.axhline(
        gates["trajectory_p90_higher_maximum"],
        color="#8f3d35",
        linestyle="--",
        linewidth=2,
        label="Frozen trajectory p90 gate",
    )
    tail_axis.set_xticks(x, labels, rotation=14, ha="right")
    tail_axis.set_ylim(0, 1.0)
    tail_axis.set_ylabel("Target error, trajectory p90-higher")
    tail_axis.set_title("C. All five trajectory tails fail", loc="left", fontsize=12.2, weight="bold")
    tail_axis.grid(axis="y", alpha=0.2)
    tail_axis.legend(frameon=False, fontsize=8.5, loc="upper left")

    fig.suptitle(
        "v144: the frozen 155D local-neighborhood hypothesis fails",
        x=0.055,
        y=0.985,
        ha="left",
        fontsize=18.2,
        weight="bold",
        color="#18262d",
    )
    fig.text(
        0.055,
        0.947,
        "20 preregistered sentinels | 5 opened PoolFire trajectories | 5/7/9/12 cameras | fixed k=8 | 0 trainable parameters | +0A/+0A^T",
        fontsize=10.3,
        color="#52636b",
    )
    fig.text(
        0.055,
        0.025,
        "*Same-trajectory neighbors are a post-open identifiability diagnostic, not a deployable model. Cross 1/20; within 8/20; no GPU or neural training authorized.",
        fontsize=10.1,
        color="#6b3430",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
