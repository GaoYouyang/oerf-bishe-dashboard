from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / (
    "docs/poolfire_exact_k1_dual_view_ridge_v142_4_public_summary.json"
)
OUTPUT = ROOT / "assets/figures/poolfire_exact_k1_dual_view_ridge_v142_4.png"


def build_figure() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    formal = data["arm_results"]["formal_view"]
    trajectories = formal["trajectory_summary"]
    labels = [row["group"].replace("kw_size", " kW / size ") for row in trajectories]
    metric_specs = [
        ("field_relative_l2", "Field", "#28786a"),
        ("gradient_relative_l2", "Full gradient", "#cb6d51"),
        ("interior_gradient_relative_l2", "Interior gradient", "#547da3"),
        ("reported_observation_relative_l2", "Observation", "#d5a33f"),
    ]

    fig = plt.figure(figsize=(15.5, 8.8), facecolor="#f4f7f6")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.7, 1], wspace=0.28)

    ax = fig.add_subplot(grid[0, 0])
    x = np.arange(len(labels))
    width = 0.19
    for index, (metric, label, color) in enumerate(metric_specs):
        values = [row["metrics"][metric]["p90_higher"] for row in trajectories]
        ax.bar(
            x + (index - 1.5) * width,
            values,
            width,
            label=label,
            color=color,
        )
    ax.axhline(1.02, color="#8f3d35", linestyle="--", linewidth=2, label="Frozen p90 gate")
    ax.set_xticks(x, labels, rotation=14, ha="right")
    ax.set_ylim(0.96, 1.98)
    ax.set_ylabel("p90-higher error ratio to Zero-CGLS K4")
    ax.set_title(
        "A. Every complete trajectory misses the matched-accuracy gate",
        loc="left",
        fontsize=13,
        weight="bold",
    )
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper left")

    right = fig.add_subplot(grid[0, 1])
    names = ["Formal feature\nview", "Independent\nfeature view", "Joint-LS\ncontrol"]
    arms = ["formal_view", "independent_view", "joint_ls_warm_restart_k1"]
    maxima = [data["arm_results"][name]["maximum_metric_ratio"] for name in arms]
    colors = ["#28786a", "#547da3", "#8d6b9f"]
    bars = right.bar(np.arange(3), maxima, color=colors, width=0.62)
    right.axhline(1.05, color="#8f3d35", linestyle="--", linewidth=2)
    right.set_xticks(np.arange(3), names)
    right.set_ylim(0.9, 2.05)
    right.set_ylabel("Worst per-cell metric ratio")
    right.set_title(
        "B. Two implementations agree on the failure",
        loc="left",
        fontsize=13,
        weight="bold",
    )
    right.grid(axis="y", alpha=0.2)
    for bar, arm, value in zip(bars, arms, maxima, strict=True):
        passed = data["arm_results"][arm]["passing_cells"]
        right.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.035,
            f"{value:.3f}\n{passed}/3700 pass",
            ha="center",
            va="bottom",
            fontsize=9.5,
            weight="bold",
        )

    fig.suptitle(
        "v142.4: shared linear K1-dual predictor fails complete-trajectory transfer",
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
        "3,700 opened PoolFire proxy cells | 3A+3Aᵀ candidate vs 4A+4Aᵀ Zero-CGLS K4 | dual-view maximum ratio difference 2.59e-11",
        fontsize=10.4,
        color="#52636b",
    )
    fig.text(
        0.055,
        0.025,
        "Negative result: fixed-teacher capacity does not become a deployable cross-trajectory linear predictor. algorithm_breakthrough=false",
        fontsize=10.2,
        color="#6b3430",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
