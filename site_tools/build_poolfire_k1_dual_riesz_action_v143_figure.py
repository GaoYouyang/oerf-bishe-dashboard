from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_k1_dual_riesz_action_v143_public_summary.json"
OUTPUT = ROOT / "assets/figures/poolfire_k1_dual_riesz_action_v143.png"


def build_figure() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    target = data["target_space"]
    controls = data["diagnostic_comparators"]

    fig = plt.figure(figsize=(15.5, 8.8), facecolor="#f4f7f6")
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 1.55], wspace=0.28)

    left = fig.add_subplot(grid[0, 0])
    names = ["Raw coefficient\nridge", "Joint-LS\ncontrol", "Riesz-action\nridge"]
    values = [
        controls["median_raw_v142_4_coefficient_error"],
        controls["median_joint_ls_error"],
        controls["median_riesz_action_prediction_error"],
    ]
    colors = ["#547da3", "#8d6b9f", "#cb6d51"]
    bars = left.bar(np.arange(3), values, color=colors, width=0.62)
    left.axhline(
        target["predicted_error_each_cell_gate"],
        color="#8f3d35",
        linestyle="--",
        linewidth=2,
        label="Frozen per-cell error gate",
    )
    left.set_xticks(np.arange(3), names)
    left.set_ylim(0, 1.08)
    left.set_ylabel("Median scale-invariant dual error")
    left.set_title(
        "A. Action coordinates still fail",
        loc="left",
        fontsize=12.6,
        weight="bold",
    )
    left.grid(axis="y", alpha=0.2)
    left.legend(frameon=False, fontsize=9, loc="upper left")
    for bar, value in zip(bars, values, strict=True):
        left.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )

    right = fig.add_subplot(grid[0, 1])
    trajectory_p90 = target["trajectory_p90_higher"]
    labels = [name.replace("kw_size", " kW / size ") for name in trajectory_p90]
    values = list(trajectory_p90.values())
    bars = right.bar(np.arange(len(labels)), values, color="#cb6d51", width=0.68)
    right.axhline(
        target["trajectory_p90_higher_gate"],
        color="#8f3d35",
        linestyle="--",
        linewidth=2,
        label="Frozen trajectory p90 gate",
    )
    right.set_xticks(np.arange(len(labels)), labels, rotation=13, ha="right")
    right.set_ylim(0, 1.08)
    right.set_ylabel("Held-out Riesz-action error, p90-higher")
    right.set_title(
        "B. All five trajectory tails fail",
        loc="left",
        fontsize=12.6,
        weight="bold",
    )
    right.grid(axis="y", alpha=0.2)
    right.legend(frameon=False, fontsize=9, loc="upper left")
    for bar, value in zip(bars, values, strict=True):
        right.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.016,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=8.8,
            weight="bold",
        )

    fig.suptitle(
        "v143: shared-linear Riesz-action prediction fails the mechanism sentinel",
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
        "20 preregistered sentinels | 5 opened PoolFire trajectories | 5/7/9/12 cameras | no additional exact A/A^T calls",
        fontsize=10.4,
        color="#52636b",
    )
    fig.text(
        0.055,
        0.025,
        "Oracle inverse max error 0.00308 <= 0.02, but prediction cosine median 0.0269 and 0/20 sentinels pass. No GPU or neural rescue authorized.",
        fontsize=10.2,
        color="#6b3430",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
