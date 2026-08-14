"""Build the public v145 global camera-state diagnostic figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_k1_dual_global_camera_state_v145_public_summary.json"
OUTPUT = ROOT / "assets/figures/poolfire_k1_dual_global_camera_state_v145.png"


def main() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    if payload["scientific_decision"] != "FAIL_GLOBAL_CAMERA_STATE_IDENTIFIABILITY_V145":
        raise RuntimeError("refusing to plot an unexpected v145 decision")

    methods = payload["methods"]
    labels = ["Count mean", "Cross\nmoments", "Cross\ncoupled", "Within*\nmoments", "Within*\ncoupled"]
    keys = [
        "camera_count_mean_control",
        "cross_trajectory_moments_knn",
        "cross_trajectory_observable_pose_coupled_knn",
        "same_trajectory_moments_diagnostic",
        "same_trajectory_observable_pose_coupled_diagnostic",
    ]
    medians = [methods[key]["scale_invariant_relative_l2_median"] for key in keys]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 10,
    })
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 7.6))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.16, top=0.84, wspace=0.30)
    fig.patch.set_facecolor("#f7f8f6")
    colors = ["#7a8793", "#c45d4b", "#b8493d", "#3f7f78", "#276c66"]

    ax = axes[0]
    y = np.arange(len(labels))
    bars = ax.barh(y, medians, color=colors, height=0.62)
    ax.axvline(0.45, color="#212529", linestyle=(0, (4, 3)), linewidth=1.5, label="Frozen gate 0.45")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.04)
    ax.set_xlabel("Median scale-invariant relative-L2")
    ax.set_title("A  Global signatures miss the target", loc="left", weight="bold")
    ax.legend(frameon=False, loc="lower right")
    for bar, value in zip(bars, medians, strict=True):
        ax.text(value - 0.015, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", ha="right", va="center", color="white", weight="bold")

    ax = axes[1]
    trajectories = ["p14", "p22", "p33", "p45", "p58"]
    cross = list(methods["cross_trajectory_observable_pose_coupled_knn"]["trajectory_p90_higher"].values())
    within = list(methods["same_trajectory_observable_pose_coupled_diagnostic"]["trajectory_p90_higher"].values())
    x = np.arange(len(trajectories))
    width = 0.35
    ax.bar(x - width / 2, cross, width, label="Cross coupled", color="#b8493d")
    ax.bar(x + width / 2, within, width, label="Within* coupled", color="#276c66")
    ax.axhline(0.35, color="#212529", linestyle=(0, (4, 3)), linewidth=1.5, label="Frozen gate 0.35")
    ax.set_xticks(x, trajectories)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Trajectory p90-higher error")
    ax.set_title("B  Every trajectory misses the tail gate", loc="left", weight="bold")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[2]
    audit = payload["post_result_camera_count_audit"]
    audit_labels = ["Cross\nmoments", "Cross\ncoupled", "Within*\nmoments", "Within*\ncoupled"]
    fractions = np.array([
        audit["cross_trajectory_moments_same_count_neighbor_edge_fraction"],
        audit["cross_trajectory_coupled_same_count_neighbor_edge_fraction"],
        audit["same_trajectory_moments_same_count_neighbor_edge_fraction"],
        audit["same_trajectory_coupled_same_count_neighbor_edge_fraction"],
    ]) * 100
    bars = ax.bar(np.arange(4), fractions, color=["#c45d4b", "#b8493d", "#3f7f78", "#276c66"], width=0.64)
    ax.axhline(95, color="#212529", linestyle=(0, (4, 3)), linewidth=1.5, label="95% reference")
    ax.set_xticks(np.arange(4), audit_labels)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Same-camera-count neighbor edges (%)")
    ax.set_title("C  Count mixing is not the sole cause", loc="left", weight="bold")
    ax.legend(frameon=False, loc="upper left")
    for bar, value in zip(bars, fractions, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.1f}%", ha="center", va="bottom", weight="bold")

    for ax in axes:
        ax.set_facecolor("#ffffff")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#d9ddda", linewidth=0.7, alpha=0.75)
        ax.set_axisbelow(True)

    fig.suptitle("v145 | Global camera-set state does not identify the shared action target", fontsize=18, weight="bold", x=0.04, y=0.97, ha="left")
    fig.text(0.04, 0.045, "All five methods: 0/20 sentinels, 0/3700 cells, 0/5 trajectories | +0A/+0A^T | no GPU | algorithm_breakthrough=false", fontsize=10, color="#39434a")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
