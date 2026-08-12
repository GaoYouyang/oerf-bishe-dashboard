from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_k1_dual_pair_depth_sentinel_v142_public_summary.json"
OUTPUT = ROOT / "assets/figures/poolfire_k1_dual_pair_depth_sentinel_v142.png"


def build_figure() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    cameras = ["5", "7", "9", "12"]
    primary = data["primary_result"]["maximum_metric_ratio_by_camera_count"]
    initializer = data["lower_cost_initializer_diagnostic"][
        "maximum_metric_ratio_by_camera_count"
    ]
    controls = data["same_cost_controls"]["maximum_metric_ratio_by_camera_count"]

    labels = [
        "Initializer only\n2A+2Aᵀ",
        "Warm K1 primary\n3A+3Aᵀ",
        "Zero K3\n3A+3Aᵀ",
        "Scaled BP + K2\n3A+3Aᵀ",
        "Geometry PCGLS K3\n3A+3Aᵀ",
    ]
    series = [
        [initializer[camera] for camera in cameras],
        [primary[camera] for camera in cameras],
        [controls["zero_cgls_k3"][camera] for camera in cameras],
        [controls["scaled_bp_k2"][camera] for camera in cameras],
        [controls["geometry_pcgls_k3"][camera] for camera in cameras],
    ]
    colors = ["#d29b48", "#28786a", "#7c8d95", "#b85c4b", "#6d5d88"]

    fig = plt.figure(figsize=(15.5, 8.8), facecolor="#f3f7f5")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.75, 1], wspace=0.27)

    ax = fig.add_subplot(grid[0, 0])
    x = np.arange(len(cameras))
    width = 0.15
    for index, (label, values, color) in enumerate(
        zip(labels, series, colors, strict=True)
    ):
        offset = (index - 2) * width
        ax.bar(x + offset, values, width, label=label, color=color)
    ax.axhline(1.05, color="#9b3f36", linestyle="--", linewidth=2, label="Frozen gate 1.05")
    ax.set_xticks(x, [f"{camera} cameras" for camera in cameras])
    ax.set_ylim(0.85, 1.46)
    ax.set_ylabel("Worst of four metric ratios to Zero-CGLS K4")
    ax.set_title(
        "A. K1-dual warm start clears four fixed difficult cells",
        loc="left",
        fontsize=13,
        weight="bold",
    )
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper left")

    boundary = fig.add_subplot(grid[0, 1])
    boundary.axis("off")
    boundary.set_title(
        "B. What this result does and does not establish",
        loc="left",
        fontsize=13,
        weight="bold",
        pad=14,
    )
    boxes = [
        (0.96, "4 / 4", "Warm K1 primary passes", "#28786a"),
        (0.75, "4 / 4", "Initializer-only also passes", "#d29b48"),
        (0.54, "0 / 3", "Same-cost control families pass all four", "#b85c4b"),
        (0.33, "pending", "Full 3,700-cell independent gate", "#7c8d95"),
        (0.12, "false", "algorithm_breakthrough", "#9b3f36"),
    ]
    for y, value, label, color in boxes:
        boundary.text(
            0.03,
            y,
            value,
            transform=boundary.transAxes,
            fontsize=23,
            weight="bold",
            color=color,
            va="top",
        )
        boundary.text(
            0.03,
            y - 0.07,
            label,
            transform=boundary.transAxes,
            fontsize=10.5,
            color="#53636b",
            va="top",
            wrap=True,
        )

    fig.suptitle(
        "v142: exact K1-dual pair-depth warm-start sentinel",
        x=0.055,
        y=0.985,
        ha="left",
        fontsize=18.5,
        weight="bold",
        color="#18262d",
    )
    fig.text(
        0.055,
        0.947,
        "four preregistered post-open difficult cells | no learned upstream | full-roster result pending",
        fontsize=10.5,
        color="#53636b",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
