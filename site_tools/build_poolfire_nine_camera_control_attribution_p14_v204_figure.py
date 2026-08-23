#!/usr/bin/env python3
"""Build the redacted v203-v204 attribution figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets/figures/poolfire_nine_camera_control_attribution_p14_v204.png"


def build() -> Path:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.facecolor": "#f7f8f5",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#c9d0cc",
            "axes.titleweight": "bold",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), constrained_layout=True)
    fig.suptitle(
        "Nine-camera information rescue and all-nine control attribution",
        fontsize=15,
        fontweight="bold",
        color="#1d2a27",
    )

    rescue_labels = ("Five-camera K2", "All-nine K2")
    rescue_values = (0, 24)
    rescue_colors = ("#c95752", "#258a7a")
    bars = axes[0].bar(rescue_labels, rescue_values, color=rescue_colors, width=0.58)
    axes[0].set_ylim(0, 27)
    axes[0].set_ylabel("Strict passes among 24 audited failures")
    axes[0].set_title("v203: four added views rescue 24/24")
    axes[0].grid(axis="y", color="#e5e9e6", linewidth=0.8)
    axes[0].set_axisbelow(True)
    for bar, value in zip(bars, rescue_values, strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.7,
            f"{value}/24",
            ha="center",
            va="bottom",
            fontweight="bold",
            color="#24302d",
        )
    axes[0].text(
        0.5,
        -0.19,
        "Same frozen K2 reference and strict field / gradient / observation gates",
        transform=axes[0].transAxes,
        ha="center",
        color="#55615d",
        fontsize=9,
    )

    labels = (
        "Zero",
        "BP-CGLS1",
        "Zero-CGLS K2",
        "Jacobi-PCGLS1",
        "Dual ridge",
        "Initializer only",
        "Full-DCT K1",
        "Fixed-id K1",
        "Full-DCT K2",
    )
    values = np.asarray((0, 0, 0, 0, 42, 654, 1313, 1313, 1313))
    colors = (
        "#9aa5a0",
        "#9aa5a0",
        "#9aa5a0",
        "#9aa5a0",
        "#d9a441",
        "#d9a441",
        "#287bb5",
        "#258a7a",
        "#584f9e",
    )
    y = np.arange(len(labels))
    bars = axes[1].barh(y, values, color=colors, height=0.65)
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1425)
    axes[1].set_xlabel("Strict-safe cells out of 1,313")
    axes[1].set_title("v204: only dense full-DCT K1/K2 pass all cells")
    axes[1].grid(axis="x", color="#e5e9e6", linewidth=0.8)
    axes[1].set_axisbelow(True)
    axes[1].axvline(1313, color="#1d2a27", linewidth=1, linestyle="--")
    for bar, value in zip(bars, values, strict=True):
        axes[1].text(
            max(value + 18, 20),
            bar.get_y() + bar.get_height() / 2,
            f"{value}",
            va="center",
            fontweight="bold" if value == 1313 else "normal",
            color="#24302d",
        )
    axes[1].text(
        0.5,
        -0.19,
        "Full-DCT K1: 2A+1AT; K2 reference: 3A+2AT. Dense cache not included.",
        transform=axes[1].transAxes,
        ha="center",
        color="#55615d",
        fontsize=9,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return OUTPUT


if __name__ == "__main__":
    print(build())
