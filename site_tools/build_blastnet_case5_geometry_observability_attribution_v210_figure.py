#!/usr/bin/env python3
"""Build the redacted v210 geometry-observability attribution figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets/figures/blastnet_case5_geometry_observability_attribution_v210.png"

FAMILIES = ("Supplied 9", "Virtual ring 9", "Virtual ring 12\n(diagnostic)")
COLORS = ("#b95a4e", "#267c72", "#315f93")
PRIMARY = np.asarray(
    (
        (0.011857128995937665, 0.02146072143939357, 0.13158748032840895),
        (0.10931577130040626, 0.3075649138064044, 0.3400795270294168),
        (0.3939928942564763, 0.405610225049606, 0.42762720935676435),
    )
)
CONDITION = np.asarray(
    (
        (22.9197319562878, 152.12990819173936, 282.54127470973987),
        (10.023845529918965, 11.653634696962563, 30.753265659817686),
        (8.01352091780168, 8.562402048141596, 8.80993167813531),
    )
)


def _range_panel(ax: plt.Axes, values: np.ndarray, *, condition: bool = False) -> None:
    for index, (low, median, high) in enumerate(values):
        ax.vlines(index, low, high, color=COLORS[index], linewidth=5, alpha=0.84)
        ax.scatter(index, median, s=92, color=COLORS[index], edgecolor="white", zorder=3)
        ax.text(
            index,
            high * (1.20 if condition else 1.05),
            f"{median:.3g}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=COLORS[index],
        )
    ax.set_xticks(range(3), FAMILIES)
    ax.grid(axis="y", color="#e2e8e5", linewidth=0.8)
    ax.set_axisbelow(True)


def build() -> Path:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "figure.facecolor": "#f5f7f6",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#c9d3cf",
            "axes.labelcolor": "#263631",
            "xtick.color": "#42524d",
            "ytick.color": "#42524d",
            "axes.titleweight": "bold",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.9))
    fig.subplots_adjust(left=0.055, right=0.985, top=0.79, bottom=0.25, wspace=0.24)
    fig.suptitle(
        "v210: geometry-only observability moves strongly, but does not strictly separate",
        fontsize=15,
        fontweight="bold",
        color="#1c2c27",
    )

    _range_panel(axes[0], PRIMARY)
    axes[0].set_title("Fixed 64-mode spectral floor")
    axes[0].set_ylabel("Min - median - max (higher is better)")
    axes[0].annotate(
        "ranges overlap",
        xy=(0.48, 0.122),
        xytext=(1.28, 0.052),
        arrowprops={"arrowstyle": "->", "color": "#6b7773"},
        fontsize=8.5,
        color="#5a6763",
    )

    passed = 167
    total = 169
    fraction = passed / total
    axes[1].barh([0], [1.0], color="#e6ebe8", height=0.44)
    axes[1].barh([0], [fraction], color="#267c72", height=0.44)
    axes[1].axvline(1.0, color="#1d2a27", linestyle="--", linewidth=1.4)
    axes[1].text(fraction / 2, 0, f"{passed} / {total}\n98.8166%", ha="center", va="center", color="white", fontweight="bold", fontsize=12)
    axes[1].text(0.988, -0.39, "2 reversed pairs", ha="right", va="center", color="#b95a4e", fontweight="bold", fontsize=9)
    axes[1].set_xlim(0, 1.02)
    axes[1].set_ylim(-0.75, 0.75)
    axes[1].set_yticks([])
    axes[1].set_xlabel("Virtual-nine > supplied-nine pair fraction")
    axes[1].set_title("Strong direction, failed strict gate")
    axes[1].grid(axis="x", color="#e2e8e5", linewidth=0.8)
    axes[1].set_axisbelow(True)

    _range_panel(axes[2], CONDITION, condition=True)
    axes[2].set_yscale("log")
    axes[2].set_title("Normalized condition number")
    axes[2].set_ylabel("Min - median - max (lower is better, log scale)")

    fig.text(
        0.5,
        0.105,
        "167/169, not 169/169: geometry and conditioning matter, but this fixed spectral floor is not a sufficient classifier.",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color="#2c3935",
    )
    fig.text(
        0.5,
        0.052,
        "Post-open synthetic attribution only. No predictor, exact-call reduction, wall/RSS, external-generalization, or real-BOST claim.",
        ha="center",
        fontsize=8.8,
        color="#56635f",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    with Image.open(OUTPUT) as image:
        image.convert("RGB").save(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(build())
