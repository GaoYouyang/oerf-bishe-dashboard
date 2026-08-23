#!/usr/bin/env python3
"""Build the redacted v209 Case 5 virtual-camera attribution figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets/figures/blastnet_case5_virtual_camera_information_v209.png"

ARMS = ("Supplied 9\n(v208)", "Virtual ring 9", "Virtual ring 12")
COLORS = ("#c95752", "#287bb5", "#258a7a")
RANGES = {
    "Field relative L2": (
        np.asarray((0.7252081324, 0.3199100706, 0.2763756373)),
        np.asarray((0.7607689701, 0.3515339756, 0.3073605362)),
        0.50,
    ),
    "Gradient relative L2": (
        np.asarray((0.5626643031, 0.6151958905, 0.5515774376)),
        np.asarray((0.6795003866, 0.6580897217, 0.5835577091)),
        0.75,
    ),
    "Observation relative L2": (
        np.asarray((0.0473007485, 0.0625066262, 0.0572689975)),
        np.asarray((0.0566662357, 0.0689341168, 0.0621708597)),
        0.20,
    ),
}


def _panel(
    ax: plt.Axes,
    title: str,
    lower: np.ndarray,
    upper: np.ndarray,
    gate: float,
) -> None:
    centers = (lower + upper) / 2.0
    errors = np.vstack((centers - lower, upper - centers))
    for index, (center, color) in enumerate(zip(centers, COLORS, strict=True)):
        ax.errorbar(
            index,
            center,
            yerr=errors[:, index : index + 1],
            fmt="o",
            markersize=9,
            capsize=8,
            linewidth=3,
            color=color,
        )
        ax.text(
            index,
            upper[index] + 0.018,
            f"{lower[index]:.3f}-{upper[index]:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color=color,
        )
    ax.axhline(
        gate,
        color="#1f2523",
        linestyle="--",
        linewidth=1.4,
        label=f"Frozen p90 gate = {gate:.2f}",
    )
    ax.set_xticks(range(len(ARMS)), ARMS)
    ax.set_title(title)
    ax.set_ylabel("Range of p90-higher across 13 groups")
    ax.grid(axis="y", color="#e5e9e6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="best", fontsize=8.5)


def build() -> Path:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "figure.facecolor": "#f7f8f5",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#c9d0cc",
            "axes.titleweight": "bold",
            "axes.labelcolor": "#24302d",
            "xtick.color": "#3f4c48",
            "ytick.color": "#3f4c48",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.8))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.80, bottom=0.23, wspace=0.22)
    fig.suptitle(
        "v209: virtual-ring geometry rescues Case 5 before camera count increases",
        fontsize=14.5,
        fontweight="bold",
        color="#1d2a27",
    )
    for ax, (title, (lower, upper, gate)) in zip(axes, RANGES.items(), strict=True):
        _panel(ax, title, lower, upper, gate)
    fig.text(
        0.5,
        0.085,
        "Virtual ring 9: 546/546 cells and 13/13 groups. Virtual ring 12: 546/546 and 13/13.",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color="#2f3a37",
    )
    fig.text(
        0.5,
        0.035,
        "Because nine virtual cameras already pass, the rescue is attributed to geometry or coverage, not to three additional cameras. Lower is better.",
        ha="center",
        fontsize=8.8,
        color="#55615d",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    with Image.open(OUTPUT) as image:
        image.convert("RGB").save(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(build())
