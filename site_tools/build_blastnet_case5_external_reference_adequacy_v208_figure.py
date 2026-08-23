#!/usr/bin/env python3
"""Build the redacted v207-v208 Case 5 reference-adequacy figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets/figures/blastnet_case5_external_reference_adequacy_v208.png"

ITERATIONS = np.asarray((4, 8, 16), dtype=np.int64)
COLORS = ("#c95752", "#287bb5", "#258a7a")
RANGES = {
    "Field relative L2": (
        np.asarray((0.8155410194, 0.7643824959, 0.7252081324)),
        np.asarray((0.8421659116, 0.7967179762, 0.7607689701)),
        0.50,
    ),
    "Gradient relative L2": (
        np.asarray((0.5950645641, 0.5381497439, 0.5626643031)),
        np.asarray((0.6745598036, 0.6764148585, 0.6795003866)),
        0.75,
    ),
    "Observation relative L2": (
        np.asarray((0.2123685521, 0.1065016103, 0.0473007485)),
        np.asarray((0.2456470465, 0.1258027257, 0.0566662357)),
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
    for index, (iteration, center, color) in enumerate(
        zip(ITERATIONS, centers, COLORS, strict=True)
    ):
        ax.errorbar(
            index,
            center,
            yerr=errors[:, index : index + 1],
            fmt="o",
            markersize=8,
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
    ax.axhline(gate, color="#1f2523", linestyle="--", linewidth=1.4, label=f"Frozen p90 gate = {gate:.2f}")
    ax.set_xticks(range(len(ITERATIONS)), [f"K{value}" for value in ITERATIONS])
    ax.set_title(title)
    ax.set_ylabel("Range of p90-higher across 13 calibrations")
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
    fig.subplots_adjust(left=0.06, right=0.985, top=0.80, bottom=0.22, wspace=0.22)
    fig.suptitle(
        "v208: zero-start CGLS K16 still lacks an adequate Case 5 field reference",
        fontsize=14.5,
        fontweight="bold",
        color="#1d2a27",
    )
    for ax, (title, (lower, upper, gate)) in zip(axes, RANGES.items(), strict=True):
        _panel(ax, title, lower, upper, gate)
    fig.text(
        0.5,
        0.08,
        "Every arm: 0/546 strict-safe cells and 0/13 complete calibration groups. Lower is better.",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color="#2f3a37",
    )
    fig.text(
        0.5,
        0.035,
        "K16 clears gradient and observation p90 limits, but field p90 remains 0.725-0.761 versus the frozen 0.50 gate. The v207 external comparison is therefore inconclusive.",
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
