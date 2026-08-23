#!/usr/bin/env python3
"""Build the redacted v217.1 global PCGLS depth-qualification figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets/figures/blastnet_case5_global_pcgls_depth_qualification_v217_1.png"
DEPTHS = np.arange(8, 17)
ABSOLUTE_SAFE = np.asarray([0, 0, 0, 96, 318, 467, 526, 544, 546])
MATCHED_SAFE = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 546])
ABSOLUTE_RIGS = np.asarray([0, 0, 0, 0, 0, 5, 8, 11, 13])
MATCHED_RIGS = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 13])


def build() -> Path:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "figure.facecolor": "#f4f7f6",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#c8d2cf",
            "axes.labelcolor": "#25352f",
            "xtick.color": "#42514c",
            "ytick.color": "#42514c",
            "axes.titleweight": "bold",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(15.6, 6.3))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.78, bottom=0.2, wspace=0.23)
    fig.suptitle(
        "v217.1: K16 remains the lowest globally adequate PCGLS depth",
        x=0.03,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#162033",
    )

    x = np.arange(len(DEPTHS))
    axes[0].plot(x, ABSOLUTE_SAFE, "o-", color="#16836d", linewidth=2.4, label="absolute gate")
    axes[0].plot(x, MATCHED_SAFE, "s-", color="#c34832", linewidth=2.4, label="matched to K16")
    axes[0].axhline(546, color="#879790", linewidth=1.2, linestyle="--")
    axes[0].set_title("Cells passing all four gates")
    axes[0].set_xlabel("Globally fixed geometry-Jacobi PCGLS depth")
    axes[0].set_ylabel("Passing cells out of 546")
    axes[0].set_xticks(x, [f"K{depth}" for depth in DEPTHS])
    axes[0].set_ylim(-20, 590)
    axes[0].grid(axis="y", color="#d9e1de", linewidth=0.8)
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].annotate(
        "K15: 544 absolute\n0 matched",
        xy=(7, 544),
        xytext=(5.2, 390),
        arrowprops={"arrowstyle": "->", "color": "#5a6b65"},
        color="#6a4e42",
        fontweight="bold",
    )

    width = 0.36
    axes[1].bar(x - width / 2, ABSOLUTE_RIGS, width, color="#16836d", label="absolute gate")
    axes[1].bar(x + width / 2, MATCHED_RIGS, width, color="#c34832", label="matched to K16")
    axes[1].axhline(13, color="#879790", linewidth=1.2, linestyle="--")
    axes[1].set_title("Complete geometries passing all gates")
    axes[1].set_xlabel("Globally fixed geometry-Jacobi PCGLS depth")
    axes[1].set_ylabel("Passing geometries out of 13")
    axes[1].set_xticks(x, [f"K{depth}" for depth in DEPTHS])
    axes[1].set_ylim(0, 14.5)
    axes[1].grid(axis="y", color="#d9e1de", linewidth=0.8)
    axes[1].legend(frameon=False, loc="upper left")
    axes[1].text(
        0.44,
        0.6,
        "Only K16 reaches\n13/13 absolute and matched",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontsize=11,
        color="#8b3425",
        fontweight="bold",
    )

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
    fig.text(
        0.03,
        0.064,
        "The v217.1 execution repair restores labeled observation blocks by camera ID; thresholds, solver, data, and science arrays are unchanged.",
        fontsize=10,
        color="#526071",
    )
    fig.text(
        0.03,
        0.022,
        "Reference qualification only: no learned initializer, exact-call reduction, wall/RSS gain, external result, or real-BOST claim.",
        fontsize=10,
        color="#6a4e42",
        fontweight="bold",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    with Image.open(OUTPUT) as image:
        image.convert("RGB").save(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(build())
