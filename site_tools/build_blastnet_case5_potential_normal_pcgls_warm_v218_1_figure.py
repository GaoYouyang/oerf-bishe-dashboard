#!/usr/bin/env python3
"""Build the redacted v218.1 warm-replay comparison figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets/figures/blastnet_case5_potential_normal_pcgls_warm_v218_1.png"
LABELS = ["Potential\nK14", "Low-64\nK10", "Low-64\nK11", "Norm. BP\nK14", "Reference\nK16"]
MATCHED_CELLS = np.asarray([0, 164, 546, 0, 546])
MATCHED_RIGS = np.asarray([0, 0, 13, 0, 13])
TOTAL_CALLS = np.asarray([30, 21, 23, 30, 32])
COLORS = ["#c34832", "#d8a33d", "#16836d", "#6d8196", "#34495e"]


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
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 6.5))
    fig.subplots_adjust(left=0.055, right=0.985, top=0.78, bottom=0.25, wspace=0.31)
    fig.suptitle(
        "v218.1: the primary fails; Low-64 K11 establishes deterministic headroom",
        x=0.03,
        ha="left",
        fontsize=17,
        fontweight="bold",
        color="#162033",
    )
    x = np.arange(len(LABELS))

    axes[0].bar(x, MATCHED_CELLS, color=COLORS, width=0.72)
    axes[0].axhline(546, color="#879790", linewidth=1.2, linestyle="--")
    axes[0].set_title("K16-matched cells")
    axes[0].set_ylabel("Passing cells out of 546")
    axes[0].set_ylim(0, 600)

    axes[1].bar(x, MATCHED_RIGS, color=COLORS, width=0.72)
    axes[1].axhline(13, color="#879790", linewidth=1.2, linestyle="--")
    axes[1].set_title("Complete matched rigs")
    axes[1].set_ylabel("Passing rigs out of 13")
    axes[1].set_ylim(0, 14.5)

    axes[2].bar(x, TOTAL_CALLS, color=COLORS, width=0.72)
    axes[2].axhline(32, color="#879790", linewidth=1.2, linestyle="--")
    axes[2].set_title("Logical exact-call ledger")
    axes[2].set_ylabel("A + A^T calls")
    axes[2].set_ylim(0, 35)
    axes[2].annotate(
        "23 vs 32\n28.125% fewer",
        xy=(2, 23),
        xytext=(0.55, 29),
        arrowprops={"arrowstyle": "->", "color": "#4f625b"},
        color="#126a59",
        fontweight="bold",
    )

    for axis, values in zip(axes, [MATCHED_CELLS, MATCHED_RIGS, TOTAL_CALLS], strict=True):
        axis.set_xticks(x, LABELS)
        axis.grid(axis="y", color="#d9e1de", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
        for idx, value in enumerate(values):
            axis.text(idx, value + axis.get_ylim()[1] * 0.022, str(value), ha="center", va="bottom", fontweight="bold")

    fig.text(
        0.03,
        0.085,
        "Potential-normal remains 0/546 at K14. Low-64 K11 is the first frozen control depth to reach 546/546 cells and 13/13 rigs.",
        fontsize=10,
        color="#42514c",
    )
    fig.text(
        0.03,
        0.035,
        "Opened virtual Case 5 control result only: no learned model, wall/RSS gain, external generalization, or real-BOST claim.",
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
