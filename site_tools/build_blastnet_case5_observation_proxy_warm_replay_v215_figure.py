#!/usr/bin/env python3
"""Build the redacted v215 reference-adequacy figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets/figures/blastnet_case5_observation_proxy_warm_replay_v215.png"
P90 = np.asarray(
    [
        0.7864362757,
        0.7242772577,
        0.7257028778,
        0.7302757544,
        0.7288271610,
        0.7598427674,
        0.7444072341,
        0.7129557647,
        0.7277926075,
        0.7432295328,
        0.7273409469,
        0.7637142875,
        0.7677609209,
    ]
)
WORST = np.asarray(
    [
        0.8185496956,
        0.7570550247,
        0.7593479621,
        0.7596454365,
        0.7656963490,
        0.7992919110,
        0.7733659182,
        0.7443572585,
        0.7634041877,
        0.7686397541,
        0.7543963288,
        0.7932594971,
        0.7986071843,
    ]
)
SAFE = np.asarray([17, 40, 39, 40, 40, 32, 39, 42, 40, 39, 40, 30, 28])


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
        "v215: warm replay stops at reference adequacy",
        x=0.03,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#162033",
    )
    rigs = np.arange(13)
    axes[0].plot(rigs, P90, "o-", color="#2469b2", linewidth=2.1, label="interior-gradient p90")
    axes[0].plot(rigs, WORST, "s--", color="#c86b22", linewidth=1.8, label="worst cell")
    axes[0].axhline(0.75, color="#b1282e", linewidth=2, label="cell / p90 gate 0.75")
    axes[0].set_title("Zero-CGLS K16 interior-gradient error")
    axes[0].set_xlabel("Virtual nine-camera rig")
    axes[0].set_ylabel("Relative error")
    axes[0].set_xticks(rigs)
    axes[0].set_ylim(0.69, 0.84)
    axes[0].grid(axis="y", color="#d9e1de", linewidth=0.8)
    axes[0].legend(frameon=False, loc="upper right")

    colors = np.where(SAFE == 42, "#16836d", "#c86b22")
    axes[1].bar(rigs, SAFE, color=colors, width=0.72)
    axes[1].axhline(42, color="#16836d", linewidth=1.8)
    axes[1].set_title("Cells passing all four absolute gates")
    axes[1].set_xlabel("Virtual nine-camera rig")
    axes[1].set_ylabel("Passing cells out of 42")
    axes[1].set_xticks(rigs)
    axes[1].set_ylim(0, 45)
    axes[1].grid(axis="y", color="#d9e1de", linewidth=0.8)
    axes[1].text(
        0.98,
        0.08,
        "1 / 13 complete rigs\n466 / 546 strict cells",
        transform=axes[1].transAxes,
        ha="right",
        fontsize=12,
        fontweight="bold",
        color="#8b4c1d",
    )
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
    fig.text(
        0.03,
        0.064,
        "All 80 cell failures are interior-gradient only; field, full-gradient, and observation violations are zero.",
        fontsize=10,
        color="#526071",
    )
    fig.text(
        0.03,
        0.022,
        "Sealed outcome: inconclusive. No proxy depth, exact-call reduction, resource speedup, or algorithm breakthrough is adjudicated.",
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
