#!/usr/bin/env python3
"""Build the redacted v216 matched-accuracy figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets/figures/blastnet_case5_pcgls_reference_warm_readjudication_v216.png"
DEPTHS = np.asarray([0, 1, 2, 4, 8])
ABSOLUTE_SAFE = np.asarray([0, 0, 0, 390, 546])
MATCHED_SAFE = np.asarray([0, 0, 0, 0, 0])
K8_VIOLATIONS = np.asarray([545, 546, 23, 546])
METRICS = ("Field", "Full gradient", "Interior gradient", "Observation")


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
        "v216: absolute accuracy is not matched accuracy",
        x=0.03,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#162033",
    )

    x = np.arange(len(DEPTHS))
    width = 0.36
    axes[0].bar(x - width / 2, ABSOLUTE_SAFE, width, color="#16836d", label="absolute gate")
    axes[0].bar(x + width / 2, MATCHED_SAFE, width, color="#c34832", label="matched to PCGLS K16")
    axes[0].axhline(546, color="#16836d", linewidth=1.5)
    axes[0].set_title("Low-64 proxy cells passing all four gates")
    axes[0].set_xlabel("Unchanged CGLS refinement depth")
    axes[0].set_ylabel("Passing cells out of 546")
    axes[0].set_xticks(x, [f"K{depth}" for depth in DEPTHS])
    axes[0].set_ylim(0, 590)
    axes[0].grid(axis="y", color="#d9e1de", linewidth=0.8)
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].text(
        0.98,
        0.86,
        "K8: 546 absolute\n0 matched",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=12,
        fontweight="bold",
        color="#8b3425",
    )

    colors = ("#326da8", "#6d58a5", "#c58a2d", "#c34832")
    bars = axes[1].bar(np.arange(4), K8_VIOLATIONS, color=colors, width=0.68)
    axes[1].axhline(546, color="#c34832", linewidth=1.5)
    axes[1].set_title("K8 cells exceeding the 1.05 matched limit")
    axes[1].set_xlabel("Metric")
    axes[1].set_ylabel("Violating cells out of 546")
    axes[1].set_xticks(np.arange(4), METRICS, rotation=12, ha="right")
    axes[1].set_ylim(0, 590)
    axes[1].grid(axis="y", color="#d9e1de", linewidth=0.8)
    axes[1].bar_label(bars, padding=4, fontweight="bold")

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
    fig.text(
        0.03,
        0.064,
        "Adequate reference: geometry-Jacobi PCGLS K16 passes 546/546 cells and 13/13 complete rigs at 16A + 16A^T.",
        fontsize=10,
        color="#526071",
    )
    fig.text(
        0.03,
        0.022,
        "Sealed outcome: the fixed low-64 proxy warm start is closed. No exact-call, resource, external, or real-BOST gain is established.",
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
