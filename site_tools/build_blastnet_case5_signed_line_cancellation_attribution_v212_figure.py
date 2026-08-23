#!/usr/bin/env python3
"""Build the redacted v212 signed-line cancellation attribution figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "assets/figures/blastnet_case5_signed_line_cancellation_attribution_v212.png"
)

FAMILIES = ("Supplied 9", "Virtual ring 9", "Virtual ring 12\n(diagnostic)")
COLORS = ("#df7b08", "#3477e6", "#1f8c76")
PRIMARY = (
    np.asarray(
        [
            0.6459675936131006,
            0.6459675936131006,
            0.6670212055105766,
            0.6627962696286572,
            0.6391321013376926,
            0.6435947991974268,
            0.6321200540548987,
            0.6459669420442566,
            0.6538070330809294,
            0.6409308161436096,
            0.6417811997947827,
            0.6459675936131006,
            0.6328368192788449,
        ]
    ),
    np.asarray(
        [
            0.6274971203639474,
            0.6190214865199252,
            0.6218786539605594,
            0.6401525012853682,
            0.6271404845671232,
            0.6369007168700865,
            0.6292190015049202,
            0.6296986168082152,
            0.6271420959910982,
            0.6316111829815131,
            0.620113972330897,
            0.636869286802728,
            0.6294352101319115,
        ]
    ),
    np.asarray(
        [
            0.6282349808668689,
            0.6281921000034222,
            0.6293398894905959,
            0.6312175116135308,
            0.6276719111879917,
            0.62807414486859,
            0.630300277205873,
            0.6296603824195783,
            0.6274232254622569,
            0.6316893017798126,
            0.629203558269585,
            0.6282116099794487,
            0.6284784884249431,
        ]
    ),
)


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
    fig = plt.figure(figsize=(15.4, 6.0))
    grid = fig.add_gridspec(
        1,
        3,
        left=0.055,
        right=0.985,
        top=0.79,
        bottom=0.19,
        wspace=0.40,
        width_ratios=(1.05, 1.05, 1.05),
    )
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    fig.suptitle(
        "v212 geometry-only signed-line cancellation attribution",
        x=0.025,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#162033",
    )

    offsets = np.linspace(-0.08, 0.08, 13)
    for family_index, values in enumerate(PRIMARY):
        axes[0].scatter(
            family_index + offsets,
            values,
            s=48,
            color=COLORS[family_index],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        axes[0].hlines(
            float(np.median(values)),
            family_index - 0.24,
            family_index + 0.24,
            color="#172033",
            linewidth=2.4,
            zorder=4,
        )
    axes[0].set_xticks(range(3), FAMILIES)
    axes[0].set_ylabel("Signed-line coherence ratio\n(higher means less cancellation)")
    axes[0].set_title("Fixed scalar does not separate families")
    axes[0].grid(axis="y", color="#dbe1df", linewidth=0.8)
    axes[0].set_axisbelow(True)
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].barh([1], [162], color="#df7b08", height=0.42)
    axes[1].barh([0], [7], color="#3477e6", height=0.42)
    axes[1].text(153, 1, "162/169", ha="right", va="center", fontweight="bold")
    axes[1].text(10, 0, "7/169", ha="left", va="center", fontweight="bold")
    axes[1].set_yticks([1, 0], ["Opposite direction", "Expected direction"])
    axes[1].set_xlim(0, 169)
    axes[1].set_xlabel("Cross-family comparisons")
    axes[1].set_title("Preregistered strict gate fails")
    axes[1].grid(axis="x", color="#dbe1df", linewidth=0.8)
    axes[1].set_axisbelow(True)
    axes[1].spines[["top", "right", "left"]].set_visible(False)

    axes[2].axis("off")
    axes[2].set_title("What the evidence now says", pad=8)
    evidence = (
        (0.83, "v210 actual low-mode Gram", "167/169 expected-direction wins", "#2863df"),
        (0.59, "v211 unsigned local coverage", "0/169 expected-direction wins", "#cf6c00"),
        (0.35, "v212 signed-line cancellation", "7/169 expected-direction wins", "#14866f"),
    )
    for y, title, detail, color in evidence:
        axes[2].text(
            0.02,
            y,
            title,
            color=color,
            fontsize=11.5,
            fontweight="bold",
            transform=axes[2].transAxes,
        )
        axes[2].text(
            0.02,
            y - 0.08,
            detail,
            color="#253044",
            fontsize=10.5,
            transform=axes[2].transAxes,
        )
    axes[2].text(
        0.02,
        0.08,
        "Close this fixed scalar only. Still false:\npredictor, speedup, external gate, real BOST.",
        color="#56606f",
        fontsize=10,
        fontweight="bold",
        linespacing=1.35,
        transform=axes[2].transAxes,
    )

    fig.text(
        0.025,
        0.045,
        "39 geometries | 64 fixed sine modes | 0A+0A^T logical deployment ledger | independently recomputed",
        fontsize=9.5,
        color="#526071",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    with Image.open(OUTPUT) as image:
        image.convert("RGB").save(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(build())
