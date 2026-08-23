#!/usr/bin/env python3
"""Build the redacted v211 local ray-coverage attribution figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets/figures/blastnet_case5_local_ray_coverage_attribution_v211.png"

FAMILIES = ("Supplied 9", "Virtual ring 9", "Virtual ring 12\n(diagnostic)")
COLORS = ("#df7b08", "#3477e6", "#1f8c76")
PRIMARY = (
    np.asarray(
        [
            0.12500802525805454,
            0.12500802525805454,
            0.15384966672431724,
            0.1560060092860445,
            0.10450482124234327,
            0.1519032087044301,
            0.15119781987442577,
            0.12500831003255441,
            0.16574280281132603,
            0.10381494565708,
            0.10482342104517318,
            0.12500802525805454,
            0.15257048808128984,
        ]
    ),
    np.asarray(
        [
            0.07912234851471091,
            0.08496753631983783,
            0.08236351738483813,
            0.07918827628045948,
            0.07681908191593811,
            0.07423606887817756,
            0.07731164117651607,
            0.07348347064991055,
            0.07990467025405291,
            0.08295292625061025,
            0.08569582312999037,
            0.07691122255366266,
            0.07685109685207385,
        ]
    ),
    np.asarray(
        [
            0.09623957978253374,
            0.09683121037401686,
            0.09534366738948706,
            0.09488408755124596,
            0.09571026399139906,
            0.09657694328462559,
            0.0953495652424014,
            0.096815282802358,
            0.09608739448753323,
            0.09900132480931355,
            0.09777708258652984,
            0.09688150780976919,
            0.09519335183666854,
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
        "v211 geometry-only local ray-coverage attribution",
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
    axes[0].set_ylabel(
        "Normalized local transverse floor\n10th percentile (higher is stronger)"
    )
    axes[0].set_title("Fixed lower-tail scalar reverses direction")
    axes[0].grid(axis="y", color="#dbe1df", linewidth=0.8)
    axes[0].set_axisbelow(True)
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].barh([1], [169], color="#df7b08", height=0.42)
    axes[1].barh([0], [0], color="#3477e6", height=0.42)
    axes[1].text(160, 1, "169/169", ha="right", va="center", fontweight="bold")
    axes[1].text(3, 0, "0/169", ha="left", va="center", fontweight="bold")
    axes[1].set_yticks([1, 0], ["Opposite direction", "Expected direction"])
    axes[1].set_xlim(0, 169)
    axes[1].set_xlabel("Cross-family comparisons")
    axes[1].set_title("Preregistered strict gate fails")
    axes[1].grid(axis="x", color="#dbe1df", linewidth=0.8)
    axes[1].set_axisbelow(True)
    axes[1].spines[["top", "right", "left"]].set_visible(False)

    axes[2].axis("off")
    axes[2].set_title("What the negative result changes", pad=8)
    axes[2].text(
        0.02,
        0.82,
        "v210 global low-mode floor",
        color="#2863df",
        fontsize=12,
        fontweight="bold",
        transform=axes[2].transAxes,
    )
    axes[2].text(
        0.02,
        0.75,
        "167/169 expected-direction wins",
        color="#253044",
        fontsize=11,
        transform=axes[2].transAxes,
    )
    axes[2].text(
        0.02,
        0.57,
        "v211 local lower-tail floor",
        color="#cf6c00",
        fontsize=12,
        fontweight="bold",
        transform=axes[2].transAxes,
    )
    axes[2].text(
        0.02,
        0.50,
        "0/169 expected-direction wins",
        color="#253044",
        fontsize=11,
        transform=axes[2].transAxes,
    )
    axes[2].text(
        0.02,
        0.31,
        "Scientific boundary",
        color="#14866f",
        fontsize=12,
        fontweight="bold",
        transform=axes[2].transAxes,
    )
    axes[2].text(
        0.02,
        0.18,
        "This fixed normalized lower-10% local\nscalar is not the reference-adequacy classifier.",
        color="#253044",
        fontsize=10.5,
        linespacing=1.35,
        transform=axes[2].transAxes,
    )
    axes[2].text(
        0.02,
        0.02,
        "Still false: predictor, speedup, external gate,\nor real-BOST success.",
        color="#56606f",
        fontsize=10,
        fontweight="bold",
        linespacing=1.35,
        transform=axes[2].transAxes,
    )

    fig.text(
        0.025,
        0.045,
        "39 geometries | 5,880 interior tensors per geometry | 0A+0A^T deployment ledger | independently recomputed",
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
