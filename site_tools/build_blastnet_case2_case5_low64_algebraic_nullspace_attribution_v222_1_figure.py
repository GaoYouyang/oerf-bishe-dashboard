#!/usr/bin/env python3
"""Build the public v222.1 algebraic null-space attribution figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case2_case5_low64_algebraic_nullspace_attribution_v222_1_public_summary.json"
OUTPUT = ROOT / "assets/figures/blastnet_case2_case5_low64_algebraic_nullspace_attribution_v222_1.png"


def build() -> Path:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    case5 = payload["outcomes"]["case5"]
    case2 = payload["outcomes"]["case2"]
    validation = payload["independent_validation"]

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
    fig.subplots_adjust(left=0.06, right=0.985, top=0.78, bottom=0.25, wspace=0.31)
    fig.suptitle(
        "v222.1: orthogonal null(A) removal preserves Case 5 but not Case 2",
        x=0.035,
        ha="left",
        fontsize=17,
        fontweight="bold",
        color="#162033",
    )

    conditions = ["Case 5", "Case 2"]
    totals = np.asarray([case5["total_cells"], case2["total_cells"]], dtype=float)
    x = np.arange(2)
    width = 0.24
    algebraic = np.asarray([case5["matched_strict_safe"], case2["matched_strict_safe"]]) / totals
    direct = np.asarray([case5["direct_low64_k11_matched_strict_safe"], case2["direct_low64_k11_matched_strict_safe"]]) / totals
    spectral = np.asarray([case5["v221_spectral_lift_matched_strict_safe"], case2["v221_spectral_lift_matched_strict_safe"]]) / totals
    for offset, values, label, color in [
        (-width, algebraic, "Orthogonal algebraic", "#16836d"),
        (0.0, direct, "Direct Low-64 K11", "#2d6f91"),
        (width, spectral, "v221 A^T A lift", "#c34832"),
    ]:
        axes[0].bar(x + offset, 100 * values, width, label=label, color=color)
        for idx, value in enumerate(values):
            axes[0].text(idx + offset, max(1.2, 100 * value + 1.2), f"{100 * value:.1f}%", ha="center", fontsize=8)
    axes[0].set_title("K16-matched cells")
    axes[0].set_ylabel("Passing cells (%)")
    axes[0].set_ylim(0, 108)
    axes[0].set_xticks(x, conditions)
    axes[0].legend(frameon=False, loc="lower left", fontsize=8)

    algebraic_rigs = np.asarray([case5["complete_rigs_passed"], case2["complete_rigs_passed"]])
    direct_rigs = np.asarray([case5["direct_low64_k11_complete_rigs_passed"], case2["direct_low64_k11_complete_rigs_passed"]])
    spectral_rigs = np.asarray([case5["v221_spectral_lift_complete_rigs_passed"], case2["v221_spectral_lift_complete_rigs_passed"]])
    for offset, values, label, color in [
        (-width, algebraic_rigs, "Orthogonal algebraic", "#16836d"),
        (0.0, direct_rigs, "Direct Low-64 K11", "#2d6f91"),
        (width, spectral_rigs, "v221 A^T A lift", "#c34832"),
    ]:
        axes[1].bar(x + offset, values, width, label=label, color=color)
        for idx, value in enumerate(values):
            axes[1].text(idx + offset, max(0.18, value + 0.18), f"{value}/13", ha="center", fontsize=8)
    axes[1].set_title("Complete rigs")
    axes[1].set_ylabel("Passing rigs")
    axes[1].set_ylim(0, 14.2)
    axes[1].set_xticks(x, conditions)
    axes[1].legend(frameon=False, loc="upper right", fontsize=8)

    labels = ["Observation", "Initializer", "Field", "Metric", "Summary"]
    values = np.asarray(
        [
            validation["maximum_independent_observation_relative_difference"],
            validation["maximum_projected_initializer_relative_difference"],
            validation["maximum_algebraic_field_relative_difference"],
            validation["maximum_primary_metric_absolute_difference"],
            validation["maximum_summary_absolute_difference"],
        ]
    )
    axes[2].bar(np.arange(len(labels)), values, color=["#2d6f91", "#5b8f79", "#16836d", "#8a6d3b", "#7a8793"])
    axes[2].set_yscale("log")
    axes[2].set_ylim(1e-15, 1e-11)
    axes[2].set_title("Independent max differences")
    axes[2].set_ylabel("Relative / absolute difference")
    axes[2].set_xticks(np.arange(len(labels)), labels, rotation=20, ha="right")
    for idx, value in enumerate(values):
        axes[2].text(idx, value * 1.25, f"{value:.1e}", ha="center", fontsize=8)

    for axis in axes:
        axis.grid(axis="y", color="#d9e1de", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)

    fig.text(
        0.035,
        0.095,
        "v222 remains INCONCLUSIVE: frozen residual-equivalence 1.439e-9 > 1e-9. v222.1 is post-open retrospective attribution only.",
        color="#42514c",
        fontsize=9.5,
    )
    fig.text(
        0.035,
        0.055,
        "Decision: null(A) content is not required for Case 5 and does not cause Case 2 harm; attribution points to v221 spectral reweighting.",
        color="#1f6658",
        fontsize=10,
        fontweight="bold",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    with Image.open(OUTPUT) as image:
        image.convert("RGB").save(OUTPUT, optimize=True)
    return OUTPUT


if __name__ == "__main__":
    print(build())
