#!/usr/bin/env python3
"""Render the redacted public v246 Case 19 envelope-decision figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_reference_envelope_v246_public_summary.json"
OUTPUT = ROOT / "assets/figures/blastnet_case19_reference_envelope_v246.png"


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    result = data["v246_two_implementation_envelope"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(18.5, 6.6))
    figure.patch.set_facecolor("#f4f6f3")
    figure.subplots_adjust(left=0.055, right=0.985, bottom=0.17, top=0.80, wspace=0.22)
    for axis in axes:
        axis.set_facecolor("#ffffff")

    safe = np.array(
        [result["k14_definitely_safe_cells"], result["k16_definitely_safe_cells"]]
    )
    bars = axes[0].bar((0, 1), safe, color=("#6f7b86", "#26766f"), width=0.58)
    axes[0].axhline(429, color="#252a28", linestyle="--", linewidth=1.4)
    axes[0].set_xticks((0, 1), ("K14", "K16"))
    axes[0].set_ylim(0, 465)
    axes[0].set_ylabel("Definitely-safe cells out of 429")
    axes[0].set_title("A  Safety improves by 104 cells")
    axes[0].grid(axis="y", color="#dfe3df", linewidth=0.8)
    for bar, value in zip(bars, safe, strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 9,
            str(int(value)),
            ha="center",
            fontweight="bold",
        )

    component_values = np.array(
        [
            result["positive_worst_case_gain_components"],
            result["nonpositive_worst_case_gain_components"],
        ]
    )
    component_bars = axes[1].bar(
        (0, 1), component_values, color=("#315f93", "#bd5c45"), width=0.58
    )
    axes[1].set_xticks((0, 1), ("Positive gain", "No positive gain"))
    axes[1].set_ylim(0, 13.2)
    axes[1].set_ylabel("Possibly unsafe K16 components")
    axes[1].set_title("B  One component blocks deeper iteration")
    axes[1].grid(axis="y", color="#dfe3df", linewidth=0.8)
    for bar, value in zip(component_bars, component_values, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.35,
            str(int(value)),
            ha="center",
            fontweight="bold",
        )

    margins = np.array(
        [result["minimum_robust_p90_margin"], result["minimum_robust_worst_margin"]]
    )
    margin_bars = axes[2].bar(
        (0, 1), margins, color=("#8a651b", "#634f83"), width=0.58
    )
    axes[2].axhline(0.0, color="#252a28", linewidth=1.2)
    axes[2].set_xticks((0, 1), ("p90 margin", "Worst margin"))
    axes[2].set_ylim(0, 0.0165)
    axes[2].set_ylabel("K14 lower tail minus K16 upper tail")
    axes[2].set_title("C  Complete-rig tails still improve")
    axes[2].grid(axis="y", color="#dfe3df", linewidth=0.8)
    for bar, value in zip(margin_bars, margins, strict=True):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.00045,
            f"{value:.5f}",
            ha="center",
            fontweight="bold",
        )

    figure.suptitle(
        "v246 Case 19: broad K16 improvement, but one zero-gain component rejects K20",
        fontsize=16.5,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.035,
        "Independent recomputation: 16/16 checks, exact v246 agreement. Post-open diagnostic only; no wall/RSS or external claim.",
        ha="center",
        color="#545b57",
        fontsize=10,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


if __name__ == "__main__":
    main()
