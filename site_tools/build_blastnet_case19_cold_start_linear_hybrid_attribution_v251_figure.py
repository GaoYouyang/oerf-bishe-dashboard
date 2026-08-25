#!/usr/bin/env python3
"""Render the redacted public v251 Case 19 hybrid-attribution figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT
    / "docs/blastnet_case19_cold_start_linear_hybrid_attribution_v251_public_summary.json"
)
OUTPUT = (
    ROOT
    / "assets/figures/blastnet_case19_cold_start_linear_hybrid_attribution_v251.png"
)


def _label_bars(axis: plt.Axes, bars: object, denominator: int) -> None:
    for bar in bars:
        value = int(bar.get_height())
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + denominator * 0.025,
            f"{value}/{denominator}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    results = data["independent_results"]

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
    figure.subplots_adjust(left=0.055, right=0.985, bottom=0.25, top=0.79, wspace=0.28)
    for axis in axes:
        axis.set_facecolor("#ffffff")

    cell_counts = np.asarray(
        (
            results["hybrid_robust_absolute_cells"],
            results["hybrid_robust_matched_cells"],
        )
    )
    cell_bars = axes[0].bar(
        (0, 1), cell_counts, color=("#26766f", "#b94f3c"), width=0.58
    )
    axes[0].set_xticks((0, 1), ("Absolute", "K16-matched"))
    axes[0].set_ylim(0, 458)
    axes[0].set_ylabel("Robust cells out of 429")
    axes[0].set_title("A  Absolute pass is not matched accuracy")
    axes[0].grid(axis="y", color="#dfe3df", linewidth=0.8)
    _label_bars(axes[0], cell_bars, 429)

    rig_counts = np.asarray(
        (
            results["hybrid_robust_absolute_complete_rigs"],
            results["hybrid_robust_matched_complete_rigs"],
            results["reference_robust_absolute_complete_rigs"],
        )
    )
    rig_bars = axes[1].bar(
        np.arange(3),
        rig_counts,
        color=("#26766f", "#b94f3c", "#6f7d9b"),
        width=0.62,
    )
    axes[1].set_xticks(
        np.arange(3), ("Hybrid\nabsolute", "Hybrid\nmatched", "K16 ref\nabsolute")
    )
    axes[1].set_ylim(0, 14.2)
    axes[1].set_ylabel("Complete rigs out of 13")
    axes[1].set_title("B  No rig closes the matched gate")
    axes[1].grid(axis="y", color="#dfe3df", linewidth=0.8)
    _label_bars(axes[1], rig_bars, 13)

    ratios = np.asarray(results["frame_zero_observation_cell_ratios"], dtype=np.float64)
    positions = np.arange(1, ratios.size + 1)
    axes[2].bar(positions, ratios, color="#d49a3a", width=0.68)
    axes[2].axhline(
        results["matched_cell_ratio_limit"],
        color="#252a28",
        linestyle="--",
        linewidth=1.5,
        label="Frozen matched limit 1.05",
    )
    axes[2].set_xticks(positions)
    axes[2].set_ylim(0, 3.55)
    axes[2].set_xlabel("Rig index")
    axes[2].set_ylabel("Frame-zero observation / K16")
    axes[2].set_title("C  All 13 misses are frame-zero observation")
    axes[2].grid(axis="y", color="#dfe3df", linewidth=0.8)
    axes[2].legend(
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#dfe3df",
        framealpha=1.0,
        loc="upper left",
    )

    figure.suptitle(
        "v251 Case 19: the direct cold-start hybrid fails matched accuracy",
        fontsize=16.2,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.025,
        "429/429 absolute cells, but 416/429 matched cells and 0/13 complete matched rigs; nominal 9.375% is not effective call reduction.",
        ha="center",
        color="#545b57",
        fontsize=10,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


if __name__ == "__main__":
    main()
