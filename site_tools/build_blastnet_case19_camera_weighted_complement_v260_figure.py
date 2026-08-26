#!/usr/bin/env python3
"""Render the redacted public v260 camera-weighting verdict figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_camera_weighted_complement_v260_public_summary.json"
OUTPUT = ROOT / "assets/figures/blastnet_case19_camera_weighted_complement_v260.png"


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    primary = data["primary"]
    unweighted = data["controls"]["unweighted_v258_equal_call"]
    labels = ("Field", "Full gradient", "Interior gradient", "Observation")
    p90 = np.asarray(primary["matched_ratio_p90_higher"])
    worst = np.asarray(primary["matched_ratio_worst"])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(17, 7))
    figure.patch.set_facecolor("#f3f5f2")
    figure.subplots_adjust(left=0.07, right=0.97, bottom=0.20, top=0.78, wspace=0.25)
    for axis in axes:
        axis.set_facecolor("#ffffff")

    positions = np.arange(len(labels))
    width = 0.36
    bars_p90 = axes[0].bar(positions - width / 2, p90, width, color="#2f8073", label="p90-higher")
    bars_worst = axes[0].bar(positions + width / 2, worst, width, color="#c04f3d", label="worst")
    axes[0].axhline(1.05, color="#252a28", linestyle="--", linewidth=1.5, label="Frozen matched gate: 1.05")
    axes[0].set_xticks(positions, labels)
    axes[0].set_ylim(0, 1.52)
    axes[0].set_ylabel("v260 / K16 error ratio")
    axes[0].set_title("A  Observation remains the only matched blocker")
    axes[0].grid(axis="y", color="#dfe3df", linewidth=0.8)
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].bar_label(bars_p90, labels=[f"{value:.3f}" for value in p90], padding=3, fontsize=9)
    axes[0].bar_label(bars_worst, labels=[f"{value:.3f}" for value in worst], padding=3, fontsize=9)

    comparison_labels = ("p90-higher", "worst")
    unweighted_values = np.asarray(
        [
            unweighted["observation_matched_ratio_p90_higher"],
            unweighted["observation_matched_ratio_worst"],
        ]
    )
    weighted_values = np.asarray([p90[-1], worst[-1]])
    positions = np.arange(len(comparison_labels))
    bars_unweighted = axes[1].bar(
        positions - width / 2,
        unweighted_values,
        width,
        color="#7d8581",
        label="v258 unweighted",
    )
    bars_weighted = axes[1].bar(
        positions + width / 2,
        weighted_values,
        width,
        color="#d5a139",
        label="v260 camera-weighted",
    )
    axes[1].axhline(1.05, color="#252a28", linestyle="--", linewidth=1.5, label="Frozen matched gate: 1.05")
    axes[1].set_xticks(positions, comparison_labels)
    axes[1].set_ylim(1.0, 1.45)
    axes[1].set_ylabel("Observation error / K16 error")
    axes[1].set_title("B  Weighting is slightly worse than equal-call v258")
    axes[1].grid(axis="y", color="#dfe3df", linewidth=0.8)
    axes[1].legend(frameon=False, loc="upper left")
    axes[1].bar_label(bars_unweighted, labels=[f"{value:.4f}" for value in unweighted_values], padding=3, fontsize=9)
    axes[1].bar_label(bars_weighted, labels=[f"{value:.4f}" for value in weighted_values], padding=3, fontsize=9)

    figure.suptitle(
        "v260 Case 19: residual-energy camera weighting does not repair matched accuracy",
        fontsize=17,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.055,
        "Independent 52/52; 13/13 absolute, 0/13 matched; 15A+14AT; post-open mechanism evidence, not an algorithm claim.",
        ha="center",
        color="#545b57",
        fontsize=10.5,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


if __name__ == "__main__":
    main()
