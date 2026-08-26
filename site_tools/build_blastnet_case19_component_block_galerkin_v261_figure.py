#!/usr/bin/env python3
"""Render the redacted public v261 component-block verdict figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_component_block_galerkin_v261_public_summary.json"
OUTPUT = ROOT / "assets/figures/blastnet_case19_component_block_galerkin_v261.png"


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    primary = data["primary"]
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
    axes[0].set_ylim(0.95, 1.42)
    axes[0].set_ylabel("v261 / K16 error ratio")
    axes[0].set_title("A  All four matched metrics fail")
    axes[0].grid(axis="y", color="#dfe3df", linewidth=0.8)
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].bar_label(bars_p90, labels=[f"{value:.3f}" for value in p90], padding=3, fontsize=9)
    axes[0].bar_label(bars_worst, labels=[f"{value:.3f}" for value in worst], padding=3, fontsize=9)

    controls = data["controls"]
    method_labels = ("v261\nblock", "K15\nsame cost", "v258\ncheaper", "K14\ncheaper", "K16\nreference")
    absolute = np.asarray(
        [
            primary["absolute_cells"],
            controls["same_cost_k15"]["absolute_cells"],
            controls["cheaper_v258"]["absolute_cells"],
            controls["raw_k14"]["absolute_cells"],
            controls["k16_reference"]["absolute_cells"],
        ]
    )
    matched = np.asarray(
        [
            primary["matched_cells"],
            controls["same_cost_k15"]["matched_cells"],
            controls["cheaper_v258"]["matched_cells"],
            controls["raw_k14"]["matched_cells"],
            controls["k16_reference"]["matched_cells"],
        ]
    )
    positions = np.arange(len(method_labels))
    bars_absolute = axes[1].bar(positions - width / 2, absolute, width, color="#d5a139", label="Absolute cells")
    bars_matched = axes[1].bar(positions + width / 2, matched, width, color="#3c668f", label="K16-matched cells")
    axes[1].set_xticks(positions, method_labels)
    axes[1].set_ylim(0, 14.5)
    axes[1].set_yticks(np.arange(0, 14, 2))
    axes[1].set_ylabel("Passing rigs out of 13")
    axes[1].set_title("B  The primary is dominated by cheaper controls")
    axes[1].grid(axis="y", color="#dfe3df", linewidth=0.8)
    axes[1].legend(frameon=False, loc="upper left")
    axes[1].bar_label(bars_absolute, padding=3, fontsize=9)
    axes[1].bar_label(bars_matched, padding=3, fontsize=9)

    figure.suptitle(
        "v261 Case 19: signed-component block Galerkin fails the frame-zero gate",
        fontsize=17,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.055,
        "Independent 41/41; primary 3/13 absolute and 0/13 matched; 15A+15AT; post-open mechanism evidence, not an algorithm claim.",
        ha="center",
        color="#545b57",
        fontsize=10.5,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


if __name__ == "__main__":
    main()
