#!/usr/bin/env python3
"""Render the redacted public v258 matched-accuracy verdict figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_krylov_complement_heat_v258_public_summary.json"
OUTPUT = ROOT / "assets/figures/blastnet_case19_krylov_complement_heat_v258.png"


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    envelope = data["worst_implementation_envelope"]
    methods = ("Krylov complement", "Linear heat", "Raw K14")
    absolute = np.asarray(
        [
            envelope["primary"]["absolute_cells"],
            envelope["equal_call_linear_heat_control"]["absolute_cells"],
            envelope["k14_control"]["absolute_cells"],
        ]
    )
    matched = np.asarray(
        [
            envelope["primary"]["matched_cells"],
            envelope["equal_call_linear_heat_control"]["matched_cells"],
            envelope["k14_control"]["matched_cells"],
        ]
    )
    observation_p90 = np.asarray(
        [
            envelope["primary"]["matched_ratio_p90_higher"][3],
            envelope["equal_call_linear_heat_control"]["observation_matched_ratio_p90_higher"],
            envelope["k14_control"]["observation_matched_ratio_p90_higher"],
        ]
    )
    observation_worst = np.asarray(
        [
            envelope["primary"]["matched_ratio_worst"][3],
            envelope["equal_call_linear_heat_control"]["observation_matched_ratio_worst"],
            envelope["k14_control"]["observation_matched_ratio_worst"],
        ]
    )

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

    positions = np.arange(len(methods))
    width = 0.34
    absolute_bars = axes[0].bar(
        positions - width / 2,
        absolute,
        width,
        label="Absolute gate",
        color="#2f8073",
    )
    matched_bars = axes[0].bar(
        positions + width / 2,
        matched,
        width,
        label="K16-matched gate",
        color="#c04f3d",
    )
    axes[0].set_xticks(positions, methods)
    axes[0].set_ylim(0, 14.3)
    axes[0].set_ylabel("Passing cells / 13")
    axes[0].set_title("A  Absolute success does not imply matched accuracy")
    axes[0].grid(axis="y", color="#dfe3df", linewidth=0.8)
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].bar_label(absolute_bars, padding=3, fontweight="bold")
    axes[0].bar_label(matched_bars, padding=3, fontweight="bold")

    p90_bars = axes[1].bar(
        positions - width / 2,
        observation_p90,
        width,
        label="p90-higher",
        color="#d5a139",
    )
    worst_bars = axes[1].bar(
        positions + width / 2,
        observation_worst,
        width,
        label="Worst",
        color="#c04f3d",
    )
    axes[1].axhline(1.05, color="#252a28", linestyle="--", linewidth=1.5, label="Frozen limit 1.05x")
    axes[1].set_xticks(positions, methods)
    axes[1].set_ylim(0, 3.65)
    axes[1].set_ylabel("Observation error / K16 reference")
    axes[1].set_title("B  Observation remains the only systematic blocker")
    axes[1].grid(axis="y", color="#dfe3df", linewidth=0.8)
    axes[1].legend(frameon=False, loc="upper left")
    axes[1].bar_label(p90_bars, labels=[f"{value:.3f}x" for value in observation_p90], padding=3)
    axes[1].bar_label(worst_bars, labels=[f"{value:.3f}x" for value in observation_worst], padding=3)

    figure.suptitle(
        "v258 Case 19: 13/13 absolute cells, 0/13 K16-matched cells",
        fontsize=17,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.055,
        "Independent validation passes 47/47 checks. The route closes without full-sequence expansion, training, GPU, or speedup claims.",
        ha="center",
        color="#545b57",
        fontsize=10.5,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


if __name__ == "__main__":
    main()
