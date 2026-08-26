#!/usr/bin/env python3
"""Render the redacted public v259 residual-attribution figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_observation_excess_attribution_v259_public_summary.json"
OUTPUT = ROOT / "assets/figures/blastnet_case19_observation_excess_attribution_v259.png"


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    primary = data["primary_attribution"]
    labels = ("Camera top-3", "Component", "High frequency")
    localized = np.asarray(
        [
            primary["camera"]["localized_rigs"],
            primary["component"]["localized_rigs"],
            primary["frequency"]["localized_rigs"],
        ]
    )
    concentration = np.asarray(
        [
            primary["camera"]["top_three_share_p50_higher"],
            primary["component"]["component_positive_share_p50"][1],
            primary["frequency"]["dominant_share_p50_higher"],
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

    colors = ("#2f8073", "#d5a139", "#c04f3d")
    positions = np.arange(len(labels))
    bars = axes[0].bar(positions, localized, color=colors, width=0.62)
    axes[0].axhline(10, color="#252a28", linestyle="--", linewidth=1.5, label="Frozen gate: 10 rigs")
    axes[0].set_xticks(positions, labels)
    axes[0].set_ylim(0, 14.3)
    axes[0].set_ylabel("Localized rigs / 13")
    axes[0].set_title("A  Excess is structured across all three views")
    axes[0].grid(axis="y", color="#dfe3df", linewidth=0.8)
    axes[0].legend(frameon=False, loc="lower right")
    axes[0].bar_label(bars, labels=[f"{value}/13" for value in localized], padding=3, fontweight="bold")

    bars = axes[1].bar(positions, concentration, color=colors, width=0.62)
    axes[1].axhline(0.75, color="#252a28", linestyle="--", linewidth=1.5, label="Frozen share gate: 0.75")
    axes[1].set_xticks(positions, labels)
    axes[1].set_ylim(0, 1.12)
    axes[1].set_ylabel("Median dominant positive-excess share")
    axes[1].set_title("B  Concentration is strongest in component 2")
    axes[1].grid(axis="y", color="#dfe3df", linewidth=0.8)
    axes[1].legend(frameon=False, loc="lower right")
    axes[1].bar_label(bars, labels=[f"{value:.3f}" for value in concentration], padding=3, fontweight="bold")

    figure.suptitle(
        "v259 Case 19: residual excess is structured, not diffuse",
        fontsize=17,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.055,
        "Independent 18/18; 0A+0AT; post-open attribution only. Frozen priority authorizes one camera-local diagnostic, not an algorithm claim.",
        ha="center",
        color="#545b57",
        fontsize=10.5,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


if __name__ == "__main__":
    main()
