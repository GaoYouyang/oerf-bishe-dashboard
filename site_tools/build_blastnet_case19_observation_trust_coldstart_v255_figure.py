#!/usr/bin/env python3
"""Render the redacted public v255 independent-contract boundary figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_observation_trust_coldstart_v255_public_summary.json"
OUTPUT = ROOT / "assets/figures/blastnet_case19_observation_trust_coldstart_v255.png"


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    validation = data["independent_validation"]
    diagnostic = data["diagnostic_only"]
    ratios = np.asarray(
        [
            validation["maximum_alpha_absolute_difference"] / validation["alpha_absolute_tolerance"],
            validation["maximum_residual_relative_difference"] / validation["residual_relative_tolerance"],
            validation["maximum_metric_absolute_difference"] / validation["metric_absolute_tolerance"],
        ]
    )
    labels = ("Blend coefficient", "Residual", "Cell metric")

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
    figure.patch.set_facecolor("#f4f6f3")
    figure.subplots_adjust(left=0.12, right=0.965, bottom=0.18, top=0.78, wspace=0.25)
    for axis in axes:
        axis.set_facecolor("#ffffff")

    positions = np.arange(len(labels))
    bars = axes[0].barh(positions, ratios, color=("#b94f3c", "#cf7358", "#d49a3a"), height=0.58)
    axes[0].axvline(1.0, color="#252a28", linestyle="--", linewidth=1.5, label="Frozen pass limit")
    axes[0].set_xscale("log")
    axes[0].set_xlim(0.6, 2.0e4)
    axes[0].set_yticks(positions, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Observed disagreement / frozen tolerance (log scale)")
    axes[0].set_title("A  Three independent checks exceed 1x")
    axes[0].grid(axis="x", which="both", color="#dfe3df", linewidth=0.8)
    axes[0].legend(loc="lower right", frameon=False)
    for bar, ratio in zip(bars, ratios, strict=True):
        axes[0].text(
            ratio * 1.08,
            bar.get_y() + bar.get_height() / 2,
            f"{ratio:.2f}x",
            va="center",
            fontweight="bold",
        )

    axes[1].axis("off")
    axes[1].set_title("B  Claim boundary")
    axes[1].text(0.04, 0.83, "Independent validation", fontsize=12, color="#545b57")
    axes[1].text(
        0.04,
        0.70,
        f"{validation['checks_passed']}/{validation['checks_total']} checks",
        fontsize=24,
        fontweight="bold",
        color="#b94f3c",
    )
    primary = diagnostic["primary"]
    axes[1].text(0.04, 0.52, "DISCRETE DIAGNOSTIC ONLY", fontsize=12, fontweight="bold", color="#6f746f")
    axes[1].text(
        0.04,
        0.36,
        f"Primary: {primary['absolute_safe_cells']}/429 absolute\n"
        f"and {primary['matched_safe_cells']}/429 matched",
        fontsize=15,
        color="#6f746f",
    )
    axes[1].text(
        0.04,
        0.13,
        "Not admissible as a pass, fail, headroom,\ncall reduction, wall, RSS, or speed result.",
        fontsize=12,
        color="#545b57",
    )

    figure.suptitle(
        "v255 Case 19: observation-trust cold start remains INCONCLUSIVE",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.055,
        "Frozen numerical agreement is part of the scientific gate. No tolerance relaxation or rerun; algorithm_breakthrough=false.",
        ha="center",
        color="#545b57",
        fontsize=10.5,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


if __name__ == "__main__":
    main()
