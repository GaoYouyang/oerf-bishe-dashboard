#!/usr/bin/env python3
"""Render the redacted public v257 independent-contract boundary figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_observable_active_support_v257_public_summary.json"
OUTPUT = ROOT / "assets/figures/blastnet_case19_observable_active_support_v257.png"


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    validation = data["independent_validation"]
    ratios = np.asarray(
        [
            validation["maximum_final_field_relative_difference"]
            / validation["field_relative_tolerance"],
            validation["maximum_observation_normalized_residual_difference"]
            / validation["observation_normalized_residual_tolerance"],
            validation["maximum_metric_absolute_difference"]
            / validation["metric_and_summary_absolute_tolerance"],
            validation["maximum_summary_absolute_difference"]
            / validation["metric_and_summary_absolute_tolerance"],
        ]
    )
    labels = ("Final field", "Normalized residual", "Cell metric", "Summary")
    colors = ["#5b7f71" if ratio <= 1.0 else "#b94f3c" for ratio in ratios]

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
    figure.subplots_adjust(left=0.13, right=0.965, bottom=0.18, top=0.78, wspace=0.28)
    for axis in axes:
        axis.set_facecolor("#ffffff")

    positions = np.arange(len(labels))
    bars = axes[0].barh(positions, ratios, color=colors, height=0.58)
    axes[0].axvline(1.0, color="#252a28", linestyle="--", linewidth=1.5, label="Frozen pass limit")
    axes[0].set_xscale("log")
    axes[0].set_xlim(1.0e-2, 3.0)
    axes[0].set_yticks(positions, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Observed disagreement / frozen tolerance (log scale)")
    axes[0].set_title("A  One preregistered check exceeds 1x")
    axes[0].grid(axis="x", which="both", color="#dfe3df", linewidth=0.8)
    axes[0].legend(loc="upper right", frameon=False)
    for bar, ratio in zip(bars, ratios, strict=True):
        label = f"{ratio:.3f}x" if ratio >= 0.1 else f"{ratio:.3f}x"
        axes[0].text(
            max(ratio * 1.08, 0.012),
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontweight="bold",
            color="#2b302d",
        )

    axes[1].axis("off")
    axes[1].set_title("B  Claim boundary")
    axes[1].text(0.04, 0.84, "Independent validation", fontsize=12, color="#545b57")
    axes[1].text(
        0.04,
        0.70,
        f"{validation['checks_passed']}/{validation['checks_total']} checks",
        fontsize=25,
        fontweight="bold",
        color="#b94f3c",
    )
    axes[1].text(
        0.04,
        0.56,
        f"Sole failure: normalized residual at {validation['residual_limit_ratio']:.3f}x",
        fontsize=13,
        fontweight="bold",
        color="#b94f3c",
    )
    axes[1].text(0.04, 0.40, "EXACT DISCRETE AGREEMENT", fontsize=12, fontweight="bold", color="#5b7f71")
    axes[1].text(
        0.04,
        0.27,
        "Support masks and selected counts agree exactly,\nbut cannot override the continuous failure.",
        fontsize=14,
        color="#545b57",
    )
    axes[1].text(
        0.04,
        0.07,
        "No performance counts, effective call reduction,\nwall, RSS, external, or real-BOST claim is admissible.",
        fontsize=12,
        color="#545b57",
    )

    figure.suptitle(
        "v257 Case 19: observable active-support correction remains INCONCLUSIVE",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.055,
        "The mechanism closes without tolerance relaxation, rerun, threshold search, or full-sequence expansion; algorithm_breakthrough=false.",
        ha="center",
        color="#545b57",
        fontsize=10.5,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


if __name__ == "__main__":
    main()
