#!/usr/bin/env python3
"""Render the redacted public v254 paired-K1 subspace verdict figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_k1_set_subspace_v254_public_summary.json"
OUTPUT = ROOT / "assets/figures/blastnet_case19_k1_set_subspace_v254.png"


def _bar_labels(axis: plt.Axes, bars: object, denominator: int) -> None:
    for bar in bars:
        value = int(bar.get_height())
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + denominator * 0.022,
            f"{value}/{denominator}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    results = data["results"]
    validation = data["independent_validation"]
    costs = data["cost_ledger"]

    labels = ("K1 set\nsubspace", "Self-K1\nrestart", "Zero PCGLS\nK15", "FIFO16 K14\ndiagnostic", "Robust K16\nreference")
    colors = ("#b94f3c", "#8b6d42", "#d49a3a", "#6f7d9b", "#447c69")
    absolute = np.asarray(
        (
            results["primary"]["absolute_safe_cells"],
            results["self_k1_restart_control"]["absolute_safe_cells"],
            results["zero_pcgls_k15_control"]["absolute_safe_cells"],
            results["sealed_chronological_fifo16_k14_diagnostic"]["absolute_safe_cells"],
            results["robust_k16_reference"]["absolute_safe_cells"],
        )
    )
    matched = np.asarray(
        (
            results["primary"]["matched_safe_cells"],
            results["self_k1_restart_control"]["matched_safe_cells"],
            results["zero_pcgls_k15_control"]["matched_safe_cells"],
            results["sealed_chronological_fifo16_k14_diagnostic"]["matched_safe_cells"],
        )
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.2,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(19, 6.8))
    figure.patch.set_facecolor("#f4f6f3")
    figure.subplots_adjust(left=0.05, right=0.985, bottom=0.27, top=0.79, wspace=0.28)
    for axis in axes:
        axis.set_facecolor("#ffffff")

    bars = axes[0].bar(np.arange(5), absolute, color=colors, width=0.64)
    bars[3].set_hatch("///")
    axes[0].axhline(429, color="#252a28", linestyle="--", linewidth=1.3)
    axes[0].set_xticks(np.arange(5), labels)
    axes[0].set_ylim(0, 470)
    axes[0].set_ylabel("Absolute-safe cells out of 429")
    axes[0].set_title("A  Primary misses absolute accuracy")
    axes[0].grid(axis="y", color="#dfe3df", linewidth=0.8)
    _bar_labels(axes[0], bars, 429)

    bars = axes[1].bar(np.arange(4), matched, color=colors[:4], width=0.64)
    bars[3].set_hatch("///")
    axes[1].axhline(429, color="#252a28", linestyle="--", linewidth=1.3)
    axes[1].set_xticks(np.arange(4), labels[:4])
    axes[1].set_ylim(0, 470)
    axes[1].set_ylabel("K16-matched cells out of 429")
    axes[1].set_title("B  Primary has zero matched cells")
    axes[1].grid(axis="y", color="#dfe3df", linewidth=0.8)
    _bar_labels(axes[1], bars, 429)

    axes[2].axis("off")
    axes[2].set_title("C  Validation and claim boundary")
    checks = validation["checks_passed"]
    nominal = costs["nominal_combined_exact_call_reduction_fraction"] * 100
    axes[2].text(0.04, 0.77, f"Independent checks\n{checks}/{validation['checks_total']}", fontsize=18, fontweight="bold")
    axes[2].text(0.04, 0.49, "Primary whole sequence\n495A + 495A^T", fontsize=14)
    axes[2].text(0.04, 0.28, "K16 reference\n528A + 528A^T", fontsize=14)
    axes[2].text(0.04, 0.08, f"Nominal difference {nominal:.2f}%\nNo effective call, wall, RSS, or speed claim", fontsize=11.5, color="#545b57")

    figure.suptitle(
        "v254 Case 19: unordered paired-K1 rank-16 subspace fails matched accuracy",
        fontsize=15.8,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.035,
        "The chronological diagnostic is not an unordered-set method and still misses absolute accuracy. algorithm_breakthrough=false.",
        ha="center",
        color="#545b57",
        fontsize=9.7,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


if __name__ == "__main__":
    main()
