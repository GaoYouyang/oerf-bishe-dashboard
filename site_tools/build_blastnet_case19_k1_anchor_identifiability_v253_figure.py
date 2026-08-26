#!/usr/bin/env python3
"""Render the redacted public v253 K1-anchor identifiability figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_k1_anchor_identifiability_v253_public_summary.json"
OUTPUT = ROOT / "assets/figures/blastnet_case19_k1_anchor_identifiability_v253.png"


def _label_bars(axis: plt.Axes, bars: object, denominator: int) -> None:
    for bar in bars:
        value = int(bar.get_height())
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + denominator * 0.024,
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
    figure.subplots_adjust(left=0.055, right=0.985, bottom=0.27, top=0.79, wspace=0.28)
    for axis in axes:
        axis.set_facecolor("#ffffff")

    labels = ("K1 residual\nprimary", "Minimum norm\ncontrol", "Cosine medoid\ncontrol", "Midpoint\ndiagnostic")
    safe_rigs = np.asarray(
        (
            results["primary"]["safe_rigs"],
            results["minimum_norm_control"]["safe_rigs"],
            results["cosine_medoid_control"]["safe_rigs"],
            results["fixed_midpoint_diagnostic"]["safe_rigs"],
        )
    )
    bars = axes[0].bar(
        np.arange(4), safe_rigs, color=("#b94f3c", "#8b6d42", "#d49a3a", "#6f7d9b"), width=0.62
    )
    bars[-1].set_hatch("///")
    axes[0].axhline(13, color="#252a28", linestyle="--", linewidth=1.4)
    axes[0].set_xticks(np.arange(4), labels)
    axes[0].set_ylim(0, 14.4)
    axes[0].set_ylabel("Safe complete rigs out of 13")
    axes[0].set_title("A  Primary misses the complete-rig gate")
    axes[0].grid(axis="y", color="#dfe3df", linewidth=0.8)
    _label_bars(axes[0], bars, 13)

    screen_calls = np.asarray((66, 0, 0))
    call_bars = axes[1].bar(
        np.arange(3), screen_calls, color=("#b94f3c", "#8b6d42", "#d49a3a"), width=0.62
    )
    axes[1].set_xticks(np.arange(3), ("K1 residual", "Minimum norm", "Cosine medoid"))
    axes[1].set_ylim(0, 75)
    axes[1].set_ylabel("Exact A plus A^T calls per rig")
    axes[1].set_title("B  Screening cost is not isolated value")
    axes[1].grid(axis="y", color="#dfe3df", linewidth=0.8)
    for bar, value in zip(call_bars, screen_calls, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            str(int(value)),
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    checks = validation["checks_passed"]
    nominal_percent = costs["nominal_combined_arithmetic_difference_fraction"] * 100
    axes[2].axis("off")
    axes[2].set_title("C  Validation and claim boundary")
    axes[2].text(0.05, 0.76, f"Independent checks\n{checks}/{validation['checks_total']}", fontsize=18, fontweight="bold")
    axes[2].text(0.05, 0.48, "Primary = minimum-norm control\nSame anchors and same 9/13 roster", fontsize=13)
    axes[2].text(0.05, 0.22, f"Nominal arithmetic difference: {nominal_percent:.4f}%\nNo full traversal, wall/RSS, or speed claim", fontsize=12, color="#545b57")

    figure.suptitle(
        "v253 Case 19: K1 residual-contraction anchoring fails identifiability",
        fontsize=15.8,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.035,
        "The midpoint diagnostic is time-index dependent and inadmissible. The K1 anchor hypothesis closes; algorithm_breakthrough=false.",
        ha="center",
        color="#545b57",
        fontsize=9.8,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


if __name__ == "__main__":
    main()
