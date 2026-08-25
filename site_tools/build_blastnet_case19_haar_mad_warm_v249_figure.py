#!/usr/bin/env python3
"""Render the redacted public v249 Case 19 validation-boundary figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_haar_mad_warm_v249_public_summary.json"
OUTPUT = ROOT / "assets/figures/blastnet_case19_haar_mad_warm_v249.png"


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    agreement = data["formal_independent_agreement"]
    execution = data["execution"]
    diagnostic = data["diagnostic_only_not_an_admissible_performance_verdict"]

    labels = (
        "Haar coeffs",
        "Initializer",
        "Field",
        "Residual",
        "Observation",
        "Metrics",
        "Summary",
    )
    ratios = np.asarray(
        (
            agreement["haar_coefficient_relative_maximum"]
            / agreement["haar_coefficient_relative_limit"],
            agreement["corrected_initializer_relative_maximum"]
            / agreement["corrected_initializer_relative_limit"],
            agreement["field_relative_maximum"] / agreement["field_relative_limit"],
            agreement["normalized_residual_difference"]
            / agreement["normalized_residual_difference_limit"],
            agreement["observation_normalized_difference"]
            / agreement["observation_normalized_difference_limit"],
            agreement["metric_absolute_maximum"] / agreement["metric_absolute_limit"],
            agreement["summary_absolute_maximum"] / agreement["summary_absolute_limit"],
        ),
        dtype=np.float64,
    )

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
    figure.subplots_adjust(left=0.055, right=0.985, bottom=0.29, top=0.79, wspace=0.27)
    for axis in axes:
        axis.set_facecolor("#ffffff")

    checks = np.asarray(
        (
            execution["independent_checks_passed"],
            execution["independent_checks_total"] - execution["independent_checks_passed"],
        )
    )
    bars = axes[0].bar((0, 1), checks, color=("#26766f", "#b94f3c"), width=0.58)
    axes[0].set_xticks((0, 1), ("Passed", "Failed"))
    axes[0].set_ylim(0, 38)
    axes[0].set_ylabel("Independent validation checks")
    axes[0].set_title("A  Validation is 33/35, not a pass")
    axes[0].grid(axis="y", color="#dfe3df", linewidth=0.8)
    for bar, value in zip(bars, checks, strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.75,
            str(int(value)),
            ha="center",
            fontweight="bold",
        )

    positions = np.arange(len(labels))
    colors = np.where(ratios > 1.0, "#b94f3c", "#26766f")
    axes[1].barh(positions, ratios, color=colors, height=0.62)
    axes[1].axvline(1.0, color="#252a28", linestyle="--", linewidth=1.4)
    axes[1].set_xscale("log")
    axes[1].set_yticks(positions, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Observed difference / frozen limit")
    axes[1].set_title("B  Coefficient agreement misses by 18.2x")
    axes[1].grid(axis="x", which="both", color="#dfe3df", linewidth=0.8)
    axes[1].text(
        0.5,
        -0.34,
        "Haar pair transforms agree near 6.83e-16.\nThe frozen gate still applies to propagated coefficients.",
        transform=axes[1].transAxes,
        ha="center",
        color="#8b3024",
        fontweight="bold",
        fontsize=9.2,
    )

    diagnostic_values = np.asarray(
        (
            diagnostic["primary_strict_safe_cells"],
            diagnostic["approximation_only_control_strict_safe_cells"],
            diagnostic["raw_k14_control_strict_safe_cells"],
            diagnostic["k16_reference_strict_safe_cells"],
        )
    )
    diagnostic_labels = ("Haar-MAD", "LLL only", "Raw K14", "K16 ref")
    diagnostic_bars = axes[2].bar(
        np.arange(4),
        diagnostic_values,
        color=("#26766f", "#9aa09c", "#9aa09c", "#9aa09c"),
        edgecolor="#5d625f",
        hatch="//",
        width=0.62,
    )
    axes[2].set_xticks(np.arange(4), diagnostic_labels, rotation=12)
    axes[2].set_ylim(0, 14)
    axes[2].set_ylabel("Strict-safe frame-zero cells out of 13")
    axes[2].set_title("C  Diagnostic only: primary is 13/13")
    axes[2].grid(axis="y", color="#dfe3df", linewidth=0.8)
    for bar, value in zip(diagnostic_bars, diagnostic_values, strict=True):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.35,
            str(int(value)),
            ha="center",
            fontweight="bold",
        )
    axes[2].text(
        0.5,
        -0.28,
        "Matching discrete counts do not repair the independent gate",
        transform=axes[2].transAxes,
        ha="center",
        color="#8b3024",
        fontweight="bold",
        fontsize=9.2,
    )

    figure.suptitle(
        "v249 Case 19: Haar-MAD frame-zero gate is inconclusive",
        fontsize=16.5,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.025,
        "Coefficient agreement misses the frozen limit. No relaxed rerun or full 429-cell sequence.",
        ha="center",
        color="#545b57",
        fontsize=10,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


if __name__ == "__main__":
    main()
