#!/usr/bin/env python3
"""Render the redacted public v248 Case 19 validation-boundary figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_geometry_voxel_block_jacobi_v248_public_summary.json"
OUTPUT = ROOT / "assets/figures/blastnet_case19_geometry_voxel_block_jacobi_v248.png"


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    agreement = data["formal_independent_agreement"]
    execution = data["execution"]
    diagnostic = data["diagnostic_only_not_an_admissible_performance_verdict"]

    labels = ("Blocks", "Inverses", "Fields", "Residuals", "Metrics", "Summaries")
    ratios = np.asarray(
        (
            agreement["exact_block_relative_maximum"] / agreement["exact_block_relative_limit"],
            agreement["inverse_block_relative_maximum"] / agreement["inverse_block_relative_limit"],
            agreement["field_relative_maximum"] / agreement["field_relative_limit"],
            agreement["normalized_residual_difference"] / agreement["normalized_residual_difference_limit"],
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
    figure.subplots_adjust(left=0.055, right=0.985, bottom=0.28, top=0.79, wspace=0.25)
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
    axes[0].set_ylim(0, 30)
    axes[0].set_ylabel("Independent validation checks")
    axes[0].set_title("A  Validation is 26/28, not a pass")
    axes[0].grid(axis="y", color="#dfe3df", linewidth=0.8)
    for bar, value in zip(bars, checks, strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.65,
            str(int(value)),
            ha="center",
            fontweight="bold",
        )

    positions = np.arange(len(labels))
    axes[1].barh(positions, ratios, color="#26766f", height=0.62)
    axes[1].axvline(1.0, color="#252a28", linestyle="--", linewidth=1.4)
    axes[1].set_xscale("log")
    axes[1].set_yticks(positions, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Observed difference / frozen limit")
    axes[1].set_title("B  Six tolerant comparisons pass")
    axes[1].grid(axis="x", which="both", color="#dfe3df", linewidth=0.8)
    axes[1].text(
        0.5,
        -0.32,
        "Observation arrays differ by 1.33e-15.\nFrozen requirement: exact equality.",
        transform=axes[1].transAxes,
        ha="center",
        color="#8b3024",
        fontweight="bold",
        fontsize=9.5,
    )

    diagnostic_values = np.asarray(
        (
            diagnostic["primary_strict_safe_cells"],
            diagnostic["diagonal_jacobi_control_strict_safe_cells"],
            diagnostic["unpreconditioned_control_strict_safe_cells"],
        )
    )
    diagnostic_labels = ("Voxel block", "Diagonal", "CGLS")
    diagnostic_bars = axes[2].bar(
        np.arange(3),
        diagnostic_values,
        color=("#b94f3c", "#9aa09c", "#9aa09c"),
        edgecolor="#5d625f",
        hatch="//",
        width=0.62,
    )
    axes[2].set_xticks(np.arange(3), diagnostic_labels)
    axes[2].set_ylim(0, 14)
    axes[2].set_ylabel("Strict-safe frame-zero cells out of 13")
    axes[2].set_title("C  Diagnostic only: primary is 0/13")
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
        -0.26,
        "Not an independently admissible performance verdict",
        transform=axes[2].transAxes,
        ha="center",
        color="#8b3024",
        fontweight="bold",
    )

    figure.suptitle(
        "v248 Case 19: fixed voxel-block frame-zero gate is inconclusive and retired",
        fontsize=16.5,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.025,
        "No tolerance relaxation or full-sequence run. The fixed 4x2x2 mechanism is retired.",
        ha="center",
        color="#545b57",
        fontsize=10,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


if __name__ == "__main__":
    main()
