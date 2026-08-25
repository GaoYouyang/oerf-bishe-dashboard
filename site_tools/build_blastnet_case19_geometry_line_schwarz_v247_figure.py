#!/usr/bin/env python3
"""Render the redacted public v247 Case 19 validation-boundary figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_geometry_line_schwarz_v247_public_summary.json"
OUTPUT = ROOT / "assets/figures/blastnet_case19_geometry_line_schwarz_v247.png"


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    agreement = data["formal_independent_agreement"]
    execution = data["execution"]
    diagnostic = data["diagnostic_only_not_an_admissible_performance_verdict"]

    labels = ("Line blocks", "Line inverses", "Fields", "Residuals", "Metrics", "Summaries")
    values = np.array(
        [
            agreement["line_exact_block_relative_maximum"] / agreement["line_exact_block_relative_limit"],
            agreement["line_inverse_block_relative_maximum"] / agreement["line_inverse_block_relative_limit"],
            agreement["field_relative_maximum"] / agreement["field_relative_limit"],
            agreement["residual_relative_maximum"] / agreement["residual_relative_limit"],
            agreement["metric_absolute_maximum"] / agreement["metric_absolute_limit"],
            agreement["summary_absolute_maximum"] / agreement["summary_absolute_limit"],
        ]
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
    figure.subplots_adjust(left=0.055, right=0.985, bottom=0.19, top=0.79, wspace=0.24)
    for axis in axes:
        axis.set_facecolor("#ffffff")

    colors = ["#26766f" if value <= 1.0 else "#b94f3c" for value in values]
    positions = np.arange(len(labels))
    axes[0].barh(positions, values, color=colors, height=0.62)
    axes[0].axvline(1.0, color="#252a28", linestyle="--", linewidth=1.4)
    axes[0].set_xscale("log")
    axes[0].set_yticks(positions, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Observed difference / frozen limit")
    axes[0].set_title("A  Residual agreement exceeds its limit")
    axes[0].grid(axis="x", which="both", color="#dfe3df", linewidth=0.8)
    axes[0].text(values[3] * 1.08, 3, f"{values[3]:.2f}x", va="center", fontweight="bold", color="#8b3024")

    checks = np.array(
        [execution["independent_checks_passed"], execution["independent_checks_total"] - execution["independent_checks_passed"]]
    )
    check_bars = axes[1].bar((0, 1), checks, color=("#26766f", "#b94f3c"), width=0.58)
    axes[1].set_xticks((0, 1), ("Passed", "Failed"))
    axes[1].set_ylim(0, 29)
    axes[1].set_ylabel("Independent validation checks")
    axes[1].set_title("B  Validation is 25/27, not a pass")
    axes[1].grid(axis="y", color="#dfe3df", linewidth=0.8)
    for bar, value in zip(check_bars, checks, strict=True):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.65, str(int(value)), ha="center", fontweight="bold")

    diagnostic_values = np.array(
        [
            diagnostic["primary_strict_safe_cells"],
            diagnostic["k16_reference_strict_safe_cells"],
            diagnostic["zero_k14_control_strict_safe_cells"],
            diagnostic["normalized_bp_k13_control_strict_safe_cells"],
        ]
    )
    diagnostic_labels = ("Primary", "K16 ref", "Zero K14", "BP K13")
    diagnostic_bars = axes[2].bar(
        np.arange(4), diagnostic_values, color="#9aa09c", edgecolor="#5d625f", hatch="//", width=0.62
    )
    axes[2].set_xticks(np.arange(4), diagnostic_labels)
    axes[2].set_ylim(0, 92)
    axes[2].set_ylabel("Strict-safe cells out of 429")
    axes[2].set_title("C  Diagnostic only: 0/13 rigs for every arm")
    axes[2].grid(axis="y", color="#dfe3df", linewidth=0.8)
    for bar, value in zip(diagnostic_bars, diagnostic_values, strict=True):
        axes[2].text(bar.get_x() + bar.get_width() / 2, value + 2.2, str(int(value)), ha="center", fontweight="bold")
    axes[2].text(0.5, -0.22, "Not an independently admissible performance verdict", transform=axes[2].transAxes, ha="center", color="#8b3024", fontweight="bold")

    figure.suptitle(
        "v247 Case 19: exact line-Schwarz validation remains inconclusive",
        fontsize=16.5,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.035,
        "No rerun or tolerance relaxation. The mechanism is retired operationally, not proven mathematically impossible.",
        ha="center",
        color="#545b57",
        fontsize=10,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


if __name__ == "__main__":
    main()
