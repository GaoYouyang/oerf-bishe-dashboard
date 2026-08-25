#!/usr/bin/env python3
"""Render the redacted public v244.2 Case 19 decision figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_canonical_camera_actual_k14_external_v244_2_public_summary.json"
OUTPUT = ROOT / "assets/figures/blastnet_case19_canonical_camera_actual_k14_external_v244_2.png"


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    diagnostics = data["post_open_diagnostic_only"]
    primary = diagnostics["primary_warm_k14"]
    reference = diagnostics["k16_reference"]
    controls = diagnostics["controls"]
    validation = data["independent_validation"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(19, 6.8), constrained_layout=True)
    figure.patch.set_facecolor("#f5f7f4")
    for axis in axes:
        axis.set_facecolor("#ffffff")

    check_values = np.array(
        [validation["checks_passed"], validation["checks_total"] - validation["checks_passed"]]
    )
    axes[0].bar((0, 1), check_values, color=("#23766f", "#bd5c45"), width=0.58)
    axes[0].set_xticks((0, 1), ("Passed", "Failed"))
    axes[0].set_ylim(0, 31)
    axes[0].set_ylabel("Independent checks")
    axes[0].set_title("A  Independent validation: 26 / 29")
    axes[0].grid(axis="y", color="#dfe3df", linewidth=0.8)
    for index, value in enumerate(check_values):
        axes[0].text(index, value + 0.7, str(int(value)), ha="center", fontweight="bold")

    labels = ("Warm K14", "K16 ref", "BP J-K13", "BP K13", "Zero J-K14", "Zero K14")
    absolute_rigs = np.array(
        [
            primary["absolute_complete_rigs_passed"],
            reference["absolute_complete_rigs_passed"],
            *[row["absolute_complete_rigs_passed"] for row in controls],
        ]
    )
    colors = ["#315f93", "#66727d", "#8a651b", "#8a651b", "#8a651b", "#8a651b"]
    bars = axes[1].bar(np.arange(len(labels)), absolute_rigs, color=colors)
    axes[1].axhline(13, color="#202523", linestyle="--", linewidth=1.5, label="Required 13 / 13")
    axes[1].set_xticks(np.arange(len(labels)), labels, rotation=21, ha="right")
    axes[1].set_ylim(0, 14.7)
    axes[1].set_ylabel("Absolute complete rigs out of 13")
    axes[1].set_title("B  Post-open diagnostic: reference inadequate")
    axes[1].grid(axis="y", color="#dfe3df", linewidth=0.8)
    axes[1].legend(frameon=False, loc="upper right")
    for bar, value in zip(bars, absolute_rigs, strict=True):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.35, str(int(value)), ha="center", fontweight="bold")

    maxima = validation["maximum_differences"]
    limits = validation["frozen_tolerances"]
    ratio_labels = ("Field", "Residual", "Metric", "Summary")
    ratios = np.array(
        [
            maxima["field_relative"] / limits["field_relative"],
            maxima["residual_relative"] / limits["residual_relative"],
            maxima["metric_absolute"] / limits["metric_absolute"],
            maxima["summary_absolute"] / limits["summary_absolute"],
        ]
    )
    ratio_colors = ["#23766f" if value <= 1.0 else "#bd5c45" for value in ratios]
    ratio_bars = axes[2].bar(np.arange(4), ratios, color=ratio_colors)
    axes[2].axhline(1.0, color="#202523", linestyle="--", linewidth=1.5, label="Frozen limit")
    axes[2].set_xticks(np.arange(4), ratio_labels, rotation=18, ha="right")
    axes[2].set_yscale("log")
    axes[2].set_ylim(0.01, 20)
    axes[2].set_ylabel("Formal / independent difference divided by limit")
    axes[2].set_title("C  Residual and metric checks exceed limits")
    axes[2].grid(axis="y", color="#dfe3df", linewidth=0.8, which="both")
    axes[2].legend(frameon=False, loc="upper left")
    for bar, value in zip(ratio_bars, ratios, strict=True):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            value * (1.23 if value >= 0.1 else 1.5),
            f"{value:.3g}x",
            ha="center",
            fontweight="bold",
        )

    figure.suptitle(
        "v244.2 Case 19: one-shot confirmation remains inconclusive",
        fontsize=17,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.012,
        "Authoritative decision is inconclusive; 9/13 reference and 12/13 primary are post-open diagnostics only. No wall/RSS gate.",
        ha="center",
        color="#545b57",
        fontsize=10,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


if __name__ == "__main__":
    main()
