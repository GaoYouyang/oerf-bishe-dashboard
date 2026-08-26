#!/usr/bin/env python3
"""Render the redacted public v252 observation-graph audit figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_observation_graph_traversal_v252_public_summary.json"
OUTPUT = ROOT / "assets/figures/blastnet_case19_observation_graph_traversal_v252.png"


def _label_bars(axis: plt.Axes, bars: object, denominator: int) -> None:
    for bar in bars:
        value = int(bar.get_height())
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + denominator * 0.023,
            f"{value}/{denominator}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    validation = data["independent_validation"]
    diagnostic = data["post_open_discrete_diagnostic"]

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

    checks = validation["numeric_checks"]
    ratios = np.asarray([check["observed"] / check["limit"] for check in checks])
    colors = ["#b94f3c" if ratio > 1 else "#26766f" for ratio in ratios]
    agreement_bars = axes[0].bar(np.arange(4), ratios, color=colors, width=0.62)
    axes[0].axhline(1.0, color="#252a28", linestyle="--", linewidth=1.4)
    axes[0].set_yscale("log")
    axes[0].set_ylim(0.05, 80)
    axes[0].set_xticks(np.arange(4), ("Field", "Metric", "Residual", "Summary"))
    axes[0].set_ylabel("Observed difference / frozen limit")
    axes[0].set_title("A  Independent numeric gate fails")
    axes[0].grid(axis="y", color="#dfe3df", linewidth=0.8)
    for bar, ratio in zip(agreement_bars, ratios, strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            ratio * 1.22,
            f"{ratio:.2f}x",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    labels = ("Graph", "Midpoint\ncontrol", "Chronological", "K16 ref")
    cell_counts = np.asarray(
        (
            diagnostic["graph_absolute_cells"],
            diagnostic["midpoint_control_absolute_cells"],
            diagnostic["chronological_control_absolute_cells"],
            diagnostic["k16_reference_absolute_cells"],
        )
    )
    cell_bars = axes[1].bar(
        np.arange(4), cell_counts, color=("#b94f3c", "#26766f", "#d49a3a", "#6f7d9b"), width=0.62
    )
    axes[1].set_xticks(np.arange(4), labels)
    axes[1].set_ylim(0, 458)
    axes[1].set_ylabel("Absolute cells out of 429")
    axes[1].set_title("B  Post-open discrete diagnostic only")
    axes[1].grid(axis="y", color="#dfe3df", linewidth=0.8)
    _label_bars(axes[1], cell_bars, 429)

    rig_counts = np.asarray(
        (
            diagnostic["graph_absolute_complete_rigs"],
            diagnostic["midpoint_control_absolute_complete_rigs"],
            diagnostic["chronological_control_absolute_complete_rigs"],
            diagnostic["k16_reference_absolute_complete_rigs"],
        )
    )
    rig_bars = axes[2].bar(
        np.arange(4), rig_counts, color=("#b94f3c", "#26766f", "#d49a3a", "#6f7d9b"), width=0.62
    )
    axes[2].set_xticks(np.arange(4), labels)
    axes[2].set_ylim(0, 14.2)
    axes[2].set_ylabel("Absolute complete rigs out of 13")
    axes[2].set_title("C  Equal-cost control passes all rigs")
    axes[2].grid(axis="y", color="#dfe3df", linewidth=0.8)
    _label_bars(axes[2], rig_bars, 13)

    figure.suptitle(
        "v252 Case 19: numeric closure fails; observation-graph traversal remains INCONCLUSIVE",
        fontsize=15.8,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.035,
        "Discrete counts are post-open diagnostics, not validated performance. The graph route closes without a scientific PASS/FAIL upgrade; nominal 9.09% is not effective reduction.",
        ha="center",
        color="#545b57",
        fontsize=9.8,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


if __name__ == "__main__":
    main()
