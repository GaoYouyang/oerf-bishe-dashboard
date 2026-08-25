#!/usr/bin/env python3
"""Render the redacted public v250 Case 19 control-attribution figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_charbonnier_graph_diffusion_v250_public_summary.json"
OUTPUT = ROOT / "assets/figures/blastnet_case19_charbonnier_graph_diffusion_v250.png"


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    execution = data["execution"]
    results = data["independent_results"]

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
    figure.subplots_adjust(left=0.055, right=0.985, bottom=0.28, top=0.79, wspace=0.27)
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
    axes[0].set_ylim(0, 41)
    axes[0].set_ylabel("Independent validation checks")
    axes[0].set_title("A  Independent recomputation closes")
    axes[0].grid(axis="y", color="#dfe3df", linewidth=0.8)
    for bar, value in zip(bars, checks, strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.75,
            str(int(value)),
            ha="center",
            fontweight="bold",
        )

    counts = np.asarray(
        (
            results["charbonnier_primary"]["strict_safe_cells"],
            results["equal_call_linear_heat_control"]["strict_safe_cells"],
            results["raw_k14_control"]["strict_safe_cells"],
            results["k16_reference"]["strict_safe_cells"],
        )
    )
    labels = ("Charbonnier", "Linear heat", "Raw K14", "K16 ref")
    count_bars = axes[1].bar(
        np.arange(4),
        counts,
        color=("#26766f", "#d49a3a", "#8f9692", "#6f7d9b"),
        width=0.64,
    )
    axes[1].set_xticks(np.arange(4), labels, rotation=10)
    axes[1].set_ylim(0, 14)
    axes[1].set_ylabel("Strict-safe frame-zero cells out of 13")
    axes[1].set_title("B  Equal-call control also reaches 13/13")
    axes[1].grid(axis="y", color="#dfe3df", linewidth=0.8)
    for bar, value in zip(count_bars, counts, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.35,
            str(int(value)),
            ha="center",
            fontweight="bold",
        )

    metric_labels = ("Field", "Full grad", "Interior grad", "Observation")
    limits = np.asarray(results["absolute_limits"], dtype=np.float64)
    primary = np.asarray(results["charbonnier_primary"]["p90_higher"], dtype=np.float64) / limits
    control = np.asarray(
        results["equal_call_linear_heat_control"]["p90_higher"], dtype=np.float64
    ) / limits
    positions = np.arange(4)
    width = 0.35
    axes[2].bar(positions - width / 2, primary, width, label="Charbonnier", color="#26766f")
    axes[2].bar(positions + width / 2, control, width, label="Linear heat", color="#d49a3a")
    axes[2].axhline(1.0, color="#252a28", linestyle="--", linewidth=1.4)
    axes[2].set_xticks(positions, metric_labels, rotation=12)
    axes[2].set_ylim(0, 1.12)
    axes[2].set_ylabel("p90 / frozen frame-zero limit")
    axes[2].set_title("C  Both remain inside every p90 limit")
    axes[2].legend(
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#dfe3df",
        framealpha=1.0,
        loc="upper left",
    )
    axes[2].grid(axis="y", color="#dfe3df", linewidth=0.8)

    figure.suptitle(
        "v250 Case 19: frame-zero smoothing passes, nonlinear advantage is not isolated",
        fontsize=16.2,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.025,
        "Equal-call linear heat also passes 13/13. No 429-cell continuation, wall/RSS, or breakthrough claim.",
        ha="center",
        color="#545b57",
        fontsize=10,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


if __name__ == "__main__":
    main()
