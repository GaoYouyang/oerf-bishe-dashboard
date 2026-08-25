#!/usr/bin/env python3
"""Render the public v240 causal reachable-span capacity figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case7_causal_reachable_span_capacity_v240_public_summary.json"
OUTPUT = ROOT / "assets/figures/blastnet_case7_causal_reachable_span_capacity_v240.png"


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    results = data["results"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(18, 6.5), constrained_layout=True)
    figure.patch.set_facecolor("#f7f8f6")
    for axis in axes:
        axis.set_facecolor("#ffffff")

    labels = ("Field", "Full grad", "Interior grad", "Observation")
    failures = np.array(list(results["metric_cell_failures"].values()))
    colors = ("#315f93", "#8a651b", "#247a70", "#d05a42")
    axes[0].bar(np.arange(4), failures, color=colors)
    axes[0].set_xticks(np.arange(4), labels, rotation=16, ha="right")
    axes[0].set_ylim(0, 575)
    axes[0].set_ylabel("Failed later cells out of 533")
    axes[0].set_title("A  Every later cell fails a necessary bound")
    axes[0].grid(axis="y", color="#dfe3df", linewidth=0.8)
    for index, value in enumerate(failures):
        axes[0].text(index, value + 10, str(int(value)), ha="center", fontweight="bold")
    axes[0].text(
        1.5,
        525,
        "necessary-safe: 0 / 533\ncomplete rigs: 0 / 13",
        ha="center",
        va="top",
        color="#7f3528",
        fontweight="bold",
    )

    p90 = np.array(list(results["minimum_metric_p90_higher"].values()))
    limits = np.array(list(results["absolute_limits"].values()))
    x = np.arange(4)
    width = 0.35
    axes[1].bar(x - width / 2, p90, width, color="#d05a42", label="Oracle lower bound")
    axes[1].bar(x + width / 2, limits, width, color="#247a70", label="Absolute limit")
    axes[1].set_xticks(x, labels, rotation=16, ha="right")
    axes[1].set_ylim(0, 0.82)
    axes[1].set_ylabel("Relative error")
    axes[1].set_title("B  Metric-wise oracle p90 lower bounds")
    axes[1].grid(axis="y", color="#dfe3df", linewidth=0.8)
    axes[1].legend(frameon=False, loc="upper left")
    for index, value in enumerate(p90):
        axes[1].text(index - width / 2, value + 0.015, f"{value:.3f}", ha="center")

    matched = np.array(results["matched_observation_p90_lower_bounds_by_rig"])
    rig_x = np.arange(1, len(matched) + 1)
    axes[2].plot(rig_x, matched, marker="o", color="#d05a42", linewidth=2)
    axes[2].axhline(
        results["matched_complete_rig_p90_limit"],
        color="#212121",
        linestyle="--",
        linewidth=1.6,
        label="Matched p90 limit 1.02",
    )
    axes[2].set_xticks(rig_x)
    axes[2].set_ylim(0, 7.65)
    axes[2].set_xlabel("Rig")
    axes[2].set_ylabel("Observation / K16 p90 lower bound")
    axes[2].set_title("C  Observation capacity fails all 13 rigs")
    axes[2].grid(color="#dfe3df", linewidth=0.8)
    axes[2].legend(frameon=False, loc="lower left")
    axes[2].text(
        7,
        5.55,
        "best possible range\n6.35 to 7.11",
        ha="center",
        color="#7f3528",
        fontweight="bold",
    )

    figure.suptitle(
        "v240 Case 7: the frozen causal reachable span lacks necessary capacity",
        fontsize=18,
        fontweight="bold",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


if __name__ == "__main__":
    main()
