#!/usr/bin/env python3
"""Render the public v241 maximal strict-savings span capacity figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case7_maximal_strict_savings_span_capacity_v241_public_summary.json"
OUTPUT = ROOT / "assets/figures/blastnet_case7_maximal_strict_savings_span_capacity_v241.png"


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    parent = data["parent_comparison"]
    results = data["results"]
    calls = data["logical_exact_call_ledger"]

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
    figure.patch.set_facecolor("#f5f7f4")
    for axis in axes:
        axis.set_facecolor("#ffffff")

    labels = ("Field", "Full grad", "Interior grad", "Observation")
    x = np.arange(4)
    width = 0.34
    k1_failures = np.array(list(parent["v240_metric_cell_failures"].values()))
    k14_failures = np.array(list(results["metric_cell_failures"].values()))
    axes[0].bar(x - width / 2, k1_failures, width, color="#bd5c45", label="K1 span (v240)")
    axes[0].bar(x + width / 2, k14_failures, width, color="#23766f", label="K14 span (v241)")
    axes[0].set_xticks(x, labels, rotation=16, ha="right")
    axes[0].set_ylim(0, 575)
    axes[0].set_ylabel("Failed later cells out of 533")
    axes[0].set_title("A  Necessary-capacity failures")
    axes[0].grid(axis="y", color="#dfe3df", linewidth=0.8)
    axes[0].legend(frameon=False, loc="upper left")
    for index, value in enumerate(k1_failures):
        axes[0].text(index - width / 2, value + 10, str(int(value)), ha="center", fontweight="bold")
        axes[0].text(index + width / 2, 12, "0", ha="center", color="#165f59", fontweight="bold")

    k1_p90 = np.array(list(parent["v240_minimum_metric_p90_higher"].values()))
    k14_p90 = np.array(list(results["minimum_metric_p90_higher"].values()))
    limits = np.array(list(results["absolute_limits"].values()))
    axes[1].bar(x - width / 2, k1_p90, width, color="#bd5c45", label="K1 span p90")
    axes[1].bar(x + width / 2, k14_p90, width, color="#315f93", label="K14 span p90")
    axes[1].scatter(x, limits, marker="_", s=500, linewidths=2.2, color="#1f1f1f", label="Absolute limit")
    axes[1].set_xticks(x, labels, rotation=16, ha="right")
    axes[1].set_ylim(0, 0.82)
    axes[1].set_ylabel("Metric-specific minimum relative error")
    axes[1].set_title("B  K14 clears every absolute bound")
    axes[1].grid(axis="y", color="#dfe3df", linewidth=0.8)
    axes[1].legend(frameon=False, loc="upper left")

    call_labels = ("Exact A", "Exact A^T", "Total")
    k14_calls = np.array(
        [calls["k14_sequence_A"], calls["k14_sequence_AT"], calls["k14_sequence_A"] + calls["k14_sequence_AT"]]
    )
    k16_calls = np.array(
        [calls["k16_sequence_A"], calls["k16_sequence_AT"], calls["k16_sequence_A"] + calls["k16_sequence_AT"]]
    )
    call_x = np.arange(3)
    axes[2].bar(call_x - width / 2, k14_calls, width, color="#8a651b", label="K14 logical ledger")
    axes[2].bar(call_x + width / 2, k16_calls, width, color="#66727d", label="K16 reference")
    axes[2].set_xticks(call_x, call_labels)
    axes[2].set_ylim(0, 1510)
    axes[2].set_ylabel("Calls per 42-frame rig")
    axes[2].set_title("C  Strict-savings capacity ledger")
    axes[2].grid(axis="y", color="#dfe3df", linewidth=0.8)
    axes[2].legend(frameon=False, loc="upper left")
    axes[2].text(
        1.0,
        1040,
        "9.15% nominal total reduction\ncapacity only, not effective savings",
        ha="center",
        va="center",
        color="#6f4f15",
        fontweight="bold",
    )

    figure.suptitle(
        "v241 Case 7: maximal strict-savings K14 restores necessary span capacity",
        fontsize=18,
        fontweight="bold",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


if __name__ == "__main__":
    main()
