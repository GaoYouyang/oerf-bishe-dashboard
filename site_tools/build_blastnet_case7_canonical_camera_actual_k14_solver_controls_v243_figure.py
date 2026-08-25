#!/usr/bin/env python3
"""Render the public v243 canonical-camera actual-K14 result figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case7_canonical_camera_actual_k14_solver_controls_v243_public_summary.json"
OUTPUT = ROOT / "assets/figures/blastnet_case7_canonical_camera_actual_k14_solver_controls_v243.png"


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    primary = data["results"]["primary"]
    controls = data["results"]["equal_or_cheaper_controls"]
    calls = data["logical_exact_call_ledger_per_complete_rig"]

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

    arm_labels = ("Warm K14", "Zero J-K14", "Zero K14", "BP J-K13", "BP K13")
    absolute = np.array([primary["absolute_cells_passed"], *[row["absolute_cells_passed"] for row in controls]])
    matched = np.array([primary["matched_cells_passed"], *[row["matched_cells_passed"] for row in controls]])
    x = np.arange(len(arm_labels))
    width = 0.34
    axes[0].bar(x - width / 2, absolute, width, color="#315f93", label="Absolute-safe")
    axes[0].bar(x + width / 2, matched, width, color="#23766f", label="K16-matched")
    axes[0].set_xticks(x, arm_labels, rotation=20, ha="right")
    axes[0].set_ylim(0, 590)
    axes[0].set_ylabel("Passing cells out of 546")
    axes[0].set_title("A  Actual solver and controls")
    axes[0].grid(axis="y", color="#dfe3df", linewidth=0.8)
    axes[0].legend(frameon=False, loc="upper right")
    for index, value in enumerate(matched):
        axes[0].text(index + width / 2, value + 11, str(int(value)), ha="center", fontweight="bold", color="#165f59")

    metric_labels = ("Field", "Full grad", "Interior grad", "Observation")
    p90 = np.array(list(primary["p90_higher"].values()))
    limits = np.array(list(primary["absolute_limits"].values()))
    ratios = p90 / limits
    colors = ["#23766f" if value <= 1.0 else "#bd5c45" for value in ratios]
    bars = axes[1].bar(np.arange(4), ratios, color=colors)
    axes[1].axhline(1.0, color="#202523", linestyle="--", linewidth=1.5, label="Frozen limit")
    axes[1].set_xticks(np.arange(4), metric_labels, rotation=18, ha="right")
    axes[1].set_ylim(0, 1.12)
    axes[1].set_ylabel("Primary p90 / absolute limit")
    axes[1].set_title("B  Warm K14 clears all p90 gates")
    axes[1].grid(axis="y", color="#dfe3df", linewidth=0.8)
    axes[1].legend(frameon=False, loc="upper left")
    for bar, value in zip(bars, ratios, strict=True):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center", fontweight="bold")

    call_labels = ("Warm K14", "K16 ref", "Each control")
    exact_a = np.array([calls["primary_A"], calls["reference_k16_A"], calls["each_equal_or_cheaper_control_A"]])
    exact_at = np.array([calls["primary_AT"], calls["reference_k16_AT"], calls["each_equal_or_cheaper_control_AT"]])
    call_x = np.arange(3)
    axes[2].bar(call_x - width / 2, exact_a, width, color="#8a651b", label="Exact A")
    axes[2].bar(call_x + width / 2, exact_at, width, color="#66727d", label="Exact A^T")
    axes[2].set_xticks(call_x, call_labels)
    axes[2].set_ylim(0, 760)
    axes[2].set_ylabel("Logical calls per 42-frame rig")
    axes[2].set_title("C  Call ledger, not wall time")
    axes[2].grid(axis="y", color="#dfe3df", linewidth=0.8)
    axes[2].legend(frameon=False, loc="upper right")
    figure.suptitle(
        "v243 opened Case 7: canonical-camera warm K14 passes; equal-or-cheaper controls fail",
        fontsize=17,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.012,
        "Post-open Case 7 mechanism evidence only; no external, wall/RSS, curved-ray, or real-BOST claim.",
        ha="center",
        color="#545b57",
        fontsize=10,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


if __name__ == "__main__":
    main()
