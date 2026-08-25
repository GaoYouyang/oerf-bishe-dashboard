#!/usr/bin/env python3
"""Render the public v238 fixed-octant tail-capacity figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case7_octant_tail_subspace_capacity_v238_public_summary.json"
OUTPUT = ROOT / "assets/figures/blastnet_case7_octant_tail_subspace_capacity_v238.png"


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    results = data["results"]
    global_rank = results["global_rank64_control"]
    octant = results["octant_rank8_primary"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    figure.patch.set_facecolor("#f7f8f6")
    for axis in axes:
        axis.set_facecolor("#ffffff")

    labels = ("p50", "p90", "worst")
    global_values = np.array(
        [global_rank["global_p50"], global_rank["global_p90_higher"], global_rank["global_worst"]]
    )
    octant_values = np.array(
        [octant["global_p50"], octant["global_p90_higher"], octant["global_worst"]]
    )
    x = np.arange(len(labels))
    width = 0.34
    axes[0].bar(x - width / 2, global_values, width, color="#227c70", label="Global rank 64")
    axes[0].bar(x + width / 2, octant_values, width, color="#d05a42", label="Octant rank 8 x 8")
    axes[0].axhline(
        data["question"]["p90_relative_residual_limit"],
        color="#315f93",
        linestyle="--",
        linewidth=2,
        label="p90 limit 0.316",
    )
    axes[0].axhline(
        data["question"]["worst_relative_residual_limit"],
        color="#8a651b",
        linestyle=":",
        linewidth=2,
        label="worst limit 0.500",
    )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 0.92)
    axes[0].set_ylabel("Relative correction residual")
    axes[0].set_title("A  Equal dimension: localization worsens the tail fit")
    axes[0].grid(axis="y", color="#dfe3df", linewidth=0.8)
    axes[0].legend(frameon=False, loc="upper left")
    for index, value in enumerate(global_values):
        axes[0].text(index - width / 2, value + 0.018, f"{value:.3f}", ha="center")
    for index, value in enumerate(octant_values):
        axes[0].text(index + width / 2, value + 0.018, f"{value:.3f}", ha="center")

    all_delta = np.asarray(octant["all_frame_p90_by_rig"]) - np.asarray(
        global_rank["all_frame_p90_by_rig"]
    )
    late_delta = np.asarray(octant["late_frame_p90_by_rig"]) - np.asarray(
        global_rank["late_frame_p90_by_rig"]
    )
    rigs = np.arange(1, 14)
    axes[1].bar(rigs - width / 2, all_delta, width, color="#315f93", label="All-frame p90 delta")
    axes[1].bar(rigs + width / 2, late_delta, width, color="#8a651b", label="Late-frame p90 delta")
    axes[1].axhline(0, color="#17252b", linewidth=1.4)
    axes[1].set_xticks(rigs)
    axes[1].set_xlabel("Held-out rig index")
    axes[1].set_ylabel("Octant minus global residual")
    axes[1].set_title("B  Octant p90 is worse on every held-out rig")
    axes[1].grid(axis="y", color="#dfe3df", linewidth=0.8)
    axes[1].legend(frameon=False, loc="upper left")
    axes[1].text(
        13.25,
        max(float(np.max(all_delta)), float(np.max(late_delta))) * 0.78,
        "13 / 13\npositive",
        color="#6d2f24",
        fontweight="bold",
        ha="right",
    )

    figure.suptitle(
        "v238 Case 7: fixed octant locality does not recover cross-rig transfer",
        fontsize=19,
        fontweight="bold",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


if __name__ == "__main__":
    main()
