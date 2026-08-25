#!/usr/bin/env python3
"""Render the public v236 Case 7 tail-subspace attribution figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case7_tail_subspace_attribution_v236_public_summary.json"
OUTPUT = ROOT / "assets/figures/blastnet_case7_tail_subspace_attribution_v236.png"


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    result = data["results"]
    keys = (
        "frozen_low64_control",
        "loro_rank16",
        "loro_rank32",
        "loro_rank64_primary",
    )
    labels = ("Fixed\nLow64", "LORO\nrank 16", "LORO\nrank 32", "LORO\nrank 64")
    p90 = np.array([result[key]["global_p90_higher"] for key in keys])
    worst = np.array([result[key]["global_worst"] for key in keys])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(16, 6.6), constrained_layout=True)
    figure.patch.set_facecolor("#f7f8f6")
    for axis in axes:
        axis.set_facecolor("#ffffff")

    x = np.arange(len(labels))
    width = 0.34
    axes[0].bar(x - width / 2, p90, width, color="#227c70", label="global p90")
    axes[0].bar(x + width / 2, worst, width, color="#d05a42", label="global worst")
    axes[0].axhline(
        data["question"]["p90_relative_residual_limit"],
        color="#145da0",
        linewidth=2,
        linestyle="--",
        label="p90 limit 0.316",
    )
    axes[0].axhline(
        data["question"]["worst_relative_residual_limit"],
        color="#7a5195",
        linewidth=2,
        linestyle=":",
        label="worst limit 0.500",
    )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("Relative correction residual")
    axes[0].set_title("A  Leave-one-rig tail capacity: 0 / 13 rigs pass")
    axes[0].grid(axis="y", color="#dfe3df", linewidth=0.8)
    axes[0].legend(frameon=False, ncols=2, loc="upper right")
    for index, value in enumerate(p90):
        axes[0].text(index - width / 2, value + 0.025, f"{value:.3f}", ha="center")
    for index, value in enumerate(worst):
        axes[0].text(index + width / 2, value + 0.025, f"{value:.3f}", ha="center")

    frames = np.arange(42)
    failed = np.zeros(42, dtype=int)
    failed[25] = 8
    failed[26:] = 13
    axes[1].step(frames, failed, where="mid", color="#d05a42", linewidth=3)
    axes[1].fill_between(
        frames,
        0,
        failed,
        step="mid",
        color="#d05a42",
        alpha=0.18,
    )
    axes[1].axvline(25.5, color="#7a5195", linewidth=2, linestyle="--")
    axes[1].text(26.3, 1.2, "all rigs fail from frame 26", color="#6d2f24")
    axes[1].text(12, 0.75, "0 failed rigs through frame 24", ha="center")
    axes[1].set_xlim(-0.5, 41.5)
    axes[1].set_ylim(0, 14.2)
    axes[1].set_yticks((0, 4, 8, 13))
    axes[1].set_xlabel("Case 7 frame")
    axes[1].set_ylabel("Rigs failing matched accuracy")
    axes[1].set_title("B  Failure is temporally aligned, but spatially rig-specific")
    axes[1].grid(axis="y", color="#dfe3df", linewidth=0.8)

    figure.suptitle(
        "v236 Case 7 tail attribution: compact jointly, non-transferable across rigs",
        fontsize=19,
        fontweight="bold",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


if __name__ == "__main__":
    main()
