#!/usr/bin/env python3
"""Build the public v153 coordinate-support audit figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_k1_coordinate_canonicalization_v153_public_summary.json"
OUTPUT = ROOT / "assets/figures/poolfire_k1_coordinate_canonicalization_v153.png"


def _values(mapping: dict[str, float], labels: list[str]) -> np.ndarray:
    return np.asarray([mapping[label] for label in labels], dtype=float)


def main() -> int:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    cameras = ["5", "7", "9", "12"]
    p33 = payload["p33_size01_camera_support"]
    raw = _values(p33["v152_raw"], cameras)
    affine = _values(p33["v153_affine"], cameras)
    monotone = _values(p33["v153_monotone"], cameras)

    trajectory_labels = ["p14-s05", "p22-s03", "p33-s01", "p45-s05", "p58-s03", "p33-s03"]
    trajectory_keys = ["p14_size05", "p22_size03", "p33_size01", "p45_size05", "p58_size03", "p33_size03"]
    trajectory = _values(payload["monotone_trajectory_support"], trajectory_keys)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.edgecolor": "#8996a3",
            "axes.labelcolor": "#26323d",
            "xtick.color": "#425466",
            "ytick.color": "#425466",
            "figure.facecolor": "#f7fafc",
            "axes.facecolor": "#ffffff",
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), dpi=170)
    figure.subplots_adjust(left=0.065, right=0.985, top=0.78, bottom=0.18, wspace=0.24)
    figure.suptitle(
        "v153 target-free coordinate canonicalization: support gets worse",
        fontsize=15,
        color="#17212b",
    )

    x = np.arange(len(cameras))
    width = 0.24
    bars = [
        axes[0].bar(x - width, 100.0 * raw, width, label="v152 raw", color="#4c78a8"),
        axes[0].bar(x, 100.0 * affine, width, label="v153 affine", color="#e0a12f"),
        axes[0].bar(x + width, 100.0 * monotone, width, label="v153 monotone", color="#d45d4c"),
    ]
    axes[0].axhline(90.0, color="#2f855a", linestyle="--", linewidth=1.4, label="Frozen 90% gate")
    axes[0].set_xticks(x, cameras)
    axes[0].set_xlabel("Active cameras")
    axes[0].set_ylabel("Supported camera rows (%)")
    axes[0].set_ylim(58.0, 103.0)
    axes[0].set_title("p33-s01: normalization harms passed strata")
    axes[0].grid(axis="y", color="#d9e2ea", linewidth=0.8, alpha=0.8)
    axes[0].set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.065, 0.895),
        ncol=4,
        fontsize=8,
    )
    for group, values in zip(bars, [raw, affine, monotone], strict=True):
        for bar, value in zip(group, values, strict=True):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2.0,
                100.0 * value + 0.55,
                f"{100.0 * value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.4,
                fontweight="bold",
            )

    colors = ["#2f855a" if value >= 0.9 else "#d45d4c" for value in trajectory]
    trajectory_bars = axes[1].bar(trajectory_labels, 100.0 * trajectory, color=colors, width=0.66)
    axes[1].axhline(90.0, color="#2f855a", linestyle="--", linewidth=1.4)
    axes[1].set_ylabel("Supported camera rows (%)")
    axes[1].set_ylim(0.0, 105.0)
    axes[1].set_title("Monotone support by held-out trajectory")
    axes[1].grid(axis="y", color="#d9e2ea", linewidth=0.8, alpha=0.8)
    axes[1].set_axisbelow(True)
    axes[1].tick_params(axis="x", labelrotation=22, labelsize=8)
    for bar, value in zip(trajectory_bars, trajectory, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            100.0 * value + 1.3,
            f"{100.0 * value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
    axes[1].annotate(
        "p45: 7.6%",
        xy=(3, 100.0 * trajectory[3]),
        xytext=(3.85, 37),
        arrowprops={"arrowstyle": "->", "color": "#8f3b31"},
        color="#8f3b31",
        fontsize=9,
        fontweight="bold",
        ha="center",
    )

    figure.text(
        0.5,
        0.055,
        "FAIL_TARGET_FREE_MONOTONE_COORDINATE_SUPPORT_V153  |  current predictor route closed  |  algorithm_breakthrough=false",
        ha="center",
        color="#5b6773",
        fontsize=9.2,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, bbox_inches="tight")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
