#!/usr/bin/env python3
"""Build the public v152 support-audit figure from redacted summary data."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_k1_expanded_train_support_v152_public_summary.json"
OUTPUT = ROOT / "assets/figures/poolfire_k1_expanded_train_support_v152.png"


def _values(mapping: dict[str, float], cameras: list[str]) -> np.ndarray:
    return np.asarray([mapping[camera] for camera in cameras], dtype=float)


def main() -> int:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    primary = data["primary_raw_camera_state"]
    cameras = ["5", "7", "9", "12"]
    baseline = _values(primary["p33_size01_baseline_by_camera_count"], cameras)
    expanded = _values(primary["p33_size01_expanded_by_camera_count"], cameras)
    added_holdout = _values(primary["p33_size03_heldout_by_camera_count"], cameras)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.edgecolor": "#9aa5b1",
            "axes.labelcolor": "#26323d",
            "xtick.color": "#425466",
            "ytick.color": "#425466",
            "figure.facecolor": "#f7fafc",
            "axes.facecolor": "#ffffff",
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), dpi=170)
    figure.subplots_adjust(left=0.07, right=0.985, top=0.83, bottom=0.17, wspace=0.25)
    figure.suptitle(
        "v152 target-free train-coverage audit: sparse-view size gap remains",
        fontsize=15,
        color="#17212b",
    )

    x = np.arange(len(cameras))
    width = 0.34
    first = axes[0].bar(
        x - width / 2,
        100.0 * baseline,
        width,
        label="p33-s01 before",
        color="#8aa7c4",
    )
    second = axes[0].bar(
        x + width / 2,
        100.0 * expanded,
        width,
        label="p33-s01 after adding p33-s03",
        color="#d45d4c",
    )
    axes[0].axhline(90.0, color="#2f855a", linestyle="--", linewidth=1.4)
    axes[0].set_xticks(x, cameras)
    axes[0].set_xlabel("Active cameras")
    axes[0].set_ylabel("Supported camera rows (%)")
    axes[0].set_ylim(65.0, 102.0)
    axes[0].set_title("Original p33 size condition")
    axes[0].grid(axis="y", color="#d9e2ea", linewidth=0.8, alpha=0.8)
    axes[0].set_axisbelow(True)
    axes[0].legend(frameon=False, loc="lower right", fontsize=8.5)
    for bars, values in [(first, baseline), (second, expanded)]:
        for bar, value in zip(bars, values, strict=True):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2.0,
                100.0 * value + 0.8,
                f"{100.0 * value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )

    colors = ["#2f855a" if value >= 0.9 else "#d45d4c" for value in added_holdout]
    holdout_bars = axes[1].bar(cameras, 100.0 * added_holdout, color=colors, width=0.62)
    axes[1].axhline(90.0, color="#2f855a", linestyle="--", linewidth=1.4, label="Frozen 90% gate")
    axes[1].set_xlabel("Active cameras")
    axes[1].set_ylabel("Supported camera rows (%)")
    axes[1].set_ylim(65.0, 102.0)
    axes[1].set_title("Added p33-s03 held out")
    axes[1].grid(axis="y", color="#d9e2ea", linewidth=0.8, alpha=0.8)
    axes[1].set_axisbelow(True)
    axes[1].legend(frameon=False, loc="lower right", fontsize=8.5)
    for bar, value in zip(holdout_bars, added_holdout, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            100.0 * value + 0.8,
            f"{100.0 * value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    axes[1].text(
        0.53,
        0.57,
        "265 p33-s01 rows rescued\n5-camera gate still fails",
        transform=axes[1].transAxes,
        color="#5b6773",
        fontsize=9,
        ha="center",
        va="center",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f7fafc", "edgecolor": "#cbd5df"},
    )

    figure.text(
        0.5,
        0.055,
        "FAIL_P33_SAME_POWER_MUTUAL_SUPPORT_V152  |  algorithm_breakthrough=false",
        ha="center",
        color="#5b6773",
        fontsize=9.5,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, bbox_inches="tight")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
