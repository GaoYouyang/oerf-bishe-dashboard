#!/usr/bin/env python3
"""Build the public v154 broader-train-coverage figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_k1_broader_train_coverage_v154_public_summary.json"
OUTPUT = ROOT / "assets/figures/poolfire_k1_broader_train_coverage_v154.png"


def _values(mapping: dict[str, float], labels: list[str]) -> np.ndarray:
    return np.asarray([mapping[label] for label in labels], dtype=float)


def main() -> int:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    trajectory_keys = [
        "p14_size05",
        "p22_size03",
        "p33_size01",
        "p45_size05",
        "p58_size03",
        "p33_size03",
        "p33_size05",
        "p45_size01",
        "p45_size03",
        "p58_size05",
    ]
    trajectory_labels = [
        "p14-s05",
        "p22-s03",
        "p33-s01",
        "p45-s05",
        "p58-s03",
        "p33-s03",
        "p33-s05",
        "p45-s01",
        "p45-s03",
        "p58-s05",
    ]
    trajectory = _values(payload["trajectory_support"], trajectory_keys)
    added = np.asarray([False, False, False, False, False, False, True, True, True, True])

    cameras = ["5", "7", "9", "12"]
    failed = payload["failed_trajectory_camera_support"]
    p45 = _values(failed["p45_size05"], cameras)
    p58s03 = _values(failed["p58_size03"], cameras)
    p58s05 = _values(failed["p58_size05"], cameras)

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
    figure, axes = plt.subplots(1, 2, figsize=(14.5, 5.7), dpi=170)
    figure.subplots_adjust(left=0.06, right=0.985, top=0.79, bottom=0.22, wspace=0.22)
    figure.suptitle(
        "v154 broader public-train coverage: 7 of 10 trajectories pass",
        fontsize=15,
        color="#17212b",
    )

    colors = []
    for value, is_added in zip(trajectory, added, strict=True):
        if value < 0.9:
            colors.append("#d45d4c")
        elif is_added:
            colors.append("#2f9e8f")
        else:
            colors.append("#4c78a8")
    bars = axes[0].bar(trajectory_labels, 100.0 * trajectory, color=colors, width=0.68)
    axes[0].axhline(90.0, color="#2f855a", linestyle="--", linewidth=1.4)
    axes[0].set_ylabel("Supported camera rows (%)")
    axes[0].set_ylim(0.0, 105.0)
    axes[0].set_title("Complete-trajectory leave-one-out support")
    axes[0].grid(axis="y", color="#d9e2ea", linewidth=0.8, alpha=0.8)
    axes[0].set_axisbelow(True)
    axes[0].tick_params(axis="x", labelrotation=28, labelsize=7.8)
    for bar, value in zip(bars, trajectory, strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            100.0 * value + 1.2,
            f"{100.0 * value:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.2,
            fontweight="bold",
        )
    axes[0].text(
        0.01,
        0.04,
        "blue: existing pass   teal: added pass   red: fail",
        transform=axes[0].transAxes,
        fontsize=8,
        color="#5b6773",
    )

    x = np.arange(len(cameras))
    width = 0.24
    groups = [
        axes[1].bar(x - width, 100.0 * p45, width, label="p45-s05", color="#d45d4c"),
        axes[1].bar(x, 100.0 * p58s03, width, label="p58-s03", color="#e0a12f"),
        axes[1].bar(x + width, 100.0 * p58s05, width, label="p58-s05 (added)", color="#756bb1"),
    ]
    axes[1].axhline(90.0, color="#2f855a", linestyle="--", linewidth=1.4, label="Frozen 90% gate")
    axes[1].set_xticks(x, cameras)
    axes[1].set_xlabel("Active cameras")
    axes[1].set_ylabel("Supported camera rows (%)")
    axes[1].set_ylim(0.0, 105.0)
    axes[1].set_title("All camera counts must pass; three trajectories do not")
    axes[1].grid(axis="y", color="#d9e2ea", linewidth=0.8, alpha=0.8)
    axes[1].set_axisbelow(True)
    axes[1].legend(
        frameon=False,
        fontsize=7.6,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.17),
        ncol=4,
    )
    for group, values in zip(groups, [p45, p58s03, p58s05], strict=True):
        for bar, value in zip(group, values, strict=True):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2.0,
                100.0 * value + 1.0,
                f"{100.0 * value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
            )

    figure.text(
        0.5,
        0.055,
        "FAIL_BROADER_TRAIN_COVERAGE_V154  |  predictor and GPU remain closed  |  algorithm_breakthrough=false",
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
