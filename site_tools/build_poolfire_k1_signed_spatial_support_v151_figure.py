#!/usr/bin/env python3
"""Build the public v151 support-audit figure from redacted summary data."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_k1_signed_spatial_support_v151_public_summary.json"
OUTPUT = ROOT / "assets/figures/poolfire_k1_signed_spatial_support_v151.png"


def main() -> int:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    baseline = data["baseline"]
    signed = data["signed_spatial_peer_state"]
    labels = list(signed["trajectory_supported_fractions"])
    signed_values = np.asarray(
        list(signed["trajectory_supported_fractions"].values()), dtype=float
    )

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
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), dpi=170)
    figure.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.16, wspace=0.28)
    figure.suptitle(
        "v151 target-free support audit: signed spatial state exposes shift",
        fontsize=15,
        color="#17212b",
    )

    global_values = np.asarray(
        [baseline["global_supported_fraction"], signed["global_supported_fraction"]]
    )
    bars = axes[0].bar(
        ["Scalar summary", "Signed spatial + peer"],
        100.0 * global_values,
        color=["#4c78a8", "#d45d4c"],
        width=0.62,
    )
    axes[0].axhline(90.0, color="#2f855a", linestyle="--", linewidth=1.4)
    axes[0].set_ylim(0.0, 105.0)
    axes[0].set_ylabel("Supported active groups (%)")
    axes[0].set_title("Global cross-trajectory support")
    axes[0].grid(axis="y", color="#d9e2ea", linewidth=0.8, alpha=0.8)
    axes[0].set_axisbelow(True)
    for bar, value in zip(bars, global_values, strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            100.0 * value + 2.0,
            f"{100.0 * value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    axes[0].text(
        1.0,
        7.0,
        "No target model\nNo physical replay",
        ha="center",
        color="#5b6773",
        fontsize=9,
    )

    colors = ["#2f855a" if value >= 0.9 else "#d45d4c" for value in signed_values]
    trajectory_bars = axes[1].bar(labels, 100.0 * signed_values, color=colors, width=0.66)
    axes[1].axhline(90.0, color="#2f855a", linestyle="--", linewidth=1.4, label="Frozen 90% gate")
    axes[1].set_ylim(0.0, 105.0)
    axes[1].set_ylabel("Supported active groups (%)")
    axes[1].set_title("Signed-spatial support by held-out trajectory")
    axes[1].grid(axis="y", color="#d9e2ea", linewidth=0.8, alpha=0.8)
    axes[1].set_axisbelow(True)
    axes[1].legend(frameon=False, loc="lower left")
    for bar, value in zip(trajectory_bars, signed_values, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            100.0 * value + 1.5,
            f"{100.0 * value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    figure.text(
        0.5,
        0.055,
        "FAIL_SIGNED_SPATIAL_CROSS_TRAJECTORY_SUPPORT_V151  |  algorithm_breakthrough=false",
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
