#!/usr/bin/env python3
"""Build the redacted v205 compact-cache evidence figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets/figures/poolfire_potential_normal_compact_cache_p14_v205.png"


def build() -> Path:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.facecolor": "#f7f8f5",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#c9d0cc",
            "axes.titleweight": "bold",
            "axes.labelcolor": "#24302d",
            "xtick.color": "#3f4c48",
            "ytick.color": "#3f4c48",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 5.6), constrained_layout=True)
    fig.suptitle(
        "v205: compact potential-normal cache reproduces dense full-DCT K1",
        fontsize=15,
        fontweight="bold",
        color="#1d2a27",
    )

    labels = ("Five cameras", "All nine")
    dense = np.asarray((2_900_875, 5_221_575), dtype=float)
    packed = np.asarray((509_545, 509_545), dtype=float)
    x = np.arange(len(labels))
    width = 0.34
    axes[0].bar(x - width / 2, dense, width, label="Dense response", color="#c95752")
    axes[0].bar(x + width / 2, packed, width, label="Packed normal cache", color="#258a7a")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Retained scalar values")
    axes[0].set_title("Retained state is 5.69x / 10.25x smaller")
    axes[0].grid(axis="y", color="#e5e9e6", linewidth=0.8)
    axes[0].set_axisbelow(True)
    axes[0].legend(frameon=False, loc="upper left")
    for xpos, value in zip(x + width / 2, packed, strict=True):
        axes[0].text(xpos, value + 95_000, "509,545", ha="center", fontweight="bold")
    axes[0].text(0, dense[0] + 95_000, "2.90M", ha="center", fontweight="bold")
    axes[0].text(1, dense[1] + 95_000, "5.22M", ha="center", fontweight="bold")

    error_labels = ("Coordinates\nvs formal", "Field\nvs parent", "Permutation\ncoordinates")
    errors = np.asarray((1.4294906475e-12, 9.8393566159e-13, 4.2415319475e-13))
    bars = axes[1].bar(error_labels, errors, color=("#287bb5", "#584f9e", "#258a7a"), width=0.62)
    axes[1].set_yscale("log")
    axes[1].set_ylim(1e-14, 3e-9)
    axes[1].axhline(1e-9, color="#c95752", linestyle="--", linewidth=1, label="Frozen 1e-9 gate")
    axes[1].set_ylabel("Maximum relative difference")
    axes[1].set_title("Independent equivalence stays near 1e-12")
    axes[1].grid(axis="y", which="both", color="#e5e9e6", linewidth=0.8)
    axes[1].set_axisbelow(True)
    axes[1].legend(frameon=False, loc="upper right")
    for bar, value in zip(bars, errors, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.45,
            f"{value:.2e}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    call_labels = ("Dense K1", "Compact K1", "K2 reference")
    calls_a = np.asarray((2, 2, 3))
    calls_at = np.asarray((1, 2, 2))
    x = np.arange(len(call_labels))
    axes[2].bar(x, calls_a, color="#287bb5", width=0.58, label="A")
    axes[2].bar(x, calls_at, bottom=calls_a, color="#d9a441", width=0.58, label="AT")
    axes[2].set_xticks(x, call_labels)
    axes[2].set_ylim(0, 5.7)
    axes[2].set_ylabel("Logical exact calls")
    axes[2].set_title("Compact K1 adds one adjoint vs dense K1")
    axes[2].grid(axis="y", color="#e5e9e6", linewidth=0.8)
    axes[2].set_axisbelow(True)
    axes[2].legend(frameon=False, loc="upper left")
    for xpos, a_count, at_count in zip(x, calls_a, calls_at, strict=True):
        axes[2].text(
            xpos,
            a_count + at_count + 0.15,
            f"{a_count}A+{at_count}AT",
            ha="center",
            fontweight="bold",
        )
    axes[2].text(
        0.5,
        -0.22,
        "Cache compression is established; wall time and peak RSS are not.",
        transform=axes[2].transAxes,
        ha="center",
        color="#55615d",
        fontsize=9,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    with Image.open(OUTPUT) as image:
        image.convert("RGB").save(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(build())
