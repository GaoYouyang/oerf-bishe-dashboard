#!/usr/bin/env python3
"""Render the redacted public v262 detector-integrability verdict figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_detector_integrability_no_go_v262_public_summary.json"
OUTPUT = ROOT / "assets/figures/blastnet_case19_detector_integrability_no_go_v262.png"


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    sources = data["sources"]
    names = ("Observation y", "K14 prediction", "K14 residual")
    keys = ("observation", "prediction_k14", "residual_k14")
    p50 = np.asarray([sources[key]["observation_normalized_defect_p50_higher"] for key in keys])
    p90 = np.asarray([sources[key]["observation_normalized_defect_p90_higher"] for key in keys])
    worst = np.asarray([sources[key]["observation_normalized_defect_worst"] for key in keys])
    removed = np.asarray([sources[key]["removed_energy_fraction_p90_higher"] for key in keys])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(17, 7))
    figure.patch.set_facecolor("#f3f5f2")
    figure.subplots_adjust(left=0.07, right=0.97, bottom=0.20, top=0.78, wspace=0.26)
    for axis in axes:
        axis.set_facecolor("#ffffff")

    positions = np.arange(len(names))
    width = 0.24
    axes[0].bar(positions - width, p50, width, color="#3c668f", label="p50-higher")
    axes[0].bar(positions, p90, width, color="#2f8073", label="p90-higher")
    axes[0].bar(positions + width, worst, width, color="#c04f3d", label="worst")
    axes[0].axhline(1e-8, color="#252a28", linestyle="--", linewidth=1.5, label="Frozen invariance gate")
    axes[0].set_yscale("log")
    axes[0].set_ylim(5e-9, 0.3)
    axes[0].set_xticks(positions, names)
    axes[0].set_ylabel("Projection defect / camera observation norm")
    axes[0].set_title("A  Neither y nor Ax14 is forward-invariant")
    axes[0].grid(axis="y", color="#dfe3df", linewidth=0.8, which="both")
    axes[0].legend(frameon=False, loc="lower right")
    for index, value in enumerate(p90):
        axes[0].text(index, value * 1.35, f"p90 {value:.3f}", ha="center", fontsize=9)

    bars = axes[1].bar(positions, 100.0 * removed, color=("#3c668f", "#2f8073", "#c04f3d"), width=0.58)
    axes[1].set_xticks(positions, names)
    axes[1].set_ylim(0, 65)
    axes[1].set_ylabel("p90-higher energy removed (%)")
    axes[1].set_title("B  Projection removes substantial residual content")
    axes[1].grid(axis="y", color="#dfe3df", linewidth=0.8)
    axes[1].bar_label(bars, labels=[f"{100.0 * value:.2f}%" for value in removed], padding=4)
    axes[1].text(
        0.5,
        0.91,
        "0 / 117 invariant camera blocks for every source",
        transform=axes[1].transAxes,
        ha="center",
        fontsize=11,
        color="#252a28",
    )

    figure.suptitle(
        "v262 Case 19: detector integrability is not an exact discrete-forward invariant",
        fontsize=17,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.055,
        "Independent 24/24; rank 255; zero new A/AT calls; post-open no-go evidence, not a reconstruction or algorithm claim.",
        ha="center",
        color="#545b57",
        fontsize=10.5,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


if __name__ == "__main__":
    main()
