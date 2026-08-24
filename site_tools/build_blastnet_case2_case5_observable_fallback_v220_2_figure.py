#!/usr/bin/env python3
"""Build the public v220.2 observable-fallback adjudication figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case2_case5_observable_fallback_v220_2_public_summary.json"
OUTPUT = ROOT / "assets/figures/blastnet_case2_case5_observable_fallback_v220_2.png"


def build() -> Path:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    nominal = payload["nominal_recomputation"]
    validation = payload["independent_validation"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "figure.facecolor": "#f4f7f6",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#c8d2cf",
            "axes.labelcolor": "#25352f",
            "xtick.color": "#42514c",
            "ytick.color": "#42514c",
            "axes.titleweight": "bold",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 6.5))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.78, bottom=0.25, wspace=0.31)
    fig.suptitle(
        "v220.2: the observable fallback does not establish a validated cross-condition pass",
        x=0.035,
        ha="left",
        fontsize=17,
        fontweight="bold",
        color="#162033",
    )

    labels = ["Case 5\nfallback", "Case 2\nfallback", "Case 2\nzero K16"]
    matched_fraction = np.asarray(
        [
            nominal["case5"]["matched_cells_passed"] / nominal["case5"]["total_cells"],
            nominal["case2"]["matched_cells_passed"] / nominal["case2"]["total_cells"],
            1.0,
        ]
    )
    complete_rigs = np.asarray(
        [nominal["case5"]["complete_rigs_passed"], nominal["case2"]["complete_rigs_passed"], 13]
    )
    colors = ["#16836d", "#c34832", "#34495e"]
    x = np.arange(len(labels))
    axes[0].bar(x, 100.0 * matched_fraction, color=colors, width=0.68)
    axes[0].axhline(100.0, color="#879790", linewidth=1.2, linestyle="--")
    axes[0].set_title("Nominal matched cells")
    axes[0].set_ylabel("Passing cells (%)")
    axes[0].set_ylim(0, 108)

    axes[1].bar(x, complete_rigs, color=colors, width=0.68)
    axes[1].axhline(13, color="#879790", linewidth=1.2, linestyle="--")
    axes[1].set_title("Complete matched rigs")
    axes[1].set_ylabel("Passing rigs out of 13")
    axes[1].set_ylim(0, 14.5)

    check_names = ["Formal vs.\nindependent", "Camera\npermutation"]
    observed = np.asarray(
        [
            validation["failed_checks"]["formal_field_relative_difference"]["observed"],
            validation["failed_checks"]["camera_permutation_field_relative_difference"]["observed"],
        ]
    )
    gate = validation["failed_checks"]["formal_field_relative_difference"]["frozen_tolerance"]
    check_x = np.arange(len(check_names))
    axes[2].bar(check_x, observed / gate, color=["#a34f43", "#d18a32"], width=0.58)
    axes[2].axhline(1.0, color="#34495e", linewidth=1.4, linestyle="--", label="Frozen gate")
    axes[2].set_title("Independent field closure")
    axes[2].set_ylabel("Observed difference / frozen tolerance")
    axes[2].set_ylim(0, 1.72)
    axes[2].legend(frameon=False, loc="upper right")

    for axis in axes[:2]:
        axis.set_xticks(x, labels)
    axes[2].set_xticks(check_x, check_names)
    for axis in axes:
        axis.grid(axis="y", color="#d9e1de", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
    for idx, value in enumerate(100.0 * matched_fraction):
        axes[0].text(idx, value + 2.0, f"{value:.1f}%", ha="center", fontweight="bold")
    for idx, value in enumerate(complete_rigs):
        axes[1].text(idx, value + 0.35, f"{value}/13", ha="center", fontweight="bold")
    for idx, value in enumerate(observed / gate):
        axes[2].text(idx, value + 0.045, f"{value:.3f}x", ha="center", fontweight="bold")

    fig.text(
        0.035,
        0.085,
        "Case 5 passes nominally; Case 2 reaches 0/13 complete rigs even with fallback. Both field-level validation checks exceed the frozen 1e-8 gate.",
        fontsize=10,
        color="#42514c",
    )
    fig.text(
        0.035,
        0.035,
        "Official status: INCONCLUSIVE. No post-result tolerance change, resource gate, model training, or breakthrough claim.",
        fontsize=10,
        color="#6a4e42",
        fontweight="bold",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    with Image.open(OUTPUT) as image:
        image.convert("RGB").save(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(build())
