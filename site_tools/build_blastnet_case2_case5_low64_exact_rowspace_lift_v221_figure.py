#!/usr/bin/env python3
"""Build the public v221 exact-row-space-lift verdict figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case2_case5_low64_exact_rowspace_lift_v221_public_summary.json"
OUTPUT = ROOT / "assets/figures/blastnet_case2_case5_low64_exact_rowspace_lift_v221.png"


def build() -> Path:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    case5 = payload["outcomes"]["case5"]
    case2 = payload["outcomes"]["case2"]

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
        "v221: exact row-space lift removes useful Low-64 warm-start information",
        x=0.035,
        ha="left",
        fontsize=17,
        fontweight="bold",
        color="#162033",
    )

    conditions = ["Case 5", "Case 2"]
    x = np.arange(2)
    width = 0.24
    primary_matched = np.asarray([case5["matched_strict_safe"] / 546, case2["matched_strict_safe"] / 715])
    direct_matched = np.asarray([1.0, case2["controls"]["direct_low64_k11_matched_cells"] / 715])
    reference_matched = np.ones(2)
    axes[0].bar(x - width, 100 * primary_matched, width, label="Row-space lift K10", color="#c34832")
    axes[0].bar(x, 100 * direct_matched, width, label="Direct Low-64 K11", color="#16836d")
    axes[0].bar(x + width, 100 * reference_matched, width, label="Zero K16 reference", color="#34495e")
    axes[0].set_title("K16-matched cells")
    axes[0].set_ylabel("Passing cells (%)")
    axes[0].set_ylim(0, 108)
    axes[0].set_xticks(x, conditions)
    axes[0].legend(frameon=False, loc="lower left", fontsize=8)
    for offset, values in [(-width, primary_matched), (0.0, direct_matched), (width, reference_matched)]:
        for idx, value in enumerate(values):
            axes[0].text(idx + offset, max(1.2, 100 * value + 1.2), f"{100 * value:.1f}%", ha="center", fontsize=8)

    primary_absolute = np.asarray([case5["absolute_strict_safe"] / 546, case2["absolute_strict_safe"] / 715])
    zero_k11_absolute = np.asarray([case5["controls"]["zero_k11_absolute_cells"] / 546, case2["controls"]["zero_k11_absolute_cells"] / 715])
    axes[1].bar(x - width / 2, 100 * primary_absolute, width, label="Row-space lift K10", color="#d18a32")
    axes[1].bar(x + width / 2, 100 * zero_k11_absolute, width, label="Zero K11", color="#7a8793")
    axes[1].set_title("Absolute strict-safe cells")
    axes[1].set_ylabel("Passing cells (%)")
    axes[1].set_ylim(0, 108)
    axes[1].set_xticks(x, conditions)
    axes[1].legend(frameon=False, loc="lower right", fontsize=8)
    for offset, values in [(-width / 2, primary_absolute), (width / 2, zero_k11_absolute)]:
        for idx, value in enumerate(values):
            axes[1].text(idx + offset, 100 * value + 1.2, f"{100 * value:.1f}%", ha="center", fontsize=8)

    metric_names = ["Field", "Full grad.", "Interior grad.", "Observation"]
    limits = np.asarray([0.5, 0.75, 0.75, 0.2])
    metric_x = np.arange(4)
    case5_ratio = np.asarray(case5["primary_absolute_p90_higher"]) / limits
    case2_ratio = np.asarray(case2["primary_absolute_p90_higher"]) / limits
    axes[2].bar(metric_x - 0.18, case5_ratio, 0.36, label="Case 5", color="#a34f43")
    axes[2].bar(metric_x + 0.18, case2_ratio, 0.36, label="Case 2", color="#2d6f91")
    axes[2].axhline(1.0, color="#34495e", linewidth=1.4, linestyle="--", label="Absolute p90 gate")
    axes[2].set_title("Primary p90 / absolute gate")
    axes[2].set_ylabel("Ratio to frozen gate")
    axes[2].set_ylim(0, 1.18)
    axes[2].set_xticks(metric_x, metric_names, rotation=15, ha="right")
    axes[2].legend(frameon=False, loc="upper left", fontsize=8)

    for axis in axes:
        axis.grid(axis="y", color="#d9e1de", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)

    fig.text(
        0.035,
        0.095,
        "Independent replay: 32/32 checks pass; max field difference 3.03e-9; permutation difference 7.56e-14; call-ledger difference 0.",
        color="#42514c",
        fontsize=9.5,
    )
    fig.text(
        0.035,
        0.055,
        "Decision: 0/13 complete rigs in both conditions. Close this construction; no exact-call, wall/RSS, external, GPU, or real-BOST claim.",
        color="#8f2f24",
        fontsize=10,
        fontweight="bold",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    with Image.open(OUTPUT) as image:
        image.convert("RGB").save(OUTPUT, optimize=True)
    return OUTPUT


if __name__ == "__main__":
    print(build())
