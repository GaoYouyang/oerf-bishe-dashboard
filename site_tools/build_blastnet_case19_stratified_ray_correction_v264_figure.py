from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_stratified_ray_correction_v264_public_summary.json"
OUTPUT = ROOT / "assets/figures/blastnet_case19_stratified_ray_correction_v264.png"


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    envelope = data["two_implementation_envelope"]
    controls = data["controls"]

    labels = ["v258", "Zero-PCGLS K15", "v264", "K16 reference"]
    absolute = np.asarray(
        [
            controls["sealed_v258"]["absolute_pass_rigs"],
            controls["zero_pcgls_k15"]["absolute_pass_rigs"],
            envelope["candidate_absolute_pass_rigs"],
            controls["k16_reference"]["absolute_pass_rigs"],
        ]
    )
    matched = np.asarray(
        [
            controls["sealed_v258"]["matched_pass_rigs"],
            controls["zero_pcgls_k15"]["matched_pass_rigs"],
            envelope["candidate_matched_pass_rigs"],
            controls["k16_reference"]["matched_pass_rigs"],
        ]
    )

    fig = plt.figure(figsize=(18.5, 7), facecolor="#f6f8f7")
    grid = fig.add_gridspec(1, 2, width_ratios=(1.05, 1.35), wspace=0.2)
    left = fig.add_subplot(grid[0, 0])
    right = fig.add_subplot(grid[0, 1])

    x = np.arange(len(labels))
    width = 0.34
    left.bar(x - width / 2, absolute, width, color="#168f73", label="Absolute gate")
    left.bar(x + width / 2, matched, width, color="#2878b5", label="K16-matched gate")
    left.set_xticks(x, labels, rotation=12, ha="right")
    left.set_ylim(0, 14.3)
    left.set_ylabel("Passing rigs out of 13")
    left.set_title("v264 clears both gates on all rigs", loc="left", fontweight="bold")
    left.grid(axis="y", alpha=0.2)
    left.legend(frameon=False, loc="lower right")
    for offset, values in ((-width / 2, absolute), (width / 2, matched)):
        for index, value in enumerate(values):
            left.text(index + offset, value + 0.25, f"{value}/13", ha="center", fontsize=9)

    metric_labels = ["Field", "Full grad.", "Interior grad.", "Observation"]
    p90 = np.asarray(envelope["candidate_matched_ratio_p90_higher"])
    worst = np.asarray(envelope["candidate_matched_ratio_worst"])
    right.bar(x - width / 2, p90, width, color="#2878b5", label="p90-higher ratio")
    right.bar(x + width / 2, worst, width, color="#f2a134", label="worst ratio")
    right.axhline(1.0, color="#2878b5", linewidth=1.7, linestyle="--", label="p90 gate")
    right.axhline(1.05, color="#b63a3a", linewidth=1.7, linestyle=":", label="worst gate")
    right.set_xticks(x, metric_labels, rotation=12, ha="right")
    right.set_ylim(0, 1.13)
    right.set_ylabel("Ratio to conservative K16 reference")
    right.set_title("Matched tails remain within frozen limits", loc="left", fontweight="bold")
    right.grid(axis="y", alpha=0.2)
    right.legend(
        frameon=True,
        facecolor="#ffffff",
        edgecolor="none",
        framealpha=0.95,
        loc="upper left",
        ncols=2,
    )
    for offset, values in ((-width / 2, p90), (width / 2, worst)):
        for index, value in enumerate(values):
            right.text(index + offset, value + 0.025, f"{value:.3f}", ha="center", fontsize=9)

    fig.suptitle(
        "v264 | Fixed even-quincunx half-ray correction: 13/13 joint passes",
        x=0.055,
        y=0.98,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.015,
        "Independent validation 32/32 | 15.5A+14.5AT ray-equivalent vs K16 16A+16AT | opened Case 19 frame zero only; no wall/RSS claim",
        fontsize=10,
        color="#444444",
    )
    for axis in (left, right):
        axis.set_facecolor("#ffffff")
        axis.spines[["top", "right"]].set_visible(False)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
