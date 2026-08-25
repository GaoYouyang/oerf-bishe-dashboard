#!/usr/bin/env python3
"""Render the public v237.2 causal Krylov-recycling figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT
    / "docs/blastnet_case7_causal_krylov_recycling_v237_2_public_summary.json"
)
OUTPUT = (
    ROOT / "assets/figures/blastnet_case7_causal_krylov_recycling_v237_2.png"
)


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    results = data["results"]
    primary = results["causal_fifo16_primary"]
    controls = results["same_ledger_controls"]
    cost = data["cost_accounting"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(18, 6.5), constrained_layout=True)
    figure.patch.set_facecolor("#f7f8f6")
    for axis in axes:
        axis.set_facecolor("#ffffff")

    labels = ("Causal\nFIFO16", "Fixed\nanchor", "Previous\nfield", "Zero\nK2")
    absolute = np.array(
        [
            primary["absolute_strict_safe_cells"],
            controls["fixed_frame_zero_cache"]["absolute_strict_safe_cells"],
            controls["previous_candidate_field"]["absolute_strict_safe_cells"],
            controls["zero_geometry_jacobi_pcgls_k2"]["absolute_strict_safe_cells"],
        ]
    )
    matched = np.array(
        [
            primary["matched_cells"],
            controls["fixed_frame_zero_cache"]["matched_cells"],
            controls["previous_candidate_field"]["matched_cells"],
            controls["zero_geometry_jacobi_pcgls_k2"]["matched_cells"],
        ]
    )
    x = np.arange(len(labels))
    width = 0.34
    axes[0].bar(
        x - width / 2,
        absolute,
        width,
        color="#247a70",
        label="Absolute-safe cells",
    )
    axes[0].bar(
        x + width / 2,
        matched,
        width,
        color="#d05a42",
        label="K16-matched cells",
    )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 170)
    axes[0].set_ylabel("Cells out of 546")
    axes[0].set_title("A  Dynamic cache helps, but not enough")
    axes[0].grid(axis="y", color="#dfe3df", linewidth=0.8)
    axes[0].legend(frameon=False, loc="upper right")
    for offset, values in ((-width / 2, absolute), (width / 2, matched)):
        for index, value in enumerate(values):
            axes[0].text(index + offset, value + 3.5, str(int(value)), ha="center")
    axes[0].text(
        0,
        116,
        "matched = 13 anchors\n+ 0 / 533 later frames",
        ha="center",
        color="#7f3528",
        fontweight="bold",
    )

    metric_labels = ("Field", "Full grad", "Interior grad", "Observation")
    ratio_stats = primary["matched_ratio_p50_p90_worst"]
    p90 = np.array(
        [
            ratio_stats["field"][1],
            ratio_stats["full_gradient"][1],
            ratio_stats["interior_gradient"][1],
            ratio_stats["observation"][1],
        ]
    )
    colors = ("#315f93", "#8a651b", "#247a70", "#d05a42")
    axes[1].bar(np.arange(4), p90, color=colors)
    axes[1].axhline(
        1.02,
        color="#212121",
        linestyle="--",
        linewidth=1.6,
        label="Complete-rig p90 limit 1.02",
    )
    axes[1].set_xticks(np.arange(4), metric_labels, rotation=16, ha="right")
    axes[1].set_ylim(0, 8.55)
    axes[1].set_ylabel("Candidate / K16 p90 ratio")
    axes[1].set_title("B  Every matched-accuracy tail fails")
    axes[1].grid(axis="y", color="#dfe3df", linewidth=0.8)
    axes[1].legend(frameon=False, loc="upper left")
    for index, value in enumerate(p90):
        axes[1].text(index, value + 0.16, f"{value:.2f}", ha="center")

    method_labels = ("Causal FIFO16\n+ K1", "Per-frame\nK16 reference")
    calls_a = np.array(
        [
            cost["primary_sequence_exact_calls_A"],
            cost["reference_sequence_exact_calls_A"],
        ]
    )
    calls_at = np.array(
        [
            cost["primary_sequence_exact_calls_AT"],
            cost["reference_sequence_exact_calls_AT"],
        ]
    )
    call_x = np.arange(2)
    axes[2].bar(call_x, calls_a, color="#315f93", label="A")
    axes[2].bar(call_x, calls_at, bottom=calls_a, color="#d6a63f", label="A transpose")
    axes[2].set_xticks(call_x, method_labels)
    axes[2].set_ylim(0, 1460)
    axes[2].set_ylabel("Exact calls over 42 frames")
    axes[2].set_title("C  Nominal reduction is not claimable")
    axes[2].grid(axis="y", color="#dfe3df", linewidth=0.8)
    axes[2].legend(frameon=False, loc="upper left")
    totals = calls_a + calls_at
    for index, total in enumerate(totals):
        axes[2].text(index, total + 30, f"{int(total)}", ha="center", fontweight="bold")
    axes[2].text(
        0,
        520,
        "88.5% fewer calls\nwithout matched accuracy",
        ha="center",
        color="#7f3528",
        fontweight="bold",
    )

    figure.suptitle(
        "v237.2 Case 7: causal Krylov recycling fails after every anchor",
        fontsize=18,
        fontweight="bold",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


if __name__ == "__main__":
    main()
