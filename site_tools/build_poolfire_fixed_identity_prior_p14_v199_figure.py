#!/usr/bin/env python3
"""Build the public v199 p14 reference-adequacy figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_fixed_identity_prior_p14_v199_public_summary.json"
OUTPUT = ROOT / "assets/figures/poolfire_fixed_identity_prior_p14_v199.png"


def main() -> None:
    payload = json.loads(SUMMARY.read_text())
    methods = payload["methods"]
    keys = ("fixed_identity_k1", "full_dct_k1_parent", "full_dct_k2_reference")
    labels = ("Fixed identity\nK1", "Unregularized\nK1", "Full-DCT K2\nreference")
    colors = ("#1f8a70", "#7b8491", "#d89522")
    all_nine_color = "#2878b5"
    five_camera_color = "#d1495b"

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.8))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.87, bottom=0.16, hspace=0.46, wspace=0.24)
    fig.suptitle(
        "v199  Fixed regularization helps, but the p14 K2 reference is inadequate",
        fontsize=16,
        fontweight="bold",
    )

    x = np.arange(3)
    width = 0.34
    nine_cells = [methods[key]["all_nine"]["strict_safe_cells"] / 1313 for key in keys]
    five_cells = [methods[key]["five_camera"]["strict_safe_cells"] / 1313 for key in keys]
    axes[0, 0].bar(x - width / 2, nine_cells, width, color=all_nine_color, label="All nine")
    axes[0, 0].bar(x + width / 2, five_cells, width, color=five_camera_color, label="Five camera")
    axes[0, 0].axhline(1.0, color="#222222", linewidth=1, linestyle="--")
    axes[0, 0].set_ylim(0.86, 1.01)
    axes[0, 0].set_xticks(x, labels, fontsize=9)
    axes[0, 0].set_ylabel("Strict-safe cell fraction")
    axes[0, 0].set_title("Cellwise safety", loc="left", fontweight="bold")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(axis="y", alpha=0.2)
    for index, key in enumerate(keys):
        axes[0, 0].text(index - width / 2, nine_cells[index] + 0.002, "1313/1313", ha="center", fontsize=8)
        axes[0, 0].text(
            index + width / 2,
            five_cells[index] + 0.002,
            f'{methods[key]["five_camera"]["strict_safe_cells"]}/1313',
            ha="center",
            fontsize=8,
        )

    nine_groups = [methods[key]["all_nine"]["complete_groups_passed"] for key in keys]
    five_groups = [methods[key]["five_camera"]["complete_groups_passed"] for key in keys]
    axes[0, 1].bar(x - width / 2, nine_groups, width, color=all_nine_color, label="All nine")
    axes[0, 1].bar(x + width / 2, five_groups, width, color=five_camera_color, label="Five camera")
    axes[0, 1].axhline(13, color="#222222", linewidth=1, linestyle="--")
    axes[0, 1].set_ylim(0, 14)
    axes[0, 1].set_xticks(x, labels, fontsize=9)
    axes[0, 1].set_ylabel("Complete calibration groups")
    axes[0, 1].set_title("Reference fails the five-camera group gate", loc="left", fontweight="bold")
    axes[0, 1].grid(axis="y", alpha=0.2)
    for index in range(3):
        axes[0, 1].text(index - width / 2, nine_groups[index] + 0.25, "13/13", ha="center", fontsize=8)
        axes[0, 1].text(index + width / 2, five_groups[index] + 0.25, f"{five_groups[index]}/13", ha="center", fontsize=8)

    metric_names = ("field", "gradient", "observation")
    metric_labels = ("Field", "Gradient", "Observation")
    metric_x = np.arange(3)
    bar_width = 0.24
    for index, key in enumerate(keys):
        values = [methods[key]["five_camera"]["p90"][name] for name in metric_names]
        axes[1, 0].bar(metric_x + (index - 1) * bar_width, values, bar_width, color=colors[index], label=labels[index].replace("\n", " "))
    for index, limit in enumerate((0.5, 0.75, 0.2)):
        axes[1, 0].hlines(limit, index - 0.43, index + 0.43, color="#222222", linestyles="--", linewidth=1)
    axes[1, 0].set_xticks(metric_x, metric_labels)
    axes[1, 0].set_ylim(0, 0.82)
    axes[1, 0].set_ylabel("Five-camera p90 error")
    axes[1, 0].set_title("Five-camera p90 values", loc="left", fontweight="bold")
    axes[1, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].grid(axis="y", alpha=0.2)

    for index, key in enumerate(keys):
        values = [methods[key]["five_camera"]["worst"][name] for name in metric_names]
        axes[1, 1].bar(metric_x + (index - 1) * bar_width, values, bar_width, color=colors[index], label=labels[index].replace("\n", " "))
    for index, limit in enumerate((0.55, 0.95, 0.22)):
        axes[1, 1].hlines(limit, index - 0.43, index + 0.43, color="#222222", linestyles="--", linewidth=1)
    axes[1, 1].set_xticks(metric_x, metric_labels)
    axes[1, 1].set_ylim(0, 1.08)
    axes[1, 1].set_ylabel("Five-camera worst error")
    axes[1, 1].set_title("Worst-case gates expose incomplete groups", loc="left", fontweight="bold")
    axes[1, 1].grid(axis="y", alpha=0.2)

    fig.text(
        0.5,
        0.035,
        "Historically exposed p14 development only. Reference inadequate: no exact-call, wall/RSS, external, or real-BOST claim.",
        ha="center",
        fontsize=10,
        color="#4d5560",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
