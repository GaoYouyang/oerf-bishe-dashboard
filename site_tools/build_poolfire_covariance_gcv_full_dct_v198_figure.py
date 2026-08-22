#!/usr/bin/env python3
"""Build the public v198 control-attribution figure from redacted evidence."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_covariance_gcv_full_dct_v198_public_summary.json"
OUTPUT = ROOT / "assets/figures/poolfire_covariance_gcv_full_dct_v198.png"


def main() -> None:
    payload = json.loads(SUMMARY.read_text())
    methods = payload["methods"]
    keys = (
        "empirical_covariance_gcv_full_dct_k1",
        "identity_gcv_full_dct_k1",
        "full_dct_k1_parent",
        "full_dct_k2_reference",
    )
    labels = ("Empirical\ncovariance GCV", "Identity GCV", "Full-DCT K1\nparent", "Full-DCT K2\nreference")
    colors = ("#1f8a70", "#2878b5", "#7b8491", "#d89522")

    cell_fraction = [methods[key]["strict_safe_cells"] / 2626 for key in keys]
    group_fraction = [methods[key]["complete_groups_passed"] / 26 for key in keys]
    forward = [methods[key]["logical_online_exact_calls"]["A"] for key in keys]
    adjoint = [methods[key]["logical_online_exact_calls"]["AT"] for key in keys]

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.8))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.15, hspace=0.45, wspace=0.22)
    fig.suptitle(
        "v198  Equal-cost identity-GCV explains the covariance candidate",
        fontsize=17,
        fontweight="bold",
    )

    for axis, values, title, denominator in (
        (axes[0, 0], cell_fraction, "Strict-safe cells", 2626),
        (axes[0, 1], group_fraction, "Complete calibration groups", 26),
    ):
        bars = axis.bar(np.arange(4), values, color=colors, width=0.7)
        axis.set_ylim(0.94, 1.006)
        axis.axhline(1.0, color="#222222", linewidth=1, linestyle="--")
        axis.set_xticks(np.arange(4), labels, fontsize=9)
        axis.set_ylabel("Pass fraction")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="y", alpha=0.2)
        for bar, key in zip(bars, keys, strict=True):
            count_key = "strict_safe_cells" if denominator == 2626 else "complete_groups_passed"
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.0008,
                f"{methods[key][count_key]}/{denominator}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    x = np.arange(4)
    axes[1, 0].bar(x, forward, color=colors, width=0.68, label="A")
    axes[1, 0].bar(x, adjoint, bottom=forward, color="#d7dde3", edgecolor=colors, width=0.68, label="A^T")
    axes[1, 0].set_xticks(x, labels, fontsize=9)
    axes[1, 0].set_ylabel("Logical exact calls per cell")
    axes[1, 0].set_title("Online exact-call ledger", loc="left", fontweight="bold")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(axis="y", alpha=0.2)

    metric_names = ("field", "gradient", "observation")
    metric_labels = ("Field", "Gradient", "Observation")
    compare_keys = (keys[0], keys[1], keys[3])
    compare_labels = ("Empirical covariance GCV", "Identity GCV", "K2 reference")
    width = 0.23
    metric_x = np.arange(3)
    for index, (key, label) in enumerate(zip(compare_keys, compare_labels, strict=True)):
        values = [methods[key]["five_camera_p90"][name] for name in metric_names]
        axes[1, 1].bar(metric_x + (index - 1) * width, values, width, label=label, color=colors[index])
    for index, limit in enumerate((0.5, 0.75, 0.2)):
        axes[1, 1].hlines(limit, index - 0.42, index + 0.42, color="#222222", linestyles="--", linewidth=1)
    axes[1, 1].set_xticks(metric_x, metric_labels)
    axes[1, 1].set_ylim(0, 0.82)
    axes[1, 1].set_ylabel("Five-camera p90 error")
    axes[1, 1].set_title("All three methods clear the absolute p90 gates", loc="left", fontweight="bold")
    axes[1, 1].legend(frameon=False, fontsize=8)
    axes[1, 1].grid(axis="y", alpha=0.2)

    fig.text(
        0.5,
        0.025,
        "Development evidence only: p22 is already open; no p14, wall/RSS, external, or real-BOST claim.",
        ha="center",
        fontsize=10,
        color="#4d5560",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
