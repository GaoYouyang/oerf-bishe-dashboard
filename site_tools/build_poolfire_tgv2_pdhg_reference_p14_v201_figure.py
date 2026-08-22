#!/usr/bin/env python3
"""Build the public v201 TGV2-PDHG reference-attribution figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_tgv2_pdhg_reference_p14_v201_public_summary.json"
OUTPUT = ROOT / "assets/figures/poolfire_tgv2_pdhg_reference_p14_v201.png"


def main() -> None:
    payload = json.loads(SUMMARY.read_text())
    tgv2 = payload["fixed_primary"]["five_camera"]
    huber = payload["controls"]["huber_pdhg_parent_v200"]["five_camera"]
    k2 = payload["controls"]["full_dct_k2_parent"]["five_camera"]
    attribution = payload["mechanism_attribution"]
    labels = ("Full-DCT K2\nparent", "Huber-TV\nparent", "TGV2\nreference")
    colors = ("#7b8491", "#2878b5", "#d1495b")

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 5.2))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.80, bottom=0.22, wspace=0.34)
    fig.suptitle(
        "v201  TGV2 fits observations better but rescues no five-camera failure",
        fontsize=15,
        fontweight="bold",
    )

    x = np.arange(3)
    safe = (k2["strict_safe_cells"], huber["strict_safe_cells"], tgv2["strict_safe_cells"])
    axes[0].bar(x, safe, color=colors, width=0.68)
    axes[0].axhline(1313, color="#20262e", linestyle="--", linewidth=1.2)
    axes[0].set_ylim(1180, 1325)
    axes[0].set_xticks(x, labels, fontsize=9)
    axes[0].set_ylabel("Strict-safe cells / 1313")
    axes[0].set_title("No failed cell rescued", loc="left", fontweight="bold")
    axes[0].grid(axis="y", alpha=0.18)
    for index, value in enumerate(safe):
        axes[0].text(index, value + 5, f"{value}/1313", ha="center", fontsize=9)

    groups = (
        k2["complete_groups_passed"],
        huber["complete_groups_passed"],
        tgv2["complete_groups_passed"],
    )
    axes[1].bar(x, groups, color=colors, width=0.68)
    axes[1].axhline(13, color="#20262e", linestyle="--", linewidth=1.2)
    axes[1].set_ylim(0, 14)
    axes[1].set_xticks(x, labels, fontsize=9)
    axes[1].set_ylabel("Complete calibration groups / 13")
    axes[1].set_title("Complete-group gate unchanged", loc="left", fontweight="bold")
    axes[1].grid(axis="y", alpha=0.18)
    for index, value in enumerate(groups):
        axes[1].text(index, value + 0.3, f"{value}/13", ha="center", fontsize=9)

    metric_names = ("field", "gradient", "observation")
    metric_labels = ("Field", "Gradient", "Observation")
    metric_x = np.arange(3)
    width = 0.34
    huber_values = [huber["p90"][name] for name in metric_names]
    tgv2_values = [tgv2["p90"][name] for name in metric_names]
    axes[2].bar(metric_x - width / 2, huber_values, width, color=colors[1], label="Huber-TV")
    axes[2].bar(metric_x + width / 2, tgv2_values, width, color=colors[2], label="TGV2")
    for index, limit in enumerate((0.5, 0.75, 0.2)):
        axes[2].hlines(limit, index - 0.43, index + 0.43, color="#20262e", linestyle="--", linewidth=1.2)
    axes[2].set_xticks(metric_x, metric_labels)
    axes[2].set_ylim(0, 0.82)
    axes[2].set_ylabel("Five-camera p90 error")
    axes[2].set_title("Same gradient tail; lower observation", loc="left", fontweight="bold")
    axes[2].legend(frameon=False, fontsize=9)
    axes[2].grid(axis="y", alpha=0.18)
    axes[2].text(
        2,
        max(huber_values[2], tgv2_values[2]) + 0.045,
        f"lower in {attribution['observation_error_improved_cells']}/1313 cells",
        ha="center",
        fontsize=8.5,
        color="#4d5560",
    )

    fig.text(
        0.5,
        0.055,
        "Historically exposed p14 development only. Failure overlap: 24/24; zero rescues. No call, wall/RSS, external, or real-BOST claim.",
        ha="center",
        fontsize=9.5,
        color="#4d5560",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
