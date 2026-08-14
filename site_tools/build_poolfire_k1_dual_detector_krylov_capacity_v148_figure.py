"""Build the public v148 detector-group Krylov capacity figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_k1_dual_detector_krylov_capacity_v148_public_summary.json"
OUTPUT = ROOT / "assets/figures/poolfire_k1_dual_detector_krylov_capacity_v148.png"


def main() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    if payload["scientific_decision"] != "HEADROOM_BLOCK_DETECTOR_KRYLOV4_V148":
        raise RuntimeError("refusing to plot an unexpected v148 decision")

    methods = payload["methods"]
    keys = ["visible_seed_control", "global_krylov4", "block_krylov4"]
    labels = ["Visible\nseed", "Global\nKrylov-4*", "Block\nKrylov-4*"]
    colors = ["#64727a", "#d47754", "#276c66"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 9,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(14.8, 7.2))
    figure.subplots_adjust(left=0.07, right=0.98, bottom=0.18, top=0.82, wspace=0.27)
    figure.patch.set_facecolor("#f7f8f6")

    axis = axes[0]
    x = np.arange(len(keys))
    sentinels = [methods[key]["sentinel_pass_count"] for key in keys]
    trajectories = [methods[key]["trajectory_pass_count"] for key in keys]
    width = 0.34
    bars1 = axis.bar(x - width / 2, sentinels, width, color=colors, label="Sentinels / 20")
    bars2 = axis.bar(
        x + width / 2,
        [value * 4 for value in trajectories],
        width,
        color=colors,
        alpha=0.42,
        edgecolor=colors,
        linewidth=1.1,
        label="Trajectory tails / 5 (scaled x4)",
    )
    axis.axhline(20, color="#202629", linestyle=(0, (4, 3)), linewidth=1.4)
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 21.8)
    axis.set_ylabel("Pass count")
    axis.set_title("A  Only the groupwise spectral model clears both gates", loc="left", weight="bold")
    axis.legend(frameon=False, loc="lower right")
    for bar, value in zip(bars1, sentinels, strict=True):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.45, f"{value}/20", ha="center", weight="bold")
    for bar, value in zip(bars2, trajectories, strict=True):
        axis.text(bar.get_x() + bar.get_width() / 2, value * 4 + 0.45, f"{value}/5", ha="center", color="#39434a")

    axis = axes[1]
    trajectory_labels = ["p14", "p22", "p33", "p45", "p58"]
    global_tail = list(methods["global_krylov4"]["trajectory_p90_higher"].values())
    block_tail = list(methods["block_krylov4"]["trajectory_p90_higher"].values())
    x = np.arange(len(trajectory_labels))
    axis.bar(x - width / 2, global_tail, width, color="#d47754", label="Global Krylov-4*")
    axis.bar(x + width / 2, block_tail, width, color="#276c66", label="Block Krylov-4*")
    axis.axhline(0.35, color="#202629", linestyle=(0, (4, 3)), linewidth=1.4, label="Tail gate 0.35")
    axis.set_xticks(x, trajectory_labels)
    axis.set_ylim(0, 0.62)
    axis.set_ylabel("Four-sentinel p90-higher error")
    axis.set_title("B  Groupwise coefficients remove the p45/p58 tail failure", loc="left", weight="bold")
    axis.legend(frameon=False, loc="upper left")

    for axis in axes:
        axis.set_facecolor("#ffffff")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#d9ddda", linewidth=0.7, alpha=0.75)
        axis.set_axisbelow(True)

    figure.suptitle(
        "v148 | Detector-group Krylov capacity passes the frozen sentinel and trajectory gates",
        fontsize=18,
        weight="bold",
        x=0.04,
        y=0.96,
        ha="left",
    )
    figure.text(
        0.04,
        0.06,
        "* Truth-aware post-open coefficient oracle | 20 frozen sentinels | +0A/+0A^T | no physical replay | "
        "no GPU | algorithm_breakthrough=false",
        fontsize=9.8,
        color="#39434a",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=170, facecolor=figure.get_facecolor())
    plt.close(figure)


if __name__ == "__main__":
    main()
