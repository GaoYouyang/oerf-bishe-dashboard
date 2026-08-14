"""Build the public v146 direction-conditioned identifiability figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / (
    "docs/poolfire_k1_dual_direction_conditioned_identifiability_v146_"
    "public_summary.json"
)
OUTPUT = ROOT / (
    "assets/figures/poolfire_k1_dual_direction_conditioned_"
    "identifiability_v146.png"
)


def main() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    if payload["scientific_decision"] != "FAIL_DIRECTION_CONDITIONED_IDENTIFIABILITY_V146":
        raise RuntimeError("refusing to plot an unexpected v146 decision")

    methods = payload["methods"]
    labels = [
        "Count mean",
        "Cross global",
        "Cross local",
        "Cross local +\npose coupling",
        "Within* local",
        "Within* local +\npose coupling",
    ]
    keys = [
        "count_mean",
        "cross_global_pose_coupled",
        "cross_local_only",
        "cross_local_pose_coupled",
        "within_local_only",
        "within_local_pose_coupled",
    ]
    colors = ["#78848d", "#b8493d", "#c45d4b", "#d47a57", "#276c66", "#3f7f78"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(15.8, 7.5))
    figure.subplots_adjust(
        left=0.065, right=0.985, bottom=0.17, top=0.84, wspace=0.31
    )
    figure.patch.set_facecolor("#f7f8f6")

    axis = axes[0]
    sentinel_passes = [methods[key]["sentinel_pass_count"] for key in keys]
    bars = axis.barh(np.arange(len(keys)), sentinel_passes, color=colors, height=0.62)
    axis.axvline(
        20,
        color="#212529",
        linestyle=(0, (4, 3)),
        linewidth=1.5,
        label="Required: 20/20",
    )
    axis.set_yticks(np.arange(len(keys)), labels)
    axis.invert_yaxis()
    axis.set_xlim(0, 21.5)
    axis.set_xlabel("Frozen sentinels passing")
    axis.set_title("A  No method passes Stage A", loc="left", weight="bold")
    axis.legend(frameon=False, loc="lower right")
    for bar, value in zip(bars, sentinel_passes, strict=True):
        axis.text(
            value + 0.35,
            bar.get_y() + bar.get_height() / 2,
            f"{value}/20",
            va="center",
            weight="bold",
        )

    axis = axes[1]
    medians = [methods[key]["error_median"] for key in keys]
    p90s = [methods[key]["error_p90_higher"] for key in keys]
    x = np.arange(len(keys))
    width = 0.36
    axis.bar(x - width / 2, medians, width, color=colors, alpha=0.95, label="Median")
    axis.bar(
        x + width / 2,
        p90s,
        width,
        color=colors,
        alpha=0.45,
        edgecolor=colors,
        linewidth=1.1,
        label="p90-higher",
    )
    axis.axhline(
        0.45,
        color="#212529",
        linestyle=(0, (4, 3)),
        linewidth=1.5,
        label="Per-sentinel error gate 0.45",
    )
    axis.set_xticks(x, ["Mean", "X-global", "X-local", "X-local+", "W-local*", "W-local+*"])
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Scale-invariant relative-L2")
    axis.set_title("B  Local state helps, but remains insufficient", loc="left", weight="bold")
    axis.legend(frameon=False, loc="upper right")

    axis = axes[2]
    trajectories = ["p14", "p22", "p33", "p45", "p58"]
    cross = list(methods["cross_local_only"]["trajectory_p90_higher"].values())
    within = list(methods["within_local_only"]["trajectory_p90_higher"].values())
    x = np.arange(len(trajectories))
    axis.bar(x - width / 2, cross, width, color="#c45d4b", label="Cross local")
    axis.bar(x + width / 2, within, width, color="#276c66", label="Within* local")
    axis.axhline(
        0.35,
        color="#212529",
        linestyle=(0, (4, 3)),
        linewidth=1.5,
        label="Trajectory gate 0.35",
    )
    axis.set_xticks(x, trajectories)
    axis.set_ylim(0, 0.96)
    axis.set_ylabel("Four-sentinel trajectory p90-higher")
    axis.set_title("C  Every trajectory misses the tail gate", loc="left", weight="bold")
    axis.legend(frameon=False, loc="upper left")

    for axis in axes:
        axis.set_facecolor("#ffffff")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#d9ddda", linewidth=0.7, alpha=0.75)
        axis.set_axisbelow(True)

    figure.suptitle(
        "v146 | Hard-count direction conditioning still does not identify the action target",
        fontsize=18,
        weight="bold",
        x=0.04,
        y=0.97,
        ha="left",
    )
    figure.text(
        0.04,
        0.045,
        "Stage A only: 20 frozen sentinels | Stage B 3700-cell run not started | "
        "+0A/+0A^T | no GPU | algorithm_breakthrough=false",
        fontsize=10,
        color="#39434a",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=170, facecolor=figure.get_facecolor())
    plt.close(figure)


if __name__ == "__main__":
    main()
