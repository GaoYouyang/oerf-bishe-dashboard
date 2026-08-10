from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_residual_adaptive_local_window_capacity_v136_public_summary.json"
OUTPUT = ROOT / "assets/figures/poolfire_residual_adaptive_local_window_capacity_v136.png"


def build_figure() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    gate = data["strict_gate"]

    fig = plt.figure(figsize=(16, 9), facecolor="#f3f7f5")
    grid = fig.add_gridspec(2, 2, width_ratios=(1, 1.35), hspace=0.38, wspace=0.24)

    cascade_ax = fig.add_subplot(grid[0, 0])
    stage_labels = ["v133\nGlobal", "v134\nPareto", "v135\nFixed local", "v136\nAdaptive local"]
    stage_values = [gate["v133_passed"], gate["v134_passed"], gate["v135_passed"], gate["v136_passed"]]
    stage_colors = ["#8c999f", "#6b9688", "#3f806f", "#c76a52"]
    bars = cascade_ax.bar(np.arange(4), stage_values, color=stage_colors, width=0.68)
    cascade_ax.axhline(3700, color="#26383f", linestyle="--", linewidth=1.4)
    cascade_ax.set_xticks(np.arange(4), stage_labels)
    cascade_ax.set_ylim(0, 4000)
    cascade_ax.set_ylabel("Cells passing all four metrics")
    cascade_ax.set_title("A. Adaptation adds only 53 strict passes", loc="left", fontsize=12.8, weight="bold")
    cascade_ax.grid(axis="y", alpha=0.18)
    for bar, value in zip(bars, stage_values, strict=True):
        cascade_ax.text(bar.get_x() + bar.get_width() / 2, value + 60, str(value), ha="center", weight="bold")

    camera_ax = fig.add_subplot(grid[0, 1])
    camera_counts = ["5", "7", "9", "12"]
    x = np.arange(len(camera_counts))
    by_camera = data["cell_pass_by_camera_count"]
    v135 = [by_camera[count]["v135_passed"] for count in camera_counts]
    v136 = [by_camera[count]["v136_passed"] for count in camera_counts]
    width = 0.34
    camera_ax.bar(x - width / 2, v135, width, label="v135 fixed", color="#8c999f")
    camera_ax.bar(x + width / 2, v136, width, label="v136 adaptive", color="#c76a52")
    camera_ax.set_xticks(x, [f"{count} cameras" for count in camera_counts])
    camera_ax.set_ylim(0, 980)
    camera_ax.set_ylabel("Passing cells (of 925)")
    camera_ax.set_title("B. Five-camera sparse views still dominate", loc="left", fontsize=12.8, weight="bold")
    camera_ax.grid(axis="y", alpha=0.18)
    camera_ax.legend(frameon=False, loc="upper left")
    for i, count in enumerate(camera_counts):
        camera_ax.text(i + width / 2, v136[i] + 20, str(v136[i]), ha="center", fontsize=9.5, weight="bold")
        camera_ax.text(i, 48, f"remain {by_camera[count]['remaining']}", ha="center", fontsize=9, color="#9a4337")

    trajectory_ax = fig.add_subplot(grid[1, 0])
    rows = data["trajectory_rows"]
    labels = [row["trajectory"].replace("p=", "").replace("kw_size=", "-") for row in rows]
    remaining = [row["remaining"] for row in rows]
    bars = trajectory_ax.bar(np.arange(len(rows)), remaining, color=["#79a99b", "#79a99b", "#79a99b", "#c25b49", "#d08a43"])
    trajectory_ax.set_xticks(np.arange(len(rows)), labels, rotation=15)
    trajectory_ax.set_ylim(0, 260)
    trajectory_ax.set_ylabel("Remaining observation-only failures")
    trajectory_ax.set_title("C. p45 and p58 still carry 409/485 failures", loc="left", fontsize=12.8, weight="bold")
    trajectory_ax.grid(axis="y", alpha=0.18)
    for bar, value in zip(bars, remaining, strict=True):
        trajectory_ax.text(bar.get_x() + bar.get_width() / 2, value + 6, str(value), ha="center", weight="bold")

    tail_ax = fig.add_subplot(grid[1, 1])
    p90 = [row["observation_p90_higher"] for row in rows]
    worst = [row["observation_worst"] for row in rows]
    tail_ax.plot(np.arange(len(rows)), p90, marker="o", linewidth=2.4, label="observation p90-higher", color="#2b6f62")
    tail_ax.plot(np.arange(len(rows)), worst, marker="s", linewidth=2.0, label="observation worst", color="#c25b49")
    tail_ax.axhline(1.02, color="#2b6f62", linestyle="--", linewidth=1.2, label="trajectory p90 gate 1.02")
    tail_ax.axhline(1.05, color="#26383f", linestyle=":", linewidth=1.5, label="worst gate 1.05")
    tail_ax.set_xticks(np.arange(len(rows)), labels, rotation=15)
    tail_ax.set_ylim(1.0, 1.27)
    tail_ax.set_ylabel("Candidate / Zero-K4 error ratio")
    tail_ax.set_title("D. Observation tails remain above both gates", loc="left", fontsize=12.8, weight="bold")
    tail_ax.grid(axis="y", alpha=0.18)
    tail_ax.legend(frameon=False, loc="upper left")

    fig.suptitle(
        "v136: residual-centroid/width adaptation is insufficient",
        x=0.055,
        y=0.985,
        ha="left",
        fontsize=19.5,
        weight="bold",
        color="#18262d",
    )
    fig.text(
        0.055,
        0.947,
        "3,215/3,700 strict cells | +53 over v135 | 485 observation-only failures | 0/5 complete trajectories | independently recomputed | algorithm_breakthrough=false",
        fontsize=10.9,
        color="#53636b",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
