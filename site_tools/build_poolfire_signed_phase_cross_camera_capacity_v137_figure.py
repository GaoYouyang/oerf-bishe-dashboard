from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_signed_phase_cross_camera_capacity_v137_public_summary.json"
OUTPUT = ROOT / "assets/figures/poolfire_signed_phase_cross_camera_capacity_v137.png"


def build_figure() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    gate = data["strict_gate"]

    fig = plt.figure(figsize=(16, 9), facecolor="#f3f7f5")
    grid = fig.add_gridspec(2, 2, width_ratios=(1, 1.35), hspace=0.38, wspace=0.24)

    cascade_ax = fig.add_subplot(grid[0, 0])
    stage_labels = ["v135\nFixed local", "v136\nAdaptive", "v137\nSigned + peer"]
    stage_values = [gate["v135_passed"], gate["v136_passed"], gate["v137_passed"]]
    stage_colors = ["#8c999f", "#6b9688", "#c76a52"]
    bars = cascade_ax.bar(np.arange(3), stage_values, color=stage_colors, width=0.66)
    cascade_ax.axhline(3700, color="#26383f", linestyle="--", linewidth=1.4)
    cascade_ax.set_xticks(np.arange(3), stage_labels)
    cascade_ax.set_ylim(0, 4000)
    cascade_ax.set_ylabel("Cells passing all four metrics")
    cascade_ax.set_title("A. Phase and peer geometry rescue 136 cells", loc="left", fontsize=12.8, weight="bold")
    cascade_ax.grid(axis="y", alpha=0.18)
    for bar, value in zip(bars, stage_values, strict=True):
        cascade_ax.text(bar.get_x() + bar.get_width() / 2, value + 60, str(value), ha="center", weight="bold")

    camera_ax = fig.add_subplot(grid[0, 1])
    camera_counts = ["5", "7", "9", "12"]
    x = np.arange(len(camera_counts))
    by_camera = data["cell_pass_by_camera_count"]
    v136 = [by_camera[count]["v136_passed"] for count in camera_counts]
    v137 = [by_camera[count]["v137_passed"] for count in camera_counts]
    width = 0.34
    camera_ax.bar(x - width / 2, v136, width, label="v136 adaptive", color="#8c999f")
    camera_ax.bar(x + width / 2, v137, width, label="v137 signed + peer", color="#c76a52")
    camera_ax.set_xticks(x, [f"{count} cameras" for count in camera_counts])
    camera_ax.set_ylim(0, 990)
    camera_ax.set_ylabel("Passing cells (of 925)")
    camera_ax.set_title("B. The unresolved tail collapses into five cameras", loc="left", fontsize=12.8, weight="bold")
    camera_ax.grid(axis="y", alpha=0.18)
    camera_ax.legend(frameon=False, loc="upper left")
    for i, count in enumerate(camera_counts):
        camera_ax.text(i + width / 2, v137[i] + 17, str(v137[i]), ha="center", fontsize=9.5, weight="bold")
        camera_ax.text(i, 48, f"remain {by_camera[count]['remaining']}", ha="center", fontsize=9, color="#9a4337")

    attribution_ax = fig.add_subplot(grid[1, 0])
    attribution_labels = ["Self signed\nphase", "Peer geometry\nincrement", "Still\nunresolved"]
    attribution_values = [92, 44, 349]
    attribution_colors = ["#3f806f", "#5f86a6", "#c25b49"]
    bars = attribution_ax.bar(np.arange(3), attribution_values, color=attribution_colors, width=0.65)
    attribution_ax.set_xticks(np.arange(3), attribution_labels)
    attribution_ax.set_ylim(0, 390)
    attribution_ax.set_ylabel("Cells")
    attribution_ax.set_title("C. Cross-camera coupling helps, but is insufficient", loc="left", fontsize=12.8, weight="bold")
    attribution_ax.grid(axis="y", alpha=0.18)
    for bar, value in zip(bars, attribution_values, strict=True):
        attribution_ax.text(bar.get_x() + bar.get_width() / 2, value + 8, str(value), ha="center", weight="bold")

    tail_ax = fig.add_subplot(grid[1, 1])
    rows = data["trajectory_rows"]
    labels = [row["trajectory"].replace("p=", "").replace("kw_size=", "-") for row in rows]
    p90 = [row["observation_p90_higher"] for row in rows]
    worst = [row["observation_worst"] for row in rows]
    tail_ax.plot(np.arange(len(rows)), p90, marker="o", linewidth=2.4, label="observation p90-higher", color="#2b6f62")
    tail_ax.plot(np.arange(len(rows)), worst, marker="s", linewidth=2.0, label="observation worst", color="#c25b49")
    tail_ax.axhline(1.02, color="#2b6f62", linestyle="--", linewidth=1.2, label="trajectory p90 gate 1.02")
    tail_ax.axhline(1.05, color="#26383f", linestyle=":", linewidth=1.5, label="worst gate 1.05")
    tail_ax.set_xticks(np.arange(len(rows)), labels, rotation=15)
    tail_ax.set_ylim(1.0, 1.18)
    tail_ax.set_ylabel("Candidate / Zero-K4 error ratio")
    tail_ax.set_title("D. p45 and p58 remain above trajectory gates", loc="left", fontsize=12.8, weight="bold")
    tail_ax.grid(axis="y", alpha=0.18)
    tail_ax.legend(frameon=False, loc="upper left")

    fig.suptitle(
        "v137: signed phase helps; sparse-view geometry still fails",
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
        "3,351/3,700 strict cells | +92 self-phase +44 peer-geometry rescues | 343/349 failures at five cameras | 0/5 complete trajectories | independently recomputed | algorithm_breakthrough=false",
        fontsize=10.5,
        color="#53636b",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
