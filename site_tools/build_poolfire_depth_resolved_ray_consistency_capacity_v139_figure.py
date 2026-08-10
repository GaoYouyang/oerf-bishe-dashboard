from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_depth_resolved_ray_consistency_capacity_v139_public_summary.json"
OUTPUT = ROOT / "assets/figures/poolfire_depth_resolved_ray_consistency_capacity_v139.png"


def build_figure() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    gate = data["strict_gate"]
    fig = plt.figure(figsize=(16, 9), facecolor="#f3f7f5")
    grid = fig.add_gridspec(2, 2, width_ratios=(1, 1.35), hspace=0.38, wspace=0.24)

    cascade = fig.add_subplot(grid[0, 0])
    labels = ["v137\nSigned + peer", "v138\nRay overlap", "v139\n3D consistency"]
    values = [gate["v137_passed"], gate["v138_passed"], gate["v139_passed"]]
    bars = cascade.bar(np.arange(3), values, color=["#8c999f", "#6b9688", "#c76a52"], width=0.66)
    cascade.axhline(3700, color="#26383f", linestyle="--", linewidth=1.4)
    cascade.set_xticks(np.arange(3), labels)
    cascade.set_ylim(0, 4000)
    cascade.set_ylabel("Cells passing all four metrics")
    cascade.set_title("A. Depth-resolved consistency rescues 152 cells", loc="left", fontsize=12.8, weight="bold")
    cascade.grid(axis="y", alpha=0.18)
    for bar, value in zip(bars, values, strict=True):
        cascade.text(bar.get_x() + bar.get_width() / 2, value + 60, str(value), ha="center", weight="bold")

    camera = fig.add_subplot(grid[0, 1])
    counts = ["5", "7", "9", "12"]
    x = np.arange(len(counts))
    rows = data["cell_pass_by_camera_count"]
    v138 = [rows[count]["v138_passed"] for count in counts]
    v139 = [rows[count]["v139_passed"] for count in counts]
    width = 0.34
    camera.bar(x - width / 2, v138, width, label="v138 ray overlap", color="#8c999f")
    camera.bar(x + width / 2, v139, width, label="v139 3D consistency", color="#c76a52")
    camera.set_xticks(x, [f"{count} cameras" for count in counts])
    camera.set_ylim(0, 990)
    camera.set_ylabel("Passing cells (of 925)")
    camera.set_title("B. Every unresolved cell now uses five cameras", loc="left", fontsize=12.8, weight="bold")
    camera.grid(axis="y", alpha=0.18)
    camera.legend(frameon=False, loc="upper left")
    for i, count in enumerate(counts):
        camera.text(i + width / 2, v139[i] + 17, str(v139[i]), ha="center", fontsize=9.5, weight="bold")
        camera.text(i, 48, f"remain {rows[count]['remaining']}", ha="center", fontsize=9, color="#9a4337")

    attribution = fig.add_subplot(grid[1, 0])
    labels = ["3D-consistency\nrescues", "Still\nunresolved", "Cheap control\npasses"]
    values = [152, 151, 0]
    bars = attribution.bar(np.arange(3), values, color=["#3f806f", "#c25b49", "#8c999f"], width=0.65)
    attribution.set_xticks(np.arange(3), labels)
    attribution.set_ylim(0, 180)
    attribution.set_ylabel("Cells")
    attribution.set_title("C. The mechanism halves the tail but does not close it", loc="left", fontsize=12.8, weight="bold")
    attribution.grid(axis="y", alpha=0.18)
    for bar, value in zip(bars, values, strict=True):
        attribution.text(bar.get_x() + bar.get_width() / 2, value + 5, str(value), ha="center", weight="bold")

    tail = fig.add_subplot(grid[1, 1])
    rows = data["trajectory_rows"]
    labels = [row["trajectory"].replace("p=", "").replace("kw_size=", "-") for row in rows]
    p90 = [row["observation_p90_higher"] for row in rows]
    worst = [row["observation_worst"] for row in rows]
    tail.plot(np.arange(len(rows)), p90, marker="o", linewidth=2.4, label="observation p90-higher", color="#2b6f62")
    tail.plot(np.arange(len(rows)), worst, marker="s", linewidth=2.0, label="observation worst", color="#c25b49")
    tail.axhline(1.02, color="#2b6f62", linestyle="--", linewidth=1.2, label="trajectory p90 gate 1.02")
    tail.axhline(1.05, color="#26383f", linestyle=":", linewidth=1.5, label="worst gate 1.05")
    tail.set_xticks(np.arange(len(rows)), labels, rotation=15)
    tail.set_ylim(1.0, 1.095)
    tail.set_ylabel("Candidate / Zero-K4 error ratio")
    tail.set_title("D. p45 and p58 retain the decisive observation tail", loc="left", fontsize=12.8, weight="bold")
    tail.grid(axis="y", alpha=0.18)
    tail.legend(frameon=False, loc="upper left")

    fig.suptitle(
        "v139: 3D ray consistency helps strongly; five-camera depth ambiguity remains",
        x=0.055,
        y=0.985,
        ha="left",
        fontsize=18.5,
        weight="bold",
        color="#18262d",
    )
    fig.text(
        0.055,
        0.947,
        "3,549/3,700 strict cells | +152 rescues | all 151 failures at five cameras | 0/5 complete trajectories | independently recomputed | algorithm_breakthrough=false",
        fontsize=10.5,
        color="#53636b",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
