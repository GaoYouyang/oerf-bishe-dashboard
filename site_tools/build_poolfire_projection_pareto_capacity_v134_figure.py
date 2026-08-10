from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_projection_pareto_capacity_v134_public_summary.json"
OUTPUT = ROOT / "assets/figures/poolfire_projection_pareto_capacity_v134.png"


def build_figure() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    trajectories = data["trajectory_order"]
    labels = ["14kW-05", "22kW-03", "33kW-01", "45kW-05", "58kW-03"]
    x = np.arange(len(labels))

    fig = plt.figure(figsize=(16, 9), facecolor="#f4f7f6")
    grid = fig.add_gridspec(2, 2, width_ratios=(1.25, 1), hspace=0.38, wspace=0.22)

    trajectory_ax = fig.add_subplot(grid[0, :])
    v133 = [
        1.04521,
        1.06888,
        1.07250,
        1.14836,
        1.20331,
    ]
    v134 = [data["selected_p90_higher"][name]["observation"] for name in trajectories]
    width = 0.34
    trajectory_ax.bar(x - width / 2, v133, width, label="v133 equal-weight", color="#8b989e")
    trajectory_ax.bar(x + width / 2, v134, width, label="v134 Pareto-selected", color="#3e8b78")
    trajectory_ax.axhline(1.02, color="#26383f", linestyle="--", linewidth=1.5, label="p90 gate 1.02")
    trajectory_ax.set_xticks(x, labels)
    trajectory_ax.set_ylim(0.98, 1.235)
    trajectory_ax.set_ylabel("Observation error ratio p90")
    trajectory_ax.set_title(
        "A. Pareto selection helps every trajectory, but none clears the strict tail gate",
        loc="left",
        fontsize=14.2,
        weight="bold",
    )
    trajectory_ax.grid(axis="y", alpha=0.18)
    trajectory_ax.legend(frameon=False, ncols=3, loc="upper left")

    camera_ax = fig.add_subplot(grid[1, 0])
    camera_counts = ["5", "7", "9", "12"]
    camera_x = np.arange(len(camera_counts))
    v133_camera = [data["cell_pass_by_camera_count"][count]["v133_passed"] for count in camera_counts]
    v134_camera = [data["cell_pass_by_camera_count"][count]["v134_passed"] for count in camera_counts]
    total = data["cell_pass_by_camera_count"]["5"]["total"]
    camera_ax.bar(camera_x - 0.18, np.asarray(v133_camera) / total * 100, 0.36, label="v133", color="#8b989e")
    camera_ax.bar(camera_x + 0.18, np.asarray(v134_camera) / total * 100, 0.36, label="v134", color="#3e8b78")
    camera_ax.set_xticks(camera_x, [f"{count} cameras" for count in camera_counts])
    camera_ax.set_ylim(0, 98)
    camera_ax.set_ylabel("Cells passing all four metrics (%)")
    camera_ax.set_title("B. Sparse camera sets retain the largest capacity gap", loc="left", fontsize=14.2, weight="bold")
    camera_ax.grid(axis="y", alpha=0.18)
    camera_ax.legend(frameon=False, loc="upper left")

    metric_ax = fig.add_subplot(grid[1, 1])
    metric_labels = ["Field", "Full grad.", "Interior grad.", "Observation"]
    metric_keys = ["field", "full_gradient", "interior_gradient", "observation"]
    metric_values = [data["metric_cell_pass_counts"][key] for key in metric_keys]
    colors = ["#3e8b78", "#3e8b78", "#3e8b78", "#c45746"]
    bars = metric_ax.bar(np.arange(4), metric_values, color=colors)
    metric_ax.set_xticks(np.arange(4), metric_labels)
    metric_ax.set_ylim(0, 4050)
    metric_ax.set_ylabel("Cells at ratio <= 1.05 (of 3,700)")
    metric_ax.set_title("C. All 1,109 failures remain observation-only", loc="left", fontsize=14.2, weight="bold")
    metric_ax.grid(axis="y", alpha=0.18)
    for bar, value in zip(bars, metric_values, strict=True):
        metric_ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 65,
            f"{value}",
            ha="center",
            fontsize=10.5,
            weight="bold",
        )

    fig.suptitle(
        "v134: objective retuning narrows but does not close fixed-span capacity",
        x=0.055,
        y=0.985,
        ha="left",
        fontsize=20.5,
        weight="bold",
        color="#18262d",
    )
    fig.text(
        0.055,
        0.947,
        "2,591/3,700 strict cells | 0/5 complete trajectories | 2A+2A^T shell | independently recomputed | algorithm_breakthrough=false",
        fontsize=11.1,
        color="#53636b",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
