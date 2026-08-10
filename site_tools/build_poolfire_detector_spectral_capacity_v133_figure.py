from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_detector_spectral_capacity_v133_public_summary.json"
OUTPUT = ROOT / "assets/figures/poolfire_detector_spectral_capacity_v133.png"


def _values(data: dict, family: str) -> list[float]:
    return [data[family][trajectory] for trajectory in data["trajectory_order"]]


def build_figure() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    labels = ["14kW-05", "22kW-03", "33kW-01", "45kW-05", "58kW-03"]
    x = np.arange(len(labels))

    fig = plt.figure(figsize=(16, 9), facecolor="#f4f7f6")
    grid = fig.add_gridspec(2, 2, width_ratios=(1.25, 1), hspace=0.36, wspace=0.22)

    observation_ax = fig.add_subplot(grid[0, :])
    families = [
        ("parent_v131_1_observation_p90_higher", "v131.1 learned proposal", "#8b989e"),
        ("v132_scalar_oracle_observation_p90_higher", "v132 scalar oracle", "#d5a94c"),
    ]
    width = 0.24
    for index, (family, label, color) in enumerate(families):
        observation_ax.bar(
            x + (index - 1) * width,
            _values(data, family),
            width,
            label=label,
            color=color,
        )
    v133 = [data["oracle_p90_higher"][trajectory]["observation"] for trajectory in data["trajectory_order"]]
    observation_ax.bar(x + width, v133, width, label="v133 spectral oracle", color="#4e8f78")
    observation_ax.axhline(1.02, color="#1f3037", linestyle="--", linewidth=1.5, label="p90 gate 1.02")
    observation_ax.set_xticks(x, labels)
    observation_ax.set_ylim(0.98, 1.53)
    observation_ax.set_ylabel("Observation error ratio p90")
    observation_ax.set_title("A. Spectral shape sharply reduces the observation tail, but 0/5 trajectories pass", loc="left", fontsize=14.5, weight="bold")
    observation_ax.grid(axis="y", alpha=0.18)
    observation_ax.legend(frameon=False, ncols=4, loc="upper left")

    camera_ax = fig.add_subplot(grid[1, 0])
    camera_counts = ["5", "7", "9", "12"]
    camera_x = np.arange(len(camera_counts))
    v132_counts = [data["cell_pass_by_camera_count"][count]["v132_passed"] for count in camera_counts]
    v133_counts = [data["cell_pass_by_camera_count"][count]["v133_passed"] for count in camera_counts]
    total = data["cell_pass_by_camera_count"]["5"]["total"]
    camera_ax.bar(camera_x - 0.18, np.asarray(v132_counts) / total * 100, 0.36, label="v132 scalar", color="#d5a94c")
    camera_ax.bar(camera_x + 0.18, np.asarray(v133_counts) / total * 100, 0.36, label="v133 spectral", color="#4e8f78")
    camera_ax.set_xticks(camera_x, [f"{count} cameras" for count in camera_counts])
    camera_ax.set_ylim(0, 92)
    camera_ax.set_ylabel("Cells passing all four metrics (%)")
    camera_ax.set_title("B. More cameras help; strict tail failures remain", loc="left", fontsize=14.5, weight="bold")
    camera_ax.grid(axis="y", alpha=0.18)
    camera_ax.legend(frameon=False, loc="upper left")

    metric_ax = fig.add_subplot(grid[1, 1])
    metric_labels = ["Field", "Full grad.", "Interior grad.", "Observation"]
    metric_keys = ["field", "full_gradient", "interior_gradient", "observation"]
    metric_values = [data["metric_cell_pass_counts"][key] for key in metric_keys]
    colors = ["#4e8f78", "#4e8f78", "#4e8f78", "#d45a45"]
    bars = metric_ax.bar(np.arange(4), metric_values, color=colors)
    metric_ax.set_xticks(np.arange(4), metric_labels)
    metric_ax.set_ylim(0, 4050)
    metric_ax.set_ylabel("Cells at ratio <= 1.05 (of 3,700)")
    metric_ax.set_title("C. All 1,347 failures are observation-only", loc="left", fontsize=14.5, weight="bold")
    metric_ax.grid(axis="y", alpha=0.18)
    for bar, value in zip(bars, metric_values, strict=True):
        metric_ax.text(bar.get_x() + bar.get_width() / 2, value + 65, f"{value}", ha="center", fontsize=10.5, weight="bold")

    fig.suptitle(
        "v133: spectral representation localizes the remaining failure to observation space",
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
        "3,700 opened PoolFire cells | 2A+2A^T candidate shell | independently recomputed | algorithm_breakthrough=false",
        fontsize=11.3,
        color="#53636b",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
