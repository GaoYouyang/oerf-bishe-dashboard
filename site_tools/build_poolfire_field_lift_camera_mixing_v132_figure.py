from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_field_lift_camera_mixing_v132_public_summary.json"
OUTPUT = ROOT / "assets/figures/poolfire_field_lift_camera_mixing_v132.png"


def _trajectory_values(data: dict, family: str, metric: str) -> list[float]:
    return [data[family][trajectory][metric] for trajectory in data["trajectory_order"]]


def build_figure() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    labels = ["14kW-05", "22kW-03", "33kW-01", "45kW-05", "58kW-03"]
    x = np.arange(len(labels))
    width = 0.25

    fig = plt.figure(figsize=(16, 9), facecolor="#f4f7f6")
    grid = fig.add_gridspec(2, 2, width_ratios=(1.2, 1), hspace=0.38, wspace=0.22)

    observation_ax = fig.add_subplot(grid[0, :])
    families = [
        ("parent_v131_1_p90_higher", "Parent v131.1", "#8b989e"),
        ("oracle_p90_higher", "Truth-aware camera-scalar oracle", "#d45a45"),
        ("rms_control_p90_higher", "Observable RMS control", "#d5a94c"),
    ]
    for index, (family, label, color) in enumerate(families):
        values = _trajectory_values(data, family, "observation")
        observation_ax.bar(x + (index - 1) * width, values, width, label=label, color=color)
    observation_ax.axhline(1.02, color="#1f3037", linestyle="--", linewidth=1.5, label="p90 gate 1.02")
    observation_ax.set_xticks(x, labels)
    observation_ax.set_ylim(1.0, 1.55)
    observation_ax.set_ylabel("Observation error ratio p90")
    observation_ax.set_title("A. Per-camera scalar mixing does not repair observation-space shape", loc="left", fontsize=15, weight="bold")
    observation_ax.grid(axis="y", alpha=0.18)
    observation_ax.legend(frameon=False, ncols=4, loc="upper left")

    field_ax = fig.add_subplot(grid[1, 0])
    for index, (family, label, color) in enumerate(families):
        values = _trajectory_values(data, family, "field")
        field_ax.bar(x + (index - 1) * width, values, width, label=label, color=color)
    field_ax.axhline(1.02, color="#1f3037", linestyle="--", linewidth=1.5)
    field_ax.set_xticks(x, labels)
    field_ax.set_ylim(1.0, 1.18)
    field_ax.set_ylabel("Field error ratio p90")
    field_ax.set_title("B. Field improvement remains below the frozen gate", loc="left", fontsize=15, weight="bold")
    field_ax.grid(axis="y", alpha=0.18)

    diagnostic_ax = fig.add_subplot(grid[1, 1])
    diagnostic = data["oracle_diagnostic"]
    diagnostic_names = ["Field lift", "Projected lift"]
    diagnostic_keys = [
        "field_lift_relative_l2_before_line_search",
        "projected_lift_relative_l2_before_line_search",
    ]
    diagnostic_x = np.arange(2)
    diagnostic_width = 0.23
    for index, (statistic, color) in enumerate(
        [("p50", "#5f8795"), ("p90_higher", "#d5a94c"), ("worst", "#d45a45")]
    ):
        values = [diagnostic[key][statistic] for key in diagnostic_keys]
        diagnostic_ax.bar(
            diagnostic_x + (index - 1) * diagnostic_width,
            values,
            diagnostic_width,
            label=statistic.replace("_higher", ""),
            color=color,
        )
    diagnostic_ax.set_xticks(diagnostic_x, diagnostic_names)
    diagnostic_ax.set_ylim(0.0, 1.28)
    diagnostic_ax.set_ylabel("Relative L2 before line search")
    diagnostic_ax.set_title(
        "C. 0/5 pass: the remaining mismatch is spatial / spectral",
        loc="left",
        fontsize=15,
        weight="bold",
    )
    diagnostic_ax.grid(axis="y", alpha=0.18)
    diagnostic_ax.legend(frameon=False, ncols=3, loc="upper left")

    fig.suptitle(
        "v132: camera-relative gain is not the missing mechanism",
        x=0.055,
        y=0.985,
        ha="left",
        fontsize=21,
        weight="bold",
        color="#18262d",
    )
    fig.text(
        0.055,
        0.947,
        "3,700 opened PoolFire cells | truth-aware capacity diagnostic | independently recomputed",
        fontsize=11.5,
        color="#53636b",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
