from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_observation_adaptive_qdeim_capacity_v192_public_summary.json"
OUTPUT = ROOT / "assets/figures/poolfire_observation_adaptive_qdeim_capacity_v192.png"


def main() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    safe = payload["strict_safe_cells"]
    failures = payload["primary_failure_modes"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 16,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
        }
    )
    fig = plt.figure(figsize=(20, 10.34), facecolor="#f7f8f6")
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=(1.3, 1.0),
        left=0.085,
        right=0.975,
        top=0.79,
        bottom=0.18,
        wspace=0.24,
    )
    ax_safe = fig.add_subplot(grid[0, 0])
    ax_fail = fig.add_subplot(grid[0, 1])

    fig.text(
        0.085,
        0.94,
        "v192  Observation-adaptive QDEIM improves, but still fails",
        fontsize=28,
        fontweight="bold",
        color="#17201d",
    )
    fig.text(
        0.085,
        0.885,
        "Normal-defect contribution rescues cells at the same 1280-coordinate budget; the strict gate remains 52/52 in both arms.",
        fontsize=17,
        color="#4d5753",
    )

    methods = ["Fixed geometry\nv190", "Magnitude\ncontrol", "Normal contribution\nprimary"]
    five = np.array(
        [
            safe["fixed_geometry_qdeim_v190"]["five_camera"],
            safe["observation_magnitude_control"]["five_camera"],
            safe["normal_contribution_primary"]["five_camera"],
        ]
    )
    nine = np.array(
        [
            safe["fixed_geometry_qdeim_v190"]["all_nine"],
            safe["observation_magnitude_control"]["all_nine"],
            safe["normal_contribution_primary"]["all_nine"],
        ]
    )
    x = np.arange(len(methods))
    width = 0.34
    bars_five = ax_safe.bar(x - width / 2, five, width, color="#2b7a78", label="Five cameras")
    bars_nine = ax_safe.bar(x + width / 2, nine, width, color="#e08b3e", label="All nine")
    ax_safe.axhline(52, color="#b33a3a", linewidth=2.2, linestyle="--", label="Required 52/52")
    for bars in (bars_five, bars_nine):
        for bar in bars:
            ax_safe.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.0,
                f"{int(bar.get_height())}/52",
                ha="center",
                va="bottom",
                fontsize=15,
                fontweight="bold",
            )
    ax_safe.set_xticks(x, methods)
    ax_safe.set_ylim(0, 57)
    ax_safe.set_ylabel("Strict-safe cells")
    ax_safe.set_title("Partial recovery, not complete capacity", loc="left", pad=18)
    ax_safe.grid(axis="y", color="#d9ddda", linewidth=1, alpha=0.85)
    ax_safe.set_axisbelow(True)
    ax_safe.legend(loc="upper left", frameon=False, ncol=3, fontsize=12)
    ax_safe.spines[["top", "right"]].set_visible(False)

    metric_labels = ["Field", "Gradient", "Observation"]
    five_failures = np.array(
        [
            failures["five_camera"]["field_failures"],
            failures["five_camera"]["gradient_failures"],
            failures["five_camera"]["observation_failures"],
        ]
    )
    nine_failures = np.array(
        [
            failures["all_nine"]["field_failures"],
            failures["all_nine"]["gradient_failures"],
            failures["all_nine"]["observation_failures"],
        ]
    )
    y = np.arange(len(metric_labels))
    height = 0.34
    ax_fail.barh(y - height / 2, five_failures, height, color="#2b7a78", label="Five cameras")
    ax_fail.barh(y + height / 2, nine_failures, height, color="#e08b3e", label="All nine")
    for row, values in enumerate((five_failures, nine_failures)):
        offset = -height / 2 if row == 0 else height / 2
        for metric_index, value in enumerate(values):
            ax_fail.text(value + 0.25, metric_index + offset, str(int(value)), va="center", fontweight="bold")
    ax_fail.set_yticks(y, metric_labels)
    ax_fail.invert_yaxis()
    ax_fail.set_xlim(0, 13.5)
    ax_fail.set_xlabel("Metric violations among failed cells")
    ax_fail.set_title("The remaining failure differs by sensor count", loc="left", pad=18)
    ax_fail.grid(axis="x", color="#d9ddda", linewidth=1, alpha=0.85)
    ax_fail.set_axisbelow(True)
    ax_fail.legend(loc="upper right", frameon=False)
    ax_fail.spines[["top", "right", "left"]].set_visible(False)

    fig.text(
        0.085,
        0.055,
        "Five-camera failures are gradient-dominated; all-nine failures are observation-only. Metric counts can overlap within a failed cell.",
        fontsize=14,
        color="#4d5753",
    )
    fig.text(
        0.085,
        0.02,
        "Post-open capacity diagnostic only: no predictor, exact-call reduction, speedup, external result, or real-BOST claim.",
        fontsize=14,
        color="#4d5753",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    with Image.open(OUTPUT) as image:
        image.convert("RGB").save(OUTPUT)


if __name__ == "__main__":
    main()
