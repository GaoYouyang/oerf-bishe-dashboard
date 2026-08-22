from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_qdeim_normal_metric_attribution_v191_1_public_summary.json"
OUTPUT = ROOT / "assets/figures/poolfire_qdeim_normal_metric_attribution_v191_1.png"


def main() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    outcomes = payload["setup_outcomes"]
    diagnostics = payload["normal_metric_diagnostics"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 16,
        "axes.titleweight": "bold",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
    })
    fig = plt.figure(figsize=(20, 10.34), facecolor="#f7f8f6")
    grid = fig.add_gridspec(1, 2, width_ratios=(0.92, 1.35), left=0.095, right=0.975,
                            top=0.79, bottom=0.18, wspace=0.24)
    ax_outcomes = fig.add_subplot(grid[0, 0])
    ax_metrics = fig.add_subplot(grid[0, 1])

    fig.text(0.095, 0.94, "v191.1  Why one fixed QDEIM subset fails",
             fontsize=29, fontweight="bold", color="#17201d")
    fig.text(
        0.095,
        0.885,
        "Same reported geometry can pass and fail across frames; discarded columns carry most of the missing response energy.",
        fontsize=17,
        color="#4d5753",
    )

    arms = ["Five cameras", "All nine"]
    y = np.arange(2)
    mixed = np.array([
        outcomes["five_camera"]["mixed_pass_fail_setups"],
        outcomes["all_nine"]["mixed_pass_fail_setups"],
    ])
    all_pass = np.array([
        outcomes["five_camera"]["all_pass_setups"],
        outcomes["all_nine"]["all_pass_setups"],
    ])
    all_fail = np.array([
        outcomes["five_camera"]["all_fail_setups"],
        outcomes["all_nine"]["all_fail_setups"],
    ])
    colors = {"mixed": "#d56a50", "pass": "#268a6a", "fail": "#595f66"}
    ax_outcomes.barh(y, mixed, color=colors["mixed"], height=0.48, label="Mixed frames")
    ax_outcomes.barh(y, all_pass, left=mixed, color=colors["pass"], height=0.48,
                     label="All four pass")
    ax_outcomes.barh(y, all_fail, left=mixed + all_pass, color=colors["fail"],
                     height=0.48, label="All four fail")
    for index in range(2):
        ax_outcomes.text(mixed[index] / 2, index, f"{mixed[index]} mixed", ha="center",
                         va="center", color="white", fontweight="bold", fontsize=17)
        ax_outcomes.text(mixed[index] + all_pass[index] / 2, index, str(all_pass[index]),
                         ha="center", va="center", color="white", fontweight="bold")
        ax_outcomes.text(mixed[index] + all_pass[index] + all_fail[index] / 2, index,
                         str(all_fail[index]), ha="center", va="center", color="white",
                         fontweight="bold")
    ax_outcomes.set_yticks(y, arms)
    ax_outcomes.invert_yaxis()
    ax_outcomes.set_xlim(0, 13)
    ax_outcomes.set_xticks(range(0, 14, 2))
    ax_outcomes.set_xlabel("Fixed sensor/calibration setups (13 per arm)")
    ax_outcomes.set_title("21 of 26 setups mix pass and fail", loc="left", pad=18)
    ax_outcomes.grid(axis="x", color="#d9ddda", linewidth=1, alpha=0.85)
    ax_outcomes.set_axisbelow(True)
    legend_handles, legend_labels = ax_outcomes.get_legend_handles_labels()
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower left",
        bbox_to_anchor=(0.09, 0.075),
        ncol=3,
        frameon=False,
        fontsize=13,
    )
    ax_outcomes.spines[["top", "right", "left"]].set_visible(False)

    metric_labels = ["Coordinate\ndiscrepancy", "Discarded response\nenergy",
                     "Selected directional\nweight"]
    five_values = np.array([
        diagnostics["coordinate_discrepancy_p50"]["five_camera"],
        diagnostics["discarded_delta_energy_fraction_p50"]["five_camera"],
        diagnostics["trace_normalized_directional_ratio_p50"]["five_camera"],
    ]) * 100
    nine_values = np.array([
        diagnostics["coordinate_discrepancy_p50"]["all_nine"],
        diagnostics["discarded_delta_energy_fraction_p50"]["all_nine"],
        diagnostics["trace_normalized_directional_ratio_p50"]["all_nine"],
    ]) * 100
    x = np.arange(len(metric_labels))
    width = 0.34
    bars_five = ax_metrics.bar(x - width / 2, five_values, width, color="#2b7a78",
                               label="Five cameras")
    bars_nine = ax_metrics.bar(x + width / 2, nine_values, width, color="#e08b3e",
                               label="All nine")
    for bars in (bars_five, bars_nine):
        for bar in bars:
            ax_metrics.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2.0,
                            f"{bar.get_height():.1f}%", ha="center", va="bottom",
                            fontsize=15, fontweight="bold")
    ax_metrics.set_xticks(x, metric_labels)
    ax_metrics.set_ylim(0, 105)
    ax_metrics.set_ylabel("Median magnitude")
    ax_metrics.set_title("The reduced metric omits the dominant response", loc="left", pad=18)
    ax_metrics.grid(axis="y", color="#d9ddda", linewidth=1, alpha=0.85)
    ax_metrics.set_axisbelow(True)
    ax_metrics.legend(loc="upper left", frameon=False)
    ax_metrics.spines[["top", "right"]].set_visible(False)

    fig.text(
        0.095,
        0.025,
        "Attribution only: selected solves are stationary for their own objective, but the full-DCT normal metric is not preserved. "
        "No predictor, speedup, external result, or real-BOST claim.",
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
