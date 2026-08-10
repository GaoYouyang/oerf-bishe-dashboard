from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_pair_resolved_depth_cost_capacity_v140_public_summary.json"
OUTPUT = ROOT / "assets/figures/poolfire_pair_resolved_depth_cost_capacity_v140.png"


def build_figure() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    stage = data["stage_a"]
    independent = data["independent_recomputation"]
    audit = data["numerical_audit"]

    fig = plt.figure(figsize=(16, 9), facecolor="#f3f7f5")
    grid = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28)

    progression = fig.add_subplot(grid[0, 0])
    labels = ["v139\nall opened cells", "v140 Stage A\nfixed hard set"]
    passed = [3549, stage["passed_count"]]
    totals = [3700, stage["evaluated_count"]]
    bars = progression.bar(np.arange(2), passed, color=["#78978d", "#c76a52"], width=0.62)
    progression.set_xticks(np.arange(2), labels)
    progression.set_ylabel("Passing cells within each evaluated roster")
    progression.set_ylim(0, 4000)
    progression.set_title("A. v140 closes the fixed 151-cell hard set", loc="left", fontsize=12.8, weight="bold")
    progression.grid(axis="y", alpha=0.18)
    for bar, value, total in zip(bars, passed, totals, strict=True):
        progression.text(
            bar.get_x() + bar.get_width() / 2,
            value + 65,
            f"{value}/{total}",
            ha="center",
            weight="bold",
        )
    progression.text(
        0.5,
        330,
        "Rosters differ: Stage A is not a 3,700-cell result",
        ha="center",
        fontsize=9.6,
        color="#8f4036",
    )

    sources = fig.add_subplot(grid[0, 1])
    histogram = data["selected_source_histogram"]
    source_labels = ["projection\nonly", "w=1", "w=4", "w=16", "w=64", "w=256+", "cheap LS"]
    source_values = [
        histogram["pair_depth_projection_only"],
        histogram["pair_depth_projection_weight_1"],
        histogram["pair_depth_projection_weight_4"],
        histogram["pair_depth_projection_weight_16"],
        histogram["pair_depth_projection_weight_64"],
        histogram["pair_depth_projection_weight_256"] + histogram["pair_depth_projection_weight_1024"],
        histogram["pair_depth_residual_joint_ls"],
    ]
    bars = sources.bar(np.arange(len(source_labels)), source_values, color=["#39796a"] + ["#78a99a"] * 4 + ["#9ba7aa", "#c25b49"])
    sources.set_xticks(np.arange(len(source_labels)), source_labels)
    sources.set_ylabel("Selected hard-set cells")
    sources.set_ylim(0, 145)
    sources.set_title("B. Pair-depth projection carries nearly all rescues", loc="left", fontsize=12.8, weight="bold")
    sources.grid(axis="y", alpha=0.18)
    for bar, value in zip(bars, source_values, strict=True):
        sources.text(bar.get_x() + bar.get_width() / 2, value + 3, str(value), ha="center", fontsize=9.4, weight="bold")

    ladder = fig.add_subplot(grid[1, 0])
    ladder_labels = ["Formal\nStage A", "Independent\nrecompute", "Cheap\ncontrol", "Stage B\ncompleted"]
    ladder_values = [151, 151, 0, 0]
    bars = ladder.bar(np.arange(4), ladder_values, color=["#39796a", "#4f8c7c", "#c25b49", "#9ba7aa"], width=0.65)
    ladder.set_xticks(np.arange(4), ladder_labels)
    ladder.set_ylabel("Passing or completed items")
    ladder.set_ylim(0, 175)
    ladder.set_title("C. Independent Stage A passes; Stage B remains pending", loc="left", fontsize=12.8, weight="bold")
    ladder.grid(axis="y", alpha=0.18)
    annotations = ["151/151", "151/151", "0/151", "0/2199"]
    for bar, value, label in zip(bars, ladder_values, annotations, strict=True):
        ladder.text(bar.get_x() + bar.get_width() / 2, value + 6, label, ha="center", weight="bold")

    audit_ax = fig.add_subplot(grid[1, 1])
    audit_labels = ["Scientific\nmetric", "Condition\nrelative", "Coefficient\nabsolute"]
    observed = [
        independent["maximum_selected_metric_difference"],
        independent["maximum_condition_relative_difference"],
        independent["maximum_coefficient_difference"],
    ]
    limits = [
        audit["science_metric_absolute_tolerance"],
        audit["condition_relative_tolerance"],
        audit["coefficient_absolute_tolerance"],
    ]
    x = np.arange(3)
    width = 0.34
    audit_ax.bar(x - width / 2, observed, width, label="Observed difference", color="#39796a")
    audit_ax.bar(x + width / 2, limits, width, label="Typed tolerance", color="#d39a4b")
    audit_ax.set_yscale("log")
    audit_ax.set_xticks(x, audit_labels)
    audit_ax.set_ylabel("Difference / tolerance (log scale)")
    audit_ax.set_title("D. Post-open typed audit passes a fresh full recomputation", loc="left", fontsize=12.8, weight="bold")
    audit_ax.grid(axis="y", which="both", alpha=0.18)
    audit_ax.legend(frameon=False, loc="upper left")

    fig.suptitle(
        "v140 Stage A: pair-resolved depth cost closes the 151-cell hard set",
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
        "truth-aware mechanism capacity only | Stage B 2,199-cell tail pending | predictor/GPU unauthorized | algorithm_breakthrough=false",
        fontsize=10.5,
        color="#53636b",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
