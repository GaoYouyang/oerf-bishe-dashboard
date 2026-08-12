from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / (
    "docs/poolfire_pair_resolved_depth_cost_stage_b_v140_4_public_summary.json"
)
OUTPUT = ROOT / (
    "assets/figures/poolfire_pair_resolved_depth_cost_stage_b_v140_4.png"
)


def build_figure() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    stage_b = data["stage_b"]
    independent = data["independent_recomputation"]

    fig = plt.figure(figsize=(16, 9), facecolor="#f3f7f5")
    grid = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28)

    progression = fig.add_subplot(grid[0, 0])
    labels = ["v139\nparent", "v140\nStage A", "v140.4\nfull roster"]
    values = [3549 / 3700 * 100, 151 / 151 * 100, 3700 / 3700 * 100]
    bars = progression.bar(
        np.arange(3), values, color=["#78978d", "#d39a4b", "#39796a"], width=0.64
    )
    progression.set_xticks(np.arange(3), labels)
    progression.set_ylabel("Pass rate within each evaluated roster (%)")
    progression.set_ylim(0, 108)
    progression.set_title(
        "A. The full 3,700-cell capacity gate now closes",
        loc="left",
        fontsize=12.8,
        weight="bold",
    )
    progression.grid(axis="y", alpha=0.18)
    annotations = ["3549/3700", "151/151", "3700/3700"]
    for bar, label in zip(bars, annotations, strict=True):
        progression.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2.2,
            label,
            ha="center",
            weight="bold",
        )

    cameras = fig.add_subplot(grid[0, 1])
    camera_labels = ["5", "7", "9", "12"]
    camera_totals = [
        stage_b["active_tail_camera_counts"][label] for label in camera_labels
    ]
    bars = cameras.bar(
        np.arange(4), camera_totals, color=["#c76a52", "#d39a4b", "#78a99a", "#39796a"]
    )
    cameras.set_xticks(np.arange(4), camera_labels)
    cameras.set_xlabel("Active cameras")
    cameras.set_ylabel("Stage-B active-tail cells")
    cameras.set_title(
        "B. Every variable-camera stratum passes",
        loc="left",
        fontsize=12.8,
        weight="bold",
    )
    cameras.grid(axis="y", alpha=0.18)
    for bar, total in zip(bars, camera_totals, strict=True):
        cameras.text(
            bar.get_x() + bar.get_width() / 2,
            total + 20,
            f"{total}/{total}",
            ha="center",
            weight="bold",
        )

    target = fig.add_subplot(grid[1, 0])
    target_labels = ["Fixed primary\nactive tail", "Fixed primary\nfull roster", "Complete\ntrajectories", "Cheap LS\nactive tail"]
    target_values = [100.0, 100.0, 100.0, 0.0]
    bars = target.bar(
        np.arange(4), target_values, color=["#39796a", "#4f8c7c", "#78a99a", "#c25b49"]
    )
    target.set_xticks(np.arange(4), target_labels)
    target.set_ylabel("Gate pass rate (%)")
    target.set_ylim(0, 112)
    target.set_title(
        "C. One preregistered truth-free target is sufficient",
        loc="left",
        fontsize=12.8,
        weight="bold",
    )
    target.grid(axis="y", alpha=0.18)
    target_annotations = ["2199/2199", "3700/3700", "5/5", "0/2199"]
    for bar, label in zip(bars, target_annotations, strict=True):
        target.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            label,
            ha="center",
            weight="bold",
        )

    audit = fig.add_subplot(grid[1, 1])
    audit_labels = ["Physical\nfield", "Physical\nprojection", "Fixed-target\nsummary"]
    observed = [
        independent["field_image_normalized_difference"],
        independent["projection_image_normalized_difference"],
        independent["maximum_fixed_candidate_summary_difference"],
    ]
    limits = [
        independent["physical_image_tolerance"],
        independent["physical_image_tolerance"],
        1e-6,
    ]
    x = np.arange(3)
    width = 0.34
    audit.bar(x - width / 2, observed, width, label="Observed difference", color="#39796a")
    audit.bar(x + width / 2, limits, width, label="Frozen bound", color="#d39a4b")
    audit.set_yscale("log")
    audit.set_xticks(x, audit_labels)
    audit.set_ylabel("Difference / bound (log scale)")
    audit.set_title(
        "D. Independent physical reconstruction stays inside bounds",
        loc="left",
        fontsize=12.8,
        weight="bold",
    )
    audit.grid(axis="y", which="both", alpha=0.18)
    audit.legend(frameon=False, loc="upper left")

    fig.suptitle(
        "v140.4: full-roster pair-resolved depth-cost capacity passes",
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
        "fixed-target capacity only | complete 3,700-cell teacher still required before fitting | algorithm_breakthrough=false",
        fontsize=10.5,
        color="#53636b",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
