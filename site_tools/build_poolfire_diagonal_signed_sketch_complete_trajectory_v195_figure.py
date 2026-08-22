from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_diagonal_signed_sketch_complete_trajectory_v195_public_summary.json"
OUTPUT = ROOT / "assets/figures/poolfire_diagonal_signed_sketch_complete_trajectory_v195.png"


def main() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    primary = payload["primary_diagonal_k1"]
    full_dct = payload["controls"]["full_dct_k1"]

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
        width_ratios=(1.2, 1.0),
        left=0.075,
        right=0.975,
        top=0.79,
        bottom=0.18,
        wspace=0.25,
    )
    ax_cells = fig.add_subplot(grid[0, 0])
    ax_groups = fig.add_subplot(grid[0, 1])

    fig.text(
        0.075,
        0.94,
        "v195.2  Frozen diagonal correction fails the complete p22 trajectory",
        fontsize=27,
        fontweight="bold",
        color="#17201d",
    )
    fig.text(
        0.075,
        0.885,
        "101 frames x 13 calibrations per sensor arm; independent recomputation passes 27/27 checks.",
        fontsize=16,
        color="#4d5753",
    )

    methods = ["Diagonal primary", "Full-DCT control"]
    five = np.array(
        [
            primary["five_camera"]["strict_safe_cells"],
            full_dct["five_camera"]["strict_safe_cells"],
        ]
    )
    nine = np.array(
        [
            primary["all_nine"]["strict_safe_cells"],
            full_dct["all_nine"]["strict_safe_cells"],
        ]
    )
    x = np.arange(len(methods))
    width = 0.34
    bars_five = ax_cells.bar(x - width / 2, five, width, color="#2b7a78", label="Five cameras")
    bars_nine = ax_cells.bar(x + width / 2, nine, width, color="#e08b3e", label="All nine")
    ax_cells.axhline(1313, color="#b33a3a", linewidth=2.2, linestyle="--", label="Required 1313/1313")
    for bars in (bars_five, bars_nine):
        for bar in bars:
            ax_cells.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 22,
                f"{int(bar.get_height())}/1313",
                ha="center",
                va="bottom",
                fontsize=13,
                fontweight="bold",
            )
    ax_cells.set_xticks(x, methods)
    ax_cells.set_ylim(0, 1435)
    ax_cells.set_ylabel("Strict-safe K1 cells")
    ax_cells.set_title("Cellwise safety is incomplete", loc="left", pad=18)
    ax_cells.grid(axis="y", color="#d9ddda", linewidth=1, alpha=0.85)
    ax_cells.set_axisbelow(True)
    ax_cells.legend(loc="upper left", frameon=False, ncol=3, fontsize=12)
    ax_cells.spines[["top", "right"]].set_visible(False)

    five_groups = np.array(
        [
            primary["five_camera"]["complete_calibration_groups_passed"],
            full_dct["five_camera"]["complete_calibration_groups_passed"],
        ]
    )
    nine_groups = np.array(
        [
            primary["all_nine"]["complete_calibration_groups_passed"],
            full_dct["all_nine"]["complete_calibration_groups_passed"],
        ]
    )
    bars_five = ax_groups.bar(x - width / 2, five_groups, width, color="#2b7a78", label="Five cameras")
    bars_nine = ax_groups.bar(x + width / 2, nine_groups, width, color="#e08b3e", label="All nine")
    ax_groups.axhline(13, color="#b33a3a", linewidth=2.2, linestyle="--", label="Required 13/13")
    for bars in (bars_five, bars_nine):
        for bar in bars:
            ax_groups.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.28,
                f"{int(bar.get_height())}/13",
                ha="center",
                va="bottom",
                fontsize=13,
                fontweight="bold",
            )
    ax_groups.set_xticks(x, methods)
    ax_groups.set_ylim(0, 14.4)
    ax_groups.set_ylabel("Complete calibration groups")
    ax_groups.set_title("Primary failure closes the route", loc="left", pad=18)
    ax_groups.grid(axis="y", color="#d9ddda", linewidth=1, alpha=0.85)
    ax_groups.set_axisbelow(True)
    ax_groups.legend(loc="upper left", frameon=False, ncol=2, fontsize=12)
    ax_groups.spines[["top", "right"]].set_visible(False)

    fig.text(
        0.075,
        0.055,
        "The full-DCT control is stronger, but five-camera still misses one complete group; it is diagnostic, not a post-hoc replacement.",
        fontsize=14,
        color="#4d5753",
    )
    fig.text(
        0.075,
        0.02,
        "Fixed diagonal route closed. Fresh p14 validation, resource tests, neural training, and GPU use remain unauthorized.",
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
