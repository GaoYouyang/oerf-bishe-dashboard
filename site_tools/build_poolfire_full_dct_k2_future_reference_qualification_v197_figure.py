from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT
    / "docs/poolfire_full_dct_k2_future_reference_qualification_v197_public_summary.json"
)
OUTPUT = (
    ROOT / "assets/figures/poolfire_full_dct_k2_future_reference_qualification_v197.png"
)


def main() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    qualification = payload["qualification"]
    coverage = np.asarray(
        [
            qualification["strict_cells_safe"] / qualification["strict_cells_total"],
            qualification["complete_groups_passed"]
            / qualification["complete_groups_total"],
            qualification["call_rows_matching"] / qualification["call_rows_total"],
        ],
        dtype=np.float64,
    )
    margins = np.asarray(
        [
            qualification["minimum_strict_cell_margin"],
            qualification["minimum_complete_group_p90_margin"],
            qualification["minimum_complete_group_worst_margin"],
        ],
        dtype=np.float64,
    )
    labels = ["Strict cells", "Complete groups", "Call rows"]
    margin_labels = ["Cell gate", "Group p90", "Group worst"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
        }
    )
    fig = plt.figure(figsize=(18, 10), facecolor="#f7f8f6")
    grid = fig.add_gridspec(
        1, 2, left=0.12, right=0.97, top=0.76, bottom=0.19, wspace=0.25
    )
    ax_coverage = fig.add_subplot(grid[0, 0])
    ax_margin = fig.add_subplot(grid[0, 1])

    fig.text(
        0.075,
        0.93,
        "v197  Future-only full-DCT K2 reference qualification",
        fontsize=24,
        fontweight="bold",
        color="#17201d",
    )
    fig.text(
        0.075,
        0.87,
        "Already-opened p22 development roster; no candidate result, p14 validation, or test was opened.",
        fontsize=15,
        color="#4d5753",
    )
    fig.text(
        0.075,
        0.825,
        "Independent recomputation agrees exactly with formal qualification (maximum numeric difference 0).",
        fontsize=15,
        color="#4d5753",
    )

    y = np.arange(len(labels))
    bars = ax_coverage.barh(y, coverage * 100.0, color="#287c78", height=0.56)
    ax_coverage.axvline(100.0, color="#aa3434", linewidth=2, linestyle="--")
    for bar, value in zip(bars, coverage, strict=True):
        ax_coverage.text(
            98.0,
            bar.get_y() + bar.get_height() / 2,
            f"{value * 100:.1f}%",
            ha="right",
            va="center",
            color="white",
            fontweight="bold",
        )
    ax_coverage.set_yticks(y, labels)
    ax_coverage.set_xlim(0, 105)
    ax_coverage.set_xlabel("Qualified coverage (%)")
    ax_coverage.set_title("Every discrete gate is complete", loc="left", pad=16)

    margin_bars = ax_margin.barh(
        y, margins, color=("#315f93", "#8a651b", "#a34f43"), height=0.56
    )
    ax_margin.axvline(0.0, color="#222222", linewidth=1.5)
    for bar, value in zip(margin_bars, margins, strict=True):
        ax_margin.text(
            value + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"+{value:.6f}",
            ha="left",
            va="center",
            fontweight="bold",
        )
    ax_margin.set_yticks(y, margin_labels)
    ax_margin.set_xlim(0, max(margins) * 1.35)
    ax_margin.set_xlabel("Minimum positive margin to frozen limit")
    ax_margin.set_title("No accepted value sits on the threshold", loc="left", pad=16)

    for axis in (ax_coverage, ax_margin):
        axis.grid(axis="x", color="#d9ddda", linewidth=1, alpha=0.85)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.invert_yaxis()

    fig.text(
        0.075,
        0.105,
        "PASS for future contracts only: 2626/2626 cells, 26/26 groups, and 2626/2626 [3A, 2A^T] call rows.",
        fontsize=15,
        fontweight="bold",
        color="#146f66",
    )
    fig.text(
        0.075,
        0.06,
        "This does not revise v196 and is not a candidate algorithm, compact initializer, call reduction, speedup, or breakthrough.",
        fontsize=13.5,
        color="#4d5753",
    )
    fig.text(
        0.075,
        0.025,
        "Next: freeze one physically distinct candidate before reading results; p14, tests, wall/RSS, training, and GPU remain closed.",
        fontsize=13.5,
        color="#4d5753",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    with Image.open(OUTPUT) as image:
        image.convert("RGB").save(OUTPUT)


if __name__ == "__main__":
    main()
