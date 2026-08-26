from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_half_ray_spillover_attribution_v266_public_summary.json"
OUTPUT = ROOT / "assets/figures/blastnet_case19_half_ray_spillover_attribution_v266.png"


def add_stacked_labels(axis: plt.Axes, bars: tuple, values: tuple[int, int]) -> None:
    for bar, value in zip(bars, values, strict=True):
        if value:
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + bar.get_height() / 2,
                str(value),
                ha="center",
                va="center",
                color="white",
                fontsize=11,
                fontweight="bold",
            )


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    attribution = data["attribution"]

    cell_complement = attribution["cell_failure_classes"]["complement_only"]
    cell_both = attribution["cell_failure_classes"]["both_selected_and_complement"]
    anchors = ("p90 tail", "Worst tail")
    anchor_complement = np.asarray(
        [
            attribution["p90_violation_classes"]["complement_only"],
            attribution["worst_violation_classes"]["complement_only"],
        ]
    )
    anchor_both = np.asarray(
        [
            attribution["p90_violation_classes"]["both_selected_and_complement"],
            attribution["worst_violation_classes"]["both_selected_and_complement"],
        ]
    )

    fig = plt.figure(figsize=(18.5, 7.3), facecolor="#f4f6f5")
    grid = fig.add_gridspec(1, 2, width_ratios=(1.15, 1.0), wspace=0.22)
    left = fig.add_subplot(grid[0, 0])
    right = fig.add_subplot(grid[0, 1])

    bars_complement = left.bar(
        [0],
        [cell_complement],
        width=0.56,
        color="#2878b5",
        label="Unselected complement only",
    )
    bars_both = left.bar(
        [0],
        [cell_both],
        width=0.56,
        bottom=[cell_complement],
        color="#b64045",
        label="Both selected and complement",
    )
    add_stacked_labels(left, tuple(bars_complement), (cell_complement,))
    add_stacked_labels(left, tuple(bars_both), (cell_both,))
    left.text(0, cell_complement + cell_both + 7, "229 failed cells", ha="center", fontweight="bold")
    left.set_xticks([0], ["K16-matched failures"])
    left.set_ylim(0, 260)
    left.set_ylabel("Cell count")
    left.set_title("Failure is mixed, not pure spillover", loc="left", fontweight="bold")
    left.grid(axis="y", alpha=0.2)
    left.legend(frameon=False, loc="upper right")

    x = np.arange(len(anchors))
    right_complement = right.bar(x, anchor_complement, width=0.56, color="#2878b5")
    right_both = right.bar(x, anchor_both, width=0.56, bottom=anchor_complement, color="#b64045")
    add_stacked_labels(right, tuple(right_complement), tuple(int(v) for v in anchor_complement))
    add_stacked_labels(right, tuple(right_both), tuple(int(v) for v in anchor_both))
    right.set_xticks(x, anchors)
    right.set_ylim(0, 15.5)
    right.set_ylabel("Violating rigs out of 13")
    right.set_title("The severe tail is two-sided", loc="left", fontweight="bold")
    right.grid(axis="y", alpha=0.2)
    for index, total in enumerate(anchor_complement + anchor_both):
        right.text(index, total + 0.45, f"{total}/13", ha="center", fontweight="bold")

    fig.suptitle(
        "v266 | Exact residual partition attributes the v265.1 observation failure",
        x=0.055,
        y=0.985,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.018,
        "Selected residual non-increase: 429/429 | Independent checks: 18/18 | 0A+0AT | no new candidate, wall/RSS, or external claim",
        fontsize=10,
        color="#444444",
    )
    for axis in (left, right):
        axis.set_facecolor("#ffffff")
        axis.spines[["top", "right"]].set_visible(False)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
