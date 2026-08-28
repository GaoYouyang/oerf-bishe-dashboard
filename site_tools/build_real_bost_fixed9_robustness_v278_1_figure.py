from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/real_bost_fixed9_robustness_v278_1_public_summary.json"
OUTPUT = ROOT / "assets/figures/real_bost_fixed9_robustness_v278_1.png"


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    conditions = data["conditions"]
    labels = ("Clean", "Observation\nnoise", "Pose", "Intrinsic", "Combined")
    values = np.asarray(
        [data["reference_maximum_normalized_gate_burden"][condition] for condition in conditions],
        dtype=np.float64,
    )
    counts = np.asarray(
        [data["reference_pass_counts_by_condition"][condition] for condition in conditions],
        dtype=np.float64,
    )

    fig = plt.figure(figsize=(17.6, 7.5), facecolor="#f3f5f4")
    grid = fig.add_gridspec(1, 2, width_ratios=(1.45, 0.85), wspace=0.28)
    heat_ax = fig.add_subplot(grid[0, 0])
    count_ax = fig.add_subplot(grid[0, 1])

    image = heat_ax.imshow(values, cmap="RdYlGn_r", vmin=0.85, vmax=1.85, aspect="auto")
    heat_ax.set_xticks(np.arange(4), ("t=0", "t=.25", "t=.75", "t=1"))
    heat_ax.set_yticks(np.arange(5), labels)
    heat_ax.set_title("K16 reference: maximum normalized gate burden", loc="left", fontweight="bold")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            heat_ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="#ffffff" if value > 1.35 else "#17252b",
                fontweight="bold",
            )
    colorbar = fig.colorbar(image, ax=heat_ax, fraction=0.045, pad=0.035)
    colorbar.set_label("worst gate ratio; <= 1 passes")

    colors = ["#258779" if count == 4 else "#c9564a" for count in counts]
    bars = count_ax.bar(np.arange(5), counts, color=colors, width=0.68)
    count_ax.axhline(4, color="#202629", linewidth=1.3, linestyle="--")
    count_ax.set_xticks(np.arange(5), labels)
    count_ax.set_ylim(0, 4.8)
    count_ax.set_ylabel("Time strata passing absolute reference gates")
    count_ax.set_title("Reference adequacy stops at 12/20", loc="left", fontweight="bold")
    count_ax.grid(axis="y", alpha=0.2)
    for bar, count in zip(bars, counts, strict=True):
        count_ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.12,
            f"{int(count)}/4",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    fig.suptitle(
        "v278.1 | Existing 3D fields reach a fixed-nine-camera robustness gate",
        x=0.045,
        y=0.985,
        ha="left",
        fontsize=16.5,
        fontweight="bold",
    )
    fig.text(
        0.045,
        0.018,
        "2,340 controlled virtual-BOS cells | independent 21/21 | pose and combined strata invalidate K16 | candidate diagnostics not adjudicated",
        fontsize=10,
        color="#40474b",
    )
    for axis in (heat_ax, count_ax):
        axis.set_facecolor("#ffffff")
        axis.spines[["top", "right"]].set_visible(False)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
