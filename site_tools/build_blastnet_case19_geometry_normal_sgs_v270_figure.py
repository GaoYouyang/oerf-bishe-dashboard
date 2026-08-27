from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_geometry_normal_sgs_v270_public_summary.json"
OUTPUT = ROOT / "assets/figures/blastnet_case19_geometry_normal_sgs_v270.png"


def _label_counts(axis: plt.Axes, bars: plt.Container) -> None:
    for bar in bars:
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.22,
            f"{int(round(bar.get_height()))}/13",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    diagnostic = data["diagnostic_results"]
    reference = data["reference_adequacy"]

    fig = plt.figure(figsize=(17.8, 7.4), facecolor="#f3f5f4")
    grid = fig.add_gridspec(1, 2, width_ratios=(0.92, 1.42), wspace=0.28)
    count_ax = fig.add_subplot(grid[0, 0])
    ratio_ax = fig.add_subplot(grid[0, 1])

    labels = ("Normal-SGS\nK14", "Jacobi\nK14", "CGLS\nK14", "Jacobi\nK16 ref")
    counts = (
        diagnostic["primary_absolute_pass_cells"],
        diagnostic["geometry_jacobi_k14_absolute_pass_cells"],
        diagnostic["unpreconditioned_cgls_k14_absolute_pass_cells"],
        reference["absolute_pass_cells"],
    )
    colors = ("#c94f4f", "#d08a2f", "#6f7d84", "#3279a8")
    bars = count_ax.bar(np.arange(4), counts, color=colors, width=0.68)
    _label_counts(count_ax, bars)
    count_ax.axhline(13, color="#202629", linewidth=1.4, linestyle="--")
    count_ax.set_xticks(np.arange(4), labels)
    count_ax.set_ylim(0, 15.2)
    count_ax.set_ylabel("Rigs passing every absolute metric")
    count_ax.set_title("Reference stops at 12/13", loc="left", fontweight="bold")
    count_ax.text(
        0.02,
        0.91,
        "one K16 interior-gradient value\n0.758223 > frozen 0.750000",
        transform=count_ax.transAxes,
        fontsize=10,
        color="#9b3b3b",
        va="top",
        fontweight="bold",
    )
    count_ax.grid(axis="y", alpha=0.2)

    limits = np.asarray((0.5, 0.75, 0.75, 0.2), dtype=np.float64)
    primary = np.asarray(diagnostic["primary_metric_worst"], dtype=np.float64) / limits
    ref = np.asarray(reference["worst"], dtype=np.float64) / limits
    positions = np.arange(4)
    ratio_ax.bar(positions - 0.19, primary, width=0.38, color="#c94f4f", label="Normal-SGS K14")
    ratio_ax.bar(positions + 0.19, ref, width=0.38, color="#3279a8", label="Jacobi K16 reference")
    ratio_ax.axhline(1.0, color="#202629", linewidth=1.4, linestyle="--", label="absolute limit")
    ratio_ax.set_yscale("log")
    ratio_ax.set_xticks(positions, ("Field", "Full\ngradient", "Interior\ngradient", "Observation"))
    ratio_ax.set_ylim(0.2, 14.0)
    ratio_ax.set_ylabel("Worst error / frozen absolute limit (log scale)")
    ratio_ax.set_title("SGS diagnostics are poor, but reference inadequacy adjudicates first", loc="left", fontweight="bold")
    ratio_ax.legend(frameon=False, fontsize=9, loc="upper right")
    ratio_ax.grid(axis="y", alpha=0.2, which="both")

    fig.suptitle(
        "v270 | Exact-normal SGS control is independently valid but scientifically inconclusive",
        x=0.045,
        y=0.985,
        ha="left",
        fontsize=16.5,
        fontweight="bold",
    )
    fig.text(
        0.045,
        0.018,
        "Formal 21/21 | independent 32/32 | K16 reference 12/13 | primary 0/13 diagnostic only | no call, wall/RSS, or algorithm claim",
        fontsize=10,
        color="#40474b",
    )
    for axis in (count_ax, ratio_ax):
        axis.set_facecolor("#ffffff")
        axis.spines[["top", "right"]].set_visible(False)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
