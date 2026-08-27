from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_haar_irls_null_line_attribution_v272_public_summary.json"
OUTPUT = ROOT / "assets/figures/blastnet_case19_haar_irls_null_line_attribution_v272.png"


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    result = data["formal_result"]
    audit = data["independent_validation"]

    fig = plt.figure(figsize=(15.8, 7.2), facecolor="#f3f5f4")
    grid = fig.add_gridspec(1, 2, width_ratios=(0.9, 1.55), wspace=0.28)
    count_ax = fig.add_subplot(grid[0, 0])
    audit_ax = fig.add_subplot(grid[0, 1])

    labels = ("Observation-null\nlines", "Independent\nendpoint descent", "Formal inward\nendpoint descent")
    values = (
        result["observation_null_lines_within_limit"],
        result["independent_endpoints_with_strict_inward_descent"],
        result["formal_endpoints_with_strict_inward_descent"],
    )
    colors = ("#3279a8", "#5c8f68", "#c94f4f")
    bars = count_ax.bar(np.arange(3), values, color=colors, width=0.62)
    count_ax.axhline(13, color="#202629", linewidth=1.3, linestyle="--")
    count_ax.set_xticks(np.arange(3), labels)
    count_ax.set_ylim(0, 15.5)
    count_ax.set_ylabel("Rigs")
    count_ax.set_title("The segment result is one-sided", loc="left", fontweight="bold")
    for bar, value in zip(bars, values, strict=True):
        count_ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.28,
            f"{value}/13",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    independent_descent = np.asarray(
        (
            audit["minimum_independent_relative_descent"],
            audit["median_independent_relative_descent"],
            audit["maximum_independent_relative_descent"],
        ),
        dtype=np.float64,
    )
    positions = np.arange(3)
    audit_ax.bar(positions, independent_descent * 1e4, color=("#5c8f68", "#3279a8", "#d39b3a"), width=0.65)
    audit_ax.set_xticks(positions, ("Minimum", "Median", "Maximum"))
    audit_ax.set_ylabel("Independent-endpoint relative descent (x 1e-4)")
    audit_ax.set_title("All independent endpoints improve toward formal", loc="left", fontweight="bold")
    for index, value in enumerate(independent_descent):
        audit_ax.text(index, value * 1e4 + 0.04, f"{value:.3e}", ha="center", va="bottom", fontsize=10)

    fig.suptitle(
        "v272 | Observation-null endpoint comparison narrows, but does not close, the v271 root cause",
        x=0.045,
        y=0.985,
        ha="left",
        fontsize=16.2,
        fontweight="bold",
    )
    fig.text(
        0.045,
        0.018,
        "Independent validation 14/14 | maximum scalar difference 1.42e-14 | verdict MIXED_OR_NEAR_FLAT | fixed Haar-IRLS remains closed",
        fontsize=10,
        color="#40474b",
    )
    for axis in (count_ax, audit_ax):
        axis.set_facecolor("#ffffff")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
