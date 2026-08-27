from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_haar_irls_reference_v271_public_summary.json"
OUTPUT = ROOT / "assets/figures/blastnet_case19_haar_irls_reference_v271.png"


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    audit = data["independent_validation"]
    limit = float(audit["cross_implementation_limit"])
    labels = ("Field", "Metric", "Objective", "Summary", "Residual", "Observation")
    values = np.asarray(
        (
            audit["maximum_field_relative_difference"],
            audit["maximum_metric_absolute_difference"],
            audit["maximum_objective_relative_difference"],
            audit["maximum_summary_absolute_difference"],
            audit["maximum_normalized_residual_difference"],
            audit["maximum_normalized_observation_difference"],
        ),
        dtype=np.float64,
    )
    ratios = values / limit
    passed = ratios <= 1.0

    fig = plt.figure(figsize=(15.8, 7.2), facecolor="#f3f5f4")
    grid = fig.add_gridspec(1, 2, width_ratios=(0.7, 1.65), wspace=0.28)
    count_ax = fig.add_subplot(grid[0, 0])
    audit_ax = fig.add_subplot(grid[0, 1])

    bars = count_ax.bar((0, 1), (13, 13), color=("#3279a8", "#5c8f68"), width=0.58)
    count_ax.axhline(13, color="#202629", linewidth=1.3, linestyle="--")
    count_ax.set_xticks((0, 1), ("Formal\nLSMR", "Independent\nLSQR"))
    count_ax.set_ylim(0, 15.4)
    count_ax.set_ylabel("Rigs passing every absolute metric")
    count_ax.set_title("Each path alone reaches 13/13", loc="left", fontweight="bold")
    for bar in bars:
        count_ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.25,
            "13/13",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    colors = np.where(passed, "#5c8f68", "#c94f4f")
    positions = np.arange(len(labels))
    audit_ax.bar(positions, np.maximum(ratios, 1e-13), color=colors, width=0.68)
    audit_ax.axhline(1.0, color="#202629", linewidth=1.4, linestyle="--", label="frozen agreement limit")
    audit_ax.set_yscale("log")
    audit_ax.set_ylim(1e-13, 1e3)
    audit_ax.set_xticks(positions, labels)
    audit_ax.set_ylabel("Formal-independent difference / frozen limit (log scale)")
    audit_ax.set_title("Field and metric agreement fail the preregistered audit", loc="left", fontweight="bold")
    audit_ax.legend(frameon=False, loc="upper right")
    for index, ratio in enumerate(ratios):
        label = f"{ratio:.1f}x" if ratio >= 0.01 else f"{ratio:.1e}x"
        audit_ax.text(index, max(ratio, 1e-13) * 1.35, label, ha="center", va="bottom", fontsize=9)

    fig.suptitle(
        "v271 | A 13/13 result is not reproducible until both numerical paths agree",
        x=0.045,
        y=0.985,
        ha="left",
        fontsize=16.5,
        fontweight="bold",
    )
    fig.text(
        0.045,
        0.018,
        "Formal 18/18 | independent 24/31 | both paths 13/13 absolute | authoritative verdict INCONCLUSIVE | fixed reference closed",
        fontsize=10,
        color="#40474b",
    )
    for axis in (count_ax, audit_ax):
        axis.set_facecolor("#ffffff")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2, which="both")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
