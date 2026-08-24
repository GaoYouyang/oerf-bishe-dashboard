#!/usr/bin/env python3
"""Build the public v228 retrospective dual-PRESS union figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT / "docs/blastnet_case2_case5_low64_dual_press_union_v228_public_summary.json"
)
OUTPUT = ROOT / "assets/figures/blastnet_case2_case5_low64_dual_press_union_v228.png"


def main() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    case5 = payload["parent_comparison"]["case5"]
    case2 = payload["parent_comparison"]["case2"]
    raw = np.asarray(case5["per_rig_raw"], dtype=float)
    studentized = np.asarray(case5["per_rig_studentized"], dtype=float)
    union = np.asarray(case5["per_rig_union"], dtype=float)

    plt.rcParams.update({"font.size": 12, "axes.titleweight": "bold"})
    fig = plt.figure(figsize=(18, 8.5), facecolor="#f6f7f4")
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.0, 1.75),
        height_ratios=(1, 1),
        left=0.045,
        right=0.985,
        bottom=0.12,
        top=0.88,
        wspace=0.12,
        hspace=0.31,
    )

    ax_accept = fig.add_subplot(grid[0, 0])
    labels = ["Case 5\nraw", "Case 5\nwhitened", "Case 5\nfixed OR", "Case 2\nfixed OR"]
    values = [
        case5["raw_accepted"],
        case5["studentized_accepted"],
        case5["union_accepted"],
        case2["union_accepted_safe"],
    ]
    totals = [546, 546, 546, 518]
    colors = ["#607a8a", "#8b6f9c", "#2d8a6d", "#2d8a6d"]
    bars = ax_accept.bar(
        labels, np.asarray(values) / np.asarray(totals) * 100, color=colors, width=0.68
    )
    ax_accept.set_ylabel("Safe accept fraction (%)")
    ax_accept.set_ylim(0, 70)
    ax_accept.grid(axis="y", alpha=0.25)
    for bar, value, total in zip(bars, values, totals, strict=True):
        ax_accept.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{value}/{total}",
            ha="center",
            fontsize=11,
        )
    ax_accept.set_title("Fixed OR recovers safe support")

    ax_parts = fig.add_subplot(grid[1, 0])
    part_labels = ["Both", "Raw only", "Whitened only"]
    c5_parts = [
        case5["both_accept"],
        case5["raw_only_accept"],
        case5["studentized_only_accept"],
    ]
    c2_parts = [
        case2["both_accept"],
        case2["raw_only_accept"],
        case2["studentized_only_accept"],
    ]
    x = np.arange(3)
    width = 0.36
    ax_parts.bar(x - width / 2, c5_parts, width, label="Case 5", color="#c95d45")
    ax_parts.bar(x + width / 2, c2_parts, width, label="Case 2", color="#3975a8")
    ax_parts.set_xticks(x, part_labels)
    ax_parts.set_ylabel("Accepted cells")
    ax_parts.set_yscale("symlog", linthresh=20)
    ax_parts.grid(axis="y", alpha=0.25)
    ax_parts.legend(frameon=False)
    ax_parts.set_title("The two scores contribute distinct safe cells")
    for index, (left, right) in enumerate(zip(c5_parts, c2_parts, strict=True)):
        ax_parts.text(
            index - width / 2,
            left + max(1, left * 0.08),
            str(left),
            ha="center",
            fontsize=10,
        )
        ax_parts.text(
            index + width / 2,
            right + max(1, right * 0.08),
            str(right),
            ha="center",
            fontsize=10,
        )

    ax_rigs = fig.add_subplot(grid[:, 1])
    rigs = np.arange(13)
    ax_rigs.plot(rigs, raw, "o-", color="#607a8a", linewidth=2, label="v226 raw")
    ax_rigs.plot(
        rigs, studentized, "s-", color="#8b6f9c", linewidth=2, label="v227 whitened"
    )
    ax_rigs.plot(
        rigs, union, "D-", color="#2d8a6d", linewidth=2.8, label="v228 fixed OR"
    )
    ax_rigs.axhline(
        5, color="#c44f3d", linestyle="--", linewidth=1.8, label="Frozen minimum: 5/42"
    )
    ax_rigs.fill_between(rigs, 0, 5, color="#c44f3d", alpha=0.08)
    ax_rigs.set_xticks(rigs)
    ax_rigs.set_xlabel("Held-out Case 5 rig")
    ax_rigs.set_ylabel("Accepted safe cells out of 42")
    ax_rigs.set_ylim(0, 27)
    ax_rigs.grid(alpha=0.24)
    ax_rigs.legend(frameon=False, ncol=2, loc="upper left")
    ax_rigs.set_title("Complementarity repairs both former one-frame failures")
    for rig in (4, 11):
        ax_rigs.annotate(
            f"rig {rig}: {int(raw[rig])}/{int(studentized[rig])}/{int(union[rig])}",
            xy=(rig, union[rig]),
            xytext=(rig - 1.8 if rig > 6 else rig + 0.5, union[rig] + 5.5),
            arrowprops={"arrowstyle": "->", "color": "#33434b"},
            fontsize=10,
        )

    fig.suptitle(
        "v228 retrospective diagnostic: raw OR whitened PRESS",
        fontsize=20,
        fontweight="bold",
        y=0.97,
    )
    fig.text(
        0.5,
        0.035,
        "Opened Cases 2/5 only. Zero unsafe accepts, but not a preregistered deployment result; algorithm_breakthrough=false.",
        ha="center",
        fontsize=11,
        color="#4f5a5f",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
