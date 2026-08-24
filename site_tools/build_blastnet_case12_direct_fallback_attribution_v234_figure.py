#!/usr/bin/env python3
"""Build the redacted v234 Case 12 fallback-attribution figure."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets/figures/blastnet_case12_direct_fallback_attribution_v234.png"


def main() -> None:
    labels = ["Direct K11", "Zero K16", "dual-PRESS policy"]
    cells = np.asarray([598, 594, 595])
    rigs = np.asarray([13, 11, 11])
    colors = ["#146f66", "#315f93", "#a34f43"]

    plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6), constrained_layout=True)

    x = np.arange(len(labels))
    bars = axes[0].bar(x, cells, color=colors, width=0.62)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(570, 604)
    axes[0].set_ylabel("strict-safe cells (of 598)")
    axes[0].set_title("Fixed direct K11 passes every Case 12 cell")
    axes[0].grid(axis="y", alpha=0.2)
    for bar, cell_count, rig_count in zip(bars, cells, rigs, strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            cell_count + 0.7,
            f"{cell_count}/598\n{rig_count}/13 rigs",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    axes[1].axis("off")
    axes[1].set_title("What the fixed fallback changed", pad=12)
    rows = [
        ("Policy accepts", "437", "neutral"),
        ("Policy rejects", "161", "neutral"),
        ("Rejected direct cells already safe", "161 / 161", "pass"),
        ("Failures caused only by fallback", "3 / 3", "fail"),
        ("Unsafe K16 cell rescued by direct", "1", "pass"),
        ("Direct calls per cell", "12A + 11A^T", "pass"),
        ("Policy mean calls per cell", "13.077A + 12.346A^T", "fail"),
    ]
    y_pos = 0.92
    for title, value, state in rows:
        color = {"pass": "#146f66", "fail": "#a34f43"}.get(state, "#315f93")
        axes[1].text(0.02, y_pos, title, transform=axes[1].transAxes, fontweight="bold", color="#17252b")
        axes[1].text(0.98, y_pos, value, transform=axes[1].transAxes, ha="right", fontweight="bold", color=color)
        axes[1].plot(
            [0.02, 0.98],
            [y_pos - 0.065, y_pos - 0.065],
            transform=axes[1].transAxes,
            color="#cad6d2",
            linewidth=0.8,
        )
        y_pos -= 0.115
    axes[1].text(
        0.02,
        0.025,
        "Post-open attribution only\nThe next condition must be preregistered",
        transform=axes[1].transAxes,
        fontsize=10,
        fontweight="bold",
        color="#a34f43",
    )

    fig.suptitle(
        "v234 Case 12: the fixed fallback creates all three policy failures",
        fontsize=14,
        fontweight="bold",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    main()
