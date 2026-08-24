#!/usr/bin/env python3
"""Build the redacted v230.1 Case 12 reference-adequacy figure."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets/figures/blastnet_case12_low64_nested_dual_press_external_v230_1.png"


def main() -> None:
    rigs = np.arange(13)
    p90 = np.asarray(
        [
            0.7417735811,
            0.6781181866,
            0.6932131014,
            0.6775240033,
            0.6700045488,
            0.7220446822,
            0.7024255497,
            0.6701573084,
            0.6694012919,
            0.7345700217,
            0.6883817229,
            0.7117305287,
            0.7416843821,
        ]
    )
    worst = np.asarray(
        [
            0.7531661853,
            0.6947551658,
            0.7048123615,
            0.6928811116,
            0.6934287833,
            0.7421409801,
            0.7098967583,
            0.6823921241,
            0.6792194362,
            0.7424879176,
            0.7004917706,
            0.7201123378,
            0.7546212459,
        ]
    )
    failure_labels = ["rig 0 / f11", "rig 0 / f42", "rig 12 / f11", "rig 12 / f42"]
    failure_values = np.asarray([0.7531661853, 0.7517267964, 0.7546212459, 0.7528645713])

    plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), constrained_layout=True)

    width = 0.36
    axes[0].bar(rigs - width / 2, p90, width, label="p90-higher", color="#3d7ea6")
    colors = np.where(worst > 0.75, "#bd493a", "#3f826d")
    axes[0].bar(rigs + width / 2, worst, width, label="worst cell", color=colors)
    axes[0].axhline(0.75, color="#252525", linestyle="--", linewidth=1.5, label="strict cell gate = 0.75")
    axes[0].set_title("Interior-gradient reference adequacy by rig")
    axes[0].set_xlabel("Case 12 rig index")
    axes[0].set_ylabel("relative error")
    axes[0].set_xticks(rigs)
    axes[0].set_ylim(0.64, 0.765)
    axes[0].grid(axis="y", alpha=0.22)
    axes[0].legend(fontsize=9, loc="lower left")
    axes[0].annotate("2 failing rigs", xy=(12.18, worst[-1]), xytext=(8.0, 0.759), arrowprops={"arrowstyle": "->", "color": "#bd493a"}, color="#8f2f25", fontweight="bold")

    bars = axes[1].barh(np.arange(4), failure_values, color=["#bd493a", "#d16f43", "#bd493a", "#d16f43"])
    axes[1].axvline(0.75, color="#252525", linestyle="--", linewidth=1.5)
    axes[1].set_yticks(np.arange(4), failure_labels)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0.7495, 0.7552)
    axes[1].set_xlabel("interior-gradient relative error")
    axes[1].set_title("Four strict failures; reference 594/598 cells, 11/13 rigs")
    axes[1].grid(axis="x", alpha=0.22)
    for bar, value in zip(bars, failure_values, strict=True):
        axes[1].text(value + 0.00008, bar.get_y() + bar.get_height() / 2, f"{value:.6f}", va="center", fontsize=9)

    fig.suptitle(
        "v230.1 Case 12: numerical adjudication passes, but the frozen K16 reference is inadequate",
        fontsize=13.5,
        fontweight="bold",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    main()
