from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets/figures/blastnet_case19_bulk_advection_warm_v275.png"


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#9aaba9",
            "axes.labelcolor": "#233338",
            "xtick.color": "#43555a",
            "ytick.color": "#43555a",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), facecolor="#f4f7f5")
    colors = ("#18736a", "#c06050")

    labels = ("K14 primary", "K16 reference")
    cells = np.array([428 / 429, 428 / 429]) * 100
    rigs = np.array([12 / 13, 12 / 13]) * 100
    x = np.arange(2)
    width = 0.33
    axes[0].bar(x - width / 2, cells, width, label="cells", color=colors[0])
    axes[0].bar(x + width / 2, rigs, width, label="complete rigs", color="#4e73a8")
    axes[0].axhline(100, color="#7d5a19", linewidth=1.4, linestyle="--")
    axes[0].set_ylim(0, 106)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("absolute gate pass (%)")
    axes[0].set_title("Reference is not adequate")
    axes[0].legend(frameon=False, loc="lower left")
    for index, value in enumerate(cells):
        axes[0].text(index - width / 2, value + 1.3, "428/429", ha="center", fontsize=10)
    for index, value in enumerate(rigs):
        axes[0].text(index + width / 2, value + 1.3, "12/13", ha="center", fontsize=10)

    axes[1].bar([0, 1], [13 / 429 * 100, 0], color=("#b47b26", colors[1]), width=0.58)
    axes[1].set_xticks([0, 1], ["matched cells", "complete rigs"])
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("primary matched pass (%)")
    axes[1].set_title("Matched accuracy fails")
    axes[1].text(0, 13 / 429 * 100 + 2.2, "13/429", ha="center", fontweight="bold")
    axes[1].text(1, 2.2, "0/13", ha="center", fontweight="bold")

    names = ("transport", "transport audit", "field", "residual", "metric")
    ratios = np.array(
        [
            3.326820085024774e-9 / 1e-12,
            1.2153782147361625e-10 / 1e-12,
            1.5191937770901028e-8 / 1e-8,
            1.5673454416080782e-6 / 1e-8,
            1.021917933607952e-8 / 1e-8,
        ]
    )
    y = np.arange(len(names))
    axes[2].barh(y, ratios, color=("#6f5795", "#d18a39", "#4e73a8", "#c06050", "#18736a"))
    axes[2].axvline(1, color="#151f22", linewidth=1.5, linestyle="--")
    axes[2].set_xscale("log")
    axes[2].set_yticks(y, names)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("observed difference / frozen limit")
    axes[2].set_title("Independent closure: 26/31")
    for index, value in enumerate(ratios):
        axes[2].text(value * 1.08, index, f"{value:.3g}x", va="center", fontsize=10)

    for axis in axes:
        axis.grid(axis="y", color="#d8e0dd", linewidth=0.8, alpha=0.8)
        axis.set_axisbelow(True)
    fig.suptitle(
        "v275 Case 19 bulk-advection warm start: diagnostic counts are not a pass",
        fontsize=17,
        fontweight="bold",
        color="#17252b",
    )
    fig.text(
        0.5,
        0.015,
        "Post-open straight-ray proxy | reference inadequate | numerical closure failed | algorithm_breakthrough=false",
        ha="center",
        color="#52656a",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.91), w_pad=2.1)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
