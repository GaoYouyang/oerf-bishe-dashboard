#!/usr/bin/env python3
"""Build the redacted v229 comparison figure from public aggregate values."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets/figures/blastnet_case2_case5_low64_nested_dual_press_calibration_v229.png"


def main() -> None:
    rigs = np.arange(13)
    raw = np.asarray([6, 5, 14, 12, 5, 5, 22, 10, 5, 23, 10, 4, 5])
    studentized = np.asarray([7, 9, 15, 9, 4, 6, 17, 9, 5, 20, 9, 6, 7])
    fixed_or = np.asarray([7, 9, 17, 12, 5, 6, 22, 10, 5, 24, 10, 6, 7])
    nested = np.asarray([7, 8, 16, 12, 5, 6, 22, 10, 5, 24, 9, 6, 6])

    plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), constrained_layout=True)
    width = 0.19
    axes[0].bar(rigs - 1.5 * width, raw, width, label="raw PRESS", color="#5b8db8")
    axes[0].bar(
        rigs - 0.5 * width,
        studentized,
        width,
        label="studentized PRESS",
        color="#d39b45",
    )
    axes[0].bar(rigs + 0.5 * width, fixed_or, width, label="fixed OR", color="#8a69a7")
    axes[0].bar(
        rigs + 1.5 * width,
        nested,
        width,
        label="nested calibration",
        color="#26796c",
    )
    axes[0].axhline(5, color="#b84d3a", linestyle="--", linewidth=1.4, label="5/42 gate")
    axes[0].set_title("Case 5 accepted cells per held-out rig")
    axes[0].set_xlabel("rig index")
    axes[0].set_ylabel("accepted safe cells (of 42)")
    axes[0].set_xticks(rigs)
    axes[0].grid(axis="y", alpha=0.22)
    axes[0].legend(fontsize=8, ncol=2, loc="upper left")

    labels = ["Case 5", "Case 2"]
    totals = np.asarray([[126, 123, 140, 136], [297, 323, 324, 318]])
    x = np.arange(2)
    colors = ["#5b8db8", "#d39b45", "#8a69a7", "#26796c"]
    names = ["raw PRESS", "studentized PRESS", "fixed OR", "nested calibration"]
    for index, (name, color) in enumerate(zip(names, colors, strict=True)):
        axes[1].bar(x + (index - 1.5) * width, totals[:, index], width, label=name, color=color)
        for xi, value in zip(x, totals[:, index], strict=True):
            axes[1].text(
                xi + (index - 1.5) * width,
                value + 5,
                str(value),
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axes[1].set_title("Safe accepts; unsafe accepts remain zero")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("accepted safe cells")
    axes[1].set_ylim(0, 365)
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].legend(fontsize=8, loc="upper left")
    fig.suptitle(
        "v229 nested dual-PRESS calibration: fold-local utility without target-rig score leakage",
        fontsize=13,
        fontweight="bold",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    main()
