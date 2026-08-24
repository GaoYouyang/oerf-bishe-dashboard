#!/usr/bin/env python3
"""Build the redacted v233 Case 12 absolute-reference figure."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets/figures/blastnet_case12_absolute_spectral_reference_v233.png"


def main() -> None:
    labels = ["Field", "Full gradient", "Interior gradient", "Observation"]
    p90 = np.asarray([0.8201801181 / 0.5, 1.2315448676 / 0.75, 0.7791635436 / 0.75, 0.1339570388 / 0.2])
    worst = np.asarray([0.9140750074 / 0.75, 1.3672605383 / 1.0, 1.0554716506 / 1.0, 0.2024905699 / 0.35])
    y = np.arange(len(labels))

    plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.2, 5.5),
        gridspec_kw={"width_ratios": [1.55, 1]},
        constrained_layout=True,
    )

    width = 0.34
    axes[0].barh(y - width / 2, p90, height=width, color="#315f93", label="p90-higher / limit")
    axes[0].barh(y + width / 2, worst, height=width, color="#a34f43", label="worst / limit")
    axes[0].axvline(1.0, color="#17252b", linestyle="--", linewidth=1.5, label="frozen limit")
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("normalized error (1.0 = frozen limit)")
    axes[0].set_title("Observation passes; 3D field and gradients fail")
    axes[0].set_xlim(0, 1.85)
    axes[0].grid(axis="x", alpha=0.2)
    axes[0].legend(loc="lower right", fontsize=9)
    for row, values in enumerate(zip(p90, worst, strict=True)):
        for offset, value in zip((-width / 2, width / 2), values, strict=True):
            axes[0].text(value + 0.025, row + offset, f"{value:.2f}x", va="center", fontsize=9)

    axes[1].axis("off")
    axes[1].set_title("Independent certificate", pad=12)
    rows = [
        ("Coverage", "598 / 598 cells", "complete"),
        ("Independent checks", "17 / 17", "pass"),
        ("Field agreement", "1.25e-13 rel", "pass"),
        ("Metric agreement", "6.22e-14 abs", "pass"),
        ("Strict-safe cells", "0 / 598", "fail"),
        ("Complete rigs", "0 / 13", "fail"),
    ]
    y_pos = 0.91
    for title, value, state in rows:
        color = "#146f66" if state != "fail" else "#a34f43"
        axes[1].text(0.02, y_pos, title, transform=axes[1].transAxes, fontweight="bold", color="#17252b")
        axes[1].text(0.98, y_pos, value, transform=axes[1].transAxes, ha="right", fontweight="bold", color=color)
        axes[1].plot([0.02, 0.98], [y_pos - 0.075, y_pos - 0.075], transform=axes[1].transAxes, color="#cad6d2", linewidth=0.8)
        y_pos -= 0.135
    axes[1].text(
        0.02,
        0.035,
        "Stable solve, inadequate 3D reference\nNo DCT or ridge retuning",
        transform=axes[1].transAxes,
        fontsize=10,
        fontweight="bold",
        color="#a34f43",
    )

    fig.suptitle(
        "v233 Case 12: projection agreement does not guarantee 3D recovery",
        fontsize=14,
        fontweight="bold",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    main()
