#!/usr/bin/env python3
"""Build the redacted v232.1 Case 12 numerical-fragility figure."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets/figures/blastnet_case12_canonical_pcgls_reference_depth_v232.png"


def main() -> None:
    depths = np.asarray([16, 17, 18, 19, 26, 32, 64])
    field_difference = np.asarray(
        [
            3.2779949375200197e-9,
            1.6742878875458043e-8,
            1.0607716237219124e-7,
            5.170604830304387e-7,
            1.1792744948564483e-2,
            6.052596117453523e-5,
            1.67805131225258e-3,
        ]
    )

    plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), constrained_layout=True)

    axes[0].plot(depths, field_difference, color="#315f93", marker="o", linewidth=2.1)
    axes[0].axhline(
        1e-8,
        color="#a34f43",
        linestyle="--",
        linewidth=1.6,
        label="frozen field tolerance = 1e-8",
    )
    axes[0].scatter([17], [field_difference[1]], color="#a34f43", s=72, zorder=3)
    axes[0].annotate(
        "first failure: K17",
        xy=(17, field_difference[1]),
        xytext=(27, 2.5e-8),
        arrowprops={"arrowstyle": "->", "color": "#a34f43"},
        color="#8b4038",
        fontweight="bold",
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("PCGLS depth")
    axes[0].set_ylabel("maximum formal-independent field difference")
    axes[0].set_title("Roundoff-scale input drift is amplified")
    axes[0].grid(alpha=0.2, which="both")
    axes[0].legend(loc="lower right", fontsize=9)

    axes[1].axis("off")
    rows = [
        ("Canonical observation", "camera reversal", "exact"),
        ("Within-implementation fields", "camera reversal", "exact"),
        ("Jacobi inverse", "cross implementation", "2.25e-16 rel"),
        ("K17 field", "cross implementation", "1.67e-8"),
        ("Released depth", "numerical contract", "none"),
    ]
    axes[1].set_title("What the repaired audit establishes", pad=14)
    y = 0.88
    for title, detail, value in rows:
        color = "#146f66" if value == "exact" else "#a34f43"
        axes[1].text(
            0.02,
            y,
            title,
            transform=axes[1].transAxes,
            fontsize=11,
            fontweight="bold",
            color="#17252b",
        )
        axes[1].text(
            0.02,
            y - 0.07,
            detail,
            transform=axes[1].transAxes,
            fontsize=9.5,
            color="#5c6d74",
        )
        axes[1].text(
            0.98,
            y - 0.025,
            value,
            transform=axes[1].transAxes,
            ha="right",
            fontsize=11,
            fontweight="bold",
            color=color,
        )
        axes[1].plot(
            [0.02, 0.98],
            [y - 0.11, y - 0.11],
            transform=axes[1].transAxes,
            color="#cad6d2",
            linewidth=0.8,
        )
        y -= 0.17
    axes[1].text(
        0.02,
        0.015,
        "Verdict: INCONCLUSIVE; current deep-PCGLS reference shell closed",
        transform=axes[1].transAxes,
        fontsize=9.8,
        fontweight="bold",
        color="#a34f43",
    )

    fig.suptitle(
        "v232.1 Case 12: canonical ordering succeeds, reference stability does not",
        fontsize=13.5,
        fontweight="bold",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    main()
