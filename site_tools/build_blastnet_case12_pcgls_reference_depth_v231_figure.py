#!/usr/bin/env python3
"""Build the redacted v231 Case 12 numerical-adjudication figure."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets/figures/blastnet_case12_pcgls_reference_depth_v231.png"


def main() -> None:
    labels = ["Jacobi", "field", "residual", "metric"]
    formal = np.asarray([2.3275616258261983e-16, 1.0849561650503294e-2, 3.4456680188956207e-1, 8.401649796350552e-3])
    independent = np.asarray([4.53960666122118e-16, 8.710700933541895e-3, 3.5101635224533595e-1, 7.005696721767485e-3])

    plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), constrained_layout=True)

    x = np.arange(len(labels))
    width = 0.36
    axes[0].bar(x - width / 2, formal, width, color="#315f93", label="formal")
    axes[0].bar(x + width / 2, independent, width, color="#a34f43", label="independent")
    axes[0].axhline(1e-8, color="#17252b", linestyle="--", linewidth=1.5, label="frozen tolerance = 1e-8")
    axes[0].set_yscale("log")
    axes[0].set_ylim(1e-17, 1.2)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("maximum discrepancy (log scale)")
    axes[0].set_title("Camera-order invariance fails for deep PCGLS")
    axes[0].grid(axis="y", which="both", alpha=0.18)
    axes[0].legend(fontsize=9, loc="lower right")

    axes[1].axis("off")
    rows = [
        ("Formal trajectory", "598 x 64 checkpoints", "complete"),
        ("Independent trajectory", "598 x 64 checkpoints", "complete"),
        ("K16 parent continuity", "field / residual / metric", "exact"),
        ("Physical residual closure", "independent maximum", "8.35e-14"),
        ("Selected global depth", "blocked by numerical gate", "none"),
    ]
    axes[1].set_title("What is and is not established", pad=14)
    y = 0.88
    for title, detail, value in rows:
        color = "#146f66" if value in {"complete", "exact"} else "#8a651b"
        axes[1].text(0.02, y, title, transform=axes[1].transAxes, fontsize=11, fontweight="bold", color="#17252b")
        axes[1].text(0.02, y - 0.07, detail, transform=axes[1].transAxes, fontsize=9.5, color="#5c6d74")
        axes[1].text(0.98, y - 0.025, value, transform=axes[1].transAxes, ha="right", fontsize=11, fontweight="bold", color=color)
        axes[1].plot([0.02, 0.98], [y - 0.11, y - 0.11], transform=axes[1].transAxes, color="#cad6d2", linewidth=0.8)
        y -= 0.17
    axes[1].text(0.02, 0.015, "Verdict: INCONCLUSIVE; no K1-K64 science depth released", transform=axes[1].transAxes, fontsize=10, fontweight="bold", color="#a34f43")

    fig.suptitle("v231 Case 12: reference-depth audit stops at the numerical invariance gate", fontsize=13.5, fontweight="bold")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    main()
