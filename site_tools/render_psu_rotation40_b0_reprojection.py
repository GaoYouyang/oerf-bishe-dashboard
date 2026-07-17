#!/usr/bin/env python3
"""Render the public aggregate-only PSU rotation-40 B0 diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def render(summary_path: Path, output_dir: Path) -> None:
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = report["per_camera"]
    labels = [f"Camera {row['camera_id']}" for row in rows]
    relative = np.array([row["vector_relative_l2"] for row in rows])
    measured = np.array([row["measured_vector_rms_px"] for row in rows])
    predicted = np.array([row["predicted_vector_rms_px"] for row in rows])
    p95 = np.array([row["residual_magnitude_p95_px"] for row in rows])
    counts = np.array([row["ray_count"] for row in rows])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5), constrained_layout=True)
    fig.patch.set_facecolor("#f5f7f4")
    colors = ["#2d6a73", "#d67a45", "#6e5a8a"]

    axes[0].bar(labels, relative, color=colors, width=0.66)
    axes[0].axhline(
        report["aggregate"]["vector_relative_l2"],
        color="#20282d",
        linestyle="--",
        linewidth=1.2,
        label=f"pooled {report['aggregate']['vector_relative_l2']:.3f}",
    )
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("vector relative-L2")
    axes[0].set_title("Held-out image-space mismatch")
    axes[0].legend(frameon=False, loc="lower right")
    for index, value in enumerate(relative):
        axes[0].text(index, value + 0.025, f"{value:.3f}", ha="center", fontsize=9)

    x = np.arange(len(labels))
    width = 0.34
    axes[1].bar(x - width / 2, measured, width, label="measured", color="#2d6a73")
    axes[1].bar(x + width / 2, predicted, width, label="frozen B0", color="#e6a15c")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("vector RMS / px")
    axes[1].set_title("Signal amplitude is underpredicted")
    axes[1].legend(frameon=False)

    axes[2].bar(labels, p95, color=colors, width=0.66)
    axes[2].set_ylabel("residual magnitude p95 / px")
    axes[2].set_title("Tail residual by physical camera")
    for index, (value, count) in enumerate(zip(p95, counts, strict=True)):
        axes[2].text(index, value + 0.018, f"{value:.3f}\n{count/1e6:.2f}M rays", ha="center", fontsize=9)
    axes[2].set_ylim(0, max(p95) * 1.25)

    for axis in axes:
        axis.set_facecolor("#ffffff")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#d7ded8", linewidth=0.7, alpha=0.7)
        axis.set_axisbelow(True)
    fig.suptitle(
        "PSU rotation-40 development baseline | frozen support CGLS field\n"
        "Real BOS observations; no volumetric truth, no candidate comparison, no final-audit rotation opened.",
        fontsize=15,
        fontweight="bold",
        color="#20282d",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "diagnostic.png", dpi=180, facecolor=fig.get_facecolor())
    fig.savefig(output_dir / "diagnostic.pdf", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    render(args.summary, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
