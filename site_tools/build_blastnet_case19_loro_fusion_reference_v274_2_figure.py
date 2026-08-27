from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_loro_fusion_reference_v274_2_public_summary.json"
OUTPUT = ROOT / "assets/figures/blastnet_case19_loro_fusion_reference_v274_2.png"


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    formal = data["formal_execution"]
    audit = data["independent_validation"]

    labels = (
        "Labeled-rig\npermutation",
        "Camera\npermutation",
        "Independent\npermutation",
        "Field",
        "Target\nprojection",
        "Target\nresidual",
    )
    values = np.asarray(
        (
            max(formal["maximum_labeled_rig_permutation_relative_difference"], 1e-18),
            formal["maximum_camera_permutation_relative_difference"],
            audit["maximum_independent_permutation_relative_difference"],
            audit["maximum_formal_independent_field_relative_difference"],
            audit["maximum_formal_independent_target_projection_relative_difference"],
            audit["maximum_formal_independent_target_residual_relative_difference"],
        ),
        dtype=np.float64,
    )
    limits = np.asarray((1e-10, 1e-10, 1e-10, 1e-9, 1e-9, 1e-9), dtype=np.float64)
    ratios = values / limits
    passed = ratios <= 1.0

    fig = plt.figure(figsize=(15.8, 7.2), facecolor="#f3f5f4")
    grid = fig.add_gridspec(1, 2, width_ratios=(1.65, 0.9), wspace=0.22)
    ax = fig.add_subplot(grid[0, 0])
    verdict = fig.add_subplot(grid[0, 1])

    colors = np.where(passed, "#2e7d68", "#c85a4b")
    bars = ax.bar(np.arange(len(labels)), ratios, color=colors, width=0.66)
    ax.set_yscale("log")
    ax.axhline(1.0, color="#343c40", linewidth=1.4, linestyle="--", label="frozen limit")
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_ylabel("Observed difference / frozen limit")
    ax.set_title("Reproducibility gates", loc="left", fontweight="bold")
    ax.set_ylim(1e-9, 3e3)
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    for bar, ratio, value in zip(bars, ratios, values, strict=True):
        label = "0" if value <= 1e-18 else f"{ratio:.1f}x"
        y = max(ratio, 1e-8)
        ax.text(bar.get_x() + bar.get_width() / 2, y * 1.35, label, ha="center", va="bottom", fontsize=9, fontweight="bold")

    verdict.axis("off")
    verdict.set_title("Scientific decision", loc="left", fontweight="bold")
    verdict.text(0.02, 0.76, "INCONCLUSIVE", transform=verdict.transAxes, fontsize=23, fontweight="bold", color="#c85a4b")
    verdict.text(0.02, 0.61, "Rig-label ordering repaired", transform=verdict.transAxes, fontsize=12.5, fontweight="bold", color="#2e7d68")
    verdict.text(0.02, 0.51, "Camera permutation and\ncross-implementation limits fail", transform=verdict.transAxes, fontsize=13, color="#343c40")
    verdict.text(0.02, 0.31, "429/429 is diagnostic only", transform=verdict.transAxes, fontsize=13, fontweight="bold", color="#c85a4b")
    verdict.text(0.02, 0.16, "Fixed LORO K16 reference closed\nNo further numerical repair", transform=verdict.transAxes, fontsize=11.5, color="#343c40")

    fig.suptitle("v274.2 | Numerical repair does not clear the frozen reproducibility contract", x=0.045, y=0.985, ha="left", fontsize=16.2, fontweight="bold")
    fig.text(0.045, 0.018, "Formal 14/15 | independent 27/32 | accuracy counts are not a scientific pass | algorithm_breakthrough=false", fontsize=10, color="#40474b")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
