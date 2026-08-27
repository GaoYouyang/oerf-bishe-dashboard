from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_volume_hodge_equivalence_v273_public_summary.json"
OUTPUT = ROOT / "assets/figures/blastnet_case19_volume_hodge_equivalence_v273.png"


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    independent = data["independent_validation"]

    fig = plt.figure(figsize=(15.8, 7.2), facecolor="#f3f5f4")
    grid = fig.add_gridspec(1, 3, width_ratios=(1.18, 1.0, 1.18), wspace=0.28)
    factor_ax = fig.add_subplot(grid[0, 0])
    error_ax = fig.add_subplot(grid[0, 1])
    verdict_ax = fig.add_subplot(grid[0, 2])

    factor_ax.axis("off")
    factor_ax.set_title("Frozen operator factorization", loc="left", fontweight="bold")
    boxes = ((0.02, "Scalar field\nx"), (0.375, "Gradient\nD x"), (0.73, "Observation\nM D x"))
    box_width = 0.245
    for x, label in boxes:
        factor_ax.add_patch(FancyBboxPatch((x, 0.57), box_width, 0.18, boxstyle="round,pad=0.018", transform=factor_ax.transAxes, facecolor="#ffffff", edgecolor="#7b8b8f", linewidth=1.2))
        factor_ax.text(x + box_width / 2, 0.66, label, transform=factor_ax.transAxes, ha="center", va="center", fontsize=11.5, fontweight="bold")
    for x_start, x_end, label in ((0.275, 0.365, "D"), (0.63, 0.72, "M")):
        factor_ax.annotate("", xy=(x_end, 0.66), xytext=(x_start, 0.66), xycoords="axes fraction", textcoords="axes fraction", arrowprops={"arrowstyle": "->", "lw": 1.8})
        factor_ax.text((x_start + x_end) / 2, 0.695, label, transform=factor_ax.transAxes, ha="center", va="bottom", fontsize=10.5)
    factor_ax.text(0.02, 0.35, "A = M D", transform=factor_ax.transAxes, fontsize=23, fontweight="bold", color="#236b63")
    factor_ax.text(0.02, 0.22, "Verified on 13/13 reported geometries", transform=factor_ax.transAxes, fontsize=11, color="#40474b")

    labels = ("A = M D", "Hodge lift", "Poisson residual")
    values = np.asarray((independent["maximum_active_operator_factorization_relative"], independent["maximum_hodge_lift_equivalence_relative"], independent["maximum_poisson_solve_relative_residual"]), dtype=np.float64)
    bars = error_ax.bar(np.arange(3), values, color=("#3279a8", "#d39b3a", "#5c8f68"), width=0.62)
    error_ax.set_yscale("log")
    error_ax.axhline(1e-9, color="#c94f4f", linewidth=1.3, linestyle="--", label="frozen agreement limit")
    error_ax.set_xticks(np.arange(3), labels, rotation=16, ha="right")
    error_ax.set_ylim(1e-16, 2e-8)
    error_ax.set_ylabel("Maximum relative error")
    error_ax.set_title("Independent numerical closure", loc="left", fontweight="bold")
    error_ax.legend(frameon=False, loc="upper left")
    for bar, value in zip(bars, values, strict=True):
        error_ax.text(bar.get_x() + bar.get_width() / 2, value * 1.8, f"{value:.2e}", ha="center", va="bottom", fontsize=9)

    verdict_ax.axis("off")
    verdict_ax.set_title("Scientific verdict", loc="left", fontweight="bold")
    verdict_ax.text(0.02, 0.72, "Direct volume Hodge lift", transform=verdict_ax.transAxes, fontsize=13, fontweight="bold")
    verdict_ax.text(0.02, 0.60, "=", transform=verdict_ax.transAxes, fontsize=20, fontweight="bold", color="#236b63")
    verdict_ax.text(0.02, 0.48, "Poisson-preconditioned\nscalar adjoint", transform=verdict_ax.transAxes, fontsize=15, fontweight="bold", color="#236b63")
    verdict_ax.text(0.02, 0.27, "Reparameterization, not a new\nphysical direction", transform=verdict_ax.transAxes, fontsize=12, color="#40474b")
    verdict_ax.text(0.02, 0.10, "Route closed | 0A+0AT | no truth read", transform=verdict_ax.transAxes, fontsize=10.5, color="#a13c3c", fontweight="bold")

    fig.suptitle("v273 | A direct 3D Hodge lift collapses to the existing Poisson-preconditioned adjoint", x=0.045, y=0.985, ha="left", fontsize=16.2, fontweight="bold")
    fig.text(0.045, 0.018, f"Independent validity 16/16 | maximum formal-independent common-metric difference {independent['maximum_formal_independent_common_metric_absolute_difference']:.2e} | algorithm_breakthrough=false", fontsize=10, color="#40474b")
    error_ax.set_facecolor("#ffffff")
    error_ax.spines[["top", "right"]].set_visible(False)
    error_ax.grid(axis="y", alpha=0.2)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
