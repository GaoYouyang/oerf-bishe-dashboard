"""Build the public v149.1 predictor and reproducibility figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_k1_dual_group_krylov_predictor_v149_public_summary.json"
OUTPUT = ROOT / "assets/figures/poolfire_k1_dual_group_krylov_predictor_v149.png"


def main() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    if payload["independent_status"] != "INCONCLUSIVE_INDEPENDENT_RECOMPUTATION_GROUP_KRYLOV_PREDICTOR_V149":
        raise RuntimeError("refusing to plot an unexpected v149.1 status")

    methods = payload["methods"]
    keys = [
        "oracle_block_krylov4",
        "visible_seed_control",
        "fit_mean_control",
        "linear_set_ridge",
        "rff_set_ridge",
    ]
    labels = ["Oracle*", "Visible\nseed", "Fit mean", "Linear\nridge", "RFF\nridge"]
    formal = [methods[key]["formal_cell_pass_count"] for key in keys]
    independent = [methods[key]["independent_cell_pass_count"] for key in keys]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 9,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(14.8, 7.2))
    figure.subplots_adjust(left=0.07, right=0.98, bottom=0.18, top=0.77, wspace=0.27)
    figure.patch.set_facecolor("#f7f8f6")

    axis = axes[0]
    x = np.arange(len(keys))
    width = 0.36
    formal_bars = axis.bar(x - width / 2, formal, width, color="#276c66", label="Formal")
    independent_bars = axis.bar(
        x + width / 2,
        independent,
        width,
        color="#77a7a1",
        edgecolor="#276c66",
        linewidth=0.8,
        label="Independent",
    )
    axis.axhline(3700, color="#202629", linestyle=(0, (4, 3)), linewidth=1.4, label="Required 3700")
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 4050)
    axis.set_ylabel("Cells passing the frozen target-space gate")
    axis.set_title("A  Capacity exists, but no observation-only\npredictor clears the gate", loc="left", weight="bold")
    axis.legend(frameon=False, loc="upper right")
    for bars, values in ((formal_bars, formal), (independent_bars, independent)):
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 65,
                str(value),
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

    axis = axes[1]
    diagnostic_labels = ["Group feature\nmax diff", "Linear prediction\nmax diff", "RFF prediction\nmax diff", "RFF lengthscale\nmax diff"]
    diagnostic_values = [9.06219543850284e-15, 3.6692870963861424e-13, 0.011682141913622823, 0.46576694922187656]
    colors = ["#276c66", "#276c66", "#c85b42", "#c85b42"]
    bars = axis.bar(np.arange(4), diagnostic_values, color=colors)
    axis.axhline(1e-8, color="#202629", linestyle=(0, (4, 3)), linewidth=1.4, label="Prediction tolerance 1e-8")
    axis.set_yscale("log")
    axis.set_ylim(1e-16, 2.0)
    axis.set_xticks(np.arange(4), diagnostic_labels)
    axis.set_ylabel("Maximum absolute difference (log scale)")
    axis.set_title("B  Independent mismatch is isolated to brittle\nRFF subsampling", loc="left", weight="bold")
    axis.legend(frameon=False, loc="upper left")
    for bar, value in zip(bars, diagnostic_values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.8,
            f"{value:.2e}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    for axis in axes:
        axis.set_facecolor("#ffffff")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#d9ddda", linewidth=0.7, alpha=0.75)
        axis.set_axisbelow(True)

    figure.suptitle(
        "v149.1 | Observation-only group-coordinate prediction\nFormal failure; independent RFF audit remains inconclusive",
        fontsize=15,
        weight="bold",
        x=0.04,
        y=0.96,
        ha="left",
    )
    figure.text(
        0.04,
        0.055,
        "* Truth-aware capacity upper bound | five opened PoolFire trajectories | +0A/+0A^T | no physical replay | "
        "no GPU | algorithm_breakthrough=false",
        fontsize=9.6,
        color="#39434a",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=170, facecolor=figure.get_facecolor())
    plt.close(figure)


if __name__ == "__main__":
    main()
