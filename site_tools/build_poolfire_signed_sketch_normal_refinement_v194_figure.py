from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_signed_sketch_normal_refinement_v194_public_summary.json"
OUTPUT = ROOT / "assets/figures/poolfire_signed_sketch_normal_refinement_v194.png"


def main() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    safe = payload["strict_safe_cells"]
    diagnostics = payload["mechanism_diagnostics"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 16,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
        }
    )
    fig = plt.figure(figsize=(20, 10.34), facecolor="#f7f8f6")
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=(1.28, 1.0),
        left=0.075,
        right=0.975,
        top=0.79,
        bottom=0.18,
        wspace=0.24,
    )
    ax_safe = fig.add_subplot(grid[0, 0])
    ax_scale = fig.add_subplot(grid[0, 1])

    fig.text(
        0.075,
        0.94,
        "v194  Full-Hessian unit step fails; diagonal scaling exposes the mechanism",
        fontsize=27,
        fontweight="bold",
        color="#17201d",
    )
    fig.text(
        0.075,
        0.885,
        "Same sealed seed and unchanged physical K1 across 104 opened development cells; the preregistered primary still decides the verdict.",
        fontsize=16,
        color="#4d5753",
    )

    methods = ["v193 seed", "Full Hessian\nprimary", "Diagonal\ncontrol"]
    five = np.array(
        [
            safe["v193_seed_k1"]["five_camera"],
            safe["full_hessian_primary_k1"]["five_camera"],
            safe["diagonal_control_k1"]["five_camera"],
        ]
    )
    nine = np.array(
        [
            safe["v193_seed_k1"]["all_nine"],
            safe["full_hessian_primary_k1"]["all_nine"],
            safe["diagonal_control_k1"]["all_nine"],
        ]
    )
    x = np.arange(len(methods))
    width = 0.34
    bars_five = ax_safe.bar(x - width / 2, five, width, color="#2b7a78", label="Five cameras")
    bars_nine = ax_safe.bar(x + width / 2, nine, width, color="#e08b3e", label="All nine")
    ax_safe.axhline(52, color="#b33a3a", linewidth=2.2, linestyle="--", label="Required 52/52")
    for bars in (bars_five, bars_nine):
        for bar in bars:
            ax_safe.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.7,
                f"{int(bar.get_height())}/52",
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
            )
    ax_safe.set_xticks(x, methods)
    ax_safe.set_ylim(0, 57)
    ax_safe.set_ylabel("Strict-safe K1 cells")
    ax_safe.set_title("Primary failure is decisive", loc="left", pad=18)
    ax_safe.grid(axis="y", color="#d9ddda", linewidth=1, alpha=0.85)
    ax_safe.set_axisbelow(True)
    ax_safe.legend(loc="upper left", frameon=False, ncol=3, fontsize=12)
    ax_safe.spines[["top", "right"]].set_visible(False)

    labels = ["Correction norm\np50", "Normal residual ratio\np50"]
    primary = np.array(
        [
            diagnostics["coordinate_correction_norm_p50"]["full_hessian_primary"],
            diagnostics["full_normal_residual_ratio_p50"]["full_hessian_primary"],
        ]
    )
    diagonal = np.array(
        [
            diagnostics["coordinate_correction_norm_p50"]["diagonal_control"],
            diagnostics["full_normal_residual_ratio_p50"]["diagonal_control"],
        ]
    )
    y = np.arange(len(labels))
    height = 0.34
    bars_primary = ax_scale.barh(y - height / 2, primary, height, color="#b33a3a", label="Full Hessian")
    bars_diagonal = ax_scale.barh(y + height / 2, diagonal, height, color="#6a8f3d", label="Diagonal")
    for bars in (bars_primary, bars_diagonal):
        for bar in bars:
            ax_scale.text(
                bar.get_width() * 1.08,
                bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.3g}",
                va="center",
                fontweight="bold",
            )
    ax_scale.set_yticks(y, labels)
    ax_scale.invert_yaxis()
    ax_scale.set_xscale("log")
    ax_scale.set_xlim(0.5, 160)
    ax_scale.set_xlabel("Median magnitude (log scale)")
    ax_scale.set_title("Off-diagonal unit step strongly overshoots", loc="left", pad=18)
    ax_scale.grid(axis="x", color="#d9ddda", linewidth=1, alpha=0.85, which="both")
    ax_scale.set_axisbelow(True)
    ax_scale.legend(loc="lower right", frameon=False)
    ax_scale.spines[["top", "right", "left"]].set_visible(False)

    fig.text(
        0.075,
        0.055,
        "The diagonal control passes 104/104, but the preregistered order does not allow it to replace a failed primary after results are visible.",
        fontsize=14,
        color="#4d5753",
    )
    fig.text(
        0.075,
        0.02,
        "Post-open mechanism diagnostic only: no complete trajectory, deployable initializer, resource gain, external result, or real-BOST claim.",
        fontsize=14,
        color="#4d5753",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    with Image.open(OUTPUT) as image:
        image.convert("RGB").save(OUTPUT)


if __name__ == "__main__":
    main()
