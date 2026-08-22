from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_full_dct_k2_complete_trajectory_v196_public_summary.json"
OUTPUT = ROOT / "assets/figures/poolfire_full_dct_k2_complete_trajectory_v196.png"


def main() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    arms = payload["arms"]
    names = [
        "Full-DCT K1",
        "Full-DCT K2",
        "Zero K2",
        "Zero K3",
        "Zero K4 ref",
    ]
    keys = [
        "full_dct_k1_parent",
        "full_dct_k2",
        "zero_cgls_k2",
        "zero_cgls_k3",
        "zero_cgls_k4_reference",
    ]

    def arm_value(key: str, sensor: str, metric: str) -> int:
        arm = arms[key]
        if sensor in arm:
            return int(arm[sensor][metric])
        prefix = "five_camera" if sensor == "five_camera" else "all_nine"
        suffix = "strict_safe_cells" if metric == "strict_safe_cells" else "complete_groups"
        return int(arm[f"{prefix}_{suffix}"])

    five_cells = np.array([arm_value(key, "five_camera", "strict_safe_cells") for key in keys])
    nine_cells = np.array([arm_value(key, "all_nine", "strict_safe_cells") for key in keys])
    five_groups = np.array(
        [arm_value(key, "five_camera", "complete_calibration_groups_passed") for key in keys]
    )
    nine_groups = np.array(
        [arm_value(key, "all_nine", "complete_calibration_groups_passed") for key in keys]
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 15,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
        }
    )
    fig = plt.figure(figsize=(20, 11), facecolor="#f7f8f6")
    grid = fig.add_gridspec(
        1,
        2,
        left=0.065,
        right=0.975,
        top=0.76,
        bottom=0.2,
        wspace=0.22,
    )
    ax_cells = fig.add_subplot(grid[0, 0])
    ax_groups = fig.add_subplot(grid[0, 1])

    fig.text(
        0.065,
        0.94,
        "v196  Dense K2 passes, but the frozen Zero-K4 reference is inadequate",
        fontsize=25,
        fontweight="bold",
        color="#17201d",
    )
    fig.text(
        0.065,
        0.885,
        "Complete opened p22 development trajectory: 101 frames x 13 calibrations per sensor arm.",
        fontsize=16,
        color="#4d5753",
    )
    fig.text(
        0.065,
        0.84,
        "Independent recomputation: 23/23 checks; formal/independent scientific arrays agree exactly.",
        fontsize=16,
        color="#4d5753",
    )

    x = np.arange(len(names))
    width = 0.34
    colors = ("#287c78", "#d98235")

    for ax, five, nine, required, ylabel, title in (
        (ax_cells, five_cells, nine_cells, 1313, "Strict-safe cells", "Cellwise absolute accuracy"),
        (ax_groups, five_groups, nine_groups, 13, "Complete calibration groups", "Complete-group gate"),
    ):
        five_bars = ax.bar(x - width / 2, five, width, color=colors[0], label="Five cameras")
        nine_bars = ax.bar(x + width / 2, nine, width, color=colors[1], label="All nine")
        ax.axhline(required, color="#aa3434", linewidth=2.2, linestyle="--", label=f"Required {required}")
        base_offset = 18 if required == 1313 else 0.18
        series_offset = 28 if required == 1313 else 0.34
        for series_index, bars in enumerate((five_bars, nine_bars)):
            for bar in bars:
                value = int(bar.get_height())
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + base_offset + series_index * series_offset,
                    f"{value}/{required}",
                    ha="center",
                    va="bottom",
                    fontsize=10.5,
                    fontweight="bold",
                )
        ax.set_xticks(x, names, rotation=12, ha="right")
        ax.set_ylim(0, required * 1.12)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", pad=18)
        ax.grid(axis="y", color="#d9ddda", linewidth=1, alpha=0.85)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper right", frameon=False, fontsize=11)

    fig.text(
        0.065,
        0.105,
        "INCONCLUSIVE: the preregistered Zero-K4 reference fails every cell and group, so relative headroom cannot be adjudicated.",
        fontsize=16,
        fontweight="bold",
        color="#8b2f2f",
    )
    fig.text(
        0.065,
        0.06,
        "Dense K2 is a strong post-open diagnostic, not a compact or learned initializer; no call reduction, wall/RSS, p14, external, or GPU claim.",
        fontsize=14,
        color="#4d5753",
    )
    fig.text(
        0.065,
        0.025,
        "Next action: audit the identity of the previously accepted reference before freezing another scientific gate.",
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
