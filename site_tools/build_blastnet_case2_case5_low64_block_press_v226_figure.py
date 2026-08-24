#!/usr/bin/env python3
"""Build the public v226 block-PRESS certificate figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case2_case5_low64_block_press_v226_public_summary.json"
OUTPUT = ROOT / "assets/figures/blastnet_case2_case5_low64_block_press_v226.png"


CASE5_ACCEPTED = np.array([6, 5, 14, 12, 5, 5, 22, 10, 5, 23, 10, 4, 5])
CASE5_TOTAL = 42


def build() -> Path:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    primary = payload["primary_block_press_certificate"]
    control = payload["cheap_full_fit_residual_control"]
    validation = payload["independent_validation"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "figure.facecolor": "#f4f7f5",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#c8d2cd",
            "axes.labelcolor": "#25342f",
            "xtick.color": "#42514c",
            "ytick.color": "#42514c",
            "axes.titleweight": "bold",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.8))
    fig.subplots_adjust(left=0.075, right=0.97, top=0.77, bottom=0.20, wspace=0.23)
    fig.suptitle(
        "v226 block-PRESS: safe transfer, utility gate misses by one frame",
        x=0.055,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#17231f",
    )
    fig.text(
        0.055,
        0.845,
        "Nine-camera held-out prediction  |  Case 5-only calibration  |  decisions sealed before truth",
        color="#52605b",
        fontsize=11,
    )

    rig_labels = [f"R{index:02d}" for index in range(1, 14)]
    acceptance_percent = 100.0 * CASE5_ACCEPTED / CASE5_TOTAL
    colors = np.where(CASE5_ACCEPTED >= 5, "#2d8a6d", "#c44f3d")
    axes[0].bar(rig_labels, acceptance_percent, color=colors, width=0.72)
    axes[0].axhline(10.0, color="#8d3429", linestyle="--", linewidth=1.8)
    axes[0].text(
        12.8,
        58.5,
        "Frozen gate: at least 5/42",
        color="#8d3429",
        ha="right",
        va="top",
        fontsize=9,
    )
    for index, count in enumerate(CASE5_ACCEPTED):
        if count <= 5:
            axes[0].text(
                index,
                acceptance_percent[index] + 1.3,
                f"{count}/42",
                ha="center",
                fontsize=8.5,
            )
    axes[0].set_ylim(0.0, 61.0)
    axes[0].set_ylabel("Case 5 leave-one-rig-out acceptance")
    axes[0].set_title("One rig accepts 4/42; passing requires 5/42")
    axes[0].grid(axis="y", color="#dce3df", linewidth=0.8)
    axes[0].spines[["top", "right", "left"]].set_visible(False)
    axes[0].tick_params(axis="x", rotation=45, length=0)
    axes[0].tick_params(axis="y", length=0)

    labels = ["Block-PRESS", "Full-fit residual control"]
    safe = np.array(
        [
            primary["case2"]["accepted_safe_cells"],
            control["case2"]["accepted_safe_cells"],
        ]
    )
    unsafe = np.array(
        [
            primary["case2"]["accepted_unsafe_cells"],
            control["case2"]["accepted_unsafe_cells"],
        ]
    )
    y = np.arange(2)
    axes[1].barh(y, safe, color="#2d8a6d", label="Accepted safe")
    axes[1].barh(y, unsafe, left=safe, color="#c44f3d", label="Accepted unsafe")
    for index, (safe_count, unsafe_count) in enumerate(zip(safe, unsafe, strict=True)):
        axes[1].text(
            8,
            index,
            f"safe {safe_count}",
            va="center",
            color="white",
            fontweight="bold",
        )
        axes[1].text(
            safe_count + unsafe_count + 10,
            index,
            f"unsafe {unsafe_count}",
            va="center",
            color="#8d3429" if unsafe_count else "#2b6354",
            fontweight="bold",
        )
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0.0, 620.0)
    axes[1].set_xlabel("Accepted Case 2 cells")
    axes[1].set_title("PRESS rejects all 197 unsafe Case 2 cells")
    axes[1].grid(axis="x", color="#dce3df", linewidth=0.8)
    axes[1].spines[["top", "right", "left"]].set_visible(False)
    axes[1].tick_params(axis="y", length=0)
    axes[1].legend(frameon=False, loc="lower right", fontsize=9)

    fig.text(
        0.055,
        0.095,
        (
            "Independent recomputation: "
            f"{validation['required_checks_passed']}/{validation['required_checks_total']} checks; "
            f"maximum feature difference {validation['maximum_formal_independent_feature_difference']:.2e}."
        ),
        color="#42514c",
        fontsize=10,
    )
    fig.text(
        0.055,
        0.05,
        "Formal decision: FAIL. Close this exact certificate; no threshold retuning, deployment, resource, or real-BOST claim.",
        color="#8d3429",
        fontsize=10.5,
        fontweight="bold",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    with Image.open(OUTPUT) as image:
        image.convert("RGB").save(OUTPUT, optimize=True)
    return OUTPUT


if __name__ == "__main__":
    print(build())
