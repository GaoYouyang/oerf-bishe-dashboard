#!/usr/bin/env python3
"""Build the public v227 studentized block-PRESS figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT
    / "docs/blastnet_case2_case5_low64_studentized_block_press_v227_public_summary.json"
)
OUTPUT = (
    ROOT / "assets/figures/blastnet_case2_case5_low64_studentized_block_press_v227.png"
)

V227_CASE5_ACCEPTED = np.array([7, 9, 15, 9, 4, 6, 17, 9, 5, 20, 9, 6, 7])
CASE5_TOTAL = 42


def build() -> Path:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    primary = payload["primary_studentized_block_press_certificate"]
    parent = payload["parent_v226_block_press_control"]
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
        "v227 geometry whitening: safer acceptance rises, utility still fails",
        x=0.055,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#17231f",
    )
    fig.text(
        0.055,
        0.845,
        "Nine-camera predictive-covariance whitening  |  Case 5-only calibration  |  zero unsafe Case 2 accepts",
        color="#52605b",
        fontsize=11,
    )

    rig_labels = [f"R{index:02d}" for index in range(1, 14)]
    acceptance_percent = 100.0 * V227_CASE5_ACCEPTED / CASE5_TOTAL
    colors = np.where(V227_CASE5_ACCEPTED >= 5, "#2d8a6d", "#c44f3d")
    axes[0].bar(rig_labels, acceptance_percent, color=colors, width=0.72)
    axes[0].axhline(10.0, color="#8d3429", linestyle="--", linewidth=1.8)
    axes[0].text(
        12.8,
        49.0,
        "Frozen gate: at least 5/42",
        color="#8d3429",
        ha="right",
        va="top",
        fontsize=9,
    )
    for index, count in enumerate(V227_CASE5_ACCEPTED):
        if count <= 6:
            axes[0].text(
                index,
                acceptance_percent[index] + 1.1,
                f"{count}/42",
                ha="center",
                fontsize=8.5,
            )
    axes[0].set_ylim(0.0, 51.0)
    axes[0].set_ylabel("Case 5 leave-one-rig-out acceptance")
    axes[0].set_title("Worst rig remains 4/42; failure moves to R05")
    axes[0].grid(axis="y", color="#dce3df", linewidth=0.8)
    axes[0].spines[["top", "right", "left"]].set_visible(False)
    axes[0].tick_params(axis="x", rotation=45, length=0)
    axes[0].tick_params(axis="y", length=0)

    labels = ["v226 raw PRESS", "v227 geometry-whitened"]
    safe = np.array(
        [
            parent["case2_accepted_safe_cells"],
            primary["case2"]["accepted_safe_cells"],
        ]
    )
    unsafe = np.array(
        [
            parent["case2_accepted_unsafe_cells"],
            primary["case2"]["accepted_unsafe_cells"],
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
            safe_count + 8,
            index,
            f"unsafe {unsafe_count}",
            va="center",
            color="#2b6354",
            fontweight="bold",
        )
    axes[1].annotate(
        "+26 safe accepts",
        xy=(323, 1),
        xytext=(345, 0.55),
        arrowprops={"arrowstyle": "->", "color": "#476b5f"},
        color="#2b6354",
        fontweight="bold",
    )
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0.0, 410.0)
    axes[1].set_xlabel("Accepted Case 2 cells")
    axes[1].set_title("Whitening adds safe accepts and preserves zero unsafe")
    axes[1].grid(axis="x", color="#dce3df", linewidth=0.8)
    axes[1].spines[["top", "right", "left"]].set_visible(False)
    axes[1].tick_params(axis="y", length=0)

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
        "Formal decision: FAIL. Whitening changes the score but does not solve cross-rig utility stability.",
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
