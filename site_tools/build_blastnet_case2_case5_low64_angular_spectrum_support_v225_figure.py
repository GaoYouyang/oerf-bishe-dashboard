#!/usr/bin/env python3
"""Build the public v225 angular-spectrum support figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT
    / "docs/blastnet_case2_case5_low64_angular_spectrum_support_v225_public_summary.json"
)
OUTPUT = (
    ROOT
    / "assets/figures/blastnet_case2_case5_low64_angular_spectrum_support_v225.png"
)


def build() -> Path:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    primary = payload["primary_angular_spectrum_policy"]
    control = payload["cheap_two_scalar_control"]
    validation = payload["independent_validation"]
    rows = [
        ("Angular 27D / Case 5", primary["case5"]),
        ("Angular 27D / Case 2", primary["case2"]),
        ("Scalar 2D / Case 5", control["case5"]),
        ("Scalar 2D / Case 2", control["case2"]),
    ]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "figure.facecolor": "#f5f7f4",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#c9d2ce",
            "axes.labelcolor": "#25342f",
            "xtick.color": "#42514c",
            "ytick.color": "#42514c",
            "axes.titleweight": "bold",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.6))
    fig.subplots_adjust(left=0.18, right=0.97, top=0.78, bottom=0.20, wspace=0.28)
    fig.suptitle(
        "v225: full multiview structure still does not yield a safe fallback",
        x=0.055,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#17231f",
    )
    fig.text(
        0.055,
        0.85,
        "27-D angular spectrum  |  one-class Case 5 support  |  Case 2 decisions sealed before truth",
        color="#52605b",
        fontsize=11,
    )

    labels = [label for label, _ in rows]
    y = np.arange(len(rows))
    minimum = np.array([row["minimum_rig_accept_fraction"] for _, row in rows])
    maximum = np.array([row["maximum_rig_accept_fraction"] for _, row in rows])
    axes[0].hlines(y, minimum * 100.0, maximum * 100.0, color="#4d7d93", linewidth=8)
    axes[0].scatter(minimum * 100.0, y, color="#1e5f76", s=55, zorder=3, label="Minimum")
    axes[0].scatter(maximum * 100.0, y, color="#77aaba", s=55, zorder=3, label="Maximum")
    axes[0].axvline(10.0, color="#c14b3a", linestyle="--", linewidth=1.8, label="10% gate")
    for index, value in enumerate(minimum * 100.0):
        axes[0].text(value + 1.2, index - 0.12, f"min {value:.1f}%", fontsize=9)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(-1.0, 105.0)
    axes[0].set_xlabel("Per-rig acceptance range")
    axes[0].set_title("Utility is not stable across rigs")
    axes[0].grid(axis="x", color="#dce3df", linewidth=0.8)
    axes[0].spines[["top", "right", "left"]].set_visible(False)
    axes[0].tick_params(axis="y", length=0)
    axes[0].legend(frameon=False, loc="lower right", fontsize=9)

    accepted_safe = np.array([252, 378, 339, 54], dtype=np.float64)
    accepted_unsafe = np.array([0, 145, 0, 132], dtype=np.float64)
    axes[1].barh(y, accepted_safe, color="#26866b", label="Accepted safe")
    axes[1].barh(
        y,
        accepted_unsafe,
        left=accepted_safe,
        color="#c14b3a",
        label="Accepted unsafe",
    )
    for index, (safe_count, unsafe_count) in enumerate(
        zip(accepted_safe, accepted_unsafe, strict=True)
    ):
        axes[1].text(
            safe_count + unsafe_count + 8,
            index,
            f"unsafe {int(unsafe_count)}",
            va="center",
            fontsize=9,
            color="#8d3429" if unsafe_count else "#42514c",
        )
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0.0, 590.0)
    axes[1].set_xlabel("Accepted cells")
    axes[1].set_title("Cross-condition acceptance is not fail-closed")
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
        "Decision: close this fixed angular-spectrum mutual-support policy; no deployment, call-saving, or real-BOST claim.",
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
