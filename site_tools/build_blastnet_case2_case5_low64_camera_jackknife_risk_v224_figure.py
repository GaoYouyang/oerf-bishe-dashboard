#!/usr/bin/env python3
"""Build the public v224 camera-jackknife overlap figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT
    / "docs/blastnet_case2_case5_low64_camera_jackknife_risk_v224_public_summary.json"
)
OUTPUT = (
    ROOT
    / "assets/figures/blastnet_case2_case5_low64_camera_jackknife_risk_v224.png"
)


def _interval(
    ax, y: float, low: float, high: float, color: str, label: str
) -> None:
    ax.plot(
        [low, high],
        [y, y],
        color=color,
        linewidth=12,
        solid_capstyle="round",
        label=label,
    )
    ax.scatter([low, high], [y, y], color=color, s=45, zorder=3)
    ax.text(low, y + 0.14, f"{low:.4f}", ha="center", va="bottom", fontsize=9)
    ax.text(high, y + 0.14, f"{high:.4f}", ha="center", va="bottom", fontsize=9)


def build() -> Path:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    primary = payload["primary_camera_jackknife_instability"]
    control = payload["cheap_max_camera_residual_control"]
    validation = payload["independent_validation"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "figure.facecolor": "#f5f7f4",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#c9d2ce",
            "axes.labelcolor": "#25342f",
            "xtick.color": "#42514c",
            "ytick.color": "#42514c",
            "axes.titleweight": "bold",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2))
    fig.subplots_adjust(left=0.065, right=0.98, top=0.76, bottom=0.25, wspace=0.24)
    fig.suptitle(
        "v224: camera-deletion stability still overlaps",
        x=0.055,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#17231f",
    )
    fig.text(
        0.055,
        0.845,
        "Safe 1,064 cells  |  Unsafe 197 cells  |  all 9 reduced systems rank 64",
        color="#52605b",
        fontsize=11,
    )

    panels = [
        (axes[0], primary, "Worst camera-deletion drift J"),
        (axes[1], control, "Maximum per-camera fit residual C"),
    ]
    for ax, values, title in panels:
        _interval(
            ax,
            1.0,
            values["safe_minimum"],
            values["safe_maximum"],
            "#26866b",
            "Safe",
        )
        _interval(
            ax,
            0.0,
            values["unsafe_minimum"],
            values["unsafe_maximum"],
            "#c14b3a",
            "Unsafe",
        )
        overlap_low = max(values["safe_minimum"], values["unsafe_minimum"])
        overlap_high = min(values["safe_maximum"], values["unsafe_maximum"])
        ax.axvspan(
            overlap_low,
            overlap_high,
            color="#d9a441",
            alpha=0.18,
            label="Overlap",
        )
        ax.set_title(title)
        ax.set_xlabel(
            f"Lower is safer; strict margin = {values['strict_separation_margin']:.4f}"
        )
        ax.set_yticks([0.0, 1.0], ["Unsafe", "Safe"])
        ax.set_ylim(-0.5, 1.55)
        lower = min(values["safe_minimum"], values["unsafe_minimum"])
        upper = max(values["safe_maximum"], values["unsafe_maximum"])
        pad = 0.08 * (upper - lower)
        ax.set_xlim(lower - pad, upper + pad)
        ax.grid(axis="x", color="#dce3df", linewidth=0.8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="lower right", fontsize=9)

    max_diff = max(
        validation["maximum_camera_permutation_difference"],
        validation["maximum_formal_independent_feature_difference"],
        validation["maximum_separation_difference"],
    )
    fig.text(
        0.055,
        0.10,
        f"Independent recomputation passes; maximum reported difference = {max_diff:.2e}.",
        color="#42514c",
        fontsize=10,
    )
    fig.text(
        0.055,
        0.055,
        "Decision: close this scalar camera-jackknife fallback; no deployment, resource, or real-BOST claim.",
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
