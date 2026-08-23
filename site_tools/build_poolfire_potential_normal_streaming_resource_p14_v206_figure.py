#!/usr/bin/env python3
"""Build the redacted v206 streamed-resource evidence figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets/figures/poolfire_potential_normal_streaming_resource_p14_v206.png"


def _ratio_panel(ax: plt.Axes, title: str, p50: tuple[float, ...], p90: tuple[float, ...]) -> None:
    labels = ("Outer\nwall", "Setup\nwall", "Worker\nRSS", "Tree\nRSS", "Pipeline\nRSS")
    x = np.arange(len(labels))
    width = 0.34
    p50_bars = ax.bar(x - width / 2, p50, width, color="#258a7a", label="p50")
    p90_bars = ax.bar(x + width / 2, p90, width, color="#287bb5", label="p90-higher")
    ax.axhline(1.0, color="#c95752", linestyle="--", linewidth=1.2, label="Control = 1")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.62, 1.06)
    ax.set_ylabel("Paired primary / control ratio")
    ax.set_title(title)
    ax.grid(axis="y", color="#e5e9e6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    for bars in (p50_bars, p90_bars):
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.008,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                rotation=90,
            )


def build() -> Path:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "figure.facecolor": "#f7f8f5",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#c9d0cc",
            "axes.titleweight": "bold",
            "axes.labelcolor": "#24302d",
            "xtick.color": "#3f4c48",
            "ytick.color": "#3f4c48",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.8))
    fig.subplots_adjust(left=0.055, right=0.985, top=0.82, bottom=0.22, wspace=0.16)
    fig.suptitle(
        "v206: streamed setup resource headroom on exposed p14, all nine cameras only",
        fontsize=14.5,
        fontweight="bold",
        color="#1d2a27",
    )

    error_labels = ("Coordinates\nvs formal", "Factor\nreconstruction", "Regularization")
    errors = np.asarray((1.4795374899614921e-12, 2.0678989589435617e-13, 1.252346784746338e-13))
    bars = axes[0].bar(error_labels, errors, color=("#287bb5", "#584f9e", "#258a7a"), width=0.62)
    axes[0].set_yscale("log")
    axes[0].set_ylim(1e-14, 3e-9)
    axes[0].axhline(1e-9, color="#c95752", linestyle="--", linewidth=1.2, label="Frozen 1e-9 gate")
    axes[0].set_ylabel("Maximum relative difference")
    axes[0].set_title("Independent streamed-setup equivalence")
    axes[0].grid(axis="y", which="both", color="#e5e9e6", linewidth=0.8)
    axes[0].set_axisbelow(True)
    axes[0].legend(frameon=False, loc="upper right")
    for bar, value in zip(bars, errors, strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.45,
            f"{value:.2e}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    _ratio_panel(
        axes[1],
        "Streaming compact K1 vs dense K1",
        (0.8602935674, 0.7800770691, 0.6886209996, 0.6907444012, 0.7100309531),
        (0.8728758587, 0.7972855405, 0.7159629889, 0.7192390718, 0.7370300583),
    )
    _ratio_panel(
        axes[2],
        "Streaming compact K1 vs dense K2",
        (0.7395327153, 0.7804364096, 0.6882850797, 0.6933994659, 0.7122364390),
        (0.7503400179, 0.7990946455, 0.7123550596, 0.7148566567, 0.7338916581),
    )
    fig.text(
        0.5,
        0.035,
        "Fresh workers, randomized adjacent complete blocks; lower is better. This is post-open p14 resource evidence, not global speedup or validation.",
        ha="center",
        color="#55615d",
        fontsize=9,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    with Image.open(OUTPUT) as image:
        image.convert("RGB").save(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(build())
