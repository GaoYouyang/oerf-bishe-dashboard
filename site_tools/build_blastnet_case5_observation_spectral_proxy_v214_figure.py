#!/usr/bin/env python3
"""Build the redacted v214 observation spectral-proxy figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from site_tools.build_blastnet_case5_source_weighted_observability_v213_figure import (
    SOURCE_BLIND,
    SOURCE_WEIGHTED,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets/figures/blastnet_case5_observation_spectral_proxy_v214.png"
FAMILIES = ("Supplied 9", "Virtual ring 9", "Virtual ring 12\n(diagnostic)")
COLORS = ("#c86b22", "#2469b2", "#16836d")
OBSERVATION_PROXY = (
    np.asarray(
        [
            0.32483357212448377,
            0.32483357212448377,
            0.5918615724542198,
            0.5057521593371227,
            0.19783320416375275,
            0.5686757443793572,
            0.5456448876275309,
            0.32482156001007834,
            0.5358189199040678,
            0.21281021763303876,
            0.29603665460939627,
            0.32483357212448377,
            0.5230114392170737,
        ]
    ),
    np.asarray(
        [
            1.0754598313747312,
            1.117027025411324,
            1.1066637466107987,
            0.9891662523122507,
            1.078047061262989,
            1.026074436296239,
            1.0417177600870065,
            1.0013493994096392,
            1.085934059407244,
            1.0473396218247233,
            1.11455022852757,
            1.024913086262889,
            1.0657380616646421,
        ]
    ),
    np.asarray(
        [
            1.1018383672375722,
            1.105835101604453,
            1.0981466345952255,
            1.0846508328504774,
            1.0992295183962086,
            1.1087213163078915,
            1.0883119028503647,
            1.0993668295638304,
            1.1089644797500136,
            1.092494077411477,
            1.0923439677243407,
            1.1043354881228225,
            1.101940951924991,
        ]
    ),
)


def _scatter(ax: plt.Axes, values: tuple[np.ndarray, ...]) -> None:
    offsets = np.linspace(-0.08, 0.08, 13)
    for family_index, rows in enumerate(values):
        ax.scatter(
            family_index + offsets,
            rows,
            s=43,
            color=COLORS[family_index],
            edgecolor="white",
            linewidth=0.7,
            alpha=0.92,
            zorder=3,
        )
        ax.hlines(
            float(np.median(rows)),
            family_index - 0.24,
            family_index + 0.24,
            color="#1d2934",
            linewidth=2.3,
            zorder=4,
        )
    ax.set_xticks(range(3), FAMILIES)
    ax.grid(axis="y", color="#d9e1de", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def build() -> Path:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "figure.facecolor": "#f4f7f6",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#c8d2cf",
            "axes.labelcolor": "#25352f",
            "xtick.color": "#42514c",
            "ytick.color": "#42514c",
            "axes.titleweight": "bold",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 6.4))
    fig.subplots_adjust(left=0.055, right=0.985, top=0.78, bottom=0.2, wspace=0.34)
    fig.suptitle(
        "v214: observation-only spectral alignment recovers strict separation",
        x=0.03,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#162033",
    )

    panels = (
        (SOURCE_BLIND, "Source-blind geometry", "Spectral floor", "167 / 169"),
        (SOURCE_WEIGHTED, "v213 truth-aware source", "Harmonic observability", "169 / 169"),
        (OBSERVATION_PROXY, "v214 observation + geometry", "Proxy harmonic observability", "169 / 169"),
    )
    for axis, (values, title, ylabel, badge) in zip(axes, panels, strict=True):
        _scatter(axis, values)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.text(
            0.04,
            0.92,
            badge,
            transform=axis.transAxes,
            fontsize=13,
            fontweight="bold",
            color="#136d5b" if badge == "169 / 169" else "#a95b20",
        )
    axes[2].axhspan(
        float(np.max(OBSERVATION_PROXY[0])),
        float(np.min(OBSERVATION_PROXY[1])),
        color="#d5efe7",
        alpha=0.8,
        zorder=0,
    )
    axes[2].text(
        0.96,
        0.08,
        "strict gap 0.39730",
        transform=axes[2].transAxes,
        ha="right",
        fontsize=11.5,
        fontweight="bold",
        color="#136d5b",
    )
    fig.text(
        0.03,
        0.06,
        "42 opened frames | 39 geometries | proxy input: known M and current 2D y only | 19/19 independent checks",
        fontsize=9.5,
        color="#526071",
    )
    fig.text(
        0.03,
        0.018,
        "Post-open mechanism evidence only: no warm-start replay, matched accuracy, resource speedup, external gate, or real BOST.",
        fontsize=9.5,
        color="#6a4e42",
        fontweight="bold",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    with Image.open(OUTPUT) as image:
        image.convert("RGB").save(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(build())
