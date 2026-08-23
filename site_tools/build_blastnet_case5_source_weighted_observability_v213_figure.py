#!/usr/bin/env python3
"""Build the redacted v213 source-weighted observability figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "assets/figures/blastnet_case5_source_weighted_observability_v213.png"
)
FAMILIES = ("Supplied 9", "Virtual ring 9", "Virtual ring 12\n(diagnostic)")
COLORS = ("#d97706", "#2563eb", "#0f8a72")
SOURCE_WEIGHTED = (
    np.asarray(
        [
            0.5281140815983209,
            0.5281140815983209,
            0.603597483922606,
            0.5650275504060371,
            0.5218983089869261,
            0.5861463599312317,
            0.6009737856800578,
            0.5281144017287176,
            0.5499599528318576,
            0.5163212094083833,
            0.5134731135409792,
            0.5281140815983209,
            0.5904968235949026,
        ]
    ),
    np.asarray(
        [
            0.9864604616591823,
            1.038116355123667,
            1.0267162833240397,
            0.8902768620487518,
            0.9800514013737411,
            0.9363037902962249,
            0.9599293284744601,
            0.9603329183542352,
            0.9823375171686584,
            0.9678140440919724,
            1.0350051024726228,
            0.936571101160451,
            0.9786919783513098,
        ]
    ),
    np.asarray(
        [
            1.0035109571806675,
            1.0089292628324509,
            1.000911496347898,
            0.990736540621029,
            1.0007550214018581,
            1.0110777050724067,
            0.9914056714437174,
            1.0019895107041576,
            1.006200037984106,
            1.001672130142775,
            0.996344489531054,
            1.0082671022684584,
            1.004049911732935,
        ]
    ),
)
SOURCE_BLIND = (
    np.asarray(
        [
            0.02146072143939376,
            0.02146072143939376,
            0.09546337783981249,
            0.10524302536339249,
            0.011857128995937239,
            0.08719189913088762,
            0.09517462599680361,
            0.02146057333988629,
            0.13158748032841025,
            0.011937048484721633,
            0.011961643743599313,
            0.02146072143939376,
            0.09902616796099835,
        ]
    ),
    np.asarray(
        [
            0.3197769327139064,
            0.30756491380640427,
            0.33528434505563154,
            0.109315771300406,
            0.32588575367865225,
            0.3400795270294155,
            0.28142037679986964,
            0.13157091581967736,
            0.3314819978056536,
            0.2612243936313664,
            0.14432922421321795,
            0.3314652803771565,
            0.26860806861696457,
        ]
    ),
    np.asarray(
        [
            0.4056102250496042,
            0.3939928942564759,
            0.4150855749752553,
            0.4247564953166286,
            0.4035968172261717,
            0.3960635138967179,
            0.4273163822142751,
            0.4148945018082526,
            0.396334875293179,
            0.42762720935676385,
            0.41509570983778316,
            0.40237107499814817,
            0.405331783334788,
        ]
    ),
)


def _scatter(ax: plt.Axes, values: tuple[np.ndarray, ...]) -> None:
    offsets = np.linspace(-0.08, 0.08, 13)
    for family_index, rows in enumerate(values):
        ax.scatter(
            family_index + offsets,
            rows,
            s=44,
            color=COLORS[family_index],
            alpha=0.9,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        ax.hlines(
            float(np.median(rows)),
            family_index - 0.24,
            family_index + 0.24,
            color="#182230",
            linewidth=2.4,
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
            "font.size": 10,
            "figure.facecolor": "#f4f7f6",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#c8d2cf",
            "axes.labelcolor": "#25352f",
            "xtick.color": "#42514c",
            "ytick.color": "#42514c",
            "axes.titleweight": "bold",
        }
    )
    fig = plt.figure(figsize=(15.4, 6.2))
    grid = fig.add_gridspec(
        1,
        3,
        left=0.055,
        right=0.985,
        top=0.79,
        bottom=0.19,
        wspace=0.38,
    )
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    fig.suptitle(
        "v213 actual-source spectral observability attribution",
        x=0.025,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#162033",
    )

    _scatter(axes[0], SOURCE_BLIND)
    axes[0].set_title("Source-blind floor still overlaps")
    axes[0].set_ylabel("Trace-normalized spectral floor")
    axes[0].annotate(
        "167 / 169",
        xy=(0.05, 0.91),
        xycoords="axes fraction",
        ha="left",
        fontsize=13,
        fontweight="bold",
        color="#b45309",
    )

    _scatter(axes[1], SOURCE_WEIGHTED)
    axes[1].set_title("Actual-source direction separates")
    axes[1].set_ylabel("Worst-frame harmonic observability")
    axes[1].axhspan(
        float(np.max(SOURCE_WEIGHTED[0])),
        float(np.min(SOURCE_WEIGHTED[1])),
        color="#d9f2e9",
        alpha=0.9,
        zorder=0,
    )
    axes[1].annotate(
        "169 / 169\nstrict gap 0.28668",
        xy=(0.97, 0.08),
        xycoords="axes fraction",
        ha="right",
        fontsize=12,
        fontweight="bold",
        color="#08765f",
    )

    axes[2].axis("off")
    axes[2].set_title("What changed scientifically", pad=8)
    statements = (
        (0.82, "v210 source-blind geometry", "167/169; two family overlaps", "#b45309"),
        (0.59, "v213 actual-source alignment", "169/169; no family overlap", "#1d63d8"),
        (0.36, "Low-64 source energy", "77.69% to 79.38% per frame", "#0f806b"),
    )
    for y, title, detail, color in statements:
        axes[2].text(
            0.02,
            y,
            title,
            color=color,
            fontsize=11.5,
            fontweight="bold",
            transform=axes[2].transAxes,
        )
        axes[2].text(
            0.02,
            y - 0.08,
            detail,
            color="#253044",
            fontsize=10.5,
            transform=axes[2].transAxes,
        )
    axes[2].text(
        0.02,
        0.08,
        "Mechanism evidence only. Still false:\ndeployable warm start, speedup, external gate, real BOST.",
        color="#586272",
        fontsize=10,
        fontweight="bold",
        linespacing=1.35,
        transform=axes[2].transAxes,
    )

    fig.text(
        0.025,
        0.045,
        "42 opened density frames | 39 geometries | 64 fixed low modes | 19/19 independent checks",
        fontsize=9.5,
        color="#526071",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    with Image.open(OUTPUT) as image:
        image.convert("RGB").save(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(build())
