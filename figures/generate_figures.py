#!/usr/bin/env python3
"""Generate simple original figures for the OERF thesis workbench."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).resolve().parent

COLORS = {
    "ink": "#1f2a2e",
    "muted": "#5f6e76",
    "line": "#c9d5d8",
    "green": "#0f766e",
    "blue": "#2563eb",
    "amber": "#b7791f",
    "red": "#b91c1c",
    "bg": "#f7faf9",
}


def setup(width: float = 12, height: float = 6):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    return fig, ax


def box(ax, xy, w, h, title, subtitle="", fc="#ffffff", ec=None, title_size=12):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.3,
        edgecolor=ec or COLORS["line"],
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=title_size, color=COLORS["ink"], weight="bold")
    if subtitle:
        ax.text(x + w / 2, y + h * 0.30, subtitle, ha="center", va="center", fontsize=9.5, color=COLORS["muted"])
    return patch


def arrow(ax, start, end, color=None, rad=0.0, lw=1.8):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=lw,
            color=color or COLORS["muted"],
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def title(ax, text, sub=None):
    ax.text(0.03, 0.94, text, fontsize=18, weight="bold", color=COLORS["ink"], ha="left", va="top")
    if sub:
        ax.text(0.03, 0.885, sub, fontsize=10.5, color=COLORS["muted"], ha="left", va="top")


def save(fig, name):
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def physical_chain():
    fig, ax = setup(13.5, 5.2)
    title(ax, "BOST physical chain", "From reacting-flow thermodynamics to image displacement and reconstruction")
    labels = [
        ("Temperature", "T(x,y,z,t)"),
        ("Density", "rho field"),
        ("Refractive index", "n - 1 = K rho"),
        ("Ray deflection", "integral of grad n"),
        ("Image displacement", "BOS / optical flow"),
        ("3D reconstruction", "BOST / NeRIF"),
    ]
    xs = [0.04, 0.20, 0.36, 0.52, 0.68, 0.84]
    for i, (main, sub) in enumerate(labels):
        fc = "#e7f5f2" if i in (2, 5) else "#ffffff"
        box(ax, (xs[i], 0.48), 0.12, 0.18, main, sub, fc=fc, ec="#9ccac2")
        if i < len(labels) - 1:
            arrow(ax, (xs[i] + 0.12, 0.57), (xs[i + 1], 0.57), color=COLORS["green"])
    box(ax, (0.10, 0.16), 0.30, 0.16, "What the experiment sees", "multi-view background displacement", fc="#f8fbff", ec="#a9c4f5")
    box(ax, (0.60, 0.16), 0.30, 0.16, "What the thesis reconstructs", "continuous refractive-index / density field", fc="#fffaf0", ec="#e5c785")
    arrow(ax, (0.40, 0.24), (0.60, 0.24), color=COLORS["amber"])
    ax.text(0.50, 0.07, "Key message: the project is an optical-diagnostics inverse problem, not generic AI.", ha="center", fontsize=11, color=COLORS["ink"], weight="bold")
    save(fig, "bost_physical_chain.png")


def oerf_position():
    fig, ax = setup(11.5, 7.0)
    title(ax, "Where this thesis fits inside OERF", "Choose the computational reconstruction layer while staying connected to experiments")
    centers = {
        "Optical diagnostics": (0.24, 0.66),
        "Computational imaging": (0.52, 0.66),
        "AI / neural fields": (0.80, 0.66),
        "Reacting-flow physics": (0.24, 0.36),
        "Data assimilation": (0.52, 0.36),
        "Thesis role": (0.80, 0.36),
    }
    subtitles = {
        "Optical diagnostics": "BOS, PLIF, TAS, holography",
        "Computational imaging": "tomography, light field, inverse problems",
        "AI / neural fields": "NeRIF, NeDF, 4D priors",
        "Reacting-flow physics": "flame / aero-engine / hypersonic flow",
        "Data assimilation": "measurement + model fusion",
        "Thesis role": "reproducible reconstruction + error analysis",
    }
    for name, (x, y) in centers.items():
        fc = "#e7f5f2" if name == "Thesis role" else "#ffffff"
        ec = COLORS["green"] if name == "Thesis role" else COLORS["line"]
        box(ax, (x - 0.12, y - 0.07), 0.24, 0.14, name, subtitles[name], fc=fc, ec=ec, title_size=11)
    connections = [
        ("Optical diagnostics", "Computational imaging"),
        ("Computational imaging", "AI / neural fields"),
        ("Reacting-flow physics", "Computational imaging"),
        ("Computational imaging", "Data assimilation"),
        ("AI / neural fields", "Thesis role"),
        ("Data assimilation", "Thesis role"),
        ("Optical diagnostics", "Thesis role"),
    ]
    for a, b in connections:
        ax1, ay1 = centers[a]
        bx, by = centers[b]
        arrow(ax, (ax1, ay1), (bx, by), color=COLORS["muted"], lw=1.2)
    ax.text(0.50, 0.12, "Best thesis identity: BOST/NeRIF reproducibility, robustness, data interface, and reporting tools.", ha="center", fontsize=11, color=COLORS["ink"], weight="bold")
    save(fig, "oerf_position_map.png")


def nerif_pipeline():
    fig, ax = setup(13.5, 6.5)
    title(ax, "NeRIF-style reconstruction pipeline", "A thesis-safe abstraction of the full Physics of Fluids method")
    top = [
        ("Coordinates", "(x,y,z) samples"),
        ("Encoding", "Fourier / hash"),
        ("Coordinate field", "MLP or RFF model"),
        ("n and grad n", "physical field"),
        ("Ray integration", "predict displacement"),
        ("Loss", "reprojection + reg."),
    ]
    xs = [0.04, 0.20, 0.36, 0.52, 0.68, 0.84]
    for i, (main, sub) in enumerate(top):
        fc = "#e7f5f2" if i in (2, 3) else "#ffffff"
        box(ax, (xs[i], 0.58), 0.12, 0.16, main, sub, fc=fc, ec="#9ccac2")
        if i < len(top) - 1:
            arrow(ax, (xs[i] + 0.12, 0.66), (xs[i + 1], 0.66), color=COLORS["green"])
    box(ax, (0.10, 0.24), 0.20, 0.14, "Observed data", "multi-view BOS displacement", fc="#f8fbff", ec="#a9c4f5")
    box(ax, (0.40, 0.24), 0.20, 0.14, "Baseline", "voxel / regularized inverse", fc="#fffaf0", ec="#e5c785")
    box(ax, (0.70, 0.24), 0.20, 0.14, "Validation", "L2, CC, SSIM, reprojection", fc="#fff5f5", ec="#e8aaaa")
    arrow(ax, (0.20, 0.38), (0.72, 0.58), color=COLORS["blue"], rad=-0.12)
    arrow(ax, (0.84, 0.58), (0.80, 0.38), color=COLORS["red"], rad=-0.15)
    arrow(ax, (0.50, 0.38), (0.80, 0.38), color=COLORS["amber"])
    ax.text(0.50, 0.09, "M0 demo currently implements the same logic in 2D with random Fourier features; full NeRIF is a later upgrade.", ha="center", fontsize=10.5, color=COLORS["ink"], weight="bold")
    save(fig, "nerif_pipeline.png")


def data_interface():
    fig, ax = setup(12.5, 6.2)
    title(ax, "Data interface to request from He Yuanzhe", "Ask for files and coordinate conventions, not vague data")
    columns = [
        ("Raw images", ["flow-off", "flow-on", "view id", "bit depth"]),
        ("Geometry", ["intrinsics", "extrinsics", "mask", "volume bounds"]),
        ("Displacement", ["u,v fields", "window size", "noise estimate", "units"]),
        ("Reference", ["voxel result", "NeRIF slices", "metrics", "allowed figures"]),
        ("Upgrade data", ["PIV pairs", "timestamps", "4D frames", "calibration"]),
    ]
    for i, (head, lines) in enumerate(columns):
        x = 0.04 + i * 0.19
        box(ax, (x, 0.56), 0.15, 0.12, head, "", fc="#e7f5f2" if i in (0, 2) else "#ffffff", ec="#9ccac2")
        for j, line in enumerate(lines):
            box(ax, (x, 0.40 - j * 0.075), 0.15, 0.052, line, "", fc="#ffffff", ec=COLORS["line"], title_size=8.5)
        if i < len(columns) - 1:
            arrow(ax, (x + 0.15, 0.62), (x + 0.19, 0.62), color=COLORS["green"], lw=1.4)
    ax.text(0.50, 0.06, "Minimum useful package: one 9-view sample + masks + geometry + displacement + one reference reconstruction.", ha="center", fontsize=10.5, color=COLORS["ink"], weight="bold")
    save(fig, "data_interface_checklist.png")


def decision_tree():
    fig, ax = setup(12.0, 7.0)
    title(ax, "Topic decision after meeting He Yuanzhe", "Keep the same core; choose the second-stage branch by data availability")
    box(ax, (0.36, 0.70), 0.28, 0.11, "Core thesis", "BOST / NeRIF reproducible reconstruction", fc="#e7f5f2", ec=COLORS["green"])
    branches = [
        ((0.08, 0.45), "BOST sample available", "real-data interface + robustness"),
        ((0.36, 0.45), "No internal data yet", "synthetic + open-source BOS benchmark"),
        ((0.64, 0.45), "PIV-BOST available", "2D PIV compensation toy"),
        ((0.36, 0.20), "4D data / need", "low-rank temporal prior toy"),
    ]
    for (x, y), main, sub in branches:
        box(ax, (x, y), 0.28, 0.12, main, sub, fc="#ffffff", ec=COLORS["line"], title_size=10.5)
        arrow(ax, (0.50, 0.70), (x + 0.14, y + 0.12), color=COLORS["muted"], rad=0.04)
    box(ax, (0.08, 0.075), 0.84, 0.075, "Always preserve the baseline deliverable", "synthetic phantom + forward model + baseline + coordinate field + metrics + figures", fc="#fffaf0", ec="#e5c785")
    ax.text(0.50, 0.025, "Do not redefine the thesis around a high-risk branch until data and mentor need are confirmed.", ha="center", fontsize=10.5, color=COLORS["red"], weight="bold")
    save(fig, "topic_decision_tree.png")


def roadmap():
    fig, ax = setup(12.8, 6.2)
    title(ax, "Three-month pre-opening roadmap", "Each phase must leave a visible artifact")
    phases = [
        ("Weeks 1-2", "Fluid/BOST variables", "chain diagram + glossary"),
        ("Weeks 3-4", "Forward model", "2D/3D phantom + deflection"),
        ("Weeks 5-6", "Baseline + metrics", "tomography baseline + figures"),
        ("Weeks 7-8", "NeRIF-style inverse", "coordinate-field reconstruction"),
        ("Weeks 9-10", "Robustness / branch", "views, noise, PIV or 4D toy"),
        ("Weeks 11-12", "Opening package", "PPT + demo + data request"),
    ]
    for i, (week, main, artifact) in enumerate(phases):
        x = 0.04 + i * 0.155
        y = 0.52 if i % 2 == 0 else 0.32
        box(ax, (x, y), 0.135, 0.16, week, main, fc="#e7f5f2" if i in (0, 3) else "#ffffff", ec="#9ccac2")
        ax.text(x + 0.0675, y - 0.035, artifact, ha="center", va="top", fontsize=8.5, color=COLORS["muted"])
        if i < len(phases) - 1:
            arrow(ax, (x + 0.135, y + 0.08), (x + 0.155, (0.52 if (i + 1) % 2 == 0 else 0.32) + 0.08), color=COLORS["green"], lw=1.4)
    ax.text(0.50, 0.12, "Weekly rule: one figure + one runnable artifact + one question for He Yuanzhe.", ha="center", fontsize=11, color=COLORS["ink"], weight="bold")
    save(fig, "three_month_roadmap.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    physical_chain()
    oerf_position()
    nerif_pipeline()
    data_interface()
    decision_tree()
    roadmap()
    for path in sorted(OUT.glob("*.png")):
        print(path.name)


if __name__ == "__main__":
    main()
