#!/usr/bin/env python3
"""M2 robustness scan for the synthetic BOST/coordinate-prior demos.

The goal is not to reproduce the full NeRIF paper. The goal is to create a
compact, repeatable experiment matrix that can be discussed with He Yuanzhe:
view count, noise level, and coordinate-field capacity all change whether a
coordinate prior is useful.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
RESULTS = ROOT / "results"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo_m1.run_m1_3d_stack_bost import (
    baseline_stack,
    make_volume,
    metrics,
    random_fourier_features_3d,
    synthesize_deflection_stack,
)


class CoordinateRegularizer:
    def __init__(self, n: int, nz: int, num_freq: int = 180, ridge: float = 1e-3) -> None:
        self.n = n
        self.nz = nz
        self.phi = random_fourier_features_3d(n=n, nz=nz, num_freq=num_freq)
        normal = self.phi.T @ self.phi + ridge * np.eye(self.phi.shape[1])
        self.solve_matrix = np.linalg.solve(normal, self.phi.T)

    def apply(self, stack_reconstruction: np.ndarray) -> np.ndarray:
        weights = self.solve_matrix @ stack_reconstruction.reshape(-1)
        return (self.phi @ weights).reshape(self.nz, self.n, self.n)


def write_rows(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def add_noise(deflection: np.ndarray, noise_level: float, rng: np.random.Generator) -> np.ndarray:
    if noise_level == 0:
        return deflection.copy()
    scale = float(np.std(deflection))
    return deflection + noise_level * scale * rng.normal(size=deflection.shape)


def run_noise_view_scan(reference: np.ndarray, n: int, nz: int) -> list[dict[str, float | int | str]]:
    view_counts = [3, 5, 7, 9]
    noise_levels = [0.0, 0.03, 0.06, 0.10]
    regularizer = CoordinateRegularizer(n=n, nz=nz, num_freq=180, ridge=1e-3)
    rng = np.random.default_rng(20260706)
    rows: list[dict[str, float | int | str]] = []

    for views in view_counts:
        angles = np.linspace(0, 180, views, endpoint=False)
        clean_deflection = synthesize_deflection_stack(reference, angles)
        for noise in noise_levels:
            noisy_deflection = add_noise(clean_deflection, noise, rng)
            base = baseline_stack(noisy_deflection, angles, n)
            coord = regularizer.apply(base)
            base_l2 = metrics("baseline", base, reference)["rel_l2"]
            coord_l2 = metrics("coord_regularized", coord, reference)["rel_l2"]
            rows.append(
                {
                    "views": views,
                    "noise_level": noise,
                    "baseline_rel_l2": base_l2,
                    "coord_regularized_rel_l2": coord_l2,
                    "improvement_baseline_minus_coord": base_l2 - coord_l2,
                    "winner": "coord_regularized" if coord_l2 < base_l2 else "baseline",
                }
            )
    return rows


def run_capacity_scan(reference: np.ndarray, n: int, nz: int) -> list[dict[str, float | int | str]]:
    views = 5
    noise = 0.06
    angles = np.linspace(0, 180, views, endpoint=False)
    rng = np.random.default_rng(606)
    deflection = add_noise(synthesize_deflection_stack(reference, angles), noise, rng)
    base = baseline_stack(deflection, angles, n)
    base_l2 = metrics("baseline", base, reference)["rel_l2"]
    rows: list[dict[str, float | int | str]] = []

    for num_freq in [40, 80, 120, 180, 260, 360]:
        regularizer = CoordinateRegularizer(n=n, nz=nz, num_freq=num_freq, ridge=1e-3)
        coord = regularizer.apply(base)
        coord_l2 = metrics("coord_regularized", coord, reference)["rel_l2"]
        rows.append(
            {
                "views": views,
                "noise_level": noise,
                "num_freq": num_freq,
                "baseline_rel_l2": base_l2,
                "coord_regularized_rel_l2": coord_l2,
                "improvement_baseline_minus_coord": base_l2 - coord_l2,
            }
        )
    return rows


def plot_noise_view_scan(rows: list[dict[str, float | int | str]]) -> None:
    view_counts = sorted({int(row["views"]) for row in rows})
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), sharey=True, constrained_layout=True)
    for ax, views in zip(axes.flat, view_counts):
        subset = [row for row in rows if int(row["views"]) == views]
        xs = [float(row["noise_level"]) for row in subset]
        base = [float(row["baseline_rel_l2"]) for row in subset]
        coord = [float(row["coord_regularized_rel_l2"]) for row in subset]
        ax.plot(xs, base, marker="o", label="stack baseline")
        ax.plot(xs, coord, marker="o", label="3D coord regularized")
        ax.set_title(f"{views} views")
        ax.set_xlabel("Deflection noise level")
        ax.set_ylabel("Volume relative L2")
        ax.grid(True, alpha=0.3)
    axes.flat[0].legend()
    fig.suptitle("M2 robustness scan: noise x view count", fontsize=13)
    fig.savefig(RESULTS / "m2_noise_view_scan.png", dpi=180)
    plt.close(fig)


def plot_improvement_heatmap(rows: list[dict[str, float | int | str]]) -> None:
    view_counts = sorted({int(row["views"]) for row in rows})
    noise_levels = sorted({float(row["noise_level"]) for row in rows})
    grid = np.zeros((len(noise_levels), len(view_counts)))
    for i, noise in enumerate(noise_levels):
        for j, views in enumerate(view_counts):
            match = next(row for row in rows if int(row["views"]) == views and float(row["noise_level"]) == noise)
            grid[i, j] = float(match["improvement_baseline_minus_coord"])

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    vmax = max(abs(float(grid.min())), abs(float(grid.max())), 1e-6)
    im = ax.imshow(grid, cmap="coolwarm", vmin=-vmax, vmax=vmax, origin="lower")
    ax.set_xticks(range(len(view_counts)), labels=view_counts)
    ax.set_yticks(range(len(noise_levels)), labels=[f"{x:.2f}" for x in noise_levels])
    ax.set_xlabel("Number of views")
    ax.set_ylabel("Deflection noise level")
    ax.set_title("Positive values mean coordinate regularization improves L2")
    for i in range(len(noise_levels)):
        for j in range(len(view_counts)):
            ax.text(j, i, f"{grid[i, j]:+.3f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="baseline L2 - coord L2")
    fig.savefig(RESULTS / "m2_improvement_heatmap.png", dpi=180)
    plt.close(fig)


def plot_capacity_scan(rows: list[dict[str, float | int | str]]) -> None:
    xs = [int(row["num_freq"]) for row in rows]
    coord = [float(row["coord_regularized_rel_l2"]) for row in rows]
    base = float(rows[0]["baseline_rel_l2"])
    fig, ax = plt.subplots(figsize=(7.8, 4.6), constrained_layout=True)
    ax.axhline(base, color="tab:blue", linestyle="--", label="stack baseline")
    ax.plot(xs, coord, color="tab:orange", marker="o", label="3D coord regularized")
    ax.set_xlabel("Random Fourier feature count")
    ax.set_ylabel("Volume relative L2")
    ax.set_title("M2 capacity scan: 5 views, noise level 0.06")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(RESULTS / "m2_capacity_scan.png", dpi=180)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    n = 34
    nz = 16
    reference = make_volume(n=n, nz=nz)

    grid_rows = run_noise_view_scan(reference, n=n, nz=nz)
    write_rows(RESULTS / "m2_metrics_grid.csv", grid_rows)
    plot_noise_view_scan(grid_rows)
    plot_improvement_heatmap(grid_rows)

    capacity_rows = run_capacity_scan(reference, n=n, nz=nz)
    write_rows(RESULTS / "m2_capacity_metrics.csv", capacity_rows)
    plot_capacity_scan(capacity_rows)

    winners = {}
    for row in grid_rows:
        winners[row["winner"]] = winners.get(row["winner"], 0) + 1
    print("M2 robustness scan complete")
    print(f"grid rows: {len(grid_rows)}")
    print(f"winner counts: {winners}")
    print(f"results: {RESULTS}")


if __name__ == "__main__":
    main()
