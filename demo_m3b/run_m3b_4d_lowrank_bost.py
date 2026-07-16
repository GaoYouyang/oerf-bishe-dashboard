#!/usr/bin/env python3
"""M3B 4D BOST low-rank temporal toy.

This is a small, repeatable experiment inspired by tensor-decomposition 4D BOST.
It generates a moving 3D refractive-index volume, reconstructs each frame with a
sparse-view stack baseline, then applies a low-rank temporal model with SVD.
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

from demo_m1.run_m1_3d_stack_bost import baseline_stack, metrics, synthesize_deflection_stack


def make_grid_3d(n: int, nz: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(-1.0, 1.0, n)
    y = np.linspace(-1.0, 1.0, n)
    z = np.linspace(-1.0, 1.0, nz)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    return xx, yy, zz


def gaussian3d(xx: np.ndarray, yy: np.ndarray, zz: np.ndarray, center, widths, amp: float) -> np.ndarray:
    x0, y0, z0 = center
    sx, sy, sz = widths
    return amp * np.exp(-(((xx - x0) ** 2) / (2 * sx**2) + ((yy - y0) ** 2) / (2 * sy**2) + ((zz - z0) ** 2) / (2 * sz**2)))


def make_4d_sequence(n: int = 30, nz: int = 12, nt: int = 18, dynamics: str = "smooth") -> np.ndarray:
    """Generate a compact 3D+time phantom with selectable dynamics.

    ``smooth`` preserves the original M3B sequence exactly. The other modes
    are used by the six-axis sweep to probe faster motion, frequency drift,
    and a localized transient without changing the reconstruction code.
    """
    if dynamics not in {"smooth", "fast", "chirp", "transient"}:
        raise ValueError(f"unknown dynamics mode: {dynamics}")
    xx, yy, zz = make_grid_3d(n, nz)
    frames = []
    for t in range(nt):
        fraction = t / max(nt - 1, 1)
        if dynamics == "fast":
            phase = 4 * np.pi * t / nt
        elif dynamics == "chirp":
            phase = 2 * np.pi * (0.35 * fraction + 1.45 * fraction**2)
        else:
            phase = 2 * np.pi * t / nt
        c1 = (-0.30 + 0.16 * np.sin(phase), 0.06 * np.cos(phase), -0.18 + 0.10 * np.sin(phase + 0.5))
        c2 = (0.27 + 0.07 * np.sin(phase + 1.1), -0.22 + 0.13 * np.cos(phase), 0.20 * np.cos(phase + 0.2))
        width_scale = 1.0 + 0.10 * np.sin(2 * phase)
        volume = (
            gaussian3d(xx, yy, zz, c1, (0.20 * width_scale, 0.30, 0.42), 1.0)
            + gaussian3d(xx, yy, zz, c2, (0.17, 0.22 * width_scale, 0.34), 0.72)
        )
        sheet_center = 0.12 * np.sin(3.5 * yy + 2.0 * zz + phase)
        sheet = 0.22 * np.exp(-((xx - sheet_center) ** 2) / (2 * 0.052**2)) * np.exp(-(yy**2) / 0.82) * np.exp(-(zz**2) / 1.25)
        volume = volume + sheet
        if dynamics == "transient":
            event_center = 0.56 * max(nt - 1, 1)
            event_weight = np.exp(-0.5 * ((t - event_center) / 0.75) ** 2)
            event = gaussian3d(xx, yy, zz, (0.03, 0.34, -0.08), (0.11, 0.13, 0.18), 0.62)
            volume = volume + event_weight * event
        volume -= volume.min()
        volume /= volume.max() + 1e-12
        frames.append(volume)
    return np.stack(frames, axis=0)


def add_deflection_noise(deflection: np.ndarray, rng: np.random.Generator, noise_level: float = 0.055) -> np.ndarray:
    return deflection + noise_level * float(np.std(deflection)) * rng.normal(size=deflection.shape)


def reconstruct_framewise(reference: np.ndarray, n: int, angles: np.ndarray, noise_level: float = 0.055) -> np.ndarray:
    rng = np.random.default_rng(20260706)
    frames = []
    for frame in reference:
        deflection = synthesize_deflection_stack(frame, angles)
        noisy = add_deflection_noise(deflection, rng=rng, noise_level=noise_level)
        frames.append(baseline_stack(noisy, angles, n))
    return np.stack(frames, axis=0)


def low_rank_temporal(sequence: np.ndarray, rank: int) -> np.ndarray:
    nt = sequence.shape[0]
    matrix = sequence.reshape(nt, -1)
    mean = matrix.mean(axis=0, keepdims=True)
    centered = matrix - mean
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    rank = min(rank, len(s))
    reconstructed = (u[:, :rank] * s[:rank]) @ vt[:rank] + mean
    return reconstructed.reshape(sequence.shape)


def centroid_x(sequence: np.ndarray) -> np.ndarray:
    n = sequence.shape[-1]
    x = np.linspace(-1.0, 1.0, n)
    weights = np.maximum(sequence, 0.0)
    numerator = (weights * x.reshape(1, 1, 1, n)).sum(axis=(1, 2, 3))
    denominator = weights.sum(axis=(1, 2, 3)) + 1e-12
    return numerator / denominator


def temporal_smoothness(sequence: np.ndarray) -> float:
    diffs = np.diff(sequence, axis=0)
    return float(np.mean([np.linalg.norm(d) / (np.linalg.norm(sequence[i]) + 1e-12) for i, d in enumerate(diffs)]))


def sequence_metrics(name: str, recon: np.ndarray, reference: np.ndarray) -> dict[str, float | str]:
    per_frame = [metrics("frame", recon[i], reference[i])["rel_l2"] for i in range(reference.shape[0])]
    cx_ref = centroid_x(reference)
    cx_rec = centroid_x(recon)
    return {
        "method": name,
        "mean_rel_l2": float(np.mean(per_frame)),
        "median_rel_l2": float(np.median(per_frame)),
        "max_rel_l2": float(np.max(per_frame)),
        "temporal_smoothness": temporal_smoothness(recon),
        "centroid_rmse": float(np.sqrt(np.mean((cx_rec - cx_ref) ** 2))),
    }


def write_rows(path: Path, rows: list[dict[str, float | str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def center_slices(sequence: np.ndarray, frame_idx: int) -> np.ndarray:
    zc = sequence.shape[1] // 2
    return sequence[frame_idx, zc]


def plot_summary(reference: np.ndarray, baseline: np.ndarray, lowrank: np.ndarray, rows: list[dict[str, float | str]], rank: int) -> None:
    t = reference.shape[0] // 2
    panels = [
        ("GT mid-frame slice", center_slices(reference, t), "viridis"),
        ("Framewise baseline", center_slices(baseline, t), "viridis"),
        (f"Low-rank rank {rank}", center_slices(lowrank, t), "viridis"),
        ("Baseline abs error", np.abs(center_slices(baseline - reference, t)), "magma"),
        ("Low-rank abs error", np.abs(center_slices(lowrank - reference, t)), "magma"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.5), constrained_layout=True)
    for ax, (title, image, cmap) in zip(axes.flat[:5], panels):
        im = ax.imshow(image, origin="lower", cmap=cmap)
        ax.set_title(title, fontsize=9.5)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax = axes.flat[5]
    labels = [row["method"] for row in rows]
    values = [float(row["mean_rel_l2"]) for row in rows]
    ax.bar(labels, values, color=["tab:blue", "tab:orange"])
    ax.set_title("Mean frame relative L2")
    ax.set_ylabel("Relative L2")
    ax.tick_params(axis="x", rotation=12)
    for i, value in enumerate(values):
        ax.text(i, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("M3B 4D BOST low-rank temporal toy", fontsize=13)
    fig.savefig(RESULTS / "m3b_4d_summary.png", dpi=180)
    plt.close(fig)


def plot_rank_scan(rows: list[dict[str, float | str]]) -> None:
    ranks = [int(row["rank"]) for row in rows]
    l2 = [float(row["mean_rel_l2"]) for row in rows]
    smooth = [float(row["temporal_smoothness"]) for row in rows]
    centroid = [float(row["centroid_rmse"]) for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    axes[0].plot(ranks, l2, marker="o", color="tab:blue")
    axes[0].set_title("Mean relative L2")
    axes[0].set_xlabel("Rank")
    axes[0].set_ylabel("Lower is better")
    axes[1].plot(ranks, smooth, marker="o", color="tab:orange")
    axes[1].set_title("Temporal smoothness")
    axes[1].set_xlabel("Rank")
    axes[2].plot(ranks, centroid, marker="o", color="tab:green")
    axes[2].set_title("Centroid trajectory RMSE")
    axes[2].set_xlabel("Rank")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle("M3B rank trade-off: accuracy, smoothness, motion tracking", fontsize=13)
    fig.savefig(RESULTS / "m3b_rank_scan.png", dpi=180)
    plt.close(fig)


def plot_temporal_trace(reference: np.ndarray, baseline: np.ndarray, lowrank: np.ndarray, rank: int) -> None:
    t = np.arange(reference.shape[0])
    fig, ax = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    ax.plot(t, centroid_x(reference), marker="o", label="ground truth")
    ax.plot(t, centroid_x(baseline), marker="o", label="framewise baseline")
    ax.plot(t, centroid_x(lowrank), marker="o", label=f"low-rank rank {rank}")
    ax.set_xlabel("Frame")
    ax.set_ylabel("x-centroid")
    ax.set_title("M3B temporal motion trace")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(RESULTS / "m3b_temporal_trace.png", dpi=180)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    n = 30
    nz = 12
    nt = 18
    angles = np.linspace(0, 180, 5, endpoint=False)
    reference = make_4d_sequence(n=n, nz=nz, nt=nt)
    baseline = reconstruct_framewise(reference, n=n, angles=angles, noise_level=0.14)

    rank_rows: list[dict[str, float | str]] = []
    lowrank_by_rank = {}
    for rank in [1, 2, 3, 5, 8, 12]:
        lowrank = low_rank_temporal(baseline, rank=rank)
        row = sequence_metrics(f"low-rank rank {rank}", lowrank, reference)
        row["rank"] = rank
        rank_rows.append(row)
        lowrank_by_rank[rank] = lowrank
    write_rows(RESULTS / "rank_metrics.csv", rank_rows)

    selected_rank = 3
    lowrank = lowrank_by_rank[selected_rank]
    rows = [
        sequence_metrics("framewise baseline", baseline, reference),
        sequence_metrics(f"low-rank rank {selected_rank}", lowrank, reference),
    ]
    write_rows(RESULTS / "metrics.csv", rows)
    plot_summary(reference, baseline, lowrank, rows, rank=selected_rank)
    plot_rank_scan(rank_rows)
    plot_temporal_trace(reference, baseline, lowrank, rank=selected_rank)

    print("M3B 4D low-rank BOST toy complete")
    for row in rows:
        print(row)
    best = min(rank_rows, key=lambda item: float(item["mean_rel_l2"]))
    print(f"best rank by mean L2: {best['rank']}, mean L2={best['mean_rel_l2']:.4f}")
    print(f"results: {RESULTS}")


if __name__ == "__main__":
    main()
