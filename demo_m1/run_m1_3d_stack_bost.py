#!/usr/bin/env python3
"""M1 2.5D / 3D-stack BOST toy demo.

The implementation is deliberately lightweight and uses only numpy/matplotlib.
It extends the M0 2D demo by reconstructing a volume slice by slice.

This is not a full NeRIF reproduction. It is a stress test for one thesis-scale
idea: a 3D coordinate prior can regularize sparse-view stack reconstructions,
but it should not be expected to beat a clean, well-sampled classical baseline.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def make_grid_2d(n: int) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(-1.0, 1.0, n)
    return np.meshgrid(axis, axis)


def make_grid_3d(n: int, nz: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(-1.0, 1.0, n)
    y = np.linspace(-1.0, 1.0, n)
    z = np.linspace(-1.0, 1.0, nz)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    return xx, yy, zz


def gaussian3d(xx, yy, zz, center, widths, amp):
    x0, y0, z0 = center
    sx, sy, sz = widths
    return amp * np.exp(-(((xx - x0) ** 2) / (2 * sx**2) + ((yy - y0) ** 2) / (2 * sy**2) + ((zz - z0) ** 2) / (2 * sz**2)))


def make_volume(n: int = 48, nz: int = 24) -> np.ndarray:
    xx, yy, zz = make_grid_3d(n, nz)
    volume = (
        gaussian3d(xx, yy, zz, (-0.35, 0.05, -0.20), (0.22, 0.34, 0.48), 1.0)
        + gaussian3d(xx, yy, zz, (0.28, -0.20, 0.22), (0.18, 0.24, 0.35), 0.72)
        - gaussian3d(xx, yy, zz, (0.08, 0.26, 0.05), (0.14, 0.16, 0.28), 0.28)
    )
    sheet_center = 0.10 * np.sin(4.0 * yy + 2.0 * zz)
    flame_sheet = 0.26 * np.exp(-((xx - sheet_center) ** 2) / (2 * 0.045**2)) * np.exp(-(yy**2) / 0.9) * np.exp(-(zz**2) / 1.3)
    volume = volume + flame_sheet
    volume -= volume.min()
    volume /= volume.max() + 1e-12
    return volume.astype(np.float64)


def bilinear_sample(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = image.shape[0]
    px = (x + 1.0) * 0.5 * (n - 1)
    py = (y + 1.0) * 0.5 * (n - 1)
    x0 = np.floor(px).astype(int)
    y0 = np.floor(py).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    valid = (x0 >= 0) & (y0 >= 0) & (x1 < n) & (y1 < n)
    out = np.zeros_like(px, dtype=np.float64)
    if not np.any(valid):
        return out
    wx = px - x0
    wy = py - y0
    v00 = image[y0[valid], x0[valid]]
    v10 = image[y0[valid], x1[valid]]
    v01 = image[y1[valid], x0[valid]]
    v11 = image[y1[valid], x1[valid]]
    out[valid] = (
        (1 - wx[valid]) * (1 - wy[valid]) * v00
        + wx[valid] * (1 - wy[valid]) * v10
        + (1 - wx[valid]) * wy[valid] * v01
        + wx[valid] * wy[valid] * v11
    )
    return out


def project_image(image: np.ndarray, angles: np.ndarray, detector_count: int | None = None) -> np.ndarray:
    n = image.shape[0]
    detector_count = detector_count or n
    offsets = np.linspace(-1.0, 1.0, detector_count)
    ray_axis = np.linspace(-1.45, 1.45, n * 2)
    dv = ray_axis[1] - ray_axis[0]
    sinogram = np.zeros((detector_count, len(angles)), dtype=np.float64)
    for j, theta_deg in enumerate(angles):
        theta = np.deg2rad(theta_deg)
        sx, sy = np.cos(theta), np.sin(theta)
        tx, ty = -np.sin(theta), np.cos(theta)
        u, v = np.meshgrid(offsets, ray_axis, indexing="ij")
        x = u * tx + v * sx
        y = u * ty + v * sy
        sinogram[:, j] = bilinear_sample(image, x, y).sum(axis=1) * dv
    return sinogram


def deflection_sinogram(field: np.ndarray, angles: np.ndarray) -> np.ndarray:
    n = field.shape[0]
    spacing = 2.0 / (n - 1)
    grad_y, grad_x = np.gradient(field, spacing, spacing)
    columns = []
    for theta_deg in angles:
        theta = np.deg2rad(theta_deg)
        tx = -np.sin(theta)
        ty = np.cos(theta)
        directional_gradient = tx * grad_x + ty * grad_y
        columns.append(project_image(directional_gradient, np.array([theta_deg]), detector_count=n)[:, 0])
    return np.stack(columns, axis=1)


def integrate_deflection(deflection: np.ndarray) -> np.ndarray:
    projection = np.cumsum(deflection, axis=0)
    projection = projection - projection.mean(axis=0, keepdims=True)
    return projection


def ramp_filter(projection: np.ndarray) -> np.ndarray:
    m = projection.shape[0]
    freq = np.fft.fftfreq(m).reshape(-1, 1)
    return np.real(np.fft.ifft(np.fft.fft(projection, axis=0) * np.abs(freq), axis=0))


def filtered_backprojection(projection: np.ndarray, angles: np.ndarray, n: int) -> np.ndarray:
    filtered = ramp_filter(projection)
    offsets = np.linspace(-1.0, 1.0, projection.shape[0])
    x, y = make_grid_2d(n)
    recon = np.zeros((n, n), dtype=np.float64)
    for j, theta_deg in enumerate(angles):
        theta = np.deg2rad(theta_deg)
        tx, ty = -np.sin(theta), np.cos(theta)
        u = x * tx + y * ty
        recon += np.interp(u.reshape(-1), offsets, filtered[:, j], left=0.0, right=0.0).reshape(n, n)
    return np.nan_to_num(recon * math.pi / max(len(angles), 1))


def baseline_slice(deflection: np.ndarray, angles: np.ndarray, n: int) -> np.ndarray:
    return filtered_backprojection(integrate_deflection(deflection), angles, n)


def random_fourier_features(n: int, num_freq: int = 54, seed: int = 11) -> np.ndarray:
    x, y = make_grid_2d(n)
    coords = np.stack([x.ravel(), y.ravel()], axis=1)
    rng = np.random.default_rng(seed)
    frequencies = rng.normal(scale=3.3, size=(num_freq, 2))
    phases = 2 * math.pi * rng.random(num_freq)
    arg = coords @ frequencies.T + phases
    return np.concatenate([np.sin(arg), np.cos(arg), coords[:, :1], coords[:, 1:2]], axis=1)


def coordinate_forward_matrix(n: int, angles: np.ndarray, num_freq: int = 54) -> tuple[np.ndarray, np.ndarray]:
    phi = random_fourier_features(n, num_freq=num_freq)
    columns = []
    for j in range(phi.shape[1]):
        basis_image = phi[:, j].reshape(n, n)
        columns.append(deflection_sinogram(basis_image, angles).reshape(-1))
    return phi, np.stack(columns, axis=1)


def coordinate_stack_inverse(deflection_stack: np.ndarray, angles: np.ndarray, n: int, num_freq: int = 54, ridge: float = 2e-2) -> np.ndarray:
    phi, a = coordinate_forward_matrix(n, angles, num_freq=num_freq)
    solve_matrix = np.linalg.solve(a.T @ a + ridge * np.eye(a.shape[1]), a.T)
    slices = []
    for z in range(deflection_stack.shape[0]):
        target = deflection_stack[z].reshape(-1)
        weights = solve_matrix @ target
        slices.append((phi @ weights).reshape(n, n))
    return np.stack(slices, axis=0)


def random_fourier_features_3d(n: int, nz: int, num_freq: int = 300, seed: int = 7, scale: float = 2.0) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, n)
    y = np.linspace(-1.0, 1.0, n)
    z = np.linspace(-1.0, 1.0, nz)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    coords = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    rng = np.random.default_rng(seed)
    frequencies = rng.normal(scale=scale, size=(num_freq, 3))
    phases = 2 * math.pi * rng.random(num_freq)
    arg = coords @ frequencies.T + phases
    low_order = np.concatenate([np.ones((coords.shape[0], 1)), coords, coords**2], axis=1)
    return np.concatenate([low_order, np.sin(arg), np.cos(arg)], axis=1)


def coordinate_regularized_stack(
    stack_reconstruction: np.ndarray,
    n: int,
    nz: int,
    num_freq: int = 300,
    ridge: float = 1e-3,
) -> np.ndarray:
    """Fit a compact 3D coordinate representation to a stack reconstruction.

    This mimics a low-capacity implicit-volume prior: it preserves broad 3D
    structure while suppressing sparse-view streaks and slice-to-slice noise.
    """
    phi = random_fourier_features_3d(n, nz, num_freq=num_freq)
    target = stack_reconstruction.reshape(-1)
    normal = phi.T @ phi + ridge * np.eye(phi.shape[1])
    weights = np.linalg.solve(normal, phi.T @ target)
    return (phi @ weights).reshape(nz, n, n)


def synthesize_deflection_stack(volume: np.ndarray, angles: np.ndarray) -> np.ndarray:
    return np.stack([deflection_sinogram(volume[z], angles) for z in range(volume.shape[0])], axis=0)


def baseline_stack(deflection_stack: np.ndarray, angles: np.ndarray, n: int) -> np.ndarray:
    return np.stack([baseline_slice(deflection_stack[z], angles, n) for z in range(deflection_stack.shape[0])], axis=0)


def align_to_reference(recon: np.ndarray, reference: np.ndarray) -> np.ndarray:
    x = recon.reshape(-1)
    y = reference.reshape(-1)
    design = np.stack([x, np.ones_like(x)], axis=1)
    scale, offset = np.linalg.lstsq(design, y, rcond=None)[0]
    return scale * recon + offset


def global_ssim(reference: np.ndarray, recon: np.ndarray, data_range: float) -> float:
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x = reference.mean()
    mu_y = recon.mean()
    var_x = reference.var()
    var_y = recon.var()
    cov_xy = ((reference - mu_x) * (recon - mu_y)).mean()
    return ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2) + 1e-12)


def metrics(name: str, recon: np.ndarray, reference: np.ndarray) -> dict[str, float | str]:
    aligned = align_to_reference(recon, reference)
    err = aligned - reference
    data_range = float(reference.max() - reference.min())
    mse = float(np.mean(err**2))
    return {
        "method": name,
        "rel_l2": float(np.linalg.norm(err) / (np.linalg.norm(reference) + 1e-12)),
        "cc": float(np.corrcoef(aligned.reshape(-1), reference.reshape(-1))[0, 1]),
        "ssim_proxy": float(global_ssim(reference, aligned, data_range=data_range)),
        "psnr": float(20 * np.log10(data_range / (np.sqrt(mse) + 1e-12))),
    }


def save_metrics(rows: list[dict[str, float | str]], filename: str = "metrics.csv") -> None:
    with (RESULTS / filename).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def extract_slices(volume: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    zc = volume.shape[0] // 2
    yc = volume.shape[1] // 2
    xc = volume.shape[2] // 2
    return volume[zc], volume[:, yc, :], volume[:, :, xc]


def plot_volume_summary(reference: np.ndarray, baseline: np.ndarray, coord: np.ndarray, rows: list[dict[str, float | str]], view_count: int) -> None:
    baseline_a = align_to_reference(baseline, reference)
    coord_a = align_to_reference(coord, reference)
    panels = [
        ("GT xy slice", extract_slices(reference)[0], "viridis"),
        ("Baseline xy slice", extract_slices(baseline_a)[0], "viridis"),
        ("3D coord-regularized xy", extract_slices(coord_a)[0], "viridis"),
        ("GT xz slice", extract_slices(reference)[1], "viridis"),
        ("Baseline abs error xy", np.abs(extract_slices(baseline_a - reference)[0]), "magma"),
        ("Coord-regularized abs error xy", np.abs(extract_slices(coord_a - reference)[0]), "magma"),
        ("GT yz slice", extract_slices(reference)[2], "viridis"),
        ("Baseline xz slice", extract_slices(baseline_a)[1], "viridis"),
        ("Coord-regularized xz slice", extract_slices(coord_a)[1], "viridis"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 10))
    for ax, (name, img, cmap) in zip(axes.flat, panels):
        im = ax.imshow(img, origin="lower", cmap=cmap)
        ax.set_title(name, fontsize=9.5)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"M1 sparse-view 3D-stack BOST toy: {view_count} views, volume slices and errors", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(RESULTS / "m1_volume_summary.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4), constrained_layout=True)
    text = "M1 volume metrics\n\n"
    for row in rows:
        text += f"{row['method']}\n"
        text += f"  rel L2:     {row['rel_l2']:.4f}\n"
        text += f"  CC:         {row['cc']:.4f}\n"
        text += f"  SSIM proxy: {row['ssim_proxy']:.4f}\n"
        text += f"  PSNR:       {row['psnr']:.2f} dB\n\n"
    ax.axis("off")
    ax.text(0.02, 0.98, text, ha="left", va="top", family="monospace", fontsize=11)
    fig.savefig(RESULTS / "m1_metrics_card.png", dpi=180)
    plt.close(fig)


def view_count_scan(reference: np.ndarray, n: int, counts=(3, 5, 7, 9, 13)) -> None:
    rows = []
    for count in counts:
        angles = np.linspace(0, 180, count, endpoint=False)
        deflection = synthesize_deflection_stack(reference, angles)
        base = baseline_stack(deflection, angles, n)
        coord = coordinate_regularized_stack(base, n=n, nz=reference.shape[0], num_freq=300, ridge=1e-3)
        rows.append(
            {
                "views": count,
                "baseline_l2": metrics("baseline", base, reference)["rel_l2"],
                "coord_l2": metrics("coordinate", coord, reference)["rel_l2"],
            }
        )
    with (RESULTS / "view_count_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["views", "baseline_l2", "coord_l2"])
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    ax.plot([r["views"] for r in rows], [r["baseline_l2"] for r in rows], marker="o", label="stack baseline")
    ax.plot([r["views"] for r in rows], [r["coord_l2"] for r in rows], marker="o", label="3D coord-regularized stack")
    ax.set_xlabel("Number of views")
    ax.set_ylabel("Volume relative L2, lower is better")
    ax.set_title("M1 3D-stack view-count sensitivity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(RESULTS / "m1_view_count_curve.png", dpi=180)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    n = 44
    nz = 22
    view_count = 5
    angles = np.linspace(0, 180, view_count, endpoint=False)
    reference = make_volume(n=n, nz=nz)
    deflection = synthesize_deflection_stack(reference, angles)
    base = baseline_stack(deflection, angles, n)
    coord = coordinate_regularized_stack(base, n=n, nz=nz, num_freq=300, ridge=1e-3)
    rows = [
        metrics(f"{view_count}-view stack baseline", base, reference),
        metrics(f"{view_count}-view 3D coord-regularized stack", coord, reference),
    ]
    save_metrics(rows)
    plot_volume_summary(reference, base, coord, rows, view_count=view_count)
    view_count_scan(reference, n)
    print("M1 demo complete")
    for row in rows:
        print(row)
    print(f"results: {RESULTS}")


if __name__ == "__main__":
    main()
