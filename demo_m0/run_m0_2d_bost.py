#!/usr/bin/env python3
"""M0 2D BOST / coordinate-field toy demo.

This script intentionally keeps dependencies light: numpy and matplotlib only.
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


def make_grid(n: int = 64) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(-1.0, 1.0, n)
    x, y = np.meshgrid(axis, axis)
    return x, y


def gaussian(x: np.ndarray, y: np.ndarray, x0: float, y0: float, sx: float, sy: float, amp: float) -> np.ndarray:
    return amp * np.exp(-(((x - x0) ** 2) / (2 * sx**2) + ((y - y0) ** 2) / (2 * sy**2)))


def make_phantom(n: int = 64) -> np.ndarray:
    x, y = make_grid(n)
    field = (
        gaussian(x, y, -0.32, 0.08, 0.20, 0.34, 1.00)
        + gaussian(x, y, 0.28, -0.18, 0.18, 0.22, 0.72)
        - gaussian(x, y, 0.05, 0.28, 0.12, 0.16, 0.34)
    )
    flame_sheet = 0.24 * np.exp(-((x + 0.12 * np.sin(4 * y)) ** 2) / (2 * 0.035**2)) * np.exp(-(y**2) / 0.85)
    field = field + flame_sheet
    field = field - field.min()
    field = field / (field.max() + 1e-12)
    return field.astype(np.float64)


def bilinear_sample(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample image on coordinates x,y in [-1, 1] with zero outside."""

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
    """Simple parallel-beam line integral projector."""

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
        samples = bilinear_sample(image, x, y)
        sinogram[:, j] = samples.sum(axis=1) * dv
    return sinogram


def deflection_sinogram(field: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Approximate BOS deflection sinograms from a refractive-index field.

    For a view angle theta, the ray direction is s=(cos theta, sin theta).
    The detector/transverse direction is t=(-sin theta, cos theta). Under a
    straight-ray small-angle model, the measured deflection is approximated by
    line-integrating t dot grad(n).
    """

    n = field.shape[0]
    spacing = 2.0 / (n - 1)
    grad_y, grad_x = np.gradient(field, spacing, spacing)
    columns = []
    for theta_deg in angles:
        theta = np.deg2rad(theta_deg)
        tx = -np.sin(theta)
        ty = np.cos(theta)
        directional_gradient = tx * grad_x + ty * grad_y
        sino = project_image(directional_gradient, np.array([theta_deg]), detector_count=n)
        columns.append(sino[:, 0])
    return np.stack(columns, axis=1)


def integrate_deflection_to_projection(deflection: np.ndarray) -> np.ndarray:
    """Recover a proxy projection by integrating detector-direction derivative."""

    projection = np.cumsum(deflection, axis=0)
    projection = projection - projection.mean(axis=0, keepdims=True)
    return projection


def baseline_reconstruction(deflection: np.ndarray, angles: np.ndarray, n: int) -> np.ndarray:
    projection = integrate_deflection_to_projection(deflection)
    recon = filtered_backprojection(projection, angles, n)
    return recon


def ramp_filter(projection: np.ndarray) -> np.ndarray:
    """Apply a simple ramp filter along detector coordinate."""

    m = projection.shape[0]
    freq = np.fft.fftfreq(m).reshape(-1, 1)
    filt = np.abs(freq)
    return np.real(np.fft.ifft(np.fft.fft(projection, axis=0) * filt, axis=0))


def filtered_backprojection(projection: np.ndarray, angles: np.ndarray, n: int) -> np.ndarray:
    filtered = ramp_filter(projection)
    offsets = np.linspace(-1.0, 1.0, projection.shape[0])
    x, y = make_grid(n)
    recon = np.zeros((n, n), dtype=np.float64)
    for j, theta_deg in enumerate(angles):
        theta = np.deg2rad(theta_deg)
        tx, ty = -np.sin(theta), np.cos(theta)
        u = x * tx + y * ty
        recon += np.interp(u.reshape(-1), offsets, filtered[:, j], left=0.0, right=0.0).reshape(n, n)
    recon *= math.pi / max(len(angles), 1)
    return np.nan_to_num(recon)


def random_fourier_features(n: int, num_freq: int = 72, seed: int = 7) -> np.ndarray:
    x, y = make_grid(n)
    coords = np.stack([x.ravel(), y.ravel()], axis=1)
    rng = np.random.default_rng(seed)
    frequencies = rng.normal(scale=3.5, size=(num_freq, 2))
    phases = 2 * math.pi * rng.random(num_freq)
    arg = coords @ frequencies.T + phases
    features = [np.sin(arg), np.cos(arg), coords[:, :1], coords[:, 1:2]]
    return np.concatenate(features, axis=1)


def coordinate_field_inverse(deflection: np.ndarray, angles: np.ndarray, n: int, num_freq: int = 72, ridge: float = 1e-2) -> np.ndarray:
    """Linear coordinate-field inverse using random Fourier feature basis.

    This is a torch-free stand-in for the coordinate-field idea: each basis
    image is pushed through the same BOS forward model, then ridge regression
    solves the inverse problem in feature space.
    """

    phi = random_fourier_features(n, num_freq=num_freq)
    target = deflection.reshape(-1)
    columns = []
    for j in range(phi.shape[1]):
        basis_image = phi[:, j].reshape(n, n)
        columns.append(deflection_sinogram(basis_image, angles).reshape(-1))
    a = np.stack(columns, axis=1)
    lhs = a.T @ a + ridge * np.eye(a.shape[1])
    rhs = a.T @ target
    weights = np.linalg.solve(lhs, rhs)
    recon = (phi @ weights).reshape(n, n)
    return recon


def align_to_reference(recon: np.ndarray, reference: np.ndarray) -> np.ndarray:
    x = recon.reshape(-1)
    y = reference.reshape(-1)
    design = np.stack([x, np.ones_like(x)], axis=1)
    scale, offset = np.linalg.lstsq(design, y, rcond=None)[0]
    return scale * recon + offset


def metrics(name: str, recon: np.ndarray, reference: np.ndarray) -> dict[str, float | str]:
    recon = align_to_reference(recon, reference)
    data_range = float(reference.max() - reference.min())
    err = recon - reference
    rel_l2 = float(np.linalg.norm(err) / (np.linalg.norm(reference) + 1e-12))
    cc = float(np.corrcoef(recon.reshape(-1), reference.reshape(-1))[0, 1])
    ssim = float(global_ssim(reference, recon, data_range=data_range))
    mse = float(np.mean(err**2))
    psnr = float(20 * np.log10(data_range / (np.sqrt(mse) + 1e-12)))
    return {"method": name, "rel_l2": rel_l2, "cc": cc, "ssim": ssim, "psnr": psnr}


def global_ssim(reference: np.ndarray, recon: np.ndarray, data_range: float) -> float:
    """Small global SSIM proxy without external image libraries."""

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x = reference.mean()
    mu_y = recon.mean()
    var_x = reference.var()
    var_y = recon.var()
    cov_xy = ((reference - mu_x) * (recon - mu_y)).mean()
    numerator = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    return numerator / (denominator + 1e-12)


def save_metrics(rows: list[dict[str, float | str]]) -> None:
    path = RESULTS / "metrics.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "rel_l2", "cc", "ssim", "psnr"])
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(reference: np.ndarray, baseline: np.ndarray, coord: np.ndarray, rows: list[dict[str, float | str]]) -> None:
    baseline_a = align_to_reference(baseline, reference)
    coord_a = align_to_reference(coord, reference)
    vmax = max(reference.max(), baseline_a.max(), coord_a.max())
    vmin = min(reference.min(), baseline_a.min(), coord_a.min())

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    items = [
        ("Ground truth refractive-index phantom", reference, "viridis", vmin, vmax),
        ("Voxel/tomography baseline", baseline_a, "viridis", vmin, vmax),
        ("Coordinate-field inverse", coord_a, "viridis", vmin, vmax),
        ("Baseline absolute error", np.abs(baseline_a - reference), "magma", 0, None),
        ("Coordinate-field absolute error", np.abs(coord_a - reference), "magma", 0, None),
    ]

    for ax, (title, img, cmap, lo, hi) in zip(axes.flat[:5], items):
        im = ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, origin="lower")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    axes.flat[5].axis("off")
    text = "M0 2D BOST toy metrics\n\n"
    for row in rows:
        text += f"{row['method']}\n"
        text += f"  rel L2: {row['rel_l2']:.4f}\n"
        text += f"  CC:     {row['cc']:.4f}\n"
        text += f"  SSIM:   {row['ssim']:.4f}\n"
        text += f"  PSNR:   {row['psnr']:.2f} dB\n\n"
    axes.flat[5].text(0.02, 0.98, text, va="top", ha="left", family="monospace", fontsize=9)

    fig.suptitle("M0: synthetic refractive-index field -> BOS deflection -> reconstruction", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(RESULTS / "m0_summary.png", dpi=180)
    plt.close(fig)


def view_count_scan(reference: np.ndarray, n: int) -> None:
    counts = [3, 5, 7, 9, 13]
    rows = []
    for count in counts:
        angles = np.linspace(0, 180, count, endpoint=False)
        deflection = deflection_sinogram(reference, angles)
        base = baseline_reconstruction(deflection, angles, n)
        coord = coordinate_field_inverse(deflection, angles, n, num_freq=72, ridge=2e-2)
        rows.append(
            {
                "views": count,
                "baseline_l2": metrics("baseline", base, reference)["rel_l2"],
                "coord_l2": metrics("coordinate", coord, reference)["rel_l2"],
            }
        )

    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    ax.plot([r["views"] for r in rows], [r["baseline_l2"] for r in rows], marker="o", label="voxel/tomography baseline")
    ax.plot([r["views"] for r in rows], [r["coord_l2"] for r in rows], marker="o", label="coordinate-field inverse")
    ax.set_xlabel("Number of views")
    ax.set_ylabel("Relative L2 error, lower is better")
    ax.set_title("M0 view-count sensitivity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(RESULTS / "view_count_curve.png", dpi=180)
    plt.close(fig)

    with (RESULTS / "view_count_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["views", "baseline_l2", "coord_l2"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    n = 64
    reference = make_phantom(n)
    angles = np.linspace(0, 180, 9, endpoint=False)
    deflection = deflection_sinogram(reference, angles)
    baseline = baseline_reconstruction(deflection, angles, n)
    coord = coordinate_field_inverse(deflection, angles, n, num_freq=72, ridge=2e-2)
    rows = [metrics("voxel/tomography baseline", baseline, reference), metrics("coordinate-field inverse", coord, reference)]
    save_metrics(rows)
    plot_summary(reference, baseline, coord, rows)
    view_count_scan(reference, n)
    print("M0 demo complete")
    for row in rows:
        print(row)
    print(f"results: {RESULTS}")


if __name__ == "__main__":
    main()
