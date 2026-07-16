"""Small linear BOST forward model and synthetic 3D field families.

This module deliberately mirrors the geometry used by demo_m1, but exposes a
matrix form so the same operator can be used by NumPy data generation and a
differentiable PyTorch reprojection loss.
"""

from __future__ import annotations

import math

import numpy as np


def make_grid_2d(n: int) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(-1.0, 1.0, n)
    return np.meshgrid(axis, axis)


def make_grid_3d(n: int, depth: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(-1.0, 1.0, n)
    y = np.linspace(-1.0, 1.0, n)
    z = np.linspace(-1.0, 1.0, depth)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    return xx, yy, zz


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


def project_image(image: np.ndarray, angles: np.ndarray) -> np.ndarray:
    n = image.shape[0]
    offsets = np.linspace(-1.0, 1.0, n)
    ray_axis = np.linspace(-1.45, 1.45, n * 2)
    dv = ray_axis[1] - ray_axis[0]
    sinogram = np.zeros((n, len(angles)), dtype=np.float64)
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
        columns.append(project_image(directional_gradient, np.array([theta_deg]))[:, 0])
    return np.stack(columns, axis=1)


def build_forward_matrix(n: int, angles: np.ndarray) -> np.ndarray:
    """Return A with shape [view, detector, pixel]."""
    columns = []
    for pixel in range(n * n):
        basis = np.zeros((n, n), dtype=np.float64)
        basis.flat[pixel] = 1.0
        columns.append(deflection_sinogram(basis, angles).T)
    return np.stack(columns, axis=-1).astype(np.float32)


def forward_volume(volume: np.ndarray, operator: np.ndarray) -> np.ndarray:
    """Project [depth, y, x] to [depth, view, detector]."""
    flat = volume.reshape(volume.shape[0], -1)
    return np.einsum("vnp,dp->dvn", operator, flat, optimize=True)


def integrate_deflection(deflection: np.ndarray) -> np.ndarray:
    projection = np.cumsum(deflection, axis=0)
    return projection - projection.mean(axis=0, keepdims=True)


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
        values = np.interp(u.reshape(-1), offsets, filtered[:, j], left=0.0, right=0.0)
        recon += values.reshape(n, n)
    return np.nan_to_num(recon * math.pi / max(len(angles), 1))


def baseline_lift(observation: np.ndarray, angles: np.ndarray, n: int) -> np.ndarray:
    """Lift [depth, view, detector] into a stack of FBP slices."""
    slices = []
    for z_index in range(observation.shape[0]):
        deflection = observation[z_index].T
        slices.append(filtered_backprojection(integrate_deflection(deflection), angles, n))
    return np.stack(slices).astype(np.float32)


def support_window(n: int, depth: int) -> np.ndarray:
    xx, yy, zz = make_grid_3d(n, depth)
    radial = np.sqrt(xx**2 + yy**2)
    radial_window = np.clip((0.98 - radial) / 0.16, 0.0, 1.0)
    axial_window = np.clip((0.98 - np.abs(zz)) / 0.22, 0.0, 1.0)
    return (0.5 - 0.5 * np.cos(np.pi * radial_window)) * (0.5 - 0.5 * np.cos(np.pi * axial_window))


def _gaussian(xx, yy, zz, center, width, amplitude):
    x0, y0, z0 = center
    sx, sy, sz = width
    exponent = ((xx - x0) / sx) ** 2 + ((yy - y0) / sy) ** 2 + ((zz - z0) / sz) ** 2
    return amplitude * np.exp(-0.5 * exponent)


def make_phantom(family: str, n: int, depth: int, rng: np.random.Generator) -> np.ndarray:
    xx, yy, zz = make_grid_3d(n, depth)
    field = np.zeros_like(xx)

    if family == "gaussian":
        for _ in range(int(rng.integers(2, 5))):
            center = rng.uniform([-0.45, -0.45, -0.45], [0.45, 0.45, 0.45])
            width = rng.uniform([0.14, 0.18, 0.22], [0.36, 0.42, 0.56])
            field += _gaussian(xx, yy, zz, center, width, rng.uniform(0.45, 1.0))
    elif family == "flame":
        phase = rng.uniform(0.0, 2.0 * np.pi)
        centerline = (
            rng.uniform(-0.18, 0.18)
            + rng.uniform(0.08, 0.20) * np.sin(rng.uniform(2.0, 4.5) * yy + phase)
            + rng.uniform(0.04, 0.13) * np.sin(rng.uniform(2.0, 4.0) * zz - phase)
        )
        width = rng.uniform(0.055, 0.12)
        sheet = np.exp(-0.5 * ((xx - centerline) / width) ** 2)
        envelope = np.exp(-0.5 * (yy / rng.uniform(0.55, 0.9)) ** 2) * np.exp(-0.5 * (zz / rng.uniform(0.55, 0.95)) ** 2)
        field = rng.uniform(0.65, 1.0) * sheet * envelope
        for _ in range(2):
            center = rng.uniform([-0.35, -0.35, -0.35], [0.35, 0.35, 0.35])
            width3 = rng.uniform([0.12, 0.16, 0.20], [0.25, 0.32, 0.45])
            field += _gaussian(xx, yy, zz, center, width3, rng.uniform(0.2, 0.5))
    elif family == "thin_front":
        phase = rng.uniform(0.0, 2.0 * np.pi)
        radius = np.sqrt((xx - rng.uniform(-0.12, 0.12)) ** 2 + (yy - rng.uniform(-0.12, 0.12)) ** 2)
        target = rng.uniform(0.32, 0.58) + rng.uniform(0.05, 0.12) * np.sin(rng.uniform(2.0, 4.0) * zz + phase)
        shell = np.exp(-0.5 * ((radius - target) / rng.uniform(0.025, 0.055)) ** 2)
        curved = np.exp(-0.5 * ((xx - 0.22 * np.sin(3.0 * yy + 2.0 * zz + phase)) / rng.uniform(0.025, 0.05)) ** 2)
        field = shell * np.exp(-0.5 * (zz / 0.78) ** 2) + rng.uniform(0.25, 0.55) * curved
    else:
        raise ValueError(f"Unknown phantom family: {family}")

    field *= support_window(n, depth)
    field -= field.min()
    field /= field.max() + 1e-8
    return field.astype(np.float32)
