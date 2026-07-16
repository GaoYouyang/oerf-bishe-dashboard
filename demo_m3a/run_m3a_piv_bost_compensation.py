#!/usr/bin/env python3
"""M3A PIV-BOST compensation toy.

This script models the vector-field error induced when a particle image is
refractively displaced at two consecutive time instants. It is a lightweight
bridge to simultaneous PIV-BOST: BOST estimates the refractive displacement,
then the PIV vector field can be corrected by subtracting the change in that
displacement along the particle motion.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def make_grid(n: int = 76) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(-1.0, 1.0, n)
    return np.meshgrid(axis, axis)


def refractive_index_field(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    blob = np.exp(-(((x + 0.22) ** 2) / (2 * 0.28**2) + ((y - 0.08) ** 2) / (2 * 0.22**2)))
    plume = 0.55 * np.exp(-((x - 0.18 * np.sin(3.0 * y)) ** 2) / (2 * 0.055**2)) * np.exp(-(y**2) / 0.85)
    pocket = -0.22 * np.exp(-(((x - 0.38) ** 2) / (2 * 0.18**2) + ((y + 0.28) ** 2) / (2 * 0.14**2)))
    field = blob + plume + pocket
    field -= field.min()
    field /= field.max() + 1e-12
    return field


def velocity_field(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = 0.075 + 0.025 * np.sin(np.pi * y) - 0.018 * x * y
    v = 0.026 * np.cos(0.7 * np.pi * x) + 0.014 * np.sin(np.pi * x * y)
    return u, v


def refractive_displacement(index_field: np.ndarray, spacing: float, alpha: float = 0.010) -> tuple[np.ndarray, np.ndarray]:
    grad_y, grad_x = np.gradient(index_field, spacing, spacing)
    return alpha * grad_x, alpha * grad_y


def bilinear_sample(field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = field.shape[0]
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
    v00 = field[y0[valid], x0[valid]]
    v10 = field[y0[valid], x1[valid]]
    v01 = field[y1[valid], x0[valid]]
    v11 = field[y1[valid], x1[valid]]
    out[valid] = (
        (1 - wx[valid]) * (1 - wy[valid]) * v00
        + wx[valid] * (1 - wy[valid]) * v10
        + (1 - wx[valid]) * wy[valid] * v01
        + wx[valid] * wy[valid] * v11
    )
    return out


def smooth_noise(shape: tuple[int, int], rng: np.random.Generator, strength: float = 0.08) -> np.ndarray:
    noise = rng.normal(size=shape)
    for _ in range(5):
        noise = (
            noise
            + np.roll(noise, 1, axis=0)
            + np.roll(noise, -1, axis=0)
            + np.roll(noise, 1, axis=1)
            + np.roll(noise, -1, axis=1)
        ) / 5.0
    noise /= np.std(noise) + 1e-12
    return strength * noise


def vector_metrics(name: str, err_u: np.ndarray, err_v: np.ndarray, ref_u: np.ndarray, ref_v: np.ndarray) -> dict[str, float | str]:
    err_mag = np.sqrt(err_u**2 + err_v**2)
    ref_mag = np.sqrt(ref_u**2 + ref_v**2)
    rmse = float(np.sqrt(np.mean(err_mag**2)))
    return {
        "field": name,
        "rmse": rmse,
        "relative_rmse": float(rmse / (np.sqrt(np.mean(ref_mag**2)) + 1e-12)),
        "p95_error": float(np.percentile(err_mag, 95)),
        "max_error": float(np.max(err_mag)),
    }


def write_metrics(rows: list[dict[str, float | str]]) -> None:
    with (RESULTS / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(
    index_field: np.ndarray,
    refr_mag: np.ndarray,
    vel_mag: np.ndarray,
    obs_err: np.ndarray,
    comp_err: np.ndarray,
    rows: list[dict[str, float | str]],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.8), constrained_layout=True)
    panels = [
        ("Synthetic refractive-index field", index_field, "viridis"),
        ("Refractive displacement magnitude", refr_mag, "magma"),
        ("True velocity magnitude", vel_mag, "viridis"),
        ("Observed PIV velocity error", obs_err, "magma"),
        ("After BOST-style compensation", comp_err, "magma"),
    ]
    for ax, (title, image, cmap) in zip(axes.flat[:5], panels):
        im = ax.imshow(image, origin="lower", cmap=cmap)
        ax.set_title(title, fontsize=9.5)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes.flat[5]
    labels = [row["field"] for row in rows]
    values = [float(row["rmse"]) for row in rows]
    ax.bar(labels, values, color=["tab:blue", "tab:orange"])
    ax.set_title("Vector RMSE")
    ax.set_ylabel("RMSE")
    ax.tick_params(axis="x", rotation=12)
    for i, value in enumerate(values):
        ax.text(i, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("M3A PIV-BOST compensation toy", fontsize=13)
    fig.savefig(RESULTS / "m3a_compensation_summary.png", dpi=180)
    plt.close(fig)


def plot_profile(x: np.ndarray, obs_err: np.ndarray, comp_err: np.ndarray) -> None:
    center = obs_err.shape[0] // 2
    fig, ax = plt.subplots(figsize=(7.8, 4.4), constrained_layout=True)
    ax.plot(x[center], obs_err[center], label="observed PIV error")
    ax.plot(x[center], comp_err[center], label="after compensation")
    ax.set_xlabel("x at centerline")
    ax.set_ylabel("Velocity error magnitude")
    ax.set_title("M3A centerline error profile")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(RESULTS / "m3a_error_profile.png", dpi=180)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260706)
    n = 76
    x, y = make_grid(n)
    spacing = 2.0 / (n - 1)

    index_field = refractive_index_field(x, y)
    true_u, true_v = velocity_field(x, y)
    refr_u, refr_v = refractive_displacement(index_field, spacing)

    r0_u = refr_u
    r0_v = refr_v
    r1_u = bilinear_sample(refr_u, x + true_u, y + true_v)
    r1_v = bilinear_sample(refr_v, x + true_u, y + true_v)
    observed_u = true_u + (r1_u - r0_u)
    observed_v = true_v + (r1_v - r0_v)

    est_refr_u = refr_u * (1.0 + smooth_noise(refr_u.shape, rng, strength=0.020))
    est_refr_v = refr_v * (1.0 + smooth_noise(refr_v.shape, rng, strength=0.020))
    est_r0_u = est_refr_u
    est_r0_v = est_refr_v
    est_r1_u = bilinear_sample(est_refr_u, x + observed_u, y + observed_v)
    est_r1_v = bilinear_sample(est_refr_v, x + observed_u, y + observed_v)

    compensated_u = observed_u - (est_r1_u - est_r0_u)
    compensated_v = observed_v - (est_r1_v - est_r0_v)

    obs_err_u = observed_u - true_u
    obs_err_v = observed_v - true_v
    comp_err_u = compensated_u - true_u
    comp_err_v = compensated_v - true_v
    obs_err = np.sqrt(obs_err_u**2 + obs_err_v**2)
    comp_err = np.sqrt(comp_err_u**2 + comp_err_v**2)

    rows = [
        vector_metrics("observed PIV", obs_err_u, obs_err_v, true_u, true_v),
        vector_metrics("BOST compensated", comp_err_u, comp_err_v, true_u, true_v),
    ]
    write_metrics(rows)

    refr_mag = np.sqrt(refr_u**2 + refr_v**2)
    vel_mag = np.sqrt(true_u**2 + true_v**2)
    plot_summary(index_field, refr_mag, vel_mag, obs_err, comp_err, rows)
    plot_profile(x, obs_err, comp_err)

    print("M3A PIV-BOST compensation toy complete")
    for row in rows:
        print(row)
    print(f"results: {RESULTS}")


if __name__ == "__main__":
    main()
