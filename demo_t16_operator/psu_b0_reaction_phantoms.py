"""Analytic reaction-flow morphology proxies for PSU-geometry inverse tests.

The fields in this module are deterministic synthetic perturbations. They
approximate plume, flame-front, and shock-like morphology, but they are not CFD
solutions and must not be reported as experimental or simulation truth.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from .psu_b0_reconstruction_interface import project_dirichlet_gauge


PHANTOM_SCHEMA = "psu-b0-reaction-morphology-proxy-1.0"
SUPPORTED_FAMILIES = (
    "plume",
    "wavy_front",
    "thin_front",
    "double_front",
    "annular_kernel",
    "oblique_shock",
    "vortex_pair",
    "multi_plume",
)


def _coordinates(
    grid_size: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    size = int(grid_size)
    if size < 8:
        raise ValueError("grid_size must be at least eight")
    axis = torch.linspace(-1.0, 1.0, size, dtype=dtype, device=device)
    zz, yy, xx = torch.meshgrid(axis, axis, axis, indexing="ij")
    return xx, yy, zz


def _window(
    xx: torch.Tensor,
    yy: torch.Tensor,
    zz: torch.Tensor,
) -> torch.Tensor:
    return (
        (1.0 - xx.square()).clamp_min(0.0)
        * (1.0 - yy.square()).clamp_min(0.0)
        * (1.0 - zz.square()).clamp_min(0.0)
    ).square()


def _plume(
    xx: torch.Tensor,
    yy: torch.Tensor,
    zz: torch.Tensor,
    rng: np.random.Generator,
) -> torch.Tensor:
    center = rng.uniform(-0.24, 0.24, size=3)
    widths = rng.uniform(0.20, 0.42, size=3)
    primary = torch.exp(
        -0.5
        * (
            ((xx - center[0]) / widths[0]).square()
            + ((yy - center[1]) / widths[1]).square()
            + ((zz - center[2]) / widths[2]).square()
        )
    )
    wake_shift = rng.uniform(0.20, 0.42)
    wake = torch.exp(
        -0.5
        * (
            ((xx - center[0] - wake_shift) / (0.8 * widths[0])).square()
            + ((yy - center[1] + 0.10) / (1.15 * widths[1])).square()
            + ((zz - center[2] - 0.08) / (0.85 * widths[2])).square()
        )
    )
    waviness = 1.0 + 0.18 * torch.sin(
        rng.uniform(2.0, 4.5) * yy
        + rng.uniform(1.5, 3.5) * zz
        + rng.uniform(-np.pi, np.pi)
    )
    return rng.uniform(0.75, 1.15) * primary * waviness - rng.uniform(
        0.28,
        0.58,
    ) * wake


def _front(
    xx: torch.Tensor,
    yy: torch.Tensor,
    zz: torch.Tensor,
    rng: np.random.Generator,
    *,
    thin: bool,
) -> torch.Tensor:
    width = rng.uniform(0.028, 0.052) if thin else rng.uniform(0.075, 0.14)
    offset = rng.uniform(-0.22, 0.22)
    amplitude_y = rng.uniform(0.08, 0.24)
    amplitude_z = rng.uniform(0.06, 0.20)
    phase_y = rng.uniform(-np.pi, np.pi)
    phase_z = rng.uniform(-np.pi, np.pi)
    surface = (
        offset
        + amplitude_y * torch.sin(rng.uniform(2.0, 4.2) * yy + phase_y)
        + amplitude_z * torch.cos(rng.uniform(1.7, 3.8) * zz + phase_z)
        + rng.uniform(-0.10, 0.10) * yy * zz
    )
    transition = torch.tanh((xx - surface) / width)
    envelope = torch.exp(
        -0.5
        * (
            ((yy - rng.uniform(-0.16, 0.16)) / rng.uniform(0.48, 0.78)).square()
            + ((zz - rng.uniform(-0.16, 0.16)) / rng.uniform(0.48, 0.78)).square()
        )
    )
    return rng.uniform(0.48, 0.82) * transition * envelope


def _double_front(
    xx: torch.Tensor,
    yy: torch.Tensor,
    zz: torch.Tensor,
    rng: np.random.Generator,
) -> torch.Tensor:
    width_a = rng.uniform(0.035, 0.070)
    width_b = rng.uniform(0.045, 0.085)
    surface_a = (
        rng.uniform(-0.34, -0.08)
        + rng.uniform(0.08, 0.20) * torch.sin(2.4 * yy + rng.uniform(-2.0, 2.0))
        + rng.uniform(0.04, 0.16) * torch.cos(2.1 * zz + rng.uniform(-2.0, 2.0))
    )
    surface_b = (
        rng.uniform(0.08, 0.34)
        + rng.uniform(0.05, 0.18) * torch.cos(2.8 * yy + rng.uniform(-2.0, 2.0))
        - rng.uniform(0.05, 0.17) * torch.sin(2.0 * zz + rng.uniform(-2.0, 2.0))
    )
    first = torch.tanh((xx - surface_a) / width_a)
    second = torch.tanh((surface_b - xx) / width_b)
    transverse = torch.exp(
        -0.5
        * (
            (yy / rng.uniform(0.55, 0.82)).square()
            + (zz / rng.uniform(0.48, 0.78)).square()
        )
    )
    return rng.uniform(0.42, 0.72) * (first + second) * transverse


def _annular_kernel(
    xx: torch.Tensor,
    yy: torch.Tensor,
    zz: torch.Tensor,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Expanding flame-kernel proxy with a finite-thickness thermal shell."""

    center = rng.uniform(-0.16, 0.16, size=3)
    axes = rng.uniform(0.82, 1.18, size=3)
    radius = rng.uniform(0.34, 0.58)
    width = rng.uniform(0.035, 0.085)
    radial = torch.sqrt(
        ((xx - center[0]) / axes[0]).square()
        + ((yy - center[1]) / axes[1]).square()
        + ((zz - center[2]) / axes[2]).square()
        + 1e-12
    )
    wrinkle = (
        rng.uniform(0.025, 0.075)
        * torch.sin(rng.uniform(2.0, 4.5) * yy + rng.uniform(-np.pi, np.pi))
        * torch.cos(rng.uniform(1.8, 4.0) * zz + rng.uniform(-np.pi, np.pi))
    )
    shell = torch.exp(-0.5 * ((radial - radius - wrinkle) / width).square())
    core = torch.exp(-0.5 * (radial / (0.62 * radius)).square())
    return rng.uniform(0.72, 1.08) * shell - rng.uniform(0.12, 0.34) * core


def _oblique_shock(
    xx: torch.Tensor,
    yy: torch.Tensor,
    zz: torch.Tensor,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Curved oblique compression-front proxy."""

    normal = rng.normal(size=3)
    normal = normal / np.linalg.norm(normal)
    offset = rng.uniform(-0.24, 0.24)
    curvature = rng.uniform(-0.18, 0.18)
    coordinate = (
        normal[0] * xx
        + normal[1] * yy
        + normal[2] * zz
        - offset
        + curvature * (yy.square() - 0.6 * zz.square())
    )
    width = rng.uniform(0.025, 0.060)
    footprint = torch.exp(
        -0.5
        * (
            ((yy - rng.uniform(-0.12, 0.12)) / rng.uniform(0.58, 0.86)).square()
            + ((zz - rng.uniform(-0.12, 0.12)) / rng.uniform(0.52, 0.82)).square()
        )
    )
    return rng.uniform(0.46, 0.78) * torch.tanh(coordinate / width) * footprint


def _vortex_pair(
    xx: torch.Tensor,
    yy: torch.Tensor,
    zz: torch.Tensor,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Opposite-signed rolled-up scalar structures in a reacting shear layer."""

    separation = rng.uniform(0.24, 0.46)
    angle = rng.uniform(-np.pi, np.pi)
    dx = 0.5 * separation * np.cos(angle)
    dy = 0.5 * separation * np.sin(angle)
    center_z = rng.uniform(-0.16, 0.16)
    width_xy = rng.uniform(0.13, 0.25)
    width_z = rng.uniform(0.30, 0.56)

    def core(cx: float, cy: float) -> torch.Tensor:
        return torch.exp(
            -0.5
            * (
                ((xx - cx) / width_xy).square()
                + ((yy - cy) / width_xy).square()
                + ((zz - center_z) / width_z).square()
            )
        )

    first = core(dx, dy)
    second = core(-dx, -dy)
    bridge = torch.sin(
        rng.uniform(2.2, 4.4) * xx
        + rng.uniform(1.8, 3.8) * yy
        + rng.uniform(-np.pi, np.pi)
    ) * torch.exp(-0.5 * (zz / rng.uniform(0.42, 0.68)).square())
    return rng.uniform(0.70, 1.10) * (first - second) + rng.uniform(
        0.06,
        0.16,
    ) * bridge


def _multi_plume(
    xx: torch.Tensor,
    yy: torch.Tensor,
    zz: torch.Tensor,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Interacting multi-source plume proxy with signed wakes."""

    field = torch.zeros_like(xx)
    count = int(rng.integers(2, 5))
    for index in range(count):
        center = rng.uniform(-0.42, 0.42, size=3)
        widths = rng.uniform(0.12, 0.30, size=3)
        blob = torch.exp(
            -0.5
            * (
                ((xx - center[0]) / widths[0]).square()
                + ((yy - center[1]) / widths[1]).square()
                + ((zz - center[2]) / widths[2]).square()
            )
        )
        sign = 1.0 if index % 2 == 0 else -rng.uniform(0.35, 0.75)
        field = field + sign * rng.uniform(0.55, 1.05) * blob
    shear = rng.uniform(0.06, 0.16) * torch.sin(
        rng.uniform(2.0, 4.8) * yy
        + rng.uniform(1.6, 3.8) * zz
        + rng.uniform(-np.pi, np.pi)
    )
    return field + shear * torch.exp(-0.5 * (xx / 0.65).square())


def reaction_morphology_field(
    *,
    grid_size: int,
    family: str,
    seed: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return one normalized sign-changing scalar perturbation field."""

    name = str(family)
    if name not in SUPPORTED_FAMILIES:
        raise ValueError(f"unsupported reaction morphology family: {name}")
    target_device = torch.device(device)
    xx, yy, zz = _coordinates(
        int(grid_size),
        dtype=dtype,
        device=target_device,
    )
    rng = np.random.default_rng(int(seed))
    if name == "plume":
        field = _plume(xx, yy, zz, rng)
    elif name == "wavy_front":
        field = _front(xx, yy, zz, rng, thin=False)
    elif name == "thin_front":
        field = _front(xx, yy, zz, rng, thin=True)
    elif name == "double_front":
        field = _double_front(xx, yy, zz, rng)
    elif name == "annular_kernel":
        field = _annular_kernel(xx, yy, zz, rng)
    elif name == "oblique_shock":
        field = _oblique_shock(xx, yy, zz, rng)
    elif name == "vortex_pair":
        field = _vortex_pair(xx, yy, zz, rng)
    else:
        field = _multi_plume(xx, yy, zz, rng)
    field = field * _window(xx, yy, zz)
    field = project_dirichlet_gauge(field[None, None], boundary_width=1)
    rms = torch.sqrt(torch.mean(field.square())).clamp_min(1e-8)
    return field / rms


def reaction_morphology_batch(
    *,
    grid_size: int,
    families: Sequence[str],
    seeds: Sequence[int],
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build a deterministic batch with one family and seed per sample."""

    if len(families) != len(seeds) or not families:
        raise ValueError("families and seeds must be nonempty and aligned")
    return torch.cat(
        [
            reaction_morphology_field(
                grid_size=grid_size,
                family=family,
                seed=int(seed),
                dtype=dtype,
                device=device,
            )
            for family, seed in zip(families, seeds, strict=True)
        ],
        dim=0,
    )
