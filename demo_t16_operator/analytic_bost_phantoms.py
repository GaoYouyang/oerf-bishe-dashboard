"""Independent continuous phantoms and analytic-gradient BOST rendering.

These deterministic fields are morphology proxies, not CFD or experimental
truth.  Their analytic gradients generate synthetic observations without using
the voxel finite-difference or trilinear-interpolation chain employed by the
inverse operator.  This removes one narrow inverse-crime mechanism while
leaving the much larger synthetic-to-experiment gap explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch


ANALYTIC_PHANTOM_SCHEMA = "analytic-bost-phantom-1.0"
ANALYTIC_RENDERER_SCHEMA = "analytic-gradient-fixed-denominator-bost-1.0"
ANALYTIC_FAMILIES = (
    "smooth_plume",
    "wrinkled_density_interface",
    "oblique_compression_sheet",
    "shock_expansion_pair",
)


@dataclass(frozen=True)
class AnalyticPhantomSpec:
    family: str
    seed: int
    parameters: tuple[tuple[str, float], ...]
    domain_minimum_xyz: tuple[float, float, float]
    domain_maximum_xyz: tuple[float, float, float]
    scalar_contract: str = "normalized_scalar_perturbation_not_physical_density"

    def parameter_dict(self) -> dict[str, float]:
        return dict(self.parameters)


@dataclass(frozen=True)
class AnalyticPhantomEvaluation:
    field: torch.Tensor
    gradient_xyz: torch.Tensor
    level_sets: torch.Tensor
    level_set_gradients_xyz: torch.Tensor


def _validated_domain(
    minimum_xyz: Any,
    maximum_xyz: Any,
) -> tuple[np.ndarray, np.ndarray]:
    minimum = np.asarray(minimum_xyz, dtype=np.float64).reshape(-1)
    maximum = np.asarray(maximum_xyz, dtype=np.float64).reshape(-1)
    if minimum.shape != (3,) or maximum.shape != (3,):
        raise ValueError("domain bounds must contain x, y, and z")
    if not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
        raise ValueError("domain bounds must be finite")
    if np.any(maximum <= minimum):
        raise ValueError("domain maximum must exceed domain minimum")
    return minimum, maximum


def make_analytic_phantom(
    *,
    family: str,
    seed: int,
    domain_minimum_xyz: Any = (-1.0, -1.0, -1.0),
    domain_maximum_xyz: Any = (1.0, 1.0, 1.0),
) -> AnalyticPhantomSpec:
    """Freeze one deterministic continuous morphology proxy."""

    name = str(family)
    if name not in ANALYTIC_FAMILIES:
        raise ValueError(f"unsupported analytic phantom family: {name}")
    minimum, maximum = _validated_domain(domain_minimum_xyz, domain_maximum_xyz)
    rng = np.random.default_rng(int(seed))
    parameters: dict[str, float]
    if name == "smooth_plume":
        parameters = {
            "amplitude": float(rng.uniform(0.65, 1.15)),
            "center_x": float(rng.uniform(-0.18, 0.18)),
            "center_y": float(rng.uniform(-0.18, 0.18)),
            "center_z": float(rng.uniform(-0.18, 0.18)),
            "width_x": float(rng.uniform(0.24, 0.38)),
            "width_y": float(rng.uniform(0.28, 0.44)),
            "width_z": float(rng.uniform(0.28, 0.46)),
        }
    elif name == "wrinkled_density_interface":
        parameters = {
            "amplitude": float(rng.uniform(0.45, 0.82)),
            "offset": float(rng.uniform(-0.14, 0.14)),
            "amplitude_y": float(rng.uniform(0.07, 0.15)),
            "amplitude_z": float(rng.uniform(0.06, 0.14)),
            "frequency_y": float(rng.uniform(2.0, 3.5)),
            "frequency_z": float(rng.uniform(1.8, 3.2)),
            "phase_y": float(rng.uniform(-math.pi, math.pi)),
            "phase_z": float(rng.uniform(-math.pi, math.pi)),
            "cross": float(rng.uniform(-0.055, 0.055)),
            "width": float(rng.uniform(0.10, 0.16)),
        }
    elif name == "oblique_compression_sheet":
        raw_normal = np.asarray(
            [
                rng.uniform(0.75, 1.0),
                rng.uniform(-0.45, 0.45),
                rng.uniform(-0.35, 0.35),
            ],
            dtype=np.float64,
        )
        normal = raw_normal / np.linalg.norm(raw_normal)
        parameters = {
            "amplitude": float(rng.uniform(0.42, 0.76)),
            "normal_x": float(normal[0]),
            "normal_y": float(normal[1]),
            "normal_z": float(normal[2]),
            "offset": float(rng.uniform(-0.16, 0.16)),
            "curvature": float(rng.uniform(-0.08, 0.08)),
            "width": float(rng.uniform(0.085, 0.14)),
        }
    else:
        parameters = {
            "compression_amplitude": float(rng.uniform(0.42, 0.70)),
            "expansion_amplitude": float(rng.uniform(0.20, 0.46)),
            "offset_a": float(rng.uniform(-0.30, -0.16)),
            "offset_b": float(rng.uniform(0.16, 0.30)),
            "waviness_a": float(rng.uniform(0.035, 0.075)),
            "waviness_b": float(rng.uniform(0.025, 0.065)),
            "frequency_y": float(rng.uniform(1.8, 3.0)),
            "frequency_z": float(rng.uniform(1.7, 2.8)),
            "phase_y": float(rng.uniform(-math.pi, math.pi)),
            "phase_z": float(rng.uniform(-math.pi, math.pi)),
            "width_a": float(rng.uniform(0.085, 0.13)),
            "width_b": float(rng.uniform(0.095, 0.15)),
        }
    return AnalyticPhantomSpec(
        family=name,
        seed=int(seed),
        parameters=tuple(sorted(parameters.items())),
        domain_minimum_xyz=tuple(float(value) for value in minimum),
        domain_maximum_xyz=tuple(float(value) for value in maximum),
    )


def _window_and_gradient(normalized_xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    factors = (1.0 - normalized_xyz.square()).clamp_min(0.0)
    squared = factors.square()
    window = torch.prod(squared, dim=-1)
    gradients = []
    for axis in range(3):
        other = torch.ones_like(window)
        for other_axis in range(3):
            if other_axis != axis:
                other = other * squared[..., other_axis]
        inside = factors[..., axis] > 0.0
        derivative = -4.0 * normalized_xyz[..., axis] * factors[..., axis] * other
        gradients.append(torch.where(inside, derivative, torch.zeros_like(derivative)))
    return window, torch.stack(gradients, dim=-1)


def _transition(
    level: torch.Tensor,
    level_gradient: torch.Tensor,
    *,
    amplitude: float,
    width: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    hyperbolic = torch.tanh(level / width)
    derivative = float(amplitude) * (1.0 - hyperbolic.square()) / float(width)
    return float(amplitude) * hyperbolic, derivative[..., None] * level_gradient


def evaluate_analytic_phantom(
    spec: AnalyticPhantomSpec,
    points_xyz: Any,
) -> AnalyticPhantomEvaluation:
    """Evaluate the scalar, analytic gradient, and declared level sets."""

    if spec.family not in ANALYTIC_FAMILIES:
        raise ValueError("phantom spec has an unsupported family")
    points = torch.as_tensor(points_xyz)
    if not points.is_floating_point():
        points = points.to(torch.float64)
    if points.ndim < 2 or points.shape[-1] != 3:
        raise ValueError("points_xyz must have shape [...,3]")
    if torch.any(~torch.isfinite(points)):
        raise ValueError("points_xyz must be finite")
    minimum = torch.as_tensor(
        spec.domain_minimum_xyz,
        dtype=points.dtype,
        device=points.device,
    )
    maximum = torch.as_tensor(
        spec.domain_maximum_xyz,
        dtype=points.dtype,
        device=points.device,
    )
    center = 0.5 * (minimum + maximum)
    half_extent = 0.5 * (maximum - minimum)
    normalized = (points - center) / half_extent
    window, window_gradient = _window_and_gradient(normalized)
    p = spec.parameter_dict()
    x, y, z = normalized.unbind(dim=-1)

    levels: list[torch.Tensor] = []
    level_gradients: list[torch.Tensor] = []
    if spec.family == "smooth_plume":
        displacement = torch.stack(
            (
                x - p["center_x"],
                y - p["center_y"],
                z - p["center_z"],
            ),
            dim=-1,
        )
        widths = torch.as_tensor(
            [p["width_x"], p["width_y"], p["width_z"]],
            dtype=points.dtype,
            device=points.device,
        )
        base = -p["amplitude"] * torch.exp(
            -0.5 * torch.sum((displacement / widths).square(), dim=-1)
        )
        base_gradient = base[..., None] * (-displacement / widths.square())
    elif spec.family == "wrinkled_density_interface":
        level = (
            x
            - p["offset"]
            - p["amplitude_y"] * torch.sin(p["frequency_y"] * y + p["phase_y"])
            - p["amplitude_z"] * torch.cos(p["frequency_z"] * z + p["phase_z"])
            - p["cross"] * y * z
        )
        level_gradient = torch.stack(
            (
                torch.ones_like(x),
                -p["amplitude_y"]
                * p["frequency_y"]
                * torch.cos(p["frequency_y"] * y + p["phase_y"])
                - p["cross"] * z,
                p["amplitude_z"]
                * p["frequency_z"]
                * torch.sin(p["frequency_z"] * z + p["phase_z"])
                - p["cross"] * y,
            ),
            dim=-1,
        )
        base, base_gradient = _transition(
            level,
            level_gradient,
            amplitude=p["amplitude"],
            width=p["width"],
        )
        levels.append(level)
        level_gradients.append(level_gradient)
    elif spec.family == "oblique_compression_sheet":
        level = (
            p["normal_x"] * x
            + p["normal_y"] * y
            + p["normal_z"] * z
            - p["offset"]
            + p["curvature"] * (y.square() - 0.6 * z.square())
        )
        level_gradient = torch.stack(
            (
                torch.full_like(x, p["normal_x"]),
                p["normal_y"] + 2.0 * p["curvature"] * y,
                p["normal_z"] - 1.2 * p["curvature"] * z,
            ),
            dim=-1,
        )
        base, base_gradient = _transition(
            level,
            level_gradient,
            amplitude=0.5 * p["amplitude"],
            width=p["width"],
        )
        levels.append(level)
        level_gradients.append(level_gradient)
    else:
        level_a = (
            x
            - p["offset_a"]
            - p["waviness_a"]
            * torch.sin(p["frequency_y"] * y + p["phase_y"])
        )
        gradient_a = torch.stack(
            (
                torch.ones_like(x),
                -p["waviness_a"]
                * p["frequency_y"]
                * torch.cos(p["frequency_y"] * y + p["phase_y"]),
                torch.zeros_like(z),
            ),
            dim=-1,
        )
        level_b = (
            x
            - p["offset_b"]
            - p["waviness_b"]
            * torch.cos(p["frequency_z"] * z + p["phase_z"])
        )
        gradient_b = torch.stack(
            (
                torch.ones_like(x),
                torch.zeros_like(y),
                p["waviness_b"]
                * p["frequency_z"]
                * torch.sin(p["frequency_z"] * z + p["phase_z"]),
            ),
            dim=-1,
        )
        first, first_gradient = _transition(
            level_a,
            gradient_a,
            amplitude=0.5 * p["compression_amplitude"],
            width=p["width_a"],
        )
        second, second_gradient = _transition(
            level_b,
            gradient_b,
            amplitude=-0.5 * p["expansion_amplitude"],
            width=p["width_b"],
        )
        base = first + second
        base_gradient = first_gradient + second_gradient
        levels.extend((level_a, level_b))
        level_gradients.extend((gradient_a, gradient_b))

    field = base * window
    gradient_normalized = base_gradient * window[..., None] + base[..., None] * window_gradient
    gradient_physical = gradient_normalized / half_extent
    if levels:
        stacked_levels = torch.stack(levels, dim=-1)
        stacked_level_gradients = torch.stack(level_gradients, dim=-2) / half_extent
    else:
        stacked_levels = torch.empty(
            (*field.shape, 0),
            dtype=field.dtype,
            device=field.device,
        )
        stacked_level_gradients = torch.empty(
            (*field.shape, 0, 3),
            dtype=field.dtype,
            device=field.device,
        )
    return AnalyticPhantomEvaluation(
        field=field,
        gradient_xyz=gradient_physical,
        level_sets=stacked_levels,
        level_set_gradients_xyz=stacked_level_gradients,
    )


def analytic_phantom_grid(
    spec: AnalyticPhantomSpec,
    *,
    grid_shape: tuple[int, int, int],
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> AnalyticPhantomEvaluation:
    """Evaluate one spec on a regular ``[z,y,x]`` reconstruction grid."""

    shape = tuple(int(value) for value in grid_shape)
    if len(shape) != 3 or any(value < 4 for value in shape):
        raise ValueError("grid_shape must contain three dimensions of at least four")
    minimum = spec.domain_minimum_xyz
    maximum = spec.domain_maximum_xyz
    x = torch.linspace(minimum[0], maximum[0], shape[2], dtype=dtype, device=device)
    y = torch.linspace(minimum[1], maximum[1], shape[1], dtype=dtype, device=device)
    z = torch.linspace(minimum[2], maximum[2], shape[0], dtype=dtype, device=device)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    return evaluate_analytic_phantom(spec, torch.stack((xx, yy, zz), dim=-1))


def render_analytic_bost(
    spec: AnalyticPhantomSpec,
    *,
    sample_points_xyz: Any,
    projection_u_xyz: Any,
    projection_v_xyz: Any,
    line_length: Any,
    system_constant: Any,
    sample_valid: Any | None = None,
) -> torch.Tensor:
    """Integrate the analytic gradient with the fixed sample-count denominator."""

    points = torch.as_tensor(sample_points_xyz)
    if not points.is_floating_point():
        points = points.to(torch.float64)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("sample_points_xyz must have shape [ray,sample,3]")
    ray_count, sample_count, _ = points.shape
    if ray_count < 1 or sample_count < 1:
        raise ValueError("renderer requires at least one ray and sample")
    dtype, device = points.dtype, points.device
    projection_u = torch.as_tensor(projection_u_xyz, dtype=dtype, device=device)
    projection_v = torch.as_tensor(projection_v_xyz, dtype=dtype, device=device)
    length = torch.as_tensor(line_length, dtype=dtype, device=device).reshape(-1)
    constant = torch.as_tensor(system_constant, dtype=dtype, device=device).reshape(-1)
    if projection_u.shape != (ray_count, 3) or projection_v.shape != (ray_count, 3):
        raise ValueError("projection vectors must have shape [ray,3]")
    if length.shape != (ray_count,) or constant.shape != (ray_count,):
        raise ValueError("line length and system constant need one value per ray")
    if torch.any(~torch.isfinite(projection_u)) or torch.any(~torch.isfinite(projection_v)):
        raise ValueError("projection vectors must be finite")
    if torch.any(~torch.isfinite(length)) or torch.any(length < 0.0):
        raise ValueError("line lengths must be finite and nonnegative")
    if torch.any(~torch.isfinite(constant)):
        raise ValueError("system constants must be finite")
    if sample_valid is None:
        valid = torch.ones((ray_count, sample_count), dtype=torch.bool, device=device)
    else:
        valid = torch.as_tensor(sample_valid, dtype=torch.bool, device=device)
        if valid.shape != (ray_count, sample_count):
            raise ValueError("sample_valid must have shape [ray,sample]")
    evaluation = evaluate_analytic_phantom(spec, points)
    gradient = torch.where(valid[..., None], evaluation.gradient_xyz, 0.0)
    projected_u = torch.sum(gradient * projection_u[:, None, :], dim=-1)
    projected_v = torch.sum(gradient * projection_v[:, None, :], dim=-1)
    projected = torch.stack(
        (projected_u.sum(dim=1), projected_v.sum(dim=1)),
        dim=-1,
    )
    scale = length * constant / float(sample_count)
    output = projected * scale[:, None]
    if torch.any(~torch.isfinite(output)):
        raise RuntimeError("analytic BOST renderer produced non-finite output")
    return output
