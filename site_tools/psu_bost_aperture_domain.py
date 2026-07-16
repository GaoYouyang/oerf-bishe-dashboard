"""Source-independent NumPy float64 B2 finite-aperture domain primitives.

The released PSU BOST sampler places each finite-aperture point at

``center(s) + radius(s) * (u * Rx + v * Ry)``,

where the center and radius are linearly interpolated between their endpoint
values.  This module reproduces that geometry deterministically and applies a
sample-level B0 box or B1 one-nappe-cone indicator.  It never renormalizes by
the number of samples that survive the domain test.
"""

from __future__ import annotations

from typing import Any

import numpy as np


CONTRACT_VERSION = "psu-bost-aperture-domain-1.0"
_DEFAULT_RTOL = 64.0 * np.finfo(np.float64).eps
_DEFAULT_BASIS_TOLERANCE = 1e-6

APERTURE_DOMAIN_CONTRACT: dict[str, Any] = {
    "version": CONTRACT_VERSION,
    "provenance": (
        "Self-contained NumPy float64 implementation. It does not import or "
        "execute the released TensorFlow sampler."
    ),
    "sample": (
        "center(s) + radius(s) * (u * Rx + v * Ry), with center and radius "
        "linearly interpolated between endpoints"
    ),
    "quadrature": (
        "deterministic Cartesian product of longitudinal probes and a disk "
        "center plus symmetric rings, including endpoint and disk-boundary probes"
    ),
    "B0": "closed axis-aligned box point membership",
    "B1": "B0 intersect normalized closed one-nappe cone point membership",
    "normalization": (
        "LoS sums are divided by the original fixed sample count, never by "
        "the number of in-domain samples"
    ),
    "dtype": "all generated points and numeric summaries are numpy.float64",
}


def _validate_tolerances(atol: float, rtol: float) -> tuple[float, float]:
    absolute = float(atol)
    relative = float(rtol)
    if not np.isfinite(absolute) or absolute < 0.0:
        raise ValueError("atol must be a finite non-negative float")
    if not np.isfinite(relative) or relative < 0.0:
        raise ValueError("rtol must be a finite non-negative float")
    return absolute, relative


def _as_rows(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one ray")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _as_vector3(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.size != 3:
        raise ValueError(f"{name} must contain exactly three values")
    vector = np.ascontiguousarray(array.reshape(3), dtype=np.float64)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _as_ray_scalars(value: Any, name: str, ray_count: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.shape != (ray_count,):
        raise ValueError(f"{name} must have shape (N,)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _as_fractions(value: Any) -> np.ndarray:
    fractions = np.asarray(value, dtype=np.float64)
    if fractions.ndim != 1 or fractions.size == 0:
        raise ValueError(
            "longitudinal_fractions must be a non-empty one-dimensional array"
        )
    if not np.all(np.isfinite(fractions)):
        raise ValueError("longitudinal_fractions must contain only finite values")
    if np.any((fractions < 0.0) | (fractions > 1.0)):
        raise ValueError(
            "longitudinal_fractions must lie in the closed interval [0, 1]"
        )
    return np.ascontiguousarray(fractions, dtype=np.float64)


def _as_disk_offsets(value: Any, *, tolerance: float = 1e-12) -> np.ndarray:
    offsets = np.asarray(value, dtype=np.float64)
    if offsets.ndim != 2 or offsets.shape[1] != 2 or offsets.shape[0] == 0:
        raise ValueError("unit_disk_offsets must have non-empty shape (S, 2)")
    if not np.all(np.isfinite(offsets)):
        raise ValueError("unit_disk_offsets must contain only finite values")
    radius_squared = np.sum(offsets * offsets, axis=1)
    if np.any(radius_squared > 1.0 + float(tolerance)):
        raise ValueError("unit_disk_offsets must lie inside the closed unit disk")
    return np.ascontiguousarray(offsets, dtype=np.float64)


def _validate_basis(
    rx: np.ndarray,
    ry: np.ndarray,
    *,
    tolerance: float,
) -> None:
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("basis_tolerance must be a finite non-negative float")

    rx_norm = np.linalg.norm(rx, axis=1)
    ry_norm = np.linalg.norm(ry, axis=1)
    invalid_norm = (
        ~np.isfinite(rx_norm)
        | ~np.isfinite(ry_norm)
        | (np.abs(rx_norm - 1.0) > tolerance)
        | (np.abs(ry_norm - 1.0) > tolerance)
    )
    if np.any(invalid_norm):
        indices = np.flatnonzero(invalid_norm).tolist()
        raise ValueError(
            f"Rx and Ry must be finite unit vectors; invalid ray indices {indices}"
        )

    dot = np.sum(rx * ry, axis=1)
    invalid_orthogonality = np.abs(dot) > tolerance
    if np.any(invalid_orthogonality):
        indices = np.flatnonzero(invalid_orthogonality).tolist()
        raise ValueError(
            f"Rx and Ry must be mutually orthogonal; invalid ray indices {indices}"
        )


def deterministic_unit_disk_offsets(
    *,
    ring_radii: Any = (0.5, 1.0),
    angles_per_ring: int = 8,
) -> np.ndarray:
    """Return a center point and symmetric disk rings with boundary probes.

    ``angles_per_ring`` must be a positive multiple of four so each ring has
    antipodal symmetry and exact probes on the positive/negative coordinate
    axes.  A unit-radius ring is appended when it is not supplied.
    """

    if isinstance(angles_per_ring, bool) or int(angles_per_ring) != angles_per_ring:
        raise ValueError("angles_per_ring must be an integer")
    angle_count = int(angles_per_ring)
    if angle_count < 4 or angle_count % 4:
        raise ValueError("angles_per_ring must be a positive multiple of four")

    radii = np.asarray(ring_radii, dtype=np.float64)
    if radii.ndim != 1:
        raise ValueError("ring_radii must be one-dimensional")
    if not np.all(np.isfinite(radii)):
        raise ValueError("ring_radii must contain only finite values")
    if np.any((radii <= 0.0) | (radii > 1.0)):
        raise ValueError("ring_radii must lie in the interval (0, 1]")
    radii = np.unique(radii)
    if radii.size == 0 or not np.isclose(radii[-1], 1.0, atol=0.0, rtol=0.0):
        radii = np.append(radii, 1.0)

    angles = 2.0 * np.pi * np.arange(angle_count, dtype=np.float64) / angle_count
    rings = [
        radius * np.column_stack((np.cos(angles), np.sin(angles))) for radius in radii
    ]
    offsets = np.vstack((np.zeros((1, 2), dtype=np.float64), *rings))
    offsets[np.abs(offsets) < 8.0 * np.finfo(np.float64).eps] = 0.0
    return np.ascontiguousarray(offsets, dtype=np.float64)


def deterministic_aperture_quadrature(
    *,
    longitudinal_count: int = 5,
    ring_radii: Any = (0.5, 1.0),
    angles_per_ring: int = 8,
) -> dict[str, Any]:
    """Build paired fixed samples from a longitudinal-by-disk product design."""

    if (
        isinstance(longitudinal_count, bool)
        or int(longitudinal_count) != longitudinal_count
    ):
        raise ValueError("longitudinal_count must be an integer")
    count = int(longitudinal_count)
    if count < 2:
        raise ValueError("longitudinal_count must be at least two")

    path = np.linspace(0.0, 1.0, count, dtype=np.float64)
    disk = deterministic_unit_disk_offsets(
        ring_radii=ring_radii,
        angles_per_ring=angles_per_ring,
    )
    fractions = np.repeat(path, disk.shape[0])
    offsets = np.tile(disk, (path.size, 1))
    return {
        "contract_version": CONTRACT_VERSION,
        "longitudinal_fractions": np.ascontiguousarray(fractions, dtype=np.float64),
        "unit_disk_offsets": np.ascontiguousarray(offsets, dtype=np.float64),
        "longitudinal_nodes": path,
        "disk_offsets": disk,
        "longitudinal_count": int(path.size),
        "disk_point_count": int(disk.shape[0]),
        "sample_count": int(fractions.size),
    }


def _van_der_corput(indices: np.ndarray, base: int) -> np.ndarray:
    values = np.zeros(indices.shape, dtype=np.float64)
    denominator = 1.0
    working = np.asarray(indices, dtype=np.int64).copy()
    while np.any(working):
        working, remainder = np.divmod(working, base)
        denominator *= base
        values += remainder / denominator
    return values


def deterministic_paired_uniform_aperture_samples(
    sample_count: int = 16,
) -> dict[str, Any]:
    """Build paired low-discrepancy samples for the released random sampler.

    The released implementation independently draws a uniform longitudinal
    coordinate, a radius with ``sqrt(U)`` disk-area law, and a uniform angle.
    This deterministic Hammersley-style design mirrors those marginals without
    using endpoints or assigning special weight to the disk boundary.  It is an
    integration probe, not a proof that the continuous aperture is contained.
    """

    if isinstance(sample_count, bool) or int(sample_count) != sample_count:
        raise ValueError("sample_count must be an integer")
    count = int(sample_count)
    if count < 2:
        raise ValueError("sample_count must be at least two")
    index = np.arange(count, dtype=np.int64)
    fractions = (index.astype(np.float64) + 0.5) / count
    radial_uniform = _van_der_corput(index + 1, 2)
    angular_uniform = _van_der_corput(index + 1, 3)
    radius = np.sqrt(radial_uniform)
    angle = 2.0 * np.pi * angular_uniform
    offsets = np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))
    offsets[np.abs(offsets) < 8.0 * np.finfo(np.float64).eps] = 0.0
    return {
        "contract_version": CONTRACT_VERSION,
        "design": "PAIRED_HAMMERSLEY_UNIFORM_PATH_AND_DISK_INTERIOR",
        "longitudinal_fractions": np.ascontiguousarray(
            fractions, dtype=np.float64
        ),
        "unit_disk_offsets": np.ascontiguousarray(offsets, dtype=np.float64),
        "sample_count": count,
        "contains_longitudinal_endpoints": False,
        "contains_disk_boundary": False,
    }


def generate_aperture_sample_points(
    ipf: Any,
    epf: Any,
    rx: Any,
    ry: Any,
    rin: Any,
    rout: Any,
    longitudinal_fractions: Any,
    unit_disk_offsets: Any,
    *,
    basis_tolerance: float = _DEFAULT_BASIS_TOLERANCE,
) -> np.ndarray:
    """Generate released-form finite-aperture points with shape ``(N, S, 3)``.

    Fractions and disk offsets are paired sample by sample.  Use
    :func:`deterministic_aperture_quadrature` to construct a deterministic
    Cartesian-product design.
    """

    start = _as_rows(ipf, "ipf")
    stop = _as_rows(epf, "epf")
    basis_x = _as_rows(rx, "Rx")
    basis_y = _as_rows(ry, "Ry")
    ray_count = start.shape[0]
    for name, array in (("epf", stop), ("Rx", basis_x), ("Ry", basis_y)):
        if array.shape[0] != ray_count:
            raise ValueError(f"{name} must contain the same number of rays as ipf")

    radius_in = _as_ray_scalars(rin, "Rin", ray_count)
    radius_out = _as_ray_scalars(rout, "Rout", ray_count)
    if np.any(radius_in < 0.0) or np.any(radius_out < 0.0):
        raise ValueError("Rin and Rout must contain non-negative radii")
    _validate_basis(basis_x, basis_y, tolerance=basis_tolerance)

    fractions = _as_fractions(longitudinal_fractions)
    offsets = _as_disk_offsets(unit_disk_offsets)
    if fractions.size != offsets.shape[0]:
        raise ValueError(
            "longitudinal_fractions and unit_disk_offsets must have the same "
            "sample count"
        )

    fraction = fractions[None, :, None]
    center = start[:, None, :] + fraction * (stop - start)[:, None, :]
    radius = radius_in[:, None] + fractions[None, :] * (radius_out - radius_in)[:, None]
    transverse = (
        offsets[None, :, 0:1] * basis_x[:, None, :]
        + offsets[None, :, 1:2] * basis_y[:, None, :]
    )
    points = center + radius[:, :, None] * transverse
    if not np.all(np.isfinite(points)):
        raise ValueError("generated aperture points overflowed float64")
    return np.ascontiguousarray(points, dtype=np.float64)


def evaluate_aperture_domain(
    points: Any,
    box_minimum: Any,
    box_maximum: Any,
    *,
    cone_vertex: Any | None = None,
    cone_axis: Any | None = None,
    cone_theta: float | None = None,
    atol: float = 0.0,
    rtol: float = _DEFAULT_RTOL,
) -> dict[str, Any]:
    """Evaluate sample-level B0 or B1 membership without changing sample count."""

    absolute, relative = _validate_tolerances(atol, rtol)
    samples = np.asarray(points, dtype=np.float64)
    if samples.ndim != 3 or samples.shape[2] != 3:
        raise ValueError("points must have shape (N, S, 3)")
    if samples.shape[0] == 0 or samples.shape[1] == 0:
        raise ValueError("points must contain at least one ray and one sample")
    if not np.all(np.isfinite(samples)):
        raise ValueError("points must contain only finite values")
    samples = np.ascontiguousarray(samples, dtype=np.float64)

    lower = _as_vector3(box_minimum, "box_minimum")
    upper = _as_vector3(box_maximum, "box_maximum")
    if np.any(lower > upper):
        raise ValueError("box_minimum must be componentwise <= box_maximum")

    coordinate_scale = np.maximum.reduce(
        (
            np.ones_like(samples),
            np.abs(samples),
            np.broadcast_to(np.abs(lower), samples.shape),
            np.broadcast_to(np.abs(upper), samples.shape),
        )
    )
    box_tolerance = absolute + relative * coordinate_scale
    box_indicator = np.all(
        (samples >= lower[None, None, :] - box_tolerance)
        & (samples <= upper[None, None, :] + box_tolerance),
        axis=2,
    )

    supplied = (
        cone_vertex is not None,
        cone_axis is not None,
        cone_theta is not None,
    )
    if any(supplied) and not all(supplied):
        raise ValueError(
            "cone_vertex, cone_axis, and cone_theta must be supplied together"
        )

    cone_indicator = np.ones(box_indicator.shape, dtype=np.bool_)
    cone_axis_unit: np.ndarray | None = None
    if all(supplied):
        vertex = _as_vector3(cone_vertex, "cone_vertex")
        axis = _as_vector3(cone_axis, "cone_axis")
        axis_norm = float(np.linalg.norm(axis))
        if not np.isfinite(axis_norm) or axis_norm == 0.0:
            raise ValueError("cone_axis must have a finite nonzero norm")
        cone_axis_unit = np.ascontiguousarray(axis / axis_norm, dtype=np.float64)

        theta = float(cone_theta)
        if not np.isfinite(theta) or not (0.0 < theta < 0.5 * np.pi):
            raise ValueError(
                "cone_theta must be finite and strictly between 0 and pi/2"
            )
        tangent = float(np.tan(theta))
        if not np.isfinite(tangent):
            raise ValueError("cone_theta produces a non-finite cone slope")

        displacement = samples - vertex[None, None, :]
        alpha = np.sum(
            displacement * cone_axis_unit[None, None, :],
            axis=2,
        )
        perpendicular = displacement - alpha[:, :, None] * cone_axis_unit[None, None, :]
        perpendicular_norm = np.linalg.norm(perpendicular, axis=2)
        cone_scale = np.maximum(
            1.0,
            np.maximum(
                np.linalg.norm(displacement, axis=2),
                np.abs(alpha),
            ),
        )
        cone_tolerance = absolute + relative * cone_scale
        cone_indicator = (alpha >= -cone_tolerance) & (
            perpendicular_norm <= np.maximum(alpha, 0.0) * tangent + cone_tolerance
        )

    indicator = np.ascontiguousarray(
        box_indicator & cone_indicator,
        dtype=np.bool_,
    )
    all_in_domain = np.all(indicator, axis=1)
    any_out_of_domain = np.any(~indicator, axis=1)
    in_domain_fraction = np.mean(indicator, axis=1, dtype=np.float64)
    empty_domain = ~np.any(indicator, axis=1)
    all_in_domain = np.ascontiguousarray(all_in_domain, dtype=np.bool_)
    any_out_of_domain = np.ascontiguousarray(any_out_of_domain, dtype=np.bool_)

    return {
        "contract_version": CONTRACT_VERSION,
        "domain": "B1" if all(supplied) else "B0",
        "indicator": indicator,
        "box_indicator": np.ascontiguousarray(box_indicator, dtype=np.bool_),
        "cone_indicator": np.ascontiguousarray(cone_indicator, dtype=np.bool_),
        "all_in_domain": all_in_domain,
        "all_in": all_in_domain,
        "any_out_of_domain": any_out_of_domain,
        "any_out": any_out_of_domain,
        "in_domain_fraction": np.ascontiguousarray(
            in_domain_fraction, dtype=np.float64
        ),
        "empty_domain": np.ascontiguousarray(empty_domain, dtype=np.bool_),
        "sample_count": int(indicator.shape[1]),
        "cone_axis_unit": cone_axis_unit,
    }


def _broadcast_ray_vectors(value: Any, name: str, ray_count: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (3,):
        array = np.broadcast_to(array, (ray_count, 3))
    elif array.shape != (ray_count, 3):
        raise ValueError(f"{name} must have shape (3,) or (N, 3)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _broadcast_ray_values(value: Any, name: str, ray_count: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(ray_count, float(array), dtype=np.float64)
    elif array.shape != (ray_count,):
        raise ValueError(f"{name} must be scalar or have shape (N,)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def constant_gradient_los_smoke(
    gradient: Any,
    sensitivity: Any,
    path_length: Any,
    indicator: Any,
    *,
    system_constant: Any = 1.0,
) -> dict[str, Any]:
    """Evaluate a constant-gradient LoS sum with the original denominator.

    For ray ``i`` this computes

    ``C_i * L_i / S * sum_j indicator[i, j] * dot(sensitivity_i, gradient_i)``,

    where ``S`` is the fixed input sample count.  Empty rays therefore produce
    zero, and partially clipped apertures retain their reduced physical weight.
    """

    raw_indicator = np.asarray(indicator)
    if raw_indicator.ndim != 2 or raw_indicator.shape[0] == 0:
        raise ValueError("indicator must have shape (N, S)")
    if raw_indicator.shape[1] == 0:
        raise ValueError("indicator must contain at least one sample")
    if np.issubdtype(raw_indicator.dtype, np.number):
        numeric_indicator = np.asarray(raw_indicator, dtype=np.float64)
        if not np.all(np.isfinite(numeric_indicator)):
            raise ValueError("indicator must contain only finite values")
        if np.any((numeric_indicator != 0.0) & (numeric_indicator != 1.0)):
            raise ValueError("numeric indicator values must be exactly zero or one")
        mask = numeric_indicator.astype(np.bool_)
    else:
        mask = np.asarray(raw_indicator, dtype=np.bool_)

    ray_count, original_sample_count = mask.shape
    gradients = _broadcast_ray_vectors(gradient, "gradient", ray_count)
    sensitivities = _broadcast_ray_vectors(sensitivity, "sensitivity", ray_count)
    lengths = _broadcast_ray_values(path_length, "path_length", ray_count)
    constants = _broadcast_ray_values(system_constant, "system_constant", ray_count)
    if np.any(lengths < 0.0):
        raise ValueError("path_length must be non-negative")

    projected_gradient = np.sum(gradients * sensitivities, axis=1)
    surviving_sample_count = np.count_nonzero(mask, axis=1)
    weighted_sum = projected_gradient * surviving_sample_count
    prediction = constants * lengths * weighted_sum / np.float64(original_sample_count)
    return {
        "contract_version": CONTRACT_VERSION,
        "prediction": np.ascontiguousarray(prediction, dtype=np.float64),
        "projected_gradient": np.ascontiguousarray(
            projected_gradient, dtype=np.float64
        ),
        "original_sample_count": int(original_sample_count),
        "surviving_sample_count": np.ascontiguousarray(
            surviving_sample_count, dtype=np.int64
        ),
        "in_domain_fraction": np.ascontiguousarray(
            surviving_sample_count / np.float64(original_sample_count),
            dtype=np.float64,
        ),
    }


# Short descriptive aliases for callers that use B2 terminology.
generate_b2_sample_points = generate_aperture_sample_points
evaluate_b2_domain = evaluate_aperture_domain
b2_constant_gradient_los_smoke = constant_gradient_los_smoke


__all__ = [
    "CONTRACT_VERSION",
    "APERTURE_DOMAIN_CONTRACT",
    "deterministic_unit_disk_offsets",
    "deterministic_aperture_quadrature",
    "deterministic_paired_uniform_aperture_samples",
    "generate_aperture_sample_points",
    "generate_b2_sample_points",
    "evaluate_aperture_domain",
    "evaluate_b2_domain",
    "constant_gradient_los_smoke",
    "b2_constant_gradient_los_smoke",
]
