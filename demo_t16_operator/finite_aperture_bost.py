"""Deterministic finite-aperture operators for a prescribed weak-BOST surrogate.

The model here is only a linear weak-deflection model on a prescribed cone/curve
geometry.  Each detector pixel is replaced by the arithmetic mean of a fixed,
deterministic set of circular-disk sub-rays.  This is a geometry surrogate, not
nonlinear ray tracing and not a complete reproduction of the model of Molnar et
al.  In particular, it should be used as a controlled finite-aperture baseline,
not as a claim of physical or paper-level model equivalence.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

try:
    from .independent_reaction_bost import _derivative_matrix, _trilinear_row
except ImportError:
    from independent_reaction_bost import _derivative_matrix, _trilinear_row


def _disk_subrays(count: int) -> np.ndarray:
    """Return deterministic unit-disk coordinates, including the disk center."""

    if int(count) < 1:
        raise ValueError("aperture_samples must be at least one")
    count = int(count)
    if count == 1:
        return np.zeros((1, 2), dtype=np.float64)
    indices = np.arange(count - 1, dtype=np.float64)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    radii = np.sqrt((indices + 0.5) / float(count - 1))
    angles = golden_angle * indices
    ring = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    return np.vstack((np.zeros((1, 2), dtype=np.float64), ring))


def _validate_radii(radii: Sequence[float]) -> np.ndarray:
    values = np.asarray(tuple(radii), dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("radii must be a non-empty one-dimensional sequence")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("radii must contain finite non-negative values")
    return values


def _validate_disk_points(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] == 0:
        raise ValueError("normalized_disk_points must have shape [sample, 2]")
    if np.any(~np.isfinite(values)):
        raise ValueError("normalized_disk_points must be finite")
    if np.any(np.sum(values * values, axis=1) > 1.0 + 1e-12):
        raise ValueError("normalized_disk_points must lie in the unit disk")
    return values


def _raw_operator(
    n: int,
    depth: int,
    angles_degrees: np.ndarray,
    *,
    aperture_radius: float,
    aperture_samples: int,
    path_samples: int,
    cone_u: float,
    cone_z: float,
    bend: float,
) -> np.ndarray:
    shape = (int(depth), int(n), int(n))
    dz = _derivative_matrix(shape, 0)
    dy = _derivative_matrix(shape, 1)
    dx = _derivative_matrix(shape, 2)
    detector_x = np.linspace(-0.82, 0.82, int(n))
    detector_z = np.linspace(-0.82, 0.82, int(depth))
    path = np.linspace(-1.5, 1.5, int(path_samples))
    dl = float(path[1] - path[0])
    disk = _disk_subrays(aperture_samples)
    rows = np.zeros((depth, len(angles_degrees), n, depth * n * n), dtype=np.float64)

    for z_index, z0 in enumerate(detector_z):
        for view_index, angle in enumerate(np.asarray(angles_degrees, dtype=float)):
            theta = np.deg2rad(angle)
            line = np.array([np.cos(theta), np.sin(theta), 0.0])
            transverse = np.array([-np.sin(theta), np.cos(theta), 0.0])
            for detector_index, u0 in enumerate(detector_x):
                pixel_row = np.zeros(depth * n * n, dtype=np.float64)
                for disk_x, disk_z in disk:
                    subray_u = float(u0 + aperture_radius * disk_x)
                    subray_z = float(z0 + aperture_radius * disk_z)
                    direction = line + float(cone_u) * subray_u * transverse
                    direction = direction + float(cone_z) * subray_z * np.array([0.0, 0.0, 1.0])
                    direction /= np.linalg.norm(direction)
                    sensitivity = transverse - np.dot(transverse, direction) * direction
                    sensitivity /= np.linalg.norm(sensitivity)
                    interpolation = np.zeros(depth * n * n, dtype=np.float64)
                    for distance in path:
                        normalized = distance / max(abs(path[0]), abs(path[-1]))
                        curve = (
                            float(bend)
                            * (1.0 - normalized * normalized)
                            * (0.35 + abs(subray_u))
                            * transverse
                        )
                        point = subray_u * transverse + subray_z * np.array([0.0, 0.0, 1.0])
                        point = point + distance * direction + curve
                        interpolation += _trilinear_row(
                            float(point[0]), float(point[1]), float(point[2]), shape
                        ) * dl
                    derivative = (
                        sensitivity[0] * dx + sensitivity[1] * dy + sensitivity[2] * dz
                    )
                    pixel_row += interpolation @ derivative
                rows[z_index, view_index, detector_index] = pixel_row / float(len(disk))
    return rows


def _raw_subray_operator_bank(
    n: int,
    depth: int,
    angles_degrees: np.ndarray,
    *,
    aperture_radius: float,
    normalized_disk_points: np.ndarray,
    path_samples: int,
    cone_u: float,
    cone_z: float,
    bend: float,
) -> np.ndarray:
    """Return one unnormalised operator for each prescribed aperture point."""

    shape = (int(depth), int(n), int(n))
    dz = _derivative_matrix(shape, 0)
    dy = _derivative_matrix(shape, 1)
    dx = _derivative_matrix(shape, 2)
    detector_x = np.linspace(-0.82, 0.82, int(n))
    detector_z = np.linspace(-0.82, 0.82, int(depth))
    path = np.linspace(-1.5, 1.5, int(path_samples))
    dl = float(path[1] - path[0])
    disk = _validate_disk_points(normalized_disk_points)
    rows = np.zeros(
        (len(disk), depth, len(angles_degrees), n, depth * n * n),
        dtype=np.float64,
    )

    for sample_index, (disk_x, disk_z) in enumerate(disk):
        for z_index, z0 in enumerate(detector_z):
            for view_index, angle in enumerate(np.asarray(angles_degrees, dtype=float)):
                theta = np.deg2rad(angle)
                line = np.array([np.cos(theta), np.sin(theta), 0.0])
                transverse = np.array([-np.sin(theta), np.cos(theta), 0.0])
                for detector_index, u0 in enumerate(detector_x):
                    subray_u = float(u0 + aperture_radius * disk_x)
                    subray_z = float(z0 + aperture_radius * disk_z)
                    direction = line + float(cone_u) * subray_u * transverse
                    direction = direction + float(cone_z) * subray_z * np.array(
                        [0.0, 0.0, 1.0]
                    )
                    direction /= np.linalg.norm(direction)
                    sensitivity = transverse - np.dot(transverse, direction) * direction
                    sensitivity /= np.linalg.norm(sensitivity)
                    interpolation = np.zeros(depth * n * n, dtype=np.float64)
                    for distance in path:
                        normalized = distance / max(abs(path[0]), abs(path[-1]))
                        curve = (
                            float(bend)
                            * (1.0 - normalized * normalized)
                            * (0.35 + abs(subray_u))
                            * transverse
                        )
                        point = subray_u * transverse + subray_z * np.array(
                            [0.0, 0.0, 1.0]
                        )
                        point = point + distance * direction + curve
                        interpolation += _trilinear_row(
                            float(point[0]), float(point[1]), float(point[2]), shape
                        ) * dl
                    derivative = (
                        sensitivity[0] * dx
                        + sensitivity[1] * dy
                        + sensitivity[2] * dz
                    )
                    rows[sample_index, z_index, view_index, detector_index] = (
                        interpolation @ derivative
                    )
    return rows


def finite_aperture_reference_scale(
    n: int,
    depth: int,
    angles_degrees: np.ndarray,
    *,
    aperture_samples: int = 17,
    path_samples: int = 22,
    cone_u: float = 0.07,
    cone_z: float = 0.05,
    bend: float = 0.035,
) -> float:
    """Return the physical scale used to normalize one renderer discretization.

    The scale is the median nonzero row norm of the radius-zero operator.  It is
    exposed so truth and reconstruction renderers can either share one declared
    scale or report their scale mismatch explicitly.  Exposing the value does
    not change the historical default behavior of this module.
    """

    reference = _raw_operator(
        n,
        depth,
        np.asarray(angles_degrees, dtype=float),
        aperture_radius=0.0,
        aperture_samples=aperture_samples,
        path_samples=path_samples,
        cone_u=cone_u,
        cone_z=cone_z,
        bend=bend,
    )
    norms = np.linalg.vector_norm(reference.reshape(-1, reference.shape[-1]), axis=1)
    nonzero = norms[norms > 1e-10]
    scale = float(np.median(nonzero)) if nonzero.size else 0.0
    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError("finite-aperture operator contains no valid rays")
    return scale


def build_finite_aperture_operator(
    n: int,
    depth: int,
    angles_degrees: np.ndarray,
    *,
    aperture_radius: float = 0.0,
    aperture_samples: int = 17,
    path_samples: int = 22,
    cone_u: float = 0.07,
    cone_z: float = 0.05,
    bend: float = 0.035,
    normalization_scale: float | None = None,
) -> np.ndarray:
    """Build one normalized ``[depth,view,detector,voxel]`` operator.

    ``aperture_radius=0`` is the single center ray (the repeated center samples
    average exactly to that ray).  The normalization is the median nonzero row
    norm of this radius-zero operator, so changing the aperture does not change
    the reference scale.  This remains a linear weak-BOST prescribed geometry
    surrogate, not nonlinear ray tracing or a full Molnar et al. reproduction.
    """

    if n < 3 or depth < 2 or path_samples < 8:
        raise ValueError("grid and path sampling are too small")
    radius = float(aperture_radius)
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("aperture_radius must be finite and non-negative")
    angles = np.asarray(angles_degrees, dtype=float)
    if angles.ndim != 1 or angles.size == 0:
        raise ValueError("angles_degrees must be a non-empty one-dimensional array")
    scale = (
        finite_aperture_reference_scale(
            n,
            depth,
            angles,
            aperture_samples=aperture_samples,
            path_samples=path_samples,
            cone_u=cone_u,
            cone_z=cone_z,
            bend=bend,
        )
        if normalization_scale is None
        else float(normalization_scale)
    )
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("normalization_scale must be finite and strictly positive")
    raw = _raw_operator(
        n, depth, angles, aperture_radius=radius, aperture_samples=aperture_samples,
        path_samples=path_samples, cone_u=cone_u, cone_z=cone_z, bend=bend,
    )
    output = (raw / scale).astype(np.float32)
    if not np.all(np.isfinite(output)):
        raise RuntimeError("finite-aperture operator contains non-finite values")
    return output


def build_finite_aperture_operator_bank(
    n: int,
    depth: int,
    angles_degrees: np.ndarray,
    radii: Sequence[float],
    *,
    aperture_samples: int = 17,
    path_samples: int = 22,
    cone_u: float = 0.07,
    cone_z: float = 0.05,
    bend: float = 0.035,
    normalization_scale: float | None = None,
) -> np.ndarray:
    """Build ``[radius,depth,view,detector,voxel]`` with one shared scale."""

    values = _validate_radii(radii)
    scale = (
        finite_aperture_reference_scale(
            n,
            depth,
            angles_degrees,
            aperture_samples=aperture_samples,
            path_samples=path_samples,
            cone_u=cone_u,
            cone_z=cone_z,
            bend=bend,
        )
        if normalization_scale is None
        else float(normalization_scale)
    )
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("normalization_scale must be finite and strictly positive")
    operators = [
        build_finite_aperture_operator(
            n, depth, angles_degrees, aperture_radius=float(radius),
            aperture_samples=aperture_samples, path_samples=path_samples,
            cone_u=cone_u, cone_z=cone_z, bend=bend,
            normalization_scale=scale,
        )
        for radius in values
    ]
    bank = np.stack(operators, axis=0).astype(np.float32, copy=False)
    if not np.all(np.isfinite(bank)):
        raise RuntimeError("finite-aperture operator bank contains non-finite values")
    return bank


def build_aperture_subray_operator_bank(
    n: int,
    depth: int,
    angles_degrees: np.ndarray,
    normalized_disk_points: np.ndarray,
    *,
    aperture_radius: float,
    path_samples: int = 22,
    cone_u: float = 0.07,
    cone_z: float = 0.05,
    bend: float = 0.035,
    normalization_scale: float | None = None,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    """Build one operator per prescribed point on the normalized aperture disk.

    The result has shape ``[sample,depth,view,detector,voxel]``.  This interface
    is intended for quadrature and variance-reduction audits: it exposes the
    aperture integrand without changing the historical deterministic renderer.
    It remains the same prescribed weak-deflection geometry surrogate and is
    not nonlinear ray tracing or an experimental camera model.
    """

    if n < 3 or depth < 2 or path_samples < 8:
        raise ValueError("grid and path sampling are too small")
    radius = float(aperture_radius)
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("aperture_radius must be finite and non-negative")
    angles = np.asarray(angles_degrees, dtype=float)
    if angles.ndim != 1 or angles.size == 0:
        raise ValueError("angles_degrees must be a non-empty one-dimensional array")
    disk = _validate_disk_points(normalized_disk_points)
    scale = (
        finite_aperture_reference_scale(
            n,
            depth,
            angles,
            aperture_samples=1,
            path_samples=path_samples,
            cone_u=cone_u,
            cone_z=cone_z,
            bend=bend,
        )
        if normalization_scale is None
        else float(normalization_scale)
    )
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("normalization_scale must be finite and strictly positive")
    raw = _raw_subray_operator_bank(
        n,
        depth,
        angles,
        aperture_radius=radius,
        normalized_disk_points=disk,
        path_samples=path_samples,
        cone_u=cone_u,
        cone_z=cone_z,
        bend=bend,
    )
    output = (raw / scale).astype(dtype, copy=False)
    if not np.all(np.isfinite(output)):
        raise RuntimeError("aperture subray operator bank contains non-finite values")
    return output
