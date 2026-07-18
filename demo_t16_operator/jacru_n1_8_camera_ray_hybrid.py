"""Camera/ray-conditioned hybrid measurement frames for JACRU N1.8.

Every basis is assembled from deployment-visible quantities, two applications
of the current solver-consistent ``A P A^T`` map, and optionally two centered
fit-only mismatch modes.  Truth from the evaluated case is never an input to
basis construction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

import torch

from .jacru_synthetic_fixture import JACRUGeometry


Tensor = torch.Tensor
LinearMap = Callable[[Tensor], Tensor]
SUPPORTED_BASIS_KINDS = (
    "krylov4_total",
    "fit_pca2_krylov6_total",
    "camera_block6_total",
    "pose_fourier_krylov6_total",
    "detector_moment_krylov6_total",
)


def _finite_observation(values: Tensor, *, name: str) -> Tensor:
    observation = torch.as_tensor(values, dtype=torch.float64)
    if observation.ndim < 1 or observation.numel() < 1:
        raise ValueError(f"{name} must be a nonempty tensor")
    if not bool(torch.all(torch.isfinite(observation))):
        raise ValueError(f"{name} must contain only finite values")
    return observation


@dataclass(frozen=True)
class HybridMeasurementBasis:
    """Orthonormal total-correction basis with an explicit setup ledger."""

    kind: str
    names: tuple[str, ...]
    vectors: Tensor
    raw_norms: tuple[float, ...]
    dropped_names: tuple[str, ...]
    orthonormality_defect: float
    setup_forward_calls: int
    setup_adjoint_calls: int
    fit_mode_count: int
    uses_evaluated_case_truth: bool = False

    @property
    def rank(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.vectors.shape[1:])

    def synthesize(self, coefficients: Tensor) -> Tensor:
        values = torch.as_tensor(coefficients, dtype=torch.float64).reshape(-1)
        if values.numel() != self.rank:
            raise ValueError("coefficient count must match basis rank")
        if not bool(torch.all(torch.isfinite(values))):
            raise ValueError("coefficients must be finite")
        return torch.einsum("r,r...->...", values, self.vectors)


def _orthonormalize(
    *,
    kind: str,
    named_vectors: Sequence[tuple[str, Tensor]],
    observation_shape: tuple[int, ...],
    dependence_tolerance: float,
    setup_forward_calls: int,
    setup_adjoint_calls: int,
    fit_mode_count: int,
) -> HybridMeasurementBasis:
    tolerance = float(dependence_tolerance)
    if not math.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
        raise ValueError("dependence_tolerance must lie in (0,1)")
    accepted_names: list[str] = []
    accepted: list[Tensor] = []
    raw_norms: list[float] = []
    dropped: list[str] = []
    for name, raw in named_vectors:
        vector = _finite_observation(raw, name=name)
        if tuple(vector.shape) != observation_shape:
            raise ValueError("all basis vectors must match the observation shape")
        flat = vector.reshape(-1).clone()
        raw_norm = float(torch.linalg.vector_norm(flat))
        raw_norms.append(raw_norm)
        if not math.isfinite(raw_norm) or raw_norm <= 1e-30:
            dropped.append(name)
            continue
        candidate = flat / raw_norm
        for _ in range(2):
            for existing in accepted:
                candidate = candidate - torch.dot(candidate, existing) * existing
        remaining = float(torch.linalg.vector_norm(candidate))
        if remaining <= tolerance:
            dropped.append(name)
            continue
        accepted_names.append(name)
        accepted.append(candidate / remaining)
    if not accepted:
        raise RuntimeError("all hybrid basis vectors were numerically dependent")
    matrix = torch.stack(accepted)
    gram = matrix @ matrix.T
    defect = float(
        torch.max(torch.abs(gram - torch.eye(matrix.shape[0], dtype=matrix.dtype)))
    )
    return HybridMeasurementBasis(
        kind=kind,
        names=tuple(accepted_names),
        vectors=matrix.reshape((matrix.shape[0],) + observation_shape),
        raw_norms=tuple(raw_norms),
        dropped_names=tuple(dropped),
        orthonormality_defect=defect,
        setup_forward_calls=int(setup_forward_calls),
        setup_adjoint_calls=int(setup_adjoint_calls),
        fit_mode_count=int(fit_mode_count),
    )


def _normal_probes(
    *,
    seeds: Sequence[tuple[str, Tensor]],
    forward: LinearMap,
    adjoint: LinearMap,
    support: Tensor,
) -> list[tuple[str, Tensor]]:
    mask = torch.as_tensor(support, dtype=torch.float64)
    if mask.ndim < 1 or mask.numel() < 1 or not bool(torch.all(torch.isfinite(mask))):
        raise ValueError("support must be a nonempty finite tensor")
    if not bool(torch.any(mask != 0.0)):
        raise ValueError("support must retain at least one field element")
    if not seeds:
        return []
    reference = _finite_observation(seeds[0][1], name=seeds[0][0])
    outputs: list[tuple[str, Tensor]] = []
    for seed_name, seed_value in seeds:
        seed = _finite_observation(seed_value, name=seed_name)
        if seed.shape != reference.shape:
            raise ValueError("normal-probe seeds must share one observation shape")
        name = f"normal_{seed_name}"
        field = torch.as_tensor(adjoint(seed), dtype=torch.float64)
        if field.shape != mask.shape or not bool(torch.all(torch.isfinite(field))):
            raise ValueError("support shape must match finite adjoint output")
        projected = _finite_observation(forward(field * mask), name=name)
        if projected.shape != reference.shape:
            raise ValueError("forward(adjoint(seed)) changed the observation shape")
        outputs.append((name, projected))
    return outputs


def _centered_unit_rms(values: Tensor) -> Tensor:
    weights = torch.as_tensor(values, dtype=torch.float64)
    weights = weights - torch.mean(weights)
    rms = torch.sqrt(torch.mean(weights.square()))
    return torch.zeros_like(weights) if float(rms) <= 1e-14 else weights / rms


def camera_block_vectors(
    *,
    damping: Tensor,
    warm_residual: Tensor,
    geometry: JACRUGeometry,
) -> list[tuple[str, Tensor]]:
    """Return per-camera masked damping and residual directions."""

    anchor = _finite_observation(damping, name="damping")
    residual = _finite_observation(warm_residual, name="warm_residual")
    if anchor.shape != (geometry.ray_count, 2) or residual.shape != anchor.shape:
        raise ValueError("camera-block inputs must have shape [ray,2]")
    camera_index = geometry.camera_index.to(torch.int64)
    expected_labels = torch.arange(geometry.camera_count, dtype=torch.int64)
    if not torch.equal(torch.unique(camera_index, sorted=True), expected_labels):
        raise ValueError("camera_index labels must be contiguous from zero")
    vectors: list[tuple[str, Tensor]] = []
    for camera in range(geometry.camera_count):
        mask = (camera_index == camera)[:, None]
        if not bool(torch.any(mask)):
            raise ValueError("every camera must contain at least one ray")
        vectors.extend(
            [
                (f"camera_{camera}_damping", torch.where(mask, anchor, 0.0)),
                (
                    f"camera_{camera}_warm_residual",
                    torch.where(mask, residual, 0.0),
                ),
            ]
        )
    return vectors


def pose_fourier_weighted_residuals(
    warm_residual: Tensor, *, geometry: JACRUGeometry
) -> tuple[Tensor, Tensor]:
    """Return first-harmonic azimuth modulations of the visible residual."""

    residual = _finite_observation(warm_residual, name="warm_residual")
    if residual.shape != (geometry.ray_count, 2):
        raise ValueError("warm_residual must have shape [ray,2]")
    azimuth = torch.deg2rad(
        torch.as_tensor(geometry.camera_azimuth_degrees, dtype=torch.float64)
    )
    if azimuth.shape != (geometry.camera_count,):
        raise ValueError("camera azimuth count drifted")
    camera_index = geometry.camera_index.to(torch.int64)
    sine = _centered_unit_rms(torch.sin(azimuth))[camera_index]
    cosine = _centered_unit_rms(torch.cos(azimuth))[camera_index]
    return residual * sine[:, None], residual * cosine[:, None]


def detector_moment_weighted_residuals(
    warm_residual: Tensor, *, geometry: JACRUGeometry
) -> tuple[Tensor, Tensor]:
    """Return detector-u/v first-moment modulations for the current fixture.

    The current synthetic payload does not store per-ray detector coordinates,
    so this helper explicitly rejects non-camera-major or missing-ray layouts.
    A real-BOST adapter must pass authoritative pixel coordinates instead.
    """

    residual = _finite_observation(warm_residual, name="warm_residual")
    if residual.shape != (geometry.ray_count, 2):
        raise ValueError("warm_residual must have shape [ray,2]")
    rows, columns = (int(value) for value in geometry.detector_shape)
    rays_per_camera = rows * columns
    expected_camera_index = torch.arange(
        geometry.camera_count, dtype=torch.int64
    ).repeat_interleave(rays_per_camera)
    if not torch.equal(geometry.camera_index.to(torch.int64), expected_camera_index):
        raise ValueError(
            "detector-moment candidate requires authoritative camera-major ray ordering"
        )
    detector_v, detector_u = torch.meshgrid(
        torch.linspace(-1.0, 1.0, rows, dtype=torch.float64),
        torch.linspace(-1.0, 1.0, columns, dtype=torch.float64),
        indexing="ij",
    )
    u = _centered_unit_rms(detector_u.reshape(-1)).repeat(geometry.camera_count)
    v = _centered_unit_rms(detector_v.reshape(-1)).repeat(geometry.camera_count)
    return residual * u[:, None], residual * v[:, None]


def visible_total_correction_radius(
    damping: Tensor,
    warm_residual: Tensor,
    *,
    damping_floor_multiplier: float = 1.0,
    warm_residual_multiplier: float = 16.0,
    damping_cap_multiplier: float = 2.0,
) -> float:
    """Return a visible bound on the norm of the entire correction.

    The damping floor guarantees that the simple component-damping baseline is
    representable.  Unlike N1.7, this radius applies to the total correction,
    not only to a residual added after an uncounted anchor.
    """

    anchor = _finite_observation(damping, name="damping")
    residual = _finite_observation(warm_residual, name="warm_residual")
    if anchor.shape != residual.shape:
        raise ValueError("damping and warm_residual must have identical shape")
    floor_multiplier = float(damping_floor_multiplier)
    residual_multiplier = float(warm_residual_multiplier)
    cap_multiplier = float(damping_cap_multiplier)
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (floor_multiplier, residual_multiplier, cap_multiplier)
    ):
        raise ValueError("radius multipliers must be positive and finite")
    if floor_multiplier > cap_multiplier:
        raise ValueError("damping floor cannot exceed damping cap")
    damping_norm = float(torch.linalg.vector_norm(anchor))
    residual_norm = float(torch.linalg.vector_norm(residual))
    if not math.isfinite(damping_norm) or damping_norm <= 1e-30:
        raise ValueError("damping norm must be positive")
    return min(
        cap_multiplier * damping_norm,
        max(floor_multiplier * damping_norm, residual_multiplier * residual_norm),
    )


def build_camera_ray_hybrid_basis(
    *,
    kind: str,
    damping: Tensor,
    warm_residual: Tensor,
    forward: LinearMap,
    adjoint: LinearMap,
    support: Tensor,
    geometry: JACRUGeometry,
    fit_modes: Tensor | None = None,
    dependence_tolerance: float = 1e-10,
) -> HybridMeasurementBasis:
    """Build one of the frozen N1.8 total-correction candidate frames."""

    basis_kind = str(kind)
    if basis_kind not in SUPPORTED_BASIS_KINDS:
        raise ValueError(f"unsupported N1.8 basis kind: {basis_kind}")
    anchor = _finite_observation(damping, name="damping")
    residual = _finite_observation(warm_residual, name="warm_residual")
    expected = (geometry.ray_count, 2)
    if anchor.shape != expected or residual.shape != expected:
        raise ValueError(f"damping and warm_residual must have shape {expected}")
    named: list[tuple[str, Tensor]] = []
    probe_seeds: list[tuple[str, Tensor]] = []
    fit_mode_count = 0
    if basis_kind == "krylov4_total":
        named.extend([("damping", anchor), ("warm_residual", residual)])
        probe_seeds = [("damping", anchor), ("warm_residual", residual)]
    elif basis_kind == "fit_pca2_krylov6_total":
        modes = torch.as_tensor(fit_modes, dtype=torch.float64)
        if modes.shape != (2, anchor.numel()) or not bool(torch.all(torch.isfinite(modes))):
            raise ValueError("fit_pca2 basis requires two finite flattened fit modes")
        named.extend([("damping", anchor), ("warm_residual", residual)])
        named.extend(
            (f"fit_pca_mode_{index + 1}", mode.reshape_as(anchor))
            for index, mode in enumerate(modes)
        )
        probe_seeds = [("damping", anchor), ("warm_residual", residual)]
        fit_mode_count = 2
    elif basis_kind == "camera_block6_total":
        named.extend(
            camera_block_vectors(
                damping=anchor, warm_residual=residual, geometry=geometry
            )
        )
    elif basis_kind == "pose_fourier_krylov6_total":
        sine_residual, cosine_residual = pose_fourier_weighted_residuals(
            residual, geometry=geometry
        )
        named.extend(
            [
                ("damping", anchor),
                ("warm_residual", residual),
                ("sin_azimuth_warm_residual", sine_residual),
                ("cos_azimuth_warm_residual", cosine_residual),
            ]
        )
        probe_seeds = [
            ("sin_azimuth_warm_residual", sine_residual),
            ("cos_azimuth_warm_residual", cosine_residual),
        ]
    elif basis_kind == "detector_moment_krylov6_total":
        u_residual, v_residual = detector_moment_weighted_residuals(
            residual, geometry=geometry
        )
        named.extend(
            [
                ("damping", anchor),
                ("warm_residual", residual),
                ("detector_u_warm_residual", u_residual),
                ("detector_v_warm_residual", v_residual),
            ]
        )
        probe_seeds = [
            ("detector_u_warm_residual", u_residual),
            ("detector_v_warm_residual", v_residual),
        ]
    if fit_modes is not None and basis_kind != "fit_pca2_krylov6_total":
        raise ValueError("fit_modes are only allowed for fit_pca2_krylov6_total")
    probed = _normal_probes(
        seeds=probe_seeds,
        forward=forward,
        adjoint=adjoint,
        support=support,
    )
    named.extend(probed)
    return _orthonormalize(
        kind=basis_kind,
        named_vectors=named,
        observation_shape=tuple(anchor.shape),
        dependence_tolerance=dependence_tolerance,
        setup_forward_calls=len(probed),
        setup_adjoint_calls=len(probed),
        fit_mode_count=fit_mode_count,
    )


__all__ = [
    "HybridMeasurementBasis",
    "SUPPORTED_BASIS_KINDS",
    "build_camera_ray_hybrid_basis",
    "camera_block_vectors",
    "detector_moment_weighted_residuals",
    "pose_fourier_weighted_residuals",
    "visible_total_correction_radius",
]
