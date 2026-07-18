"""Geometry-conditioned measurement Krylov spaces for JACRU N1.7.

The deployable representation is generated only from a component-damping
anchor, the measured warm residual, and the current low-order ``A/A^T`` pair.
Truth-derived targets are accepted only by the explicitly named evaluator
projection functions below.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

import torch


Tensor = torch.Tensor
LinearMap = Callable[[Tensor], Tensor]


def _finite_observation(values: Tensor, *, name: str) -> Tensor:
    observation = torch.as_tensor(values, dtype=torch.float64)
    if observation.ndim < 1 or observation.numel() < 1:
        raise ValueError(f"{name} must be a nonempty tensor")
    if not bool(torch.all(torch.isfinite(observation))):
        raise ValueError(f"{name} must contain only finite values")
    return observation


def _finite_coefficients(values: Tensor, *, rank: int) -> Tensor:
    coefficients = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    if coefficients.numel() != rank:
        raise ValueError("coefficient count must match basis rank")
    if not bool(torch.all(torch.isfinite(coefficients))):
        raise ValueError("coefficients must be finite")
    return coefficients


@dataclass(frozen=True)
class GeometryKrylovBasis:
    """Per-case orthonormal measurement basis with an explicit call ledger."""

    names: tuple[str, ...]
    vectors: Tensor
    raw_norms: tuple[float, ...]
    dropped_names: tuple[str, ...]
    orthonormality_defect: float
    setup_forward_calls: int
    setup_adjoint_calls: int

    @property
    def rank(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.vectors.shape[1:])

    def synthesize(self, coefficients: Tensor) -> Tensor:
        values = _finite_coefficients(coefficients, rank=self.rank)
        return torch.einsum("r,r...->...", values, self.vectors)


def build_geometry_krylov_basis(
    *,
    damping: Tensor,
    warm_residual: Tensor,
    forward: LinearMap,
    adjoint: LinearMap,
    support: Tensor,
    dependence_tolerance: float = 1e-10,
) -> GeometryKrylovBasis:
    """Build ``span(d, r, AP A^T d, AP A^T r)`` with two operator pairs.

    The ordered, twice-reorthogonalized Gram-Schmidt construction avoids SVD
    sign ambiguity.  Each raw vector is normalized before dependence testing,
    so the tolerance is scale independent.
    """

    anchor = _finite_observation(damping, name="damping")
    residual = _finite_observation(warm_residual, name="warm_residual")
    if anchor.shape != residual.shape:
        raise ValueError("damping and warm_residual must have identical shape")
    if not callable(forward) or not callable(adjoint):
        raise TypeError("forward and adjoint must be callable")
    mask = torch.as_tensor(support, dtype=torch.float64)
    if mask.ndim < 1 or mask.numel() < 1 or not bool(torch.all(torch.isfinite(mask))):
        raise ValueError("support must be a nonempty finite tensor")
    if not bool(torch.any(mask != 0.0)):
        raise ValueError("support must retain at least one field element")
    tolerance = float(dependence_tolerance)
    if not math.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
        raise ValueError("dependence_tolerance must lie in (0,1)")

    probed: list[Tensor] = []
    for name, seed in (("normal_damping", anchor), ("normal_warm_residual", residual)):
        field = torch.as_tensor(adjoint(seed), dtype=torch.float64)
        if field.numel() < 1 or not bool(torch.all(torch.isfinite(field))):
            raise ValueError(f"adjoint returned invalid values for {name}")
        if field.shape != mask.shape:
            raise ValueError("support shape must match the adjoint field")
        projected = _finite_observation(forward(field * mask), name=name)
        if projected.shape != anchor.shape:
            raise ValueError("forward(adjoint(seed)) changed the observation shape")
        probed.append(projected)

    raw_names = ("damping", "warm_residual", "normal_damping", "normal_warm_residual")
    raw_vectors = (anchor, residual, probed[0], probed[1])
    accepted_names: list[str] = []
    accepted: list[Tensor] = []
    raw_norms: list[float] = []
    dropped: list[str] = []
    for name, raw in zip(raw_names, raw_vectors, strict=True):
        flat = raw.reshape(-1).clone()
        raw_norm = float(torch.linalg.vector_norm(flat))
        raw_norms.append(raw_norm)
        if not math.isfinite(raw_norm) or raw_norm <= 1e-30:
            dropped.append(name)
            continue
        candidate = flat / raw_norm
        for _ in range(2):
            for vector in accepted:
                candidate = candidate - torch.dot(candidate, vector) * vector
        remaining = float(torch.linalg.vector_norm(candidate))
        if remaining <= tolerance:
            dropped.append(name)
            continue
        accepted_names.append(name)
        accepted.append(candidate / remaining)
    if not accepted:
        raise RuntimeError("all Krylov seed vectors were numerically dependent")

    matrix = torch.stack(accepted)
    gram = matrix @ matrix.T
    defect = float(
        torch.max(torch.abs(gram - torch.eye(matrix.shape[0], dtype=matrix.dtype)))
    )
    vectors = matrix.reshape((matrix.shape[0],) + tuple(anchor.shape))
    return GeometryKrylovBasis(
        names=tuple(accepted_names),
        vectors=vectors,
        raw_norms=tuple(raw_norms),
        dropped_names=tuple(dropped),
        orthonormality_defect=defect,
        setup_forward_calls=2,
        setup_adjoint_calls=2,
    )


def project_to_l2_ball(coefficients: Tensor, *, radius: float) -> Tensor:
    """Project coefficients onto a target-independent Euclidean trust region."""

    values = torch.as_tensor(coefficients, dtype=torch.float64).reshape(-1)
    bound = float(radius)
    if values.numel() < 1 or not bool(torch.all(torch.isfinite(values))):
        raise ValueError("coefficients must be nonempty and finite")
    if not math.isfinite(bound) or bound <= 0.0:
        raise ValueError("radius must be positive and finite")
    norm = float(torch.linalg.vector_norm(values))
    return values if norm <= bound else values * (bound / norm)


@dataclass(frozen=True)
class ProjectionOracle:
    coefficients: Tensor
    unconstrained_coefficients: Tensor
    coefficient_norm: float
    clipped: bool
    residual_ratio: float
    evaluator_forward_calls: int
    evaluator_adjoint_calls: int


def measurement_projection_oracle(
    basis: GeometryKrylovBasis,
    target_residual: Tensor,
    *,
    coefficient_radius: float,
) -> ProjectionOracle:
    """Evaluator-only measurement-L2 projection of an exact residual target."""

    target = _finite_observation(target_residual, name="target_residual")
    if tuple(target.shape) != basis.observation_shape:
        raise ValueError("target_residual shape must match the basis")
    unconstrained = torch.einsum("r..., ...->r", basis.vectors, target)
    coefficients = project_to_l2_ball(unconstrained, radius=coefficient_radius)
    approximation = basis.synthesize(coefficients)
    denominator = max(float(torch.linalg.vector_norm(target)), 1e-30)
    ratio = float(torch.linalg.vector_norm(target - approximation)) / denominator
    return ProjectionOracle(
        coefficients=coefficients,
        unconstrained_coefficients=unconstrained,
        coefficient_norm=float(torch.linalg.vector_norm(coefficients)),
        clipped=not bool(torch.equal(coefficients, unconstrained)),
        residual_ratio=ratio,
        evaluator_forward_calls=0,
        evaluator_adjoint_calls=0,
    )


def adjoint_projection_oracle(
    basis: GeometryKrylovBasis,
    target_residual: Tensor,
    *,
    adjoint: LinearMap,
    support: Tensor,
    coefficient_radius: float,
    l2: float,
) -> ProjectionOracle:
    """Evaluator-only trust-region optimum in the ``P A^T`` induced norm."""

    target = _finite_observation(target_residual, name="target_residual")
    if tuple(target.shape) != basis.observation_shape:
        raise ValueError("target_residual shape must match the basis")
    penalty = float(l2)
    if not callable(adjoint):
        raise TypeError("adjoint must be callable")
    if not math.isfinite(penalty) or penalty < 0.0:
        raise ValueError("l2 must be finite and nonnegative")
    mask = torch.as_tensor(support, dtype=torch.float64)

    def effective(values: Tensor) -> Tensor:
        field = torch.as_tensor(adjoint(values), dtype=torch.float64)
        if field.shape != mask.shape:
            raise ValueError("support shape must match the adjoint field")
        return (field * mask).reshape(-1)

    target_adjoint = effective(target)
    columns = [effective(vector) for vector in basis.vectors]
    if any(column.shape != target_adjoint.shape for column in columns):
        raise ValueError("adjoint output shape drifted across basis vectors")
    design = torch.stack(columns, dim=1)
    identity = torch.eye(basis.rank, dtype=design.dtype)
    gram = design.T @ design + penalty * identity
    right = design.T @ target_adjoint
    unconstrained = torch.linalg.solve(gram, right)
    radius = float(coefficient_radius)
    if float(torch.linalg.vector_norm(unconstrained)) <= radius:
        coefficients = unconstrained
    else:
        lower = 0.0
        upper = 1.0
        while float(torch.linalg.vector_norm(torch.linalg.solve(gram + upper * identity, right))) > radius:
            upper *= 2.0
            if upper > 1e30:
                raise RuntimeError("failed to bracket the adjoint trust-region multiplier")
        for _ in range(100):
            middle = 0.5 * (lower + upper)
            trial = torch.linalg.solve(gram + middle * identity, right)
            if float(torch.linalg.vector_norm(trial)) > radius:
                lower = middle
            else:
                upper = middle
        coefficients = torch.linalg.solve(gram + upper * identity, right)
    approximation = design @ coefficients
    denominator = max(float(torch.linalg.vector_norm(target_adjoint)), 1e-30)
    ratio = float(torch.linalg.vector_norm(target_adjoint - approximation)) / denominator
    return ProjectionOracle(
        coefficients=coefficients,
        unconstrained_coefficients=unconstrained,
        coefficient_norm=float(torch.linalg.vector_norm(coefficients)),
        clipped=float(torch.linalg.vector_norm(unconstrained)) > radius,
        residual_ratio=ratio,
        evaluator_forward_calls=0,
        evaluator_adjoint_calls=basis.rank + 1,
    )


__all__ = [
    "GeometryKrylovBasis",
    "ProjectionOracle",
    "adjoint_projection_oracle",
    "build_geometry_krylov_basis",
    "measurement_projection_oracle",
    "project_to_l2_ball",
]
