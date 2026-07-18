from __future__ import annotations

import inspect

import pytest
import torch

from demo_t16_operator.jacru_n1_7_krylov_correction import (
    GeometryKrylovBasis,
    adjoint_projection_oracle,
    build_geometry_krylov_basis,
    measurement_projection_oracle,
    project_to_l2_ball,
)


class MatrixPair:
    def __init__(self, matrix: torch.Tensor) -> None:
        self.matrix = matrix
        self.forward_calls = 0
        self.adjoint_calls = 0
        self.forward_inputs: list[torch.Tensor] = []

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        self.forward_inputs.append(field.clone())
        return (self.matrix @ field.reshape(-1)).reshape(2, 2)

    def adjoint(self, observation: torch.Tensor) -> torch.Tensor:
        self.adjoint_calls += 1
        return self.matrix.T @ observation.reshape(-1)


def test_geometry_basis_is_deployable_orthonormal_and_exactly_two_pairs() -> None:
    matrix = torch.tensor(
        [
            [1.0, 0.2, 0.0],
            [0.1, 1.0, 0.3],
            [0.4, 0.0, 0.8],
            [0.0, 0.5, 1.0],
        ],
        dtype=torch.float64,
    )
    pair = MatrixPair(matrix)
    damping = torch.tensor([[1.0, 0.0], [0.2, -0.1]], dtype=torch.float64)
    warm_residual = torch.tensor([[0.0, 1.0], [-0.3, 0.4]], dtype=torch.float64)
    support = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64)
    basis = build_geometry_krylov_basis(
        damping=damping,
        warm_residual=warm_residual,
        forward=pair.forward,
        adjoint=pair.adjoint,
        support=support,
    )
    assert pair.forward_calls == pair.adjoint_calls == 2
    assert basis.setup_forward_calls == basis.setup_adjoint_calls == 2
    assert torch.equal(pair.forward_inputs[0], (matrix.T @ damping.reshape(-1)) * support)
    assert torch.equal(
        pair.forward_inputs[1], (matrix.T @ warm_residual.reshape(-1)) * support
    )
    assert basis.rank >= 2
    flattened = basis.vectors.reshape(basis.rank, -1)
    assert torch.allclose(
        flattened @ flattened.T,
        torch.eye(basis.rank, dtype=torch.float64),
        atol=1e-12,
    )
    assert basis.orthonormality_defect <= 1e-12


def test_dependent_normal_vectors_are_dropped_without_rank_fabrication() -> None:
    pair = MatrixPair(torch.eye(4, 3, dtype=torch.float64))
    damping = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    basis = build_geometry_krylov_basis(
        damping=damping,
        warm_residual=2.0 * damping,
        forward=pair.forward,
        adjoint=pair.adjoint,
        support=torch.ones(3, dtype=torch.float64),
    )
    assert basis.rank == 1
    assert set(basis.dropped_names) == {
        "warm_residual",
        "normal_damping",
        "normal_warm_residual",
    }


def test_measurement_projection_is_bounded_and_reduces_target_residual() -> None:
    pair = MatrixPair(
        torch.tensor(
            [[1.0, 0.0, 0.2], [0.0, 1.0, 0.1], [0.3, 0.0, 1.0], [0.0, 0.4, 1.0]],
            dtype=torch.float64,
        )
    )
    basis = build_geometry_krylov_basis(
        damping=torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64),
        warm_residual=torch.tensor([[0.0, 1.0], [0.5, -0.2]], dtype=torch.float64),
        forward=pair.forward,
        adjoint=pair.adjoint,
        support=torch.ones(3, dtype=torch.float64),
    )
    target = 0.3 * basis.vectors[0] - 0.2 * basis.vectors[1]
    oracle = measurement_projection_oracle(
        basis, target, coefficient_radius=0.25
    )
    assert oracle.coefficient_norm <= 0.25 + 1e-12
    assert oracle.clipped is True
    assert oracle.residual_ratio < 1.0
    assert oracle.evaluator_adjoint_calls == 0


def test_adjoint_projection_reports_truth_only_evaluator_calls() -> None:
    pair = MatrixPair(
        torch.tensor(
            [[1.0, 0.0, 0.2], [0.0, 1.0, 0.1], [0.3, 0.0, 1.0], [0.0, 0.4, 1.0]],
            dtype=torch.float64,
        )
    )
    basis = build_geometry_krylov_basis(
        damping=torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64),
        warm_residual=torch.tensor([[0.0, 1.0], [0.5, -0.2]], dtype=torch.float64),
        forward=pair.forward,
        adjoint=pair.adjoint,
        support=torch.ones(3, dtype=torch.float64),
    )
    pair.adjoint_calls = 0
    target = torch.tensor([[0.2, -0.1], [0.3, 0.0]], dtype=torch.float64)
    oracle = adjoint_projection_oracle(
        basis,
        target,
        adjoint=pair.adjoint,
        support=torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64),
        coefficient_radius=1.0,
        l2=1e-8,
    )
    assert pair.adjoint_calls == basis.rank + 1
    assert oracle.evaluator_adjoint_calls == basis.rank + 1
    assert 0.0 <= oracle.residual_ratio <= 1.0 + 1e-9


def test_l2_ball_projection_never_changes_an_interior_vector() -> None:
    interior = torch.tensor([0.3, 0.4], dtype=torch.float64)
    assert torch.equal(project_to_l2_ball(interior, radius=1.0), interior)
    exterior = project_to_l2_ball(torch.tensor([3.0, 4.0]), radius=2.0)
    assert torch.allclose(exterior, torch.tensor([1.2, 1.6], dtype=torch.float64))


def test_deployable_builder_signature_has_no_truth_target_input() -> None:
    names = set(inspect.signature(build_geometry_krylov_basis).parameters)
    assert names == {
        "damping",
        "warm_residual",
        "forward",
        "adjoint",
        "support",
        "dependence_tolerance",
    }


def test_zero_delta_exactly_recovers_the_damping_anchor() -> None:
    pair = MatrixPair(torch.eye(4, 3, dtype=torch.float64))
    damping = torch.tensor([[1.0, -0.2], [0.3, 0.0]], dtype=torch.float64)
    basis = build_geometry_krylov_basis(
        damping=damping,
        warm_residual=torch.tensor([[0.0, 0.4], [-0.1, 0.2]], dtype=torch.float64),
        forward=pair.forward,
        adjoint=pair.adjoint,
        support=torch.ones(3, dtype=torch.float64),
    )
    correction = damping + basis.synthesize(
        torch.zeros(basis.rank, dtype=torch.float64)
    )
    assert torch.equal(correction, damping)


def test_adjoint_ball_oracle_solves_the_active_elliptic_trust_region() -> None:
    basis = GeometryKrylovBasis(
        names=("x", "y"),
        vectors=torch.eye(2, dtype=torch.float64),
        raw_norms=(1.0, 1.0, 1.0, 1.0),
        dropped_names=(),
        orthonormality_defect=0.0,
        setup_forward_calls=2,
        setup_adjoint_calls=2,
    )
    matrix = torch.diag(torch.tensor([8.0, 1.0], dtype=torch.float64))

    def adjoint(values: torch.Tensor) -> torch.Tensor:
        return matrix @ values.reshape(-1)

    target = torch.tensor([1.0, 1.0], dtype=torch.float64)
    oracle = adjoint_projection_oracle(
        basis,
        target,
        adjoint=adjoint,
        support=torch.ones(2, dtype=torch.float64),
        coefficient_radius=0.25,
        l2=0.0,
    )
    assert oracle.clipped is True
    assert torch.linalg.vector_norm(oracle.coefficients) == pytest.approx(0.25)

    def objective(coefficients: torch.Tensor) -> float:
        residual = adjoint(target - coefficients)
        return float(torch.sum(residual.square()))

    angles = torch.linspace(0.0, 2.0 * torch.pi, 2001, dtype=torch.float64)
    circle = 0.25 * torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
    sampled_best = min(objective(point) for point in circle)
    assert objective(oracle.coefficients) <= sampled_best + 1e-5
