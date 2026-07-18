from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from demo_t16_operator.jacru_n1_8_camera_ray_hybrid import (
    SUPPORTED_BASIS_KINDS,
    build_camera_ray_hybrid_basis,
    visible_total_correction_radius,
)
from demo_t16_operator.jacru_synthetic_fixture import (
    JACRUSyntheticFixtureConfig,
    build_jacru_geometry,
)


def _geometry():
    return build_jacru_geometry(
        split="development",
        base_seed=991,
        config=JACRUSyntheticFixtureConfig(detector_shape=(3, 3)),
    )


def _maps(observation_shape: tuple[int, ...], field_shape: tuple[int, ...]):
    calls = {"forward": 0, "adjoint": 0}
    observation_size = int(torch.tensor(observation_shape).prod())
    field_size = int(torch.tensor(field_shape).prod())
    matrix = torch.arange(
        1, observation_size * field_size + 1, dtype=torch.float64
    ).reshape(observation_size, field_size)
    matrix = matrix / torch.linalg.vector_norm(matrix)

    def forward(field: torch.Tensor) -> torch.Tensor:
        calls["forward"] += 1
        return (matrix @ field.reshape(-1)).reshape(observation_shape)

    def adjoint(observation: torch.Tensor) -> torch.Tensor:
        calls["adjoint"] += 1
        return (matrix.T @ observation.reshape(-1)).reshape(field_shape)

    return forward, adjoint, calls


@pytest.mark.parametrize(
    ("kind", "expected_raw_rank", "expected_calls"),
    (
        ("krylov4_total", 4, 2),
        ("fit_pca2_krylov6_total", 6, 2),
        ("camera_block6_total", 6, 0),
        ("pose_fourier_krylov6_total", 6, 2),
        ("detector_moment_krylov6_total", 6, 2),
    ),
)
def test_hybrid_basis_respects_frozen_operator_ledger(
    kind: str, expected_raw_rank: int, expected_calls: int
) -> None:
    assert SUPPORTED_BASIS_KINDS == (
        "krylov4_total",
        "fit_pca2_krylov6_total",
        "camera_block6_total",
        "pose_fourier_krylov6_total",
        "detector_moment_krylov6_total",
    )
    geometry = _geometry()
    observation_shape = (geometry.ray_count, 2)
    field_shape = (3, 3, 3)
    forward, adjoint, calls = _maps(observation_shape, field_shape)
    generator = torch.Generator().manual_seed(7)
    damping = torch.randn(observation_shape, generator=generator, dtype=torch.float64)
    residual = torch.randn(observation_shape, generator=generator, dtype=torch.float64)
    fit_modes = torch.randn((2, damping.numel()), generator=generator, dtype=torch.float64)
    basis = build_camera_ray_hybrid_basis(
        kind=kind,
        damping=damping,
        warm_residual=residual,
        forward=forward,
        adjoint=adjoint,
        support=torch.ones(field_shape, dtype=torch.float64),
        geometry=geometry,
        fit_modes=fit_modes if kind == "fit_pca2_krylov6_total" else None,
    )
    assert 2 <= basis.rank <= expected_raw_rank
    assert calls == {"forward": expected_calls, "adjoint": expected_calls}
    assert (basis.setup_forward_calls, basis.setup_adjoint_calls) == (
        expected_calls,
        expected_calls,
    )
    assert basis.uses_evaluated_case_truth is False
    assert basis.observation_shape == observation_shape
    assert basis.orthonormality_defect < 1e-12
    assert basis.fit_mode_count == (2 if kind == "fit_pca2_krylov6_total" else 0)


def test_total_radius_contains_damping_and_caps_at_twice_its_norm() -> None:
    damping = torch.tensor([3.0, 4.0], dtype=torch.float64)
    tiny = torch.tensor([1e-8, 0.0], dtype=torch.float64)
    medium = torch.tensor([0.4, 0.0], dtype=torch.float64)
    large = torch.tensor([10.0, 0.0], dtype=torch.float64)
    assert visible_total_correction_radius(damping, tiny) == pytest.approx(5.0)
    assert visible_total_correction_radius(damping, medium) == pytest.approx(6.4)
    assert visible_total_correction_radius(damping, large) == pytest.approx(10.0)


def test_total_correction_norm_equals_coefficient_norm() -> None:
    geometry = _geometry()
    observation_shape = (geometry.ray_count, 2)
    forward, adjoint, _ = _maps(observation_shape, (3, 3, 3))
    generator = torch.Generator().manual_seed(19)
    basis = build_camera_ray_hybrid_basis(
        kind="pose_fourier_krylov6_total",
        damping=torch.randn(observation_shape, generator=generator, dtype=torch.float64),
        warm_residual=torch.randn(observation_shape, generator=generator, dtype=torch.float64),
        forward=forward,
        adjoint=adjoint,
        support=torch.ones((3, 3, 3), dtype=torch.float64),
        geometry=geometry,
    )
    coefficients = torch.linspace(0.1, 0.1 * basis.rank, basis.rank, dtype=torch.float64)
    assert float(torch.linalg.vector_norm(basis.synthesize(coefficients))) == pytest.approx(
        float(torch.linalg.vector_norm(coefficients)), rel=1e-12, abs=1e-12
    )


def test_detector_moment_fails_closed_on_reordered_rays() -> None:
    geometry = _geometry()
    reordered = replace(geometry, camera_index=torch.roll(geometry.camera_index, 1))
    observation_shape = (geometry.ray_count, 2)
    forward, adjoint, calls = _maps(observation_shape, (3, 3, 3))
    with pytest.raises(ValueError, match="authoritative camera-major"):
        build_camera_ray_hybrid_basis(
            kind="detector_moment_krylov6_total",
            damping=torch.ones(observation_shape, dtype=torch.float64),
            warm_residual=torch.ones(observation_shape, dtype=torch.float64),
            forward=forward,
            adjoint=adjoint,
            support=torch.ones((3, 3, 3), dtype=torch.float64),
            geometry=reordered,
        )
    assert calls == {"forward": 0, "adjoint": 0}


def test_fit_modes_are_rejected_outside_fit_pca_candidate() -> None:
    geometry = _geometry()
    observation_shape = (geometry.ray_count, 2)
    forward, adjoint, _ = _maps(observation_shape, (3, 3, 3))
    with pytest.raises(ValueError, match="only allowed"):
        build_camera_ray_hybrid_basis(
            kind="krylov4_total",
            damping=torch.ones(observation_shape, dtype=torch.float64),
            warm_residual=torch.ones(observation_shape, dtype=torch.float64),
            forward=forward,
            adjoint=adjoint,
            support=torch.ones((3, 3, 3), dtype=torch.float64),
            geometry=geometry,
            fit_modes=torch.ones((2, int(torch.tensor(observation_shape).prod()))),
        )
