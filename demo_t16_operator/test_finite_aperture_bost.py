from __future__ import annotations

import numpy as np
import torch

try:
    from .finite_aperture_bost import (
        _disk_subrays,
        build_aperture_subray_operator_bank,
        build_finite_aperture_operator_bank,
        finite_aperture_reference_scale,
    )
    from .independent_reaction_bost import build_curved_cone_operator
    from .measurement_contract import BOSTBatch, DenseVolumeLinearBOST
except ImportError:
    from finite_aperture_bost import (
        _disk_subrays,
        build_aperture_subray_operator_bank,
        build_finite_aperture_operator_bank,
        finite_aperture_reference_scale,
    )
    from independent_reaction_bost import build_curved_cone_operator
    from measurement_contract import BOSTBatch, DenseVolumeLinearBOST


def _bank() -> np.ndarray:
    return build_finite_aperture_operator_bank(
        n=4,
        depth=3,
        angles_degrees=np.array([0.0, 47.0, 113.0]),
        radii=[0.0, 0.16],
        aperture_samples=13,
        path_samples=8,
    )


def test_bank_is_float32_deterministic_and_finite():
    first = _bank()
    second = _bank()
    assert first.shape == (2, 3, 3, 4, 48)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    assert not np.array_equal(first[0], first[1])


def test_zero_radius_reproduces_existing_single_ray_operator():
    angles = np.array([0.0, 47.0, 113.0])
    bank_zero = build_finite_aperture_operator_bank(
        4, 3, angles, [0.0], aperture_samples=13, path_samples=8
    )[0]
    single_ray = build_curved_cone_operator(
        4, 3, angles, path_samples=8
    )
    assert np.array_equal(bank_zero, single_ray)


def test_explicit_reference_scale_preserves_default_renderer():
    angles = np.array([0.0, 47.0, 113.0])
    kwargs = {"aperture_samples": 13, "path_samples": 8}
    scale = finite_aperture_reference_scale(4, 3, angles, **kwargs)
    default = build_finite_aperture_operator_bank(
        4, 3, angles, [0.0, 0.08, 0.16], **kwargs
    )
    explicit = build_finite_aperture_operator_bank(
        4,
        3,
        angles,
        [0.0, 0.08, 0.16],
        normalization_scale=scale,
        **kwargs,
    )
    assert np.array_equal(default, explicit)


def test_invalid_explicit_reference_scale_is_rejected():
    angles = np.array([0.0, 47.0, 113.0])
    with np.testing.assert_raises(ValueError):
        build_finite_aperture_operator_bank(
            4,
            3,
            angles,
            [0.0],
            aperture_samples=13,
            path_samples=8,
            normalization_scale=0.0,
        )


def test_dense_volume_linear_bost_adjoint_identity_for_finite_aperture():
    matrix = torch.from_numpy(_bank()[1])
    operator = DenseVolumeLinearBOST(matrix, (3, 4, 4))
    generator = torch.Generator().manual_seed(123)
    volume = torch.randn((2, 1, 3, 4, 4), generator=generator, dtype=torch.float32)
    observation = torch.zeros((2, 3, 3, 4), dtype=torch.float32)
    batch = BOSTBatch(
        observation=observation,
        view_mask=torch.ones((2, 3), dtype=torch.float32),
        noise_std=torch.ones((1, 1, 3, 1), dtype=torch.float32),
        view_angles_degrees=torch.tensor([[0.0, 47.0, 113.0]] * 2),
        support=torch.ones((1, 1, 3, 4, 4), dtype=torch.float32),
        geometry_ids=("finite-a", "finite-b"),
    )
    assert operator.adjoint_relative_error(batch, seed=29) < 1e-6
    residual = torch.randn(observation.shape, generator=generator)
    lhs = torch.sum(operator.forward(volume, batch) * residual)
    rhs = torch.sum(volume * operator.adjoint(residual, batch))
    torch.testing.assert_close(lhs, rhs, rtol=1e-6, atol=1e-6)


def test_prescribed_subray_bank_averages_to_historical_renderer():
    angles = np.array([0.0, 47.0, 113.0])
    disk = _disk_subrays(13)
    scale = finite_aperture_reference_scale(
        4, 3, angles, aperture_samples=13, path_samples=8
    )
    subrays = build_aperture_subray_operator_bank(
        4,
        3,
        angles,
        disk,
        aperture_radius=0.16,
        path_samples=8,
        normalization_scale=scale,
    )
    historical = build_finite_aperture_operator_bank(
        4,
        3,
        angles,
        [0.16],
        aperture_samples=13,
        path_samples=8,
        normalization_scale=scale,
    )[0]
    assert subrays.shape == (13, 3, 3, 4, 48)
    np.testing.assert_allclose(
        np.mean(subrays, axis=0), historical, rtol=2e-7, atol=2e-7
    )


def test_prescribed_subray_bank_rejects_points_outside_disk():
    with np.testing.assert_raises(ValueError):
        build_aperture_subray_operator_bank(
            4,
            3,
            np.array([0.0, 90.0]),
            np.array([[1.01, 0.0]]),
            aperture_radius=0.1,
            path_samples=8,
        )
