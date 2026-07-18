from __future__ import annotations

import numpy as np

from demo_t16_operator.aperture_control_variate import (
    centered_pupil_basis,
    concentric_square_to_disk,
    cross_fitted_control_variate,
    disk_product_quadrature,
    sample_uniform_disk_antithetic,
    sample_uniform_disk_iid,
    weighted_operator_mean,
)


def test_concentric_mapping_stays_in_disk_and_is_deterministic() -> None:
    square = np.array([[0.5, 0.5], [0.0, 0.0], [1.0, 1.0], [0.2, 0.8]])
    first = concentric_square_to_disk(square)
    second = concentric_square_to_disk(square)
    assert np.array_equal(first, second)
    assert np.max(np.sum(first * first, axis=1)) <= 1.0 + 1e-12
    np.testing.assert_array_equal(first[0], np.zeros(2))


def test_iid_and_antithetic_samplers_obey_contract() -> None:
    iid = sample_uniform_disk_iid(20, seed=9)
    paired = sample_uniform_disk_antithetic(20, seed=9)
    assert iid.shape == paired.shape == (20, 2)
    np.testing.assert_allclose(paired[0::2], -paired[1::2], rtol=0.0, atol=0.0)
    assert not np.array_equal(iid, paired)


def test_centered_basis_has_correct_quadrature_mean() -> None:
    points, weights = disk_product_quadrature(10, 40)
    for basis in ("affine", "quadratic"):
        design = centered_pupil_basis(points, basis=basis)
        mean = weights @ design
        np.testing.assert_allclose(mean[0], 1.0, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(mean[1:], 0.0, rtol=0.0, atol=1e-14)


def test_cross_fitted_estimator_exactly_removes_affine_variation() -> None:
    points = sample_uniform_disk_iid(64, seed=31)
    values = (
        2.75
        + 1.2 * points[:, 0]
        - 0.8 * points[:, 1]
    )[:, None]
    result = cross_fitted_control_variate(
        points, values, basis="affine", ridge=0.0
    )
    np.testing.assert_allclose(result.estimate, np.array([2.75]), rtol=1e-12, atol=1e-12)
    assert abs(float(result.plain_mean[0]) - 2.75) > 1e-3


def test_cross_fitted_quadratic_estimator_is_exact_for_known_basis() -> None:
    points = sample_uniform_disk_iid(64, seed=71)
    design = centered_pupil_basis(points, basis="quadratic")
    coefficients = np.array([1.4, -0.7, 0.2, 0.5, -0.3, 0.8])
    values = (design @ coefficients)[:, None, None]
    result = cross_fitted_control_variate(
        points, values, basis="quadratic", ridge=0.0
    )
    np.testing.assert_allclose(
        result.estimate, np.array([[coefficients[0]]]), rtol=1e-11, atol=1e-11
    )


def test_weighted_operator_mean_preserves_trailing_shape() -> None:
    points, weights = disk_product_quadrature(4, 8)
    values = np.stack(
        [np.eye(3) * (1.0 + point[0]) for point in points], axis=0
    )
    mean = weighted_operator_mean(values, weights)
    np.testing.assert_allclose(mean, np.eye(3), rtol=0.0, atol=1e-14)
