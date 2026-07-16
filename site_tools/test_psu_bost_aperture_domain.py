from __future__ import annotations

import numpy as np
import pytest

from site_tools.psu_bost_aperture_domain import (
    APERTURE_DOMAIN_CONTRACT,
    CONTRACT_VERSION,
    constant_gradient_los_smoke,
    deterministic_aperture_quadrature,
    deterministic_paired_uniform_aperture_samples,
    evaluate_aperture_domain,
    generate_aperture_sample_points,
)


def test_paired_uniform_design_is_deterministic_and_interior() -> None:
    first = deterministic_paired_uniform_aperture_samples(16)
    second = deterministic_paired_uniform_aperture_samples(16)
    np.testing.assert_array_equal(
        first["longitudinal_fractions"], second["longitudinal_fractions"]
    )
    np.testing.assert_array_equal(
        first["unit_disk_offsets"], second["unit_disk_offsets"]
    )
    assert first["sample_count"] == 16
    assert np.all(first["longitudinal_fractions"] > 0.0)
    assert np.all(first["longitudinal_fractions"] < 1.0)
    radius = np.linalg.norm(first["unit_disk_offsets"], axis=1)
    assert np.all(radius > 0.0)
    assert np.all(radius < 1.0)


@pytest.mark.parametrize("value", [True, 1, 2.5])
def test_paired_uniform_design_rejects_invalid_count(value: object) -> None:
    with pytest.raises(ValueError, match="sample_count"):
        deterministic_paired_uniform_aperture_samples(value)  # type: ignore[arg-type]


RX = np.array([[0.0, 1.0, 0.0]])
RY = np.array([[0.0, 0.0, 1.0]])


def test_all_in_box_matches_released_center_radius_offset_formula() -> None:
    fractions = np.array([0.0, 0.5, 1.0])
    offsets = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, -1.0]])
    points = generate_aperture_sample_points(
        [[-0.5, 0.0, 0.0]],
        [[0.5, 0.0, 0.0]],
        RX,
        RY,
        [0.1],
        [0.3],
        fractions,
        offsets,
    )

    expected = np.array([[[-0.5, 0.0, 0.0], [0.0, 0.2, 0.0], [0.5, 0.0, -0.3]]])
    np.testing.assert_allclose(points, expected, atol=1e-15, rtol=0.0)
    assert points.dtype == np.float64

    membership = evaluate_aperture_domain(points, [-1, -1, -1], [1, 1, 1])
    assert membership["domain"] == "B0"
    assert membership["indicator"].tolist() == [[True, True, True]]
    assert membership["all_in_domain"].tolist() == [True]
    assert membership["any_out_of_domain"].tolist() == [False]
    assert membership["empty_domain"].tolist() == [False]
    np.testing.assert_allclose(membership["in_domain_fraction"], [1.0])


def test_aperture_crossing_box_face_reports_sample_level_fraction() -> None:
    points = generate_aperture_sample_points(
        [[0.75, 0.0, 0.0]],
        [[0.75, 0.0, 0.0]],
        [[1.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0]],
        [0.5],
        [0.5],
        [0.5, 0.5, 0.5],
        [[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]],
    )
    membership = evaluate_aperture_domain(points, [-1, -1, -1], [1, 1, 1])

    assert membership["indicator"].tolist() == [[True, False, True]]
    assert membership["all_in_domain"].tolist() == [False]
    assert membership["all_in"].tolist() == [False]
    assert membership["any_out_of_domain"].tolist() == [True]
    assert membership["any_out"].tolist() == [True]
    assert membership["empty_domain"].tolist() == [False]
    np.testing.assert_allclose(membership["in_domain_fraction"], [2.0 / 3.0])


def test_b1_rejects_negative_nappe_even_inside_box() -> None:
    points = np.array(
        [
            [
                [-2.0, 0.0, 0.0],
                [-2.0, 0.5, 0.0],
                [2.0, 0.5, 0.0],
            ]
        ]
    )
    membership = evaluate_aperture_domain(
        points,
        [-3, -3, -3],
        [3, 3, 3],
        cone_vertex=[0, 0, 0],
        cone_axis=[4, 0, 0],
        cone_theta=np.pi / 4,
    )

    assert membership["domain"] == "B1"
    assert membership["box_indicator"].tolist() == [[True, True, True]]
    assert membership["cone_indicator"].tolist() == [[False, False, True]]
    assert membership["indicator"].tolist() == [[False, False, True]]
    np.testing.assert_allclose(membership["cone_axis_unit"], [1.0, 0.0, 0.0])


def test_zero_radius_is_exact_thin_centerline_for_every_disk_offset() -> None:
    fractions = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    offsets = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    points = generate_aperture_sample_points(
        [[0.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0]],
        RX,
        RY,
        [0.0],
        [0.0],
        fractions,
        offsets,
    )

    expected = np.column_stack((fractions, np.zeros((fractions.size, 2))))
    np.testing.assert_array_equal(points[0], expected)


def test_basis_validation_rejects_nonunit_and_nonorthogonal_vectors() -> None:
    common = (
        [[0.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0]],
    )
    with pytest.raises(ValueError, match="unit vectors"):
        generate_aperture_sample_points(
            *common,
            [[0.0, 2.0, 0.0]],
            RY,
            [0.1],
            [0.1],
            [0.5],
            [[0.0, 0.0]],
        )

    with pytest.raises(ValueError, match="mutually orthogonal"):
        generate_aperture_sample_points(
            *common,
            RX,
            RX,
            [0.1],
            [0.1],
            [0.5],
            [[0.0, 0.0]],
        )


def test_negative_radius_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative radii"):
        generate_aperture_sample_points(
            [[0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            RX,
            RY,
            [-0.1],
            [0.1],
            [0.5],
            [[0.0, 0.0]],
        )


def test_constant_gradient_los_uses_original_sample_count() -> None:
    smoke = constant_gradient_los_smoke(
        gradient=[2.0, 0.0, 0.0],
        sensitivity=[1.0, 0.0, 0.0],
        path_length=4.0,
        indicator=[[True, False, True, False]],
        system_constant=3.0,
    )

    # 3 * 4 / 4 * (2 + 0 + 2 + 0) = 12. Surviving-count
    # renormalization would incorrectly produce 24.
    np.testing.assert_allclose(smoke["prediction"], [12.0])
    assert smoke["original_sample_count"] == 4
    assert smoke["surviving_sample_count"].tolist() == [2]
    np.testing.assert_allclose(smoke["in_domain_fraction"], [0.5])

    empty = constant_gradient_los_smoke(
        gradient=[2.0, 0.0, 0.0],
        sensitivity=[1.0, 0.0, 0.0],
        path_length=4.0,
        indicator=[[False, False, False, False]],
    )
    np.testing.assert_array_equal(empty["prediction"], [0.0])


def test_deterministic_quadrature_is_reproducible_symmetric_and_has_boundaries() -> (
    None
):
    first = deterministic_aperture_quadrature(
        longitudinal_count=3,
        ring_radii=(0.5,),
        angles_per_ring=8,
    )
    second = deterministic_aperture_quadrature(
        longitudinal_count=3,
        ring_radii=(0.5,),
        angles_per_ring=8,
    )

    np.testing.assert_array_equal(
        first["longitudinal_fractions"],
        second["longitudinal_fractions"],
    )
    np.testing.assert_array_equal(
        first["unit_disk_offsets"],
        second["unit_disk_offsets"],
    )
    assert first["sample_count"] == 3 * (1 + 2 * 8)
    assert {0.0, 1.0}.issubset(set(first["longitudinal_fractions"]))

    disk = first["disk_offsets"]
    assert np.any(np.all(disk == [0.0, 0.0], axis=1))
    assert np.any(np.all(disk == [1.0, 0.0], axis=1))
    assert np.any(np.all(disk == [-1.0, 0.0], axis=1))
    for offset in disk:
        assert np.any(np.all(np.isclose(disk, -offset, atol=1e-15), axis=1))

    assert APERTURE_DOMAIN_CONTRACT["version"] == CONTRACT_VERSION
    assert "original fixed sample count" in APERTURE_DOMAIN_CONTRACT["normalization"]
