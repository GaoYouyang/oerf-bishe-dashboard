from __future__ import annotations

import warnings

import numpy as np
import pytest

from site_tools.psu_bost_forward_geometry import (
    CONTRACT_VERSION,
    FORWARD_GEOMETRY_CONTRACT,
    b0_forward_ray_box_intersection,
    b1_forward_ray_box_cone_intersection,
    intersect_forward_ray_box,
    intersect_forward_ray_box_cone,
)


UNIT_BOX_MINIMUM = np.array([-1.0, -1.0, -1.0])
UNIT_BOX_MAXIMUM = np.array([1.0, 1.0, 1.0])
WIDE_BOX_MINIMUM = np.array([-10.0, -10.0, -10.0])
WIDE_BOX_MAXIMUM = np.array([10.0, 10.0, 10.0])
CONE_VERTEX = np.zeros(3)
CONE_AXIS_X = np.array([1.0, 0.0, 0.0])
CONE_THETA = np.pi / 4.0


def _assert_interval(
    result: dict,
    enter: list[float],
    exit_: list[float],
    length: list[float],
    hit: list[bool],
    *,
    atol: float = 1e-12,
) -> None:
    np.testing.assert_allclose(result["enter"], enter, atol=atol, rtol=0.0)
    np.testing.assert_allclose(result["exit"], exit_, atol=atol, rtol=0.0)
    np.testing.assert_allclose(result["length"], length, atol=atol, rtol=0.0)
    assert result["hit"].tolist() == hit
    assert result["enter"].dtype == np.float64
    assert result["exit"].dtype == np.float64
    assert result["length"].dtype == np.float64
    assert result["hit"].dtype == np.bool_


def test_contract_is_explicit_metric_forward_and_source_independent() -> None:
    assert FORWARD_GEOMETRY_CONTRACT["version"] == CONTRACT_VERSION
    assert "normalized_direction" in FORWARD_GEOMETRY_CONTRACT["ray"]
    assert "strictly positive-measure" in FORWARD_GEOMETRY_CONTRACT["interval"]
    assert "does not import" in FORWARD_GEOMETRY_CONTRACT["provenance"]
    assert b0_forward_ray_box_intersection is intersect_forward_ray_box
    assert b1_forward_ray_box_cone_intersection is intersect_forward_ray_box_cone


def test_b0_exact_secant_inside_and_behind_intervals() -> None:
    secant = intersect_forward_ray_box(
        [-2.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        UNIT_BOX_MINIMUM,
        UNIT_BOX_MAXIMUM,
    )
    _assert_interval(secant, [1.0], [3.0], [2.0], [True])
    assert secant["direction_norm"].tolist() == [4.0]
    np.testing.assert_allclose(secant["direction_unit"], [[1.0, 0.0, 0.0]])

    inside = intersect_forward_ray_box(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        UNIT_BOX_MINIMUM,
        UNIT_BOX_MAXIMUM,
    )
    _assert_interval(inside, [0.0], [1.0], [1.0], [True])
    assert inside["origin_inside"].tolist() == [True]
    assert inside["forward_clipped"].tolist() == [True]

    behind = intersect_forward_ray_box(
        [2.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        UNIT_BOX_MINIMUM,
        UNIT_BOX_MAXIMUM,
    )
    _assert_interval(behind, [0.0], [0.0], [0.0], [False])
    assert behind["behind_origin"].tolist() == [True]


def test_b0_parallel_outside_and_on_face_are_warning_free() -> None:
    origins = np.array(
        [
            [-2.0, 2.0, 0.0],
            [-2.0, 1.0, 0.0],
        ]
    )
    directions = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = intersect_forward_ray_box(
            origins,
            directions,
            UNIT_BOX_MINIMUM,
            UNIT_BOX_MAXIMUM,
        )

    _assert_interval(result, [0.0, 1.0], [0.0, 3.0], [0.0, 2.0], [False, True])
    assert result["parallel_outside"].tolist() == [True, False]
    assert result["parallel_outside_axis"][:, 1].tolist() == [True, False]
    assert result["parallel_on_boundary_axis"][:, 1].tolist() == [False, True]


def test_b0_rejects_zero_directions_and_ambiguous_layout() -> None:
    with pytest.raises(ValueError, match="zero direction"):
        intersect_forward_ray_box(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            UNIT_BOX_MINIMUM,
            UNIT_BOX_MAXIMUM,
        )

    ambiguous_origins = np.zeros((3, 3))
    ambiguous_directions = np.eye(3)
    with pytest.raises(ValueError, match="ambiguous shape"):
        intersect_forward_ray_box(
            ambiguous_origins,
            ambiguous_directions,
            UNIT_BOX_MINIMUM,
            UNIT_BOX_MAXIMUM,
        )

    explicit = intersect_forward_ray_box(
        ambiguous_origins,
        ambiguous_directions,
        UNIT_BOX_MINIMUM,
        UNIT_BOX_MAXIMUM,
        layout="rows",
    )
    assert explicit["hit"].tolist() == [True, True, True]


def test_b0_accepts_unambiguous_column_layout() -> None:
    origins = np.array(
        [
            [-2.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    directions = np.array(
        [
            [2.0, 0.0],
            [0.0, -3.0],
            [0.0, 0.0],
        ]
    )
    result = intersect_forward_ray_box(
        origins,
        directions,
        UNIT_BOX_MINIMUM,
        UNIT_BOX_MAXIMUM,
    )
    _assert_interval(result, [1.0, 0.0], [3.0, 1.0], [2.0, 1.0], [True, True])


def test_b1_exact_transverse_secant_and_metric_normalization() -> None:
    result = intersect_forward_ray_box_cone(
        [2.0, 3.0, 0.0],
        [0.0, -4.0, 0.0],
        WIDE_BOX_MINIMUM,
        WIDE_BOX_MAXIMUM,
        CONE_VERTEX,
        CONE_AXIS_X,
        CONE_THETA,
    )
    _assert_interval(result, [1.0], [5.0], [4.0], [True])
    np.testing.assert_allclose(result["cone_enter"], [1.0])
    np.testing.assert_allclose(result["cone_exit"], [5.0])
    np.testing.assert_allclose(result["cone_length"], [4.0])
    assert result["cone_hit"].tolist() == [True]
    assert result["origin_in_cone"].tolist() == [False]


@pytest.mark.parametrize("direction", ([0.0, 1.0, 0.0], [0.0, -1.0, 0.0]))
def test_b1_origin_inside_cone_has_forward_exit_length_two(direction: list[float]) -> None:
    result = intersect_forward_ray_box_cone(
        [2.0, 0.0, 0.0],
        direction,
        WIDE_BOX_MINIMUM,
        WIDE_BOX_MAXIMUM,
        CONE_VERTEX,
        CONE_AXIS_X,
        CONE_THETA,
    )
    _assert_interval(result, [0.0], [2.0], [2.0], [True])
    assert result["origin_in_cone"].tolist() == [True]


def test_b1_rejects_negative_nappe_and_axis_flip_selects_it() -> None:
    positive_axis = intersect_forward_ray_box_cone(
        [-2.0, 3.0, 0.0],
        [0.0, -1.0, 0.0],
        WIDE_BOX_MINIMUM,
        WIDE_BOX_MAXIMUM,
        CONE_VERTEX,
        CONE_AXIS_X,
        CONE_THETA,
    )
    _assert_interval(positive_axis, [0.0], [0.0], [0.0], [False])
    assert positive_axis["double_cone_forward_hit"].tolist() == [True]
    assert positive_axis["nappe_rejected"].tolist() == [True]

    flipped_axis = intersect_forward_ray_box_cone(
        [-2.0, 3.0, 0.0],
        [0.0, -1.0, 0.0],
        WIDE_BOX_MINIMUM,
        WIDE_BOX_MAXIMUM,
        CONE_VERTEX,
        -CONE_AXIS_X,
        CONE_THETA,
    )
    _assert_interval(flipped_axis, [1.0], [5.0], [4.0], [True])
    np.testing.assert_allclose(flipped_axis["cone_axis_unit"], [-1.0, 0.0, 0.0])


def test_b1_apex_and_tangent_are_zero_measure_contacts() -> None:
    origins = np.array(
        [
            [0.0, 1.0, 0.0],
            [2.0, 3.0, 2.0],
        ]
    )
    directions = np.array(
        [
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    result = intersect_forward_ray_box_cone(
        origins,
        directions,
        WIDE_BOX_MINIMUM,
        WIDE_BOX_MAXIMUM,
        CONE_VERTEX,
        CONE_AXIS_X,
        CONE_THETA,
    )
    _assert_interval(result, [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [False, False])
    assert result["cone_point_touch"].tolist() == [True, True]
    assert result["discriminant_tangent"].tolist() == [True, True]
    assert result["apex_on_forward_ray"].tolist() == [True, False]
    assert result["apex_in_box_interval"].tolist() == [True, False]


def test_b1_near_linear_generator_and_small_a_signs() -> None:
    generator = np.array([np.cos(CONE_THETA), np.sin(CONE_THETA), 0.0])
    linear = intersect_forward_ray_box_cone(
        [1.0, 0.0, 0.0],
        generator,
        [0.0, -10.0, -1.0],
        [3.0, 10.0, 1.0],
        CONE_VERTEX,
        CONE_AXIS_X,
        CONE_THETA,
    )
    expected_exit = 2.0 / np.cos(CONE_THETA)
    _assert_interval(
        linear,
        [0.0],
        [expected_exit],
        [expected_exit],
        [True],
        atol=1e-11,
    )
    assert linear["quadratic_degenerate"].tolist() == [True]
    assert linear["used_linear_inequality"].tolist() == [True]

    delta = 1e-9
    directions = np.array(
        [
            [np.cos(CONE_THETA - delta), np.sin(CONE_THETA - delta), 0.0],
            [np.cos(CONE_THETA + delta), np.sin(CONE_THETA + delta), 0.0],
        ]
    )
    signs = intersect_forward_ray_box_cone(
        np.zeros((2, 3)),
        directions,
        [0.0, -10.0, -1.0],
        [3.0, 10.0, 1.0],
        CONE_VERTEX,
        CONE_AXIS_X,
        CONE_THETA,
    )
    assert signs["coefficient_a"][0] < 0.0
    assert signs["coefficient_a"][1] > 0.0
    assert signs["quadratic_degenerate"].tolist() == [False, False]
    assert signs["hit"].tolist() == [True, False]
    assert signs["cone_point_touch"].tolist() == [False, True]


def test_b1_slightly_positive_and_negative_discriminants() -> None:
    offset = 1e-8
    origins = np.array(
        [
            [2.0, 3.0, 2.0 - offset],
            [2.0, 3.0, 2.0 + offset],
        ]
    )
    directions = np.array(
        [
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    result = intersect_forward_ray_box_cone(
        origins,
        directions,
        WIDE_BOX_MINIMUM,
        WIDE_BOX_MAXIMUM,
        CONE_VERTEX,
        CONE_AXIS_X,
        CONE_THETA,
    )
    radius = np.sqrt(4.0 - (2.0 - offset) ** 2)
    _assert_interval(
        result,
        [3.0 - radius, 0.0],
        [3.0 + radius, 0.0],
        [2.0 * radius, 0.0],
        [True, False],
        atol=1e-10,
    )
    assert result["discriminant_positive"].tolist() == [True, False]
    assert result["discriminant_negative"].tolist() == [False, True]


def test_b1_reports_valid_box_and_cone_intervals_with_no_overlap() -> None:
    result = intersect_forward_ray_box_cone(
        [2.0, 3.0, 0.0],
        [0.0, -1.0, 0.0],
        [1.0, -5.0, -1.0],
        [3.0, -4.0, 1.0],
        CONE_VERTEX,
        CONE_AXIS_X,
        CONE_THETA,
    )
    _assert_interval(result, [0.0], [0.0], [0.0], [False])
    np.testing.assert_allclose(result["box_enter"], [7.0])
    np.testing.assert_allclose(result["box_exit"], [8.0])
    np.testing.assert_allclose(result["cone_enter"], [1.0])
    np.testing.assert_allclose(result["cone_exit"], [5.0])
    assert result["box_hit"].tolist() == [True]
    assert result["cone_hit"].tolist() == [True]
    assert result["box_cone_disjoint"].tolist() == [True]
    assert result["cone_no_box_overlap"].tolist() == [True]


def test_b1_rejects_invalid_cone_and_zero_ray_inputs() -> None:
    with pytest.raises(ValueError, match="cone_axis"):
        intersect_forward_ray_box_cone(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            UNIT_BOX_MINIMUM,
            UNIT_BOX_MAXIMUM,
            CONE_VERTEX,
            [0.0, 0.0, 0.0],
            CONE_THETA,
        )

    with pytest.raises(ValueError, match="theta"):
        intersect_forward_ray_box_cone(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            UNIT_BOX_MINIMUM,
            UNIT_BOX_MAXIMUM,
            CONE_VERTEX,
            CONE_AXIS_X,
            np.pi / 2.0,
        )

    with pytest.raises(ValueError, match="zero direction"):
        intersect_forward_ray_box_cone(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            UNIT_BOX_MINIMUM,
            UNIT_BOX_MAXIMUM,
            CONE_VERTEX,
            CONE_AXIS_X,
            CONE_THETA,
        )
