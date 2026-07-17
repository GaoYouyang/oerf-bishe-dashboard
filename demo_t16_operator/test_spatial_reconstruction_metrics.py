from __future__ import annotations

import numpy as np
import pytest

from demo_t16_operator.spatial_reconstruction_metrics import (
    interface_surface_from_level_set,
    level_set_surface_metrics,
    normal_angle_metrics,
    scalar_grid_gradient,
    surface_distance_metrics,
    synthetic_field_metrics,
)


def test_scalar_grid_gradient_uses_xyz_component_order() -> None:
    x = np.linspace(-1.0, 1.0, 7)
    y = np.linspace(-2.0, 2.0, 8)
    z = np.linspace(-3.0, 3.0, 9)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    field = 2.0 * xx - 3.0 * yy + 4.0 * zz
    gradient = scalar_grid_gradient(
        field,
        spacing_xyz=(x[1] - x[0], y[1] - y[0], z[1] - z[0]),
    )
    assert np.allclose(gradient, np.array([2.0, -3.0, 4.0]))


def test_exact_linear_field_has_zero_field_and_h1_error() -> None:
    axis = np.linspace(-1.0, 1.0, 9)
    zz, yy, xx = np.meshgrid(axis, axis, axis, indexing="ij")
    truth = 0.5 * xx - 0.25 * yy + 0.75 * zz
    truth_gradient = np.broadcast_to([0.5, -0.25, 0.75], (*truth.shape, 3))
    metrics = synthetic_field_metrics(
        truth,
        truth,
        analytic_truth_gradient_xyz=truth_gradient,
        spacing_xyz=(axis[1] - axis[0],) * 3,
    )
    assert metrics["field_relative_l2"] == 0.0
    assert metrics["field_rmse"] == 0.0
    assert metrics["h1_seminorm_relative_error"] < 1e-14


def test_interface_surface_ignores_domain_wall_without_sign_change() -> None:
    level = np.ones((8, 8, 8), dtype=np.float64)
    assert np.count_nonzero(interface_surface_from_level_set(level)) == 0
    x = np.linspace(-1.0, 1.0, 8)
    plane = np.broadcast_to(x, (8, 8, 8))
    surface = interface_surface_from_level_set(plane)
    assert np.count_nonzero(surface) == 2 * 8 * 8


def test_identical_surface_has_zero_distance_and_unit_f1() -> None:
    surface = np.zeros((9, 9, 9), dtype=bool)
    surface[:, :, 4] = True
    metrics = surface_distance_metrics(
        surface,
        surface,
        spacing_xyz=(0.25, 0.5, 0.75),
        tolerance_distances=(0.0, 0.25),
    )
    assert metrics["surface_assd"] == 0.0
    assert metrics["surface_hd95"] == 0.0
    assert metrics["surface_f1_at_0"] == 1.0
    assert metrics["surface_f1_at_0p25"] == 1.0


def test_shifted_planes_report_physical_distance_and_tolerance_f1() -> None:
    truth = np.zeros((9, 9, 9), dtype=bool)
    predicted = np.zeros_like(truth)
    truth[:, :, 3] = True
    predicted[:, :, 5] = True
    metrics = surface_distance_metrics(
        predicted,
        truth,
        spacing_xyz=(0.2, 0.3, 0.4),
        tolerance_distances=(0.39, 0.4, 0.41),
    )
    assert metrics["surface_assd"] == pytest.approx(0.4)
    assert metrics["surface_hd95"] == pytest.approx(0.4)
    assert metrics["surface_f1_at_0p39"] == 0.0
    assert metrics["surface_f1_at_0p4"] == 1.0
    assert metrics["surface_f1_at_0p41"] == 1.0


def test_level_set_wrapper_matches_identical_interface() -> None:
    x = np.linspace(-1.0, 1.0, 9)
    level = np.broadcast_to(x, (9, 9, 9))
    metrics = level_set_surface_metrics(
        level,
        level,
        spacing_xyz=(0.25, 0.25, 0.25),
        tolerance_distances=(0.25, 0.5),
    )
    assert metrics["surface_assd"] == 0.0
    assert metrics["surface_f1_at_0p25"] == 1.0


def test_normal_angles_report_signed_and_unoriented_cases() -> None:
    truth = np.zeros((3, 3, 3, 3), dtype=np.float64)
    predicted = np.zeros_like(truth)
    truth[..., 0] = 1.0
    predicted[..., 0] = -1.0
    metrics = normal_angle_metrics(
        predicted,
        truth,
        evaluation_mask=np.ones((3, 3, 3), dtype=bool),
    )
    assert metrics["normal_angle_median_degrees"] == 180.0
    assert metrics["normal_angle_unoriented_median_degrees"] == 0.0


def test_empty_surface_and_empty_normal_mask_fail_closed() -> None:
    empty = np.zeros((4, 4, 4), dtype=bool)
    with pytest.raises(ValueError, match="at least one"):
        surface_distance_metrics(
            empty,
            empty,
            spacing_xyz=(1.0, 1.0, 1.0),
            tolerance_distances=(1.0,),
        )
    gradient = np.ones((4, 4, 4, 3))
    with pytest.raises(ValueError, match="no valid"):
        normal_angle_metrics(
            gradient,
            gradient,
            evaluation_mask=empty,
        )
