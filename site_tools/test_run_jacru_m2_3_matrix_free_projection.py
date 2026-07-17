import pytest
import torch

from site_tools.run_jacru_m2_3_matrix_free_projection import (
    _convex_quadratic_feasible_interval,
    _convex_quadratic_minimizer,
    _dense_camera_block_preconditioner,
    _projection_closure_relative_error,
    _safe_reduction_retention,
)


def test_dense_camera_block_preconditioner_matches_explicit_block_solves() -> None:
    matrix = torch.tensor(
        [
            [2.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 1.0, 2.0],
        ],
        dtype=torch.float64,
    )
    apply, metadata = _dense_camera_block_preconditioner(
        matrix=matrix,
        camera_index=torch.tensor([0, 1]),
        measurement_shape=(2, 2),
        damping=0.1,
    )
    value = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    observed = apply(value)
    gram = matrix @ matrix.mT + 0.1 * torch.eye(4, dtype=torch.float64)
    expected = torch.cat(
        (
            torch.linalg.solve(gram[:2, :2], value[0]),
            torch.linalg.solve(gram[2:, 2:], value[1]),
        )
    ).reshape(2, 2)
    torch.testing.assert_close(observed, expected)
    assert metadata["block_count"] == 2
    assert metadata["largest_block_size"] == 2
    assert metadata["minimum_block_eigenvalue"] > 0.0
    assert metadata["maximum_block_condition_number"] >= 1.0


def test_dense_camera_block_preconditioner_rejects_layout_mismatch() -> None:
    with pytest.raises(ValueError, match="one value per measurement ray"):
        _dense_camera_block_preconditioner(
            matrix=torch.eye(4, dtype=torch.float64),
            camera_index=torch.tensor([0]),
            measurement_shape=(2, 2),
            damping=0.0,
        )


def test_oracle_reduction_retention_is_undefined_without_positive_headroom() -> None:
    value, defined = _safe_reduction_retention(
        base_error=0.5,
        candidate_error=0.4,
        oracle_error=0.5,
    )
    assert value == 0.0
    assert defined is False


def test_projection_closure_relative_error_matches_affine_identity() -> None:
    residual = torch.tensor([1.0, -2.0], dtype=torch.float64)
    dual = torch.tensor([0.5, 1.5], dtype=torch.float64)
    damping = 0.2
    visible = residual + damping * dual
    observed = _projection_closure_relative_error(
        visible=visible,
        system_residual=residual,
        dual=dual,
        damping=damping,
        initial_system_norm=3.0,
    )
    assert observed == pytest.approx(0.0, abs=1e-15)


def test_projection_closure_relative_error_rejects_shape_drift() -> None:
    with pytest.raises(ValueError, match="share one shape"):
        _projection_closure_relative_error(
            visible=torch.ones(2),
            system_residual=torch.ones(3),
            dual=torch.ones(2),
            damping=0.0,
            initial_system_norm=1.0,
        )


def test_convex_quadratic_feasible_interval_clips_two_roots() -> None:
    observed = _convex_quadratic_feasible_interval(
        quadratic=1.0,
        linear=-1.0,
        constant=0.16,
    )
    assert observed == pytest.approx((0.2, 0.8))


def test_convex_quadratic_feasible_interval_handles_constant_cases() -> None:
    assert _convex_quadratic_feasible_interval(
        quadratic=0.0,
        linear=0.0,
        constant=-1.0,
    ) == (0.0, 1.0)
    assert _convex_quadratic_feasible_interval(
        quadratic=0.0,
        linear=0.0,
        constant=1.0,
    ) is None


def test_convex_quadratic_minimizer_clips_vertex() -> None:
    assert _convex_quadratic_minimizer(
        quadratic=2.0,
        linear=-8.0,
        interval=(0.1, 0.7),
    ) == pytest.approx(0.7)
