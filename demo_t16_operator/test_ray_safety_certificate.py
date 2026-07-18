from __future__ import annotations

import math

import pytest
import torch

try:
    from .automatic_discrete_multifidelity import SyntheticRayRig, smoothstep_grid_field
    from .ray_safety_certificate import (
        low_path_safety_certificate,
        smoothstep_derivative_bounds,
    )
except ImportError:
    from automatic_discrete_multifidelity import SyntheticRayRig, smoothstep_grid_field
    from ray_safety_certificate import (
        low_path_safety_certificate,
        smoothstep_derivative_bounds,
    )


def _rig() -> SyntheticRayRig:
    return SyntheticRayRig(
        rig_id="certificate-test",
        view_angle_degrees=20.0,
        detector_u=0.02,
        detector_z=-0.01,
        aperture_radius=0.02,
        path_half_length=0.55,
        cone_u=0.02,
        cone_z=0.015,
    )


def _states(count: int = 8) -> torch.Tensor:
    return torch.quasirandom.SobolEngine(2, scramble=True, seed=91).draw(count).to(
        torch.float64
    )


def _smooth_positive_grid(size: int = 9) -> torch.Tensor:
    axis = torch.linspace(-1.0, 1.0, size, dtype=torch.float64)
    z, y, x = torch.meshgrid(axis, axis, axis, indexing="ij")
    return torch.exp(-4.0 * (x.square() + 0.8 * y.square() + 0.6 * z.square()))


def test_constant_field_has_zero_derivative_bounds() -> None:
    grid = torch.full((7, 8, 9), 0.4, dtype=torch.float64)
    bounds = smoothstep_derivative_bounds(grid)
    assert bounds.scalar_minimum == pytest.approx(0.4)
    assert bounds.scalar_maximum == pytest.approx(0.4)
    assert bounds.gradient_norm_bound == 0.0
    assert bounds.hessian_frobenius_bound == 0.0


def test_bounds_dominate_sampled_coordinate_gradients() -> None:
    grid = _smooth_positive_grid()
    bounds = smoothstep_derivative_bounds(grid)
    points = 1.8 * torch.rand((512, 3), generator=torch.Generator().manual_seed(7)) - 0.9
    points = points.to(torch.float64).requires_grad_(True)
    field = smoothstep_grid_field(grid, points)
    gradient = torch.autograd.grad(field.sum(), points)[0]
    assert float(torch.max(torch.linalg.vector_norm(gradient, dim=-1))) <= (
        bounds.gradient_norm_bound * (1.0 + 1e-12)
    )


def test_bounds_dominate_sampled_hessian_frobenius_norms() -> None:
    grid = _smooth_positive_grid()
    bounds = smoothstep_derivative_bounds(grid)
    points = 1.7 * torch.rand((64, 3), generator=torch.Generator().manual_seed(13)) - 0.85
    points = points.to(torch.float64).requires_grad_(True)
    field = smoothstep_grid_field(grid, points)
    gradient = torch.autograd.grad(field.sum(), points, create_graph=True)[0]
    rows = [
        torch.autograd.grad(gradient[:, axis].sum(), points, retain_graph=True)[0]
        for axis in range(3)
    ]
    hessian = torch.stack(rows, dim=1)
    assert float(torch.max(torch.linalg.matrix_norm(hessian, ord="fro"))) <= (
        bounds.hessian_frobenius_bound * (1.0 + 1e-12)
    )


def test_constant_field_certificate_is_safe_and_zero_risk() -> None:
    grid = torch.full((9, 9, 9), 0.25, dtype=torch.float64)
    result = low_path_safety_certificate(
        grid,
        _states(),
        _rig(),
        refractivity_scale=3e-4,
        difference_step=0.002,
        support_threshold=0.1,
        frustum_half_width_u=0.005,
        frustum_half_width_v=0.005,
        support_interval_count=32,
    )
    assert torch.all(result.safe_mask)
    assert all(count == 0 for count in result.straight_support_crossings_per_ray)
    assert result.continuous_path_deviation_bound == 0.0
    assert result.continuous_direction_change_bound == 0.0
    assert torch.max(result.local_deviation_proxy) == 0.0
    assert torch.all(result.residual_risk_proxy > 0.0)
    assert all(not codes for codes in result.failure_codes_per_ray)


def test_frustum_bound_fails_closed_as_refractivity_grows() -> None:
    grid = _smooth_positive_grid()
    common = dict(
        values_zyx=grid,
        pupil_states=_states(),
        rig=_rig(),
        difference_step=0.002,
        support_threshold=0.0,
        frustum_half_width_u=0.005,
        frustum_half_width_v=0.005,
        support_interval_count=32,
    )
    low = low_path_safety_certificate(refractivity_scale=1e-5, **common)
    high = low_path_safety_certificate(refractivity_scale=0.03, **common)
    assert low.continuous_path_deviation_bound < high.continuous_path_deviation_bound
    assert int(torch.sum(high.safe_mask)) <= int(torch.sum(low.safe_mask))
    assert not torch.any(high.domain_frustum_safe_mask)
    assert all("FAIL_FRUSTUM_BOUND" in codes for codes in high.failure_codes_per_ray)


def test_support_threshold_ambiguity_fails_closed() -> None:
    grid = _smooth_positive_grid()
    states = _states(4)
    preliminary = low_path_safety_certificate(
        grid,
        states,
        _rig(),
        refractivity_scale=1e-5,
        difference_step=0.002,
        support_threshold=0.1,
        frustum_half_width_u=0.2,
        frustum_half_width_v=0.2,
        support_interval_count=16,
    )
    threshold = float(torch.max(grid))
    ambiguous = low_path_safety_certificate(
        grid,
        states,
        _rig(),
        refractivity_scale=1e-5,
        difference_step=0.002,
        support_threshold=threshold,
        frustum_half_width_u=0.2,
        frustum_half_width_v=0.2,
        support_interval_count=16,
    )
    assert preliminary.residual_risk_proxy.shape == (4,)
    assert not torch.all(ambiguous.support_topology_safe_mask)
    assert any(
        "FAIL_SUPPORT_TOPOLOGY_BOUND" in codes
        for codes in ambiguous.failure_codes_per_ray
    )


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("refractivity_scale", 0.0),
        ("difference_step", -0.1),
        ("frustum_half_width_u", 0.0),
        ("support_interval_count", 3),
        ("numerical_path_buffer", -1e-3),
    ],
)
def test_invalid_certificate_inputs_fail(keyword: str, value: float) -> None:
    kwargs = dict(
        refractivity_scale=3e-4,
        difference_step=0.002,
        support_threshold=0.1,
        frustum_half_width_u=0.005,
        frustum_half_width_v=0.005,
        support_interval_count=16,
        numerical_path_buffer=0.0,
    )
    kwargs[keyword] = value
    with pytest.raises(ValueError):
        low_path_safety_certificate(
            _smooth_positive_grid(),
            _states(2),
            _rig(),
            **kwargs,
        )


def test_derivative_bound_scaling_is_finite() -> None:
    grid = _smooth_positive_grid(11)
    bounds = smoothstep_derivative_bounds(grid)
    assert math.isfinite(bounds.gradient_norm_bound)
    assert math.isfinite(bounds.hessian_frobenius_bound)
    assert bounds.gradient_norm_bound > 0.0
    assert bounds.hessian_frobenius_bound > 0.0
