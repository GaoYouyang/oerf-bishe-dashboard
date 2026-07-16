from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

try:
    from .adjoint_landweber import (
        adjoint_inner_product_relative_error,
        feasibility_project,
        finite_difference_gradient_relative_error,
        forward_project,
        geometry_normalization,
        landweber_trajectory,
        masked_adjoint,
    )
except ImportError:
    from adjoint_landweber import (
        adjoint_inner_product_relative_error,
        feasibility_project,
        finite_difference_gradient_relative_error,
        forward_project,
        geometry_normalization,
        landweber_trajectory,
        masked_adjoint,
    )


def toy_problem(seed: int = 4):
    rng = np.random.default_rng(seed)
    operator = rng.normal(size=(4, 3, 6))
    volume = rng.normal(size=(2, 2, 2, 3))
    masks = np.asarray([[1, 1, 0, 1], [0, 1, 1, 1]], dtype=np.float64)
    observation = forward_project(volume, operator)
    return rng, operator, volume, masks, observation


def test_forward_and_masked_adjoint_have_expected_shapes():
    _, operator, volume, masks, observation = toy_problem()
    adjoint = masked_adjoint(observation, operator, masks, volume.shape[1:])
    assert observation.shape == (2, 2, 4, 3)
    assert adjoint.shape == volume.shape


def test_adjoint_inner_product_identity_for_each_geometry():
    rng, operator, volume, masks, _ = toy_problem()
    projection = rng.normal(size=(1, 2, 4, 3))
    for index, mask in enumerate(masks):
        error = adjoint_inner_product_relative_error(
            volume[index : index + 1], projection, operator, mask
        )
        assert error < 1e-12


def test_data_fidelity_gradient_matches_finite_difference():
    rng, operator, volume, masks, observation = toy_problem()
    direction = rng.normal(size=volume[0:1].shape)
    error = finite_difference_gradient_relative_error(
        volume[0:1] + 0.15,
        direction,
        observation[0:1] - 0.2,
        operator,
        masks[0],
    )
    assert error < 2e-7


def test_feasibility_projection_is_nonnegative_and_support_limited():
    volume = np.asarray([[[[-1.0, 2.0], [3.0, -4.0]]]])
    support = np.asarray([[[1.0, 0.0], [0.5, 1.0]]])
    projected = feasibility_project(volume, support)
    assert np.array_equal(projected, np.asarray([[[[0.0, 0.0], [3.0, 0.0]]]]))
    assert np.array_equal(feasibility_project(projected, support), projected)


def test_geometry_normalization_matches_direct_spectral_norm():
    _, operator, _, masks, _ = toy_problem()
    normalization = geometry_normalization(operator, masks)
    for mask in masks:
        key = "".join(str(int(value)) for value in mask)
        active = operator[mask > 0.5].reshape(-1, operator.shape[-1])
        expected = np.linalg.svd(active, compute_uv=False)[0] ** 2
        assert np.isclose(normalization[key]["spectral_constant"], expected)
        assert float(normalization[key]["preconditioned_spectral_constant"]) > 0.0


def test_inactive_camera_values_cannot_change_landweber_update():
    _, operator, volume, masks, observation = toy_problem()
    support = np.ones(volume.shape[1:])
    start = feasibility_project(volume, support)
    normalization = geometry_normalization(operator, masks)
    corrupted = observation.copy()
    corrupted[0, :, masks[0] < 0.5] += 1e6
    first = landweber_trajectory(
        start[0:1],
        observation[0:1],
        operator,
        masks[0:1],
        support,
        1.0,
        [2],
        normalization,
    )[2]
    second = landweber_trajectory(
        start[0:1],
        corrupted[0:1],
        operator,
        masks[0:1],
        support,
        1.0,
        [2],
        normalization,
    )[2]
    assert np.array_equal(first, second)


@pytest.mark.parametrize("method", ["standard", "jacobi"])
def test_landweber_reduces_noiseless_data_residual(method: str):
    rng = np.random.default_rng(12)
    operator = rng.uniform(0.05, 1.0, size=(3, 4, 6))
    masks = np.ones((1, 3), dtype=np.float64)
    support = np.ones((2, 2, 3), dtype=np.float64)
    truth = rng.uniform(0.1, 1.0, size=(1, 2, 2, 3))
    observation = forward_project(truth, operator)
    start = np.zeros_like(truth)
    normalization = geometry_normalization(operator, masks)
    trajectory = landweber_trajectory(
        start,
        observation,
        operator,
        masks,
        support,
        1.0,
        [1, 2, 4, 8],
        normalization,
        method=method,
    )
    errors = [
        np.linalg.norm(forward_project(trajectory[iteration], operator) - observation)
        for iteration in [1, 2, 4, 8]
    ]
    assert all(right <= left + 1e-10 for left, right in zip(errors, errors[1:]))


def test_landweber_rejects_unstable_normalized_step():
    _, operator, volume, masks, observation = toy_problem()
    normalization = geometry_normalization(operator, masks)
    with pytest.raises(ValueError, match="must lie"):
        landweber_trajectory(
            volume,
            observation,
            operator,
            masks,
            np.ones(volume.shape[1:]),
            2.0,
            [1],
            normalization,
        )


def test_v3k_c_protocol_keeps_selection_and_claim_boundaries_locked():
    config_path = Path(__file__).parent / "configs" / "v3k_c_adjoint_landweber_gate.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["selection_split"] == "val"
    assert config["numerical_protocol"]["audit_camera_used_for_selection"] is False
    assert max(config["numerical_protocol"]["step_fractions"]) < 2.0
    assert config["claims_boundary"]["superiority_tested"] is False
    assert config["claims_boundary"]["blind_final_opened"] is False
