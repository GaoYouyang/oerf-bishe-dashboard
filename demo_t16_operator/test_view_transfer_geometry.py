from __future__ import annotations

import numpy as np
import pytest

from demo_t16_operator.dual_regularization import error_reduction_percent
from demo_t16_operator.view_transfer_geometry import (
    gram_cosine,
    group_predictive_leverage,
    operator_change_cosine,
    projection_similarity,
    residual_field_transfer,
    similarity_weighted_gain,
)


def test_projection_similarity_detects_equal_and_orthogonal_row_spaces() -> None:
    first = np.array([[1.0, 0.0, 0.0]])
    equal = 7.0 * first
    orthogonal = np.array([[0.0, 1.0, 0.0]])
    assert projection_similarity(first, equal) == pytest.approx(1.0)
    assert projection_similarity(first, orthogonal) == pytest.approx(0.0)
    assert projection_similarity(first, orthogonal) == projection_similarity(
        orthogonal, first
    )


def test_gram_and_operator_change_cosines_are_scale_aware_as_declared() -> None:
    first = np.array([[1.0, 2.0], [0.0, 1.0]])
    assert gram_cosine(first, 3.0 * first) == pytest.approx(1.0)
    assert operator_change_cosine(
        2.0 * first, first, 4.0 * first, 2.0 * first
    ) == pytest.approx(1.0)


def test_group_predictive_leverage_has_hand_computable_diagonal() -> None:
    identity = np.eye(2)
    result = group_predictive_leverage(identity, identity, ridge_lambda=1.0)
    assert result.total == pytest.approx(1.0)
    assert result.mean_per_measurement == pytest.approx(0.5)
    assert result.maximum_diagonal == pytest.approx(0.5)


def test_similarity_weighted_gain_has_parameter_free_zero_weight_fallback() -> None:
    assert similarity_weighted_gain([1.0, 3.0], [2.0, 6.0]) == pytest.approx(5.0)
    assert similarity_weighted_gain([0.0, -1.0], [2.0, 6.0]) == pytest.approx(4.0)


def _two_view_problem(target_observation_scale: float = 1.0):
    operator = np.zeros((1, 2, 2, 2), dtype=np.float64)
    operator[0, 0] = np.eye(2)
    operator[0, 1] = np.array([[2.0, 0.0], [0.0, 0.5]])
    residual_field = np.array([2.0, 1.0])
    observation = np.einsum("dvnp,p->dvn", operator, residual_field)
    observation[:, 1, :] *= target_observation_scale
    return operator, observation, residual_field


def test_residual_transfer_matches_exact_source_recovery() -> None:
    operator, observation, residual_field = _two_view_problem()
    baseline_field = np.zeros(2)
    candidate_field = np.array([1.0, 0.0])
    result = residual_field_transfer(
        operator,
        operator,
        baseline_field,
        candidate_field,
        observation,
        np.ones(2),
        (0,),
        (1,),
        np.ones(2, dtype=bool),
        1e-12,
    )
    baseline_residual = operator[:, 1, :, :] @ residual_field
    candidate_residual = baseline_residual - operator[:, 1, :, :] @ candidate_field
    expected = error_reduction_percent(
        float(np.sqrt(np.mean(candidate_residual**2))),
        float(np.sqrt(np.mean(baseline_residual**2))),
    )
    assert result.residual_field == pytest.approx(residual_field, abs=3e-12)
    assert result.predicted_error_reductions_percent[0] == pytest.approx(expected)


def test_residual_transfer_cannot_read_target_observation() -> None:
    operator, observation, _ = _two_view_problem()
    poisoned = observation.copy()
    poisoned[:, 1, :] = 1e9
    kwargs = dict(
        baseline_operator=operator,
        candidate_operator=operator,
        baseline_field=np.zeros(2),
        candidate_field=np.array([1.0, 0.0]),
        noise_std=np.ones(2),
        source_views=(0,),
        target_views=(1,),
        support=np.ones(2, dtype=bool),
        ridge_lambda=1e-6,
    )
    clean_result = residual_field_transfer(observation=observation, **kwargs)
    poisoned_result = residual_field_transfer(observation=poisoned, **kwargs)
    assert poisoned_result.predicted_error_reductions_percent == pytest.approx(
        clean_result.predicted_error_reductions_percent
    )


def test_residual_transfer_rejects_overlapping_source_and_target() -> None:
    operator, observation, _ = _two_view_problem()
    with pytest.raises(ValueError, match="disjoint"):
        residual_field_transfer(
            operator,
            operator,
            np.zeros(2),
            np.ones(2),
            observation,
            np.ones(2),
            (0,),
            (0,),
            np.ones(2, dtype=bool),
            1e-3,
        )
