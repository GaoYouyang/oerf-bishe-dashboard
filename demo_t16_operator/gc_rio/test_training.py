from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from .model import GeometryConditionedResidualInverseOperator
from .training import (
    FORBIDDEN_FIT_KEYS,
    ObjectiveWeights,
    cluster_mean_metric,
    predictor_tensors,
    row_whitened_rmse,
    training_objective,
)


def _output():
    model = GeometryConditionedResidualInverseOperator(
        (2, 2, 2), hidden_channels=4, residual_blocks=1
    )
    generator = torch.Generator().manual_seed(7)
    return model(
        source_operator=torch.randn(2, 2, 3, 8, generator=generator) * 0.1,
        target_operator=torch.randn(2, 3, 8, generator=generator) * 0.1,
        source_residual=torch.randn(2, 2, 3, generator=generator) * 0.02,
        source_sigma=torch.full((2, 2), 0.04),
        target_sigma=torch.full((2,), 0.05),
        base_field=torch.zeros(2, 8),
        analytic_correction=torch.randn(2, 8, generator=generator) * 0.01,
        support=torch.ones(2, 8),
    )


def test_training_objective_signature_has_no_truth_inputs() -> None:
    names = set(inspect.signature(training_objective).parameters)
    assert not names & FORBIDDEN_FIT_KEYS


def test_predictor_tensors_rejects_truth_bearing_keys() -> None:
    valid = {
        "source_operator": np.zeros((1, 2, 3, 8)),
        "target_operator": np.zeros((1, 3, 8)),
        "source_residual": np.zeros((1, 2, 3)),
        "source_sigma": np.ones((1, 2)),
        "target_sigma": np.ones(1),
        "base_field": np.zeros((1, 8)),
        "analytic_correction": np.zeros((1, 8)),
        "support": np.ones((1, 8)),
    }
    assert set(predictor_tensors(valid, "cpu")) == set(valid)
    valid["truth_field"] = np.ones((1, 8))
    with pytest.raises(ValueError, match="forbidden"):
        predictor_tensors(valid, "cpu")


def test_target_objective_is_noise_normalized() -> None:
    output = _output()
    label = output.target_residual_prediction.detach() + 0.10
    terms = training_objective(
        output, label, torch.full((2,), 0.05), ObjectiveWeights(1.0, 0.0, 0.0)
    )
    torch.testing.assert_close(terms.target, torch.tensor(4.0))
    torch.testing.assert_close(terms.total, terms.target)


def test_cluster_metric_weights_rig_family_cells_equally() -> None:
    metric = np.asarray([1.0, 1.0, 9.0])
    overall, cells = cluster_mean_metric(
        metric, ["rig-a", "rig-a", "rig-b"], ["family-a", "family-a", "family-b"]
    )
    assert overall == pytest.approx(5.0)
    assert [cell["row_count"] for cell in cells] == [2, 1]


def test_row_whitened_rmse_matches_manual_value() -> None:
    prediction = np.asarray([[2.0, 4.0], [1.0, 1.0]])
    label = np.asarray([[1.0, 2.0], [1.5, 0.5]])
    result = row_whitened_rmse(prediction, label, np.asarray([1.0, 0.5]))
    np.testing.assert_allclose(result, [np.sqrt(2.5), 1.0])
