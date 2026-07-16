from __future__ import annotations

import numpy as np

try:
    from .run_v5b_forward_model_convergence import rank_operator_bank
except ImportError:
    from run_v5b_forward_model_convergence import rank_operator_bank


def _operator(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(1, 1, 1, -1)


def test_profiled_gain_removes_only_global_scale():
    truth = _operator([1.0, 2.0])
    bank = np.stack([_operator([2.0, 4.0]), _operator([1.0, -2.0])])
    metrics = rank_operator_bank(
        truth,
        bank,
        [0],
        candidate_reference_scale=1.0,
        truth_reference_scale=1.0,
    )
    assert int(np.argmin(metrics["profiled_gain"])) == 0
    assert metrics["profiled_gain"][0] < 1e-12
    assert metrics["native"][0] > 0.9


def test_shared_scale_uses_declared_reference_ratio():
    truth = _operator([1.0, 2.0])
    bank = np.stack([_operator([2.0, 4.0]), _operator([1.0, -2.0])])
    metrics = rank_operator_bank(
        truth,
        bank,
        [0],
        candidate_reference_scale=0.5,
        truth_reference_scale=1.0,
    )
    assert metrics["shared_scale"][0] < 1e-12
    assert metrics["native"][0] > 0.9


def test_invalid_views_are_rejected():
    truth = _operator([1.0, 2.0])
    bank = np.stack([truth])
    with np.testing.assert_raises(ValueError):
        rank_operator_bank(
            truth,
            bank,
            [],
            candidate_reference_scale=1.0,
            truth_reference_scale=1.0,
        )
