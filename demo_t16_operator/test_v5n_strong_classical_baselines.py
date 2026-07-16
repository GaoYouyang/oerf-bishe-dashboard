from __future__ import annotations

import numpy as np

from .run_v5n_strong_classical_baselines import (
    projected_bb_correction,
    support_ridge_correction,
)


def _row() -> dict:
    operator = np.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
    return {
        "source_operator": operator,
        "source_residual": np.asarray([[0.2, -0.1]], dtype=np.float32),
        "source_sigma": np.asarray([1.0], dtype=np.float32),
        "support": np.asarray([1.0, 1.0], dtype=np.float32),
        "base_field": np.asarray([0.5, 0.5], dtype=np.float32),
    }


def test_support_ridge_is_finite_and_shrinks_with_regularization() -> None:
    weak = support_ridge_correction(_row(), 1e-6)
    strong = support_ridge_correction(_row(), 10.0)
    assert np.all(np.isfinite(weak))
    assert np.linalg.norm(strong) < np.linalg.norm(weak)


def test_projected_bb_zero_iteration_is_exact_zero() -> None:
    np.testing.assert_equal(
        projected_bb_correction(_row(), 0), np.zeros(2, dtype=np.float32)
    )


def test_projected_bb_zero_iteration_preserves_feasible_warm_start() -> None:
    initial = np.asarray([0.1, -0.8], dtype=np.float32)
    correction = projected_bb_correction(
        _row(), 0, initial_correction=initial
    )
    np.testing.assert_allclose(correction, [0.1, -0.5])


def test_projected_bb_reduces_source_residual_without_negative_field() -> None:
    correction = projected_bb_correction(_row(), 4)
    assert np.linalg.norm(correction - _row()["source_residual"].reshape(-1)) < 1e-4
    assert np.all(_row()["base_field"] + correction >= 0.0)


def test_projected_bb_counts_executed_source_operator_calls() -> None:
    counter: dict[str, int] = {}
    projected_bb_correction(_row(), 4, operator_call_counter=counter)
    assert counter == {"source_forward": 4, "source_adjoint": 4}
