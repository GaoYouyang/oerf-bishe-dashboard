from __future__ import annotations

import numpy as np
import pytest

from .run_v5o_prior_anchored_frontier import prior_anchored_bb_correction


def _row() -> dict:
    return {
        "source_operator": np.asarray(
            [[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32
        ),
        "source_residual": np.asarray([[0.2, -0.1]], dtype=np.float32),
        "source_sigma": np.asarray([1.0], dtype=np.float32),
        "support": np.asarray([1.0, 1.0], dtype=np.float32),
        "base_field": np.asarray([0.5, 0.5], dtype=np.float32),
    }


def test_zero_refinement_returns_projected_prior() -> None:
    result = prior_anchored_bb_correction(
        _row(),
        np.asarray([0.1, -0.8], dtype=np.float32),
        iterations=0,
        relative_anchor=0.1,
    )
    np.testing.assert_allclose(result, [0.1, -0.5])


def test_zero_anchor_converges_toward_source_residual() -> None:
    result = prior_anchored_bb_correction(
        _row(),
        np.asarray([-0.2, 0.2], dtype=np.float32),
        iterations=8,
        relative_anchor=0.0,
    )
    np.testing.assert_allclose(result, [0.2, -0.1], atol=1e-4)


def test_anchor_keeps_solution_closer_to_prior() -> None:
    prior = np.asarray([0.0, 0.2], dtype=np.float32)
    weak = prior_anchored_bb_correction(
        _row(), prior, iterations=8, relative_anchor=0.0
    )
    strong = prior_anchored_bb_correction(
        _row(), prior, iterations=8, relative_anchor=10.0
    )
    assert np.linalg.norm(strong - prior) < np.linalg.norm(weak - prior)


def test_invalid_prior_is_rejected() -> None:
    with pytest.raises(ValueError, match="prior_correction"):
        prior_anchored_bb_correction(
            _row(),
            np.asarray([np.nan, 0.0], dtype=np.float32),
            iterations=1,
            relative_anchor=0.0,
        )


def test_prior_anchored_bb_counts_executed_source_operator_calls() -> None:
    counter: dict[str, int] = {}
    prior_anchored_bb_correction(
        _row(),
        np.zeros(2, dtype=np.float32),
        iterations=5,
        relative_anchor=0.1,
        operator_call_counter=counter,
    )
    assert counter == {"source_forward": 5, "source_adjoint": 5}
