from __future__ import annotations

import math

import numpy as np
import pytest

from demo_t16_operator.mixed_scale_adjoint_certificate import (
    FLOAT64_UNIT_ROUNDOFF,
    evaluate_mixed_scale_adjoint,
    gamma_n,
    summarize_probe_set,
)


def _normalize(value: np.ndarray) -> np.ndarray:
    return value / np.linalg.norm(value)


def test_gamma_n_matches_binary64_formula() -> None:
    expected = 4913 * FLOAT64_UNIT_ROUNDOFF / (
        1.0 - 4913 * FLOAT64_UNIT_ROUNDOFF
    )
    assert gamma_n(4913) == expected
    assert gamma_n(8) < gamma_n(4913)


@pytest.mark.parametrize("bad", [0, -1])
def test_gamma_n_rejects_nonpositive_counts(bad: int) -> None:
    with pytest.raises(ValueError):
        gamma_n(bad)


def test_correct_low_signal_operator_can_fail_relative_but_not_normwise() -> None:
    rng = np.random.default_rng(11)
    matrix = rng.normal(size=(8, 97)) / math.sqrt(97)
    tangent = _normalize(rng.normal(size=97))
    jvp = matrix @ tangent
    cotangent = rng.normal(size=8)
    cotangent = cotangent - jvp * float(cotangent @ jvp) / float(jvp @ jvp)
    cotangent = _normalize(cotangent)
    vjp = matrix.T @ cotangent

    evidence = evaluate_mixed_scale_adjoint(jvp, cotangent, tangent, vjp)

    assert evidence.dot_relative_defect > 1e-10
    assert evidence.gamma_scaled_normwise_score < 0.1
    assert evidence.dot_condition_proxy > 1e12


def test_aligned_vjp_fault_is_visible_to_the_adjoint_probe() -> None:
    rng = np.random.default_rng(12)
    matrix = rng.normal(size=(5, 101)) / math.sqrt(101)
    tangent = _normalize(rng.normal(size=101))
    cotangent = _normalize(rng.normal(size=5))
    jvp = matrix @ tangent
    vjp = matrix.T @ cotangent
    clean = evaluate_mixed_scale_adjoint(jvp, cotangent, tangent, vjp)
    faulty = evaluate_mixed_scale_adjoint(
        jvp,
        cotangent,
        tangent,
        vjp + 1e-8 * np.linalg.norm(vjp) * tangent,
    )

    assert clean.gamma_scaled_normwise_score < 0.1
    assert faulty.gamma_scaled_normwise_score > 1e4


def test_one_probe_can_miss_an_orthogonal_vjp_fault() -> None:
    matrix = np.eye(3, dtype=np.float64)
    tangent_0 = _normalize(np.array([1.0, 1.0, 0.0]))
    tangent_1 = np.array([0.0, 0.0, 1.0])
    cotangent = _normalize(np.array([1.0, 2.0, 3.0]))
    true_vjp = matrix.T @ cotangent
    blind_error = _normalize(np.array([1.0, -1.0, 1.0]))
    blind_error = blind_error - tangent_0 * float(blind_error @ tangent_0)
    blind_error = _normalize(blind_error)
    faulty_vjp = true_vjp + 1e-8 * blind_error
    first = evaluate_mixed_scale_adjoint(
        matrix @ tangent_0, cotangent, tangent_0, faulty_vjp
    )
    second = evaluate_mixed_scale_adjoint(
        matrix @ tangent_1, cotangent, tangent_1, faulty_vjp
    )

    assert first.gamma_scaled_normwise_score < 1.0
    assert second.gamma_scaled_normwise_score > 1e6
    summary = summarize_probe_set((first, second))
    assert summary.probe_count == 2
    assert summary.maximum_gamma_scaled_normwise_score == (
        second.gamma_scaled_normwise_score
    )


def test_shape_mismatch_fails_closed() -> None:
    with pytest.raises(ValueError):
        evaluate_mixed_scale_adjoint(
            np.ones(2), np.ones(3), np.ones(4), np.ones(4)
        )
