from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from demo_t16_operator.side_weighted_adjoint_certificate import (
    UNIT_ROUNDOFF,
    evaluate_side_weighted_adjoint,
    gamma_n,
    summarize_side_weighted_probes,
)


def _normalize(value: np.ndarray) -> np.ndarray:
    return value / np.linalg.norm(value)


def test_gamma_n_matches_binary64_formula_and_is_monotone() -> None:
    expected = 4913 * UNIT_ROUNDOFF / (1.0 - 4913 * UNIT_ROUNDOFF)

    assert gamma_n(4913) == expected
    assert gamma_n(1) < gamma_n(8) < gamma_n(4913)


@pytest.mark.parametrize("bad", [True, False, 1.5, "2"])
def test_gamma_n_rejects_noninteger_counts(bad: object) -> None:
    with pytest.raises(TypeError):
        gamma_n(bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [0, -1, -100])
def test_gamma_n_rejects_nonpositive_counts(bad: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        gamma_n(bad)


def test_gamma_n_rejects_count_outside_roundoff_model() -> None:
    first_invalid_count = int(1.0 / UNIT_ROUNDOFF)

    with pytest.raises(ValueError, match="too large"):
        gamma_n(first_invalid_count)


def test_side_specific_gamma_formula_uses_each_contraction_dimension() -> None:
    jvp = np.array([3.0, 4.0])
    cotangent = np.array([12.0, -5.0])
    tangent = np.array([1.0, 2.0, 2.0])
    vjp = np.array([4.0, -1.0, 2.0])

    evidence = evaluate_side_weighted_adjoint(
        jvp, cotangent, tangent, vjp
    )

    expected_left_action = np.linalg.norm(jvp) * np.linalg.norm(cotangent)
    expected_right_action = np.linalg.norm(tangent) * np.linalg.norm(vjp)
    expected_side_scale = (
        gamma_n(jvp.size) * expected_left_action
        + gamma_n(tangent.size) * expected_right_action
    )
    expected_contraction_envelope = gamma_n(jvp.size) * np.sum(
        np.abs(jvp * cotangent)
    ) + gamma_n(tangent.size) * np.sum(np.abs(tangent * vjp))
    wrong_shared_gamma_scale = gamma_n(tangent.size) * (
        expected_left_action + expected_right_action
    )

    assert evidence.output_dimension == 2
    assert evidence.input_dimension == 3
    assert evidence.output_gamma == gamma_n(2)
    assert evidence.input_gamma == gamma_n(3)
    assert evidence.left_action_scale == pytest.approx(expected_left_action)
    assert evidence.right_action_scale == pytest.approx(expected_right_action)
    assert evidence.maximum_action_scale == pytest.approx(expected_left_action)
    assert evidence.lhs == pytest.approx(16.0)
    assert evidence.rhs == pytest.approx(6.0)
    assert evidence.absolute_defect == pytest.approx(10.0)
    assert evidence.signal_relative_defect == pytest.approx(10.0 / 16.0)
    assert evidence.normwise_defect == pytest.approx(
        10.0 / expected_left_action
    )
    assert evidence.side_weighted_gamma_scale == pytest.approx(
        expected_side_scale
    )
    assert evidence.side_weighted_gamma_scale < wrong_shared_gamma_scale
    assert evidence.side_weighted_gamma_score == pytest.approx(
        10.0 / expected_side_scale
    )
    assert evidence.contraction_roundoff_envelope == pytest.approx(
        expected_contraction_envelope
    )
    assert evidence.contraction_envelope_ratio == pytest.approx(
        10.0 / expected_contraction_envelope
    )
    assert evidence.finite


@pytest.mark.parametrize(
    ("jvp", "cotangent", "tangent", "vjp", "message"),
    [
        (np.ones(2), np.ones(3), np.ones(4), np.ones(4), "jvp"),
        (np.ones(2), np.ones(2), np.ones(3), np.ones(4), "tangent"),
    ],
)
def test_evaluate_rejects_dimension_mismatch(
    jvp: np.ndarray,
    cotangent: np.ndarray,
    tangent: np.ndarray,
    vjp: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_side_weighted_adjoint(jvp, cotangent, tangent, vjp)


@pytest.mark.parametrize("position", range(4))
@pytest.mark.parametrize(
    "bad_vector",
    [
        np.array([], dtype=np.float64),
        np.array([0.0, np.nan]),
        np.array([0.0, np.inf]),
        np.array([0.0, -np.inf]),
    ],
    ids=["empty", "nan", "positive-inf", "negative-inf"],
)
def test_evaluate_rejects_empty_or_nonfinite_vector_inputs(
    position: int, bad_vector: np.ndarray
) -> None:
    arguments = [np.ones(2), np.ones(2), np.ones(3), np.ones(3)]
    arguments[position] = bad_vector

    with pytest.raises(ValueError, match="nonempty finite vector"):
        evaluate_side_weighted_adjoint(*arguments)


@pytest.mark.parametrize("bad_floor", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_evaluate_rejects_nonpositive_or_nonfinite_denominator_floor(
    bad_floor: float,
) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        evaluate_side_weighted_adjoint(
            np.ones(2),
            np.ones(2),
            np.ones(3),
            np.ones(3),
            denominator_floor=bad_floor,
        )


def test_probe_summary_reports_extrema_and_propagates_finiteness() -> None:
    rows = [
        evaluate_side_weighted_adjoint(
            [1.0, 2.0], [2.0, -1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]
        ),
        evaluate_side_weighted_adjoint(
            [2.0, -3.0], [1.0, 4.0], [0.0, 2.0, 1.0], [3.0, -1.0, 2.0]
        ),
        evaluate_side_weighted_adjoint(
            [0.5, 1.5], [-2.0, 3.0], [2.0, 1.0, -1.0], [0.0, 4.0, 1.0]
        ),
    ]
    rows[1] = replace(rows[1], finite=False)

    summary = summarize_side_weighted_probes(row for row in rows)

    assert summary.probe_count == len(rows)
    assert not summary.all_finite
    assert summary.maximum_signal_relative_defect == max(
        row.signal_relative_defect for row in rows
    )
    assert summary.maximum_side_weighted_gamma_score == max(
        row.side_weighted_gamma_score for row in rows
    )
    assert summary.maximum_contraction_envelope_ratio == max(
        row.contraction_envelope_ratio for row in rows
    )
    assert summary.minimum_signal == min(
        max(abs(row.lhs), abs(row.rhs)) for row in rows
    )
    assert summary.maximum_condition_proxy == max(
        row.dot_condition_proxy for row in rows
    )
    assert summary.to_dict()["probe_count"] == len(rows)


def test_probe_summary_rejects_empty_iterable() -> None:
    with pytest.raises(ValueError, match="at least one probe"):
        summarize_side_weighted_probes(iter(()))


def test_low_bilinear_signal_is_not_a_large_normwise_defect() -> None:
    rng = np.random.default_rng(11)
    matrix = rng.normal(size=(8, 97)) / math.sqrt(97)
    tangent = _normalize(rng.normal(size=97))
    jvp = matrix @ tangent
    cotangent = rng.normal(size=8)
    cotangent -= jvp * float(cotangent @ jvp) / float(jvp @ jvp)
    cotangent = _normalize(cotangent)
    vjp = matrix.T @ cotangent

    evidence = evaluate_side_weighted_adjoint(
        jvp, cotangent, tangent, vjp
    )

    assert evidence.signal_relative_defect > 0.1
    assert evidence.dot_condition_proxy > 1e12
    assert evidence.normwise_defect < 1e-14
    assert evidence.side_weighted_gamma_score < 0.1
    assert evidence.signal_relative_defect > 1e12 * evidence.normwise_defect
