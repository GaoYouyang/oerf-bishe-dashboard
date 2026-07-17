from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from demo_t16_operator.spatial_support_gate import (
    SUPPORT_METHODS,
    calibrate_support_gate,
    conformal_upper_quantile,
    decide_support,
    deployment_signature_audit,
    fit_support_score_model,
    knn_leave_one_out_scores,
    score_support_features,
    validate_score_model,
    validate_support_gate,
)


FEATURE_NAMES = ("view_fraction", "angle_span", "initial_residual_rms")
DEPLOYMENT_CONTEXT = {
    "feature_units": ["fraction", "radian", "pixel"],
    "grid_shape": [16, 16, 16],
    "renderer": "synthetic-interface-test",
}


def _fit_features() -> np.ndarray:
    rng = np.random.default_rng(173)
    return rng.normal(size=(36, len(FEATURE_NAMES)))


def _calibration_features() -> np.ndarray:
    rng = np.random.default_rng(281)
    return rng.normal(scale=0.9, size=(24, len(FEATURE_NAMES)))


def _fit(method: str = "robust_diagonal", **kwargs):
    return fit_support_score_model(
        _fit_features(),
        feature_names=FEATURE_NAMES,
        deployment_context=DEPLOYMENT_CONTEXT,
        method=method,
        **kwargs,
    )


@pytest.mark.parametrize("method", SUPPORT_METHODS)
def test_all_support_methods_are_deterministic_and_finite(method: str) -> None:
    kwargs = {"knn_k": 4} if method == "knn" else {}
    first = _fit(method, **kwargs)
    second = _fit(method, **kwargs)
    assert first == second
    scores = score_support_features(first, _calibration_features())
    assert scores.shape == (24,)
    assert np.all(np.isfinite(scores))


def test_split_conformal_quantile_uses_exact_finite_sample_rank() -> None:
    scores = np.arange(9, dtype=np.float64)
    threshold, rank = conformal_upper_quantile(scores, alpha=0.2)
    assert rank == 8
    assert threshold == 7.0


def test_split_conformal_quantile_rejects_unsupported_alpha() -> None:
    with pytest.raises(ValueError, match="too small"):
        conformal_upper_quantile(np.arange(8), alpha=0.1)


def test_constant_feature_becomes_an_exact_match_contract() -> None:
    features = np.column_stack((np.ones(20) * 7.0, np.linspace(-1.0, 1.0, 20)))
    model = fit_support_score_model(
        features,
        feature_names=("constant", "varying"),
        deployment_context=DEPLOYMENT_CONTEXT,
        method="robust_diagonal",
    )
    calibration = np.column_stack((np.ones(12) * 7.0, np.linspace(-0.8, 0.8, 12)))
    policy = calibrate_support_gate(
        model,
        calibration,
        alpha=0.1,
        policy_label="constant-feature-test",
    )
    decisions = decide_support(
        policy,
        np.array([[7.0, 0.0], [7.01, 0.0]]),
        feature_names=("constant", "varying"),
        deployment_context=DEPLOYMENT_CONTEXT,
    )
    assert model.exact_match_mask == (True, False)
    assert decisions[0].accepted is True
    assert decisions[1].accepted is False
    assert decisions[1].reason == "EXACT_MATCH_CONTRACT_FAILED_USE_FALLBACK"


def test_calibration_cannot_redefine_an_exact_match_feature() -> None:
    fit = np.column_stack((np.ones(20), np.linspace(-1.0, 1.0, 20)))
    model = fit_support_score_model(
        fit,
        feature_names=("constant", "varying"),
        deployment_context=DEPLOYMENT_CONTEXT,
        method="robust_diagonal",
    )
    calibration = np.column_stack((np.ones(12), np.linspace(-0.8, 0.8, 12)))
    calibration[3, 0] = 1.01
    with pytest.raises(ValueError, match="exact-match"):
        calibrate_support_gate(
            model,
            calibration,
            alpha=0.1,
            policy_label="invalid-calibration",
        )


def test_shrinkage_precision_is_positive_definite_under_collinearity() -> None:
    axis = np.linspace(-2.0, 2.0, 30)
    features = np.column_stack((axis, 2.0 * axis, np.ones_like(axis)))
    model = fit_support_score_model(
        features,
        feature_names=FEATURE_NAMES,
        deployment_context=DEPLOYMENT_CONTEXT,
        method="shrinkage_mahalanobis",
        covariance_shrinkage=0.2,
    )
    eigenvalues = np.linalg.eigvalsh(np.asarray(model.precision))
    assert np.all(eigenvalues > 0.0)
    validate_score_model(model)


def test_knn_leave_one_out_never_uses_zero_self_distance() -> None:
    features = np.arange(18, dtype=np.float64).reshape(6, 3)
    scores = knn_leave_one_out_scores(features, k=1)
    assert np.all(scores > 0.0)


def test_deployment_feature_reordering_fails_closed() -> None:
    model = _fit()
    policy = calibrate_support_gate(
        model,
        _calibration_features(),
        alpha=0.1,
        policy_label="unit-test",
    )
    with pytest.raises(ValueError, match="feature order"):
        decide_support(
            policy,
            _calibration_features()[:1],
            feature_names=tuple(reversed(FEATURE_NAMES)),
            deployment_context=DEPLOYMENT_CONTEXT,
        )


def test_deployment_context_mismatch_fails_closed() -> None:
    model = _fit()
    policy = calibrate_support_gate(
        model,
        _calibration_features(),
        alpha=0.1,
        policy_label="unit-test",
    )
    with pytest.raises(ValueError, match="deployment context"):
        decide_support(
            policy,
            _calibration_features()[:1],
            feature_names=FEATURE_NAMES,
            deployment_context={**DEPLOYMENT_CONTEXT, "grid_shape": [32, 32, 32]},
        )


def test_policy_and_model_tampering_fail_closed() -> None:
    model = _fit("axis_envelope")
    policy = calibrate_support_gate(
        model,
        _calibration_features(),
        alpha=0.1,
        policy_label="unit-test",
    )
    with pytest.raises(ValueError, match="model contract"):
        validate_score_model(replace(model, center=(99.0, *model.center[1:])))
    with pytest.raises(ValueError, match="policy contract"):
        validate_support_gate(replace(policy, threshold=policy.threshold + 1.0))


def test_nonfinite_or_wrong_width_features_are_rejected() -> None:
    model = _fit()
    with pytest.raises(ValueError, match="finite"):
        score_support_features(model, [[0.0, np.nan, 0.0]])
    with pytest.raises(ValueError, match="width"):
        score_support_features(model, [[0.0, 0.0]])


def test_calibrated_gate_accepts_center_and_rejects_extreme_shift() -> None:
    model = _fit("shrinkage_mahalanobis")
    policy = calibrate_support_gate(
        model,
        _calibration_features(),
        alpha=0.1,
        policy_label="unit-test",
    )
    center = np.asarray(model.center)[None]
    queries = np.concatenate((center, center + 100.0), axis=0)
    decisions = decide_support(
        policy,
        queries,
        feature_names=FEATURE_NAMES,
        deployment_context=DEPLOYMENT_CONTEXT,
    )
    assert decisions[0].accepted is True
    assert decisions[1].accepted is False
    assert decisions[1].reason == "OUTSIDE_CALIBRATED_SUPPORT_USE_FALLBACK"
    assert all(not row.uses_truth for row in decisions)
    assert all(not row.uses_target for row in decisions)


def test_deployment_signature_contains_no_truth_or_error_input() -> None:
    audit = deployment_signature_audit()
    assert audit["passed"] is True
    assert audit["forbidden_parameters_present"] == []
