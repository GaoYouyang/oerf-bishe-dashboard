"""Truth-free support gates for spatial BOST and neural-operator studies.

The score model is fit on one set of complete geometry clusters and its
threshold is calibrated on a disjoint in-support set.  Deployment accepts only
declared observable features.  This module deliberately does not select a
reconstruction method or make a quality claim; it only answers whether a
frozen policy authorizes a learned method to leave its classical fallback.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import hashlib
import inspect
import json
import math
from typing import Any, Literal

import numpy as np


SCHEMA_VERSION = "spatial-support-gate-1.0"
STATUS = "DEVELOPMENT_ONLY_TRUTH_FREE_SUPPORT_INTERFACE"
SupportMethod = Literal[
    "axis_envelope",
    "robust_diagonal",
    "shrinkage_mahalanobis",
    "knn",
]
SUPPORT_METHODS: tuple[SupportMethod, ...] = (
    "axis_envelope",
    "robust_diagonal",
    "shrinkage_mahalanobis",
    "knn",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _validated_feature_names(feature_names: tuple[str, ...]) -> tuple[str, ...]:
    names = tuple(str(name) for name in feature_names)
    if not names or any(not name.strip() for name in names):
        raise ValueError("feature_names must contain nonempty names")
    if len(set(names)) != len(names):
        raise ValueError("feature_names must be unique")
    return names


def feature_schema_sha256(feature_names: tuple[str, ...]) -> str:
    """Hash the exact observable feature order used at deployment."""

    names = _validated_feature_names(feature_names)
    return _sha256_json(
        {
            "dtype": "numpy.float64",
            "feature_names": list(names),
            "source": "declared_observable_geometry_or_initial_residual_features",
            "version": 1,
        }
    )


def deployment_context_sha256(context: Mapping[str, Any]) -> str:
    """Bind units, grid, renderer, and acquisition declarations exactly."""

    if not isinstance(context, Mapping) or not context:
        raise ValueError("deployment context must be a nonempty mapping")
    normalized = {str(key): value for key, value in context.items()}
    if any(not key.strip() for key in normalized):
        raise ValueError("deployment context keys must be nonempty")
    return _sha256_json({"deployment_context": normalized, "version": 1})


def _as_feature_matrix(
    features: Any,
    *,
    width: int | None = None,
    minimum_rows: int = 1,
) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < minimum_rows:
        raise ValueError(f"features must be a matrix with at least {minimum_rows} rows")
    if width is not None and values.shape[1] != width:
        raise ValueError("feature width does not match the frozen schema")
    if values.shape[1] < 1:
        raise ValueError("features must contain at least one column")
    if not np.all(np.isfinite(values)):
        raise ValueError("features must contain only finite values")
    return values


def _robust_center_scale(
    features: np.ndarray,
    *,
    scale_epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center = np.median(features, axis=0)
    mad_scale = 1.4826 * np.median(np.abs(features - center), axis=0)
    sample_scale = np.std(features, axis=0, ddof=1)
    absolute_floor = scale_epsilon * np.maximum(1.0, np.abs(center))
    exact_match = np.ptp(features, axis=0) <= absolute_floor
    scale = np.maximum(np.maximum(mad_scale, sample_scale), absolute_floor)
    scale = np.where(exact_match, 1.0, scale)
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise RuntimeError("robust feature scaling failed")
    return center, scale, exact_match, absolute_floor


def _tuple_vector(values: np.ndarray) -> tuple[float, ...]:
    return tuple(float(value) for value in np.asarray(values).reshape(-1))


def _tuple_matrix(values: np.ndarray) -> tuple[tuple[float, ...], ...]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("expected a matrix")
    return tuple(tuple(float(value) for value in row) for row in matrix)


@dataclass(frozen=True)
class SupportScoreModel:
    """Frozen transform used to calculate one observable support score."""

    method: str
    feature_names: tuple[str, ...]
    feature_schema_sha256: str
    deployment_context_sha256: str
    training_count: int
    center: tuple[float, ...]
    scale: tuple[float, ...]
    exact_match_mask: tuple[bool, ...]
    exact_match_tolerance: tuple[float, ...]
    lower_standardized: tuple[float, ...]
    upper_standardized: tuple[float, ...]
    precision: tuple[tuple[float, ...], ...]
    reference_standardized: tuple[tuple[float, ...], ...]
    knn_k: int
    covariance_shrinkage: float
    scale_epsilon: float
    model_contract_sha256: str


def _score_model_payload(model: SupportScoreModel) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "method": model.method,
        "feature_names": list(model.feature_names),
        "feature_schema_sha256": model.feature_schema_sha256,
        "deployment_context_sha256": model.deployment_context_sha256,
        "training_count": model.training_count,
        "center": list(model.center),
        "scale": list(model.scale),
        "exact_match_mask": list(model.exact_match_mask),
        "exact_match_tolerance": list(model.exact_match_tolerance),
        "lower_standardized": list(model.lower_standardized),
        "upper_standardized": list(model.upper_standardized),
        "precision": [list(row) for row in model.precision],
        "reference_standardized": [
            list(row) for row in model.reference_standardized
        ],
        "knn_k": model.knn_k,
        "covariance_shrinkage": model.covariance_shrinkage,
        "scale_epsilon": model.scale_epsilon,
    }


def score_model_contract_sha256(model: SupportScoreModel) -> str:
    return _sha256_json(_score_model_payload(model))


def validate_score_model(model: SupportScoreModel) -> None:
    """Fail closed if a serialized or replaced score model is inconsistent."""

    names = _validated_feature_names(model.feature_names)
    width = len(names)
    if model.method not in SUPPORT_METHODS:
        raise ValueError("support method is not recognized")
    if model.feature_schema_sha256 != feature_schema_sha256(names):
        raise ValueError("feature schema hash does not match the declared order")
    if len(model.deployment_context_sha256) != 64:
        raise ValueError("deployment context digest is malformed")
    if model.training_count < 2:
        raise ValueError("support model requires at least two training clusters")
    center = np.asarray(model.center, dtype=np.float64)
    scale = np.asarray(model.scale, dtype=np.float64)
    if center.shape != (width,) or scale.shape != (width,):
        raise ValueError("center and scale do not match the feature schema")
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(scale)):
        raise ValueError("center and scale must be finite")
    if np.any(scale <= 0.0):
        raise ValueError("feature scales must be positive")
    exact_match = np.asarray(model.exact_match_mask, dtype=np.bool_)
    exact_tolerance = np.asarray(model.exact_match_tolerance, dtype=np.float64)
    if exact_match.shape != (width,) or exact_tolerance.shape != (width,):
        raise ValueError("exact-match contract does not match the feature schema")
    if np.any(~np.isfinite(exact_tolerance)) or np.any(exact_tolerance <= 0.0):
        raise ValueError("exact-match tolerances must be finite and positive")
    if np.any(exact_match & (scale != 1.0)):
        raise ValueError("exact-match features must be removed from distance scaling")
    if not math.isfinite(model.scale_epsilon) or model.scale_epsilon <= 0.0:
        raise ValueError("scale_epsilon must be finite and positive")

    lower = np.asarray(model.lower_standardized, dtype=np.float64)
    upper = np.asarray(model.upper_standardized, dtype=np.float64)
    precision = np.asarray(model.precision, dtype=np.float64)
    reference = np.asarray(model.reference_standardized, dtype=np.float64)
    if model.method == "axis_envelope":
        if lower.shape != (width,) or upper.shape != (width,):
            raise ValueError("axis envelope must bind every feature")
        if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
            raise ValueError("axis envelope bounds must be finite")
        if np.any(lower > upper):
            raise ValueError("axis envelope lower bounds exceed upper bounds")
    elif lower.size or upper.size:
        raise ValueError("non-envelope models must not carry envelope bounds")

    if model.method == "shrinkage_mahalanobis":
        if precision.shape != (width, width):
            raise ValueError("Mahalanobis precision has the wrong shape")
        if not np.all(np.isfinite(precision)) or not np.allclose(
            precision, precision.T, rtol=1e-10, atol=1e-12
        ):
            raise ValueError("Mahalanobis precision must be finite and symmetric")
        if float(np.min(np.linalg.eigvalsh(precision))) <= 0.0:
            raise ValueError("Mahalanobis precision must be positive definite")
        if not 0.0 < model.covariance_shrinkage <= 1.0:
            raise ValueError("covariance_shrinkage must lie in (0,1]")
    elif precision.size:
        raise ValueError("only the Mahalanobis model may carry a precision matrix")

    if model.method == "knn":
        if reference.shape != (model.training_count, width):
            raise ValueError("kNN reference matrix does not match the contract")
        if not np.all(np.isfinite(reference)):
            raise ValueError("kNN reference features must be finite")
        if not 1 <= model.knn_k <= model.training_count:
            raise ValueError("knn_k exceeds the available reference clusters")
    elif reference.size or model.knn_k != 0:
        raise ValueError("only the kNN model may carry reference features")

    if model.model_contract_sha256 != score_model_contract_sha256(model):
        raise ValueError("support score model contract hash mismatch")


def fit_support_score_model(
    support_fit_features: Any,
    *,
    feature_names: tuple[str, ...],
    deployment_context: Mapping[str, Any],
    method: SupportMethod,
    knn_k: int = 3,
    covariance_shrinkage: float = 0.15,
    scale_epsilon: float = 1e-9,
) -> SupportScoreModel:
    """Fit a support score without using threshold-calibration clusters."""

    names = _validated_feature_names(feature_names)
    values = _as_feature_matrix(
        support_fit_features,
        width=len(names),
        minimum_rows=2,
    )
    if method not in SUPPORT_METHODS:
        raise ValueError("method is not one of the frozen support methods")
    if not math.isfinite(scale_epsilon) or scale_epsilon <= 0.0:
        raise ValueError("scale_epsilon must be finite and positive")
    center, scale, exact_match, exact_tolerance = _robust_center_scale(
        values,
        scale_epsilon=scale_epsilon,
    )
    standardized = (values - center) / scale

    lower = np.empty(0, dtype=np.float64)
    upper = np.empty(0, dtype=np.float64)
    precision = np.empty((0, 0), dtype=np.float64)
    reference = np.empty((0, 0), dtype=np.float64)
    frozen_k = 0
    frozen_shrinkage = 0.0
    if method == "axis_envelope":
        lower = np.min(standardized, axis=0)
        upper = np.max(standardized, axis=0)
    elif method == "shrinkage_mahalanobis":
        if not math.isfinite(covariance_shrinkage) or not (
            0.0 < covariance_shrinkage <= 1.0
        ):
            raise ValueError("covariance_shrinkage must lie in (0,1]")
        covariance = np.cov(standardized, rowvar=False, ddof=1)
        covariance = np.atleast_2d(covariance).astype(np.float64)
        identity = np.eye(len(names), dtype=np.float64)
        covariance = (
            (1.0 - covariance_shrinkage) * covariance
            + covariance_shrinkage * identity
        )
        precision = np.linalg.inv(covariance)
        frozen_shrinkage = float(covariance_shrinkage)
    elif method == "knn":
        frozen_k = int(knn_k)
        if not 1 <= frozen_k <= len(values):
            raise ValueError("knn_k must lie between one and the fit count")
        reference = standardized

    provisional = SupportScoreModel(
        method=method,
        feature_names=names,
        feature_schema_sha256=feature_schema_sha256(names),
        deployment_context_sha256=deployment_context_sha256(deployment_context),
        training_count=int(values.shape[0]),
        center=_tuple_vector(center),
        scale=_tuple_vector(scale),
        exact_match_mask=tuple(bool(value) for value in exact_match),
        exact_match_tolerance=_tuple_vector(exact_tolerance),
        lower_standardized=_tuple_vector(lower),
        upper_standardized=_tuple_vector(upper),
        precision=_tuple_matrix(precision),
        reference_standardized=_tuple_matrix(reference),
        knn_k=frozen_k,
        covariance_shrinkage=frozen_shrinkage,
        scale_epsilon=float(scale_epsilon),
        model_contract_sha256="",
    )
    model = replace(
        provisional,
        model_contract_sha256=score_model_contract_sha256(provisional),
    )
    validate_score_model(model)
    return model


def _standardized_features(model: SupportScoreModel, features: Any) -> np.ndarray:
    values = _as_feature_matrix(features, width=len(model.feature_names))
    return (values - np.asarray(model.center)) / np.asarray(model.scale)


def _exact_match_passes(model: SupportScoreModel, features: Any) -> np.ndarray:
    values = _as_feature_matrix(features, width=len(model.feature_names))
    mask = np.asarray(model.exact_match_mask, dtype=np.bool_)
    if not np.any(mask):
        return np.ones(len(values), dtype=np.bool_)
    center = np.asarray(model.center)[mask]
    tolerance = np.asarray(model.exact_match_tolerance)[mask]
    return np.all(np.abs(values[:, mask] - center[None]) <= tolerance[None], axis=1)


def _knn_query_scores(
    query: np.ndarray,
    reference: np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    difference = query[:, None, :] - reference[None, :, :]
    distances = np.linalg.norm(difference, axis=2) / math.sqrt(query.shape[1])
    return np.partition(distances, k - 1, axis=1)[:, k - 1]


def knn_leave_one_out_scores(features: Any, *, k: int) -> np.ndarray:
    """Return kNN reference scores while explicitly excluding each row itself."""

    values = _as_feature_matrix(features, minimum_rows=2)
    if not 1 <= int(k) < len(values):
        raise ValueError("leave-one-out k must be smaller than the row count")
    difference = values[:, None, :] - values[None, :, :]
    distances = np.linalg.norm(difference, axis=2) / math.sqrt(values.shape[1])
    np.fill_diagonal(distances, np.inf)
    return np.partition(distances, int(k) - 1, axis=1)[:, int(k) - 1]


def score_support_features(model: SupportScoreModel, features: Any) -> np.ndarray:
    """Calculate truth-free nonconformity scores under a frozen score model."""

    validate_score_model(model)
    standardized = _standardized_features(model, features)
    if model.method == "axis_envelope":
        lower = np.asarray(model.lower_standardized)
        upper = np.asarray(model.upper_standardized)
        scores = np.max(
            np.maximum(lower[None] - standardized, standardized - upper[None]),
            axis=1,
        )
    elif model.method == "robust_diagonal":
        scores = np.linalg.norm(standardized, axis=1) / math.sqrt(
            standardized.shape[1]
        )
    elif model.method == "shrinkage_mahalanobis":
        precision = np.asarray(model.precision)
        squared = np.einsum("bi,ij,bj->b", standardized, precision, standardized)
        scores = np.sqrt(np.maximum(squared, 0.0) / standardized.shape[1])
    elif model.method == "knn":
        scores = _knn_query_scores(
            standardized,
            np.asarray(model.reference_standardized),
            k=model.knn_k,
        )
    else:  # pragma: no cover - validate_score_model already rejects this.
        raise RuntimeError("unreachable support method")
    if not np.all(np.isfinite(scores)):
        raise RuntimeError("support scoring produced a non-finite value")
    return scores


def conformal_upper_quantile(
    calibration_scores: Any,
    *,
    alpha: float,
) -> tuple[float, int]:
    """Finite-sample split-conformal upper quantile and one-based rank."""

    scores = np.asarray(calibration_scores, dtype=np.float64).reshape(-1)
    if scores.size < 1 or not np.all(np.isfinite(scores)):
        raise ValueError("calibration scores must be a nonempty finite vector")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    rank = int(math.ceil((scores.size + 1) * (1.0 - alpha)))
    if rank > scores.size:
        raise ValueError(
            "calibration split is too small for the requested finite-sample alpha"
        )
    threshold = float(np.partition(scores, rank - 1)[rank - 1])
    return threshold, rank


@dataclass(frozen=True)
class FrozenSupportGate:
    """Score model plus a threshold calibrated on disjoint in-support clusters."""

    score_model: SupportScoreModel
    alpha: float
    calibration_count: int
    quantile_rank: int
    threshold: float
    calibration_scores_sha256: str
    policy_label: str
    policy_contract_sha256: str


def _policy_payload(policy: FrozenSupportGate) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "score_model_contract_sha256": policy.score_model.model_contract_sha256,
        "alpha": policy.alpha,
        "calibration_count": policy.calibration_count,
        "quantile_rank": policy.quantile_rank,
        "threshold": policy.threshold,
        "calibration_scores_sha256": policy.calibration_scores_sha256,
        "policy_label": policy.policy_label,
        "acceptance_rule": "score <= threshold",
        "deployment_inputs": ["observable_features", "frozen_policy"],
    }


def support_policy_contract_sha256(policy: FrozenSupportGate) -> str:
    return _sha256_json(_policy_payload(policy))


def validate_support_gate(policy: FrozenSupportGate) -> None:
    validate_score_model(policy.score_model)
    if not math.isfinite(policy.alpha) or not 0.0 < policy.alpha < 1.0:
        raise ValueError("policy alpha must lie strictly between zero and one")
    expected_rank = int(
        math.ceil((policy.calibration_count + 1) * (1.0 - policy.alpha))
    )
    if policy.calibration_count < 1 or policy.quantile_rank != expected_rank:
        raise ValueError("policy conformal rank is inconsistent")
    if policy.quantile_rank > policy.calibration_count:
        raise ValueError("policy calibration count cannot support its alpha")
    if not math.isfinite(policy.threshold):
        raise ValueError("policy threshold must be finite")
    if len(policy.calibration_scores_sha256) != 64:
        raise ValueError("policy calibration digest is malformed")
    if not policy.policy_label.strip():
        raise ValueError("policy_label must be nonempty")
    if policy.policy_contract_sha256 != support_policy_contract_sha256(policy):
        raise ValueError("support gate policy contract hash mismatch")


def calibrate_support_gate(
    score_model: SupportScoreModel,
    support_calibration_features: Any,
    *,
    alpha: float,
    policy_label: str,
) -> FrozenSupportGate:
    """Calibrate a score threshold without re-fitting the score model."""

    validate_score_model(score_model)
    calibration = _as_feature_matrix(
        support_calibration_features,
        width=len(score_model.feature_names),
    )
    if not np.all(_exact_match_passes(score_model, calibration)):
        raise ValueError(
            "support calibration violates a fit-time exact-match feature contract"
        )
    scores = score_support_features(score_model, calibration)
    threshold, rank = conformal_upper_quantile(scores, alpha=alpha)
    provisional = FrozenSupportGate(
        score_model=score_model,
        alpha=float(alpha),
        calibration_count=int(len(scores)),
        quantile_rank=rank,
        threshold=threshold,
        calibration_scores_sha256=_sha256_json([float(value) for value in scores]),
        policy_label=str(policy_label),
        policy_contract_sha256="",
    )
    policy = replace(
        provisional,
        policy_contract_sha256=support_policy_contract_sha256(provisional),
    )
    validate_support_gate(policy)
    return policy


@dataclass(frozen=True)
class SupportDecision:
    row_index: int
    score: float
    threshold: float
    accepted: bool
    reason: str
    feature_schema_sha256: str
    policy_contract_sha256: str
    uses_truth: bool = False
    uses_target: bool = False
    uses_reconstruction_error: bool = False


def decide_support(
    policy: FrozenSupportGate,
    observable_features: Any,
    *,
    feature_names: tuple[str, ...],
    deployment_context: Mapping[str, Any],
) -> tuple[SupportDecision, ...]:
    """Apply a frozen gate using observable features only."""

    validate_support_gate(policy)
    declared_names = _validated_feature_names(feature_names)
    if declared_names != policy.score_model.feature_names:
        raise ValueError("deployment feature order does not match the frozen schema")
    if (
        deployment_context_sha256(deployment_context)
        != policy.score_model.deployment_context_sha256
    ):
        raise ValueError("deployment context does not match the frozen contract")
    scores = score_support_features(policy.score_model, observable_features)
    exact_passes = _exact_match_passes(policy.score_model, observable_features)
    return tuple(
        SupportDecision(
            row_index=index,
            score=float(score),
            threshold=policy.threshold,
            accepted=bool(exact_passes[index] and score <= policy.threshold),
            reason=(
                "EXACT_MATCH_CONTRACT_FAILED_USE_FALLBACK"
                if not exact_passes[index]
                else "INSIDE_CALIBRATED_SUPPORT"
                if score <= policy.threshold
                else "OUTSIDE_CALIBRATED_SUPPORT_USE_FALLBACK"
            ),
            feature_schema_sha256=policy.score_model.feature_schema_sha256,
            policy_contract_sha256=policy.policy_contract_sha256,
        )
        for index, score in enumerate(scores)
    )


def deployment_signature_audit() -> dict[str, Any]:
    """Machine-check that deployment exposes no obvious oracle argument."""

    parameter_names = tuple(inspect.signature(decide_support).parameters)
    forbidden = {
        "truth",
        "target",
        "reference_field",
        "field_error",
        "reconstruction_error",
        "harm_label",
    }
    return {
        "parameter_names": list(parameter_names),
        "forbidden_parameter_names": sorted(forbidden),
        "forbidden_parameters_present": sorted(forbidden.intersection(parameter_names)),
        "passed": not forbidden.intersection(parameter_names),
    }
