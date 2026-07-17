"""Observable-only risk fallback for the certified grouped-majorizer smoke.

This module is development-only synthetic infrastructure.  Offline stages may
use reconstruction errors to fit and calibrate a small rule, but deployment
selection accepts only ``DeploymentGeometry`` plus frozen model/calibration
objects.  Every selected partition remains subject to the deterministic
grouped-majorizer certificate implemented by ``certified_grouped_majorizer``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Any

import torch

from demo_t16_operator.certified_grouped_majorizer import (
    DeploymentGeometry,
    TinyPrimitiveRig,
    deployment_geometry_from_rig,
    generate_tiny_primitive_rigs,
)


SCHEMA_VERSION = "observable-risk-fallback-interface-1.4"
STATUS = "DEVELOPMENT_ONLY_SYNTHETIC_RCCF_INTERFACE_GATE"
SPLIT_ROLES = (
    "train",
    "model_selection",
    "risk_calibration",
    "fresh_geometry_ood",
)
FALLBACK_PARTITION = "paired_cross"
ALLOWED_CANDIDATES = (
    "singleton_factor",
    "paired_local",
    "paired_cross",
    "triad_bridge",
)
FORBIDDEN_ORACLE = "all_in_one_exact"
FIELD_ENDPOINT = "field_relative_l2"
RESIDUAL_ENDPOINT = "normalized_residual_l2"
JOINT_HARM_ENDPOINTS = (FIELD_ENDPOINT, RESIDUAL_ENDPOINT)
MULTIPLICITY_CORRECTION = "BONFERRONI_FROZEN_FINITE_GRID"
FEATURE_NAMES = (
    "regime",
    "angle",
    "aperture",
    "shear",
    "frequency",
    "cancellation",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def feature_schema_sha256() -> str:
    payload = {
        "dtype": "torch.float64",
        "feature_names": list(FEATURE_NAMES),
        "source": "DeploymentGeometry.geometry_features",
        "version": 1,
    }
    return hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()


@dataclass(frozen=True)
class ObservableRiskRule:
    """Frozen depth-one rule and leaf risk scores."""

    feature_index: int
    threshold: float
    left_partition: str
    right_partition: str
    left_risk_score: float
    right_risk_score: float
    fallback_partition: str
    allowed_partitions: tuple[str, ...]
    forbidden_oracle: str
    feature_schema_sha256: str
    support_minimums: tuple[float, ...]
    support_maximums: tuple[float, ...]


@dataclass(frozen=True)
class RiskCalibration:
    """Frozen threshold selected only on the risk-calibration split."""

    acceptance_threshold: float
    confidence_alpha: float
    coverage_confidence_alpha: float
    multiplicity_correction: str
    threshold_candidate_count: int
    threshold_grid: tuple[float, ...]
    rule_contract_sha256: str
    policy_contract_sha256: str
    joint_harm_endpoints: tuple[str, ...]
    harm_tolerances: tuple[tuple[str, float], ...]
    corrected_risk_alpha: float
    corrected_coverage_alpha: float
    maximum_risk_upper: float
    minimum_takeover_coverage: float
    accepted_count: int
    failure_count: int
    risk_upper_bound: float
    calibration_count: int
    takeover_coverage: float
    takeover_coverage_lower_bound: float
    authorized_takeover_coverage: float
    authorized_takeover_coverage_lower_bound: float
    development_gate_passed: bool
    feature_schema_sha256: str


@dataclass(frozen=True)
class SelectionDecision:
    rig_id: str
    candidate_partition: str
    selected_partition: str
    fallback_partition: str
    fallback_used: bool
    fallback_reason: str
    risk_score: float
    acceptance_threshold: float
    risk_upper_bound: float
    support_gate_passed: bool
    observable_feature_sha256: str
    observable_feature_schema_sha256: str
    uses_truth: bool = False
    uses_target: bool = False
    uses_primitives: bool = False
    uses_signed_matrix: bool = False
    uses_exact_abs_operator: bool = False
    uses_solver_trajectory: bool = False


def frozen_threshold_grid(rule: ObservableRiskRule) -> tuple[float, ...]:
    """Return the finite threshold grid fixed by the trained rule."""

    return tuple(sorted({-1.0, rule.left_risk_score, rule.right_risk_score}))


def rule_contract_sha256(rule: ObservableRiskRule) -> str:
    """Bind calibration to the exact rule, schema, catalogue, and support."""

    payload = {
        "feature_index": rule.feature_index,
        "threshold": rule.threshold,
        "left_partition": rule.left_partition,
        "right_partition": rule.right_partition,
        "left_risk_score": rule.left_risk_score,
        "right_risk_score": rule.right_risk_score,
        "fallback_partition": rule.fallback_partition,
        "allowed_partitions": list(rule.allowed_partitions),
        "forbidden_oracle": rule.forbidden_oracle,
        "feature_schema_sha256": rule.feature_schema_sha256,
        "support_minimums": list(rule.support_minimums),
        "support_maximums": list(rule.support_maximums),
    }
    return hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()


def _normalized_harm_tolerances(
    harm_tolerances: Mapping[str, float],
) -> tuple[tuple[str, float], ...]:
    if set(harm_tolerances) != set(JOINT_HARM_ENDPOINTS):
        raise ValueError("harm tolerances must cover the frozen joint endpoints exactly")
    normalized = tuple(
        (endpoint, float(harm_tolerances[endpoint]))
        for endpoint in JOINT_HARM_ENDPOINTS
    )
    if any(not math.isfinite(value) or value < 0.0 for _, value in normalized):
        raise ValueError("harm tolerances must be finite and nonnegative")
    return normalized


def calibration_policy_contract_sha256(
    rule: ObservableRiskRule,
    *,
    harm_tolerances: Mapping[str, float],
    confidence_alpha: float,
    coverage_confidence_alpha: float,
    maximum_risk_upper: float,
    minimum_takeover_coverage: float,
) -> str:
    """Bind every preregistered calibration choice to an expected policy."""

    tolerances = _normalized_harm_tolerances(harm_tolerances)
    thresholds = frozen_threshold_grid(rule)
    payload = {
        "version": 1,
        "rule_contract_sha256": rule_contract_sha256(rule),
        "multiplicity_correction": MULTIPLICITY_CORRECTION,
        "threshold_candidate_count": len(thresholds),
        "threshold_grid": list(thresholds),
        "confidence_alpha": float(confidence_alpha),
        "coverage_confidence_alpha": float(coverage_confidence_alpha),
        "maximum_risk_upper": float(maximum_risk_upper),
        "minimum_takeover_coverage": float(minimum_takeover_coverage),
        "joint_harm_endpoints": list(JOINT_HARM_ENDPOINTS),
        "harm_tolerances": {name: value for name, value in tolerances},
    }
    return hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()


def generate_four_way_rigs(
    *,
    split_assignments: Mapping[str, str],
    geometry_seed: int,
    noise_seed: int,
    row_count: int,
    column_count: int,
) -> list[TinyPrimitiveRig]:
    """Reuse the frozen v3 generator while preserving four explicit roles."""

    if not split_assignments or set(split_assignments.values()).difference(SPLIT_ROLES):
        raise ValueError("four-way assignments contain an unknown split role")
    rigs: list[TinyPrimitiveRig] = []
    for rig_id, role in sorted(split_assignments.items()):
        generator_role = (
            "fresh_geometry_ood"
            if role == "fresh_geometry_ood"
            else "train"
            if role == "train"
            else "safety_calibration"
        )
        generator_id = f"{role}::{rig_id}"
        generated = generate_tiny_primitive_rigs(
            split_assignments={generator_id: generator_role},
            geometry_seed=int(geometry_seed),
            noise_seed=int(noise_seed),
            row_count=int(row_count),
            column_count=int(column_count),
            dtype=torch.float64,
        )
        rigs.append(replace(generated[0], rig_id=rig_id, split_role=role))
    return rigs


def split_rigs_four_way(
    rigs: Sequence[TinyPrimitiveRig],
) -> tuple[dict[str, list[TinyPrimitiveRig]], dict[str, Any]]:
    if not rigs or len({rig.rig_id for rig in rigs}) != len(rigs):
        raise ValueError("rigs must be nonempty complete units with unique ids")
    grouped = {role: [rig for rig in rigs if rig.split_role == role] for role in SPLIT_ROLES}
    if any(not grouped[role] for role in SPLIT_ROLES):
        raise ValueError("all four split roles require at least one complete rig")
    if sum(len(values) for values in grouped.values()) != len(rigs):
        raise ValueError("rig contains an unknown split role")
    return grouped, {
        "split_unit": "COMPLETE_RIG",
        "random_ray_or_pixel_split_used": False,
        "roles_are_disjoint": True,
        "role_rig_ids": {
            role: [rig.rig_id for rig in grouped[role]] for role in SPLIT_ROLES
        },
    }


def audit_operator_decomposition(
    primitives: torch.Tensor, signed_matrix: torch.Tensor
) -> dict[str, float | int | bool]:
    """Verify the actual solver operator equals the certified primitive sum."""

    if primitives.ndim != 3 or signed_matrix.ndim != 2:
        raise ValueError("primitive and signed operator dimensions are invalid")
    if tuple(primitives.shape[1:]) != tuple(signed_matrix.shape):
        raise ValueError("primitive and signed operator shapes differ")
    if not bool(torch.all(torch.isfinite(primitives))) or not bool(
        torch.all(torch.isfinite(signed_matrix))
    ):
        raise ValueError("primitive decomposition contains non-finite values")
    rebuilt = torch.sum(primitives, dim=0)
    difference = torch.abs(rebuilt - signed_matrix)
    scale = torch.maximum(torch.ones_like(signed_matrix), torch.abs(signed_matrix))
    tolerance = 256.0 * torch.finfo(signed_matrix.dtype).eps * scale
    mismatch_count = int(torch.count_nonzero(difference > tolerance))
    return {
        "operator_decomposition_mismatch_count": mismatch_count,
        "operator_decomposition_max_abs_error": float(torch.amax(difference)),
        "operator_decomposition_verified": mismatch_count == 0,
    }


def _validate_candidate_catalogue(candidate_names: Sequence[str]) -> tuple[str, ...]:
    candidates = tuple(candidate_names)
    if not candidates or len(candidates) != len(set(candidates)):
        raise ValueError("candidate partitions must be nonempty and unique")
    if candidates != ALLOWED_CANDIDATES:
        raise ValueError("candidate partitions differ from the frozen catalogue")
    if FORBIDDEN_ORACLE in candidates:
        raise ValueError("all-in-one exact oracle cannot enter the selector")
    if FALLBACK_PARTITION not in candidates:
        raise ValueError("paired_cross fallback must remain available")
    return candidates


def _endpoint_score(
    score_table: Mapping[tuple[str, str], Mapping[str, float]],
    rig_id: str,
    partition: str,
    endpoint: str,
) -> float:
    if endpoint not in JOINT_HARM_ENDPOINTS:
        raise ValueError("unknown RCCF harm endpoint")
    value = float(score_table[(rig_id, partition)][endpoint])
    if not math.isfinite(value):
        raise ValueError("offline score must be finite")
    return value


def _joint_harm_record(
    score_table: Mapping[tuple[str, str], Mapping[str, float]],
    rig_id: str,
    partition: str,
    *,
    harm_tolerances: Mapping[str, float],
) -> dict[str, float | bool]:
    if set(harm_tolerances) != set(JOINT_HARM_ENDPOINTS):
        raise ValueError("joint harm tolerances differ from frozen endpoints")
    harms = {
        endpoint: _endpoint_score(score_table, rig_id, partition, endpoint)
        - _endpoint_score(score_table, rig_id, FALLBACK_PARTITION, endpoint)
        for endpoint in JOINT_HARM_ENDPOINTS
    }
    if any(
        not math.isfinite(float(harm_tolerances[endpoint]))
        or float(harm_tolerances[endpoint]) < 0.0
        for endpoint in JOINT_HARM_ENDPOINTS
    ):
        raise ValueError("joint harm tolerances must be finite and nonnegative")
    return {
        "field_harm": harms[FIELD_ENDPOINT],
        "residual_harm": harms[RESIDUAL_ENDPOINT],
        "harm_failure": any(
            harms[endpoint] > float(harm_tolerances[endpoint])
            for endpoint in JOINT_HARM_ENDPOINTS
        ),
    }


def _partition_for_rule(
    feature_value: float, *, threshold: float, left: str, right: str
) -> str:
    return left if feature_value <= threshold else right


def _leaf_risk(
    rigs: Sequence[TinyPrimitiveRig],
    score_table: Mapping[tuple[str, str], Mapping[str, float]],
    *,
    feature_index: int,
    threshold: float,
    left_partition: str,
    right_partition: str,
    left_leaf: bool,
    harm_tolerances: Mapping[str, float],
) -> float:
    labels: list[bool] = []
    for rig in rigs:
        is_left = float(rig.geometry_features[feature_index]) <= threshold
        if is_left != left_leaf:
            continue
        partition = left_partition if is_left else right_partition
        labels.append(
            bool(
                _joint_harm_record(
                    score_table,
                    rig.rig_id,
                    partition,
                    harm_tolerances=harm_tolerances,
                )["harm_failure"]
            )
        )
    if not labels:
        raise ValueError("every frozen risk leaf requires train support")
    return (sum(labels) + 1.0) / (len(labels) + 2.0)


def fit_observable_risk_rule(
    train_rigs: Sequence[TinyPrimitiveRig],
    model_selection_rigs: Sequence[TinyPrimitiveRig],
    score_table: Mapping[tuple[str, str], Mapping[str, float]],
    *,
    candidate_names: Sequence[str],
    train_top_k: int,
    harm_tolerances: Mapping[str, float],
    failure_penalty: float,
) -> tuple[ObservableRiskRule, dict[str, Any]]:
    """Fit candidates on train and choose one only on model-selection rigs."""

    candidates = _validate_candidate_catalogue(candidate_names)
    if not train_rigs or not model_selection_rigs:
        raise ValueError("train and model-selection splits must be nonempty")
    if train_top_k < 1:
        raise ValueError("train_top_k must be positive")
    if failure_penalty < 0.0:
        raise ValueError("failure penalty must be nonnegative")
    feature_count = len(FEATURE_NAMES)
    if any(int(rig.geometry_features.numel()) != feature_count for rig in (*train_rigs, *model_selection_rigs)):
        raise ValueError("observable geometry feature count differs")

    rule_candidates: list[tuple[float, int, float, str, str]] = []
    for feature_index in range(feature_count):
        values = sorted(float(rig.geometry_features[feature_index]) for rig in train_rigs)
        thresholds = [0.5 * (a + b) for a, b in zip(values, values[1:]) if a < b]
        for threshold in thresholds:
            for left in candidates:
                for right in candidates:
                    values_for_rule = [
                        _endpoint_score(
                            score_table,
                            rig.rig_id,
                            _partition_for_rule(
                                float(rig.geometry_features[feature_index]),
                                threshold=threshold,
                                left=left,
                                right=right,
                            ),
                            FIELD_ENDPOINT,
                        )
                        for rig in train_rigs
                    ]
                    rule_candidates.append(
                        (sum(values_for_rule) / len(values_for_rule), feature_index, threshold, left, right)
                    )
    if not rule_candidates:
        raise ValueError("train split cannot form a depth-one rule")
    rule_candidates.sort(key=lambda item: (item[0], item[3] == item[4], item[1], item[2], item[3], item[4]))
    finalists = rule_candidates[: min(train_top_k, len(rule_candidates))]

    def model_selection_objective(item: tuple[float, int, float, str, str]) -> tuple[float, float, tuple[Any, ...]]:
        _, feature_index, threshold, left, right = item
        field_harms: list[float] = []
        failures: list[bool] = []
        for rig in model_selection_rigs:
            partition = _partition_for_rule(
                float(rig.geometry_features[feature_index]),
                threshold=threshold,
                left=left,
                right=right,
            )
            record = _joint_harm_record(
                score_table,
                rig.rig_id,
                partition,
                harm_tolerances=harm_tolerances,
            )
            field_harms.append(float(record["field_harm"]))
            failures.append(bool(record["harm_failure"]))
        failure_rate = sum(failures) / len(failures)
        objective = (
            sum(field_harms) / len(field_harms)
            + failure_penalty * failure_rate
        )
        return objective, failure_rate, (feature_index, threshold, left, right)

    selected = min(finalists, key=model_selection_objective)
    train_score, feature_index, threshold, left, right = selected
    selection_objective, selection_failure_rate, _ = model_selection_objective(selected)
    support_rigs = (*train_rigs, *model_selection_rigs)
    support_minimums = tuple(
        min(float(rig.geometry_features[index]) for rig in support_rigs)
        for index in range(feature_count)
    )
    support_maximums = tuple(
        max(float(rig.geometry_features[index]) for rig in support_rigs)
        for index in range(feature_count)
    )
    rule = ObservableRiskRule(
        feature_index=feature_index,
        threshold=threshold,
        left_partition=left,
        right_partition=right,
        left_risk_score=_leaf_risk(
            train_rigs,
            score_table,
            feature_index=feature_index,
            threshold=threshold,
            left_partition=left,
            right_partition=right,
            left_leaf=True,
            harm_tolerances=harm_tolerances,
        ),
        right_risk_score=_leaf_risk(
            train_rigs,
            score_table,
            feature_index=feature_index,
            threshold=threshold,
            left_partition=left,
            right_partition=right,
            left_leaf=False,
            harm_tolerances=harm_tolerances,
        ),
        fallback_partition=FALLBACK_PARTITION,
        allowed_partitions=candidates,
        forbidden_oracle=FORBIDDEN_ORACLE,
        feature_schema_sha256=feature_schema_sha256(),
        support_minimums=support_minimums,
        support_maximums=support_maximums,
    )
    return rule, {
        "model_class": "DEPTH_ONE_OBSERVABLE_GEOMETRY_RISK_RULE",
        "feature_index": feature_index,
        "feature_name": FEATURE_NAMES[feature_index],
        "threshold": threshold,
        "left_partition": left,
        "right_partition": right,
        "left_risk_score": rule.left_risk_score,
        "right_risk_score": rule.right_risk_score,
        "train_mean_field_score": train_score,
        "model_selection_objective": selection_objective,
        "model_selection_failure_rate": selection_failure_rate,
        "train_candidate_count": len(rule_candidates),
        "model_selection_finalist_count": len(finalists),
        "feature_schema_sha256": rule.feature_schema_sha256,
        "support_source": "TRAIN_PLUS_MODEL_SELECTION_FEATURE_ENVELOPE",
        "support_minimums": list(rule.support_minimums),
        "support_maximums": list(rule.support_maximums),
        "joint_harm_endpoints": list(JOINT_HARM_ENDPOINTS),
        "truth_used_offline_for_harm_labels": True,
        "fresh_truth_used_for_selection": False,
    }


def _binomial_cdf(k: int, n: int, probability: float) -> float:
    return sum(
        math.comb(n, index)
        * probability**index
        * (1.0 - probability) ** (n - index)
        for index in range(k + 1)
    )


def clopper_pearson_upper(failures: int, count: int, alpha: float) -> float:
    """Return the deterministic one-sided ``1-alpha`` binomial upper bound."""

    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    if count < 0 or failures < 0 or failures > count:
        raise ValueError("binomial counts are invalid")
    if count == 0 or failures == count:
        return 1.0
    lower, upper = 0.0, 1.0
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        if _binomial_cdf(failures, count, midpoint) > alpha:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def clopper_pearson_lower(successes: int, count: int, alpha: float) -> float:
    """Return the deterministic one-sided ``1-alpha`` binomial lower bound."""

    if count < 0 or successes < 0 or successes > count:
        raise ValueError("binomial counts are invalid")
    return 1.0 - clopper_pearson_upper(count - successes, count, alpha)


def selection_conditional_harm_rate(
    failure_count: int, takeover_count: int
) -> float | None:
    """Return harm among actual takeovers, never diluted by fallback cases."""

    if takeover_count < 0 or failure_count < 0 or failure_count > takeover_count:
        raise ValueError("selection-conditional counts are invalid")
    return failure_count / takeover_count if takeover_count else None


def _rule_prediction(
    rule: ObservableRiskRule, deployment: DeploymentGeometry
) -> tuple[str, float]:
    if not isinstance(deployment, DeploymentGeometry):
        raise TypeError("risk inference requires DeploymentGeometry")
    if rule.feature_schema_sha256 != feature_schema_sha256():
        raise ValueError("observable feature schema hash mismatch")
    _validate_candidate_catalogue(rule.allowed_partitions)
    if rule.fallback_partition != FALLBACK_PARTITION:
        raise ValueError("fallback partition must remain paired_cross")
    if rule.forbidden_oracle != FORBIDDEN_ORACLE:
        raise ValueError("forbidden oracle contract differs")
    if not 0 <= rule.feature_index < len(FEATURE_NAMES):
        raise ValueError("rule feature index is invalid")
    if rule.left_partition not in rule.allowed_partitions or rule.right_partition not in rule.allowed_partitions:
        raise ValueError("rule selected a partition outside the frozen catalogue")
    if FORBIDDEN_ORACLE in {rule.left_partition, rule.right_partition}:
        raise ValueError("all-in-one exact oracle cannot enter the frozen rule")
    if not math.isfinite(rule.threshold):
        raise ValueError("rule threshold must be finite")
    if not 0.0 <= rule.left_risk_score <= 1.0 or not 0.0 <= rule.right_risk_score <= 1.0:
        raise ValueError("rule risk scores must lie in [0,1]")
    if int(deployment.geometry_features.numel()) != len(FEATURE_NAMES):
        raise ValueError("deployment geometry feature count differs")
    if (
        len(rule.support_minimums) != len(FEATURE_NAMES)
        or len(rule.support_maximums) != len(FEATURE_NAMES)
        or any(
            not math.isfinite(lower)
            or not math.isfinite(upper)
            or lower > upper
            for lower, upper in zip(rule.support_minimums, rule.support_maximums)
        )
    ):
        raise ValueError("observable support envelope is invalid")
    value = float(deployment.geometry_features[rule.feature_index])
    if not math.isfinite(value):
        raise ValueError("deployment geometry contains non-finite values")
    if value <= rule.threshold:
        return rule.left_partition, rule.left_risk_score
    return rule.right_partition, rule.right_risk_score


def _inside_observable_support(
    rule: ObservableRiskRule, deployment: DeploymentGeometry
) -> bool:
    if int(deployment.geometry_features.numel()) != len(FEATURE_NAMES):
        return False
    values = tuple(float(value) for value in deployment.geometry_features)
    if any(not math.isfinite(value) for value in values):
        return False
    return all(
        lower <= value <= upper
        for value, lower, upper in zip(
            values, rule.support_minimums, rule.support_maximums
        )
    )


def calibrate_acceptance_threshold(
    rule: ObservableRiskRule,
    risk_calibration_rigs: Sequence[TinyPrimitiveRig],
    score_table: Mapping[tuple[str, str], Mapping[str, float]],
    *,
    harm_tolerances: Mapping[str, float],
    confidence_alpha: float,
    coverage_confidence_alpha: float,
    maximum_risk_upper: float,
    minimum_takeover_coverage: float,
) -> tuple[RiskCalibration, list[dict[str, Any]]]:
    """Freeze only the acceptance threshold on the risk-calibration split."""

    if not risk_calibration_rigs:
        raise ValueError("risk-calibration split must be nonempty")
    if not 0.0 < confidence_alpha < 1.0 or not 0.0 < coverage_confidence_alpha < 1.0:
        raise ValueError("risk and coverage alpha must lie in (0,1)")
    if not 0.0 <= maximum_risk_upper <= 1.0:
        raise ValueError("maximum risk upper must lie in [0,1]")
    if not 0.0 <= minimum_takeover_coverage <= 1.0:
        raise ValueError("minimum takeover coverage must lie in [0,1]")
    normalized_harm_tolerances = _normalized_harm_tolerances(harm_tolerances)
    frozen_harm_tolerances = dict(normalized_harm_tolerances)
    records: list[dict[str, Any]] = []
    for rig in risk_calibration_rigs:
        deployment = deployment_geometry_from_rig(rig)
        candidate, risk_score = _rule_prediction(rule, deployment)
        harm = _joint_harm_record(
            score_table,
            rig.rig_id,
            candidate,
            harm_tolerances=frozen_harm_tolerances,
        )
        records.append(
            {
                "rig_id": rig.rig_id,
                "candidate_partition": candidate,
                "risk_score": risk_score,
                "support_gate_passed": _inside_observable_support(rule, deployment),
                "observed_field_harm_vs_fallback": harm["field_harm"],
                "observed_residual_harm_vs_fallback": harm["residual_harm"],
                "harm_failure": harm["harm_failure"],
            }
        )
    # This finite grid is determined by the frozen train/model-selection rule,
    # not by which leaves happen to appear in risk calibration.
    thresholds = frozen_threshold_grid(rule)
    threshold_count = len(thresholds)
    corrected_risk_alpha = confidence_alpha / threshold_count
    corrected_coverage_alpha = coverage_confidence_alpha / threshold_count
    choices: list[tuple[int, float, float, float, int]] = []
    evaluated: list[tuple[int, float, float, float, int]] = []
    for threshold in thresholds:
        accepted = [
            row
            for row in records
            if row["support_gate_passed"]
            and row["candidate_partition"] != FALLBACK_PARTITION
            and float(row["risk_score"]) <= threshold
        ]
        failures = sum(bool(row["harm_failure"]) for row in accepted)
        upper = clopper_pearson_upper(
            failures, len(accepted), corrected_risk_alpha
        )
        coverage_lower = clopper_pearson_lower(
            len(accepted), len(records), corrected_coverage_alpha
        )
        if accepted:
            evaluated.append(
                (len(accepted), -upper, coverage_lower, threshold, failures)
            )
        if upper <= maximum_risk_upper and coverage_lower >= minimum_takeover_coverage:
            choices.append(
                (len(accepted), -upper, coverage_lower, threshold, failures)
            )
    if choices:
        accepted_count, neg_upper, coverage_lower, threshold, failures = max(choices)
        risk_upper = -neg_upper
        gate = True
    elif evaluated:
        accepted_count, neg_upper, coverage_lower, threshold, failures = max(evaluated)
        risk_upper = -neg_upper
        gate = False
    else:
        threshold = -1.0
        failures = accepted_count = 0
        risk_upper = 1.0
        coverage_lower = 0.0
        gate = False
    calibration = RiskCalibration(
        acceptance_threshold=float(threshold),
        confidence_alpha=float(confidence_alpha),
        coverage_confidence_alpha=float(coverage_confidence_alpha),
        multiplicity_correction=MULTIPLICITY_CORRECTION,
        threshold_candidate_count=threshold_count,
        threshold_grid=thresholds,
        rule_contract_sha256=rule_contract_sha256(rule),
        policy_contract_sha256=calibration_policy_contract_sha256(
            rule,
            harm_tolerances=frozen_harm_tolerances,
            confidence_alpha=confidence_alpha,
            coverage_confidence_alpha=coverage_confidence_alpha,
            maximum_risk_upper=maximum_risk_upper,
            minimum_takeover_coverage=minimum_takeover_coverage,
        ),
        joint_harm_endpoints=JOINT_HARM_ENDPOINTS,
        harm_tolerances=normalized_harm_tolerances,
        corrected_risk_alpha=float(corrected_risk_alpha),
        corrected_coverage_alpha=float(corrected_coverage_alpha),
        maximum_risk_upper=float(maximum_risk_upper),
        minimum_takeover_coverage=float(minimum_takeover_coverage),
        accepted_count=int(accepted_count),
        failure_count=int(failures),
        risk_upper_bound=float(risk_upper),
        calibration_count=len(records),
        takeover_coverage=float(accepted_count / len(records)),
        takeover_coverage_lower_bound=float(coverage_lower),
        authorized_takeover_coverage=float(
            accepted_count / len(records) if gate else 0.0
        ),
        authorized_takeover_coverage_lower_bound=float(
            coverage_lower if gate else 0.0
        ),
        development_gate_passed=gate,
        feature_schema_sha256=feature_schema_sha256(),
    )
    return calibration, records


def select_with_risk_fallback(
    deployment: DeploymentGeometry,
    rule: ObservableRiskRule,
    calibration: RiskCalibration,
    *,
    expected_policy_contract_sha256: str,
) -> SelectionDecision:
    """Select from observable geometry only; all sensitive objects are absent."""

    if not isinstance(deployment, DeploymentGeometry):
        raise TypeError("fresh selection requires DeploymentGeometry")
    if calibration.feature_schema_sha256 != feature_schema_sha256():
        raise ValueError("calibration feature schema hash mismatch")
    if not math.isfinite(calibration.acceptance_threshold):
        raise ValueError("calibration threshold must be finite")
    if calibration.multiplicity_correction != MULTIPLICITY_CORRECTION:
        raise ValueError("calibration multiplicity correction differs")
    expected_grid = frozen_threshold_grid(rule)
    if (
        calibration.threshold_candidate_count != len(expected_grid)
        or calibration.threshold_grid != expected_grid
        or calibration.acceptance_threshold not in expected_grid
    ):
        raise ValueError("calibration threshold grid is invalid")
    if calibration.rule_contract_sha256 != rule_contract_sha256(rule):
        raise ValueError("calibration is not bound to the current rule")
    if (
        not 0.0 < calibration.confidence_alpha < 1.0
        or not 0.0 < calibration.coverage_confidence_alpha < 1.0
        or calibration.confidence_alpha + calibration.coverage_confidence_alpha
        > 0.05
    ):
        raise ValueError("joint risk and coverage alpha budget exceeds 0.05")
    if not math.isclose(
        calibration.corrected_risk_alpha,
        calibration.confidence_alpha / calibration.threshold_candidate_count,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("corrected risk alpha differs")
    if not math.isclose(
        calibration.corrected_coverage_alpha,
        calibration.coverage_confidence_alpha
        / calibration.threshold_candidate_count,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("corrected coverage alpha differs")
    if calibration.joint_harm_endpoints != JOINT_HARM_ENDPOINTS:
        raise ValueError("calibration joint harm endpoints differ")
    try:
        normalized_harm_tolerances = _normalized_harm_tolerances(
            dict(calibration.harm_tolerances)
        )
    except (TypeError, ValueError) as error:
        raise ValueError("calibration harm tolerances are invalid") from error
    if calibration.harm_tolerances != normalized_harm_tolerances:
        raise ValueError("calibration harm tolerance contract differs")
    recomputed_policy_contract = calibration_policy_contract_sha256(
        rule,
        harm_tolerances=dict(normalized_harm_tolerances),
        confidence_alpha=calibration.confidence_alpha,
        coverage_confidence_alpha=calibration.coverage_confidence_alpha,
        maximum_risk_upper=calibration.maximum_risk_upper,
        minimum_takeover_coverage=calibration.minimum_takeover_coverage,
    )
    if calibration.policy_contract_sha256 != recomputed_policy_contract:
        raise ValueError("calibration policy contract is internally inconsistent")
    if (
        not isinstance(expected_policy_contract_sha256, str)
        or len(expected_policy_contract_sha256) != 64
        or any(char not in "0123456789abcdef" for char in expected_policy_contract_sha256)
        or calibration.policy_contract_sha256 != expected_policy_contract_sha256
    ):
        raise ValueError("calibration policy differs from the independently expected policy")
    if not 0.0 <= calibration.risk_upper_bound <= 1.0:
        raise ValueError("calibration risk upper bound must lie in [0,1]")
    if (
        calibration.calibration_count <= 0
        or calibration.accepted_count < 0
        or calibration.accepted_count > calibration.calibration_count
        or calibration.failure_count < 0
        or calibration.failure_count > calibration.accepted_count
        or not 0.0 <= calibration.maximum_risk_upper <= 1.0
        or not 0.0 <= calibration.minimum_takeover_coverage <= 1.0
    ):
        raise ValueError("calibration counts or gate limits are invalid")
    expected_risk_upper = clopper_pearson_upper(
        calibration.failure_count,
        calibration.accepted_count,
        calibration.corrected_risk_alpha,
    )
    expected_coverage = calibration.accepted_count / calibration.calibration_count
    expected_coverage_lower = clopper_pearson_lower(
        calibration.accepted_count,
        calibration.calibration_count,
        calibration.corrected_coverage_alpha,
    )
    expected_gate = (
        expected_risk_upper <= calibration.maximum_risk_upper
        and expected_coverage_lower >= calibration.minimum_takeover_coverage
    )
    if not math.isclose(
        calibration.risk_upper_bound,
        expected_risk_upper,
        rel_tol=2e-13,
        abs_tol=2e-14,
    ):
        raise ValueError("calibration risk upper bound is internally inconsistent")
    if not math.isclose(
        calibration.takeover_coverage,
        expected_coverage,
        rel_tol=0.0,
        abs_tol=1e-15,
    ) or not math.isclose(
        calibration.takeover_coverage_lower_bound,
        expected_coverage_lower,
        rel_tol=2e-13,
        abs_tol=2e-14,
    ):
        raise ValueError("calibration coverage is internally inconsistent")
    if calibration.development_gate_passed is not expected_gate:
        raise ValueError("calibration gate is internally inconsistent")
    if not 0.0 <= calibration.takeover_coverage_lower_bound <= calibration.takeover_coverage <= 1.0:
        raise ValueError("calibration coverage bounds are invalid")
    if not (
        0.0
        <= calibration.authorized_takeover_coverage_lower_bound
        <= calibration.authorized_takeover_coverage
        <= calibration.takeover_coverage
    ):
        raise ValueError("authorized calibration coverage bounds are invalid")
    expected_authorized_coverage = expected_coverage if expected_gate else 0.0
    expected_authorized_lower = expected_coverage_lower if expected_gate else 0.0
    if not math.isclose(
        calibration.authorized_takeover_coverage,
        expected_authorized_coverage,
        rel_tol=0.0,
        abs_tol=1e-15,
    ) or not math.isclose(
        calibration.authorized_takeover_coverage_lower_bound,
        expected_authorized_lower,
        rel_tol=2e-13,
        abs_tol=2e-14,
    ):
        raise ValueError("authorized calibration coverage is internally inconsistent")
    deployment_schema_valid = (
        isinstance(deployment.geometry_features, torch.Tensor)
        and deployment.geometry_features.ndim == 1
        and deployment.geometry_features.dtype == torch.float64
        and deployment.geometry_features.device.type == "cpu"
        and int(deployment.geometry_features.numel()) == len(FEATURE_NAMES)
    )
    deployment_values = (
        tuple(float(value) for value in deployment.geometry_features)
        if deployment_schema_valid
        else ()
    )
    deployment_schema_valid = deployment_schema_valid and all(
        math.isfinite(value) for value in deployment_values
    )
    if deployment_schema_valid:
        candidate, risk_score = _rule_prediction(rule, deployment)
        support_gate_passed = _inside_observable_support(rule, deployment)
    else:
        candidate, risk_score = FALLBACK_PARTITION, 1.0
        support_gate_passed = False
    if not deployment_schema_valid:
        selected = FALLBACK_PARTITION
        reason = "OBSERVABLE_SCHEMA_MISMATCH"
    elif not support_gate_passed:
        selected = FALLBACK_PARTITION
        reason = "OBSERVABLE_SUPPORT_MISMATCH"
    elif candidate == FALLBACK_PARTITION:
        selected = FALLBACK_PARTITION
        reason = "RULE_SELECTED_FALLBACK"
    elif not calibration.development_gate_passed:
        selected = FALLBACK_PARTITION
        reason = "RISK_CALIBRATION_NOT_AUTHORIZED"
    elif risk_score > calibration.acceptance_threshold:
        selected = FALLBACK_PARTITION
        reason = "RISK_SCORE_ABOVE_FROZEN_THRESHOLD"
    else:
        selected = candidate
        reason = "CALIBRATED_CANDIDATE_TAKEOVER"
    observable_payload = (
        ",".join(format(value, ".17g") for value in deployment_values)
        if deployment_schema_valid
        else _canonical_json(
            {
                "device": str(getattr(deployment.geometry_features, "device", "missing")),
                "dtype": str(getattr(deployment.geometry_features, "dtype", "missing")),
                "shape": list(getattr(deployment.geometry_features, "shape", ())),
            }
        )
    )
    observable_hash = hashlib.sha256(observable_payload.encode("ascii")).hexdigest()
    return SelectionDecision(
        rig_id=deployment.rig_id,
        candidate_partition=candidate,
        selected_partition=selected,
        fallback_partition=FALLBACK_PARTITION,
        fallback_used=selected == FALLBACK_PARTITION,
        fallback_reason=reason,
        risk_score=float(risk_score),
        acceptance_threshold=calibration.acceptance_threshold,
        risk_upper_bound=calibration.risk_upper_bound,
        support_gate_passed=support_gate_passed,
        observable_feature_sha256=observable_hash,
        observable_feature_schema_sha256=feature_schema_sha256(),
    )
