from __future__ import annotations

from dataclasses import replace
import math

import pytest
from scipy.stats import beta as beta_distribution
import torch

from demo_t16_operator.certified_grouped_majorizer import deployment_geometry_from_rig
from demo_t16_operator.observable_risk_fallback import (
    ALLOWED_CANDIDATES,
    FALLBACK_PARTITION,
    FIELD_ENDPOINT,
    FORBIDDEN_ORACLE,
    RESIDUAL_ENDPOINT,
    ObservableRiskRule,
    RiskCalibration,
    audit_operator_decomposition,
    calibration_policy_contract_sha256,
    calibrate_acceptance_threshold,
    clopper_pearson_lower,
    clopper_pearson_upper,
    feature_schema_sha256,
    fit_observable_risk_rule,
    frozen_threshold_grid,
    generate_four_way_rigs,
    rule_contract_sha256,
    selection_conditional_harm_rate,
    select_with_risk_fallback,
    split_rigs_four_way,
)


def _assignments() -> dict[str, str]:
    return {
        "train-00": "train",
        "train-01": "train",
        "model-00": "model_selection",
        "risk-00": "risk_calibration",
        "fresh-00": "fresh_geometry_ood",
    }


def _rigs():
    return generate_four_way_rigs(
        split_assignments=_assignments(),
        geometry_seed=101,
        noise_seed=202,
        row_count=8,
        column_count=6,
    )


def _rule() -> ObservableRiskRule:
    return ObservableRiskRule(
        feature_index=0,
        threshold=0.0,
        left_partition="paired_local",
        right_partition="triad_bridge",
        left_risk_score=0.1,
        right_risk_score=0.2,
        fallback_partition=FALLBACK_PARTITION,
        allowed_partitions=ALLOWED_CANDIDATES,
        forbidden_oracle=FORBIDDEN_ORACLE,
        feature_schema_sha256=feature_schema_sha256(),
        support_minimums=(-10.0,) * 6,
        support_maximums=(10.0,) * 6,
    )


_HARM_TOLERANCES = {FIELD_ENDPOINT: 0.02, RESIDUAL_ENDPOINT: 0.02}


def _policy_contract(*, maximum_risk_upper: float = 0.8) -> str:
    return calibration_policy_contract_sha256(
        _rule(),
        harm_tolerances=_HARM_TOLERANCES,
        confidence_alpha=0.025,
        coverage_confidence_alpha=0.025,
        maximum_risk_upper=maximum_risk_upper,
        minimum_takeover_coverage=0.25,
    )


def _calibration(*, maximum_risk_upper: float = 0.8) -> RiskCalibration:
    rule = _rule()
    grid = frozen_threshold_grid(rule)
    corrected_alpha = 0.025 / len(grid)
    risk_upper = clopper_pearson_upper(0, 4, corrected_alpha)
    coverage_lower = clopper_pearson_lower(4, 4, corrected_alpha)
    gate = risk_upper <= maximum_risk_upper and coverage_lower >= 0.25
    return RiskCalibration(
        acceptance_threshold=0.2,
        confidence_alpha=0.025,
        coverage_confidence_alpha=0.025,
        multiplicity_correction="BONFERRONI_FROZEN_FINITE_GRID",
        threshold_candidate_count=len(grid),
        threshold_grid=grid,
        rule_contract_sha256=rule_contract_sha256(rule),
        policy_contract_sha256=_policy_contract(
            maximum_risk_upper=maximum_risk_upper
        ),
        joint_harm_endpoints=(FIELD_ENDPOINT, RESIDUAL_ENDPOINT),
        harm_tolerances=tuple(_HARM_TOLERANCES.items()),
        corrected_risk_alpha=corrected_alpha,
        corrected_coverage_alpha=corrected_alpha,
        maximum_risk_upper=maximum_risk_upper,
        minimum_takeover_coverage=0.25,
        accepted_count=4,
        failure_count=0,
        risk_upper_bound=risk_upper,
        calibration_count=4,
        takeover_coverage=1.0,
        takeover_coverage_lower_bound=coverage_lower,
        authorized_takeover_coverage=1.0 if gate else 0.0,
        authorized_takeover_coverage_lower_bound=coverage_lower if gate else 0.0,
        development_gate_passed=gate,
        feature_schema_sha256=feature_schema_sha256(),
    )


def _select(
    deployment: object,
    *,
    rule: ObservableRiskRule | None = None,
    calibration: RiskCalibration | None = None,
    expected_policy_contract_sha256: str | None = None,
):
    return select_with_risk_fallback(
        deployment,  # type: ignore[arg-type]
        rule or _rule(),
        calibration or _calibration(),
        expected_policy_contract_sha256=(
            expected_policy_contract_sha256 or _policy_contract()
        ),
    )


def test_clopper_pearson_upper_matches_closed_form_zero_failure_case() -> None:
    observed = clopper_pearson_upper(0, 4, 0.05)
    assert observed == pytest.approx(1.0 - 0.05 ** 0.25, rel=1e-12)
    assert clopper_pearson_upper(4, 4, 0.05) == 1.0
    assert clopper_pearson_upper(0, 0, 0.05) == 1.0


def test_clopper_pearson_lower_matches_closed_form_all_success_case() -> None:
    observed = clopper_pearson_lower(4, 4, 0.05)
    assert observed == pytest.approx(0.05 ** 0.25, rel=1e-12)
    assert clopper_pearson_lower(0, 4, 0.05) == 0.0


@pytest.mark.parametrize(("successes", "count"), [(1, 5), (3, 8), (7, 9)])
def test_clopper_pearson_bounds_match_scipy_beta_quantiles(
    successes: int, count: int
) -> None:
    alpha = 0.037
    failures = count - successes
    expected_upper = beta_distribution.ppf(
        1.0 - alpha, failures + 1, count - failures
    )
    expected_lower = beta_distribution.ppf(
        alpha, successes, count - successes + 1
    )
    assert clopper_pearson_upper(failures, count, alpha) == pytest.approx(
        expected_upper, rel=2e-13
    )
    assert clopper_pearson_lower(successes, count, alpha) == pytest.approx(
        expected_lower, rel=2e-13
    )


@pytest.mark.parametrize(("failures", "count"), [(-1, 4), (5, 4), (1, -1)])
def test_clopper_pearson_rejects_invalid_counts(failures: int, count: int) -> None:
    with pytest.raises(ValueError, match="counts"):
        clopper_pearson_upper(failures, count, 0.05)


def test_four_way_split_keeps_complete_disjoint_rigs() -> None:
    rigs = _rigs()
    grouped, contract = split_rigs_four_way(rigs)
    assert set(grouped) == {"train", "model_selection", "risk_calibration", "fresh_geometry_ood"}
    assert sum(len(values) for values in grouped.values()) == len(rigs)
    assert contract["split_unit"] == "COMPLETE_RIG"
    assert contract["random_ray_or_pixel_split_used"] is False
    assert {role: len(values) for role, values in grouped.items()} == {
        "train": 2,
        "model_selection": 1,
        "risk_calibration": 1,
        "fresh_geometry_ood": 1,
    }


def test_fresh_selection_is_invariant_to_sensitive_rig_mutations() -> None:
    rig = next(rig for rig in _rigs() if rig.split_role == "fresh_geometry_ood")
    first = _select(deployment_geometry_from_rig(rig))
    changed = replace(
        rig,
        truth=torch.full_like(rig.truth, 99.0),
        target=torch.full_like(rig.target, -77.0),
        primitives=torch.zeros_like(rig.primitives),
        signed_matrix=torch.zeros_like(rig.signed_matrix),
    )
    second = _select(deployment_geometry_from_rig(changed))
    assert first == second
    assert not any(
        (
            first.uses_truth,
            first.uses_target,
            first.uses_primitives,
            first.uses_signed_matrix,
            first.uses_exact_abs_operator,
            first.uses_solver_trajectory,
        )
    )
    with pytest.raises(TypeError, match="DeploymentGeometry"):
        _select(rig)


def test_fresh_selection_falls_back_when_calibration_is_not_authorized() -> None:
    rig = next(rig for rig in _rigs() if rig.split_role == "fresh_geometry_ood")
    calibration = _calibration(maximum_risk_upper=0.6)
    decision = _select(
        deployment_geometry_from_rig(rig),
        calibration=calibration,
        expected_policy_contract_sha256=_policy_contract(maximum_risk_upper=0.6),
    )
    assert decision.selected_partition == FALLBACK_PARTITION
    assert decision.fallback_used is True
    assert decision.fallback_reason == "RISK_CALIBRATION_NOT_AUTHORIZED"


def test_fresh_selection_falls_back_outside_observable_support() -> None:
    rig = next(rig for rig in _rigs() if rig.split_role == "fresh_geometry_ood")
    deployment = deployment_geometry_from_rig(rig)
    deployment = replace(
        deployment, geometry_features=torch.full_like(deployment.geometry_features, 99.0)
    )
    decision = _select(deployment)
    assert decision.fallback_used is True
    assert decision.support_gate_passed is False
    assert decision.fallback_reason == "OBSERVABLE_SUPPORT_MISMATCH"


@pytest.mark.parametrize(
    "features",
    [
        torch.zeros((6, 1), dtype=torch.float64),
        torch.zeros(6, dtype=torch.float32),
    ],
    ids=["rank-two", "float32"],
)
def test_fresh_selection_falls_back_on_observable_schema_mismatch(
    features: torch.Tensor,
) -> None:
    rig = next(rig for rig in _rigs() if rig.split_role == "fresh_geometry_ood")
    deployment = replace(deployment_geometry_from_rig(rig), geometry_features=features)
    decision = _select(deployment)
    assert decision.fallback_used is True
    assert decision.support_gate_passed is False
    assert decision.fallback_reason == "OBSERVABLE_SCHEMA_MISMATCH"


def test_calibration_must_match_rule_and_frozen_threshold_grid() -> None:
    rig = next(rig for rig in _rigs() if rig.split_role == "fresh_geometry_ood")
    deployment = deployment_geometry_from_rig(rig)
    with pytest.raises(ValueError, match="threshold grid"):
        _select(
            deployment,
            calibration=replace(_calibration(), acceptance_threshold=0.19),
        )
    with pytest.raises(ValueError, match="bound to the current rule"):
        _select(
            deployment,
            rule=replace(_rule(), threshold=0.5),
        )


def test_calibration_statistics_must_be_internally_consistent() -> None:
    rig = next(rig for rig in _rigs() if rig.split_role == "fresh_geometry_ood")
    deployment = deployment_geometry_from_rig(rig)
    with pytest.raises(ValueError, match="risk upper bound is internally inconsistent"):
        _select(
            deployment,
            calibration=replace(_calibration(), risk_upper_bound=0.4),
        )
    with pytest.raises(ValueError, match="alpha budget"):
        _select(
            deployment,
            calibration=replace(
                _calibration(),
                confidence_alpha=0.0,
                corrected_risk_alpha=0.0,
            ),
        )


def test_selector_rejects_rehashed_but_weakened_policy() -> None:
    rig = next(rig for rig in _rigs() if rig.split_role == "fresh_geometry_ood")
    weakened_limit = 0.9
    weakened = replace(
        _calibration(),
        maximum_risk_upper=weakened_limit,
        policy_contract_sha256=_policy_contract(
            maximum_risk_upper=weakened_limit
        ),
    )
    with pytest.raises(ValueError, match="independently expected policy"):
        _select(
            deployment_geometry_from_rig(rig),
            calibration=weakened,
            expected_policy_contract_sha256=_policy_contract(),
        )


def test_actual_solver_operator_mismatch_is_detected() -> None:
    rig = _rigs()[0]
    valid = audit_operator_decomposition(rig.primitives, rig.signed_matrix)
    assert valid["operator_decomposition_verified"] is True
    changed = rig.signed_matrix.clone()
    changed[0, 0] += 0.01
    invalid = audit_operator_decomposition(rig.primitives, changed)
    assert invalid["operator_decomposition_verified"] is False
    assert int(invalid["operator_decomposition_mismatch_count"]) == 1


def test_model_fit_rejects_all_in_one_selector_injection() -> None:
    rigs = _rigs()
    train = [rig for rig in rigs if rig.split_role == "train"]
    model = [rig for rig in rigs if rig.split_role == "model_selection"]
    score_table = {
        (rig.rig_id, partition): {
            FIELD_ENDPOINT: 1.0,
            RESIDUAL_ENDPOINT: 1.0,
        }
        for rig in [*train, *model]
        for partition in (*ALLOWED_CANDIDATES, FORBIDDEN_ORACLE)
    }
    with pytest.raises(ValueError, match="catalogue"):
        fit_observable_risk_rule(
            train,
            model,
            score_table,
            candidate_names=(*ALLOWED_CANDIDATES[:-1], FORBIDDEN_ORACLE),
            train_top_k=1,
            harm_tolerances={FIELD_ENDPOINT: 0.0, RESIDUAL_ENDPOINT: 0.0},
            failure_penalty=1.0,
        )


def test_cp_upper_is_monotone_in_failures() -> None:
    values = [clopper_pearson_upper(k, 8, 0.1) for k in range(9)]
    assert all(math.isfinite(value) for value in values)
    assert values == sorted(values)


def test_selection_conditional_harm_rate_never_uses_all_fresh_rigs() -> None:
    assert selection_conditional_harm_rate(1, 1) == 1.0
    assert selection_conditional_harm_rate(0, 0) is None
    with pytest.raises(ValueError, match="conditional counts"):
        selection_conditional_harm_rate(2, 1)


def test_calibration_uses_frozen_grid_bonferroni_and_coverage_lower_bound() -> None:
    rig = next(rig for rig in _rigs() if rig.split_role == "risk_calibration")
    score_table = {
        (rig.rig_id, partition): {
            FIELD_ENDPOINT: 1.0,
            RESIDUAL_ENDPOINT: 1.0,
        }
        for partition in ALLOWED_CANDIDATES
    }
    calibration, _ = calibrate_acceptance_threshold(
        _rule(),
        [rig],
        score_table,
        harm_tolerances={FIELD_ENDPOINT: 0.0, RESIDUAL_ENDPOINT: 0.0},
        confidence_alpha=0.2,
        coverage_confidence_alpha=0.3,
        maximum_risk_upper=1.0,
        minimum_takeover_coverage=0.0,
    )
    assert calibration.threshold_candidate_count == 3
    assert calibration.threshold_grid == frozen_threshold_grid(_rule())
    assert calibration.rule_contract_sha256 == rule_contract_sha256(_rule())
    assert calibration.multiplicity_correction == "BONFERRONI_FROZEN_FINITE_GRID"
    assert calibration.corrected_risk_alpha == pytest.approx(0.2 / 3.0)
    assert calibration.corrected_coverage_alpha == pytest.approx(0.1)
    assert calibration.accepted_count == 1
    assert calibration.failure_count == 0
    assert calibration.risk_upper_bound == pytest.approx(1.0 - 0.2 / 3.0)
    assert calibration.takeover_coverage == 1.0
    assert calibration.takeover_coverage_lower_bound == pytest.approx(0.1)


def test_residual_only_harm_triggers_joint_failure() -> None:
    rig = next(rig for rig in _rigs() if rig.split_role == "risk_calibration")
    score_table = {}
    for partition in ALLOWED_CANDIDATES:
        score_table[(rig.rig_id, partition)] = {
            FIELD_ENDPOINT: 1.0,
            RESIDUAL_ENDPOINT: 1.0 if partition == FALLBACK_PARTITION else 1.03,
        }
    calibration, records = calibrate_acceptance_threshold(
        _rule(),
        [rig],
        score_table,
        harm_tolerances={FIELD_ENDPOINT: 0.02, RESIDUAL_ENDPOINT: 0.02},
        confidence_alpha=0.2,
        coverage_confidence_alpha=0.3,
        maximum_risk_upper=1.0,
        minimum_takeover_coverage=0.0,
    )
    assert records[0]["observed_field_harm_vs_fallback"] == pytest.approx(0.0)
    assert records[0]["observed_residual_harm_vs_fallback"] == pytest.approx(0.03)
    assert records[0]["harm_failure"] is True
    assert calibration.failure_count == 1
    audit_operator_decomposition,
