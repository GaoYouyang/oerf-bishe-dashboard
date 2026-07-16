from __future__ import annotations

from site_tools.analyze_psu_b0_covariance_stopping_rules import (
    choose_one_threshold,
    choose_rollback_continue,
    gate_checks,
    summarize,
)


def _stage(stage: int, **features: float) -> dict[str, object]:
    return {
        "stage": stage,
        "field_gain_vs_component_s3_k4_percent": 3.0,
        "gradient_gain_vs_component_s3_k4_percent": 2.0,
        "front_gain_vs_component_s3_k4": 0.01,
        "relative_whitened_residual_objective": 0.1,
        "residual_objective_reduction_from_previous_stage": 0.02,
        "previous_pcgls_beta": 0.4,
        "relative_volume_update": 0.2,
        **features,
    }


def test_one_threshold_switches_only_between_stages_four_and_five() -> None:
    stages = {stage: _stage(stage) for stage in (2, 3, 4, 5)}
    assert choose_one_threshold(
        stages,
        feature="relative_whitened_residual_objective",
        threshold=0.2,
        direction="le",
    )["stage"] == 5
    assert choose_one_threshold(
        stages,
        feature="relative_whitened_residual_objective",
        threshold=0.05,
        direction="le",
    )["stage"] == 4


def test_rollback_continue_rule_has_three_declared_actions() -> None:
    stages = {stage: _stage(stage) for stage in (2, 3, 4, 5)}
    stages[4]["previous_pcgls_beta"] = 0.8
    stages[4]["relative_volume_update"] = 0.5
    assert choose_rollback_continue(
        stages,
        beta_minimum=0.7,
        rollback_update_minimum=0.4,
        extension_reduction_maximum=0.03,
    )["stage"] == 3
    stages[4]["relative_volume_update"] = 0.2
    assert choose_rollback_continue(
        stages,
        beta_minimum=0.7,
        rollback_update_minimum=0.4,
        extension_reduction_maximum=0.03,
    )["stage"] == 5
    stages[4]["previous_pcgls_beta"] = 0.2
    stages[4]["residual_objective_reduction_from_previous_stage"] = 0.1
    assert choose_rollback_continue(
        stages,
        beta_minimum=0.7,
        rollback_update_minimum=0.4,
        extension_reduction_maximum=0.03,
    )["stage"] == 4


def test_summary_and_gates_enforce_tail_and_compute_budget() -> None:
    selected = [_stage(4), _stage(4), _stage(5)]
    summary = summarize(selected)
    checks = gate_checks(
        summary,
        gates={
            "mean_field_gain_percent_minimum": 2.0,
            "field_gain_p10_percent_minimum": 0.0,
            "field_harm_over_one_percent_rate_maximum": 0.05,
            "worst_field_gain_percent_minimum": -2.0,
            "mean_gradient_gain_percent_minimum": 1.0,
            "mean_front_f1_gain_minimum": 0.005,
            "mean_stage_maximum": 4.5,
        },
    )
    assert all(checks.values())
