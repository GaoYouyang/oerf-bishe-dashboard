from __future__ import annotations

import numpy as np

from site_tools.run_psu_b0_observable_morphology_probe import (
    ridge_scores,
)
from site_tools.run_psu_b0_rq_ogse_pcgls_development import (
    _secondary_metric_safety_audit,
    action_gain_targets,
    fit_l1_quantile_multioutput,
    fit_ridge_logistic_multioutput,
    numpy_risk_actions,
    route_gain_metrics,
    select_screen_candidate,
)


def test_quantile_fit_is_finite_and_tracks_lower_relation() -> None:
    x = np.linspace(-2.0, 2.0, 40)[:, None]
    target = np.stack((x[:, 0], -0.5 * x[:, 0]), axis=1)
    model = fit_l1_quantile_multioutput(
        x,
        target,
        quantile=0.2,
        regularization=0.01,
    )
    prediction = ridge_scores(model, x)
    assert prediction.shape == target.shape
    assert np.all(np.isfinite(prediction))
    assert np.corrcoef(prediction[:, 0], target[:, 0])[0, 1] > 0.99


def test_logistic_fit_separates_a_simple_harm_boundary() -> None:
    x = np.linspace(-3.0, 3.0, 60)[:, None]
    target = (x[:, 0] > 0.0).astype(np.float64)[:, None]
    model = fit_ridge_logistic_multioutput(
        x,
        target,
        regularization=0.1,
    )
    logits = ridge_scores(model, np.asarray([[-2.0], [2.0]]))
    assert logits[0, 0] < 0.0 < logits[1, 0]


def test_numpy_route_rejects_to_baseline() -> None:
    actions, accepted, _ = numpy_risk_actions(
        mean_scores=np.asarray([[0.0, 2.0], [0.0, 2.0]]),
        lower_scores=np.asarray([[0.0, 1.0], [0.0, 0.1]]),
        harm_logits=np.asarray([[-20.0, -3.0], [-20.0, 3.0]]),
        baseline_expert_index=0,
        route_mode="mean_quantile_harm",
        minimum_score=0.5,
        maximum_harm_probability=0.25,
    )
    assert actions.tolist() == [1, 0]
    assert accepted.tolist() == [True, False]


def test_action_gain_targets_and_route_metrics() -> None:
    baseline = [
        {"sample_id": "a", "field_relative_l2": 1.0},
        {"sample_id": "b", "field_relative_l2": 2.0},
    ]
    expert = [
        {"sample_id": "a", "field_relative_l2": 0.9},
        {"sample_id": "b", "field_relative_l2": 2.2},
    ]
    targets = action_gain_targets(
        sample_ids=["a", "b"],
        baseline_rows=baseline,
        action_rows_by_expert={0: baseline, 1: expert},
        expert_count=2,
        baseline_expert_index=0,
    )
    metrics = route_gain_metrics(
        targets,
        np.asarray([1, 0]),
        np.asarray([True, False]),
    )
    assert np.allclose(targets[:, 0], 0.0)
    assert np.isclose(targets[0, 1], 10.0)
    assert metrics["coverage"] == 0.5
    assert np.isclose(metrics["mean_field_gain_percent"], 5.0)


def test_strict_screen_selection_honors_risk_gate() -> None:
    screen = [
        {
            "coverage": 0.5,
            "mean_field_gain_percent": 2.0,
            "p10_field_gain_percent": 0.0,
            "harm_over_one_percent_rate": 0.0,
            "accepted_harm_over_one_percent_rate": 0.0,
            "accepted_mean_field_gain_percent": 4.0,
            "interpolation_fraction": 0.5,
        },
        {
            "coverage": 0.8,
            "mean_field_gain_percent": 4.0,
            "p10_field_gain_percent": -2.0,
            "harm_over_one_percent_rate": 0.1,
            "accepted_harm_over_one_percent_rate": 0.125,
            "accepted_mean_field_gain_percent": 5.0,
            "interpolation_fraction": 1.0,
        },
    ]
    selected = select_screen_candidate(
        screen,
        gate={
            "minimum_coverage": 0.2,
            "minimum_mean_field_gain_percent": 0.0,
            "minimum_p10_field_gain_percent": 0.0,
            "maximum_harm_over_one_percent_rate": 0.05,
            "maximum_accepted_harm_over_one_percent_rate": 0.05,
        },
    )
    assert selected["strict_gate_pass"] is True
    assert selected["mean_field_gain_percent"] == 2.0


def test_secondary_audit_rejects_front_degradation() -> None:
    summaries = []
    for split in ("risk_validation", "risk_calibration"):
        summaries.append(
            {
                "candidate_method": "candidate",
                "split": split,
                "secondary_metric_gain": {
                    "gradient_relative_l2": {
                        "mean_gain_percent": 0.5,
                        "p10_gain_percent": -0.5,
                    },
                    "front_top10_f1": {
                        "mean_gain_percent": (
                            0.2 if split == "risk_validation" else -0.1
                        ),
                        "p10_gain_percent": -0.5,
                    },
                },
            }
        )
    audit = _secondary_metric_safety_audit(
        summaries,
        method="candidate",
        gate={
            "minimum_gradient_mean_gain_percent": 0.0,
            "minimum_gradient_p10_gain_percent": -1.0,
            "minimum_front_mean_gain_percent": 0.0,
            "minimum_front_p10_gain_percent": -1.0,
        },
    )
    assert audit["pass"] is False
    assert audit["checks"]["risk_calibration_front_mean"] is False
