from __future__ import annotations

import numpy as np

from site_tools.run_psu_b0_mo_rq_ogse_pcgls_development import (
    front_delta_targets,
    multiobjective_route_metrics,
    numpy_multiobjective_actions,
    select_multiobjective_candidate,
)


def test_front_delta_targets_use_absolute_f1_change() -> None:
    baseline = [
        {"sample_id": "a", "front_top10_f1": 0.4},
        {"sample_id": "b", "front_top10_f1": 0.8},
    ]
    expert = [
        {"sample_id": "a", "front_top10_f1": 0.45},
        {"sample_id": "b", "front_top10_f1": 0.7},
    ]
    targets = front_delta_targets(
        sample_ids=["a", "b"],
        baseline_rows=baseline,
        action_rows_by_expert={0: baseline, 1: expert},
        expert_count=2,
        baseline_expert_index=0,
    )
    assert np.allclose(targets[:, 0], 0.0)
    assert np.allclose(targets[:, 1], [0.05, -0.1])


def test_numpy_multiobjective_actions_veto_front_harm() -> None:
    actions, accepted, _, _ = numpy_multiobjective_actions(
        field_mean_scores=np.asarray([[0.0, 3.0], [0.0, 3.0]]),
        field_lower_scores=np.asarray([[0.0, 1.0], [0.0, 1.0]]),
        field_harm_logits=np.asarray([[-20.0, -3.0], [-20.0, -3.0]]),
        front_lower_scores=np.asarray([[0.0, 0.01], [0.0, -0.03]]),
        front_harm_logits=np.asarray([[-20.0, -3.0], [-20.0, 3.0]]),
        baseline_expert_index=0,
        minimum_field_score=1.0,
        maximum_field_harm_probability=0.25,
        minimum_front_lower_delta=-0.01,
        maximum_front_harm_probability=0.25,
    )
    assert actions.tolist() == [1, 0]
    assert accepted.tolist() == [True, False]


def test_multiobjective_metrics_keep_field_and_front_separate() -> None:
    metrics = multiobjective_route_metrics(
        field_gain_targets=np.asarray([[0.0, 4.0], [0.0, -2.0]]),
        front_delta_values=np.asarray([[0.0, 0.03], [0.0, -0.04]]),
        actions=np.asarray([1, 0]),
        accepted=np.asarray([True, False]),
        field_harm_threshold=-1.0,
        front_harm_threshold=-0.02,
    )
    assert metrics["mean_field_gain_percent"] == 2.0
    assert np.isclose(metrics["mean_front_f1_absolute_delta"], 0.015)
    assert metrics["front_harm_rate"] == 0.0


def test_multiobjective_selection_rejects_front_unsafe_row() -> None:
    safe = {
        "coverage": 0.3,
        "mean_field_gain_percent": 1.5,
        "p10_field_gain_percent": 0.0,
        "field_harm_rate": 0.0,
        "accepted_field_harm_rate": 0.0,
        "mean_front_f1_absolute_delta": 0.01,
        "p10_front_f1_absolute_delta": 0.0,
        "front_harm_rate": 0.0,
        "accepted_front_harm_rate": 0.0,
        "interpolation_fraction": 0.5,
    }
    unsafe = {
        **safe,
        "mean_field_gain_percent": 3.0,
        "mean_front_f1_absolute_delta": -0.02,
        "front_harm_rate": 0.1,
    }
    selected = select_multiobjective_candidate(
        [safe, unsafe],
        gate={
            "minimum_coverage": 0.2,
            "minimum_mean_field_gain_percent": 0.0,
            "minimum_p10_field_gain_percent": 0.0,
            "maximum_field_harm_rate": 0.05,
            "maximum_accepted_field_harm_rate": 0.05,
            "minimum_mean_front_f1_absolute_delta": 0.0,
            "minimum_p10_front_f1_absolute_delta": -0.01,
            "maximum_front_harm_rate": 0.05,
            "maximum_accepted_front_harm_rate": 0.05,
        },
    )
    assert selected["strict_gate_pass"] is True
    assert selected["mean_field_gain_percent"] == 1.5
