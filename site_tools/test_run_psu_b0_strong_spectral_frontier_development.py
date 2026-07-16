from __future__ import annotations

from site_tools.run_psu_b0_strong_spectral_frontier_development import (
    _generalized_candidates,
    _pcgls_candidates,
    _schedule_candidates,
    select_spectral_candidate,
)


def test_candidate_builders_are_deterministic() -> None:
    config = {
        "generalized_sobolev": {
            "strength_grid": [4.0, 5.0],
            "epsilon_grid": [0.05],
            "axis_weight_patterns_xyz": {
                "iso": [1.0, 1.0, 1.0],
                "x": [2.0, 1.0, 1.0],
            },
        },
        "scheduled_sobolev": {
            "schedules": [[5.0, 5.0, 5.0, 5.0], [6.0, 5.0, 4.0, 3.0]]
        },
        "sobolev_pcgls": {
            "stage_counts": [3, 4],
            "strength_grid": [4.0],
            "epsilon_grid": [0.05, 0.1],
        },
    }
    assert len(_generalized_candidates(config)) == 4
    assert len(_schedule_candidates(config)) == 2
    assert len(_pcgls_candidates(config)) == 4
    assert _generalized_candidates(config) == _generalized_candidates(config)


def test_spectral_selection_uses_metric_then_identifier() -> None:
    screen = [
        {
            "family": "scheduled_sobolev",
            "candidate_id": "b",
            "mean_combined_loss": 0.4,
            "mean_field_relative_l2": 0.5,
        },
        {
            "family": "scheduled_sobolev",
            "candidate_id": "a",
            "mean_combined_loss": 0.4,
            "mean_field_relative_l2": 0.5,
        },
    ]
    selected = select_spectral_candidate(
        screen,
        family="scheduled_sobolev",
    )
    assert selected["candidate_id"] == "a"
