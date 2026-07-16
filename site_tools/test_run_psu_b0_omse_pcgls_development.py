from __future__ import annotations

from site_tools.run_psu_b0_omse_pcgls_development import (
    select_omse_screen_candidate,
)


def test_strict_screen_selects_best_feasible_candidate() -> None:
    gate = {
        "minimum_coverage": 0.25,
        "minimum_mean_field_gain_percent": 0.0,
        "minimum_p10_field_gain_percent": 0.0,
        "maximum_harm_over_one_percent_rate": 0.05,
        "maximum_accepted_harm_over_one_percent_rate": 0.05,
    }
    screen = [
        {
            "mean_field_gain_percent": 1.0,
            "p10_field_gain_percent": 0.0,
            "coverage": 0.3,
            "harm_over_one_percent_rate": 0.0,
            "accepted_harm_over_one_percent_rate": 0.0,
            "maximum_blend": 0.5,
            "temperature": 1.0,
        },
        {
            "mean_field_gain_percent": 1.5,
            "p10_field_gain_percent": 0.2,
            "coverage": 0.4,
            "harm_over_one_percent_rate": 0.0,
            "accepted_harm_over_one_percent_rate": 0.0,
            "maximum_blend": 0.25,
            "temperature": 0.5,
        },
        {
            "mean_field_gain_percent": 3.0,
            "p10_field_gain_percent": -2.0,
            "coverage": 0.5,
            "harm_over_one_percent_rate": 0.2,
            "accepted_harm_over_one_percent_rate": 0.4,
            "maximum_blend": 1.0,
            "temperature": 0.5,
        },
    ]
    selected = select_omse_screen_candidate(screen, gate=gate)
    assert selected["strict_gate_pass"]
    assert selected["mean_field_gain_percent"] == 1.5


def test_strict_screen_returns_no_candidate_when_tail_gate_fails() -> None:
    selected = select_omse_screen_candidate(
        [
            {
                "mean_field_gain_percent": 2.0,
                "p10_field_gain_percent": -0.1,
                "coverage": 0.5,
                "harm_over_one_percent_rate": 0.1,
                "accepted_harm_over_one_percent_rate": 0.2,
                "maximum_blend": 1.0,
                "temperature": 1.0,
            }
        ],
        gate={
            "minimum_coverage": 0.25,
            "minimum_mean_field_gain_percent": 0.0,
            "minimum_p10_field_gain_percent": 0.0,
            "maximum_harm_over_one_percent_rate": 0.05,
            "maximum_accepted_harm_over_one_percent_rate": 0.05,
        },
    )
    assert not selected["strict_gate_pass"]
