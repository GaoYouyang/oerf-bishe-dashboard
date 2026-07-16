from __future__ import annotations

from site_tools.run_psu_b0_dg_wpcgls_multiseed import (
    _aggregate,
    _decision,
    _paired_gain_rows,
)


def _metric(
    *,
    replicate: int,
    sample: int,
    noise: str,
    method: str,
    field: float,
    front: float = 0.4,
) -> dict:
    return {
        "replicate": replicate,
        "sample_index": sample,
        "reaction_family": "plume",
        "noise_family": noise,
        "relative_noise": 0.04,
        "method": method,
        "field_relative_l2": field,
        "gradient_relative_l2": field + 0.1,
        "front_top10_f1": front,
        "solver_elapsed_seconds": 1.0,
        "forward_calls": 4,
        "adjoint_calls": 4,
    }


def test_replicate_aggregation_uses_paired_field_gains() -> None:
    rows = []
    for replicate in range(3):
        for noise in ("component_iid", "graph_heat"):
            for sample, baseline in enumerate((0.5, 1.0)):
                rows.append(
                    _metric(
                        replicate=replicate,
                        sample=sample,
                        noise=noise,
                        method="component_iid",
                        field=baseline,
                    )
                )
                rows.append(
                    _metric(
                        replicate=replicate,
                        sample=sample,
                        noise=noise,
                        method="dg_covgate",
                        field=0.9 * baseline,
                        front=0.45,
                    )
                )
    gains = _paired_gain_rows(rows, baseline_method="component_iid")
    replicate_rows, summaries = _aggregate(gains)
    assert len(replicate_rows) == 12
    gated = next(
        row
        for row in summaries
        if row["noise_family"] == "graph_heat"
        and row["method"] == "dg_covgate"
    )
    assert abs(gated["replicate_mean_field_gain_percent"] - 10.0) < 1e-12
    assert gated["replicate_mean_field_gain_ci95_lower"] > 9.999
    assert gated["front_f1_gain_mean"] > 0.0


def test_decision_checks_correct_model_against_oracle_and_wrong_model() -> None:
    summaries = []
    for noise in ("component_iid", "graph_heat"):
        values = (
            ("dg_covgate", 2.0 if noise == "graph_heat" else 0.0, 0.0),
            ("oracle_covariance", 2.4 if noise == "graph_heat" else 0.0, 0.0),
            (
                "wrong_graph_assumption",
                0.5 if noise == "graph_heat" else 0.0,
                0.2,
            ),
        )
        for method, gain, harm in values:
            summaries.append(
                {
                    "noise_family": noise,
                    "method": method,
                    "replicate_mean_field_gain_percent": gain,
                    "replicate_mean_field_gain_ci95_lower": gain - 0.2,
                    "replicate_mean_field_gain_ci95_upper": gain + 0.2,
                    "field_gain_percent_p10": gain - 0.3,
                    "field_harm_over_one_percent_rate": harm,
                    "front_f1_gain_mean": 0.01,
                }
            )
    decision = _decision(
        summaries,
        criteria={
            "graph_heat_candidate_mean_gain_percent_minimum": 1.0,
            "graph_heat_candidate_gain_ci95_lower_minimum": 0.25,
            "graph_heat_candidate_p10_field_gain_percent_minimum": -0.5,
            "graph_heat_candidate_harm_over_one_percent_rate_maximum": 0.1,
            "graph_heat_candidate_front_f1_gain_mean_minimum": 0.005,
            "graph_heat_oracle_gain_retention_minimum": 0.75,
            "graph_heat_candidate_minus_wrong_mean_gain_percent_minimum": 0.5,
            "graph_heat_candidate_harm_rate_not_above_wrong": True,
            "component_iid_candidate_absolute_mean_gain_percent_maximum": 0.2,
            "component_iid_candidate_harm_over_one_percent_rate_maximum": 0.02,
        },
    )
    assert decision["all_checks_pass"] is True
    assert "REPLICATED" in decision["verdict"]
