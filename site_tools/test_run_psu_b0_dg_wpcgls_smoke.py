from __future__ import annotations

from site_tools.run_psu_b0_dg_wpcgls_smoke import (
    _mechanism_decision,
    _summarize,
)


def _row(
    *,
    sample: int,
    noise: str,
    method: str,
    field: float,
    front: float = 0.5,
) -> dict:
    return {
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


def test_summary_is_paired_against_component_iid() -> None:
    rows = []
    for noise in ("component_iid", "graph_heat"):
        for sample, baseline in enumerate((0.5, 1.0)):
            rows.append(
                _row(
                    sample=sample,
                    noise=noise,
                    method="component_iid",
                    field=baseline,
                )
            )
            rows.append(
                _row(
                    sample=sample,
                    noise=noise,
                    method="dg_covgate",
                    field=0.9 * baseline,
                    front=0.55,
                )
            )
    summary = _summarize(rows)
    gated = next(
        row
        for row in summary
        if row["noise_family"] == "graph_heat"
        and row["method"] == "dg_covgate"
    )
    assert abs(gated["field_gain_vs_component_iid_percent_median"] - 10.0) < 1e-12
    assert gated["front_f1_gain_vs_component_iid_mean"] > 0.0


def test_mechanism_decision_requires_oracle_and_misspecification_controls() -> None:
    rows = []
    for noise in ("component_iid", "graph_heat"):
        for method, gain, harm in (
            ("dg_covgate", 2.0 if noise == "graph_heat" else -0.1, 0.0),
            ("oracle_covariance", 4.0 if noise == "graph_heat" else 0.0, 0.0),
            ("wrong_graph_assumption", 0.0, 0.0),
        ):
            rows.append(
                {
                    "noise_family": noise,
                    "method": method,
                    "field_gain_vs_component_iid_percent_median": gain,
                    "field_harm_over_one_percent_rate": harm,
                }
            )
    decision = _mechanism_decision(
        rows,
        criteria={
            "graph_heat_oracle_median_field_gain_percent_minimum": 1.0,
            "graph_heat_dg_covgate_median_field_gain_percent_minimum": 0.25,
            "graph_heat_dg_covgate_oracle_gain_retention_minimum": 0.2,
            "component_iid_dg_covgate_median_field_harm_percent_maximum": 0.5,
            "component_iid_dg_covgate_harm_over_one_percent_rate_maximum": 0.25,
            "wrong_graph_must_trail_oracle_field_gain_percent": 0.5,
        },
    )
    assert decision["all_checks_pass"] is True
    assert "ADVANCE" in decision["verdict"]
