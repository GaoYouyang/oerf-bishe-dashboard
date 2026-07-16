from __future__ import annotations

from pathlib import Path

from site_tools.plot_psu_b0_dg_wpcgls_multiseed import build_figure


def _summary(noise: str, method: str, gain: float) -> dict:
    return {
        "noise_family": noise,
        "method": method,
        "replicate_mean_field_gain_percent": gain,
        "replicate_mean_field_gain_ci95_lower": gain - 0.2,
        "replicate_mean_field_gain_ci95_upper": gain + 0.2,
        "field_gain_percent_p10": gain - 0.5,
        "field_harm_over_one_percent_rate": 0.05,
    }


def test_build_figure_writes_png_and_pdf(tmp_path: Path) -> None:
    methods = [
        "component_iid",
        "dg_covgate",
        "oracle_covariance",
        "diagonal_shrinkage",
        "wrong_graph_assumption",
    ]
    report = {
        "summaries": [
            _summary(noise, method, float(index))
            for noise in ("component_iid", "graph_heat")
            for index, method in enumerate(methods)
        ],
        "family_summaries": [
            {
                "noise_family": "graph_heat",
                "method": method,
                "reaction_family": family,
                "field_gain_percent_mean": float(index + offset),
            }
            for index, family in enumerate(("plume", "thin_front"))
            for offset, method in enumerate(
                ("dg_covgate", "oracle_covariance")
            )
        ],
        "decision": {"verdict": "TEST_ONLY"},
    }
    replicate_rows = [
        {
            "noise_family": "graph_heat",
            "method": method,
            "field_gain_percent_mean": str(index + replicate * 0.1),
        }
        for index, method in enumerate(
            (
                "dg_covgate",
                "oracle_covariance",
                "wrong_graph_assumption",
            )
        )
        for replicate in range(4)
    ]
    output = tmp_path / "dg-wpcgls.png"
    build_figure(report, replicate_rows, output)
    assert output.stat().st_size > 1000
    assert output.with_suffix(".pdf").stat().st_size > 1000
