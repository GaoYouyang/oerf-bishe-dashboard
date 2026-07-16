from __future__ import annotations

from pathlib import Path

from site_tools.plot_psu_b0_detector_graph_covariance_gate import (
    build_figure,
)


def _summary(family: str, repeat: int, model: str) -> dict:
    return {
        "truth_family": family,
        "repeat_count": repeat,
        "model": model,
        "nll_gain_vs_component_iid": {
            "median": 0.01,
            "p10": 0.0,
            "p90": 0.02,
        },
        "absolute_coverage_error": {"p90": 0.05},
        "graph_activation_rate": 0.8 if family != "component_iid" else 0.1,
    }


def test_build_figure_writes_png_and_pdf(tmp_path: Path) -> None:
    repeats = [4, 8]
    families = [
        "component_iid",
        "graph_heat",
        "nonstationary_drift",
    ]
    models = [
        "diagonal_shrinkage",
        "always_graph",
        "always_low_rank_drift",
        "dg_covgate",
    ]
    report = {
        "configuration": {"simulation": {"repeat_counts": repeats}},
        "summary": [
            _summary(family, repeat, model)
            for family in families
            for repeat in repeats
            for model in models
        ],
        "empirical_covariance_rank_ceiling": [
            {"repeat_count": repeat, "rank_fraction": (repeat - 1) / 512}
            for repeat in repeats
        ],
        "acquisition_gate": {
            "minimum_repeat_count_passing_synthetic_gate": 8
        },
    }
    output = tmp_path / "covariance-gate.png"
    build_figure(report, output)
    assert output.stat().st_size > 1000
    assert output.with_suffix(".pdf").stat().st_size > 1000
