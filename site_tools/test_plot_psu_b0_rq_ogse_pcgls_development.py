from __future__ import annotations

from pathlib import Path

from site_tools.plot_psu_b0_rq_ogse_pcgls_development import build_figure


def _summary(method: str, split: str, field: float, front: float) -> dict:
    return {
        "candidate_method": method,
        "split": split,
        "mean_field_gain_percent": field,
        "secondary_metric_gain": {
            "front_top10_f1": {"mean_gain_percent": front}
        },
    }


def test_build_figure_writes_png_and_pdf(tmp_path: Path) -> None:
    rq_methods = (
        "rq_ogse_strict",
        "rq_ogse_ablation_mean_only",
        "rq_ogse_ablation_quantile_only",
        "rq_ogse_ablation_quantile_harm",
        "rq_ogse_ablation_mean_quantile_harm",
    )
    rq = {
        "expert_bank": {"baseline_candidate_id": "base"},
        "finite_action_bank": [
            {
                "expert_candidate_id": expert,
                "interpolation_fraction": blend,
                "mean_field_gain_percent": 1.0 - blend,
                "harm_over_one_percent_rate": 0.1 * blend,
            }
            for expert in ("a", "b", "c")
            for blend in (0.25, 0.5, 0.75, 1.0)
        ],
        "paired_gain_summary": [
            _summary(
                method,
                split,
                2.8 - 0.4 * method_index - 0.3 * split_index,
                0.5 - 0.4 * method_index - 0.2 * split_index,
            )
            for method_index, method in enumerate(rq_methods)
            for split_index, split in enumerate(
                ("risk_validation", "risk_calibration")
            )
        ],
        "overall_decision": {
            "primary_strict_gate_pass": True,
            "secondary_metric_safety_pass": False,
        },
    }
    multi = {
        "paired_gain_summary": [
            _summary(
                "mo_rq_ogse_strict",
                split,
                1.4 - 0.2 * split_index,
                0.2 - 0.3 * split_index,
            )
            for split_index, split in enumerate(
                ("risk_validation", "risk_calibration")
            )
        ],
        "overall_decision": {
            "primary_strict_gate_pass": False,
            "secondary_metric_safety_pass": False,
        },
    }
    output = tmp_path / "figure.png"
    build_figure(rq, multi, output)
    assert output.stat().st_size > 1000
    assert output.with_suffix(".pdf").stat().st_size > 1000
