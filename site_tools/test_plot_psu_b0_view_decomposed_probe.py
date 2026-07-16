from __future__ import annotations

from pathlib import Path

from site_tools.plot_psu_b0_view_decomposed_probe import build_figure


def _row(method: str, split: str, field: float, front: float) -> dict:
    return {
        "candidate_method": method,
        "split": split,
        "mean_field_gain_percent": field,
        "harm_over_one_percent_rate": 0.05,
        "secondary_metric_gain": {
            "front_top10_f1": {"mean_gain_percent": front}
        },
    }


def test_build_figure_writes_png_and_pdf(tmp_path: Path) -> None:
    methods = (
        "vd0_pooled_initial_normal_strict",
        "vd0_view_conflict_strict",
        "vd0_pooled_plus_view_conflict_strict",
    )
    feature_sets = (
        "pooled_initial_normal",
        "view_conflict",
        "pooled_plus_view_conflict",
    )
    report = {
        "paired_gain_summary": [
            _row(
                method,
                split,
                3.0 - method_index - 0.2 * split_index,
                0.5 - method_index - 0.3 * split_index,
            )
            for method_index, method in enumerate(methods)
            for split_index, split in enumerate(
                ("risk_validation", "risk_calibration")
            )
        ],
        "stress_audits": {
            feature_set: {
                "strict": {
                    stress: {
                        "mean_field_gain_percent": (
                            1.0 - feature_index - stress_index
                        )
                    }
                    for stress_index, stress in enumerate(
                        (
                            "leave_one_morphology_family_out",
                            "leave_one_noise_profile_out",
                        )
                    )
                }
            }
            for feature_index, feature_set in enumerate(feature_sets)
        },
        "regeneration_checks": {
            "maximum_group_sum_relative_error": 1e-7
        },
        "runtime": {
            "wall_seconds": 6.0,
            "maximum_rss_bytes": 430_000_000,
        },
    }
    output = tmp_path / "figure.png"
    build_figure(report, output)
    assert output.stat().st_size > 1000
    assert output.with_suffix(".pdf").stat().st_size > 1000
