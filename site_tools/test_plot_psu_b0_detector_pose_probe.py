from __future__ import annotations

from pathlib import Path

from site_tools.plot_psu_b0_detector_pose_probe import build_figure


def _row(method: str, split: str, field: float, front: float) -> dict:
    return {
        "candidate_method": method,
        "split": split,
        "mean_field_gain_percent": field,
        "harm_over_one_percent_rate": 0.08,
        "secondary_metric_gain": {
            "front_top10_f1": {"mean_gain_percent": front}
        },
    }


def test_build_figure_writes_png_and_pdf(tmp_path: Path) -> None:
    methods = (
        (
            "vd0b_pooled_initial_normal_strict",
            "pooled_initial_normal",
        ),
        (
            "vd0b_detector_graph_front_strict",
            "detector_graph_front",
        ),
        (
            "vd0b_pooled_plus_detector_graph_front_strict",
            "pooled_plus_detector_graph_front",
        ),
    )
    report = {
        "paired_gain_summary": [
            _row(
                method,
                split,
                3.1 - method_index - 0.1 * split_index,
                0.4 - method_index - 0.2 * split_index,
            )
            for method_index, (method, _) in enumerate(methods)
            for split_index, split in enumerate(
                ("risk_validation", "risk_calibration")
            )
        ],
        "stress_audits": {
            feature_set: {
                "strict": {
                    stress_name: {
                        "mean_field_gain_percent": (
                            1.0 - feature_index - stress_index
                        )
                    }
                    for stress_index, stress_name in enumerate(
                        (
                            "leave_one_morphology_family_out",
                            "leave_one_noise_profile_out",
                        )
                    )
                }
            }
            for feature_index, (_, feature_set) in enumerate(methods)
        },
        "detector_graph_diagnostics": {
            "view_count": 9,
            "rays_per_view": 256,
            "neighbor_count": 8,
            "nearest_neighbor_distance_median": 0.015,
            "furthest_selected_neighbor_distance_median": 0.048,
        },
        "runtime": {
            "wall_seconds": 3.5,
            "maximum_rss_bytes": 429_000_000,
        },
    }
    output = tmp_path / "figure.png"
    build_figure(report, output)
    assert output.stat().st_size > 1000
    assert output.with_suffix(".pdf").stat().st_size > 1000
