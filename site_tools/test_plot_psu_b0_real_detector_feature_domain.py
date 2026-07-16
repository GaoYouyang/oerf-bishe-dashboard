from __future__ import annotations

from pathlib import Path

from site_tools.plot_psu_b0_real_detector_feature_domain import build_figure


def _comparison(center: float, nearest: float, outside: float) -> dict:
    return {
        "robust_center_distance": {"median": center},
        "nearest_reference_distance": {"median": nearest},
        "candidate_rows_outside_any_train_95pct_feature_envelope": outside,
        "mean_informative_feature_outside_fraction": 0.2 * outside,
    }


def test_build_figure_writes_png_and_pdf(tmp_path: Path) -> None:
    report = {
        "configuration": {
            "real_physical_field_count": 1,
            "real_camera_subset_count": 130,
        },
        "synthetic_internal_comparisons": {
            "risk_validation_vs_risk_train": _comparison(1.2, 0.7, 0.3),
            "risk_calibration_vs_risk_train": _comparison(1.4, 0.8, 0.6),
        },
        "real_measurement_comparison": {
            **_comparison(3.1, 1.9, 1.0),
            "highest_outside_fraction_features": [
                {
                    "feature": f"log_local_jacobian_rms_{name}",
                    "train_q025": 1.0 + index,
                    "train_q975": 2.0 + index,
                    "candidate_median": 3.0 + index,
                }
                for index, name in enumerate(
                    ("mean", "min", "max", "std")
                )
            ],
        },
        "runtime": {
            "wall_seconds": 1.2,
            "maximum_rss_bytes": 398_000_000,
        },
    }
    output = tmp_path / "domain.png"
    build_figure(report, output)
    assert output.stat().st_size > 1000
    assert output.with_suffix(".pdf").stat().st_size > 1000


def test_build_figure_accepts_private_report_configuration(tmp_path: Path) -> None:
    report = {
        "configuration_public": {
            "real_physical_field_count": 1,
            "real_camera_subset_count": 130,
        },
        "synthetic_internal_comparisons": {
            "risk_validation_vs_risk_train": _comparison(1.2, 0.7, 0.3),
            "risk_calibration_vs_risk_train": _comparison(1.4, 0.8, 0.6),
        },
        "real_measurement_comparison": {
            **_comparison(3.1, 1.9, 1.0),
            "highest_outside_fraction_features": [
                {
                    "feature": f"neighbor_contrast_to_signal_ratio_{name}",
                    "train_q025": 0.5 + index,
                    "train_q975": 1.0 + index,
                    "candidate_median": 1.5 + index,
                }
                for index, name in enumerate(
                    ("mean", "min", "max", "std")
                )
            ],
        },
        "runtime": {
            "wall_seconds": 1.2,
            "maximum_rss_bytes": 398_000_000,
        },
    }
    output = tmp_path / "private-domain.png"
    build_figure(report, output)
    assert output.stat().st_size > 1000
