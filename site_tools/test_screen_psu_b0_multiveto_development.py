from __future__ import annotations

import numpy as np

from demo_t16_operator.psu_b0_residual_risk import (
    RISK_FEATURE_NAMES,
    RidgeRiskFit,
)
from site_tools.screen_psu_b0_multiveto_development import (
    selection_metrics,
    select_candidate,
)


def test_selection_metrics_separate_coverage_from_selected_harm() -> None:
    metrics = selection_metrics(
        np.asarray([2.0, -3.0, -8.0]),
        np.asarray([True, True, False]),
        np.asarray([6, 6, 7]),
    )
    assert metrics["coverage"] == 2 / 3
    assert metrics["harm_over_one_percent_count"] == 1
    assert metrics["harm_over_one_percent_rate"] == 1 / 3
    assert metrics["accepted_minimum_raw_gain_percent"] == -3.0


def test_candidate_selection_prefers_stricter_equal_performance_veto() -> None:
    width = len(RISK_FEATURE_NAMES)
    fit = RidgeRiskFit(
        feature_mean=np.zeros(width),
        feature_scale=np.ones(width),
        coefficients=np.zeros(width),
        intercept=2.0,
        ridge_lambda=1.0,
        validation_rmse=0.0,
    )
    validation = {
        "features": np.zeros((4, width)),
        "gain": np.asarray([2.0, 2.0, 2.0, 2.0]),
        "seed": np.asarray([1, 1, 1, 1]),
        "views": np.asarray([6, 7, 8, 9]),
        "spectral": np.asarray([0.1, 0.2, 0.3, 0.4]),
        "camera": np.asarray([0.1, 0.2, 0.3, 0.4]),
    }
    selected = select_candidate(
        validation=validation,
        fit=fit,
        quantile_by_seed={1: 1.0},
        distance_threshold=10.0,
        minimum_lower_gain_percent=0.0,
        spectral_grid=[0.4, 1e9],
        camera_grid=[0.4, 1e9],
        six_view_backoff_grid=[0.0, 0.5],
        coverage_minimum=0.2,
        overall_harm_maximum=0.05,
        per_view_harm_maximum=0.1,
    )["selected"]
    assert selected["spectral_stress_threshold"] == 0.4
    assert selected["camera_stress_threshold"] == 0.4
    assert selected["six_view_extra_margin_percent"] == 0.0
