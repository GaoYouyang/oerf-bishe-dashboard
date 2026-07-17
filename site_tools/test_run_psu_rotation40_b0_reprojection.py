from __future__ import annotations

import numpy as np
import pytest

from site_tools.run_psu_rotation40_b0_reprojection import metric_row, validate_config


def _config() -> dict:
    return {
        "schema_version": "psu-rotation40-b0-reprojection-config-1.0",
        "status": "FROZEN_AFTER_NON_TUNING_3K_INTERFACE_SMOKE_BEFORE_ALL_ACTIVE_SCORE",
        "dataset": {"camera_ids": [2, 3, 4], "selection_mode": "all_rotation40_active_rows"},
        "forward": {
            "grid_shape_zyx": [32, 32, 32],
            "grid_minimum_xyz_m": [-0.11, -0.11, -0.11],
            "grid_maximum_xyz_m": [0.11, 0.11, 0.11],
            "gauge": "zero_one_voxel_outer_boundary",
            "finite_aperture_sample_count": 16,
            "chunk_rays": 32768,
            "dtype": "float64",
            "device": "cpu",
            "torch_threads": 8,
        },
        "claim_firewall": {
            "algorithm_superiority": False,
            "candidate_compared": False,
            "experimental_field_truth_available": False,
            "final_rotations_opened": False,
            "field_relative_l2_reported": False,
            "publish_predictions_or_measurement_arrays": False,
        },
    }


def test_metric_row_exact_and_nonzero_residual() -> None:
    measured = np.array([[1.0, 0.0], [0.0, 1.0]])
    exact = metric_row(measured, measured)
    assert exact["vector_relative_l2"] == 0.0
    shifted = metric_row(measured, np.zeros_like(measured))
    assert shifted["vector_relative_l2"] == 1.0
    assert shifted["component_rmse_px"] == pytest.approx(np.sqrt(0.5))


def test_metric_row_rejects_zero_measurement_norm() -> None:
    with pytest.raises(ValueError, match="norm"):
        metric_row(np.zeros((2, 2)), np.zeros((2, 2)))


def test_config_rejects_post_smoke_row_subsampling() -> None:
    config = _config()
    validate_config(config)
    config["dataset"]["selection_mode"] = "quantiles"
    with pytest.raises(ValueError, match="full active-row"):
        validate_config(config)
