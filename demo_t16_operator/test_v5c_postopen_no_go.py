from __future__ import annotations

from pathlib import Path

import pytest

from demo_t16_operator.analyze_v5c_postopen_no_go import summarize


ROOT = Path(__file__).resolve().parent


def test_frozen_v5c_no_go_semantics_are_reproduced() -> None:
    diagnosis, plotting = summarize(
        ROOT / "results" / "v5c_nested_crossview_first_open"
    )
    primary = diagnosis["primary_findings"]
    assert diagnosis["verdict"] == "NO_GO_JOINT_RADIUS_KAPPA_CROSSVIEW_SELECTION"
    assert primary["nearest_bank_match_blocks"] == 3
    assert primary["fixed_ridge_nearest_bank_match_blocks"] == 5
    assert primary["operator_matrix_oracle_match_blocks"] == 6
    assert primary["clean_truth_oracle_match_blocks"] == 6
    assert primary["noisy_truth_oracle_match_blocks"] == 6
    assert primary["mean_true_camera_deletion_radius_stability"] == pytest.approx(
        0.375
    )
    assert primary["fully_camera_deletion_stable_blocks"] == 0
    assert primary["selected_kappa_upper_boundary_blocks"] == 6
    assert primary["strictly_decreasing_best_cv_profiles"] == 6
    assert primary["changed_operator_blocks"] == 2
    assert primary["no_action_sample_rows"] == 32
    assert primary["accepted_sample_rows"] == 0
    assert diagnosis["changed_candidate_only"][
        "raw_field_harm_below_minus_1_count"
    ] == 7
    assert diagnosis["changed_candidate_only"]["raw_audit_increase_count"] == 8
    assert len(plotting["blocks"]) == 6
    assert len(plotting["changed_samples"]) == 16
