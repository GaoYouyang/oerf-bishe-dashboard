from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "general_operator_research_lab.html"
RESULT = (
    ROOT
    / "demo_t16_operator/results/jacru_n1_7_geometry_krylov_postopen_full1"
)
RESULT_N18 = (
    ROOT
    / "demo_t16_operator/results/jacru_n1_8_hybrid_design_screen_postopen_audit_amended_full1"
)
RESULT_N19 = (
    ROOT
    / "demo_t16_operator/results/jacru_n1_9_global_contrast_postopen_full1"
)


def test_focused_page_exposes_n1_7_verdict_without_success_language() -> None:
    html = PAGE.read_text(encoding="utf-8")
    assert 'id="n1-7-result"' in html
    assert "REPRESENTATION_NO_GO_STOP_BEFORE_LEARNER" in html
    assert "+4.83%" in html
    assert "56.72%" in html
    assert "33,780F/33,780Aᵀ" in html
    assert "74,010F/74,010Aᵀ" in html
    assert "+6.186%" in html
    assert "16/17" in html
    assert "learner 未启动" in html
    assert "docs%2Fjacru_n1_7_geometry_krylov_no_go_2026-07-18.md" in html
    assert "docs%2Fjacru_n1_7_advisor_review_brief_2026-07-18.md" in html
    assert "docs%2Fjacru_n1_7_radius_sensitivity_audit_2026-07-18.md" in html
    assert "jacru_n1_7_radius_4x_posthoc_audit_full1/summary.json" in html


def test_page_machine_summary_matches_frozen_no_go() -> None:
    summary = json.loads((RESULT / "summary.json").read_text())
    assert summary["status"] == "REPRESENTATION_NO_GO_STOP_BEFORE_LEARNER"
    assert summary["primary_representation_gate"]["passed"] is False
    assert summary["finite_k_diagnostic_gate"]["passed"] is False
    assert summary["learner_was_trained"] is False
    assert summary["opens_ood_fresh_or_final"] is False
    assert summary["primary_development_aggregate"][
        "mean_field_gain_over_low_cgls24"
    ] < 0.05
    assert summary["finite_k_evaluator_total_forward_calls"] == 33780
    assert (
        summary["finite_k_evaluator_total_forward_calls"]
        == summary["finite_k_evaluator_total_adjoint_calls"]
    )


def test_public_n1_7_package_contains_no_model_checkpoint() -> None:
    forbidden = {"pt", "pth", "ckpt", "npz", "npy", "mat"}
    assert RESULT.is_dir()
    assert not {path.suffix.lower().lstrip(".") for path in RESULT.iterdir()} & forbidden


def test_radius_audit_is_posthoc_and_does_not_authorize_a_learner() -> None:
    audit = ROOT / "demo_t16_operator/results/jacru_n1_7_radius_4x_posthoc_audit_full1"
    config = json.loads(
        (
            ROOT
            / "demo_t16_operator/configs/jacru_n1_7_radius_4x_posthoc_audit_v1.json"
        ).read_text()
    )
    summary = json.loads((audit / "summary.json").read_text())
    assert config["audit_context"]["may_change_n1_7_verdict"] is False
    assert config["audit_context"]["may_select_a_learner"] is False
    assert config["claim_boundary"]["may_authorize_later_learner_preregistration"] is False
    assert summary["evidence_level"].endswith("POSTHOC_RADIUS_SENSITIVITY")
    assert summary["status"] == "REPRESENTATION_NO_GO_STOP_BEFORE_LEARNER"
    assert summary["learner_was_trained"] is False
    assert summary["primary_representation_gate"]["passed_count"] == 15
    assert summary["finite_k_diagnostic_gate"]["passed_count"] == 16
    assert summary["finite_k_evaluator_development_forward_calls"] == 74010
    assert summary["finite_k_evaluator_calibration_forward_calls"] == 42680
    assert summary["finite_k_evaluator_package_forward_calls"] == 116690


def test_focused_page_exposes_n1_8_no_auth_without_success_language() -> None:
    html = PAGE.read_text(encoding="utf-8")
    assert 'id="n1-8-result"' in html
    assert "NO-AUTH" in html
    assert "+6.343%" in html
    assert "57.071%" in html
    assert "9.474%" in html
    assert "fresh 均未启动" in html
    assert "docs%2Fjacru_n1_8_hybrid_design_no_auth_2026-07-18.md" in html
    assert "docs%2Fjacru_n1_8_advisor_review_brief_2026-07-18.md" in html
    assert "docs%2Fjacru_n1_8_hybrid_design_freeze_2026-07-18.md" in html
    assert "jacru_n1_8_hybrid_design_screen_postopen_audit_amended_full1/summary.json" in html
    assert "jacru_n1_8_hybrid_design_screen_postopen_audit_amended_full1/checksums.sha256" in html


def test_n1_8_machine_summary_keeps_fresh_and_learner_closed() -> None:
    summary = json.loads((RESULT_N18 / "summary.json").read_text())
    assert summary["status"] == "NO_N1_8_CONFIRMATION_AUTHORIZATION"
    assert summary["selection"]["authorized"] is False
    assert summary["selection"]["selected_candidate_id"] is None
    assert len(summary["candidate_gates"]) == 5
    assert summary["learner_was_trained"] is False
    assert summary["opens_new_geometry"] is False
    assert summary["n1_7_development_case_identity_verified"] is True
    camera = next(
        row
        for row in summary["candidate_gates"]
        if row["candidate_id"] == "camera_block6_total_measurement_oracle"
    )
    assert camera["reconstruction_passed_count"] == 16
    assert camera["reconstruction_passed"] is False
    assert camera["physics_fidelity_passed"] is False
    assert camera["extra_headroom_retention_over_component_damping"] < 0.6


def test_public_n1_8_package_contains_no_model_checkpoint_or_array() -> None:
    forbidden = {"pt", "pth", "ckpt", "npz", "npy", "mat"}
    assert RESULT_N18.is_dir()
    assert not {
        path.suffix.lower().lstrip(".") for path in RESULT_N18.iterdir()
    } & forbidden


def test_focused_page_exposes_n1_9_branch_closure_without_success_language() -> None:
    html = PAGE.read_text(encoding="utf-8")
    assert 'id="n1-9-result"' in html
    assert "N1.9 · BRANCH CLOSED" in html
    assert "+6.207%" in html
    assert "51.408%" in html
    assert "35.787%" in html
    assert "learner 与新 split 均未启动" in html
    assert "Residual 算法胜出" in html
    assert "docs%2Fjacru_n1_9_global_contrast_branch_closed_2026-07-18.md" in html
    assert "docs%2Fjacru_n1_9_advisor_review_brief_2026-07-18.md" in html
    assert "docs%2Fjacru_n1_9_global_contrast_freeze_2026-07-18.md" in html
    assert "jacru_n1_9_global_contrast_postopen_full1/summary.json" in html
    assert "jacru_n1_9_global_contrast_postopen_full1/checksums.sha256" in html


def test_n1_9_machine_summary_closes_only_the_frozen_candidate_pair() -> None:
    summary = json.loads((RESULT_N19 / "summary.json").read_text())
    assert summary["status"] == "N1_9_RANK6_CAMERA_GLOBAL_K_BRANCH_CLOSED"
    assert summary["selection"]["authorized"] is False
    assert summary["selection"]["selected_candidate_id"] is None
    assert summary["selection"]["rank6_camera_global_k_branch_closed"] is True
    assert summary["learner_was_trained"] is False
    assert summary["opens_new_geometry"] is False
    assert summary["n1_7_development_case_identity_verified"] is True
    assert summary["n1_8_development_case_identity_verified"] is True
    assert summary["schur_gate_status"] == (
        "NOT_APPLICABLE_NO_COVARIANCE_OR_MAJORIZER"
    )
    assert summary["deployable_method_end_to_end_cost_was_not_claimed"] is True
    gates = {row["candidate_id"]: row for row in summary["candidate_gates"]}
    residual = gates["residual_contrast_global_k6_total_measurement_oracle"]
    damping = gates["damping_contrast_global_k6_total_measurement_oracle"]
    assert residual["reconstruction_passed_count"] == 16
    assert damping["reconstruction_passed_count"] == 15
    assert residual["reconstruction_checks"]["extra_headroom_retention"] is False
    assert damping["reconstruction_checks"]["exact_gain_retention"] is False
    assert damping["reconstruction_checks"]["extra_headroom_retention"] is False
    assert residual["operational_cost_passed"] is True
    assert damping["operational_cost_passed"] is True


def test_public_n1_9_package_contains_no_checkpoint_or_array() -> None:
    forbidden = {"pt", "pth", "ckpt", "npz", "npy", "mat"}
    assert RESULT_N19.is_dir()
    assert not {
        path.suffix.lower().lstrip(".") for path in RESULT_N19.iterdir()
    } & forbidden
