from __future__ import annotations

import json
from pathlib import Path

import pytest

from site_tools import run_jacru_n1_9_global_contrast_screen as runner


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "demo_t16_operator/configs/jacru_n1_9_global_contrast_screen_postopen_v1.json"
)


def _config() -> dict:
    return json.loads(CONFIG.read_text())


def test_config_inherits_n18_gates_and_forbids_claims() -> None:
    config = _config()
    runner._validate_config(config, seed_limit=None)
    assert config["claim_boundary"][
        "close_rank6_camera_global_k_branch_if_both_fail"
    ] is True
    assert config["claim_boundary"]["may_train_a_learner"] is False
    assert config["claim_boundary"]["opens_new_geometry"] is False
    assert config["basis"]["minimum_accepted_rank"] == 6


def test_config_rejects_post_result_gate_relaxation() -> None:
    config = _config()
    config["design_selection_gate"][
        "extra_headroom_retention_over_component_damping_minimum"
    ] = 0.57
    with pytest.raises(ValueError, match="inherit the frozen N1.8"):
        runner._validate_config(config, seed_limit=None)


def test_config_rejects_source_redirect_or_hash_drift() -> None:
    redirected = _config()
    redirected["source_n1_8_result"] = "demo_t16_operator/results/other"
    with pytest.raises(ValueError, match="source path drifted"):
        runner._validate_config(redirected, seed_limit=None)
    rehashed = _config()
    rehashed["source_integrity"]["source_n1_8_summary_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source hash drifted"):
        runner._validate_config(rehashed, seed_limit=None)


def test_config_rejects_candidate_or_budget_drift() -> None:
    config = _config()
    config["candidate_schedules"]["residual_contrast_global_k6_total"][
        "refine_iterations"
    ] = 11
    with pytest.raises(ValueError, match="schedule drifted"):
        runner._validate_config(config, seed_limit=None)


def test_config_rejects_orthonormality_or_role_allowlist_drift() -> None:
    relaxed = _config()
    relaxed["basis"]["maximum_orthonormality_defect"] = 1.0
    with pytest.raises(ValueError, match="orthonormalization contract drifted"):
        runner._validate_config(relaxed, seed_limit=None)
    relabeled = _config()
    relabeled["selection_rule"]["eligible_representation_roles"].append(
        "REPRESENTATION_NO_GO"
    )
    with pytest.raises(ValueError, match="branch-closure rule drifted"):
        runner._validate_config(relabeled, seed_limit=None)


def test_config_rejects_cost_or_schur_applicability_drift() -> None:
    loosened = _config()
    loosened["operational_cost_gate"]["p90_ratio_maximum"] = 2.0
    with pytest.raises(ValueError, match="operational cost gate drifted"):
        runner._validate_config(loosened, seed_limit=None)
    mislabeled = _config()
    mislabeled["numerical_applicability"]["schur_gate"] = "ZERO_VIOLATIONS"
    with pytest.raises(ValueError, match="Schur applicability contract drifted"):
        runner._validate_config(mislabeled, seed_limit=None)


def test_operational_cost_gate_uses_paired_case_ratios() -> None:
    config = _config()
    rows = []
    for candidate_id, seconds in {
        "component_damping": (1.0, 1.0, 1.0),
        "candidate": (1.0, 1.2, 1.4),
    }.items():
        for seed, value in enumerate(seconds, start=1):
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "base_seed": seed,
                    "family": "smooth",
                    "end_to_end_seconds": value,
                }
            )
    gate = runner._operational_cost_gate(
        rows, candidate_id="candidate", config=config
    )
    assert gate["paired_case_count"] == 3
    assert gate["median_oracle_excluded_solver_path_seconds_ratio"] == pytest.approx(
        1.2
    )
    assert gate["p90_oracle_excluded_solver_path_seconds_ratio"] == pytest.approx(
        1.36
    )
    assert gate["passed"] is True
    assert gate["deployable_method_end_to_end_claim"] is False


def test_role_claim_label_does_not_oversell_nonnegative_adjoint_screen() -> None:
    assert runner._role_claim_label("SOLVER_AWARE_REPRESENTATION_ELIGIBLE") == (
        "NONNEGATIVE_SUPPORT_ADJOINT_SCREEN_ONLY"
    )
    assert runner._role_claim_label(
        "FORWARD_CORRECTION_REPRESENTATION_ELIGIBLE"
    ) == "FORWARD_CORRECTION_REPRESENTATION_ELIGIBLE"
    assert runner._role_claim_label("REPRESENTATION_NO_GO") == "REPRESENTATION_NO_GO"


def test_selection_closes_branch_when_both_candidates_fail() -> None:
    gates = [
        {
            "candidate_id": "a",
            "reconstruction_passed": False,
            "role": "REPRESENTATION_NO_GO",
        },
        {
            "candidate_id": "b",
            "reconstruction_passed": False,
            "role": "REPRESENTATION_NO_GO",
        },
    ]
    selection = runner._select(
        gates,
        aggregates={},
        decisive=True,
        eligible_roles=(
            "SOLVER_AWARE_REPRESENTATION_ELIGIBLE",
            "FORWARD_CORRECTION_REPRESENTATION_ELIGIBLE",
        ),
    )
    assert selection == {
        "authorized": False,
        "selected_candidate_id": None,
        "rank6_camera_global_k_branch_closed": True,
        "status": "N1_9_RANK6_CAMERA_GLOBAL_K_BRANCH_CLOSED",
    }


def test_selection_cannot_authorize_reconstruction_pass_with_no_go_role() -> None:
    gates = [
        {
            "candidate_id": "a",
            "reconstruction_passed": True,
            "operational_cost_passed": True,
            "role": "REPRESENTATION_NO_GO",
        }
    ]
    assert runner._select(
        gates,
        aggregates={},
        decisive=True,
        eligible_roles=(
            "SOLVER_AWARE_REPRESENTATION_ELIGIBLE",
            "FORWARD_CORRECTION_REPRESENTATION_ELIGIBLE",
        ),
    )["authorized"] is False


def test_selection_requires_operational_cost_noninferiority() -> None:
    gates = [
        {
            "candidate_id": "a",
            "reconstruction_passed": True,
            "operational_cost_passed": False,
            "role": "FORWARD_CORRECTION_REPRESENTATION_ELIGIBLE",
        }
    ]
    assert runner._select(
        gates,
        aggregates={},
        decisive=True,
        eligible_roles=("FORWARD_CORRECTION_REPRESENTATION_ELIGIBLE",),
    )["authorized"] is False


def test_smoke_selection_is_always_nondecisive() -> None:
    gates = [
        {
            "candidate_id": "a",
            "reconstruction_passed": True,
            "role": "FORWARD_CORRECTION_REPRESENTATION_ELIGIBLE",
        }
    ]
    selection = runner._select(
        gates,
        aggregates={"a": {"mean_field_gain_over_low_cgls24": 1.0}},
        decisive=False,
        eligible_roles=("FORWARD_CORRECTION_REPRESENTATION_ELIGIBLE",),
    )
    assert selection == {
        "authorized": False,
        "selected_candidate_id": None,
        "rank6_camera_global_k_branch_closed": False,
        "status": "N1_9_SMOKE_NONDECISIVE",
    }


def test_smoke_package_is_truth_free_and_nonconfirmatory(tmp_path: Path) -> None:
    output = tmp_path / "n19-global-contrast-smoke"
    old_argv = runner.sys.argv
    runner.sys.argv = [
        "run_jacru_n1_9_global_contrast_screen.py",
        "--config",
        str(CONFIG),
        "--output-dir",
        str(output),
        "--seed-limit",
        "2",
    ]
    try:
        assert runner.main() == 0
    finally:
        runner.sys.argv = old_argv
    expected = {
        "README.md",
        "aggregate_metrics.csv",
        "basis_diagnostics.csv",
        "case_manifest.csv",
        "case_metrics.csv",
        "checksums.sha256",
        "config_snapshot.json",
        "diagnostic.pdf",
        "diagnostic.png",
        "provenance.json",
        "summary.json",
        "target_diagnostics.csv",
    }
    assert {path.name for path in output.iterdir()} == expected
    summary = json.loads((output / "summary.json").read_text())
    assert summary["learner_was_trained"] is False
    assert summary["opens_new_geometry"] is False
    assert summary["finite_k_truth_search_was_run"] is False
    assert summary["opened_geometry_cluster_count"] == 2
    assert summary["opened_case_count"] == 4
    assert len(summary["candidate_gates"]) == 2
    assert summary["n1_7_development_case_identity_verified"] is True
    assert summary["n1_8_development_case_identity_verified"] is True
    assert summary["source_n1_8_no_auth_status_verified"] is True
    assert summary["status"] == "N1_9_SMOKE_NONDECISIVE"
    assert summary["selection"]["authorized"] is False
    assert summary["selection"]["rank6_camera_global_k_branch_closed"] is False
    assert summary["schur_gate_status"] == (
        "NOT_APPLICABLE_NO_COVARIANCE_OR_MAJORIZER"
    )
    assert summary["schur_violation_was_not_reported_as_zero"] is True
    assert summary["deployable_method_end_to_end_cost_was_not_claimed"] is True
    assert all(
        gate["reconstruction_check_count"] == 17
        for gate in summary["candidate_gates"]
    )
    assert all("operational_cost" in gate for gate in summary["candidate_gates"])
