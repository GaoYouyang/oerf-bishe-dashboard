from __future__ import annotations

import json
from pathlib import Path

import pytest

from site_tools import run_jacru_n1_8_hybrid_design_screen as runner


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "demo_t16_operator/configs/jacru_n1_8_hybrid_design_screen_postopen_v1.json"
)


def _config() -> dict:
    return json.loads(CONFIG.read_text())


def test_design_config_forbids_new_geometry_and_learner() -> None:
    config = _config()
    runner._validate_config(config, seed_limit=None)
    assert config["claim_boundary"]["opens_new_geometry"] is False
    assert config["claim_boundary"]["may_train_a_learner"] is False
    assert config["total_correction_trust_region"]["applies_to_entire_correction"] is True
    assert config["oracle_screen"]["finite_k_truth_search_enabled"] is False


def test_design_config_rejects_evaluated_truth_in_basis() -> None:
    config = _config()
    config["basis"]["evaluated_case_truth_is_forbidden"] = False
    with pytest.raises(ValueError, match="truth must be forbidden"):
        runner._validate_config(config, seed_limit=None)


def test_design_config_rejects_unmatched_candidate_budget() -> None:
    config = _config()
    config["candidate_schedules"]["camera_block6_total"]["refine_iterations"] = 10
    with pytest.raises(ValueError, match="schedule drifted"):
        runner._validate_config(config, seed_limit=None)


def test_role_classification_fails_closed_on_negative_adjoint_gain() -> None:
    gates = _config()["design_selection_gate"]
    assert runner._representation_role(
        reconstruction_passed=True, adjoint_gain=-1e-12, gates=gates
    ) == "REPRESENTATION_NO_GO"
    assert runner._representation_role(
        reconstruction_passed=True, adjoint_gain=0.2, gates=gates
    ) == "SOLVER_AWARE_REPRESENTATION_ELIGIBLE"
    assert runner._representation_role(
        reconstruction_passed=True, adjoint_gain=0.5, gates=gates
    ) == "FORWARD_CORRECTION_REPRESENTATION_ELIGIBLE"
    assert runner._representation_role(
        reconstruction_passed=False, adjoint_gain=1.0, gates=gates
    ) == "REPRESENTATION_NO_GO"


def test_development_case_identity_detects_digest_drift() -> None:
    current = [
        {
            "partition": "development",
            "base_seed": 1,
            "family": "smooth",
            "case_id": "case-a",
            "geometry_digest": "geometry-a",
        }
    ]
    runner._assert_same_development_cases(current, [dict(current[0])])
    runner._assert_same_development_cases(
        current, [dict(current[0]), dict(current[0], base_seed=2)], allow_current_subset=True
    )
    changed = [dict(current[0], geometry_digest="geometry-b")]
    with pytest.raises(RuntimeError, match="digest drifted"):
        runner._assert_same_development_cases(current, changed)


def test_extra_headroom_uses_ratio_of_summed_error_differences() -> None:
    rows = []
    values = {
        "component_damping": (0.60, 0.40),
        "candidate": (0.45, 0.30),
        "exact_mismatch_oracle": (0.30, 0.20),
    }
    for candidate_id, errors in values.items():
        for seed, error in enumerate(errors, start=1):
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "base_seed": seed,
                    "family": "smooth",
                    "field_relative_l2": error,
                }
            )
    assert runner._extra_headroom_retention(rows, candidate_id="candidate") == pytest.approx(
        0.5
    )


def test_smoke_package_has_no_learner_or_new_geometry(tmp_path: Path) -> None:
    output = tmp_path / "n18-design-smoke"
    old_argv = runner.sys.argv
    runner.sys.argv = [
        "run_jacru_n1_8_hybrid_design_screen.py",
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
    assert len(summary["candidate_gates"]) == 5
    assert summary["radius_applies_to_entire_correction"] is True
    assert summary["n1_7_development_case_identity_verified"] is True
