from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "docs/poolfire_field_lift_camera_mixing_v132_public_summary.json"
EVIDENCE_PATH = ROOT / "operator-learning/current-evidence.json"
FIGURE_PATH = ROOT / "assets/figures/poolfire_field_lift_camera_mixing_v132.png"
PUBLIC_PAGES = [
    ROOT / "index.html",
    ROOT / "operator-learning/index.html",
    ROOT / "operator-learning/daily-progress.html",
    ROOT / "docs/poolfire_field_lift_camera_mixing_v132_result_2026-08-10.md",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_public_summary_preserves_capacity_failure_and_claim_boundary() -> None:
    summary = load_json(SUMMARY_PATH)

    assert summary["formal_status"] == "FAIL_V132_FIELD_LIFT_CAMERA_MIXING_CAPACITY"
    assert summary["independent_status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_FIELD_LIFT_CAMERA_MIXING_V132"
    )
    assert summary["scope"]["sample_count"] == 3700
    assert summary["gate"]["oracle_accuracy_gate_passed"] is False
    assert summary["gate"]["rms_control_accuracy_gate_passed"] is False
    assert summary["gate"]["coefficient_predictor_authorized"] is False
    assert summary["claim_boundary"]["per_camera_scalar_mixing_capacity"] is False
    assert summary["claim_boundary"]["algorithm_breakthrough"] is False
    assert summary["claim_boundary"]["paper_success"] is False


def test_independent_recomputation_is_precise_without_overclaiming() -> None:
    independent = load_json(SUMMARY_PATH)["independent_recomputation"]

    assert independent["maximum_oracle_coefficient_difference"] < 1e-13
    assert independent["maximum_diagnostic_difference"] < 1e-13
    assert independent["maximum_oracle_metric_difference"] < 1e-13
    assert independent["maximum_summary_difference"] < 1e-13
    assert independent["maximum_k1_residual_difference"] < 1e-12
    assert independent["validation_truth_read"] is False
    assert independent["test_truth_read"] is False
    assert independent["end_to_end_physics_independence_proven"] is False


def test_historical_v132_evidence_remains_traceable_after_v133() -> None:
    evidence = load_json(EVIDENCE_PATH)
    page_texts = [path.read_text(encoding="utf-8") for path in PUBLIC_PAGES]

    assert evidence["v132_field_lift_camera_mixing_formal_status"] == (
        "FAIL_V132_FIELD_LIFT_CAMERA_MIXING_CAPACITY"
    )
    assert evidence["v132_field_lift_camera_mixing_independent_status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_FIELD_LIFT_CAMERA_MIXING_V132"
    )
    assert evidence["metrics"]["v132_sample_count"] == 3700
    assert evidence["metrics"]["v132_trajectory_gate_passed"] == 0
    assert evidence["current_decision"]["v132_camera_coefficient_predictor_authorized"] is False

    for text in page_texts:
        assert "v132" in text
        assert "algorithm_breakthrough=false" in text

    joined = "\n".join(page_texts)
    assert "poolfire_field_lift_camera_mixing_v132_result_2026-08-10.md" in joined
    assert FIGURE_PATH.stat().st_size > 100_000


def test_public_artifacts_do_not_expose_private_execution_identifiers() -> None:
    paths = PUBLIC_PAGES + [SUMMARY_PATH, EVIDENCE_PATH]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    forbidden_fragments = [
        "/Users/",
        "private_results",
        "private_worktrees",
        "TEST_RELEASE.json",
        "VALIDATED_READY",
    ]
    assert all(fragment not in text for fragment in forbidden_fragments)
    assert re.search(r"\b[0-9a-f]{40,64}\b", text, flags=re.IGNORECASE) is None
