from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "docs/poolfire_detector_spectral_capacity_v133_public_summary.json"
EVIDENCE_PATH = ROOT / "operator-learning/current-evidence.json"
FIGURE_PATH = ROOT / "assets/figures/poolfire_detector_spectral_capacity_v133.png"
PUBLIC_PAGES = [
    ROOT / "index.html",
    ROOT / "operator-learning/index.html",
    ROOT / "operator-learning/daily-progress.html",
    ROOT / "docs/poolfire_detector_spectral_capacity_v133_result_2026-08-10.md",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_summary_preserves_strict_failure_and_localized_signal() -> None:
    summary = load_json(SUMMARY_PATH)

    assert summary["formal_status"] == "FAIL_V133_DETECTOR_SPECTRAL_CAPACITY"
    assert summary["independent_status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_DETECTOR_SPECTRAL_CAPACITY_V133"
    )
    assert summary["gate"]["oracle_cell_pass_count"] == 2353
    assert summary["gate"]["oracle_cell_total"] == 3700
    assert summary["gate"]["oracle_trajectory_gate_passed"] is False
    assert summary["gate"]["spectral_ls_control_cell_pass_count"] == 0
    assert summary["metric_cell_pass_counts"] == {
        "field": 3700,
        "full_gradient": 3700,
        "interior_gradient": 3700,
        "observation": 2353,
    }
    assert summary["failure_patterns"]["observation_only_failure"] == 1347
    assert summary["claim_boundary"]["algorithm_breakthrough"] is False
    assert summary["claim_boundary"]["coefficient_predictor_authorized"] is False


def test_camera_counts_and_independent_recomputation_are_consistent() -> None:
    summary = load_json(SUMMARY_PATH)
    by_camera = summary["cell_pass_by_camera_count"]
    independent = summary["independent_recomputation"]

    assert sum(row["v132_passed"] for row in by_camera.values()) == 61
    assert sum(row["v133_passed"] for row in by_camera.values()) == 2353
    assert all(row["total"] == 925 for row in by_camera.values())
    assert independent["maximum_oracle_coefficient_difference"] < 2e-12
    assert independent["maximum_diagnostic_difference"] < 2e-10
    assert independent["maximum_oracle_metric_difference"] < 2e-15
    assert independent["candidate_receipt_failures"] == 0
    assert independent["validation_truth_read"] is False
    assert independent["test_truth_read"] is False
    assert independent["end_to_end_physics_independence_proven"] is False


def test_current_evidence_and_pages_point_to_v133() -> None:
    evidence = load_json(EVIDENCE_PATH)
    page_texts = [path.read_text(encoding="utf-8") for path in PUBLIC_PAGES]

    assert evidence["formal_status"] == "FAIL_V133_DETECTOR_SPECTRAL_CAPACITY"
    assert evidence["engineering_status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_DETECTOR_SPECTRAL_CAPACITY_V133"
    )
    assert evidence["metrics"]["v133_oracle_cell_pass_count"] == 2353
    assert evidence["metrics"]["v133_observation_only_failure_count"] == 1347
    assert evidence["current_decision"]["v133_coefficient_predictor_authorized"] is False

    for text in page_texts:
        assert "v133" in text
        assert "algorithm_breakthrough=false" in text

    joined = "\n".join(page_texts)
    assert "poolfire_detector_spectral_capacity_v133_result_2026-08-10.md" in joined
    assert "poolfire_detector_spectral_capacity_v133.png" in joined
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
