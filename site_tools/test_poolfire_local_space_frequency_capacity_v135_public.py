from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "docs/poolfire_local_space_frequency_capacity_v135_public_summary.json"
EVIDENCE_PATH = ROOT / "operator-learning/current-evidence.json"
RESULT_PATH = ROOT / "docs/poolfire_local_space_frequency_capacity_v135_result_2026-08-10.md"
FIGURE_PATH = ROOT / "assets/figures/poolfire_local_space_frequency_capacity_v135.png"
PUBLIC_PAGES = [
    ROOT / "index.html",
    ROOT / "operator-learning/index.html",
    ROOT / "operator-learning/daily-progress.html",
    RESULT_PATH,
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_summary_preserves_partial_gain_and_strict_failure() -> None:
    summary = load_json(SUMMARY_PATH)
    gate = summary["strict_gate"]
    metrics = summary["metric_pass_counts"]

    assert summary["status"] == "FAIL_V135_LOCAL_SPACE_FREQUENCY_CAPACITY"
    assert summary["independent_status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_LOCAL_SPACE_FREQUENCY_V135_1"
    )
    assert gate["v134_passed"] == 2591
    assert gate["v135_passed"] == 3162
    assert gate["v135_rescued_from_v134"] == 571
    assert gate["v135_remaining_failures"] == 538
    assert gate["complete_trajectories_passed"] == 0
    assert metrics == {
        "field_relative_l2": 3700,
        "gradient_relative_l2": 3700,
        "interior_gradient_relative_l2": 3700,
        "reported_observation_relative_l2": 3162,
    }
    assert summary["remaining_failure_pattern"]["observation_only"] == 538
    assert summary["remaining_failure_pattern"]["field_or_gradient"] == 0
    assert summary["controls"]["deployment_visible_local_residual_joint_ls_passed"] == 0
    assert summary["controls"]["minimal_predictor_authorized"] is False
    assert summary["claim_boundary"]["algorithm_breakthrough"] is False
    assert summary["claim_boundary"]["real_bost"] is False


def test_camera_tail_and_independent_recomputation_are_consistent() -> None:
    summary = load_json(SUMMARY_PATH)
    by_camera = summary["cell_pass_by_camera_count"]
    independent = summary["independent_recomputation"]

    assert sum(row["v135_passed"] for row in by_camera.values()) == 3162
    assert sum(row["rescued"] for row in by_camera.values()) == 571
    assert sum(row["remaining"] for row in by_camera.values()) == 538
    assert by_camera["5"]["remaining"] == 442
    assert all(row["total"] == 925 for row in by_camera.values())
    assert independent["maximum_selected_metric_difference"] < 3e-15
    assert independent["maximum_coefficient_difference"] < 1e-11
    assert independent["maximum_diagnostic_scaled_difference"] < 1.0
    assert independent["maximum_summary_difference"] < 1e-15
    assert independent["exact_array_failures"] == 0


def test_current_evidence_and_pages_point_to_v135_in_both_languages() -> None:
    evidence = load_json(EVIDENCE_PATH)
    page_texts = [path.read_text(encoding="utf-8") for path in PUBLIC_PAGES]

    assert evidence["v135_local_space_frequency_formal_status"] == (
        "FAIL_V135_LOCAL_SPACE_FREQUENCY_CAPACITY"
    )
    assert evidence["v135_local_space_frequency_independent_status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_LOCAL_SPACE_FREQUENCY_V135_1"
    )
    assert evidence["metrics"]["v135_selected_cell_pass_count"] == 3162
    assert evidence["metrics"]["v135_observation_only_failure_count"] == 538
    assert evidence["metrics"]["v135_five_camera_remaining_failure_count"] == 442
    assert evidence["current_decision"]["v135_fixed_2x2_representation_closed"] is True
    assert evidence["current_decision"]["v135_minimal_predictor_authorized"] is False
    assert evidence["current_decision"]["v136_adaptive_residual_window_capacity_required"] is False
    assert evidence["current_decision"]["v136_residual_adaptive_local_window_evaluated"] is True

    for text in page_texts:
        assert "v135" in text
        assert "algorithm_breakthrough=false" in text

    joined = "\n".join(page_texts)
    assert "固定 2x2 局部" in joined
    assert "fixed 2x2" in joined.lower()
    assert "poolfire_local_space_frequency_capacity_v135_result_2026-08-10.md" in joined
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
