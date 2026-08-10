from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "docs/poolfire_residual_adaptive_local_window_capacity_v136_public_summary.json"
EVIDENCE_PATH = ROOT / "operator-learning/current-evidence.json"
RESULT_PATH = ROOT / "docs/poolfire_residual_adaptive_local_window_capacity_v136_result_2026-08-10.md"
FIGURE_PATH = ROOT / "assets/figures/poolfire_residual_adaptive_local_window_capacity_v136.png"
PUBLIC_PAGES = [
    ROOT / "index.html",
    ROOT / "operator-learning/index.html",
    ROOT / "operator-learning/daily-progress.html",
    RESULT_PATH,
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_summary_preserves_small_gain_and_strict_failure() -> None:
    summary = load_json(SUMMARY_PATH)
    gate = summary["strict_gate"]
    metrics = summary["metric_pass_counts"]

    assert summary["status"] == "FAIL_V136_RESIDUAL_ADAPTIVE_LOCAL_WINDOW_CAPACITY"
    assert summary["independent_status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_RESIDUAL_ADAPTIVE_LOCAL_WINDOW_V136_1"
    )
    assert gate["v135_passed"] == 3162
    assert gate["v136_passed"] == 3215
    assert gate["v136_rescued_from_v135"] == 53
    assert gate["v136_remaining_failures"] == 485
    assert gate["complete_trajectories_passed"] == 0
    assert metrics == {
        "field_relative_l2": 3700,
        "gradient_relative_l2": 3700,
        "interior_gradient_relative_l2": 3700,
        "reported_observation_relative_l2": 3215,
    }
    assert summary["controls"]["deployment_visible_adaptive_projection_only_passed"] == 0
    assert summary["controls"]["minimal_predictor_authorized"] is False
    assert summary["claim_boundary"]["algorithm_breakthrough"] is False
    assert summary["claim_boundary"]["real_bost"] is False


def test_camera_tail_mechanism_and_independent_recomputation_are_consistent() -> None:
    summary = load_json(SUMMARY_PATH)
    by_camera = summary["cell_pass_by_camera_count"]
    diagnosis = summary["mechanism_diagnosis"]
    independent = summary["independent_recomputation"]

    assert sum(row["v136_passed"] for row in by_camera.values()) == 3215
    assert sum(row["rescued"] for row in by_camera.values()) == 53
    assert sum(row["remaining"] for row in by_camera.values()) == 485
    assert by_camera["5"]["remaining"] == 409
    assert diagnosis["cells_with_observation_improvement"] == 447
    assert diagnosis["cells_unchanged"] == 91
    assert independent["maximum_center_difference"] == 0
    assert independent["maximum_half_width_difference"] == 0
    assert independent["maximum_selected_metric_difference"] < 2e-15
    assert independent["maximum_coefficient_difference"] < 1e-11
    assert independent["maximum_diagnostic_scaled_difference"] < 1.0
    assert independent["exact_array_failures"] == 0


def test_current_evidence_and_pages_preserve_v136_as_archived_parent() -> None:
    evidence = load_json(EVIDENCE_PATH)
    page_texts = [path.read_text(encoding="utf-8") for path in PUBLIC_PAGES]

    assert evidence["metrics"]["v136_selected_cell_pass_count"] == 3215
    assert evidence["metrics"]["v136_observation_only_failure_count"] == 485
    assert evidence["metrics"]["v136_five_camera_remaining_failure_count"] == 409
    assert evidence["current_decision"]["v136_residual_centroid_width_representation_closed"] is True
    assert evidence["current_decision"]["v136_minimal_predictor_authorized"] is False
    assert evidence["current_decision"]["v136_residual_adaptive_local_window_independently_recomputed"] is True

    for text in page_texts:
        assert "v136" in text
        assert "algorithm_breakthrough=false" in text

    joined = "\n".join(page_texts)
    assert "正负相位" in joined
    assert "sign/phase" in joined.lower()
    assert "poolfire_residual_adaptive_local_window_capacity_v136_result_2026-08-10.md" in joined
    assert "poolfire_residual_adaptive_local_window_capacity_v136.png" in joined
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
