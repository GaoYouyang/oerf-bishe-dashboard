from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "docs/poolfire_ray_overlap_epipolar_capacity_v138_public_summary.json"
EVIDENCE_PATH = ROOT / "operator-learning/current-evidence.json"
RESULT_PATH = ROOT / "docs/poolfire_ray_overlap_epipolar_capacity_v138_result_2026-08-10.md"
FIGURE_PATH = ROOT / "assets/figures/poolfire_ray_overlap_epipolar_capacity_v138.png"
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

    assert summary["status"] == "FAIL_V138_RAY_OVERLAP_EPIPOLAR_CAPACITY"
    assert summary["independent_status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_RAY_OVERLAP_EPIPOLAR_V138_3"
    )
    assert gate["v137_passed"] == 3351
    assert gate["v138_passed"] == 3397
    assert gate["v138_rescued_from_v137"] == 46
    assert gate["v138_remaining_failures"] == 303
    assert gate["complete_trajectories_passed"] == 0
    assert metrics == {
        "field_relative_l2": 3700,
        "gradient_relative_l2": 3700,
        "interior_gradient_relative_l2": 3700,
        "reported_observation_relative_l2": 3397,
    }
    assert summary["controls"]["deployment_visible_ray_overlap_joint_ls_passed"] == 0
    assert summary["controls"]["minimal_predictor_authorized"] is False
    assert summary["claim_boundary"]["algorithm_breakthrough"] is False


def test_sparse_view_tail_and_independent_recomputation_are_consistent() -> None:
    summary = load_json(SUMMARY_PATH)
    by_camera = summary["cell_pass_by_camera_count"]
    independent = summary["independent_recomputation"]

    assert sum(row["v138_passed"] for row in by_camera.values()) == 3397
    assert sum(row["rescued"] for row in by_camera.values()) == 46
    assert sum(row["remaining"] for row in by_camera.values()) == 303
    assert by_camera["5"]["remaining"] == 298
    assert by_camera["9"]["remaining"] == 0
    assert by_camera["12"]["remaining"] == 0
    assert independent["maximum_selected_metric_difference"] < 7e-15
    assert independent["maximum_coefficient_difference"] < 6e-11
    assert independent["exact_array_failures"] == 0
    assert independent["formal_tree_unchanged"] is True
    assert summary["audit_note"]["all_15_numeric_arrays_byte_identical"] is True


def test_current_evidence_and_pages_preserve_v138_history_in_both_languages() -> None:
    evidence = load_json(EVIDENCE_PATH)
    page_texts = [path.read_text(encoding="utf-8") for path in PUBLIC_PAGES]

    assert evidence["v138_ray_overlap_epipolar_formal_status"] == (
        "FAIL_V138_RAY_OVERLAP_EPIPOLAR_CAPACITY"
    )
    assert evidence["v138_ray_overlap_epipolar_independent_status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_RAY_OVERLAP_EPIPOLAR_V138_3"
    )
    assert evidence["metrics"]["v138_selected_cell_pass_count"] == 3397
    assert evidence["metrics"]["v138_observation_only_failure_count"] == 303
    assert evidence["metrics"]["v138_five_camera_remaining_failure_count"] == 298
    assert evidence["current_decision"]["v138_ray_average_representation_closed"] is True
    assert evidence["current_decision"]["v138_minimal_predictor_authorized"] is False

    for text in page_texts:
        assert "v138" in text
        assert "algorithm_breakthrough=false" in text

    joined = "\n".join(page_texts)
    assert "深度" in joined
    assert "depth-resolved" in joined
    assert "poolfire_ray_overlap_epipolar_capacity_v138_result_2026-08-10.md" in joined
    assert "poolfire_ray_overlap_epipolar_capacity_v138.png" in joined
    assert FIGURE_PATH.stat().st_size > 100_000


def test_public_artifacts_do_not_expose_private_execution_identifiers() -> None:
    paths = PUBLIC_PAGES + [SUMMARY_PATH, EVIDENCE_PATH]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    forbidden = ["/Users/", "private_results", "private_worktrees", "VALIDATED_READY"]
    assert all(fragment not in text for fragment in forbidden)
    assert re.search(r"\b[0-9a-f]{40,64}\b", text, flags=re.IGNORECASE) is None
