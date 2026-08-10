from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "docs/poolfire_depth_resolved_ray_consistency_capacity_v139_public_summary.json"
EVIDENCE_PATH = ROOT / "operator-learning/current-evidence.json"
RESULT_PATH = ROOT / "docs/poolfire_depth_resolved_ray_consistency_capacity_v139_result_2026-08-11.md"
FIGURE_PATH = ROOT / "assets/figures/poolfire_depth_resolved_ray_consistency_capacity_v139.png"
PUBLIC_PAGES = [
    ROOT / "index.html",
    ROOT / "operator-learning/index.html",
    ROOT / "operator-learning/daily-progress.html",
    RESULT_PATH,
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_summary_preserves_large_gain_and_strict_failure() -> None:
    summary = load_json(SUMMARY_PATH)
    gate = summary["strict_gate"]
    metrics = summary["metric_pass_counts"]

    assert summary["status"] == "FAIL_V139_DEPTH_RESOLVED_RAY_CONSISTENCY_CAPACITY"
    assert summary["independent_status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_DEPTH_RESOLVED_RAY_CONSISTENCY_V139_3"
    )
    assert gate["v138_passed"] == 3397
    assert gate["v139_passed"] == 3549
    assert gate["v139_rescued_from_v138"] == 152
    assert gate["v139_remaining_failures"] == 151
    assert gate["complete_trajectories_passed"] == 0
    assert metrics == {
        "field_relative_l2": 3700,
        "gradient_relative_l2": 3700,
        "interior_gradient_relative_l2": 3700,
        "reported_observation_relative_l2": 3549,
    }
    assert summary["controls"]["deployment_visible_depth_resolved_joint_ls_passed"] == 0
    assert summary["controls"]["minimal_predictor_authorized"] is False
    assert summary["claim_boundary"]["algorithm_breakthrough"] is False


def test_five_camera_tail_and_independent_recomputation_are_consistent() -> None:
    summary = load_json(SUMMARY_PATH)
    by_camera = summary["cell_pass_by_camera_count"]
    independent = summary["independent_recomputation"]

    assert sum(row["v139_passed"] for row in by_camera.values()) == 3549
    assert sum(row["rescued"] for row in by_camera.values()) == 152
    assert sum(row["remaining"] for row in by_camera.values()) == 151
    assert by_camera["5"]["remaining"] == 151
    assert all(by_camera[count]["remaining"] == 0 for count in ("7", "9", "12"))
    assert independent["maximum_selected_metric_difference"] < 1.6e-11
    assert independent["maximum_coefficient_difference"] < 2.7e-8
    assert independent["exact_array_failures"] == 0
    assert independent["formal_tree_unchanged"] is True


def test_current_evidence_and_pages_point_to_v139_in_both_languages() -> None:
    evidence = load_json(EVIDENCE_PATH)
    page_texts = [path.read_text(encoding="utf-8") for path in PUBLIC_PAGES]

    assert evidence["formal_status"] == "FAIL_V139_DEPTH_RESOLVED_RAY_CONSISTENCY_CAPACITY"
    assert evidence["engineering_status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_DEPTH_RESOLVED_RAY_CONSISTENCY_V139_3"
    )
    assert evidence["headline"].startswith("v139 backprojects signed K1 residuals")
    assert evidence["headline_zh"].startswith("v139 将 signed K1 residual")
    assert evidence["metrics"]["v139_selected_cell_pass_count"] == 3549
    assert evidence["metrics"]["v139_observation_only_failure_count"] == 151
    assert evidence["metrics"]["v139_five_camera_remaining_failure_count"] == 151
    assert evidence["current_decision"]["v139_fixed_p1_p2_depth_moments_closed"] is True
    assert evidence["current_decision"]["v139_minimal_predictor_authorized"] is False

    for text in page_texts:
        assert "v139" in text
        assert "algorithm_breakthrough=false" in text

    joined = "\n".join(page_texts)
    assert "多假设深度" in joined
    assert "multi-hypothesis depth" in joined
    assert "poolfire_depth_resolved_ray_consistency_capacity_v139_result_2026-08-11.md" in joined
    assert "poolfire_depth_resolved_ray_consistency_capacity_v139.png" in joined
    assert FIGURE_PATH.stat().st_size > 100_000


def test_public_artifacts_do_not_expose_private_execution_identifiers() -> None:
    paths = PUBLIC_PAGES + [SUMMARY_PATH, EVIDENCE_PATH]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    forbidden = ["/Users/", "private_results", "private_worktrees", "VALIDATED_READY"]
    assert all(fragment not in text for fragment in forbidden)
    assert re.search(r"\b[0-9a-f]{40,64}\b", text, flags=re.IGNORECASE) is None
