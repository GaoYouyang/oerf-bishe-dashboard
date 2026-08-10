from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "docs/poolfire_signed_phase_cross_camera_capacity_v137_public_summary.json"
EVIDENCE_PATH = ROOT / "operator-learning/current-evidence.json"
RESULT_PATH = ROOT / "docs/poolfire_signed_phase_cross_camera_capacity_v137_result_2026-08-10.md"
FIGURE_PATH = ROOT / "assets/figures/poolfire_signed_phase_cross_camera_capacity_v137.png"
PUBLIC_PAGES = [
    ROOT / "index.html",
    ROOT / "operator-learning/index.html",
    ROOT / "operator-learning/daily-progress.html",
    RESULT_PATH,
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_summary_preserves_gain_and_strict_failure() -> None:
    summary = load_json(SUMMARY_PATH)
    gate = summary["strict_gate"]
    metrics = summary["metric_pass_counts"]
    attribution = summary["rescue_attribution"]

    assert summary["status"] == "FAIL_V137_SIGNED_PHASE_CROSS_CAMERA_CAPACITY"
    assert summary["independent_status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_SIGNED_PHASE_CROSS_CAMERA_V137_1"
    )
    assert gate["v136_passed"] == 3215
    assert gate["v137_passed"] == 3351
    assert gate["v137_rescued_from_v136"] == 136
    assert gate["v137_remaining_failures"] == 349
    assert gate["complete_trajectories_passed"] == 0
    assert metrics == {
        "field_relative_l2": 3700,
        "gradient_relative_l2": 3700,
        "interior_gradient_relative_l2": 3700,
        "reported_observation_relative_l2": 3351,
    }
    assert attribution["self_phase_rescues"] == 92
    assert attribution["peer_geometry_incremental_rescues"] == 44
    assert summary["controls"]["deployment_visible_signed_phase_joint_ls_passed"] == 0
    assert summary["controls"]["minimal_predictor_authorized"] is False
    assert summary["claim_boundary"]["algorithm_breakthrough"] is False
    assert summary["claim_boundary"]["real_bost"] is False


def test_sparse_view_tail_and_independent_recomputation_are_consistent() -> None:
    summary = load_json(SUMMARY_PATH)
    by_camera = summary["cell_pass_by_camera_count"]
    attribution = summary["rescue_attribution"]
    independent = summary["independent_recomputation"]

    assert sum(row["v137_passed"] for row in by_camera.values()) == 3351
    assert sum(row["rescued"] for row in by_camera.values()) == 136
    assert sum(row["remaining"] for row in by_camera.values()) == 349
    assert by_camera["5"]["remaining"] == 343
    assert by_camera["9"]["remaining"] == 0
    assert by_camera["12"]["remaining"] == 0
    assert attribution["remaining_p45_p58"] == 318
    assert independent["maximum_self_phase_difference"] == 0
    assert independent["maximum_peer_phase_difference"] == 0
    assert independent["maximum_peer_residual_rms_difference"] == 0
    assert independent["maximum_selected_metric_difference"] < 5e-15
    assert independent["maximum_coefficient_difference"] < 2e-11
    assert independent["exact_array_failures"] == 0
    assert independent["formal_tree_unchanged"] is True
    assert independent["parent_trees_unchanged"] is True


def test_current_evidence_and_pages_point_to_v137_in_both_languages() -> None:
    evidence = load_json(EVIDENCE_PATH)
    page_texts = [path.read_text(encoding="utf-8") for path in PUBLIC_PAGES]

    assert evidence["formal_status"] == "FAIL_V137_SIGNED_PHASE_CROSS_CAMERA_CAPACITY"
    assert evidence["engineering_status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_SIGNED_PHASE_CROSS_CAMERA_V137_1"
    )
    assert evidence["latest_valid_mechanism_formal_status"] == (
        "FAIL_V137_SIGNED_PHASE_CROSS_CAMERA_CAPACITY"
    )
    assert evidence["headline"].startswith("v137 preserves signed residual phase")
    assert evidence["headline_zh"].startswith("v137 保留 residual 正负相位")
    assert evidence["metrics"]["v137_selected_cell_pass_count"] == 3351
    assert evidence["metrics"]["v137_observation_only_failure_count"] == 349
    assert evidence["metrics"]["v137_five_camera_remaining_failure_count"] == 343
    assert evidence["current_decision"]["v137_same_pixel_peer_representation_closed"] is True
    assert evidence["current_decision"]["v137_minimal_predictor_authorized"] is False
    assert evidence["current_decision"]["v138_ray_overlap_epipolar_capacity_required"] is True

    for text in page_texts:
        assert "v137" in text
        assert "algorithm_breakthrough=false" in text

    joined = "\n".join(page_texts)
    assert "射线" in joined
    assert "epipolar" in joined.lower()
    assert "poolfire_signed_phase_cross_camera_capacity_v137_result_2026-08-10.md" in joined
    assert "poolfire_signed_phase_cross_camera_capacity_v137.png" in joined
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
