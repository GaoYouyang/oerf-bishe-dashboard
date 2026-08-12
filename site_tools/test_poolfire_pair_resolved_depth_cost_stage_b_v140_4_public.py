from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / (
    "docs/poolfire_pair_resolved_depth_cost_stage_b_v140_4_public_summary.json"
)
RESULT = ROOT / (
    "docs/poolfire_pair_resolved_depth_cost_stage_b_v140_4_result_2026-08-13.md"
)
FIGURE = ROOT / (
    "assets/figures/poolfire_pair_resolved_depth_cost_stage_b_v140_4.png"
)
EVIDENCE = ROOT / "operator-learning/current-evidence.json"
PAGES = [
    ROOT / "index.html",
    ROOT / "operator-learning/index.html",
    ROOT / "operator-learning/daily-progress.html",
    ROOT / "docs/operator_3d_learning_log.md",
    RESULT,
]


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_full_roster_and_fixed_target_capacity_are_published() -> None:
    summary = _read(SUMMARY)
    stage = summary["stage_b"]
    target = summary["pre_registered_fixed_target"]

    assert summary["formal_status"] == "PASS_V140_4_STAGE_B_HIERARCHICAL_STABLE_CAPACITY"
    assert summary["independent_status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_PAIR_RESOLVED_DEPTH_COST_STAGE_B_V140_4"
    )
    assert stage["passed_count"] == stage["active_tail_count"] == 2199
    assert stage["merged_complete_passed_count"] == stage["merged_complete_count"] == 3700
    assert stage["complete_trajectory_passed_count"] == stage["complete_trajectory_count"] == 5
    assert stage["cheap_control_passed_count"] == 0
    assert all(value == 0 for value in stage["remaining_failures_by_camera_count"].values())
    assert target["label"] == "pair_depth_projection_only"
    assert target["teacher_uses_cfd_truth"] is False
    assert target["complete_passed_count"] == 3700
    assert target["complete_trajectory_passed_count"] == 5
    assert target["post_result_target_switch_used"] is False


def test_independent_audit_and_claim_boundary_are_narrow() -> None:
    summary = _read(SUMMARY)
    audit = summary["independent_recomputation"]
    authorization = summary["authorization"]
    claims = summary["claim_boundary"]

    assert audit["field_image_normalized_difference"] < audit["physical_image_tolerance"]
    assert audit["projection_image_normalized_difference"] < audit["physical_image_tolerance"]
    assert audit["exact_array_failures"] == 0
    assert audit["call_receipt_failures"] == 0
    assert audit["end_to_end_physics_independence_proven"] is False
    assert authorization["complete_3700_fixed_teacher_bundle_generation_authorized"] is True
    assert authorization["predictor_training_authorized_now"] is False
    assert authorization["gpu_rental_recommended_now"] is False
    assert claims["fixed_target_capacity_proven"] is True
    assert claims["complete_teacher_bundle_proven"] is False
    assert claims["observation_only_prediction_proven"] is False
    assert all(
        claims[key] is False
        for key in (
            "algorithm_breakthrough",
            "paper_success",
            "external_generalization",
            "resource_speedup",
            "curved_ray_validated",
            "real_bost",
        )
    )


def test_current_evidence_and_bilingual_surfaces_show_the_new_gate() -> None:
    evidence = _read(EVIDENCE)
    text = "\n".join(path.read_text(encoding="utf-8") for path in PAGES)

    assert evidence["updated"] == "2026-08-13"
    assert evidence["metrics"]["v140_4_stage_b_pass_count"] == 2199
    assert evidence["metrics"]["v140_4_full_pass_count"] == 3700
    assert evidence["metrics"]["v140_4_trajectory_pass_count"] == 5
    assert evidence["current_decision"]["v140_4_fixed_target_capacity_proven"] is True
    assert evidence["current_decision"]["v141_complete_teacher_bundle_proven"] is False
    assert evidence["current_decision"]["v141_predictor_training_authorized"] is False
    assert "3700/3700" in text
    assert "3,700/3,700" in text
    assert "pair_depth_projection_only" in text
    assert "active-tail" in text
    assert "algorithm_breakthrough=false" in text
    assert "poolfire_pair_resolved_depth_cost_stage_b_v140_4_result_2026-08-13.md" in text
    assert "poolfire_pair_resolved_depth_cost_stage_b_v140_4.png" in text
    assert FIGURE.stat().st_size > 100_000


def test_mobile_primary_navigation_keeps_bilingual_labels_readable() -> None:
    focused_page = (ROOT / "operator-learning/index.html").read_text(encoding="utf-8")

    assert "@media (max-width: 720px)" in focused_page
    assert "overflow-x: auto" in focused_page
    assert "flex-wrap: nowrap" in focused_page
    assert "white-space: nowrap" in focused_page
    assert "word-break: normal" in focused_page


def test_new_public_artifacts_do_not_expose_private_execution_details() -> None:
    paths = [*PAGES[:3], RESULT, SUMMARY, EVIDENCE]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    learning_log = PAGES[3].read_text(encoding="utf-8")
    text += "\n" + learning_log.split("## 2026-08-12：", maxsplit=1)[1]
    forbidden = [
        "/Users/",
        "private_results",
        "private_worktrees",
        "VALIDATED_READY",
        "stageb_797c",
        "validation_ed2219",
    ]
    assert all(fragment not in text for fragment in forbidden)
    assert re.search(r"\b[0-9a-f]{40,64}\b", text, flags=re.IGNORECASE) is None
