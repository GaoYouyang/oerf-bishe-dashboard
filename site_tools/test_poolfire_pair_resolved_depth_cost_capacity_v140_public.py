from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "docs/poolfire_pair_resolved_depth_cost_capacity_v140_public_summary.json"
EVIDENCE_PATH = ROOT / "operator-learning/current-evidence.json"
RESULT_PATH = ROOT / "docs/poolfire_pair_resolved_depth_cost_capacity_v140_result_2026-08-11.md"
FIGURE_PATH = ROOT / "assets/figures/poolfire_pair_resolved_depth_cost_capacity_v140.png"
PUBLIC_PAGES = [
    ROOT / "index.html",
    ROOT / "operator-learning/index.html",
    ROOT / "operator-learning/daily-progress.html",
    RESULT_PATH,
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_stage_a_capacity_is_not_mislabeled_as_full_roster_success() -> None:
    summary = load_json(SUMMARY_PATH)
    stage = summary["stage_a"]

    assert summary["formal_status"] == "PASS_V140_STAGE_A_HARD_FAILURE_CAPACITY"
    assert summary["independent_status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_PAIR_RESOLVED_DEPTH_COST_V140"
    )
    assert stage["evaluated_count"] == 151
    assert stage["passed_count"] == 151
    assert stage["remaining_failure_count"] == 0
    assert stage["cheap_control_passed_count"] == 0
    assert stage["stage_b_active_tail_count"] == 2199
    assert stage["all_3700_cells_proven"] is False
    assert stage["all_five_complete_trajectories_proven"] is False
    assert summary["controls"]["minimal_predictor_authorized"] is False
    assert summary["claim_boundary"]["algorithm_breakthrough"] is False


def test_independent_recomputation_and_numerical_audit_are_explicit() -> None:
    summary = load_json(SUMMARY_PATH)
    independent = summary["independent_recomputation"]
    audit = summary["numerical_audit"]

    assert independent["maximum_selected_metric_difference"] < 9e-12
    assert independent["maximum_pair_diagnostic_difference"] < 9e-16
    assert independent["maximum_condition_relative_difference"] < audit["condition_relative_tolerance"]
    assert independent["maximum_coefficient_difference"] < audit["coefficient_absolute_tolerance"]
    assert independent["exact_array_failures"] == 0
    assert independent["call_receipt_failures"] == 0
    assert independent["end_to_end_physics_independence_proven"] is False
    assert audit["post_open_typed_tolerance_repair"] is True
    assert audit["scientific_roster_thresholds_candidates_and_selectors_unchanged"] is True


def test_current_evidence_keeps_v140_stage_a_as_history_after_v140_4() -> None:
    evidence = load_json(EVIDENCE_PATH)
    texts = [path.read_text(encoding="utf-8") for path in PUBLIC_PAGES]
    joined = "\n".join(texts)

    assert evidence["formal_status"] == "PASS_V140_4_STAGE_B_HIERARCHICAL_STABLE_CAPACITY"
    assert evidence["engineering_status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_PAIR_RESOLVED_DEPTH_COST_STAGE_B_V140_4"
    )
    assert evidence["metrics"]["v140_stage_a_pass_count"] == 151
    assert evidence["metrics"]["v140_stage_b_active_tail_count"] == 2199
    assert evidence["current_decision"]["v140_all_3700_cells_proven"] is True
    assert evidence["current_decision"]["v140_complete_trajectory_gate_proven"] is True
    assert evidence["current_decision"]["v141_complete_teacher_bundle_proven"] is False
    assert evidence["current_decision"]["v141_predictor_training_authorized"] is False

    for text in texts:
        assert "v140" in text
        assert "algorithm_breakthrough=false" in text
    assert "成对深度" in joined
    assert "pair-resolved depth" in joined.lower()
    assert "151/151" in joined
    assert "2199" in joined
    assert "post-open" in joined
    assert "poolfire_pair_resolved_depth_cost_capacity_v140_result_2026-08-11.md" in joined
    assert "poolfire_pair_resolved_depth_cost_capacity_v140.png" in joined
    assert "poolfire_pair_resolved_depth_cost_stage_b_v140_4_result_2026-08-13.md" in joined
    assert FIGURE_PATH.stat().st_size > 100_000


def test_public_artifacts_do_not_expose_private_execution_identifiers() -> None:
    paths = PUBLIC_PAGES + [SUMMARY_PATH, EVIDENCE_PATH]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    forbidden = ["/Users/", "private_results", "private_worktrees", "VALIDATED_READY"]
    assert all(fragment not in text for fragment in forbidden)
    assert re.search(r"\b[0-9a-f]{40,64}\b", text, flags=re.IGNORECASE) is None
