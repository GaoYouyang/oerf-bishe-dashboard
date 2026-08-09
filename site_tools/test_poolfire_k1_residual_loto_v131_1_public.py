from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "docs/poolfire_k1_residual_loto_v131_1_public_summary.json"
EVIDENCE_PATH = ROOT / "operator-learning/current-evidence.json"
PUBLIC_PAGES = [
    ROOT / "index.html",
    ROOT / "operator-learning/index.html",
    ROOT / "operator-learning/daily-progress.html",
    ROOT / "docs/poolfire_k1_residual_loto_v131_1_result_2026-08-10.md",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_public_summary_preserves_the_negative_scientific_verdict() -> None:
    summary = load_json(SUMMARY_PATH)

    assert summary["formal_status"] == "FAIL_V131_1_PRIMARY_HELDOUT_ACCURACY"
    assert summary["independent_status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_POOLFIRE_K1_RESIDUAL_SCORE_V131_1"
    )
    assert summary["gate"]["trajectory_passed"] == 0
    assert summary["gate"]["trajectory_total"] == 5
    assert summary["gate"]["accuracy_gate_passed"] is False
    assert summary["scope"]["candidate_exact_A"] == 2
    assert summary["scope"]["candidate_exact_AT"] == 2
    assert summary["scope"]["reference_exact_A"] == 4
    assert summary["scope"]["reference_exact_AT"] == 4
    assert summary["claim_boundary"]["current_dual_l2_representation_closed"] is True
    assert summary["claim_boundary"]["algorithm_breakthrough"] is False
    assert summary["claim_boundary"]["paper_success"] is False


def test_independent_recomputation_is_exact_without_overclaiming_independence() -> None:
    independent = load_json(SUMMARY_PATH)["independent_recomputation"]
    difference_fields = [key for key in independent if "difference" in key]

    assert difference_fields
    assert all(independent[key] == 0.0 for key in difference_fields)
    assert independent["actual_exact_A_calls"] == 7400
    assert independent["actual_exact_AT_calls"] == 7400
    assert independent["validation_truth_read"] is False
    assert independent["test_truth_read"] is False
    assert independent["end_to_end_physics_independence_proven"] is False


def test_current_evidence_and_pages_point_to_the_same_final_result() -> None:
    evidence = load_json(EVIDENCE_PATH)
    page_texts = [path.read_text(encoding="utf-8") for path in PUBLIC_PAGES]

    assert evidence["formal_status"] == "FAIL_V131_1_PRIMARY_HELDOUT_ACCURACY"
    assert evidence["engineering_status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_POOLFIRE_K1_RESIDUAL_SCORE_V131_1"
    )
    assert evidence["metrics"]["v131_1_trajectory_gate_passed"] == 0
    assert evidence["metrics"]["v131_1_trajectory_gate_total"] == 5

    for text in page_texts:
        assert "v131.1" in text
        assert "0/5" in text
        assert "algorithm_breakthrough=false" in text

    joined = "\n".join(page_texts)
    assert "poolfire_k1_residual_loto_v131_1_result_2026-08-10.md" in joined
    assert "poolfire_k1_residual_loto_v131_1.png" in joined


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
