from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "docs/poolfire_projection_pareto_capacity_v134_public_summary.json"
EVIDENCE_PATH = ROOT / "operator-learning/current-evidence.json"
FIGURE_PATH = ROOT / "assets/figures/poolfire_projection_pareto_capacity_v134.png"
PUBLIC_PAGES = [
    ROOT / "index.html",
    ROOT / "operator-learning/index.html",
    ROOT / "operator-learning/daily-progress.html",
    ROOT / "docs/poolfire_projection_pareto_capacity_v134_result_2026-08-10.md",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_summary_preserves_negative_result_and_exact_gain() -> None:
    summary = load_json(SUMMARY_PATH)

    assert summary["formal_status"] == "FAIL_V134_PROJECTION_PARETO_CAPACITY"
    assert summary["independent_status"] == "PASS_INDEPENDENT_RECOMPUTATION_PROJECTION_PARETO_V134"
    assert summary["gate"]["selected_all_metric_pass_count"] == 2591
    assert summary["gate"]["selected_all_metric_total"] == 3700
    assert summary["gate"]["complete_trajectory_pass_count"] == 0
    assert summary["comparison"]["absolute_gain"] == 238
    assert summary["comparison"]["projection_only_cell_pass_count"] == 2564
    assert summary["comparison"]["finite_pareto_gain_over_projection_only"] == 27
    assert summary["metric_cell_pass_counts"] == {
        "field": 3700,
        "full_gradient": 3700,
        "interior_gradient": 3700,
        "observation": 2591,
    }
    assert summary["failure_patterns"]["observation_only_failure"] == 1109
    assert summary["claim_boundary"]["continuous_span_impossibility_proven"] is False
    assert summary["claim_boundary"]["minimal_predictor_authorized"] is False
    assert summary["claim_boundary"]["algorithm_breakthrough"] is False


def test_roster_camera_counts_and_independent_recomputation_are_consistent() -> None:
    summary = load_json(SUMMARY_PATH)
    by_camera = summary["cell_pass_by_camera_count"]
    selected = summary["selected_candidate_counts"]
    independent = summary["independent_recomputation"]

    assert sum(row["v133_passed"] for row in by_camera.values()) == 2353
    assert sum(row["v134_passed"] for row in by_camera.values()) == 2591
    assert sum(selected.values()) == 3700
    assert all(row["total"] == 925 for row in by_camera.values())
    assert independent["maximum_candidate_coefficient_difference"] < 2e-12
    assert independent["maximum_diagnostic_difference"] < 3e-10
    assert independent["maximum_candidate_metric_difference"] < 3e-15
    assert independent["exact_array_failure_count"] == 0
    assert independent["validation_truth_read"] is False
    assert independent["test_truth_read"] is False


def test_current_evidence_preserves_v134_as_historical_parent() -> None:
    evidence = load_json(EVIDENCE_PATH)
    page_texts = [path.read_text(encoding="utf-8") for path in PUBLIC_PAGES]

    assert evidence["v134_projection_pareto_formal_status"] == (
        "FAIL_V134_PROJECTION_PARETO_CAPACITY"
    )
    assert evidence["v134_projection_pareto_independent_status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_PROJECTION_PARETO_V134"
    )
    assert evidence["metrics"]["v134_selected_cell_pass_count"] == 2591
    assert evidence["metrics"]["v134_observation_only_failure_count"] == 1109
    assert evidence["current_decision"]["v134_minimal_predictor_authorized"] is False

    for text in page_texts:
        assert "v134" in text
        assert "algorithm_breakthrough=false" in text

    joined = "\n".join(page_texts)
    assert "poolfire_projection_pareto_capacity_v134_result_2026-08-10.md" in joined
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
