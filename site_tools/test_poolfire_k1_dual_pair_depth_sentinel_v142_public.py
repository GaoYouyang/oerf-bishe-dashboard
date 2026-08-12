from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_k1_dual_pair_depth_sentinel_v142_public_summary.json"
RESULT = ROOT / "docs/poolfire_k1_dual_pair_depth_sentinel_v142_result_2026-08-13.md"
FIGURE = ROOT / "assets/figures/poolfire_k1_dual_pair_depth_sentinel_v142.png"
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


def test_four_cell_result_and_lower_cost_boundary_are_published() -> None:
    summary = _read(SUMMARY)

    assert summary["independent_status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_K1_DUAL_SENTINEL_V142"
    )
    assert summary["scientific_decision"] == (
        "PASS_K1_DUAL_PAIR_DEPTH_WARM_SENTINEL_V142"
    )
    assert summary["primary_result"]["passed_cells"] == 4
    assert summary["primary_result"]["required_cells"] == 4
    assert summary["primary_result"]["maximum_metric_ratio_to_zero_cgls_k4"] < 1.05
    assert summary["lower_cost_initializer_diagnostic"][
        "passed_the_same_four_sentinel_metric_gates"
    ] is True
    assert all(
        summary["same_cost_controls"][key] is False
        for key in (
            "zero_cgls_k3_passed_all_four",
            "scaled_bp_k2_passed_all_four",
            "geometry_pcgls_k3_passed_all_four",
        )
    )


def test_route_decision_and_claim_boundary_remain_narrow() -> None:
    summary = _read(SUMMARY)
    route = summary["route_decision"]
    claims = summary["claim_boundary"]

    assert "before predictions or scores" in route["invalidated_parent_path"]
    assert route["full_3700_run_in_progress"] is True
    assert route["predictor_training_authorized"] is False
    assert claims["four_cell_mechanism_headroom"] is True
    assert claims["full_3700_capacity_proven"] is False
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


def test_current_surfaces_point_to_v142_without_overclaiming() -> None:
    evidence = _read(EVIDENCE)
    text = "\n".join(path.read_text(encoding="utf-8") for path in PAGES)

    assert evidence["latest_valid_mechanism_status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_K1_DUAL_SENTINEL_V142"
    )
    assert evidence["metrics"]["v142_sentinel_pass_count"] == 4
    assert evidence["current_decision"]["v142_full_3700_capacity_proven"] is False
    assert evidence["current_decision"]["v142_predictor_training_authorized"] is False
    assert "initializer-only" in text
    assert "Initializer-only" in text
    assert "algorithm_breakthrough=false" in text
    assert "poolfire_k1_dual_pair_depth_sentinel_v142_result_2026-08-13.md" in text
    assert "poolfire_k1_dual_pair_depth_sentinel_v142.png" in text
    assert FIGURE.stat().st_size > 100_000


def test_public_artifacts_do_not_expose_private_execution_details() -> None:
    current_surfaces = [
        ROOT / "index.html",
        ROOT / "operator-learning/index.html",
        ROOT / "operator-learning/daily-progress.html",
        RESULT,
        SUMMARY,
        EVIDENCE,
    ]
    log = (ROOT / "docs/operator_3d_learning_log.md").read_text(encoding="utf-8")
    marker = "## 2026-08-13：v141 泄漏链作废"
    assert marker in log
    current_log_section = marker + log.split(marker, maxsplit=1)[1]
    text = "\n".join(
        [*(path.read_text(encoding="utf-8") for path in current_surfaces), current_log_section]
    )
    forbidden = [
        "/Users/",
        "private_results",
        "private_worktrees",
        "VALIDATED_READY",
        "f9cb76cf",
        "be082626",
    ]
    assert all(fragment not in text for fragment in forbidden)
    assert re.search(r"\b[0-9a-f]{40,64}\b", text, flags=re.IGNORECASE) is None
