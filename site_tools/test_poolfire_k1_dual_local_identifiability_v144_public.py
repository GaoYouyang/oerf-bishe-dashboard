from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_k1_dual_local_identifiability_v144_public_summary.json"
RESULT = ROOT / "docs/poolfire_k1_dual_local_identifiability_v144_result_2026-08-14.md"
FIGURE = ROOT / "assets/figures/poolfire_k1_dual_local_identifiability_v144.png"
EVIDENCE = ROOT / "operator-learning/current-evidence.json"
SURFACES = [
    ROOT / "index.html",
    ROOT / "operator-learning/index.html",
    ROOT / "operator-learning/daily-progress.html",
    RESULT,
]


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_v144_local_identifiability_negative_result_is_exact() -> None:
    summary = _read(SUMMARY)
    gates = summary["frozen_gates"]
    methods = summary["methods"]
    cross = methods["cross_trajectory_knn"]
    within = methods["same_trajectory_diagnostic_knn"]
    mean = methods["structural_mean_control"]

    assert summary["scientific_decision"] == "FAIL_LOCAL_OBSERVABLE_IDENTIFIABILITY_V144"
    assert cross["sentinel_pass_count"] == 1
    assert within["sentinel_pass_count"] == 8
    assert mean["sentinel_pass_count"] == 0
    assert cross["trajectory_pass_count"] == 0
    assert within["trajectory_pass_count"] == 0
    assert all(value > gates["trajectory_p90_higher_maximum"] for value in cross["trajectory_p90_higher"].values())
    assert all(value > gates["trajectory_p90_higher_maximum"] for value in within["trajectory_p90_higher"].values())
    assert within["scale_invariant_relative_l2_median"] < cross["scale_invariant_relative_l2_median"]


def test_independent_recomputation_and_post_result_audit_are_disclosed() -> None:
    summary = _read(SUMMARY)
    independent = summary["independent_recomputation"]
    audit = summary["audit_repair"]

    assert independent["integer_neighbor_arrays_exact"] is True
    assert independent["maximum_float_array_absolute_difference"] < 1e-12
    assert independent["scientific_decisions_exact"] is True
    assert audit["post_result"] is True
    assert audit["scientific_gate_changed"] is False
    assert audit["neighbor_or_prediction_changed"] is False
    assert audit["trajectory_tail_maximum_absolute_difference"] < audit["trajectory_tail_audit_tolerance"]


def test_route_closes_without_gpu_or_breakthrough_claim() -> None:
    summary = _read(SUMMARY)
    evidence = _read(EVIDENCE)

    assert summary["route_action"]["frozen_local_neighborhood_hypothesis_closed"] is True
    assert summary["route_action"]["all_nonlinear_models_ruled_out"] is False
    assert summary["route_action"]["gpu_rental_authorized"] is False
    assert summary["route_action"]["neural_training_authorized"] is False
    assert evidence["current_decision"]["v144_frozen_local_neighborhood_hypothesis_closed"] is True
    assert evidence["current_decision"]["gpu_rental_recommended_now"] is False
    assert all(value is False for value in summary["claim_boundary"].values())


def test_current_surfaces_are_bilingual_and_link_v144() -> None:
    log = (ROOT / "docs/operator_3d_learning_log.md").read_text(encoding="utf-8")
    marker = "## 2026-08-14：v144 局部邻域仍不能辨识 Riesz-action 目标"
    assert marker in log
    text = "\n".join([*(path.read_text(encoding="utf-8") for path in SURFACES), marker + log.split(marker, maxsplit=1)[1]])

    assert "poolfire_k1_dual_local_identifiability_v144_result_2026-08-14.md" in text
    assert "poolfire_k1_dual_local_identifiability_v144.png" in text
    assert "1/20" in text and "8/20" in text and "0/20" in text
    assert "local identifiability" in text and "局部可辨识" in text
    assert "No GPU" in text or "不租" in text
    assert "algorithm_breakthrough=false" in text
    assert FIGURE.stat().st_size > 100_000


def test_daily_progress_keeps_one_latest_date() -> None:
    page = (ROOT / "operator-learning/daily-progress.html").read_text(encoding="utf-8")
    dates = re.findall(r'<time datetime="(2026-\d{2}-\d{2})"', page)

    assert dates.count("2026-08-14") == 1
    assert len(dates) == len(set(dates))
    assert page.count('class="day-entry latest"') == 1


def test_public_artifacts_do_not_expose_private_execution_details() -> None:
    text = "\n".join(path.read_text(encoding="utf-8") for path in [*SURFACES, SUMMARY, EVIDENCE])
    forbidden = ["/Users/", "private_results", "private_worktrees", "12141f53", "d580c00a", "9e213238"]

    assert all(fragment not in text for fragment in forbidden)
    assert re.search(r"\b[0-9a-f]{40,64}\b", text, flags=re.IGNORECASE) is None
