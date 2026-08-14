from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_exact_k1_dual_view_ridge_v142_4_public_summary.json"
RESULT = ROOT / "docs/poolfire_exact_k1_dual_view_ridge_v142_4_result_2026-08-14.md"
FIGURE = ROOT / "assets/figures/poolfire_exact_k1_dual_view_ridge_v142_4.png"
EVIDENCE = ROOT / "operator-learning/current-evidence.json"
CURRENT_PAGES = [
    ROOT / "index.html",
    ROOT / "operator-learning/index.html",
    ROOT / "operator-learning/daily-progress.html",
    RESULT,
]


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_independent_negative_result_is_published_exactly() -> None:
    summary = _read(SUMMARY)
    arms = summary["arm_results"]

    assert summary["execution_status"] == (
        "PASS_INDEPENDENT_EXECUTION_EXACT_K1_DUAL_VIEW_RIDGE_V142_4_1"
    )
    assert summary["scientific_decision"] == (
        "FAIL_SHARED_LINEAR_RIDGE_REPRESENTATION_V142_4"
    )
    assert summary["evaluation"]["sample_count"] == 3700
    assert summary["evaluation"]["trajectory_count"] == 5
    assert arms["formal_view"]["passing_cells"] == 1
    assert arms["independent_view"]["passing_cells"] == 1
    assert arms["joint_ls_warm_restart_k1"]["passing_cells"] == 0
    assert all(arms[name]["passing_trajectories"] == 0 for name in arms)
    assert arms["formal_view"]["maximum_metric_ratio"] == 1.9333594179092188
    assert arms["independent_view"]["maximum_metric_ratio"] == 1.933359417909224
    assert arms["joint_ls_warm_restart_k1"]["maximum_metric_ratio"] == (
        1.7435105806532756
    )


def test_dual_views_integrity_and_call_boundary_are_explicit() -> None:
    summary = _read(SUMMARY)
    reproducibility = summary["dual_view_reproducibility"]
    integrity = summary["integrity"]
    evaluation = summary["evaluation"]

    assert reproducibility["pass"] is True
    assert reproducibility["maximum_per_cell_metric_ratio_difference"] < 1e-10
    assert reproducibility["cross_view_dual_relative_l2_difference"] < 1e-8
    assert integrity["checks_passed"] == integrity["checks_total"] == 19
    assert integrity["partial_arrays_reused"] is False
    assert evaluation["candidate_exact_calls_per_sample"] == {"A": 3, "AT": 3}
    assert evaluation["reference_exact_calls_per_sample"] == {"A": 4, "AT": 4}


def test_route_closes_without_overclaiming_or_gpu_rescue() -> None:
    summary = _read(SUMMARY)
    action = summary["route_action"]
    claims = summary["claim_boundary"]
    evidence = _read(EVIDENCE)
    decision = evidence["current_decision"]

    assert action["shared_linear_ridge_closed"] is True
    assert action["post_result_target_switch_allowed"] is False
    assert action["lambda_retuning_allowed"] is False
    assert action["larger_model_rescue_allowed"] is False
    assert decision["v142_4_shared_linear_ridge_closed"] is True
    assert decision["gpu_rental_recommended_now"] is False
    assert decision["wall_or_rss_authorized"] is False
    assert claims["fixed_teacher_mechanism_capacity_proven"] is True
    assert all(
        claims[key] is False
        for key in (
            "deployable_linear_predictor_proven",
            "matched_accuracy_call_reduction_proven",
            "algorithm_breakthrough",
            "paper_success",
            "external_generalization",
            "resource_speedup",
            "curved_ray_validated",
            "real_bost",
        )
    )


def test_current_surfaces_are_bilingual_and_point_to_v142_4() -> None:
    log = (ROOT / "docs/operator_3d_learning_log.md").read_text(encoding="utf-8")
    marker = "## 2026-08-14：v142.4 独立复算关闭当前共享线性预测器"
    assert marker in log
    text = "\n".join(
        [
            *(path.read_text(encoding="utf-8") for path in CURRENT_PAGES),
            marker + log.split(marker, maxsplit=1)[1],
        ]
    )

    assert "poolfire_exact_k1_dual_view_ridge_v142_4_result_2026-08-14.md" in text
    assert "poolfire_exact_k1_dual_view_ridge_v142_4.png" in text
    assert "1/3700" in text and "1/3,700" in text
    assert "0/5" in text
    assert "共享线性" in text and "shared linear" in text.lower()
    assert "algorithm_breakthrough=false" in text
    assert "v142.1 全量公平审计仍在运行" not in text
    assert "v142.1 full fair audit is still running" not in text
    assert FIGURE.stat().st_size > 100_000


def test_daily_progress_has_one_latest_day_and_unique_dates() -> None:
    page = (ROOT / "operator-learning/daily-progress.html").read_text(encoding="utf-8")
    dates = re.findall(r'<time datetime="(2026-\d{2}-\d{2})"', page)

    assert dates.count("2026-08-14") == 1
    assert len(dates) == len(set(dates))
    assert page.count('class="day-entry latest"') == 1


def test_public_artifacts_do_not_expose_private_execution_details() -> None:
    public_files = [*CURRENT_PAGES, SUMMARY, EVIDENCE]
    log = (ROOT / "docs/operator_3d_learning_log.md").read_text(encoding="utf-8")
    marker = "## 2026-08-14：v142.4 独立复算关闭当前共享线性预测器"
    current_log_section = marker + log.split(marker, maxsplit=1)[1]
    text = "\n".join(
        [*(path.read_text(encoding="utf-8") for path in public_files), current_log_section]
    )
    forbidden = [
        "/Users/",
        "private_results",
        "private_worktrees",
        "VALIDATED_READY",
        "validation_56d6e040",
        "formal_740cc28e",
        "OPENED receipt",
    ]

    assert all(fragment not in text for fragment in forbidden)
    assert re.search(r"\b[0-9a-f]{40,64}\b", text, flags=re.IGNORECASE) is None
