from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_k1_dual_riesz_action_v143_public_summary.json"
RESULT = ROOT / "docs/poolfire_k1_dual_riesz_action_v143_result_2026-08-14.md"
FIGURE = ROOT / "assets/figures/poolfire_k1_dual_riesz_action_v143.png"
EVIDENCE = ROOT / "operator-learning/current-evidence.json"
SURFACES = [
    ROOT / "index.html",
    ROOT / "operator-learning/index.html",
    ROOT / "operator-learning/daily-progress.html",
    RESULT,
]


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_v143_negative_result_and_audit_boundary_are_exact() -> None:
    summary = _read(SUMMARY)
    target = summary["target_space"]
    audit = summary["audit_repair"]

    assert summary["scientific_decision"] == "FAIL_SHARED_LINEAR_RIESZ_ACTION_PREDICTABILITY_V143"
    assert summary["evaluation"]["sentinel_pass_count"] == 0
    assert summary["evaluation"]["sentinel_count"] == 20
    assert target["oracle_inverse_scale_invariant_relative_l2_maximum"] < target["oracle_inverse_gate"]
    assert target["predicted_scale_invariant_relative_l2_minimum"] > target["predicted_error_each_cell_gate"]
    assert target["predicted_cosine_maximum"] < target["predicted_cosine_each_cell_gate"]
    assert all(value > target["trajectory_p90_higher_gate"] for value in target["trajectory_p90_higher"].values())
    assert audit["post_result"] is True
    assert audit["scientific_gate_changed"] is False
    assert audit["condition_number_maximum_symmetric_relative_difference"] < 1e-9


def test_route_closes_without_gpu_or_breakthrough_claim() -> None:
    summary = _read(SUMMARY)
    evidence = _read(EVIDENCE)

    assert summary["route_action"]["shared_linear_riesz_action_hypothesis_closed"] is True
    assert summary["route_action"]["gpu_rental_authorized"] is False
    assert summary["route_action"]["neural_training_authorized"] is False
    assert evidence["current_decision"]["v143_shared_linear_riesz_action_closed"] is True
    assert evidence["current_decision"]["gpu_rental_recommended_now"] is False
    assert all(value is False for value in summary["claim_boundary"].values())


def test_current_surfaces_are_bilingual_and_link_v143() -> None:
    log = (ROOT / "docs/operator_3d_learning_log.md").read_text(encoding="utf-8")
    marker = "## 2026-08-14：v143 Riesz-action 坐标未恢复共享线性可预测性"
    assert marker in log
    text = "\n".join([*(path.read_text(encoding="utf-8") for path in SURFACES), marker + log.split(marker, maxsplit=1)[1]])

    assert "poolfire_k1_dual_riesz_action_v143_result_2026-08-14.md" in text
    assert "poolfire_k1_dual_riesz_action_v143.png" in text
    assert "0/20" in text and "0/20" in text
    assert "Riesz-action" in text and "共享线性" in text
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
    forbidden = ["/Users/", "private_results", "private_worktrees", "11845406", "2bbdae12", "61ee6ff2"]

    assert all(fragment not in text for fragment in forbidden)
    assert re.search(r"\b[0-9a-f]{40,64}\b", text, flags=re.IGNORECASE) is None
