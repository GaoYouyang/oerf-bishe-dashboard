from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_k1_dual_global_camera_state_v145_public_summary.json"
RESULT = ROOT / "docs/poolfire_k1_dual_global_camera_state_v145_result_2026-08-14.md"
FIGURE = ROOT / "assets/figures/poolfire_k1_dual_global_camera_state_v145.png"
EVIDENCE = ROOT / "operator-learning/current-evidence.json"
SURFACES = [ROOT / "index.html", ROOT / "operator-learning/index.html", ROOT / "operator-learning/daily-progress.html"]


def test_summary_preserves_the_negative_scientific_boundary() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert payload["scientific_decision"] == "FAIL_GLOBAL_CAMERA_STATE_IDENTIFIABILITY_V145"
    assert payload["independent_status"] == "PASS_INDEPENDENT_RECOMPUTATION_GLOBAL_CAMERA_STATE_V145"
    assert payload["post_result_camera_count_audit"]["scientific_decision_changed"] is False
    assert payload["independent_recomputation"]["all_checks_passed"] is True
    assert payload["independent_recomputation"]["maximum_float_array_absolute_difference"] <= 2e-12
    for method in payload["methods"].values():
        assert method["sentinel_pass_count"] == 0
        assert method["full_cell_pass_count"] == 0
        assert method["trajectory_pass_count"] == 0
    boundary = payload["claim_boundary"]
    assert all(value is False for value in boundary.values())
    action = payload["route_action"]
    assert action["shared_global_neighbor_metric_closed"] is True
    assert action["direction_conditioned_local_global_metric_ruled_out"] is False
    assert action["gpu_rental_authorized"] is False


def test_public_surfaces_are_bilingual_and_point_to_v145() -> None:
    log = (ROOT / "docs/operator_3d_learning_log.md").read_text(encoding="utf-8")
    marker = "## 2026-08-14：v145 全局 camera-set 状态仍不能辨识共享目标"
    assert marker in log
    text = "\n".join([*(path.read_text(encoding="utf-8") for path in SURFACES), RESULT.read_text(encoding="utf-8"), marker + log.split(marker, 1)[1]])
    assert "poolfire_k1_dual_global_camera_state_v145_result_2026-08-14.md" in text
    assert "poolfire_k1_dual_global_camera_state_v145.png" in text
    assert "0/20" in text and "0/3700" in text and "1.31e-12" in text
    assert "direction-conditioned" in text and "方向条件" in text
    assert "Do not rent" in text or "不租 GPU" in text
    assert "algorithm_breakthrough=false" in text


def test_daily_progress_keeps_one_latest_date() -> None:
    page = (ROOT / "operator-learning/daily-progress.html").read_text(encoding="utf-8")
    dates = re.findall(r'<time datetime="(2026-\d{2}-\d{2})"', page)
    assert dates.count("2026-08-14") == 1
    assert len(dates) == len(set(dates))
    assert page.count('class="day-entry latest"') == 1


def test_current_manifest_retains_v145_history_without_blocking_newer_evidence() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    metrics = evidence["metrics"]
    decision = evidence["current_decision"]
    assert metrics["v145_cross_coupled_pass_count"] == 0
    assert metrics["v145_full_cell_pass_count"] == 0
    assert metrics["v145_maximum_float_array_absolute_difference"] <= 2e-12
    assert decision["v145_shared_global_neighbor_metric_closed"] is True
    assert decision["v145_direction_conditioned_local_global_metric_ruled_out"] is False
    assert decision["gpu_rental_recommended_now"] is False
    assert decision["algorithm_breakthrough"] is False
    assert RESULT.exists()
    assert SUMMARY.exists()
    assert "v160" in evidence["public_evidence"]["result"]


def test_figure_is_large_and_readable() -> None:
    assert FIGURE.stat().st_size > 100_000
    with Image.open(FIGURE) as image:
        assert image.width >= 2400
        assert image.height >= 1200


def test_public_artifacts_do_not_expose_private_execution_details() -> None:
    text = "\n".join(path.read_text(encoding="utf-8") for path in [*SURFACES, SUMMARY, RESULT, EVIDENCE])
    forbidden = ["/Users/", "private_results", "private_worktrees", "c0e5736e", "eafdab4f", "45cb821e"]
    assert all(fragment not in text for fragment in forbidden)
    assert re.search(r"\b[0-9a-f]{40,64}\b", text, flags=re.IGNORECASE) is None
