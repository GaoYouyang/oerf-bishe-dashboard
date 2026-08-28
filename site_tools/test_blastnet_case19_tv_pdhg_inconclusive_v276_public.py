import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "operator-learning/current-evidence.json"
NOTE = ROOT / "docs/blastnet_case19_tv_basis_pursuit_reference_v276_execution_note_2026-08-28.md"


def test_v276_is_recorded_as_pre_scoring_execution_failure() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    decision = current["current_decision"]

    assert decision["v276_prediction_ready_generated"] is False
    assert decision["v276_metrics_generated"] is False
    assert decision["v276_independent_validation_run"] is False
    assert decision["v276_scientific_result_interpretable"] is False
    assert decision["v276_fixed_tv_pdhg_execution_closed"] is True
    assert decision["v276_rerun_authorized"] is False
    assert decision["v276_algorithm_breakthrough"] is False


def test_v276_public_note_is_bilingual_and_keeps_claim_boundary() -> None:
    note = NOTE.read_text(encoding="utf-8")

    assert "## Chinese" in note
    assert "## English" in note
    assert "failed closed" in note
    assert "`PREDICTION_READY`" in note
    assert "No field, gradient, observation, matched-accuracy, or exact-call result exists" in note
    assert "not a scientific negative result" in note
    assert "algorithm_breakthrough=false" in note


def test_v276_is_visible_without_replacing_v275_scientific_figure() -> None:
    root = (ROOT / "index.html").read_text(encoding="utf-8")
    focus = (ROOT / "operator-learning/index.html").read_text(encoding="utf-8")
    daily = (ROOT / "operator-learning/daily-progress.html").read_text(encoding="utf-8")

    for page in (root, focus, daily):
        assert "v276" in page
        assert (
            "没有正式指标" in page
            or "正式指标0项" in page
            or "正式指标生成数" in page
            or "无科学指标" in page
            or "指标、通过数" in page
        )
        assert "v275" in page
        assert NOTE.name in page
        assert "blastnet_case19_bulk_advection_warm_v275.png" in page
