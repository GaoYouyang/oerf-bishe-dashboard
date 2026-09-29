"""Regression checks for the de-identified v298 post-open attribution."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "operator-learning/index.html"
LOG = ROOT / "learning_log.html"
DAILY = ROOT / "operator-learning/daily-progress.html"
NOTE = ROOT / "docs/poolfire_v298_forward_mismatch_decomposition_2026-09-30.md"
SUMMARY = ROOT / "docs/poolfire_v298_forward_mismatch_decomposition_2026-09-30.json"
LEARNING = ROOT / "docs/operator_3d_learning_log.md"


def test_v298_metrics_and_evidence_limits_are_visible() -> None:
    page, log, daily, note, learning = (
        path.read_text(encoding="utf-8")
        for path in (PAGE, LOG, DAILY, NOTE, LEARNING)
    )
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    for text in (page, log, daily):
        assert "v298-forward-mismatch" in text
        assert "0.239" in text and "0.290" in text
        assert "25/25" in text
        assert "not" in text.lower() or "不是" in text
        assert "real BOST" in text
    assert "3.82e-14" in note
    assert "already-open" in note
    assert "does not change the prior failure" in note
    assert summary["scope"]["prospective_test"] is False
    assert summary["matched_budget_2A_plus_2AT"]["learned_warm_lower_error_than_zero"]["rows"] == 25
    assert summary["matched_budget_17A_plus_17AT"]["observation_residual_rows"] == {
        "learned_warm_lower": 21,
        "learned_warm_higher": 4,
        "total": 25,
    }
    assert "algorithm_breakthrough=false" in learning
    assert "algorithm_breakthrough=false" in note


def test_v298_public_artifacts_exclude_private_identity() -> None:
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (PAGE, LOG, DAILY, NOTE, SUMMARY, LEARNING)
    ).lower()
    for forbidden in (
        "private_results",
        "/users/gaoyouyang",
        "checkpoint_sha256",
        "operator_sha256",
        "raw_data_path",
    ):
        assert forbidden not in text


def test_v298_bilingual_links_resolve() -> None:
    page = PAGE.read_text(encoding="utf-8")
    log = LOG.read_text(encoding="utf-8")
    daily = DAILY.read_text(encoding="utf-8")
    assert 'id="v298-forward-mismatch"' in page
    assert 'id="research-update-2026-09-30-v298"' in log
    assert 'id="latest"' in daily and "2026-09-30" in daily
    for path in (NOTE, SUMMARY):
        assert path.is_file()
    for text in (page, log, daily):
        assert "Read" in text or "阅读" in text
        assert "View" in text or "查看" in text
