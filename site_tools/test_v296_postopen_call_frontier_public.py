"""Regression checks for the de-identified v296 post-open diagnostic."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "operator-learning/index.html"
LOG = ROOT / "learning_log.html"
NOTE = ROOT / "docs/poolfire_v296_postopen_call_frontier_2026-09-30.md"
SUMMARY = ROOT / "docs/poolfire_v296_postopen_call_frontier_2026-09-30.json"
DAILY_LOG = ROOT / "docs/operator_3d_learning_log.md"


def test_v296_call_frontier_and_evidence_limits() -> None:
    page, log, note = (p.read_text(encoding="utf-8") for p in (PAGE, LOG, NOTE))
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    for text in (page, log):
        assert "v296-postopen-call-frontier" in text
        assert "24/25" in text
        assert "23/25" in text
        assert "post-open" in text
        assert "real BOST" in text
    assert "At target budget 3" in note
    assert "not a prospective outer test" in note.lower()
    assert summary["warm_matches_control_target_at_lower_budget"]["target_budget_3"] == {
        "zero": 24,
        "bp": 23,
        "applications_saved_per_operator": 1,
    }
    assert summary["warm_matches_control_target_at_lower_budget"]["target_budget_5"] == {
        "zero": 4,
        "bp": 7,
        "applications_saved_per_operator": 2,
    }
    assert summary["scope_limits"][-2:] == [
        "algorithm_breakthrough=false",
        "paper_success=false",
    ]


def test_v296_public_artifacts_exclude_private_identity() -> None:
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (PAGE, LOG, NOTE, SUMMARY)
    ).lower()
    for forbidden in (
        "private_results",
        "/users/gaoyouyang",
        "checkpoint_sha256",
        "operator_sha256",
    ):
        assert forbidden not in text


def test_v296_bilingual_log_and_focus_links_resolve() -> None:
    page = PAGE.read_text(encoding="utf-8")
    learning_log = LOG.read_text(encoding="utf-8")
    daily_log = DAILY_LOG.read_text(encoding="utf-8")
    assert '<section id="v296-postopen-call-frontier"' in page
    assert 'id="research-update-2026-09-30-v296"' in learning_log
    assert "### English checkpoint" in daily_log.split("## 2026-09-27", 1)[0]
    assert "2026-09-30" in daily_log
    for path in (NOTE, SUMMARY):
        assert path.is_file()
    for document in (page, learning_log):
        assert "24/25" in document and "23/25" in document
        assert "2.44e-15" in document or "2.44×10⁻¹⁵" in document
