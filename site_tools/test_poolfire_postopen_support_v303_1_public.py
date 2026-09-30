"""Public regression for the redacted V303.1 support-sensitivity update."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_postopen_train_support_v303_1_public_summary_2026-09-30.json"
NOTE = ROOT / "docs/poolfire_postopen_train_support_v303_1_2026-09-30.md"
FOCUS = ROOT / "operator-learning/index.html"
DAILY = ROOT / "operator-learning/daily-progress.html"
LOG = ROOT / "learning_log.html"


def test_v303_1_public_summary_is_scoped_and_redacted() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["scientific_decision"] == "FAIL_V303_1_POSTOPEN_TRAIN_ROSTER_SUPPORT_SENSITIVITY"
    assert summary["post_open"] is True
    assert summary["trajectory_count"] == 11
    assert summary["sample_count"] == 8140
    assert summary["active_camera_rows"] == 67155
    assert summary["overall_support_fraction"] < summary["frozen_support_threshold"]
    assert summary["added_trajectory_support_fraction"] < summary["frozen_support_threshold"]
    assert summary["integrity_checks_passed"] == summary["integrity_checks_total"] == 25
    assert summary["support_audit_exact_forward_calls"] == 0
    assert summary["support_audit_exact_adjoint_calls"] == 0
    assert summary["predictor_training_authorized"] is False
    assert summary["algorithm_breakthrough"] is False


def test_v303_1_is_linked_from_bilingual_research_surfaces() -> None:
    focus = FOCUS.read_text(encoding="utf-8")
    daily = DAILY.read_text(encoding="utf-8")
    log = LOG.read_text(encoding="utf-8")
    note = NOTE.read_text(encoding="utf-8")
    for content in (focus, daily, log, note):
        assert "86.89%" in content
        assert "16.13%" in content
        assert "post-open" in content.lower() or "已开封" in content
        assert "real BOST" in content or "真实 BOST" in content
    assert 'id="v303-1-train-support"' in focus
    assert 'id="latest"' in daily
    assert 'id="research-update-2026-09-30-v303-1"' in log
    assert 'data-i18n-html-en=' in focus and 'data-i18n-html-zh=' in focus
    assert 'id="latest-2026-09-30-v302"' in daily
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["algorithm_breakthrough"] is False


def test_v303_1_public_payload_contains_no_local_or_private_identifiers() -> None:
    payload = "\n".join(
        (
            NOTE.read_text(encoding="utf-8"),
            SUMMARY.read_text(encoding="utf-8"),
            FOCUS.read_text(encoding="utf-8"),
            DAILY.read_text(encoding="utf-8"),
            LOG.read_text(encoding="utf-8"),
        )
    ).lower()
    for forbidden in ("/users/gaoyouyang", "private_results", "checkpoint_sha256", "raw_data_path"):
        assert forbidden not in payload
