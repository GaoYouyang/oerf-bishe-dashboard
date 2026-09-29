"""Public regression for the de-identified V302 component-basis diagnostic."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "operator-learning/index.html"
DAILY = ROOT / "operator-learning/daily-progress.html"
LOG = ROOT / "learning_log.html"
LEARNING = ROOT / "docs/operator_3d_learning_log.md"
NOTE = ROOT / "docs/poolfire_v302_component_basis_audit_2026-09-30.md"
SUMMARY = ROOT / "docs/poolfire_v302_component_basis_audit_2026-09-30.json"


def test_v302_summary_preserves_the_frozen_scope_and_result() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["status"] == "POSTOPEN_V302_SHARED_COMPONENT_BASIS_AUDIT_COMPLETE"
    assert summary["scope"]["already_open_trajectories"] == 5
    assert summary["scope"]["rows"] == 25
    assert summary["scope"]["views"] == 9
    assert summary["scope"]["new_data_opened"] is False
    assert summary["scope"]["new_split_opened"] is False
    assert summary["measurement_interface"]["unit"] == "radian"
    assert summary["measurement_interface"]["pixel_displacement_mapping_in_this_branch"] is False
    assert summary["measurement_interface"]["separate_v282_virtual_pixel_forward_exists"] is True
    assert summary["full_2x2_vs_diagonal"]["folds_improved"] == 1
    assert summary["full_2x2_vs_diagonal"]["folds_worsened"] == 4
    assert summary["independent_recomputation"]["upstream_seals_unchanged"] is True


def test_v302_public_surfaces_are_bilingual_and_keep_claim_limits() -> None:
    page = PAGE.read_text(encoding="utf-8")
    daily = DAILY.read_text(encoding="utf-8")
    log = LOG.read_text(encoding="utf-8")
    learning = LEARNING.read_text(encoding="utf-8")
    note = NOTE.read_text(encoding="utf-8")
    combined = "\n".join((page, daily, log, learning, note, SUMMARY.read_text(encoding="utf-8"))).lower()
    for required in ("algorithm_breakthrough=false", "real_bost=false", "v284", "2.66e-15", "radians", "弧度"):
        assert required in combined
    assert 'id="v302-component-basis"' in page
    assert 'data-i18n-zh=' in page and 'data-i18n-en=' in page
    assert 'id="research-update-2026-09-30-v302"' in log
    assert 'id="latest-2026-09-30-v302"' in daily
    assert NOTE.exists() and SUMMARY.exists()
    assert "docs%2Fpoolfire_v302_component_basis_audit_2026-09-30.md" in page
    assert "docs%2Fpoolfire_v302_component_basis_audit_2026-09-30.md" in daily
    for forbidden in ("/users/gaoyouyang", "private_results", "raw_data_path", "checkpoint_sha256"):
        assert forbidden not in combined
