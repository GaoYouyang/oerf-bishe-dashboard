"""Public regression for the redacted p14 composition-sensitivity replication."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTE = ROOT / "docs/poolfire_composition_refractivity_sensitivity_postopen_2026-09-30.md"
FOCUS = ROOT / "operator-learning/index.html"
DAILY = ROOT / "operator-learning/daily-progress.html"
LOG = ROOT / "docs/operator_3d_learning_log.md"


def test_p14_replication_is_bilingual_and_scoped() -> None:
    note = NOTE.read_text(encoding="utf-8")
    focus = FOCUS.read_text(encoding="utf-8")
    daily = DAILY.read_text(encoding="utf-8")
    log = LOG.read_text(encoding="utf-8")

    for text in (note, focus, daily, log):
        assert "2.5816%" in text
        assert "2.6189%" in text
        assert "2.6406%" in text
        assert "algorithm_breakthrough=false" in text or "算法突破" in text
        assert "real BOST" in text or "真实 BOST" in text

    assert "2.75e-12" in note and "2.75e-12" in daily and "2.75e-12" in log
    assert "no p14 projection difference was recomputed" in note
    assert "p14 本次未重新计算二维投影" in focus
    assert 'id="latest"' in daily
    assert "p14 independently reproduces" in daily


def test_public_note_contains_no_private_machine_identifiers() -> None:
    payload = "\n".join(
        path.read_text(encoding="utf-8") for path in (NOTE, FOCUS, DAILY, LOG)
    ).lower()
    for forbidden in ("/users/gaoyouyang", "private_results", "checkpoint_sha256", "raw_data_path"):
        assert forbidden not in payload
