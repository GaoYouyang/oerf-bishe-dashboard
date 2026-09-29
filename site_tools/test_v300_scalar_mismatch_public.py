"""Public regression for de-identified v300 scalar mismatch attribution."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "operator-learning/index.html"
LOG = ROOT / "learning_log.html"
DAILY = ROOT / "operator-learning/daily-progress.html"
NOTE = ROOT / "docs/poolfire_v300_scalar_mismatch_attribution_2026-09-30.md"
SUMMARY = ROOT / "docs/poolfire_v300_scalar_mismatch_attribution_2026-09-30.json"
LEARNING = ROOT / "docs/operator_3d_learning_log.md"


def test_v300_numbers_and_independent_receipt() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    pooled = summary["pooled_both_components"]
    assert abs(pooled["best_fit_gain"] - 1.1944451482742084) < 1e-15
    assert abs(pooled["scalar_adjusted_relative_residual"] - 0.19904150249610986) < 1e-15
    assert abs(summary["independent_recomputation"]["max_absolute_difference"] - 2.3092638912203256e-14) < 1e-27
    assert summary["scope"]["rows"] == 25
    assert summary["independent_recomputation"]["input_trees_unchanged"] is True


def test_v300_bilingual_artifacts_preserve_claim_limits() -> None:
    page, log, daily, note, learning = (
        path.read_text(encoding="utf-8")
        for path in (PAGE, LOG, DAILY, NOTE, LEARNING)
    )
    for text in (page, log, daily, note, learning):
        assert "0.199042" in text
        assert "not" in text.lower() or "不是" in text
        assert "real BOST" in text
        assert "v284" in text.lower()
    assert "algorithm_breakthrough=false" in note and "algorithm_breakthrough=false" in learning
    assert 'id="v300-scalar-mismatch"' in page
    assert 'id="research-update-2026-09-30-v300"' in log
    assert "v284" in note.lower() and "v284" in page.lower()
    public_text = "\n".join((page, log, daily, note, SUMMARY.read_text(), learning)).lower()
    for forbidden in ("private_results", "/users/gaoyouyang", "checkpoint_sha256", "raw_data_path"):
        assert forbidden not in public_text
