"""Public regression for the de-identified v301 camera attribution."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "operator-learning/index.html"
LOG = ROOT / "learning_log.html"
DAILY = ROOT / "operator-learning/daily-progress.html"
NOTE = ROOT / "docs/poolfire_v301_camera_residual_heterogeneity_2026-09-30.md"
SUMMARY = ROOT / "docs/poolfire_v301_camera_residual_heterogeneity_2026-09-30.json"
LEARNING = ROOT / "docs/operator_3d_learning_log.md"


def test_v301_summary_matches_frozen_scope_and_result() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["status"] == "PASS_V301_CAMERA_RESIDUAL_HETEROGENEITY_POSTOPEN"
    assert summary["scope"]["already_open_trajectories"] == 5
    assert summary["scope"]["rows"] == 25
    assert summary["scope"]["views"] == 9
    assert summary["scope"]["new_data_opened"] is False
    assert summary["scope"]["new_split_opened"] is False
    assert summary["scope"]["per_view_gain_fit"] is False
    assert abs(summary["camera_residual_heterogeneity"]["pooled_camera_ratio_cv"] - 0.05527924480623661) < 1e-15
    assert abs(summary["independent_recomputation"]["max_summary_absolute_difference"] - 1.1685097334179773e-14) < 1e-27
    assert summary["independent_recomputation"]["upstream_inputs_unchanged"] is True


def test_v301_bilingual_artifacts_keep_evidence_boundary_and_privacy() -> None:
    texts = [
        path.read_text(encoding="utf-8")
        for path in (PAGE, LOG, DAILY, NOTE, LEARNING, SUMMARY)
    ]
    joined = "\n".join(texts).lower()
    for required in ("v301", "no single view", "real bost", "algorithm_breakthrough=false", "v284"):
        assert required in joined
    for required in ("data-i18n-zh=", "data-i18n-en=", "v301-camera-residual"):
        assert required in PAGE.read_text(encoding="utf-8")
    assert 'id="research-update-2026-09-30-v301"' in LOG.read_text(encoding="utf-8")
    for forbidden in ("private_results", "/users/gaoyouyang", "checkpoint_sha256", "raw_data_path", "/private/"):
        assert forbidden not in joined
