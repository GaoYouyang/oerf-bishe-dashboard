"""Public regression for the redacted V307 boundary-support audit."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_fullres_boundary_support_sensitivity_v307_public_summary_2026-09-30.json"
NOTE = ROOT / "docs/poolfire_fullres_boundary_support_sensitivity_v307_2026-09-30.md"
FOCUS = ROOT / "operator-learning/index.html"
DAILY = ROOT / "operator-learning/daily-progress.html"
LOG = ROOT / "learning_log.html"


def test_v307_summary_preserves_scope_and_frozen_conclusion() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["experiment"] == "V307"
    assert summary["evidence_role"] == "post_open_descriptive_representation_sensitivity"
    assert summary["input_scope"]["already_opened_train_trajectories"] == 11
    assert summary["input_scope"]["total_frames"] == 55
    assert summary["input_scope"]["validation_test_external_accessed"] is False
    assert summary["centered_l2_removed_fraction"]["p50"] == 0.19076054534881995
    assert summary["centered_energy_removed_fraction"]["p50"] == 0.036389585661779196
    assert summary["independent_recomputation"]["status"] == "PASS"
    assert summary["interpretation"].startswith("non-negligible numerical representation sensitivity only")
    v306 = summary["companion_v306_native_grid_gradient_stencil_audit"]
    assert v306["total_frames"] == 55
    assert v306["full_gradient_relative_l2_difference"]["p50"] == 0.06593981184283161
    assert v306["strict_interior_relative_l2_difference"]["worst"] == 0.0
    assert v306["independent_recomputation"]["status"] == "PASS_V306_INDEPENDENT_RECOMPUTATION"
    assert v306["independent_recomputation"]["float_metrics_checked"] == 715
    assert v306["independent_recomputation"]["integer_checks"] == 110
    for key in ("algorithm_breakthrough", "paper_success", "resource_speedup", "external_generalization", "real_bost"):
        assert summary[key] is False


def test_v307_is_bilingual_and_linked_without_private_identifiers() -> None:
    contents = {
        "focus": FOCUS.read_text(encoding="utf-8"),
        "daily": DAILY.read_text(encoding="utf-8"),
        "log": LOG.read_text(encoding="utf-8"),
        "note": NOTE.read_text(encoding="utf-8"),
    }
    assert 'id="v307-boundary-support"' in contents["focus"]
    assert 'id="latest"' in contents["daily"]
    assert 'id="research-update-2026-09-30-v307-boundary-support"' in contents["log"]
    assert contents["daily"].count('id="latest"') == 1
    for name, content in contents.items():
        assert "19.08%" in content or "19.076%" in content, name
    assert "不是边界条件确认" in contents["log"]
    assert "not a confirmed boundary condition" in contents["log"]
    assert "不是 BOS 投影" in contents["focus"]
    assert "not a new warm start" in contents["note"]
    assert "6.59%" in contents["focus"] and "6.59%" in contents["daily"]
    assert "715 个浮点指标" in contents["log"]
    assert "9.36e-16" in contents["note"]
    payload = "\n".join(contents.values()).lower()
    for forbidden in ("/users/gaoyouyang", "private_results", "checkpoint_sha256", "raw_data_path"):
        assert forbidden not in payload
