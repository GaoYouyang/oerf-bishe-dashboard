from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "general_operator_research_lab.html"
RESULT_ROOT = ROOT / "demo_t16_operator/results/n2_pvgr_n1_variational_development_v1"
RESULT = RESULT_ROOT / "result.json"
MANIFEST = RESULT_ROOT / "manifest.json"
CONFIG = ROOT / "demo_t16_operator/configs/n2_pvgr_n1_variational_development_v1.json"
PROTOCOL = ROOT / "docs/n2_pvgr_n0_1_shared_state_and_variational_protocol_2026-07-18.md"
GUIDE = ROOT / "docs/n2_pvgr_n1_variational_learning_guide_2026-07-18.md"
LEARNING_LOG = ROOT / "docs/operator_3d_learning_log.md"


def _section() -> str:
    html = PAGE.read_text(encoding="utf-8")
    return html.split('id="n2-pvgr-n1"', 1)[1].split('id="n2-pvgr-n0"', 1)[0]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_latest_n1_decision_is_first_viewport_and_cannot_be_misread() -> None:
    html = PAGE.read_text(encoding="utf-8")
    section = _section()
    assert 'href="#n2-pvgr-n1"' in html
    assert "看 N1 变分修正 7/9" in html
    assert html.index("N2-PVGR-N1 · VARIATIONAL DEFECT") < html.index(
        "N2-PVGR-N0 · TRIFIDELITY DEVELOPMENT"
    )
    assert "7 / 9 SCREEN" in section
    assert "2 REFERENCE NO-HARM FAILURES" in section
    assert "NO AUDIT AUTHORIZATION" in section
    assert "7/9 只允许继续开发" in section
    assert "不允许写“已优于 DeepONet/FNO”" in section


def test_displayed_nine_rows_match_the_machine_result() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    section = _section()
    assert result["machine_decision"] == "DEVELOPMENT_ONLY_NO_AUDIT_AUTHORIZATION"
    assert result["development_screen_pass_count"] == 7
    assert result["case_scale_count"] == 9
    failed = [
        (row["case_id"], row["dimensionless_stress_multiplier"])
        for row in result["rows"]
        if not row["all_development_screens_pass"]
    ]
    assert failed == [
        ("wrinkled_wide_aperture", 3.0),
        ("wrinkled_wide_aperture", 10.0),
    ]
    expected_display = {
        ("smooth_narrow_aperture", 1.0): ("0.0464", "0.0032", "0.9957", "1.000"),
        ("smooth_narrow_aperture", 3.0): ("0.0467", "0.0032", "0.9958", "1.000"),
        ("smooth_narrow_aperture", 10.0): ("0.0486", "0.0032", "0.9961", "1.006"),
        ("wrinkled_wide_aperture", 1.0): ("0.0675", "0.0220", "0.9262", "1.034"),
        ("wrinkled_wide_aperture", 3.0): ("0.0675", "0.0219", "0.9262", "1.143"),
        ("wrinkled_wide_aperture", 10.0): ("0.0685", "0.0226", "0.9259", "1.774"),
        ("smooth_wide_aperture", 1.0): ("0.0514", "0.0030", "0.9932", "1.002"),
        ("smooth_wide_aperture", 3.0): ("0.0514", "0.0030", "0.9938", "1.005"),
        ("smooth_wide_aperture", 10.0): ("0.0520", "0.0030", "0.9930", "1.029"),
    }
    for row in result["rows"]:
        key = (row["case_id"], row["dimensionless_stress_multiplier"])
        for displayed in expected_display[key]:
            assert displayed in section


def test_page_links_evidence_code_learning_log_and_primary_sources() -> None:
    section = _section()
    required = (
        "demo_t16_operator/results/n2_pvgr_n1_variational_development_v1/summary.md",
        "demo_t16_operator/results/n2_pvgr_n1_variational_development_v1/result.json",
        "demo_t16_operator/results/n2_pvgr_n1_variational_development_v1/metrics.csv",
        "demo_t16_operator/results/n2_pvgr_n1_variational_development_v1/manifest.json",
        "n2_pvgr_n1_variational_development.png",
        "docs%2Fn2_pvgr_n1_variational_learning_guide_2026-07-18.md",
        "docs%2Fn2_pvgr_n0_1_shared_state_and_variational_protocol_2026-07-18.md",
        "anchor=106-变分预测第一次真正超过旧代理-但-7-9-不能写成成功",
        "demo_t16_operator/run_n2_pvgr_n1_variational_development.py",
        "demo_t16_operator/trajectory_variational_predictor.py",
        "demo_t16_operator/shared_straight_state.py",
        "https://link.springer.com/article/10.1007/s00348-015-1927-5",
        "https://opg.optica.org/josaa/abstract.cfm?uri=josaa-4-10-1919",
        "https://opg.optica.org/ao/abstract.cfm?uri=ao-26-18-3919",
        "https://imaging.cs.cmu.edu/adjoint_nonlinear_tracing/",
        "https://arxiv.org/html/2402.15954",
        "https://openaccess.thecvf.com/content/CVPR2024/",
        "https://arxiv.org/html/2409.14722v2",
        "https://doi.org/10.1145/3809488",
    )
    for target in required:
        assert target in section


def test_manifest_hashes_sources_outputs_and_leaks_no_local_path() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    result = RESULT.read_text(encoding="utf-8")
    assert "/Users/" not in json.dumps(manifest)
    assert "/Users/" not in result
    source_paths = {
        "runner": ROOT
        / "demo_t16_operator/run_n2_pvgr_n1_variational_development.py",
        "config": CONFIG,
        "shared_straight_state": ROOT / "demo_t16_operator/shared_straight_state.py",
        "trajectory_variational_predictor": ROOT
        / "demo_t16_operator/trajectory_variational_predictor.py",
        "field_dependent_ray": ROOT / "demo_t16_operator/field_dependent_ray.py",
    }
    for key, path in source_paths.items():
        assert manifest["source_sha256"][key] == _sha256(path)
    for filename, expected in manifest["files"].items():
        assert _sha256(RESULT_ROOT / filename) == expected


def test_reserved_families_stay_closed_and_docs_teach_the_failure() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    protocol = PROTOCOL.read_text(encoding="utf-8")
    guide = GUIDE.read_text(encoding="utf-8")
    log = LEARNING_LOG.read_text(encoding="utf-8")
    reserved = ["oblique_compression_sheet", "shock_expansion_pair"]
    assert config["reserved_audit_families_not_opened"] == reserved
    assert result["reserved_audit_families_not_opened"] == reserved
    assert "matched residual 收敛不推出 mixed closure 收敛" in protocol
    assert "## 6. 给何远哲的五项最小数据合同" in guide
    assert "## 106. 变分预测第一次真正超过旧代理" in log
