from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "general_operator_research_lab.html"
RESULT_DIR = (
    ROOT / "demo_t16_operator/results/n2_pvgr_n5_d4c_msra_semantic_v2"
)
REPORT = ROOT / "docs/n2_pvgr_n5_d4c_msra_semantic_v2_2026-07-19.md"
V1_REPORT = ROOT / "docs/n2_pvgr_n5_d4c_msra_development_2026-07-19.md"


def _section() -> str:
    html = PAGE.read_text(encoding="utf-8")
    return html.split('id="n5-d4c"', 1)[1].split('id="n5-d4b"', 1)[0]


def test_page_centers_semantic_v2_and_records_v1_no_go() -> None:
    html = PAGE.read_text(encoding="utf-8")
    section = _section()

    assert 'href="#n5-d4c">D4c-v2 语义审计</a>' in html
    assert "D4c-v1 虽通过文件完整性检查" in html
    assert "语义 NO-GO" in html
    assert "protocol commit 09a50d1" in section
    assert "720 个 case" in section
    assert "34,560" in section
    assert "36,000" in section
    assert "0 pooled scores" in section
    assert "74.72" not in html
    assert "8 probes → 24/24" not in html


def test_page_links_v2_evidence_and_keeps_claims_closed() -> None:
    section = _section()
    required_targets = (
        "document_reader.html?doc=docs%2Fn2_pvgr_n5_d4c_msra_semantic_v2_2026-07-19.md",
        "demo_t16_operator/results/n2_pvgr_n5_d4c_msra_semantic_v2/result.json",
        "demo_t16_operator/results/n2_pvgr_n5_d4c_msra_semantic_v2/validation_report.json",
        "demo_t16_operator/results/n2_pvgr_n5_d4c_msra_semantic_v2/semantic_v2.png",
        "demo_t16_operator/configs/n2_pvgr_n5_d4c_msra_semantic_v2_preregistered.json",
    )
    for target in required_targets:
        assert target in section

    result = json.loads((RESULT_DIR / "result.json").read_text(encoding="utf-8"))
    validation = json.loads(
        (RESULT_DIR / "validation_report.json").read_text(encoding="utf-8")
    )
    assert result["protocol_commit"].startswith("09a50d1")
    assert result["threshold_selected"] is None
    assert result["counts"]["variant_count"] == 720
    assert result["counts"]["fd_rows"] == 34560
    assert result["counts"]["decision_rows"] == 36000
    assert validation["valid"] is True
    assert validation["independence_contract"]["runner_imported"] is False
    assert all(value is False for value in result["claim_authorizations"].values())
    assert all(
        value is False
        for key, value in validation["claim_boundary"].items()
        if key.endswith("authorized")
    )


def test_reports_explain_semantic_boundary_and_primary_sources() -> None:
    report = REPORT.read_text(encoding="utf-8")
    historical = V1_REPORT.read_text(encoding="utf-8")

    assert "valid=true" in report
    assert "synthetic explicit-matrix" in report
    assert "PASS_STRONG_SIGNAL" in report
    assert "不表示候选导数已被证明正确" in report
    assert "10.1137/1.9780898718027" in report
    assert "10.1137/1.9780898717761" in report
    assert "10.1145/3476576.3476671" in report
    assert "10.1145/3272127.3275109" in report
    assert "10.1063/5.0250899" in report
    assert "语义主张已撤回" in historical
    assert "`fd_relative_error` 是候选 JVP 与参考 JVP 的差" in historical
