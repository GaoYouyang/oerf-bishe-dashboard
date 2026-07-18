from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "general_operator_research_lab.html"
RESULT_DIR = (
    ROOT / "demo_t16_operator/results/n5_d5_minimum_interface_bridge_synthetic_v1"
)
REPORT = ROOT / "docs/n5_d5_minimum_real_interface_bridge_2026-07-19.md"


def _section() -> str:
    html = PAGE.read_text(encoding="utf-8")
    return html.split('id="n5-d5"', 1)[1].split('id="n5-d5-private-readiness"', 1)[0]


def test_page_centers_n5_d5_as_current_interface_bridge() -> None:
    html = PAGE.read_text(encoding="utf-8")
    section = _section()

    assert 'href="#n5-d5">N5-D5 接口桥</a>' in html
    assert html.index('href="#n5-d5"') < html.index('href="#n5-d4c"')
    assert "protocol commit ee792fd" in section
    assert "53" in section
    assert "42 F + 6 Jv + 3 Jᵀq" in section
    assert "1,370" in section
    assert "SYNTHETIC_PROTOCOL_PASS_NO_LAB_AUTHORIZATION" in section
    assert "D4c-v2 现在是历史前置证据" in html


def test_page_links_schema_code_raw_evidence_and_independent_replay() -> None:
    section = _section()
    required_targets = (
        "document_reader.html?doc=docs%2Fn5_d5_minimum_real_interface_bridge_2026-07-19.md",
        "data_templates/n5_d5_minimum_bost_interface.schema.json",
        "data_templates/n5_d5_lab_interface.placeholder.json",
        "demo_t16_operator/configs/n5_d5_minimum_interface_bridge_preregistered_v1.json",
        "demo_t16_operator/results/n5_d5_minimum_interface_bridge_synthetic_v1/result.json",
        "demo_t16_operator/results/n5_d5_minimum_interface_bridge_synthetic_v1/validation_report.json",
        "demo_t16_operator/results/n5_d5_minimum_interface_bridge_synthetic_v1/interface_bridge.png",
    )
    for target in required_targets:
        assert target in section


def test_formal_result_and_page_keep_all_scientific_claims_closed() -> None:
    result = json.loads((RESULT_DIR / "result.json").read_text(encoding="utf-8"))
    validation = json.loads(
        (RESULT_DIR / "validation_report.json").read_text(encoding="utf-8")
    )
    section = _section()

    assert result["protocol_commit"].startswith("ee792fd")
    assert result["machine_decision"] == (
        "SYNTHETIC_PROTOCOL_PASS_NO_LAB_AUTHORIZATION"
    )
    assert result["counts"]["request_count"] == 53
    assert result["counts"]["forward_api_calls"] == 42
    assert result["counts"]["jvp_api_calls"] == 6
    assert result["counts"]["vjp_api_calls"] == 3
    assert validation["valid"] is True
    assert validation["check_count"] == 1370
    assert not any(result["claim_authorizations"].values())
    assert not any(validation["claim_boundary"].values())
    for phrase in (
        "不证明真实 BOST",
        "不证明物理正确",
        "不证明完整 Jacobian",
        "不证明三维重建",
        "不证明算法优越",
        "不证明泛化",
        "不证明论文成立",
    ):
        assert phrase in section


def test_report_has_beginner_route_senior_message_and_primary_sources() -> None:
    report = REPORT.read_text(encoding="utf-8")

    assert "初学者先理解三个函数" in report
    assert "给何远哲师兄的最小请求" in report
    assert "收到接口后的 72 小时路线" in report
    assert "10.1063/5.0250899" in report
    assert "10.1007/s00348-025-04093-y" in report
    assert "10.1007/s00348-020-2912-1" in report
    assert "10.1145/3272127.3275109" in report
    assert "10.1145/3450626.3459775" in report
    assert "10.1137/1.9780898718027" in report
    assert "10.1137/1.9780898717761" in report
