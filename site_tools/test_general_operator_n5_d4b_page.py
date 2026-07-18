from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "general_operator_research_lab.html"
RESULT_ROOT = (
    ROOT
    / "demo_t16_operator"
    / "results"
    / "n2_pvgr_n5_d4b_population_field_derivative_v1"
)
RESULT = RESULT_ROOT / "result.json"
AUDIT = (
    ROOT
    / "docs"
    / "n2_pvgr_n5_d4b_population_field_derivative_result_audit_2026-07-19.md"
)
LEARNING_LOG = ROOT / "docs" / "operator_3d_learning_log.md"


def _section() -> str:
    html = PAGE.read_text(encoding="utf-8")
    return html.split('id="n5-d4b"', 1)[1].split('id="pdhg-no-go"', 1)[0]


def test_current_homepage_leads_with_the_validated_d4b_fail_closed_result() -> None:
    html = PAGE.read_text(encoding="utf-8")
    section = _section()

    assert 'href="#n5-d4b"' in html
    assert "D4b 严格 FAIL-CLOSED" in html
    assert "完整 32-cell 导数普查" in section
    assert "D4B_DERIVATIVE_CONTEXT_CHANGED_FAIL_CLOSED" in section
    assert "254 / 256" in section
    assert "128 / 128" in section
    assert "58 / 64" in section
    assert "333.070 s" in section


def test_displayed_d4b_counts_and_claim_boundary_match_machine_result() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    section = _section()

    assert result["machine_decision"] == (
        "D4B_DERIVATIVE_CONTEXT_CHANGED_FAIL_CLOSED"
    )
    assert result["counts"]["map_context_count"] == 256
    assert result["counts"]["passing_map_context_count"] == 254
    assert result["counts"]["structural_control_count"] == 128
    assert result["counts"]["passing_structural_control_count"] == 128
    assert result["counts"]["topology_context_count"] == 64
    assert result["counts"]["stable_topology_context_count"] == 58
    assert result["budget"]["d4b_total_logical_queries"] == 12_558_336
    assert all(value is False for value in result["authorizations"].values())

    for displayed in (
        "1.842e-10",
        "1.534e-10",
        "12,558,336",
        "全部授权 false",
    ):
        assert displayed in section


def test_d4b_section_links_only_public_explanatory_artifacts() -> None:
    section = _section()
    required = (
        "docs%2Fn2_pvgr_n5_d4b_population_field_derivative_result_audit_2026-07-19.md",
        "docs%2Fn2_pvgr_n5_d4b_population_field_derivative_preregistration_2026-07-19.md",
        "n2_pvgr_n5_d4b_population_field_derivative_v1/result.json",
        "n2_pvgr_n5_d4b_population_field_derivative_v1/validation_report.json",
        "n2_pvgr_n5_d4b_population_field_derivative.png",
        "anchor=113-d4b-没有通过它帮我们看见了两种不能交给大网络掩盖的问题",
    )
    for target in required:
        assert target in PAGE.read_text(encoding="utf-8")

    assert ".npz" not in section
    assert RESULT.is_file()
    assert (RESULT_ROOT / "validation_report.json").is_file()
    assert (RESULT_ROOT / "n2_pvgr_n5_d4b_population_field_derivative.png").is_file()


def test_learning_artifacts_preserve_the_d4b_failure_and_next_legal_gate() -> None:
    audit = AUDIT.read_text(encoding="utf-8")
    log = LEARNING_LOG.read_text(encoding="utf-8")

    assert "254/256" in audit
    assert "58/64" in audit
    assert "不能把 `254/256` 四舍五入成总体成功" in audit
    assert "结果前预注册 D4b-R" in audit
    assert "## 113. D4b 没有通过" in log
    assert "不能改变 D4b 的历史判决" in log
