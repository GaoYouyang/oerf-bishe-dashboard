from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "general_operator_research_lab.html"
REPORT = ROOT / "docs/psu_n2_contract_rehearsal_public_summary.json"
LEARNING_LOG = ROOT / "docs/operator_3d_learning_log.md"
TECHNICAL_NOTE = ROOT / "docs/psu_n2_public_rehearsal_2026-07-18.md"


def test_focused_page_links_public_psu_rehearsal_artifacts() -> None:
    html = PAGE.read_text(encoding="utf-8")
    assert 'id="n2-psu-rehearsal"' in html
    assert "公开 PSU 是接口考场，不是真实有限孔径算法的成绩单" in html
    assert "2000 acquired ≠ 0 exposed" in html
    assert "8000 points / pixel" in html
    assert "docs/psu_n2_contract_rehearsal_public_summary.json" in html
    assert "docs/psu_primary_source_fact_audit_2026-07-18.json" in html
    assert "docs%2Fpsu_n2_public_rehearsal_2026-07-18.md" in html
    assert "anchor=101-公开-psu-是接口考场不是有限孔径算法成绩单" in html


def test_page_keeps_interface_and_algorithm_claims_separate() -> None:
    html = PAGE.read_text(encoding="utf-8")
    section = html.split('id="n2-psu-rehearsal"', 1)[1].split(
        '<section id="algorithm"', 1
    )[0]
    assert "0 / 7 N2 AUTHORIZATION" in section
    assert "N2 operator gate 仍为 false" in section
    assert "下一候选是设计假设，不是已提出的新算法" in section
    assert "当前不授权训练" in section
    assert "真实 finite-aperture superiority" in section


def test_machine_report_is_fail_closed_and_matches_page_counts() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["status"] == "PUBLIC_PSU_INTERFACE_REHEARSAL_ONLY_N2_BLOCKED"
    assert report["decision"] == "GO_INTERFACE_REHEARSAL_STOP_N2_ALGORITHM_CLAIMS"
    assert report["field_status_counts"] == {
        "FORBIDDEN_TO_INFER": 3,
        "LOCAL_VERIFICATION_REQUIRED": 3,
        "MISSING": 2,
        "PUBLIC_NEGATIVE": 2,
        "PUBLIC_SUPPORTED": 6,
    }
    assert not any(report["authorization"].values())
    assert all(not row["n2_gate_passed"] for row in report["gate_results"])


def test_learning_artifacts_explain_acquired_vs_accessible_repeats() -> None:
    log = LEARNING_LOG.read_text(encoding="utf-8")
    note = TECHNICAL_NOTE.read_text(encoding="utf-8")
    assert "## 101. 公开 PSU 是接口考场" in log
    assert "实验中采过 2000 张" in log
    assert "不能写成“我们当前拥有 2000 个 repeats”" in log
    assert "论文报告每次 test 实际采集了 2000 张" in note
    assert "每固定条件 `0` 个可访问独立 flow-off repeats" in note
    assert "f/22" in note and "f/32" in note
