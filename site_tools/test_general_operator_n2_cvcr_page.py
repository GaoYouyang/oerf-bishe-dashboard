from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "general_operator_research_lab.html"
LEARNING_LOG = ROOT / "docs/operator_3d_learning_log.md"
TECHNICAL_NOTE = (
    ROOT / "docs/n2_cvcr_n0_postopen_reference_and_pivot_2026-07-18.md"
)
HELD_REPORT = (
    ROOT
    / "demo_t16_operator/results/n2_cvcr_n0_mechanism_gate_v1/report.json"
)
SENSITIVITY_ROOT = (
    ROOT
    / "demo_t16_operator/results/n2_cvcr_n0_reference_sensitivity_postopen_v1"
)
SENSITIVITY_SUMMARY = SENSITIVITY_ROOT / "summary.json"


def _n2_cvcr_section() -> str:
    html = PAGE.read_text(encoding="utf-8")
    return html.split('id="n2-cvcr-n0"', 1)[1].split(
        '<section id="algorithm"', 1
    )[0]


def test_page_exposes_machine_result_and_postopen_boundary() -> None:
    html = PAGE.read_text(encoding="utf-8")
    section = _n2_cvcr_section()
    assert 'href="#n2-cvcr-n0"' in html
    assert "HOLD · REFERENCE NOT CONVERGED" in section
    assert "0.1234%" in section
    assert "2.168×" in section
    assert "6.25%" in section
    assert "原 HOLD、候选分数和所有授权均未改变" in section
    assert "4096 点自身" in section and "不能据此称为 exact truth" in section
    assert "下一候选是设计假设，不是已提出的新算法" in section


def test_page_uses_frustum_wording_and_never_relabels_points_as_pupil_samples() -> None:
    section = _n2_cvcr_section()
    assert "8000 points/pixel 指 frustum data-operator 积分点" in section
    assert "不是全部 pupil samples" in section
    assert "8000 aperture samples" not in section
    assert "8000 pupil samples" not in section


def test_page_links_reproducible_result_assets_and_primary_sources() -> None:
    section = _n2_cvcr_section()
    required = (
        "demo_t16_operator/results/n2_cvcr_n0_mechanism_gate_v1/report.json",
        "demo_t16_operator/results/n2_cvcr_n0_mechanism_gate_v1/aggregate_metrics.csv",
        "demo_t16_operator/results/n2_cvcr_n0_reference_sensitivity_postopen_v1/summary.json",
        "demo_t16_operator/results/n2_cvcr_n0_reference_sensitivity_postopen_v1/reference_ladder.csv",
        "demo_t16_operator/results/n2_cvcr_n0_reference_sensitivity_postopen_v1/reference_sensitivity.png",
        "demo_t16_operator/run_n2_cvcr_n0_reference_sensitivity.py",
        "https://arxiv.org/html/2409.14722v2",
        "https://arxiv.org/html/2402.15954",
        "https://arxiv.org/abs/2211.07422",
        "https://arxiv.org/abs/2008.06722",
        "https://arxiv.org/abs/1606.02261",
        "https://arxiv.org/abs/2006.01524",
    )
    for target in required:
        assert target in section


def test_machine_files_keep_original_hold_and_all_authorizations_closed() -> None:
    held = json.loads(HELD_REPORT.read_text(encoding="utf-8"))
    sensitivity = json.loads(SENSITIVITY_SUMMARY.read_text(encoding="utf-8"))
    assert held["gate_report"]["decision"] == (
        "HOLD_REFERENCE_QUADRATURE_NOT_CONVERGED"
    )
    assert not any(held["authorizations"].values())
    assert sensitivity["status"] == (
        "POSTOPEN_REFERENCE_SENSITIVITY_UNRESOLVED_AT_4096"
    )
    assert sensitivity["original_decision_unchanged"] is True
    assert sensitivity["maximum_last_step_relative_l2"] == 0.0012338918300273773
    assert not any(sensitivity["authorizations"].values())


def test_learning_artifacts_explain_failure_and_next_candidate_in_plain_language() -> None:
    log = LEARNING_LOG.read_text(encoding="utf-8")
    note = TECHNICAL_NOTE.read_text(encoding="utf-8")
    assert "## 102. 第一个孔径控制变量没有过关" in log
    assert "这不是电脑卡住" in log
    assert "NeRIF 同时输出折射率" in log
    assert "N0 不再调参，只保留为 BOST 专用基线" in note
    assert "E[h] = E[c] + E[h-c]" in note
    assert "同随机状态的 forward/JVP/VJP" in note


def test_reference_sensitivity_public_package_is_complete() -> None:
    expected = {
        "README.md",
        "checksums.sha256",
        "reference_ladder.csv",
        "reference_sensitivity.pdf",
        "reference_sensitivity.png",
        "summary.json",
    }
    assert expected <= {path.name for path in SENSITIVITY_ROOT.iterdir()}
