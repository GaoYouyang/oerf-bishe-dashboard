from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PAGE = REPO_ROOT / "operator-learning" / "index.html"
CURRICULUM = REPO_ROOT / "operator-learning" / "curriculum.js"


def test_focused_learning_home_uses_n3_fail_closed_mainline() -> None:
    html = PAGE.read_text(encoding="utf-8")

    assert "N3 已严格 NO-AUTH" in html
    assert "先把参考解与 field adjoint 做稳" in html
    assert "Picard-1 物理预条件之后" in html
    assert "selectedTrack: 'pvgr-residual'" in html
    assert "当前科学判断：主线转为 Base-Correction CG-PDNO" not in html


def test_focused_learning_home_links_the_frozen_n3_evidence() -> None:
    html = PAGE.read_text(encoding="utf-8")
    expected = (
        "../general_operator_research_lab.html#n2-pvgr-n3",
        "../document_reader.html?doc=docs%2Fn2_pvgr_n3_grouped_factorial_result_audit_2026-07-18.md",
        "../demo_t16_operator/results/n2_pvgr_n3_grouped_factorial_v1/n2_pvgr_n3_grouped_factorial.png",
    )
    for target in expected:
        assert target in html

    assert (
        REPO_ROOT
        / "demo_t16_operator"
        / "results"
        / "n2_pvgr_n3_grouped_factorial_v1"
        / "n2_pvgr_n3_grouped_factorial.png"
    ).is_file()
    assert (
        REPO_ROOT
        / "docs"
        / "n2_pvgr_n3_grouped_factorial_result_audit_2026-07-18.md"
    ).is_file()


def test_curriculum_exposes_current_residual_operator_route_and_boundaries() -> None:
    source = CURRICULUM.read_text(encoding="utf-8")

    assert 'version: "2026.07.18-n3"' in source
    assert 'id:"pvgr-residual", rank:1' in source
    assert 'id:"n3-grouped-audit"' in source
    assert 'id:"n3-field-adjoint"' in source
    assert 'id:"n3-recovery-disclosure"' in source
    assert "H-P1 不高于 evaluator 或实验噪声底" in source
    assert 'title:"历史支线：Base-Correction CG-PDNO"' in source
