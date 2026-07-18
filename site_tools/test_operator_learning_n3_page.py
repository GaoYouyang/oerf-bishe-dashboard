from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PAGE = REPO_ROOT / "operator-learning" / "index.html"
CURRICULUM = REPO_ROOT / "operator-learning" / "curriculum.js"


def test_focused_learning_home_uses_n4_1_fail_closed_mainline() -> None:
    html = PAGE.read_text(encoding="utf-8")

    assert "N4.1 仍是 NO-AUTH" in html
    assert "两个小残差格仍未取得 reference 许可" in html
    assert "先解决两格小残差 reference" in html
    assert "cancellation-aware" in html
    assert "selectedTrack: 'pvgr-residual'" in html
    assert "当前科学判断：主线转为 Base-Correction CG-PDNO" not in html


def test_focused_learning_home_links_the_frozen_n4_1_evidence() -> None:
    html = PAGE.read_text(encoding="utf-8")
    expected = (
        "../document_reader.html?doc=docs%2Fn2_pvgr_n4_1_evaluator_convergence_result_audit_2026-07-18.md",
        "../document_reader.html?doc=docs%2Fn2_pvgr_n5_cancellation_aware_reference_plan_2026-07-18.md",
        "../demo_t16_operator/results/n2_pvgr_n4_1_evaluator_convergence_v1/n2_pvgr_n4_1_evaluator_convergence.png",
    )
    for target in expected:
        assert target in html

    assert (
        REPO_ROOT
        / "demo_t16_operator"
        / "results"
        / "n2_pvgr_n4_1_evaluator_convergence_v1"
        / "n2_pvgr_n4_1_evaluator_convergence.png"
    ).is_file()
    assert (
        REPO_ROOT
        / "docs"
        / "n2_pvgr_n4_1_evaluator_convergence_result_audit_2026-07-18.md"
    ).is_file()


def test_curriculum_exposes_n4_1_reference_gate_and_residual_route() -> None:
    source = CURRICULUM.read_text(encoding="utf-8")

    assert 'version: "2026.07.18-n4.1"' in source
    assert 'id:"pvgr-residual", rank:1' in source
    assert 'id:"n4-evaluator-audit"' in source
    assert 'id:"n5-reference-plan"' in source
    assert 'id:"n3-grouped-audit"' in source
    assert 'id:"n3-field-adjoint"' in source
    assert 'id:"n3-recovery-disclosure"' in source
    assert "H-P1 低于实验噪声" in source
    assert "30/32 不能四舍五入成成功" in source
    assert 'title:"历史支线：Base-Correction CG-PDNO"' in source
