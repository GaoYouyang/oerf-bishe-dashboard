from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PAGE = REPO_ROOT / "operator-learning" / "index.html"
CURRICULUM = REPO_ROOT / "operator-learning" / "curriculum.js"
D4_AUDIT = (
    REPO_ROOT
    / "docs"
    / "n2_pvgr_n5_d4_tiny_field_derivative_result_audit_2026-07-18.md"
)
D4_RESULT = (
    REPO_ROOT
    / "demo_t16_operator"
    / "results"
    / "n2_pvgr_n5_d4_tiny_field_derivative_v1"
)


def test_focused_learning_home_uses_d4_selected_derivative_boundary() -> None:
    html = PAGE.read_text(encoding="utf-8")

    assert "D4 的 tiny selected grid-field JVP/VJP 证书已通过独立校验" in html
    assert "D4 证明 tiny grid-field 导数闭合，但没有跳过逆问题" in html
    assert "D4b 32-cell derivative expansion" in html
    assert "decoder(theta) → field → detector" in html
    assert "selectedTrack: 'pvgr-residual'" in html
    assert "不授权 decoder、三维重建、模型或真实数据结论" in html
    assert "当前科学判断：主线转为 Base-Correction CG-PDNO" not in html


def test_focused_learning_home_links_the_validated_d4_evidence() -> None:
    html = PAGE.read_text(encoding="utf-8")
    expected = (
        "../document_reader.html?doc=docs%2Fn2_pvgr_n5_d4_tiny_field_derivative_result_audit_2026-07-18.md",
        "../demo_t16_operator/results/n2_pvgr_n5_d4_tiny_field_derivative_v1/validation_report.json",
        "../demo_t16_operator/results/n2_pvgr_n5_d4_tiny_field_derivative_v1/n2_pvgr_n5_d4_tiny_field_derivative.png",
    )
    for target in expected:
        assert target in html

    assert D4_AUDIT.is_file()
    assert (D4_RESULT / "validation_report.json").is_file()
    assert (D4_RESULT / "n2_pvgr_n5_d4_tiny_field_derivative.png").is_file()


def test_curriculum_exposes_d4_gate_and_next_legal_route() -> None:
    source = CURRICULUM.read_text(encoding="utf-8")

    assert 'version: "2026.07.18-n5-d4"' in source
    assert 'id:"pvgr-residual", rank:1' in source
    assert 'id:"n5-d4-field-derivative"' in source
    assert "32/32 maps、16/16 structures、8/8 topology" in source
    assert "D4b 32-cell derivative→decoder-chain dot/FD→6+2 view reconstruction" in source
    assert "reconstruction/model/real claims false" in source
    assert 'title:"历史支线：Base-Correction CG-PDNO"' in source
