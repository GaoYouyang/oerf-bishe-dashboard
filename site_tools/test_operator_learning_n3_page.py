from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PAGE = REPO_ROOT / "operator-learning" / "index.html"
CURRICULUM = REPO_ROOT / "operator-learning" / "curriculum.js"
D4B_AUDIT = (
    REPO_ROOT
    / "docs"
    / "n2_pvgr_n5_d4b_population_field_derivative_result_audit_2026-07-19.md"
)
D4B_RESULT = (
    REPO_ROOT
    / "demo_t16_operator"
    / "results"
    / "n2_pvgr_n5_d4b_population_field_derivative_v1"
)


def test_focused_learning_home_uses_d4b_fail_closed_boundary() -> None:
    html = PAGE.read_text(encoding="utf-8")

    assert "D4b 完整 32-cell 导数普查严格 FAIL-CLOSED" in html
    assert "D4b 证明 tiny pass 不能直接外推到完整开发总体" in html
    assert "254/256" in html
    assert "58/64" in html
    assert "topology-certified D4b-R" in html
    assert "selectedTrack: 'pvgr-residual'" in html
    assert "不得跳到 decoder、NeRIF、DeepONet、FNO/FFNO 或三维重建" in html
    assert "当前科学判断：主线转为 Base-Correction CG-PDNO" not in html


def test_focused_learning_home_links_the_validated_d4b_evidence() -> None:
    html = PAGE.read_text(encoding="utf-8")
    expected = (
        "../document_reader.html?doc=docs%2Fn2_pvgr_n5_d4b_population_field_derivative_result_audit_2026-07-19.md",
        "../demo_t16_operator/results/n2_pvgr_n5_d4b_population_field_derivative_v1/validation_report.json",
        "../demo_t16_operator/results/n2_pvgr_n5_d4b_population_field_derivative_v1/n2_pvgr_n5_d4b_population_field_derivative.png",
    )
    for target in expected:
        assert target in html

    assert D4B_AUDIT.is_file()
    assert (D4B_RESULT / "validation_report.json").is_file()
    assert (D4B_RESULT / "n2_pvgr_n5_d4b_population_field_derivative.png").is_file()


def test_curriculum_exposes_d4b_failure_and_next_legal_route() -> None:
    source = CURRICULUM.read_text(encoding="utf-8")

    assert 'version: "2026.07.19-n5-d4b"' in source
    assert 'id:"pvgr-residual", rank:1' in source
    assert 'id:"n5-d4b-population-derivative"' in source
    assert "254/256 maps、128/128 structures、58/64 topology" in source
    assert "结果前 D4b-R topology-certified contract" in source
    assert "机器判决 FAIL-CLOSED；全部授权 false" in source
    assert 'title:"历史支线：Base-Correction CG-PDNO"' in source
