from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "general_operator_research_lab.html"
RESULT = ROOT / "docs/psu_rotation40_multiresolution_diagnosis_result_2026-07-19.md"
PROTOCOL = ROOT / "docs/psu_rotation40_multiresolution_diagnosis_protocol_2026-07-19.md"
AUDIT = ROOT / "docs/psu_rotation40_multiresolution_diagnosis_independent_audit_2026-07-19.md"


def _page() -> str:
    return PAGE.read_text(encoding="utf-8")


def _section() -> str:
    raw = _page()
    start = raw.index('<section id="multires-diagnosis"')
    stop = raw.index('<div class="hero-grid">', start)
    return raw[start:stop]


def test_first_viewport_and_nav_surface_latest_diagnosis() -> None:
    raw = _page()
    assert '<a href="#multires-diagnosis">多分辨率诊断</a>' in raw
    lead = raw[raw.index('<p class="lead">') : raw.index("</p>", raw.index('<p class="lead">'))]
    assert "U(x16)" in lead
    assert "camera 2" in lead
    assert "camera 3/4" in lead
    assert "不能把迁移反转便宜地归因" in lead


def test_section_reports_exact_machine_result_and_boundaries() -> None:
    section = _section()
    for token in (
        "E69 · MECHANISM UNRESOLVED",
        "0.013541",
        "0.111838",
        "-0.052812",
        "+0.276931",
        "-0.137083/-0.162454",
        "mechanism unresolved",
        "不授权高频过拟合",
        "不授权高频过拟合、CGLS4 唯一归因",
        "N = 1 ROTATION BLOCK",
    ):
        assert token in section
    assert "算法已经成功" not in section
    assert "因果机制已经证明" not in section
    assert "private_library/" not in section


def test_section_has_rendered_figure_and_all_local_targets() -> None:
    section = _section()
    assert (
        'src="demo_t16_operator/results/psu_rotation40_multiresolution_diagnosis_public_v1/diagnostic.png"'
        in section
    )
    assert 'width="2700" height="1800"' in section
    targets = re.findall(r'href="([^"]+)"', section)
    local = [
        target
        for target in targets
        if not target.startswith(("http://", "https://"))
    ]
    assert len(local) >= 6
    for target in local:
        path = target.split("?", 1)[0].split("#", 1)[0]
        if path in {"document_reader.html", "asset_viewer.html"}:
            assert (ROOT / path).is_file()
        else:
            assert (ROOT / path).is_file(), target


def test_primary_source_links_match_the_design_claims() -> None:
    section = _section()
    for url in (
        "https://pubs.aip.org/aip/pof/article/37/1/017143/3331552/Neural-refractive-index-field-Unlocking-the",
        "https://link.springer.com/article/10.1007/s00348-025-04153-3",
        "https://proceedings.neurips.cc/paper_files/paper/2023/hash/dc35c593e61f6df62db541b976d09dcf-Abstract-Conference.html",
        "https://proceedings.iclr.cc/paper_files/paper/2024/hash/eb3c8135137c8a60425a0320869ad87e-Abstract-Conference.html",
        "https://www.aimsciences.org/article/doi/10.3934/ipi.2022037?viewType=HTML",
    ):
        assert url.replace("&", "&amp;") in section or url in section


def test_result_and_protocol_keep_postopen_and_nonnested_grid_caveats() -> None:
    result = RESULT.read_text(encoding="utf-8")
    protocol = PROTOCOL.read_text(encoding="utf-8")
    audit = AUDIT.read_text(encoding="utf-8")
    for token in (
        "DU != I",
        "端点对齐三线性重采样",
        "一个 pooled 标量掩盖",
        "SUPPORT-LORO-MECH-1",
        "不能说哪个三维场更准",
    ):
        assert token in result
    assert "这不是新的盲测或预注册确认实验" in protocol
    assert "不能证明唯一因果机制" in protocol
    assert "final rotations 继续封存" in protocol
    assert "87f5e79539f64de8172720b80ebb2efaef7871b3" in audit
    assert "完整场差" in audit
    assert "不是连续物理能量" in audit
    assert "没有重放 CGLS iteration path" in audit
