from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "general_operator_research_lab.html"


def _section(raw: str) -> str:
    start = raw.index('<section id="support-loro-preflight"')
    stop = raw.index('<div class="hero-grid">', start)
    return raw[start:stop]


def test_support_loro_preflight_section_has_evidence_and_boundaries() -> None:
    raw = PAGE.read_text(encoding="utf-8")
    section = _section(raw)
    assert '<a href="#support-loro-preflight">LORO 前置证据</a>' in raw
    for needle in (
        "E70 · PREFLIGHT PASS",
        "NO LORO SCORE YET",
        "view 0",
        "view 8",
        "10,628,822",
        "verify_hashes=True",
        "k=0,1,2,3,4,6,8,12",
        "psu_support_rotation_loro_preflight_2026-07-19.md",
        "psu_support_rotation_loro_preflight_public_summary.json",
        "audit_psu_support_rotation_loro_preflight.py",
        "讲人话日志 125",
    ):
        assert needle in section
    assert "没有 CGLS LORO 分数" in section
    assert "没有 field relative-L2" in section


def test_support_loro_public_assets_exist() -> None:
    for path in (
        ROOT / "docs/psu_support_rotation_loro_preflight_2026-07-19.md",
        ROOT / "docs/psu_support_rotation_loro_preflight_public_summary.json",
        ROOT / "site_tools/audit_psu_support_rotation_loro_preflight.py",
    ):
        assert path.is_file(), path
