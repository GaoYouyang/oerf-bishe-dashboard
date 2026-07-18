from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "general_operator_research_lab.html"
RESULT_ROOT = (
    ROOT / "demo_t16_operator/results/n2_adrc_n1_development_pilot_v1"
)
RESULT = RESULT_ROOT / "result.json"
MANIFEST = RESULT_ROOT / "manifest.json"
LEARNING_LOG = ROOT / "docs/operator_3d_learning_log.md"
PROTOCOL = ROOT / "docs/n2_adrc_n1_development_protocol_2026-07-18.md"
SOURCE_AUDIT = (
    ROOT / "docs/n2_neural_refractive_primitive_source_audit_2026-07-18.md"
)


def _section() -> str:
    html = PAGE.read_text(encoding="utf-8")
    return html.split('id="n2-adrc-n1"', 1)[1].split(
        '<section id="algorithm"', 1
    )[0]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_page_exposes_the_hard_development_boundary_and_four_cases() -> None:
    html = PAGE.read_text(encoding="utf-8")
    section = _section()
    assert 'href="#n2-adrc-n1"' in html
    assert "N2-ADRC-N1 · DEVELOPMENT" in section
    assert "NO AUDIT / PAPER AUTHORIZATION" in section
    assert "2 / 4 PROMOTION SCREEN" in section
    assert "DEVELOPMENT ONLY" in section
    for case_id in (
        "smooth_gradient_only",
        "wrinkled_gradient_only",
        "smooth_coupled_bend",
        "wrinkled_coupled_bend",
    ):
        assert case_id in section


def test_page_numbers_match_the_machine_result() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    section = _section()
    assert result["machine_decision"] == (
        "DEVELOPMENT_ONLY_NO_AUDIT_AUTHORIZATION"
    )
    assert result["case_screen_count"] == 2
    assert result["case_count"] == 4
    assert result["promotion_screen_met"] is True
    expected = {
        "smooth_gradient_only": ("1.784×", "0.07696%", "1.397×"),
        "wrinkled_gradient_only": ("1.744×", "0.12172%", "1.512×"),
        "smooth_coupled_bend": ("1.359×", "0.08182%", "1.102×"),
        "wrinkled_coupled_bend": ("1.396×", "0.77863%", "1.195×"),
    }
    for case in result["cases"]:
        assert case["case_id"] in expected
        for displayed in expected[case["case_id"]]:
            assert displayed in section
    assert result["reserved_audit_families_not_opened"] == [
        "oblique_compression_sheet",
        "shock_expansion_pair",
    ]


def test_page_links_reproducible_assets_protocol_and_primary_sources() -> None:
    section = _section()
    required = (
        "demo_t16_operator/results/n2_adrc_n1_development_pilot_v1/summary.md",
        "demo_t16_operator/results/n2_adrc_n1_development_pilot_v1/result.json",
        "demo_t16_operator/results/n2_adrc_n1_development_pilot_v1/metrics.csv",
        "demo_t16_operator/results/n2_adrc_n1_development_pilot_v1/manifest.json",
        "demo_t16_operator/results/n2_adrc_n1_development_pilot_v1/n2_adrc_n1_development_pilot.png",
        "docs%2Fn2_adrc_n1_development_protocol_2026-07-18.md",
        "docs%2Fn2_neural_refractive_primitive_source_audit_2026-07-18.md",
        "demo_t16_operator/run_n2_adrc_n1_development_pilot.py",
        "demo_t16_operator/automatic_discrete_multifidelity.py",
        "https://arxiv.org/html/2605.11454",
        "https://arxiv.org/html/2409.14722v2",
        "https://arxiv.org/html/2409.19971",
        "https://doi.org/10.1137/15M1046472",
    )
    for target in required:
        assert target in section


def test_manifest_has_relative_sources_and_matching_hashes() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    serialized = json.dumps(manifest)
    assert "/Users/" not in serialized
    assert manifest["config_source"] == (
        "demo_t16_operator/configs/n2_adrc_n1_development_pilot_v1.json"
    )
    for relative, expected in manifest["source_files"].items():
        assert _sha256(ROOT / relative) == expected
    for filename, expected in manifest["files"].items():
        assert _sha256(RESULT_ROOT / filename) == expected


def test_learning_and_source_artifacts_explain_the_candidate_in_plain_language() -> None:
    log = LEARNING_LOG.read_text(encoding="utf-8")
    protocol = PROTOCOL.read_text(encoding="utf-8")
    audit = SOURCE_AUDIT.read_text(encoding="utf-8")
    assert "## 103. 自动梯度加离散梯度不是新算法" in log
    assert "现在最多能说“值得出一张更严格的" in log
    assert "新试卷”，不能说“新算法已经赢了”" in log
    assert "DEVELOPMENT_ONLY_NO_AUDIT_AUTHORIZATION" in protocol
    assert "An unbiased forward estimator does not make a squared loss" in protocol
    assert "2026 paper" in audit
    assert "Prohibited claims" in audit
    assert "No repository-level license file" in audit
