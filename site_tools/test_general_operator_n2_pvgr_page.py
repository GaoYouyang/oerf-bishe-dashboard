from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "general_operator_research_lab.html"
RESULT_ROOT = (
    ROOT / "demo_t16_operator/results/n2_pvgr_n0_trifidelity_development_v1"
)
RESULT = RESULT_ROOT / "result.json"
MANIFEST = RESULT_ROOT / "manifest.json"
CONFIG = ROOT / "demo_t16_operator/configs/n2_pvgr_n0_trifidelity_development_v1.json"
REPORT = ROOT / "docs/n2_pvgr_n0_trifidelity_development_2026-07-18.md"
NEXT = ROOT / "docs/n2_pvgr_next_algorithm_candidates_2026-07-18.md"
LEARNING_LOG = ROOT / "docs/operator_3d_learning_log.md"


def _section() -> str:
    html = PAGE.read_text(encoding="utf-8")
    return html.split('id="n2-pvgr-n0"', 1)[1].split(
        '<div class="alert info"',
        1,
    )[0]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_page_makes_the_latest_no_go_easy_to_find_and_hard_to_misread() -> None:
    html = PAGE.read_text(encoding="utf-8")
    section = _section()
    assert 'href="#n2-pvgr-n0"' in html
    assert "看最新三级路由 NO-GO" in html
    assert "N2-PVGR-N0 · TRIFIDELITY DEVELOPMENT" in section
    assert "9 / 9 CONTRACT" in section
    assert "0 / 9 PROXY SCREEN" in section
    assert "ORACLE HEADROOM 4 / 9" in section
    assert "NO AUDIT AUTHORIZATION" in section
    assert "9/9 合同通过不是算法胜出" in section


def test_displayed_decision_and_base_scale_numbers_match_machine_result() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    section = _section()
    assert result["machine_decision"] == (
        "DEVELOPMENT_ONLY_NO_AUDIT_AUTHORIZATION"
    )
    assert result["development_diagnosis"] == (
        "ORACLE_HEADROOM_CURRENT_PROXY_AND_IMPLEMENTATION_NO_GO"
    )
    assert result["numerical_contract_count"] == result["case_scale_count"] == 9
    assert result["mechanism_headroom_count"] == 4
    assert result["proxy_router_screen_count"] == 0
    assert result["oracle_router_headroom_count"] == 4
    assert result["derivative_contract_count"] == 3
    assert result["implementation_cost_screen_count"] == 0

    expected = {
        "smooth_narrow_aperture": (
            "0.0481",
            "1.0971",
            "0.8344",
            "1 / 64",
            "2.51",
        ),
        "wrinkled_wide_aperture": (
            "0.00679",
            "0.9734",
            "0.9068",
            "0 / 64",
            "2.51",
        ),
        "smooth_wide_aperture": (
            "0.0917",
            "0.9943",
            "0.6682",
            "10 / 64",
            "2.48",
        ),
    }
    base_rows = [
        row for row in result["rows"] if row["dimensionless_stress_multiplier"] == 1
    ]
    assert len(base_rows) == 3
    for row in base_rows:
        assert row["case_id"] in expected
        for displayed in expected[row["case_id"]]:
            assert displayed in section


def test_page_links_results_code_learning_artifacts_and_primary_sources() -> None:
    section = _section()
    required = (
        "demo_t16_operator/results/n2_pvgr_n0_trifidelity_development_v1/summary.md",
        "demo_t16_operator/results/n2_pvgr_n0_trifidelity_development_v1/result.json",
        "demo_t16_operator/results/n2_pvgr_n0_trifidelity_development_v1/metrics.csv",
        "demo_t16_operator/results/n2_pvgr_n0_trifidelity_development_v1/manifest.json",
        "n2_pvgr_n0_trifidelity_development.png",
        "n2_pvgr_n0_derivative_contract.png",
        "docs%2Fn2_pvgr_n0_trifidelity_development_2026-07-18.md",
        "docs%2Fn2_pvgr_next_algorithm_candidates_2026-07-18.md",
        "anchor=105-三级路线找到了机制余量但第一版路由和实现都应该判失败",
        "demo_t16_operator/run_n2_pvgr_n0_trifidelity_development.py",
        "demo_t16_operator/topology_certified_routing.py",
        "demo_t16_operator/ray_safety_certificate.py",
        "https://doi.org/10.1080/01621459.1952.10483446",
        "https://proceedings.neurips.cc/paper/2015/",
        "https://imaging.cs.cmu.edu/adjoint_nonlinear_tracing/",
        "https://openaccess.thecvf.com/content/CVPR2024/",
        "https://doi.org/10.1063/5.0250899",
        "https://doi.org/10.1145/3809488",
    )
    for target in required:
        assert target in section


def test_manifest_hashes_the_exact_sources_and_generated_files() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    serialized = json.dumps(manifest)
    assert "/Users/" not in serialized
    source_paths = {
        "runner": ROOT / "demo_t16_operator/run_n2_pvgr_n0_trifidelity_development.py",
        "config": CONFIG,
        "field_dependent_ray": ROOT / "demo_t16_operator/field_dependent_ray.py",
        "ray_safety_certificate": ROOT / "demo_t16_operator/ray_safety_certificate.py",
        "topology_certified_routing": ROOT
        / "demo_t16_operator/topology_certified_routing.py",
        "analytic_phantoms": ROOT / "demo_t16_operator/analytic_bost_phantoms.py",
    }
    for key, path in source_paths.items():
        assert manifest["source_sha256"][key] == _sha256(path)
    for filename, expected in manifest["files"].items():
        assert _sha256(RESULT_ROOT / filename) == expected


def test_reserved_families_remain_closed_and_docs_preserve_the_boundary() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    report = REPORT.read_text(encoding="utf-8")
    candidates = NEXT.read_text(encoding="utf-8")
    log = LEARNING_LOG.read_text(encoding="utf-8")
    reserved = ["oblique_compression_sheet", "shock_expansion_pair"]
    assert config["reserved_audit_families_not_opened"] == reserved
    assert result["reserved_audit_families_not_opened"] == reserved
    assert "ORACLE_HEADROOM_CURRENT_PROXY_AND_IMPLEMENTATION_NO_GO" in report
    assert "这不是算法成功" in report
    assert "建议立即并行做 `N0.1 + N1`" in candidates
    assert "## 105. 三级路线找到了机制余量" in log
