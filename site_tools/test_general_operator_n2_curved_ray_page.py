from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "general_operator_research_lab.html"
RESULT_ROOT = (
    ROOT / "demo_t16_operator/results/n2_adrc_n1_curved_ray_rehearsal_v1"
)
RESULT = RESULT_ROOT / "result.json"
MANIFEST = RESULT_ROOT / "manifest.json"
PROTOCOL = ROOT / "docs/n2_adrc_n1_curved_ray_rehearsal_2026-07-18.md"
HYPOTHESIS = ROOT / "docs/n2_topology_certified_routing_hypothesis_2026-07-18.md"
LEARNING_LOG = ROOT / "docs/operator_3d_learning_log.md"


def _section() -> str:
    html = PAGE.read_text(encoding="utf-8")
    return html.split('id="n2-curved-ray"', 1)[1].split(
        '<div class="alert info"',
        1,
    )[0]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_page_keeps_the_curved_ray_result_at_rehearsal_level() -> None:
    html = PAGE.read_text(encoding="utf-8")
    section = _section()
    assert 'href="#n2-curved-ray"' in html
    assert "FIELD-DEPENDENT RK4 RAY" in section
    assert "3 / 3 NUMERICAL VALIDITY" in section
    assert "E0 SYNTHETIC REHEARSAL" in section
    assert "RESERVED FAMILIES UNOPENED" in section
    assert "REHEARSAL ONLY" in section
    assert "不授权开封、重建、实验或论文主张" in section


def test_displayed_numbers_match_the_machine_result() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    section = _section()
    assert result["machine_decision"] == (
        "CURVED_RAY_REHEARSAL_ONLY_RESERVED_FAMILIES_UNOPENED"
    )
    assert result["case_screen_count"] == result["case_count"] == 3
    expected = {
        "smooth_narrow_aperture": ("0.3183%", "0.2147%", "8.14e-9", "0.0830%", "30×"),
        "wrinkled_wide_aperture": ("0.2609%", "0.0349%", "4.10e-9", "0.0225%", "30×"),
        "smooth_wide_aperture": ("0.3230%", "0.1961%", "1.06e-8", "0.0215%", "100×"),
    }
    for case in result["cases"]:
        assert case["case_id"] in section
        for displayed in expected[case["case_id"]]:
            assert displayed in section
    assert result["reserved_audit_families_not_opened"] == [
        "oblique_compression_sheet",
        "shock_expansion_pair",
    ]


def test_page_links_results_protocol_code_and_primary_sources() -> None:
    section = _section()
    required = (
        "demo_t16_operator/results/n2_adrc_n1_curved_ray_rehearsal_v1/summary.md",
        "demo_t16_operator/results/n2_adrc_n1_curved_ray_rehearsal_v1/result.json",
        "demo_t16_operator/results/n2_adrc_n1_curved_ray_rehearsal_v1/metrics.csv",
        "demo_t16_operator/results/n2_adrc_n1_curved_ray_rehearsal_v1/manifest.json",
        "demo_t16_operator/results/n2_adrc_n1_curved_ray_rehearsal_v1/n2_adrc_n1_curved_ray_rehearsal.png",
        "demo_t16_operator/results/n2_adrc_n1_curved_ray_rehearsal_v1/n2_adrc_n1_refractivity_envelope.png",
        "docs%2Fn2_adrc_n1_curved_ray_rehearsal_2026-07-18.md",
        "docs%2Fn2_topology_certified_routing_hypothesis_2026-07-18.md",
        "demo_t16_operator/run_n2_adrc_n1_curved_ray_rehearsal.py",
        "demo_t16_operator/field_dependent_ray.py",
        "https://arxiv.org/html/2409.19971",
        "https://imaging.cs.cmu.edu/adjoint_nonlinear_tracing/",
        "https://arxiv.org/html/2402.15954",
        "https://arxiv.org/html/2605.11454",
    )
    for target in required:
        assert target in section


def test_manifest_has_relative_sources_and_matching_generated_hashes() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    serialized = json.dumps(manifest)
    assert "/Users/" not in serialized
    assert manifest["config"] == (
        "demo_t16_operator/configs/n2_adrc_n1_curved_ray_rehearsal_v1.json"
    )
    assert manifest["reserved_audit_families_opened"] is False
    assert manifest["ray_module_sha256"] == _sha256(
        ROOT / "demo_t16_operator/field_dependent_ray.py"
    )
    assert manifest["runner_sha256"] == _sha256(
        ROOT / "demo_t16_operator/run_n2_adrc_n1_curved_ray_rehearsal.py"
    )
    for filename, expected in manifest["files"].items():
        assert _sha256(RESULT_ROOT / filename) == expected


def test_learning_artifacts_explain_the_real_decision_in_plain_language() -> None:
    protocol = PROTOCOL.read_text(encoding="utf-8")
    hypothesis = HYPOTHESIS.read_text(encoding="utf-8")
    log = LEARNING_LOG.read_text(encoding="utf-8")
    assert "CURVED_RAY_REHEARSAL_ONLY_RESERVED_FAMILIES_UNOPENED" in protocol
    assert "trajectory sensitivity is only" in protocol
    assert "This is a research hypothesis, not an implemented" in hypothesis
    assert "widehat H" in hypothesis
    assert "## 104. 曲光线导数写对了" in log
    assert "基础折射尺度下 trajectory JVP 只占" in log

