from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "general_operator_research_lab.html"
RESULT = (
    ROOT
    / "demo_t16_operator/results/psu_rotation40_resolution_transfer_public_v1"
)


def _text() -> str:
    return PAGE.read_text(encoding="utf-8")


def _summary() -> dict:
    return json.loads((RESULT / "summary.json").read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_latest_real_result_is_a_first_viewport_signal() -> None:
    text = _text()
    assert text.index('id="resolution-transfer"') < text.index('class="hero-grid"')
    assert "E68 · 16³ SCORE PRE-REGISTERED NO-GO" in text
    assert "3,847,050 REAL RAYS" in text
    assert "N = 1 ROTATION BLOCK" in text
    assert "冻结的 32³+CGLS4 support 优势在 rotation-40 完全反转" in text


def test_page_numbers_match_machine_summary() -> None:
    text = _text()
    comparison = _summary()["comparison"]
    assert f"{comparison['pooled_vector_relative_l2_16']:.6f}" in text
    assert f"{comparison['pooled_vector_relative_l2_32']:.6f}" in text
    assert f"{comparison['pooled_absolute_improvement_16_minus_32']:.6f}" in text
    for row in comparison["camera_rows"]:
        assert f"{row['absolute_improvement_16_minus_32']:.6f}" in text


def test_page_names_rotation_holdout_and_single_block() -> None:
    text = _text()
    assert "ROTATION HOLDOUT · NOT CAMERA HOLDOUT" in text
    assert "三台相机共享一个 rotation-40 运行" in text
    assert "1 rotation block" in text
    assert "未见相机几何与旋转" not in text


def test_page_keeps_scientific_claims_closed() -> None:
    text = _text()
    assert "NO FIELD TRUTH" in text
    assert "不能说 16³ 的真实三维场更准" in text
    assert "RTG-MRC 已是原创算法" in text
    assert "concatenated-all-ray vector relative-L2" in text
    assert "网格与四步 CGLS 的谱滤波轨迹" in text
    assert "32³ 冻结为低分辨率 reference" not in text


def test_new_public_assets_and_docs_exist() -> None:
    required = (
        ROOT / "docs/psu_rotation40_resolution_transfer_result_2026-07-19.md",
        ROOT / "docs/psu_rotation40_resolution_transfer_prereg_2026-07-19.md",
        ROOT / "docs/psu_rotation40_resolution_transfer_independent_audit_2026-07-19.md",
        ROOT / "docs/psu_rotation40_resolution_transfer_replay_2026-07-19.md",
        ROOT / "docs/psu_rotation40_resolution_transfer_environment_2026-07-19.json",
        RESULT / "summary.json",
        RESULT / "comparison_rows.csv",
        RESULT / "diagnostic.png",
        RESULT / "diagnostic.pdf",
        RESULT / "README.md",
        RESULT / "checksums.sha256",
        ROOT / "site_tools/validate_psu_rotation40_resolution_transfer_public.py",
    )
    assert all(path.is_file() and path.stat().st_size > 0 for path in required)


def test_result_checksums_cover_every_public_asset() -> None:
    lines = (RESULT / "checksums.sha256").read_text(encoding="ascii").splitlines()
    entries = {}
    for line in lines:
        digest, filename = line.split("  ", 1)
        entries[filename] = digest
    expected = {
        path.name
        for path in RESULT.iterdir()
        if path.is_file() and path.name != "checksums.sha256"
    }
    assert set(entries) == expected
    assert all(_sha256(RESULT / filename) == digest for filename, digest in entries.items())


def test_public_summary_contains_no_private_path_or_arrays() -> None:
    raw = (RESULT / "summary.json").read_text(encoding="utf-8")
    assert "/Users/" not in raw
    assert "private_library/" not in raw
    summary = json.loads(raw)
    assert summary["public_export_policy"] == {
        "contains_geometry_arrays": False,
        "contains_measurements": False,
        "contains_only_aggregate_metrics": True,
        "contains_predictions": False,
        "contains_volumes": False,
    }


def test_result_document_translates_no_go_into_bounded_next_algorithm() -> None:
    text = (
        ROOT / "docs/psu_rotation40_resolution_transfer_result_2026-07-19.md"
    ).read_text(encoding="utf-8")
    assert "support-fit / held-out-reprojection reversal" in text
    assert "Rotation-Transfer-Gated Multiresolution Correction" in text
    assert "工作名，必须完成一级文献原创性检索" in text
    assert "不能说 16³ 恢复了更准确的真实三维密度场" in text
    assert "Post-open metric-label erratum" in text
    assert "32³ + CGLS4" in text


def test_learning_log_has_plain_language_section_123() -> None:
    text = (ROOT / "docs/operator_3d_learning_log.md").read_text(encoding="utf-8")
    assert "## 123. 更细网格在练习题上赢了，换一个旋转角却输了" in text
    assert "0.843263 / 0.959591" in text
    assert "机器判决是明确 NO-GO" in text
