from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "general_operator_research_lab.html"
REPORT = ROOT / "docs/n5_d5_private_adapter_handoff_2026-07-19.md"
READINESS = ROOT / "site_tools/n5_d5_private_lab_readiness.py"


def _section() -> str:
    html = PAGE.read_text(encoding="utf-8")
    return html.split('id="n5-d5-private-readiness"', 1)[1].split(
        'id="n5-d4c"', 1
    )[0]


def test_page_exposes_private_readiness_as_current_locked_gate() -> None:
    html = PAGE.read_text(encoding="utf-8")
    section = _section()

    assert 'href="#n5-d5-private-readiness">D5 私有接线</a>' in html
    assert html.index('href="#n5-d5"') < html.index(
        'href="#n5-d5-private-readiness"'
    ) < html.index('href="#n5-d4c"')
    for phrase in (
        "REAL ADAPTER NOT RECEIVED",
        "STATIC PREFLIGHT IMPLEMENTED",
        "53-CALL FORMAL REPLAY LOCKED",
        "MODEL TRAINING LOCKED",
        "当前没有绿色实验室报告",
        "静态绿灯只解锁人工源码复核与私有 describe 准备",
    ):
        assert phrase in section


def test_page_links_handoff_template_auditor_and_tests() -> None:
    section = _section()
    for target in (
        "document_reader.html?doc=docs%2Fn5_d5_private_adapter_handoff_2026-07-19.md",
        "data_templates/n5_d5_lab_interface.placeholder.json",
        "data_templates/n5_d5_private_adapter_skeleton.py",
        "site_tools/n5_d5_private_lab_readiness.py",
        "site_tools/test_n5_d5_private_lab_readiness.py",
    ):
        assert target in section
    for function in (
        "describe_renderer()",
        "forward_renderer()",
        "jvp_renderer()",
        "vjp_renderer()",
        "canonical_field_vector()",
        "source_review_notes()",
    ):
        assert function in section


def test_private_readiness_section_keeps_science_and_execution_locked() -> None:
    section = _section()

    assert "七类科学授权仍全部为 false" in section
    assert "formal_53_call_replay_authorized=false" in section
    assert "不会 import 或执行私有 adapter" in section
    assert "不证明真实 BOST" in section
    assert "不证明三维重建" in section
    assert "不证明算法优越" in section
    assert "不证明泛化" in section
    assert "不证明论文成立" in section


def test_static_auditor_does_not_import_runner_or_private_adapter() -> None:
    tree = ast.parse(READINESS.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)

    assert "importlib" not in imports
    assert "site_tools.run_n5_d5_minimum_interface_bridge" not in imports
    assert "demo_t16_operator.n5_d5_adapter_protocol" not in imports
    assert not any("private_library" in module for module in imports)


def test_handoff_explains_private_public_conflict_and_future_l2() -> None:
    report = REPORT.read_text(encoding="utf-8")

    for phrase in (
        "public/private provenance 冲突",
        "STATIC_PRIVATE_INTAKE_READY_FORMAL_REPLAY_LOCKED",
        "FORMAL_REPLAY_CLOSED_WORLD_MANIFEST_AVAILABLE",
        "LAB_PUBLIC_SUMMARY_HARD_GUARD_AVAILABLE",
        "UNPREDICTABLE_PRIVATE_PROBES_AVAILABLE",
        "可直接发给何远哲师兄",
        "没有使用、复制或发布任何受限论文",
    ):
        assert phrase in report
