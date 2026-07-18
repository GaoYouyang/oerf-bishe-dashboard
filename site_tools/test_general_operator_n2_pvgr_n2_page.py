from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "general_operator_research_lab.html"
RESULT_ROOT = (
    ROOT / "demo_t16_operator/results/n2_pvgr_n2_operator_consistent_bridge_v1"
)
RESULT = RESULT_ROOT / "result.json"
MANIFEST = RESULT_ROOT / "manifest.json"
REPORT = ROOT / "docs/n2_pvgr_n2_operator_consistent_bridge_2026-07-18.md"
AUDIT = ROOT / "docs/n2_pvgr_n2_independent_evidence_audit_2026-07-18.md"
MANUSCRIPT = (
    ROOT
    / "docs/n2_pvgr_operator_consistent_manuscript_working_draft_2026-07-18.md"
)
LEARNING_LOG = ROOT / "docs/operator_3d_learning_log.md"


def _section() -> str:
    html = PAGE.read_text(encoding="utf-8")
    return html.split('id="n2-pvgr-n2"', 1)[1].split(
        'id="n2-pvgr-n1"', 1
    )[0]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_latest_n2_bridge_is_first_and_claim_boundary_is_visible() -> None:
    html = PAGE.read_text(encoding="utf-8")
    section = _section()
    assert 'href="#n2-pvgr-n2"' in html
    assert "看 N2 算子一致桥接" in html
    assert html.index("N2-PVGR-N2 · OPERATOR-CONSISTENT") < html.index(
        "N2-PVGR-N1 · VARIATIONAL DEFECT DEVELOPMENT"
    )
    assert "9 / 9 MECHANISM BRIDGE" in section
    assert "60 CORE TESTS" in section
    assert "PICARD STRONGER" in section
    assert "NO PAPER / REAL-DATA AUTHORIZATION" in section
    assert "带警告通过，仅限开发阶段机制证据" in section
    assert "11 项警告" in section
    assert "不得写成“算法成功”" in section
    assert "版本 2026.07.18-N2-PVGR-N2" in html
    assert "PASS WITH WARNINGS — DEVELOPMENT MECHANISM EVIDENCE ONLY" in html


def test_displayed_worst_case_values_match_machine_result() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    section = _section()
    assert result["machine_decision"] == (
        "MECHANISM_BRIDGE_SIGNAL_ONLY_96_CELL_RECONSTRUCTION_AND_REAL_DATA_GATES_CLOSED"
    )
    assert result["primary_screen_pass_count"] == 9
    assert result["primary_screen_required_count"] == 9
    assert result["teacher_screen_pass_count"] == 9
    assert result["reference_sentinel_pass_count"] == 9
    assert result["timing_screen_pass_count"] == 3
    assert result["point_query_screen_pass_count"] == 9
    assert result["development_bridge_authorization"] is False
    assert result["paper_claim_authorization"] is False
    assert result["real_data_authorization"] is False

    rows_by_method: dict[str, list[dict[str, object]]] = {}
    for row in result["method_rows"]:
        rows_by_method.setdefault(row["method_id"], []).append(row)

    expected = {
        "continuous_affine_n1": ("0.06854", "1.77360", "1.68768", "0.92592"),
        "operator_consistent_homotopy": (
            "0.01337",
            "1.06441",
            "1.05611",
            "0.99867",
        ),
        "picard_1": ("0.001709", "1.001015", "1.000944", "0.99982"),
        "picard_2": ("0.000498", "1.000986", "1.000908", "0.99991"),
    }
    for method_id, displayed_values in expected.items():
        assert len(rows_by_method[method_id]) == 9
        for value in displayed_values:
            assert value in section

    teacher_worst = max(
        row["metrics"]["output_relative_l2"]
        for row in result["teacher_rows"]
    )
    assert teacher_worst < 2.2e-14
    assert "2.16e-14" in section


def test_page_links_results_code_learning_and_primary_sources() -> None:
    section = _section()
    required = (
        "demo_t16_operator/results/n2_pvgr_n2_operator_consistent_bridge_v1/summary.md",
        "demo_t16_operator/results/n2_pvgr_n2_operator_consistent_bridge_v1/result.json",
        "demo_t16_operator/results/n2_pvgr_n2_operator_consistent_bridge_v1/metrics.csv",
        "demo_t16_operator/results/n2_pvgr_n2_operator_consistent_bridge_v1/teacher_metrics.csv",
        "demo_t16_operator/results/n2_pvgr_n2_operator_consistent_bridge_v1/reference_sentinel.csv",
        "demo_t16_operator/results/n2_pvgr_n2_operator_consistent_bridge_v1/timing.csv",
        "demo_t16_operator/results/n2_pvgr_n2_operator_consistent_bridge_v1/manifest.json",
        "n2_pvgr_n2_operator_consistent_bridge.png",
        "docs%2Fn2_pvgr_n2_operator_consistent_bridge_2026-07-18.md",
        "docs%2Fn2_pvgr_n2_independent_evidence_audit_2026-07-18.md",
        "docs%2Fn2_pvgr_operator_consistent_manuscript_working_draft_2026-07-18.md",
        "docs%2Fn2_pvgr_cone_ray_baseline_design_2026-07-18.md",
        "anchor=107-精确离散-jvp-修掉了-7-9-但-picard-又把我们打醒了",
        "demo_t16_operator/run_n2_pvgr_n2_operator_consistent_bridge.py",
        "demo_t16_operator/operator_consistent_homotopy_predictor.py",
        "demo_t16_operator/discrete_rk4_jvp_predictor.py",
        "demo_t16_operator/picard_curved_ray_baseline.py",
        "https://opg.optica.org/josaa/abstract.cfm?uri=josaa-4-10-1919",
        "https://academic.oup.com/gji/article/79/1/89/601880",
        "https://arxiv.org/html/2402.15954",
        "https://arxiv.org/html/2409.14722v2",
    )
    for target in required:
        assert target in section


def test_manifest_hashes_all_sources_and_outputs_without_local_paths() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    result_text = RESULT.read_text(encoding="utf-8")
    assert "/Users/" not in json.dumps(manifest)
    assert "/Users/" not in result_text
    for entry in manifest["files"].values():
        path = ROOT / entry["path"]
        assert path.is_file()
        assert path.stat().st_size == entry["bytes"]
        assert _sha256(path) == entry["sha256"]


def test_docs_preserve_erratum_picard_result_and_closed_gates() -> None:
    report = REPORT.read_text(encoding="utf-8")
    audit = AUDIT.read_text(encoding="utf-8")
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    log = LEARNING_LOG.read_text(encoding="utf-8")
    assert "development mechanism bridge" in report
    assert "Picard-1/2" in report
    assert "float64 容差内一致" in report
    assert "不能把 9/9 机制桥接写成论文成功" in report
    assert "完成了 96 physical cells" not in report
    assert "= 96 physical cells" in report
    assert "PASS WITH WARNINGS — DEVELOPMENT MECHANISM EVIDENCE ONLY" in audit
    assert "| W11 | WARN |" in audit
    assert "WORKING DRAFT / NOT SUBMISSION READY" in manuscript
    assert "paper claim authorization | `false`" in manuscript
    assert "## 107. 精确离散 JVP 修掉了 7/9" in log
    assert "不能把 OCBH 包装成“自有算法已经胜出”" in log
