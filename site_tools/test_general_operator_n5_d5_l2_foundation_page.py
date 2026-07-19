from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "general_operator_research_lab.html"
REPORT = ROOT / "docs/n5_d5_l2_private_replay_foundation_2026-07-19.md"
L2B_REPORT = ROOT / "docs/n5_d5_l2b_dual_v2_mechanism_2026-07-19.md"
L2C_REPORT = ROOT / "docs/n5_d5_l2c_external_witness_and_cost_capsule_2026-07-19.md"
L2D_REPORT = ROOT / "docs/n5_d5_l2d_offline_prefix_and_role_evidence_2026-07-19.md"
ADVISOR_REPORT = ROOT / "docs/n5_d5_advisor_first_contact_2026-07-19.md"
PLAN = ROOT / "data_templates/n5_d5_private_replay_plan.placeholder.json"


def _section() -> str:
    html = PAGE.read_text(encoding="utf-8")
    return html.split('id="n5-d5-l2-foundation"', 1)[1].split('id="n5-d4c"', 1)[0]


def test_page_orders_l2_after_l1_and_before_archived_d4c() -> None:
    html = PAGE.read_text(encoding="utf-8")
    section = _section()

    assert 'href="#n5-d5-l2-foundation">D5 L2-B/C/D0 / 双路径门</a>' in html
    assert (
        html.index('href="#n5-d5-private-readiness"')
        < html.index('href="#n5-d5-l2-foundation"')
        < html.index('href="#n5-d4c"')
    )
    assert "N5-D5-L2-D0 offline prefix + L2-C external witness" in section
    assert "development test double 可跑，生产路径在授权前 fail closed" in section


def test_page_explains_exact_triple_and_dual_authorization_budgets() -> None:
    section = _section()

    for phrase in (
        "2 + 53 + 53 + 3 × (2 + 2 + 2 × 2 × 3) = 156 requests",
        "2 + 36 + 36 + 2 × (2 + 2 + 2 × 2 × 3) = 106 requests",
        "TRIPLE TOTAL 156",
        "DUAL TOTAL 106",
        "dual-v2 已实现并接入 L2-A",
        "不得用 wrapper 相减补第三路",
    ):
        assert phrase in section


def test_page_links_public_l2_code_templates_tests_and_beginner_report() -> None:
    section = _section()
    for target in (
        "document_reader.html?doc=docs%2Fn5_d5_l2b_dual_v2_mechanism_2026-07-19.md",
        "document_reader.html?doc=docs%2Fn5_d5_l2_private_replay_foundation_2026-07-19.md",
        "data_templates/n5_d5_l2b_describe_authorization.schema.json",
        "data_templates/n5_d5_minimum_bost_interface_dual_v2.schema.json",
        "data_templates/n5_d5_private_replay_plan.schema.json",
        "data_templates/n5_d5_private_replay_plan.placeholder.json",
        "data_templates/n5_d5_private_physical_contract.placeholder.json",
        "data_templates/n5_d5_private_environment_lock.placeholder.json",
        "site_tools/n5_d5_private_replay_foundation.py",
        "site_tools/n5_d5_l2b_describe_runner.py",
        "site_tools/n5_d5_private_lab_readiness_dual_v2.py",
        "site_tools/test_n5_d5_l2b_describe_runner.py",
        "site_tools/test_n5_d5_private_lab_readiness_dual_v2.py",
        "document_reader.html?doc=docs%2Fn5_d5_l2c_external_witness_and_cost_capsule_2026-07-19.md",
        "data_templates/n5_d5_l2c_trust_policy.schema.json",
        "data_templates/n5_d5_l2c_trust_anchor_registry.json",
        "data_templates/n5_d5_l2c_external_witness_bundle.schema.json",
        "site_tools/n5_d5_l2c_external_witness.py",
        "site_tools/test_n5_d5_l2c_external_witness.py",
        "site_tools/requirements-n5-d5.txt",
        "document_reader.html?doc=docs%2Fn5_d5_l2d_offline_prefix_and_role_evidence_2026-07-19.md",
        "document_reader.html?doc=docs%2Fn5_d5_advisor_first_contact_2026-07-19.md",
        "data_templates/n5_d5_l2d_replay_role_policy.schema.json",
        "data_templates/n5_d5_l2d_trust_anchor_registry.json",
        "data_templates/n5_d5_l2d_replay_role_bundle.schema.json",
        "site_tools/n5_d5_l2d_replay_role_verifier.py",
        "site_tools/test_n5_d5_l2d_replay_role_verifier.py",
        "anchor=122-l2-d0终于能检查签名前缀里只出现一次但还不能说全局只运行一次",
    ):
        assert target in section


def test_page_and_report_keep_execution_science_and_training_locked() -> None:
    section = _section()
    report = (
        REPORT.read_text(encoding="utf-8")
        + L2B_REPORT.read_text(encoding="utf-8")
        + L2C_REPORT.read_text(encoding="utf-8")
        + L2D_REPORT.read_text(encoding="utf-8")
        + ADVISOR_REPORT.read_text(encoding="utf-8")
    )

    for phrase in (
        "133 TARGETED MECHANISM TESTS PASS",
        "L2-D0 31",
        "L2-C 21",
        "FAST MATRIX 257",
        "MEDIUM: 3 FROZEN FAILURES",
        "medium 并行层实测为 2242 passed / 3 个冻结 N2/D4c 失败",
        "实测 21 passed",
        "PRODUCTION HOST BLOCKED",
        "0 REAL ADAPTER CALLS",
        "0 EXTERNAL WITNESS RUNS",
        "0 EXTERNAL LEDGER RUNS",
        "POST_LAUNCH_EXEC_REPLACEMENT_NOT_DENIED",
        "PRIVATE_INPUT_ROOT_EXTERNAL_MUTATION_NOT_DENIED",
        "DURABLE_NONCE_LEDGER_ROOT_NOT_PROTECTED",
        "OUTPUT_ROOT_EXTERNAL_MUTATION_NOT_DENIED",
        "BACKEND_CAPABILITY_ATTESTATION_NOT_EXTERNALLY_VERIFIED",
        "global_nonce_uniqueness_proven=false",
        "production_execution_authorized=false",
        "NO LAB RESULT",
        "formal_replay_authorized=false",
        "这不是算法成功，也不是论文结果",
        "没有训练模型",
        "没有真实 BOST、三维重建、算法基线比较、泛化或论文结果",
    ):
        assert phrase in section or phrase in report


def test_page_separates_signed_claims_from_truth_and_model_cost() -> None:
    section = _section()

    for phrase in (
        "2 distinct key roles",
        "14 chained events",
        "8 bound digests",
        "0 auto authorization",
        "host_capability_truth_independently_proven=false",
        "role_operational_independence_proven=false",
        "one_time_acceptance_proven=false",
        "target_semantic_nonce_unique_in_supplied_signed_prefix=true",
        "prefix_completeness_external_to_bundle_proven=false",
        "anti_rollback_protection_proven=false",
        "log_fork_freedom_proven=false",
        "online_atomic_consume_observed=false",
        "l2c_report_authenticity_proven=false",
        "same_uid_trust_root_replacement_excluded=false",
        "verifier_binary_integrity_proven=false",
        "VERIFIER_CONTROLLED_REGISTRY_BYTES_CHANGED",
        "production_backend_authorized=false",
        "当前 cost capsule 为什么不能比较 FNO",
        "36 或 106 是接口预算，不是算法速度",
        "真实 signer/evidence 运行 0 次",
        "真实 ledger/monitor/gossip 运行 0 次",
    ):
        assert phrase in section


def test_default_private_plan_is_honest_dual_path_with_106_request_budget() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))

    assert plan["capability_profile"]["direct_residual_supported"] is False
    assert plan["capability_profile"]["direct_residual_semantics"] == "unavailable"
    assert (
        plan["capability_profile"]["precomputed_probe_arrays_are_sufficient"] is False
    )
    assert plan["authorization_budget"] == {
        "describe_only_requests": 2,
        "primary_requests": 36,
        "validator_base_replay_requests": 36,
        "validator_private_probe_requests": 32,
        "total_requests": 106,
        "separate_authorization_tokens_required": True,
    }
