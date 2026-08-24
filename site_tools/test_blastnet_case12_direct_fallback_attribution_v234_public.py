from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case12_direct_fallback_attribution_v234_public_summary.json"
RESULT = ROOT / "docs/blastnet_case12_direct_fallback_attribution_v234_result_2026-08-25.md"
FIGURE = ROOT / "assets/figures/blastnet_case12_direct_fallback_attribution_v234.png"


def test_v234_summary_closes_the_fallback_attribution() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert data["scope"]["evidence_role"] == "post-open mechanism attribution"
    assert data["scope"]["pristine_external_result"] is False
    assert data["execution"]["independent_checks_passed"] == 14
    accuracy = data["strict_accuracy"]
    assert accuracy["direct_low64_pcgls_k11"]["strict_safe_cells"] == 598
    assert accuracy["direct_low64_pcgls_k11"]["complete_rigs_passed"] == 13
    assert accuracy["zero_geometry_jacobi_pcgls_k16"]["strict_safe_cells"] == 594
    assert accuracy["fixed_v229_dual_press_policy"]["strict_safe_cells"] == 595
    attribution = data["causal_attribution"]
    assert attribution["policy_accepts"] + attribution["policy_rejects"] == 598
    assert attribution["rejected_direct_cells_already_safe"] == attribution["policy_rejects"]
    assert attribution["failures_caused_by_fallback"] == attribution["policy_failures"] == 3
    assert attribution["all_policy_failures_explained_by_fallback"]
    assert data["logical_exact_calls"]["direct_total_savings_over_policy"] == {"A": 644, "AT": 805}
    assert all(value is False for value in data["claims_fixed_false"].values())


def test_v234_result_is_bilingual_and_preserves_the_evidence_boundary() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v234：" in text and "# v234:" in text
    for token in ("598/598", "594/598", "595/598", "437", "161", "644A", "805A^T", "14/14"):
        assert token in text
    assert "结果已开的机制归因" in text
    assert "post-open mechanism attribution" in text
    assert "algorithm_breakthrough=false" in text


def test_v234_figure_is_rendered() -> None:
    assert FIGURE.is_file() and FIGURE.stat().st_size > 50_000
    with Image.open(FIGURE) as image:
        assert image.width >= 1800
        assert image.height >= 700


def test_v234_current_surfaces_and_log_are_synchronized() -> None:
    current = json.loads((ROOT / "operator-learning/current-evidence.json").read_text(encoding="utf-8"))
    assert current["scientific_status"] == (
        "POST_OPEN_CASE12_DIRECT_LOW64_K11_CONTRACT_DOMINATES_FIXED_DUAL_PRESS_FALLBACK_V234"
    )
    assert current["metrics"]["v234_direct_strict_safe_cells"] == 598
    assert current["metrics"]["v234_policy_strict_safe_cells"] == 595
    assert current["metrics"]["v234_fallback_caused_failures"] == 3
    assert current["current_decision"]["v234_current_v229_fallback_shell_closed"] is True
    assert current["current_decision"]["v234_external_generalization"] is False
    assert "unopened" in current["next_scientific_gate"].lower()
    assert "direct_fallback_attribution_v234" in current["public_evidence"]["result"]
    for relative in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html"):
        content = (ROOT / relative).read_text(encoding="utf-8")
        assert "v234" in content
        assert "blastnet_case12_direct_fallback_attribution_v234.png" in content
        assert "598/598" in content
    log = (ROOT / "docs/operator_3d_learning_log.md").read_text(encoding="utf-8")
    assert "v234" in log
    assert "added fallback is both more expensive" in log


def test_v234_public_artifacts_do_not_expose_private_execution_details() -> None:
    text = SUMMARY.read_text(encoding="utf-8") + RESULT.read_text(encoding="utf-8")
    forbidden = (
        "/Users/",
        "private_results",
        "private_worktrees",
        "source_commit",
        "sha256",
        "checkpoint.pt",
        "62e42c34",
    )
    assert all(token not in text for token in forbidden)
