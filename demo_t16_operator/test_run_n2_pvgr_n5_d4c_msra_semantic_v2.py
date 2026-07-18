from __future__ import annotations

import json
from pathlib import Path

import pytest

from demo_t16_operator.run_n2_pvgr_n5_d4c_msra_semantic_v2 import (
    EXPECTED_SCENARIOS,
    ROOT,
    _build_trial_variants,
    _evaluate_variant,
    _validate_config,
    run,
)


CONFIG = (
    ROOT
    / "demo_t16_operator/configs/"
    "n2_pvgr_n5_d4c_msra_semantic_v2_preregistered.json"
)


def _config() -> dict[str, object]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _small_config(tmp_path: Path) -> Path:
    config = _config()
    config["trial_count"] = 4
    config["probe_counts"] = [1, 2, 4]
    config["h_values"] = [0.001, 0.01]
    config["side_weighted_gamma_threshold_grid"] = [0.5, 2.0]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def test_config_forbids_claim_authorization() -> None:
    config = _config()
    config["claim_authorizations"]["field_derivative_interface"] = True
    with pytest.raises(ValueError, match="pre-authorize"):
        _validate_config(config)


def test_config_requires_every_semantic_obligation() -> None:
    config = _config()
    config["semantic_requirements"]["finite_difference"] = ""
    with pytest.raises(ValueError, match="finite_difference"):
        _validate_config(config)


def test_forward_states_distinguish_diagnostic_and_actual_branch() -> None:
    config = _config()
    config["probe_counts"] = [1, 2, 4]
    variants = _build_trial_variants(config, 0)
    diagnostic = next(
        item for item in variants if item.scenario == "diagnostic_only_support_flip"
    )
    branch = next(
        item
        for item in variants
        if item.scenario == "actual_piecewise_branch_crossing"
    )

    _, diagnostic_fd, _, diagnostic_evidence = _evaluate_variant(config, diagnostic)
    _, branch_fd, _, branch_evidence = _evaluate_variant(config, branch)

    assert any(row["diagnostic_state_changed"] for row in diagnostic_fd)
    assert not any(row["actual_forward_branch_changed"] for row in diagnostic_fd)
    assert all(row["diagnostic_state_changed"] for row in diagnostic_evidence)
    assert not any(
        row["actual_forward_branch_changed"] for row in diagnostic_evidence
    )
    assert all(row["actual_forward_branch_changed"] for row in branch_fd)
    assert all(row["actual_forward_branch_changed"] for row in branch_evidence)


def test_three_path_fault_can_pass_direct_fd_but_fail_structure() -> None:
    config = _config()
    config["probe_counts"] = [1, 2, 4]
    variant = next(
        item
        for item in _build_trial_variants(config, 0)
        if item.scenario == "three_path_structure_mismatch"
        and item.parameter_value == 1e-6
    )
    _, _, structure_rows, evidence_rows = _evaluate_variant(config, variant)

    assert structure_rows
    assert max(
        row["maximum_structure_relative_error"] for row in structure_rows
    ) > float(config["structure_relative_threshold"])
    assert max(
        row["maximum_fd_relative_error"] for row in evidence_rows
    ) <= float(config["finite_difference_relative_threshold"])


def test_semantic_v2_run_emits_raw_evidence_without_selecting_threshold(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result"
    config_path = _small_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = run(config_path, output, require_committed_source=False)

    assert result["machine_decision"] == (
        "D4C_MSRA_SEMANTIC_V2_CHARACTERIZED_NO_BOST_AUTHORIZATION"
    )
    assert result["threshold_selected"] is None
    assert result["threshold_selection_forbidden"] is True
    assert result["counts"]["trial_count"] == 4
    assert result["counts"]["scenario_count"] == len(EXPECTED_SCENARIOS)
    assert result["h_values_consumed"] == [0.001, 0.01]
    assert result["headline_diagnostics"]["diagnostic_actual_branch_change_count"] == 0
    assert result["headline_diagnostics"]["actual_branch_change_count"] == 4
    assert all(value is False for value in result["claim_authorizations"].values())
    assert not any(
        "overall" in key or "accuracy" in key
        for row in result["scenario_summary"]
        for key in row
    )

    variants_per_trial = 2 + 2 * len(config["cancellation_deltas"]) + (
        5 * len(config["fault_relative_magnitudes"])
    ) + 2
    assert result["counts"]["variant_count"] == 4 * variants_per_trial
    assert result["counts"]["fd_rows"] == (
        result["counts"]["variant_count"]
        * max(config["probe_counts"])
        * len(config["h_values"])
    )

    for name in (
        "case_specs.jsonl",
        "probe_rows.csv",
        "fd_rows.csv",
        "structure_rows.csv",
        "evidence_rows.csv",
        "decision_rows.csv",
        "scenario_summary.csv",
        "result.json",
        "summary.md",
        "semantic_v2.png",
        "manifest.json",
    ):
        assert (output / name).is_file()


def test_semantic_v2_run_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "occupied"
    output.mkdir()
    with pytest.raises(FileExistsError):
        run(_small_config(tmp_path), output, require_committed_source=False)
