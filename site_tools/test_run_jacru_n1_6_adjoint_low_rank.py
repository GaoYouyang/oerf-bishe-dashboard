from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import pytest

from site_tools import run_jacru_n1_6_adjoint_low_rank as runner


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "demo_t16_operator/configs/"
    "jacru_n1_6_adjoint_low_rank_development_v1.json"
)


def _payloads():
    config = json.loads(CONFIG.read_text())
    n15a = json.loads((ROOT / config["source_n1_5_a_config"]).read_text())
    return config, n15a


def test_config_accepts_postopen_deployment_contract() -> None:
    config, n15a = _payloads()
    runner._validate_config(config, n15a, seed_limit=None)


def test_config_rejects_high_order_deployment_call() -> None:
    config, n15a = _payloads()
    changed = copy.deepcopy(config)
    changed["deployment_contract"]["high_order_forward_calls"] = 1
    with pytest.raises(ValueError, match="zero high-order"):
        runner._validate_config(changed, n15a, seed_limit=None)


def test_config_rejects_truth_contract_drift() -> None:
    config, n15a = _payloads()
    changed = copy.deepcopy(config)
    changed["deployment_contract"]["forbidden_inputs"].remove("truth_volume")
    with pytest.raises(ValueError, match="forbidden-input"):
        runner._validate_config(changed, n15a, seed_limit=None)


def test_config_rejects_unmatched_reference_budget() -> None:
    config, n15a = _payloads()
    changed = copy.deepcopy(config)
    changed["budget"]["low_reference_cgls_iterations"] = 25
    with pytest.raises(ValueError, match="reference physical-call"):
        runner._validate_config(changed, n15a, seed_limit=None)


def test_seed_limited_smoke_writes_auditable_package(tmp_path: Path) -> None:
    output = tmp_path / "n16-smoke"
    old_argv = runner.sys.argv
    runner.sys.argv = [
        "run_jacru_n1_6_adjoint_low_rank.py",
        "--config",
        str(CONFIG),
        "--output-dir",
        str(output),
        "--seed-limit",
        "1",
    ]
    try:
        assert runner.main() == 0
    finally:
        runner.sys.argv = old_argv
    expected = {
        "README.md",
        "calibration_adjoint_diagnostic_rows.csv",
        "calibration_case_rows.csv",
        "calibration_model_rows.csv",
        "case_manifest.csv",
        "checksums.sha256",
        "config_snapshot.json",
        "diagnostic.pdf",
        "diagnostic.png",
        "fit_adjoint_target_rows.csv",
        "provenance.json",
        "selected_aggregate_metrics.csv",
        "selected_adjoint_diagnostic_rows.csv",
        "selected_case_metrics.csv",
        "selected_model.json",
        "summary.json",
    }
    assert {path.name for path in output.iterdir()} == expected
    summary = json.loads((output / "summary.json").read_text())
    assert summary["may_claim_confirmed_algorithm_gain"] is False
    assert summary["opens_ood_fresh_or_final"] is False
    assert summary["deployment_contract"]["high_order_forward_calls"] == 0
    assert summary["budget"]["deployable_total_low_forward_calls"] == 25
    assert summary["budget"]["deployable_total_low_adjoint_calls"] == 24
    assert summary["fit_geometry_cluster_count"] == 1
    assert summary["calibration_geometry_cluster_count"] == 1
    assert summary["development_geometry_cluster_count"] == 1
    assert summary["runtime_observation_source"] == "case.inference.observations_uv"
    assert summary["split_integrity_audit"]["partition_seed_disjoint"] is True
    with (output / "selected_case_metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if row["candidate_id"] == "selected_fail_closed"]
    teachers = [row for row in rows if row["candidate_id"] == "high_order_teacher_b0p75"]
    assert selected and teachers
    assert "adjoint_residual_ratio_to_component_damping" not in selected[0]
    assert "hidden_evaluator_adjoint_calls" not in selected[0]
    assert all(row["low_forward_calls"] == "25" for row in selected)
    assert all(row["low_adjoint_calls"] == "24" for row in selected)
    assert all(row["high_order_forward_calls"] == "0" for row in selected)
    assert all(row["evaluator_only"] == "False" for row in selected)
    assert all(row["high_order_forward_calls"] == "1" for row in teachers)
    assert all(row["evaluator_only"] == "True" for row in teachers)
    with (output / "selected_adjoint_diagnostic_rows.csv").open(newline="") as handle:
        diagnostics = list(csv.DictReader(handle))
    selected_diagnostics = [
        row for row in diagnostics if row["candidate_id"] == "selected_fail_closed"
    ]
    assert selected_diagnostics
    assert all(row["fresh_exact_mismatch_access"] == "True" for row in selected_diagnostics)
    assert all(row["evaluator_only"] == "True" for row in selected_diagnostics)


def test_duplicate_candidate_case_is_rejected() -> None:
    row = {
        "candidate_id": "candidate",
        "base_seed": 7,
        "family": "smooth_no_interface",
    }
    with pytest.raises(RuntimeError, match="duplicate candidate case"):
        runner._index_rows([row, dict(row)], "candidate")
