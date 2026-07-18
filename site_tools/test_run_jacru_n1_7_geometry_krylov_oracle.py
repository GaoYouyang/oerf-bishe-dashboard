from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import pytest
import torch

from site_tools import run_jacru_n1_7_geometry_krylov_oracle as runner


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "demo_t16_operator/configs/jacru_n1_7_geometry_krylov_postopen_v1.json"
)


def _payloads():
    config = json.loads(CONFIG.read_text())
    n15a = json.loads((ROOT / config["source_n1_5_a_config"]).read_text())
    return config, n15a


def test_config_accepts_frozen_postopen_representation_contract() -> None:
    config, n15a = _payloads()
    runner._validate_config(config, n15a, seed_limit=None)


def test_config_rejects_unmatched_probe_budget() -> None:
    config, n15a = _payloads()
    changed = copy.deepcopy(config)
    changed["budget"]["krylov_probe_forward_calls"] = 1
    with pytest.raises(ValueError, match="candidate physical-call"):
        runner._validate_config(changed, n15a, seed_limit=None)


def test_config_rejects_target_conditioned_trust_radius() -> None:
    config, n15a = _payloads()
    changed = copy.deepcopy(config)
    changed["coefficient_trust_region"]["uses_exact_target"] = True
    with pytest.raises(ValueError, match="must not access"):
        runner._validate_config(changed, n15a, seed_limit=None)


def test_deployable_case_view_cannot_hold_evaluator_labels_or_truth() -> None:
    fields = set(runner.DeployableCaseView.__dataclass_fields__)
    assert fields == {
        "measured_observation",
        "signal_scale",
        "warm_field",
        "warm_projection",
        "damping_normalized",
        "shared_warm_seconds",
        "operator",
    }
    assert not fields & {"truth_volume", "mismatch_normalized", "family", "record"}


def test_visible_radius_obeys_floor_expansion_and_cap() -> None:
    config, _ = _payloads()
    damping = torch.tensor([3.0, 4.0], dtype=torch.float64)
    tiny = torch.tensor([0.0, 0.0], dtype=torch.float64)
    medium = torch.tensor([0.2, 0.0], dtype=torch.float64)
    large = torch.tensor([10.0, 0.0], dtype=torch.float64)
    assert runner._coefficient_radius(damping, tiny, config) == pytest.approx(0.5)
    assert runner._coefficient_radius(damping, medium, config) == pytest.approx(0.8)
    assert runner._coefficient_radius(damping, large, config) == pytest.approx(2.5)


def test_call_delta_is_stage_local() -> None:
    assert runner._call_delta(
        {"forward_calls": 12, "adjoint_calls": 9},
        {"forward_calls": 14, "adjoint_calls": 11},
    ) == {"forward_calls": 2, "adjoint_calls": 2}


def test_seed_limited_smoke_writes_separated_route_and_oracle_ledgers(
    tmp_path: Path,
) -> None:
    output = tmp_path / "n17-smoke"
    old_argv = runner.sys.argv
    runner.sys.argv = [
        "run_jacru_n1_7_geometry_krylov_oracle.py",
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
        "aggregate_metrics.csv",
        "basis_diagnostics.csv",
        "case_manifest.csv",
        "case_metrics.csv",
        "checksums.sha256",
        "config_snapshot.json",
        "diagnostic.pdf",
        "diagnostic.png",
        "finite_k_search_diagnostics.csv",
        "provenance.json",
        "summary.json",
        "target_diagnostics.csv",
    }
    assert {path.name for path in output.iterdir()} == expected
    summary = json.loads((output / "summary.json").read_text())
    assert summary["learner_was_trained"] is False
    assert summary["opens_ood_fresh_or_final"] is False
    assert summary["development_case_count"] == 2
    assert summary["finite_k_evaluator_total_forward_calls"] > 0
    assert (
        summary["finite_k_evaluator_total_forward_calls"]
        == summary["finite_k_evaluator_total_adjoint_calls"]
    )
    assert summary["finite_k_evaluator_total_scope"] == (
        "development_only_legacy_field_name"
    )
    assert summary["finite_k_evaluator_total_forward_calls"] == summary[
        "finite_k_evaluator_development_forward_calls"
    ]
    assert summary["finite_k_evaluator_total_adjoint_calls"] == summary[
        "finite_k_evaluator_development_adjoint_calls"
    ]
    assert summary["finite_k_evaluator_package_forward_calls"] == (
        summary["finite_k_evaluator_development_forward_calls"]
        + summary["finite_k_evaluator_calibration_forward_calls"]
    )
    assert summary["finite_k_evaluator_package_adjoint_calls"] == (
        summary["finite_k_evaluator_development_adjoint_calls"]
        + summary["finite_k_evaluator_calibration_adjoint_calls"]
    )
    assert (
        summary["finite_k_evaluator_package_forward_calls"]
        == summary["finite_k_evaluator_package_adjoint_calls"]
    )
    rows = list(csv.DictReader((output / "case_metrics.csv").open(newline="")))
    assert rows
    assert all(row["low_forward_calls"] == "25" for row in rows)
    assert all(row["low_adjoint_calls"] == "24" for row in rows)
    field_oracle = [
        row
        for row in rows
        if row["candidate_id"] == "truth_conditioned_finite_k_oracle_search"
    ]
    assert field_oracle
    assert all(int(row["evaluator_search_forward_calls"]) > 0 for row in field_oracle)
    assert all(row["evaluator_only"] == "True" for row in field_oracle)
