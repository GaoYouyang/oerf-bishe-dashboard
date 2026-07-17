from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from site_tools import run_jacru_n1_5_approximation_error_headroom as runner


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "demo_t16_operator/configs/jacru_n1_5_approximation_error_headroom_development_v1.json"
SOURCE = ROOT / "demo_t16_operator/configs/jacru_m2_learned_residual_t0_v1.json"


def _configs():
    return json.loads(CONFIG.read_text()), json.loads(SOURCE.read_text())


def test_config_has_disjoint_complete_seed_contract() -> None:
    config, source = _configs()
    runner._validate_config(config, source, None)


def test_config_rejects_ood_access() -> None:
    config, source = _configs()
    changed = copy.deepcopy(config)
    changed["may_construct_or_evaluate_ood"] = True
    with pytest.raises(RuntimeError, match="OOD"):
        runner._validate_config(changed, source, None)


def test_config_rejects_train_partition_leakage() -> None:
    config, source = _configs()
    changed = copy.deepcopy(config)
    changed["calibration"]["base_seeds"].append(changed["fit"]["base_seeds"][0])
    with pytest.raises(ValueError, match="disjoint"):
        runner._validate_config(changed, source, None)


def test_config_rejects_pca_oracle_gate() -> None:
    config, source = _configs()
    changed = copy.deepcopy(config)
    changed["pca_oracle"]["participates_in_gate"] = True
    with pytest.raises(ValueError, match="cannot participate"):
        runner._validate_config(changed, source, None)


def test_seed_limited_smoke_writes_auditable_package(tmp_path: Path) -> None:
    output = tmp_path / "n15-smoke"
    old_argv = runner.sys.argv
    runner.sys.argv = [
        "run_jacru_n1_5_approximation_error_headroom.py",
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
        "calibration_alpha_rows.csv",
        "candidate_summary.csv",
        "case_manifest.csv",
        "case_metrics.csv",
        "checksums.sha256",
        "diagnostic.pdf",
        "diagnostic.png",
        "geometry_cluster_metrics.csv",
        "pca_oracle_rows.csv",
        "provenance.json",
        "selected_ridge_models.json",
        "summary.json",
    }
    assert {path.name for path in output.iterdir()} == expected
    summary = json.loads((output / "summary.json").read_text())
    assert summary["opens_ood_fresh_or_final"] is False
    assert summary["pca_oracle_participates_in_gate"] is False
    assert summary["fit_geometry_cluster_count"] == 1
    assert summary["development_geometry_cluster_count"] == 1
