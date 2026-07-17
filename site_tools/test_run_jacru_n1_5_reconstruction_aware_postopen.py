from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from site_tools import run_jacru_n1_5_reconstruction_aware_postopen as runner


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "demo_t16_operator/configs/jacru_n1_5_reconstruction_aware_postopen_development_v1.json"


def _payloads():
    config = json.loads(CONFIG.read_text())
    summary = json.loads((ROOT / config["source_n1_5_a_results"] / "summary.json").read_text())
    return config, summary


def test_config_accepts_frozen_postopen_contract() -> None:
    config, summary = _payloads()
    runner._validate_config(config, summary, None)


def test_config_rejects_confirmed_claim() -> None:
    config, summary = _payloads()
    changed = copy.deepcopy(config)
    changed["claim_boundary"]["may_claim_confirmed_algorithm_gain"] = True
    with pytest.raises(ValueError, match="cannot claim"):
        runner._validate_config(changed, summary, None)


def test_config_rejects_call_budget_drift() -> None:
    config, summary = _payloads()
    changed = copy.deepcopy(config)
    changed["budget"]["corrected_total_low_forward_calls"] = 24
    with pytest.raises(ValueError, match="forward-call"):
        runner._validate_config(changed, summary, None)


def test_seed_limited_smoke_writes_result_package(tmp_path: Path) -> None:
    output = tmp_path / "n15b-smoke"
    old_argv = runner.sys.argv
    runner.sys.argv = [
        "run_jacru_n1_5_reconstruction_aware_postopen.py",
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
        "case_manifest.csv",
        "case_metrics.csv",
        "checksums.sha256",
        "diagnostic.pdf",
        "diagnostic.png",
        "provenance.json",
        "summary.json",
    }
    assert {path.name for path in output.iterdir()} == expected
    summary = json.loads((output / "summary.json").read_text())
    assert summary["may_claim_confirmed_algorithm_gain"] is False
    assert summary["opens_ood_fresh_or_final"] is False
    assert summary["calibration_geometry_cluster_count"] == 1
    assert summary["development_geometry_cluster_count"] == 1
