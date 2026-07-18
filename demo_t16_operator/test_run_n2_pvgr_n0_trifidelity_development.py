from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

try:
    from .run_n2_pvgr_n0_trifidelity_development import DEFAULT_CONFIG, run
except ImportError:
    from run_n2_pvgr_n0_trifidelity_development import DEFAULT_CONFIG, run


def _reduced_config(tmp_path: Path) -> Path:
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["population_count"] = 8
    config["dimensionless_stress_scale_multipliers"] = [1]
    config["route_step_counts"] = {
        "low_automatic": 8,
        "medium_straight_central": 8,
        "high_curved_central": 8,
        "high_reference": 16,
    }
    config["certificate"]["support_interval_count"] = 16
    config["routing"]["monte_carlo_replicates"] = 512
    config["routing"]["quadratic_loss_replicates"] = 256
    config["derivative_contract"].update(
        {
            "ray_count": 2,
            "step_count": 4,
            "monte_carlo_replicates": 256,
        }
    )
    config["timing"] = {"warmup_repeats": 0, "measured_repeats": 1}
    config["development_cases"] = config["development_cases"][:1]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def test_reduced_runner_writes_replayable_development_artifacts(tmp_path: Path) -> None:
    config_path = _reduced_config(tmp_path)
    output = tmp_path / "result"
    result = run(config_path, output)
    assert result["machine_decision"] == "DEVELOPMENT_ONLY_NO_AUDIT_AUTHORIZATION"
    assert result["case_scale_count"] == 1
    assert result["reserved_audit_families_not_opened"] == [
        "oblique_compression_sheet",
        "shock_expansion_pair",
    ]
    row = result["rows"][0]
    assert row["routing"]["sparse_execution"]["sparse_to_replay_relative_l2"] < 1e-10
    assert row["derivative_contract"]["probabilities_detached"] is True
    expected = {
        "result.json",
        "config_snapshot.json",
        "metrics.csv",
        "summary.md",
        "n2_pvgr_n0_trifidelity_development.png",
        "n2_pvgr_n0_derivative_contract.png",
        "manifest.json",
    }
    assert expected == {path.name for path in output.iterdir()}
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for name, expected_hash in manifest["files"].items():
        actual = hashlib.sha256((output / name).read_bytes()).hexdigest()
        assert actual == expected_hash


def test_runner_refuses_reserved_family_before_execution(tmp_path: Path) -> None:
    config_path = _reduced_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["development_cases"][0]["phantom_family"] = "shock_expansion_pair"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="reserved audit families"):
        run(config_path, tmp_path / "blocked")
