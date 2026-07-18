from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from demo_t16_operator.run_n2_adrc_n1_development_pilot import (
    DEFAULT_CONFIG,
    read_json,
    run_experiment,
    write_outputs,
)


def _tiny_config() -> dict:
    config = copy.deepcopy(read_json(DEFAULT_CONFIG))
    config["grid_shape_zyx"] = [9, 9, 9]
    config["calibration_population_count"] = 32
    config["reference_population_count"] = 64
    config["estimator_replicates"] = 48
    config["bootstrap_replicates"] = 64
    config["matched_cost_high_equivalent_samples"] = 16
    config["fixed_state_derivative_check"] = {
        "low_count": 8,
        "residual_count": 4,
    }
    config["timing"] = {"warmup_repeats": 1, "measured_repeats": 2}
    config["development_cases"] = config["development_cases"][:1]
    screens = config["development_promotion_screens"]
    screens["minimum_predicted_measured_cost_gain"] = 0.0
    screens["minimum_conservative_timing_efficiency_gain"] = 0.0
    screens["minimum_empirical_mse_gain"] = 0.0
    screens["minimum_empirical_gain_bootstrap_lower_95"] = 0.0
    screens["maximum_reference_half_to_full_relative_l2"] = 1.0
    screens["minimum_cases_passing_efficiency_screen"] = 1
    return config


def test_tiny_development_run_keeps_the_hard_conclusion(tmp_path: Path) -> None:
    config = _tiny_config()
    result = run_experiment(config)
    assert result["machine_decision"] == "DEVELOPMENT_ONLY_NO_AUDIT_AUTHORIZATION"
    assert result["promotion_screen_meaning"] == "may_design_unseen_audit_only"
    assert result["case_count"] == 1
    assert result["cases"][0]["fixed_state_derivative"][
        "trajectory_sensitivity_audited"
    ] is False
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "result"
    write_outputs(output, config=config, result=result, config_path=config_path)
    expected = {
        "config_snapshot.json",
        "manifest.json",
        "metrics.csv",
        "n2_adrc_n1_development_pilot.png",
        "result.json",
        "summary.md",
    }
    assert {path.name for path in output.iterdir()} == expected
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["files"]) == expected - {"manifest.json"}
    assert manifest["config_source"] == "config.json"
    assert set(manifest["source_files"]) == {
        "demo_t16_operator/analytic_bost_phantoms.py",
        "demo_t16_operator/automatic_discrete_multifidelity.py",
        "demo_t16_operator/run_n2_adrc_n1_development_pilot.py",
    }
    assert "/Users/" not in json.dumps(manifest)


def test_development_runner_rejects_a_relaxed_machine_conclusion() -> None:
    config = _tiny_config()
    config["hard_conclusion"] = "PASS"
    with pytest.raises(ValueError, match="hard conclusion"):
        run_experiment(config)
