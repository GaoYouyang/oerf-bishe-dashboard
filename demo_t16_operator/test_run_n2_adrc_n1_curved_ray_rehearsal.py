from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from demo_t16_operator import run_n2_adrc_n1_curved_ray_rehearsal as runner


def _small_config() -> dict:
    config = runner.read_json(runner.DEFAULT_CONFIG)
    config["grid_shape_zyx"] = [9, 9, 9]
    config["population_count"] = 8
    config["reference_step_counts"] = [4, 8]
    config["dimensionless_stress_scale_multipliers"] = [1]
    config["derivative_check"].update(
        {
            "ray_count": 2,
            "step_count": 4,
            "finite_difference_epsilon": 2e-5,
        }
    )
    config["timing"] = {"warmup_repeats": 0, "measured_repeats": 1}
    config["development_rehearsal_cases"] = [
        copy.deepcopy(config["development_rehearsal_cases"][0])
    ]
    config["rehearsal_screens"]["maximum_high_half_to_full_relative_l2"] = 1.0
    config["rehearsal_screens"]["maximum_exit_vs_integral_relative_l2"] = 1.0
    config["rehearsal_screens"]["maximum_momentum_balance_relative_l2"] = 1.0
    return config


def test_rehearsal_writes_fail_closed_public_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.json"
    output = tmp_path / "result"
    config_path.write_text(json.dumps(_small_config()), encoding="utf-8")
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)

    result = runner.run(config_path, output)

    assert result["machine_decision"] == (
        "CURVED_RAY_REHEARSAL_ONLY_RESERVED_FAMILIES_UNOPENED"
    )
    assert result["case_count"] == 1
    assert result["reserved_audit_families_not_opened"] == [
        "oblique_compression_sheet",
        "shock_expansion_pair",
    ]
    case = result["cases"][0]
    assert case["dimensionless_stress_envelope"][0]["scale_multiplier"] == 1.0
    assert "momentum_balance" in case["rehearsal_screen_checks"]
    for name in (
        "result.json",
        "config_snapshot.json",
        "metrics.csv",
        "manifest.json",
        "summary.md",
        "n2_adrc_n1_curved_ray_rehearsal.png",
        "n2_adrc_n1_refractivity_envelope.png",
    ):
        assert (output / name).is_file()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"] == "config.json"
    assert manifest["reserved_audit_families_opened"] is False
    assert set(manifest["files"]) == {
        "result.json",
        "config_snapshot.json",
        "metrics.csv",
        "summary.md",
        "n2_adrc_n1_curved_ray_rehearsal.png",
        "n2_adrc_n1_refractivity_envelope.png",
    }


def test_rehearsal_refuses_a_reserved_family(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _small_config()
    config["development_rehearsal_cases"][0][
        "phantom_family"
    ] = "oblique_compression_sheet"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)

    with pytest.raises(RuntimeError, match="reserved family"):
        runner.run(config_path, tmp_path / "result")
