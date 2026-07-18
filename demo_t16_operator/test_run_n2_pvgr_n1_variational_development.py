from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

try:
    from .run_n2_pvgr_n1_variational_development import (
        DEFAULT_CONFIG,
        ROOT,
        _norm_ratio,
        run,
    )
except ImportError:
    from run_n2_pvgr_n1_variational_development import (
        DEFAULT_CONFIG,
        ROOT,
        _norm_ratio,
        run,
    )


def _reduced_contract(tmp_path: Path) -> tuple[Path, Path]:
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    source = json.loads(
        (ROOT / config["source_config"]).read_text(encoding="utf-8")
    )
    source["population_count"] = 8
    source["dimensionless_stress_scale_multipliers"] = [1]
    source["development_cases"] = source["development_cases"][:1]
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")

    convergence = {
        "machine_decision": "RESIDUAL_TARGET_128_ACCEPTED_DEVELOPMENT_ONLY",
        "accepted_execution_step_count": 8,
        "reference_step_count": 16,
        "reserved_audit_families_not_opened": source[
            "reserved_audit_families_not_opened"
        ],
    }
    convergence_path = tmp_path / "convergence.json"
    convergence_path.write_text(
        json.dumps(convergence, indent=2) + "\n",
        encoding="utf-8",
    )
    config.update(
        {
            "source_config": str(source_path),
            "residual_convergence_audit": str(convergence_path),
            "matched_execution_step_count": 8,
            "reference_step_count": 16,
            "timing": {
                "warmup_repeats": 0,
                "measured_repeats": 5,
                "interleave_seed": 74001,
            },
            "development_screens": {
                "maximum_matched_residual_prediction_relative_l2": 2.0,
                "maximum_corrected_residual_variance_ratio": 2.0,
                "minimum_per_ray_risk_spearman": -1.0,
                "minimum_valid_ray_fraction": 1.0,
                "maximum_candidate_p90_to_full_high_p10_wall_time_ratio": 10.0,
                "maximum_candidate_to_high_reference_relative_l2": 2.0,
                "maximum_candidate_reference_error_to_high_execution_reference_error_ratio": 10.0,
            },
            "hard_conclusion_if_all_development_screens_pass": "TEST_SIGNAL",
            "hard_conclusion_otherwise": "TEST_NO_SIGNAL",
        }
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config_path, source_path


def test_norm_ratio_measures_remaining_residual_not_residual_difference() -> None:
    reference = torch.tensor([[3.0, 4.0]], dtype=torch.float64)
    assert _norm_ratio(torch.zeros_like(reference), reference) == pytest.approx(0.0)
    assert _norm_ratio(0.1 * reference, reference) == pytest.approx(0.1)


def test_reduced_runner_writes_replayable_artifacts(tmp_path: Path) -> None:
    config_path, _ = _reduced_contract(tmp_path)
    output = tmp_path / "result"
    result = run(config_path, output)

    assert result["machine_decision"] == "TEST_SIGNAL"
    assert result["case_scale_count"] == 1
    assert result["development_screen_pass_count"] == 1
    assert result["reserved_audit_families_not_opened"] == [
        "oblique_compression_sheet",
        "shock_expansion_pair",
    ]
    row = result["rows"][0]
    assert row["all_development_screens_pass"]
    assert row["metrics"]["valid_ray_fraction"] == pytest.approx(1.0)
    assert result["query_accounting"]["full_high"][
        "logical_scalar_grid_point_queries"
    ] == 35 * 8 * 8
    expected = {
        "config_snapshot.json",
        "manifest.json",
        "metrics.csv",
        "n2_pvgr_n1_variational_development.png",
        "result.json",
        "summary.md",
    }
    assert expected == {path.name for path in output.iterdir()}
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for name, expected_hash in manifest["files"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected_hash
    assert b"\r\n" not in (output / "metrics.csv").read_bytes()


def test_runner_refuses_reserved_family_before_output(tmp_path: Path) -> None:
    config_path, source_path = _reduced_contract(tmp_path)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["development_cases"][0]["phantom_family"] = "shock_expansion_pair"
    source_path.write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="reserved audit family"):
        run(config_path, tmp_path / "blocked")
    assert not (tmp_path / "blocked").exists()
