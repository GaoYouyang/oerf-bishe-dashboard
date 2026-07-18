from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import demo_t16_operator.run_n2_pvgr_n2_operator_consistent_bridge as bridge


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "demo_t16_operator"
    / "configs"
    / "n2_pvgr_n2_operator_consistent_bridge_v1.json"
)


def test_frozen_config_keeps_reserved_families_closed() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source = json.loads(
        (ROOT / config["source_config"]).read_text(encoding="utf-8")
    )
    bridge._validate_contracts(config, source)
    assert config["execution_step_count"] == 128
    assert config["reference_step_count"] == 256
    assert config["reference_sentinel_step_count"] == 512
    assert config["reserved_audit_families_not_opened"] == [
        "oblique_compression_sheet",
        "shock_expansion_pair",
    ]


def test_contract_rejects_reserved_family_opening() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source = json.loads(
        (ROOT / config["source_config"]).read_text(encoding="utf-8")
    )
    source["development_cases"][0]["phantom_family"] = (
        "oblique_compression_sheet"
    )
    with pytest.raises(ValueError, match="reserved"):
        bridge._validate_contracts(config, source)


def test_method_metrics_separate_matched_and_reference_errors() -> None:
    medium = torch.tensor([[1.0, 0.0], [0.9, 0.1], [1.1, -0.1]], dtype=torch.float64)
    high128 = medium + torch.tensor(
        [[0.02, 0.01], [0.01, 0.02], [0.03, 0.01]],
        dtype=torch.float64,
    )
    high256 = high128 + torch.tensor(
        [[0.001, -0.001], [0.001, 0.0], [0.0, 0.001]],
        dtype=torch.float64,
    )
    output = high128.clone()
    risk = torch.linalg.vector_norm(output - medium, dim=-1)
    metrics = bridge._method_metrics(
        output,
        risk,
        torch.ones(3, dtype=torch.bool),
        medium128=medium,
        high128=high128,
        high256=high256,
    )
    assert metrics["matched_residual_prediction_relative_l2"] == pytest.approx(0.0)
    assert metrics["corrected_residual_variance_ratio"] == pytest.approx(0.0)
    assert metrics[
        "candidate_reference_error_to_high_execution_reference_error_ratio"
    ] == pytest.approx(1.0)
    assert metrics[
        "candidate_q95_reference_error_to_high_execution_q95_ratio"
    ] == pytest.approx(1.0)


def test_primary_gates_fail_the_reference_tail_independently() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    screens = config["development_screens"]
    metrics = {
        "matched_residual_prediction_relative_l2": 0.001,
        "corrected_residual_variance_ratio": 0.001,
        "per_ray_risk_spearman": 0.999,
        "valid_ray_fraction": 1.0,
        "candidate_to_high_reference_relative_l2": 0.001,
        "candidate_reference_error_to_high_execution_reference_error_ratio": 1.0,
        "candidate_q95_reference_error_to_high_execution_q95_ratio": 1.2,
    }
    gates = bridge._primary_gates(metrics, screens)
    assert all(value for key, value in gates.items() if key != "q95_reference_no_harm_gate_met")
    assert not gates["q95_reference_no_harm_gate_met"]


def test_run_writes_machine_readable_artifacts_without_opening_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_bundle(
        case: dict[str, object],
        source: dict[str, object],
        config: dict[str, object],
        *,
        stress: float,
    ) -> dict[str, object]:
        del source, config
        method_row = {
            "case_id": str(case["id"]),
            "phantom_family": str(case["phantom_family"]),
            "phantom_seed": int(case["phantom_seed"]),
            "dimensionless_stress_multiplier": float(stress),
            "refractivity_scale": 3e-4 * float(stress),
            "method_id": "operator_consistent_homotopy",
            "ray_count": 64,
            "step_count": 128,
            "correction_applied": True,
            "metrics": {
                "matched_residual_prediction_relative_l2": 0.001,
                "corrected_residual_variance_ratio": 1e-5,
                "per_ray_risk_spearman": 0.999,
                "valid_ray_fraction": 1.0,
                "candidate_to_high_reference_relative_l2": 3e-4,
                "high_execution_to_high_reference_relative_l2": 3e-4,
                "candidate_reference_error_to_high_execution_reference_error_ratio": 1.0,
                "candidate_q95_reference_error": 1e-5,
                "high_execution_q95_reference_error": 1e-5,
                "candidate_q95_reference_error_to_high_execution_q95_ratio": 1.0,
                "candidate_q95_reference_error_normalized_by_high256_rms": 1e-4,
                "base_output_relative_l2": 0.0,
            },
            "gates": {"all_fake_gate_met": True},
            "all_primary_gates_pass": True,
        }
        return {
            "method_rows": [method_row],
            "primary_all_pass": True,
            "teacher_row": {
                "case_id": str(case["id"]),
                "dimensionless_stress_multiplier": float(stress),
                "metrics": {
                    "output_relative_l2": 1e-14,
                    "position_tangent_relative_l2": 1e-14,
                    "direction_tangent_relative_l2": 1e-14,
                    "teacher_valid_ray_fraction": 1.0,
                },
                "gates": {"teacher_fake_gate_met": True},
                "all_gates_pass": True,
            },
            "sentinel_row": {
                "case_id": str(case["id"]),
                "dimensionless_stress_multiplier": float(stress),
                "metrics": {
                    "high256_to_high512_output_relative_l2": 1e-5,
                    "matched_residual_256_to_512_relative_l2": 1e-3,
                },
                "gates": {"sentinel_fake_gate_met": True},
                "all_gates_pass": True,
            },
            "query_accounting": {
                "operator_consistent_homotopy": {
                    "logical_scalar_grid_point_queries": 100,
                },
                "teacher_discrete_jvp": {},
                "picard_1": {},
                "picard_2": {},
                "high128": {"logical_scalar_grid_point_queries": 300},
            },
        }

    def fake_timing(
        case: dict[str, object],
        source: dict[str, object],
        config: dict[str, object],
    ) -> list[dict[str, object]]:
        del source, config
        return [
            {
                "case_id": str(case["id"]),
                "dimensionless_stress_multiplier": 1.0,
                "method_id": "operator_consistent_homotopy",
                "sample_count": 2,
                "p10_seconds": 0.01,
                "p50_seconds": 0.01,
                "p90_seconds": 0.01,
                "candidate_p90_to_high128_p10_wall_time_ratio": 0.1,
                "timing_gate_met": True,
            },
            {
                "case_id": str(case["id"]),
                "dimensionless_stress_multiplier": 1.0,
                "method_id": "high128",
                "sample_count": 2,
                "p10_seconds": 0.1,
                "p50_seconds": 0.1,
                "p90_seconds": 0.1,
                "candidate_p90_to_high128_p10_wall_time_ratio": 1.0,
            },
        ]

    monkeypatch.setattr(bridge, "_case_scale_bundle", fake_bundle)
    monkeypatch.setattr(bridge, "_timing_bundle", fake_timing)
    monkeypatch.setattr(
        bridge,
        "_write_figure",
        lambda path, *args: path.write_bytes(b"fake-png"),
    )
    result = bridge.run(CONFIG, tmp_path)
    assert result["machine_decision"].startswith("MECHANISM_BRIDGE_SIGNAL_ONLY")
    assert not result["development_bridge_authorization"]
    assert not result["paper_claim_authorization"]
    for name in (
        "result.json",
        "metrics.csv",
        "teacher_metrics.csv",
        "reference_sentinel.csv",
        "timing.csv",
        "manifest.json",
        "summary.md",
        "n2_pvgr_n2_operator_consistent_bridge.png",
    ):
        assert (tmp_path / name).is_file()
