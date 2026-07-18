from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

import demo_t16_operator.run_n2_pvgr_n3_grouped_factorial as grouped


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "demo_t16_operator/configs/n2_pvgr_n3_grouped_factorial_preregistered_v1.json"
)


def _config() -> dict[str, object]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _source(config: dict[str, object]) -> dict[str, object]:
    return json.loads((ROOT / str(config["source_config"])).read_text(encoding="utf-8"))


def test_factorial_is_exactly_eight_field_units_and_96_repeated_conditions() -> None:
    config = _config()
    source = _source(config)
    grouped._validate_contract(config, source)
    cases = grouped.expand_factorial_cases(config)
    assert len(cases) == 32
    assert len({case["field_unit_id"] for case in cases}) == 8
    assert len({case["id"] for case in cases}) == 32
    assert (
        len(cases) * len(config["factorial"]["dimensionless_stress_scale_multipliers"])
        == 96
    )
    for field_unit in {case["field_unit_id"] for case in cases}:
        assert sum(case["field_unit_id"] == field_unit for case in cases) == 4


def test_reserved_family_or_replaced_seed_contract_is_rejected() -> None:
    config = _config()
    source = _source(config)
    config["factorial"]["field_families"][0]["phantom_family"] = "shock_expansion_pair"
    with pytest.raises(ValueError, match="field-family or field-seed"):
        grouped._validate_contract(config, source)

    config = _config()
    config["factorial"]["field_families"][0]["field_seeds"] = [1, 1, 2, 3]
    with pytest.raises(ValueError, match="field-family or field-seed"):
        grouped._validate_contract(config, source)

    config = _config()
    config["factorial"]["common_sobol_prefix_by_field_unit"] = False
    with pytest.raises(ValueError, match="common Sobol prefix"):
        grouped._validate_contract(config, source)


def test_field_unit_execution_case_reuses_common_sobol_prefix() -> None:
    config = _config()
    source = grouped._source_for_run(
        _source(config),
        config,
        grouped.expand_factorial_cases(config),
    )
    cases = grouped.expand_factorial_cases(config)
    same_field = [case for case in cases if case["field_unit_id"] == "smooth-s1729"]
    first_values, first_states, _ = grouped.bridge._build_case_context(
        grouped._execution_case(same_field[0]), source
    )
    for case in same_field[1:]:
        values, states, _ = grouped.bridge._build_case_context(
            grouped._execution_case(case), source
        )
        assert torch.equal(values, first_values)
        assert torch.equal(states, first_states)
    assert len(first_states) == 256
    assert len({case["id"] for case in same_field}) == 4


def test_symmetric_gate_uses_absolute_fallback_for_unscoreable_ratio() -> None:
    config = _config()
    screens = config["development_screens"]
    metrics = {
        "matched_residual_prediction_relative_l2": 0.001,
        "corrected_residual_variance_ratio": 0.001,
        "per_ray_risk_spearman": 0.999,
        "valid_ray_fraction": 1.0,
        "candidate_to_high_reference_relative_l2": 0.001,
        "high_execution_to_high_reference_relative_l2": 1e-14,
        "candidate_reference_error_to_high_execution_reference_error_ratio": 1e8,
        "candidate_q95_reference_error": 5e-13,
        "high_execution_q95_reference_error": 1e-14,
        "candidate_q95_reference_error_to_high_execution_q95_ratio": 50.0,
    }
    gates = grouped._symmetric_method_gates(metrics, screens)
    assert gates["absolute_reference_gate_met"]
    assert gates["global_no_harm_gate_met"]
    assert gates["q95_no_harm_gate_met"]
    metrics["candidate_to_high_reference_relative_l2"] = 0.003
    assert not grouped._symmetric_method_gates(metrics, screens)[
        "global_no_harm_gate_met"
    ]
    metrics["candidate_to_high_reference_relative_l2"] = 0.001
    metrics["per_ray_risk_spearman"] = 0.98
    assert not grouped._symmetric_method_gates(metrics, screens)[
        "risk_spearman_gate_met"
    ]


def test_floor_stabilized_dominance_ratio_fails_closed_at_machine_zero() -> None:
    ratio, scoreable = grouped._floor_stabilized_ratio(
        9e-13,
        1e-14,
        scoreability_floor=1e-12,
    )
    assert not scoreable
    assert ratio == pytest.approx(0.9)
    ratio, scoreable = grouped._floor_stabilized_ratio(
        2e-12,
        1e-14,
        scoreability_floor=1e-12,
    )
    assert not scoreable
    assert ratio == pytest.approx(2.0)


def _metric(matched: float) -> dict[str, float]:
    return {
        "matched_residual_prediction_relative_l2": matched,
        "corrected_residual_variance_ratio": 0.001,
        "per_ray_risk_spearman": 0.999,
        "valid_ray_fraction": 1.0,
        "candidate_to_high_reference_relative_l2": 0.001,
        "high_execution_to_high_reference_relative_l2": 0.001,
        "candidate_reference_error_to_high_execution_reference_error_ratio": 1.0,
        "candidate_q95_reference_error": 1e-5,
        "high_execution_q95_reference_error": 1e-5,
        "candidate_q95_reference_error_to_high_execution_q95_ratio": 1.0,
        "candidate_q95_reference_error_normalized_by_high256_rms": 1e-4,
    }


def _synthetic_dominance_inputs() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    method_rows: list[dict[str, object]] = []
    query_rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []
    matched = {
        "continuous_affine_n1": 0.018,
        "operator_consistent_homotopy": 0.010,
        "picard_1": 0.005,
        "picard_2": 0.004,
    }
    for family in ("smooth", "wrinkled"):
        for seed_index in range(4):
            field_unit = f"{family}-s{seed_index}"
            for orientation in ("o1", "o2"):
                for aperture in ("a1", "a2"):
                    for stress in (1.0, 3.0, 10.0):
                        for method in grouped.METHODS:
                            method_rows.append(
                                {
                                    "field_unit_id": field_unit,
                                    "family_id": family,
                                    "phantom_family": family,
                                    "phantom_seed": seed_index,
                                    "orientation_id": orientation,
                                    "aperture_id": aperture,
                                    "dimensionless_stress_multiplier": stress,
                                    "method_id": method,
                                    "metrics": _metric(matched[method]),
                                    "symmetric_gates": {
                                        "matched_residual_gate_met": True,
                                        "residual_variance_gate_met": True,
                                        "risk_spearman_gate_met": True,
                                        "valid_fraction_gate_met": True,
                                        "absolute_reference_gate_met": True,
                                        "global_no_harm_gate_met": True,
                                        "q95_no_harm_gate_met": True,
                                    },
                                    "all_symmetric_gates_pass": True,
                                }
                            )
                        query_rows.append(
                            {
                                "field_unit_id": field_unit,
                                "orientation_id": orientation,
                                "aperture_id": aperture,
                                "dimensionless_stress_multiplier": stress,
                                "query_accounting": {
                                    "operator_consistent_homotopy": {
                                        "logical_scalar_grid_point_queries": 101
                                    },
                                    "picard_1": {
                                        "logical_scalar_grid_point_queries": 100
                                    },
                                    "picard_2": {
                                        "logical_scalar_grid_point_queries": 150
                                    },
                                    "high128": {
                                        "logical_scalar_grid_point_queries": 250
                                    },
                                },
                            }
                        )
    for orientation in ("o1", "o2"):
        for aperture in ("a1", "a2"):
            package = f"{orientation}__{aperture}"
            for method, p90 in (
                ("operator_consistent_homotopy", 1.0),
                ("picard_1", 0.5),
                ("picard_2", 0.4),
                ("continuous_affine_n1", 0.8),
                ("high128", 2.0),
            ):
                timing_rows.append(
                    {
                        "timing_package_id": package,
                        "method_id": method,
                        "p90_seconds": p90,
                    }
                )
    return method_rows, timing_rows, query_rows


def test_grouped_dominance_uses_field_pairs_and_query_ratio_vs_ocbh() -> None:
    config = _config()
    method_rows, timing_rows, query_rows = _synthetic_dominance_inputs()
    field_rows = grouped._field_unit_summaries(method_rows, config)
    decisions = grouped._dominance_rows(
        method_rows,
        field_rows,
        timing_rows,
        query_rows,
        config,
    )
    by_method = {row["method_id"]: row for row in decisions}
    assert by_method["picard_1"]["dominates_ocbh_forward_role"]
    assert by_method["picard_1"][
        "maximum_logical_point_query_ratio_vs_ocbh"
    ] == pytest.approx(100 / 101)
    assert not by_method["picard_2"]["dominates_ocbh_forward_role"]
    assert not by_method["picard_2"]["dominance_gates"]["query_gate"]
    assert (
        by_method["picard_1"]["field_units_with_lower_geometric_mean_matched_error"]
        == 8
    )
    assert by_method["picard_1"]["global_scoreable_condition_count"] == 96
    assert by_method["picard_1"]["q95_scoreable_condition_count"] == 96


def test_blocked_bootstrap_is_deterministic_and_family_balanced() -> None:
    rows = [
        {
            "field_unit_id": f"{family}-{index}",
            "family_id": family,
            "matched_ratio_vs_ocbh": ratio,
        }
        for family, ratios in (
            ("smooth", (0.4, 0.5, 0.6, 0.7)),
            ("wrinkled", (0.3, 0.4, 0.5, 0.6)),
        )
        for index, ratio in enumerate(ratios)
    ]
    first = grouped._blocked_bootstrap_ci(
        rows, seed=7, replicates=1000, confidence_level=0.95
    )
    second = grouped._blocked_bootstrap_ci(
        rows, seed=7, replicates=1000, confidence_level=0.95
    )
    assert first == second
    assert first[1] < first[0] < first[2]


def test_committed_attestation_validates_when_present() -> None:
    config = _config()
    attestation = ROOT / str(config["pre_registration_attestation"])
    if not attestation.exists():
        formal_output = ROOT / str(config["formal_output"])
        formal_work_output = ROOT / str(config["formal_work_output"])
        assert not formal_output.exists()
        assert not formal_work_output.exists()
        pytest.skip("attestation is intentionally created in the second prereg commit")
    validated = grouped._validate_preregistration(config, CONFIG)
    assert validated["formal_results_absent_at_creation"] is True
    assert validated["formal_work_output_absent_at_creation"] is True


def test_execution_case_does_not_mutate_visible_condition_identity() -> None:
    case = grouped.expand_factorial_cases(_config())[0]
    original = copy.deepcopy(case)
    execution = grouped._execution_case(case)
    assert case == original
    assert execution["id"] == case["field_unit_id"]
    assert execution["rig"] == case["rig"]


def test_final_artifact_staging_replaces_only_its_own_partial_directory(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    cells = work / "cells"
    cells.mkdir(parents=True)
    (cells / "keep.json").write_text("{}", encoding="utf-8")
    partial = work / "final_artifacts_staging"
    partial.mkdir()
    (partial / "partial.json").write_text("{}", encoding="utf-8")
    staging = grouped._prepare_staging_dir(work)
    assert staging == partial
    assert list(staging.iterdir()) == []
    assert (cells / "keep.json").is_file()
