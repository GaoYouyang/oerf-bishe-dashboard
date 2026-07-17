from __future__ import annotations

from pathlib import Path

import numpy as np

from site_tools.validate_psu_b0_exact_absolute_diagnostic import (
    power_iteration_estimate,
    parse_checksum_manifest,
    primitive_exact_absolute_certificate,
    recompute_decision,
    recompute_summaries,
    replay_primal_dual_recurrence,
    validate_metric_rows,
    validate_config,
    validate_tightness_rows,
)


def test_frozen_config_requires_all_mapped_sources_but_not_this_validator() -> None:
    config = validate_config(
        Path(
            "demo_t16_operator/configs/"
            "psu_b0_exact_absolute_root_cause_v3_cpu64_audit_amendment.json"
        )
    )
    mapped = set(config["source_paths"].values())
    assert "site_tools/validate_psu_b0_exact_absolute_diagnostic.py" not in mapped
    assert "site_tools/test_validate_psu_b0_exact_absolute_diagnostic.py" not in mapped


def test_checksum_contract_requires_trajectory_rows(tmp_path: Path) -> None:
    digest = "a" * 64
    manifest = tmp_path / "checksums.sha256"
    manifest.write_text(
        "".join(
            f"{digest}  {name}\n"
            for name in (
                "report.json",
                "trajectory_rows.csv",
                "tightness_rows.csv",
                "audit_rows.json",
            )
        ),
        encoding="ascii",
    )
    assert set(parse_checksum_manifest(manifest)) == {
        "report.json",
        "trajectory_rows.csv",
        "tightness_rows.csv",
        "audit_rows.json",
    }


def test_tiny_float64_fixture_reconstructs_signed_a_factor_m_and_schur_certificate() -> None:
    # Primitive frozen arrays: the fourth column is M-active but A-null.
    signed_a = np.array([[1.0, -2.0, 0.0, 0.0], [0.0, 1.0, -1.0, 0.0], [-1.0, 0.0, 1.0, 0.0]], dtype=np.float64)
    factor_m = np.array([[1.2, 2.1, 0.0, 0.2], [0.0, 1.2, 1.1, 0.3], [1.1, 0.0, 1.1, 0.4]], dtype=np.float64)
    oracle = primitive_exact_absolute_certificate(signed_a, factor_m)

    assert np.all(factor_m >= np.abs(signed_a))
    np.testing.assert_array_equal(oracle["exact_row_sums"], np.array([3.0, 2.0, 2.0]))
    np.testing.assert_array_equal(oracle["exact_column_sums"], np.array([2.0, 3.0, 2.0, 0.0]))
    np.testing.assert_array_equal(oracle["m_active_columns"], np.array([0, 1, 2, 3]))
    np.testing.assert_array_equal(oracle["a_nonzero_columns"], np.array([0, 1, 2]))
    np.testing.assert_array_equal(oracle["m_active_a_null_columns"], np.array([3]))
    assert oracle["nullspace_dimension"] == 1
    assert oracle["schur_norm_squared"] <= oracle["eta_squared"] + 1e-12


def test_tiny_float64_fixture_adjoint_recurrence_and_power_estimate() -> None:
    signed_a = np.array([[1.0, -2.0, 0.0], [0.0, 1.0, -1.0], [-1.0, 0.0, 1.0]], dtype=np.float64)
    factor_m = np.abs(signed_a) + np.array([[0.2, 0.1, 0.0], [0.0, 0.2, 0.1], [0.1, 0.0, 0.1]], dtype=np.float64)
    oracle = primitive_exact_absolute_certificate(signed_a, factor_m)
    probe_x, probe_y = np.array([0.3, -0.7, 1.1]), np.array([-0.2, 0.4, 0.9])
    assert np.isclose(np.dot(signed_a @ probe_x, probe_y), np.dot(probe_x, signed_a.T @ probe_y), atol=1e-14)

    tau = 0.7 / oracle["factor_column_sums"]
    sigma = 0.7 / oracle["factor_row_sums"]
    target = np.array([0.25, -0.4, 0.8])
    observed = replay_primal_dual_recurrence(signed_a, target, tau, sigma, steps=4)
    # An explicit primitive replay guards the update order and theta=1 extrapolation.
    x = np.zeros(3); x_bar = x.copy(); dual = np.zeros(3)
    for _ in range(4):
        dual = (dual + sigma * (signed_a @ x_bar - target)) / (1.0 + sigma)
        next_x = x - tau * (signed_a.T @ dual)
        x_bar, x = next_x + (next_x - x), next_x
    for actual, expected in zip(observed, (x, x_bar, dual), strict=True):
        np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-14)

    estimate = power_iteration_estimate(signed_a, tau, sigma, iterations=32, seed=7)
    assert estimate <= oracle["eta_squared"] + 1e-12


def test_independent_csv_arithmetic_preserves_descriptive_no_reopen_label() -> None:
    methods = {
        "scalar_a_only_pdhg": 1.2, "formal_factor_view_a_only_pdhg": 1.0,
        "factor_row_hybrid_a_only_pdhg": 0.7, "exact_abs_view_a_only_pdhg": 0.6,
        "exact_abs_row_a_only_pdhg": 0.5, "graph_pcgls": 0.4,
    }
    checkpoints = (4, 8, 16, 32, 64, 128)
    families = ("plume", "wavy_front", "thin_front", "double_front", "annular_kernel", "oblique_shock", "vortex_pair", "multi_plume")
    rows = [{
        "replicate": replicate, "sample_index": sample, "reaction_family": family,
        "method": method, "iterations": iteration, "forward_calls": iteration,
        "adjoint_calls": iteration, "field_relative_l2": error,
        "gradient_relative_l2": error / 2, "front_top10_f1": 0.5,
        "data_coupled_relative_l2": error, "data_null_support_relative_l2": error,
        "data_coupled_error_energy": error, "data_null_support_error_energy": error,
        "data_null_support_reconstruction_energy": error,
        "normalized_data_residual_l2": "" if method == "graph_pcgls" else error,
        "trajectory_elapsed_seconds": 0.01,
    } for replicate in (0, 8) for sample, family in enumerate(families)
        for method, error in methods.items() for iteration in checkpoints]
    tightness = [{
        "replicate": replicate, "sample_index": sample, "reaction_family": family,
        "row_ratio_minimum": 0.4, "row_ratio_p05": 0.4, "row_ratio_median": 0.4, "row_ratio_mean": 0.4,
        "column_ratio_minimum": 0.4, "column_ratio_p05": 0.4, "column_ratio_median": 0.4, "column_ratio_mean": 0.4,
        "global_exact_to_factor_mass_ratio": 0.4, "global_slack_mass": 0.6,
        "exact_zero_row_count": 0, "exact_zero_column_count": 0, "factor_only_nonzero_count": 1,
        "exact_only_nonzero_count": 0, "factor_majorizer_active_coordinate_count": 2322,
        "signed_A_nonzero_coordinate_count": 2322, "M_active_A_zero_coordinate_count": 0,
        "nullspace_dimension_claimed": False, "dominance_violation_maximum": 0.0,
        "dominance_relative_violation_maximum": 0.0,
            "setup_factor_row_relative_error": 0.0, "setup_factor_column_relative_error": 0.0,
            "audit_content_sha256": "a" * 64,
            "mps_repeat_content_sha256": (
                "a" * 64 if replicate == 0 and sample == 0 else ""
            ),
            "mps_repeat_required_for_this_row": replicate == 0 and sample == 0,
            "solver_mps_setup_call_ledger": {"signed_data_forward_calls": 1},
            "audit_cpu64_setup_call_ledger": {"signed_data_forward_calls": 1},
        "exact_streaming_replay_call_ledger": {
            "signed_data_forward_calls": 19, "signed_data_transpose_calls": 0,
            "absolute_data_forward_calls": 19, "absolute_data_transpose_calls": 0,
            "signed_tv_forward_calls": 0, "signed_tv_transpose_calls": 0,
            "absolute_tv_forward_calls": 0, "absolute_tv_transpose_calls": 0,
        },
    } for replicate in (0, 8) for sample, family in enumerate(families)]
    thresholds = {
        "dominance_absolute_tolerance": 1e-12, "dominance_relative_tolerance": 1e-12,
        "factor_replay_relative_error_maximum": 1e-12, "descriptive_endpoint_k": 128,
        "material_residual_gain_percent_min": 10.0, "material_paired_sample_count_min": 12,
        "material_high_quantile_slack_min": 0.1, "nonbinding_exact_to_graph_field_ratio_min": 1.5,
    }
    validate_metric_rows(rows)
    validate_tightness_rows(tightness, thresholds)
    assert len(recompute_summaries(rows)) == len(methods) * len(checkpoints)
    decision = recompute_decision(rows, tightness, thresholds)
    assert decision["status"] == "FACTOR_MAJORIZER_CANCELLATION_MATERIAL_DESCRIPTIVE"
    assert decision["formal_gate_b_reopened"] is False
