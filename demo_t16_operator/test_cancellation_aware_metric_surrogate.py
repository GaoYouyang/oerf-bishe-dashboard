from __future__ import annotations

from dataclasses import fields
import math

import pytest
import torch

from demo_t16_operator.cancellation_aware_metric_surrogate import (
    SCHEMA_VERSION,
    STATUS,
    ExactMassInstrumentation,
    InferenceRigFeatures,
    apply_calibration_envelope,
    audit_schur_safety,
    build_diagonal_metric,
    exact_mass_access_scope,
    exact_masses_from_signed_matrix,
    fit_calibration_envelope,
    fit_metric_surrogate,
    generate_tiny_rigs,
    inference_features_from_rig,
    predict_metric_masses,
    run_signed_residual_trajectory,
    split_rigs_three_way,
)


def _assignments() -> dict[str, str]:
    return {
        "train-00": "train",
        "train-01": "train",
        "train-02": "train",
        "cal-00": "safety_calibration",
        "cal-01": "safety_calibration",
        "ood-00": "fresh_geometry_ood",
        "ood-01": "fresh_geometry_ood",
    }


def _rigs() -> list[object]:
    return generate_tiny_rigs(
        split_assignments=_assignments(),
        geometry_seed=2026071711,
        noise_seed=2026071813,
        row_count=10,
        column_count=8,
        dtype=torch.float64,
    )


def test_status_schema_and_inference_contract_exclude_all_teachers() -> None:
    assert SCHEMA_VERSION == "cancellation-aware-metric-surrogate-smoke-2.0"
    assert STATUS == "DEVELOPMENT_ONLY_SYNTHETIC_INTERFACE_SMOKE"
    assert {field.name for field in fields(InferenceRigFeatures)} == {
        "rig_id",
        "row_features",
        "column_features",
    }
    forbidden = {"signed_matrix", "exact_row_mass", "exact_column_mass", "truth", "target"}
    assert forbidden.isdisjoint({field.name for field in fields(InferenceRigFeatures)})


def test_independent_geometry_generation_is_deterministic_and_three_way() -> None:
    first = _rigs()
    second = _rigs()
    for left, right in zip(first, second, strict=True):
        torch.testing.assert_close(left.geometry_features, right.geometry_features, rtol=0, atol=0)
        torch.testing.assert_close(left.signed_matrix, right.signed_matrix, rtol=0, atol=0)
        torch.testing.assert_close(left.target, right.target, rtol=0, atol=0)
        exact_row, exact_column = exact_masses_from_signed_matrix(left.signed_matrix)
        torch.testing.assert_close(left.exact_row_mass, exact_row)
        torch.testing.assert_close(left.exact_column_mass, exact_column)
        assert torch.all(left.factor_majorizer >= torch.abs(left.signed_matrix))
    train, calibration, fresh, contract = split_rigs_three_way(first)
    assert [len(train), len(calibration), len(fresh)] == [3, 2, 2]
    assert contract["geometry_parameters_independently_sampled"] is True
    assert contract["noise_seed_is_not_geometry_seed"] is True
    assert set(contract["train_rig_ids"]).isdisjoint(contract["fresh_geometry_ood_rig_ids"])
    # Five geometry features are not a deterministic one-dimensional index curve.
    geometry = torch.stack([rig.geometry_features for rig in first])
    assert int(torch.linalg.matrix_rank(geometry - geometry.mean(dim=0))) >= 3


def test_per_rig_sha_seed_is_invariant_to_reordering_and_unrelated_addition() -> None:
    original = _assignments()
    reordered = dict(reversed(list(original.items())))
    extended = {"unrelated-train": "train", **reordered}

    def generate(assignments: dict[str, str]) -> dict[str, object]:
        return {
            rig.rig_id: rig
            for rig in generate_tiny_rigs(
                split_assignments=assignments,
                geometry_seed=2026071711,
                noise_seed=2026071813,
                row_count=10,
                column_count=8,
                dtype=torch.float64,
            )
        }

    baseline, changed = generate(original), generate(extended)
    for rig_id in original:
        left, right = baseline[rig_id], changed[rig_id]
        assert left.geometry_seed == right.geometry_seed
        assert left.noise_seed == right.noise_seed
        assert left.geometry_parameters == right.geometry_parameters
        assert left.geometry_parameters_sha256 == right.geometry_parameters_sha256
        torch.testing.assert_close(left.signed_matrix, right.signed_matrix, rtol=0, atol=0)
        torch.testing.assert_close(left.target, right.target, rtol=0, atol=0)


def test_prediction_accepts_only_deployment_features_and_calibration_is_frozen() -> None:
    train, calibration, fresh, _ = split_rigs_three_way(_rigs())
    estimator, training = fit_metric_surrogate(
        train, hidden_dim=12, steps=35, learning_rate=0.02, seed=2026071917
    )
    assert training["exact_teacher_rig_count"] == len(train)
    features = inference_features_from_rig(fresh[0])
    predicted_row, predicted_column = predict_metric_masses(estimator, features)
    assert torch.all(predicted_row > 0) and torch.all(predicted_column > 0)
    with pytest.raises(TypeError, match="InferenceRigFeatures"):
        predict_metric_masses(estimator, fresh[0])  # type: ignore[arg-type]

    instrumentation = ExactMassInstrumentation.empty()
    envelope = fit_calibration_envelope(
        estimator,
        calibration,
        margin=1.02,
        instrumentation=instrumentation,
    )
    row, column = apply_calibration_envelope(predicted_row, predicted_column, envelope)
    torch.testing.assert_close(row, predicted_row * envelope.row_scale)
    torch.testing.assert_close(column, predicted_column * envelope.column_scale)
    assert envelope.exact_mass_materializations == len(calibration)
    assert envelope.factor_mass_vector_accesses == 2 * len(calibration)
    assert fresh[0].rig_id not in envelope.calibration_rig_ids


def test_fresh_exact_access_guard_blocks_candidate_phase_and_records_attempt() -> None:
    _, _, fresh, _ = split_rigs_three_way(_rigs())
    rig = fresh[0]
    instrumentation = ExactMassInstrumentation.empty()
    with pytest.raises(RuntimeError, match="fresh exact-mass access blocked"):
        with exact_mass_access_scope(
            instrumentation,
            rig_id=rig.rig_id,
            split_role=rig.split_role,
            phase="learned_prediction_setup",
            fresh_access_allowed=False,
        ):
            exact_masses_from_signed_matrix(rig.signed_matrix)
    summary = instrumentation.summary()
    assert summary["blocked_exact_mass_access_count"] == 1
    assert summary["fresh_candidate_exact_mass_access_count"] == 1
    assert summary["fresh_candidate_exact_access"] is True


def test_schur_audit_recomputes_exact_and_counts_spectral_violation() -> None:
    rig = _rigs()[0]
    exact_row, exact_column = exact_masses_from_signed_matrix(rig.signed_matrix)
    safe = build_diagonal_metric(exact_row, exact_column, eta=0.7)
    safe_audit = audit_schur_safety(rig.signed_matrix, safe, eta=0.7)
    assert safe_audit["total_violation_count"] == 0
    assert safe_audit["spectral_violation_count"] == 0
    assert safe_audit["exact_masses_recomputed_from_signed_a"] == 1

    unsafe = build_diagonal_metric(0.1 * exact_row, 0.1 * exact_column, eta=0.7)
    unsafe_audit = audit_schur_safety(rig.signed_matrix, unsafe, eta=0.7)
    assert unsafe_audit["row_violation_count"] > 0
    assert unsafe_audit["column_violation_count"] > 0
    assert unsafe_audit["spectral_violation_count"] == 1
    assert unsafe_audit["total_violation_count"] == (
        unsafe_audit["row_violation_count"]
        + unsafe_audit["column_violation_count"]
        + unsafe_audit["spectral_violation_count"]
    )


def test_trajectory_reports_frozen_field_relative_l2_and_equal_call_budget() -> None:
    rig = _rigs()[0]
    checkpoints = (0, 1, 2, 4, 8)
    results = {}
    for name, masses in {
        "factor": (rig.factor_row_mass, rig.factor_column_mass),
        "exact": exact_masses_from_signed_matrix(rig.signed_matrix),
    }.items():
        results[name] = run_signed_residual_trajectory(
            rig.signed_matrix,
            rig.target,
            rig.truth,
            build_diagonal_metric(*masses, eta=0.7),
            checkpoints=checkpoints,
            theta=1.0,
        )
    for result in results.values():
        assert [row["iteration"] for row in result.rows] == list(checkpoints)
        assert all(math.isfinite(float(row["field_relative_l2"])) for row in result.rows)
        assert result.rows[0]["field_relative_l2"] == pytest.approx(1.0)
        assert result.ledger == {
            "signed_forward_solver_calls": 8,
            "signed_transpose_solver_calls": 8,
            "signed_forward_evaluation_calls": len(checkpoints),
            "field_error_evaluation_calls": len(checkpoints),
            "iteration_budget": 8,
        }
    assert results["exact"].rows[-1]["field_relative_l2"] < results["factor"].rows[-1]["field_relative_l2"]
