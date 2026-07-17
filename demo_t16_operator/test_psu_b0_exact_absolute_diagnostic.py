from __future__ import annotations

import torch

from demo_t16_operator.psu_b0_exact_absolute_diagnostic import (
    build_exact_absolute_audit,
    dense_normalized_spectral_norm_squared,
    describe_exact_absolute_metric,
    run_exact_absolute_trajectory,
    schur_safety_certificate_squared,
    summarize_tightness,
)
from demo_t16_operator.psu_b0_gate_a_fixture import (
    build_gate_a_fixture,
    load_gate_a_config,
)
from demo_t16_operator.psu_b0_gate_b_data_only import (
    build_single_sample_factor_setup,
)
from demo_t16_operator.test_psu_b0_gate_b_data_only import _SourceWhitening


def _fixture() -> tuple[object, object]:
    fixture = build_gate_a_fixture(load_gate_a_config())
    config = fixture.config["fixture"]
    source = _SourceWhitening(
        torch.as_tensor(config["whitening_matrix_by_view"], dtype=torch.float64),
        torch.as_tensor(config["scale_by_view"], dtype=torch.float64),
    )
    setup = build_single_sample_factor_setup(
        voxel_operator=fixture.setup.pipeline.voxel_operator,
        source_whitening=source,
        sample_index=0,
        measurement_scale=float(config["measurement_scale"]),
        eta=0.7,
    )
    return fixture, setup


def _dense_signed_and_factor(setup: object) -> tuple[torch.Tensor, torch.Tensor]:
    count = setup.active_primal_count
    basis = torch.zeros((count, setup.pipeline.n_active), dtype=torch.float64)
    basis[torch.arange(count), setup.active_primal_indices] = 1.0
    signed = setup.pipeline.signed_data_forward(basis).reshape(count, -1).T
    factor = setup.pipeline.absolute_data_forward(basis).reshape(count, -1).T
    return signed, factor


def test_streamed_exact_absolute_audit_matches_independent_dense_oracle() -> None:
    _, setup = _fixture()
    signed, factor = _dense_signed_and_factor(setup)
    audit = build_exact_absolute_audit(setup, batch_size=2)

    assert torch.allclose(audit.exact_row_sums, torch.abs(signed).sum(1))
    assert torch.allclose(audit.exact_column_sums, torch.abs(signed).sum(0))
    assert torch.allclose(audit.factor_row_sums, factor.sum(1))
    assert torch.allclose(audit.factor_column_sums, factor.sum(0))
    assert torch.all(torch.abs(signed) <= factor + 1e-12)
    assert audit.dominance_violation_maximum <= 1e-12
    assert audit.exact_only_nonzero_count == 0
    assert audit.setup_factor_row_relative_error <= 1e-12
    assert audit.setup_factor_column_relative_error <= 1e-12
    summary = summarize_tightness(audit)
    assert 0.0 < summary["global_exact_to_factor_mass_ratio"] <= 1.0
    assert 0.0 <= summary["global_slack_mass"] < 1.0


def test_all_diagnostic_metrics_are_schur_safe_in_dense_oracle() -> None:
    _, setup = _fixture()
    signed, _ = _dense_signed_and_factor(setup)
    audit = build_exact_absolute_audit(setup, batch_size=3)

    for mode in ("factor_row", "exact_view", "exact_row"):
        metric = describe_exact_absolute_metric(setup, audit, mode)
        squared_norm = dense_normalized_spectral_norm_squared(signed, metric)
        assert squared_norm <= setup.eta**2 + 1e-12
        assert schur_safety_certificate_squared(
            setup,
            audit,
            metric,
            dominance_absolute_tolerance=1e-12,
            dominance_relative_tolerance=1e-12,
        ) == setup.eta**2


def test_exact_row_recurrence_matches_independent_dense_replay() -> None:
    fixture, setup = _fixture()
    dense_a, _ = _dense_signed_and_factor(setup)
    audit = build_exact_absolute_audit(setup, batch_size=2)
    metric = describe_exact_absolute_metric(setup, audit, "exact_row")
    target = fixture.target.reshape(-1)

    x = torch.zeros(setup.active_primal_count, dtype=torch.float64)
    x_bar = torch.zeros_like(x)
    dual = torch.zeros_like(target)
    for _ in range(4):
        dual = (
            dual + metric.sigma_rows * (dense_a @ x_bar - target)
        ) / (1.0 + metric.sigma_rows)
        dual = torch.where(metric.sigma_rows > 0.0, dual, 0.0)
        next_x = x - metric.tau * (dense_a.T @ dual)
        x_bar = next_x + (next_x - x)
        x = next_x

    setup.pipeline.reset_call_ledger()
    observed = run_exact_absolute_trajectory(
        setup,
        fixture.target,
        metric,
        checkpoints=(4,),
    )[4]
    assert torch.allclose(observed.x, x, atol=1e-12, rtol=1e-12)
    assert torch.allclose(observed.x_bar, x_bar, atol=1e-12, rtol=1e-12)
    assert torch.allclose(observed.data_dual.reshape(-1), dual, atol=1e-12, rtol=1e-12)
    assert setup.pipeline.call_ledger().signed_data_forward_calls == 4
    assert setup.pipeline.call_ledger().signed_data_transpose_calls == 4
