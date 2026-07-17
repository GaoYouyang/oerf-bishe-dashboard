from __future__ import annotations

import pytest
import torch
from torch import nn

from demo_t16_operator.psu_b0_factor_majorizer_pipeline import (
    FactorPipelineCallLedger,
)
from demo_t16_operator.psu_b0_gate_a_fixture import (
    build_gate_a_fixture,
    load_gate_a_config,
)
from demo_t16_operator.psu_b0_gate_b_data_only import (
    SingleSampleViewLocalWhitening,
    build_single_sample_factor_setup,
    describe_gate_b_metric,
    factor_state_to_volume,
    run_gate_b_data_only_trajectory,
)


class _SourceWhitening(nn.Module):
    def __init__(self, matrix: torch.Tensor, scales: torch.Tensor) -> None:
        super().__init__()
        self.view_count = int(matrix.shape[0])
        self.rays_per_view = int(matrix.shape[1] // 2)
        self.register_buffer("matrix", matrix)
        self.register_buffer("scale_by_view", scales)
        self.register_buffer(
            "calibration_mean",
            torch.zeros(
                (self.view_count, 2 * self.rays_per_view),
                dtype=matrix.dtype,
                device=matrix.device,
            ),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        canonical = values.reshape(
            len(values),
            self.view_count,
            2 * self.rays_per_view,
        )
        projected = torch.einsum("vij,bvj->bvi", self.matrix, canonical)
        projected = projected / self.scale_by_view[:, :, None]
        return projected.reshape_as(values)


def _gate_b_fixture() -> tuple[object, _SourceWhitening, object]:
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
    return fixture, source, setup


def test_single_sample_whitening_matches_source_row_and_transpose_identity() -> None:
    _, source, _ = _gate_b_fixture()
    frozen = SingleSampleViewLocalWhitening(source, 0)
    generator = torch.Generator().manual_seed(1707)
    detector = torch.randn((1, 2, 2), generator=generator, dtype=torch.float64)
    dual = torch.randn((1, 2, 2), generator=generator, dtype=torch.float64)

    assert torch.equal(frozen(detector), source(detector))
    lhs = torch.sum(frozen(detector) * dual)
    rhs = torch.sum(detector * frozen.transpose(dual))
    assert float(torch.abs(lhs - rhs)) <= 1e-12
    assert frozen.whitening_block_scope == "view_local"
    assert frozen.independent_whitening_blocks is True
    assert frozen.covariance_block_ids == (0, 1)


def test_single_sample_whitening_accepts_production_canonical_calibration_mean() -> None:
    _, source, _ = _gate_b_fixture()
    source.calibration_mean = source.calibration_mean.reshape(
        source.view_count,
        source.rays_per_view,
        2,
    )
    frozen = SingleSampleViewLocalWhitening(source, 0)
    assert frozen.calibration_mean.shape == (
        source.view_count,
        2 * source.rays_per_view,
    )
    values = torch.zeros((1, 2, 2), dtype=torch.float64)
    assert torch.equal(frozen.center_observation(values), values)


def test_factor_setup_matches_measurement_scaled_whitened_physical_forward() -> None:
    fixture, source, setup = _gate_b_fixture()
    voxel = fixture.setup.pipeline.voxel_operator
    generator = torch.Generator().manual_seed(42)
    volume = torch.randn(
        (1, 1, *voxel.grid_shape),
        generator=generator,
        dtype=torch.float64,
    )
    volume = volume * voxel.support[None, None]
    active = setup.pipeline.restrict_active(volume)
    observed = setup.pipeline.signed_data_forward(active)
    expected = source(voxel(volume)) * float(
        fixture.config["fixture"]["measurement_scale"]
    )
    assert torch.allclose(observed, expected, atol=1e-12, rtol=1e-12)


def test_a_only_setup_uses_no_tv_factor_and_exact_data_column_steps() -> None:
    _, _, setup = _gate_b_fixture()
    expected = FactorPipelineCallLedger(
        absolute_data_forward_calls=1,
        absolute_data_transpose_calls=1,
    )
    assert setup.setup_call_ledger == expected
    assert setup.setup_physical_call_ledger == expected
    selected_columns = setup.data_column_sums.index_select(
        0,
        setup.active_primal_indices,
    )
    assert torch.equal(setup.tau_voxel_factor, setup.eta / selected_columns)
    assert setup.active_primal_count == setup.pipeline.n_active


@pytest.mark.parametrize("mode", ["scalar", "view_block", "voxel_factor"])
def test_all_a_only_metrics_match_independent_dense_recurrence(mode: str) -> None:
    fixture, _, setup = _gate_b_fixture()
    metric = describe_gate_b_metric(setup, mode)
    reduced_count = setup.active_primal_count
    active_basis = torch.zeros(
        (reduced_count, setup.pipeline.n_active),
        dtype=torch.float64,
    )
    active_basis[
        torch.arange(reduced_count),
        setup.active_primal_indices,
    ] = 1.0
    dense_a = setup.pipeline.signed_data_forward(active_basis).reshape(
        reduced_count,
        -1,
    ).T
    setup.pipeline.reset_call_ledger()

    target = fixture.target.reshape(-1)
    sigma = metric.sigma_data_by_view[:, None, None].expand_as(
        setup.data_row_mask
    )
    sigma = torch.where(setup.data_row_mask, sigma, torch.zeros_like(sigma)).reshape(-1)
    x = torch.zeros(reduced_count, dtype=torch.float64)
    x_bar = torch.zeros_like(x)
    dual = torch.zeros_like(target)
    for _ in range(4):
        next_dual = (dual + sigma * (dense_a @ x_bar - target)) / (1.0 + sigma)
        next_dual = torch.where(setup.data_row_mask.reshape(-1), next_dual, 0.0)
        next_x = x - metric.tau * (dense_a.T @ next_dual)
        x_bar = next_x + (next_x - x)
        x = next_x
        dual = next_dual

    observed = run_gate_b_data_only_trajectory(
        setup,
        fixture.target,
        checkpoints=(4,),
        mode=mode,
    )[4]
    assert torch.allclose(observed.x, x, atol=1e-12, rtol=1e-12)
    assert torch.allclose(observed.x_bar, x_bar, atol=1e-12, rtol=1e-12)
    assert torch.allclose(observed.data_dual.reshape(-1), dual, atol=1e-12, rtol=1e-12)
    ledger = setup.pipeline.call_ledger()
    assert ledger == FactorPipelineCallLedger(
        signed_data_forward_calls=4,
        signed_data_transpose_calls=4,
    )


def test_scalar_to_factor_ablation_is_nested_and_finite() -> None:
    fixture, _, setup = _gate_b_fixture()
    scalar = describe_gate_b_metric(setup, "scalar")
    block = describe_gate_b_metric(setup, "view_block")
    factor = describe_gate_b_metric(setup, "voxel_factor")
    assert torch.all(scalar.tau == torch.min(factor.tau))
    assert torch.equal(block.tau, scalar.tau)
    assert torch.all(scalar.sigma_data_by_view <= block.sigma_data_by_view)
    assert torch.equal(block.sigma_data_by_view, factor.sigma_data_by_view)
    checkpoints = run_gate_b_data_only_trajectory(
        setup,
        fixture.target,
        checkpoints=(1, 2, 4),
        mode="voxel_factor",
    )
    final = checkpoints[4]
    assert torch.all(torch.isfinite(final.x))
    assert torch.count_nonzero(final.tv_dual) == 0
    volume = factor_state_to_volume(setup, final)
    assert volume.shape == (1, 1, *setup.pipeline.grid_shape)
    assert torch.count_nonzero(volume * (1.0 - setup.pipeline.voxel_operator.support)) == 0


def test_setup_mutation_is_detected_before_solver_calls() -> None:
    fixture, _, setup = _gate_b_fixture()
    setup.pipeline.reset_call_ledger()
    setup.tau_voxel_factor.data.mul_(2.0)
    with pytest.raises(RuntimeError, match="derived A-only metric changed"):
        run_gate_b_data_only_trajectory(
            setup,
            fixture.target,
            checkpoints=(1,),
            mode="voxel_factor",
        )
    assert setup.pipeline.call_ledger() == FactorPipelineCallLedger()


@pytest.mark.parametrize(
    ("checkpoints", "message"),
    [
        ((), "positive iterations"),
        ((0, 2), "positive iterations"),
    ],
)
def test_gate_b_trajectory_rejects_invalid_checkpoints(
    checkpoints: tuple[int, ...],
    message: str,
) -> None:
    fixture, _, setup = _gate_b_fixture()
    with pytest.raises(ValueError, match=message):
        run_gate_b_data_only_trajectory(
            setup,
            fixture.target,
            checkpoints=checkpoints,
            mode="view_block",
        )


def test_single_sample_whitening_rejects_cross_batch_use() -> None:
    _, source, _ = _gate_b_fixture()
    frozen = SingleSampleViewLocalWhitening(source, 0)
    values = torch.zeros((2, 2, 2), dtype=torch.float64)
    with pytest.raises(ValueError, match="batch size one"):
        frozen(values)
