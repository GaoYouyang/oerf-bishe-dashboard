"""Tiny CPU/float64 audits for the production factor-majorizer pipeline."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
from torch import nn

from .psu_b0_absolute_measurement_factor import ExactAbsoluteMeasurementFactor
from .psu_b0_active_coordinates import CoordinateSupportGauge
from .psu_b0_factor_majorizer_pipeline import (
    FactorPipelineCallLedger,
    PSUB0FactorMajorizerSetup,
    build_deleted_data_ledger,
    build_psu_b0_factor_majorizer_pipeline,
    factor_pdhg_objective,
    run_factor_pdhg,
)
from .psu_b0_primal_dual import ForwardNeumannRegularizationOperator
from .psu_b0_reconstruction_interface import (
    PSUB0VoxelGradientOperator,
    absolute_finite_difference_gradient,
    build_trilinear_stencil,
    finite_difference_gradient,
)
from .psu_b0_signed_factor_majorizer import (
    SignedFactorSystem,
    build_majorizer_setup,
    pdhg_objective,
    run_pdhg,
)


SHAPE_ZYX = (3, 3, 3)
SPACING_XYZ = (1.0, 1.0, 1.0)


class _Whitening(nn.Module):
    def __init__(self, *, dtype: torch.dtype = torch.float64) -> None:
        super().__init__()
        self.view_count = 2
        self.rays_per_view = 1
        self.register_buffer(
            "matrix",
            torch.tensor(
                [
                    [[1.0, -0.4], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0]],
                ],
                dtype=dtype,
            ),
        )
        self.register_buffer(
            "scale_by_view",
            torch.tensor([[1.25, 0.75]], dtype=dtype),
        )


class _DoubleAbsoluteMeasurementFactor(ExactAbsoluteMeasurementFactor):
    def absolute_forward(self, sampled_gradient: torch.Tensor) -> torch.Tensor:
        super().absolute_forward(sampled_gradient)
        return super().absolute_forward(sampled_gradient)


def _fixture(
    *,
    support: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float64,
    factor_sample_count: int = 2,
    regularization: ForwardNeumannRegularizationOperator | None = None,
    factor_class: type[ExactAbsoluteMeasurementFactor] = (
        ExactAbsoluteMeasurementFactor
    ),
) -> tuple[
    CoordinateSupportGauge,
    PSUB0VoxelGradientOperator,
    ExactAbsoluteMeasurementFactor,
    ForwardNeumannRegularizationOperator,
]:
    if support is None:
        support = torch.ones(SHAPE_ZYX, dtype=torch.float64)
        support[0, 0, 0] = 0.0
        support[2, 0, 1] = 0.0
    gauge = CoordinateSupportGauge.from_support(support)
    points = np.array(
        [
            [[-0.7, -0.5, -0.2], [0.4, 0.3, 0.6]],
            [[-0.4, 0.6, -0.5], [0.7, -0.2, 0.4]],
        ],
        dtype=np.float64,
    )
    stencil = build_trilinear_stencil(
        points,
        grid_shape=SHAPE_ZYX,
        grid_minimum_xyz=(-1.0, -1.0, -1.0),
        grid_maximum_xyz=(1.0, 1.0, 1.0),
        dtype=dtype,
    )
    projection_u = torch.tensor(
        [[1.0, -0.5, 0.25], [-0.4, 0.8, 0.3]],
        dtype=dtype,
    )
    projection_v = torch.tensor(
        [[-0.2, 0.7, 0.4], [0.9, -0.3, -0.6]],
        dtype=dtype,
    )
    voxel = PSUB0VoxelGradientOperator(
        stencil=stencil,
        projection_u_xyz=projection_u,
        projection_v_xyz=projection_v,
        line_length=torch.tensor([2.0, 1.5], dtype=dtype),
        system_constant=torch.tensor([0.8, -1.1], dtype=dtype),
        grid_minimum_xyz=(-1.0, -1.0, -1.0),
        grid_maximum_xyz=(1.0, 1.0, 1.0),
        support=support.to(dtype=dtype),
        dtype=dtype,
    )
    factor = factor_class(
        _Whitening(dtype=dtype),
        projection_u_xyz=projection_u,
        projection_v_xyz=projection_v,
        ray_scale=voxel.ray_scale,
        sample_count=factor_sample_count,
        measurement_scale=0.6,
    )
    tv = regularization or ForwardNeumannRegularizationOperator(SPACING_XYZ)
    return gauge, voxel, factor, tv


def _setup(**fixture_kwargs: object) -> PSUB0FactorMajorizerSetup:
    gauge, voxel, factor, tv = _fixture(**fixture_kwargs)
    return build_psu_b0_factor_majorizer_pipeline(
        gauge=gauge,
        voxel_operator=voxel,
        measurement_factor=factor,
        regularization_operator=tv,
        eta=0.7,
    )


def _basis_matrix(forward: object, columns: int) -> torch.Tensor:
    basis = torch.eye(columns, dtype=torch.float64)
    values = forward(basis)  # type: ignore[operator]
    return values.reshape(columns, -1).T.contiguous()


def _dense_factors(setup: PSUB0FactorMajorizerSetup) -> tuple[torch.Tensor, ...]:
    pipeline = setup.pipeline
    gauge = pipeline.gauge
    voxel = pipeline.voxel_operator
    factor = pipeline.measurement_factor
    full_count = gauge.full_primal_count
    sampled_count = 3 * pipeline.ray_count * pipeline.sample_count

    full_basis = torch.eye(full_count, dtype=torch.float64).reshape(
        full_count, 1, *SHAPE_ZYX
    )
    E = gauge.E
    G = finite_difference_gradient(
        full_basis[:, 0],
        spacing_xyz=SPACING_XYZ,
    ).reshape(full_count, -1).T
    absolute_G = absolute_finite_difference_gradient(
        full_basis[:, 0],
        spacing_xyz=SPACING_XYZ,
    ).reshape(full_count, -1).T

    gradient_basis = torch.eye(3 * full_count, dtype=torch.float64).reshape(
        3 * full_count, 3, *SHAPE_ZYX
    )
    P = voxel.trilinear_interpolation(gradient_basis).reshape(
        3 * full_count, sampled_count
    ).T
    sampled_basis = torch.eye(sampled_count, dtype=torch.float64).reshape(
        sampled_count,
        3,
        pipeline.ray_count,
        pipeline.sample_count,
    )
    W = factor.signed_forward(sampled_basis).reshape(sampled_count, -1).T
    absolute_W = factor.absolute_forward(sampled_basis).reshape(
        sampled_count, -1
    ).T

    D_plus = pipeline.regularization_operator(full_basis[:, 0]).reshape(
        full_count, -1
    ).T
    absolute_D_plus = pipeline.regularization_operator.absolute_forward(
        full_basis[:, 0]
    ).reshape(full_count, -1).T
    return (
        W @ P @ G @ E,
        D_plus @ E,
        absolute_W @ P @ absolute_G @ E,
        absolute_D_plus @ E,
    )


def _scaled_norm_squared(
    setup: PSUB0FactorMajorizerSetup,
    A: torch.Tensor,
    D: torch.Tensor,
    *,
    tau_scale: float = 1.0,
) -> float:
    data_mask = setup.data_row_mask.reshape(-1)
    tv_site_rows = setup.tv_site_mask.unsqueeze(0).expand(3, *SHAPE_ZYX)
    primal = setup.active_primal_mask
    K = torch.cat((A[data_mask][:, primal], D[tv_site_rows.reshape(-1)][:, primal]))
    sigma_data = setup.data_sigma_rows.reshape(-1)[data_mask]
    sigma_tv = setup.sigma_tv_by_site.unsqueeze(0).expand(
        3, *SHAPE_ZYX
    ).reshape(-1)[tv_site_rows.reshape(-1)]
    sigma = torch.cat((sigma_data, sigma_tv))
    scaled = (
        torch.sqrt(sigma)[:, None]
        * K
        * torch.sqrt(setup.tau * tau_scale)[None, :]
    )
    return float(torch.linalg.svdvals(scaled)[0].square())


def _component_major_to_site_major(matrix: torch.Tensor) -> torch.Tensor:
    """Convert [component,z,y,x] rows to [z,y,x,component] rows."""

    return (
        matrix.reshape(3, *SHAPE_ZYX, matrix.shape[1])
        .movedim(0, -2)
        .reshape(-1, matrix.shape[1])
    )


def _dense_site_major_oracle_system(
    setup: PSUB0FactorMajorizerSetup,
) -> SignedFactorSystem:
    pipeline = setup.pipeline
    gauge = pipeline.gauge
    voxel = pipeline.voxel_operator
    factor = pipeline.measurement_factor
    full_count = gauge.full_primal_count
    sampled_count = 3 * pipeline.ray_count * pipeline.sample_count

    full_basis = torch.eye(full_count, dtype=torch.float64).reshape(
        full_count,
        1,
        *SHAPE_ZYX,
    )
    G = finite_difference_gradient(
        full_basis[:, 0],
        spacing_xyz=SPACING_XYZ,
    ).reshape(full_count, -1).T
    gradient_basis = torch.eye(3 * full_count, dtype=torch.float64).reshape(
        3 * full_count,
        3,
        *SHAPE_ZYX,
    )
    P = voxel.trilinear_interpolation(gradient_basis).reshape(
        3 * full_count,
        sampled_count,
    ).T
    sampled_basis = torch.eye(sampled_count, dtype=torch.float64).reshape(
        sampled_count,
        3,
        pipeline.ray_count,
        pipeline.sample_count,
    )
    W = factor.signed_forward(sampled_basis).reshape(sampled_count, -1).T
    D_component_major = pipeline.regularization_operator(
        full_basis[:, 0]
    ).reshape(full_count, -1).T
    D_site_major = _component_major_to_site_major(D_component_major)
    rows_per_view = 2 * pipeline.rays_per_view
    return SignedFactorSystem(
        E=gauge.E.detach().cpu().numpy(),
        P_blocks=tuple(
            P.detach().cpu().numpy() for _ in range(pipeline.view_count)
        ),
        G=G.detach().cpu().numpy(),
        W_blocks=tuple(
            W[
                view * rows_per_view : (view + 1) * rows_per_view
            ].detach().cpu().numpy()
            for view in range(pipeline.view_count)
        ),
        D_plus=D_site_major.detach().cpu().numpy(),
        tv_site_count=full_count,
    )


def test_basis_dense_chains_match_production_maps_and_physical_chain() -> None:
    setup = _setup()
    pipeline = setup.pipeline
    A, D, M, N = _dense_factors(setup)

    torch.testing.assert_close(_basis_matrix(pipeline.signed_data_forward, pipeline.n_active), A)
    torch.testing.assert_close(_basis_matrix(pipeline.signed_tv_forward, pipeline.n_active), D)
    torch.testing.assert_close(_basis_matrix(pipeline.absolute_data_forward, pipeline.n_active), M)
    torch.testing.assert_close(_basis_matrix(pipeline.absolute_tv_forward, pipeline.n_active), N)

    active_basis = torch.eye(pipeline.n_active, dtype=torch.float64)
    physical = pipeline.voxel_operator(pipeline.embed_active(active_basis))
    whitening = _Whitening()
    expected = torch.einsum(
        "vij,bvj->bvi",
        whitening.matrix,
        physical.reshape(pipeline.n_active, 2, 2),
    ) / whitening.scale_by_view[:, :, None]
    expected = 0.6 * expected.reshape(pipeline.n_active, 2, 2)
    torch.testing.assert_close(
        pipeline.signed_data_forward(active_basis),
        expected,
        atol=2e-14,
        rtol=2e-14,
    )


def test_all_four_pairs_are_adjoint_and_majorizers_dominate() -> None:
    setup = _setup()
    pipeline = setup.pipeline
    A, D, M, N = _dense_factors(setup)
    generator = torch.Generator().manual_seed(1707)
    x = torch.randn((2, pipeline.n_active), generator=generator, dtype=torch.float64)
    y = torch.randn((2, pipeline.ray_count, 2), generator=generator, dtype=torch.float64)
    q = torch.randn((2, 3, *SHAPE_ZYX), generator=generator, dtype=torch.float64)

    for forward, transpose, dual in (
        (pipeline.signed_data_forward, pipeline.signed_data_transpose, y),
        (pipeline.absolute_data_forward, pipeline.absolute_data_transpose, y),
        (pipeline.signed_tv_forward, pipeline.signed_tv_transpose, q),
        (pipeline.absolute_tv_forward, pipeline.absolute_tv_transpose, q),
    ):
        torch.testing.assert_close(
            torch.sum(forward(x) * dual),
            torch.sum(x * transpose(dual)),
            atol=2e-13,
            rtol=2e-13,
        )
    assert torch.all(M >= A.abs() - 2e-14)
    assert torch.all(N >= D.abs() - 2e-14)


def test_one_pass_sums_masks_and_shared_steps_are_exact() -> None:
    setup = _setup()
    A, D, M, N = _dense_factors(setup)
    del A, D
    torch.testing.assert_close(setup.data_row_sums.reshape(-1), M.sum(dim=1))
    torch.testing.assert_close(setup.data_column_sums, M.sum(dim=0))
    torch.testing.assert_close(setup.tv_row_sums.reshape(-1), N.sum(dim=1))
    torch.testing.assert_close(setup.tv_column_sums, N.sum(dim=0))
    assert setup.setup_call_ledger == FactorPipelineCallLedger(
        absolute_data_forward_calls=1,
        absolute_data_transpose_calls=1,
        absolute_tv_forward_calls=1,
        absolute_tv_transpose_calls=1,
    )
    assert setup.setup_physical_call_ledger == setup.setup_call_ledger

    assert setup.data_row_mask.shape == (2, 1, 2)
    assert setup.data_row_mask[0, 0].tolist() == [True, False]
    assert not torch.any(setup.data_row_mask[1])
    assert setup.rho_data_by_view[1] == 0.0
    assert setup.sigma_data_by_view[1] == 0.0
    assert setup.data_sigma_rows[0, 0, 0] == setup.sigma_data_by_view[0]
    assert setup.data_sigma_rows[0, 0, 1] == 0.0

    assert not setup.tv_site_mask[-1, -1, -1]
    assert setup.rho_tv_by_site[-1, -1, -1] == 0.0
    assert setup.sigma_tv_by_site[-1, -1, -1] == 0.0
    active_site = (0, 0, 1)
    assert torch.all(
        setup.tv_sigma_rows[:, active_site[0], active_site[1], active_site[2]]
        == setup.sigma_tv_by_site[active_site]
    )
    assert torch.all(torch.isfinite(setup.tau))
    assert torch.all(setup.tau > 0.0)
    assert not torch.any(torch.isinf(setup.sigma_data_by_view))
    assert not torch.any(torch.isinf(setup.sigma_tv_by_site))


def test_production_tv_contract_matches_site_major_dense_oracle() -> None:
    """Cross the component-major PyTorch boundary explicitly before the oracle."""

    setup = _setup()
    pipeline = setup.pipeline
    full_count = pipeline.gauge.full_primal_count
    full_basis = torch.eye(full_count, dtype=torch.float64).reshape(
        full_count,
        *SHAPE_ZYX,
    )
    dense_component_major = pipeline.regularization_operator(full_basis).reshape(
        full_count,
        -1,
    ).T
    dense_site_major = _component_major_to_site_major(dense_component_major)

    identity = np.eye(full_count, dtype=np.float64)
    oracle = build_majorizer_setup(
        SignedFactorSystem(
            E=pipeline.gauge.E.detach().cpu().numpy(),
            P_blocks=(identity,),
            G=identity,
            W_blocks=(np.zeros((1, full_count), dtype=np.float64),),
            D_plus=dense_site_major.detach().cpu().numpy(),
            tv_site_count=full_count,
        ),
        eta=setup.eta,
    )

    active = torch.randn(
        (2, pipeline.n_active),
        generator=torch.Generator().manual_seed(1708),
        dtype=torch.float64,
    )
    production_site_major = (
        pipeline.signed_tv_forward(active)
        .movedim(1, -1)
        .reshape(active.shape[0], -1)
    )
    expected = active @ torch.from_numpy(oracle.D).T
    torch.testing.assert_close(production_site_major, expected)

    production_rows = setup.tv_row_sums.movedim(0, -1).reshape(-1)
    torch.testing.assert_close(
        production_rows,
        torch.from_numpy(oracle.omega_tv_rows),
    )
    torch.testing.assert_close(
        setup.rho_tv_by_site.reshape(-1),
        torch.from_numpy(oracle.rho_tv_sites),
    )


def test_exact_dense_svd_is_bounded_and_enlarged_steps_fail() -> None:
    setup = _setup()
    A, D, _, _ = _dense_factors(setup)
    observed = _scaled_norm_squared(setup, A, D)
    assert observed <= setup.eta**2 + 1e-12
    enlarged = _scaled_norm_squared(setup, A, D, tau_scale=16.0)
    assert enlarged > setup.eta**2 + 1e-12


def test_six_step_production_recurrence_matches_site_major_dense_oracle() -> None:
    setup = _setup()
    oracle = build_majorizer_setup(
        _dense_site_major_oracle_system(setup),
        eta=setup.eta,
    )
    np.testing.assert_array_equal(
        setup.active_primal_indices.detach().cpu().numpy(),
        oracle.active_primal,
    )
    torch.testing.assert_close(setup.tau, torch.from_numpy(oracle.tau))
    torch.testing.assert_close(
        setup.rho_data_by_view,
        torch.as_tensor(oracle.rho_data, dtype=torch.float64),
    )
    torch.testing.assert_close(
        setup.rho_tv_by_site.reshape(-1),
        torch.from_numpy(oracle.rho_tv_sites),
    )

    target = torch.tensor(
        [[[0.25, -0.40]], [[0.70, -0.15]]],
        dtype=torch.float64,
    )
    oracle_targets = tuple(
        target[view].reshape(-1).numpy()[mask]
        for view, mask in enumerate(oracle.active_data_rows)
    )
    physical_before = setup.pipeline.physical_call_ledger()
    production_states = run_factor_pdhg(
        setup,
        target,
        iterations=6,
        regularization_weight=0.15,
        penalty="huber",
        huber_delta=0.3,
    )
    physical_after = setup.pipeline.physical_call_ledger()
    physical_delta = FactorPipelineCallLedger(
        **{
            name: getattr(physical_after, name) - getattr(physical_before, name)
            for name in FactorPipelineCallLedger.__dataclass_fields__
        }
    )
    assert physical_delta == FactorPipelineCallLedger(
        signed_data_forward_calls=6,
        signed_data_transpose_calls=6,
        signed_tv_forward_calls=6,
        signed_tv_transpose_calls=6,
    )
    oracle_states = run_pdhg(
        oracle,
        oracle_targets,
        iterations=6,
        regularization_weight=0.15,
        penalty="huber",
        huber_delta=0.3,
    )
    for production, expected in zip(production_states, oracle_states):
        torch.testing.assert_close(production.x, torch.from_numpy(expected.x))
        torch.testing.assert_close(
            production.x_bar,
            torch.from_numpy(expected.x_bar),
        )
        for view, mask in enumerate(oracle.active_data_rows):
            torch.testing.assert_close(
                production.data_dual[view].reshape(-1)[torch.from_numpy(mask)],
                torch.from_numpy(expected.data_dual[view]),
            )
        production_tv = production.tv_dual.movedim(0, -1).reshape(-1, 3)
        torch.testing.assert_close(
            production_tv[torch.from_numpy(oracle.active_tv_sites)],
            torch.from_numpy(expected.tv_dual),
        )

    assert setup.pipeline.call_ledger() == FactorPipelineCallLedger(
        signed_data_forward_calls=6,
        signed_data_transpose_calls=6,
        absolute_data_forward_calls=1,
        absolute_data_transpose_calls=1,
        signed_tv_forward_calls=6,
        signed_tv_transpose_calls=6,
        absolute_tv_forward_calls=1,
        absolute_tv_transpose_calls=1,
    )
    production_objective = factor_pdhg_objective(
        setup,
        production_states[-1],
        target,
        regularization_weight=0.15,
        penalty="huber",
        huber_delta=0.3,
    )
    deleted_ledger = build_deleted_data_ledger(setup, target)
    deleted_constant = sum(
        0.5 * float(np.square(target[view].reshape(-1).numpy()[~mask]).sum())
        for view, mask in enumerate(oracle.active_data_rows)
    )
    assert float(deleted_ledger.objective_constant) == pytest.approx(
        deleted_constant,
        abs=0.0,
        rel=0.0,
    )
    np.testing.assert_array_equal(
        deleted_ledger.deleted_flat_indices.numpy(),
        np.flatnonzero(~setup.data_row_mask.reshape(-1).numpy()),
    )
    expected_objective = pdhg_objective(
        oracle,
        oracle_states[-1],
        oracle_targets,
        regularization_weight=0.15,
        penalty="huber",
        huber_delta=0.3,
    ) + deleted_constant
    assert float(production_objective) == pytest.approx(
        expected_objective,
        abs=2e-13,
        rel=2e-13,
    )


class _DoubleAbsoluteRegularization(ForwardNeumannRegularizationOperator):
    def absolute_forward(self, volume: torch.Tensor) -> torch.Tensor:
        super().absolute_forward(volume)
        return super().absolute_forward(volume)


def test_contract_validation_and_setup_signature_fail_closed() -> None:
    gauge, voxel, factor, tv = _fixture()
    bad_support = gauge.support.clone()
    bad_support[0, 0, 0] = 1.0
    with pytest.raises(ValueError, match="support.*gauge"):
        build_psu_b0_factor_majorizer_pipeline(
            gauge=CoordinateSupportGauge.from_support(bad_support),
            voxel_operator=voxel,
            measurement_factor=factor,
            regularization_operator=tv,
        )
    _, _, bad_samples, _ = _fixture(factor_sample_count=3)
    with pytest.raises(ValueError, match="sample counts"):
        build_psu_b0_factor_majorizer_pipeline(
            gauge=gauge,
            voxel_operator=voxel,
            measurement_factor=bad_samples,
            regularization_operator=tv,
        )
    with pytest.raises(ValueError, match=r"\(0,1\)"):
        build_psu_b0_factor_majorizer_pipeline(
            gauge=gauge,
            voxel_operator=voxel,
            measurement_factor=factor,
            regularization_operator=tv,
            eta=1.0,
        )

    parameters = inspect.signature(
        build_psu_b0_factor_majorizer_pipeline
    ).parameters
    assert not ({"truth", "morphology", "target", "targets"} & set(parameters))


def test_setup_rejects_multiple_measurement_scale_instances() -> None:
    gauge, voxel, factor, tv = _fixture()
    factor.scale_by_view = factor.scale_by_view.expand(2, -1).clone()
    with pytest.raises(ValueError, match="exactly one frozen"):
        build_psu_b0_factor_majorizer_pipeline(
            gauge=gauge,
            voxel_operator=voxel,
            measurement_factor=factor,
            regularization_operator=tv,
            batch_index=1,
        )


@pytest.mark.parametrize(
    "fixture_kwargs",
    [
        {"factor_class": _DoubleAbsoluteMeasurementFactor},
        {
            "regularization": _DoubleAbsoluteRegularization(
                SPACING_XYZ
            )
        },
    ],
)
def test_setup_rejects_subclasses_that_could_spoof_physical_ledgers(
    fixture_kwargs: dict[str, object],
) -> None:
    gauge, voxel, factor, tv = _fixture(**fixture_kwargs)
    with pytest.raises(TypeError, match="sealed"):
        build_psu_b0_factor_majorizer_pipeline(
            gauge=gauge,
            voxel_operator=voxel,
            measurement_factor=factor,
            regularization_operator=tv,
        )


@pytest.mark.parametrize("storage_bypass", [False, True])
def test_solver_rejects_factor_mutation_after_setup(
    storage_bypass: bool,
) -> None:
    setup = _setup()
    target = torch.zeros((2, 1, 2), dtype=torch.float64)
    state = run_factor_pdhg(setup, target, iterations=1)[0]
    kernel = setup.pipeline.measurement_factor.signed_kernel
    if storage_bypass:
        kernel.data[0, 0, 0, 0] += 0.25
    else:
        kernel[0, 0, 0, 0].add_(0.25)
    with pytest.raises(RuntimeError, match="changed after majorizer setup"):
        factor_pdhg_objective(setup, state, target)


@pytest.mark.parametrize(
    "mutation",
    [
        "tau",
        "sigma_data",
        "data_mask",
        "active_primal_indices",
        "pipeline_active_indices",
        "voxel_spacing",
    ],
)
def test_solver_rejects_derived_setup_or_geometry_mutation(
    mutation: str,
) -> None:
    setup = _setup()
    target = torch.zeros((2, 1, 2), dtype=torch.float64)
    if mutation == "tau":
        setup.tau.data[0] *= 1.1
    elif mutation == "sigma_data":
        setup.sigma_data_by_view.data[0] *= 1.1
    elif mutation == "data_mask":
        setup.data_row_mask.data.reshape(-1)[0] = False
    elif mutation == "active_primal_indices":
        setup.active_primal_indices.data[0] += 1
    elif mutation == "pipeline_active_indices":
        setup.pipeline._active_indices.data[0] += 1
    elif mutation == "voxel_spacing":
        setup.pipeline.voxel_operator.spacing_xyz = (1.1, 1.0, 1.0)
    else:  # pragma: no cover - parametrization is exhaustive.
        raise AssertionError(mutation)
    with pytest.raises(
        RuntimeError,
        match="changed after (majorizer )?setup",
    ):
        run_factor_pdhg(setup, target, iterations=1)


def test_embed_restrict_are_device_dtype_strict_and_do_not_use_dense_E() -> None:
    setup = _setup()
    pipeline = setup.pipeline
    original_E = pipeline.gauge.E
    object.__setattr__(pipeline.gauge, "E", torch.full_like(original_E, float("nan")))
    active = torch.arange(pipeline.n_active, dtype=torch.float64)[None]
    full = pipeline.embed_active(active)
    torch.testing.assert_close(pipeline.restrict_active(full), active)
    with pytest.raises(ValueError, match="dtype"):
        pipeline.embed_active(active.to(torch.float32))
    object.__setattr__(pipeline.gauge, "E", original_E)
