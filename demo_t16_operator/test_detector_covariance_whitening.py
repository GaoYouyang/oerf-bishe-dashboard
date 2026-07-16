from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from .detector_covariance_whitening import (
    DetectorCovarianceWhitening,
    WhitenedMeasurementOperator,
    spatially_tempered_covariance_fit,
)
from .detector_graph_covariance import (
    CovarianceFit,
    apply_covariance_whitening,
    detector_graph_spectral_basis,
    fit_graph_separable_covariance,
    simulate_flowoff_repeats,
)
from .psu_b0_classical_baselines import (
    preconditioned_cgls_reconstruction,
)
from .psu_b0_detector_graph_features import build_detector_knn_graph
from .psu_b0_reconstruction_interface import (
    PSUB0VoxelGradientOperator,
    build_trilinear_stencil,
)
from .psu_b0_spectral_preconditioner import IdentityDirection


def _graph_fit() -> tuple[CovarianceFit, np.ndarray, np.ndarray]:
    axis = np.linspace(-1.0, 1.0, 4)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    coordinates = np.stack((xx.ravel(), yy.ravel()), axis=1)
    graph = build_detector_knn_graph(
        coordinates,
        view_count=1,
        rays_per_view=len(coordinates),
        neighbor_count=5,
    )
    eigenvalues, eigenvectors = detector_graph_spectral_basis(
        graph,
        view_index=0,
    )
    repeats, _ = simulate_flowoff_repeats(
        family="graph_heat",
        repeat_count=64,
        rng=np.random.default_rng(8101),
        laplacian_eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        detector_xy=coordinates,
    )
    fit = fit_graph_separable_covariance(
        repeats,
        laplacian_eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        diffusion_times=(0.5, 1.2, 2.0),
        spatial_fractions=(0.0, 0.5, 0.75, 0.9),
        correlations=(0.0, 0.2, 0.35, 0.5),
        variance_ratios=(0.85, 1.0, 1.25, 1.35, 1.55),
    )
    return fit, eigenvalues, eigenvectors


def _diagonal_fit(rays: int, view: int) -> CovarianceFit:
    variance = np.linspace(
        0.7 + 0.1 * view,
        1.4 + 0.1 * view,
        2 * rays,
    ).reshape(rays, 2)
    return CovarianceFit(
        kind="diagonal_shrinkage",
        mean=np.zeros((rays, 2), dtype=np.float64),
        sigma2=None,
        spatial_eigenvalues=None,
        component_covariance=None,
        node_amplitude=None,
        low_rank_vectors=None,
        low_rank_eigenvalues=None,
        diagonal_variance=variance,
        parameters={"calibration_repeat_count": 50.0},
        training_nll_per_dimension=0.0,
    )


def _operator() -> PSUB0VoxelGradientOperator:
    rng = np.random.default_rng(8102)
    rays = 21
    points = rng.uniform(-0.85, 0.85, size=(rays, 5, 3))
    stencil = build_trilinear_stencil(
        points,
        grid_shape=(8, 8, 8),
        grid_minimum_xyz=(-1.0, -1.0, -1.0),
        grid_maximum_xyz=(1.0, 1.0, 1.0),
        dtype=torch.float64,
    )
    return PSUB0VoxelGradientOperator(
        stencil=stencil,
        projection_u_xyz=rng.normal(size=(rays, 3)),
        projection_v_xyz=rng.normal(size=(rays, 3)),
        line_length=np.ones(rays),
        system_constant=np.ones(rays),
        grid_minimum_xyz=(-1.0, -1.0, -1.0),
        grid_maximum_xyz=(1.0, 1.0, 1.0),
        dtype=torch.float64,
    )


def test_dense_whitening_matches_factorized_graph_transform() -> None:
    fit, _, eigenvectors = _graph_fit()
    correction = 1.0 + 1.0 / 64.0
    rng = np.random.default_rng(8103)
    values = rng.normal(size=(5, 16, 2))
    expected = apply_covariance_whitening(
        fit,
        values,
        eigenvectors=eigenvectors,
        covariance_scale=correction,
    )
    whitening = DetectorCovarianceWhitening(
        [fit],
        eigenvectors_by_view=[eigenvectors],
        scale_by_view=[1.0],
        predictive_mean_correction=True,
        dtype=torch.float64,
    )
    actual = whitening(torch.as_tensor(values, dtype=torch.float64))
    assert np.allclose(actual.numpy(), expected, atol=1e-11, rtol=1e-11)


def test_whitening_and_transpose_satisfy_detector_adjoint_identity() -> None:
    fits = [_diagonal_fit(7, view) for view in range(3)]
    whitening = DetectorCovarianceWhitening(
        fits,
        eigenvectors_by_view=[np.eye(7) for _ in fits],
        scale_by_view=[[0.8, 1.1, 1.4], [1.2, 0.9, 0.7]],
        dtype=torch.float64,
    )
    generator = torch.Generator().manual_seed(8104)
    left = torch.randn((2, 21, 2), generator=generator, dtype=torch.float64)
    right = torch.randn((2, 21, 2), generator=generator, dtype=torch.float64)
    lhs = torch.sum(whitening(left) * right)
    rhs = torch.sum(left * whitening.transpose(right))
    error = torch.abs(lhs - rhs) / torch.maximum(
        torch.abs(lhs),
        torch.abs(rhs),
    ).clamp_min(1e-18)
    assert float(error) < 1e-12


def test_wrapped_bost_operator_is_adjoint_and_call_accounted() -> None:
    base = _operator()
    fits = [_diagonal_fit(7, view) for view in range(3)]
    whitening = DetectorCovarianceWhitening(
        fits,
        eigenvectors_by_view=[np.eye(7) for _ in fits],
        scale_by_view=[[0.8, 1.1, 1.4], [1.2, 0.9, 0.7]],
        dtype=torch.float64,
    )
    wrapped = WhitenedMeasurementOperator(base, whitening)
    assert wrapped.spacing_xyz == tuple(
        float(value) for value in base.spacing_xyz
    )
    generator = torch.Generator().manual_seed(8105)
    volume = torch.randn(
        (2, 1, *base.grid_shape),
        generator=generator,
        dtype=torch.float64,
    )
    residual = torch.randn(
        (2, base.ray_count, 2),
        generator=generator,
        dtype=torch.float64,
    )
    lhs = torch.sum(wrapped(volume) * residual)
    rhs = torch.sum(volume * wrapped.adjoint(residual))
    error = torch.abs(lhs - rhs) / torch.maximum(
        torch.abs(lhs),
        torch.abs(rhs),
    ).clamp_min(1e-18)
    assert float(error) < 1e-11

    observation = wrapped.prepare_observation(base(volume).detach())
    wrapped.reset_call_counts()
    result = preconditioned_cgls_reconstruction(
        wrapped,
        observation,
        sigma_by_view=torch.ones((2, 3), dtype=torch.float64),
        view_mask=torch.ones((2, 3), dtype=torch.float64),
        rays_per_view=7,
        stages=3,
        preconditioner=IdentityDirection(),
    )
    assert result.forward_calls == 3
    assert result.adjoint_calls == 3
    assert wrapped.call_report() == {"forward_calls": 3, "adjoint_calls": 3}


def test_spatial_tempering_has_iid_and_graph_spectrum_endpoints() -> None:
    component, _, _ = _graph_fit()
    graph = replace(
        component,
        kind="gated_graph",
        spatial_eigenvalues=np.linspace(
            0.4,
            1.6,
            len(component.spatial_eigenvalues),
        ),
        parameters={
            **component.parameters,
            "spatial_fraction": 0.75,
            "diffusion_time": 1.2,
        },
    )
    graph = replace(
        graph,
        spatial_eigenvalues=(
            graph.spatial_eigenvalues
            / np.mean(graph.spatial_eigenvalues)
        ),
    )

    iid = spatially_tempered_covariance_fit(
        component,
        graph,
        spatial_exponent=0.0,
    )
    full_spatial = spatially_tempered_covariance_fit(
        component,
        graph,
        spatial_exponent=1.0,
    )
    assert np.allclose(
        iid.spatial_eigenvalues,
        np.ones_like(component.spatial_eigenvalues),
    )
    assert np.allclose(
        full_spatial.spatial_eigenvalues,
        graph.spatial_eigenvalues,
    )
    assert np.allclose(
        full_spatial.component_covariance,
        component.component_covariance,
    )
    assert full_spatial.sigma2 == component.sigma2
    assert full_spatial.parameters["component_parameters_held_fixed"] == 1.0


def test_spatial_tempering_zero_matches_component_whitening() -> None:
    fitted, _, eigenvectors = _graph_fit()
    component = replace(
        fitted,
        kind="component_iid",
        spatial_eigenvalues=np.ones_like(fitted.spatial_eigenvalues),
        parameters={
            **fitted.parameters,
            "spatial_fraction": 0.0,
            "diffusion_time": 0.0,
        },
    )
    graph = replace(
        component,
        kind="gated_graph",
        spatial_eigenvalues=np.linspace(
            0.5,
            1.5,
            len(component.spatial_eigenvalues),
        ),
    )
    tempered = spatially_tempered_covariance_fit(
        component,
        graph,
        spatial_exponent=0.0,
    )
    baseline = DetectorCovarianceWhitening(
        [component],
        eigenvectors_by_view=[eigenvectors],
        scale_by_view=[1.0],
        predictive_mean_correction=True,
        dtype=torch.float64,
    )
    endpoint = DetectorCovarianceWhitening(
        [tempered],
        eigenvectors_by_view=[eigenvectors],
        scale_by_view=[1.0],
        predictive_mean_correction=True,
        dtype=torch.float64,
    )
    assert torch.equal(baseline.matrix, endpoint.matrix)


@pytest.mark.parametrize("value", [-0.01, 1.01, np.nan, np.inf])
def test_spatial_tempering_rejects_invalid_exponent(value: float) -> None:
    component, _, _ = _graph_fit()
    with pytest.raises(ValueError, match="spatial_exponent"):
        spatially_tempered_covariance_fit(
            component,
            component,
            spatial_exponent=value,
        )
