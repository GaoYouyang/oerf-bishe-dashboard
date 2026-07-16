from __future__ import annotations

import numpy as np

from demo_t16_operator.detector_graph_covariance import (
    detector_graph_spectral_basis,
    empirical_covariance_rank_upper_bound,
    evaluate_covariance_fit,
    fit_amplitude_modulated_graph_covariance,
    fit_component_iid_covariance,
    fit_gated_graph_covariance,
    fit_graph_separable_covariance,
    fit_low_rank_drift_covariance,
    simulate_flowoff_repeats,
)
from demo_t16_operator.psu_b0_detector_graph_features import (
    build_detector_knn_graph,
)


def _toy_graph() -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    axis = np.linspace(-1.0, 1.0, 8)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    coordinates = np.stack((xx.ravel(), yy.ravel()), axis=1)
    graph = build_detector_knn_graph(
        coordinates,
        view_count=1,
        rays_per_view=len(coordinates),
        neighbor_count=6,
    )
    eigenvalues, eigenvectors = detector_graph_spectral_basis(
        graph,
        view_index=0,
    )
    return graph, coordinates, eigenvalues, eigenvectors


def _grid() -> dict[str, tuple[float, ...]]:
    return {
        "diffusion_times": (0.5, 1.2, 2.0),
        "spatial_fractions": (0.0, 0.5, 0.75, 0.9),
        "correlations": (0.0, 0.25, 0.35, 0.5),
        "variance_ratios": (0.85, 1.0, 1.25, 1.35, 1.55),
    }


def test_detector_graph_laplacian_has_valid_spectrum() -> None:
    _, _, eigenvalues, eigenvectors = _toy_graph()
    assert eigenvalues.shape == (64,)
    assert eigenvectors.shape == (64, 64)
    assert eigenvalues[0] < 1e-10
    assert np.all(eigenvalues >= 0.0)
    assert np.allclose(eigenvectors.T @ eigenvectors, np.eye(64), atol=1e-10)


def test_graph_model_improves_heldout_likelihood_in_class() -> None:
    _, coordinates, eigenvalues, eigenvectors = _toy_graph()
    rng = np.random.default_rng(701)
    calibration, _ = simulate_flowoff_repeats(
        family="graph_heat",
        repeat_count=64,
        rng=rng,
        laplacian_eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        detector_xy=coordinates,
    )
    heldout, _ = simulate_flowoff_repeats(
        family="graph_heat",
        repeat_count=256,
        rng=rng,
        laplacian_eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        detector_xy=coordinates,
    )
    grid = _grid()
    iid = fit_component_iid_covariance(
        calibration,
        laplacian_eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        correlations=grid["correlations"],
        variance_ratios=grid["variance_ratios"],
    )
    graph = fit_graph_separable_covariance(
        calibration,
        laplacian_eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        diffusion_times=grid["diffusion_times"],
        spatial_fractions=grid["spatial_fractions"],
        correlations=grid["correlations"],
        variance_ratios=grid["variance_ratios"],
    )
    iid_score = evaluate_covariance_fit(
        iid,
        heldout,
        eigenvectors=eigenvectors,
    )
    graph_score = evaluate_covariance_fit(
        graph,
        heldout,
        eigenvectors=eigenvectors,
    )
    assert (
        graph_score["mean_nll_per_dimension"]
        < iid_score["mean_nll_per_dimension"] - 0.01
    )
    assert graph.parameters["spatial_fraction"] >= 0.5


def test_gated_fit_returns_auditable_validation_decision() -> None:
    _, coordinates, eigenvalues, eigenvectors = _toy_graph()
    rng = np.random.default_rng(702)
    calibration, _ = simulate_flowoff_repeats(
        family="component_iid",
        repeat_count=24,
        rng=rng,
        laplacian_eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        detector_xy=coordinates,
    )
    fit, gate = fit_gated_graph_covariance(
        calibration,
        laplacian_eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        graph_grid=_grid(),
        validation_fraction=0.25,
        minimum_validation_gain_per_dimension=0.0025,
    )
    assert gate["fit_repeat_count"] == 18
    assert gate["validation_repeat_count"] == 6
    assert isinstance(gate["graph_activated"], bool)
    assert fit.kind in {"component_iid", "gated_graph"}


def test_empirical_covariance_rank_ceiling_exposes_small_repeat_limit() -> None:
    assert empirical_covariance_rank_upper_bound(
        repeat_count=20,
        dimension=512,
    ) == 19
    assert empirical_covariance_rank_upper_bound(
        repeat_count=600,
        dimension=512,
    ) == 512


def test_rank_one_drift_improves_nonstationary_heldout_likelihood() -> None:
    _, coordinates, eigenvalues, eigenvectors = _toy_graph()
    gains = []
    for seed in (910, 911, 912):
        rng = np.random.default_rng(seed)
        calibration, _ = simulate_flowoff_repeats(
            family="nonstationary_drift",
            repeat_count=50,
            rng=rng,
            laplacian_eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            detector_xy=coordinates,
        )
        heldout, _ = simulate_flowoff_repeats(
            family="nonstationary_drift",
            repeat_count=400,
            rng=rng,
            laplacian_eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            detector_xy=coordinates,
        )
        amplitude = fit_amplitude_modulated_graph_covariance(
            calibration,
            laplacian_eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            graph_grid=_grid(),
            smoothing_strengths=(0.5, 2.0, 8.0, 32.0),
        )
        low_rank = fit_low_rank_drift_covariance(
            calibration,
            laplacian_eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            graph_grid=_grid(),
            amplitude_smoothing_strengths=(0.5, 2.0, 8.0, 32.0),
            rank_options=(1,),
            shrinkage_strengths=(0.25, 0.5, 0.75, 1.0),
            base_amplitude_fit=amplitude,
        )
        amplitude_score = evaluate_covariance_fit(
            amplitude,
            heldout,
            eigenvectors=eigenvectors,
        )
        low_rank_score = evaluate_covariance_fit(
            low_rank,
            heldout,
            eigenvectors=eigenvectors,
        )
        gains.append(
            amplitude_score["mean_nll_per_dimension"]
            - low_rank_score["mean_nll_per_dimension"]
        )
    assert min(gains) > 0.005
