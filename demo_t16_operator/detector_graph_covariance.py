"""Low-parameter covariance calibration on an irregular BOS detector graph.

The routines in this module are intentionally small and auditable.  They do
not estimate a dense covariance matrix.  Instead, they fit a separable model

    Sigma = sigma^2 K_graph(tau, alpha) kron B_uv(rho, ratio)

where ``K_graph`` is a heat kernel on the measured detector-neighborhood
graph.  The model is useful as an acquisition-planning hypothesis, but it must
still be checked on held-out flow-off repeats before it can be used for BOS
whitening or reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy.stats import chi2


@dataclass(frozen=True)
class CovarianceFit:
    """A fitted covariance model and its calibration mean."""

    kind: str
    mean: np.ndarray
    sigma2: float | None
    spatial_eigenvalues: np.ndarray | None
    component_covariance: np.ndarray | None
    node_amplitude: np.ndarray | None
    low_rank_vectors: np.ndarray | None
    low_rank_eigenvalues: np.ndarray | None
    diagonal_variance: np.ndarray | None
    parameters: dict[str, float]
    training_nll_per_dimension: float


def detector_graph_spectral_basis(
    graph: dict[str, Any],
    *,
    view_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return eigenpairs of a symmetric normalized detector Laplacian."""

    neighbor_index = np.asarray(
        graph["neighbor_index"].detach().cpu().numpy(),
        dtype=np.int64,
    )
    neighbor_weight = np.asarray(
        graph["neighbor_weight"].detach().cpu().numpy(),
        dtype=np.float64,
    )
    view = int(view_index)
    if neighbor_index.ndim != 3 or neighbor_weight.shape != neighbor_index.shape:
        raise ValueError("detector graph tensors must have shape [view,node,k]")
    if not 0 <= view < neighbor_index.shape[0]:
        raise ValueError("view_index is outside the detector graph")

    node_count = int(neighbor_index.shape[1])
    adjacency = np.zeros((node_count, node_count), dtype=np.float64)
    rows = np.arange(node_count, dtype=np.int64)[:, None]
    np.add.at(
        adjacency,
        (np.broadcast_to(rows, neighbor_index[view].shape), neighbor_index[view]),
        neighbor_weight[view],
    )
    adjacency = 0.5 * (adjacency + adjacency.T)
    degree = np.sum(adjacency, axis=1)
    if np.any(degree <= 0.0):
        raise ValueError("detector graph contains an isolated node")
    inverse_sqrt = 1.0 / np.sqrt(degree)
    normalized = inverse_sqrt[:, None] * adjacency * inverse_sqrt[None, :]
    laplacian = np.eye(node_count, dtype=np.float64) - normalized
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    return eigenvalues, eigenvectors


def graph_heat_eigenvalues(
    laplacian_eigenvalues: np.ndarray,
    *,
    diffusion_time: float,
    spatial_fraction: float,
) -> np.ndarray:
    """Return unit-mean eigenvalues for an identity/heat-kernel mixture."""

    spectrum = np.asarray(laplacian_eigenvalues, dtype=np.float64)
    tau = float(diffusion_time)
    fraction = float(spatial_fraction)
    if spectrum.ndim != 1 or np.any(~np.isfinite(spectrum)):
        raise ValueError("laplacian_eigenvalues must be a finite vector")
    if tau < 0.0 or not 0.0 <= fraction < 1.0:
        raise ValueError("invalid graph heat-kernel parameters")
    heat = np.exp(-tau * spectrum)
    heat = heat / np.mean(heat)
    values = (1.0 - fraction) + fraction * heat
    if np.any(values <= 0.0):
        raise ValueError("graph covariance spectrum must be positive")
    return values


def component_covariance(
    *,
    correlation: float,
    variance_ratio: float,
) -> np.ndarray:
    """Build the two-component covariance before the global scale."""

    rho = float(correlation)
    ratio = float(variance_ratio)
    if not -0.98 < rho < 0.98 or ratio <= 0.0:
        raise ValueError("invalid component covariance parameters")
    cross = rho * np.sqrt(ratio)
    covariance = np.asarray(
        [[1.0, cross], [cross, ratio]],
        dtype=np.float64,
    )
    if np.linalg.det(covariance) <= 0.0:
        raise ValueError("component covariance is not positive definite")
    return covariance


def _component_candidates(
    correlations: Iterable[float],
    variance_ratios: Iterable[float],
) -> list[tuple[np.ndarray, np.ndarray, float, float, float]]:
    output = []
    for correlation in correlations:
        for variance_ratio in variance_ratios:
            covariance = component_covariance(
                correlation=float(correlation),
                variance_ratio=float(variance_ratio),
            )
            output.append(
                (
                    covariance,
                    np.linalg.inv(covariance),
                    float(np.linalg.slogdet(covariance)[1]),
                    float(correlation),
                    float(variance_ratio),
                )
            )
    if not output:
        raise ValueError("component parameter grid cannot be empty")
    return output


def _graph_training_objective(
    centered: np.ndarray,
    *,
    eigenvectors: np.ndarray,
    spatial_eigenvalues: np.ndarray,
    component_inverse: np.ndarray,
    component_logdet: float,
) -> tuple[float, float]:
    repeat_count, node_count, component_count = centered.shape
    if component_count != 2:
        raise ValueError("BOS displacement repeats require two components")
    spectral = np.einsum(
        "ij,rjc->ric",
        eigenvectors.T,
        centered,
        optimize=True,
    )
    scaled = spectral / np.sqrt(spatial_eigenvalues)[None, :, None]
    scatter = np.einsum("ric,rid->cd", scaled, scaled, optimize=True)
    quadratic = float(
        np.einsum("cd,dc->", component_inverse, scatter, optimize=True)
    )
    dimension = node_count * component_count
    sigma2 = max(quadratic / float(repeat_count * dimension), 1e-12)
    spatial_logdet = float(np.sum(np.log(spatial_eigenvalues)))
    logdet = (
        dimension * np.log(sigma2)
        + component_count * spatial_logdet
        + node_count * component_logdet
    )
    objective = 0.5 * (
        repeat_count * (dimension * np.log(2.0 * np.pi) + logdet)
        + quadratic / sigma2
    )
    return float(objective / (repeat_count * dimension)), sigma2


def fit_graph_separable_covariance(
    repeats: np.ndarray,
    *,
    laplacian_eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    diffusion_times: Iterable[float],
    spatial_fractions: Iterable[float],
    correlations: Iterable[float],
    variance_ratios: Iterable[float],
    kind: str = "graph_separable",
) -> CovarianceFit:
    """Fit a low-parameter graph/component covariance by Gaussian likelihood."""

    values = np.asarray(repeats, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 2 or values.shape[0] < 2:
        raise ValueError("repeats must have shape [repeat,node,2] with repeat >= 2")
    if np.any(~np.isfinite(values)):
        raise ValueError("repeats must be finite")
    node_count = int(values.shape[1])
    spectrum = np.asarray(laplacian_eigenvalues, dtype=np.float64)
    basis = np.asarray(eigenvectors, dtype=np.float64)
    if spectrum.shape != (node_count,) or basis.shape != (node_count, node_count):
        raise ValueError("graph eigenpairs do not match repeat dimensions")

    mean = np.mean(values, axis=0)
    centered = values - mean[None]
    components = _component_candidates(correlations, variance_ratios)
    spectral = np.einsum(
        "ij,rjc->ric",
        basis.T,
        centered,
        optimize=True,
    )
    component_covariances = np.stack(
        [item[0] for item in components],
        axis=0,
    )
    component_inverses = np.stack(
        [item[1] for item in components],
        axis=0,
    )
    component_logdets = np.asarray(
        [item[2] for item in components],
        dtype=np.float64,
    )
    component_correlations = np.asarray(
        [item[3] for item in components],
        dtype=np.float64,
    )
    component_ratios = np.asarray(
        [item[4] for item in components],
        dtype=np.float64,
    )
    repeat_count = int(values.shape[0])
    dimension = 2 * node_count
    best: tuple[
        float,
        float,
        np.ndarray,
        np.ndarray,
        float,
        float,
        float,
        float,
    ] | None = None
    for diffusion_time in diffusion_times:
        for spatial_fraction in spatial_fractions:
            spatial = graph_heat_eigenvalues(
                spectrum,
                diffusion_time=float(diffusion_time),
                spatial_fraction=float(spatial_fraction),
            )
            scaled = spectral / np.sqrt(spatial)[None, :, None]
            scatter = np.einsum(
                "ric,rid->cd",
                scaled,
                scaled,
                optimize=True,
            )
            quadratic = np.einsum(
                "mcd,dc->m",
                component_inverses,
                scatter,
                optimize=True,
            )
            sigma2 = np.maximum(
                quadratic / float(repeat_count * dimension),
                1e-12,
            )
            spatial_logdet = float(np.sum(np.log(spatial)))
            logdet = (
                dimension * np.log(sigma2)
                + 2.0 * spatial_logdet
                + node_count * component_logdets
            )
            objectives = 0.5 * (
                repeat_count
                * (dimension * np.log(2.0 * np.pi) + logdet)
                + quadratic / sigma2
            ) / float(repeat_count * dimension)
            component_index = int(np.argmin(objectives))
            candidate = (
                float(objectives[component_index]),
                float(sigma2[component_index]),
                spatial,
                component_covariances[component_index],
                float(diffusion_time),
                float(spatial_fraction),
                float(component_correlations[component_index]),
                float(component_ratios[component_index]),
            )
            if best is None or candidate[0] < best[0]:
                best = candidate
    if best is None:
        raise ValueError("graph covariance parameter grid cannot be empty")
    objective, sigma2_mle, spatial, covariance, tau, fraction, rho, ratio = best
    repeat_count = int(values.shape[0])
    sigma2 = sigma2_mle * repeat_count / (repeat_count - 1)
    return CovarianceFit(
        kind=str(kind),
        mean=mean,
        sigma2=float(sigma2),
        spatial_eigenvalues=np.asarray(spatial, dtype=np.float64),
        component_covariance=np.asarray(covariance, dtype=np.float64),
        node_amplitude=None,
        low_rank_vectors=None,
        low_rank_eigenvalues=None,
        diagonal_variance=None,
        parameters={
            "diffusion_time": float(tau),
            "spatial_fraction": float(fraction),
            "component_correlation": float(rho),
            "component_variance_ratio": float(ratio),
            "calibration_repeat_count": float(repeat_count),
            "scale_estimator": "UNBIASED_AFTER_MLE_SHAPE_SELECTION",
        },
        training_nll_per_dimension=float(objective),
    )


def fit_component_iid_covariance(
    repeats: np.ndarray,
    *,
    laplacian_eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    correlations: Iterable[float],
    variance_ratios: Iterable[float],
) -> CovarianceFit:
    """Fit a spatially IID model while retaining u-v covariance."""

    return fit_graph_separable_covariance(
        repeats,
        laplacian_eigenvalues=laplacian_eigenvalues,
        eigenvectors=eigenvectors,
        diffusion_times=(0.0,),
        spatial_fractions=(0.0,),
        correlations=correlations,
        variance_ratios=variance_ratios,
        kind="component_iid",
    )


def fit_diagonal_shrinkage_covariance(
    repeats: np.ndarray,
    *,
    prior_strength: float,
) -> CovarianceFit:
    """Fit a positive diagonal covariance shrunk to componentwise variance."""

    values = np.asarray(repeats, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 2 or values.shape[0] < 2:
        raise ValueError("repeats must have shape [repeat,node,2] with repeat >= 2")
    strength = float(prior_strength)
    if strength <= 0.0:
        raise ValueError("prior_strength must be positive")
    mean = np.mean(values, axis=0)
    centered = values - mean[None]
    empirical = np.sum(np.square(centered), axis=0) / (
        values.shape[0] - 1
    )
    component_prior = np.mean(empirical, axis=0, keepdims=True)
    weight = float(values.shape[0] / (values.shape[0] + strength))
    variance = weight * empirical + (1.0 - weight) * component_prior
    floor = max(float(np.mean(variance)) * 1e-8, 1e-12)
    variance = np.maximum(variance, floor)
    quadratic = np.sum(np.square(centered) / variance[None])
    logdet = float(np.sum(np.log(variance)))
    dimension = int(np.prod(variance.shape))
    objective = 0.5 * (
        values.shape[0] * (dimension * np.log(2.0 * np.pi) + logdet)
        + quadratic
    )
    return CovarianceFit(
        kind="diagonal_shrinkage",
        mean=mean,
        sigma2=None,
        spatial_eigenvalues=None,
        component_covariance=None,
        node_amplitude=None,
        low_rank_vectors=None,
        low_rank_eigenvalues=None,
        diagonal_variance=variance,
        parameters={
            "prior_strength": strength,
            "empirical_weight": weight,
            "calibration_repeat_count": float(values.shape[0]),
            "scale_estimator": "UNBIASED_DIAGONAL_WITH_SHRINKAGE",
        },
        training_nll_per_dimension=float(
            objective / (values.shape[0] * dimension)
        ),
    )


def _graph_whitened_coordinates(
    fit: CovarianceFit,
    centered: np.ndarray,
    *,
    eigenvectors: np.ndarray,
    covariance_scale: float,
) -> np.ndarray:
    if (
        fit.sigma2 is None
        or fit.spatial_eigenvalues is None
        or fit.component_covariance is None
    ):
        raise ValueError("graph covariance fit is incomplete")
    node_count = int(centered.shape[1])
    amplitude = (
        np.ones(node_count, dtype=np.float64)
        if fit.node_amplitude is None
        else np.asarray(fit.node_amplitude, dtype=np.float64)
    )
    if amplitude.shape != (node_count,) or np.any(amplitude <= 0.0):
        raise ValueError("node amplitude does not match the covariance fit")
    normalized_centered = centered / amplitude[None, :, None]
    spectral = np.einsum(
        "ij,rjc->ric",
        np.asarray(eigenvectors, dtype=np.float64).T,
        normalized_centered,
        optimize=True,
    )
    spectral = spectral / np.sqrt(
        fit.spatial_eigenvalues
    )[None, :, None]
    component_inverse_cholesky = np.linalg.inv(
        np.linalg.cholesky(fit.component_covariance)
    )
    whitened = np.einsum(
        "cd,rid->ric",
        component_inverse_cholesky,
        spectral,
        optimize=True,
    )
    return whitened / np.sqrt(float(covariance_scale * fit.sigma2))


def _graph_logdet(
    fit: CovarianceFit,
    *,
    node_count: int,
    component_count: int,
    covariance_scale: float,
) -> float:
    if (
        fit.sigma2 is None
        or fit.spatial_eigenvalues is None
        or fit.component_covariance is None
    ):
        raise ValueError("graph covariance fit is incomplete")
    amplitude = (
        np.ones(node_count, dtype=np.float64)
        if fit.node_amplitude is None
        else np.asarray(fit.node_amplitude, dtype=np.float64)
    )
    dimension = node_count * component_count
    return float(
        dimension * np.log(float(covariance_scale * fit.sigma2))
        + component_count * np.sum(np.log(fit.spatial_eigenvalues))
        + node_count * np.linalg.slogdet(fit.component_covariance)[1]
        + 2.0 * component_count * np.sum(np.log(amplitude))
    )


def evaluate_covariance_fit(
    fit: CovarianceFit,
    repeats: np.ndarray,
    *,
    eigenvectors: np.ndarray,
) -> dict[str, float]:
    """Evaluate held-out likelihood and chi-square whitening calibration."""

    values = np.asarray(repeats, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 2:
        raise ValueError("repeats must have shape [repeat,node,2]")
    if fit.mean.shape != values.shape[1:]:
        raise ValueError("fit mean does not match repeats")
    centered = values - fit.mean[None]
    repeat_count, node_count, component_count = centered.shape
    dimension = node_count * component_count
    calibration_repeat_count = int(
        round(float(fit.parameters.get("calibration_repeat_count", 0.0)))
    )
    if calibration_repeat_count < 2:
        raise ValueError("fit is missing its calibration repeat count")
    predictive_scale = 1.0 + 1.0 / calibration_repeat_count
    if fit.diagonal_variance is not None:
        variance = (
            predictive_scale
            * np.asarray(fit.diagonal_variance, dtype=np.float64)
        )
        quadratic = np.sum(
            np.square(centered) / variance[None],
            axis=(1, 2),
        )
        logdet = float(np.sum(np.log(variance)))
    else:
        whitened = _graph_whitened_coordinates(
            fit,
            centered,
            eigenvectors=eigenvectors,
            covariance_scale=predictive_scale,
        )
        flat = whitened.reshape(repeat_count, dimension)
        quadratic = np.sum(np.square(flat), axis=1)
        logdet = _graph_logdet(
            fit,
            node_count=node_count,
            component_count=component_count,
            covariance_scale=predictive_scale,
        )
        if fit.low_rank_vectors is not None:
            vectors = np.asarray(fit.low_rank_vectors, dtype=np.float64)
            eigenvalues = np.asarray(
                fit.low_rank_eigenvalues,
                dtype=np.float64,
            )
            if (
                vectors.ndim != 2
                or vectors.shape[1] != dimension
                or eigenvalues.shape != (vectors.shape[0],)
                or np.any(eigenvalues < 0.0)
            ):
                raise ValueError("low-rank covariance factors are invalid")
            projection = flat @ vectors.T
            quadratic = quadratic - np.sum(
                (
                    eigenvalues
                    / np.maximum(1.0 + eigenvalues, 1e-12)
                )[None]
                * np.square(projection),
                axis=1,
            )
            quadratic = np.maximum(quadratic, 0.0)
            logdet += float(np.sum(np.log1p(eigenvalues)))
    nll = 0.5 * (
        dimension * np.log(2.0 * np.pi) + logdet + quadratic
    ) / dimension
    coverage = float(np.mean(quadratic <= chi2.ppf(0.95, dimension)))
    energy_ratio = float(np.mean(quadratic) / dimension)
    return {
        "mean_nll_per_dimension": float(np.mean(nll)),
        "p90_nll_per_dimension": float(np.quantile(nll, 0.9)),
        "whitened_energy_ratio": energy_ratio,
        "absolute_log_energy_ratio": float(
            abs(np.log(max(energy_ratio, 1e-12)))
        ),
        "chi_square_95_coverage": coverage,
        "absolute_coverage_error": float(abs(coverage - 0.95)),
        "predictive_mean_estimation_scale": float(predictive_scale),
    }


def fit_amplitude_modulated_graph_covariance(
    repeats: np.ndarray,
    *,
    laplacian_eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    graph_grid: dict[str, Iterable[float]],
    smoothing_strengths: Iterable[float],
    kind: str = "amplitude_graph",
) -> CovarianceFit:
    """Fit graph covariance with a graph-smoothed nodewise noise envelope."""

    values = np.asarray(repeats, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 2 or values.shape[0] < 2:
        raise ValueError("repeats must have shape [repeat,node,2] with repeat >= 2")
    node_count = int(values.shape[1])
    spectrum = np.asarray(laplacian_eigenvalues, dtype=np.float64)
    basis = np.asarray(eigenvectors, dtype=np.float64)
    if spectrum.shape != (node_count,) or basis.shape != (node_count, node_count):
        raise ValueError("graph eigenpairs do not match repeat dimensions")
    strengths = tuple(float(value) for value in smoothing_strengths)
    if not strengths or any(value < 0.0 for value in strengths):
        raise ValueError("smoothing_strengths must be nonempty and nonnegative")

    mean = np.mean(values, axis=0)
    centered = values - mean[None]
    base = fit_graph_separable_covariance(
        values,
        laplacian_eigenvalues=spectrum,
        eigenvectors=basis,
        diffusion_times=graph_grid["diffusion_times"],
        spatial_fractions=graph_grid["spatial_fractions"],
        correlations=graph_grid["correlations"],
        variance_ratios=graph_grid["variance_ratios"],
    )
    if (
        base.sigma2 is None
        or base.spatial_eigenvalues is None
        or base.component_covariance is None
    ):
        raise ValueError("base graph covariance fit is incomplete")
    graph_diagonal = np.sum(
        np.square(basis) * base.spatial_eigenvalues[None, :],
        axis=1,
    )
    expected_energy = (
        float(base.sigma2)
        * graph_diagonal
        * float(np.trace(base.component_covariance))
    )
    empirical_energy = np.mean(
        np.sum(np.square(centered), axis=2),
        axis=0,
    )
    raw_log_amplitude = 0.5 * np.log(
        np.maximum(empirical_energy, 1e-12)
        / np.maximum(expected_energy, 1e-12)
    )
    raw_log_amplitude = (
        raw_log_amplitude - np.mean(raw_log_amplitude)
    )

    best: tuple[float, CovarianceFit] | None = None
    dimension = 2 * node_count
    for strength in strengths:
        coefficients = basis.T @ raw_log_amplitude
        filtered = coefficients / (1.0 + strength * spectrum)
        log_amplitude = basis @ filtered
        amplitude = np.exp(log_amplitude)
        amplitude = amplitude / np.sqrt(np.mean(np.square(amplitude)))
        normalized = centered / amplitude[None, :, None]
        normalized_fit = fit_graph_separable_covariance(
            normalized,
            laplacian_eigenvalues=spectrum,
            eigenvectors=basis,
            diffusion_times=graph_grid["diffusion_times"],
            spatial_fractions=graph_grid["spatial_fractions"],
            correlations=graph_grid["correlations"],
            variance_ratios=graph_grid["variance_ratios"],
            kind=kind,
        )
        objective = (
            normalized_fit.training_nll_per_dimension
            + 2.0 * float(np.sum(np.log(amplitude))) / dimension
        )
        fit = CovarianceFit(
            kind=str(kind),
            mean=mean,
            sigma2=normalized_fit.sigma2,
            spatial_eigenvalues=normalized_fit.spatial_eigenvalues,
            component_covariance=normalized_fit.component_covariance,
            node_amplitude=amplitude,
            low_rank_vectors=None,
            low_rank_eigenvalues=None,
            diagonal_variance=None,
            parameters={
                **normalized_fit.parameters,
                "amplitude_smoothing_strength": float(strength),
                "amplitude_log_standard_deviation": float(
                    np.std(np.log(amplitude))
                ),
            },
            training_nll_per_dimension=float(objective),
        )
        if best is None or objective < best[0]:
            best = (float(objective), fit)
    if best is None:
        raise ValueError("no amplitude-modulated covariance candidate was fit")
    return best[1]


def fit_low_rank_drift_covariance(
    repeats: np.ndarray,
    *,
    laplacian_eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    graph_grid: dict[str, Iterable[float]],
    amplitude_smoothing_strengths: Iterable[float],
    rank_options: Iterable[int],
    shrinkage_strengths: Iterable[float],
    base_amplitude_fit: CovarianceFit | None = None,
    kind: str = "amplitude_graph_low_rank_drift",
) -> CovarianceFit:
    """Add a small temporal drift subspace in base-whitened coordinates."""

    values = np.asarray(repeats, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 2 or values.shape[0] < 3:
        raise ValueError("low-rank drift fitting needs at least three repeats")
    ranks = tuple(int(value) for value in rank_options)
    shrinkages = tuple(float(value) for value in shrinkage_strengths)
    if (
        not ranks
        or min(ranks) < 1
        or max(ranks) >= values.shape[0]
        or not shrinkages
        or min(shrinkages) <= 0.0
        or max(shrinkages) > 1.0
    ):
        raise ValueError("invalid low-rank drift parameter grid")
    base = (
        base_amplitude_fit
        if base_amplitude_fit is not None
        else fit_amplitude_modulated_graph_covariance(
            values,
            laplacian_eigenvalues=laplacian_eigenvalues,
            eigenvectors=eigenvectors,
            graph_grid=graph_grid,
            smoothing_strengths=amplitude_smoothing_strengths,
        )
    )
    mean = np.mean(values, axis=0)
    centered = values - mean[None]
    whitened = _graph_whitened_coordinates(
        base,
        centered,
        eigenvectors=eigenvectors,
        covariance_scale=1.0,
    )
    repeat_count, node_count, component_count = whitened.shape
    dimension = node_count * component_count
    flat = whitened.reshape(repeat_count, dimension)
    _, singular_values, right = np.linalg.svd(
        flat,
        full_matrices=False,
    )
    empirical_eigenvalues = np.square(singular_values) / (
        repeat_count - 1
    )
    base_logdet = _graph_logdet(
        base,
        node_count=node_count,
        component_count=component_count,
        covariance_scale=1.0,
    )
    base_energy = np.sum(np.square(flat), axis=1)
    best: tuple[float, CovarianceFit] | None = None
    for rank in ranks:
        vectors = np.asarray(right[:rank], dtype=np.float64)
        projection = flat @ vectors.T
        for shrinkage in shrinkages:
            low_rank_eigenvalues = np.maximum(
                (empirical_eigenvalues[:rank] - 1.0) * shrinkage,
                0.0,
            )
            correction = (
                low_rank_eigenvalues
                / np.maximum(1.0 + low_rank_eigenvalues, 1e-12)
            )
            quadratic = base_energy - np.sum(
                correction[None] * np.square(projection),
                axis=1,
            )
            logdet = base_logdet + float(
                np.sum(np.log1p(low_rank_eigenvalues))
            )
            objective = 0.5 * (
                repeat_count
                * (dimension * np.log(2.0 * np.pi) + logdet)
                + float(np.sum(quadratic))
            ) / float(repeat_count * dimension)
            fit = CovarianceFit(
                kind=str(kind),
                mean=mean,
                sigma2=base.sigma2,
                spatial_eigenvalues=base.spatial_eigenvalues,
                component_covariance=base.component_covariance,
                node_amplitude=base.node_amplitude,
                low_rank_vectors=vectors,
                low_rank_eigenvalues=low_rank_eigenvalues,
                diagonal_variance=None,
                parameters={
                    **base.parameters,
                    "low_rank_drift_rank": float(rank),
                    "low_rank_drift_shrinkage": float(shrinkage),
                    "largest_base_whitened_empirical_eigenvalue": float(
                        empirical_eigenvalues[0]
                    ),
                    "retained_low_rank_eigenvalue_sum": float(
                        np.sum(low_rank_eigenvalues)
                    ),
                },
                training_nll_per_dimension=float(objective),
            )
            if best is None or objective < best[0]:
                best = (float(objective), fit)
    if best is None:
        raise ValueError("no low-rank drift covariance candidate was fit")
    return best[1]


def _sample_separable(
    *,
    repeat_count: int,
    rng: np.random.Generator,
    eigenvectors: np.ndarray,
    spatial_eigenvalues: np.ndarray,
    component: np.ndarray,
    sigma: float,
) -> np.ndarray:
    node_count = int(len(spatial_eigenvalues))
    white = rng.normal(size=(int(repeat_count), node_count, 2))
    spectral = white * np.sqrt(spatial_eigenvalues)[None, :, None]
    spatial = np.einsum(
        "ij,rjc->ric",
        eigenvectors,
        spectral,
        optimize=True,
    )
    return (
        float(sigma)
        * np.einsum(
            "rni,ji->rnj",
            spatial,
            np.linalg.cholesky(component),
            optimize=True,
        )
    )


def simulate_flowoff_repeats(
    *,
    family: str,
    repeat_count: int,
    rng: np.random.Generator,
    laplacian_eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    detector_xy: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate controlled flow-off repeats on a measured detector graph."""

    name = str(family)
    coordinates = np.asarray(detector_xy, dtype=np.float64)
    if coordinates.shape != (len(laplacian_eigenvalues), 2):
        raise ValueError("detector_xy does not match graph size")
    if int(repeat_count) < 1:
        raise ValueError("repeat_count must be positive")

    if name == "component_iid":
        parameters = {
            "diffusion_time": 0.0,
            "spatial_fraction": 0.0,
            "component_correlation": 0.2,
            "component_variance_ratio": 1.25,
            "sigma": 1.0,
        }
    elif name in {"graph_heat", "nonstationary_drift"}:
        parameters = {
            "diffusion_time": 1.2,
            "spatial_fraction": 0.75,
            "component_correlation": 0.35,
            "component_variance_ratio": 1.35,
            "sigma": 1.0,
        }
    else:
        raise ValueError(f"unknown flow-off simulation family: {name}")

    spatial = graph_heat_eigenvalues(
        laplacian_eigenvalues,
        diffusion_time=float(parameters["diffusion_time"]),
        spatial_fraction=float(parameters["spatial_fraction"]),
    )
    component = component_covariance(
        correlation=float(parameters["component_correlation"]),
        variance_ratio=float(parameters["component_variance_ratio"]),
    )
    samples = _sample_separable(
        repeat_count=int(repeat_count),
        rng=rng,
        eigenvectors=np.asarray(eigenvectors, dtype=np.float64),
        spatial_eigenvalues=spatial,
        component=component,
        sigma=float(parameters["sigma"]),
    )
    metadata: dict[str, Any] = {
        "family": name,
        "base_parameters": parameters,
        "model_is_in_graph_family": name in {"component_iid", "graph_heat"},
    }
    if name == "nonstationary_drift":
        standardized = (
            coordinates - np.mean(coordinates, axis=0, keepdims=True)
        ) / np.maximum(np.std(coordinates, axis=0, keepdims=True), 1e-12)
        amplitude = np.exp(
            0.30
            * (
                0.65 * standardized[:, 0]
                + 0.35 * np.sin(np.pi * standardized[:, 1])
            )
        )
        amplitude = amplitude / np.sqrt(np.mean(np.square(amplitude)))
        samples = samples * amplitude[None, :, None]
        mode = (
            0.7 * eigenvectors[:, 1]
            + 0.3 * standardized[:, 0]
        )
        mode = mode / np.sqrt(np.mean(np.square(mode)))
        component_mode = np.asarray([1.0, -0.55], dtype=np.float64)
        component_mode = component_mode / np.linalg.norm(component_mode)
        drift = rng.normal(scale=0.45, size=(int(repeat_count), 1, 1))
        samples = (
            samples
            + drift * mode[None, :, None] * component_mode[None, None, :]
        )
        metadata["misspecification"] = {
            "nodewise_amplitude_modulation": True,
            "low_rank_temporal_drift_scale": 0.45,
        }
    return samples, metadata


def fit_gated_graph_covariance(
    repeats: np.ndarray,
    *,
    laplacian_eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    graph_grid: dict[str, Iterable[float]],
    validation_fraction: float,
    minimum_validation_gain_per_dimension: float,
    full_graph_fit: CovarianceFit | None = None,
    full_iid_fit: CovarianceFit | None = None,
    amplitude_smoothing_strengths: Iterable[float] | None = None,
    full_amplitude_fit: CovarianceFit | None = None,
    low_rank_drift: dict[str, Iterable[float] | Iterable[int]] | None = None,
    full_low_rank_fit: CovarianceFit | None = None,
) -> tuple[CovarianceFit, dict[str, Any]]:
    """Use held-out repeats to decide whether spatial covariance is warranted."""

    values = np.asarray(repeats, dtype=np.float64)
    repeat_count = int(values.shape[0])
    fraction = float(validation_fraction)
    if repeat_count < 4 or not 0.1 <= fraction <= 0.5:
        raise ValueError("gated fitting needs at least four repeats and a valid split")
    validation_count = max(1, int(np.ceil(repeat_count * fraction)))
    fit_count = repeat_count - validation_count
    if fit_count < 2:
        raise ValueError("gated fitting leaves fewer than two fit repeats")
    fit_values = values[:fit_count]
    validation_values = values[fit_count:]

    iid_pre = fit_component_iid_covariance(
        fit_values,
        laplacian_eigenvalues=laplacian_eigenvalues,
        eigenvectors=eigenvectors,
        correlations=graph_grid["correlations"],
        variance_ratios=graph_grid["variance_ratios"],
    )
    graph_pre = fit_graph_separable_covariance(
        fit_values,
        laplacian_eigenvalues=laplacian_eigenvalues,
        eigenvectors=eigenvectors,
        diffusion_times=graph_grid["diffusion_times"],
        spatial_fractions=graph_grid["spatial_fractions"],
        correlations=graph_grid["correlations"],
        variance_ratios=graph_grid["variance_ratios"],
    )
    amplitude_pre = (
        None
        if amplitude_smoothing_strengths is None
        else fit_amplitude_modulated_graph_covariance(
            fit_values,
            laplacian_eigenvalues=laplacian_eigenvalues,
            eigenvectors=eigenvectors,
            graph_grid=graph_grid,
            smoothing_strengths=amplitude_smoothing_strengths,
        )
    )
    low_rank_pre = (
        None
        if low_rank_drift is None or amplitude_pre is None
        else fit_low_rank_drift_covariance(
            fit_values,
            laplacian_eigenvalues=laplacian_eigenvalues,
            eigenvectors=eigenvectors,
            graph_grid=graph_grid,
            amplitude_smoothing_strengths=(
                amplitude_smoothing_strengths or ()
            ),
            rank_options=low_rank_drift["rank_options"],
            shrinkage_strengths=low_rank_drift[
                "shrinkage_strengths"
            ],
            base_amplitude_fit=amplitude_pre,
        )
    )
    iid_validation = evaluate_covariance_fit(
        iid_pre,
        validation_values,
        eigenvectors=eigenvectors,
    )
    graph_validation = evaluate_covariance_fit(
        graph_pre,
        validation_values,
        eigenvectors=eigenvectors,
    )
    candidates = {
        "component_iid": (iid_pre, iid_validation),
        "graph_separable": (graph_pre, graph_validation),
    }
    if amplitude_pre is not None:
        candidates["amplitude_graph"] = (
            amplitude_pre,
            evaluate_covariance_fit(
                amplitude_pre,
                validation_values,
                eigenvectors=eigenvectors,
            ),
        )
    if low_rank_pre is not None:
        candidates["amplitude_graph_low_rank_drift"] = (
            low_rank_pre,
            evaluate_covariance_fit(
                low_rank_pre,
                validation_values,
                eigenvectors=eigenvectors,
            ),
        )
    selected_name = min(
        candidates,
        key=lambda name: candidates[name][1]["mean_nll_per_dimension"],
    )
    selected_validation = candidates[selected_name][1]
    gain = (
        iid_validation["mean_nll_per_dimension"]
        - selected_validation["mean_nll_per_dimension"]
    )
    selected_pre = candidates[selected_name][0]
    spatial_fraction = float(
        selected_pre.parameters.get("spatial_fraction", 0.0)
    )
    activate = bool(
        selected_name != "component_iid"
        and gain >= float(minimum_validation_gain_per_dimension)
        and spatial_fraction > 0.0
    )
    amplitude_activate = bool(
        activate and selected_name == "amplitude_graph"
    )
    low_rank_activate = bool(
        activate and selected_name == "amplitude_graph_low_rank_drift"
    )
    if low_rank_activate:
        if full_low_rank_fit is None:
            final = fit_low_rank_drift_covariance(
                values,
                laplacian_eigenvalues=laplacian_eigenvalues,
                eigenvectors=eigenvectors,
                graph_grid=graph_grid,
                amplitude_smoothing_strengths=(
                    amplitude_smoothing_strengths or ()
                ),
                rank_options=low_rank_drift["rank_options"],
                shrinkage_strengths=low_rank_drift[
                    "shrinkage_strengths"
                ],
                kind="gated_amplitude_graph_low_rank_drift",
            )
        else:
            final = CovarianceFit(
                kind="gated_amplitude_graph_low_rank_drift",
                mean=full_low_rank_fit.mean,
                sigma2=full_low_rank_fit.sigma2,
                spatial_eigenvalues=full_low_rank_fit.spatial_eigenvalues,
                component_covariance=full_low_rank_fit.component_covariance,
                node_amplitude=full_low_rank_fit.node_amplitude,
                low_rank_vectors=full_low_rank_fit.low_rank_vectors,
                low_rank_eigenvalues=full_low_rank_fit.low_rank_eigenvalues,
                diagonal_variance=full_low_rank_fit.diagonal_variance,
                parameters=full_low_rank_fit.parameters,
                training_nll_per_dimension=(
                    full_low_rank_fit.training_nll_per_dimension
                ),
            )
    elif amplitude_activate:
        if full_amplitude_fit is None:
            final = fit_amplitude_modulated_graph_covariance(
                values,
                laplacian_eigenvalues=laplacian_eigenvalues,
                eigenvectors=eigenvectors,
                graph_grid=graph_grid,
                smoothing_strengths=amplitude_smoothing_strengths or (),
                kind="gated_amplitude_graph",
            )
        else:
            final = CovarianceFit(
                kind="gated_amplitude_graph",
                mean=full_amplitude_fit.mean,
                sigma2=full_amplitude_fit.sigma2,
                spatial_eigenvalues=full_amplitude_fit.spatial_eigenvalues,
                component_covariance=full_amplitude_fit.component_covariance,
                node_amplitude=full_amplitude_fit.node_amplitude,
                low_rank_vectors=full_amplitude_fit.low_rank_vectors,
                low_rank_eigenvalues=full_amplitude_fit.low_rank_eigenvalues,
                diagonal_variance=full_amplitude_fit.diagonal_variance,
                parameters=full_amplitude_fit.parameters,
                training_nll_per_dimension=(
                    full_amplitude_fit.training_nll_per_dimension
                ),
            )
    elif activate:
        if full_graph_fit is None:
            final = fit_graph_separable_covariance(
                values,
                laplacian_eigenvalues=laplacian_eigenvalues,
                eigenvectors=eigenvectors,
                diffusion_times=graph_grid["diffusion_times"],
                spatial_fractions=graph_grid["spatial_fractions"],
                correlations=graph_grid["correlations"],
                variance_ratios=graph_grid["variance_ratios"],
                kind="gated_graph",
            )
        else:
            final = CovarianceFit(
                kind="gated_graph",
                mean=full_graph_fit.mean,
                sigma2=full_graph_fit.sigma2,
                spatial_eigenvalues=full_graph_fit.spatial_eigenvalues,
                component_covariance=full_graph_fit.component_covariance,
                node_amplitude=full_graph_fit.node_amplitude,
                low_rank_vectors=full_graph_fit.low_rank_vectors,
                low_rank_eigenvalues=full_graph_fit.low_rank_eigenvalues,
                diagonal_variance=full_graph_fit.diagonal_variance,
                parameters=full_graph_fit.parameters,
                training_nll_per_dimension=(
                    full_graph_fit.training_nll_per_dimension
                ),
            )
    else:
        final = (
            full_iid_fit
            if full_iid_fit is not None
            else fit_component_iid_covariance(
                values,
                laplacian_eigenvalues=laplacian_eigenvalues,
                eigenvectors=eigenvectors,
                correlations=graph_grid["correlations"],
                variance_ratios=graph_grid["variance_ratios"],
            )
        )
    return final, {
        "fit_repeat_count": fit_count,
        "validation_repeat_count": validation_count,
        "validation_nll_gain_per_dimension": float(gain),
        "selected_preliminary_candidate": selected_name,
        "preliminary_graph_spatial_fraction": spatial_fraction,
        "graph_activated": activate,
        "amplitude_activated": amplitude_activate,
        "low_rank_drift_activated": low_rank_activate,
        "selected_kind": final.kind,
    }


def empirical_covariance_rank_upper_bound(
    *,
    repeat_count: int,
    dimension: int,
) -> int:
    """Return the rank ceiling of a mean-centered empirical covariance."""

    repeats = int(repeat_count)
    size = int(dimension)
    if repeats < 1 or size < 1:
        raise ValueError("repeat_count and dimension must be positive")
    return min(repeats - 1, size)
