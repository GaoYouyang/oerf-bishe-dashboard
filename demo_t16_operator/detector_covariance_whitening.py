"""Detector-domain covariance whitening for matrix-free BOST reconstruction."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

from .detector_graph_covariance import (
    CovarianceFit,
    covariance_whitening_matrix,
)


WHITENING_SCHEMA = "psu-b0-detector-covariance-whitening-1.0"


def spatially_tempered_covariance_fit(
    component_fit: CovarianceFit,
    graph_fit: CovarianceFit,
    *,
    spatial_exponent: float,
) -> CovarianceFit:
    """Interpolate only the detector-graph covariance spectrum.

    The component-IID fit supplies the mean, global scale, and u/v covariance.
    The graph fit supplies only its unit-mean spatial spectrum. Raising that
    spectrum to ``spatial_exponent`` provides an auditable path from spatial
    IID (zero) to the fitted graph spectrum (one) without changing component
    calibration at the same time.
    """

    exponent = float(spatial_exponent)
    if not np.isfinite(exponent) or not 0.0 <= exponent <= 1.0:
        raise ValueError("spatial_exponent must lie in [0,1]")
    for name, fit in (
        ("component_fit", component_fit),
        ("graph_fit", graph_fit),
    ):
        if (
            fit.diagonal_variance is not None
            or fit.sigma2 is None
            or fit.spatial_eigenvalues is None
            or fit.component_covariance is None
        ):
            raise ValueError(f"{name} must be a separable graph-style fit")
        if fit.node_amplitude is not None or fit.low_rank_vectors is not None:
            raise ValueError(
                f"{name} cannot contain amplitude or low-rank corrections"
            )
    if component_fit.mean.shape != graph_fit.mean.shape:
        raise ValueError("component and graph fits must use the same detector")
    if not np.allclose(
        component_fit.mean,
        graph_fit.mean,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("component and graph fits must use the same mean")

    target = np.asarray(
        graph_fit.spatial_eigenvalues,
        dtype=np.float64,
    )
    baseline = np.asarray(
        component_fit.spatial_eigenvalues,
        dtype=np.float64,
    )
    if target.shape != baseline.shape or np.any(target <= 0.0):
        raise ValueError("component and graph spectra must align and be positive")
    tempered = np.exp(exponent * np.log(target))
    tempered = tempered / np.mean(tempered)
    parameters = {
        **component_fit.parameters,
        "spatial_tempering_exponent": exponent,
        "source_graph_kind": str(graph_fit.kind),
        "source_graph_spatial_fraction": float(
            graph_fit.parameters.get("spatial_fraction", 0.0)
        ),
        "source_graph_diffusion_time": float(
            graph_fit.parameters.get("diffusion_time", 0.0)
        ),
        "component_parameters_held_fixed": 1.0,
    }
    return CovarianceFit(
        kind=f"spatially_tempered_graph_{exponent:g}",
        mean=np.asarray(component_fit.mean, dtype=np.float64).copy(),
        sigma2=float(component_fit.sigma2),
        spatial_eigenvalues=tempered,
        component_covariance=np.asarray(
            component_fit.component_covariance,
            dtype=np.float64,
        ).copy(),
        node_amplitude=None,
        low_rank_vectors=None,
        low_rank_eigenvalues=None,
        diagonal_variance=None,
        parameters=parameters,
        training_nll_per_dimension=float("nan"),
    )


class DetectorCovarianceWhitening(nn.Module):
    """Apply one fitted detector covariance model per camera view.

    The fitted covariance describes unit-scale calibration noise. A scalar
    ``scale_by_view`` may vary by batch item and carries the absolute noise
    scale without changing the detector covariance shape.
    """

    def __init__(
        self,
        fits: Sequence[CovarianceFit],
        *,
        eigenvectors_by_view: Sequence[np.ndarray],
        scale_by_view: Any,
        predictive_mean_correction: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if not fits or len(fits) != len(eigenvectors_by_view):
            raise ValueError("fits and eigenvectors must contain the same views")
        node_counts = {int(fit.mean.shape[0]) for fit in fits}
        if len(node_counts) != 1:
            raise ValueError("all covariance fits must use one rays-per-view")
        rays_per_view = node_counts.pop()
        if rays_per_view < 1:
            raise ValueError("rays_per_view must be positive")

        matrices = []
        means = []
        predictive_scales = []
        for fit, eigenvectors in zip(
            fits,
            eigenvectors_by_view,
            strict=True,
        ):
            repeat_count = int(
                round(float(fit.parameters.get("calibration_repeat_count", 0.0)))
            )
            if predictive_mean_correction and repeat_count < 2:
                raise ValueError(
                    "predictive correction needs calibration_repeat_count >= 2"
                )
            predictive_scale = (
                1.0 + 1.0 / repeat_count
                if predictive_mean_correction
                else 1.0
            )
            matrices.append(
                covariance_whitening_matrix(
                    fit,
                    eigenvectors=np.asarray(eigenvectors, dtype=np.float64),
                    covariance_scale=predictive_scale,
                )
            )
            means.append(np.asarray(fit.mean, dtype=np.float64))
            predictive_scales.append(float(predictive_scale))

        scales = torch.as_tensor(scale_by_view, dtype=dtype)
        if scales.ndim == 1:
            scales = scales[None]
        if scales.ndim != 2 or scales.shape[1] != len(fits):
            raise ValueError("scale_by_view must have shape [batch,view] or [view]")
        if torch.any(~torch.isfinite(scales)) or torch.any(scales <= 0.0):
            raise ValueError("scale_by_view must be finite and positive")

        self.view_count = len(fits)
        self.rays_per_view = rays_per_view
        self.measurement_dimension_per_view = 2 * rays_per_view
        self.predictive_mean_correction = bool(predictive_mean_correction)
        self.register_buffer(
            "matrix",
            torch.as_tensor(np.stack(matrices), dtype=dtype),
        )
        self.register_buffer(
            "calibration_mean",
            torch.as_tensor(np.stack(means), dtype=dtype),
        )
        self.register_buffer("scale_by_view", scales)
        self.register_buffer(
            "predictive_scale_by_view",
            torch.as_tensor(predictive_scales, dtype=dtype),
        )

    def _canonical(self, values: torch.Tensor) -> torch.Tensor:
        expected_rays = self.view_count * self.rays_per_view
        if values.ndim != 3 or values.shape[1:] != (expected_rays, 2):
            raise ValueError(
                "detector values must have shape [batch,view*rays_per_view,2]"
            )
        return values.to(self.matrix).reshape(
            len(values),
            self.view_count,
            self.measurement_dimension_per_view,
        )

    def _expanded_scale(self, batch_size: int) -> torch.Tensor:
        if self.scale_by_view.shape[0] not in {1, int(batch_size)}:
            raise ValueError(
                "scale_by_view batch must be one or match detector values"
            )
        return self.scale_by_view.expand(int(batch_size), -1)

    def center_observation(self, observation_uv: torch.Tensor) -> torch.Tensor:
        """Subtract the scaled calibration mean without applying whitening."""

        canonical = self._canonical(observation_uv)
        scale = self._expanded_scale(len(observation_uv))
        mean = self.calibration_mean.reshape(
            self.view_count,
            self.measurement_dimension_per_view,
        )
        centered = canonical - scale[:, :, None] * mean[None]
        return centered.reshape_as(observation_uv)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        canonical = self._canonical(values)
        scale = self._expanded_scale(len(values))
        whitened = torch.einsum(
            "vij,bvj->bvi",
            self.matrix,
            canonical,
        )
        whitened = whitened / scale[:, :, None]
        return whitened.reshape_as(values)

    def transpose(self, values: torch.Tensor) -> torch.Tensor:
        """Apply the exact transpose of :meth:`forward`."""

        canonical = self._canonical(values)
        scale = self._expanded_scale(len(values))
        scaled = canonical / scale[:, :, None]
        output = torch.einsum(
            "vji,bvj->bvi",
            self.matrix,
            scaled,
        )
        return output.reshape_as(values)

    def prepare_observation(self, observation_uv: torch.Tensor) -> torch.Tensor:
        """Center and whiten a measured observation for the wrapped inverse."""

        return self(self.center_observation(observation_uv))


class WhitenedMeasurementOperator(nn.Module):
    """Compose detector whitening with a matrix-free physical forward map."""

    def __init__(
        self,
        base_operator: nn.Module,
        whitening: DetectorCovarianceWhitening,
    ) -> None:
        super().__init__()
        expected_rays = whitening.view_count * whitening.rays_per_view
        if int(getattr(base_operator, "ray_count", -1)) != expected_rays:
            raise ValueError("base operator ray count does not match whitening")
        self.base_operator = base_operator
        self.whitening = whitening
        self.grid_shape = tuple(int(value) for value in base_operator.grid_shape)
        self.spacing_xyz = tuple(
            float(value) for value in base_operator.spacing_xyz
        )
        self.ray_count = expected_rays

    @property
    def support(self) -> torch.Tensor:
        return self.base_operator.support

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        return self.whitening(self.base_operator(volume))

    def adjoint(self, residual_uv: torch.Tensor) -> torch.Tensor:
        return self.base_operator.adjoint(
            self.whitening.transpose(residual_uv)
        )

    def prepare_observation(self, observation_uv: torch.Tensor) -> torch.Tensor:
        return self.whitening.prepare_observation(observation_uv)

    def reset_call_counts(self) -> None:
        self.base_operator.reset_call_counts()

    def call_report(self) -> dict[str, int]:
        return self.base_operator.call_report()


__all__ = [
    "DetectorCovarianceWhitening",
    "WHITENING_SCHEMA",
    "WhitenedMeasurementOperator",
    "spatially_tempered_covariance_fit",
]
