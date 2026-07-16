"""Observable-morphology mixtures of fixed positive PCGLS experts."""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn

from .psu_b0_conditioned_pcgls import (
    MaterializedPositiveSpectralDirection,
)
from .psu_b0_initial_normal_features import (
    initial_normal_spectral_features,
)


MORPHOLOGY_EXPERT_SCHEMA = "psu-b0-observable-morphology-spectral-experts-1.0"


def materialize_log_expert_mixture(
    scores: torch.Tensor,
    *,
    expert_log_gains: torch.Tensor,
    baseline_expert_index: int,
    temperature: float,
    confidence_threshold: float,
    maximum_blend: float,
) -> MaterializedPositiveSpectralDirection:
    """Create a fixed positive log-space expert mixture from selector scores."""

    if scores.ndim != 2:
        raise ValueError("scores must have shape [batch,expert]")
    if expert_log_gains.ndim != 4:
        raise ValueError(
            "expert_log_gains must have shape [expert,z,y,x_half]"
        )
    if scores.shape[1] != expert_log_gains.shape[0]:
        raise ValueError("scores and expert bank do not align")
    if not 0 <= int(baseline_expert_index) < scores.shape[1]:
        raise ValueError("baseline_expert_index is out of range")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    if not 0.0 <= float(maximum_blend) <= 1.0:
        raise ValueError("maximum_blend must lie in [0,1]")
    top = torch.topk(scores, k=2, dim=1)
    margin = top.values[:, 0] - top.values[:, 1]
    accepted = (
        margin >= float(confidence_threshold)
    ) & (top.indices[:, 0] != int(baseline_expert_index))
    blend = accepted.to(scores) * float(maximum_blend)
    weights = torch.softmax(scores / float(temperature), dim=1)
    expert_logs = expert_log_gains.to(scores)
    mixture_log = torch.einsum("be,ezyp->bzyp", weights, expert_logs)
    baseline_log = expert_logs[int(baseline_expert_index)][None]
    log_gain = baseline_log + blend[:, None, None, None] * (
        mixture_log - baseline_log
    )
    log_gain = log_gain - log_gain.mean(
        dim=(1, 2, 3),
        keepdim=True,
    )
    gain = torch.exp(log_gain)
    return MaterializedPositiveSpectralDirection(
        gain=gain,
        controller_coefficients=weights,
        log_correction=log_gain - baseline_log,
    )


class ObservableMorphologyExpertFactory(nn.Module):
    """Map the shared first adjoint field to one fixed SPD expert mixture."""

    def __init__(
        self,
        *,
        expert_log_gains: torch.Tensor,
        expert_candidate_ids: Sequence[str],
        baseline_expert_index: int,
        feature_mean: torch.Tensor,
        feature_scale: torch.Tensor,
        ridge_weights: torch.Tensor,
        temperature: float,
        confidence_threshold: float,
        maximum_blend: float,
    ) -> None:
        super().__init__()
        expert_ids = tuple(str(value) for value in expert_candidate_ids)
        logs = torch.as_tensor(expert_log_gains, dtype=torch.float32)
        mean = torch.as_tensor(feature_mean, dtype=torch.float32).flatten()
        scale = torch.as_tensor(feature_scale, dtype=torch.float32).flatten()
        weights = torch.as_tensor(ridge_weights, dtype=torch.float32)
        if logs.ndim != 4 or len(logs) != len(expert_ids):
            raise ValueError("expert ids and gains must align")
        if weights.ndim != 2 or weights.shape[1] != len(expert_ids):
            raise ValueError("ridge weights must output one score per expert")
        if weights.shape[0] != len(mean) + 1 or mean.shape != scale.shape:
            raise ValueError("ridge feature schema does not align")
        if torch.any(scale <= 0.0):
            raise ValueError("feature scale must be positive")
        self.expert_candidate_ids = expert_ids
        self.baseline_expert_index = int(baseline_expert_index)
        self.temperature = float(temperature)
        self.confidence_threshold = float(confidence_threshold)
        self.maximum_blend = float(maximum_blend)
        self.register_buffer("expert_log_gains", logs)
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_scale", scale)
        self.register_buffer("ridge_weights", weights)

    def selector_scores(
        self,
        initial_normal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features, _ = initial_normal_spectral_features(initial_normal)
        normalized = (
            features - self.feature_mean.to(features)[None]
        ) / self.feature_scale.to(features)[None]
        design = torch.cat(
            (
                torch.ones(
                    (len(features), 1),
                    dtype=features.dtype,
                    device=features.device,
                ),
                normalized,
            ),
            dim=1,
        )
        scores = design @ self.ridge_weights.to(features)
        return scores, features

    def forward(
        self,
        initial_normal: torch.Tensor,
        **_: Any,
    ) -> MaterializedPositiveSpectralDirection:
        scores, _ = self.selector_scores(initial_normal)
        return materialize_log_expert_mixture(
            scores,
            expert_log_gains=self.expert_log_gains,
            baseline_expert_index=self.baseline_expert_index,
            temperature=self.temperature,
            confidence_threshold=self.confidence_threshold,
            maximum_blend=self.maximum_blend,
        )


__all__ = [
    "MORPHOLOGY_EXPERT_SCHEMA",
    "ObservableMorphologyExpertFactory",
    "materialize_log_expert_mixture",
]
