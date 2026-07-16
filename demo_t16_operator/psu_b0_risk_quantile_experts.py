"""Risk-quantile routing along baseline-to-single-expert SPD paths."""

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


RISK_QUANTILE_EXPERT_SCHEMA = "psu-b0-risk-quantile-experts-1.0"


def risk_quantile_actions(
    *,
    mean_scores: torch.Tensor,
    lower_scores: torch.Tensor,
    harm_logits: torch.Tensor,
    baseline_expert_index: int,
    route_mode: str,
    minimum_score: float,
    maximum_harm_probability: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Choose one expert or reject to the exact baseline."""

    if mean_scores.ndim != 2:
        raise ValueError("mean_scores must have shape [batch,expert]")
    if lower_scores.shape != mean_scores.shape:
        raise ValueError("lower_scores must align with mean_scores")
    if harm_logits.shape != mean_scores.shape:
        raise ValueError("harm_logits must align with mean_scores")
    if not 0 <= int(baseline_expert_index) < mean_scores.shape[1]:
        raise ValueError("baseline_expert_index is out of range")
    if route_mode not in {
        "mean_only",
        "quantile_only",
        "quantile_harm",
        "mean_quantile_harm",
    }:
        raise ValueError(f"unsupported route_mode: {route_mode}")
    if not 0.0 <= float(maximum_harm_probability) <= 1.0:
        raise ValueError("maximum_harm_probability must lie in [0,1]")

    rank_scores = (
        mean_scores if route_mode == "mean_only" else lower_scores
    )
    selected = torch.argmax(rank_scores, dim=1)
    row = torch.arange(len(selected), device=selected.device)
    selected_rank = rank_scores[row, selected]
    selected_mean = mean_scores[row, selected]
    selected_harm = torch.sigmoid(harm_logits[row, selected])
    accepted = (
        (selected != int(baseline_expert_index))
        & (selected_rank >= float(minimum_score))
    )
    if route_mode in {"quantile_harm", "mean_quantile_harm"}:
        accepted = accepted & (
            selected_harm <= float(maximum_harm_probability)
        )
    if route_mode == "mean_quantile_harm":
        accepted = accepted & (selected_mean >= 0.0)
    actions = torch.where(
        accepted,
        selected,
        torch.full_like(selected, int(baseline_expert_index)),
    )
    return actions, accepted, selected_harm


def multiobjective_risk_actions(
    *,
    field_mean_scores: torch.Tensor,
    field_lower_scores: torch.Tensor,
    field_harm_logits: torch.Tensor,
    front_lower_scores: torch.Tensor,
    front_harm_logits: torch.Tensor,
    baseline_expert_index: int,
    minimum_field_score: float,
    maximum_field_harm_probability: float,
    minimum_front_lower_delta: float,
    maximum_front_harm_probability: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Route only when both field and front-risk heads accept an expert."""

    matrices = (
        field_mean_scores,
        field_lower_scores,
        field_harm_logits,
        front_lower_scores,
        front_harm_logits,
    )
    if field_mean_scores.ndim != 2 or any(
        value.shape != field_mean_scores.shape for value in matrices[1:]
    ):
        raise ValueError("multiobjective score matrices must align")
    if not 0 <= int(baseline_expert_index) < field_mean_scores.shape[1]:
        raise ValueError("baseline_expert_index is out of range")
    if not 0.0 <= float(maximum_field_harm_probability) <= 1.0:
        raise ValueError("maximum_field_harm_probability must lie in [0,1]")
    if not 0.0 <= float(maximum_front_harm_probability) <= 1.0:
        raise ValueError("maximum_front_harm_probability must lie in [0,1]")

    selected = torch.argmax(field_mean_scores, dim=1)
    row = torch.arange(len(selected), device=selected.device)
    field_mean = field_mean_scores[row, selected]
    field_lower = field_lower_scores[row, selected]
    field_harm = torch.sigmoid(field_harm_logits[row, selected])
    front_lower = front_lower_scores[row, selected]
    front_harm = torch.sigmoid(front_harm_logits[row, selected])
    accepted = (
        (selected != int(baseline_expert_index))
        & (field_mean >= float(minimum_field_score))
        & (field_lower >= 0.0)
        & (
            field_harm
            <= float(maximum_field_harm_probability)
        )
        & (front_lower >= float(minimum_front_lower_delta))
        & (
            front_harm
            <= float(maximum_front_harm_probability)
        )
    )
    actions = torch.where(
        accepted,
        selected,
        torch.full_like(selected, int(baseline_expert_index)),
    )
    return actions, accepted, field_harm, front_harm


def materialize_single_expert_path(
    action_indices: torch.Tensor,
    *,
    expert_log_gains: torch.Tensor,
    baseline_expert_index: int,
    interpolation_fraction: float,
) -> MaterializedPositiveSpectralDirection:
    """Interpolate in log space from the baseline to one selected expert."""

    actions = torch.as_tensor(action_indices, dtype=torch.long).flatten()
    if expert_log_gains.ndim != 4:
        raise ValueError(
            "expert_log_gains must have shape [expert,z,y,x_half]"
        )
    if not 0 <= int(baseline_expert_index) < len(expert_log_gains):
        raise ValueError("baseline_expert_index is out of range")
    if torch.any(actions < 0) or torch.any(actions >= len(expert_log_gains)):
        raise ValueError("action_indices contain an invalid expert")
    if not 0.0 <= float(interpolation_fraction) <= 1.0:
        raise ValueError("interpolation_fraction must lie in [0,1]")

    logs = expert_log_gains.to(actions.device)
    baseline = logs[int(baseline_expert_index)][None]
    selected = logs[actions]
    active = (actions != int(baseline_expert_index)).to(logs)
    fraction = active * float(interpolation_fraction)
    log_gain = baseline + fraction[:, None, None, None] * (
        selected - baseline
    )
    log_gain = log_gain - log_gain.mean(
        dim=(1, 2, 3),
        keepdim=True,
    )
    gain = torch.exp(log_gain)
    coefficients = torch.nn.functional.one_hot(
        actions,
        num_classes=len(logs),
    ).to(gain)
    return MaterializedPositiveSpectralDirection(
        gain=gain,
        controller_coefficients=coefficients,
        log_correction=log_gain - baseline,
    )


class RiskQuantileSingleExpertFactory(nn.Module):
    """Map the shared first adjoint field to a risk-gated SPD expert path."""

    def __init__(
        self,
        *,
        expert_log_gains: torch.Tensor,
        expert_candidate_ids: Sequence[str],
        baseline_expert_index: int,
        feature_mean: torch.Tensor,
        feature_scale: torch.Tensor,
        mean_weights: torch.Tensor,
        lower_weights: torch.Tensor,
        harm_weights: torch.Tensor,
        route_mode: str,
        minimum_score: float,
        maximum_harm_probability: float,
        interpolation_fraction: float,
    ) -> None:
        super().__init__()
        expert_ids = tuple(str(value) for value in expert_candidate_ids)
        logs = torch.as_tensor(expert_log_gains, dtype=torch.float32)
        mean = torch.as_tensor(feature_mean, dtype=torch.float32).flatten()
        scale = torch.as_tensor(feature_scale, dtype=torch.float32).flatten()
        mean_map = torch.as_tensor(mean_weights, dtype=torch.float32)
        lower_map = torch.as_tensor(lower_weights, dtype=torch.float32)
        harm_map = torch.as_tensor(harm_weights, dtype=torch.float32)
        expected = (len(mean) + 1, len(expert_ids))
        if logs.ndim != 4 or len(logs) != len(expert_ids):
            raise ValueError("expert ids and gains must align")
        if mean.shape != scale.shape or torch.any(scale <= 0.0):
            raise ValueError("feature normalization is invalid")
        if any(
            weights.shape != expected
            for weights in (mean_map, lower_map, harm_map)
        ):
            raise ValueError("risk heads do not align with the feature schema")
        self.expert_candidate_ids = expert_ids
        self.baseline_expert_index = int(baseline_expert_index)
        self.route_mode = str(route_mode)
        self.minimum_score = float(minimum_score)
        self.maximum_harm_probability = float(
            maximum_harm_probability
        )
        self.interpolation_fraction = float(interpolation_fraction)
        self.register_buffer("expert_log_gains", logs)
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_scale", scale)
        self.register_buffer("mean_weights", mean_map)
        self.register_buffer("lower_weights", lower_map)
        self.register_buffer("harm_weights", harm_map)

    def risk_scores(
        self,
        initial_normal: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
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
        return (
            design @ self.mean_weights.to(features),
            design @ self.lower_weights.to(features),
            design @ self.harm_weights.to(features),
            features,
        )

    def forward(
        self,
        initial_normal: torch.Tensor,
        **_: Any,
    ) -> MaterializedPositiveSpectralDirection:
        mean, lower, harm, _ = self.risk_scores(initial_normal)
        actions, _, _ = risk_quantile_actions(
            mean_scores=mean,
            lower_scores=lower,
            harm_logits=harm,
            baseline_expert_index=self.baseline_expert_index,
            route_mode=self.route_mode,
            minimum_score=self.minimum_score,
            maximum_harm_probability=self.maximum_harm_probability,
        )
        return materialize_single_expert_path(
            actions,
            expert_log_gains=self.expert_log_gains,
            baseline_expert_index=self.baseline_expert_index,
            interpolation_fraction=self.interpolation_fraction,
        )


class MultiObjectiveRiskSingleExpertFactory(nn.Module):
    """Risk-gate one expert using field and front-preservation heads."""

    def __init__(
        self,
        *,
        expert_log_gains: torch.Tensor,
        expert_candidate_ids: Sequence[str],
        baseline_expert_index: int,
        feature_mean: torch.Tensor,
        feature_scale: torch.Tensor,
        field_mean_weights: torch.Tensor,
        field_lower_weights: torch.Tensor,
        field_harm_weights: torch.Tensor,
        front_lower_weights: torch.Tensor,
        front_harm_weights: torch.Tensor,
        minimum_field_score: float,
        maximum_field_harm_probability: float,
        minimum_front_lower_delta: float,
        maximum_front_harm_probability: float,
        interpolation_fraction: float,
    ) -> None:
        super().__init__()
        expert_ids = tuple(str(value) for value in expert_candidate_ids)
        logs = torch.as_tensor(expert_log_gains, dtype=torch.float32)
        mean = torch.as_tensor(feature_mean, dtype=torch.float32).flatten()
        scale = torch.as_tensor(feature_scale, dtype=torch.float32).flatten()
        weight_tensors = tuple(
            torch.as_tensor(value, dtype=torch.float32)
            for value in (
                field_mean_weights,
                field_lower_weights,
                field_harm_weights,
                front_lower_weights,
                front_harm_weights,
            )
        )
        expected = (len(mean) + 1, len(expert_ids))
        if logs.ndim != 4 or len(logs) != len(expert_ids):
            raise ValueError("expert ids and gains must align")
        if mean.shape != scale.shape or torch.any(scale <= 0.0):
            raise ValueError("feature normalization is invalid")
        if any(weights.shape != expected for weights in weight_tensors):
            raise ValueError(
                "multiobjective heads do not align with the feature schema"
            )
        self.expert_candidate_ids = expert_ids
        self.baseline_expert_index = int(baseline_expert_index)
        self.minimum_field_score = float(minimum_field_score)
        self.maximum_field_harm_probability = float(
            maximum_field_harm_probability
        )
        self.minimum_front_lower_delta = float(
            minimum_front_lower_delta
        )
        self.maximum_front_harm_probability = float(
            maximum_front_harm_probability
        )
        self.interpolation_fraction = float(interpolation_fraction)
        self.register_buffer("expert_log_gains", logs)
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_scale", scale)
        for name, weights in zip(
            (
                "field_mean_weights",
                "field_lower_weights",
                "field_harm_weights",
                "front_lower_weights",
                "front_harm_weights",
            ),
            weight_tensors,
            strict=True,
        ):
            self.register_buffer(name, weights)

    def risk_scores(
        self,
        initial_normal: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
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
        return (
            design @ self.field_mean_weights.to(features),
            design @ self.field_lower_weights.to(features),
            design @ self.field_harm_weights.to(features),
            design @ self.front_lower_weights.to(features),
            design @ self.front_harm_weights.to(features),
            features,
        )

    def forward(
        self,
        initial_normal: torch.Tensor,
        **_: Any,
    ) -> MaterializedPositiveSpectralDirection:
        field_mean, field_lower, field_harm, front_lower, front_harm, _ = (
            self.risk_scores(initial_normal)
        )
        actions, _, _, _ = multiobjective_risk_actions(
            field_mean_scores=field_mean,
            field_lower_scores=field_lower,
            field_harm_logits=field_harm,
            front_lower_scores=front_lower,
            front_harm_logits=front_harm,
            baseline_expert_index=self.baseline_expert_index,
            minimum_field_score=self.minimum_field_score,
            maximum_field_harm_probability=(
                self.maximum_field_harm_probability
            ),
            minimum_front_lower_delta=self.minimum_front_lower_delta,
            maximum_front_harm_probability=(
                self.maximum_front_harm_probability
            ),
        )
        return materialize_single_expert_path(
            actions,
            expert_log_gains=self.expert_log_gains,
            baseline_expert_index=self.baseline_expert_index,
            interpolation_fraction=self.interpolation_fraction,
        )


__all__ = [
    "MultiObjectiveRiskSingleExpertFactory",
    "RISK_QUANTILE_EXPERT_SCHEMA",
    "RiskQuantileSingleExpertFactory",
    "materialize_single_expert_path",
    "multiobjective_risk_actions",
    "risk_quantile_actions",
]
