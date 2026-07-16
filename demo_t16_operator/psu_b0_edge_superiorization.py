"""Edge-preserving superiorization for matrix-free PSU B0 reconstruction.

This module implements a three-dimensional analogue of the bounded
nonascending perturbations used by superiorized PCG.  The data step remains a
fixed-SPD preconditioned conjugate-gradient update on the whitened normal
equations.  Perturbing the iterate invalidates the cheap residual recurrence,
so the implementation explicitly records the extra forward projection needed
to rebuild the residual.  No operator call is hidden inside the regularizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

from .psu_b0_classical_baselines import _weighted_measurement_terms
from .psu_b0_reconstruction_interface import (
    finite_difference_gradient,
    finite_difference_gradient_adjoint,
)
from .psu_b0_spectral_preconditioner import (
    IterativeReconstruction,
    SearchDirection,
)


EDGE_SUPERIORIZATION_SCHEMA = "psu-b0-edge-superiorization-1.0"
EdgePenalty = Literal["tv", "huber"]


@dataclass(frozen=True)
class EdgePerturbation:
    """One batch of accepted nonascending edge perturbations."""

    volume: torch.Tensor
    exponent: torch.Tensor
    penalty_before: torch.Tensor
    penalty_after: torch.Tensor
    perturbation_norm: torch.Tensor
    trial_count: torch.Tensor


def edge_penalty_and_gradient(
    volume: torch.Tensor,
    *,
    spacing_xyz: tuple[float, float, float],
    penalty: EdgePenalty,
    smoothing: float = 1e-3,
    huber_delta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return an isotropic 3-D edge penalty and its analytic gradient.

    ``tv`` uses a differentiable ``sqrt(|D x|^2 + smoothing^2)`` surrogate.
    ``huber`` applies the scalar Huber function to the same isotropic gradient
    magnitude.  The additive smoothing constant changes the penalty offset but
    not the constant-field gradient.
    """

    if volume.ndim != 5 or volume.shape[1] != 1:
        raise ValueError("volume must have shape [batch,1,z,y,x]")
    epsilon = float(smoothing)
    delta = float(huber_delta)
    if not torch.isfinite(torch.as_tensor(epsilon)) or epsilon <= 0.0:
        raise ValueError("smoothing must be finite and positive")
    if not torch.isfinite(torch.as_tensor(delta)) or delta <= 0.0:
        raise ValueError("huber_delta must be finite and positive")
    gradient = finite_difference_gradient(
        volume[:, 0],
        spacing_xyz=spacing_xyz,
    )
    magnitude = torch.sqrt(
        torch.sum(gradient.square(), dim=1) + epsilon**2
    )
    if penalty == "tv":
        values = magnitude
        multiplier = magnitude.reciprocal()
    elif penalty == "huber":
        quadratic = magnitude <= delta
        values = torch.where(
            quadratic,
            0.5 * magnitude.square() / delta,
            magnitude - 0.5 * delta,
        )
        multiplier = torch.where(
            quadratic,
            torch.full_like(magnitude, 1.0 / delta),
            magnitude.reciprocal(),
        )
    else:
        raise ValueError("penalty must be 'tv' or 'huber'")
    derivative = finite_difference_gradient_adjoint(
        multiplier[:, None] * gradient,
        spacing_xyz=spacing_xyz,
    )[:, None]
    return torch.sum(values, dim=(1, 2, 3)), derivative


def nonascending_edge_perturbation(
    volume: torch.Tensor,
    *,
    support: torch.Tensor,
    spacing_xyz: tuple[float, float, float],
    penalty: EdgePenalty,
    exponent: torch.Tensor,
    inner_steps: int,
    initial_step: float,
    decay: float,
    smoothing: float = 1e-3,
    huber_delta: float = 0.1,
    maximum_backtracks: int = 64,
    acceptance_tolerance: float = 1e-10,
) -> EdgePerturbation:
    """Apply summable per-sample perturbations that do not raise the penalty."""

    if volume.ndim != 5 or volume.shape[1] != 1:
        raise ValueError("volume must have shape [batch,1,z,y,x]")
    if support.shape != volume.shape[-3:]:
        raise ValueError("support must match the spatial volume shape")
    steps = int(inner_steps)
    if steps < 0:
        raise ValueError("inner_steps must be nonnegative")
    gamma = float(initial_step)
    shrink = float(decay)
    if not torch.isfinite(torch.as_tensor(gamma)) or gamma < 0.0:
        raise ValueError("initial_step must be finite and nonnegative")
    if not 0.0 < shrink < 1.0:
        raise ValueError("decay must lie strictly between zero and one")
    if int(maximum_backtracks) < 1:
        raise ValueError("maximum_backtracks must be positive")
    current_exponent = torch.as_tensor(
        exponent,
        dtype=torch.int64,
        device=volume.device,
    ).reshape(-1)
    if current_exponent.shape != (len(volume),):
        raise ValueError("exponent must contain one value per sample")
    if torch.any(current_exponent < 0):
        raise ValueError("exponent must be nonnegative")

    mask = support.to(volume)[None, None]
    current = volume * mask
    penalty_before, _ = edge_penalty_and_gradient(
        current,
        spacing_xyz=spacing_xyz,
        penalty=penalty,
        smoothing=smoothing,
        huber_delta=huber_delta,
    )
    if steps == 0 or gamma == 0.0:
        zero = torch.zeros(
            len(volume),
            dtype=volume.dtype,
            device=volume.device,
        )
        return EdgePerturbation(
            volume=current,
            exponent=current_exponent,
            penalty_before=penalty_before,
            penalty_after=penalty_before,
            perturbation_norm=zero,
            trial_count=zero.to(torch.int64),
        )

    accepted_volume = current
    accepted_penalty = penalty_before
    trial_count = torch.zeros_like(current_exponent)
    for _ in range(steps):
        _, derivative = edge_penalty_and_gradient(
            accepted_volume,
            spacing_xyz=spacing_xyz,
            penalty=penalty,
            smoothing=smoothing,
            huber_delta=huber_delta,
        )
        direction = -derivative * mask
        direction_norm = torch.linalg.vector_norm(
            direction.flatten(1),
            dim=1,
        )
        direction = direction / direction_norm.clamp_min(1e-20)[
            :, None, None, None, None
        ]
        direction = torch.where(
            (direction_norm > 0.0)[:, None, None, None, None],
            direction,
            torch.zeros_like(direction),
        )
        pending = torch.ones(
            len(volume),
            dtype=torch.bool,
            device=volume.device,
        )
        next_volume = accepted_volume.clone()
        next_penalty = accepted_penalty.clone()
        for _ in range(int(maximum_backtracks)):
            if not bool(torch.any(pending)):
                break
            step_length = gamma * torch.pow(
                torch.full(
                    (len(volume),),
                    shrink,
                    dtype=volume.dtype,
                    device=volume.device,
                ),
                current_exponent.to(volume.dtype),
            )
            proposal = (
                accepted_volume
                + step_length[:, None, None, None, None] * direction
            ) * mask
            proposal_penalty, _ = edge_penalty_and_gradient(
                proposal,
                spacing_xyz=spacing_xyz,
                penalty=penalty,
                smoothing=smoothing,
                huber_delta=huber_delta,
            )
            accepted = pending & (
                proposal_penalty
                <= accepted_penalty + float(acceptance_tolerance)
            )
            next_volume = torch.where(
                accepted[:, None, None, None, None],
                proposal,
                next_volume,
            )
            next_penalty = torch.where(
                accepted,
                proposal_penalty,
                next_penalty,
            )
            trial_count = trial_count + pending.to(torch.int64)
            current_exponent = current_exponent + pending.to(torch.int64)
            pending = pending & ~accepted
        if bool(torch.any(pending)):
            raise RuntimeError(
                "edge perturbation failed to find a nonascending step"
            )
        accepted_volume = next_volume
        accepted_penalty = next_penalty

    perturbation_norm = torch.linalg.vector_norm(
        (accepted_volume - current).flatten(1),
        dim=1,
    )
    return EdgePerturbation(
        volume=accepted_volume,
        exponent=current_exponent,
        penalty_before=penalty_before,
        penalty_after=accepted_penalty,
        perturbation_norm=perturbation_norm,
        trial_count=trial_count,
    )


def superiorized_pcgls_reconstruction(
    operator: Any,
    observation_uv: torch.Tensor,
    *,
    sigma_by_view: torch.Tensor,
    view_mask: torch.Tensor,
    rays_per_view: int,
    stages: int,
    preconditioner: SearchDirection,
    penalty: EdgePenalty,
    perturbation_steps: int,
    perturbation_initial_step: float,
    perturbation_decay: float,
    smoothing: float = 1e-3,
    huber_delta: float = 0.1,
    maximum_backtracks: int = 64,
    denominator_floor: float = 1e-20,
) -> IterativeReconstruction:
    """Run fixed-SPD PCGLS with explicit TV/Huber superiorization.

    The first PCGLS update is unperturbed, matching the published SupPCG
    ordering.  Before every later update, a bounded nonascending perturbation
    is applied.  Rebuilding the perturbed residual costs one extra forward
    projection; the call report is therefore ``2*K-1`` forward and ``K``
    adjoint calls for ``K`` stages.
    """

    count = int(stages)
    if count < 1:
        raise ValueError("stages must be positive")
    active, sigma = _weighted_measurement_terms(
        observation_uv,
        sigma_by_view=sigma_by_view,
        view_mask=view_mask,
        rays_per_view=rays_per_view,
    )
    support = operator.support[None, None].to(observation_uv)
    support_3d = operator.support.to(observation_uv)
    spacing_xyz = tuple(float(value) for value in operator.spacing_xyz)
    current = torch.zeros(
        (len(observation_uv), 1, *operator.grid_shape),
        dtype=observation_uv.dtype,
        device=observation_uv.device,
    )
    residual_white = active * observation_uv / sigma
    initial_objective = torch.sum(
        residual_white.square(),
        dim=(1, 2),
    ).clamp_min(float(denominator_floor))
    normal = operator.adjoint(active * residual_white / sigma)
    preconditioned, diagnostics = preconditioner(
        normal,
        residual_uv=residual_white * sigma,
        sigma_by_view=sigma_by_view,
        view_mask=view_mask,
        rays_per_view=rays_per_view,
        stage_fraction=1.0 / count,
    )
    preconditioned = preconditioned * support
    direction = preconditioned
    projected_white = active * operator(direction) / sigma
    denominator = torch.sum(
        projected_white.square(),
        dim=(1, 2),
    ).clamp_min(float(denominator_floor))
    alpha = torch.sum(
        residual_white * projected_white,
        dim=(1, 2),
    ) / denominator
    objective_before = torch.sum(
        residual_white.square(),
        dim=(1, 2),
    ) / initial_objective
    current = current + alpha[:, None, None, None, None] * direction
    residual_white = (
        residual_white - alpha[:, None, None] * projected_white
    )
    objective_after = torch.sum(
        residual_white.square(),
        dim=(1, 2),
    ) / initial_objective
    penalty_value, _ = edge_penalty_and_gradient(
        current,
        spacing_xyz=spacing_xyz,
        penalty=penalty,
        smoothing=smoothing,
        huber_delta=huber_delta,
    )
    zeros = torch.zeros_like(alpha)
    history: list[dict[str, torch.Tensor]] = [
        {
            "stage": torch.ones_like(alpha, dtype=torch.int64),
            "alpha": alpha,
            "beta": zeros,
            "relative_objective_before": objective_before,
            "relative_objective_after": objective_after,
            "edge_penalty_before": penalty_value,
            "edge_penalty_after": penalty_value,
            "perturbation_norm": zeros,
            "perturbation_trials": zeros.to(torch.int64),
            "perturbation_exponent": zeros.to(torch.int64),
            **diagnostics,
        }
    ]
    exponent = torch.zeros(
        len(observation_uv),
        dtype=torch.int64,
        device=observation_uv.device,
    )

    for stage in range(1, count):
        perturbation = nonascending_edge_perturbation(
            current,
            support=support_3d,
            spacing_xyz=spacing_xyz,
            penalty=penalty,
            exponent=exponent,
            inner_steps=int(perturbation_steps),
            initial_step=float(perturbation_initial_step),
            decay=float(perturbation_decay),
            smoothing=float(smoothing),
            huber_delta=float(huber_delta),
            maximum_backtracks=int(maximum_backtracks),
        )
        exponent = perturbation.exponent
        superiorized = perturbation.volume
        perturbation_volume = superiorized - current
        perturbation_projected_white = (
            active * operator(perturbation_volume) / sigma
        )
        residual_half = (
            residual_white - perturbation_projected_white
        )
        normal = operator.adjoint(active * residual_half / sigma)
        next_preconditioned, diagnostics = preconditioner(
            normal,
            residual_uv=residual_half * sigma,
            sigma_by_view=sigma_by_view,
            view_mask=view_mask,
            rays_per_view=rays_per_view,
            stage_fraction=(stage + 1) / count,
        )
        next_preconditioned = next_preconditioned * support
        next_projected_white = (
            active * operator(next_preconditioned) / sigma
        )
        previous_denominator = torch.sum(
            projected_white.square(),
            dim=(1, 2),
        ).clamp_min(float(denominator_floor))
        beta = -torch.sum(
            next_projected_white * projected_white,
            dim=(1, 2),
        ) / previous_denominator
        direction = (
            next_preconditioned
            + beta[:, None, None, None, None] * direction
        )
        projected_white = (
            next_projected_white
            + beta[:, None, None] * projected_white
        )
        denominator = torch.sum(
            projected_white.square(),
            dim=(1, 2),
        ).clamp_min(float(denominator_floor))
        alpha = torch.sum(
            residual_half * projected_white,
            dim=(1, 2),
        ) / denominator
        objective_before = torch.sum(
            residual_half.square(),
            dim=(1, 2),
        ) / initial_objective
        current = (
            superiorized
            + alpha[:, None, None, None, None] * direction
        ) * support
        residual_white = (
            residual_half - alpha[:, None, None] * projected_white
        )
        objective_after = torch.sum(
            residual_white.square(),
            dim=(1, 2),
        ) / initial_objective
        edge_after_update, _ = edge_penalty_and_gradient(
            current,
            spacing_xyz=spacing_xyz,
            penalty=penalty,
            smoothing=smoothing,
            huber_delta=huber_delta,
        )
        history.append(
            {
                "stage": torch.full_like(
                    alpha,
                    stage + 1,
                    dtype=torch.int64,
                ),
                "alpha": alpha,
                "beta": beta,
                "relative_objective_before": objective_before,
                "relative_objective_after": objective_after,
                "edge_penalty_before": perturbation.penalty_before,
                "edge_penalty_after_perturbation": (
                    perturbation.penalty_after
                ),
                "edge_penalty_after": edge_after_update,
                "perturbation_norm": perturbation.perturbation_norm,
                "perturbation_trials": perturbation.trial_count,
                "perturbation_exponent": exponent.clone(),
                **diagnostics,
            }
        )

    return IterativeReconstruction(
        volume=current,
        residual_uv=residual_white * sigma,
        history=history,
        forward_calls=2 * count - 1,
        adjoint_calls=count,
    )


__all__ = [
    "EDGE_SUPERIORIZATION_SCHEMA",
    "EdgePerturbation",
    "edge_penalty_and_gradient",
    "nonascending_edge_perturbation",
    "superiorized_pcgls_reconstruction",
]
