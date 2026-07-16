"""Call-accounted quadratic classical baselines for the PSU B0 inverse.

The solvers in this module use the same masked, whitened data objective as the
spectral pilot.  Each stage evaluates one exact adjoint and one exact forward
projection.  The only extra work is a local quadratic regularizer, so operator
calls remain directly comparable with the four-stage learned direction.
"""

from __future__ import annotations

from math import ceil
from typing import Any, Literal, Sequence

import torch
from torch import nn

from .psu_b0_reconstruction_interface import (
    finite_difference_gradient,
    finite_difference_gradient_adjoint,
)
from .psu_b0_spectral_preconditioner import (
    IterativeReconstruction,
    SearchDirection,
)


CLASSICAL_BASELINE_SCHEMA = "psu-b0-call-accounted-classical-baselines-1.0"
QuadraticRegularizer = Literal["identity", "h1"]


def _frequency_radius(
    grid_shape: tuple[int, int, int],
    *,
    axis_weights_xyz: tuple[float, float, float],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    nz, ny, nx = (int(value) for value in grid_shape)
    if min(nz, ny, nx) < 4:
        raise ValueError("grid_shape must contain dimensions of at least four")
    weights = tuple(float(value) for value in axis_weights_xyz)
    if len(weights) != 3 or any(value <= 0.0 for value in weights):
        raise ValueError("axis_weights_xyz must contain three positive values")
    fz = torch.fft.fftfreq(nz, dtype=dtype)
    fy = torch.fft.fftfreq(ny, dtype=dtype)
    fx = torch.fft.rfftfreq(nx, dtype=dtype)
    zz, yy, xx = torch.meshgrid(fz, fy, fx, indexing="ij")
    scale = torch.as_tensor(0.5, dtype=dtype)
    return (
        weights[0] * (xx / scale).square()
        + weights[1] * (yy / scale).square()
        + weights[2] * (zz / scale).square()
    )


class GeneralizedSobolevDirection(nn.Module):
    """Validation-tunable static anisotropic inverse-Sobolev direction."""

    def __init__(
        self,
        grid_shape: tuple[int, int, int],
        *,
        strength: float,
        epsilon: float = 0.05,
        axis_weights_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        super().__init__()
        if float(strength) < 0.0:
            raise ValueError("strength must be nonnegative")
        if float(epsilon) <= 0.0:
            raise ValueError("epsilon must be positive")
        radius = _frequency_radius(
            grid_shape,
            axis_weights_xyz=axis_weights_xyz,
        )
        gain = (float(epsilon) + radius).pow(-float(strength))
        geometric = torch.exp(torch.mean(torch.log(gain.clamp_min(1e-8))))
        self.strength = float(strength)
        self.epsilon = float(epsilon)
        self.axis_weights_xyz = tuple(float(value) for value in axis_weights_xyz)
        self.register_buffer("gain", gain / geometric)

    def forward(
        self,
        gradient: torch.Tensor,
        **_: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        gain = self.gain.to(gradient)
        spectrum = torch.fft.rfftn(gradient, dim=(-3, -2, -1))
        direction = torch.fft.irfftn(
            spectrum * gain[None, None],
            s=gradient.shape[-3:],
            dim=(-3, -2, -1),
        )
        batch = len(gradient)
        return direction, {
            "gain_minimum": gain.amin().expand(batch),
            "gain_maximum": gain.amax().expand(batch),
            "gain_geometric_mean": torch.exp(
                torch.mean(torch.log(gain))
            ).expand(batch),
            "sobolev_strength": torch.full(
                (batch,),
                self.strength,
                dtype=gradient.dtype,
                device=gradient.device,
            ),
        }


class ScheduledSobolevDirection(nn.Module):
    """Use one preregistered static Sobolev strength at each solver stage."""

    def __init__(
        self,
        grid_shape: tuple[int, int, int],
        *,
        strengths: Sequence[float],
        epsilon: float = 0.05,
    ) -> None:
        super().__init__()
        values = tuple(float(value) for value in strengths)
        if not values:
            raise ValueError("strengths must be nonempty")
        self.strengths = values
        self.directions = nn.ModuleList(
            [
                GeneralizedSobolevDirection(
                    grid_shape,
                    strength=value,
                    epsilon=float(epsilon),
                )
                for value in values
            ]
        )

    def forward(
        self,
        gradient: torch.Tensor,
        *,
        stage_fraction: float,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        fraction = float(stage_fraction)
        if not 0.0 < fraction <= 1.0 + 1e-8:
            raise ValueError("stage_fraction must lie in (0,1]")
        index = min(max(ceil(fraction * len(self.directions)) - 1, 0), len(self.directions) - 1)
        direction, diagnostics = self.directions[index](
            gradient,
            stage_fraction=stage_fraction,
            **kwargs,
        )
        diagnostics["sobolev_schedule_index"] = torch.full(
            (len(gradient),),
            index,
            dtype=gradient.dtype,
            device=gradient.device,
        )
        return direction, diagnostics


def _view_expansion(
    values: torch.Tensor,
    *,
    rays_per_view: int,
    ray_count: int,
) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError("view values must have shape [batch,view]")
    expanded = values.repeat_interleave(int(rays_per_view), dim=1)
    if expanded.shape[1] != int(ray_count):
        raise ValueError("view count and rays_per_view do not cover the operator")
    return expanded[:, :, None]


def _weighted_measurement_terms(
    observation_uv: torch.Tensor,
    *,
    sigma_by_view: torch.Tensor,
    view_mask: torch.Tensor,
    rays_per_view: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if observation_uv.ndim != 3 or observation_uv.shape[-1] != 2:
        raise ValueError("observation_uv must have shape [batch,ray,2]")
    batch = len(observation_uv)
    if sigma_by_view.shape != view_mask.shape or sigma_by_view.shape[0] != batch:
        raise ValueError("sigma_by_view and view_mask must align with observations")
    if torch.any(sigma_by_view <= 0.0):
        raise ValueError("sigma_by_view must be strictly positive")
    if torch.any((view_mask < 0.0) | (view_mask > 1.0)):
        raise ValueError("view_mask must lie in [0,1]")
    if torch.any(torch.sum(view_mask > 0.5, dim=1) < 1):
        raise ValueError("each sample needs at least one active view")
    active = _view_expansion(
        view_mask.to(observation_uv),
        rays_per_view=rays_per_view,
        ray_count=observation_uv.shape[1],
    )
    sigma = _view_expansion(
        sigma_by_view.to(observation_uv),
        rays_per_view=rays_per_view,
        ray_count=observation_uv.shape[1],
    )
    return active, sigma


def _regularizer_terms(
    volume: torch.Tensor,
    *,
    regularizer: QuadraticRegularizer,
    spacing_xyz: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``R x`` and the per-sample quadratic energy ``||L x||^2``."""

    if regularizer == "identity":
        energy = torch.sum(volume.square(), dim=(1, 2, 3, 4))
        return volume, energy
    if regularizer == "h1":
        gradient = finite_difference_gradient(
            volume[:, 0],
            spacing_xyz=spacing_xyz,
        )
        normal = finite_difference_gradient_adjoint(
            gradient,
            spacing_xyz=spacing_xyz,
        )[:, None]
        energy = torch.sum(gradient.square(), dim=(1, 2, 3, 4))
        return normal, energy
    raise ValueError("regularizer must be 'identity' or 'h1'")


def quadratic_tikhonov_reconstruction(
    operator: Any,
    observation_uv: torch.Tensor,
    *,
    sigma_by_view: torch.Tensor,
    view_mask: torch.Tensor,
    rays_per_view: int,
    stages: int,
    regularization_lambda: float,
    regularizer: QuadraticRegularizer = "h1",
    denominator_floor: float = 1e-20,
) -> IterativeReconstruction:
    """Run exact-line steepest descent on a quadratic Tikhonov objective.

    The minimized objective is

    ``0.5 * ||M(Ax-y)/sigma||^2 + 0.5 * lambda * ||Lx||^2``.

    ``L`` is either the identity or the declared voxel-centered 3-D gradient.
    The search is restricted to the operator support.  Because the objective is
    quadratic, every stage has an analytic line minimizer and must not increase
    the total objective apart from floating-point roundoff.
    """

    count = int(stages)
    weight = float(regularization_lambda)
    if count < 1:
        raise ValueError("stages must be positive")
    if not torch.isfinite(torch.as_tensor(weight)) or weight < 0.0:
        raise ValueError("regularization_lambda must be finite and nonnegative")
    active, sigma = _weighted_measurement_terms(
        observation_uv,
        sigma_by_view=sigma_by_view,
        view_mask=view_mask,
        rays_per_view=rays_per_view,
    )
    support = operator.support[None, None].to(observation_uv)
    spacing_xyz = tuple(float(value) for value in operator.spacing_xyz)
    current = torch.zeros(
        (len(observation_uv), 1, *operator.grid_shape),
        dtype=observation_uv.dtype,
        device=observation_uv.device,
    )
    residual = active * observation_uv
    initial_data = torch.sum(
        (residual / sigma).square(),
        dim=(1, 2),
    ).clamp_min(float(denominator_floor))
    history: list[dict[str, torch.Tensor]] = []
    for stage in range(count):
        data_before = torch.sum(
            (residual / sigma).square(),
            dim=(1, 2),
        )
        regularizer_normal, regularizer_before = _regularizer_terms(
            current,
            regularizer=regularizer,
            spacing_xyz=spacing_xyz,
        )
        data_negative_gradient = operator.adjoint(residual / sigma.square())
        search = (
            data_negative_gradient - weight * regularizer_normal
        ) * support
        projected = active * operator(search)
        _, search_regularizer_energy = _regularizer_terms(
            search,
            regularizer=regularizer,
            spacing_xyz=spacing_xyz,
        )
        numerator = torch.sum(search.square(), dim=(1, 2, 3, 4))
        denominator = (
            torch.sum((projected / sigma).square(), dim=(1, 2))
            + weight * search_regularizer_energy
        ).clamp_min(float(denominator_floor))
        alpha = torch.clamp_min(numerator / denominator, 0.0)
        current = current + alpha[:, None, None, None, None] * search
        residual = residual - alpha[:, None, None] * projected
        data_after = torch.sum(
            (residual / sigma).square(),
            dim=(1, 2),
        )
        _, regularizer_after = _regularizer_terms(
            current,
            regularizer=regularizer,
            spacing_xyz=spacing_xyz,
        )
        total_before = data_before + weight * regularizer_before
        total_after = data_after + weight * regularizer_after
        history.append(
            {
                "stage": torch.full_like(alpha, stage + 1, dtype=torch.int64),
                "alpha": alpha,
                "negative_gradient_norm_squared": numerator,
                "relative_data_objective_before": data_before / initial_data,
                "relative_data_objective_after": data_after / initial_data,
                "relative_regularizer_objective_before": (
                    weight * regularizer_before / initial_data
                ),
                "relative_regularizer_objective_after": (
                    weight * regularizer_after / initial_data
                ),
                "relative_total_objective_before": total_before / initial_data,
                "relative_total_objective_after": total_after / initial_data,
            }
        )
    return IterativeReconstruction(
        volume=current,
        residual_uv=residual,
        history=history,
        forward_calls=count,
        adjoint_calls=count,
    )


def preconditioned_cgls_reconstruction(
    operator: Any,
    observation_uv: torch.Tensor,
    *,
    sigma_by_view: torch.Tensor,
    view_mask: torch.Tensor,
    rays_per_view: int,
    stages: int,
    preconditioner: SearchDirection | None = None,
    preconditioner_factory: Any | None = None,
    denominator_floor: float = 1e-20,
) -> IterativeReconstruction:
    """Run fixed-depth fixed-SPD PCGLS on the whitened data objective.

    This is preconditioned conjugate gradient on the normal equations.  A
    stage-dependent or residual-adaptive preconditioner would break the usual
    conjugacy argument, so confirmatory use should pass one fixed SPD map such
    as :class:`GeneralizedSobolevDirection`. ``preconditioner_factory`` may use
    the already-required first normal residual to materialize one fixed map;
    this shares the first adjoint instead of adding a hidden operator call.
    The final residual normal is not evaluated because no subsequent direction
    is needed. A fixed ``K``-stage reconstruction therefore uses exactly ``K``
    forward and ``K`` adjoint operator calls.
    """

    count = int(stages)
    return _preconditioned_cgls_trajectory(
        operator,
        observation_uv,
        sigma_by_view=sigma_by_view,
        view_mask=view_mask,
        rays_per_view=rays_per_view,
        stages=count,
        checkpoint_stages=(count,),
        preconditioner=preconditioner,
        preconditioner_factory=preconditioner_factory,
        denominator_floor=denominator_floor,
    )[count]


def preconditioned_cgls_trajectory(
    operator: Any,
    observation_uv: torch.Tensor,
    *,
    sigma_by_view: torch.Tensor,
    view_mask: torch.Tensor,
    rays_per_view: int,
    checkpoint_stages: Sequence[int],
    preconditioner: SearchDirection,
    denominator_floor: float = 1e-20,
) -> dict[int, IterativeReconstruction]:
    """Return selected iterates from one fixed-preconditioner PCGLS solve.

    This execution helper removes duplicated prefix work when several fixed
    iteration counts are compared. Each returned reconstruction retains its
    logical call budget, while the wrapped operator performs only the maximum
    checkpoint count. The intermediate volumes are identical to independent
    solves when the supplied fixed preconditioner is stage-fraction invariant,
    as are the static Sobolev maps used by the covariance diagnosis.
    """

    checkpoints = tuple(sorted({int(value) for value in checkpoint_stages}))
    if not checkpoints or checkpoints[0] < 1:
        raise ValueError("checkpoint_stages must contain positive integers")
    return _preconditioned_cgls_trajectory(
        operator,
        observation_uv,
        sigma_by_view=sigma_by_view,
        view_mask=view_mask,
        rays_per_view=rays_per_view,
        stages=checkpoints[-1],
        checkpoint_stages=checkpoints,
        preconditioner=preconditioner,
        preconditioner_factory=None,
        denominator_floor=denominator_floor,
    )


def _preconditioned_cgls_trajectory(
    operator: Any,
    observation_uv: torch.Tensor,
    *,
    sigma_by_view: torch.Tensor,
    view_mask: torch.Tensor,
    rays_per_view: int,
    stages: int,
    checkpoint_stages: Sequence[int],
    preconditioner: SearchDirection | None,
    preconditioner_factory: Any | None,
    denominator_floor: float,
) -> dict[int, IterativeReconstruction]:
    count = int(stages)
    if count < 1:
        raise ValueError("stages must be positive")
    checkpoints = tuple(sorted({int(value) for value in checkpoint_stages}))
    if (
        not checkpoints
        or checkpoints[0] < 1
        or checkpoints[-1] != count
    ):
        raise ValueError(
            "checkpoint_stages must be positive and include stages"
        )
    if (preconditioner is None) == (preconditioner_factory is None):
        raise ValueError(
            "provide exactly one preconditioner or preconditioner_factory"
        )
    active, sigma = _weighted_measurement_terms(
        observation_uv,
        sigma_by_view=sigma_by_view,
        view_mask=view_mask,
        rays_per_view=rays_per_view,
    )
    support = operator.support[None, None].to(observation_uv)
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
    residual_uv = residual_white * sigma
    if preconditioner_factory is not None:
        fixed_preconditioner = preconditioner_factory(
            normal,
            observation_uv=observation_uv,
            residual_uv=residual_uv,
            sigma_by_view=sigma_by_view,
            view_mask=view_mask,
            rays_per_view=rays_per_view,
        )
    else:
        fixed_preconditioner = preconditioner
    if fixed_preconditioner is None:
        raise RuntimeError("preconditioner materialization failed")
    preconditioned, diagnostics = fixed_preconditioner(
        normal,
        residual_uv=residual_uv,
        sigma_by_view=sigma_by_view,
        view_mask=view_mask,
        rays_per_view=rays_per_view,
        stage_fraction=1.0 / count,
    )
    preconditioned = preconditioned * support
    direction = preconditioned.clone()
    gamma = torch.sum(normal * preconditioned, dim=(1, 2, 3, 4))
    if torch.any(gamma < -float(denominator_floor)):
        raise ValueError("preconditioner is not positive on the initial normal")
    history: list[dict[str, torch.Tensor]] = []
    outputs: dict[int, IterativeReconstruction] = {}
    for stage in range(count):
        projected_white = active * operator(direction) / sigma
        denominator = torch.sum(
            projected_white.square(),
            dim=(1, 2),
        ).clamp_min(float(denominator_floor))
        alpha = gamma / denominator
        objective_before = torch.sum(
            residual_white.square(),
            dim=(1, 2),
        ) / initial_objective
        current = current + alpha[:, None, None, None, None] * direction
        residual_white = residual_white - alpha[:, None, None] * projected_white
        objective_after = torch.sum(
            residual_white.square(),
            dim=(1, 2),
        ) / initial_objective
        completed = stage + 1
        terminal_row = {
            "stage": torch.full_like(
                alpha,
                completed,
                dtype=torch.int64,
            ),
            "alpha": alpha,
            "relative_objective_before": objective_before,
            "relative_objective_after": objective_after,
            **diagnostics,
        }
        if completed in checkpoints:
            outputs[completed] = IterativeReconstruction(
                volume=current.clone(),
                residual_uv=(residual_white * sigma).clone(),
                history=[*history, terminal_row],
                forward_calls=completed,
                adjoint_calls=completed,
            )
        if completed == count:
            history.append(terminal_row)
            break
        next_normal = operator.adjoint(active * residual_white / sigma)
        next_residual_uv = residual_white * sigma
        next_preconditioned, next_diagnostics = fixed_preconditioner(
            next_normal,
            residual_uv=next_residual_uv,
            sigma_by_view=sigma_by_view,
            view_mask=view_mask,
            rays_per_view=rays_per_view,
            stage_fraction=min((stage + 2) / count, 1.0),
        )
        next_preconditioned = next_preconditioned * support
        next_gamma = torch.sum(
            next_normal * next_preconditioned,
            dim=(1, 2, 3, 4),
        )
        if torch.any(next_gamma < -float(denominator_floor)):
            raise ValueError("preconditioner lost positive definiteness")
        beta = next_gamma / gamma.clamp_min(float(denominator_floor))
        direction = (
            next_preconditioned
            + beta[:, None, None, None, None] * direction
        )
        gamma = next_gamma
        history.append(
            {
                **terminal_row,
                "beta": beta,
            }
        )
        diagnostics = next_diagnostics
    if set(outputs) != set(checkpoints):
        raise RuntimeError("PCGLS trajectory did not emit every checkpoint")
    return outputs


__all__ = [
    "CLASSICAL_BASELINE_SCHEMA",
    "GeneralizedSobolevDirection",
    "ScheduledSobolevDirection",
    "preconditioned_cgls_reconstruction",
    "preconditioned_cgls_trajectory",
    "quadratic_tikhonov_reconstruction",
]
