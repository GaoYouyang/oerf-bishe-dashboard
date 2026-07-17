"""Strict matrix-free baselines for a single 3-D BOST reconstruction.

The two public solvers deliberately expose the physical ``forward`` and
``adjoint`` maps instead of depending on a data-set-specific operator class.
They share three comparison rules:

* the reconstruction API has no reference-field input;
* ``iterations=K`` performs exactly ``K`` forward and ``K`` adjoint calls;
* histories use quantities already available inside the recurrence, so no
  uncounted terminal projection is used for reporting.

``edge_preserving_pdhg_baseline`` follows the forward-Neumann gradient and
Huber/TV conjugate proximal conventions in :mod:`psu_b0_primal_dual`.
The supplied data-operator norm bound is setup metadata: estimating or tuning
that bound is outside the reported solve budget and must be accounted for by
the caller when methods are compared.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
from typing import Literal, TypeAlias

import torch

from .psu_b0_primal_dual import (
    gradient_operator_norm_squared_bound,
    proximal_edge_conjugate,
    regularization_gradient,
    regularization_gradient_adjoint,
)


Tensor = torch.Tensor
LinearMap: TypeAlias = Callable[[Tensor], Tensor]
HistoryValue: TypeAlias = float | int | bool | str | None
HistoryRow: TypeAlias = dict[str, HistoryValue]
EdgePenalty = Literal["tv", "huber"]


@dataclass(frozen=True)
class InterfaceBaselineResult:
    """A reconstruction and its complete physical-operator call ledger."""

    field: Tensor
    history: list[HistoryRow]
    forward_calls: int
    adjoint_calls: int


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if parsed < 1 or parsed != value:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _finite_scalar(
    value: float,
    *,
    name: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _validated_spacing(spacing_xyz: Sequence[float]) -> tuple[float, float, float]:
    try:
        spacing = tuple(float(value) for value in spacing_xyz)
    except (TypeError, ValueError) as error:
        raise ValueError("spacing_xyz must contain three positive finite values") from error
    if len(spacing) != 3 or any(
        not math.isfinite(value) or value <= 0.0 for value in spacing
    ):
        raise ValueError("spacing_xyz must contain three positive finite values")
    return spacing[0], spacing[1], spacing[2]


def _validated_observation(observation: Tensor) -> Tensor:
    if not isinstance(observation, torch.Tensor):
        raise TypeError("observation must be a torch.Tensor")
    if not observation.is_floating_point() or observation.is_complex():
        raise TypeError("observation must have a real floating dtype")
    if observation.numel() == 0:
        raise ValueError("observation must be nonempty")
    if not bool(torch.all(torch.isfinite(observation))):
        raise ValueError("observation must contain only finite values")
    return observation


def _validated_support(support: Tensor, *, reference: Tensor) -> Tensor:
    if not isinstance(support, torch.Tensor):
        raise TypeError("support must be a torch.Tensor")
    if support.ndim != 3 or support.numel() == 0:
        raise ValueError("support must have shape [z,y,x]")
    values = support.to(device=reference.device)
    if values.is_complex():
        raise TypeError("support must be real")
    if values.dtype != torch.bool:
        if not bool(torch.all(torch.isfinite(values))):
            raise ValueError("support must contain only finite values")
        floating = values.to(dtype=reference.dtype)
        if not bool(torch.all((floating == 0.0) | (floating == 1.0))):
            raise ValueError("support must be binary")
        mask = floating
    else:
        mask = values.to(dtype=reference.dtype)
    if not bool(torch.any(mask > 0.5)):
        raise ValueError("support must retain at least one voxel")
    return mask


def _dot(left: Tensor, right: Tensor) -> Tensor:
    return torch.sum(left * right)


class _CountedLinearOperator:
    """Validate a matrix-free pair and count every physical invocation."""

    def __init__(
        self,
        forward: LinearMap,
        adjoint: LinearMap,
        *,
        field_reference: Tensor,
        observation_reference: Tensor,
    ) -> None:
        if not callable(forward) or not callable(adjoint):
            raise TypeError("forward and adjoint must be callable")
        self._forward = forward
        self._adjoint = adjoint
        self._field_reference = field_reference
        self._observation_reference = observation_reference
        self.forward_calls = 0
        self.adjoint_calls = 0

    @staticmethod
    def _validated_output(value: Tensor, *, reference: Tensor, name: str) -> Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must return a torch.Tensor")
        if value.shape != reference.shape:
            raise ValueError(
                f"{name} returned shape {tuple(value.shape)}, expected "
                f"{tuple(reference.shape)}"
            )
        if value.dtype != reference.dtype or value.device != reference.device:
            raise ValueError(f"{name} must preserve the declared dtype and device")
        if not bool(torch.all(torch.isfinite(value))):
            raise FloatingPointError(f"{name} returned non-finite values")
        return value

    def forward(self, field: Tensor) -> Tensor:
        self.forward_calls += 1
        return self._validated_output(
            self._forward(field),
            reference=self._observation_reference,
            name="forward",
        )

    def adjoint(self, measurement: Tensor) -> Tensor:
        self.adjoint_calls += 1
        return self._validated_output(
            self._adjoint(measurement),
            reference=self._field_reference,
            name="adjoint",
        )


def _result(
    *,
    field: Tensor,
    history: list[HistoryRow],
    operator: _CountedLinearOperator,
    expected_calls: int,
) -> InterfaceBaselineResult:
    if operator.forward_calls != expected_calls or operator.adjoint_calls != expected_calls:
        raise RuntimeError(
            "physical call contract violated: expected "
            f"{expected_calls} forward/{expected_calls} adjoint, observed "
            f"{operator.forward_calls} forward/{operator.adjoint_calls} adjoint"
        )
    return InterfaceBaselineResult(
        field=field,
        history=history,
        forward_calls=operator.forward_calls,
        adjoint_calls=operator.adjoint_calls,
    )


@torch.no_grad()
def cgls_baseline(
    observation: Tensor,
    *,
    forward: LinearMap,
    adjoint: LinearMap,
    support: Tensor,
    spacing_xyz: Sequence[float],
    iterations: int,
    denominator_floor: float = 1e-20,
) -> InterfaceBaselineResult:
    """Run zero-start CGLS with exactly one physical pair per iteration.

    The support projection is applied to every normal residual and search
    direction. ``spacing_xyz`` is validated for interface parity with the
    regularized baseline but does not alter the data-only CGLS recurrence.
    A numerical breakdown produces zero steps while the remaining registered
    calls are still executed, preserving the fixed comparison budget.
    """

    target = _validated_observation(observation)
    mask = _validated_support(support, reference=target)
    _validated_spacing(spacing_xyz)
    count = _positive_integer(iterations, name="iterations")
    floor = _finite_scalar(denominator_floor, name="denominator_floor", positive=True)
    field = torch.zeros_like(mask)
    operator = _CountedLinearOperator(
        forward,
        adjoint,
        field_reference=field,
        observation_reference=target,
    )

    residual = target.clone()
    normal = operator.adjoint(residual) * mask
    direction = normal.clone()
    gamma = _dot(normal, normal)
    initial_residual_squared = max(float(_dot(residual, residual).item()), floor)
    history: list[HistoryRow] = []

    for index in range(count):
        projected = operator.forward(direction)
        denominator = _dot(projected, projected)
        gamma_value = float(gamma.item())
        denominator_value = float(denominator.item())
        breakdown = gamma_value <= floor or denominator_value <= floor
        alpha_value = 0.0 if breakdown else gamma_value / denominator_value
        alpha = target.new_tensor(alpha_value)
        field = (field + alpha * direction) * mask
        residual = residual - alpha * projected
        residual_squared = float(_dot(residual, residual).item())

        beta_value: float | None = None
        if index + 1 < count:
            next_normal = operator.adjoint(residual) * mask
            next_gamma = _dot(next_normal, next_normal)
            next_gamma_value = float(next_gamma.item())
            beta_value = 0.0 if gamma_value <= floor else next_gamma_value / gamma_value
            direction = (next_normal + beta_value * direction) * mask
            normal = next_normal
            gamma = next_gamma

        history.append(
            {
                "iteration": index + 1,
                "data_residual_norm": math.sqrt(max(residual_squared, 0.0)),
                "relative_data_residual_squared": residual_squared
                / initial_residual_squared,
                "normal_residual_norm_before": math.sqrt(max(gamma_value, 0.0)),
                "alpha": alpha_value,
                "beta": beta_value,
                "breakdown": breakdown,
            }
        )
        if not bool(torch.all(torch.isfinite(field))):
            raise FloatingPointError("CGLS produced a non-finite field")

    return _result(
        field=field,
        history=history,
        operator=operator,
        expected_calls=count,
    )


def _edge_value(
    gradient: Tensor,
    *,
    penalty: EdgePenalty,
    huber_delta: float,
) -> float:
    magnitude = torch.linalg.vector_norm(gradient, dim=1)
    if penalty == "tv":
        values = magnitude
    else:
        values = torch.where(
            magnitude <= huber_delta,
            0.5 * magnitude.square() / huber_delta,
            magnitude - 0.5 * huber_delta,
        )
    return float(torch.sum(values).item())


def _scalar_huber_value(residual: Tensor, *, delta: float) -> float:
    """Return the separable Huber data penalty used by the robust solver."""

    magnitude = torch.abs(residual)
    values = torch.where(
        magnitude <= delta,
        0.5 * magnitude.square() / delta,
        magnitude - 0.5 * delta,
    )
    return float(torch.sum(values).item())


def _validated_edge_weight_map(
    edge_weight_map: Tensor | None,
    *,
    mask: Tensor,
) -> Tensor | None:
    if edge_weight_map is None:
        return None
    weights = torch.as_tensor(edge_weight_map).to(mask)
    if weights.shape != mask.shape:
        raise ValueError("edge_weight_map must match support shape")
    if not bool(torch.all(torch.isfinite(weights))) or bool(torch.any(weights <= 0.0)):
        raise ValueError("edge_weight_map must contain positive finite values")
    return weights


def _proximal_weighted_edge_conjugate(
    dual: Tensor,
    *,
    regularization_weight: float,
    edge_weight_map: Tensor,
    dual_step: float,
    penalty: EdgePenalty,
    huber_delta: float,
) -> Tensor:
    if regularization_weight == 0.0:
        return torch.zeros_like(dual)
    radius = regularization_weight * edge_weight_map[None, None]
    if penalty == "huber":
        unconstrained = dual / (
            1.0 + dual_step * huber_delta / radius
        )
    else:
        unconstrained = dual
    magnitude = torch.linalg.vector_norm(unconstrained, dim=1, keepdim=True)
    scale = torch.clamp(radius / magnitude.clamp_min(1e-30), max=1.0)
    return unconstrained * scale


def _weighted_edge_value(
    gradient: Tensor,
    *,
    edge_weight_map: Tensor,
    penalty: EdgePenalty,
    huber_delta: float,
) -> float:
    magnitude = torch.linalg.vector_norm(gradient, dim=1)
    if penalty == "tv":
        values = magnitude
    else:
        values = torch.where(
            magnitude <= huber_delta,
            0.5 * magnitude.square() / huber_delta,
            magnitude - 0.5 * huber_delta,
        )
    return float(torch.sum(values * edge_weight_map[None]).item())


def _pdhg_steps(
    *,
    data_norm_squared_bound: float,
    gradient_norm_squared_bound: float,
    regularization_weight: float,
    step_safety: float,
    primal_step: float | None,
    data_dual_step: float | None,
    edge_dual_step: float | None,
) -> tuple[float, float, float, float]:
    supplied = (
        primal_step is not None,
        data_dual_step is not None,
        edge_dual_step is not None,
    )
    if any(supplied) and not all(supplied):
        raise ValueError(
            "primal_step, data_dual_step and edge_dual_step must be supplied together"
        )
    if all(supplied):
        assert primal_step is not None
        assert data_dual_step is not None
        assert edge_dual_step is not None
        tau = _finite_scalar(
            primal_step,
            name="primal_step",
            positive=True,
        )
        sigma_data = _finite_scalar(
            data_dual_step,
            name="data_dual_step",
            positive=True,
        )
        sigma_edge = _finite_scalar(
            edge_dual_step,
            name="edge_dual_step",
            positive=True,
        )
    else:
        sigma_data = 1.0 / math.sqrt(data_norm_squared_bound)
        sigma_edge = 1.0 / math.sqrt(gradient_norm_squared_bound)
        active_edge_bound = (
            gradient_norm_squared_bound if regularization_weight > 0.0 else 0.0
        )
        denominator = (
            sigma_data * data_norm_squared_bound
            + sigma_edge * active_edge_bound
        )
        tau = step_safety / denominator
    active_edge_bound = (
        gradient_norm_squared_bound if regularization_weight > 0.0 else 0.0
    )
    contract = tau * (
        sigma_data * data_norm_squared_bound
        + sigma_edge * active_edge_bound
    )
    if not math.isfinite(contract) or contract >= 1.0:
        raise ValueError(
            "PDHG steps must satisfy tau*(sigma_data*||A||^2 + "
            "sigma_edge*||D||^2) < 1"
        )
    return tau, sigma_data, sigma_edge, contract


@torch.no_grad()
def edge_preserving_pdhg_baseline(
    observation: Tensor,
    *,
    forward: LinearMap,
    adjoint: LinearMap,
    support: Tensor,
    spacing_xyz: Sequence[float],
    iterations: int,
    regularization_weight: float,
    data_norm_squared_bound: float,
    penalty: EdgePenalty = "huber",
    huber_delta: float = 0.1,
    step_safety: float = 0.99,
    extrapolation: float = 1.0,
    primal_step: float | None = None,
    data_dual_step: float | None = None,
    edge_dual_step: float | None = None,
) -> InterfaceBaselineResult:
    """Solve a least-squares plus isotropic TV/Huber objective with PDHG.

    ``data_norm_squared_bound`` must be an externally established upper bound
    for ``||A||^2``. The default block steps satisfy the strict Chambolle--Pock
    contract using the repository's analytic forward-Neumann gradient bound.
    Explicit steps may be supplied as a complete triple and are checked by the
    same contract. Every iteration performs exactly one call to each injected
    physical map; local gradient operations are not physical-operator calls.
    """

    target = _validated_observation(observation)
    mask = _validated_support(support, reference=target)
    spacing = _validated_spacing(spacing_xyz)
    count = _positive_integer(iterations, name="iterations")
    weight = _finite_scalar(
        regularization_weight,
        name="regularization_weight",
        nonnegative=True,
    )
    data_bound = _finite_scalar(
        data_norm_squared_bound,
        name="data_norm_squared_bound",
        positive=True,
    )
    delta = _finite_scalar(huber_delta, name="huber_delta", positive=True)
    safety = _finite_scalar(step_safety, name="step_safety", positive=True)
    theta = _finite_scalar(extrapolation, name="extrapolation")
    if penalty not in {"tv", "huber"}:
        raise ValueError("penalty must be 'tv' or 'huber'")
    if safety >= 1.0:
        raise ValueError("step_safety must lie in (0,1)")
    if not 0.0 <= theta <= 1.0:
        raise ValueError("extrapolation must lie in [0,1]")
    gradient_bound = gradient_operator_norm_squared_bound(spacing)
    tau, sigma_data, sigma_edge, contract = _pdhg_steps(
        data_norm_squared_bound=data_bound,
        gradient_norm_squared_bound=gradient_bound,
        regularization_weight=weight,
        step_safety=safety,
        primal_step=primal_step,
        data_dual_step=data_dual_step,
        edge_dual_step=edge_dual_step,
    )

    field = torch.zeros_like(mask)
    extrapolated = field.clone()
    data_dual = torch.zeros_like(target)
    edge_dual = torch.zeros(
        (1, 3, *mask.shape),
        dtype=target.dtype,
        device=target.device,
    )
    operator = _CountedLinearOperator(
        forward,
        adjoint,
        field_reference=field,
        observation_reference=target,
    )
    initial_data_squared = max(float(_dot(target, target).item()), 1e-30)
    history: list[HistoryRow] = []

    for index in range(count):
        projected = operator.forward(extrapolated)
        residual = projected - target
        data_dual = (data_dual + sigma_data * residual) / (1.0 + sigma_data)
        gradient = regularization_gradient(
            extrapolated[None],
            spacing_xyz=spacing,
        )
        edge_dual = proximal_edge_conjugate(
            edge_dual + sigma_edge * gradient,
            regularization_weight=weight,
            dual_step=sigma_edge,
            penalty=penalty,
            huber_delta=delta,
        )
        data_normal = operator.adjoint(data_dual) * mask
        edge_normal = regularization_gradient_adjoint(
            edge_dual,
            spacing_xyz=spacing,
        )[0]
        next_field = (
            field - tau * (data_normal + edge_normal)
        ) * mask
        update = next_field - field
        extrapolated = (next_field + theta * update) * mask
        field = next_field

        residual_squared = float(_dot(residual, residual).item())
        edge_value = _edge_value(
            gradient,
            penalty=penalty,
            huber_delta=delta,
        )
        history.append(
            {
                "iteration": index + 1,
                "extrapolated_data_residual_norm": math.sqrt(
                    max(residual_squared, 0.0)
                ),
                "extrapolated_relative_data_objective": residual_squared
                / initial_data_squared,
                "extrapolated_edge_penalty": edge_value,
                "extrapolated_total_objective": 0.5 * residual_squared
                + weight * edge_value,
                "primal_update_norm": float(torch.linalg.vector_norm(update).item()),
                "step_contract": contract,
            }
        )
        if not bool(torch.all(torch.isfinite(field))):
            raise FloatingPointError("PDHG produced a non-finite field")
        if not bool(torch.all(torch.isfinite(data_dual))):
            raise FloatingPointError("PDHG produced a non-finite data dual")
        if not bool(torch.all(torch.isfinite(edge_dual))):
            raise FloatingPointError("PDHG produced a non-finite edge dual")

    return _result(
        field=field,
        history=history,
        operator=operator,
        expected_calls=count,
    )


@torch.no_grad()
def robust_data_pdhg_baseline(
    observation: Tensor,
    *,
    forward: LinearMap,
    adjoint: LinearMap,
    support: Tensor,
    spacing_xyz: Sequence[float],
    iterations: int,
    regularization_weight: float,
    data_norm_squared_bound: float,
    data_huber_delta: float,
    edge_penalty: EdgePenalty = "huber",
    edge_huber_delta: float = 0.1,
    ridge_weight: float = 0.0,
    initial_field: Tensor | None = None,
    edge_weight_map: Tensor | None = None,
    step_safety: float = 0.99,
    extrapolation: float = 1.0,
    primal_step: float | None = None,
    data_dual_step: float | None = None,
    edge_dual_step: float | None = None,
) -> InterfaceBaselineResult:
    """Solve Huber-data plus isotropic TV/Huber regularization with PDHG.

    The caller may inject a prewhitened operator pair and prewhitened
    observation; this function never estimates covariance from the target.
    With ``rho_delta(t)=t^2/(2 delta)`` near zero and ``|t|-delta/2`` in the
    tails, the data-dual proximal is an analytic shrink followed by clipping
    to ``[-1, 1]``. Every iteration uses exactly one injected forward and one
    injected adjoint call.
    """

    target = _validated_observation(observation)
    mask = _validated_support(support, reference=target)
    spacing = _validated_spacing(spacing_xyz)
    count = _positive_integer(iterations, name="iterations")
    weight = _finite_scalar(
        regularization_weight,
        name="regularization_weight",
        nonnegative=True,
    )
    data_bound = _finite_scalar(
        data_norm_squared_bound,
        name="data_norm_squared_bound",
        positive=True,
    )
    data_delta = _finite_scalar(
        data_huber_delta,
        name="data_huber_delta",
        positive=True,
    )
    edge_delta = _finite_scalar(
        edge_huber_delta,
        name="edge_huber_delta",
        positive=True,
    )
    ridge = _finite_scalar(ridge_weight, name="ridge_weight", nonnegative=True)
    spatial_weights = _validated_edge_weight_map(edge_weight_map, mask=mask)
    safety = _finite_scalar(step_safety, name="step_safety", positive=True)
    theta = _finite_scalar(extrapolation, name="extrapolation")
    if edge_penalty not in {"tv", "huber"}:
        raise ValueError("edge_penalty must be 'tv' or 'huber'")
    if safety >= 1.0:
        raise ValueError("step_safety must lie in (0,1)")
    if not 0.0 <= theta <= 1.0:
        raise ValueError("extrapolation must lie in [0,1]")
    gradient_bound = gradient_operator_norm_squared_bound(spacing)
    tau, sigma_data, sigma_edge, contract = _pdhg_steps(
        data_norm_squared_bound=data_bound,
        gradient_norm_squared_bound=gradient_bound,
        regularization_weight=weight,
        step_safety=safety,
        primal_step=primal_step,
        data_dual_step=data_dual_step,
        edge_dual_step=edge_dual_step,
    )

    if initial_field is None:
        field = torch.zeros_like(mask)
    else:
        field = torch.as_tensor(initial_field).to(target)
        if field.shape != mask.shape:
            raise ValueError("initial_field must match support shape")
        if not bool(torch.all(torch.isfinite(field))):
            raise ValueError("initial_field must contain only finite values")
        field = field.clone() * mask
    extrapolated = field.clone()
    data_dual = torch.zeros_like(target)
    edge_dual = torch.zeros(
        (1, 3, *mask.shape),
        dtype=target.dtype,
        device=target.device,
    )
    operator = _CountedLinearOperator(
        forward,
        adjoint,
        field_reference=field,
        observation_reference=target,
    )
    history: list[HistoryRow] = []

    for index in range(count):
        projected = operator.forward(extrapolated)
        residual = projected - target
        unconstrained_data_dual = (
            data_dual + sigma_data * residual
        ) / (1.0 + sigma_data * data_delta)
        data_dual = torch.clamp(unconstrained_data_dual, min=-1.0, max=1.0)
        gradient = regularization_gradient(
            extrapolated[None],
            spacing_xyz=spacing,
        )
        ridge_value = 0.5 * ridge * float(_dot(extrapolated, extrapolated).item())
        edge_argument = edge_dual + sigma_edge * gradient
        if spatial_weights is None:
            edge_dual = proximal_edge_conjugate(
                edge_argument,
                regularization_weight=weight,
                dual_step=sigma_edge,
                penalty=edge_penalty,
                huber_delta=edge_delta,
            )
        else:
            edge_dual = _proximal_weighted_edge_conjugate(
                edge_argument,
                regularization_weight=weight,
                edge_weight_map=spatial_weights,
                dual_step=sigma_edge,
                penalty=edge_penalty,
                huber_delta=edge_delta,
            )
        data_normal = operator.adjoint(data_dual) * mask
        edge_normal = regularization_gradient_adjoint(
            edge_dual,
            spacing_xyz=spacing,
        )[0]
        next_field = (
            field - tau * (data_normal + edge_normal)
        ) / (1.0 + tau * ridge)
        next_field = next_field * mask
        update = next_field - field
        extrapolated = (next_field + theta * update) * mask
        field = next_field

        data_value = _scalar_huber_value(residual, delta=data_delta)
        edge_value = (
            _edge_value(
                gradient,
                penalty=edge_penalty,
                huber_delta=edge_delta,
            )
            if spatial_weights is None
            else _weighted_edge_value(
                gradient,
                edge_weight_map=spatial_weights,
                penalty=edge_penalty,
                huber_delta=edge_delta,
            )
        )
        history.append(
            {
                "iteration": index + 1,
                "extrapolated_data_residual_norm": float(
                    torch.linalg.vector_norm(residual).item()
                ),
                "extrapolated_robust_data_penalty": data_value,
                "extrapolated_edge_penalty": edge_value,
                "extrapolated_ridge_penalty": ridge_value,
                "extrapolated_total_objective": (
                    data_value + weight * edge_value + ridge_value
                ),
                "primal_update_norm": float(torch.linalg.vector_norm(update).item()),
                "step_contract": contract,
                "data_dual_saturation_fraction": float(
                    torch.mean((torch.abs(data_dual) >= 1.0).to(target.dtype)).item()
                ),
            }
        )
        if not bool(torch.all(torch.isfinite(field))):
            raise FloatingPointError("robust-data PDHG produced a non-finite field")
        if not bool(torch.all(torch.isfinite(data_dual))):
            raise FloatingPointError("robust-data PDHG produced a non-finite data dual")
        if not bool(torch.all(torch.isfinite(edge_dual))):
            raise FloatingPointError("robust-data PDHG produced a non-finite edge dual")

    return _result(
        field=field,
        history=history,
        operator=operator,
        expected_calls=count,
    )


__all__ = [
    "InterfaceBaselineResult",
    "cgls_baseline",
    "edge_preserving_pdhg_baseline",
    "robust_data_pdhg_baseline",
]
