"""Call-audited primal-dual reconstruction for covariance-weighted PSU B0.

The physical forward model is composed with detector-covariance whitening
before it reaches this module.  Each PDHG iteration performs exactly one
physical forward and one physical adjoint.  Three-dimensional finite
differences and their exact transpose are local voxel operations and are kept
separate from the physical call ledger.

The fixed-step contract uses separately estimated bounds for the whitened data
operator and the finite-difference gradient.  Power iteration is an explicit,
shared setup cost; it is never folded into the per-reconstruction solve count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import torch

from .psu_b0_classical_baselines import _weighted_measurement_terms
PRIMAL_DUAL_SCHEMA = "psu-b0-covariance-primal-dual-1.0"
EdgePenalty = Literal["tv", "huber"]


@dataclass(frozen=True)
class BlockOperatorNormEstimate:
    """Power-iteration estimates with explicit physical setup calls."""

    data_norm_squared_by_sample: torch.Tensor
    data_norm_squared_upper: float
    gradient_norm_squared_upper: float
    power_iterations: int
    safety_factor: float
    forward_calls: int
    adjoint_calls: int
    operator_identity: int
    operator_state_token: tuple[tuple[str, str], ...]
    batch_size: int
    measurement_scale: float
    grid_shape: tuple[int, int, int]
    spacing_xyz: tuple[float, float, float]


@dataclass(frozen=True)
class PrimalDualReconstruction:
    """PDHG output without an uncounted terminal forward projection."""

    volume: torch.Tensor
    data_dual: torch.Tensor
    edge_dual: torch.Tensor
    history: list[dict[str, torch.Tensor]]
    checkpoint_volumes: dict[int, torch.Tensor]
    forward_calls: int
    adjoint_calls: int
    gradient_calls: int
    gradient_adjoint_calls: int
    step_contract_value: float


def _forward_difference_axis(
    values: torch.Tensor,
    *,
    axis: int,
    spacing: float,
) -> torch.Tensor:
    moved = torch.movedim(values, axis, -1)
    output = torch.zeros_like(moved)
    output[..., :-1] = (moved[..., 1:] - moved[..., :-1]) / spacing
    return torch.movedim(output, -1, axis)


def _forward_difference_axis_adjoint(
    values: torch.Tensor,
    *,
    axis: int,
    spacing: float,
) -> torch.Tensor:
    moved = torch.movedim(values, axis, -1)
    output = torch.zeros_like(moved)
    output[..., :-1] -= moved[..., :-1] / spacing
    output[..., 1:] += moved[..., :-1] / spacing
    return torch.movedim(output, -1, axis)


def regularization_gradient(
    volume: torch.Tensor,
    *,
    spacing_xyz: tuple[float, float, float],
) -> torch.Tensor:
    """Forward-Neumann 3-D gradient with channels ``dx,dy,dz``."""

    if volume.ndim != 4:
        raise ValueError("volume must have shape [batch,z,y,x]")
    dx = _forward_difference_axis(
        volume,
        axis=-1,
        spacing=float(spacing_xyz[0]),
    )
    dy = _forward_difference_axis(
        volume,
        axis=-2,
        spacing=float(spacing_xyz[1]),
    )
    dz = _forward_difference_axis(
        volume,
        axis=-3,
        spacing=float(spacing_xyz[2]),
    )
    return torch.stack((dx, dy, dz), dim=1)


def regularization_gradient_adjoint(
    gradient: torch.Tensor,
    *,
    spacing_xyz: tuple[float, float, float],
) -> torch.Tensor:
    """Exact transpose of :func:`regularization_gradient`."""

    if gradient.ndim != 5 or gradient.shape[1] != 3:
        raise ValueError("gradient must have shape [batch,3,z,y,x]")
    return (
        _forward_difference_axis_adjoint(
            gradient[:, 0],
            axis=-1,
            spacing=float(spacing_xyz[0]),
        )
        + _forward_difference_axis_adjoint(
            gradient[:, 1],
            axis=-2,
            spacing=float(spacing_xyz[1]),
        )
        + _forward_difference_axis_adjoint(
            gradient[:, 2],
            axis=-3,
            spacing=float(spacing_xyz[2]),
        )
    )


class ForwardNeumannRegularizationOperator:
    """Call-audited wrapper for the standard local gradient and its transpose."""

    def __init__(self, spacing_xyz: tuple[float, float, float]) -> None:
        spacing = tuple(float(value) for value in spacing_xyz)
        gradient_operator_norm_squared_bound(spacing)
        self.spacing_xyz = spacing
        self.gradient_calls = 0
        self.gradient_adjoint_calls = 0

    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        self.gradient_calls += 1
        return regularization_gradient(
            volume,
            spacing_xyz=self.spacing_xyz,
        )

    def adjoint(self, gradient: torch.Tensor) -> torch.Tensor:
        self.gradient_adjoint_calls += 1
        return regularization_gradient_adjoint(
            gradient,
            spacing_xyz=self.spacing_xyz,
        )

    def call_report(self) -> dict[str, int]:
        return {
            "gradient_calls": self.gradient_calls,
            "gradient_adjoint_calls": self.gradient_adjoint_calls,
        }


def isotropic_edge_penalty(
    volume: torch.Tensor,
    *,
    spacing_xyz: tuple[float, float, float],
    penalty: EdgePenalty,
    huber_delta: float = 0.1,
) -> torch.Tensor:
    """Return exact isotropic TV or isotropic Huber-TV per batch item."""

    if volume.ndim != 5 or volume.shape[1] != 1:
        raise ValueError("volume must have shape [batch,1,z,y,x]")
    delta = float(huber_delta)
    if not torch.isfinite(torch.as_tensor(delta)) or delta <= 0.0:
        raise ValueError("huber_delta must be finite and positive")
    gradient = regularization_gradient(
        volume[:, 0],
        spacing_xyz=spacing_xyz,
    )
    return _edge_penalty_from_gradient(
        gradient,
        penalty=penalty,
        huber_delta=delta,
    )


def _edge_penalty_from_gradient(
    gradient: torch.Tensor,
    *,
    penalty: EdgePenalty,
    huber_delta: float,
) -> torch.Tensor:
    if gradient.ndim != 5 or gradient.shape[1] != 3:
        raise ValueError("gradient must have shape [batch,3,z,y,x]")
    delta = float(huber_delta)
    magnitude = torch.linalg.vector_norm(gradient, dim=1)
    if penalty == "tv":
        values = magnitude
    elif penalty == "huber":
        values = torch.where(
            magnitude <= delta,
            0.5 * magnitude.square() / delta,
            magnitude - 0.5 * delta,
        )
    else:
        raise ValueError("penalty must be 'tv' or 'huber'")
    return torch.sum(values, dim=(1, 2, 3))


def regularization_site_mask(support: torch.Tensor) -> torch.Tensor:
    """Return sites where a forward-Neumann support gradient may be nonzero."""

    values = torch.as_tensor(support, dtype=torch.bool)
    if values.ndim != 3:
        raise ValueError("support must have shape [z,y,x]")
    sites = values.clone()
    sites[:, :, :-1] |= values[:, :, 1:]
    sites[:, :-1, :] |= values[:, 1:, :]
    sites[:-1, :, :] |= values[1:, :, :]
    return sites


def gradient_operator_norm_squared_bound(
    spacing_xyz: tuple[float, float, float],
) -> float:
    """Conservative squared-norm bound for the declared 3-D stencil."""

    spacing = torch.as_tensor(spacing_xyz, dtype=torch.float64)
    if spacing.shape != (3,) or torch.any(~torch.isfinite(spacing)):
        raise ValueError("spacing_xyz must contain three finite values")
    if torch.any(spacing <= 0.0):
        raise ValueError("spacing_xyz must be positive")
    return float(4.0 * torch.sum(spacing.reciprocal().square()))


def proximal_edge_conjugate(
    dual: torch.Tensor,
    *,
    regularization_weight: float,
    dual_step: float,
    penalty: EdgePenalty,
    huber_delta: float = 0.1,
) -> torch.Tensor:
    """Apply ``prox_(sigma (lambda phi)^*)`` voxelwise.

    For isotropic TV this is projection onto the l2 ball of radius ``lambda``.
    The Huber conjugate adds ``delta/(2 lambda) * ||q||^2`` inside the same
    ball, so the unconstrained point is first shrunk and then projected.
    """

    if dual.ndim != 5 or dual.shape[1] != 3:
        raise ValueError("dual must have shape [batch,3,z,y,x]")
    weight = float(regularization_weight)
    step = float(dual_step)
    delta = float(huber_delta)
    if not torch.isfinite(torch.as_tensor(weight)) or weight < 0.0:
        raise ValueError("regularization_weight must be finite and nonnegative")
    if not torch.isfinite(torch.as_tensor(step)) or step <= 0.0:
        raise ValueError("dual_step must be finite and positive")
    if not torch.isfinite(torch.as_tensor(delta)) or delta <= 0.0:
        raise ValueError("huber_delta must be finite and positive")
    if weight == 0.0:
        return torch.zeros_like(dual)
    if penalty == "tv":
        candidate = dual
    elif penalty == "huber":
        candidate = dual / (1.0 + step * delta / weight)
    else:
        raise ValueError("penalty must be 'tv' or 'huber'")
    norm = torch.linalg.vector_norm(candidate, dim=1, keepdim=True)
    scale = torch.maximum(
        torch.ones_like(norm),
        norm / weight,
    )
    return candidate / scale


def _normalize_volume(
    volume: torch.Tensor,
    *,
    floor: float,
) -> torch.Tensor:
    norm = torch.linalg.vector_norm(volume.flatten(1), dim=1)
    return volume / norm.clamp_min(float(floor))[:, None, None, None, None]


def _validated_dirichlet_support(
    operator: Any,
    *,
    reference: torch.Tensor,
) -> torch.Tensor:
    support = torch.as_tensor(operator.support).to(reference)
    if support.shape != tuple(int(value) for value in operator.grid_shape):
        raise ValueError("operator support must match grid_shape")
    if not bool(
        torch.all(
            torch.isclose(
                support,
                support.round(),
                atol=1e-6,
                rtol=0.0,
            )
        )
    ):
        raise ValueError("PDHG support must be binary")
    support = support.round()
    boundary = torch.cat(
        (
            support[0].flatten(),
            support[-1].flatten(),
            support[:, 0].flatten(),
            support[:, -1].flatten(),
            support[:, :, 0].flatten(),
            support[:, :, -1].flatten(),
        )
    )
    if bool(torch.any(boundary != 0.0)):
        raise ValueError(
            "PDHG support must zero the outer boundary to fix the BOS gauge"
        )
    if not bool(torch.any(support > 0.5)):
        raise ValueError("PDHG support must retain at least one interior voxel")
    return support[None, None]


def _operator_state_token(operator: Any) -> tuple[tuple[str, str], ...]:
    """Return a cheap process-local invalidation token for fixed operator state."""

    rows: list[tuple[str, str]] = [
        (
            "operator_type",
            f"{type(operator).__module__}.{type(operator).__qualname__}",
        ),
        ("grid_shape", repr(tuple(int(item) for item in operator.grid_shape))),
        ("spacing_xyz", repr(tuple(float(item) for item in operator.spacing_xyz))),
        ("ray_count", repr(int(operator.ray_count))),
    ]
    named_buffers = getattr(operator, "named_buffers", None)
    named_parameters = getattr(operator, "named_parameters", None)
    tensors: list[tuple[str, torch.Tensor]] = []
    if callable(named_buffers):
        tensors.extend(
            (f"buffer:{name}", value)
            for name, value in named_buffers(recurse=True)
        )
    if callable(named_parameters):
        tensors.extend(
            (f"parameter:{name}", value)
            for name, value in named_parameters(recurse=True)
        )
    if not tensors:
        tensors.append(("support", torch.as_tensor(operator.support)))
    for name, tensor in sorted(tensors, key=lambda item: item[0]):
        value = torch.as_tensor(tensor)
        version = int(getattr(value, "_version", -1))
        rows.append(
            (
                name,
                repr(
                    (
                        id(tensor),
                        version,
                        tuple(int(item) for item in value.shape),
                        str(value.dtype),
                        str(value.device),
                    )
                ),
            )
        )
    return tuple(rows)


def _prewhitened_measurement_terms(
    observation_uv: torch.Tensor,
    *,
    sigma_by_view: torch.Tensor,
    view_mask: torch.Tensor,
    rays_per_view: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not bool(torch.all(view_mask == 1.0)):
        raise ValueError(
            "covariance PDHG requires the frozen all-view prewhitened operator"
        )
    if not bool(torch.all(sigma_by_view == 1.0)):
        raise ValueError(
            "sigma_by_view must be one because covariance whitening is already applied"
        )
    return _weighted_measurement_terms(
        observation_uv,
        sigma_by_view=sigma_by_view,
        view_mask=view_mask,
        rays_per_view=rays_per_view,
    )


def _tail_maximum(values: list[torch.Tensor]) -> torch.Tensor:
    start = max(len(values) // 2, 0)
    return torch.amax(torch.stack(values[start:]), dim=0)


@torch.no_grad()
def estimate_block_operator_norms(
    operator: Any,
    *,
    batch_size: int,
    sigma_by_view: torch.Tensor,
    view_mask: torch.Tensor,
    rays_per_view: int,
    power_iterations: int = 12,
    safety_factor: float = 1.25,
    measurement_scale: float = 1.0,
    seed: int = 0,
    denominator_floor: float = 1e-20,
) -> BlockOperatorNormEstimate:
    """Estimate data/gradient squared norms and expose setup call cost.

    One batch power iteration estimates every sample-specific whitened data
    block simultaneously.  The returned values are power estimates multiplied
    by a declared safety factor, not mathematically certified upper bounds.
    """

    count = int(power_iterations)
    batch = int(batch_size)
    safety = float(safety_factor)
    scale = float(measurement_scale)
    if count < 2:
        raise ValueError("power_iterations must be at least two")
    if batch < 1:
        raise ValueError("batch_size must be positive")
    if not torch.isfinite(torch.as_tensor(safety)) or safety <= 1.0:
        raise ValueError("safety_factor must be finite and greater than one")
    if not torch.isfinite(torch.as_tensor(scale)) or scale <= 0.0:
        raise ValueError("measurement_scale must be finite and positive")
    support = _validated_dirichlet_support(
        operator,
        reference=sigma_by_view,
    )
    dummy = torch.zeros(
        (batch, int(operator.ray_count), 2),
        dtype=sigma_by_view.dtype,
        device=sigma_by_view.device,
    )
    active, sigma = _prewhitened_measurement_terms(
        dummy,
        sigma_by_view=sigma_by_view,
        view_mask=view_mask,
        rays_per_view=rays_per_view,
    )
    if len(sigma_by_view) != batch:
        raise ValueError("sigma_by_view batch must match batch_size")
    before = operator.call_report()
    generator = torch.Generator().manual_seed(int(seed))
    current = torch.randn(
        (batch, 1, *operator.grid_shape),
        generator=generator,
        dtype=sigma_by_view.dtype,
    ).to(sigma_by_view.device)
    current = _normalize_volume(
        current * support,
        floor=denominator_floor,
    )
    data_rayleigh: list[torch.Tensor] = []
    for _ in range(count):
        projected = scale * active * operator(current) / sigma
        normal = (
            scale * operator.adjoint(active * projected / sigma) * support
        )
        denominator = torch.sum(current.square(), dim=(1, 2, 3, 4))
        data_rayleigh.append(
            torch.sum(current * normal, dim=(1, 2, 3, 4))
            / denominator.clamp_min(float(denominator_floor))
        )
        current = _normalize_volume(
            normal,
            floor=denominator_floor,
        )
    after = operator.call_report()
    forward_calls = int(after["forward_calls"] - before["forward_calls"])
    adjoint_calls = int(after["adjoint_calls"] - before["adjoint_calls"])
    if forward_calls != count or adjoint_calls != count:
        raise RuntimeError(
            "norm-estimation call contract violated: expected "
            f"{count} forward/{count} adjoint, observed "
            f"{forward_calls} forward/{adjoint_calls} adjoint"
        )

    spacing_xyz = tuple(float(value) for value in operator.spacing_xyz)
    gradient_current = torch.randn(
        (1, 1, *operator.grid_shape),
        generator=generator,
        dtype=sigma_by_view.dtype,
    ).to(sigma_by_view.device)
    gradient_current = _normalize_volume(
        gradient_current * support[:1],
        floor=denominator_floor,
    )
    gradient_rayleigh: list[torch.Tensor] = []
    for _ in range(count):
        gradient = regularization_gradient(
            gradient_current[:, 0],
            spacing_xyz=spacing_xyz,
        )
        normal = regularization_gradient_adjoint(
            gradient,
            spacing_xyz=spacing_xyz,
        )[:, None] * support[:1]
        denominator = torch.sum(
            gradient_current.square(),
            dim=(1, 2, 3, 4),
        )
        gradient_rayleigh.append(
            torch.sum(
                gradient_current * normal,
                dim=(1, 2, 3, 4),
            )
            / denominator.clamp_min(float(denominator_floor))
        )
        gradient_current = _normalize_volume(
            normal,
            floor=denominator_floor,
        )

    data_values = _tail_maximum(data_rayleigh).clamp_min(
        float(denominator_floor)
    ) * safety
    gradient_power = float(
        _tail_maximum(gradient_rayleigh).max().clamp_min(
            float(denominator_floor)
        )
        * safety
    )
    gradient_value = max(
        gradient_power,
        gradient_operator_norm_squared_bound(spacing_xyz),
    )
    return BlockOperatorNormEstimate(
        data_norm_squared_by_sample=data_values,
        data_norm_squared_upper=float(torch.max(data_values)),
        gradient_norm_squared_upper=gradient_value,
        power_iterations=count,
        safety_factor=safety,
        forward_calls=forward_calls,
        adjoint_calls=adjoint_calls,
        operator_identity=id(operator),
        operator_state_token=_operator_state_token(operator),
        batch_size=batch,
        measurement_scale=scale,
        grid_shape=tuple(int(value) for value in operator.grid_shape),
        spacing_xyz=spacing_xyz,
    )


@torch.no_grad()
def primal_dual_reconstruction(
    operator: Any,
    observation_uv: torch.Tensor,
    *,
    sigma_by_view: torch.Tensor,
    view_mask: torch.Tensor,
    rays_per_view: int,
    iterations: int,
    regularization_weight: float,
    penalty: EdgePenalty,
    primal_step: float,
    data_dual_step: float,
    edge_dual_step: float,
    norm_estimate: BlockOperatorNormEstimate,
    huber_delta: float = 0.1,
    measurement_scale: float = 1.0,
    extrapolation: float = 1.0,
    initial_volume: torch.Tensor | None = None,
    initial_data_dual: torch.Tensor | None = None,
    checkpoint_iterations: tuple[int, ...] | list[int] | None = None,
    denominator_floor: float = 1e-20,
    regularization_operator: Any | None = None,
) -> PrimalDualReconstruction:
    """Run fixed-step block PDHG with exactly one physical pair per iteration."""

    count = int(iterations)
    weight = float(regularization_weight)
    tau = float(primal_step)
    sigma_data = float(data_dual_step)
    sigma_edge = float(edge_dual_step)
    theta = float(extrapolation)
    scale = float(measurement_scale)
    if not isinstance(norm_estimate, BlockOperatorNormEstimate):
        raise TypeError("norm_estimate must come from estimate_block_operator_norms")
    data_bound = float(norm_estimate.data_norm_squared_upper)
    gradient_bound = float(norm_estimate.gradient_norm_squared_upper)
    data_values = torch.as_tensor(norm_estimate.data_norm_squared_by_sample)
    if count < 1:
        raise ValueError("iterations must be positive")
    checkpoints = tuple(
        sorted(
            {
                int(value)
                for value in (
                    checkpoint_iterations
                    if checkpoint_iterations is not None
                    else (count,)
                )
            }
        )
    )
    if (
        not checkpoints
        or checkpoints[0] < 1
        or checkpoints[-1] != count
    ):
        raise ValueError(
            "checkpoint_iterations must be positive and include iterations"
        )
    for name, value in (
        ("regularization_weight", weight),
        ("primal_step", tau),
        ("data_dual_step", sigma_data),
        ("edge_dual_step", sigma_edge),
        ("norm_estimate.data_norm_squared_upper", data_bound),
        ("norm_estimate.gradient_norm_squared_upper", gradient_bound),
        ("measurement_scale", scale),
    ):
        if not torch.isfinite(torch.as_tensor(value)):
            raise ValueError(f"{name} must be finite")
    if weight < 0.0:
        raise ValueError("regularization_weight must be nonnegative")
    if min(
        tau,
        sigma_data,
        sigma_edge,
        data_bound,
        gradient_bound,
        scale,
    ) <= 0.0:
        raise ValueError("steps and norm bounds must be positive")
    if data_values.shape != (len(observation_uv),):
        raise ValueError("norm_estimate samplewise data bounds do not match batch")
    if not bool(torch.all(torch.isfinite(data_values))) or bool(
        torch.any(data_values <= 0.0)
    ):
        raise ValueError("norm_estimate samplewise data bounds must be finite and positive")
    if not math.isclose(
        data_bound,
        float(torch.max(data_values)),
        rel_tol=1e-6,
        abs_tol=0.0,
    ):
        raise ValueError("norm_estimate scalar data bound is internally inconsistent")
    if (
        int(norm_estimate.power_iterations) < 2
        or int(norm_estimate.forward_calls) != int(norm_estimate.power_iterations)
        or int(norm_estimate.adjoint_calls) != int(norm_estimate.power_iterations)
        or not math.isfinite(float(norm_estimate.safety_factor))
        or float(norm_estimate.safety_factor) <= 1.0
    ):
        raise ValueError("norm_estimate setup ledger is internally inconsistent")
    if not 0.0 <= theta <= 1.0:
        raise ValueError("extrapolation must lie in [0,1]")
    active, sigma = _prewhitened_measurement_terms(
        observation_uv,
        sigma_by_view=sigma_by_view,
        view_mask=view_mask,
        rays_per_view=rays_per_view,
    )
    support = _validated_dirichlet_support(
        operator,
        reference=observation_uv,
    )
    spacing_xyz = tuple(float(value) for value in operator.spacing_xyz)
    if norm_estimate.operator_identity != id(operator):
        raise ValueError("norm_estimate belongs to a different operator instance")
    if norm_estimate.operator_state_token != _operator_state_token(operator):
        raise ValueError("operator state changed after norm estimation")
    if norm_estimate.batch_size != len(observation_uv):
        raise ValueError("norm_estimate batch does not match observations")
    if norm_estimate.measurement_scale != scale:
        raise ValueError("norm_estimate measurement scale does not match solve")
    if norm_estimate.grid_shape != tuple(int(v) for v in operator.grid_shape):
        raise ValueError("norm_estimate grid does not match solve")
    if norm_estimate.spacing_xyz != spacing_xyz:
        raise ValueError("norm_estimate spacing does not match solve")
    if gradient_bound < gradient_operator_norm_squared_bound(spacing_xyz):
        raise ValueError("norm_estimate gradient bound is below the analytic bound")
    effective_gradient_bound = gradient_bound if weight > 0.0 else 0.0
    contract = tau * (
        sigma_data * data_bound
        + sigma_edge * effective_gradient_bound
    )
    if not contract < 1.0:
        raise ValueError(
            "PDHG step contract requires tau*(sigma_data*Ld + "
            "sigma_edge*Lg) < 1"
        )

    if initial_volume is None:
        current = torch.zeros(
            (len(observation_uv), 1, *operator.grid_shape),
            dtype=observation_uv.dtype,
            device=observation_uv.device,
        )
    else:
        current = torch.as_tensor(initial_volume).to(observation_uv)
        if current.shape != (
            len(observation_uv),
            1,
            *operator.grid_shape,
        ):
            raise ValueError("initial_volume does not match the inverse batch")
        current = current.clone()
    current = current * support
    extrapolated = current.clone()
    target = active * observation_uv / sigma
    initial_objective = torch.sum(target.square(), dim=(1, 2)).clamp_min(
        float(denominator_floor)
    )
    if initial_data_dual is None:
        data_dual = torch.zeros_like(observation_uv)
    else:
        data_dual = torch.as_tensor(initial_data_dual).to(observation_uv)
        if data_dual.shape != observation_uv.shape:
            raise ValueError("initial_data_dual must match observation_uv")
        data_dual = active * data_dual
    edge_dual = torch.zeros(
        (len(observation_uv), 3, *operator.grid_shape),
        dtype=observation_uv.dtype,
        device=observation_uv.device,
    )
    history: list[dict[str, torch.Tensor]] = []
    checkpoint_volumes: dict[int, torch.Tensor] = {}
    local_regularizer = (
        ForwardNeumannRegularizationOperator(spacing_xyz)
        if regularization_operator is None
        else regularization_operator
    )
    if not callable(local_regularizer):
        raise TypeError("regularization_operator must be callable")
    if not callable(getattr(local_regularizer, "adjoint", None)):
        raise TypeError("regularization_operator must expose adjoint")
    if not callable(getattr(local_regularizer, "call_report", None)):
        raise TypeError("regularization_operator must expose call_report")
    before_calls = operator.call_report()
    before_regularization_calls = dict(local_regularizer.call_report())

    for iteration in range(count):
        previous_volume = current
        projected = scale * active * operator(extrapolated) / sigma
        extrapolated_residual = projected - target
        previous_data_dual = data_dual
        data_dual = (
            data_dual + sigma_data * extrapolated_residual
        ) / (1.0 + sigma_data)
        data_dual_update = data_dual - previous_data_dual
        gradient = local_regularizer(extrapolated[:, 0])
        previous_edge_dual = edge_dual
        edge_dual = proximal_edge_conjugate(
            edge_dual + sigma_edge * gradient,
            regularization_weight=weight,
            dual_step=sigma_edge,
            penalty=penalty,
            huber_delta=huber_delta,
        )
        edge_dual_update = edge_dual - previous_edge_dual
        data_normal = scale * operator.adjoint(active * data_dual / sigma)
        edge_normal = local_regularizer.adjoint(edge_dual)[:, None]
        next_volume = (
            current - tau * (data_normal + edge_normal)
        ) * support
        update = next_volume - current
        extrapolated = (
            next_volume + theta * update
        ) * support
        current = next_volume
        extrapolated_edge_value = _edge_penalty_from_gradient(
            gradient,
            penalty=penalty,
            huber_delta=huber_delta,
        )
        edge_norm = torch.linalg.vector_norm(
            edge_dual,
            dim=1,
        )
        primal_metric_increment = torch.linalg.vector_norm(
            update.flatten(1),
            dim=1,
        ) / math.sqrt(tau)
        data_dual_metric_increment = torch.linalg.vector_norm(
            data_dual_update.flatten(1),
            dim=1,
        ) / math.sqrt(sigma_data)
        edge_dual_metric_increment = torch.linalg.vector_norm(
            edge_dual_update.flatten(1),
            dim=1,
        ) / math.sqrt(sigma_edge)
        previous_metric_norm = torch.sqrt(
            torch.sum(previous_volume.flatten(1).square(), dim=1) / tau
            + torch.sum(previous_data_dual.flatten(1).square(), dim=1)
            / sigma_data
            + torch.sum(previous_edge_dual.flatten(1).square(), dim=1)
            / sigma_edge
        )
        next_metric_norm = torch.sqrt(
            torch.sum(current.flatten(1).square(), dim=1) / tau
            + torch.sum(data_dual.flatten(1).square(), dim=1) / sigma_data
            + torch.sum(edge_dual.flatten(1).square(), dim=1) / sigma_edge
        )
        joint_metric_increment = torch.sqrt(
            primal_metric_increment.square()
            + data_dual_metric_increment.square()
            + edge_dual_metric_increment.square()
        )
        normalized_joint_fixed_point = joint_metric_increment / torch.maximum(
            torch.maximum(previous_metric_norm, next_metric_norm),
            torch.full_like(next_metric_norm, 1e-12),
        )
        history.append(
            {
                "iteration": torch.full(
                    (len(observation_uv),),
                    iteration + 1,
                    dtype=torch.int64,
                    device=observation_uv.device,
                ),
                "extrapolated_relative_data_objective": torch.sum(
                    extrapolated_residual.square(),
                    dim=(1, 2),
                )
                / initial_objective,
                "primal_update_norm": torch.linalg.vector_norm(
                    update.flatten(1),
                    dim=1,
                ),
                "primal_metric_increment_norm": primal_metric_increment,
                "data_dual_metric_increment_norm": (
                    data_dual_metric_increment
                ),
                "edge_dual_metric_increment_norm": (
                    edge_dual_metric_increment
                ),
                "previous_metric_state_norm": previous_metric_norm,
                "next_metric_state_norm": next_metric_norm,
                "normalized_joint_fixed_point_residual": (
                    normalized_joint_fixed_point
                ),
                "data_dual_norm": torch.linalg.vector_norm(
                    data_dual.flatten(1),
                    dim=1,
                ),
                "edge_dual_maximum_norm": torch.amax(
                    edge_norm.flatten(1),
                    dim=1,
                ),
                "extrapolated_edge_penalty": extrapolated_edge_value,
                "extrapolated_total_objective": (
                    0.5
                    * torch.sum(
                        extrapolated_residual.square(),
                        dim=(1, 2),
                    )
                    + weight * extrapolated_edge_value
                ),
            }
        )
        completed = iteration + 1
        if completed in checkpoints:
            checkpoint_volumes[completed] = current.clone()
        if not bool(torch.all(torch.isfinite(current))):
            raise FloatingPointError("PDHG produced a non-finite primal iterate")
        if not bool(torch.all(torch.isfinite(data_dual))):
            raise FloatingPointError("PDHG produced a non-finite data dual")
        if not bool(torch.all(torch.isfinite(edge_dual))):
            raise FloatingPointError("PDHG produced a non-finite edge dual")

    after_calls = operator.call_report()
    forward_calls = int(
        after_calls["forward_calls"] - before_calls["forward_calls"]
    )
    adjoint_calls = int(
        after_calls["adjoint_calls"] - before_calls["adjoint_calls"]
    )
    after_regularization_calls = dict(local_regularizer.call_report())
    try:
        gradient_calls = int(
            after_regularization_calls["gradient_calls"]
            - before_regularization_calls["gradient_calls"]
        )
        gradient_adjoint_calls = int(
            after_regularization_calls["gradient_adjoint_calls"]
            - before_regularization_calls["gradient_adjoint_calls"]
        )
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            "regularization_operator call_report must contain numeric "
            "gradient_calls and gradient_adjoint_calls"
        ) from error
    expected = {
        "forward_calls": count,
        "adjoint_calls": count,
        "gradient_calls": count,
        "gradient_adjoint_calls": count,
    }
    observed = {
        "forward_calls": forward_calls,
        "adjoint_calls": adjoint_calls,
        "gradient_calls": gradient_calls,
        "gradient_adjoint_calls": gradient_adjoint_calls,
    }
    if observed != expected:
        raise RuntimeError(
            f"PDHG call contract violated: expected {expected}, observed {observed}"
        )
    return PrimalDualReconstruction(
        volume=current,
        data_dual=data_dual,
        edge_dual=edge_dual,
        history=history,
        checkpoint_volumes=checkpoint_volumes,
        forward_calls=forward_calls,
        adjoint_calls=adjoint_calls,
        gradient_calls=gradient_calls,
        gradient_adjoint_calls=gradient_adjoint_calls,
        step_contract_value=float(contract),
    )


__all__ = [
    "BlockOperatorNormEstimate",
    "ForwardNeumannRegularizationOperator",
    "PRIMAL_DUAL_SCHEMA",
    "PrimalDualReconstruction",
    "estimate_block_operator_norms",
    "gradient_operator_norm_squared_bound",
    "isotropic_edge_penalty",
    "primal_dual_reconstruction",
    "proximal_edge_conjugate",
    "regularization_gradient",
    "regularization_gradient_adjoint",
]
