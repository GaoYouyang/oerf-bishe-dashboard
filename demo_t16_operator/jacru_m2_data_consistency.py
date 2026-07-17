"""Truth-free data-consistency paths for learned residual diagnostics.

The routines in this module do not train or score a model.  They accept an
already proposed field and apply one of two deterministic projected-gradient
paths using only a forward map, its adjoint, and measured observations:

``measurement_pullback`` moves the full prediction toward the measured data;
``base_nullspace_filter`` spectrally damps the observable component of a
learned correction while preserving components in the exact null space of the
map.  A finite number of steps is a near-null filter, not an exact projector.

Both paths use one forward and one adjoint call per iteration.  A step smaller
than ``2 / ||A||^2`` gives the usual Landweber descent guarantee for a linear
map when the declared support is part of that map.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import math

import torch


Tensor = torch.Tensor


@dataclass(frozen=True)
class DataConsistencyPath:
    """Selected iterates and the exact logical operator-call count."""

    fields_by_step: dict[int, Tensor]
    forward_calls: int
    adjoint_calls: int
    mode: str
    step_size: float


def _validated_steps(snapshot_steps: Iterable[int]) -> tuple[int, ...]:
    values: list[int] = []
    for raw in snapshot_steps:
        if isinstance(raw, bool):
            raise ValueError("snapshot steps must be non-negative integers")
        value = int(raw)
        if value != raw or value < 0:
            raise ValueError("snapshot steps must be non-negative integers")
        values.append(value)
    if not values:
        raise ValueError("snapshot_steps cannot be empty")
    if len(set(values)) != len(values):
        raise ValueError("snapshot_steps cannot contain duplicates")
    if 0 not in values:
        raise ValueError("snapshot_steps must include zero")
    return tuple(sorted(values))


def _validated_field(value: Tensor, *, name: str) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 3:
        raise ValueError(f"{name} must be one three-dimensional tensor")
    if value.is_complex() or not value.dtype.is_floating_point:
        raise TypeError(f"{name} must use a real floating dtype")
    if not bool(torch.all(torch.isfinite(value))):
        raise ValueError(f"{name} must contain only finite values")
    return value


def _validated_support(support: Tensor, *, reference: Tensor) -> Tensor:
    values = torch.as_tensor(support, device=reference.device, dtype=reference.dtype)
    if tuple(values.shape) != tuple(reference.shape):
        raise ValueError("support must match the field shape")
    if not bool(torch.all(torch.isfinite(values))):
        raise ValueError("support must contain only finite values")
    if not bool(torch.all((values == 0.0) | (values == 1.0))):
        raise ValueError("support must be binary")
    if not bool(torch.any(values > 0.5)):
        raise ValueError("support must retain at least one voxel")
    return values


def data_consistency_path(
    *,
    initial_field: Tensor,
    observation: Tensor,
    forward: Callable[[Tensor], Tensor],
    adjoint: Callable[[Tensor], Tensor],
    support: Tensor,
    step_size: float,
    operator_norm_squared_bound: float,
    snapshot_steps: Iterable[int],
    mode: str,
    base_field: Tensor | None = None,
) -> DataConsistencyPath:
    """Return selected truth-free Landweber/null-space-filter iterates.

    ``mode='measurement_pullback'`` minimizes ``||A x - y||^2`` from the
    learned field.  ``mode='base_nullspace_filter'`` minimizes
    ``||A (x - x_base)||^2`` and therefore retains any exact-null-space part of
    the learned correction.  The latter mode requires ``base_field``.
    """

    initial = _validated_field(initial_field, name="initial_field")
    steps = _validated_steps(snapshot_steps)
    tau = float(step_size)
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError("step_size must be positive and finite")
    norm_bound = float(operator_norm_squared_bound)
    if not math.isfinite(norm_bound) or norm_bound <= 0.0:
        raise ValueError("operator_norm_squared_bound must be positive and finite")
    if tau >= 2.0 / norm_bound:
        raise ValueError("step_size must be smaller than 2 / operator_norm_squared_bound")
    support_value = _validated_support(support, reference=initial)
    if not isinstance(observation, Tensor) or observation.is_complex():
        raise TypeError("observation must be one real tensor")
    if not observation.dtype.is_floating_point:
        raise TypeError("observation must use a floating dtype")
    if not bool(torch.all(torch.isfinite(observation))):
        raise ValueError("observation must contain only finite values")
    if mode not in {"measurement_pullback", "base_nullspace_filter"}:
        raise ValueError("unknown data-consistency mode")

    current = initial.detach().clone() * support_value
    base: Tensor | None = None
    if mode == "base_nullspace_filter":
        if base_field is None:
            raise ValueError("base_field is required for base_nullspace_filter")
        base = _validated_field(base_field, name="base_field").to(initial)
        if tuple(base.shape) != tuple(initial.shape):
            raise ValueError("base_field must match initial_field")
        base = base.detach().clone() * support_value
        correction = (current - base) * support_value

    fields = {0: current.detach().clone()}
    forward_calls = 0
    adjoint_calls = 0
    for step in range(1, max(steps) + 1):
        if mode == "measurement_pullback":
            residual = observation - forward(current)
            forward_calls += 1
            gradient = adjoint(residual)
            adjoint_calls += 1
            if tuple(gradient.shape) != tuple(current.shape):
                raise ValueError("adjoint output must match the field shape")
            current = support_value * (current + tau * gradient.to(current))
        else:
            assert base is not None
            projected_correction = forward(correction)
            forward_calls += 1
            gradient = adjoint(projected_correction)
            adjoint_calls += 1
            if tuple(gradient.shape) != tuple(correction.shape):
                raise ValueError("adjoint output must match the field shape")
            correction = support_value * (correction - tau * gradient.to(correction))
            current = support_value * (base + correction)
        if step in steps:
            if not bool(torch.all(torch.isfinite(current))):
                raise RuntimeError("data-consistency path produced non-finite values")
            fields[step] = current.detach().clone()

    return DataConsistencyPath(
        fields_by_step=fields,
        forward_calls=forward_calls,
        adjoint_calls=adjoint_calls,
        mode=mode,
        step_size=tau,
    )


__all__ = ["DataConsistencyPath", "data_consistency_path"]
