"""Matrix-free measurement-space filtering for learned 3-D corrections.

Given a reference reconstruction ``x_ref`` and a learned proposal ``x_learned``,
this module approximately removes the component of
``delta = x_learned - x_ref`` that is visible to a declared linear map ``A``.
It runs fixed-step (preconditioned) conjugate gradients on

``(A A^T + damping * I) z = A delta``

and returns ``x_ref + delta - A^T z``.  With zero damping and an exact solve,
the retained correction belongs to the numerical kernel of ``A``.  Any finite
iteration result is only an approximate data-consistency filter, and the
kernel of a discretized inverse operator is not the true optical null space.

The implementation never accepts a truth field or an evaluation score.  It
also accumulates ``A^T z`` inside the recurrence, avoiding an uncounted final
adjoint call.  For ``K`` fixed iterations the exact logical budget is
``K + 1`` forward calls and ``K`` adjoint calls.  An optional affine mode
replaces the right-hand side by ``A x_learned - y`` and therefore projects the
learned proposal toward the measured-data set instead of toward ``A x_ref``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import math
from typing import Any

import torch


Tensor = torch.Tensor


@dataclass(frozen=True)
class MatrixFreeProjectionPath:
    """Selected iterates and a complete physical-call ledger."""

    fields_by_iteration: dict[int, Tensor]
    retained_corrections_by_iteration: dict[int, Tensor]
    removed_corrections_by_iteration: dict[int, Tensor]
    system_residuals_by_iteration: dict[int, Tensor]
    duals_by_iteration: dict[int, Tensor]
    history: list[dict[str, Any]]
    forward_calls: int
    adjoint_calls: int
    damping: float
    preconditioner: str
    preconditioner_applications: int
    target_mode: str
    fixed_iterations: int


def _validated_iterations(values: Iterable[int]) -> tuple[int, ...]:
    parsed: list[int] = []
    for raw in values:
        if isinstance(raw, bool):
            raise ValueError("snapshot iterations must be non-negative integers")
        try:
            value = int(raw)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "snapshot iterations must be non-negative integers"
            ) from error
        if value != raw or value < 0:
            raise ValueError("snapshot iterations must be non-negative integers")
        parsed.append(value)
    if not parsed:
        raise ValueError("snapshot_iterations cannot be empty")
    if len(set(parsed)) != len(parsed):
        raise ValueError("snapshot_iterations cannot contain duplicates")
    if 0 not in parsed:
        raise ValueError("snapshot_iterations must include zero")
    return tuple(sorted(parsed))


def _validated_field(value: Tensor, *, name: str) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 3 or value.numel() == 0:
        raise ValueError(f"{name} must be one nonempty three-dimensional tensor")
    if value.is_complex() or not value.dtype.is_floating_point:
        raise TypeError(f"{name} must use a real floating dtype")
    if not bool(torch.all(torch.isfinite(value))):
        raise ValueError(f"{name} must contain only finite values")
    return value


def _validated_support(support: Tensor, *, reference: Tensor) -> Tensor:
    values = torch.as_tensor(
        support,
        dtype=reference.dtype,
        device=reference.device,
    )
    if tuple(values.shape) != tuple(reference.shape):
        raise ValueError("support must match the field shape")
    if not bool(torch.all(torch.isfinite(values))):
        raise ValueError("support must contain only finite values")
    if not bool(torch.all((values == 0.0) | (values == 1.0))):
        raise ValueError("support must be binary")
    if not bool(torch.any(values > 0.5)):
        raise ValueError("support must retain at least one voxel")
    return values


def _finite_nonnegative(value: float, *, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return parsed


class _CountedPair:
    def __init__(
        self,
        *,
        forward: Callable[[Tensor], Tensor],
        adjoint: Callable[[Tensor], Tensor],
        field_reference: Tensor,
    ) -> None:
        if not callable(forward) or not callable(adjoint):
            raise TypeError("forward and adjoint must be callable")
        self._forward = forward
        self._adjoint = adjoint
        self._field_reference = field_reference
        self._measurement_reference: Tensor | None = None
        self.forward_calls = 0
        self.adjoint_calls = 0

    @staticmethod
    def _validate_output(value: Tensor, *, name: str) -> Tensor:
        if not isinstance(value, Tensor) or value.numel() == 0:
            raise TypeError(f"{name} must return one nonempty tensor")
        if value.is_complex() or not value.dtype.is_floating_point:
            raise TypeError(f"{name} must return a real floating tensor")
        if not bool(torch.all(torch.isfinite(value))):
            raise FloatingPointError(f"{name} returned non-finite values")
        return value

    def forward(self, field: Tensor) -> Tensor:
        self.forward_calls += 1
        value = self._validate_output(self._forward(field), name="forward")
        if self._measurement_reference is None:
            self._measurement_reference = value
        elif (
            value.shape != self._measurement_reference.shape
            or value.dtype != self._measurement_reference.dtype
            or value.device != self._measurement_reference.device
        ):
            raise ValueError("forward must preserve measurement shape, dtype and device")
        return value

    def adjoint(self, measurement: Tensor) -> Tensor:
        if self._measurement_reference is None:
            raise RuntimeError("forward must be called before adjoint")
        if (
            measurement.shape != self._measurement_reference.shape
            or measurement.dtype != self._measurement_reference.dtype
            or measurement.device != self._measurement_reference.device
        ):
            raise ValueError("adjoint input must match the forward output contract")
        self.adjoint_calls += 1
        value = self._validate_output(self._adjoint(measurement), name="adjoint")
        if (
            value.shape != self._field_reference.shape
            or value.dtype != self._field_reference.dtype
            or value.device != self._field_reference.device
        ):
            raise ValueError("adjoint must preserve field shape, dtype and device")
        return value


def _dot(left: Tensor, right: Tensor) -> Tensor:
    return torch.sum(left * right)


@torch.no_grad()
def matrix_free_measurement_projection_path(
    *,
    reference_field: Tensor,
    learned_field: Tensor,
    forward: Callable[[Tensor], Tensor],
    adjoint: Callable[[Tensor], Tensor],
    support: Tensor,
    snapshot_iterations: Iterable[int],
    damping: float = 0.0,
    preconditioner_diagonal: Tensor | None = None,
    preconditioner_apply: Callable[[Tensor], Tensor] | None = None,
    preconditioner_name: str | None = None,
    target_observation: Tensor | None = None,
    denominator_floor: float = 1e-30,
) -> MatrixFreeProjectionPath:
    """Return fixed-step matrix-free projections of one learned correction.

    ``preconditioner_diagonal`` represents a positive diagonal approximation to
    ``A A^T + damping I`` in measurement space.  Supplying it changes only the
    local algebra; any physical calls used to estimate it must be accounted for
    separately by the caller.

    A numerical convergence or breakdown never shortens the run.  Remaining
    iterations still execute the declared zero directions so the physical-call
    budget cannot become sample dependent.

    With ``target_observation=None``, the system right-hand side is
    ``A (x_learned - x_ref)`` and the final field approaches the affine set
    ``A x = A x_ref``.  Supplying ``target_observation=y`` instead uses
    ``A x_learned - y`` and approaches ``A x = y``.  The latter still does not
    imply consistency with an independent renderer or the real optical system.
    """

    reference = _validated_field(reference_field, name="reference_field")
    learned = _validated_field(learned_field, name="learned_field")
    if (
        learned.shape != reference.shape
        or learned.dtype != reference.dtype
        or learned.device != reference.device
    ):
        raise ValueError(
            "learned_field must match reference_field shape, dtype and device"
        )
    mask = _validated_support(support, reference=reference)
    snapshots = _validated_iterations(snapshot_iterations)
    maximum_iteration = max(snapshots)
    damping_value = _finite_nonnegative(damping, name="damping")
    floor = float(denominator_floor)
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("denominator_floor must be positive and finite")

    pair = _CountedPair(
        forward=forward,
        adjoint=adjoint,
        field_reference=reference,
    )
    reference = reference * mask
    learned = learned * mask
    correction = (learned - reference) * mask
    if target_observation is None:
        right_hand_side = pair.forward(correction)
        target_mode = "reference_reprojection"
    else:
        learned_projection = pair.forward(learned)
        target = torch.as_tensor(
            target_observation,
            dtype=learned_projection.dtype,
            device=learned_projection.device,
        )
        if target.shape != learned_projection.shape:
            raise ValueError("target_observation must match the forward output shape")
        if not bool(torch.all(torch.isfinite(target))):
            raise ValueError("target_observation must contain only finite values")
        right_hand_side = learned_projection - target
        target_mode = "affine_observation"
    dual = torch.zeros_like(right_hand_side)
    system_residual = right_hand_side.detach().clone()
    removed = torch.zeros_like(correction)

    if preconditioner_diagonal is not None and preconditioner_apply is not None:
        raise ValueError(
            "preconditioner_diagonal and preconditioner_apply are mutually exclusive"
        )
    if preconditioner_diagonal is None and preconditioner_apply is None:
        diagonal = None
        preconditioner_name = "identity"
    elif preconditioner_diagonal is not None:
        diagonal = torch.as_tensor(
            preconditioner_diagonal,
            dtype=right_hand_side.dtype,
            device=right_hand_side.device,
        )
        if diagonal.shape != right_hand_side.shape:
            raise ValueError(
                "preconditioner_diagonal must match the forward output shape"
            )
        if not bool(torch.all(torch.isfinite(diagonal))) or not bool(
            torch.all(diagonal > 0.0)
        ):
            raise ValueError("preconditioner_diagonal must be positive and finite")
        preconditioner_name = "supplied_positive_diagonal"
    else:
        diagonal = None
        if not callable(preconditioner_apply):
            raise TypeError("preconditioner_apply must be callable")
        if not isinstance(preconditioner_name, str) or not preconditioner_name.strip():
            raise ValueError(
                "preconditioner_name is required with preconditioner_apply"
            )
        preconditioner_name = preconditioner_name.strip()

    preconditioner_applications = 0
    def precondition(value: Tensor) -> Tensor:
        nonlocal preconditioner_applications
        preconditioner_applications += 1
        if preconditioner_apply is not None:
            result = preconditioner_apply(value)
            if not isinstance(result, Tensor):
                raise TypeError("preconditioner_apply must return a tensor")
            if (
                result.shape != value.shape
                or result.dtype != value.dtype
                or result.device != value.device
            ):
                raise ValueError(
                    "preconditioner_apply must preserve shape, dtype and device"
                )
            if not bool(torch.all(torch.isfinite(result))):
                raise FloatingPointError(
                    "preconditioner_apply returned non-finite values"
                )
            return result
        return value if diagonal is None else value / diagonal

    preconditioned = precondition(system_residual)
    direction = preconditioned.detach().clone()
    gamma = _dot(system_residual, preconditioned)
    initial_norm = float(torch.linalg.vector_norm(right_hand_side))
    norm_floor = math.sqrt(floor)
    inactive = initial_norm <= norm_floor
    breakdown_seen = False

    fields: dict[int, Tensor] = {}
    retained_values: dict[int, Tensor] = {}
    removed_values: dict[int, Tensor] = {}
    residual_values: dict[int, Tensor] = {}
    dual_values: dict[int, Tensor] = {}
    history: list[dict[str, Any]] = []

    def record(iteration: int, *, alpha: float | None, beta: float | None) -> None:
        retained = (correction - removed) * mask
        field = (reference + retained) * mask
        residual_norm = float(torch.linalg.vector_norm(system_residual))
        history.append(
            {
                "iteration": iteration,
                "system_residual_norm": residual_norm,
                "relative_system_residual": residual_norm
                / max(initial_norm, norm_floor),
                "alpha": alpha,
                "beta": beta,
                "converged": bool(inactive and not breakdown_seen),
                "breakdown": bool(breakdown_seen),
            }
        )
        if iteration in snapshots:
            fields[iteration] = field.detach().clone()
            retained_values[iteration] = retained.detach().clone()
            removed_values[iteration] = removed.detach().clone()
            residual_values[iteration] = system_residual.detach().clone()
            dual_values[iteration] = dual.detach().clone()

    record(0, alpha=None, beta=None)
    for iteration in range(1, maximum_iteration + 1):
        applications_before = preconditioner_applications
        active_direction = direction if not inactive else torch.zeros_like(direction)
        pulled_back = pair.adjoint(active_direction) * mask
        projected = pair.forward(pulled_back)
        system_direction = projected + damping_value * active_direction
        denominator = _dot(active_direction, system_direction)
        gamma_value = float(gamma)
        denominator_value = float(denominator)

        invalid_step = (
            not math.isfinite(gamma_value)
            or not math.isfinite(denominator_value)
            or gamma_value < -floor
            or denominator_value <= floor
        )
        alpha_value = 0.0
        beta_value = 0.0
        if not inactive and invalid_step:
            breakdown_seen = True
            inactive = True
        elif not inactive:
            alpha_value = gamma_value / denominator_value
            dual = dual + alpha_value * active_direction
            removed = (removed + alpha_value * pulled_back) * mask
            system_residual = system_residual - alpha_value * system_direction
            next_preconditioned = precondition(system_residual)
            next_gamma = _dot(system_residual, next_preconditioned)
            next_gamma_value = float(next_gamma)
            residual_norm = float(torch.linalg.vector_norm(system_residual))
            if residual_norm <= norm_floor:
                inactive = True
                direction = torch.zeros_like(direction)
                gamma = next_gamma
            elif (
                not math.isfinite(next_gamma_value)
                or next_gamma_value < -floor
                or gamma_value <= floor
            ):
                breakdown_seen = True
                inactive = True
                direction = torch.zeros_like(direction)
                gamma = next_gamma
            else:
                beta_value = next_gamma_value / gamma_value
                direction = next_preconditioned + beta_value * active_direction
                gamma = next_gamma
        if preconditioner_applications == applications_before:
            # Preserve a fixed algebraic-preconditioner budget after numerical
            # convergence or breakdown.  The padded zero result is discarded.
            precondition(torch.zeros_like(system_residual))
        if preconditioner_applications != applications_before + 1:
            raise RuntimeError("preconditioner application count drifted")
        record(iteration, alpha=alpha_value, beta=beta_value)

    expected_forward = maximum_iteration + 1
    expected_adjoint = maximum_iteration
    if (
        pair.forward_calls != expected_forward
        or pair.adjoint_calls != expected_adjoint
    ):
        raise RuntimeError(
            "matrix-free projection call contract drifted: expected "
            f"{expected_forward} forward/{expected_adjoint} adjoint, observed "
            f"{pair.forward_calls} forward/{pair.adjoint_calls} adjoint"
        )
    if set(fields) != set(snapshots):
        raise RuntimeError("matrix-free projection did not materialize every snapshot")
    if preconditioner_applications != maximum_iteration + 1:
        raise RuntimeError("fixed preconditioner application budget drifted")
    return MatrixFreeProjectionPath(
        fields_by_iteration=fields,
        retained_corrections_by_iteration=retained_values,
        removed_corrections_by_iteration=removed_values,
        system_residuals_by_iteration=residual_values,
        duals_by_iteration=dual_values,
        history=history,
        forward_calls=pair.forward_calls,
        adjoint_calls=pair.adjoint_calls,
        damping=damping_value,
        preconditioner=preconditioner_name,
        preconditioner_applications=preconditioner_applications,
        target_mode=target_mode,
        fixed_iterations=maximum_iteration,
    )


__all__ = [
    "MatrixFreeProjectionPath",
    "matrix_free_measurement_projection_path",
]
