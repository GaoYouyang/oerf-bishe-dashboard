"""Purely statistical routing primitives for paired low/high-fidelity replays.

This module deliberately knows nothing about rays, renderers, reconstruction models, or
physical validity certificates.  A caller must construct ``unsafe_mask`` outside this
module.  The routines below only enforce the statistical contract:

* every replay has a detached inclusion probability in ``[pi_min, 1]``;
* every externally certified unsafe replay is evaluated at high fidelity (``pi = 1``);
* the Horvitz--Thompson correction is unbiased conditional on fixed replay tensors and
  fixed probabilities;
* probability allocation is a deterministic, fail-closed budget calculation.

Passing these checks is not evidence of training success, physical correctness,
generalization, or research novelty.
"""

from __future__ import annotations

from typing import Literal

import torch


__all__ = [
    "RoutingValidationError",
    "allocate_inclusion_probabilities",
    "conditional_trace_variance",
    "horvitz_thompson_mean",
    "horvitz_thompson_sparse_mean",
    "two_replica_quadratic_loss",
    "validate_inclusion_probabilities",
]


class RoutingValidationError(ValueError):
    """Raised when a routing contract is invalid or cannot be satisfied safely."""


def _require_floating_vector(name: str, value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise RoutingValidationError(f"{name} must be a torch.Tensor")
    if value.ndim != 1 or value.numel() == 0:
        raise RoutingValidationError(
            f"{name} must be a non-empty one-dimensional tensor"
        )
    if not value.is_floating_point():
        raise RoutingValidationError(f"{name} must have a floating-point dtype")
    if not torch.all(torch.isfinite(value)):
        raise RoutingValidationError(f"{name} must contain only finite values")
    return value


def _prepare_unsafe_mask(
    unsafe_mask: torch.Tensor | None,
    *,
    sample_count: int,
    device: torch.device,
) -> torch.Tensor:
    if unsafe_mask is None:
        return torch.zeros(sample_count, dtype=torch.bool, device=device)
    if not isinstance(unsafe_mask, torch.Tensor):
        raise RoutingValidationError("unsafe_mask must be a torch.Tensor or None")
    if unsafe_mask.dtype != torch.bool:
        raise RoutingValidationError("unsafe_mask must have boolean dtype")
    if unsafe_mask.ndim != 1 or unsafe_mask.shape[0] != sample_count:
        raise RoutingValidationError(
            f"unsafe_mask must have shape ({sample_count},), got {tuple(unsafe_mask.shape)}"
        )
    return unsafe_mask.to(device=device)


def _validate_probability_floor(pi_min: float) -> float:
    try:
        floor = float(pi_min)
    except (TypeError, ValueError) as error:
        raise RoutingValidationError("pi_min must be a finite scalar") from error
    if not torch.isfinite(torch.tensor(floor)) or not 0.0 < floor <= 1.0:
        raise RoutingValidationError("pi_min must satisfy 0 < pi_min <= 1")
    return floor


def validate_inclusion_probabilities(
    inclusion_probabilities: torch.Tensor,
    *,
    pi_min: float,
    unsafe_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Validate and return detached per-replay inclusion probabilities.

    Probabilities are always detached here.  Learning through a sampled routing
    probability would require a different gradient estimator and is outside this
    Horvitz--Thompson contract.  Unsafe entries must be exactly one, not merely close
    to one, so that they cannot be skipped by a uniform draw.
    """

    probabilities = _require_floating_vector(
        "inclusion_probabilities", inclusion_probabilities
    )
    floor = _validate_probability_floor(pi_min)
    unsafe = _prepare_unsafe_mask(
        unsafe_mask,
        sample_count=probabilities.numel(),
        device=probabilities.device,
    )
    if torch.any(probabilities < floor):
        raise RoutingValidationError("all inclusion probabilities must be >= pi_min")
    if torch.any(probabilities > 1.0):
        raise RoutingValidationError("all inclusion probabilities must be <= 1")
    if torch.any(probabilities[unsafe] != 1.0):
        raise RoutingValidationError(
            "unsafe replays must have inclusion probability pi = 1"
        )
    return probabilities.detach()


def _prepare_replays(
    low_replay: torch.Tensor,
    high_replay: torch.Tensor,
    *,
    sample_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if not isinstance(low_replay, torch.Tensor) or not isinstance(
        high_replay, torch.Tensor
    ):
        raise RoutingValidationError("low_replay and high_replay must be torch tensors")
    if low_replay.shape != high_replay.shape:
        raise RoutingValidationError(
            "low_replay and high_replay must have identical shapes"
        )
    if low_replay.ndim == 0 or low_replay.numel() == 0:
        raise RoutingValidationError(
            "replay tensors must be non-empty and have a sample axis"
        )
    if not low_replay.is_floating_point() or not high_replay.is_floating_point():
        raise RoutingValidationError("replay tensors must have floating-point dtype")
    if low_replay.dtype != high_replay.dtype or low_replay.device != high_replay.device:
        raise RoutingValidationError("replay tensors must share dtype and device")
    if not torch.all(torch.isfinite(low_replay)) or not torch.all(
        torch.isfinite(high_replay)
    ):
        raise RoutingValidationError("replay tensors must contain only finite values")
    try:
        normalized_dim = sample_dim % low_replay.ndim
    except TypeError as error:
        raise RoutingValidationError("sample_dim must be an integer") from error
    if not isinstance(sample_dim, int):
        raise RoutingValidationError("sample_dim must be an integer")
    if not -low_replay.ndim <= sample_dim < low_replay.ndim:
        raise RoutingValidationError(
            f"sample_dim {sample_dim} is invalid for {low_replay.ndim} dimensions"
        )
    return (
        low_replay.movedim(normalized_dim, 0),
        high_replay.movedim(normalized_dim, 0),
        normalized_dim,
    )


def _prepare_uniforms(
    bernoulli_uniforms: torch.Tensor,
    *,
    sample_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(bernoulli_uniforms, torch.Tensor):
        raise RoutingValidationError("bernoulli_uniforms must be a torch.Tensor")
    if bernoulli_uniforms.ndim == 0 or bernoulli_uniforms.shape[-1] != sample_count:
        raise RoutingValidationError(
            "bernoulli_uniforms must have shape (..., sample_count), with samples last"
        )
    if not bernoulli_uniforms.is_floating_point():
        raise RoutingValidationError(
            "bernoulli_uniforms must have floating-point dtype"
        )
    if not torch.all(torch.isfinite(bernoulli_uniforms)):
        raise RoutingValidationError(
            "bernoulli_uniforms must contain only finite values"
        )
    if torch.any(bernoulli_uniforms < 0.0) or torch.any(bernoulli_uniforms >= 1.0):
        raise RoutingValidationError(
            "bernoulli_uniforms must lie in the half-open interval [0, 1)"
        )
    return bernoulli_uniforms.detach().to(device=device, dtype=dtype)


def horvitz_thompson_mean(
    low_replay: torch.Tensor,
    high_replay: torch.Tensor,
    inclusion_probabilities: torch.Tensor,
    bernoulli_uniforms: torch.Tensor,
    *,
    pi_min: float,
    unsafe_mask: torch.Tensor | None = None,
    sample_dim: int = 0,
) -> torch.Tensor:
    r"""Return the paired Horvitz--Thompson mean estimator.

    For fixed replay pairs ``(L_i, H_i)`` this computes

    ``mean_i[L_i + I_i / pi_i * (H_i - L_i)]``,

    where ``I_i = 1[U_i < pi_i]``.  Uniforms may have shape ``(N,)`` for one
    estimator or ``(..., N)`` for a batch of statistically independent replicas.
    The caller is responsible for supplying independent uniforms when independence
    is required.  Probabilities and uniforms are detached; gradients may still flow
    through the low/high replay values.
    """

    low, high, _ = _prepare_replays(low_replay, high_replay, sample_dim=sample_dim)
    sample_count = low.shape[0]
    probabilities = validate_inclusion_probabilities(
        inclusion_probabilities,
        pi_min=pi_min,
        unsafe_mask=unsafe_mask,
    )
    if probabilities.shape[0] != sample_count:
        raise RoutingValidationError(
            f"probabilities have {probabilities.shape[0]} entries, expected {sample_count}"
        )
    probabilities = probabilities.to(device=low.device, dtype=low.dtype)
    uniforms = _prepare_uniforms(
        bernoulli_uniforms,
        sample_count=sample_count,
        device=low.device,
        dtype=low.dtype,
    )

    inclusion = (uniforms < probabilities).to(dtype=low.dtype)
    weights = inclusion / probabilities
    replica_shape = weights.shape[:-1]
    output_shape = low.shape[1:]
    correction = weights.reshape(-1, sample_count) @ (high - low).reshape(
        sample_count, -1
    )
    correction = correction.reshape((*replica_shape, *output_shape)) / sample_count
    low_mean = low.mean(dim=0)
    if replica_shape:
        low_mean = low_mean.reshape(*(1 for _ in replica_shape), *output_shape)
    return low_mean + correction


def horvitz_thompson_sparse_mean(
    low_replay: torch.Tensor,
    selected_high_replay: torch.Tensor,
    selected_indices: torch.Tensor,
    inclusion_probabilities: torch.Tensor,
    *,
    pi_min: float,
    unsafe_mask: torch.Tensor | None = None,
    sample_dim: int = 0,
) -> torch.Tensor:
    r"""Evaluate an HT mean after computing high fidelity only where selected.

    This is the online execution counterpart of :func:`horvitz_thompson_mean`.
    ``selected_indices`` is the realized Bernoulli mask and
    ``selected_high_replay`` contains only those expensive outputs.  All unsafe
    indices must be present.  The function does not draw the mask, so the caller
    must preserve the declared independent Bernoulli design and its state hash.
    """

    if not isinstance(low_replay, torch.Tensor) or low_replay.ndim == 0:
        raise RoutingValidationError("low_replay must contain a sample axis")
    if not low_replay.is_floating_point() or not torch.all(torch.isfinite(low_replay)):
        raise RoutingValidationError("low_replay must be finite and floating point")
    if not isinstance(sample_dim, int) or not -low_replay.ndim <= sample_dim < low_replay.ndim:
        raise RoutingValidationError("sample_dim is invalid for low_replay")
    normalized_dim = sample_dim % low_replay.ndim
    low = low_replay.movedim(normalized_dim, 0)
    sample_count = low.shape[0]
    probabilities = validate_inclusion_probabilities(
        inclusion_probabilities,
        pi_min=pi_min,
        unsafe_mask=unsafe_mask,
    ).to(device=low.device, dtype=low.dtype)
    if probabilities.shape[0] != sample_count:
        raise RoutingValidationError(
            f"probabilities have {probabilities.shape[0]} entries, expected {sample_count}"
        )
    if not isinstance(selected_indices, torch.Tensor):
        raise RoutingValidationError("selected_indices must be a torch.Tensor")
    if selected_indices.dtype != torch.long or selected_indices.ndim != 1:
        raise RoutingValidationError("selected_indices must be a one-dimensional int64 tensor")
    indices = selected_indices.detach().to(device=low.device)
    if indices.numel() and (
        torch.any(indices < 0) or torch.any(indices >= sample_count)
    ):
        raise RoutingValidationError("selected_indices contain an out-of-range entry")
    if torch.unique(indices).numel() != indices.numel():
        raise RoutingValidationError("selected_indices must not contain duplicates")
    unsafe = _prepare_unsafe_mask(
        unsafe_mask,
        sample_count=sample_count,
        device=low.device,
    )
    if torch.any(unsafe):
        selected_mask = torch.zeros(sample_count, dtype=torch.bool, device=low.device)
        selected_mask[indices] = True
        if torch.any(unsafe & ~selected_mask):
            raise RoutingValidationError("every unsafe replay must be selected")

    if not isinstance(selected_high_replay, torch.Tensor):
        raise RoutingValidationError("selected_high_replay must be a torch.Tensor")
    expected_shape = (indices.numel(), *low.shape[1:])
    if tuple(selected_high_replay.shape) != expected_shape:
        raise RoutingValidationError(
            "selected_high_replay must have shape "
            f"{expected_shape}, got {tuple(selected_high_replay.shape)}"
        )
    if (
        selected_high_replay.dtype != low.dtype
        or selected_high_replay.device != low.device
        or not selected_high_replay.is_floating_point()
        or not torch.all(torch.isfinite(selected_high_replay))
    ):
        raise RoutingValidationError(
            "selected_high_replay must be finite and share low dtype/device"
        )
    if indices.numel() == 0:
        return low.mean(dim=0)
    selected_low = low.index_select(0, indices)
    correction = (selected_high_replay - selected_low) / probabilities.index_select(
        0, indices
    ).reshape((-1,) + (1,) * (low.ndim - 1))
    return low.mean(dim=0) + correction.sum(dim=0) / sample_count


def conditional_trace_variance(
    low_replay: torch.Tensor,
    high_replay: torch.Tensor,
    inclusion_probabilities: torch.Tensor,
    *,
    pi_min: float,
    unsafe_mask: torch.Tensor | None = None,
    sample_dim: int = 0,
) -> torch.Tensor:
    r"""Compute the exact conditional trace variance of the HT mean.

    Conditional on fixed replay pairs and independent Bernoulli decisions, the trace
    is

    ``N^-2 sum_i ((1 - pi_i) / pi_i) ||H_i - L_i||_2^2``.

    The result includes all non-sample tensor components in the squared norm.  This
    is an exact statistical identity, not a measured physical uncertainty.
    """

    low, high, _ = _prepare_replays(low_replay, high_replay, sample_dim=sample_dim)
    sample_count = low.shape[0]
    probabilities = validate_inclusion_probabilities(
        inclusion_probabilities,
        pi_min=pi_min,
        unsafe_mask=unsafe_mask,
    )
    if probabilities.shape[0] != sample_count:
        raise RoutingValidationError(
            f"probabilities have {probabilities.shape[0]} entries, expected {sample_count}"
        )
    probabilities = probabilities.to(device=low.device, dtype=low.dtype)
    residual_energy = (high - low).reshape(sample_count, -1).square().sum(dim=1)
    return torch.sum((1.0 - probabilities) / probabilities * residual_energy) / (
        sample_count**2
    )


def allocate_inclusion_probabilities(
    residual_risk: torch.Tensor,
    *,
    average_high_fidelity_budget: float,
    pi_floor: float,
    unsafe_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Allocate detached probabilities under an exact expected-cost budget.

    For equal incremental high-route costs, safe entries follow the clipped KKT form

    ``pi_i = clip(scale * residual_risk_i, pi_floor, 1)``.

    The positive scale is solved by bisection so the expected budget is exact.  This is
    variance-optimal only when ``residual_risk_i`` equals the true residual norm; with a
    cheap proxy it is a proxy allocation, not an oracle optimum.  If all eligible risks
    are zero, the budget is split uniformly.  Unsafe entries are fixed at one.

    The requested budget is the mean inclusion probability across *all* entries.  An
    invalid proxy or a budget below the mandatory unsafe-plus-floor cost raises
    ``RoutingValidationError`` instead of silently weakening the safety contract.
    Output probabilities are detached and must not be interpreted as a differentiable
    routing policy.
    """

    risk = _require_floating_vector("residual_risk", residual_risk)
    if torch.any(risk < 0.0):
        raise RoutingValidationError("residual_risk must be non-negative")
    floor = _validate_probability_floor(pi_floor)
    try:
        budget = float(average_high_fidelity_budget)
    except (TypeError, ValueError) as error:
        raise RoutingValidationError(
            "average_high_fidelity_budget must be a finite scalar"
        ) from error
    if not torch.isfinite(torch.tensor(budget)) or not 0.0 < budget <= 1.0:
        raise RoutingValidationError(
            "average_high_fidelity_budget must satisfy 0 < budget <= 1"
        )

    unsafe = _prepare_unsafe_mask(
        unsafe_mask,
        sample_count=risk.numel(),
        device=risk.device,
    )
    safe = ~unsafe
    work = torch.full(
        risk.shape,
        floor,
        dtype=torch.float64,
        device=risk.device,
    )
    work[unsafe] = 1.0
    target_sum = budget * risk.numel()
    minimum_sum = float(work.sum().item())
    tolerance = 64.0 * torch.finfo(torch.float64).eps * max(1.0, risk.numel())
    if target_sum < minimum_sum - tolerance:
        minimum_budget = minimum_sum / risk.numel()
        raise RoutingValidationError(
            "budget is infeasible: unsafe entries require pi=1 and safe entries require "
            f"pi>=pi_floor, so the minimum average budget is {minimum_budget:.17g}"
        )
    if not torch.any(safe):
        if abs(target_sum - risk.numel()) > tolerance:
            raise RoutingValidationError(
                "all entries are unsafe, so the budget must equal 1"
            )
        return work.to(dtype=risk.dtype).detach()

    remaining = max(0.0, target_sum - minimum_sum)
    risk64 = risk.detach().to(dtype=torch.float64)
    safe_indices = torch.nonzero(safe, as_tuple=False).flatten()
    safe_risk = risk64[safe_indices]
    positive = safe_risk > 0.0
    target_safe_sum = target_sum - float(torch.sum(work[unsafe]).item())
    if remaining > tolerance and not torch.any(positive):
        work[safe_indices] = target_safe_sum / safe_indices.numel()
    elif remaining > tolerance:
        positive_indices = safe_indices[positive]
        positive_risk = safe_risk[positive]
        zero_indices = safe_indices[~positive]
        maximum_positive_sum = float(positive_indices.numel())
        minimum_zero_sum = floor * float(zero_indices.numel())
        if target_safe_sum > maximum_positive_sum + minimum_zero_sum + tolerance:
            work[positive_indices] = 1.0
            work[zero_indices] = (
                target_safe_sum - maximum_positive_sum
            ) / max(1, zero_indices.numel())
        else:
            lower_scale = 0.0
            upper_scale = 1.0

            def positive_sum(scale: float) -> float:
                return float(
                    torch.clamp(
                        scale * positive_risk,
                        min=floor,
                        max=1.0,
                    ).sum()
                )

            positive_target = target_safe_sum - minimum_zero_sum
            while positive_sum(upper_scale) < positive_target - tolerance:
                upper_scale *= 2.0
                if not torch.isfinite(torch.tensor(upper_scale)):
                    raise RoutingValidationError(
                        "KKT scale search overflowed before meeting the budget"
                    )
            for _ in range(100):
                midpoint = 0.5 * (lower_scale + upper_scale)
                if positive_sum(midpoint) < positive_target:
                    lower_scale = midpoint
                else:
                    upper_scale = midpoint
            scale = 0.5 * (lower_scale + upper_scale)
            work[positive_indices] = torch.clamp(
                scale * positive_risk,
                min=floor,
                max=1.0,
            )

    # Remove round-off without changing unsafe entries or violating the floor/cap.
    discrepancy = target_sum - float(work.sum().item())
    if abs(discrepancy) > tolerance:
        if discrepancy > 0.0:
            candidates = torch.nonzero(safe & (work < 1.0), as_tuple=False).flatten()
            headroom = 1.0 - work[candidates]
        else:
            candidates = torch.nonzero(safe & (work > floor), as_tuple=False).flatten()
            headroom = work[candidates] - floor
        if candidates.numel() == 0 or float(headroom.sum().item()) + tolerance < abs(
            discrepancy
        ):
            raise RoutingValidationError(
                "round-off correction would violate routing constraints"
            )
        adjustment = abs(discrepancy) * headroom / headroom.sum()
        work[candidates] += adjustment if discrepancy > 0.0 else -adjustment

    output = work.to(dtype=risk.dtype).detach()
    return validate_inclusion_probabilities(output, pi_min=floor, unsafe_mask=unsafe)


def two_replica_quadratic_loss(
    replica_a: torch.Tensor,
    replica_b: torch.Tensor,
    target: torch.Tensor,
    *,
    reduction: Literal["none", "mean", "sum"] = "mean",
) -> torch.Tensor:
    r"""Estimate a quadratic loss using two independent unbiased replicas.

    The elementwise estimator is ``(replica_a - target) * (replica_b - target)``.
    If both replicas are independent and unbiased for the same quantity, its
    expectation equals the squared error of that quantity.  The estimate itself may
    be negative.  Independence is a caller-side sampling contract and cannot be
    inferred from tensor values or storage.
    """

    tensors = (replica_a, replica_b, target)
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise RoutingValidationError("replicas and target must be torch tensors")
    if replica_a.shape != replica_b.shape or replica_a.shape != target.shape:
        raise RoutingValidationError("replicas and target must have identical shapes")
    if replica_a.numel() == 0:
        raise RoutingValidationError("replicas and target must be non-empty")
    if not all(value.is_floating_point() for value in tensors):
        raise RoutingValidationError(
            "replicas and target must have floating-point dtype"
        )
    if not (replica_a.dtype == replica_b.dtype == target.dtype):
        raise RoutingValidationError("replicas and target must share dtype")
    if not (replica_a.device == replica_b.device == target.device):
        raise RoutingValidationError("replicas and target must share device")
    if not all(torch.all(torch.isfinite(value)) for value in tensors):
        raise RoutingValidationError(
            "replicas and target must contain only finite values"
        )

    elementwise = (replica_a - target) * (replica_b - target)
    if reduction == "none":
        return elementwise
    if reduction == "mean":
        return elementwise.mean()
    if reduction == "sum":
        return elementwise.sum()
    raise RoutingValidationError("reduction must be one of: 'none', 'mean', 'sum'")
