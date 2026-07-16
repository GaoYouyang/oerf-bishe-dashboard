"""Numerical adjoint checks and Landweber baselines for T16 v3k-C."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def mask_key(mask: np.ndarray) -> str:
    values = np.asarray(mask) > 0.5
    return "".join("1" if value else "0" for value in values)


def forward_project(volume: np.ndarray, operator: np.ndarray) -> np.ndarray:
    """Apply the depth-separable multi-view forward operator."""
    values = np.asarray(volume, dtype=np.float64)
    matrix = np.asarray(operator, dtype=np.float64)
    if values.ndim != 4 or matrix.ndim != 3:
        raise ValueError("forward_project expects [batch,depth,height,width] and [view,n,p]")
    flat = values.reshape(values.shape[0], values.shape[1], -1)
    if flat.shape[-1] != matrix.shape[-1]:
        raise ValueError("volume and forward operator have incompatible pixel counts")
    return np.einsum("vnp,bdp->bdvn", matrix, flat, optimize=True)


def masked_adjoint(
    residual: np.ndarray,
    operator: np.ndarray,
    view_masks: np.ndarray,
    volume_shape: tuple[int, int, int],
) -> np.ndarray:
    """Apply A^T after zeroing inactive camera residuals."""
    values = np.asarray(residual, dtype=np.float64)
    matrix = np.asarray(operator, dtype=np.float64)
    masks = np.asarray(view_masks, dtype=np.float64)
    if values.ndim != 4 or masks.ndim != 2:
        raise ValueError("masked_adjoint expects [batch,depth,view,n] and [batch,view]")
    if values.shape[0] != masks.shape[0] or values.shape[2] != masks.shape[1]:
        raise ValueError("residual and view masks are not aligned")
    weighted = values * masks[:, None, :, None]
    flat = np.einsum("vnp,bdvn->bdp", matrix, weighted, optimize=True)
    if flat.shape[-1] != int(np.prod(volume_shape[-2:])):
        raise ValueError("adjoint output and requested volume shape disagree")
    return flat.reshape(values.shape[0], *volume_shape)


def feasibility_project(volume: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Project onto the known nonnegative support used by the synthetic generator."""
    active = np.asarray(support, dtype=np.float64) > 0.0
    return np.clip(np.asarray(volume, dtype=np.float64), 0.0, None) * active


def geometry_normalization(
    operator: np.ndarray,
    view_masks: np.ndarray,
    diagonal_floor_relative: float = 1e-3,
) -> dict[str, dict[str, np.ndarray | float]]:
    """Return exact spectral constants and Jacobi diagonals for each camera mask."""
    matrix = np.asarray(operator, dtype=np.float64)
    output: dict[str, dict[str, np.ndarray | float]] = {}
    for mask in np.asarray(view_masks, dtype=np.float64):
        key = mask_key(mask)
        if key in output:
            continue
        selected = np.flatnonzero(mask > 0.5)
        if not len(selected):
            raise ValueError("geometry normalization received an empty camera mask")
        active = matrix[selected].reshape(-1, matrix.shape[-1])
        singular = np.linalg.svd(active, compute_uv=False)
        spectral = float(singular[0] ** 2)
        diagonal = np.sum(active * active, axis=0)
        positive = diagonal[diagonal > 0]
        if not len(positive):
            raise ValueError("normal operator diagonal is identically zero")
        floor = float(diagonal_floor_relative) * float(np.mean(positive))
        diagonal = np.maximum(diagonal, floor)
        whitened = active / np.sqrt(diagonal)[None]
        preconditioned_spectral = float(np.linalg.svd(whitened, compute_uv=False)[0] ** 2)
        output[key] = {
            "active_camera_count": float(len(selected)),
            "spectral_constant": spectral,
            "jacobi_diagonal": diagonal,
            "jacobi_floor": floor,
            "preconditioned_spectral_constant": preconditioned_spectral,
        }
    return output


def landweber_trajectory(
    start: np.ndarray,
    observation: np.ndarray,
    operator: np.ndarray,
    view_masks: np.ndarray,
    support: np.ndarray,
    step_fraction: float,
    checkpoints: Iterable[int],
    normalization: dict[str, dict[str, np.ndarray | float]],
    method: str = "standard",
) -> dict[int, np.ndarray]:
    """Run projected, geometry-normalized Landweber iterations."""
    if method not in {"standard", "jacobi"}:
        raise ValueError(f"unknown Landweber method: {method}")
    beta = float(step_fraction)
    if not 0.0 < beta < 2.0:
        raise ValueError("normalized Landweber step fraction must lie in (0, 2)")
    requested = sorted({int(value) for value in checkpoints})
    if not requested or requested[0] < 1:
        raise ValueError("Landweber checkpoints must be positive integers")
    masks = np.asarray(view_masks, dtype=np.float64)
    keys = [mask_key(mask) for mask in masks]
    spectral = np.asarray(
        [float(normalization[key]["spectral_constant"]) for key in keys]
    )[:, None, None, None]
    preconditioned = np.asarray(
        [float(normalization[key]["preconditioned_spectral_constant"]) for key in keys]
    )[:, None, None, None]
    diagonal = np.asarray(
        [np.asarray(normalization[key]["jacobi_diagonal"]) for key in keys]
    ).reshape(len(keys), 1, start.shape[-2], start.shape[-1])
    current = np.asarray(start, dtype=np.float64).copy()
    observed = np.asarray(observation, dtype=np.float64)
    output: dict[int, np.ndarray] = {}
    for iteration in range(1, requested[-1] + 1):
        residual = observed - forward_project(current, operator)
        gradient = masked_adjoint(residual, operator, masks, current.shape[1:])
        if method == "standard":
            current = current + beta * gradient / spectral
        else:
            current = current + beta * (gradient / diagonal) / preconditioned
        current = feasibility_project(current, support)
        if iteration in requested:
            output[iteration] = current.copy()
    return output


def global_landweber_trajectory(
    start: np.ndarray,
    observation: np.ndarray,
    operator: np.ndarray,
    view_masks: np.ndarray,
    support: np.ndarray,
    step_fraction: float,
    checkpoints: Iterable[int],
    global_spectral_constant: float,
) -> dict[int, np.ndarray]:
    """Run projected Landweber with one absolute step for every geometry."""
    beta = float(step_fraction)
    spectral = float(global_spectral_constant)
    if not 0.0 < beta < 2.0:
        raise ValueError("global Landweber step fraction must lie in (0, 2)")
    if not np.isfinite(spectral) or spectral <= 0.0:
        raise ValueError("global spectral constant must be finite and positive")
    requested = sorted({int(value) for value in checkpoints})
    if not requested or requested[0] < 1:
        raise ValueError("Landweber checkpoints must be positive integers")
    masks = np.asarray(view_masks, dtype=np.float64)
    current = np.asarray(start, dtype=np.float64).copy()
    observed = np.asarray(observation, dtype=np.float64)
    output: dict[int, np.ndarray] = {}
    for iteration in range(1, requested[-1] + 1):
        residual = observed - forward_project(current, operator)
        gradient = masked_adjoint(residual, operator, masks, current.shape[1:])
        current = feasibility_project(current + beta * gradient / spectral, support)
        if iteration in requested:
            output[iteration] = current.copy()
    return output


def quadratic_steepest_descent_trajectory(
    start: np.ndarray,
    observation: np.ndarray,
    operator: np.ndarray,
    view_masks: np.ndarray,
    support: np.ndarray,
    checkpoints: Iterable[int],
    geometry_spectral_constants: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[str, np.ndarray]]:
    """Use the exact quadratic step along the negative data gradient, then project.

    The closed-form step is exact for the unconstrained line through the current
    iterate. The subsequent feasibility projection means it is not an exact
    line search for the full constrained problem, so objective behavior is
    returned explicitly for audit.
    """
    requested = sorted({int(value) for value in checkpoints})
    if not requested or requested[0] < 1:
        raise ValueError("steepest-descent checkpoints must be positive integers")
    masks = np.asarray(view_masks, dtype=np.float64)
    spectral = np.asarray(geometry_spectral_constants, dtype=np.float64)
    current = np.asarray(start, dtype=np.float64).copy()
    observed = np.asarray(observation, dtype=np.float64)
    if masks.ndim != 2 or masks.shape[0] != len(current):
        raise ValueError("view masks must provide one row per start volume")
    if spectral.shape != (len(current),) or np.any(~np.isfinite(spectral)) or np.any(spectral <= 0.0):
        raise ValueError("geometry spectral constants must be one positive value per sample")

    output: dict[int, np.ndarray] = {}
    steps: list[np.ndarray] = []
    normalized_steps: list[np.ndarray] = []
    objective_before: list[np.ndarray] = []
    objective_after: list[np.ndarray] = []
    projection_change: list[np.ndarray] = []
    weight = masks[:, None, :, None]
    residual = (observed - forward_project(current, operator)) * weight
    for iteration in range(1, requested[-1] + 1):
        gradient = masked_adjoint(residual, operator, masks, current.shape[1:])
        projected_gradient = forward_project(gradient, operator) * weight
        numerator = np.sum(gradient * gradient, axis=(1, 2, 3))
        denominator = np.sum(
            projected_gradient * projected_gradient, axis=(1, 2, 3)
        )
        alpha = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 1e-24,
        )
        proposal = current + alpha[:, None, None, None] * gradient
        candidate = feasibility_project(proposal, support)
        next_residual = (observed - forward_project(candidate, operator)) * weight
        before = 0.5 * np.sum(residual * residual, axis=(1, 2, 3))
        after = 0.5 * np.sum(next_residual * next_residual, axis=(1, 2, 3))
        change = np.linalg.norm(
            (candidate - proposal).reshape(len(candidate), -1), axis=1
        ) / np.maximum(
            np.linalg.norm(proposal.reshape(len(proposal), -1), axis=1), 1e-12
        )
        steps.append(alpha)
        normalized_steps.append(alpha * spectral)
        objective_before.append(before)
        objective_after.append(after)
        projection_change.append(change)
        current = candidate
        residual = next_residual
        if iteration in requested:
            output[iteration] = current.copy()
    diagnostics = {
        "step_size": np.stack(steps),
        "normalized_step_fraction": np.stack(normalized_steps),
        "objective_before": np.stack(objective_before),
        "objective_after": np.stack(objective_after),
        "projection_change_relative_l2": np.stack(projection_change),
    }
    return output, diagnostics


def projected_bb_trajectory(
    start: np.ndarray,
    observation: np.ndarray,
    operator: np.ndarray,
    view_masks: np.ndarray,
    support: np.ndarray,
    checkpoints: Iterable[int],
    geometry_spectral_constants: np.ndarray,
    variant: str,
    initial_step_fraction: float,
    normalized_step_min: float = 0.05,
    normalized_step_max: float = 1.95,
    curvature_floor_relative: float = 1e-12,
    record_residual: bool = False,
) -> tuple[dict[int, np.ndarray], dict[str, np.ndarray]]:
    """Run safeguarded projected BB steps without a line search.

    Each iteration uses one forward and one adjoint call. BB1, BB2, or their
    deterministic alternation changes only the scalar step computed from the
    previous feasible iterate and data gradient. This is a mechanism control,
    not the globally convergent nonmonotone SPG algorithm.
    """
    if variant not in {"bb1", "bb2", "alternating"}:
        raise ValueError("projected BB variant must be bb1, bb2, or alternating")
    requested = sorted({int(value) for value in checkpoints})
    if not requested or requested[0] < 1:
        raise ValueError("projected BB checkpoints must be positive integers")
    initial = float(initial_step_fraction)
    lower = float(normalized_step_min)
    upper = float(normalized_step_max)
    curvature_floor = float(curvature_floor_relative)
    if not np.isfinite(initial) or initial <= 0.0:
        raise ValueError("initial projected BB step fraction must be positive")
    if not np.isfinite(lower) or not np.isfinite(upper) or not 0.0 < lower <= upper:
        raise ValueError("projected BB normalized step bounds are invalid")
    if not np.isfinite(curvature_floor) or curvature_floor < 0.0:
        raise ValueError("projected BB curvature floor must be nonnegative")

    masks = np.asarray(view_masks, dtype=np.float64)
    spectral = np.asarray(geometry_spectral_constants, dtype=np.float64)
    current = np.asarray(start, dtype=np.float64).copy()
    observed = np.asarray(observation, dtype=np.float64)
    if masks.ndim != 2 or masks.shape[0] != len(current):
        raise ValueError("view masks must provide one row per start volume")
    if spectral.shape != (len(current),) or np.any(~np.isfinite(spectral)) or np.any(spectral <= 0.0):
        raise ValueError("geometry spectral constants must be one positive value per sample")

    output: dict[int, np.ndarray] = {}
    raw_steps: list[np.ndarray] = []
    steps: list[np.ndarray] = []
    normalized_steps: list[np.ndarray] = []
    step_sources: list[np.ndarray] = []
    fallback_used: list[np.ndarray] = []
    clipped_low: list[np.ndarray] = []
    clipped_high: list[np.ndarray] = []
    curvature_dots: list[np.ndarray] = []
    objective_before: list[np.ndarray] = []
    projected_change: list[np.ndarray] = []
    residual_before: list[np.ndarray] = []
    previous_current: np.ndarray | None = None
    previous_gradient: np.ndarray | None = None
    weight = masks[:, None, :, None]
    fallback_step = initial / spectral

    for iteration in range(1, requested[-1] + 1):
        residual = (forward_project(current, operator) - observed) * weight
        gradient = masked_adjoint(residual, operator, masks, current.shape[1:])
        objective_before.append(0.5 * np.sum(residual * residual, axis=(1, 2, 3)))
        if record_residual:
            residual_before.append(residual.copy())

        if previous_current is None or previous_gradient is None:
            raw = fallback_step.copy()
            source = np.zeros(len(current), dtype=np.int64)
            invalid = np.zeros(len(current), dtype=bool)
            curvature = np.full(len(current), np.nan, dtype=np.float64)
        else:
            step_difference = current - previous_current
            gradient_difference = gradient - previous_gradient
            s_dot_s = np.sum(step_difference * step_difference, axis=(1, 2, 3))
            s_dot_y = np.sum(step_difference * gradient_difference, axis=(1, 2, 3))
            y_dot_y = np.sum(gradient_difference * gradient_difference, axis=(1, 2, 3))
            scale = np.sqrt(np.maximum(s_dot_s * y_dot_y, 0.0))
            valid = (
                np.isfinite(s_dot_y)
                & np.isfinite(s_dot_s)
                & np.isfinite(y_dot_y)
                & (s_dot_y > curvature_floor * np.maximum(scale, 1e-30))
                & (s_dot_s > 1e-30)
                & (y_dot_y > 1e-30)
            )
            use_bb1 = variant == "bb1" or (variant == "alternating" and iteration % 2 == 0)
            if use_bb1:
                raw = np.divide(
                    s_dot_s,
                    s_dot_y,
                    out=fallback_step.copy(),
                    where=valid,
                )
                formula_code = 1
            else:
                raw = np.divide(
                    s_dot_y,
                    y_dot_y,
                    out=fallback_step.copy(),
                    where=valid,
                )
                formula_code = 2
            valid &= np.isfinite(raw) & (raw > 0.0)
            raw = np.where(valid, raw, fallback_step)
            source = np.where(valid, formula_code, 3).astype(np.int64)
            invalid = ~valid
            curvature = s_dot_y

        normalized_raw = raw * spectral
        step = np.clip(normalized_raw, lower, upper) / spectral
        candidate = feasibility_project(
            current - step[:, None, None, None] * gradient,
            support,
        )
        change = np.linalg.norm(
            (candidate - current).reshape(len(current), -1), axis=1
        ) / np.maximum(
            np.linalg.norm(current.reshape(len(current), -1), axis=1), 1e-12
        )

        raw_steps.append(raw)
        steps.append(step)
        normalized_steps.append(step * spectral)
        step_sources.append(source)
        fallback_used.append(invalid)
        clipped_low.append(normalized_raw < lower)
        clipped_high.append(normalized_raw > upper)
        curvature_dots.append(curvature)
        projected_change.append(change)
        previous_current = current.copy()
        previous_gradient = gradient.copy()
        current = candidate
        if iteration in requested:
            output[iteration] = current.copy()

    diagnostics = {
        "raw_step_size": np.stack(raw_steps),
        "step_size": np.stack(steps),
        "normalized_step_fraction": np.stack(normalized_steps),
        "step_source_code": np.stack(step_sources),
        "fallback_used": np.stack(fallback_used),
        "clipped_low": np.stack(clipped_low),
        "clipped_high": np.stack(clipped_high),
        "curvature_dot": np.stack(curvature_dots),
        "objective_before": np.stack(objective_before),
        "projected_change_relative_l2": np.stack(projected_change),
    }
    if record_residual:
        diagnostics["residual_before"] = np.stack(residual_before)
    return output, diagnostics


def adjoint_inner_product_relative_error(
    volume: np.ndarray,
    projection: np.ndarray,
    operator: np.ndarray,
    view_mask: np.ndarray,
) -> float:
    """Check <MAx,y> = <x,A^TMy> for one geometry."""
    x = np.asarray(volume, dtype=np.float64)
    y = np.asarray(projection, dtype=np.float64)
    mask = np.asarray(view_mask, dtype=np.float64)
    if x.shape[0] != 1 or y.shape[0] != 1:
        raise ValueError("adjoint check expects a single batch item")
    projected = forward_project(x, operator) * mask[None, None, :, None]
    lhs = float(np.sum(projected * y))
    adjoint = masked_adjoint(y, operator, mask[None], x.shape[1:])
    rhs = float(np.sum(x * adjoint))
    return abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-12)


def finite_difference_gradient_relative_error(
    volume: np.ndarray,
    direction: np.ndarray,
    observation: np.ndarray,
    operator: np.ndarray,
    view_mask: np.ndarray,
    epsilon: float = 1e-6,
) -> float:
    """Check the gradient of 1/2 ||M(Ax-y)||^2 by a central difference."""
    x = np.asarray(volume, dtype=np.float64)
    vector = np.asarray(direction, dtype=np.float64)
    observed = np.asarray(observation, dtype=np.float64)
    mask = np.asarray(view_mask, dtype=np.float64)

    def objective(candidate: np.ndarray) -> float:
        residual = (forward_project(candidate, operator) - observed) * mask[
            None, None, :, None
        ]
        return 0.5 * float(np.sum(residual * residual))

    finite = (objective(x + epsilon * vector) - objective(x - epsilon * vector)) / (
        2.0 * epsilon
    )
    residual = forward_project(x, operator) - observed
    gradient = masked_adjoint(residual, operator, mask[None], x.shape[1:])
    analytic = float(np.sum(gradient * vector))
    return abs(finite - analytic) / max(abs(finite), abs(analytic), 1e-12)
