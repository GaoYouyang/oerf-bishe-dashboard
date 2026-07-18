"""Reusable float64 derivative-audit primitives for tensor field closures.

The functions in this module audit a single ``Tensor -> Tensor`` closure.  They
do not select cases, tune thresholds, write results, or make scientific claims.
Callers remain responsible for preregistering points, directions, cotangents,
finite-difference steps, and acceptance thresholds.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math

import torch


TensorClosure = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class JVPVJPAudit:
    """Forward/reverse derivative evidence for one point and one direction."""

    primal: torch.Tensor
    jvp: torch.Tensor
    vjp: torch.Tensor
    dot_jvp_cotangent: float
    dot_tangent_vjp: float
    dot_absolute_defect: float
    dot_relative_defect: float
    repeated_primal_relative_defect: float
    tangent_norm: float
    cotangent_norm: float
    jvp_norm: float
    vjp_norm: float
    dot_signal: float
    finite: bool
    nondegenerate_inputs: bool
    nondegenerate_derivative: bool
    nondegenerate_dot: bool
    nondegenerate: bool


@dataclass(frozen=True)
class CentralDifferenceLevel:
    """One fixed central-finite-difference level compared with a JVP."""

    h: float
    plus: torch.Tensor
    minus: torch.Tensor
    estimate: torch.Tensor
    symmetric_difference_norm: float
    output_scale: float
    estimate_norm: float
    absolute_error: float
    relative_error: float
    finite: bool
    nondegenerate: bool


@dataclass(frozen=True)
class CentralDifferenceSweep:
    """All requested finite-difference levels, without adaptive re-selection."""

    levels: tuple[CentralDifferenceLevel, ...]
    reference_jvp_norm: float
    all_finite: bool
    any_finite: bool
    nondegenerate: bool
    best_index: int | None
    best_h: float | None
    best_relative_error: float

    def count_at_or_below(self, relative_tolerance: float) -> int:
        """Count finite levels meeting a caller-supplied relative tolerance."""

        tolerance = _require_nonnegative_finite(
            "relative_tolerance", relative_tolerance
        )
        return sum(
            level.finite and level.relative_error <= tolerance for level in self.levels
        )


@dataclass(frozen=True)
class ClosureDerivativeAudit:
    """Combined autodiff, dot-product, and fixed multi-h FD audit."""

    autodiff: JVPVJPAudit
    finite_difference: CentralDifferenceSweep
    finite: bool
    nondegenerate: bool

    def passes(
        self,
        *,
        dot_relative_tolerance: float,
        finite_difference_relative_tolerance: float,
        minimum_finite_difference_levels: int = 1,
    ) -> bool:
        """Apply explicit caller thresholds while failing closed on bad signals."""

        dot_tolerance = _require_nonnegative_finite(
            "dot_relative_tolerance", dot_relative_tolerance
        )
        fd_tolerance = _require_nonnegative_finite(
            "finite_difference_relative_tolerance",
            finite_difference_relative_tolerance,
        )
        if (
            isinstance(minimum_finite_difference_levels, bool)
            or not isinstance(minimum_finite_difference_levels, int)
            or minimum_finite_difference_levels < 1
        ):
            raise ValueError("minimum_finite_difference_levels must be a positive int")
        return bool(
            self.finite
            and self.nondegenerate
            and self.autodiff.dot_relative_defect <= dot_tolerance
            and self.finite_difference.count_at_or_below(fd_tolerance)
            >= minimum_finite_difference_levels
        )


@dataclass(frozen=True)
class TensorConsistency:
    """Scale-aware comparison of two tensors with an explicit allclose gate."""

    absolute_l2: float
    relative_l2: float
    maximum_absolute_error: float
    finite: bool
    within_tolerance: bool


@dataclass(frozen=True)
class ResidualStructureAudit:
    """Check direct residual against curved minus straight at three levels."""

    direct: JVPVJPAudit
    curved: JVPVJPAudit
    straight: JVPVJPAudit
    primal: TensorConsistency
    jvp: TensorConsistency
    vjp: TensorConsistency
    finite: bool
    nondegenerate: bool
    consistent: bool


def _require_cpu_float64(name: str, value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.device.type != "cpu" or value.dtype != torch.float64:
        raise TypeError(f"{name} must be a CPU torch.float64 tensor")
    if value.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    return value


def _require_finite_input(name: str, value: torch.Tensor) -> None:
    if not bool(torch.all(torch.isfinite(value)).item()):
        raise ValueError(f"{name} must contain only finite values")


def _require_output(
    name: str,
    value: torch.Tensor,
    *,
    expected_shape: torch.Size | None = None,
) -> torch.Tensor:
    value = _require_cpu_float64(name, value)
    if expected_shape is not None and value.shape != expected_shape:
        raise ValueError(
            f"{name} has shape {tuple(value.shape)}, expected {tuple(expected_shape)}"
        )
    return value


def _require_nonnegative_finite(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def _require_positive_finite(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.detach().reshape(-1)).item())


def _all_finite(*values: torch.Tensor) -> bool:
    return all(bool(torch.all(torch.isfinite(value)).item()) for value in values)


def _detached(value: torch.Tensor) -> torch.Tensor:
    return value.detach().clone()


def _relative_l2(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    denominator_floor: float,
) -> float:
    if not _all_finite(candidate, reference):
        return math.inf
    error = _norm(candidate - reference)
    scale = max(_norm(candidate), _norm(reference), denominator_floor)
    return error / scale


def evaluate_jvp_vjp(
    closure: TensorClosure,
    point: torch.Tensor,
    tangent: torch.Tensor,
    cotangent: torch.Tensor,
    *,
    denominator_floor: float = 1e-30,
    nondegenerate_floor: float = 1e-14,
) -> JVPVJPAudit:
    """Evaluate JVP/VJP and their same-cotangent dot-product identity.

    The relative dot defect is
    ``|<Jv,w>-<v,J^T w>| / max(|<Jv,w>|, |<v,J^T w>|, floor)``.
    ``nondegenerate`` is separate from this ratio so a zero-over-zero case can
    never be mistaken for a successful dot-product test.
    """

    if not callable(closure):
        raise TypeError("closure must be callable")
    point = _require_cpu_float64("point", point)
    tangent = _require_cpu_float64("tangent", tangent)
    cotangent = _require_cpu_float64("cotangent", cotangent)
    if tangent.shape != point.shape:
        raise ValueError("tangent must have the same shape as point")
    _require_finite_input("point", point)
    _require_finite_input("tangent", tangent)
    _require_finite_input("cotangent", cotangent)
    denominator = _require_positive_finite("denominator_floor", denominator_floor)
    signal_floor = _require_positive_finite("nondegenerate_floor", nondegenerate_floor)

    def checked_closure(value: torch.Tensor) -> torch.Tensor:
        return _require_output("closure output", closure(value))

    jvp_primal, jvp_value = torch.func.jvp(
        checked_closure,
        (point,),
        (tangent,),
    )
    jvp_primal = _require_output("jvp primal", jvp_primal)
    jvp_value = _require_output("jvp value", jvp_value, expected_shape=jvp_primal.shape)
    vjp_primal, pullback = torch.func.vjp(checked_closure, point)
    vjp_primal = _require_output(
        "vjp primal", vjp_primal, expected_shape=jvp_primal.shape
    )
    if cotangent.shape != jvp_primal.shape:
        raise ValueError("cotangent must have the same shape as closure output")
    (vjp_value,) = pullback(cotangent)
    vjp_value = _require_output("vjp value", vjp_value, expected_shape=point.shape)

    dot_jvp_cotangent_tensor = torch.sum(jvp_value * cotangent)
    dot_tangent_vjp_tensor = torch.sum(tangent * vjp_value)
    finite = _all_finite(
        jvp_primal,
        vjp_primal,
        jvp_value,
        vjp_value,
        dot_jvp_cotangent_tensor,
        dot_tangent_vjp_tensor,
    )
    tangent_norm = _norm(tangent)
    cotangent_norm = _norm(cotangent)
    jvp_norm = _norm(jvp_value)
    vjp_norm = _norm(vjp_value)
    if finite:
        lhs = float(dot_jvp_cotangent_tensor.item())
        rhs = float(dot_tangent_vjp_tensor.item())
        absolute_defect = abs(lhs - rhs)
        dot_signal = max(abs(lhs), abs(rhs))
        relative_defect = absolute_defect / max(dot_signal, denominator)
        repeated_primal_defect = _relative_l2(
            jvp_primal,
            vjp_primal,
            denominator_floor=denominator,
        )
    else:
        lhs = math.nan
        rhs = math.nan
        absolute_defect = math.inf
        relative_defect = math.inf
        repeated_primal_defect = math.inf
        dot_signal = 0.0

    nondegenerate_inputs = bool(
        math.isfinite(tangent_norm)
        and math.isfinite(cotangent_norm)
        and tangent_norm > signal_floor
        and cotangent_norm > signal_floor
    )
    nondegenerate_derivative = bool(
        math.isfinite(jvp_norm)
        and math.isfinite(vjp_norm)
        and jvp_norm > signal_floor
        and vjp_norm > signal_floor
    )
    nondegenerate_dot = bool(finite and dot_signal > signal_floor)
    nondegenerate = bool(
        finite
        and nondegenerate_inputs
        and nondegenerate_derivative
        and nondegenerate_dot
    )
    return JVPVJPAudit(
        primal=_detached(jvp_primal),
        jvp=_detached(jvp_value),
        vjp=_detached(vjp_value),
        dot_jvp_cotangent=lhs,
        dot_tangent_vjp=rhs,
        dot_absolute_defect=absolute_defect,
        dot_relative_defect=relative_defect,
        repeated_primal_relative_defect=repeated_primal_defect,
        tangent_norm=tangent_norm,
        cotangent_norm=cotangent_norm,
        jvp_norm=jvp_norm,
        vjp_norm=vjp_norm,
        dot_signal=dot_signal,
        finite=finite,
        nondegenerate_inputs=nondegenerate_inputs,
        nondegenerate_derivative=nondegenerate_derivative,
        nondegenerate_dot=nondegenerate_dot,
        nondegenerate=nondegenerate,
    )


def central_difference_sweep(
    closure: TensorClosure,
    point: torch.Tensor,
    tangent: torch.Tensor,
    reference_jvp: torch.Tensor,
    h_values: Sequence[float],
    *,
    denominator_floor: float = 1e-30,
    nondegenerate_floor: float = 1e-14,
) -> CentralDifferenceSweep:
    """Compare a reference JVP with every level in a fixed central-FD grid."""

    if not callable(closure):
        raise TypeError("closure must be callable")
    point = _require_cpu_float64("point", point)
    tangent = _require_cpu_float64("tangent", tangent)
    reference_jvp = _require_cpu_float64("reference_jvp", reference_jvp)
    if tangent.shape != point.shape:
        raise ValueError("tangent must have the same shape as point")
    _require_finite_input("point", point)
    _require_finite_input("tangent", tangent)
    denominator = _require_positive_finite("denominator_floor", denominator_floor)
    signal_floor = _require_positive_finite("nondegenerate_floor", nondegenerate_floor)
    steps = tuple(_require_positive_finite("h", value) for value in h_values)
    if not steps:
        raise ValueError("h_values must contain at least one fixed step")
    if len(set(steps)) != len(steps):
        raise ValueError("h_values must not contain duplicate steps")

    reference_norm = _norm(reference_jvp)
    levels: list[CentralDifferenceLevel] = []
    for h in steps:
        plus = _require_output(
            f"closure(point + {h} * tangent)",
            closure(point + h * tangent),
            expected_shape=reference_jvp.shape,
        )
        minus = _require_output(
            f"closure(point - {h} * tangent)",
            closure(point - h * tangent),
            expected_shape=reference_jvp.shape,
        )
        estimate = (plus - minus) / (2.0 * h)
        finite = _all_finite(reference_jvp, plus, minus, estimate)
        estimate_norm = _norm(estimate)
        symmetric_difference_norm = _norm(plus - minus)
        output_scale = max(_norm(plus), _norm(minus), denominator)
        if finite:
            absolute_error = _norm(estimate - reference_jvp)
            relative_error = absolute_error / max(
                estimate_norm,
                reference_norm,
                denominator,
            )
        else:
            absolute_error = math.inf
            relative_error = math.inf
        nondegenerate = bool(
            finite
            and math.isfinite(reference_norm)
            and math.isfinite(estimate_norm)
            and reference_norm > signal_floor
            and estimate_norm > signal_floor
        )
        levels.append(
            CentralDifferenceLevel(
                h=h,
                plus=_detached(plus),
                minus=_detached(minus),
                estimate=_detached(estimate),
                symmetric_difference_norm=symmetric_difference_norm,
                output_scale=output_scale,
                estimate_norm=estimate_norm,
                absolute_error=absolute_error,
                relative_error=relative_error,
                finite=finite,
                nondegenerate=nondegenerate,
            )
        )

    finite_indices = [index for index, level in enumerate(levels) if level.finite]
    best_index = (
        min(finite_indices, key=lambda index: levels[index].relative_error)
        if finite_indices
        else None
    )
    return CentralDifferenceSweep(
        levels=tuple(levels),
        reference_jvp_norm=reference_norm,
        all_finite=all(level.finite for level in levels),
        any_finite=bool(finite_indices),
        nondegenerate=any(level.nondegenerate for level in levels),
        best_index=best_index,
        best_h=levels[best_index].h if best_index is not None else None,
        best_relative_error=(
            levels[best_index].relative_error if best_index is not None else math.inf
        ),
    )


def audit_tensor_closure(
    closure: TensorClosure,
    point: torch.Tensor,
    tangent: torch.Tensor,
    cotangent: torch.Tensor,
    h_values: Sequence[float],
    *,
    denominator_floor: float = 1e-30,
    nondegenerate_floor: float = 1e-14,
) -> ClosureDerivativeAudit:
    """Run the reusable JVP/VJP, dot, and fixed multi-h audit bundle."""

    autodiff = evaluate_jvp_vjp(
        closure,
        point,
        tangent,
        cotangent,
        denominator_floor=denominator_floor,
        nondegenerate_floor=nondegenerate_floor,
    )
    finite_difference = central_difference_sweep(
        closure,
        point,
        tangent,
        autodiff.jvp,
        h_values,
        denominator_floor=denominator_floor,
        nondegenerate_floor=nondegenerate_floor,
    )
    finite = bool(autodiff.finite and finite_difference.all_finite)
    return ClosureDerivativeAudit(
        autodiff=autodiff,
        finite_difference=finite_difference,
        finite=finite,
        nondegenerate=bool(
            finite and autodiff.nondegenerate and finite_difference.nondegenerate
        ),
    )


def _compare_tensors(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
    denominator_floor: float,
) -> TensorConsistency:
    finite = _all_finite(candidate, reference)
    if finite:
        delta = candidate - reference
        absolute_l2 = _norm(delta)
        relative_l2 = absolute_l2 / max(
            _norm(candidate),
            _norm(reference),
            denominator_floor,
        )
        maximum_absolute_error = float(torch.max(torch.abs(delta)).item())
        within_tolerance = bool(
            torch.allclose(
                candidate,
                reference,
                atol=absolute_tolerance,
                rtol=relative_tolerance,
            )
        )
    else:
        absolute_l2 = math.inf
        relative_l2 = math.inf
        maximum_absolute_error = math.inf
        within_tolerance = False
    return TensorConsistency(
        absolute_l2=absolute_l2,
        relative_l2=relative_l2,
        maximum_absolute_error=maximum_absolute_error,
        finite=finite,
        within_tolerance=within_tolerance,
    )


def compare_residual_structure(
    direct_residual: TensorClosure,
    curved: TensorClosure,
    straight: TensorClosure,
    point: torch.Tensor,
    tangent: torch.Tensor,
    cotangent: torch.Tensor,
    *,
    absolute_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-10,
    denominator_floor: float = 1e-30,
    nondegenerate_floor: float = 1e-14,
) -> ResidualStructureAudit:
    """Compare direct residual with ``curved(point) - straight(point)``.

    The same point, tangent, and cotangent are used for all three closures.
    Primal values, JVPs, and VJPs must each agree; no averaging can hide a
    failed component.
    """

    absolute = _require_nonnegative_finite("absolute_tolerance", absolute_tolerance)
    relative = _require_nonnegative_finite("relative_tolerance", relative_tolerance)
    denominator = _require_positive_finite("denominator_floor", denominator_floor)
    signal_floor = _require_positive_finite("nondegenerate_floor", nondegenerate_floor)
    direct_audit = evaluate_jvp_vjp(
        direct_residual,
        point,
        tangent,
        cotangent,
        denominator_floor=denominator,
        nondegenerate_floor=signal_floor,
    )
    curved_audit = evaluate_jvp_vjp(
        curved,
        point,
        tangent,
        cotangent,
        denominator_floor=denominator,
        nondegenerate_floor=signal_floor,
    )
    straight_audit = evaluate_jvp_vjp(
        straight,
        point,
        tangent,
        cotangent,
        denominator_floor=denominator,
        nondegenerate_floor=signal_floor,
    )
    expected_primal = curved_audit.primal - straight_audit.primal
    expected_jvp = curved_audit.jvp - straight_audit.jvp
    expected_vjp = curved_audit.vjp - straight_audit.vjp
    primal = _compare_tensors(
        direct_audit.primal,
        expected_primal,
        absolute_tolerance=absolute,
        relative_tolerance=relative,
        denominator_floor=denominator,
    )
    jvp = _compare_tensors(
        direct_audit.jvp,
        expected_jvp,
        absolute_tolerance=absolute,
        relative_tolerance=relative,
        denominator_floor=denominator,
    )
    vjp = _compare_tensors(
        direct_audit.vjp,
        expected_vjp,
        absolute_tolerance=absolute,
        relative_tolerance=relative,
        denominator_floor=denominator,
    )
    finite = bool(
        direct_audit.finite
        and curved_audit.finite
        and straight_audit.finite
        and primal.finite
        and jvp.finite
        and vjp.finite
    )
    expected_jvp_norm = _norm(expected_jvp)
    expected_vjp_norm = _norm(expected_vjp)
    nondegenerate = bool(
        finite
        and direct_audit.nondegenerate_inputs
        and direct_audit.jvp_norm > signal_floor
        and direct_audit.vjp_norm > signal_floor
        and expected_jvp_norm > signal_floor
        and expected_vjp_norm > signal_floor
    )
    consistent = bool(
        finite
        and nondegenerate
        and primal.within_tolerance
        and jvp.within_tolerance
        and vjp.within_tolerance
    )
    return ResidualStructureAudit(
        direct=direct_audit,
        curved=curved_audit,
        straight=straight_audit,
        primal=primal,
        jvp=jvp,
        vjp=vjp,
        finite=finite,
        nondegenerate=nondegenerate,
        consistent=consistent,
    )


__all__ = [
    "CentralDifferenceLevel",
    "CentralDifferenceSweep",
    "ClosureDerivativeAudit",
    "JVPVJPAudit",
    "ResidualStructureAudit",
    "TensorClosure",
    "TensorConsistency",
    "audit_tensor_closure",
    "central_difference_sweep",
    "compare_residual_structure",
    "evaluate_jvp_vjp",
]
