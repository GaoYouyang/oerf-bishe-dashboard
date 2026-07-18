"""Scale-aware diagnostics for randomized JVP/VJP adjoint probes.

This module deliberately does not choose an acceptance threshold.  It exposes
the traditional dot-relative defect together with a normwise defect scaled by
the classical float64 dot-product ``gamma_n`` quantity.  Protocol runners must
pre-register probe counts, threshold grids, branch semantics, and independent
finite-difference or structural controls before using these diagnostics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

import numpy as np


FLOAT64_UNIT_ROUNDOFF = float(np.finfo(np.float64).eps / 2.0)


def gamma_n(term_count: int, *, unit_roundoff: float = FLOAT64_UNIT_ROUNDOFF) -> float:
    """Return ``n*u/(1-n*u)`` for a positive dot-product term count."""

    if isinstance(term_count, bool) or int(term_count) != term_count:
        raise TypeError("term_count must be an integer")
    count = int(term_count)
    if count < 1:
        raise ValueError("term_count must be positive")
    unit = float(unit_roundoff)
    if not math.isfinite(unit) or unit <= 0.0:
        raise ValueError("unit_roundoff must be finite and positive")
    product = count * unit
    if product >= 1.0:
        raise ValueError("term_count * unit_roundoff must be smaller than one")
    return product / (1.0 - product)


def _vector(name: str, value: np.ndarray | Iterable[float]) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64).reshape(-1))
    if array.size < 1:
        raise ValueError(f"{name} must not be empty")
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{name} must contain only finite values")
    return array


@dataclass(frozen=True, slots=True)
class MixedScaleAdjointEvidence:
    """One JVP/VJP contraction reported at complementary numerical scales."""

    input_dimension: int
    output_dimension: int
    lhs_jvp_cotangent: float
    rhs_tangent_vjp: float
    signed_defect: float
    absolute_defect: float
    dot_signal: float
    dot_relative_defect: float
    maximum_action_scale: float
    summed_action_scale: float
    normwise_defect: float
    summed_action_normwise_defect: float
    dot_condition_proxy: float
    input_gamma: float
    output_gamma: float
    maximum_gamma: float
    gamma_scaled_normwise_score: float
    product_l1_scale: float
    contraction_roundoff_envelope: float
    contraction_envelope_ratio: float
    finite: bool

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


def evaluate_mixed_scale_adjoint(
    jvp: np.ndarray | Iterable[float],
    cotangent: np.ndarray | Iterable[float],
    tangent: np.ndarray | Iterable[float],
    vjp: np.ndarray | Iterable[float],
    *,
    denominator_floor: float = 1e-300,
) -> MixedScaleAdjointEvidence:
    """Evaluate relative, normwise, and gamma-scaled adjoint defects.

    ``gamma_scaled_normwise_score`` is a diagnostic, not a proof that an entire
    autodiff program obeys a dot-product-only rounding model.  A score near one
    means the observed normwise mismatch is comparable to ``gamma_n`` for the
    larger contraction.  It cannot replace finite-difference, structural, or
    real forward-branch checks.
    """

    jvp_array = _vector("jvp", jvp)
    cotangent_array = _vector("cotangent", cotangent)
    tangent_array = _vector("tangent", tangent)
    vjp_array = _vector("vjp", vjp)
    if jvp_array.shape != cotangent_array.shape:
        raise ValueError("jvp and cotangent must have the same flattened shape")
    if tangent_array.shape != vjp_array.shape:
        raise ValueError("tangent and vjp must have the same flattened shape")
    floor = float(denominator_floor)
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("denominator_floor must be finite and positive")

    left_products = jvp_array * cotangent_array
    right_products = tangent_array * vjp_array
    lhs = float(np.dot(jvp_array, cotangent_array))
    rhs = float(np.dot(tangent_array, vjp_array))
    signed = lhs - rhs
    absolute = abs(signed)
    signal = max(abs(lhs), abs(rhs))
    left_action_scale = float(np.linalg.norm(jvp_array)) * float(
        np.linalg.norm(cotangent_array)
    )
    right_action_scale = float(np.linalg.norm(tangent_array)) * float(
        np.linalg.norm(vjp_array)
    )
    maximum_action_scale = max(left_action_scale, right_action_scale)
    summed_action_scale = left_action_scale + right_action_scale
    input_gamma = gamma_n(tangent_array.size)
    output_gamma = gamma_n(jvp_array.size)
    maximum_gamma = max(input_gamma, output_gamma)
    product_l1_scale = float(np.sum(np.abs(left_products))) + float(
        np.sum(np.abs(right_products))
    )
    roundoff_envelope = output_gamma * float(np.sum(np.abs(left_products))) + (
        input_gamma * float(np.sum(np.abs(right_products)))
    )
    normwise_defect = absolute / max(maximum_action_scale, floor)
    summed_action_normwise_defect = absolute / max(summed_action_scale, floor)
    return MixedScaleAdjointEvidence(
        input_dimension=int(tangent_array.size),
        output_dimension=int(jvp_array.size),
        lhs_jvp_cotangent=lhs,
        rhs_tangent_vjp=rhs,
        signed_defect=signed,
        absolute_defect=absolute,
        dot_signal=signal,
        dot_relative_defect=absolute / max(signal, floor),
        maximum_action_scale=maximum_action_scale,
        summed_action_scale=summed_action_scale,
        normwise_defect=normwise_defect,
        summed_action_normwise_defect=summed_action_normwise_defect,
        dot_condition_proxy=maximum_action_scale / max(signal, floor),
        input_gamma=input_gamma,
        output_gamma=output_gamma,
        maximum_gamma=maximum_gamma,
        gamma_scaled_normwise_score=normwise_defect / maximum_gamma,
        product_l1_scale=product_l1_scale,
        contraction_roundoff_envelope=roundoff_envelope,
        contraction_envelope_ratio=absolute / max(roundoff_envelope, floor),
        finite=all(
            math.isfinite(value)
            for value in (
                lhs,
                rhs,
                absolute,
                signal,
                maximum_action_scale,
                summed_action_scale,
                normwise_defect,
                product_l1_scale,
                roundoff_envelope,
            )
        ),
    )


@dataclass(frozen=True, slots=True)
class ProbeSetSummary:
    """Worst-case summary over a frozen ordered set of randomized probes."""

    probe_count: int
    all_finite: bool
    maximum_dot_relative_defect: float
    maximum_gamma_scaled_normwise_score: float
    median_gamma_scaled_normwise_score: float
    minimum_dot_signal: float
    maximum_dot_condition_proxy: float

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


def summarize_probe_set(
    evidence: Iterable[MixedScaleAdjointEvidence],
) -> ProbeSetSummary:
    rows = tuple(evidence)
    if not rows:
        raise ValueError("evidence must contain at least one probe")
    scores = np.asarray(
        [row.gamma_scaled_normwise_score for row in rows], dtype=np.float64
    )
    return ProbeSetSummary(
        probe_count=len(rows),
        all_finite=all(row.finite for row in rows),
        maximum_dot_relative_defect=max(row.dot_relative_defect for row in rows),
        maximum_gamma_scaled_normwise_score=float(np.max(scores)),
        median_gamma_scaled_normwise_score=float(np.median(scores)),
        minimum_dot_signal=min(row.dot_signal for row in rows),
        maximum_dot_condition_proxy=max(row.dot_condition_proxy for row in rows),
    )


__all__ = [
    "FLOAT64_UNIT_ROUNDOFF",
    "MixedScaleAdjointEvidence",
    "ProbeSetSummary",
    "evaluate_mixed_scale_adjoint",
    "gamma_n",
    "summarize_probe_set",
]
