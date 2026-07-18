"""Side-specific float64 diagnostics for randomized adjoint probes.

Unlike the frozen D4c-v1 diagnostic, this module weights the output-side and
input-side action scales by their own ``gamma_n`` values.  It still does not
choose a pass threshold and does not claim to bound the complete autodiff
program.  Finite differences, structural identities, and actual branch-state
comparisons remain independent obligations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

import numpy as np


UNIT_ROUNDOFF = float(np.finfo(np.float64).eps / 2.0)


def gamma_n(term_count: int) -> float:
    if isinstance(term_count, bool) or int(term_count) != term_count:
        raise TypeError("term_count must be an integer")
    count = int(term_count)
    if count < 1:
        raise ValueError("term_count must be positive")
    product = count * UNIT_ROUNDOFF
    if product >= 1.0:
        raise ValueError("term_count is too large for the gamma_n model")
    return product / (1.0 - product)


def _vector(name: str, value: np.ndarray | Iterable[float]) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64).reshape(-1))
    if array.size < 1 or not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{name} must be a nonempty finite vector")
    return array


@dataclass(frozen=True, slots=True)
class SideWeightedAdjointEvidence:
    input_dimension: int
    output_dimension: int
    lhs: float
    rhs: float
    absolute_defect: float
    signal_relative_defect: float
    left_action_scale: float
    right_action_scale: float
    maximum_action_scale: float
    normwise_defect: float
    dot_condition_proxy: float
    output_gamma: float
    input_gamma: float
    side_weighted_gamma_scale: float
    side_weighted_gamma_score: float
    contraction_roundoff_envelope: float
    contraction_envelope_ratio: float
    finite: bool

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


def evaluate_side_weighted_adjoint(
    jvp: np.ndarray | Iterable[float],
    cotangent: np.ndarray | Iterable[float],
    tangent: np.ndarray | Iterable[float],
    vjp: np.ndarray | Iterable[float],
    *,
    denominator_floor: float = 1e-300,
) -> SideWeightedAdjointEvidence:
    jvp_array = _vector("jvp", jvp)
    cotangent_array = _vector("cotangent", cotangent)
    tangent_array = _vector("tangent", tangent)
    vjp_array = _vector("vjp", vjp)
    if jvp_array.shape != cotangent_array.shape:
        raise ValueError("jvp and cotangent shapes differ")
    if tangent_array.shape != vjp_array.shape:
        raise ValueError("tangent and vjp shapes differ")
    floor = float(denominator_floor)
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("denominator_floor must be finite and positive")

    left_products = jvp_array * cotangent_array
    right_products = tangent_array * vjp_array
    lhs = float(np.dot(jvp_array, cotangent_array))
    rhs = float(np.dot(tangent_array, vjp_array))
    absolute = abs(lhs - rhs)
    signal = max(abs(lhs), abs(rhs))
    left_action = float(np.linalg.norm(jvp_array)) * float(
        np.linalg.norm(cotangent_array)
    )
    right_action = float(np.linalg.norm(tangent_array)) * float(
        np.linalg.norm(vjp_array)
    )
    maximum_action = max(left_action, right_action)
    output_gamma = gamma_n(jvp_array.size)
    input_gamma = gamma_n(tangent_array.size)
    side_scale = output_gamma * left_action + input_gamma * right_action
    contraction_envelope = output_gamma * float(np.sum(np.abs(left_products))) + (
        input_gamma * float(np.sum(np.abs(right_products)))
    )
    normwise = absolute / max(maximum_action, floor)
    return SideWeightedAdjointEvidence(
        input_dimension=int(tangent_array.size),
        output_dimension=int(jvp_array.size),
        lhs=lhs,
        rhs=rhs,
        absolute_defect=absolute,
        signal_relative_defect=absolute / max(signal, floor),
        left_action_scale=left_action,
        right_action_scale=right_action,
        maximum_action_scale=maximum_action,
        normwise_defect=normwise,
        dot_condition_proxy=maximum_action / max(signal, floor),
        output_gamma=output_gamma,
        input_gamma=input_gamma,
        side_weighted_gamma_scale=side_scale,
        side_weighted_gamma_score=absolute / max(side_scale, floor),
        contraction_roundoff_envelope=contraction_envelope,
        contraction_envelope_ratio=absolute / max(contraction_envelope, floor),
        finite=all(
            math.isfinite(value)
            for value in (
                lhs,
                rhs,
                absolute,
                left_action,
                right_action,
                normwise,
                side_scale,
                contraction_envelope,
            )
        ),
    )


@dataclass(frozen=True, slots=True)
class SideWeightedProbeSummary:
    probe_count: int
    all_finite: bool
    maximum_signal_relative_defect: float
    maximum_side_weighted_gamma_score: float
    maximum_contraction_envelope_ratio: float
    minimum_signal: float
    maximum_condition_proxy: float

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


def summarize_side_weighted_probes(
    evidence: Iterable[SideWeightedAdjointEvidence],
) -> SideWeightedProbeSummary:
    rows = tuple(evidence)
    if not rows:
        raise ValueError("at least one probe is required")
    return SideWeightedProbeSummary(
        probe_count=len(rows),
        all_finite=all(row.finite for row in rows),
        maximum_signal_relative_defect=max(
            row.signal_relative_defect for row in rows
        ),
        maximum_side_weighted_gamma_score=max(
            row.side_weighted_gamma_score for row in rows
        ),
        maximum_contraction_envelope_ratio=max(
            row.contraction_envelope_ratio for row in rows
        ),
        minimum_signal=min(max(abs(row.lhs), abs(row.rhs)) for row in rows),
        maximum_condition_proxy=max(row.dot_condition_proxy for row in rows),
    )


__all__ = [
    "SideWeightedAdjointEvidence",
    "SideWeightedProbeSummary",
    "UNIT_ROUNDOFF",
    "evaluate_side_weighted_adjoint",
    "gamma_n",
    "summarize_side_weighted_probes",
]
