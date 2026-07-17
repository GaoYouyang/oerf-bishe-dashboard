"""Exact-|A| diagnostics for the closed PSU-B0 Gate-B mechanism study.

This module does not define a new Gate-B candidate.  It materializes the
absolute row and column sums of the already-frozen signed A-only operator and
replays diagnostic diagonal metrics to separate three effects:

* the factor majorizer ``M >= |A|``;
* the per-view maximum used by the formal Gate-B dual metric; and
* the remaining advantage of the exact-K graph-PCGLS comparator.

The production path streams basis columns, so it never stores the full dense
operator.  Tiny tests independently assemble the dense operator and verify the
same sums, recurrence, dominance relation, and Schur safety bound.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
from typing import Any, Literal

import torch

from .psu_b0_factor_majorizer_pipeline import (
    FactorPipelineCallLedger,
    PSUB0FactorPDHGState,
)
from .psu_b0_gate_b_data_only import (
    GateBDataOnlySetup,
    describe_gate_b_metric,
    initial_gate_b_state,
)


EXACT_ABSOLUTE_DIAGNOSTIC_SCHEMA = "psu-b0-exact-absolute-diagnostic-1.0"
DiagnosticMetricMode = Literal["factor_row", "exact_view", "exact_row"]


def _tensor_digest(digest: Any, name: str, value: torch.Tensor) -> None:
    tensor = value.detach().contiguous().cpu()
    digest.update(name.encode("utf-8"))
    digest.update(repr((tuple(tensor.shape), str(tensor.dtype))).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))


@dataclass(frozen=True)
class ExactAbsoluteAudit:
    """Frozen exact and factor row/column sums on the Gate-B reduced space."""

    exact_row_sums: torch.Tensor
    exact_column_sums: torch.Tensor
    factor_row_sums: torch.Tensor
    factor_column_sums: torch.Tensor
    exact_row_mask: torch.Tensor
    exact_column_mask: torch.Tensor
    dominance_violation_maximum: float
    dominance_relative_violation_maximum: float
    exact_nonzero_count: int
    factor_nonzero_count: int
    intersection_nonzero_count: int
    factor_only_nonzero_count: int
    exact_only_nonzero_count: int
    factor_active_column_count: int
    exact_active_column_count: int
    factor_only_active_column_count: int
    batch_size: int
    streamed_column_count: int
    setup_factor_row_relative_error: float
    setup_factor_column_relative_error: float
    replay_call_ledger: FactorPipelineCallLedger
    content_sha256: str

    @property
    def measurement_count(self) -> int:
        return int(self.exact_row_sums.numel())

    @property
    def active_primal_count(self) -> int:
        return int(self.exact_column_sums.numel())


@dataclass(frozen=True)
class ExactAbsoluteMetric:
    """One diagnostic diagonal metric using the same frozen A-only gauge."""

    mode: DiagnosticMetricMode
    tau: torch.Tensor
    sigma_rows: torch.Tensor
    content_sha256: str


def _relative_error(observed: torch.Tensor, expected: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm((observed - expected).to(torch.float64))
    denominator = torch.linalg.vector_norm(expected.to(torch.float64)).clamp_min(1e-30)
    return float((numerator / denominator).detach().cpu())


def _audit_digest(audit: ExactAbsoluteAudit) -> str:
    digest = hashlib.sha256()
    for name in (
        "exact_row_sums",
        "exact_column_sums",
        "factor_row_sums",
        "factor_column_sums",
        "exact_row_mask",
        "exact_column_mask",
    ):
        _tensor_digest(digest, name, getattr(audit, name))
    for value in (
        audit.dominance_violation_maximum,
        audit.dominance_relative_violation_maximum,
        audit.exact_nonzero_count,
        audit.factor_nonzero_count,
        audit.intersection_nonzero_count,
        audit.factor_only_nonzero_count,
        audit.exact_only_nonzero_count,
        audit.factor_active_column_count,
        audit.exact_active_column_count,
        audit.factor_only_active_column_count,
        audit.batch_size,
        audit.streamed_column_count,
        audit.setup_factor_row_relative_error,
        audit.setup_factor_column_relative_error,
        audit.replay_call_ledger,
    ):
        digest.update(repr(value).encode("ascii"))
    return digest.hexdigest()


def _metric_digest(mode: str, tau: torch.Tensor, sigma_rows: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(mode.encode("ascii"))
    _tensor_digest(digest, "tau", tau)
    _tensor_digest(digest, "sigma_rows", sigma_rows)
    return digest.hexdigest()


def _validate_audit(audit: ExactAbsoluteAudit) -> None:
    if not isinstance(audit, ExactAbsoluteAudit):
        raise TypeError("audit must be an ExactAbsoluteAudit")
    if audit.content_sha256 != _audit_digest(audit):
        raise RuntimeError("exact-absolute audit changed after construction")


def build_exact_absolute_audit(
    setup: GateBDataOnlySetup,
    *,
    batch_size: int = 64,
    nonzero_tolerance: float = 0.0,
    dominance_absolute_tolerance: float = 0.0,
    dominance_relative_tolerance: float = 0.0,
) -> ExactAbsoluteAudit:
    """Stream exact ``|A|`` and factor ``M`` sums over reduced basis columns."""

    describe_gate_b_metric(setup, "voxel_factor")
    width = int(batch_size)
    if width < 1:
        raise ValueError("batch_size must be positive")
    tolerance = float(nonzero_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("nonzero_tolerance must be finite and nonnegative")
    absolute_tolerance = float(dominance_absolute_tolerance)
    relative_tolerance = float(dominance_relative_tolerance)
    if not math.isfinite(absolute_tolerance) or absolute_tolerance < 0.0:
        raise ValueError("dominance_absolute_tolerance must be finite and nonnegative")
    if not math.isfinite(relative_tolerance) or relative_tolerance < 0.0:
        raise ValueError("dominance_relative_tolerance must be finite and nonnegative")

    pipeline = setup.pipeline
    measurement_count = pipeline.ray_count * 2
    reduced_count = setup.active_primal_count
    accumulator_device = torch.device("cpu")
    exact_rows = torch.zeros(measurement_count, dtype=torch.float64)
    factor_rows = torch.zeros_like(exact_rows)
    exact_columns = torch.zeros(reduced_count, dtype=torch.float64)
    factor_columns = torch.zeros_like(exact_columns)
    dominance_violation = 0.0
    dominance_relative_violation = 0.0
    exact_nonzero = 0
    factor_nonzero = 0
    intersection_nonzero = 0
    factor_only_nonzero = 0
    exact_only_nonzero = 0

    pipeline.reset_call_ledger()
    for start in range(0, reduced_count, width):
        stop = min(start + width, reduced_count)
        count = stop - start
        basis = torch.zeros(
            (count, pipeline.n_active),
            dtype=pipeline.dtype,
            device=pipeline.device,
        )
        selected = setup.active_primal_indices[start:stop]
        basis[
            torch.arange(count, device=pipeline.device),
            selected,
        ] = 1.0
        signed = pipeline.signed_data_forward(basis).reshape(count, -1).T
        majorizer = pipeline.absolute_data_forward(basis).reshape(count, -1).T
        if pipeline.device.type == "mps":
            torch.mps.synchronize()
        if torch.any(~torch.isfinite(signed)) or torch.any(~torch.isfinite(majorizer)):
            raise ValueError("streamed operator columns must be finite")
        if torch.any(majorizer < 0.0):
            raise ValueError("factor-majorizer columns must be nonnegative")

        exact = torch.abs(signed).to(device=accumulator_device, dtype=torch.float64)
        factor = majorizer.to(device=accumulator_device, dtype=torch.float64)
        if torch.any(~torch.isfinite(exact)) or torch.any(~torch.isfinite(factor)):
            raise ValueError("CPU-reduced exact and factor columns must be finite")
        exact_rows += torch.sum(exact, dim=1)
        factor_rows += torch.sum(factor, dim=1)
        exact_columns[start:stop] = torch.sum(exact, dim=0)
        factor_columns[start:stop] = torch.sum(factor, dim=0)
        positive_violation = torch.clamp(exact - factor, min=0.0)
        dominance_violation = max(dominance_violation, float(positive_violation.amax()))
        scale = torch.maximum(exact, factor).clamp_min(torch.finfo(torch.float64).tiny)
        dominance_relative_violation = max(
            dominance_relative_violation,
            float((positive_violation / scale).amax()),
        )
        exact_mask = exact > tolerance
        factor_mask = factor > tolerance
        exact_nonzero += int(torch.count_nonzero(exact_mask))
        factor_nonzero += int(torch.count_nonzero(factor_mask))
        intersection_nonzero += int(torch.count_nonzero(exact_mask & factor_mask))
        factor_only_nonzero += int(torch.count_nonzero((~exact_mask) & factor_mask))
        exact_only_nonzero += int(torch.count_nonzero(exact_mask & (~factor_mask)))

    expected_calls = math.ceil(reduced_count / width)
    ledger = pipeline.call_ledger()
    expected_ledger = FactorPipelineCallLedger(
        signed_data_forward_calls=expected_calls,
        absolute_data_forward_calls=expected_calls,
    )
    if ledger != expected_ledger:
        raise AssertionError("exact-absolute replay call ledger is inconsistent")

    setup_factor_rows = setup.data_row_sums.reshape(-1).detach().cpu().to(torch.float64)
    setup_factor_columns = (
        setup.data_column_sums.index_select(0, setup.active_primal_indices)
        .detach()
        .cpu()
        .to(torch.float64)
    )
    row_error = _relative_error(factor_rows, setup_factor_rows)
    column_error = _relative_error(factor_columns, setup_factor_columns)
    exact_row_mask = exact_rows > tolerance
    exact_column_mask = exact_columns > tolerance
    factor_column_mask = factor_columns > tolerance
    factor_active_column_count = int(torch.count_nonzero(factor_column_mask))
    exact_active_column_count = int(torch.count_nonzero(exact_column_mask))
    factor_only_active_column_count = int(
        torch.count_nonzero(factor_column_mask & (~exact_column_mask))
    )
    if (
        dominance_violation > absolute_tolerance
        and dominance_relative_violation > relative_tolerance
    ):
        raise ValueError(
            "streamed entrywise factor majorizer does not dominate the signed operator"
        )

    provisional = ExactAbsoluteAudit(
        exact_row_sums=exact_rows,
        exact_column_sums=exact_columns,
        factor_row_sums=factor_rows,
        factor_column_sums=factor_columns,
        exact_row_mask=exact_row_mask,
        exact_column_mask=exact_column_mask,
        dominance_violation_maximum=dominance_violation,
        dominance_relative_violation_maximum=dominance_relative_violation,
        exact_nonzero_count=exact_nonzero,
        factor_nonzero_count=factor_nonzero,
        intersection_nonzero_count=intersection_nonzero,
        factor_only_nonzero_count=factor_only_nonzero,
        exact_only_nonzero_count=exact_only_nonzero,
        factor_active_column_count=factor_active_column_count,
        exact_active_column_count=exact_active_column_count,
        factor_only_active_column_count=factor_only_active_column_count,
        batch_size=width,
        streamed_column_count=reduced_count,
        setup_factor_row_relative_error=row_error,
        setup_factor_column_relative_error=column_error,
        replay_call_ledger=ledger,
        content_sha256="",
    )
    return ExactAbsoluteAudit(
        **{
            **provisional.__dict__,
            "content_sha256": _audit_digest(provisional),
        }
    )


def describe_exact_absolute_metric(
    setup: GateBDataOnlySetup,
    audit: ExactAbsoluteAudit,
    mode: DiagnosticMetricMode,
) -> ExactAbsoluteMetric:
    """Build one frozen row/column diagnostic without changing the A-only gauge."""

    formal = describe_gate_b_metric(setup, "voxel_factor")
    _validate_audit(audit)
    if audit.measurement_count != setup.pipeline.ray_count * 2:
        raise ValueError("audit measurement count does not match setup")
    if audit.active_primal_count != setup.active_primal_count:
        raise ValueError("audit active-primal count does not match setup")

    device = setup.pipeline.device
    dtype = setup.pipeline.dtype
    exact_rows = audit.exact_row_sums.to(device=device, dtype=dtype)
    factor_rows = audit.factor_row_sums.to(device=device, dtype=dtype)
    exact_columns = audit.exact_column_sums.to(device=device, dtype=dtype)
    exact_row_mask = audit.exact_row_mask.to(device=device)

    if mode == "factor_row":
        tau = formal.tau.detach().clone()
        sigma_rows = torch.zeros_like(factor_rows)
        positive = factor_rows > 0.0
        sigma_rows[positive] = float(setup.eta) / factor_rows[positive]
    elif mode in {"exact_view", "exact_row"}:
        if not torch.all(audit.exact_column_mask):
            raise ValueError(
                "exact-|A| metric requires every frozen M-active coordinate to be A-nonzero"
            )
        tau = float(setup.eta) / exact_columns
        if mode == "exact_row":
            sigma_rows = torch.zeros_like(exact_rows)
            sigma_rows[exact_row_mask] = float(setup.eta) / exact_rows[exact_row_mask]
        else:
            reshaped = exact_rows.reshape(
                setup.pipeline.view_count,
                setup.pipeline.rays_per_view,
                2,
            )
            rho_by_view = reshaped.amax(dim=(1, 2))
            sigma_by_view = torch.zeros_like(rho_by_view)
            active_views = rho_by_view > 0.0
            sigma_by_view[active_views] = float(setup.eta) / rho_by_view[active_views]
            sigma_rows = sigma_by_view[:, None, None].expand_as(reshaped).reshape(-1)
            sigma_rows = torch.where(exact_row_mask, sigma_rows, 0.0)
    else:
        raise ValueError("unsupported exact-absolute diagnostic mode")

    if torch.any(~torch.isfinite(tau)) or torch.any(tau <= 0.0):
        raise ValueError("diagnostic tau must be finite and positive")
    if torch.any(~torch.isfinite(sigma_rows)) or torch.any(sigma_rows < 0.0):
        raise ValueError("diagnostic sigma must be finite and nonnegative")
    return ExactAbsoluteMetric(
        mode=mode,
        tau=tau,
        sigma_rows=sigma_rows,
        content_sha256=_metric_digest(mode, tau, sigma_rows),
    )


def _validate_metric(metric: ExactAbsoluteMetric) -> None:
    if not isinstance(metric, ExactAbsoluteMetric):
        raise TypeError("metric must be an ExactAbsoluteMetric")
    if metric.content_sha256 != _metric_digest(metric.mode, metric.tau, metric.sigma_rows):
        raise RuntimeError("exact-absolute metric changed after construction")


def _expanded_active(setup: GateBDataOnlySetup, reduced: torch.Tensor) -> torch.Tensor:
    active = reduced.new_zeros(setup.pipeline.n_active)
    active.index_copy_(0, setup.active_primal_indices, reduced)
    return active


def run_exact_absolute_trajectory(
    setup: GateBDataOnlySetup,
    target: Any,
    metric: ExactAbsoluteMetric,
    *,
    checkpoints: Sequence[int],
    theta: float = 1.0,
) -> Mapping[int, PSUB0FactorPDHGState]:
    """Replay one maximum-depth diagnostic trajectory with exact call accounting."""

    describe_gate_b_metric(setup, "voxel_factor")
    _validate_metric(metric)
    expected_target = (
        setup.pipeline.view_count,
        setup.pipeline.rays_per_view,
        2,
    )
    if not isinstance(target, torch.Tensor) or tuple(target.shape) != expected_target:
        raise ValueError(f"target must have shape {expected_target}")
    if target.device != setup.pipeline.device or target.dtype != setup.pipeline.dtype:
        raise ValueError("target must match setup dtype and device")
    ordered = tuple(sorted({int(value) for value in checkpoints}))
    if not ordered or ordered[0] < 1:
        raise ValueError("checkpoints must contain positive iterations")
    relaxation = float(theta)
    if not math.isfinite(relaxation) or not 0.0 <= relaxation <= 1.0:
        raise ValueError("theta must lie in [0,1]")
    if metric.tau.shape != (setup.active_primal_count,):
        raise ValueError("diagnostic tau shape does not match setup")
    if metric.sigma_rows.shape != (setup.pipeline.ray_count * 2,):
        raise ValueError("diagnostic sigma shape does not match setup")

    state = initial_gate_b_state(setup)
    output: dict[int, PSUB0FactorPDHGState] = {}
    sigma = metric.sigma_rows.reshape(expected_target)
    for iteration in range(1, max(ordered) + 1):
        active_x_bar = _expanded_active(setup, state.x_bar)
        projected = setup.pipeline.signed_data_forward(active_x_bar[None])[0].reshape(
            expected_target
        )
        candidate = (
            state.data_dual + sigma * (projected - target)
        ) / (1.0 + sigma)
        data_dual = torch.where(sigma > 0.0, candidate, torch.zeros_like(candidate))
        gradient_full = setup.pipeline.signed_data_transpose(
            data_dual.reshape(1, setup.pipeline.ray_count, 2)
        )[0]
        gradient = gradient_full.index_select(0, setup.active_primal_indices)
        next_x = state.x - metric.tau * gradient
        next_x_bar = next_x + relaxation * (next_x - state.x)
        state = PSUB0FactorPDHGState(
            x=next_x,
            x_bar=next_x_bar,
            data_dual=data_dual,
            tv_dual=torch.zeros_like(state.tv_dual),
        )
        if iteration in ordered:
            output[iteration] = state
    return output


def normalized_operator_power_estimate(
    setup: GateBDataOnlySetup,
    metric: ExactAbsoluteMetric,
    *,
    iterations: int = 32,
    seed: int = 2026071703,
) -> float:
    """Estimate ``||Sigma^(1/2) A T^(1/2)||_2^2`` by fixed power iteration.

    This is a stress check, not an upper bound.  The production safety
    certificate comes from verified entrywise majorization and Schur row/column
    sums.
    """

    describe_gate_b_metric(setup, "voxel_factor")
    _validate_metric(metric)
    count = int(iterations)
    if count < 1:
        raise ValueError("iterations must be positive")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    vector = torch.randn(
        setup.active_primal_count,
        generator=generator,
        dtype=setup.pipeline.dtype,
    ).to(setup.pipeline.device)
    vector /= torch.linalg.vector_norm(vector).clamp_min(1e-30)
    sqrt_tau = torch.sqrt(metric.tau)
    sqrt_sigma = torch.sqrt(metric.sigma_rows).reshape(
        setup.pipeline.view_count,
        setup.pipeline.rays_per_view,
        2,
    )
    eigenvalue = vector.new_tensor(0.0)
    for _ in range(count):
        active = _expanded_active(setup, sqrt_tau * vector)
        projected = setup.pipeline.signed_data_forward(active[None])[0].reshape_as(
            sqrt_sigma
        )
        weighted = sqrt_sigma * projected
        transposed = setup.pipeline.signed_data_transpose(
            (sqrt_sigma * weighted).reshape(1, setup.pipeline.ray_count, 2)
        )[0]
        next_vector = sqrt_tau * transposed.index_select(
            0,
            setup.active_primal_indices,
        )
        norm = torch.linalg.vector_norm(next_vector)
        if not torch.isfinite(norm) or float(norm.detach().cpu()) <= 0.0:
            raise ValueError("normalized operator power iteration collapsed")
        vector = next_vector / norm
        eigenvalue = norm
    return float(eigenvalue.detach().cpu())


def schur_safety_certificate_squared(
    setup: GateBDataOnlySetup,
    audit: ExactAbsoluteAudit,
    metric: ExactAbsoluteMetric,
    *,
    dominance_absolute_tolerance: float,
    dominance_relative_tolerance: float,
) -> float:
    """Return the theorem-backed Schur upper bound after numeric premises pass."""

    describe_gate_b_metric(setup, "voxel_factor")
    _validate_audit(audit)
    _validate_metric(metric)
    absolute_tolerance = float(dominance_absolute_tolerance)
    relative_tolerance = float(dominance_relative_tolerance)
    if (
        audit.dominance_violation_maximum > absolute_tolerance
        and audit.dominance_relative_violation_maximum > relative_tolerance
    ):
        raise ValueError("entrywise majorization premise failed")
    if audit.factor_only_active_column_count != 0 and metric.mode != "factor_row":
        raise ValueError("exact metric is undefined on M-active/A-zero coordinates")

    exact_rows = audit.exact_row_sums.to(metric.sigma_rows)
    exact_columns = audit.exact_column_sums.to(metric.tau)
    row_products = metric.sigma_rows * exact_rows
    column_products = metric.tau * exact_columns
    eta = float(setup.eta)
    numeric_tolerance = 32.0 * torch.finfo(metric.tau.dtype).eps
    if float(torch.amax(row_products).detach().cpu()) > eta * (1.0 + numeric_tolerance):
        raise ValueError("row scaling violates the Schur premise")
    if float(torch.amax(column_products).detach().cpu()) > eta * (
        1.0 + numeric_tolerance
    ):
        raise ValueError("column scaling violates the Schur premise")
    return eta * eta


def dense_normalized_spectral_norm_squared(
    dense_a: torch.Tensor,
    metric: ExactAbsoluteMetric,
) -> float:
    """Independent tiny-oracle safety check for a materialized signed matrix."""

    _validate_metric(metric)
    if dense_a.ndim != 2:
        raise ValueError("dense_a must be a matrix")
    if dense_a.shape != (metric.sigma_rows.numel(), metric.tau.numel()):
        raise ValueError("dense_a shape does not match metric")
    scaled = (
        torch.sqrt(metric.sigma_rows.to(dense_a))[:, None]
        * dense_a
        * torch.sqrt(metric.tau.to(dense_a))[None, :]
    )
    return float(torch.linalg.matrix_norm(scaled, ord=2).square())


def summarize_tightness(audit: ExactAbsoluteAudit) -> dict[str, Any]:
    """Return ratio statistics that avoid unstable entrywise division by zero."""

    _validate_audit(audit)
    row_mask = audit.factor_row_sums > 0.0
    column_mask = audit.factor_column_sums > 0.0
    row_ratio = audit.exact_row_sums[row_mask] / audit.factor_row_sums[row_mask]
    column_ratio = (
        audit.exact_column_sums[column_mask] / audit.factor_column_sums[column_mask]
    )
    if row_ratio.numel() == 0 or column_ratio.numel() == 0:
        raise ValueError("tightness summary requires active rows and columns")

    def stats(values: torch.Tensor) -> Mapping[str, float]:
        return {
            "minimum": float(torch.amin(values)),
            "p05": float(torch.quantile(values, 0.05)),
            "median": float(torch.median(values)),
            "mean": float(torch.mean(values)),
            "maximum": float(torch.amax(values)),
        }

    exact_mass = float(torch.sum(audit.exact_row_sums))
    factor_mass = float(torch.sum(audit.factor_row_sums))
    return {
        "row_ratio": dict(stats(row_ratio)),
        "column_ratio": dict(stats(column_ratio)),
        "global_exact_to_factor_mass_ratio": exact_mass / factor_mass,
        "global_slack_mass": 1.0 - exact_mass / factor_mass,
        "exact_zero_row_count": int(torch.count_nonzero(~audit.exact_row_mask)),
        "exact_zero_column_count": int(torch.count_nonzero(~audit.exact_column_mask)),
        "exact_nonzero_count": audit.exact_nonzero_count,
        "factor_nonzero_count": audit.factor_nonzero_count,
        "intersection_nonzero_count": audit.intersection_nonzero_count,
        "factor_only_nonzero_count": audit.factor_only_nonzero_count,
        "exact_only_nonzero_count": audit.exact_only_nonzero_count,
        "factor_active_column_count": audit.factor_active_column_count,
        "exact_active_column_count": audit.exact_active_column_count,
        "factor_only_active_column_count": audit.factor_only_active_column_count,
        "dominance_violation_maximum": audit.dominance_violation_maximum,
        "dominance_relative_violation_maximum": (
            audit.dominance_relative_violation_maximum
        ),
        "setup_factor_row_relative_error": audit.setup_factor_row_relative_error,
        "setup_factor_column_relative_error": audit.setup_factor_column_relative_error,
    }


__all__ = [
    "DiagnosticMetricMode",
    "EXACT_ABSOLUTE_DIAGNOSTIC_SCHEMA",
    "ExactAbsoluteAudit",
    "ExactAbsoluteMetric",
    "build_exact_absolute_audit",
    "dense_normalized_spectral_norm_squared",
    "describe_exact_absolute_metric",
    "normalized_operator_power_estimate",
    "run_exact_absolute_trajectory",
    "schur_safety_certificate_squared",
    "summarize_tightness",
]
