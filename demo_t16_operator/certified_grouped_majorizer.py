"""Certified grouped majorizers for a tiny signed-operator smoke test.

For primitive signed contributions ``A = sum_l C_l`` and any partition ``P``
of the primitive indices, the grouped matrix

    M_P = sum_{G in P} |sum_{l in G} C_l|

satisfies ``M_P >= |A|`` entrywise by the triangle inequality.  A selector may
therefore choose among predeclared partitions without predicting metric values
and without weakening the deterministic diagonal Schur certificate.

This module is synthetic development infrastructure.  It does not model a
complete BOST forward operator and makes no real-data or superiority claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
from time import perf_counter
from typing import Any

import torch


SCHEMA_VERSION = "certified-grouped-majorizer-smoke-1.0"
STATUS = "DEVELOPMENT_ONLY_SYNTHETIC_CERTIFIED_PARTITION_SMOKE"
SPLIT_ROLES = ("train", "safety_calibration", "fresh_geometry_ood")
PRIMITIVE_COUNT = 6


@dataclass(frozen=True)
class PartitionSpec:
    """A complete, disjoint partition of primitive indices."""

    name: str
    groups: tuple[tuple[int, ...], ...]
    oracle_only: bool = False


@dataclass(frozen=True)
class TinyPrimitiveRig:
    """Full synthetic rig; teachers and operator data never enter inference."""

    rig_id: str
    split_role: str
    geometry_seed_sha256: str
    noise_seed_sha256: str
    geometry_feature_sha256: str
    geometry_features: torch.Tensor
    primitives: torch.Tensor
    signed_matrix: torch.Tensor
    truth: torch.Tensor
    target: torch.Tensor


@dataclass(frozen=True)
class DeploymentGeometry:
    """The only object accepted by the partition selector at inference."""

    rig_id: str
    geometry_features: torch.Tensor


@dataclass(frozen=True)
class GroupedMajorizer:
    partition_name: str
    matrix: torch.Tensor
    row_mass: torch.Tensor
    column_mass: torch.Tensor
    construction_cost: dict[str, int | str]


@dataclass(frozen=True)
class DiagonalMetric:
    row_mass: torch.Tensor
    column_mass: torch.Tensor
    sigma: torch.Tensor
    tau: torch.Tensor
    eta: float


@dataclass(frozen=True)
class TrajectoryResult:
    rows: list[dict[str, float | int]]
    ledger: dict[str, int]
    wall_time_seconds: float


@dataclass(frozen=True)
class GeometryPartitionStump:
    """A depth-one, geometry-only partition selector."""

    feature_index: int
    threshold: float
    left_partition: str
    right_partition: str
    allowed_partitions: tuple[str, ...]
    exact_oracle_partition: str


def predefined_partitions(
    primitive_count: int = PRIMITIVE_COUNT,
) -> dict[str, PartitionSpec]:
    """Return the frozen partition catalogue used by this smoke test."""

    if int(primitive_count) != PRIMITIVE_COUNT:
        raise ValueError(f"this smoke requires exactly {PRIMITIVE_COUNT} primitives")
    specs = {
        "singleton_factor": PartitionSpec(
            "singleton_factor", ((0,), (1,), (2,), (3,), (4,), (5,))
        ),
        "paired_local": PartitionSpec(
            "paired_local", ((0, 1), (2, 3), (4, 5))
        ),
        "paired_cross": PartitionSpec(
            "paired_cross", ((0, 3), (1, 4), (2, 5))
        ),
        "triad_bridge": PartitionSpec(
            "triad_bridge", ((0, 1, 4), (2, 3, 5))
        ),
        "all_in_one_exact": PartitionSpec(
            "all_in_one_exact", ((0, 1, 2, 3, 4, 5),), oracle_only=True
        ),
    }
    for spec in specs.values():
        validate_partition(spec, primitive_count=primitive_count)
    return specs


def validate_partition(spec: PartitionSpec, *, primitive_count: int) -> None:
    if not isinstance(spec, PartitionSpec) or not spec.name:
        raise ValueError("partition must be a named PartitionSpec")
    if not spec.groups:
        raise ValueError("partition must contain at least one group")
    flattened: list[int] = []
    for group in spec.groups:
        if not group:
            raise ValueError("partition groups must be nonempty")
        if tuple(sorted(group)) != group or len(set(group)) != len(group):
            raise ValueError("partition groups must contain sorted unique indices")
        flattened.extend(group)
    if sorted(flattened) != list(range(int(primitive_count))):
        raise ValueError("partition must cover every primitive exactly once")


def deployment_geometry_from_rig(rig: TinyPrimitiveRig) -> DeploymentGeometry:
    """Project a complete rig onto the deployment-visible contract."""

    return DeploymentGeometry(
        rig_id=rig.rig_id,
        geometry_features=rig.geometry_features.detach().clone(),
    )


def _uniform(
    generator: torch.Generator, low: float, high: float, *, dtype: torch.dtype
) -> float:
    value = float(torch.rand((), generator=generator, dtype=dtype))
    return low + (high - low) * value


def _signed_outer_range(
    generator: torch.Generator, low: float, high: float, *, dtype: torch.dtype
) -> float:
    magnitude = _uniform(generator, low, high, dtype=dtype)
    return magnitude if _uniform(generator, 0.0, 1.0, dtype=dtype) >= 0.5 else -magnitude


def stable_rig_seed(base_seed: int, rig_id: str, split_role: str) -> tuple[int, str]:
    """Derive an order-invariant seed from stable rig identity."""

    if not isinstance(base_seed, int) or isinstance(base_seed, bool):
        raise ValueError("base_seed must be an integer")
    if not isinstance(rig_id, str) or not rig_id:
        raise ValueError("rig_id must be a nonempty string")
    if split_role not in SPLIT_ROLES:
        raise ValueError("split_role is unknown")
    payload = f"{base_seed}\x1f{rig_id}\x1f{split_role}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    seed = int.from_bytes(bytes.fromhex(digest[:16]), "big") % (2**63 - 1)
    return seed, digest


def geometry_feature_sha256(features: torch.Tensor) -> str:
    """Hash a canonical float64 text representation of one geometry vector."""

    if features.ndim != 1 or not bool(torch.all(torch.isfinite(features))):
        raise ValueError("geometry features must be a finite vector")
    payload = ",".join(format(float(value), ".17g") for value in features)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _sample_geometry(
    *, generator: torch.Generator, split_role: str, dtype: torch.dtype
) -> tuple[float, float, float, float, float, float]:
    """Sample six coordinates independently; fresh support is deliberately OOD."""

    if split_role not in SPLIT_ROLES:
        raise ValueError(f"unknown split role: {split_role}")
    if split_role == "fresh_geometry_ood":
        regime = _signed_outer_range(generator, 1.02, 1.30, dtype=dtype)
        angle = _signed_outer_range(generator, 0.90, 1.18, dtype=dtype)
        aperture = (
            _uniform(generator, 0.48, 0.62, dtype=dtype)
            if _uniform(generator, 0.0, 1.0, dtype=dtype) < 0.5
            else _uniform(generator, 1.04, 1.16, dtype=dtype)
        )
        shear = _signed_outer_range(generator, 0.44, 0.64, dtype=dtype)
        frequency = (
            _uniform(generator, 0.55, 0.70, dtype=dtype)
            if _uniform(generator, 0.0, 1.0, dtype=dtype) < 0.5
            else _uniform(generator, 1.28, 1.48, dtype=dtype)
        )
        cancellation = _uniform(generator, 0.96, 1.12, dtype=dtype)
    else:
        regime = _uniform(generator, -0.90, 0.90, dtype=dtype)
        angle = _uniform(generator, -0.80, 0.80, dtype=dtype)
        aperture = _uniform(generator, 0.66, 1.00, dtype=dtype)
        shear = _uniform(generator, -0.36, 0.36, dtype=dtype)
        frequency = _uniform(generator, 0.80, 1.20, dtype=dtype)
        cancellation = _uniform(generator, 0.70, 0.96, dtype=dtype)
    return regime, angle, aperture, shear, frequency, cancellation


def generate_tiny_primitive_rigs(
    *,
    split_assignments: Mapping[str, str],
    geometry_seed: int,
    noise_seed: int,
    row_count: int,
    column_count: int,
    primitive_count: int = PRIMITIVE_COUNT,
    dtype: torch.dtype = torch.float64,
) -> list[TinyPrimitiveRig]:
    """Generate independent geometry rigs with six signed primitive matrices."""

    if not split_assignments:
        raise ValueError("at least one split assignment is required")
    if set(split_assignments.values()).difference(SPLIT_ROLES):
        raise ValueError("split assignments contain an unknown role")
    if int(primitive_count) != PRIMITIVE_COUNT:
        raise ValueError(f"primitive_count must equal {PRIMITIVE_COUNT}")
    rows_n, columns_n = int(row_count), int(column_count)
    if rows_n < 4 or columns_n < 4:
        raise ValueError("matrix dimensions must be at least four")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("tiny rigs require a floating-point CPU dtype")

    row_coordinate = torch.linspace(-1.0, 1.0, rows_n, dtype=dtype)
    column_coordinate = torch.linspace(-1.0, 1.0, columns_n, dtype=dtype)
    rr = row_coordinate[:, None]
    cc = column_coordinate[None, :]
    rigs: list[TinyPrimitiveRig] = []

    for rig_id, split_role in split_assignments.items():
        resolved_geometry_seed, geometry_seed_digest = stable_rig_seed(
            int(geometry_seed), rig_id, split_role
        )
        resolved_noise_seed, noise_seed_digest = stable_rig_seed(
            int(noise_seed), rig_id, split_role
        )
        geometry_generator = torch.Generator(device="cpu").manual_seed(
            resolved_geometry_seed
        )
        noise_generator = torch.Generator(device="cpu").manual_seed(resolved_noise_seed)
        regime, angle, aperture, shear, frequency, cancellation = _sample_geometry(
            generator=geometry_generator, split_role=split_role, dtype=dtype
        )

        center_zero = 0.34 * rr * math.cos(angle) + 0.22 * shear
        center_one = -0.26 * rr * math.sin(angle) - 0.18 * shear
        template_zero = aperture * torch.exp(
            -0.5 * ((cc - center_zero) / (0.24 + 0.08 * aperture)).square()
        )
        template_one = (1.10 - 0.25 * aperture) * torch.exp(
            -0.5 * ((cc - center_one) / (0.21 + 0.05 * aperture)).square()
        )
        template_two = 0.62 + 0.38 * torch.cos(
            math.pi * frequency * (0.78 * rr - 0.92 * cc + 0.18 * angle)
        )
        template_zero *= 1.0 + 0.10 * torch.cos(math.pi * (rr + cc))
        template_one *= 1.0 + 0.08 * torch.sin(math.pi * (rr - 0.6 * cc))
        template_two *= 0.52 + 0.10 * torch.cos(math.pi * (rr + shear))
        signed_carrier = 0.14 * torch.sin(
            math.pi * (1.18 * rr + 0.86 * cc + 0.22 * angle)
        )

        local_weight = 0.5 * (math.tanh(2.8 * regime) + 1.0)
        cross_weight = 1.0 - local_weight
        jitter = 0.008 * torch.randn(
            (PRIMITIVE_COUNT, rows_n, columns_n),
            generator=noise_generator,
            dtype=dtype,
        )
        positive = (template_zero, template_one, template_two)
        negative = (
            local_weight * template_zero + cross_weight * template_two,
            local_weight * template_one + cross_weight * template_zero,
            local_weight * template_two + cross_weight * template_one,
        )
        primitives = torch.stack(
            (
                positive[0] + jitter[0],
                -cancellation * negative[0] + jitter[1],
                positive[1] + jitter[2],
                -cancellation * negative[1] + jitter[3],
                positive[2] + signed_carrier + jitter[4],
                -cancellation * negative[2] + jitter[5],
            )
        )
        signed_matrix = torch.sum(primitives, dim=0)
        geometry = torch.tensor(
            [regime, angle, aperture, shear, frequency, cancellation], dtype=dtype
        )
        truth = (
            0.22
            + torch.exp(-((column_coordinate - 0.24 * shear) / 0.31).square())
            + 0.16 * torch.cos(math.pi * (column_coordinate + 0.20 * angle))
        ).clamp_min(0.02)
        noiseless = signed_matrix @ truth
        noise_scale = 0.003 * torch.linalg.vector_norm(noiseless) / math.sqrt(rows_n)
        target = noiseless + noise_scale * torch.randn(
            rows_n, generator=noise_generator, dtype=dtype
        )

        if not bool(torch.all(torch.isfinite(primitives))):
            raise RuntimeError("synthetic primitives are non-finite")
        if not bool(torch.any(signed_matrix < 0.0) and torch.any(signed_matrix > 0.0)):
            raise RuntimeError("synthetic signed matrix lost sign variation")
        rigs.append(
            TinyPrimitiveRig(
                rig_id=rig_id,
                split_role=split_role,
                geometry_seed_sha256=geometry_seed_digest,
                noise_seed_sha256=noise_seed_digest,
                geometry_feature_sha256=geometry_feature_sha256(geometry),
                geometry_features=geometry,
                primitives=primitives,
                signed_matrix=signed_matrix,
                truth=truth,
                target=target,
            )
        )
    return rigs


def split_rigs_three_way(
    rigs: Sequence[TinyPrimitiveRig],
) -> tuple[list[TinyPrimitiveRig], list[TinyPrimitiveRig], list[TinyPrimitiveRig], dict[str, Any]]:
    if not rigs or len({rig.rig_id for rig in rigs}) != len(rigs):
        raise ValueError("rigs must be nonempty with unique ids")
    grouped = {role: [rig for rig in rigs if rig.split_role == role] for role in SPLIT_ROLES}
    if any(not grouped[role] for role in SPLIT_ROLES):
        raise ValueError("all three split roles require at least one complete rig")
    contract = {
        "split_unit": "COMPLETE_RIG",
        "random_ray_split_used": False,
        "geometry_parameters_independently_sampled": True,
        "noise_seed_is_not_geometry_seed": True,
        "train_rig_ids": [rig.rig_id for rig in grouped["train"]],
        "safety_calibration_rig_ids": [
            rig.rig_id for rig in grouped["safety_calibration"]
        ],
        "fresh_geometry_ood_rig_ids": [
            rig.rig_id for rig in grouped["fresh_geometry_ood"]
        ],
    }
    return (
        grouped["train"],
        grouped["safety_calibration"],
        grouped["fresh_geometry_ood"],
        contract,
    )


def construction_cost_proxy(
    *, matrix_shape: tuple[int, int], spec: PartitionSpec, primitive_count: int
) -> dict[str, int | str]:
    """Return a frozen analytic proxy, not a hardware runtime estimate."""

    validate_partition(spec, primitive_count=primitive_count)
    entries = int(matrix_shape[0]) * int(matrix_shape[1])
    group_sizes = [len(group) for group in spec.groups]
    component_entries = int(primitive_count) * entries
    signed_addition_entries = sum(size - 1 for size in group_sizes) * entries
    group_abs_entries = len(group_sizes) * entries
    materialized_group_component_entries = sum(
        size * entries for size in group_sizes if size > 1
    )
    largest_fused_group_component_entries = max(group_sizes) * entries
    proxy_units = (
        component_entries
        + signed_addition_entries
        + group_abs_entries
        + materialized_group_component_entries
        + largest_fused_group_component_entries
    )
    return {
        "definition": "ANALYTIC_PROXY_NOT_WALL_TIME",
        "primitive_component_accumulation_entries": component_entries,
        "signed_group_addition_entries": signed_addition_entries,
        "group_absolute_value_entries": group_abs_entries,
        "materialized_signed_group_count": sum(size > 1 for size in group_sizes),
        "materialized_group_component_entries": materialized_group_component_entries,
        "largest_fused_group_component_entries": largest_fused_group_component_entries,
        "full_signed_operator_materialized": int(
            len(group_sizes) == 1 and group_sizes[0] == int(primitive_count)
        ),
        "cost_proxy_units": proxy_units,
    }


def build_grouped_majorizer(
    primitives: torch.Tensor, spec: PartitionSpec
) -> GroupedMajorizer:
    """Construct ``M_P`` and verify the entrywise triangle-inequality bound."""

    if primitives.ndim != 3 or not bool(torch.all(torch.isfinite(primitives))):
        raise ValueError("primitives must be a finite [component,row,column] tensor")
    primitive_count = int(primitives.shape[0])
    validate_partition(spec, primitive_count=primitive_count)
    grouped = torch.zeros_like(primitives[0])
    for group in spec.groups:
        grouped = grouped + torch.abs(torch.sum(primitives[list(group)], dim=0))
    signed_matrix = torch.sum(primitives, dim=0)
    tolerance = 128.0 * torch.finfo(primitives.dtype).eps
    if bool(torch.any(grouped + tolerance < torch.abs(signed_matrix))):
        raise RuntimeError("grouped majorizer violated the triangle inequality")
    return GroupedMajorizer(
        partition_name=spec.name,
        matrix=grouped,
        row_mass=torch.sum(grouped, dim=1),
        column_mass=torch.sum(grouped, dim=0),
        construction_cost=construction_cost_proxy(
            matrix_shape=(int(primitives.shape[1]), int(primitives.shape[2])),
            spec=spec,
            primitive_count=primitive_count,
        ),
    )


def build_diagonal_metric(majorizer: GroupedMajorizer, *, eta: float) -> DiagonalMetric:
    if not math.isfinite(eta) or not 0.0 < eta < 1.0:
        raise ValueError("eta must lie strictly between zero and one")
    for name, mass in (("row", majorizer.row_mass), ("column", majorizer.column_mass)):
        if mass.ndim != 1 or not bool(torch.all(torch.isfinite(mass))):
            raise ValueError(f"{name} mass must be a finite vector")
        if bool(torch.any(mass < 0.0)):
            raise ValueError(f"{name} mass must be nonnegative")
    row_active = majorizer.row_mass > 0.0
    column_active = majorizer.column_mass > 0.0
    sigma = torch.zeros_like(majorizer.row_mass)
    tau = torch.zeros_like(majorizer.column_mass)
    sigma[row_active] = float(eta) / majorizer.row_mass[row_active]
    tau[column_active] = float(eta) / majorizer.column_mass[column_active]
    return DiagonalMetric(
        row_mass=majorizer.row_mass.clone(),
        column_mass=majorizer.column_mass.clone(),
        sigma=sigma,
        tau=tau,
        eta=float(eta),
    )


def audit_partition_safety(
    primitives: torch.Tensor,
    spec: PartitionSpec,
    metric: DiagonalMetric,
    *,
    eta: float,
) -> dict[str, float | int | bool | str]:
    """Reconstruct every safety quantity instead of trusting caller teachers."""

    rebuilt = build_grouped_majorizer(primitives, spec)
    signed_matrix = torch.sum(primitives, dim=0)
    if signed_matrix.shape != (metric.sigma.numel(), metric.tau.numel()):
        raise ValueError("metric and signed matrix shapes differ")
    if float(eta) != metric.eta:
        raise ValueError("eta differs from the frozen metric")
    eps = torch.finfo(primitives.dtype).eps
    tolerance = 256.0 * eps
    exact_abs = torch.abs(signed_matrix)
    pointwise_violation_count = int(
        torch.count_nonzero(rebuilt.matrix + tolerance < exact_abs)
    )
    row_mass_mismatch_count = int(
        torch.count_nonzero(
            torch.abs(rebuilt.row_mass - metric.row_mass)
            > tolerance * torch.maximum(torch.ones_like(rebuilt.row_mass), rebuilt.row_mass)
        )
    )
    column_mass_mismatch_count = int(
        torch.count_nonzero(
            torch.abs(rebuilt.column_mass - metric.column_mass)
            > tolerance
            * torch.maximum(torch.ones_like(rebuilt.column_mass), rebuilt.column_mass)
        )
    )
    exact_row = torch.sum(exact_abs, dim=1)
    exact_column = torch.sum(exact_abs, dim=0)
    support_violation_count = int(
        torch.count_nonzero((rebuilt.row_mass == 0.0) & (exact_row > tolerance))
        + torch.count_nonzero(
            (rebuilt.column_mass == 0.0) & (exact_column > tolerance)
        )
    )
    row_product = metric.sigma * exact_row
    column_product = metric.tau * exact_column
    threshold = float(eta) * (1.0 + tolerance)
    row_violation_count = int(torch.count_nonzero(row_product > threshold))
    column_violation_count = int(torch.count_nonzero(column_product > threshold))
    normalized = (
        torch.sqrt(metric.sigma)[:, None]
        * signed_matrix
        * torch.sqrt(metric.tau)[None, :]
    )
    spectral_squared = float(torch.linalg.matrix_norm(normalized, ord=2).square())
    spectral_bound = float(eta) ** 2
    spectral_violation_count = int(
        spectral_squared > spectral_bound * (1.0 + tolerance)
    )
    total = (
        pointwise_violation_count
        + row_mass_mismatch_count
        + column_mass_mismatch_count
        + support_violation_count
        + row_violation_count
        + column_violation_count
        + spectral_violation_count
    )
    return {
        "partition_name": spec.name,
        "pointwise_violation_count": pointwise_violation_count,
        "row_mass_mismatch_count": row_mass_mismatch_count,
        "column_mass_mismatch_count": column_mass_mismatch_count,
        "support_violation_count": support_violation_count,
        "row_violation_count": row_violation_count,
        "column_violation_count": column_violation_count,
        "spectral_violation_count": spectral_violation_count,
        "total_violation_count": total,
        "maximum_entrywise_slack": float(torch.amax(rebuilt.matrix - exact_abs)),
        "minimum_entrywise_slack": float(torch.amin(rebuilt.matrix - exact_abs)),
        "maximum_row_product": float(torch.amax(row_product)),
        "maximum_column_product": float(torch.amax(column_product)),
        "dense_normalized_spectral_norm_squared": spectral_squared,
        "schur_squared_upper_bound": spectral_bound,
        "active_row_count": int(torch.count_nonzero(rebuilt.row_mass > 0.0)),
        "active_column_count": int(torch.count_nonzero(rebuilt.column_mass > 0.0)),
        "exact_masses_recomputed_from_signed_primitives": True,
        "triangle_inequality_certificate": "ENTRYWISE_GROUPED_SUM_DOMINATES_ABS_TOTAL",
    }


def run_signed_pdhg_trajectory(
    signed_matrix: torch.Tensor,
    target: torch.Tensor,
    truth: torch.Tensor,
    metric: DiagonalMetric,
    *,
    checkpoints: Sequence[int],
    theta: float,
) -> TrajectoryResult:
    """Run an equal-budget A/A^T trajectory and report field relative L2."""

    if signed_matrix.ndim != 2 or target.ndim != 1 or truth.ndim != 1:
        raise ValueError("matrix, target, and truth dimensions are invalid")
    if signed_matrix.shape != (target.numel(), truth.numel()):
        raise ValueError("matrix, target, and truth shapes differ")
    if signed_matrix.shape != (metric.sigma.numel(), metric.tau.numel()):
        raise ValueError("metric shape differs from signed_matrix")
    if not math.isfinite(theta) or not 0.0 <= theta <= 1.0:
        raise ValueError("theta must lie in [0, 1]")
    ordered = sorted({int(value) for value in checkpoints})
    if not ordered or ordered[0] != 0 or ordered[-1] < 1 or any(v < 0 for v in ordered):
        raise ValueError("checkpoints must include zero and a positive endpoint")

    started = perf_counter()
    x = torch.zeros_like(truth)
    x_bar = x.clone()
    dual = torch.zeros_like(target)
    target_norm = torch.linalg.vector_norm(target).clamp_min(1e-30)
    truth_norm = torch.linalg.vector_norm(truth).clamp_min(1e-30)
    rows: list[dict[str, float | int]] = []
    forward_solver_calls = transpose_solver_calls = forward_evaluation_calls = 0

    def record(iteration: int) -> None:
        nonlocal forward_evaluation_calls
        residual = signed_matrix @ x - target
        forward_evaluation_calls += 1
        rows.append(
            {
                "iteration": iteration,
                "normalized_residual_l2": float(
                    torch.linalg.vector_norm(residual) / target_norm
                ),
                "field_relative_l2": float(
                    torch.linalg.vector_norm(x - truth) / truth_norm
                ),
                "solution_l2": float(torch.linalg.vector_norm(x)),
            }
        )

    record(0)
    checkpoint_set = set(ordered)
    for iteration in range(1, ordered[-1] + 1):
        projected = signed_matrix @ x_bar
        forward_solver_calls += 1
        dual = (dual + metric.sigma * (projected - target)) / (1.0 + metric.sigma)
        gradient = signed_matrix.T @ dual
        transpose_solver_calls += 1
        next_x = x - metric.tau * gradient
        x_bar = next_x + float(theta) * (next_x - x)
        x = next_x
        if iteration in checkpoint_set:
            record(iteration)
    wall = perf_counter() - started
    if not all(
        math.isfinite(float(row["field_relative_l2"]))
        and math.isfinite(float(row["normalized_residual_l2"]))
        for row in rows
    ):
        raise RuntimeError("trajectory produced non-finite metrics")
    return TrajectoryResult(
        rows=rows,
        ledger={
            "signed_forward_solver_calls": forward_solver_calls,
            "signed_transpose_solver_calls": transpose_solver_calls,
            "signed_forward_evaluation_calls": forward_evaluation_calls,
            "field_error_evaluation_calls": len(rows),
            "iteration_budget": ordered[-1],
        },
        wall_time_seconds=wall,
    )


def select_partition(
    selector: GeometryPartitionStump, features: DeploymentGeometry
) -> str:
    """Select only from deployment geometry; complete rigs are rejected."""

    if not isinstance(features, DeploymentGeometry):
        raise TypeError("partition inference requires DeploymentGeometry")
    if selector.exact_oracle_partition in selector.allowed_partitions:
        raise ValueError("exact all-in-one oracle cannot be selector-accessible")
    if selector.left_partition not in selector.allowed_partitions:
        raise ValueError("left partition is outside the allowed catalogue")
    if selector.right_partition not in selector.allowed_partitions:
        raise ValueError("right partition is outside the allowed catalogue")
    if not 0 <= int(selector.feature_index) < int(features.geometry_features.numel()):
        raise ValueError("selector feature index is invalid")
    value = float(features.geometry_features[int(selector.feature_index)])
    if not math.isfinite(value):
        raise ValueError("deployment geometry must be finite")
    return selector.left_partition if value <= selector.threshold else selector.right_partition


def select_global_fixed_partition(
    train_rigs: Sequence[TinyPrimitiveRig],
    score_table: Mapping[tuple[str, str], float],
    candidate_names: Sequence[str],
) -> tuple[str, dict[str, float]]:
    if not train_rigs or not candidate_names:
        raise ValueError("global selection requires train rigs and candidates")
    scores: dict[str, float] = {}
    for name in candidate_names:
        values = [float(score_table[(rig.rig_id, name)]) for rig in train_rigs]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("selection scores must be finite")
        scores[name] = sum(values) / len(values)
    selected = min(sorted(scores), key=lambda name: scores[name])
    return selected, scores


def fit_geometry_partition_stump(
    train_rigs: Sequence[TinyPrimitiveRig],
    safety_rigs: Sequence[TinyPrimitiveRig],
    score_table: Mapping[tuple[str, str], float],
    *,
    candidate_names: Sequence[str],
    exact_oracle_partition: str,
    train_top_k: int,
) -> tuple[GeometryPartitionStump, dict[str, Any]]:
    """Fit on train scores and choose among train finalists on safety rigs."""

    if not train_rigs or not safety_rigs:
        raise ValueError("selector requires nonempty train and safety splits")
    allowed = tuple(candidate_names)
    if len(allowed) != len(set(allowed)) or exact_oracle_partition in allowed:
        raise ValueError("selector candidates must be unique and exclude exact oracle")
    feature_count = int(train_rigs[0].geometry_features.numel())
    if any(int(rig.geometry_features.numel()) != feature_count for rig in (*train_rigs, *safety_rigs)):
        raise ValueError("geometry feature dimensions differ")

    candidates: list[tuple[float, int, float, str, str]] = []
    for feature_index in range(feature_count):
        values = sorted({float(rig.geometry_features[feature_index]) for rig in train_rigs})
        thresholds = [0.5 * (left + right) for left, right in zip(values, values[1:])]
        if not thresholds:
            thresholds = [values[0]]
        for threshold in thresholds:
            for left in allowed:
                for right in allowed:
                    score = sum(
                        float(
                            score_table[
                                (
                                    rig.rig_id,
                                    left
                                    if float(rig.geometry_features[feature_index]) <= threshold
                                    else right,
                                )
                            ]
                        )
                        for rig in train_rigs
                    ) / len(train_rigs)
                    candidates.append((score, feature_index, threshold, left, right))
    candidates.sort(key=lambda item: (item[0], item[3] == item[4], item[1], item[2], item[3], item[4]))
    top_k = min(max(1, int(train_top_k)), len(candidates))
    finalists = candidates[:top_k]

    def safety_score(candidate: tuple[float, int, float, str, str]) -> float:
        _, feature_index, threshold, left, right = candidate
        return sum(
            float(
                score_table[
                    (
                        rig.rig_id,
                        left if float(rig.geometry_features[feature_index]) <= threshold else right,
                    )
                ]
            )
            for rig in safety_rigs
        ) / len(safety_rigs)

    selected = min(
        finalists,
        key=lambda item: (
            safety_score(item),
            item[0],
            item[3] == item[4],
            item[1],
            item[2],
            item[3],
            item[4],
        ),
    )
    train_score, feature_index, threshold, left, right = selected
    selector = GeometryPartitionStump(
        feature_index=feature_index,
        threshold=threshold,
        left_partition=left,
        right_partition=right,
        allowed_partitions=allowed,
        exact_oracle_partition=exact_oracle_partition,
    )
    safety_selected_score = safety_score(selected)
    return selector, {
        "model_class": "DEPTH_ONE_GEOMETRY_STUMP",
        "feature_index": feature_index,
        "threshold": threshold,
        "left_partition": left,
        "right_partition": right,
        "is_constant": left == right,
        "train_mean_final_field_relative_l2": train_score,
        "safety_mean_final_field_relative_l2": safety_selected_score,
        "train_candidate_count": len(candidates),
        "safety_finalist_count": top_k,
        "parameter_count_proxy": 3,
        "exact_oracle_available": False,
    }
