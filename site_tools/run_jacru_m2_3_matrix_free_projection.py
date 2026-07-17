#!/usr/bin/env python3
"""Run the opened M2.3 fixed-budget matrix-free projection diagnostic.

The deployable candidate uses only a learned proposal, its prepared CGLS-12
base, and calls to a declared forward/adjoint pair.  A dense toy SVD is used
only after the fact to measure approximation headroom on the 12^3 fixture; it
is excluded from the algorithm, call budget, and every efficiency claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo_t16_operator.jacru_m2_exact_nullspace_oracle import (
    ExactDenseNullspaceProjector,
    build_exact_dense_nullspace_projector,
)
from demo_t16_operator.jacru_m2_matrix_free_projection import (
    matrix_free_measurement_projection_path,
)
from site_tools import run_jacru_m2_1_data_consistency_diagnostic as m21
from site_tools import run_jacru_m2_2_exact_nullspace_oracle as m22
from site_tools import run_jacru_m2_learned_residual_gate as m2


DEFAULT_CONFIG = (
    ROOT
    / "demo_t16_operator/configs/"
    "jacru_m2_3_matrix_free_projection_postopen_v1.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "demo_t16_operator/results/"
    "jacru_m2_3_matrix_free_projection_postopen_public"
)
SCHEMA = "jacru-m2-3-matrix-free-projection-postopen-report-1.0"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed-limit", type=int)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("config must contain one JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _validate_source_hashes(config: dict[str, Any]) -> dict[str, Path]:
    sources = {
        "source_t0_config": ROOT / config["source_t0_config"],
        "source_t0_summary": ROOT / config["source_t0_results"] / "summary.json",
        "source_m2_2_config": ROOT / config["source_m2_2_config"],
        "source_m2_2_summary": ROOT / config["source_m2_2_results"] / "summary.json",
    }
    expected = {
        "source_t0_config": config["source_t0_config_sha256"],
        "source_t0_summary": config["source_t0_summary_sha256"],
        "source_m2_2_config": config["source_m2_2_config_sha256"],
        "source_m2_2_summary": config["source_m2_2_summary_sha256"],
    }
    if "source_m2_3_config" in config:
        sources["source_m2_3_config"] = ROOT / config["source_m2_3_config"]
        sources["source_m2_3_summary"] = (
            ROOT / config["source_m2_3_results"] / "summary.json"
        )
        expected["source_m2_3_config"] = config["source_m2_3_config_sha256"]
        expected["source_m2_3_summary"] = config["source_m2_3_summary_sha256"]
    if "source_m2_4_config" in config:
        sources["source_m2_4_config"] = ROOT / config["source_m2_4_config"]
        sources["source_m2_4_summary"] = (
            ROOT / config["source_m2_4_results"] / "summary.json"
        )
        expected["source_m2_4_config"] = config["source_m2_4_config_sha256"]
        expected["source_m2_4_summary"] = config["source_m2_4_summary_sha256"]
    if "source_m2_5_config" in config:
        sources["source_m2_5_config"] = ROOT / config["source_m2_5_config"]
        sources["source_m2_5_summary"] = (
            ROOT / config["source_m2_5_results"] / "summary.json"
        )
        expected["source_m2_5_config"] = config["source_m2_5_config_sha256"]
        expected["source_m2_5_summary"] = config["source_m2_5_summary_sha256"]
    if "source_m2_6_config" in config:
        sources["source_m2_6_config"] = ROOT / config["source_m2_6_config"]
        sources["source_m2_6_summary"] = (
            ROOT / config["source_m2_6_results"] / "summary.json"
        )
        expected["source_m2_6_config"] = config["source_m2_6_config_sha256"]
        expected["source_m2_6_summary"] = config["source_m2_6_summary_sha256"]
    for name, path in sources.items():
        observed = _sha256(path)
        if observed != expected[name]:
            raise RuntimeError(
                f"{name} hash drifted: expected {expected[name]}, observed {observed}"
            )
    m2_2_summary = _read_json(sources["source_m2_2_summary"])
    if not bool(
        m2_2_summary.get("authorization", {}).get(
            "continue_matrix_free_projection_research", False
        )
    ):
        raise RuntimeError("M2.2 did not authorize matrix-free mechanism research")
    return sources


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise ValueError("cannot average an empty collection")
    return float(math.fsum(materialized) / len(materialized))


def _safe_reduction_retention(
    *,
    base_error: float,
    candidate_error: float,
    oracle_error: float,
) -> tuple[float, bool]:
    denominator = float(base_error) - float(oracle_error)
    if denominator <= 1e-12:
        return 0.0, False
    return (float(base_error) - float(candidate_error)) / denominator, True


def _projection_closure_relative_error(
    *,
    visible: torch.Tensor,
    system_residual: torch.Tensor,
    dual: torch.Tensor,
    damping: float,
    initial_system_norm: float,
) -> float:
    """Audit ``visible = residual + damping * dual`` at one iterate."""

    if visible.shape != system_residual.shape or visible.shape != dual.shape:
        raise ValueError("projection closure tensors must share one shape")
    closure = visible - (system_residual + float(damping) * dual)
    return float(torch.linalg.vector_norm(closure)) / max(
        float(initial_system_norm), 1e-30
    )


def _convex_quadratic_feasible_interval(
    *,
    quadratic: float,
    linear: float,
    constant: float,
    lower: float = 0.0,
    upper: float = 1.0,
    tolerance: float = 1e-14,
) -> tuple[float, float] | None:
    """Return where ``a*x^2+b*x+c <= 0`` inside one closed interval."""

    values = (quadratic, linear, constant, lower, upper, tolerance)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("quadratic interval inputs must be finite")
    a = float(quadratic)
    b = float(linear)
    c = float(constant)
    lo = float(lower)
    hi = float(upper)
    tol = float(tolerance)
    if lo > hi or tol < 0.0 or a < -tol:
        raise ValueError("expected a convex quadratic on a valid interval")
    if abs(a) <= tol:
        if abs(b) <= tol:
            return (lo, hi) if c <= tol else None
        boundary = -c / b
        candidate = (lo, min(hi, boundary)) if b > 0.0 else (max(lo, boundary), hi)
        return candidate if candidate[0] <= candidate[1] + tol else None
    discriminant = b * b - 4.0 * a * c
    scale = max(b * b, abs(4.0 * a * c), 1.0)
    if discriminant < -tol * scale:
        return None
    root = math.sqrt(max(discriminant, 0.0))
    left = (-b - root) / (2.0 * a)
    right = (-b + root) / (2.0 * a)
    candidate = (max(lo, left), min(hi, right))
    return candidate if candidate[0] <= candidate[1] + tol else None


def _convex_quadratic_minimizer(
    *,
    quadratic: float,
    linear: float,
    interval: tuple[float, float],
    tolerance: float = 1e-14,
) -> float:
    """Minimize ``a*x^2+b*x+c`` over a previously validated interval."""

    a = float(quadratic)
    b = float(linear)
    lo, hi = (float(value) for value in interval)
    if not all(math.isfinite(value) for value in (a, b, lo, hi, tolerance)):
        raise ValueError("quadratic minimizer inputs must be finite")
    if a < -float(tolerance) or lo > hi:
        raise ValueError("expected a convex quadratic on a valid interval")
    if a <= float(tolerance):
        return lo if b >= 0.0 else hi
    return min(max(-b / (2.0 * a), lo), hi)


def _dense_camera_block_preconditioner(
    *,
    matrix: torch.Tensor,
    camera_index: torch.Tensor,
    measurement_shape: tuple[int, ...],
    damping: float,
) -> tuple[Any, dict[str, Any]]:
    """Build a dense toy camera-block SPD inverse for oracle diagnosis."""

    if matrix.ndim != 2 or matrix.dtype != torch.float64 or matrix.device.type != "cpu":
        raise ValueError("dense camera-block oracle requires one CPU float64 matrix")
    if len(measurement_shape) < 2:
        raise ValueError("measurement shape must expose ray and component axes")
    ray_count = int(measurement_shape[0])
    component_count = int(np.prod(measurement_shape[1:]))
    cameras = camera_index.detach().cpu().to(torch.int64).reshape(-1)
    if cameras.numel() != ray_count:
        raise ValueError("camera_index must contain one value per measurement ray")
    row_camera = cameras.repeat_interleave(component_count)
    if row_camera.numel() != matrix.shape[0]:
        raise ValueError("camera/component row layout does not match dense matrix")
    gram = matrix @ matrix.mT
    if damping:
        gram = gram + float(damping) * torch.eye(
            gram.shape[0], dtype=gram.dtype
        )
    indices: list[torch.Tensor] = []
    factors: list[torch.Tensor] = []
    conditions: list[float] = []
    minimum_eigenvalues: list[float] = []
    for camera in torch.unique(row_camera, sorted=True):
        selected = torch.nonzero(row_camera == camera, as_tuple=False).reshape(-1)
        block = gram.index_select(0, selected).index_select(1, selected)
        eigenvalues = torch.linalg.eigvalsh(block)
        minimum = float(eigenvalues[0])
        maximum = float(eigenvalues[-1])
        if not math.isfinite(minimum) or minimum <= 0.0:
            raise RuntimeError("camera-block oracle is not strictly positive definite")
        factors.append(torch.linalg.cholesky(block))
        indices.append(selected)
        minimum_eigenvalues.append(minimum)
        conditions.append(maximum / minimum)

    def apply(value: torch.Tensor) -> torch.Tensor:
        flat = value.reshape(-1)
        if flat.dtype != torch.float64 or flat.device.type != "cpu":
            raise ValueError("camera-block oracle expects CPU float64 measurements")
        output = torch.zeros_like(flat)
        for selected, factor in zip(indices, factors):
            solved = torch.cholesky_solve(
                flat.index_select(0, selected)[:, None],
                factor,
            )[:, 0]
            output.index_copy_(0, selected, solved)
        return output.reshape(measurement_shape)

    return apply, {
        "block_count": len(indices),
        "largest_block_size": max(int(value.numel()) for value in indices),
        "minimum_block_eigenvalue": min(minimum_eigenvalues),
        "maximum_block_condition_number": max(conditions),
    }


def _score_reference_rows(
    *,
    record: m2.PreparedRecord,
    method: str,
    model_seed: int,
    field: torch.Tensor,
    optimization_forward_calls: int,
    optimization_adjoint_calls: int,
    grouped_adjoint_calls: int,
    reference_kind: str,
) -> dict[str, Any]:
    row = m2._score_prediction(
        record=record,
        method=method,
        model_seed=model_seed,
        prediction=field,
        gate=None,
        correction_rms=None,
        optimization_forward_calls=optimization_forward_calls,
        optimization_adjoint_calls=optimization_adjoint_calls,
        grouped_adjoint_calls=grouped_adjoint_calls,
        neural_inference_seconds=0.0,
    )
    row["reference_kind"] = reference_kind
    return row


def _build_norm_and_oracle_caches(
    *,
    records: list[m2.PreparedRecord],
    source_config: dict[str, Any],
    config: dict[str, Any],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, ExactDenseNullspaceProjector],
    dict[str, dict[str, Any]],
]:
    norm_cache: dict[str, dict[str, Any]] = {}
    projectors: dict[str, ExactDenseNullspaceProjector] = {}
    oracle_setup: dict[str, dict[str, Any]] = {}
    budget = source_config["physical_budget"]
    oracle_spec = config["retrospective_toy_oracle_audit"]
    for record in records:
        if record.split == "train":
            continue
        operator = record.case.inference.operator
        digest = record.case.inference.geometry.digest
        if digest not in norm_cache:
            operator.reset_call_counts()
            norm_cache[digest] = m2._dense_norm_squared_bound(
                operator,
                batch_size=int(budget["dense_norm_batch_size"]),
                safety_factor=float(budget["dense_norm_safety_factor"]),
            )
        if digest not in projectors:
            matrix, setup = m22._assemble_active_matrix_batched(
                operator,
                support=operator.support,
                batch_size=int(oracle_spec["assembly_batch_size"]),
            )
            started = time.perf_counter()
            projector = build_exact_dense_nullspace_projector(
                support=operator.support,
                dense_matrix=matrix,
                rank_rtol=float(oracle_spec["rank_relative_tolerance"]),
                rank_atol=float(oracle_spec["rank_absolute_tolerance"]),
            )
            setup.update(
                {
                    "rank": projector.rank,
                    "nullity_lower_bound": int(
                        projector.dense_active_matrix.shape[1] - projector.rank
                    ),
                    "largest_singular_value": float(projector.singular_values[0]),
                    "smallest_retained_singular_value": float(
                        projector.singular_values[projector.rank - 1]
                    )
                    if projector.rank
                    else None,
                    "rank_tolerance": projector.rank_tolerance,
                    "factorization_seconds": time.perf_counter() - started,
                    "used_by_algorithm": False,
                    "status": "RETROSPECTIVE_DENSE_TOY_ORACLE_NOT_ALGORITHM",
                }
            )
            projectors[digest] = projector
            oracle_setup[digest] = setup
    return norm_cache, projectors, oracle_setup


def _matched_baselines(
    *,
    records: list[m2.PreparedRecord],
    source_config: dict[str, Any],
    config: dict[str, Any],
    norm_cache: dict[str, dict[str, Any]],
    iterations: list[int],
) -> list[dict[str, Any]]:
    # M2.1's helper uses total=12+1+step.  Passing step=K+1 therefore
    # produces the required 14+K pair budget for this algorithm.
    helper_config = {
        "step_safety_factor": 0.98,
    }
    rows = m21._matched_baseline_rows(
        records=records,
        source_config=source_config,
        diagnostic_config=helper_config,
        norm_cache=norm_cache,
        steps=[value + 1 for value in iterations],
    )
    for row in rows:
        row["projection_iterations"] = int(row["matched_step"]) - 1
        row["paired_call_budget"] = int(row["total_calls"])
        row["matched_step_internal_offset"] = int(row["matched_step"])
    expected = {
        14 + iteration
        for iteration in iterations
    }
    if {int(row["paired_call_budget"]) for row in rows} != expected:
        raise RuntimeError("matched baseline total-call formula drifted")
    return rows


def _baseline_lookup(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, int, str], dict[str, Any]]:
    lookup: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row["case_id"]),
            int(row["projection_iterations"]),
            str(row["method"]),
        )
        if key in lookup:
            raise RuntimeError(f"duplicate matched baseline row: {key}")
        lookup[key] = row
    return lookup


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["method"]),
            int(row["model_seed"]),
            str(row["split"]),
            str(row["projection_variant"]),
            int(row["projection_iterations"]),
        )
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (method, seed, split, variant, iteration), values in sorted(grouped.items()):
        output.append(
            {
                "method": method,
                "model_seed": seed,
                "split": split,
                "projection_variant": variant,
                "projection_iterations": iteration,
                "case_count": len(values),
                "paired_call_budget": int(values[0]["paired_call_budget"]),
                "damping_fraction": float(values[0]["damping_fraction"]),
                "field_relative_l2_mean": _mean(
                    float(row["field_relative_l2"]) for row in values
                ),
                "h1_seminorm_relative_error_mean": _mean(
                    float(row["h1_seminorm_relative_error"]) for row in values
                ),
                "field_gain_to_best_matched_classical_mean": _mean(
                    float(row["field_gain_to_best_matched_classical"])
                    for row in values
                ),
                "h1_gain_to_best_matched_classical_mean": _mean(
                    float(row["h1_gain_to_best_matched_classical"])
                    for row in values
                ),
                "reprojection_ratio_to_matched_cgls_mean": _mean(
                    float(row["reprojection_ratio_to_matched_cgls"])
                    for row in values
                ),
                "visible_correction_fraction_mean": _mean(
                    float(row["visible_correction_fraction"]) for row in values
                ),
                "visible_correction_fraction_maximum": max(
                    float(row["visible_correction_fraction"]) for row in values
                ),
                "system_residual_fraction_mean": _mean(
                    float(row["system_residual_fraction"]) for row in values
                ),
                "projection_closure_relative_error_mean": _mean(
                    float(row["projection_closure_relative_error"])
                    for row in values
                ),
                "projection_closure_relative_error_maximum": max(
                    float(row["projection_closure_relative_error"])
                    for row in values
                ),
                "exact_projection_approximation_error_mean": _mean(
                    float(row["exact_projection_approximation_error"])
                    for row in values
                ),
                "exact_oracle_error_reduction_retention_mean": _mean(
                    float(row["exact_oracle_error_reduction_retention"])
                    for row in values
                ),
                "oracle_reduction_defined_rate": _mean(
                    float(row["oracle_reduction_defined"]) for row in values
                ),
                "field_harm_rate": _mean(
                    float(row["field_harm_to_best_matched_classical"])
                    for row in values
                ),
                "worst_field_gain_to_best_matched_classical": min(
                    float(row["field_gain_to_best_matched_classical"])
                    for row in values
                ),
                "breakdown_rate": _mean(float(row["breakdown"]) for row in values),
            }
        )
    return output


def _candidate_metrics(
    rows: list[dict[str, Any]],
    *,
    method: str,
    variant: str,
    iteration: int,
    split: str,
) -> dict[str, Any]:
    values = [
        row
        for row in rows
        if row["method"] == method
        and row["projection_variant"] == variant
        and int(row["projection_iterations"]) == iteration
        and row["split"] == split
    ]
    if not values:
        raise RuntimeError(
            f"missing candidate rows for {method}/{variant}/{iteration}/{split}"
        )
    paired_budgets = {int(row["paired_call_budget"]) for row in values}
    if len(paired_budgets) != 1:
        raise RuntimeError("candidate rows disagree on paired call budget")
    seed_means = []
    for seed in sorted({int(row["model_seed"]) for row in values}):
        seed_means.append(
            _mean(
                float(row["field_gain_to_best_matched_classical"])
                for row in values
                if int(row["model_seed"]) == seed
            )
        )
    return {
        "case_model_count": len(values),
        "paired_call_budget": paired_budgets.pop(),
        "field_gain_mean": _mean(
            float(row["field_gain_to_best_matched_classical"]) for row in values
        ),
        "h1_gain_mean": _mean(
            float(row["h1_gain_to_best_matched_classical"]) for row in values
        ),
        "reprojection_ratio_mean": _mean(
            float(row["reprojection_ratio_to_matched_cgls"]) for row in values
        ),
        "visible_correction_fraction_mean": _mean(
            float(row["visible_correction_fraction"]) for row in values
        ),
        "visible_correction_fraction_maximum": max(
            float(row["visible_correction_fraction"]) for row in values
        ),
        "projection_closure_relative_error_mean": _mean(
            float(row["projection_closure_relative_error"]) for row in values
        ),
        "projection_closure_relative_error_maximum": max(
            float(row["projection_closure_relative_error"]) for row in values
        ),
        "exact_oracle_error_reduction_retention_mean": _mean(
            float(row["exact_oracle_error_reduction_retention"]) for row in values
        ),
        "oracle_reduction_defined_rate": _mean(
            float(row["oracle_reduction_defined"]) for row in values
        ),
        "field_harm_rate": _mean(
            float(row["field_harm_to_best_matched_classical"]) for row in values
        ),
        "worst_field_gain": min(
            float(row["field_gain_to_best_matched_classical"]) for row in values
        ),
        "breakdown_rate": _mean(float(row["breakdown"]) for row in values),
        "per_model_seed_field_gain_means": seed_means,
    }


def _select_and_decide(
    *,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    selection = config["development_selection_rule"]
    gates = config["decision_gates"]
    decisions: dict[str, Any] = {}
    variants = {
        str(value["name"]): float(
            value["damping_fraction_of_operator_norm_squared_bound"]
        )
        for value in config["projection"]["variants"]
    }
    iterations = [int(value) for value in config["projection"]["snapshot_iterations"]]
    for method in config["methods"]:
        screened: list[dict[str, Any]] = []
        for variant, damping_fraction in variants.items():
            for iteration in iterations:
                development = _candidate_metrics(
                    rows,
                    method=str(method),
                    variant=variant,
                    iteration=iteration,
                    split="development",
                )
                eligible = bool(
                    development["visible_correction_fraction_mean"]
                    <= float(selection["maximum_mean_visible_correction_fraction"])
                    and development["visible_correction_fraction_maximum"]
                    <= float(selection["maximum_worst_case_visible_correction_fraction"])
                    and development["reprojection_ratio_mean"]
                    <= float(
                        selection[
                            "maximum_mean_reprojection_ratio_to_matched_cgls"
                        ]
                    )
                    and development["breakdown_rate"]
                    <= float(gates["maximum_breakdown_rate"])
                    and development["projection_closure_relative_error_maximum"]
                    <= float(
                        selection.get(
                            "maximum_projection_closure_relative_error",
                            math.inf,
                        )
                    )
                    and development["paired_call_budget"]
                    <= int(
                        selection.get(
                            "maximum_paired_call_budget",
                            2**31 - 1,
                        )
                    )
                )
                screened.append(
                    {
                        "projection_variant": variant,
                        "projection_iterations": iteration,
                        "damping_fraction": damping_fraction,
                        "development": development,
                        "development_eligible": eligible,
                    }
                )
        eligible = [value for value in screened if value["development_eligible"]]
        eligible.sort(
            key=lambda value: (
                -float(value["development"]["field_gain_mean"]),
                int(value["projection_iterations"]),
                float(value["damping_fraction"]),
                str(value["projection_variant"]),
            )
        )
        if not eligible:
            decisions[str(method)] = {
                "screened_candidates": screened,
                "selection": None,
                "checks": {"development_selection_exists": False},
                "passed_m2_3_mechanism_gate": False,
            }
            continue
        chosen = eligible[0]
        variant = str(chosen["projection_variant"])
        iteration = int(chosen["projection_iterations"])
        diagnostics = {
            "development": chosen["development"],
            "ood": _candidate_metrics(
                rows,
                method=str(method),
                variant=variant,
                iteration=iteration,
                split="ood",
            ),
        }
        checks: dict[str, bool] = {"development_selection_exists": True}
        for split in ("development", "ood"):
            values = diagnostics[split]
            checks[f"{split}_field_gain"] = values["field_gain_mean"] >= float(
                gates[
                    f"{split}_field_gain_to_best_matched_classical_minimum_fraction"
                ]
            )
            checks[f"{split}_h1_gain"] = values["h1_gain_mean"] >= float(
                gates[f"{split}_h1_gain_to_best_matched_classical_minimum_fraction"]
            )
            checks[f"{split}_oracle_retention"] = values[
                "exact_oracle_error_reduction_retention_mean"
            ] >= float(
                gates[
                    f"{split}_exact_oracle_error_reduction_retention_minimum_fraction"
                ]
            ) and values["oracle_reduction_defined_rate"] == 1.0
            checks[f"{split}_reprojection"] = values[
                "reprojection_ratio_mean"
            ] <= float(
                gates[f"{split}_reprojection_ratio_to_matched_cgls_maximum"]
            )
            checks[f"{split}_visible_correction"] = values[
                "visible_correction_fraction_mean"
            ] <= float(
                gates[f"{split}_visible_correction_fraction_mean_maximum"]
            )
            checks[f"{split}_harm_rate"] = values["field_harm_rate"] <= float(
                gates["field_harm_rate_maximum"]
            )
            checks[f"{split}_worst_case"] = values["worst_field_gain"] >= float(
                gates["worst_field_gain_minimum_fraction"]
            )
            checks[f"{split}_all_seed_means_positive"] = bool(
                not gates["require_all_model_seed_mean_field_gains_positive"]
                or all(
                    value > 0.0
                    for value in values["per_model_seed_field_gain_means"]
                )
            )
            checks[f"{split}_no_breakdown"] = values["breakdown_rate"] <= float(
                gates["maximum_breakdown_rate"]
            )
            checks[f"{split}_projection_closure"] = values[
                "projection_closure_relative_error_maximum"
            ] <= float(
                gates.get(
                    "maximum_projection_closure_relative_error",
                    math.inf,
                )
            )
            checks[f"{split}_paired_call_budget"] = values[
                "paired_call_budget"
            ] <= int(gates.get("maximum_paired_call_budget", 2**31 - 1))
        decisions[str(method)] = {
            "screened_candidates": screened,
            "selection": {
                "projection_variant": variant,
                "projection_iterations": iteration,
                "damping_fraction": float(chosen["damping_fraction"]),
                "used_ood_for_selection": False,
            },
            "diagnostics": diagnostics,
            "checks": checks,
            "passed_m2_3_mechanism_gate": all(checks.values()),
        }
    return decisions


def _plot(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    decisions: dict[str, Any],
    methods: list[str],
    variants: list[str],
    title: str = "M2.3 matrix-free measurement-space projection",
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.4), constrained_layout=True)
    colors = {
        "cg_undamped": "#146c94",
        "cg_damped_1e-6": "#d95f59",
        "cg_damped_1e-4": "#2a9d65",
    }
    palette = ("#146c94", "#d95f59", "#2a9d65", "#7b61a8", "#d18b24")
    for index, variant in enumerate(variants):
        colors.setdefault(variant, palette[index % len(palette)])
    method_styles = {"jacru_m2": "-", "pooled_cnn": "--"}
    method_markers = {"jacru_m2": "o", "pooled_cnn": "s"}
    labels = {"jacru_m2": "JACRU-M2", "pooled_cnn": "Pooled CNN"}
    metrics = (
        ("field_gain_to_best_matched_classical", "field gain vs best matched classic"),
        ("visible_correction_fraction", "visible learned-correction fraction"),
        ("reprojection_ratio_to_matched_cgls", "reprojection ratio to matched CGLS"),
    )
    for axis, (metric, ylabel) in zip(axes.reshape(-1)[:3], metrics):
        for method in methods:
            for variant in variants:
                values = [
                    row
                    for row in rows
                    if row["method"] == method
                    and row["projection_variant"] == variant
                    and row["split"] == "development"
                ]
                xs = sorted({int(row["projection_iterations"]) for row in values})
                ys = [
                    _mean(
                        float(row[metric])
                        for row in values
                        if int(row["projection_iterations"]) == iteration
                    )
                    for iteration in xs
                ]
                axis.plot(
                    xs,
                    ys,
                    color=colors.get(variant, "#555"),
                    linestyle=method_styles[method],
                    marker=method_markers[method],
                    linewidth=1.8,
                    markersize=4.5,
                    label=f"{labels[method]} · {variant}",
                )
        axis.set_xlabel("matrix-free CG iterations K")
        axis.set_ylabel(ylabel)
    axes[0, 0].axhline(0.0, color="#222", linewidth=1)
    axes[0, 0].axhline(0.05, color="#555", linewidth=1, linestyle=":")
    axes[0, 0].set_title("development field benefit under equal call budget")
    axes[0, 1].set_yscale("log")
    axes[0, 1].axhline(0.1, color="#222", linewidth=1, linestyle=":")
    axes[0, 1].set_title("finite CG approaches the operator kernel")
    axes[1, 0].set_yscale("log")
    axes[1, 0].axhline(1.0, color="#222", linewidth=1)
    axes[1, 0].axhline(1.1, color="#555", linewidth=1, linestyle=":")
    axes[1, 0].set_title("measured-data consistency")

    axis = axes[1, 1]
    selected_methods = [
        method for method in methods if decisions[method].get("selection") is not None
    ]
    x = np.arange(len(selected_methods), dtype=np.float64)
    width = 0.18
    for split_index, split in enumerate(("development", "ood")):
        field = [
            decisions[method]["diagnostics"][split]["field_gain_mean"]
            for method in selected_methods
        ]
        retention = [
            decisions[method]["diagnostics"][split][
                "exact_oracle_error_reduction_retention_mean"
            ]
            for method in selected_methods
        ]
        offset = (split_index - 0.5) * 2.2 * width
        axis.bar(
            x + offset - width / 2,
            field,
            width,
            color="#146c94" if split == "development" else "#6baed6",
            label=f"{split} field gain",
        )
        axis.bar(
            x + offset + width / 2,
            retention,
            width,
            color="#d95f59" if split == "development" else "#f4a6a1",
            label=f"{split} oracle retention",
        )
    axis.axhline(0.0, color="#222", linewidth=1)
    axis.axhline(0.5, color="#555", linewidth=1, linestyle=":")
    axis.set_xticks(x, [labels[value] for value in selected_methods])
    axis.set_ylabel("fraction")
    axis.set_title("development-selected candidate, OOD untouched by selection")
    axis.legend(frameon=False, fontsize=8)

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="outside lower center",
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(
        f"{title} · opened synthetic T0",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    config_path = args.config.resolve()
    config = _read_json(config_path)
    sources = _validate_source_hashes(config)
    source_config = _read_json(sources["source_t0_config"])
    methods = [str(value) for value in config["methods"]]
    iterations = [int(value) for value in config["projection"]["snapshot_iterations"]]
    variants = [dict(value) for value in config["projection"]["variants"]]
    affine_enabled = any(
        str(value.get("target_mode", "reference_reprojection"))
        == "affine_observation"
        for value in variants
    )
    preconditioner_experiment = any("preconditioner" in value for value in variants)
    if 0 not in iterations or len(iterations) != len(set(iterations)):
        raise ValueError("projection iterations must be unique and include zero")
    if any(value < 0 for value in iterations):
        raise ValueError("projection iterations must remain non-negative")
    if not set(methods).issubset(set(source_config["methods"])):
        raise ValueError("M2.3 methods must be source T0 methods")
    if args.seed_limit is not None:
        if args.seed_limit < 1:
            raise ValueError("seed-limit must be positive")
        source_config = json.loads(json.dumps(source_config))
        for split in source_config["splits"].values():
            split["base_seeds"] = split["base_seeds"][: args.seed_limit]
        source_config["training"]["model_seeds"] = source_config["training"][
            "model_seeds"
        ][: args.seed_limit]

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    fixture = m2._fixture_config(source_config)
    records = m2._prepare_records(source_config, fixture)
    device = m2._choose_device(args.device or source_config["training"]["device"])
    trained: list[dict[str, Any]] = []
    for method in methods:
        for seed in source_config["training"]["model_seeds"]:
            trained.append(
                m2._train_one(
                    method=method,
                    seed=int(seed),
                    config=source_config,
                    records=records,
                    device=device,
                    epoch_override=args.epochs,
                )
            )

    norm_cache, projectors, oracle_setup = _build_norm_and_oracle_caches(
        records=records,
        source_config=source_config,
        config=config,
    )
    baseline_rows = _matched_baselines(
        records=records,
        source_config=source_config,
        config=config,
        norm_cache=norm_cache,
        iterations=iterations,
    )
    baseline_lookup = _baseline_lookup(baseline_rows)

    base_iterations = int(config["reference_anchor"]["iterations"])
    feature_forward_calls = int(
        config["matched_budget"]["learned_feature_preparation_forward_calls"]
    )
    feature_adjoint_calls = int(
        config["matched_budget"]["learned_feature_preparation_adjoint_calls"]
    )
    maximum_iteration = max(iterations)
    rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    base_scores: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.split == "train":
            continue
        base = record.batch.base_field[0, 0].to(record.case.inference.operator.support)
        scored = _score_reference_rows(
            record=record,
            method="prepared_cgls_base_12",
            model_seed=-1,
            field=base,
            optimization_forward_calls=base_iterations,
            optimization_adjoint_calls=base_iterations,
            grouped_adjoint_calls=0,
            reference_kind="base_anchor",
        )
        base_scores[record.case.inference.case_id] = scored
        reference_rows.append(scored)

    for run in trained:
        model = run["model"]
        model_device = next(model.parameters()).device
        for record in records:
            if record.split == "train":
                continue
            kwargs = m2._to_device(record.batch.model_kwargs(), model_device)
            m2._synchronize(model_device)
            inference_started = time.perf_counter()
            with torch.no_grad():
                prediction, gate = model(**kwargs, return_gate=True)
            m2._synchronize(model_device)
            inference_seconds = time.perf_counter() - inference_started

            operator = record.case.inference.operator
            initial = prediction[0, 0].detach().cpu().to(operator.support)
            base = record.batch.base_field[0, 0].to(operator.support)
            correction = (initial - base) * operator.support
            correction_norm = float(torch.linalg.vector_norm(correction).clamp_min(1e-30))
            forward, adjoint = m2._operator_maps(operator)

            raw_score = _score_reference_rows(
                record=record,
                method=str(run["method"]),
                model_seed=int(run["model_seed"]),
                field=initial,
                optimization_forward_calls=feature_forward_calls,
                optimization_adjoint_calls=feature_adjoint_calls,
                grouped_adjoint_calls=1,
                reference_kind="raw_learned",
            )
            reference_rows.append(raw_score)

            projector = projectors[record.case.inference.geometry.digest]
            exact = projector.project(correction)
            exact_null = exact.null_space_correction.to(base)
            exact_field = (base + exact_null) * operator.support
            exact_score = _score_reference_rows(
                record=record,
                method=str(run["method"]),
                model_seed=int(run["model_seed"]),
                field=exact_field,
                optimization_forward_calls=feature_forward_calls,
                optimization_adjoint_calls=feature_adjoint_calls,
                grouped_adjoint_calls=1,
                reference_kind="retrospective_dense_oracle_base_anchor",
            )
            reference_rows.append(exact_score)
            base_score = base_scores[record.case.inference.case_id]
            base_error = float(base_score["field_relative_l2"])
            oracle_by_mode: dict[str, dict[str, Any]] = {
                "reference_reprojection": {
                    "field": exact_field,
                    "score": exact_score,
                    "rank": exact.rank,
                    "active_voxel_count": exact.active_voxel_count,
                    "approximation_denominator": correction_norm,
                    "internal_projection_residual": exact.internal_projection_residual,
                }
            }
            if affine_enabled:
                observation = record.case.inference.observations_uv[0].to(
                    operator.support
                )
                affine_exact = projector.project_field_to_observation(
                    field=initial * operator.support,
                    observation=observation,
                )
                affine_field = affine_exact.projected_field.to(base)
                affine_score = _score_reference_rows(
                    record=record,
                    method=str(run["method"]),
                    model_seed=int(run["model_seed"]),
                    field=affine_field,
                    optimization_forward_calls=feature_forward_calls,
                    optimization_adjoint_calls=feature_adjoint_calls,
                    grouped_adjoint_calls=1,
                    reference_kind="retrospective_dense_oracle_affine_observation",
                )
                reference_rows.append(affine_score)
                oracle_by_mode["affine_observation"] = {
                    "field": affine_field,
                    "score": affine_score,
                    "rank": affine_exact.rank,
                    "active_voxel_count": affine_exact.active_voxel_count,
                    "approximation_denominator": float(
                        torch.linalg.vector_norm(initial - affine_field).clamp_min(
                            1e-30
                        )
                    ),
                    "internal_projection_residual": (
                        affine_exact.internal_projection_residual
                    ),
                }

            for variant in variants:
                variant_name = str(variant["name"])
                target_mode = str(
                    variant.get("target_mode", "reference_reprojection")
                )
                if target_mode not in oracle_by_mode:
                    raise ValueError(f"unsupported projection target_mode: {target_mode}")
                oracle = oracle_by_mode[target_mode]
                target_observation = (
                    record.case.inference.observations_uv[0].to(operator.support)
                    if target_mode == "affine_observation"
                    else None
                )
                damping_fraction = float(
                    variant["damping_fraction_of_operator_norm_squared_bound"]
                )
                norm = norm_cache[record.case.inference.geometry.digest]
                damping = damping_fraction * float(norm["bound"])
                preconditioner_kind = str(variant.get("preconditioner", "identity"))
                preconditioner_diagonal = None
                preconditioner_apply = None
                supplied_preconditioner_name = None
                preconditioner_is_oracle = False
                preconditioner_setup_forward_equivalents = 0
                preconditioner_metadata = {
                    "block_count": 0,
                    "largest_block_size": 0,
                    "minimum_block_eigenvalue": 0.0,
                    "maximum_block_condition_number": 0.0,
                }
                if preconditioner_kind == "dense_exact_jacobi_oracle":
                    matrix = projector.dense_active_matrix
                    diagonal_flat = torch.sum(matrix.square(), dim=1) + damping
                    measurement_shape = tuple(
                        record.case.inference.observations_uv[0].shape
                    )
                    preconditioner_diagonal = diagonal_flat.reshape(
                        measurement_shape
                    ).to(operator.support)
                    preconditioner_is_oracle = True
                    preconditioner_setup_forward_equivalents = int(
                        matrix.shape[1] + 1
                    )
                elif preconditioner_kind == "dense_exact_camera_block_jacobi_oracle":
                    matrix = projector.dense_active_matrix
                    measurement_shape = tuple(
                        record.case.inference.observations_uv[0].shape
                    )
                    (
                        preconditioner_apply,
                        preconditioner_metadata,
                    ) = _dense_camera_block_preconditioner(
                        matrix=matrix,
                        camera_index=record.case.inference.geometry.camera_index,
                        measurement_shape=measurement_shape,
                        damping=damping,
                    )
                    supplied_preconditioner_name = preconditioner_kind
                    preconditioner_is_oracle = True
                    preconditioner_setup_forward_equivalents = int(
                        matrix.shape[1] + 1
                    )
                elif preconditioner_kind != "identity":
                    raise ValueError(
                        f"unsupported projection preconditioner: {preconditioner_kind}"
                    )
                operator.reset_call_counts()
                path = matrix_free_measurement_projection_path(
                    reference_field=base,
                    learned_field=initial,
                    forward=forward,
                    adjoint=adjoint,
                    support=operator.support,
                    snapshot_iterations=iterations,
                    damping=damping,
                    preconditioner_diagonal=preconditioner_diagonal,
                    preconditioner_apply=preconditioner_apply,
                    preconditioner_name=supplied_preconditioner_name,
                    target_observation=target_observation,
                    denominator_floor=float(
                        config["projection"]["denominator_floor"]
                    ),
                )
                if path.target_mode != target_mode:
                    raise RuntimeError("matrix-free target mode drifted")
                if preconditioner_kind == "identity":
                    expected_preconditioner_name = "identity"
                elif preconditioner_kind == "dense_exact_jacobi_oracle":
                    expected_preconditioner_name = "supplied_positive_diagonal"
                else:
                    expected_preconditioner_name = preconditioner_kind
                if path.preconditioner != expected_preconditioner_name:
                    raise RuntimeError("matrix-free preconditioner mode drifted")
                expected_algorithm_calls = {
                    "forward_calls": maximum_iteration + 1,
                    "adjoint_calls": maximum_iteration,
                }
                if operator.call_report() != expected_algorithm_calls:
                    raise RuntimeError(
                        "matrix-free runner call ledger drifted: "
                        f"{operator.call_report()} != {expected_algorithm_calls}"
                    )
                history = {int(value["iteration"]): value for value in path.history}
                initial_system_norm = float(
                    torch.linalg.vector_norm(path.system_residuals_by_iteration[0])
                )
                for iteration in iterations:
                    field = path.fields_by_iteration[iteration]
                    retained = path.retained_corrections_by_iteration[iteration]
                    operator.reset_call_counts()
                    if target_mode == "affine_observation":
                        assert target_observation is not None
                        visible = forward(field) - target_observation
                    else:
                        visible = forward(retained)
                    score = m2._score_prediction(
                        record=record,
                        method=str(run["method"]),
                        model_seed=int(run["model_seed"]),
                        prediction=field,
                        gate=float(gate[0, 0, 0, 0, 0].detach().cpu()),
                        correction_rms=float(torch.sqrt(torch.mean(retained.square()))),
                        optimization_forward_calls=feature_forward_calls + iteration + 1,
                        optimization_adjoint_calls=feature_adjoint_calls + iteration,
                        grouped_adjoint_calls=1,
                        neural_inference_seconds=inference_seconds,
                    )
                    if operator.call_report() != {
                        "forward_calls": 2,
                        "adjoint_calls": 0,
                    }:
                        raise RuntimeError("evaluation-only forward ledger drifted")
                    matched_cgls = baseline_lookup[
                        (score["case_id"], iteration, "cgls_matched")
                    ]
                    matched_huber = baseline_lookup[
                        (score["case_id"], iteration, "huber_pdhg_matched")
                    ]
                    matched_landweber = baseline_lookup[
                        (score["case_id"], iteration, "base_landweber_matched")
                    ]
                    best_field = min(
                        float(matched_cgls["field_relative_l2"]),
                        float(matched_huber["field_relative_l2"]),
                    )
                    best_h1 = min(
                        float(matched_cgls["h1_seminorm_relative_error"]),
                        float(matched_huber["h1_seminorm_relative_error"]),
                    )
                    candidate_error = float(score["field_relative_l2"])
                    retention, retention_defined = _safe_reduction_retention(
                        base_error=base_error,
                        candidate_error=candidate_error,
                        oracle_error=float(oracle["score"]["field_relative_l2"]),
                    )
                    visible_fraction = float(torch.linalg.vector_norm(visible)) / max(
                        initial_system_norm, 1e-30
                    )
                    closure_error = _projection_closure_relative_error(
                        visible=visible,
                        system_residual=path.system_residuals_by_iteration[iteration],
                        dual=path.duals_by_iteration[iteration],
                        damping=damping,
                        initial_system_norm=initial_system_norm,
                    )
                    approximation_error = float(
                        torch.linalg.vector_norm(field - oracle["field"])
                    ) / float(oracle["approximation_denominator"])
                    history_row = history[iteration]
                    paired_budget = feature_forward_calls + iteration + 1
                    if paired_budget != 14 + iteration:
                        raise RuntimeError("M2.3 paired-call formula drifted")
                    score.update(
                        {
                            "projection_variant": variant_name,
                            "projection_iterations": iteration,
                            "damping_fraction": damping_fraction,
                            "damping_absolute": damping,
                            "preconditioner": path.preconditioner,
                            "projection_forward_calls": iteration + 1,
                            "projection_adjoint_calls": iteration,
                            "paired_call_budget": paired_budget,
                            "matched_cgls_field_relative_l2": float(
                                matched_cgls["field_relative_l2"]
                            ),
                            "matched_huber_field_relative_l2": float(
                                matched_huber["field_relative_l2"]
                            ),
                            "matched_base_landweber_field_relative_l2": float(
                                matched_landweber["field_relative_l2"]
                            ),
                            "field_gain_to_best_matched_classical": (
                                best_field - candidate_error
                            )
                            / max(best_field, 1e-30),
                            "h1_gain_to_best_matched_classical": (
                                best_h1
                                - float(score["h1_seminorm_relative_error"])
                            )
                            / max(best_h1, 1e-30),
                            "reprojection_ratio_to_matched_cgls": float(
                                score["measured_reprojection_relative_l2"]
                            )
                            / max(
                                float(
                                    matched_cgls[
                                        "measured_reprojection_relative_l2"
                                    ]
                                ),
                                1e-30,
                            ),
                            "visible_correction_fraction": visible_fraction,
                            "system_residual_fraction": float(
                                torch.linalg.vector_norm(
                                    path.system_residuals_by_iteration[iteration]
                                )
                            )
                            / max(initial_system_norm, 1e-30),
                            "projection_closure_relative_error": closure_error,
                            "exact_projection_approximation_error": approximation_error,
                            "exact_oracle_error_reduction_retention": retention,
                            "oracle_reduction_defined": int(retention_defined),
                            "base_anchor_field_relative_l2": base_error,
                            "exact_oracle_field_relative_l2": float(
                                oracle["score"]["field_relative_l2"]
                            ),
                            "raw_learned_field_relative_l2": float(
                                raw_score["field_relative_l2"]
                            ),
                            "exact_oracle_rank": int(oracle["rank"]),
                            "exact_oracle_nullity_lower_bound": (
                                int(oracle["active_voxel_count"])
                                - int(oracle["rank"])
                            ),
                            "field_harm_to_best_matched_classical": int(
                                candidate_error
                                > best_field
                                * (
                                    1.0
                                    + float(
                                        config["decision_gates"][
                                            "field_harm_threshold_fraction"
                                        ]
                                    )
                                )
                            ),
                            "converged": int(bool(history_row["converged"])),
                            "breakdown": int(bool(history_row["breakdown"])),
                            "projection_diagnostic_forward_calls": 1,
                            "dense_oracle_used_by_algorithm": False,
                        }
                    )
                    if affine_enabled:
                        score.update(
                            {
                                "projection_target_mode": target_mode,
                                "exact_oracle_internal_projection_residual": float(
                                    oracle["internal_projection_residual"]
                                ),
                            }
                        )
                    if preconditioner_experiment:
                        score.update(
                            {
                                "preconditioner_kind": preconditioner_kind,
                                "preconditioner_is_oracle": int(
                                    preconditioner_is_oracle
                                ),
                                "preconditioner_setup_forward_equivalents": (
                                    preconditioner_setup_forward_equivalents
                                ),
                                "preconditioner_setup_adjoint_equivalents": 0,
                                "preconditioner_applications": (
                                    path.preconditioner_applications
                                ),
                                "preconditioner_block_count": int(
                                    preconditioner_metadata["block_count"]
                                ),
                                "preconditioner_largest_block_size": int(
                                    preconditioner_metadata["largest_block_size"]
                                ),
                                "preconditioner_minimum_block_eigenvalue": float(
                                    preconditioner_metadata[
                                        "minimum_block_eigenvalue"
                                    ]
                                ),
                                "preconditioner_maximum_block_condition_number": float(
                                    preconditioner_metadata[
                                        "maximum_block_condition_number"
                                    ]
                                ),
                            }
                        )
                    rows.append(score)

    aggregate = _aggregate(rows)
    baseline_aggregate = m21._aggregate_baselines(baseline_rows)
    decisions = _select_and_decide(rows=rows, config=config)
    any_pass = any(
        value["passed_m2_3_mechanism_gate"] for value in decisions.values()
    )
    report_status = config.get("report_status", {})
    preconditioner_oracle_only = bool(
        config.get("preconditioner_oracle_only", False)
    )
    summary = {
        "schema_version": str(config.get("report_schema_version", SCHEMA)),
        "status": str(
            report_status.get(
                "success"
                if any_pass
                else "no_go",
                "M2_3_POSTOPEN_MATRIX_FREE_MECHANISM_FOUND_NOT_CONFIRMATORY"
                if any_pass
                else "M2_3_POSTOPEN_MATRIX_FREE_PROJECTION_NO_GO",
            )
        ),
        "evidence_level": config["evidence_level"],
        "source_config_sha256": _sha256(config_path),
        "source_t0_config_sha256": _sha256(sources["source_t0_config"]),
        "source_t0_summary_sha256": _sha256(sources["source_t0_summary"]),
        "source_m2_2_config_sha256": _sha256(sources["source_m2_2_config"]),
        "source_m2_2_summary_sha256": _sha256(sources["source_m2_2_summary"]),
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
        "metric_row_count": len(rows),
        "reference_row_count": len(reference_rows),
        "matched_baseline_row_count": len(baseline_rows),
        "training_runs": [
            {
                key: value
                for key, value in run.items()
                if key not in {"model", "history"}
            }
            for run in trained
        ],
        "operator_norm_setup": norm_cache,
        "retrospective_dense_oracle_setup": oracle_setup,
        "aggregate": aggregate,
        "matched_baseline_aggregate": baseline_aggregate,
        "decisions": decisions,
        "authorization": {
            "claim_deployable_algorithm": False,
            "claim_method_superiority": False,
            "claim_real_bost_generalization": False,
            "open_fresh_or_final": False,
            "draft_new_preregistered_fresh_gate": (
                any_pass and not preconditioner_oracle_only
            ),
            "continue_matrix_free_preconditioner_research": True,
        },
        "claim_boundary": config["claim_boundary"],
        "public_export_policy": {
            "contains_model_checkpoints": False,
            "contains_restricted_papers": False,
            "contains_private_experimental_arrays": False,
        },
    }
    if preconditioner_experiment:
        summary["authorization"][
            "continue_deployable_preconditioner_estimation"
        ] = bool(any_pass and preconditioner_oracle_only)
    if "source_m2_3_config" in sources:
        summary["source_m2_3_config_sha256"] = _sha256(
            sources["source_m2_3_config"]
        )
        summary["source_m2_3_summary_sha256"] = _sha256(
            sources["source_m2_3_summary"]
        )
    if "source_m2_4_config" in sources:
        summary["source_m2_4_config_sha256"] = _sha256(
            sources["source_m2_4_config"]
        )
        summary["source_m2_4_summary_sha256"] = _sha256(
            sources["source_m2_4_summary"]
        )
    if "source_m2_5_config" in sources:
        summary["source_m2_5_config_sha256"] = _sha256(
            sources["source_m2_5_config"]
        )
        summary["source_m2_5_summary_sha256"] = _sha256(
            sources["source_m2_5_summary"]
        )
    if "source_m2_6_config" in sources:
        summary["source_m2_6_config_sha256"] = _sha256(
            sources["source_m2_6_config"]
        )
        summary["source_m2_6_summary_sha256"] = _sha256(
            sources["source_m2_6_summary"]
        )

    _write_csv(output / "metric_rows.csv", rows)
    _write_csv(output / "aggregate_rows.csv", aggregate)
    _write_csv(output / "reference_rows.csv", reference_rows)
    _write_csv(output / "matched_baseline_rows.csv", baseline_rows)
    _write_csv(output / "matched_baseline_aggregate_rows.csv", baseline_aggregate)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _plot(
        output / "diagnostic",
        rows=rows,
        decisions=decisions,
        methods=methods,
        variants=[str(value["name"]) for value in variants],
        title=str(
            config.get(
                "figure_title",
                "M2.3 matrix-free measurement-space projection",
            )
        ),
    )
    readme = f"""# {config.get('readme_title', 'JACRU-M2.3 matrix-free projection diagnostic')}

Status: `{summary['status']}`

This packet reuses the opened M2-T0 synthetic splits. The candidate runs
fixed-step CG only through the supplied forward/adjoint pair; it does not read
truth. Development selects one fixed variant/iteration per learned backbone,
and exploratory OOD is not used for that selection. Dense SVD is retrospective
toy-oracle evaluation only and is excluded from the algorithm and efficiency
claims. Finite CG is not an exact optical null-space projector.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    artifacts = (
        "README.md",
        "aggregate_rows.csv",
        "diagnostic.pdf",
        "diagnostic.png",
        "matched_baseline_aggregate_rows.csv",
        "matched_baseline_rows.csv",
        "metric_rows.csv",
        "reference_rows.csv",
        "summary.json",
    )
    checksum_lines = [f"{_sha256(output / name)}  {name}" for name in artifacts]
    (output / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="ascii",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
