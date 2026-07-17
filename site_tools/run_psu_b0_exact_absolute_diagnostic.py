#!/usr/bin/env python3
"""Run the frozen post-Gate-B exact-|A| root-cause diagnostic.

The formal Gate-B decision remains closed.  This runner reuses only its opened
PSU geometry, analytic reaction phantoms, and synthetic noise to ask why the
factor PDHG metric failed: factor cancellation, per-view aggregation, or
headroom that remains after the strongest static diagonal metric.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import torch

from demo_t16_operator.psu_b0_classical_baselines import (
    preconditioned_cgls_trajectory,
)
from demo_t16_operator.psu_b0_exact_absolute_diagnostic import (
    EXACT_ABSOLUTE_DIAGNOSTIC_SCHEMA,
    build_exact_absolute_audit,
    describe_exact_absolute_metric,
    normalized_operator_power_estimate,
    run_exact_absolute_trajectory,
    schur_safety_certificate_squared,
    summarize_tightness,
)
from demo_t16_operator.psu_b0_factor_majorizer_pipeline import (
    FactorPipelineCallLedger,
)
from demo_t16_operator.psu_b0_gate_b_data_only import (
    build_single_sample_factor_setup,
    factor_state_to_volume,
    run_gate_b_data_only_trajectory,
)
from site_tools.run_psu_b0_factor_gate_b import (
    OPENED_REPLICATES,
    REACTION_FAMILIES,
    _build_runtime,
    _single_graph_operator,
)
from site_tools.run_psu_b0_spectral_preconditioner_pilot import _field_metrics


SCHEMA = "psu-b0-exact-absolute-root-cause-report-1.0"
CONFIG_SCHEMA = "psu-b0-exact-absolute-root-cause-config-1.0"
METHODS = (
    "scalar_a_only_pdhg",
    "formal_factor_view_a_only_pdhg",
    "factor_row_hybrid_a_only_pdhg",
    "exact_abs_view_a_only_pdhg",
    "exact_abs_row_a_only_pdhg",
    "graph_pcgls",
)


def _reject_constant(raw: str) -> None:
    raise ValueError(f"invalid JSON constant: {raw}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
        object_pairs_hook=_unique_object,
    )
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def load_config(path: Path) -> dict[str, Any]:
    config = _load_json(path)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unexpected exact-absolute config schema")
    if config.get("status") != "FROZEN_PREPERFORMANCE_ROOT_CAUSE_DIAGNOSTIC":
        raise ValueError("exact-absolute protocol is not frozen")
    if tuple(config.get("replicate_indices", ())) != tuple(OPENED_REPLICATES):
        raise ValueError("replicate set differs from the opened Gate-B set")
    if tuple(config.get("reaction_families", ())) != tuple(REACTION_FAMILIES):
        raise ValueError("reaction-family set differs from Gate B")
    if tuple(config.get("methods", ())) != METHODS:
        raise ValueError("diagnostic method set changed")
    checkpoints = tuple(int(value) for value in config.get("checkpoints", ()))
    if checkpoints != (4, 8, 16, 32, 64, 128):
        raise ValueError("diagnostic checkpoints changed")
    if float(config["solver"]["eta"]) != 0.7 or float(config["solver"]["theta"]) != 1.0:
        raise ValueError("Gate-B solver constants changed")
    return config


def validate_sources(root: Path, config: Mapping[str, Any]) -> dict[str, str]:
    paths = config.get("source_paths")
    expected = config.get("source_sha256")
    if not isinstance(paths, dict) or not isinstance(expected, dict):
        raise ValueError("source path/hash maps are required")
    if set(paths) != set(expected):
        raise ValueError("source path/hash keys differ")
    observed: dict[str, str] = {}
    for key, relative in paths.items():
        path = root / str(relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        observed[key] = _sha256(path)
        if observed[key] != str(expected[key]):
            raise ValueError(f"source hash mismatch: {key}")
    parent = _load_json(root / str(paths["parent_gate_b_report"]))
    if parent.get("status") != "GATE_B_E2_MECHANISM_NO_GO":
        raise ValueError("formal Gate-B parent is not the closed NO-GO release")
    return observed


def validate_clean_repository(root: Path) -> str:
    if _git(root, "status", "--porcelain"):
        raise ValueError("formal diagnostic requires a clean repository")
    return _git(root, "rev-parse", "HEAD")


def _relative_l2(prediction: torch.Tensor, truth: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm((prediction - truth).reshape(-1))
    denominator = torch.linalg.vector_norm(truth.reshape(-1)).clamp_min(1e-12)
    return float((numerator / denominator).detach().cpu())


def _support_components(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    setup: Any,
) -> dict[str, float]:
    predicted_active = setup.pipeline.restrict_active(prediction)[0]
    truth_active = setup.pipeline.restrict_active(truth)[0]
    data_indices = setup.active_primal_indices
    data_mask = torch.zeros(
        setup.pipeline.n_active,
        dtype=torch.bool,
        device=prediction.device,
    )
    data_mask[data_indices] = True
    null_indices = torch.nonzero(~data_mask, as_tuple=False).flatten()
    data_prediction = predicted_active.index_select(0, data_indices)
    data_truth = truth_active.index_select(0, data_indices)
    null_prediction = predicted_active.index_select(0, null_indices)
    null_truth = truth_active.index_select(0, null_indices)
    return {
        "data_coupled_relative_l2": _relative_l2(data_prediction, data_truth),
        "data_null_support_relative_l2": _relative_l2(null_prediction, null_truth),
        "data_coupled_error_energy": float(
            torch.sum((data_prediction - data_truth).square()).detach().cpu()
        ),
        "data_null_support_error_energy": float(
            torch.sum((null_prediction - null_truth).square()).detach().cpu()
        ),
        "data_null_support_reconstruction_energy": float(
            torch.sum(null_prediction.square()).detach().cpu()
        ),
    }


def _metric_row(
    *,
    replicate: int,
    sample_index: int,
    family: str,
    method: str,
    iteration: int,
    prediction: torch.Tensor,
    truth: torch.Tensor,
    setup: Any,
    elapsed_seconds: float,
    normalized_data_residual_l2: float | None,
) -> dict[str, Any]:
    metrics = _field_metrics(prediction, truth)
    return {
        "replicate": int(replicate),
        "sample_index": int(sample_index),
        "reaction_family": str(family),
        "method": str(method),
        "iterations": int(iteration),
        "forward_calls": int(iteration),
        "adjoint_calls": int(iteration),
        "field_relative_l2": float(metrics["field_relative_l2"][0].detach().cpu()),
        "gradient_relative_l2": float(
            metrics["gradient_relative_l2"][0].detach().cpu()
        ),
        "front_top10_f1": float(metrics["front_top10_f1"][0].detach().cpu()),
        **_support_components(prediction, truth, setup),
        "normalized_data_residual_l2": normalized_data_residual_l2,
        "trajectory_elapsed_seconds": float(elapsed_seconds),
    }


def _run_pdhg_method(
    *,
    setup: Any,
    target: torch.Tensor,
    truth: torch.Tensor,
    output_scale: torch.Tensor,
    audit: Any,
    method: str,
    checkpoints: Sequence[int],
    theta: float,
    replicate: int,
    sample_index: int,
    family: str,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    setup.pipeline.reset_call_ledger()
    _synchronize(device)
    started = time.perf_counter()
    if method == "scalar_a_only_pdhg":
        states = run_gate_b_data_only_trajectory(
            setup,
            target,
            checkpoints=checkpoints,
            mode="scalar",
            theta=theta,
        )
    elif method == "formal_factor_view_a_only_pdhg":
        states = run_gate_b_data_only_trajectory(
            setup,
            target,
            checkpoints=checkpoints,
            mode="voxel_factor",
            theta=theta,
        )
    else:
        diagnostic_mode = {
            "factor_row_hybrid_a_only_pdhg": "factor_row",
            "exact_abs_view_a_only_pdhg": "exact_view",
            "exact_abs_row_a_only_pdhg": "exact_row",
        }.get(method)
        if diagnostic_mode is None:
            raise ValueError("unsupported PDHG diagnostic method")
        metric = describe_exact_absolute_metric(setup, audit, diagnostic_mode)
        states = run_exact_absolute_trajectory(
            setup,
            target,
            metric,
            checkpoints=checkpoints,
            theta=theta,
        )
    _synchronize(device)
    elapsed = time.perf_counter() - started
    expected = FactorPipelineCallLedger(
        signed_data_forward_calls=max(checkpoints),
        signed_data_transpose_calls=max(checkpoints),
    )
    if setup.pipeline.call_ledger() != expected:
        raise ValueError(f"{method} violated exact-K call accounting")
    solver_ledger = setup.pipeline.call_ledger()
    setup.pipeline.reset_call_ledger()
    rows: list[dict[str, Any]] = []
    target_norm = torch.linalg.vector_norm(target.reshape(-1)).clamp_min(1e-12)
    for iteration in checkpoints:
        normalized = factor_state_to_volume(setup, states[iteration])
        active = setup.pipeline.restrict_active(normalized)
        projected = setup.pipeline.signed_data_forward(active).reshape_as(target)
        residual = float(
            (
                torch.linalg.vector_norm((projected - target).reshape(-1)) / target_norm
            )
            .detach()
            .cpu()
        )
        rows.append(_metric_row(
            replicate=replicate,
            sample_index=sample_index,
            family=family,
            method=method,
            iteration=iteration,
            prediction=normalized * output_scale,
            truth=truth,
            setup=setup,
            elapsed_seconds=elapsed,
            normalized_data_residual_l2=residual,
        ))
    scorer_expected = FactorPipelineCallLedger(
        signed_data_forward_calls=len(checkpoints),
    )
    if setup.pipeline.call_ledger() != scorer_expected:
        raise ValueError(f"{method} scorer call ledger is inconsistent")
    return rows, {
        "replicate": replicate,
        "sample_index": sample_index,
        "method": method,
        "solver_ledger": solver_ledger.__dict__,
        "scorer_ledger": scorer_expected.__dict__,
        "elapsed_seconds": elapsed,
    }


def _run_graph_method(
    *,
    operator: Any,
    context: Mapping[str, Any],
    direction: Any,
    setup: Any,
    checkpoints: Sequence[int],
    sample_index: int,
    family: str,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    graph_operator = _single_graph_operator(operator, context, sample_index)
    graph_operator.reset_call_counts()
    _synchronize(device)
    started = time.perf_counter()
    trajectory = preconditioned_cgls_trajectory(
        graph_operator,
        context["graph_prepared"][sample_index : sample_index + 1],
        sigma_by_view=context["ones_sigma"][sample_index : sample_index + 1],
        view_mask=context["ones_mask"][sample_index : sample_index + 1],
        rays_per_view=int(context["rays_per_view"]),
        checkpoint_stages=checkpoints,
        preconditioner=direction,
    )
    _synchronize(device)
    elapsed = time.perf_counter() - started
    expected = {"forward_calls": max(checkpoints), "adjoint_calls": max(checkpoints)}
    if graph_operator.call_report() != expected:
        raise ValueError("graph-PCGLS violated exact-K call accounting")
    truth = context["truth"][sample_index : sample_index + 1]
    rows = [
        _metric_row(
            replicate=int(context["replicate"]),
            sample_index=sample_index,
            family=family,
            method="graph_pcgls",
            iteration=iteration,
            prediction=trajectory[iteration].volume,
            truth=truth,
            setup=setup,
            elapsed_seconds=elapsed,
            normalized_data_residual_l2=None,
        )
        for iteration in checkpoints
    ]
    return rows, {
        "replicate": int(context["replicate"]),
        "sample_index": sample_index,
        "method": "graph_pcgls",
        "ledger": expected,
        "elapsed_seconds": elapsed,
    }


def _summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), int(row["iterations"]))].append(row)
    expected_count = len(OPENED_REPLICATES) * len(REACTION_FAMILIES)
    output: list[dict[str, Any]] = []
    for (method, iteration), values in sorted(grouped.items()):
        if len(values) != expected_count:
            raise ValueError(f"metric coverage incomplete for {method} K={iteration}")
        summary = {
                "method": method,
                "iterations": iteration,
                "sample_count": len(values),
                "mean_field_relative_l2": statistics.fmean(
                    float(row["field_relative_l2"]) for row in values
                ),
                "p90_field_relative_l2": float(
                    np.quantile(
                        [float(row["field_relative_l2"]) for row in values],
                        0.90,
                    )
                ),
                "mean_gradient_relative_l2": statistics.fmean(
                    float(row["gradient_relative_l2"]) for row in values
                ),
                "mean_front_top10_f1": statistics.fmean(
                    float(row["front_top10_f1"]) for row in values
                ),
        }
        residuals = [row.get("normalized_data_residual_l2") for row in values]
        summary["mean_normalized_data_residual_l2"] = (
            None
            if any(value is None for value in residuals)
            else statistics.fmean(float(value) for value in residuals)
        )
        output.append(summary)
    return output


def classify_root_cause(
    rows: Sequence[Mapping[str, Any]],
    tightness_rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply descriptive same-operator labels; graph remains nonbinding."""

    lookup: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    paired_data: dict[tuple[int, int, str, int], float] = {}
    expected_count = len(OPENED_REPLICATES) * len(REACTION_FAMILIES)
    for row in rows:
        method = str(row["method"])
        iteration = int(row["iterations"])
        lookup[(method, iteration)].append(row)
        residual = row.get("normalized_data_residual_l2")
        if method != "graph_pcgls":
            if residual is None:
                raise ValueError("PDHG diagnostic row is missing normalized residual")
            paired_data[
                (
                    int(row["replicate"]),
                    int(row["sample_index"]),
                    method,
                    iteration,
                )
            ] = float(residual)

    def mean_metric(method: str, iteration: int, key: str) -> float:
        values = lookup[(method, iteration)]
        if len(values) != expected_count:
            raise ValueError("classification metric coverage is incomplete")
        numeric = [row.get(key) for row in values]
        if any(value is None for value in numeric):
            raise ValueError(f"classification metric {key} is missing")
        return statistics.fmean(float(value) for value in numeric)

    endpoint = int(thresholds["descriptive_endpoint_k"])
    formal = "formal_factor_view_a_only_pdhg"
    variants = {
        "factor_row_hybrid": "factor_row_hybrid_a_only_pdhg",
        "exact_abs_view": "exact_abs_view_a_only_pdhg",
        "exact_abs_row": "exact_abs_row_a_only_pdhg",
    }
    formal_mean = mean_metric(formal, endpoint, "normalized_data_residual_l2")
    mean_gain_percent = {
        name: 100.0
        * (
            formal_mean
            - mean_metric(method, endpoint, "normalized_data_residual_l2")
        )
        / formal_mean
        for name, method in variants.items()
    }
    paired_gain_percent: dict[str, list[float]] = {name: [] for name in variants}
    for replicate in OPENED_REPLICATES:
        for sample_index in range(len(REACTION_FAMILIES)):
            baseline = paired_data[(replicate, sample_index, formal, endpoint)]
            for name, method in variants.items():
                candidate = paired_data[(replicate, sample_index, method, endpoint)]
                paired_gain_percent[name].append(
                    100.0 * (baseline - candidate) / max(baseline, 1e-30)
                )

    material_gain = float(thresholds["material_residual_gain_percent_min"])
    material_count_min = int(thresholds["material_paired_sample_count_min"])
    material_counts = {
        name: sum(value >= material_gain for value in values)
        for name, values in paired_gain_percent.items()
    }
    high_quantile_slack = statistics.median(
        max(1.0 - float(row["row_ratio_p05"]), 1.0 - float(row["column_ratio_p05"]))
        for row in tightness_rows
    )
    slack_material = high_quantile_slack >= float(
        thresholds["material_high_quantile_slack_min"]
    )
    exact_graph_field_ratio = (
        mean_metric("exact_abs_row_a_only_pdhg", endpoint, "field_relative_l2")
        / mean_metric("graph_pcgls", endpoint, "field_relative_l2")
    )

    exact_view_material = (
        mean_gain_percent["exact_abs_view"] >= material_gain
        and material_counts["exact_abs_view"] >= material_count_min
        and slack_material
    )
    factor_row_material = (
        mean_gain_percent["factor_row_hybrid"] >= material_gain
        and material_counts["factor_row_hybrid"] >= material_count_min
    )
    exact_row_material = (
        mean_gain_percent["exact_abs_row"] >= material_gain
        and material_counts["exact_abs_row"] >= material_count_min
    )
    if exact_view_material:
        status = "FACTOR_MAJORIZER_CANCELLATION_MATERIAL_DESCRIPTIVE"
    elif factor_row_material:
        status = "VIEW_AGGREGATION_MATERIAL_DESCRIPTIVE"
    elif exact_row_material:
        status = "COMBINED_STATIC_DIAGONAL_MATERIAL_DESCRIPTIVE"
    elif (
        mean_gain_percent["exact_abs_row"] < material_gain
        and exact_graph_field_ratio
        >= float(thresholds["nonbinding_exact_to_graph_field_ratio_min"])
    ):
        status = "STATIC_DIAGONAL_GAIN_SMALL_GRAPH_HEADROOM_NONBINDING"
    else:
        status = "INDETERMINATE_MIXED_DESCRIPTIVE"

    return {
        "status": status,
        "formal_gate_b_reopened": False,
        "descriptive_endpoint_k": endpoint,
        "mean_normalized_residual_gain_percent": mean_gain_percent,
        "paired_material_gain_count": material_counts,
        "material_gain_threshold_percent": material_gain,
        "material_paired_sample_count_min": material_count_min,
        "median_high_quantile_factor_slack": high_quantile_slack,
        "material_factor_slack": slack_material,
        "exact_abs_row_to_graph_field_error_ratio_nonbinding": (
            exact_graph_field_ratio
        ),
        "graph_comparison_binding": False,
        "graph_support_contract_matches_pdhg": False,
        "causal_krylov_explanation_claimed": False,
        "claim": (
            "OPENED_SYNTHETIC_SAME_SIGNED_A_DIAGONAL_DIAGNOSTIC_ONLY_"
            "NO_NEW_ALGORITHM_NO_EXPERIMENTAL_OR_GENERALIZATION_CLAIM"
        ),
    }


def run_diagnostic(
    *,
    root: Path,
    config_path: Path,
    view_root: Path,
    device_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = load_config(config_path)
    source_hashes = validate_sources(root, config)
    source_commit = validate_clean_repository(root)
    if device_name != "mps" or not torch.backends.mps.is_available():
        raise ValueError("formal diagnostic requires the frozen MPS runtime")
    device = torch.device(device_name)
    torch.set_num_threads(int(config["runtime"]["torch_cpu_threads"]))
    checkpoints = tuple(int(value) for value in config["checkpoints"])
    operator, contexts, direction = _build_runtime(
        root=root,
        config=config,
        view_root=view_root.resolve(),
        device=device,
    )
    torch.mps.synchronize()
    audit_operator = copy.deepcopy(operator).to(device="cpu").to(dtype=torch.float64)
    audit_whitening_by_replicate = {
        int(context["replicate"]): copy.deepcopy(
            context["graph_operator"].whitening
        )
        .to(device="cpu")
        .to(dtype=torch.float64)
        for context in contexts
    }

    rows: list[dict[str, Any]] = []
    tightness_rows: list[dict[str, Any]] = []
    call_rows: list[dict[str, Any]] = []
    power_rows: list[dict[str, Any]] = []
    for context in contexts:
        replicate = int(context["replicate"])
        for sample_index, family in enumerate(context["families"]):
            setup = build_single_sample_factor_setup(
                voxel_operator=operator,
                source_whitening=context["graph_operator"].whitening,
                sample_index=sample_index,
                measurement_scale=float(context["measurement_scale"]),
                eta=float(config["solver"]["eta"]),
            )
            audit_setup = build_single_sample_factor_setup(
                voxel_operator=audit_operator,
                source_whitening=audit_whitening_by_replicate[replicate],
                sample_index=sample_index,
                measurement_scale=float(context["measurement_scale"]),
                eta=float(config["solver"]["eta"]),
            )
            if not torch.equal(
                setup.active_primal_indices.detach().cpu(),
                audit_setup.active_primal_indices.detach().cpu(),
            ):
                raise ValueError("MPS solver and CPU64 audit active coordinates differ")
            audit = build_exact_absolute_audit(
                audit_setup,
                batch_size=int(config["exact_absolute"]["basis_batch_size"]),
                nonzero_tolerance=float(config["exact_absolute"]["nonzero_tolerance"]),
                dominance_absolute_tolerance=float(
                    config["thresholds"]["dominance_absolute_tolerance"]
                ),
                dominance_relative_tolerance=float(
                    config["thresholds"]["dominance_relative_tolerance"]
                ),
            )
            repeat_content_sha256: str | None = None
            if replicate == int(OPENED_REPLICATES[0]) and sample_index == 0:
                repeated = build_exact_absolute_audit(
                    audit_setup,
                    batch_size=int(config["exact_absolute"]["basis_batch_size"]),
                    nonzero_tolerance=float(
                        config["exact_absolute"]["nonzero_tolerance"]
                    ),
                    dominance_absolute_tolerance=float(
                        config["thresholds"]["dominance_absolute_tolerance"]
                    ),
                    dominance_relative_tolerance=float(
                        config["thresholds"]["dominance_relative_tolerance"]
                    ),
                )
                repeat_content_sha256 = repeated.content_sha256
                if repeat_content_sha256 != audit.content_sha256:
                    raise ValueError("MPS exact/factor basis replay is not deterministic")
            tightness = summarize_tightness(audit)
            if max(
                audit.setup_factor_row_relative_error,
                audit.setup_factor_column_relative_error,
            ) > float(config["thresholds"]["factor_replay_relative_error_maximum"]):
                raise ValueError("streamed factor replay differs from setup sums")
            if audit.factor_active_column_count != int(
                config["expected_runtime_shape"]["factor_majorizer_active_coordinate_count"]
            ):
                raise ValueError("factor-majorizer active coordinate count changed")
            if audit.factor_only_active_column_count != 0:
                raise ValueError("M-active and signed-A-nonzero coordinate sets differ")
            tightness_rows.append(
                {
                    "replicate": replicate,
                    "sample_index": sample_index,
                    "reaction_family": str(family),
                    "row_ratio_minimum": tightness["row_ratio"]["minimum"],
                    "row_ratio_p05": tightness["row_ratio"]["p05"],
                    "row_ratio_median": tightness["row_ratio"]["median"],
                    "row_ratio_mean": tightness["row_ratio"]["mean"],
                    "column_ratio_minimum": tightness["column_ratio"]["minimum"],
                    "column_ratio_p05": tightness["column_ratio"]["p05"],
                    "column_ratio_median": tightness["column_ratio"]["median"],
                    "column_ratio_mean": tightness["column_ratio"]["mean"],
                    "global_exact_to_factor_mass_ratio": tightness[
                        "global_exact_to_factor_mass_ratio"
                    ],
                    "global_slack_mass": tightness["global_slack_mass"],
                    "exact_zero_row_count": tightness["exact_zero_row_count"],
                    "exact_zero_column_count": tightness["exact_zero_column_count"],
                    "factor_only_nonzero_count": tightness[
                        "factor_only_nonzero_count"
                    ],
                    "exact_only_nonzero_count": tightness["exact_only_nonzero_count"],
                    "factor_majorizer_active_coordinate_count": (
                        audit.factor_active_column_count
                    ),
                    "signed_A_nonzero_coordinate_count": (
                        audit.exact_active_column_count
                    ),
                    "M_active_A_zero_coordinate_count": (
                        audit.factor_only_active_column_count
                    ),
                    "nullspace_dimension_claimed": False,
                    "dominance_violation_maximum": audit.dominance_violation_maximum,
                    "dominance_relative_violation_maximum": (
                        audit.dominance_relative_violation_maximum
                    ),
                    "setup_factor_row_relative_error": (
                        audit.setup_factor_row_relative_error
                    ),
                    "setup_factor_column_relative_error": (
                        audit.setup_factor_column_relative_error
                    ),
                    "audit_content_sha256": audit.content_sha256,
                    "mps_repeat_content_sha256": repeat_content_sha256,
                    "mps_repeat_required_for_this_row": (
                        replicate == int(OPENED_REPLICATES[0]) and sample_index == 0
                    ),
                    "solver_mps_setup_call_ledger": setup.setup_call_ledger.__dict__,
                    "audit_cpu64_setup_call_ledger": (
                        audit_setup.setup_call_ledger.__dict__
                    ),
                    "exact_streaming_replay_call_ledger": (
                        audit.replay_call_ledger.__dict__
                    ),
                }
            )

            for mode in ("factor_row", "exact_view", "exact_row"):
                metric = describe_exact_absolute_metric(setup, audit, mode)
                audit_metric = describe_exact_absolute_metric(
                    audit_setup,
                    audit,
                    mode,
                )
                audit_setup.pipeline.reset_call_ledger()
                estimate = normalized_operator_power_estimate(
                    audit_setup,
                    audit_metric,
                    iterations=int(config["exact_absolute"]["power_iterations"]),
                    seed=int(config["exact_absolute"]["power_seed"])
                    + 100 * replicate
                    + sample_index,
                )
                certificate = schur_safety_certificate_squared(
                    setup,
                    audit,
                    metric,
                    dominance_absolute_tolerance=float(
                        config["thresholds"]["dominance_absolute_tolerance"]
                    ),
                    dominance_relative_tolerance=float(
                        config["thresholds"]["dominance_relative_tolerance"]
                    ),
                )
                if estimate > float(
                    config["thresholds"]["power_estimate_sanity_maximum"]
                ):
                    raise ValueError("normalized-operator power estimate exceeds sanity gate")
                power_rows.append(
                    {
                        "replicate": replicate,
                        "sample_index": sample_index,
                        "mode": mode,
                        "normalized_norm_squared_power_estimate": estimate,
                        "power_value_is_upper_bound": False,
                        "schur_certificate_squared_upper_bound": certificate,
                        "schur_certificate_is_theorem_backed": True,
                        "eta_squared": float(config["solver"]["eta"]) ** 2,
                        "power_call_ledger": audit_setup.pipeline.call_ledger().__dict__,
                        "power_estimate_device": "cpu",
                        "power_estimate_dtype": "torch.float64",
                        "solver_metric_device": "mps",
                        "solver_metric_dtype": "torch.float32",
                    }
                )

            target = context["b0"][sample_index].reshape(
                setup.pipeline.view_count,
                setup.pipeline.rays_per_view,
                2,
            )
            truth = context["truth"][sample_index : sample_index + 1]
            output_scale = context["amplitude_scale"][sample_index].reshape(
                1, 1, 1, 1, 1
            )
            for method in METHODS[:-1]:
                method_rows, calls = _run_pdhg_method(
                    setup=setup,
                    target=target,
                    truth=truth,
                    output_scale=output_scale,
                    audit=audit,
                    method=method,
                    checkpoints=checkpoints,
                    theta=float(config["solver"]["theta"]),
                    replicate=replicate,
                    sample_index=sample_index,
                    family=str(family),
                    device=device,
                )
                rows.extend(method_rows)
                call_rows.append(calls)
            graph_rows, graph_calls = _run_graph_method(
                operator=operator,
                context=context,
                direction=direction,
                setup=setup,
                checkpoints=checkpoints,
                sample_index=sample_index,
                family=str(family),
                device=device,
            )
            rows.extend(graph_rows)
            call_rows.append(graph_calls)

    summaries = _summaries(rows)
    decision = classify_root_cause(rows, tightness_rows, config["thresholds"])
    report = {
        "schema_version": SCHEMA,
        "exact_absolute_schema": EXACT_ABSOLUTE_DIAGNOSTIC_SCHEMA,
        "status": decision["status"],
        "source_commit": source_commit,
        "config_sha256": _sha256(config_path),
        "source_sha256": source_hashes,
        "evidence_role": config["evidence_role"],
        "data_contract": config["data_contract"],
        "claim_boundary": config["claim_boundary"],
        "operator_contract": {
            "solver_recurrence_operator": "SIGNED_A",
            "absolute_operator_role": "DIAGONAL_METRIC_ONLY",
            "factor_majorizer_relation": "ENTRYWISE_M_GREATER_OR_EQUAL_ABS_A",
            "factor_active_coordinates_are_nullspace_dimension": False,
            "power_iteration_role": "NONBINDING_STRESS_ESTIMATE_NOT_BOUND",
            "schur_certificate_role": "THEOREM_BACKED_SAFETY_UPPER_BOUND",
            "graph_pcgls_binding": False,
            "graph_full_support_matches_reduced_pdhg_support": False,
        },
        "decision": decision,
        "summaries": summaries,
        "metric_row_count": len(rows),
        "tightness_row_count": len(tightness_rows),
        "power_row_count": len(power_rows),
        "call_row_count": len(call_rows),
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "device": device_name,
            "dtype": "torch.float32",
            "audit_device": "cpu",
            "audit_dtype": "torch.float64",
        },
        "scientific_claim_boundary": (
            "POST_NO_GO_OPENED_SYNTHETIC_ROOT_CAUSE_DIAGNOSTIC_ONLY"
        ),
    }
    return report, rows, tightness_rows, [*power_rows, *call_rows]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise ValueError("CSV rows must share ordered fields")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_release(
    output: Path,
    *,
    report: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    tightness_rows: Sequence[Mapping[str, Any]],
    audit_rows: Sequence[Mapping[str, Any]],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_bytes(_canonical_bytes(report) + b"\n")
    _write_csv(output / "trajectory_rows.csv", rows)
    _write_csv(output / "tightness_rows.csv", tightness_rows)
    (output / "audit_rows.json").write_bytes(_canonical_bytes(audit_rows) + b"\n")
    files = (
        "report.json",
        "trajectory_rows.csv",
        "tightness_rows.csv",
        "audit_rows.json",
    )
    checksums = "".join(f"{_sha256(output / name)}  {name}\n" for name in files)
    (output / "checksums.sha256").write_text(checksums, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT
        / "demo_t16_operator/configs/psu_b0_exact_absolute_root_cause_v1.json",
    )
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT
        / "demo_t16_operator/results/psu_b0_exact_absolute_root_cause",
    )
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    report, rows, tightness_rows, audit_rows = run_diagnostic(
        root=REPOSITORY_ROOT,
        config_path=args.config,
        view_root=args.view_root,
        device_name=str(args.device),
    )
    write_release(
        args.output.resolve(),
        report=report,
        rows=rows,
        tightness_rows=tightness_rows,
        audit_rows=audit_rows,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
