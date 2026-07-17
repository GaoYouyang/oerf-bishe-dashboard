#!/usr/bin/env python3
"""Run the first fixed-budget JACRU mechanism gate on independent observations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch

from demo_t16_operator.analytic_bost_phantoms import analytic_phantom_grid
from demo_t16_operator.interface_baselines import (
    cgls_baseline,
    edge_preserving_pdhg_baseline,
)
from demo_t16_operator.jacru_synthetic_fixture import (
    JACRUSyntheticFixtureConfig,
    build_jacru_synthetic_case,
)
from demo_t16_operator.jump_aware_cone_ray_unrolling import (
    JumpAwareConfig,
    optimize_jump_aware_cone_ray,
    voxel_operator_gradient_forward,
)
from demo_t16_operator.multi_interface_metrics import (
    multi_interface_level_set_metrics,
)
from demo_t16_operator.phase_interface_bost import (
    PhaseInterfaceConfig,
    optimize_phase_interface_bost,
)
from demo_t16_operator.psu_b0_streaming_operator import zero_outer_boundary_support
from demo_t16_operator.spatial_reconstruction_metrics import synthetic_field_metrics


REPORT_SCHEMA = "jacru-m0-independent-renderer-development-gate-1.0"
METHODS = ("cgls", "huber_pdhg", "phase_only", "jacru_no_bias", "jacru_bias")
BASELINES = ("cgls", "huber_pdhg", "phase_only")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("config must contain one JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_l2(value: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(reference).clamp_min(1e-30)
    return float(torch.linalg.vector_norm(value - reference) / denominator)


def _operator_maps(operator) -> tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
    def forward(field: torch.Tensor) -> torch.Tensor:
        return operator(field[None, None])[0]

    def adjoint(observation: torch.Tensor) -> torch.Tensor:
        return operator.adjoint(observation[None])[0, 0]

    return forward, adjoint


@torch.no_grad()
def _dense_norm_squared_bound(
    operator,
    *,
    batch_size: int,
    safety_factor: float,
) -> dict[str, Any]:
    voxel_count = int(np.prod(operator.grid_shape))
    rows = []
    start_calls = int(operator.forward_calls)
    for start in range(0, voxel_count, int(batch_size)):
        stop = min(start + int(batch_size), voxel_count)
        indices = torch.arange(start, stop, dtype=torch.int64)
        basis = torch.zeros(
            (stop - start, voxel_count),
            dtype=operator.support.dtype,
            device=operator.support.device,
        )
        basis[torch.arange(stop - start), indices] = 1.0
        projected = operator.forward(
            basis.reshape(stop - start, 1, *operator.grid_shape)
        )
        rows.append(projected.reshape(stop - start, -1).cpu())
    matrix = torch.cat(rows, dim=0)
    singular_maximum = float(torch.linalg.svdvals(matrix).max())
    estimate = singular_maximum**2
    return {
        "matrix_shape": [int(matrix.shape[1]), int(matrix.shape[0])],
        "spectral_norm_squared": estimate,
        "bound": float(safety_factor) * estimate,
        "safety_factor": float(safety_factor),
        "setup_forward_calls": int(operator.forward_calls) - start_calls,
        "status": "DENSE_NUMERICAL_SVD_TIMES_SAFETY_FACTOR_NOT_INTERVAL_CERTIFIED",
    }


def _spacing(config: JACRUSyntheticFixtureConfig) -> tuple[float, float, float]:
    shape = config.grid_shape
    return tuple(
        (config.domain_maximum_xyz[index] - config.domain_minimum_xyz[index])
        / (shape[2 - index] - 1)
        for index in range(3)
    )


def _adjoint_scale(method_config: dict[str, Any], norm_report: dict[str, Any]) -> float:
    mode = str(method_config.get("adjoint_initialization_scale_mode", "fixed"))
    if mode == "fixed":
        return float(method_config.get("adjoint_initialization_scale", 1.0))
    if mode == "inverse_dense_bound":
        return 1.0 / float(norm_report["bound"])
    raise ValueError(f"unsupported adjoint initialization scale mode: {mode}")


def _truth_level_sets(evaluation) -> np.ndarray:
    return evaluation.level_sets[0].movedim(0, -1).cpu().numpy()


def _selected_phase_levels(
    phase_fields: torch.Tensor,
    active_mask: torch.Tensor,
) -> list[np.ndarray]:
    active = active_mask[0].detach().cpu().numpy().astype(bool)
    values = phase_fields[0].detach().cpu().numpy()
    return [values[index] for index, selected in enumerate(active) if selected]


def _score_method(
    *,
    method: str,
    field: torch.Tensor,
    measured_prediction: torch.Tensor,
    scalar_prediction: torch.Tensor,
    predicted_level_sets: list[np.ndarray],
    case,
    spacing_xyz: tuple[float, float, float],
    call_ledger: dict[str, Any],
    wall_seconds: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evaluation = case.evaluation
    truth_evaluation = analytic_phantom_grid(
        evaluation.phantom_spec,
        grid_shape=tuple(int(value) for value in field.shape),
        dtype=torch.float64,
    )
    prediction = field.detach().cpu().numpy()
    truth = evaluation.truth_volume[0, 0].cpu().numpy()
    field_metrics = synthetic_field_metrics(
        prediction,
        truth,
        analytic_truth_gradient_xyz=truth_evaluation.gradient_xyz.cpu().numpy(),
        spacing_xyz=spacing_xyz,
    )
    interface = multi_interface_level_set_metrics(
        predicted_level_sets,
        _truth_level_sets(evaluation),
        spacing_xyz=spacing_xyz,
    )
    row = {
        "case_id": case.inference.case_id,
        "split": case.inference.split,
        "family": evaluation.family,
        "base_seed": evaluation.base_seed,
        "method": method,
        **field_metrics,
        "measured_reprojection_relative_l2": _relative_l2(
            measured_prediction,
            case.inference.observations_uv[0],
        ),
        "clean_reprojection_relative_l2": _relative_l2(
            scalar_prediction,
            evaluation.clean_observations_uv[0],
        ),
        "interface_detection_f1": interface["interface_detection_f1"],
        "surface_assd": interface["penalized_surface_assd"],
        "surface_hd95": interface["penalized_surface_hd95"],
        "surface_f1_at_1dx": interface["penalized_surface_f1_at_1dx"],
        "surface_f1_at_2dx": interface["penalized_surface_f1_at_2dx"],
        "missed_truth_count": interface["missed_truth_count"],
        "false_positive_count": interface["false_positive_count"],
        "wall_seconds": float(wall_seconds),
        "optimization_forward_calls": int(call_ledger["forward"]),
        "optimization_vjp_or_adjoint_calls": int(call_ledger["vjp_or_adjoint"]),
        "initialization_adjoint_calls": int(call_ledger.get("initialization_adjoint", 0)),
        "evaluation_forward_calls": int(call_ledger.get("evaluation_forward", 1)),
    }
    if extra:
        row.update(extra)
    return row


def _run_case(
    case,
    *,
    config: dict[str, Any],
    fixture_config: JACRUSyntheticFixtureConfig,
    norm_report: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    operator = case.inference.operator
    support = zero_outer_boundary_support(
        fixture_config.grid_shape,
        dtype=torch.float64,
    )
    operator.support.copy_(support)
    spacing_xyz = _spacing(fixture_config)
    observation = case.inference.observations_uv[0]
    forward, adjoint = _operator_maps(operator)
    rows: list[dict[str, Any]] = []
    slices: dict[str, np.ndarray] = {
        "truth": case.evaluation.truth_volume[0, 0].cpu().numpy()
    }

    operator.reset_call_counts()
    started = time.perf_counter()
    cgls = cgls_baseline(
        observation,
        forward=forward,
        adjoint=adjoint,
        support=support,
        spacing_xyz=spacing_xyz,
        iterations=int(config["methods"]["cgls"]["iterations"]),
    )
    elapsed = time.perf_counter() - started
    cgls_prediction = forward(cgls.field)
    rows.append(
        _score_method(
            method="cgls",
            field=cgls.field,
            measured_prediction=cgls_prediction,
            scalar_prediction=cgls_prediction,
            predicted_level_sets=[cgls.field.cpu().numpy()],
            case=case,
            spacing_xyz=spacing_xyz,
            call_ledger={"forward": cgls.forward_calls, "vjp_or_adjoint": cgls.adjoint_calls},
            wall_seconds=elapsed,
        )
    )
    slices["cgls"] = cgls.field.cpu().numpy()

    operator.reset_call_counts()
    pdhg_config = config["methods"]["huber_pdhg"]
    started = time.perf_counter()
    pdhg = edge_preserving_pdhg_baseline(
        observation,
        forward=forward,
        adjoint=adjoint,
        support=support,
        spacing_xyz=spacing_xyz,
        iterations=int(pdhg_config["iterations"]),
        regularization_weight=float(pdhg_config["regularization_weight"]),
        data_norm_squared_bound=float(norm_report["bound"]),
        penalty="huber",
        huber_delta=float(pdhg_config["huber_delta"]),
        step_safety=float(pdhg_config["step_safety"]),
    )
    elapsed = time.perf_counter() - started
    pdhg_prediction = forward(pdhg.field)
    rows.append(
        _score_method(
            method="huber_pdhg",
            field=pdhg.field,
            measured_prediction=pdhg_prediction,
            scalar_prediction=pdhg_prediction,
            predicted_level_sets=[pdhg.field.cpu().numpy()],
            case=case,
            spacing_xyz=spacing_xyz,
            call_ledger={"forward": pdhg.forward_calls, "vjp_or_adjoint": pdhg.adjoint_calls},
            wall_seconds=elapsed,
        )
    )
    slices["huber_pdhg"] = pdhg.field.cpu().numpy()

    phase_config = config["methods"]["phase_only"]
    operator.reset_call_counts()
    started = time.perf_counter()
    phase = optimize_phase_interface_bost(
        case.inference.observations_uv,
        forward=operator.forward,
        adjoint=operator.adjoint,
        support=support,
        config=PhaseInterfaceConfig(
            grid_shape=fixture_config.grid_shape,
            max_interfaces=int(phase_config["max_interfaces"]),
            spacing_xyz=spacing_xyz,
            epsilon=float(phase_config["epsilon_voxels"]) * spacing_xyz[0],
            optimization_steps=int(phase_config["optimization_steps"]),
            learning_rate=float(phase_config["learning_rate"]),
            seed=int(case.evaluation.base_seed),
            initial_gate_logit=float(phase_config.get("initial_gate_logit", -1.5)),
            adjoint_initialization_scale=_adjoint_scale(phase_config, norm_report),
            background_smoothness_weight=float(
                phase_config["background_smoothness_weight"]
            ),
            phase_smoothness_weight=float(phase_config["phase_smoothness_weight"]),
            eikonal_weight=float(phase_config["eikonal_weight"]),
            gate_sparsity_weight=float(phase_config["gate_sparsity_weight"]),
            amplitude_weight=float(phase_config["amplitude_weight"]),
        ),
    )
    elapsed = time.perf_counter() - started
    phase_field = phase.soft_prediction[0, 0]
    phase_prediction = operator.forward(phase.soft_prediction)[0].detach()
    rows.append(
        _score_method(
            method="phase_only",
            field=phase_field,
            measured_prediction=phase_prediction,
            scalar_prediction=phase_prediction,
            predicted_level_sets=_selected_phase_levels(
                phase.phase_fields,
                phase.active_interface_mask,
            ),
            case=case,
            spacing_xyz=spacing_xyz,
            call_ledger={
                "forward": phase.forward_evaluations,
                "vjp_or_adjoint": phase.forward_evaluations,
                "initialization_adjoint": phase.adjoint_evaluations,
            },
            wall_seconds=elapsed,
            extra={
                "gate_probability_max": float(phase.gate_probabilities.max()),
                "active_interface_count": int(phase.active_interface_mask.sum()),
            },
        )
    )
    slices["phase_only"] = phase_field.detach().cpu().numpy()

    for method in ("jacru_no_bias", "jacru_bias"):
        method_config = config["methods"][method]
        use_bias = int(method_config["bias_updates_per_outer"]) > 0
        operator.reset_call_counts()
        started = time.perf_counter()
        warm_start_iterations = int(method_config.get("warm_start_cgls_iterations", 0))
        warm_start = None
        initial_upstream = None
        if warm_start_iterations > 0:
            warm_start = cgls_baseline(
                observation,
                forward=forward,
                adjoint=adjoint,
                support=support,
                spacing_xyz=spacing_xyz,
                iterations=warm_start_iterations,
            )
            initial_upstream = warm_start.field[None, None]
        result = optimize_jump_aware_cone_ray(
            case.inference.observations_uv,
            gradient_forward=lambda gradient, op=operator: voxel_operator_gradient_forward(
                op, gradient
            ),
            scalar_adjoint=(None if warm_start is not None else operator.adjoint),
            support=support,
            initial_upstream=initial_upstream,
            ray_group_index=(case.inference.geometry.camera_index if use_bias else None),
            camera_group_count=(case.inference.geometry.camera_count if use_bias else 0),
            config=JumpAwareConfig(
                grid_shape=fixture_config.grid_shape,
                spacing_xyz=spacing_xyz,
                epsilon=float(method_config["epsilon_voxels"]) * spacing_xyz[0],
                outer_steps=int(method_config["outer_steps"]),
                field_updates_per_outer=int(method_config["field_updates_per_outer"]),
                interface_updates_per_outer=int(
                    method_config["interface_updates_per_outer"]
                ),
                bias_updates_per_outer=int(method_config["bias_updates_per_outer"]),
                field_learning_rate=float(method_config["field_learning_rate"]),
                interface_learning_rate=float(method_config["interface_learning_rate"]),
                bias_learning_rate=float(method_config.get("bias_learning_rate", 0.01)),
                seed=int(case.evaluation.base_seed),
                initial_phase_mode=str(method_config.get("initial_phase_mode", "fixed_x")),
                initial_gate_logit=float(method_config.get("initial_gate_logit", 0.0)),
                initial_jump_amplitude=float(
                    method_config.get("initial_jump_amplitude", 0.05)
                ),
                learn_upstream_field=bool(
                    method_config.get("learn_upstream_field", True)
                ),
                adjoint_initialization_scale=_adjoint_scale(
                    method_config,
                    norm_report,
                ),
                side_smoothness_weight=float(
                    method_config.get("side_smoothness_weight", 1e-4)
                ),
                jump_smoothness_weight=float(
                    method_config.get("jump_smoothness_weight", 1e-4)
                ),
                jump_amplitude_weight=float(
                    method_config.get("jump_amplitude_weight", 1e-5)
                ),
                eikonal_weight=float(method_config.get("eikonal_weight", 1e-3)),
                curvature_weight=float(method_config.get("curvature_weight", 1e-4)),
                interface_localization_weight=float(
                    method_config.get("interface_localization_weight", 1e-3)
                ),
                gate_sparsity_weight=float(
                    method_config.get("gate_sparsity_weight", 1e-4)
                ),
                camera_bias_weight=float(method_config.get("camera_bias_weight", 1e-3)),
            ),
        )
        elapsed = time.perf_counter() - started
        field = result.soft_volume[0, 0]
        with torch.no_grad():
            scalar_prediction = operator(result.soft_volume)[0]
        predicted_levels = (
            [result.model.phase_field[0, 0].detach().cpu().numpy()]
            if bool(result.active_gate[0])
            else []
        )
        extra = {
            "gate_probability_max": float(result.gate_probability.max()),
            "active_interface_count": int(result.active_gate.sum()),
            "jump_rms": float(result.jump_rms[0]),
            "gradient_closure_rms": float(result.soft_gradient_split.closure_rms),
        }
        if use_bias:
            centered = result.model.centered_camera_bias()
            assert centered is not None
            truth_bias = case.evaluation.camera_bias_uv
            truth_centered = truth_bias - truth_bias.mean(dim=0, keepdim=True)
            extra["camera_bias_rmse"] = float(
                torch.sqrt(
                    torch.mean((centered[0].detach() - truth_centered).square())
                )
            )
        warm_forward_calls = 0 if warm_start is None else warm_start.forward_calls
        warm_adjoint_calls = 0 if warm_start is None else warm_start.adjoint_calls
        total_forward_calls = warm_forward_calls + result.optimization_forward_evaluations
        total_reverse_calls = warm_adjoint_calls + result.implicit_data_vjp_evaluations
        expected_pairs = int(config["budget"]["physical_forward_vjp_pairs_per_method"])
        if total_forward_calls != expected_pairs or total_reverse_calls != expected_pairs:
            raise RuntimeError(
                f"{method} violated the frozen pair budget: "
                f"{total_forward_calls}/{total_reverse_calls} != {expected_pairs}"
            )
        extra["warm_start_cgls_iterations"] = warm_start_iterations
        extra["warm_start_forward_calls"] = warm_forward_calls
        extra["warm_start_adjoint_calls"] = warm_adjoint_calls
        rows.append(
            _score_method(
                method=method,
                field=field,
                measured_prediction=result.soft_observation[0],
                scalar_prediction=scalar_prediction,
                predicted_level_sets=predicted_levels,
                case=case,
                spacing_xyz=spacing_xyz,
                call_ledger={
                    "forward": total_forward_calls,
                    "vjp_or_adjoint": total_reverse_calls,
                    "initialization_adjoint": result.adjoint_evaluations,
                    "evaluation_forward": result.reporting_forward_evaluations + 1,
                },
                wall_seconds=elapsed,
                extra=extra,
            )
        )
        slices[method] = field.detach().cpu().numpy()
    return rows, slices


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    metric_names = (
        "field_relative_l2",
        "h1_seminorm_relative_error",
        "measured_reprojection_relative_l2",
        "clean_reprojection_relative_l2",
        "surface_assd",
        "surface_f1_at_1dx",
        "wall_seconds",
    )
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        summary: dict[str, Any] = {"case_count": len(selected)}
        for metric in metric_names:
            values = [float(row[metric]) for row in selected if row.get(metric) is not None]
            summary[f"{metric}_mean"] = None if not values else float(np.mean(values))
            summary[f"{metric}_maximum"] = None if not values else float(np.max(values))
        result[method] = summary
    return result


def _decision(
    rows: list[dict[str, Any]],
    aggregate: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    candidate_name = str(config["primary_candidate"])
    candidate = aggregate[candidate_name]
    best_baseline_name = min(
        BASELINES,
        key=lambda name: float(aggregate[name]["field_relative_l2_mean"]),
    )
    baseline = aggregate[best_baseline_name]

    def gain(smaller: float, larger: float) -> float:
        return (larger - smaller) / max(abs(larger), 1e-30)

    field_gain = gain(
        float(candidate["field_relative_l2_mean"]),
        float(baseline["field_relative_l2_mean"]),
    )
    h1_gain = gain(
        float(candidate["h1_seminorm_relative_error_mean"]),
        float(baseline["h1_seminorm_relative_error_mean"]),
    )
    single_candidate = [
        row for row in rows if row["method"] == candidate_name and row["family"] == "single_interface"
    ]
    single_baseline = [
        row for row in rows if row["method"] == best_baseline_name and row["family"] == "single_interface"
    ]
    candidate_assd = float(np.mean([row["surface_assd"] for row in single_candidate]))
    baseline_assd = float(np.mean([row["surface_assd"] for row in single_baseline]))
    assd_gain = gain(candidate_assd, baseline_assd)
    candidate_f1 = float(np.mean([row["surface_f1_at_1dx"] for row in single_candidate]))
    baseline_f1 = float(np.mean([row["surface_f1_at_1dx"] for row in single_baseline]))
    f1_gain = candidate_f1 - baseline_f1
    reprojection_ratio = float(candidate["measured_reprojection_relative_l2_mean"]) / max(
        float(baseline["measured_reprojection_relative_l2_mean"]),
        1e-30,
    )
    smooth_candidate = [
        row for row in rows if row["method"] == candidate_name and row["family"] == "smooth_no_interface"
    ]
    smooth_baseline = [
        row for row in rows if row["method"] == best_baseline_name and row["family"] == "smooth_no_interface"
    ]
    smooth_harm = (
        float(np.mean([row["field_relative_l2"] for row in smooth_candidate]))
        / max(float(np.mean([row["field_relative_l2"] for row in smooth_baseline])), 1e-30)
        - 1.0
    )
    seed_wins = []
    for seed in config["cases"]["base_seeds"]:
        candidate_rows = [
            row for row in rows if row["method"] == candidate_name and row["base_seed"] == seed
        ]
        baseline_rows = [
            row for row in rows if row["method"] == best_baseline_name and row["base_seed"] == seed
        ]
        seed_wins.append(
            float(np.mean([row["field_relative_l2"] for row in candidate_rows]))
            < float(np.mean([row["field_relative_l2"] for row in baseline_rows]))
        )
    gates = config["decision_gates"]
    checks = {
        "field_gain": field_gain >= float(gates["mean_field_relative_l2_gain_minimum_fraction"]),
        "h1_gain": h1_gain >= float(gates["mean_h1_gain_minimum_fraction"]),
        "assd_gain": assd_gain >= float(gates["single_interface_assd_gain_minimum_fraction"]),
        "f1_gain": f1_gain >= float(gates["single_interface_f1_at_1dx_gain_minimum_absolute"]),
        "reprojection": reprojection_ratio <= float(gates["measured_reprojection_ratio_maximum"]),
        "smooth_harm": smooth_harm <= float(gates["smooth_case_field_harm_maximum_fraction"]),
        "every_seed_field_win": all(seed_wins),
    }
    passed = all(checks.values())
    return {
        "status": "M0_RETAIN_FOR_LARGER_TEST" if passed else "M0_NO_GO_OR_REVISE",
        "passed": passed,
        "candidate": candidate_name,
        "best_baseline_by_mean_field_l2": best_baseline_name,
        "checks": checks,
        "diagnostics": {
            "field_gain_fraction": field_gain,
            "h1_gain_fraction": h1_gain,
            "single_interface_assd_gain_fraction": assd_gain,
            "single_interface_f1_at_1dx_gain_absolute": f1_gain,
            "measured_reprojection_ratio": reprojection_ratio,
            "smooth_case_field_harm_fraction": smooth_harm,
            "per_seed_field_wins": seed_wins,
        },
        "authorization": "DEVELOPMENT_MECHANISM_ROUTING_ONLY_NOT_A_METHOD_WIN",
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _plot(
    *,
    rows: list[dict[str, Any]],
    aggregate: dict[str, dict[str, Any]],
    representative: dict[str, np.ndarray],
    decision: dict[str, Any],
    output: Path,
) -> None:
    figure = plt.figure(figsize=(18, 7.8), constrained_layout=True)
    grid = figure.add_gridspec(2, 6, height_ratios=(1.0, 1.15))
    labels = list(METHODS)
    colors = ["#64748b", "#0f766e", "#2563eb", "#d97706", "#be123c"]
    metrics = (
        ("field_relative_l2_mean", "Field relative-L2"),
        ("h1_seminorm_relative_error_mean", "H1 seminorm error"),
        ("surface_f1_at_1dx_mean", "Interface F1 @ 1dx"),
        ("measured_reprojection_relative_l2_mean", "Measured reprojection rel-L2"),
    )
    for index, (key, title) in enumerate(metrics):
        axis = figure.add_subplot(grid[0, index])
        values = [aggregate[name][key] for name in labels]
        axis.bar(range(len(labels)), values, color=colors)
        axis.set_title(title)
        axis.set_xticks(range(len(labels)), [name.replace("_", "\n") for name in labels], fontsize=7)
        axis.grid(axis="y", alpha=0.25)
    axis = figure.add_subplot(grid[0, 4:6])
    axis.axis("off")
    diagnostics = decision["diagnostics"]
    axis.text(
        0.0,
        1.0,
        "M0 DEVELOPMENT GATE\n"
        f"status: {decision['status']}\n"
        f"baseline: {decision['best_baseline_by_mean_field_l2']}\n"
        f"field gain: {diagnostics['field_gain_fraction']:+.2%}\n"
        f"H1 gain: {diagnostics['h1_gain_fraction']:+.2%}\n"
        f"ASSD gain: {diagnostics['single_interface_assd_gain_fraction']:+.2%}\n"
        f"F1 gain: {diagnostics['single_interface_f1_at_1dx_gain_absolute']:+.3f}\n\n"
        "Not experimental truth.\nNot operator-learning evidence.\nNot a method-win claim.",
        va="top",
        family="monospace",
        fontsize=9,
    )
    slice_names = ("truth", *METHODS)
    center = representative["truth"].shape[0] // 2
    limit = max(float(np.max(np.abs(representative[name][center]))) for name in slice_names)
    for index, name in enumerate(slice_names):
        axis = figure.add_subplot(grid[1, index])
        axis.imshow(
            representative[name][center],
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            origin="lower",
        )
        axis.set_title(name.replace("_", " "))
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(
        "JACRU-M0 independent-renderer development gate | fixed 24 forward/VJP pairs",
        fontsize=14,
    )
    figure.savefig(output, dpi=180)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def run_gate(*, config_path: Path, output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    config = _load_json(config_path)
    fixture = config["fixture"]
    fixture_config = JACRUSyntheticFixtureConfig(
        grid_shape=tuple(int(value) for value in fixture["grid_shape"]),
        detector_shape=tuple(int(value) for value in fixture["detector_shape"]),
        samples_per_ray=int(fixture["samples_per_ray"]),
        noise_relative_std=float(fixture["noise_relative_std"]),
        camera_bias_relative_std=float(fixture["camera_bias_relative_std"]),
        enable_noise=bool(fixture["enable_noise"]),
        enable_camera_bias=bool(fixture["enable_camera_bias"]),
    )
    cases = [
        build_jacru_synthetic_case(
            family=family,
            split=str(config["cases"]["split"]),
            base_seed=int(seed),
            config=fixture_config,
        )
        for seed in config["cases"]["base_seeds"]
        for family in config["cases"]["families"]
    ]
    norm_by_geometry: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    representative: dict[str, np.ndarray] | None = None
    for case in cases:
        geometry_digest = case.inference.geometry.digest
        if geometry_digest not in norm_by_geometry:
            support = zero_outer_boundary_support(
                fixture_config.grid_shape,
                dtype=torch.float64,
            )
            case.inference.operator.support.copy_(support)
            case.inference.operator.reset_call_counts()
            norm_by_geometry[geometry_digest] = _dense_norm_squared_bound(
                case.inference.operator,
                batch_size=int(config["budget"]["dense_norm_batch_size"]),
                safety_factor=float(config["budget"]["dense_norm_safety_factor"]),
            )
        case_rows, case_slices = _run_case(
            case,
            config=config,
            fixture_config=fixture_config,
            norm_report=norm_by_geometry[geometry_digest],
        )
        rows.extend(case_rows)
        if representative is None and case.evaluation.family == "single_interface":
            representative = case_slices
    aggregate = _aggregate(rows)
    decision = _decision(rows, aggregate, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "metric_rows.csv", rows)
    assert representative is not None
    _plot(
        rows=rows,
        aggregate=aggregate,
        representative=representative,
        decision=decision,
        output=output_dir / "diagnostic.png",
    )
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": decision["status"],
        "evidence_level": config["evidence_level"],
        "source_config_sha256": _sha256(config_path),
        "case_count": len(cases),
        "method_count": len(METHODS),
        "metric_row_count": len(rows),
        "fixture": fixture_config.manifest(),
        "case_contracts": [
            {
                "case_id": case.inference.case_id,
                "split": case.inference.split,
                "family": case.evaluation.family,
                "base_seed": case.evaluation.base_seed,
                "geometry_digest": case.inference.geometry.digest,
                "observation_digest": case.inference.observation_digest,
            }
            for case in cases
        ],
        "norm_setup": norm_by_geometry,
        "budget": config["budget"],
        "aggregate": aggregate,
        "decision": decision,
        "claim_boundary": config["claim_boundary"],
        "amendment": config.get("amendment"),
        "elapsed_seconds": time.perf_counter() - started,
        "public_export_policy": {
            "contains_truth_or_observation_arrays": False,
            "contains_geometry_arrays": False,
            "contains_only_aggregate_metrics_and_rendered_slices": True,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "# JACRU-M0 development gate\n\n"
        "Independent analytic gradients generate the observations; a discrete voxel "
        "operator performs inversion. This is a fixed-budget development mechanism "
        "test, not experimental validation, operator-learning evidence, or a method-win "
        "claim.\n",
        encoding="ascii",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_gate(config_path=args.config, output_dir=args.output)
    print(json.dumps(report["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
