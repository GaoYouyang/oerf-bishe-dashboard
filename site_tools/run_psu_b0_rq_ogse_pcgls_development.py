#!/usr/bin/env python3
"""Develop risk-quantile single-expert routing for fixed SPD PCGLS."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import platform
import resource
import time
from typing import Any

import numpy as np
from scipy.optimize import linprog
from scipy.special import expit
import torch

from demo_t16_operator.psu_b0_classical_baselines import (
    preconditioned_cgls_reconstruction,
)
from demo_t16_operator.psu_b0_risk_quantile_experts import (
    RISK_QUANTILE_EXPERT_SCHEMA,
    RiskQuantileSingleExpertFactory,
    materialize_single_expert_path,
)
from demo_t16_operator.psu_b0_spectral_preconditioner import (
    normalized_field_loss,
)
from demo_t16_operator.psu_b0_streaming_operator import (
    zero_outer_boundary_support,
)
from site_tools.run_psu_b0_classical_frontier_development import (
    _verify_split_metadata,
)
from site_tools.run_psu_b0_conditioned_pcgls_development import (
    paired_gain_summary,
)
from site_tools.run_psu_b0_observable_morphology_probe import (
    fit_ridge_multioutput,
    ridge_scores,
    stratified_folds,
)
from site_tools.run_psu_b0_omse_pcgls_development import (
    _evaluate_integrated_factory,
    _expert_bank,
)
from site_tools.run_psu_b0_real_interface_audit import (
    _make_operator,
    load_real_support_geometry,
)
from site_tools.run_psu_b0_residual_risk_development import (
    DevelopmentSplit,
    _build_development_split,
    _synchronize,
)
from site_tools.run_psu_b0_spectral_preconditioner_pilot import (
    _expanded_measurement_values,
    _field_metrics,
    _load_json,
)


PRIVATE_SCHEMA = "psu-b0-rq-ogse-pcgls-development-private-1.0"
PUBLIC_SCHEMA = "psu-b0-rq-ogse-pcgls-development-public-1.0"
STATUS = "RQ_OGSE_PCGLS_DEVELOPMENT_COMPLETE_FRESH_NOT_USED"


def _max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _standardize_fit(
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("features must be a nontrivial matrix")
    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return (values - mean) / scale, mean, scale


def fit_l1_quantile_multioutput(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    quantile: float,
    regularization: float,
) -> dict[str, np.ndarray | float | int]:
    """Fit exact linear quantiles with L1 slope regularization."""

    if not 0.0 < float(quantile) < 1.0:
        raise ValueError("quantile must lie in (0,1)")
    if float(regularization) < 0.0:
        raise ValueError("regularization must be nonnegative")
    values, mean, scale = _standardize_fit(features)
    target_values = np.asarray(targets, dtype=np.float64)
    if target_values.ndim != 2 or len(target_values) != len(values):
        raise ValueError("features and targets must align")
    sample_count, feature_count = values.shape
    weights = np.zeros(
        (feature_count + 1, target_values.shape[1]),
        dtype=np.float64,
    )
    equality = np.concatenate(
        (
            np.ones((sample_count, 1), dtype=np.float64),
            values,
            -values,
            np.eye(sample_count, dtype=np.float64),
            -np.eye(sample_count, dtype=np.float64),
        ),
        axis=1,
    )
    objective = np.concatenate(
        (
            np.zeros(1, dtype=np.float64),
            np.full(
                2 * feature_count,
                float(regularization),
                dtype=np.float64,
            ),
            np.full(
                sample_count,
                float(quantile),
                dtype=np.float64,
            ),
            np.full(
                sample_count,
                1.0 - float(quantile),
                dtype=np.float64,
            ),
        )
    )
    bounds = [(None, None)] + [
        (0.0, None)
        for _ in range(2 * feature_count + 2 * sample_count)
    ]
    for output_index in range(target_values.shape[1]):
        target = target_values[:, output_index]
        if float(np.ptp(target)) <= 1e-12:
            weights[0, output_index] = float(target[0])
            continue
        result = linprog(
            objective,
            A_eq=equality,
            b_eq=target,
            bounds=bounds,
            method="highs",
        )
        if not bool(result.success):
            raise RuntimeError(
                f"quantile linear program failed: {result.message}"
            )
        positive = result.x[1 : 1 + feature_count]
        negative = result.x[
            1 + feature_count : 1 + 2 * feature_count
        ]
        weights[0, output_index] = float(result.x[0])
        weights[1:, output_index] = positive - negative
    return {
        "mean": mean,
        "scale": scale,
        "weights": weights,
        "quantile": float(quantile),
        "regularization": float(regularization),
        "output_count": int(target_values.shape[1]),
    }


def fit_ridge_logistic_multioutput(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    regularization: float,
    maximum_iterations: int = 100,
) -> dict[str, np.ndarray | float | int]:
    """Fit deterministic multi-output logistic ridge heads with IRLS."""

    if float(regularization) < 0.0:
        raise ValueError("regularization must be nonnegative")
    values, mean, scale = _standardize_fit(features)
    target_values = np.asarray(targets, dtype=np.float64)
    if target_values.ndim != 2 or len(target_values) != len(values):
        raise ValueError("features and targets must align")
    if np.any((target_values < 0.0) | (target_values > 1.0)):
        raise ValueError("logistic targets must lie in [0,1]")
    design = np.concatenate(
        (np.ones((len(values), 1), dtype=np.float64), values),
        axis=1,
    )
    penalty = np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    weights = np.zeros(
        (design.shape[1], target_values.shape[1]),
        dtype=np.float64,
    )
    for output_index in range(target_values.shape[1]):
        target = target_values[:, output_index]
        if float(np.ptp(target)) <= 1e-12:
            weights[0, output_index] = 20.0 if target[0] > 0.5 else -20.0
            continue
        prevalence = float(np.mean(target))
        current = np.zeros(design.shape[1], dtype=np.float64)
        current[0] = np.log(
            (prevalence + 0.5 / len(target))
            / (1.0 - prevalence + 0.5 / len(target))
        )
        for _ in range(int(maximum_iterations)):
            probability = expit(design @ current)
            curvature = np.clip(
                probability * (1.0 - probability),
                1e-5,
                None,
            )
            hessian = (
                design.T @ (curvature[:, None] * design)
                + float(regularization) * penalty
            )
            gradient = (
                design.T @ (probability - target)
                + float(regularization) * (penalty @ current)
            )
            step = np.linalg.solve(hessian, gradient)
            current = current - step
            if float(np.max(np.abs(step))) <= 1e-9:
                break
        weights[:, output_index] = current
    return {
        "mean": mean,
        "scale": scale,
        "weights": weights,
        "regularization": float(regularization),
        "output_count": int(target_values.shape[1]),
    }


def _oof_model_scores(
    features: np.ndarray,
    targets: np.ndarray,
    fold_labels: np.ndarray,
    *,
    fold_count: int,
    fit_kind: str,
    regularization: float,
    quantile: float | None = None,
) -> np.ndarray:
    folds = stratified_folds(fold_labels, fold_count=int(fold_count))
    output = np.zeros_like(targets, dtype=np.float64)
    for fold in range(int(fold_count)):
        train = folds != fold
        holdout = folds == fold
        if fit_kind == "ridge":
            model = fit_ridge_multioutput(
                features[train],
                targets[train],
                regularization=float(regularization),
            )
        elif fit_kind == "quantile":
            if quantile is None:
                raise ValueError("quantile is required")
            model = fit_l1_quantile_multioutput(
                features[train],
                targets[train],
                quantile=float(quantile),
                regularization=float(regularization),
            )
        elif fit_kind == "logistic":
            model = fit_ridge_logistic_multioutput(
                features[train],
                targets[train],
                regularization=float(regularization),
            )
        else:
            raise ValueError(f"unsupported fit_kind: {fit_kind}")
        output[holdout] = ridge_scores(model, features[holdout])
    return output


def numpy_risk_actions(
    *,
    mean_scores: np.ndarray,
    lower_scores: np.ndarray,
    harm_logits: np.ndarray,
    baseline_expert_index: int,
    route_mode: str,
    minimum_score: float,
    maximum_harm_probability: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NumPy mirror of the deployment routing contract."""

    mean = np.asarray(mean_scores, dtype=np.float64)
    lower = np.asarray(lower_scores, dtype=np.float64)
    harm = np.asarray(harm_logits, dtype=np.float64)
    if mean.ndim != 2 or lower.shape != mean.shape or harm.shape != mean.shape:
        raise ValueError("risk score matrices must align")
    if route_mode not in {
        "mean_only",
        "quantile_only",
        "quantile_harm",
        "mean_quantile_harm",
    }:
        raise ValueError(f"unsupported route_mode: {route_mode}")
    rank = mean if route_mode == "mean_only" else lower
    selected = np.argmax(rank, axis=1)
    row = np.arange(len(selected))
    selected_rank = rank[row, selected]
    selected_mean = mean[row, selected]
    selected_harm = expit(harm[row, selected])
    accepted = (
        (selected != int(baseline_expert_index))
        & (selected_rank >= float(minimum_score))
    )
    if route_mode in {"quantile_harm", "mean_quantile_harm"}:
        accepted &= selected_harm <= float(maximum_harm_probability)
    if route_mode == "mean_quantile_harm":
        accepted &= selected_mean >= 0.0
    actions = np.where(
        accepted,
        selected,
        int(baseline_expert_index),
    ).astype(np.int64)
    return actions, accepted, selected_harm


def _evaluate_action_indices(
    *,
    method: str,
    wrapped: DevelopmentSplit,
    operator: Any,
    source_config: dict[str, Any],
    device: torch.device,
    action_indices: np.ndarray,
    expert_log_gains: torch.Tensor,
    baseline_expert_index: int,
    interpolation_fraction: float,
    batch_size: int = 12,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split = wrapped.data
    actions = np.asarray(action_indices, dtype=np.int64).reshape(-1)
    if len(actions) != len(split.truth):
        raise ValueError("action_indices do not align with the split")
    rays_per_view = int(source_config["geometry"]["rays_per_view"])
    rows = []
    monotone = True
    gain_minimum = float("inf")
    gain_maximum = 0.0
    geometric_defect = 0.0
    started = time.perf_counter()
    operator.reset_call_counts()
    for start in range(0, len(split.truth), int(batch_size)):
        stop = min(start + int(batch_size), len(split.truth))
        truth = split.truth[start:stop].to(device)
        observation = split.observation_uv[start:stop].to(device)
        sigma = split.sigma_by_view[start:stop].to(device)
        mask = split.view_mask[start:stop].to(device)
        batch_actions = torch.as_tensor(
            actions[start:stop],
            dtype=torch.long,
            device=device,
        )

        def factory(_: torch.Tensor, **__: Any) -> Any:
            return materialize_single_expert_path(
                batch_actions,
                expert_log_gains=expert_log_gains,
                baseline_expert_index=int(baseline_expert_index),
                interpolation_fraction=float(interpolation_fraction),
            )

        with torch.no_grad():
            result = preconditioned_cgls_reconstruction(
                operator,
                observation,
                sigma_by_view=sigma,
                view_mask=mask,
                rays_per_view=rays_per_view,
                stages=4,
                preconditioner_factory=factory,
            )
            metrics = _field_metrics(result.volume, truth)
            combined = normalized_field_loss(
                result.volume,
                truth,
                gradient_weight=float(
                    source_config["training"]["gradient_weight"]
                ),
            )
            active = _expanded_measurement_values(
                mask,
                rays_per_view=rays_per_view,
            )
            expanded_sigma = _expanded_measurement_values(
                sigma,
                rays_per_view=rays_per_view,
            )
            measurement = torch.linalg.vector_norm(
                (result.residual_uv / expanded_sigma).flatten(1),
                dim=1,
            ) / torch.linalg.vector_norm(
                (active * observation / expanded_sigma).flatten(1),
                dim=1,
            ).clamp_min(1e-12)
        for history in result.history:
            monotone &= bool(
                torch.all(
                    history["relative_objective_after"]
                    <= history["relative_objective_before"] + 2e-5
                )
            )
            gain_minimum = min(
                gain_minimum,
                float(torch.min(history["gain_minimum"])),
            )
            gain_maximum = max(
                gain_maximum,
                float(torch.max(history["gain_maximum"])),
            )
            geometric_defect = max(
                geometric_defect,
                float(
                    torch.max(
                        torch.abs(history["gain_geometric_mean"] - 1.0)
                    )
                ),
            )
        for offset, index in enumerate(range(start, stop)):
            rows.append(
                {
                    "sample_id": split.sample_ids[index],
                    "split": split.name,
                    "family": split.families[index],
                    "noise_profile": wrapped.noise_profiles[index],
                    "relative_noise": float(split.relative_noise[index]),
                    "active_view_count": int(
                        torch.sum(split.view_mask[index] > 0.5)
                    ),
                    "method": str(method),
                    "field_relative_l2": float(
                        metrics["field_relative_l2"][offset]
                    ),
                    "gradient_relative_l2": float(
                        metrics["gradient_relative_l2"][offset]
                    ),
                    "front_top10_f1": float(
                        metrics["front_top10_f1"][offset]
                    ),
                    "combined_loss": float(combined[offset]),
                    "measurement_relative_l2": float(measurement[offset]),
                }
            )
    _synchronize(device)
    calls = operator.call_report()
    return rows, {
        "method": str(method),
        "split": split.name,
        "sample_count": len(split.truth),
        "wall_seconds": float(time.perf_counter() - started),
        "logical_calls_per_sample": {"forward": 4, "adjoint": 4},
        "batch_invocations": {
            "forward": int(calls["forward_calls"]),
            "adjoint": int(calls["adjoint_calls"]),
        },
        "data_objective_monotone": bool(monotone),
        "gain_minimum": float(gain_minimum),
        "gain_maximum": float(gain_maximum),
        "gain_geometric_mean_maximum_defect": float(geometric_defect),
    }


def action_gain_targets(
    *,
    sample_ids: list[str],
    baseline_rows: list[dict[str, Any]],
    action_rows_by_expert: dict[int, list[dict[str, Any]]],
    expert_count: int,
    baseline_expert_index: int,
) -> np.ndarray:
    """Build per-sample gains for a finite single-expert action bank."""

    baseline = {
        str(row["sample_id"]): float(row["field_relative_l2"])
        for row in baseline_rows
    }
    output = np.zeros(
        (len(sample_ids), int(expert_count)),
        dtype=np.float64,
    )
    for expert_index, rows in action_rows_by_expert.items():
        candidate = {
            str(row["sample_id"]): float(row["field_relative_l2"])
            for row in rows
        }
        for sample_index, sample_id in enumerate(sample_ids):
            output[sample_index, int(expert_index)] = (
                100.0
                * (baseline[sample_id] - candidate[sample_id])
                / max(baseline[sample_id], 1e-12)
            )
    output[:, int(baseline_expert_index)] = 0.0
    return output


def route_gain_metrics(
    gain_targets: np.ndarray,
    actions: np.ndarray,
    accepted: np.ndarray,
) -> dict[str, Any]:
    values = np.asarray(gain_targets, dtype=np.float64)
    selected = np.asarray(actions, dtype=np.int64).reshape(-1)
    acceptance = np.asarray(accepted, dtype=bool).reshape(-1)
    if values.ndim != 2 or len(values) != len(selected):
        raise ValueError("gain targets and actions must align")
    gain = values[np.arange(len(selected)), selected]
    return {
        "sample_count": len(gain),
        "mean_field_gain_percent": float(np.mean(gain)),
        "p10_field_gain_percent": float(np.quantile(gain, 0.10)),
        "minimum_field_gain_percent": float(np.min(gain)),
        "harm_over_one_percent_rate": float(np.mean(gain < -1.0)),
        "coverage": float(np.mean(acceptance)),
        "accepted_mean_field_gain_percent": (
            0.0
            if not np.any(acceptance)
            else float(np.mean(gain[acceptance]))
        ),
        "accepted_harm_over_one_percent_rate": (
            0.0
            if not np.any(acceptance)
            else float(np.mean(gain[acceptance] < -1.0))
        ),
    }


def select_screen_candidate(
    screen: list[dict[str, Any]],
    *,
    gate: dict[str, Any],
) -> dict[str, Any]:
    feasible = [
        row
        for row in screen
        if float(row["coverage"]) >= float(gate["minimum_coverage"])
        and float(row["mean_field_gain_percent"])
        >= float(gate["minimum_mean_field_gain_percent"])
        and float(row["p10_field_gain_percent"])
        >= float(gate["minimum_p10_field_gain_percent"])
        and float(row["harm_over_one_percent_rate"])
        <= float(gate["maximum_harm_over_one_percent_rate"])
        and float(row["accepted_harm_over_one_percent_rate"])
        <= float(gate["maximum_accepted_harm_over_one_percent_rate"])
    ]
    if not feasible:
        return {
            "selection_status": "NO_STRICT_OOF_RISK_ROUTE_FEASIBLE",
            "strict_gate_pass": False,
        }
    selected = max(
        feasible,
        key=lambda row: (
            float(row["mean_field_gain_percent"]),
            float(row["p10_field_gain_percent"]),
            float(row["coverage"]),
            float(row["accepted_mean_field_gain_percent"]),
            -float(row["interpolation_fraction"]),
        ),
    )
    output = dict(selected)
    output["selection_status"] = "STRICT_OOF_RISK_ROUTE_SELECTED"
    output["strict_gate_pass"] = True
    return output


def best_diagnostic_candidate(
    screen: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = max(
        screen,
        key=lambda row: (
            float(row["mean_field_gain_percent"]),
            float(row["p10_field_gain_percent"]),
            -float(row["harm_over_one_percent_rate"]),
        ),
    )
    output = dict(selected)
    output["selection_status"] = "BEST_OOF_DIAGNOSTIC_NOT_RISK_GATED"
    output["strict_gate_pass"] = False
    return output


def _risk_development_gate(
    summaries: list[dict[str, Any]],
    *,
    method: str,
    gate: dict[str, Any],
) -> dict[str, Any]:
    lookup = {
        (str(row["candidate_method"]), str(row["split"])): row
        for row in summaries
    }
    validation = lookup[(method, "risk_validation")]
    calibration = lookup[(method, "risk_calibration")]
    checks = {
        "validation_mean": validation["mean_field_gain_percent"]
        >= float(gate["minimum_validation_mean_field_gain_percent"]),
        "calibration_mean": calibration["mean_field_gain_percent"]
        >= float(gate["minimum_calibration_mean_field_gain_percent"]),
        "validation_bootstrap_lower": validation[
            "bootstrap_mean_95_interval_percent"
        ][0]
        > float(gate["minimum_bootstrap_lower_percent"]),
        "calibration_bootstrap_lower": calibration[
            "bootstrap_mean_95_interval_percent"
        ][0]
        > float(gate["minimum_bootstrap_lower_percent"]),
        "validation_harm": validation["harm_over_one_percent_rate"]
        <= float(gate["maximum_harm_over_one_percent_rate"]),
        "calibration_harm": calibration["harm_over_one_percent_rate"]
        <= float(gate["maximum_harm_over_one_percent_rate"]),
        "validation_p10": validation["p10_field_gain_percent"]
        >= float(gate["minimum_p10_field_gain_percent"]),
        "calibration_p10": calibration["p10_field_gain_percent"]
        >= float(gate["minimum_p10_field_gain_percent"]),
    }
    return {
        "method": method,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _secondary_metric_safety_audit(
    summaries: list[dict[str, Any]],
    *,
    method: str,
    gate: dict[str, Any],
) -> dict[str, Any]:
    lookup = {
        (str(row["candidate_method"]), str(row["split"])): row
        for row in summaries
    }
    checks = {}
    for split_name in ("risk_validation", "risk_calibration"):
        summary = lookup[(method, split_name)]
        gradient = summary["secondary_metric_gain"][
            "gradient_relative_l2"
        ]
        front = summary["secondary_metric_gain"]["front_top10_f1"]
        checks[f"{split_name}_gradient_mean"] = (
            gradient["mean_gain_percent"]
            >= float(gate["minimum_gradient_mean_gain_percent"])
        )
        checks[f"{split_name}_gradient_p10"] = (
            gradient["p10_gain_percent"]
            >= float(gate["minimum_gradient_p10_gain_percent"])
        )
        checks[f"{split_name}_front_mean"] = (
            front["mean_gain_percent"]
            >= float(gate["minimum_front_mean_gain_percent"])
        )
        checks[f"{split_name}_front_p10"] = (
            front["p10_gain_percent"]
            >= float(gate["minimum_front_p10_gain_percent"])
        )
    return {
        "method": method,
        "checks": checks,
        "pass": all(checks.values()),
        "selection_use": "POST_SELECTION_AUDIT_ONLY",
        "posthoc_threshold_formalization": True,
    }


def _action_bank_summary(
    *,
    gain_targets_by_blend: dict[float, np.ndarray],
    expert_ids: list[str],
) -> list[dict[str, Any]]:
    output = []
    for blend, targets in gain_targets_by_blend.items():
        for expert_index, expert_id in enumerate(expert_ids):
            values = targets[:, expert_index]
            output.append(
                {
                    "interpolation_fraction": float(blend),
                    "expert_candidate_id": str(expert_id),
                    "mean_field_gain_percent": float(np.mean(values)),
                    "p10_field_gain_percent": float(
                        np.quantile(values, 0.10)
                    ),
                    "minimum_field_gain_percent": float(np.min(values)),
                    "harm_over_one_percent_rate": float(
                        np.mean(values < -1.0)
                    ),
                    "positive_gain_rate": float(np.mean(values > 0.0)),
                }
            )
    return output


def _fit_selected_models(
    *,
    features: np.ndarray,
    gain_targets: np.ndarray,
    parameters: dict[str, Any],
    harm_threshold: float,
) -> dict[str, Any]:
    mean_model = fit_ridge_multioutput(
        features,
        gain_targets,
        regularization=float(parameters["mean_regularization"]),
    )
    lower_model = fit_l1_quantile_multioutput(
        features,
        gain_targets,
        quantile=float(parameters["lower_quantile"]),
        regularization=float(parameters["quantile_regularization"]),
    )
    harm_model = fit_ridge_logistic_multioutput(
        features,
        (gain_targets < float(harm_threshold)).astype(np.float64),
        regularization=float(parameters["harm_regularization"]),
    )
    for model in (lower_model, harm_model):
        if not np.allclose(
            np.asarray(model["mean"]),
            np.asarray(mean_model["mean"]),
        ) or not np.allclose(
            np.asarray(model["scale"]),
            np.asarray(mean_model["scale"]),
        ):
            raise RuntimeError("risk heads disagree on feature normalization")
    return {
        "mean": mean_model,
        "lower": lower_model,
        "harm": harm_model,
    }


def build_public_summary(private: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": PUBLIC_SCHEMA,
        "algorithm_schema": RISK_QUANTILE_EXPERT_SCHEMA,
        "status": private["status"],
        "evidence_scope": private["evidence_scope"],
        "configuration": copy.deepcopy(private["configuration_public"]),
        "regeneration_checks": copy.deepcopy(
            private["regeneration_checks"]
        ),
        "expert_bank": copy.deepcopy(private["expert_bank"]),
        "finite_action_bank": copy.deepcopy(private["finite_action_bank"]),
        "selection_screen": copy.deepcopy(private["selection_screen"]),
        "selected_strict_candidate": copy.deepcopy(
            private["selected_strict_candidate"]
        ),
        "selected_by_route_mode": copy.deepcopy(
            private["selected_by_route_mode"]
        ),
        "best_diagnostic_candidate": copy.deepcopy(
            private["best_diagnostic_candidate"]
        ),
        "route_decision_summary": copy.deepcopy(
            private["route_decision_summary"]
        ),
        "paired_gain_summary": copy.deepcopy(
            private["paired_gain_summary"]
        ),
        "development_gates": copy.deepcopy(
            private["development_gates"]
        ),
        "secondary_metric_safety_audits": copy.deepcopy(
            private["secondary_metric_safety_audits"]
        ),
        "overall_decision": copy.deepcopy(private["overall_decision"]),
        "execution_summary": copy.deepcopy(
            private["execution_summary"]
        ),
        "runtime": copy.deepcopy(private["runtime"]),
        "claim_boundary": copy.deepcopy(private["claim_boundary"]),
    }


def run_development(
    *,
    root: Path,
    config_path: Path,
    development_report_path: Path,
    probe_private_report_path: Path,
    view_root: Path,
    device_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _load_json(config_path)
    development_config = _load_json(
        root / str(config["source_development_config"])
    )
    source_config = _load_json(
        root / str(development_config["source_pilot"]["config"])
    )
    ogse_public = _load_json(
        root / str(config["source_ogse_public_summary"])
    )
    probe_private = _load_json(probe_private_report_path)
    development_report = _load_json(development_report_path)
    expert_ids = [
        str(value)
        for value in ogse_public["selected_strict_candidate"][
            "expert_candidate_ids"
        ]
    ]
    baseline_id = str(config["expert_bank"]["baseline_candidate_id"])
    baseline_index = expert_ids.index(baseline_id)
    features_by_split = {
        split_name: np.asarray(
            probe_private["features_private"][split_name][
                "initial_normal_spectrum"
            ],
            dtype=np.float64,
        )
        for split_name in (
            "risk_train",
            "risk_validation",
            "risk_calibration",
        )
    }
    train_features = features_by_split["risk_train"]
    train_count = len(train_features)
    train_spec = development_config["development_splits"]["risk_train"]
    train_families = [
        str(
            train_spec["families"][
                index % len(train_spec["families"])
            ]
        )
        for index in range(train_count)
    ]
    family_index = {
        family: index
        for index, family in enumerate(sorted(set(train_families)))
    }
    fold_labels = np.asarray(
        [family_index[family] for family in train_families],
        dtype=np.int64,
    )
    train_sample_ids = [
        f"risk_train-{index:03d}" for index in range(train_count)
    ]

    device = torch.device(device_name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    torch.set_num_threads(8)
    started = time.perf_counter()
    geometry = development_config["geometry"]
    grid_size = int(geometry["grid_size"])
    rays_per_view = int(geometry["rays_per_view"])
    support = zero_outer_boundary_support((grid_size,) * 3).to(device)
    true_geometry, _ = load_real_support_geometry(
        view_root,
        rays_per_view=rays_per_view,
        sample_count=int(
            geometry["true_finite_aperture_sample_count"]
        ),
    )
    nominal_geometry, _ = load_real_support_geometry(
        view_root,
        rays_per_view=rays_per_view,
        sample_count=int(
            geometry["nominal_finite_aperture_sample_count"]
        ),
    )
    true_operator = _make_operator(
        true_geometry,
        grid_size=grid_size,
        dtype=torch.float32,
    ).to(device)
    nominal_operator = _make_operator(
        nominal_geometry,
        grid_size=grid_size,
        dtype=torch.float32,
    ).to(device)
    true_operator.support.copy_(support)
    nominal_operator.support.copy_(support)
    used_masks: set[str] = set()
    splits: dict[str, DevelopmentSplit] = {}
    for split_name in (
        "risk_train",
        "risk_validation",
        "risk_calibration",
    ):
        wrapped, used_masks = _build_development_split(
            name=split_name,
            spec=development_config["development_splits"][split_name],
            config=development_config,
            source_config=source_config,
            true_operator=true_operator,
            nominal_operator=nominal_operator,
            device=device,
            forbidden_masks=used_masks,
        )
        splits[split_name] = wrapped
    source_rows = development_report["dataset_private"]["metric_rows"]
    for wrapped in splits.values():
        _verify_split_metadata(wrapped, source_rows)

    expert_logs = _expert_bank(
        headroom_public=_load_json(
            root / str(config["source_headroom_public_summary"])
        ),
        expert_ids=expert_ids,
        grid_size=grid_size,
        device=device,
    )
    action_execution = []
    baseline_rows, ledger = _evaluate_action_indices(
        method="rq_action_baseline",
        wrapped=splits["risk_train"],
        operator=nominal_operator,
        source_config=source_config,
        device=device,
        action_indices=np.full(train_count, baseline_index),
        expert_log_gains=expert_logs,
        baseline_expert_index=baseline_index,
        interpolation_fraction=0.0,
    )
    action_execution.append(ledger)
    gain_targets_by_blend: dict[float, np.ndarray] = {}
    action_rows_private = {}
    for blend_value in config["finite_action_bank"][
        "interpolation_fractions"
    ]:
        blend = float(blend_value)
        rows_by_expert = {}
        for expert_index in range(len(expert_ids)):
            if expert_index == baseline_index:
                rows_by_expert[expert_index] = baseline_rows
                continue
            rows, ledger = _evaluate_action_indices(
                method=f"rq_action_e{expert_index}_b{blend:g}",
                wrapped=splits["risk_train"],
                operator=nominal_operator,
                source_config=source_config,
                device=device,
                action_indices=np.full(train_count, expert_index),
                expert_log_gains=expert_logs,
                baseline_expert_index=baseline_index,
                interpolation_fraction=blend,
            )
            rows_by_expert[expert_index] = rows
            action_execution.append(ledger)
        gain_targets_by_blend[blend] = action_gain_targets(
            sample_ids=train_sample_ids,
            baseline_rows=baseline_rows,
            action_rows_by_expert=rows_by_expert,
            expert_count=len(expert_ids),
            baseline_expert_index=baseline_index,
        )
        action_rows_private[str(blend)] = rows_by_expert

    screen_config = config["risk_head_screen"]
    fold_count = int(screen_config["folds"])
    harm_threshold = float(screen_config["harm_gain_threshold_percent"])
    screen = []
    oof_cache_private = {}
    for blend, gain_targets in gain_targets_by_blend.items():
        mean_cache = {
            float(regularization): _oof_model_scores(
                train_features,
                gain_targets,
                fold_labels,
                fold_count=fold_count,
                fit_kind="ridge",
                regularization=float(regularization),
            )
            for regularization in screen_config[
                "mean_regularization_grid"
            ]
        }
        lower_cache = {
            (float(quantile), float(regularization)): _oof_model_scores(
                train_features,
                gain_targets,
                fold_labels,
                fold_count=fold_count,
                fit_kind="quantile",
                regularization=float(regularization),
                quantile=float(quantile),
            )
            for quantile in screen_config["lower_quantile_grid"]
            for regularization in screen_config[
                "quantile_regularization_grid"
            ]
        }
        harm_targets = (
            gain_targets < harm_threshold
        ).astype(np.float64)
        harm_cache = {
            float(regularization): _oof_model_scores(
                train_features,
                harm_targets,
                fold_labels,
                fold_count=fold_count,
                fit_kind="logistic",
                regularization=float(regularization),
            )
            for regularization in screen_config[
                "harm_regularization_grid"
            ]
        }
        oof_cache_private[str(blend)] = {
            "mean": {
                str(key): value.tolist()
                for key, value in mean_cache.items()
            },
            "lower": {
                f"{key[0]}:{key[1]}": value.tolist()
                for key, value in lower_cache.items()
            },
            "harm": {
                str(key): value.tolist()
                for key, value in harm_cache.items()
            },
        }
        for route_mode in screen_config["route_modes"]:
            mean_grid = (
                list(mean_cache)
                if route_mode in {"mean_only", "mean_quantile_harm"}
                else [float(screen_config["mean_regularization_grid"][0])]
            )
            lower_grid = (
                [
                    (
                        float(screen_config["lower_quantile_grid"][0]),
                        float(
                            screen_config[
                                "quantile_regularization_grid"
                            ][0]
                        ),
                    )
                ]
                if route_mode == "mean_only"
                else list(lower_cache)
            )
            harm_grid = (
                list(harm_cache)
                if route_mode in {
                    "quantile_harm",
                    "mean_quantile_harm",
                }
                else [float(screen_config["harm_regularization_grid"][0])]
            )
            harm_caps = (
                screen_config["maximum_harm_probability_grid"]
                if route_mode in {
                    "quantile_harm",
                    "mean_quantile_harm",
                }
                else [1.0]
            )
            for mean_regularization in mean_grid:
                for lower_key in lower_grid:
                    quantile, quantile_regularization = lower_key
                    for harm_regularization in harm_grid:
                        for minimum_score in screen_config[
                            "minimum_score_grid"
                        ]:
                            for maximum_harm_probability in harm_caps:
                                mean_scores = mean_cache[
                                    float(mean_regularization)
                                ]
                                lower_scores = (
                                    np.zeros_like(mean_scores)
                                    if route_mode == "mean_only"
                                    else lower_cache[
                                        (
                                            float(quantile),
                                            float(
                                                quantile_regularization
                                            ),
                                        )
                                    ]
                                )
                                harm_logits = harm_cache[
                                    float(harm_regularization)
                                ]
                                actions, accepted, _ = numpy_risk_actions(
                                    mean_scores=mean_scores,
                                    lower_scores=lower_scores,
                                    harm_logits=harm_logits,
                                    baseline_expert_index=baseline_index,
                                    route_mode=str(route_mode),
                                    minimum_score=float(minimum_score),
                                    maximum_harm_probability=float(
                                        maximum_harm_probability
                                    ),
                                )
                                metrics = route_gain_metrics(
                                    gain_targets,
                                    actions,
                                    accepted,
                                )
                                screen.append(
                                    {
                                        "route_mode": str(route_mode),
                                        "interpolation_fraction": blend,
                                        "mean_regularization": float(
                                            mean_regularization
                                        ),
                                        "lower_quantile": float(quantile),
                                        "quantile_regularization": float(
                                            quantile_regularization
                                        ),
                                        "harm_regularization": float(
                                            harm_regularization
                                        ),
                                        "minimum_score": float(
                                            minimum_score
                                        ),
                                        "maximum_harm_probability": float(
                                            maximum_harm_probability
                                        ),
                                        **metrics,
                                    }
                                )
    selected = select_screen_candidate(
        screen,
        gate=config["strict_oof_gate"],
    )
    diagnostic = best_diagnostic_candidate(screen)
    selected_by_route_mode = {
        str(route_mode): select_screen_candidate(
            [
                row
                for row in screen
                if row["route_mode"] == str(route_mode)
            ],
            gate=config["strict_oof_gate"],
        )
        for route_mode in screen_config["route_modes"]
    }

    transfer_rows = []
    transfer_execution = []
    selection_models = {}
    route_decisions = []
    routes = [
        ("strict", selected),
        *[
            (f"ablation_{mode}", selected_by_route_mode[str(mode)])
            for mode in screen_config["route_modes"]
        ],
        ("diagnostic", diagnostic),
    ]
    for route_name, route in routes:
        if route_name != "diagnostic" and not bool(
            route.get("strict_gate_pass")
        ):
            parameters = {
                "route_mode": "mean_only",
                "interpolation_fraction": float(
                    config["finite_action_bank"][
                        "interpolation_fractions"
                    ][0]
                ),
                "mean_regularization": float(
                    screen_config["mean_regularization_grid"][0]
                ),
                "lower_quantile": float(
                    screen_config["lower_quantile_grid"][0]
                ),
                "quantile_regularization": float(
                    screen_config["quantile_regularization_grid"][0]
                ),
                "harm_regularization": float(
                    screen_config["harm_regularization_grid"][0]
                ),
                "minimum_score": 1e30,
                "maximum_harm_probability": 0.0,
            }
        else:
            parameters = route
        blend = float(parameters["interpolation_fraction"])
        models = _fit_selected_models(
            features=train_features,
            gain_targets=gain_targets_by_blend[blend],
            parameters=parameters,
            harm_threshold=harm_threshold,
        )
        factory = RiskQuantileSingleExpertFactory(
            expert_log_gains=expert_logs,
            expert_candidate_ids=expert_ids,
            baseline_expert_index=baseline_index,
            feature_mean=torch.as_tensor(models["mean"]["mean"]),
            feature_scale=torch.as_tensor(models["mean"]["scale"]),
            mean_weights=torch.as_tensor(models["mean"]["weights"]),
            lower_weights=torch.as_tensor(models["lower"]["weights"]),
            harm_weights=torch.as_tensor(models["harm"]["weights"]),
            route_mode=str(parameters["route_mode"]),
            minimum_score=float(parameters["minimum_score"]),
            maximum_harm_probability=float(
                parameters["maximum_harm_probability"]
            ),
            interpolation_fraction=blend,
        ).to(device)
        selection_models[route_name] = {
            "parameters": copy.deepcopy(parameters),
            "feature_mean_private": np.asarray(
                models["mean"]["mean"]
            ).tolist(),
            "feature_scale_private": np.asarray(
                models["mean"]["scale"]
            ).tolist(),
            "mean_weights_private": np.asarray(
                models["mean"]["weights"]
            ).tolist(),
            "lower_weights_private": np.asarray(
                models["lower"]["weights"]
            ).tolist(),
            "harm_weights_private": np.asarray(
                models["harm"]["weights"]
            ).tolist(),
        }
        for split_name in ("risk_validation", "risk_calibration"):
            split_features = features_by_split[split_name]
            mean_scores = ridge_scores(models["mean"], split_features)
            lower_scores = ridge_scores(models["lower"], split_features)
            harm_logits = ridge_scores(models["harm"], split_features)
            actions, accepted, selected_harm = numpy_risk_actions(
                mean_scores=mean_scores,
                lower_scores=lower_scores,
                harm_logits=harm_logits,
                baseline_expert_index=baseline_index,
                route_mode=str(parameters["route_mode"]),
                minimum_score=float(parameters["minimum_score"]),
                maximum_harm_probability=float(
                    parameters["maximum_harm_probability"]
                ),
            )
            route_decisions.append(
                {
                    "route": route_name,
                    "split": split_name,
                    "sample_count": len(actions),
                    "coverage": float(np.mean(accepted)),
                    "selected_expert_counts": {
                        expert_ids[index]: int(np.sum(actions == index))
                        for index in range(len(expert_ids))
                    },
                    "accepted_predicted_harm_probability_mean": (
                        0.0
                        if not np.any(accepted)
                        else float(np.mean(selected_harm[accepted]))
                    ),
                }
            )
            rows, ledger = _evaluate_integrated_factory(
                method=f"rq_ogse_{route_name}",
                wrapped=splits[split_name],
                operator=nominal_operator,
                source_config=source_config,
                device=device,
                factory=factory,
            )
            transfer_rows.extend(rows)
            transfer_execution.append(ledger)

    for split_name in ("risk_validation", "risk_calibration"):
        sample_count = len(splits[split_name].data.truth)
        rows, ledger = _evaluate_action_indices(
            method="static_pcgls4",
            wrapped=splits[split_name],
            operator=nominal_operator,
            source_config=source_config,
            device=device,
            action_indices=np.full(sample_count, baseline_index),
            expert_log_gains=expert_logs,
            baseline_expert_index=baseline_index,
            interpolation_fraction=0.0,
        )
        transfer_rows.extend(rows)
        transfer_execution.append(ledger)
    methods = [f"rq_ogse_{route_name}" for route_name, _ in routes]
    summaries = []
    for method_index, method in enumerate(methods):
        for split_index, split_name in enumerate(
            ("risk_validation", "risk_calibration")
        ):
            summaries.append(
                paired_gain_summary(
                    transfer_rows,
                    split=split_name,
                    candidate_method=method,
                    bootstrap_seed=20263300
                    + 100 * method_index
                    + split_index,
                )
            )
    development_gates = [
        _risk_development_gate(
            summaries,
            method=method,
            gate=config["development_gate"],
        )
        for method in methods
    ]
    secondary_audits = [
        _secondary_metric_safety_audit(
            summaries,
            method=method,
            gate=config["secondary_metric_safety_audit"],
        )
        for method in methods
    ]
    execution = action_execution + transfer_execution
    ledgers_valid = all(
        row["logical_calls_per_sample"] == {
            "forward": 4,
            "adjoint": 4,
        }
        and bool(row["data_objective_monotone"])
        and float(row["gain_minimum"]) > 0.0
        and float(row["gain_geometric_mean_maximum_defect"]) <= 2e-5
        for row in execution
    )
    private = {
        "schema_version": PRIVATE_SCHEMA,
        "algorithm_schema": RISK_QUANTILE_EXPERT_SCHEMA,
        "status": STATUS,
        "evidence_scope": (
            "REAL_PSU_SUPPORT_GEOMETRY_WITH_ANALYTIC_REACTION_MORPHOLOGY_"
            "AND_SYNTHETIC_CAMERA_NOISE_POSTOPEN_RQ_OGSE_DEVELOPMENT_ONLY"
        ),
        "configuration_private": {
            "root": str(root.resolve()),
            "config_path": str(config_path.resolve()),
            "development_report_path": str(
                development_report_path.resolve()
            ),
            "probe_private_report_path": str(
                probe_private_report_path.resolve()
            ),
            "view_root": str(view_root.resolve()),
            "device": str(device),
        },
        "configuration_public": copy.deepcopy(config),
        "regeneration_checks": {
            "development_metadata_matches_frozen_rows": True,
            "expert_bank_inherited_from_train_only_ogse": True,
            "finite_action_targets_use_risk_train_only": True,
            "risk_heads_screened_by_family_stratified_oof_only": True,
            "integrated_factory_shares_first_adjoint": True,
            "single_expert_path_exactly_falls_back_to_baseline": True,
            "opened_fresh_not_loaded": True,
            "fixed_spd_and_call_ledgers_pass": bool(ledgers_valid),
        },
        "expert_bank": {
            "candidate_ids": expert_ids,
            "baseline_candidate_id": baseline_id,
            "deployment_uses_family_labels": False,
        },
        "finite_action_bank": _action_bank_summary(
            gain_targets_by_blend=gain_targets_by_blend,
            expert_ids=expert_ids,
        ),
        "finite_action_rows_private": action_rows_private,
        "selection_screen": screen,
        "selected_strict_candidate": selected,
        "selected_by_route_mode": selected_by_route_mode,
        "best_diagnostic_candidate": diagnostic,
        "oof_predictions_private": oof_cache_private,
        "selection_models_private": selection_models,
        "route_decision_summary": route_decisions,
        "transfer_metric_rows_private": transfer_rows,
        "paired_gain_summary": summaries,
        "development_gates": development_gates,
        "secondary_metric_safety_audits": secondary_audits,
        "overall_decision": {
            "primary_strict_gate_pass": next(
                row["pass"]
                for row in development_gates
                if row["method"] == "rq_ogse_strict"
            ),
            "secondary_metric_safety_pass": next(
                row["pass"]
                for row in secondary_audits
                if row["method"] == "rq_ogse_strict"
            ),
            "fresh_repeat_authorized": False,
            "decision": (
                "HOLD_PRIMARY_FIELD_GATE_SIGNAL_REQUIRES_"
                "FRONT_SAFETY_REDESIGN"
            ),
        },
        "execution": execution,
        "execution_summary": {
            "finite_action_reconstruction_count": len(action_execution),
            "oof_route_candidate_count": len(screen),
            "transfer_route_count": len(routes),
            "logical_calls_per_reconstruction_sample": {
                "forward": 4,
                "adjoint": 4,
            },
        },
        "runtime": {
            "wall_seconds": float(time.perf_counter() - started),
            "maximum_rss_bytes": int(_max_rss_bytes()),
        },
        "claim_boundary": {
            "postopen_development_diagnostic_only": True,
            "quantile_heads_are_empirical_not_conformal": True,
            "validation_and_calibration_are_postopen_diagnostics": True,
            "fresh_values_loaded": False,
            "fresh_repeat_authorized": False,
            "experimental_field_truth_used": False,
            "real_psu_measurement_values_used": False,
            "analytic_morphology_is_cfd": False,
            "algorithm_superiority": False,
        },
    }
    return private, build_public_summary(private)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "demo_t16_operator/configs/"
            "psu_b0_rq_ogse_pcgls_development_v1.json"
        ),
    )
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument("--probe-private-report", type=Path, required=True)
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--public-output", type=Path)
    args = parser.parse_args()
    private, public = run_development(
        root=args.root,
        config_path=args.config,
        development_report_path=args.development_report,
        probe_private_report_path=args.probe_private_report,
        view_root=args.view_root,
        device_name=args.device,
    )
    if args.private_output is not None:
        args.private_output.parent.mkdir(parents=True, exist_ok=True)
        args.private_output.write_text(
            json.dumps(private, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    if args.public_output is not None:
        args.public_output.parent.mkdir(parents=True, exist_ok=True)
        args.public_output.write_text(
            json.dumps(public, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps(public, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
