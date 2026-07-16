#!/usr/bin/env python3
"""Screen and transfer a shared-adjoint observable morphology PCGLS mixture."""

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
import torch

from demo_t16_operator.psu_b0_classical_baselines import (
    GeneralizedSobolevDirection,
    preconditioned_cgls_reconstruction,
)
from demo_t16_operator.psu_b0_morphology_spectral_experts import (
    MORPHOLOGY_EXPERT_SCHEMA,
    ObservableMorphologyExpertFactory,
    materialize_log_expert_mixture,
)
from demo_t16_operator.psu_b0_spectral_preconditioner import (
    normalized_field_loss,
)
from demo_t16_operator.psu_b0_streaming_operator import (
    zero_outer_boundary_support,
)
from site_tools.run_psu_b0_conditioned_pcgls_development import (
    paired_gain_summary,
)
from site_tools.run_psu_b0_classical_frontier_development import (
    _verify_split_metadata,
)
from site_tools.run_psu_b0_observable_morphology_probe import (
    fit_ridge_classifier,
    fit_ridge_multioutput,
    ridge_scores,
    score_predictions,
    stratified_folds,
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


PRIVATE_SCHEMA = "psu-b0-omse-pcgls-development-private-1.0"
PUBLIC_SCHEMA = "psu-b0-omse-pcgls-development-public-1.0"
STATUS = "OMSE_PCGLS_DEVELOPMENT_COMPLETE_FRESH_NOT_USED"


def _max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _oof_scores(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    fold_count: int,
    regularization: float,
    class_count: int,
) -> np.ndarray:
    folds = stratified_folds(labels, fold_count=int(fold_count))
    output = np.zeros((len(features), int(class_count)), dtype=np.float64)
    for fold in range(int(fold_count)):
        train = folds != fold
        holdout = folds == fold
        model = fit_ridge_classifier(
            features[train],
            labels[train],
            class_count=int(class_count),
            regularization=float(regularization),
        )
        output[holdout] = ridge_scores(model, features[holdout])
    return output


def _oof_regression_scores(
    features: np.ndarray,
    targets: np.ndarray,
    labels_for_folds: np.ndarray,
    *,
    fold_count: int,
    regularization: float,
) -> np.ndarray:
    folds = stratified_folds(
        labels_for_folds,
        fold_count=int(fold_count),
    )
    output = np.zeros_like(targets, dtype=np.float64)
    for fold in range(int(fold_count)):
        train = folds != fold
        holdout = folds == fold
        model = fit_ridge_multioutput(
            features[train],
            targets[train],
            regularization=float(regularization),
        )
        output[holdout] = ridge_scores(model, features[holdout])
    return output


def _expert_gain_targets(
    *,
    sample_ids: list[str],
    candidate_rows: list[dict[str, Any]],
    expert_ids: list[str],
    baseline_candidate_id: str,
) -> np.ndarray:
    lookup = {
        (str(row["sample_id"]), str(row["candidate_id"])): row
        for row in candidate_rows
    }
    output = np.zeros(
        (len(sample_ids), len(expert_ids)),
        dtype=np.float64,
    )
    for sample_index, sample_id in enumerate(sample_ids):
        baseline = float(
            lookup[(sample_id, baseline_candidate_id)][
                "field_relative_l2"
            ]
        )
        for expert_index, candidate_id in enumerate(expert_ids):
            candidate = float(
                lookup[(sample_id, candidate_id)][
                    "field_relative_l2"
                ]
            )
            output[sample_index, expert_index] = (
                100.0
                * (baseline - candidate)
                / max(baseline, 1e-12)
            )
    return output


def _evaluate_mixture_scores(
    *,
    method: str,
    wrapped: DevelopmentSplit,
    operator: Any,
    source_config: dict[str, Any],
    device: torch.device,
    scores: np.ndarray,
    expert_log_gains: torch.Tensor,
    baseline_expert_index: int,
    temperature: float,
    confidence_threshold: float,
    maximum_blend: float,
    batch_size: int = 12,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split = wrapped.data
    if np.asarray(scores).shape != (
        len(split.truth),
        len(expert_log_gains),
    ):
        raise ValueError("selector scores do not align with the split")
    rays_per_view = int(source_config["geometry"]["rays_per_view"])
    rows = []
    monotone = True
    gain_minimum = float("inf")
    gain_maximum = 0.0
    geometric_defect = 0.0
    started = time.perf_counter()
    operator.reset_call_counts()
    score_tensor = torch.as_tensor(scores, dtype=torch.float32)
    for start in range(0, len(split.truth), int(batch_size)):
        stop = min(start + int(batch_size), len(split.truth))
        truth = split.truth[start:stop].to(device)
        observation = split.observation_uv[start:stop].to(device)
        sigma = split.sigma_by_view[start:stop].to(device)
        mask = split.view_mask[start:stop].to(device)
        batch_scores = score_tensor[start:stop].to(device)

        def factory(_: torch.Tensor, **__: Any) -> Any:
            return materialize_log_expert_mixture(
                batch_scores,
                expert_log_gains=expert_log_gains,
                baseline_expert_index=int(baseline_expert_index),
                temperature=float(temperature),
                confidence_threshold=float(confidence_threshold),
                maximum_blend=float(maximum_blend),
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


def _evaluate_integrated_factory(
    *,
    method: str,
    wrapped: DevelopmentSplit,
    operator: Any,
    source_config: dict[str, Any],
    device: torch.device,
    factory: ObservableMorphologyExpertFactory,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split = wrapped.data
    rays_per_view = int(source_config["geometry"]["rays_per_view"])
    rows = []
    monotone = True
    gain_minimum = float("inf")
    gain_maximum = 0.0
    geometric_defect = 0.0
    started = time.perf_counter()
    operator.reset_call_counts()
    for start in range(0, len(split.truth), 12):
        stop = min(start + 12, len(split.truth))
        truth = split.truth[start:stop].to(device)
        observation = split.observation_uv[start:stop].to(device)
        sigma = split.sigma_by_view[start:stop].to(device)
        mask = split.view_mask[start:stop].to(device)
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


def _paired_screen_metrics(
    candidate_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    *,
    scores: np.ndarray,
    margins: np.ndarray,
    confidence_threshold: float,
    baseline_expert_index: int,
) -> dict[str, Any]:
    baseline = {
        str(row["sample_id"]): float(row["field_relative_l2"])
        for row in baseline_rows
    }
    candidate = {
        str(row["sample_id"]): float(row["field_relative_l2"])
        for row in candidate_rows
    }
    sample_ids = sorted(baseline)
    gain = np.asarray(
        [
            100.0
            * (baseline[sample] - candidate[sample])
            / max(baseline[sample], 1e-12)
            for sample in sample_ids
        ],
        dtype=np.float64,
    )
    top = np.argmax(np.asarray(scores, dtype=np.float64), axis=1)
    accepted = (
        np.asarray(margins, dtype=np.float64)
        >= float(confidence_threshold)
    ) & (top != int(baseline_expert_index))
    return {
        "sample_count": len(sample_ids),
        "mean_field_gain_percent": float(np.mean(gain)),
        "p10_field_gain_percent": float(np.quantile(gain, 0.10)),
        "minimum_field_gain_percent": float(np.min(gain)),
        "harm_over_one_percent_rate": float(np.mean(gain < -1.0)),
        "coverage": float(np.mean(accepted)),
        "accepted_harm_over_one_percent_rate": (
            0.0
            if not np.any(accepted)
            else float(np.mean(gain[accepted] < -1.0))
        ),
    }


def select_omse_screen_candidate(
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
            "selection_status": "NO_STRICT_OOF_MIXTURE_FEASIBLE",
            "strict_gate_pass": False,
        }
    selected = max(
        feasible,
        key=lambda row: (
            float(row["mean_field_gain_percent"]),
            float(row["p10_field_gain_percent"]),
            float(row["coverage"]),
            -float(row["maximum_blend"]),
            -float(row["temperature"]),
        ),
    )
    output = dict(selected)
    output["selection_status"] = "STRICT_OOF_MIXTURE_SELECTED"
    output["strict_gate_pass"] = True
    return output


def _best_diagnostic(screen: list[dict[str, Any]]) -> dict[str, Any]:
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


def _expert_bank(
    *,
    headroom_public: dict[str, Any],
    expert_ids: list[str],
    grid_size: int,
    device: torch.device,
) -> torch.Tensor:
    candidate_lookup = {
        str(row["candidate_id"]): row
        for row in headroom_public["candidate_grid"]["candidates"]
    }
    logs = []
    for candidate_id in expert_ids:
        parameters = candidate_lookup[candidate_id]
        direction = GeneralizedSobolevDirection(
            (grid_size,) * 3,
            strength=float(parameters["strength"]),
            epsilon=float(parameters["epsilon"]),
            axis_weights_xyz=tuple(
                float(value)
                for value in parameters["axis_weights_xyz"]
            ),
        ).to(device)
        logs.append(torch.log(direction.gain.clamp_min(1e-20)))
    return torch.stack(logs, dim=0)


def _development_gate(
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
    }
    return {
        "method": method,
        "checks": checks,
        "pass": all(checks.values()),
    }


def build_public_summary(private: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": PUBLIC_SCHEMA,
        "algorithm_schema": MORPHOLOGY_EXPERT_SCHEMA,
        "status": private["status"],
        "evidence_scope": private["evidence_scope"],
        "configuration": copy.deepcopy(private["configuration_public"]),
        "regeneration_checks": copy.deepcopy(
            private["regeneration_checks"]
        ),
        "expert_bank": copy.deepcopy(private["expert_bank"]),
        "selection_screen": copy.deepcopy(private["selection_screen"]),
        "selected_strict_candidate": copy.deepcopy(
            private["selected_strict_candidate"]
        ),
        "best_diagnostic_candidate": copy.deepcopy(
            private["best_diagnostic_candidate"]
        ),
        "paired_gain_summary": copy.deepcopy(
            private["paired_gain_summary"]
        ),
        "development_gates": copy.deepcopy(
            private["development_gates"]
        ),
        "execution": copy.deepcopy(private["execution"]),
        "runtime": copy.deepcopy(private["runtime"]),
        "claim_boundary": copy.deepcopy(private["claim_boundary"]),
    }


def run_development(
    *,
    root: Path,
    config_path: Path,
    development_report_path: Path,
    probe_private_report_path: Path,
    headroom_private_report_path: Path,
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
    headroom_public = _load_json(
        root / str(config["source_headroom_public_summary"])
    )
    probe_public = _load_json(
        root / str(config["source_probe_public_summary"])
    )
    probe_private = _load_json(probe_private_report_path)
    headroom_private = _load_json(headroom_private_report_path)
    development_report = _load_json(development_report_path)
    expert_ids = [
        str(value)
        for value in probe_public["expert_bank"]["class_candidates"]
    ]
    baseline_id = str(
        config["solver_contract"]["baseline_candidate_id"]
    )
    baseline_index = expert_ids.index(baseline_id)
    family_map = probe_public["expert_bank"][
        "family_to_expert_non_deployable"
    ]
    class_index = {
        candidate_id: index
        for index, candidate_id in enumerate(expert_ids)
    }
    features = np.asarray(
        probe_private["features_private"]["risk_train"][
            "initial_normal_spectrum"
        ],
        dtype=np.float64,
    )
    train_families = development_config["development_splits"]["risk_train"][
        "families"
    ]
    train_count = int(
        development_config["development_splits"]["risk_train"]["count"]
    )
    labels = np.asarray(
        [
            class_index[
                str(
                    family_map[
                        str(train_families[index % len(train_families)])
                    ]
                )
            ]
            for index in range(train_count)
        ],
        dtype=np.int64,
    )
    train_sample_ids = [
        f"risk_train-{index:03d}" for index in range(train_count)
    ]
    gain_targets = _expert_gain_targets(
        sample_ids=train_sample_ids,
        candidate_rows=headroom_private[
            "candidate_metric_rows_private"
        ]["risk_train"],
        expert_ids=expert_ids,
        baseline_candidate_id=baseline_id,
    )

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
        headroom_public=headroom_public,
        expert_ids=expert_ids,
        grid_size=grid_size,
        device=device,
    )
    baseline_direction = GeneralizedSobolevDirection(
        (grid_size,) * 3,
        strength=4.0,
        epsilon=0.05,
    ).to(device)
    baseline_rows, baseline_ledger = _evaluate_mixture_scores(
        method="static_pcgls4",
        wrapped=splits["risk_train"],
        operator=nominal_operator,
        source_config=source_config,
        device=device,
        scores=np.tile(
            np.eye(len(expert_ids))[baseline_index],
            (train_count, 1),
        ),
        expert_log_gains=expert_logs,
        baseline_expert_index=baseline_index,
        temperature=1.0,
        confidence_threshold=1e30,
        maximum_blend=0.0,
    )
    del baseline_direction
    execution = [baseline_ledger]
    screen = []
    screen_config = config["selector_screen"]
    for selector_type in screen_config["selector_types"]:
        for regularization in screen_config["ridge_lambda_grid"]:
            if selector_type == "family_classification":
                scores = _oof_scores(
                    features,
                    labels,
                    fold_count=int(screen_config["folds"]),
                    regularization=float(regularization),
                    class_count=len(expert_ids),
                )
            elif selector_type == "expert_gain_regression":
                scores = _oof_regression_scores(
                    features,
                    gain_targets,
                    labels,
                    fold_count=int(screen_config["folds"]),
                    regularization=float(regularization),
                )
            else:
                raise ValueError(
                    f"unknown selector type: {selector_type}"
                )
            _, margins = score_predictions(scores)
            margin_quantiles = [
                float(np.quantile(margins, float(value)))
                for value in screen_config["confidence_margin_quantiles"]
            ]
            for temperature in screen_config["temperature_grid"]:
                for maximum_blend in screen_config["maximum_blend_grid"]:
                    for confidence_threshold in margin_quantiles:
                        method = (
                            f"screen_{selector_type}_"
                            f"l{float(regularization):g}_"
                            f"t{float(temperature):g}_"
                            f"b{float(maximum_blend):g}_"
                            f"q{margin_quantiles.index(confidence_threshold)}"
                        )
                        rows, ledger = _evaluate_mixture_scores(
                            method=method,
                            wrapped=splits["risk_train"],
                            operator=nominal_operator,
                            source_config=source_config,
                            device=device,
                            scores=scores,
                            expert_log_gains=expert_logs,
                            baseline_expert_index=baseline_index,
                            temperature=float(temperature),
                            confidence_threshold=float(confidence_threshold),
                            maximum_blend=float(maximum_blend),
                        )
                        metrics = _paired_screen_metrics(
                            rows,
                            baseline_rows,
                            scores=scores,
                            margins=margins,
                            confidence_threshold=float(
                                confidence_threshold
                            ),
                            baseline_expert_index=baseline_index,
                        )
                        screen.append(
                            {
                                "selector_type": str(selector_type),
                                "regularization": float(regularization),
                                "temperature": float(temperature),
                                "maximum_blend": float(maximum_blend),
                                "confidence_threshold": float(
                                    confidence_threshold
                                ),
                                "confidence_quantile": int(
                                    margin_quantiles.index(
                                        confidence_threshold
                                    )
                                ),
                                **metrics,
                            }
                        )
                        execution.append(ledger)
    selected = select_omse_screen_candidate(
        screen,
        gate=config["strict_oof_gate"],
    )
    diagnostic = _best_diagnostic(screen)

    transfer_rows = []
    transfer_execution = []
    selection_models = {}
    for route_name, route in (
        ("strict", selected),
        ("diagnostic", diagnostic),
    ):
        if not bool(route.get("strict_gate_pass")) and route_name == "strict":
            route_parameters = {
                "selector_type": "expert_gain_regression",
                "regularization": 0.1,
                "temperature": 1.0,
                "maximum_blend": 0.0,
                "confidence_threshold": 1e30,
            }
        else:
            route_parameters = route
        if route_parameters["selector_type"] == "family_classification":
            ridge = fit_ridge_classifier(
                features,
                labels,
                class_count=len(expert_ids),
                regularization=float(route_parameters["regularization"]),
            )
        elif route_parameters["selector_type"] == "expert_gain_regression":
            ridge = fit_ridge_multioutput(
                features,
                gain_targets,
                regularization=float(route_parameters["regularization"]),
            )
        else:
            raise ValueError("selected route has an unknown selector type")
        factory = ObservableMorphologyExpertFactory(
            expert_log_gains=expert_logs,
            expert_candidate_ids=expert_ids,
            baseline_expert_index=baseline_index,
            feature_mean=torch.as_tensor(ridge["mean"]),
            feature_scale=torch.as_tensor(ridge["scale"]),
            ridge_weights=torch.as_tensor(ridge["weights"]),
            temperature=float(route_parameters["temperature"]),
            confidence_threshold=float(
                route_parameters["confidence_threshold"]
            ),
            maximum_blend=float(route_parameters["maximum_blend"]),
        ).to(device)
        selection_models[route_name] = {
            "parameters": copy.deepcopy(route_parameters),
            "ridge_mean_private": np.asarray(ridge["mean"]).tolist(),
            "ridge_scale_private": np.asarray(ridge["scale"]).tolist(),
            "ridge_weights_private": np.asarray(ridge["weights"]).tolist(),
        }
        for split_name in ("risk_validation", "risk_calibration"):
            rows, ledger = _evaluate_integrated_factory(
                method=f"omse_{route_name}",
                wrapped=splits[split_name],
                operator=nominal_operator,
                source_config=source_config,
                device=device,
                factory=factory,
            )
            transfer_rows.extend(rows)
            transfer_execution.append(ledger)

    for split_name in ("risk_validation", "risk_calibration"):
        rows, ledger = _evaluate_mixture_scores(
            method="static_pcgls4",
            wrapped=splits[split_name],
            operator=nominal_operator,
            source_config=source_config,
            device=device,
            scores=np.tile(
                np.eye(len(expert_ids))[baseline_index],
                (len(splits[split_name].data.truth), 1),
            ),
            expert_log_gains=expert_logs,
            baseline_expert_index=baseline_index,
            temperature=1.0,
            confidence_threshold=1e30,
            maximum_blend=0.0,
        )
        transfer_rows.extend(rows)
        transfer_execution.append(ledger)
    summaries = []
    for method_index, method in enumerate(
        ("omse_strict", "omse_diagnostic")
    ):
        for split_index, split_name in enumerate(
            ("risk_validation", "risk_calibration")
        ):
            summaries.append(
                paired_gain_summary(
                    transfer_rows,
                    split=split_name,
                    candidate_method=method,
                    bootstrap_seed=20263100
                    + 100 * method_index
                    + split_index,
                )
            )
    development_gates = [
        _development_gate(
            summaries,
            method=method,
            gate=config["development_gate"],
        )
        for method in ("omse_strict", "omse_diagnostic")
    ]
    execution.extend(transfer_execution)
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
        "algorithm_schema": MORPHOLOGY_EXPERT_SCHEMA,
        "status": STATUS,
        "evidence_scope": (
            "REAL_PSU_SUPPORT_GEOMETRY_WITH_ANALYTIC_REACTION_MORPHOLOGY_"
            "AND_SYNTHETIC_CAMERA_NOISE_POSTOPEN_OMSE_DEVELOPMENT_ONLY"
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
            "headroom_private_report_path": str(
                headroom_private_report_path.resolve()
            ),
            "view_root": str(view_root.resolve()),
            "device": str(device),
        },
        "configuration_public": copy.deepcopy(config),
        "regeneration_checks": {
            "development_metadata_matches_frozen_rows": True,
            "expert_bank_matches_headroom_and_probe": True,
            "oof_screen_uses_risk_train_only": True,
            "integrated_factory_shares_first_adjoint": True,
            "opened_fresh_not_loaded": True,
            "fixed_spd_and_call_ledgers_pass": bool(ledgers_valid),
        },
        "expert_bank": {
            "candidate_ids": expert_ids,
            "baseline_candidate_id": baseline_id,
            "baseline_expert_index": baseline_index,
            "family_label_source_is_non_deployable": True,
        },
        "selection_screen": screen,
        "selected_strict_candidate": selected,
        "best_diagnostic_candidate": diagnostic,
        "selection_models_private": selection_models,
        "transfer_metric_rows_private": transfer_rows,
        "paired_gain_summary": summaries,
        "development_gates": development_gates,
        "execution": execution,
        "runtime": {
            "wall_seconds": float(time.perf_counter() - started),
            "maximum_rss_bytes": int(_max_rss_bytes()),
            "screen_candidate_count": len(screen),
        },
        "claim_boundary": {
            "postopen_development_diagnostic_only": True,
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
            "psu_b0_omse_pcgls_development_v1.json"
        ),
    )
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument("--probe-private-report", type=Path, required=True)
    parser.add_argument("--headroom-private-report", type=Path, required=True)
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
        headroom_private_report_path=args.headroom_private_report,
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
