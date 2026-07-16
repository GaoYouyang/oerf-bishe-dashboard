#!/usr/bin/env python3
"""Develop field-and-front risk routing for single-expert SPD PCGLS."""

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
from scipy.special import expit
import torch

from demo_t16_operator.psu_b0_risk_quantile_experts import (
    MultiObjectiveRiskSingleExpertFactory,
    RISK_QUANTILE_EXPERT_SCHEMA,
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
)
from site_tools.run_psu_b0_rq_ogse_pcgls_development import (
    _oof_model_scores,
    _risk_development_gate,
    _secondary_metric_safety_audit,
    action_gain_targets,
    fit_l1_quantile_multioutput,
    fit_ridge_logistic_multioutput,
)
from site_tools.run_psu_b0_spectral_preconditioner_pilot import (
    _load_json,
)


PRIVATE_SCHEMA = "psu-b0-mo-rq-ogse-pcgls-development-private-1.0"
PUBLIC_SCHEMA = "psu-b0-mo-rq-ogse-pcgls-development-public-1.0"
STATUS = "MO_RQ_OGSE_PCGLS_DEVELOPMENT_COMPLETE_FRESH_NOT_USED"


def _max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def front_delta_targets(
    *,
    sample_ids: list[str],
    baseline_rows: list[dict[str, Any]],
    action_rows_by_expert: dict[int, list[dict[str, Any]]],
    expert_count: int,
    baseline_expert_index: int,
) -> np.ndarray:
    """Return absolute top-10% front-F1 deltas for each finite action."""

    baseline = {
        str(row["sample_id"]): float(row["front_top10_f1"])
        for row in baseline_rows
    }
    output = np.zeros(
        (len(sample_ids), int(expert_count)),
        dtype=np.float64,
    )
    for expert_index, rows in action_rows_by_expert.items():
        candidate = {
            str(row["sample_id"]): float(row["front_top10_f1"])
            for row in rows
        }
        for sample_index, sample_id in enumerate(sample_ids):
            output[sample_index, int(expert_index)] = (
                candidate[sample_id] - baseline[sample_id]
            )
    output[:, int(baseline_expert_index)] = 0.0
    return output


def numpy_multiobjective_actions(
    *,
    field_mean_scores: np.ndarray,
    field_lower_scores: np.ndarray,
    field_harm_logits: np.ndarray,
    front_lower_scores: np.ndarray,
    front_harm_logits: np.ndarray,
    baseline_expert_index: int,
    minimum_field_score: float,
    maximum_field_harm_probability: float,
    minimum_front_lower_delta: float,
    maximum_front_harm_probability: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    matrices = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (
            field_mean_scores,
            field_lower_scores,
            field_harm_logits,
            front_lower_scores,
            front_harm_logits,
        )
    )
    field_mean, field_lower, field_harm_logit, front_lower, front_harm_logit = (
        matrices
    )
    if field_mean.ndim != 2 or any(
        value.shape != field_mean.shape for value in matrices[1:]
    ):
        raise ValueError("multiobjective score matrices must align")
    selected = np.argmax(field_mean, axis=1)
    row = np.arange(len(selected))
    field_harm = expit(field_harm_logit[row, selected])
    front_harm = expit(front_harm_logit[row, selected])
    accepted = (
        (selected != int(baseline_expert_index))
        & (
            field_mean[row, selected]
            >= float(minimum_field_score)
        )
        & (field_lower[row, selected] >= 0.0)
        & (
            field_harm
            <= float(maximum_field_harm_probability)
        )
        & (
            front_lower[row, selected]
            >= float(minimum_front_lower_delta)
        )
        & (
            front_harm
            <= float(maximum_front_harm_probability)
        )
    )
    actions = np.where(
        accepted,
        selected,
        int(baseline_expert_index),
    ).astype(np.int64)
    return actions, accepted, field_harm, front_harm


def multiobjective_route_metrics(
    *,
    field_gain_targets: np.ndarray,
    front_delta_values: np.ndarray,
    actions: np.ndarray,
    accepted: np.ndarray,
    field_harm_threshold: float,
    front_harm_threshold: float,
) -> dict[str, Any]:
    field = np.asarray(field_gain_targets, dtype=np.float64)
    front = np.asarray(front_delta_values, dtype=np.float64)
    selected = np.asarray(actions, dtype=np.int64).reshape(-1)
    acceptance = np.asarray(accepted, dtype=bool).reshape(-1)
    if field.shape != front.shape or len(field) != len(selected):
        raise ValueError("multiobjective targets and actions must align")
    row = np.arange(len(selected))
    field_gain = field[row, selected]
    front_delta = front[row, selected]
    return {
        "sample_count": len(selected),
        "coverage": float(np.mean(acceptance)),
        "mean_field_gain_percent": float(np.mean(field_gain)),
        "p10_field_gain_percent": float(np.quantile(field_gain, 0.10)),
        "minimum_field_gain_percent": float(np.min(field_gain)),
        "field_harm_rate": float(
            np.mean(field_gain < float(field_harm_threshold))
        ),
        "accepted_field_harm_rate": (
            0.0
            if not np.any(acceptance)
            else float(
                np.mean(
                    field_gain[acceptance]
                    < float(field_harm_threshold)
                )
            )
        ),
        "mean_front_f1_absolute_delta": float(np.mean(front_delta)),
        "p10_front_f1_absolute_delta": float(
            np.quantile(front_delta, 0.10)
        ),
        "minimum_front_f1_absolute_delta": float(np.min(front_delta)),
        "front_harm_rate": float(
            np.mean(front_delta < float(front_harm_threshold))
        ),
        "accepted_front_harm_rate": (
            0.0
            if not np.any(acceptance)
            else float(
                np.mean(
                    front_delta[acceptance]
                    < float(front_harm_threshold)
                )
            )
        ),
    }


def select_multiobjective_candidate(
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
        and float(row["field_harm_rate"])
        <= float(gate["maximum_field_harm_rate"])
        and float(row["accepted_field_harm_rate"])
        <= float(gate["maximum_accepted_field_harm_rate"])
        and float(row["mean_front_f1_absolute_delta"])
        >= float(gate["minimum_mean_front_f1_absolute_delta"])
        and float(row["p10_front_f1_absolute_delta"])
        >= float(gate["minimum_p10_front_f1_absolute_delta"])
        and float(row["front_harm_rate"])
        <= float(gate["maximum_front_harm_rate"])
        and float(row["accepted_front_harm_rate"])
        <= float(gate["maximum_accepted_front_harm_rate"])
    ]
    if not feasible:
        return {
            "selection_status": "NO_STRICT_OOF_MULTI_OBJECTIVE_ROUTE",
            "strict_gate_pass": False,
        }
    selected = max(
        feasible,
        key=lambda row: (
            float(row["mean_field_gain_percent"]),
            float(row["mean_front_f1_absolute_delta"]),
            float(row["coverage"]),
            -float(row["interpolation_fraction"]),
        ),
    )
    output = dict(selected)
    output["selection_status"] = "STRICT_OOF_MULTI_OBJECTIVE_ROUTE_SELECTED"
    output["strict_gate_pass"] = True
    return output


def _best_diagnostic(screen: list[dict[str, Any]]) -> dict[str, Any]:
    selected = max(
        screen,
        key=lambda row: (
            float(row["mean_field_gain_percent"]),
            float(row["mean_front_f1_absolute_delta"]),
            -float(row["field_harm_rate"]),
            -float(row["front_harm_rate"]),
        ),
    )
    output = dict(selected)
    output["selection_status"] = "BEST_OOF_MULTI_OBJECTIVE_DIAGNOSTIC"
    output["strict_gate_pass"] = False
    return output


def _fit_models(
    *,
    features: np.ndarray,
    field_targets: np.ndarray,
    front_targets: np.ndarray,
    parameters: dict[str, Any],
    field_harm_threshold: float,
    front_harm_threshold: float,
) -> dict[str, Any]:
    models = {
        "field_mean": fit_ridge_multioutput(
            features,
            field_targets,
            regularization=float(parameters["field_mean_regularization"]),
        ),
        "field_lower": fit_l1_quantile_multioutput(
            features,
            field_targets,
            quantile=float(parameters["field_lower_quantile"]),
            regularization=float(
                parameters["field_quantile_regularization"]
            ),
        ),
        "field_harm": fit_ridge_logistic_multioutput(
            features,
            (
                field_targets < float(field_harm_threshold)
            ).astype(np.float64),
            regularization=float(
                parameters["field_harm_regularization"]
            ),
        ),
        "front_lower": fit_l1_quantile_multioutput(
            features,
            front_targets,
            quantile=float(parameters["front_lower_quantile"]),
            regularization=float(
                parameters["front_quantile_regularization"]
            ),
        ),
        "front_harm": fit_ridge_logistic_multioutput(
            features,
            (
                front_targets < float(front_harm_threshold)
            ).astype(np.float64),
            regularization=float(
                parameters["front_harm_regularization"]
            ),
        ),
    }
    reference = models["field_mean"]
    for model in models.values():
        if not np.allclose(model["mean"], reference["mean"]) or not np.allclose(
            model["scale"],
            reference["scale"],
        ):
            raise RuntimeError("multiobjective heads disagree on normalization")
    return models


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
        "selection_screen": copy.deepcopy(private["selection_screen"]),
        "selected_strict_candidate": copy.deepcopy(
            private["selected_strict_candidate"]
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
    rq_private_report_path: Path,
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
    probe_private = _load_json(probe_private_report_path)
    rq_private = _load_json(rq_private_report_path)
    development_report = _load_json(development_report_path)
    expert_ids = [
        str(value)
        for value in rq_private["expert_bank"]["candidate_ids"]
    ]
    baseline_id = str(rq_private["expert_bank"]["baseline_candidate_id"])
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
    sample_ids = [f"risk_train-{index:03d}" for index in range(train_count)]

    field_targets_by_blend = {}
    front_targets_by_blend = {}
    for blend_text, raw_rows_by_expert in rq_private[
        "finite_action_rows_private"
    ].items():
        rows_by_expert = {
            int(index): rows
            for index, rows in raw_rows_by_expert.items()
        }
        baseline_rows = rows_by_expert[baseline_index]
        blend = float(blend_text)
        field_targets_by_blend[blend] = action_gain_targets(
            sample_ids=sample_ids,
            baseline_rows=baseline_rows,
            action_rows_by_expert=rows_by_expert,
            expert_count=len(expert_ids),
            baseline_expert_index=baseline_index,
        )
        front_targets_by_blend[blend] = front_delta_targets(
            sample_ids=sample_ids,
            baseline_rows=baseline_rows,
            action_rows_by_expert=rows_by_expert,
            expert_count=len(expert_ids),
            baseline_expert_index=baseline_index,
        )

    screen_config = config["multiobjective_screen"]
    fold_count = int(screen_config["folds"])
    field_harm_threshold = float(
        screen_config["field_harm_gain_threshold_percent"]
    )
    front_harm_threshold = float(
        screen_config["front_harm_absolute_delta_threshold"]
    )
    screen = []
    oof_cache_private = {}
    for blend in [
        float(value)
        for value in screen_config["interpolation_fractions"]
    ]:
        field_targets = field_targets_by_blend[blend]
        front_targets = front_targets_by_blend[blend]
        field_mean_cache = {
            float(regularization): _oof_model_scores(
                train_features,
                field_targets,
                fold_labels,
                fold_count=fold_count,
                fit_kind="ridge",
                regularization=float(regularization),
            )
            for regularization in screen_config[
                "field_mean_regularization_grid"
            ]
        }
        field_lower_cache = {
            (float(quantile), float(regularization)): _oof_model_scores(
                train_features,
                field_targets,
                fold_labels,
                fold_count=fold_count,
                fit_kind="quantile",
                regularization=float(regularization),
                quantile=float(quantile),
            )
            for quantile in screen_config["field_lower_quantile_grid"]
            for regularization in screen_config[
                "field_quantile_regularization_grid"
            ]
        }
        field_harm_cache = {
            float(regularization): _oof_model_scores(
                train_features,
                (
                    field_targets < field_harm_threshold
                ).astype(np.float64),
                fold_labels,
                fold_count=fold_count,
                fit_kind="logistic",
                regularization=float(regularization),
            )
            for regularization in screen_config[
                "field_harm_regularization_grid"
            ]
        }
        front_lower_cache = {
            (float(quantile), float(regularization)): _oof_model_scores(
                train_features,
                front_targets,
                fold_labels,
                fold_count=fold_count,
                fit_kind="quantile",
                regularization=float(regularization),
                quantile=float(quantile),
            )
            for quantile in screen_config["front_lower_quantile_grid"]
            for regularization in screen_config[
                "front_quantile_regularization_grid"
            ]
        }
        front_harm_cache = {
            float(regularization): _oof_model_scores(
                train_features,
                (
                    front_targets < front_harm_threshold
                ).astype(np.float64),
                fold_labels,
                fold_count=fold_count,
                fit_kind="logistic",
                regularization=float(regularization),
            )
            for regularization in screen_config[
                "front_harm_regularization_grid"
            ]
        }
        oof_cache_private[str(blend)] = {
            "field_mean": {
                str(key): value.tolist()
                for key, value in field_mean_cache.items()
            },
            "field_lower": {
                f"{key[0]}:{key[1]}": value.tolist()
                for key, value in field_lower_cache.items()
            },
            "field_harm": {
                str(key): value.tolist()
                for key, value in field_harm_cache.items()
            },
            "front_lower": {
                f"{key[0]}:{key[1]}": value.tolist()
                for key, value in front_lower_cache.items()
            },
            "front_harm": {
                str(key): value.tolist()
                for key, value in front_harm_cache.items()
            },
        }
        for field_mean_regularization, field_mean_scores in (
            field_mean_cache.items()
        ):
            for field_lower_key, field_lower_scores in (
                field_lower_cache.items()
            ):
                for field_harm_regularization, field_harm_logits in (
                    field_harm_cache.items()
                ):
                    for front_lower_key, front_lower_scores in (
                        front_lower_cache.items()
                    ):
                        for front_harm_regularization, front_harm_logits in (
                            front_harm_cache.items()
                        ):
                            for minimum_field_score in screen_config[
                                "minimum_field_score_grid"
                            ]:
                                for maximum_field_harm in screen_config[
                                    "maximum_field_harm_probability_grid"
                                ]:
                                    for minimum_front_lower in screen_config[
                                        "minimum_front_lower_delta_grid"
                                    ]:
                                        for maximum_front_harm in (
                                            screen_config[
                                                "maximum_front_harm_"
                                                "probability_grid"
                                            ]
                                        ):
                                            (
                                                actions,
                                                accepted,
                                                _,
                                                _,
                                            ) = (
                                                numpy_multiobjective_actions(
                                                    field_mean_scores=(
                                                        field_mean_scores
                                                    ),
                                                    field_lower_scores=(
                                                        field_lower_scores
                                                    ),
                                                    field_harm_logits=(
                                                        field_harm_logits
                                                    ),
                                                    front_lower_scores=(
                                                        front_lower_scores
                                                    ),
                                                    front_harm_logits=(
                                                        front_harm_logits
                                                    ),
                                                    baseline_expert_index=(
                                                        baseline_index
                                                    ),
                                                    minimum_field_score=float(
                                                        minimum_field_score
                                                    ),
                                                    maximum_field_harm_probability=float(
                                                        maximum_field_harm
                                                    ),
                                                    minimum_front_lower_delta=float(
                                                        minimum_front_lower
                                                    ),
                                                    maximum_front_harm_probability=float(
                                                        maximum_front_harm
                                                    ),
                                                )
                                            )
                                            metrics = (
                                                multiobjective_route_metrics(
                                                    field_gain_targets=(
                                                        field_targets
                                                    ),
                                                    front_delta_values=(
                                                        front_targets
                                                    ),
                                                    actions=actions,
                                                    accepted=accepted,
                                                    field_harm_threshold=(
                                                        field_harm_threshold
                                                    ),
                                                    front_harm_threshold=(
                                                        front_harm_threshold
                                                    ),
                                                )
                                            )
                                            screen.append(
                                                {
                                                    "interpolation_fraction": blend,
                                                    "field_mean_regularization": float(
                                                        field_mean_regularization
                                                    ),
                                                    "field_lower_quantile": float(
                                                        field_lower_key[0]
                                                    ),
                                                    "field_quantile_regularization": float(
                                                        field_lower_key[1]
                                                    ),
                                                    "field_harm_regularization": float(
                                                        field_harm_regularization
                                                    ),
                                                    "front_lower_quantile": float(
                                                        front_lower_key[0]
                                                    ),
                                                    "front_quantile_regularization": float(
                                                        front_lower_key[1]
                                                    ),
                                                    "front_harm_regularization": float(
                                                        front_harm_regularization
                                                    ),
                                                    "minimum_field_score": float(
                                                        minimum_field_score
                                                    ),
                                                    "maximum_field_harm_probability": float(
                                                        maximum_field_harm
                                                    ),
                                                    "minimum_front_lower_delta": float(
                                                        minimum_front_lower
                                                    ),
                                                    "maximum_front_harm_probability": float(
                                                        maximum_front_harm
                                                    ),
                                                    **metrics,
                                                }
                                            )
    selected = select_multiobjective_candidate(
        screen,
        gate=config["strict_oof_gate"],
    )
    diagnostic = _best_diagnostic(screen)

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

    transfer_rows = []
    transfer_execution = []
    route_decisions = []
    selection_models = {}
    for route_name, route in (
        ("strict", selected),
        ("diagnostic", diagnostic),
    ):
        if route_name == "strict" and not bool(
            route.get("strict_gate_pass")
        ):
            parameters = {
                "interpolation_fraction": float(
                    screen_config["interpolation_fractions"][0]
                ),
                "field_mean_regularization": 1.0,
                "field_lower_quantile": 0.2,
                "field_quantile_regularization": 1.0,
                "field_harm_regularization": 1.0,
                "front_lower_quantile": 0.2,
                "front_quantile_regularization": 1.0,
                "front_harm_regularization": 1.0,
                "minimum_field_score": 1e30,
                "maximum_field_harm_probability": 0.0,
                "minimum_front_lower_delta": 1e30,
                "maximum_front_harm_probability": 0.0,
            }
        else:
            parameters = route
        blend = float(parameters["interpolation_fraction"])
        models = _fit_models(
            features=train_features,
            field_targets=field_targets_by_blend[blend],
            front_targets=front_targets_by_blend[blend],
            parameters=parameters,
            field_harm_threshold=field_harm_threshold,
            front_harm_threshold=front_harm_threshold,
        )
        reference = models["field_mean"]
        factory = MultiObjectiveRiskSingleExpertFactory(
            expert_log_gains=expert_logs,
            expert_candidate_ids=expert_ids,
            baseline_expert_index=baseline_index,
            feature_mean=torch.as_tensor(reference["mean"]),
            feature_scale=torch.as_tensor(reference["scale"]),
            field_mean_weights=torch.as_tensor(
                models["field_mean"]["weights"]
            ),
            field_lower_weights=torch.as_tensor(
                models["field_lower"]["weights"]
            ),
            field_harm_weights=torch.as_tensor(
                models["field_harm"]["weights"]
            ),
            front_lower_weights=torch.as_tensor(
                models["front_lower"]["weights"]
            ),
            front_harm_weights=torch.as_tensor(
                models["front_harm"]["weights"]
            ),
            minimum_field_score=float(
                parameters["minimum_field_score"]
            ),
            maximum_field_harm_probability=float(
                parameters["maximum_field_harm_probability"]
            ),
            minimum_front_lower_delta=float(
                parameters["minimum_front_lower_delta"]
            ),
            maximum_front_harm_probability=float(
                parameters["maximum_front_harm_probability"]
            ),
            interpolation_fraction=blend,
        ).to(device)
        selection_models[route_name] = {
            "parameters": copy.deepcopy(parameters),
            **{
                f"{name}_weights_private": np.asarray(
                    model["weights"]
                ).tolist()
                for name, model in models.items()
            },
            "feature_mean_private": np.asarray(reference["mean"]).tolist(),
            "feature_scale_private": np.asarray(reference["scale"]).tolist(),
        }
        for split_name in ("risk_validation", "risk_calibration"):
            split_features = features_by_split[split_name]
            scores = {
                name: ridge_scores(model, split_features)
                for name, model in models.items()
            }
            actions, accepted, field_harm, front_harm = (
                numpy_multiobjective_actions(
                    field_mean_scores=scores["field_mean"],
                    field_lower_scores=scores["field_lower"],
                    field_harm_logits=scores["field_harm"],
                    front_lower_scores=scores["front_lower"],
                    front_harm_logits=scores["front_harm"],
                    baseline_expert_index=baseline_index,
                    minimum_field_score=float(
                        parameters["minimum_field_score"]
                    ),
                    maximum_field_harm_probability=float(
                        parameters["maximum_field_harm_probability"]
                    ),
                    minimum_front_lower_delta=float(
                        parameters["minimum_front_lower_delta"]
                    ),
                    maximum_front_harm_probability=float(
                        parameters["maximum_front_harm_probability"]
                    ),
                )
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
                    "accepted_field_harm_probability_mean": (
                        0.0
                        if not np.any(accepted)
                        else float(np.mean(field_harm[accepted]))
                    ),
                    "accepted_front_harm_probability_mean": (
                        0.0
                        if not np.any(accepted)
                        else float(np.mean(front_harm[accepted]))
                    ),
                }
            )
            rows, ledger = _evaluate_integrated_factory(
                method=f"mo_rq_ogse_{route_name}",
                wrapped=splits[split_name],
                operator=nominal_operator,
                source_config=source_config,
                device=device,
                factory=factory,
            )
            transfer_rows.extend(rows)
            transfer_execution.append(ledger)

    baseline_rows = [
        row
        for row in rq_private["transfer_metric_rows_private"]
        if row["method"] == "static_pcgls4"
    ]
    transfer_rows.extend(copy.deepcopy(baseline_rows))
    summaries = []
    methods = ("mo_rq_ogse_strict", "mo_rq_ogse_diagnostic")
    for method_index, method in enumerate(methods):
        for split_index, split_name in enumerate(
            ("risk_validation", "risk_calibration")
        ):
            summaries.append(
                paired_gain_summary(
                    transfer_rows,
                    split=split_name,
                    candidate_method=method,
                    bootstrap_seed=20263400
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
    ledgers_valid = all(
        row["logical_calls_per_sample"] == {
            "forward": 4,
            "adjoint": 4,
        }
        and bool(row["data_objective_monotone"])
        and float(row["gain_minimum"]) > 0.0
        and float(row["gain_geometric_mean_maximum_defect"]) <= 2e-5
        for row in transfer_execution
    )
    strict_primary = next(
        row["pass"]
        for row in development_gates
        if row["method"] == "mo_rq_ogse_strict"
    )
    strict_secondary = next(
        row["pass"]
        for row in secondary_audits
        if row["method"] == "mo_rq_ogse_strict"
    )
    private = {
        "schema_version": PRIVATE_SCHEMA,
        "algorithm_schema": RISK_QUANTILE_EXPERT_SCHEMA,
        "status": STATUS,
        "evidence_scope": (
            "REAL_PSU_SUPPORT_GEOMETRY_WITH_ANALYTIC_REACTION_MORPHOLOGY_"
            "AND_SYNTHETIC_CAMERA_NOISE_POSTOPEN_MULTI_OBJECTIVE_"
            "DEVELOPMENT_ONLY"
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
            "rq_private_report_path": str(
                rq_private_report_path.resolve()
            ),
            "view_root": str(view_root.resolve()),
            "device": str(device),
        },
        "configuration_public": copy.deepcopy(config),
        "regeneration_checks": {
            "development_metadata_matches_frozen_rows": True,
            "finite_action_rows_reused_without_fresh": True,
            "field_and_front_heads_screened_by_train_oof_only": True,
            "integrated_factory_shares_first_adjoint": True,
            "opened_fresh_not_loaded": True,
            "fixed_spd_and_call_ledgers_pass": bool(ledgers_valid),
        },
        "expert_bank": copy.deepcopy(rq_private["expert_bank"]),
        "selection_screen": screen,
        "selected_strict_candidate": selected,
        "best_diagnostic_candidate": diagnostic,
        "oof_predictions_private": oof_cache_private,
        "selection_models_private": selection_models,
        "route_decision_summary": route_decisions,
        "transfer_metric_rows_private": transfer_rows,
        "paired_gain_summary": summaries,
        "development_gates": development_gates,
        "secondary_metric_safety_audits": secondary_audits,
        "overall_decision": {
            "primary_strict_gate_pass": bool(strict_primary),
            "secondary_metric_safety_pass": bool(strict_secondary),
            "fresh_repeat_authorized": False,
            "decision": (
                "HOLD_POSTOPEN_MULTI_OBJECTIVE_DIAGNOSTIC_"
                "INDEPENDENT_REPEAT_NOT_AUTHORIZED"
            ),
        },
        "execution": transfer_execution,
        "execution_summary": {
            "finite_action_rows_reused": True,
            "oof_route_candidate_count": len(screen),
            "transfer_route_count": 2,
            "logical_calls_per_transfer_sample": {
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
            "front_safety_thresholds_are_posthoc_formalized": True,
            "quantile_heads_are_empirical_not_conformal": True,
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
            "psu_b0_mo_rq_ogse_pcgls_development_v1.json"
        ),
    )
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument("--probe-private-report", type=Path, required=True)
    parser.add_argument("--rq-private-report", type=Path, required=True)
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
        rq_private_report_path=args.rq_private_report,
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
