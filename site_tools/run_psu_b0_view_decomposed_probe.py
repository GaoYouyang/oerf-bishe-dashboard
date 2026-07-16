#!/usr/bin/env python3
"""Probe whether camera-wise adjoint conflict improves B0 expert routing."""

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

from demo_t16_operator.psu_b0_streaming_operator import (
    zero_outer_boundary_support,
)
from demo_t16_operator.psu_b0_view_decomposed_features import (
    VIEW_DECOMPOSED_FEATURE_SCHEMA,
    view_adjoint_conflict_features,
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
from site_tools.run_psu_b0_rq_ogse_pcgls_development import (
    _evaluate_action_indices,
    _oof_model_scores,
    _risk_development_gate,
    _secondary_metric_safety_audit,
    action_gain_targets,
    best_diagnostic_candidate,
    numpy_risk_actions,
    route_gain_metrics,
    select_screen_candidate,
)
from site_tools.run_psu_b0_spectral_preconditioner_pilot import (
    _expanded_measurement_values,
    _load_json,
)


PRIVATE_SCHEMA = "psu-b0-view-decomposed-probe-private-1.0"
PUBLIC_SCHEMA = "psu-b0-view-decomposed-probe-public-1.0"
STATUS = "VIEW_DECOMPOSED_MECHANISM_PROBE_COMPLETE_FRESH_NOT_USED"


def _max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def leave_group_out_scores(
    features: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    *,
    regularization: float,
) -> np.ndarray:
    """Predict every row from a model that excluded its complete group."""

    values = np.asarray(features, dtype=np.float64)
    target_values = np.asarray(targets, dtype=np.float64)
    group_values = np.asarray(groups).reshape(-1)
    if (
        values.ndim != 2
        or target_values.ndim != 2
        or len(values) != len(target_values)
        or len(values) != len(group_values)
    ):
        raise ValueError("features, targets, and groups must align")
    output = np.zeros_like(target_values)
    for group in np.unique(group_values):
        holdout = group_values == group
        train = ~holdout
        if not np.any(train) or not np.any(holdout):
            raise ValueError("each group split needs train and holdout rows")
        model = fit_ridge_multioutput(
            values[train],
            target_values[train],
            regularization=float(regularization),
        )
        output[holdout] = ridge_scores(model, values[holdout])
    return output


def _feature_batch(
    *,
    wrapped: DevelopmentSplit,
    operator: Any,
    device: torch.device,
    rays_per_view: int,
    batch_size: int = 12,
) -> tuple[np.ndarray, tuple[str, ...], dict[str, Any]]:
    split = wrapped.data
    feature_parts = []
    maximum_sum_error = 0.0
    started = time.perf_counter()
    operator.reset_call_counts()
    for start in range(0, len(split.truth), int(batch_size)):
        stop = min(start + int(batch_size), len(split.truth))
        observation = split.observation_uv[start:stop].to(device)
        sigma = split.sigma_by_view[start:stop].to(device)
        mask = split.view_mask[start:stop].to(device)
        active = _expanded_measurement_values(
            mask,
            rays_per_view=rays_per_view,
        )
        expanded_sigma = _expanded_measurement_values(
            sigma,
            rays_per_view=rays_per_view,
        )
        whitened = active * observation / expanded_sigma.square()
        with torch.no_grad():
            grouped = operator.adjoint_by_view(
                whitened,
                rays_per_view=rays_per_view,
            )
            pooled_reference = operator._adjoint(whitened)[:, None]
            pooled_grouped = grouped.sum(dim=1)
            relative_error = (
                torch.linalg.vector_norm(
                    (pooled_grouped - pooled_reference).flatten(1),
                    dim=1,
                )
                / torch.linalg.vector_norm(
                    pooled_reference.flatten(1),
                    dim=1,
                ).clamp_min(1e-20)
            )
            maximum_sum_error = max(
                maximum_sum_error,
                float(torch.max(relative_error)),
            )
            features, names = view_adjoint_conflict_features(
                grouped,
                view_mask=mask,
            )
        feature_parts.append(features.cpu())
    _synchronize(device)
    calls = operator.call_report()
    return (
        torch.cat(feature_parts, dim=0).numpy(),
        names,
        {
            "split": split.name,
            "sample_count": len(split.truth),
            "wall_seconds": float(time.perf_counter() - started),
            "logical_grouped_adjoint_calls_per_sample": 1,
            "per_ray_scatter_traversals": 1,
            "equal_flop_to_pooled_adjoint": False,
            "maximum_group_sum_relative_error": maximum_sum_error,
            "batch_invocations": {
                "forward": int(calls["forward_calls"]),
                "adjoint": int(calls["adjoint_calls"]),
            },
        },
    )


def _fallback_route(
    *,
    blend: float,
    regularization: float,
) -> dict[str, Any]:
    return {
        "route_mode": "mean_only",
        "interpolation_fraction": float(blend),
        "mean_regularization": float(regularization),
        "lower_quantile": 0.1,
        "quantile_regularization": 1.0,
        "harm_regularization": 1.0,
        "minimum_score": 1e30,
        "maximum_harm_probability": 1.0,
        "coverage": 0.0,
        "mean_field_gain_percent": 0.0,
        "p10_field_gain_percent": 0.0,
        "minimum_field_gain_percent": 0.0,
        "harm_over_one_percent_rate": 0.0,
        "accepted_harm_over_one_percent_rate": 0.0,
        "accepted_mean_field_gain_percent": 0.0,
        "selection_status": "NO_STRICT_ROUTE_EXACT_BASELINE_FALLBACK",
        "strict_gate_pass": False,
    }


def _route_actions(
    scores: np.ndarray,
    *,
    baseline_expert_index: int,
    minimum_score: float,
) -> tuple[np.ndarray, np.ndarray]:
    zeros = np.zeros_like(scores)
    actions, accepted, _ = numpy_risk_actions(
        mean_scores=scores,
        lower_scores=zeros,
        harm_logits=zeros,
        baseline_expert_index=int(baseline_expert_index),
        route_mode="mean_only",
        minimum_score=float(minimum_score),
        maximum_harm_probability=1.0,
    )
    return actions, accepted


def _screen_feature_set(
    *,
    feature_set: str,
    features: np.ndarray,
    gain_targets_by_blend: dict[float, np.ndarray],
    fold_labels: np.ndarray,
    baseline_expert_index: int,
    screen_config: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, Any]:
    screen = []
    fold_count = int(screen_config["folds"])
    for blend, targets in gain_targets_by_blend.items():
        for regularization in screen_config["regularization_grid"]:
            scores = _oof_model_scores(
                features,
                targets,
                fold_labels,
                fold_count=fold_count,
                fit_kind="ridge",
                regularization=float(regularization),
            )
            for minimum_score in screen_config["minimum_score_grid"]:
                actions, accepted = _route_actions(
                    scores,
                    baseline_expert_index=baseline_expert_index,
                    minimum_score=float(minimum_score),
                )
                screen.append(
                    {
                        "feature_set": str(feature_set),
                        "route_mode": "mean_only",
                        "interpolation_fraction": float(blend),
                        "mean_regularization": float(regularization),
                        "lower_quantile": 0.1,
                        "quantile_regularization": 1.0,
                        "harm_regularization": 1.0,
                        "minimum_score": float(minimum_score),
                        "maximum_harm_probability": 1.0,
                        **route_gain_metrics(
                            targets,
                            actions,
                            accepted,
                        ),
                    }
                )
    selected = select_screen_candidate(screen, gate=gate)
    diagnostic = best_diagnostic_candidate(screen)
    if not bool(selected.get("strict_gate_pass")):
        selected = _fallback_route(
            blend=min(gain_targets_by_blend),
            regularization=float(
                screen_config["regularization_grid"][0]
            ),
        )
        selected["feature_set"] = str(feature_set)
    return {
        "screen": screen,
        "strict": selected,
        "diagnostic": diagnostic,
    }


def _stress_audit(
    *,
    route: dict[str, Any],
    features: np.ndarray,
    gain_targets_by_blend: dict[float, np.ndarray],
    families: np.ndarray,
    noise_profiles: np.ndarray,
    baseline_expert_index: int,
) -> dict[str, Any]:
    blend = float(route["interpolation_fraction"])
    targets = gain_targets_by_blend[blend]
    regularization = float(route["mean_regularization"])
    output = {}
    for name, groups in (
        ("leave_one_morphology_family_out", families),
        ("leave_one_noise_profile_out", noise_profiles),
    ):
        scores = leave_group_out_scores(
            features,
            targets,
            groups,
            regularization=regularization,
        )
        actions, accepted = _route_actions(
            scores,
            baseline_expert_index=baseline_expert_index,
            minimum_score=float(route["minimum_score"]),
        )
        output[name] = {
            "group_count": int(len(np.unique(groups))),
            **route_gain_metrics(targets, actions, accepted),
        }
    return output


def build_public_summary(private: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": PUBLIC_SCHEMA,
        "feature_schema": VIEW_DECOMPOSED_FEATURE_SCHEMA,
        "status": private["status"],
        "evidence_scope": private["evidence_scope"],
        "configuration": copy.deepcopy(private["configuration_public"]),
        "regeneration_checks": copy.deepcopy(
            private["regeneration_checks"]
        ),
        "expert_bank": copy.deepcopy(private["expert_bank"]),
        "feature_schema_summary": copy.deepcopy(
            private["feature_schema_summary"]
        ),
        "train_oof_selection": copy.deepcopy(
            private["train_oof_selection"]
        ),
        "stress_audits": copy.deepcopy(private["stress_audits"]),
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


def run_probe(
    *,
    root: Path,
    config_path: Path,
    development_report_path: Path,
    pooled_probe_private_report_path: Path,
    rq_private_report_path: Path,
    view_root: Path,
    device_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _load_json(config_path)
    development_config = _load_json(
        root / str(config["source_development_config"])
    )
    rq_config = _load_json(root / str(config["source_rq_config"]))
    source_config = _load_json(
        root / str(development_config["source_pilot"]["config"])
    )
    development_report = _load_json(development_report_path)
    pooled_probe = _load_json(pooled_probe_private_report_path)
    rq_private = _load_json(rq_private_report_path)
    expert_ids = [
        str(value)
        for value in rq_private["expert_bank"]["candidate_ids"]
    ]
    baseline_id = str(rq_private["expert_bank"]["baseline_candidate_id"])
    baseline_index = expert_ids.index(baseline_id)

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

    features_by_split: dict[str, dict[str, np.ndarray]] = {}
    feature_execution = []
    conflict_names: tuple[str, ...] = ()
    pooled_names = tuple(
        pooled_probe["feature_schema_summary"][
            "initial_normal_spectrum"
        ]["feature_names"]
    )
    for split_name, wrapped in splits.items():
        conflict, conflict_names, ledger = _feature_batch(
            wrapped=wrapped,
            operator=nominal_operator,
            device=device,
            rays_per_view=rays_per_view,
        )
        pooled = np.asarray(
            pooled_probe["features_private"][split_name][
                "initial_normal_spectrum"
            ],
            dtype=np.float64,
        )
        if len(pooled) != len(conflict):
            raise ValueError("pooled and view-decomposed features do not align")
        features_by_split[split_name] = {
            "pooled_initial_normal": pooled,
            "view_conflict": conflict,
            "pooled_plus_view_conflict": np.concatenate(
                (pooled, conflict),
                axis=1,
            ),
        }
        feature_execution.append(ledger)

    train = splits["risk_train"].data
    train_sample_ids = list(train.sample_ids)
    family_names = np.asarray(list(train.families), dtype=str)
    family_lookup = {
        family: index
        for index, family in enumerate(sorted(set(family_names)))
    }
    family_labels = np.asarray(
        [family_lookup[value] for value in family_names],
        dtype=np.int64,
    )
    noise_profiles = np.asarray(
        list(splits["risk_train"].noise_profiles),
        dtype=str,
    )

    gain_targets_by_blend = {}
    for blend_key, rows_by_expert_json in rq_private[
        "finite_action_rows_private"
    ].items():
        rows_by_expert = {
            int(index): rows
            for index, rows in rows_by_expert_json.items()
        }
        gain_targets_by_blend[float(blend_key)] = action_gain_targets(
            sample_ids=train_sample_ids,
            baseline_rows=rows_by_expert[baseline_index],
            action_rows_by_expert=rows_by_expert,
            expert_count=len(expert_ids),
            baseline_expert_index=baseline_index,
        )

    selection = {}
    stress = {}
    screen_config = config["mean_route_screen"]
    for feature_set in config["feature_sets"]:
        selected = _screen_feature_set(
            feature_set=str(feature_set),
            features=features_by_split["risk_train"][feature_set],
            gain_targets_by_blend=gain_targets_by_blend,
            fold_labels=family_labels,
            baseline_expert_index=baseline_index,
            screen_config=screen_config,
            gate=rq_config["strict_oof_gate"],
        )
        selection[str(feature_set)] = selected
        stress[str(feature_set)] = {
            route_name: _stress_audit(
                route=route,
                features=features_by_split["risk_train"][feature_set],
                gain_targets_by_blend=gain_targets_by_blend,
                families=family_names,
                noise_profiles=noise_profiles,
                baseline_expert_index=baseline_index,
            )
            for route_name, route in (
                ("strict", selected["strict"]),
                ("diagnostic", selected["diagnostic"]),
            )
        }

    expert_logs = _expert_bank(
        headroom_public=_load_json(
            root / str(config["source_headroom_public_summary"])
        ),
        expert_ids=expert_ids,
        grid_size=grid_size,
        device=device,
    )
    transfer_rows = []
    transfer_execution = []
    route_decisions = []
    selection_models = {}
    methods = []
    for feature_set in config["feature_sets"]:
        train_features = features_by_split["risk_train"][feature_set]
        for route_name in ("strict", "diagnostic"):
            route = selection[feature_set][route_name]
            blend = float(route["interpolation_fraction"])
            model = fit_ridge_multioutput(
                train_features,
                gain_targets_by_blend[blend],
                regularization=float(route["mean_regularization"]),
            )
            method = f"vd0_{feature_set}_{route_name}"
            methods.append(method)
            selection_models[method] = {
                "parameters": copy.deepcopy(route),
                "feature_mean_private": np.asarray(model["mean"]).tolist(),
                "feature_scale_private": np.asarray(model["scale"]).tolist(),
                "weights_private": np.asarray(model["weights"]).tolist(),
            }
            for split_name in ("risk_validation", "risk_calibration"):
                scores = ridge_scores(
                    model,
                    features_by_split[split_name][feature_set],
                )
                actions, accepted = _route_actions(
                    scores,
                    baseline_expert_index=baseline_index,
                    minimum_score=float(route["minimum_score"]),
                )
                route_decisions.append(
                    {
                        "feature_set": str(feature_set),
                        "route": route_name,
                        "method": method,
                        "split": split_name,
                        "sample_count": len(actions),
                        "coverage": float(np.mean(accepted)),
                        "selected_expert_counts": {
                            expert_ids[index]: int(np.sum(actions == index))
                            for index in range(len(expert_ids))
                        },
                    }
                )
                rows, ledger = _evaluate_action_indices(
                    method=method,
                    wrapped=splits[split_name],
                    operator=nominal_operator,
                    source_config=source_config,
                    device=device,
                    action_indices=actions,
                    expert_log_gains=expert_logs,
                    baseline_expert_index=baseline_index,
                    interpolation_fraction=blend,
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
                    bootstrap_seed=20263400
                    + 10 * method_index
                    + split_index,
                )
            )
    development_gates = [
        _risk_development_gate(
            summaries,
            method=method,
            gate=rq_config["development_gate"],
        )
        for method in methods
    ]
    secondary_audits = [
        _secondary_metric_safety_audit(
            summaries,
            method=method,
            gate=rq_config["secondary_metric_safety_audit"],
        )
        for method in methods
    ]
    summary_lookup = {
        (row["candidate_method"], row["split"]): row
        for row in summaries
    }
    pooled_method = "vd0_pooled_initial_normal_strict"
    combined_method = "vd0_pooled_plus_view_conflict_strict"
    combined_beats_pooled = all(
        summary_lookup[(combined_method, split_name)][
            "mean_field_gain_percent"
        ]
        > summary_lookup[(pooled_method, split_name)][
            "mean_field_gain_percent"
        ]
        for split_name in ("risk_validation", "risk_calibration")
    )
    combined_gate = next(
        row["pass"]
        for row in development_gates
        if row["method"] == combined_method
    )
    execution = feature_execution + transfer_execution
    reconstruction_ledgers_valid = all(
        row["logical_calls_per_sample"] == {
            "forward": 4,
            "adjoint": 4,
        }
        and bool(row["data_objective_monotone"])
        and float(row["gain_minimum"]) > 0.0
        and float(row["gain_geometric_mean_maximum_defect"]) <= 2e-5
        for row in transfer_execution
    )
    grouped_sum_error = max(
        float(row["maximum_group_sum_relative_error"])
        for row in feature_execution
    )
    public_config = copy.deepcopy(config)
    public_config["inherited_strict_oof_gate"] = copy.deepcopy(
        rq_config["strict_oof_gate"]
    )
    public_config["inherited_development_gate"] = copy.deepcopy(
        rq_config["development_gate"]
    )
    private = {
        "schema_version": PRIVATE_SCHEMA,
        "feature_schema": VIEW_DECOMPOSED_FEATURE_SCHEMA,
        "status": STATUS,
        "evidence_scope": (
            "REAL_PSU_SUPPORT_GEOMETRY_WITH_ANALYTIC_REACTION_MORPHOLOGY_"
            "AND_SYNTHETIC_CAMERA_NOISE_POSTOPEN_VIEW_DECOMPOSITION_PROBE"
        ),
        "configuration_private": {
            "root": str(root.resolve()),
            "config_path": str(config_path.resolve()),
            "development_report_path": str(
                development_report_path.resolve()
            ),
            "pooled_probe_private_report_path": str(
                pooled_probe_private_report_path.resolve()
            ),
            "rq_private_report_path": str(
                rq_private_report_path.resolve()
            ),
            "view_root": str(view_root.resolve()),
            "device": str(device),
        },
        "configuration_public": public_config,
        "regeneration_checks": {
            "development_metadata_matches_frozen_rows": True,
            "pooled_features_reused_without_reconstruction": True,
            "finite_action_targets_reused_without_reconstruction": True,
            "feature_and_route_screen_use_risk_train_only": True,
            "grouped_adjoint_sum_matches_pooled": grouped_sum_error <= 2e-5,
            "maximum_group_sum_relative_error": grouped_sum_error,
            "grouped_operator_traverses_each_ray_scatter_once": True,
            "grouped_operator_equal_flop_to_pooled_adjoint": False,
            "opened_fresh_not_loaded": True,
            "fixed_spd_and_call_ledgers_pass": bool(
                reconstruction_ledgers_valid
            ),
        },
        "expert_bank": {
            "candidate_ids": expert_ids,
            "baseline_candidate_id": baseline_id,
            "deployment_uses_family_labels": False,
        },
        "feature_schema_summary": {
            "pooled_initial_normal": {
                "feature_count": len(pooled_names),
                "feature_names": list(pooled_names),
            },
            "view_conflict": {
                "feature_count": len(conflict_names),
                "feature_names": list(conflict_names),
            },
            "pooled_plus_view_conflict": {
                "feature_count": len(pooled_names) + len(conflict_names),
                "feature_names": [
                    *pooled_names,
                    *conflict_names,
                ],
            },
        },
        "features_private": {
            split_name: {
                feature_set: values.tolist()
                for feature_set, values in split_features.items()
            }
            for split_name, split_features in features_by_split.items()
        },
        "gain_targets_private": {
            str(blend): targets.tolist()
            for blend, targets in gain_targets_by_blend.items()
        },
        "train_oof_selection": {
            feature_set: {
                "screen": result["screen"],
                "strict": result["strict"],
                "diagnostic": result["diagnostic"],
            }
            for feature_set, result in selection.items()
        },
        "stress_audits": stress,
        "selection_models_private": selection_models,
        "route_decision_summary": route_decisions,
        "transfer_metric_rows_private": transfer_rows,
        "paired_gain_summary": summaries,
        "development_gates": development_gates,
        "secondary_metric_safety_audits": secondary_audits,
        "overall_decision": {
            "combined_strict_development_gate_pass": bool(combined_gate),
            "combined_strict_beats_pooled_strict_on_both_transfer_splits": (
                bool(combined_beats_pooled)
            ),
            "fresh_repeat_authorized": False,
            "decision": (
                "VD0_VIEW_CONFLICT_SUPPORTED_FOR_NEXT_ARCHITECTURE_STAGE"
                if combined_gate and combined_beats_pooled
                else "VD0_VIEW_CONFLICT_NOT_YET_TRANSFER_SUPPORTED"
            ),
        },
        "execution": execution,
        "execution_summary": {
            "grouped_feature_batches": int(
                sum(
                    row["batch_invocations"]["adjoint"]
                    for row in feature_execution
                )
            ),
            "reused_train_action_reconstruction_count": int(
                len(rq_private["finite_action_bank"])
            ),
            "new_train_action_reconstruction_count": 0,
            "transfer_route_count": len(methods),
            "logical_calls_per_transfer_reconstruction_sample": {
                "forward": 4,
                "adjoint": 4,
            },
        },
        "runtime": {
            "wall_seconds": float(time.perf_counter() - started),
            "maximum_rss_bytes": int(_max_rss_bytes()),
        },
        "claim_boundary": {
            "postopen_mechanism_probe_only": True,
            "validation_and_calibration_are_postopen_diagnostics": True,
            "fresh_values_loaded": False,
            "fresh_repeat_authorized": False,
            "grouped_invocation_is_equal_flop_to_pooled_adjoint": False,
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
            "psu_b0_view_decomposed_probe_v1.json"
        ),
    )
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument(
        "--pooled-probe-private-report",
        type=Path,
        required=True,
    )
    parser.add_argument("--rq-private-report", type=Path, required=True)
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--public-output", type=Path)
    args = parser.parse_args()
    private, public = run_probe(
        root=args.root,
        config_path=args.config,
        development_report_path=args.development_report,
        pooled_probe_private_report_path=(
            args.pooled_probe_private_report
        ),
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
