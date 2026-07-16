#!/usr/bin/env python3
"""Probe whether the shared first adjoint reveals useful spectral experts."""

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

from demo_t16_operator.psu_b0_initial_normal_features import (
    INITIAL_NORMAL_FEATURE_SCHEMA,
    initial_normal_spectral_features,
    measurement_metadata_features,
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
    _load_json,
)


PRIVATE_SCHEMA = "psu-b0-observable-morphology-probe-private-1.0"
PUBLIC_SCHEMA = "psu-b0-observable-morphology-probe-public-1.0"
STATUS = "OBSERVABLE_MORPHOLOGY_PROBE_COMPLETE_FRESH_NOT_USED"


def _max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def stratified_folds(
    labels: np.ndarray,
    *,
    fold_count: int,
) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int64).reshape(-1)
    if int(fold_count) < 2:
        raise ValueError("fold_count must be at least two")
    folds = np.empty(len(values), dtype=np.int64)
    for label in sorted(np.unique(values)):
        indices = np.flatnonzero(values == label)
        for offset, index in enumerate(indices):
            folds[index] = offset % int(fold_count)
    return folds


def _standardize_fit(
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(features, dtype=np.float64)
    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return (values - mean) / scale, mean, scale


def _standardize_apply(
    features: np.ndarray,
    *,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    return (np.asarray(features, dtype=np.float64) - mean) / scale


def fit_ridge_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    class_count: int,
    regularization: float,
) -> dict[str, np.ndarray | float | int]:
    target_labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(features) != len(target_labels):
        raise ValueError("features and labels must align")
    if int(class_count) < 2:
        raise ValueError("class_count must be at least two")
    targets = np.eye(int(class_count), dtype=np.float64)[target_labels]
    return fit_ridge_multioutput(
        features,
        targets,
        regularization=float(regularization),
    )


def fit_ridge_multioutput(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    regularization: float,
) -> dict[str, np.ndarray | float | int]:
    """Fit a standardized multi-output ridge map with an unpenalized bias."""

    values, mean, scale = _standardize_fit(features)
    target_values = np.asarray(targets, dtype=np.float64)
    if target_values.ndim != 2 or len(values) != len(target_values):
        raise ValueError("features and multi-output targets must align")
    design = np.concatenate(
        (np.ones((len(values), 1), dtype=np.float64), values),
        axis=1,
    )
    penalty = np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    gram = design.T @ design + float(regularization) * penalty
    weights = np.linalg.solve(gram, design.T @ target_values)
    return {
        "mean": mean,
        "scale": scale,
        "weights": weights,
        "regularization": float(regularization),
        "output_count": int(target_values.shape[1]),
    }


def ridge_scores(
    model: dict[str, np.ndarray | float | int],
    features: np.ndarray,
) -> np.ndarray:
    values = _standardize_apply(
        features,
        mean=np.asarray(model["mean"]),
        scale=np.asarray(model["scale"]),
    )
    design = np.concatenate(
        (np.ones((len(values), 1), dtype=np.float64), values),
        axis=1,
    )
    return design @ np.asarray(model["weights"])


def score_predictions(
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("scores must contain at least two classes")
    order = np.argsort(values, axis=1)
    predicted = order[:, -1]
    margin = (
        values[np.arange(len(values)), order[:, -1]]
        - values[np.arange(len(values)), order[:, -2]]
    )
    return predicted.astype(np.int64), margin


def _candidate_lookup(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    output = {}
    for row in rows:
        key = (str(row["sample_id"]), str(row["candidate_id"]))
        if key in output:
            raise ValueError(f"duplicate candidate row: {key}")
        output[key] = row
    return output


def prediction_gains(
    *,
    sample_ids: list[str],
    predicted_labels: np.ndarray,
    margins: np.ndarray,
    threshold: float,
    class_candidates: list[str],
    candidate_rows: list[dict[str, Any]],
    baseline_candidate_id: str,
) -> dict[str, Any]:
    lookup = _candidate_lookup(candidate_rows)
    gains = []
    accepted = []
    selected_ids = []
    for index, sample_id in enumerate(sample_ids):
        predicted = class_candidates[int(predicted_labels[index])]
        use_candidate = (
            float(margins[index]) >= float(threshold)
            and predicted != str(baseline_candidate_id)
        )
        selected = predicted if use_candidate else str(baseline_candidate_id)
        baseline = float(
            lookup[(sample_id, str(baseline_candidate_id))][
                "field_relative_l2"
            ]
        )
        candidate = float(
            lookup[(sample_id, selected)]["field_relative_l2"]
        )
        gains.append(
            100.0 * (baseline - candidate) / max(baseline, 1e-12)
        )
        accepted.append(use_candidate)
        selected_ids.append(selected)
    gain = np.asarray(gains, dtype=np.float64)
    acceptance = np.asarray(accepted, dtype=bool)
    return {
        "mean_field_gain_percent": float(np.mean(gain)),
        "p10_field_gain_percent": float(np.quantile(gain, 0.10)),
        "minimum_field_gain_percent": float(np.min(gain)),
        "harm_over_one_percent_rate": float(np.mean(gain < -1.0)),
        "coverage": float(np.mean(acceptance)),
        "accepted_harm_over_one_percent_rate": (
            0.0
            if not np.any(acceptance)
            else float(np.mean(gain[acceptance] < -1.0))
        ),
        "selected_candidate_ids": selected_ids,
        "gain_values": gain,
        "accepted": acceptance,
    }


def select_ridge_route(
    *,
    features: np.ndarray,
    labels: np.ndarray,
    sample_ids: list[str],
    class_candidates: list[str],
    candidate_rows: list[dict[str, Any]],
    baseline_candidate_id: str,
    fold_count: int,
    lambda_grid: list[float],
    minimum_coverage: float,
    maximum_harm_rate: float,
    maximum_accepted_harm_rate: float,
) -> dict[str, Any]:
    folds = stratified_folds(labels, fold_count=int(fold_count))
    candidates = []
    for regularization in lambda_grid:
        oof_scores = np.zeros(
            (len(features), len(class_candidates)),
            dtype=np.float64,
        )
        for fold in range(int(fold_count)):
            train = folds != fold
            holdout = folds == fold
            model = fit_ridge_classifier(
                features[train],
                labels[train],
                class_count=len(class_candidates),
                regularization=float(regularization),
            )
            oof_scores[holdout] = ridge_scores(model, features[holdout])
        predicted, margins = score_predictions(oof_scores)
        padding = max(float(np.ptp(margins)), 1.0) * 1e-6
        thresholds = np.unique(
            np.concatenate(
                (
                    np.asarray(
                        [float(np.min(margins)) - padding],
                        dtype=np.float64,
                    ),
                    margins,
                    np.asarray(
                        [float(np.max(margins)) + padding],
                        dtype=np.float64,
                    ),
                )
            )
        )
        for threshold in thresholds:
            gains = prediction_gains(
                sample_ids=sample_ids,
                predicted_labels=predicted,
                margins=margins,
                threshold=float(threshold),
                class_candidates=class_candidates,
                candidate_rows=candidate_rows,
                baseline_candidate_id=baseline_candidate_id,
            )
            candidates.append(
                {
                    "regularization": float(regularization),
                    "confidence_threshold": float(threshold),
                    "classification_accuracy": float(
                        np.mean(predicted == labels)
                    ),
                    "all_predictions_active_threshold": bool(
                        threshold == thresholds[0]
                    ),
                    **{
                        key: value
                        for key, value in gains.items()
                        if key
                        not in {
                            "gain_values",
                            "accepted",
                            "selected_candidate_ids",
                        }
                    },
                }
            )
    relaxed_feasible = [
        row
        for row in candidates
        if float(row["coverage"]) >= float(minimum_coverage)
        and float(row["harm_over_one_percent_rate"])
        <= float(maximum_harm_rate)
    ]
    strict_feasible = [
        row
        for row in relaxed_feasible
        if float(row["accepted_harm_over_one_percent_rate"])
        <= float(maximum_accepted_harm_rate)
    ]
    hard_candidates = [
        row for row in candidates if row["all_predictions_active_threshold"]
    ]

    def _best(rows: list[dict[str, Any]], status: str) -> dict[str, Any]:
        selected = max(
            rows,
            key=lambda row: (
                float(row["mean_field_gain_percent"]),
                float(row["p10_field_gain_percent"]),
                float(row["coverage"]),
                float(row["classification_accuracy"]),
                -float(row["regularization"]),
            ),
        )
        selected = dict(selected)
        selected["selection_status"] = status
        return selected

    def _fallback(status: str) -> dict[str, Any]:
        return {
            "regularization": float(lambda_grid[0]),
            "confidence_threshold": 1e30,
            "classification_accuracy": 0.0,
            "mean_field_gain_percent": 0.0,
            "p10_field_gain_percent": 0.0,
            "minimum_field_gain_percent": 0.0,
            "harm_over_one_percent_rate": 0.0,
            "coverage": 0.0,
            "accepted_harm_over_one_percent_rate": 0.0,
            "all_predictions_active_threshold": False,
            "selection_status": status,
        }

    if strict_feasible:
        strict_selected = _best(
            strict_feasible,
            "STRICT_OOF_GATE_FEASIBLE",
        )
    else:
        strict_selected = _fallback(
            "NO_STRICT_OOF_GATE_FEASIBLE_EXACT_FALLBACK"
        )
    if relaxed_feasible:
        relaxed_selected = _best(
            relaxed_feasible,
            "RELAXED_OOF_GATE_FEASIBLE_ACCEPTED_HARM_UNCONSTRAINED",
        )
    else:
        relaxed_selected = _fallback(
            "NO_RELAXED_OOF_GATE_FEASIBLE_EXACT_FALLBACK"
        )
    if hard_candidates:
        hard_selected = _best(
            hard_candidates,
            "OOF_UNGATED_DIAGNOSTIC",
        )
    else:
        hard_selected = _fallback("NO_UNGATED_ROUTE")
    return {
        "strict_selected": strict_selected,
        "relaxed_selected": relaxed_selected,
        "hard_selected": hard_selected,
        "candidate_count": len(candidates),
        "fold_assignments": folds.tolist(),
    }


def _feature_batch(
    *,
    wrapped: DevelopmentSplit,
    operator: Any,
    device: torch.device,
    rays_per_view: int,
    batch_size: int = 12,
) -> tuple[dict[str, np.ndarray], dict[str, tuple[str, ...]], dict[str, Any]]:
    split = wrapped.data
    metadata_parts = []
    normal_parts = []
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
        with torch.no_grad():
            initial_normal = operator.adjoint(
                active * observation / expanded_sigma.square()
            )
            initial_normal = (
                initial_normal
                * operator.support[None, None].to(initial_normal)
            )
            metadata, metadata_names = measurement_metadata_features(
                observation,
                sigma_by_view=sigma,
                view_mask=mask,
                rays_per_view=rays_per_view,
            )
            normal, normal_names = initial_normal_spectral_features(
                initial_normal
            )
        metadata_parts.append(metadata.cpu())
        normal_parts.append(normal.cpu())
    _synchronize(device)
    calls = operator.call_report()
    metadata_values = torch.cat(metadata_parts, dim=0).numpy()
    normal_values = torch.cat(normal_parts, dim=0).numpy()
    return (
        {
            "measurement_metadata": metadata_values,
            "initial_normal_spectrum": normal_values,
            "metadata_plus_initial_normal": np.concatenate(
                (metadata_values, normal_values),
                axis=1,
            ),
        },
        {
            "measurement_metadata": metadata_names,
            "initial_normal_spectrum": normal_names,
            "metadata_plus_initial_normal": (
                *metadata_names,
                *normal_names,
            ),
        },
        {
            "split": split.name,
            "sample_count": len(split.truth),
            "wall_seconds": float(time.perf_counter() - started),
            "logical_initial_adjoint_calls_per_sample": 1,
            "batch_invocations": {
                "forward": int(calls["forward_calls"]),
                "adjoint": int(calls["adjoint_calls"]),
            },
        },
    )


def _materialize_prediction_rows(
    *,
    method: str,
    sample_ids: list[str],
    predicted_labels: np.ndarray,
    margins: np.ndarray,
    threshold: float,
    class_candidates: list[str],
    candidate_rows: list[dict[str, Any]],
    baseline_candidate_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    outcome = prediction_gains(
        sample_ids=sample_ids,
        predicted_labels=predicted_labels,
        margins=margins,
        threshold=threshold,
        class_candidates=class_candidates,
        candidate_rows=candidate_rows,
        baseline_candidate_id=baseline_candidate_id,
    )
    lookup = _candidate_lookup(candidate_rows)
    rows = []
    for sample_id, candidate_id, accepted, margin in zip(
        sample_ids,
        outcome["selected_candidate_ids"],
        outcome["accepted"],
        margins,
    ):
        copied = dict(lookup[(sample_id, str(candidate_id))])
        copied["method"] = str(method)
        copied["selector_accepted"] = bool(accepted)
        copied["selector_margin"] = float(margin)
        copied["selected_candidate_id"] = str(candidate_id)
        rows.append(copied)
    public = {
        key: value
        for key, value in outcome.items()
        if key not in {"gain_values", "accepted", "selected_candidate_ids"}
    }
    return rows, public


def _confusion(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    class_count: int,
) -> list[list[int]]:
    matrix = np.zeros((int(class_count), int(class_count)), dtype=np.int64)
    for expected, observed in zip(truth, prediction):
        matrix[int(expected), int(observed)] += 1
    return matrix.tolist()


def build_public_summary(private: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": PUBLIC_SCHEMA,
        "feature_schema": INITIAL_NORMAL_FEATURE_SCHEMA,
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
        "transfer_classification": copy.deepcopy(
            private["transfer_classification"]
        ),
        "transfer_route_summary": copy.deepcopy(
            private["transfer_route_summary"]
        ),
        "paired_gain_summary": copy.deepcopy(
            private["paired_gain_summary"]
        ),
        "execution": copy.deepcopy(private["execution"]),
        "runtime": copy.deepcopy(private["runtime"]),
        "claim_boundary": copy.deepcopy(private["claim_boundary"]),
    }


def run_probe(
    *,
    root: Path,
    config_path: Path,
    development_report_path: Path,
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
    headroom_private = _load_json(headroom_private_report_path)
    development_report = _load_json(development_report_path)
    baseline_id = str(
        config["solver_contract"]["baseline_candidate_id"]
    )
    if (
        str(
            headroom_public["configuration"]["solver"]["baseline"][
                "candidate_id"
            ]
        )
        != baseline_id
    ):
        raise ValueError("observable probe baseline drifted from headroom audit")

    family_map = {
        str(row["stratum"][0]): str(row["candidate_id"])
        for row in headroom_public["train_selection"][
            "family_label_non_deployable"
        ]
    }
    class_candidates = sorted(set(family_map.values()))
    class_index = {
        candidate_id: index
        for index, candidate_id in enumerate(class_candidates)
    }
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

    features: dict[str, dict[str, np.ndarray]] = {}
    feature_names: dict[str, tuple[str, ...]] = {}
    execution = []
    for split_name, wrapped in splits.items():
        split_features, split_names, ledger = _feature_batch(
            wrapped=wrapped,
            operator=nominal_operator,
            device=device,
            rays_per_view=rays_per_view,
        )
        features[split_name] = split_features
        feature_names = split_names
        execution.append(ledger)

    candidate_rows_by_split = headroom_private[
        "candidate_metric_rows_private"
    ]
    train = splits["risk_train"].data
    train_sample_ids = list(train.sample_ids)
    train_labels = np.asarray(
        [
            class_index[family_map[str(family)]]
            for family in train.families
        ],
        dtype=np.int64,
    )
    selector_config = config["ridge_selector"]
    selections = {}
    transfer_rows: list[dict[str, Any]] = []
    transfer_classification = {}
    transfer_route_summary = {}
    paired_summaries = []
    for feature_set in config["feature_sets"]:
        selected = select_ridge_route(
            features=features["risk_train"][feature_set],
            labels=train_labels,
            sample_ids=train_sample_ids,
            class_candidates=class_candidates,
            candidate_rows=candidate_rows_by_split["risk_train"],
            baseline_candidate_id=baseline_id,
            fold_count=int(selector_config["folds"]),
            lambda_grid=[
                float(value)
                for value in selector_config["lambda_grid"]
            ],
            minimum_coverage=float(
                selector_config["minimum_oof_coverage"]
            ),
            maximum_harm_rate=float(
                selector_config[
                    "maximum_oof_harm_over_one_percent_rate"
                ]
            ),
            maximum_accepted_harm_rate=float(
                selector_config[
                    "maximum_oof_accepted_harm_over_one_percent_rate"
                ]
            ),
        )
        selections[feature_set] = selected
        routes = {
            "hard": selected["hard_selected"],
            "relaxed": selected["relaxed_selected"],
            "strict": selected["strict_selected"],
        }
        models = {
            route_name: fit_ridge_classifier(
                features["risk_train"][feature_set],
                train_labels,
                class_count=len(class_candidates),
                regularization=float(route["regularization"]),
            )
            for route_name, route in routes.items()
        }
        for split_index, split_name in enumerate(
            ("risk_validation", "risk_calibration")
        ):
            split = splits[split_name].data
            target_labels = np.asarray(
                [
                    class_index[family_map[str(family)]]
                    for family in split.families
                ],
                dtype=np.int64,
            )
            baseline_rows, _ = _materialize_prediction_rows(
                method="static_pcgls4",
                sample_ids=list(split.sample_ids),
                predicted_labels=np.zeros(len(split.sample_ids), dtype=np.int64),
                margins=np.full(len(split.sample_ids), -1e30),
                threshold=1e30,
                class_candidates=class_candidates,
                candidate_rows=candidate_rows_by_split[split_name],
                baseline_candidate_id=baseline_id,
            )
            transfer_rows.extend(baseline_rows)
            for route_index, (route_name, route) in enumerate(routes.items()):
                scores = ridge_scores(
                    models[route_name],
                    features[split_name][feature_set],
                )
                predicted, margins = score_predictions(scores)
                selected_rows, route_summary = _materialize_prediction_rows(
                    method=f"{feature_set}_{route_name}",
                    sample_ids=list(split.sample_ids),
                    predicted_labels=predicted,
                    margins=margins,
                    threshold=float(route["confidence_threshold"]),
                    class_candidates=class_candidates,
                    candidate_rows=candidate_rows_by_split[split_name],
                    baseline_candidate_id=baseline_id,
                )
                transfer_rows.extend(selected_rows)
                transfer_classification.setdefault(feature_set, {}).setdefault(
                    split_name,
                    {},
                )[route_name] = {
                    "sample_count": len(split.sample_ids),
                    "expert_accuracy": float(
                        np.mean(predicted == target_labels)
                    ),
                    "confusion_matrix": _confusion(
                        target_labels,
                        predicted,
                        class_count=len(class_candidates),
                    ),
                    "mean_margin": float(np.mean(margins)),
                    "p10_margin": float(np.quantile(margins, 0.10)),
                }
                transfer_route_summary.setdefault(
                    feature_set,
                    {},
                ).setdefault(split_name, {})[route_name] = route_summary
                paired_summaries.append(
                    paired_gain_summary(
                        transfer_rows,
                        split=split_name,
                        candidate_method=f"{feature_set}_{route_name}",
                        bootstrap_seed=20263000
                        + 100 * list(config["feature_sets"]).index(feature_set)
                        + 10 * split_index
                        + route_index,
                    )
                )

    private = {
        "schema_version": PRIVATE_SCHEMA,
        "feature_schema": INITIAL_NORMAL_FEATURE_SCHEMA,
        "status": STATUS,
        "evidence_scope": (
            "REAL_PSU_SUPPORT_GEOMETRY_WITH_ANALYTIC_REACTION_MORPHOLOGY_"
            "AND_SYNTHETIC_CAMERA_NOISE_POSTOPEN_OBSERVABLE_PROBE_ONLY"
        ),
        "configuration_private": {
            "root": str(root.resolve()),
            "config_path": str(config_path.resolve()),
            "development_report_path": str(
                development_report_path.resolve()
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
            "headroom_expert_bank_matches_public_summary": True,
            "selector_fit_and_oof_selection_use_risk_train_only": True,
            "validation_and_calibration_are_postopen_diagnostics": True,
            "opened_fresh_not_loaded": True,
            "initial_normal_uses_one_exact_adjoint": all(
                row["logical_initial_adjoint_calls_per_sample"] == 1
                and row["batch_invocations"]["forward"] == 0
                for row in execution
            ),
        },
        "expert_bank": {
            "class_candidates": class_candidates,
            "family_to_expert_non_deployable": family_map,
            "baseline_candidate_id": baseline_id,
        },
        "feature_schema_summary": {
            key: {
                "feature_count": len(names),
                "feature_names": list(names),
            }
            for key, names in feature_names.items()
        },
        "features_private": {
            split: {
                key: values.tolist()
                for key, values in split_features.items()
            }
            for split, split_features in features.items()
        },
        "train_oof_selection": selections,
        "transfer_classification": transfer_classification,
        "transfer_route_summary": transfer_route_summary,
        "transfer_metric_rows_private": transfer_rows,
        "paired_gain_summary": paired_summaries,
        "execution": execution,
        "runtime": {
            "wall_seconds": float(time.perf_counter() - started),
            "maximum_rss_bytes": int(_max_rss_bytes()),
        },
        "claim_boundary": {
            "postopen_development_diagnostic_only": True,
            "current_probe_is_an_integrated_solver": False,
            "fresh_values_loaded": False,
            "family_label_is_available_at_deployment": False,
            "calibration_is_still_untouched": False,
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
            "psu_b0_observable_morphology_probe_v1.json"
        ),
    )
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument("--headroom-private-report", type=Path, required=True)
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--public-output", type=Path)
    args = parser.parse_args()
    private, public = run_probe(
        root=args.root,
        config_path=args.config,
        development_report_path=args.development_report,
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
