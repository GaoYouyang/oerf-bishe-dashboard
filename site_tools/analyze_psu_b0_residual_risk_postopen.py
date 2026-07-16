#!/usr/bin/env python3
"""Diagnose the opened OCRRG fresh failures without creating new evidence.

This script deliberately treats the frozen fresh audit as post-open diagnostic
material. It regenerates only first-stage truth-free features, joins them to the
already frozen per-sample metrics, and explains which observable strata the
pooled linear risk model missed. Its output must not be used as a fresh result.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from demo_t16_operator.psu_b0_residual_risk import (
    RISK_FEATURE_NAMES,
    RidgeRiskFit,
    observable_risk_features,
    one_sided_conformal_quantile,
)
from demo_t16_operator.psu_b0_spectral_preconditioner import FixedSobolevDirection
from demo_t16_operator.psu_b0_streaming_operator import zero_outer_boundary_support
from site_tools.run_psu_b0_real_interface_audit import (
    _make_operator,
    load_real_support_geometry,
)
from site_tools.run_psu_b0_residual_risk_fresh import (
    _build_fresh_splits,
    _load_frozen_models,
    _load_risk_fit,
)
from site_tools.run_psu_b0_spectral_preconditioner_pilot import _load_json


PRIVATE_SCHEMA = "psu-b0-residual-risk-postopen-diagnosis-private-1.0"
PUBLIC_SCHEMA = "psu-b0-residual-risk-postopen-diagnosis-public-1.0"
STATUS = "POSTOPEN_DIAGNOSIS_ONLY_NEXT_CANDIDATE_NOT_FROZEN"

SPECTRAL_STRESS_FEATURES = {
    "direction_relative_correction": 1.0,
    "candidate_log_gain_span": 1.0,
    "gradient_spectral_centroid": -1.0,
    "gradient_high_frequency_fraction": -1.0,
}
CORRELATED_CAMERA_STRESS_FEATURES = {
    "white_component_correlation_abs": 1.0,
    "gradient_log_rms": -1.0,
    "white_rms_active_mean": -1.0,
    "gradient_axis_anisotropy": 1.0,
}


def _feature_index(name: str) -> int:
    return RISK_FEATURE_NAMES.index(str(name))


def _branch_score(
    standardized_features: np.ndarray,
    definition: dict[str, float],
) -> np.ndarray:
    values = np.asarray(standardized_features, dtype=np.float64)
    terms = [
        float(sign) * values[:, _feature_index(name)]
        for name, sign in definition.items()
    ]
    return np.mean(np.stack(terms, axis=1), axis=1)


def summarize_view_strata(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize trusted rows by active-view count."""

    trusted = [row for row in rows if bool(row["trusted"])]
    output = []
    for view_count in sorted({int(row["active_view_count"]) for row in trusted}):
        group = [
            row for row in trusted if int(row["active_view_count"]) == view_count
        ]
        gain = np.asarray(
            [float(row["actual_gain_percent"]) for row in group],
            dtype=np.float64,
        )
        output.append(
            {
                "active_view_count": view_count,
                "accepted_row_count": len(group),
                "mean_gain_percent": float(np.mean(gain)),
                "p10_gain_percent": float(np.quantile(gain, 0.10)),
                "minimum_gain_percent": float(np.min(gain)),
                "harm_over_one_percent_count": int(np.sum(gain < -1.0)),
                "harm_over_one_percent_rate": float(np.mean(gain < -1.0)),
            }
        )
    return output


def feature_contrasts(
    harmful: list[dict[str, Any]],
    safe: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rank observable feature shifts between harmful and safe accepted rows."""

    if not harmful or not safe:
        raise ValueError("harmful and safe accepted rows must both be nonempty")
    harm = np.asarray(
        [row["standardized_features"] for row in harmful],
        dtype=np.float64,
    )
    accepted_safe = np.asarray(
        [row["standardized_features"] for row in safe],
        dtype=np.float64,
    )
    safe_scale = np.std(accepted_safe, axis=0)
    safe_scale = np.where(safe_scale < 1e-8, 1.0, safe_scale)
    effect = (np.mean(harm, axis=0) - np.mean(accepted_safe, axis=0)) / safe_scale
    rows = []
    for index, name in enumerate(RISK_FEATURE_NAMES):
        rows.append(
            {
                "feature": name,
                "harmful_mean_training_z": float(np.mean(harm[:, index])),
                "safe_mean_training_z": float(np.mean(accepted_safe[:, index])),
                "safe_std_training_z": float(np.std(accepted_safe[:, index])),
                "harmful_vs_safe_effect": float(effect[index]),
            }
        )
    return sorted(
        rows,
        key=lambda row: abs(float(row["harmful_vs_safe_effect"])),
        reverse=True,
    )


def calibration_view_support(
    development_feature_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Count distinct fields and pooled candidate rows in each view stratum."""

    output = []
    for split in ("risk_train", "risk_validation", "risk_calibration"):
        split_rows = [
            row for row in development_feature_rows if row["split"] == split
        ]
        for view_count in range(6, 10):
            group = [
                row
                for row in split_rows
                if int(row["active_view_count"]) == view_count
            ]
            output.append(
                {
                    "split": split,
                    "active_view_count": view_count,
                    "distinct_field_count": len(
                        {str(row["sample_id"]) for row in group}
                    ),
                    "pooled_candidate_row_count": len(group),
                }
            )
    return output


def exact_view_conformal_probe(
    *,
    fit: RidgeRiskFit,
    development_feature_rows: list[dict[str, Any]],
    fresh_rows: list[dict[str, Any]],
    pooled_quantile_by_seed: dict[int, float],
    alpha: float,
    distance_threshold: float,
    minimum_lower_gain_percent: float,
) -> dict[str, Any]:
    """Post-open probe of conservative exact-view conformal quantiles.

    The probe is descriptive only. It does not modify or re-score the frozen
    audit and is not a valid new candidate because the fresh failures are open.
    """

    calibration = [
        row
        for row in development_feature_rows
        if row["split"] == "risk_calibration"
    ]
    quantile_records = []
    exact_quantiles: dict[tuple[int, int], float | None] = {}
    for seed in sorted(pooled_quantile_by_seed):
        for view_count in range(6, 10):
            group = [
                row
                for row in calibration
                if int(row["seed"]) == int(seed)
                and int(row["active_view_count"]) == view_count
            ]
            if len(group) >= 2:
                features = np.asarray(
                    [row["features"] for row in group],
                    dtype=np.float64,
                )
                actual = np.asarray(
                    [row["actual_gain_percent"] for row in group],
                    dtype=np.float64,
                )
                raw = one_sided_conformal_quantile(
                    fit.predict(features),
                    actual,
                    alpha=float(alpha),
                )
                conservative = max(float(pooled_quantile_by_seed[seed]), raw)
            else:
                raw = None
                conservative = None
            exact_quantiles[(seed, view_count)] = conservative
            quantile_records.append(
                {
                    "seed": int(seed),
                    "active_view_count": view_count,
                    "calibration_row_count": len(group),
                    "pooled_quantile": float(pooled_quantile_by_seed[seed]),
                    "exact_view_quantile": raw,
                    "conservative_max_quantile": conservative,
                    "sufficient_for_probe": len(group) >= 2,
                }
            )

    current_accepted = [row for row in fresh_rows if bool(row["trusted"])]
    evaluated = []
    for row in current_accepted:
        seed = int(row["seed"])
        view_count = int(row["active_view_count"])
        quantile = exact_quantiles.get((seed, view_count))
        if quantile is None:
            trust = False
            reason = "insufficient_exact_view_calibration"
        else:
            lower = float(row["predicted_gain_percent"]) - float(quantile)
            trust = (
                lower >= float(minimum_lower_gain_percent)
                and float(row["feature_distance"]) <= float(distance_threshold)
            )
            reason = "exact_view_quantile_available"
        evaluated.append(
            {
                "split": row["split"],
                "sample_id": row["sample_id"],
                "seed": seed,
                "active_view_count": view_count,
                "actual_gain_percent": float(row["actual_gain_percent"]),
                "postopen_probe_trusted": bool(trust),
                "reason": reason,
            }
        )
    harmful = [
        row for row in evaluated if float(row["actual_gain_percent"]) < -1.0
    ]
    return {
        "quantiles": quantile_records,
        "current_accepted_row_count": len(current_accepted),
        "probe_accepted_row_count": int(
            sum(bool(row["postopen_probe_trusted"]) for row in evaluated)
        ),
        "current_accepted_harm_count": len(harmful),
        "harmful_rows_rejected_by_exact_view_probe": int(
            sum(not bool(row["postopen_probe_trusted"]) for row in harmful)
        ),
        "harmful_rows_still_accepted_by_exact_view_probe": int(
            sum(bool(row["postopen_probe_trusted"]) for row in harmful)
        ),
        "interpretation": (
            "Exact-view quantiles alone do not repair the two observed "
            "six-view failure modes; sparse eight/nine-view calibration also "
            "forces abstention rather than informative conditional control."
        ),
    }


def build_public_summary(private: dict[str, Any]) -> dict[str, Any]:
    """Remove local paths, all feature rows, model weights, and checkpoints."""

    return {
        "schema_version": PUBLIC_SCHEMA,
        "status": private["status"],
        "evidence_scope": private["evidence_scope"],
        "source_audit": copy.deepcopy(private["source_audit_public"]),
        "accepted_harm": copy.deepcopy(private["accepted_harm_public"]),
        "view_strata": copy.deepcopy(private["view_strata"]),
        "calibration_view_support": copy.deepcopy(
            private["calibration_view_support"]
        ),
        "feature_contrasts": copy.deepcopy(private["feature_contrasts"]),
        "failure_modes": copy.deepcopy(private["failure_modes"]),
        "support_order_mismatch": copy.deepcopy(
            private["support_order_mismatch"]
        ),
        "exact_view_conformal_probe": copy.deepcopy(
            private["exact_view_conformal_probe"]
        ),
        "next_candidate_hypothesis": copy.deepcopy(
            private["next_candidate_hypothesis"]
        ),
        "literature_boundary": copy.deepcopy(private["literature_boundary"]),
        "claim_boundary": copy.deepcopy(private["claim_boundary"]),
        "public_export_policy": {
            "contains_local_paths": False,
            "contains_all_per_sample_feature_rows": False,
            "contains_model_weights_or_checkpoints": False,
            "contains_real_psu_measurement_values": False,
            "contains_opened_fresh_failure_examples": True,
        },
    }


def _metric_lookup(
    metric_rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    lookup = {}
    for row in metric_rows:
        key = (str(row["split"]), str(row["sample_id"]), str(row["method"]))
        if key in lookup:
            raise ValueError(f"duplicate metric row: {key}")
        lookup[key] = row
    return lookup


def _first_stage_feature_pair(
    *,
    split: Any,
    operator: Any,
    candidate: Any,
    fallback: Any,
    rays_per_view: int,
    device: torch.device,
    batch_size: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deployed and support-projected first-stage feature matrices."""

    deployed_rows = []
    projected_rows = []
    support = operator.support[None, None]
    operator.reset_call_counts()
    with torch.no_grad():
        for start in range(0, len(split.truth), int(batch_size)):
            observation = split.observation_uv[start : start + batch_size].to(device)
            sigma = split.sigma_by_view[start : start + batch_size].to(device)
            mask = split.view_mask[start : start + batch_size].to(device)
            active = mask.repeat_interleave(int(rays_per_view), dim=1)[:, :, None]
            expanded_sigma = sigma.repeat_interleave(
                int(rays_per_view),
                dim=1,
            )[:, :, None]
            residual = active * observation
            gradient = operator.adjoint(residual / expanded_sigma.square())
            candidate_direction, diagnostics = candidate(
                gradient,
                residual_uv=residual,
                sigma_by_view=sigma,
                view_mask=mask,
                rays_per_view=rays_per_view,
                stage_fraction=0.25,
            )
            fallback_direction, _ = fallback(
                gradient,
                residual_uv=residual,
                sigma_by_view=sigma,
                view_mask=mask,
                rays_per_view=rays_per_view,
                stage_fraction=0.25,
            )
            common = {
                "gradient": gradient,
                "residual_uv": residual,
                "sigma_by_view": sigma,
                "view_mask": mask,
                "rays_per_view": rays_per_view,
                "candidate_diagnostics": diagnostics,
            }
            deployed_rows.append(
                observable_risk_features(
                    **common,
                    candidate_direction=candidate_direction,
                    fallback_direction=fallback_direction,
                )
                .detach()
                .cpu()
                .numpy()
            )
            projected_rows.append(
                observable_risk_features(
                    **common,
                    candidate_direction=candidate_direction * support,
                    fallback_direction=fallback_direction * support,
                )
                .detach()
                .cpu()
                .numpy()
            )
    return (
        np.concatenate(deployed_rows, axis=0),
        np.concatenate(projected_rows, axis=0),
    )


def _regenerate_feature_rows(
    *,
    preregistration: dict[str, Any],
    development_config: dict[str, Any],
    source_config: dict[str, Any],
    development_report: dict[str, Any],
    fresh_report: dict[str, Any],
    view_root: Path,
    checkpoint_dir: Path,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grid_size = int(preregistration["geometry"]["grid_size"])
    rays_per_view = int(preregistration["geometry"]["rays_per_view"])
    support = zero_outer_boundary_support((grid_size,) * 3).to(device)
    true_geometry, _ = load_real_support_geometry(
        view_root,
        rays_per_view=rays_per_view,
        sample_count=int(
            preregistration["geometry"]["true_finite_aperture_sample_count"]
        ),
    )
    nominal_geometry, _ = load_real_support_geometry(
        view_root,
        rays_per_view=rays_per_view,
        sample_count=int(
            preregistration["geometry"]["nominal_finite_aperture_sample_count"]
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
    models, _ = _load_frozen_models(
        checkpoint_dir=checkpoint_dir,
        source_config=source_config,
        development_report=development_report,
        preregistration=preregistration,
        device=device,
    )
    fallback = FixedSobolevDirection(
        (grid_size,) * 3,
        strength=float(
            preregistration["frozen_source"]["selected_sobolev_strength"]
        ),
    ).to(device)
    splits, _, _ = _build_fresh_splits(
        preregistration=preregistration,
        development_config=development_config,
        source_config=source_config,
        true_operator=true_operator,
        nominal_operator=nominal_operator,
        device=device,
    )
    fit = _load_risk_fit(development_report)
    metrics = _metric_lookup(fresh_report["dataset_private"]["per_sample_metrics"])
    rows = []
    maximum_prediction_difference = 0.0
    maximum_distance_difference = 0.0
    for split_name, wrapped in splits.items():
        for seed, model in sorted(models.items()):
            deployed_features, projected_features = _first_stage_feature_pair(
                split=wrapped.data,
                operator=nominal_operator,
                candidate=model,
                fallback=fallback,
                rays_per_view=rays_per_view,
                device=device,
            )
            standardized = (
                deployed_features - fit.feature_mean
            ) / fit.feature_scale
            projected_standardized = (
                projected_features - fit.feature_mean
            ) / fit.feature_scale
            spectral_stress = _branch_score(
                standardized,
                SPECTRAL_STRESS_FEATURES,
            )
            correlated_stress = _branch_score(
                standardized,
                CORRELATED_CAMERA_STRESS_FEATURES,
            )
            predictions = fit.predict(deployed_features)
            distances = fit.distance(deployed_features)
            projected_predictions = fit.predict(projected_features)
            projected_distances = fit.distance(projected_features)
            for index, sample_id in enumerate(wrapped.data.sample_ids):
                baseline = metrics[
                    (split_name, sample_id, "sobolev_selected")
                ]
                raw = metrics[(split_name, sample_id, f"raw_seed_{seed}")]
                gated = metrics[(split_name, sample_id, f"gated_seed_{seed}")]
                baseline_error = float(baseline["field_relative_l2"])
                actual_gain = 100.0 * (
                    baseline_error - float(raw["field_relative_l2"])
                ) / max(baseline_error, 1e-12)
                maximum_prediction_difference = max(
                    maximum_prediction_difference,
                    abs(
                        float(predictions[index])
                        - float(gated["predicted_gain_percent"])
                    ),
                )
                maximum_distance_difference = max(
                    maximum_distance_difference,
                    abs(
                        float(distances[index])
                        - float(gated["feature_distance"])
                    ),
                )
                rows.append(
                    {
                        "split": split_name,
                        "sample_id": sample_id,
                        "seed": int(seed),
                        "family": wrapped.data.families[index],
                        "noise_profile": wrapped.noise_profiles[index],
                        "relative_noise": float(
                            wrapped.data.relative_noise[index]
                        ),
                        "active_view_count": int(
                            torch.sum(wrapped.data.view_mask[index] > 0.5)
                        ),
                        "trusted": bool(gated["trusted"]),
                        "actual_gain_percent": float(actual_gain),
                        "predicted_gain_percent": float(predictions[index]),
                        "lower_gain_bound_percent": float(
                            gated["lower_gain_bound_percent"]
                        ),
                        "feature_distance": float(distances[index]),
                        "spectral_correction_stress": float(
                            spectral_stress[index]
                        ),
                        "correlated_camera_stress": float(
                            correlated_stress[index]
                        ),
                        "features": deployed_features[index].tolist(),
                        "standardized_features": standardized[index].tolist(),
                        "support_projected_features": (
                            projected_features[index].tolist()
                        ),
                        "support_projected_standardized_features": (
                            projected_standardized[index].tolist()
                        ),
                        "support_projected_prediction_percent": float(
                            projected_predictions[index]
                        ),
                        "support_projected_feature_distance": float(
                            projected_distances[index]
                        ),
                    }
                )
    return rows, {
        "regenerated_row_count": len(rows),
        "maximum_prediction_difference": float(maximum_prediction_difference),
        "maximum_distance_difference": float(maximum_distance_difference),
    }


def support_order_mismatch(
    *,
    rows: list[dict[str, Any]],
    pooled_quantile_by_seed: dict[int, float],
    distance_threshold: float,
    minimum_lower_gain_percent: float,
) -> dict[str, Any]:
    """Audit calibration/deployment feature-order drift in the frozen source."""

    decision_disagreements = []
    prediction_shifts = []
    distance_shifts = []
    feature_shifts = []
    for row in rows:
        seed = int(row["seed"])
        projected_prediction = float(
            row["support_projected_prediction_percent"]
        )
        projected_distance = float(row["support_projected_feature_distance"])
        projected_trust = (
            6 <= int(row["active_view_count"]) <= 9
            and projected_prediction - float(pooled_quantile_by_seed[seed])
            >= float(minimum_lower_gain_percent)
            and projected_distance <= float(distance_threshold)
        )
        deployed_trust = bool(row["trusted"])
        prediction_shifts.append(
            float(row["predicted_gain_percent"]) - projected_prediction
        )
        distance_shifts.append(
            float(row["feature_distance"]) - projected_distance
        )
        feature_shifts.append(
            float(
                np.max(
                    np.abs(
                        np.asarray(row["features"], dtype=np.float64)
                        - np.asarray(
                            row["support_projected_features"],
                            dtype=np.float64,
                        )
                    )
                )
            )
        )
        if projected_trust != deployed_trust:
            decision_disagreements.append(
                {
                    "split": row["split"],
                    "sample_id": row["sample_id"],
                    "seed": seed,
                    "active_view_count": int(row["active_view_count"]),
                    "deployed_trust": deployed_trust,
                    "support_projected_trust": bool(projected_trust),
                    "actual_gain_percent": float(row["actual_gain_percent"]),
                }
            )
    return {
        "calibration_feature_order": (
            "candidate and fallback directions multiplied by support before "
            "direction-derived risk features"
        ),
        "deployment_feature_order": (
            "direction-derived risk features computed before the solver "
            "multiplies the selected direction by support"
        ),
        "same_scoring_function_for_calibration_and_deployment": False,
        "fresh_row_count": len(rows),
        "decision_disagreement_count": len(decision_disagreements),
        "decision_disagreement_rate": float(
            len(decision_disagreements) / max(len(rows), 1)
        ),
        "maximum_absolute_prediction_shift_percent": float(
            np.max(np.abs(prediction_shifts))
        ),
        "mean_absolute_prediction_shift_percent": float(
            np.mean(np.abs(prediction_shifts))
        ),
        "maximum_absolute_distance_shift": float(
            np.max(np.abs(distance_shifts))
        ),
        "maximum_absolute_raw_feature_shift": float(np.max(feature_shifts)),
        "disagreements": decision_disagreements,
        "interpretation": (
            "The frozen empirical reconstruction metrics remain reproducible, "
            "but the split-conformal interpretation is not valid until "
            "calibration and deployment use one identical feature function."
        ),
    }


def run_diagnosis(
    *,
    root: Path,
    preregistration_path: Path,
    development_report_path: Path,
    fresh_report_path: Path,
    view_root: Path,
    checkpoint_dir: Path,
    device_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    preregistration = _load_json(preregistration_path)
    development_report = _load_json(development_report_path)
    fresh_report = _load_json(fresh_report_path)
    development_config = _load_json(
        root / str(preregistration["frozen_source"]["development_config"])
    )
    source_config = _load_json(
        root / str(development_config["source_pilot"]["config"])
    )
    device = torch.device(device_name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    torch.set_num_threads(8)
    feature_rows, regeneration = _regenerate_feature_rows(
        preregistration=preregistration,
        development_config=development_config,
        source_config=source_config,
        development_report=development_report,
        fresh_report=fresh_report,
        view_root=view_root,
        checkpoint_dir=checkpoint_dir,
        device=device,
    )
    accepted = [row for row in feature_rows if bool(row["trusted"])]
    harmful = [
        row for row in accepted if float(row["actual_gain_percent"]) < -1.0
    ]
    safe = [
        row for row in accepted if float(row["actual_gain_percent"]) >= -1.0
    ]
    if len(harmful) != 4:
        raise ValueError("opened fresh report no longer has four accepted harm rows")
    contrast = feature_contrasts(harmful, safe)
    top_features = [str(row["feature"]) for row in contrast[:8]]
    accepted_harm_public = []
    for row in harmful:
        z_lookup = dict(zip(RISK_FEATURE_NAMES, row["standardized_features"]))
        accepted_harm_public.append(
            {
                "split": row["split"],
                "sample_id": row["sample_id"],
                "seed": int(row["seed"]),
                "family": row["family"],
                "noise_profile": row["noise_profile"],
                "active_view_count": int(row["active_view_count"]),
                "actual_gain_percent": float(row["actual_gain_percent"]),
                "predicted_gain_percent": float(row["predicted_gain_percent"]),
                "lower_gain_bound_percent": float(
                    row["lower_gain_bound_percent"]
                ),
                "feature_distance": float(row["feature_distance"]),
                "spectral_correction_stress": float(
                    row["spectral_correction_stress"]
                ),
                "correlated_camera_stress": float(
                    row["correlated_camera_stress"]
                ),
                "selected_training_z": {
                    name: float(z_lookup[name]) for name in top_features
                },
            }
        )
    development_features = development_report["dataset_private"]["feature_rows"]
    pooled_quantiles = {
        int(seed): float(value)
        for seed, value in development_report["risk_model_public"][
            "calibration_overprediction_quantile_by_seed"
        ].items()
    }
    distance_threshold = float(
        development_report["risk_model_public"]["distance_threshold"]
    )
    minimum_lower_gain_percent = float(
        development_report["risk_model_public"][
            "selected_minimum_lower_gain_percent"
        ]
    )
    order_mismatch = support_order_mismatch(
        rows=feature_rows,
        pooled_quantile_by_seed=pooled_quantiles,
        distance_threshold=distance_threshold,
        minimum_lower_gain_percent=minimum_lower_gain_percent,
    )
    fit = _load_risk_fit(development_report)
    exact_view_probe = exact_view_conformal_probe(
        fit=fit,
        development_feature_rows=development_features,
        fresh_rows=feature_rows,
        pooled_quantile_by_seed=pooled_quantiles,
        alpha=float(development_report["risk_model_public"]["conformal_alpha"]),
        distance_threshold=distance_threshold,
        minimum_lower_gain_percent=minimum_lower_gain_percent,
    )
    private = {
        "schema_version": PRIVATE_SCHEMA,
        "status": STATUS,
        "evidence_scope": (
            "POSTOPEN_ANALYSIS_OF_FROZEN_SYNTHETIC_FRESH_AUDIT_ON_REAL_PSU_"
            "SUPPORT_GEOMETRY_NO_NEW_CONFIRMATORY_EVIDENCE"
        ),
        "configuration_private": {
            "root": str(root.resolve()),
            "preregistration_path": str(preregistration_path.resolve()),
            "development_report_path": str(development_report_path.resolve()),
            "fresh_report_path": str(fresh_report_path.resolve()),
            "view_root": str(view_root.resolve()),
            "checkpoint_dir": str(checkpoint_dir.resolve()),
            "device": device_name,
        },
        "source_audit_public": {
            "source_status": fresh_report["status"],
            "fresh_protocol_was_frozen_before_opening": True,
            "fresh_values_are_open_for_this_diagnosis": True,
            "regenerated_first_stage_feature_rows": len(feature_rows),
            "accepted_row_count": len(accepted),
            "accepted_harm_over_one_percent_count": len(harmful),
            "regeneration_consistency": regeneration,
        },
        "feature_rows_private": feature_rows,
        "accepted_harm_public": accepted_harm_public,
        "view_strata": summarize_view_strata(feature_rows),
        "calibration_view_support": calibration_view_support(
            development_features
        ),
        "feature_contrasts": contrast,
        "failure_modes": [
            {
                "name": "low_frequency_plume_aggressive_correction",
                "source_samples": ["fresh_iid_support-012"],
                "observable_signature": (
                    "low gradient spectral centroid/high-frequency fraction "
                    "combined with above-average candidate correction and gain span"
                ),
                "physical_interpretation": (
                    "A smooth plume leaves a weakly constrained low-frequency "
                    "inverse direction; the learned correction can reduce the "
                    "data objective while worsening the three-dimensional field."
                ),
                "deployment_labels_allowed": False,
            },
            {
                "name": "correlated_camera_noise_shock_confusion",
                "source_samples": ["fresh_correlated_noise_ood-011"],
                "observable_signature": (
                    "large cross-component residual correlation with unusually "
                    "small whitened residual/gradient scale and high anisotropy"
                ),
                "physical_interpretation": (
                    "Structured camera noise aligns with a shock-like gradient "
                    "and appears easier than it is under IID-style whitening."
                ),
                "deployment_labels_allowed": False,
            },
        ],
        "support_order_mismatch": order_mismatch,
        "exact_view_conformal_probe": exact_view_probe,
        "next_candidate_hypothesis": {
            "working_name": "Observable Multi-Veto Residual-Risk Gate",
            "architecture": [
                "retain the pooled lower-gain predictor only as one vote",
                "add a spectral/correction stress veto",
                "add a correlated-camera stress veto",
                "apply a separately calibrated six-view risk backoff",
                "exactly fall back to validation-selected Sobolev on any veto",
            ],
            "development_requirements": [
                "use one support-order feature function in training, "
                "calibration, validation, and deployment",
                "balance six, seven, eight, and nine-view fields by construction",
                "hold out complete morphology families during model selection",
                "calibrate each veto without morphology or noise-profile labels",
                "compare against exact-view Mondrian, nonlinear ridge features, "
                "and a shallow tree under identical development data",
                "freeze all scores, thresholds, tie-breaks, seeds, and failure "
                "criteria before generating a new independent repeat",
            ],
            "not_yet_a_candidate": True,
        },
        "literature_boundary": [
            {
                "title": "Conformal Risk Control",
                "url": "https://research.google/pubs/conformal-risk-control/",
                "lesson": (
                    "finite-sample control requires a declared risk family and "
                    "exchangeable calibration; arbitrary OOD is not covered"
                ),
            },
            {
                "title": "Learn then Test",
                "url": "https://arxiv.org/abs/2110.01052",
                "lesson": (
                    "multiple candidate thresholds should be calibrated as a "
                    "risk-control selection problem, not chosen on the audit set"
                ),
            },
            {
                "title": "Automatically Adaptive Conformal Risk Control",
                "url": "https://proceedings.mlr.press/v258/blot25a.html",
                "lesson": (
                    "difficulty-adaptive conditioning is relevant, but a small "
                    "BOST calibration set cannot support unrestricted conditioning"
                ),
            },
            {
                "title": "Confidence on the Focal",
                "url": (
                    "https://academic.oup.com/jrsssb/article/87/4/1239/8113856"
                ),
                "lesson": (
                    "selection changes the target guarantee; a gate needs "
                    "selection-aware evaluation rather than marginal coverage alone"
                ),
            },
        ],
        "claim_boundary": {
            "postopen_threshold_tuning_is_confirmatory": False,
            "frozen_conformal_interpretation_valid": False,
            "new_candidate_frozen": False,
            "new_independent_repeat_run": False,
            "experimental_field_truth_used": False,
            "real_psu_measurement_values_used": False,
            "analytic_morphology_is_cfd": False,
            "camera_noise_is_measured_psu_noise": False,
            "algorithm_superiority": False,
        },
    }
    return private, build_public_summary(private)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument("--fresh-report", type=Path, required=True)
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--public-output", type=Path)
    args = parser.parse_args()
    private, public = run_diagnosis(
        root=args.root,
        preregistration_path=args.preregistration,
        development_report_path=args.development_report,
        fresh_report_path=args.fresh_report,
        view_root=args.view_root,
        checkpoint_dir=args.checkpoint_dir,
        device_name=args.device,
    )
    if args.private_output is not None:
        args.private_output.parent.mkdir(parents=True, exist_ok=True)
        args.private_output.write_text(
            json.dumps(private, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.public_output is not None:
        args.public_output.parent.mkdir(parents=True, exist_ok=True)
        args.public_output.write_text(
            json.dumps(public, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(public, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
