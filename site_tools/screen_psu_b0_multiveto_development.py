#!/usr/bin/env python3
"""Screen the v2 multi-veto hypothesis on opened development evidence."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from demo_t16_operator.psu_b0_multiveto_risk import (
    MULTIVETO_FEATURE_SCHEMA,
    observable_stress_scores,
)
from demo_t16_operator.psu_b0_residual_risk import RidgeRiskFit


PRIVATE_SCHEMA = "psu-b0-multiveto-development-screen-private-1.0"
PUBLIC_SCHEMA = "psu-b0-multiveto-development-screen-public-1.0"
STATUS = "DEVELOPMENT_SCREEN_PARTIAL_MECHANISM_SIGNAL_NOT_READY_TO_FREEZE"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("input JSON must be an object")
    return value


def _fit(report: dict[str, Any]) -> RidgeRiskFit:
    private = report["risk_model_private"]
    public = report["risk_model_public"]
    return RidgeRiskFit(
        feature_mean=np.asarray(private["feature_mean"], dtype=np.float64),
        feature_scale=np.asarray(private["feature_scale"], dtype=np.float64),
        coefficients=np.asarray(private["coefficients"], dtype=np.float64),
        intercept=float(private["intercept"]),
        ridge_lambda=float(public["ridge_lambda"]),
        validation_rmse=float(public["validation_rmse"]),
    )


def _development_arrays(
    report: dict[str, Any],
    *,
    split: str,
    fit: RidgeRiskFit,
) -> dict[str, np.ndarray]:
    rows = [
        row
        for row in report["dataset_private"]["feature_rows"]
        if row["split"] == split
    ]
    features = np.asarray([row["features"] for row in rows], dtype=np.float64)
    standardized = (features - fit.feature_mean) / fit.feature_scale
    spectral, camera = observable_stress_scores(standardized)
    return {
        "features": features,
        "gain": np.asarray(
            [row["actual_gain_percent"] for row in rows],
            dtype=np.float64,
        ),
        "seed": np.asarray([row["seed"] for row in rows], dtype=np.int64),
        "views": np.asarray(
            [row["active_view_count"] for row in rows],
            dtype=np.int64,
        ),
        "spectral": np.asarray(spectral, dtype=np.float64),
        "camera": np.asarray(camera, dtype=np.float64),
    }


def selection_metrics(
    gain: np.ndarray,
    trust: np.ndarray,
    views: np.ndarray,
) -> dict[str, Any]:
    values = np.asarray(gain, dtype=np.float64)
    accepted = np.asarray(trust, dtype=bool)
    view_count = np.asarray(views, dtype=np.int64)
    selected = np.where(accepted, values, 0.0)
    output: dict[str, Any] = {
        "row_count": len(values),
        "accepted_row_count": int(np.sum(accepted)),
        "coverage": float(np.mean(accepted)),
        "mean_selected_gain_percent": float(np.mean(selected)),
        "p10_selected_gain_percent": float(np.quantile(selected, 0.10)),
        "harm_over_one_percent_count": int(np.sum(selected < -1.0)),
        "harm_over_one_percent_rate": float(np.mean(selected < -1.0)),
        "accepted_minimum_raw_gain_percent": (
            None if not np.any(accepted) else float(np.min(values[accepted]))
        ),
    }
    output["by_active_views"] = []
    for active_views in range(6, 10):
        group = view_count == active_views
        group_selected = selected[group]
        group_trust = accepted[group]
        output["by_active_views"].append(
            {
                "active_view_count": active_views,
                "row_count": int(np.sum(group)),
                "coverage": (
                    0.0 if not np.any(group) else float(np.mean(group_trust))
                ),
                "harm_over_one_percent_rate": (
                    0.0
                    if not np.any(group)
                    else float(np.mean(group_selected < -1.0))
                ),
            }
        )
    return output


def _score(
    *,
    data: dict[str, np.ndarray],
    fit: RidgeRiskFit,
    quantile_by_seed: dict[int, float],
    distance_threshold: float,
    minimum_lower_gain_percent: float,
    spectral_threshold: float,
    camera_threshold: float,
    six_view_extra_margin_percent: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    prediction = fit.predict(data["features"])
    distance = fit.distance(data["features"])
    quantile = np.asarray(
        [quantile_by_seed[int(seed)] for seed in data["seed"]],
        dtype=np.float64,
    )
    required = float(minimum_lower_gain_percent) + (
        data["views"] == 6
    ) * float(six_view_extra_margin_percent)
    trust = (
        (prediction - quantile >= required)
        & (distance <= float(distance_threshold))
        & (data["spectral"] <= float(spectral_threshold))
        & (data["camera"] <= float(camera_threshold))
    )
    return trust, selection_metrics(data["gain"], trust, data["views"])


def select_candidate(
    *,
    validation: dict[str, np.ndarray],
    fit: RidgeRiskFit,
    quantile_by_seed: dict[int, float],
    distance_threshold: float,
    minimum_lower_gain_percent: float,
    spectral_grid: list[float],
    camera_grid: list[float],
    six_view_backoff_grid: list[float],
    coverage_minimum: float,
    overall_harm_maximum: float,
    per_view_harm_maximum: float,
) -> dict[str, Any]:
    candidates = []
    for spectral_threshold in spectral_grid:
        for camera_threshold in camera_grid:
            for six_view_backoff in six_view_backoff_grid:
                _, metrics = _score(
                    data=validation,
                    fit=fit,
                    quantile_by_seed=quantile_by_seed,
                    distance_threshold=distance_threshold,
                    minimum_lower_gain_percent=minimum_lower_gain_percent,
                    spectral_threshold=spectral_threshold,
                    camera_threshold=camera_threshold,
                    six_view_extra_margin_percent=six_view_backoff,
                )
                per_view_worst = max(
                    float(row["harm_over_one_percent_rate"])
                    for row in metrics["by_active_views"]
                )
                admissible = (
                    float(metrics["coverage"]) >= float(coverage_minimum)
                    and float(metrics["harm_over_one_percent_rate"])
                    <= float(overall_harm_maximum)
                    and per_view_worst <= float(per_view_harm_maximum)
                )
                candidates.append(
                    {
                        "spectral_stress_threshold": float(spectral_threshold),
                        "camera_stress_threshold": float(camera_threshold),
                        "six_view_extra_margin_percent": float(
                            six_view_backoff
                        ),
                        "admissible": bool(admissible),
                        "validation": metrics,
                    }
                )
    admissible = [row for row in candidates if row["admissible"]]
    if not admissible:
        raise ValueError("no admissible multi-veto development candidate")
    selected = max(
        admissible,
        key=lambda row: (
            float(row["validation"]["mean_selected_gain_percent"]),
            float(row["validation"]["coverage"]),
            -float(row["spectral_stress_threshold"]),
            -float(row["camera_stress_threshold"]),
            -float(row["six_view_extra_margin_percent"]),
        ),
    )
    return {
        "candidate_count": len(candidates),
        "admissible_candidate_count": len(admissible),
        "selected": selected,
    }


def _fresh_diagnostic(
    *,
    rows: list[dict[str, Any]],
    quantile_by_seed: dict[int, float],
    distance_threshold: float,
    minimum_lower_gain_percent: float,
    selected: dict[str, Any],
) -> dict[str, Any]:
    processed = []
    for row in rows:
        standardized = np.asarray(
            [row["support_projected_standardized_features"]],
            dtype=np.float64,
        )
        spectral, camera = observable_stress_scores(standardized)
        lower = float(row["support_projected_prediction_percent"]) - float(
            quantile_by_seed[int(row["seed"])]
        )
        base_trust = (
            6 <= int(row["active_view_count"]) <= 9
            and lower >= float(minimum_lower_gain_percent)
            and float(row["support_projected_feature_distance"])
            <= float(distance_threshold)
        )
        required = float(minimum_lower_gain_percent) + (
            int(row["active_view_count"]) == 6
        ) * float(selected["six_view_extra_margin_percent"])
        selected_trust = (
            base_trust
            and lower >= required
            and float(spectral[0])
            <= float(selected["spectral_stress_threshold"])
            and float(camera[0])
            <= float(selected["camera_stress_threshold"])
        )
        processed.append(
            {
                "split": row["split"],
                "sample_id": row["sample_id"],
                "seed": int(row["seed"]),
                "active_view_count": int(row["active_view_count"]),
                "actual_gain_percent": float(row["actual_gain_percent"]),
                "canonical_pooled_trust": bool(base_trust),
                "selected_multiveto_trust": bool(selected_trust),
            }
        )
    split_names = (
        "fresh_iid_support",
        "fresh_family_ood",
        "fresh_correlated_noise_ood",
        "fresh_family_noise_ood",
        "fresh_exact_operator_control",
    )
    aggregates = []
    for split in split_names:
        group = [row for row in processed if row["split"] == split]
        gain = np.asarray(
            [row["actual_gain_percent"] for row in group],
            dtype=np.float64,
        )
        views = np.asarray(
            [row["active_view_count"] for row in group],
            dtype=np.int64,
        )
        for name, key in (
            ("canonical_pooled", "canonical_pooled_trust"),
            ("development_selected_multiveto", "selected_multiveto_trust"),
        ):
            trust = np.asarray([row[key] for row in group], dtype=bool)
            aggregates.append(
                {
                    "split": split,
                    "method": name,
                    **selection_metrics(gain, trust, views),
                }
            )
    harmful_base = [
        row
        for row in processed
        if row["canonical_pooled_trust"]
        and float(row["actual_gain_percent"]) < -1.0
    ]
    return {
        "aggregates": aggregates,
        "canonical_pooled_harm_count": len(harmful_base),
        "selected_multiveto_harm_rejected_count": int(
            sum(not row["selected_multiveto_trust"] for row in harmful_base)
        ),
        "selected_multiveto_harm_remaining_count": int(
            sum(row["selected_multiveto_trust"] for row in harmful_base)
        ),
        "remaining_harm_sources": sorted(
            {
                str(row["sample_id"])
                for row in harmful_base
                if row["selected_multiveto_trust"]
            }
        ),
    }


def build_public_summary(private: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": PUBLIC_SCHEMA,
        "status": private["status"],
        "evidence_scope": private["evidence_scope"],
        "feature_contract": private["feature_contract"],
        "screen": copy.deepcopy(private["screen"]),
        "development_evaluation": copy.deepcopy(
            private["development_evaluation"]
        ),
        "opened_fresh_diagnostic": copy.deepcopy(
            private["opened_fresh_diagnostic"]
        ),
        "decision": copy.deepcopy(private["decision"]),
        "claim_boundary": copy.deepcopy(private["claim_boundary"]),
        "public_export_policy": {
            "contains_local_paths": False,
            "contains_feature_rows_or_model_weights": False,
            "contains_real_psu_measurement_values": False,
            "contains_opened_fresh_aggregates": True,
        },
    }


def run_screen(
    *,
    config_path: Path,
    development_report_path: Path,
    postopen_report_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _load(config_path)
    development = _load(development_report_path)
    postopen = _load(postopen_report_path)
    fit = _fit(development)
    train = _development_arrays(development, split="risk_train", fit=fit)
    validation = _development_arrays(
        development,
        split="risk_validation",
        fit=fit,
    )
    calibration = _development_arrays(
        development,
        split="risk_calibration",
        fit=fit,
    )
    public_model = development["risk_model_public"]
    quantile_by_seed = {
        int(seed): float(value)
        for seed, value in public_model[
            "calibration_overprediction_quantile_by_seed"
        ].items()
    }
    quantiles = [
        float(value)
        for value in config["threshold_grid"]["train_stress_quantiles"]
    ]
    spectral_grid = [
        float(value) for value in np.quantile(train["spectral"], quantiles)
    ]
    camera_grid = [
        float(value) for value in np.quantile(train["camera"], quantiles)
    ]
    if config["threshold_grid"]["include_no_veto"]:
        spectral_grid.append(1e9)
        camera_grid.append(1e9)
    selected_screen = select_candidate(
        validation=validation,
        fit=fit,
        quantile_by_seed=quantile_by_seed,
        distance_threshold=float(public_model["distance_threshold"]),
        minimum_lower_gain_percent=float(
            public_model["selected_minimum_lower_gain_percent"]
        ),
        spectral_grid=spectral_grid,
        camera_grid=camera_grid,
        six_view_backoff_grid=[
            float(value)
            for value in config["threshold_grid"][
                "six_view_extra_margin_percent"
            ]
        ],
        coverage_minimum=float(config["selection"]["coverage_minimum"]),
        overall_harm_maximum=float(
            config["selection"][
                "overall_harm_over_one_percent_rate_maximum"
            ]
        ),
        per_view_harm_maximum=float(
            config["selection"][
                "per_view_harm_over_one_percent_rate_maximum"
            ]
        ),
    )
    selected = selected_screen["selected"]
    development_evaluation = {}
    for name, data in (
        ("risk_train", train),
        ("risk_validation", validation),
        ("risk_calibration", calibration),
    ):
        _, metrics = _score(
            data=data,
            fit=fit,
            quantile_by_seed=quantile_by_seed,
            distance_threshold=float(public_model["distance_threshold"]),
            minimum_lower_gain_percent=float(
                public_model["selected_minimum_lower_gain_percent"]
            ),
            spectral_threshold=float(selected["spectral_stress_threshold"]),
            camera_threshold=float(selected["camera_stress_threshold"]),
            six_view_extra_margin_percent=float(
                selected["six_view_extra_margin_percent"]
            ),
        )
        development_evaluation[name] = metrics
    fresh_diagnostic = _fresh_diagnostic(
        rows=postopen["feature_rows_private"],
        quantile_by_seed=quantile_by_seed,
        distance_threshold=float(public_model["distance_threshold"]),
        minimum_lower_gain_percent=float(
            public_model["selected_minimum_lower_gain_percent"]
        ),
        selected=selected,
    )
    ready = (
        fresh_diagnostic["selected_multiveto_harm_remaining_count"] == 0
        and not fresh_diagnostic["remaining_harm_sources"]
    )
    private = {
        "schema_version": PRIVATE_SCHEMA,
        "status": STATUS,
        "evidence_scope": (
            "OPENED_DEVELOPMENT_SELECTION_WITH_OPENED_FRESH_DIAGNOSIS_ONLY_"
            "NO_NEW_INDEPENDENT_REPEAT"
        ),
        "configuration_private": {
            "config_path": str(config_path.resolve()),
            "development_report_path": str(development_report_path.resolve()),
            "postopen_report_path": str(postopen_report_path.resolve()),
        },
        "feature_contract": {
            "schema": MULTIVETO_FEATURE_SCHEMA,
            "support_projection_before_direction_features": True,
        },
        "screen": {
            **selected_screen,
            "spectral_threshold_grid": spectral_grid,
            "camera_threshold_grid": camera_grid,
            "six_view_backoff_grid": config["threshold_grid"][
                "six_view_extra_margin_percent"
            ],
        },
        "development_evaluation": development_evaluation,
        "opened_fresh_diagnostic": fresh_diagnostic,
        "decision": {
            "ready_to_freeze_independent_repeat": bool(ready),
            "selected_veto_catches_correlated_camera_tail": (
                fresh_diagnostic[
                    "selected_multiveto_harm_rejected_count"
                ]
                == 2
            ),
            "selected_veto_catches_low_frequency_plume_tail": False,
            "required_next_action": (
                "expand development with balanced view counts, low-frequency "
                "plume stress, and measured-style correlated camera covariance"
            ),
        },
        "claim_boundary": copy.deepcopy(config["claim_boundary"]),
    }
    return private, build_public_summary(private)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument("--postopen-report", type=Path, required=True)
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--public-output", type=Path)
    args = parser.parse_args()
    private, public = run_screen(
        config_path=args.config,
        development_report_path=args.development_report,
        postopen_report_path=args.postopen_report,
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
