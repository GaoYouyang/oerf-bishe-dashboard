#!/usr/bin/env python3
"""Screen deployable predictors of JACRU representation mismatch.

This opened-development diagnostic isolates the mismatch between continuous
analytic-gradient observations and the voxel finite-difference/trilinear
operator.  It does not model finite aperture, ray bending, calibration drift,
optical flow, or real BOST data.  OOD/fresh/final data are never constructed.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo_t16_operator.interface_baselines import cgls_baseline  # noqa: E402
from demo_t16_operator.jacru_n1_5_approximation_error import (  # noqa: E402
    StandardizedRidge,
    fit_standardized_ridge,
    pca_oracle_prediction,
    visible_feature_blocks,
)
from demo_t16_operator.jacru_synthetic_fixture import (  # noqa: E402
    JACRUSyntheticCase,
    build_jacru_synthetic_case,
)
from demo_t16_operator.psu_b0_streaming_operator import (  # noqa: E402
    zero_outer_boundary_support,
)
from site_tools import run_jacru_m2_learned_residual_gate as m2  # noqa: E402


DEFAULT_CONFIG = (
    ROOT
    / "demo_t16_operator/configs/"
    "jacru_n1_5_approximation_error_headroom_development_v1.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "demo_t16_operator/results/"
    "jacru_n1_5_approximation_error_headroom_development_scratch"
)
REPORT_SCHEMA = "jacru-n1-5-approximation-error-headroom-report-1.0"


@dataclass(frozen=True)
class CaseRecord:
    partition: str
    family: str
    base_seed: int
    case: JACRUSyntheticCase
    signal_scale: float
    mismatch_normalized: torch.Tensor
    observation_normalized: torch.Tensor
    features: Mapping[str, tuple[tuple[str, ...], torch.Tensor]]
    warm_forward_calls: int
    warm_adjoint_calls: int
    visible_projection_forward_calls: int
    evaluator_truth_forward_calls: int
    warm_seconds: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replace-output", action="store_true")
    parser.add_argument("--seed-limit", type=int)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_checksums(output: Path) -> None:
    files = sorted(
        path for path in output.iterdir() if path.is_file() and path.name != "checksums.sha256"
    )
    (output / "checksums.sha256").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )


def _source_manifest(config_path: Path, config: Mapping[str, Any]) -> dict[str, str]:
    paths = [
        config_path,
        ROOT / str(config["source_t0_config"]),
        ROOT / "demo_t16_operator/jacru_synthetic_fixture.py",
        ROOT / "demo_t16_operator/interface_baselines.py",
        ROOT / "demo_t16_operator/jacru_n1_5_approximation_error.py",
        Path(__file__).resolve(),
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"source manifest is incomplete: {missing}")
    return {str(path.relative_to(ROOT)): _sha256(path) for path in paths}


def _validate_config(config: Mapping[str, Any], source: Mapping[str, Any], seed_limit: int | None) -> None:
    if config.get("status") != "DEVELOPMENT_ONLY_OPENED_NOT_CONFIRMATORY":
        raise RuntimeError("runner accepts only the explicit opened-development config")
    if config.get("may_construct_or_evaluate_ood") is not False:
        raise RuntimeError("OOD construction must remain disabled")
    if seed_limit is not None and seed_limit < 1:
        raise ValueError("seed-limit must be positive")
    fit = {int(value) for value in config["fit"]["base_seeds"]}
    calibration = {int(value) for value in config["calibration"]["base_seeds"]}
    development = {int(value) for value in config["development"]["base_seeds"]}
    source_train = {int(value) for value in source["splits"]["train"]["base_seeds"]}
    source_development = {
        int(value) for value in source["splits"]["development"]["base_seeds"]
    }
    if fit & calibration or fit | calibration != source_train:
        raise ValueError("fit/calibration must be a disjoint complete partition of train seeds")
    if development != source_development:
        raise ValueError("development seed contract drifted")
    families = tuple(str(value) for value in config["families"])
    if families != tuple(str(value) for value in source["splits"]["train"]["families"]):
        raise ValueError("family contract drifted")
    if bool(config["sensor_model"]["enable_noise"]) or bool(
        config["sensor_model"]["enable_camera_bias"]
    ):
        raise ValueError("Stage-A must isolate representation mismatch from sensor nuisance")
    warm = config["warm_start"]
    iterations = int(warm["iterations"])
    if iterations != int(warm["forward_calls"]) or iterations != int(warm["adjoint_calls"]):
        raise ValueError("warm-start call budget drifted")
    feature_sets = tuple(str(value) for value in config["ridge"]["feature_sets"])
    expected_features = (
        "geometry_only",
        "geometry_observation",
        "geometry_signal",
        "curvature_visible",
    )
    if feature_sets != expected_features:
        raise ValueError("ridge feature ablation contract drifted")
    alphas = [float(value) for value in config["ridge"]["alphas"]]
    if not alphas or any(not math.isfinite(value) or value < 0.0 for value in alphas):
        raise ValueError("ridge alphas must be finite and nonnegative")
    if config["pca_oracle"].get("available_to_deployable_predictor") is not False:
        raise ValueError("PCA oracle must remain evaluator-only")
    if config["pca_oracle"].get("participates_in_gate") is not False:
        raise ValueError("PCA oracle cannot participate in the route gate")


def _operator_maps(operator):
    def forward(field: torch.Tensor) -> torch.Tensor:
        return operator(field[None, None])[0]

    def adjoint(observation: torch.Tensor) -> torch.Tensor:
        return operator.adjoint(observation[None])[0, 0]

    return forward, adjoint


def _partition_seeds(config: Mapping[str, Any], partition: str, seed_limit: int | None) -> list[int]:
    seeds = [int(value) for value in config[partition]["base_seeds"]]
    return seeds if seed_limit is None else seeds[:seed_limit]


def _prepare_records(
    config: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    seed_limit: int | None,
) -> tuple[list[CaseRecord], list[dict[str, Any]]]:
    clean_source = json.loads(json.dumps(source))
    clean_source["fixture"]["enable_noise"] = False
    clean_source["fixture"]["enable_camera_bias"] = False
    fixture = m2._fixture_config(clean_source)
    support = zero_outer_boundary_support(fixture.grid_shape, dtype=torch.float64)
    spacing = m2._spacing(fixture)
    iterations = int(config["warm_start"]["iterations"])
    records: list[CaseRecord] = []
    manifest: list[dict[str, Any]] = []

    for partition in ("fit", "calibration", "development"):
        split = str(config[partition]["split"])
        for base_seed in _partition_seeds(config, partition, seed_limit):
            geometry_digest: str | None = None
            for family in config["families"]:
                case = build_jacru_synthetic_case(
                    family=str(family), split=split, base_seed=base_seed, config=fixture
                )
                geometry = case.inference.geometry
                if geometry_digest is None:
                    geometry_digest = geometry.digest
                elif geometry.digest != geometry_digest:
                    raise RuntimeError("families in one geometry cluster must share geometry")
                operator = case.inference.operator
                observation = case.evaluation.clean_observations_uv[0]
                truth = case.evaluation.truth_volume[0, 0]

                operator.reset_call_counts()
                voxel_truth = operator(truth[None, None])[0]
                evaluator_calls = operator.call_report()
                if evaluator_calls != {"forward_calls": 1, "adjoint_calls": 0}:
                    raise RuntimeError("truth-only evaluator call contract drifted")
                mismatch = observation - voxel_truth
                signal_scale = float(torch.sqrt(torch.mean(observation.square())).clamp_min(1e-12))

                forward, adjoint = _operator_maps(operator)
                operator.reset_call_counts()
                started = time.perf_counter()
                warm = cgls_baseline(
                    observation,
                    forward=forward,
                    adjoint=adjoint,
                    support=support,
                    spacing_xyz=spacing,
                    iterations=iterations,
                )
                warm_seconds = time.perf_counter() - started
                warm_calls = operator.call_report()
                if warm_calls != {"forward_calls": iterations, "adjoint_calls": iterations}:
                    raise RuntimeError("warm-start call contract drifted")
                operator.reset_call_counts()
                warm_projection = operator(warm.field[None, None])[0]
                visible_calls = operator.call_report()
                if visible_calls != {"forward_calls": 1, "adjoint_calls": 0}:
                    raise RuntimeError("visible feature projection call contract drifted")
                features = visible_feature_blocks(
                    geometry=geometry,
                    observation_uv=observation,
                    warm_projection_uv=warm_projection,
                )
                records.append(
                    CaseRecord(
                        partition=partition,
                        family=str(family),
                        base_seed=base_seed,
                        case=case,
                        signal_scale=signal_scale,
                        mismatch_normalized=mismatch / signal_scale,
                        observation_normalized=observation / signal_scale,
                        features=features,
                        warm_forward_calls=iterations,
                        warm_adjoint_calls=iterations,
                        visible_projection_forward_calls=1,
                        evaluator_truth_forward_calls=1,
                        warm_seconds=warm_seconds,
                    )
                )
                manifest.append(
                    {
                        "partition": partition,
                        "source_split": split,
                        "base_seed": base_seed,
                        "family": str(family),
                        "case_id": case.inference.case_id,
                        "geometry_digest": geometry.digest,
                        "geometry_cluster_is_independent_unit": True,
                        "sensor_noise_enabled": False,
                        "camera_bias_enabled": False,
                        "truth_used_only_for_mismatch_evaluator_target": True,
                    }
                )
    return records, manifest


def _stack(records: list[CaseRecord], partition: str, feature_set: str) -> tuple[tuple[str, ...], torch.Tensor, torch.Tensor]:
    selected = [record for record in records if record.partition == partition]
    names = selected[0].features[feature_set][0]
    if any(record.features[feature_set][0] != names for record in selected):
        raise RuntimeError("feature names drifted across records")
    features = torch.cat([record.features[feature_set][1] for record in selected], dim=0)
    targets = torch.cat([record.mismatch_normalized.reshape(-1) for record in selected])
    return names, features, targets


def _fixed_predictors(records: list[CaseRecord]) -> dict[str, dict[str, Any]]:
    fit = [record for record in records if record.partition == "fit"]
    mismatch = torch.stack([record.mismatch_normalized for record in fit])
    observation = torch.stack([record.observation_normalized for record in fit])
    camera_count = fit[0].case.inference.geometry.camera_count
    rows, columns = fit[0].case.inference.geometry.detector_shape
    shaped = mismatch.reshape(len(fit), camera_count, rows, columns, 2)
    component_mean = torch.mean(mismatch, dim=(0, 1))
    slot_component_mean = torch.mean(shaped, dim=(0, 2, 3))
    local_detector_component_mean = torch.mean(shaped, dim=(0, 1))
    denominator = torch.sum(observation.square(), dim=(0, 1)).clamp_min(1e-20)
    damping = torch.sum(observation * mismatch, dim=(0, 1)) / denominator
    return {
        "zero": {"kind": "zero"},
        "global_component_mean": {
            "kind": "global_component_mean",
            "value": component_mean,
        },
        "nominal_slot_component_mean": {
            "kind": "nominal_slot_component_mean",
            "value": slot_component_mean,
        },
        "local_detector_component_mean": {
            "kind": "local_detector_component_mean",
            "value": local_detector_component_mean,
        },
        "component_damping": {"kind": "component_damping", "value": damping},
    }


def _predict_fixed(spec: Mapping[str, Any], record: CaseRecord) -> torch.Tensor:
    kind = str(spec["kind"])
    target = record.mismatch_normalized
    if kind == "zero":
        return torch.zeros_like(target)
    if kind == "global_component_mean":
        return torch.as_tensor(spec["value"]).reshape(1, 2).expand_as(target)
    geometry = record.case.inference.geometry
    camera_count = geometry.camera_count
    rows, columns = geometry.detector_shape
    if kind == "nominal_slot_component_mean":
        value = torch.as_tensor(spec["value"]).reshape(camera_count, 1, 2)
        return value.expand(camera_count, rows * columns, 2).reshape_as(target)
    if kind == "local_detector_component_mean":
        value = torch.as_tensor(spec["value"]).reshape(1, rows, columns, 2)
        return value.expand(camera_count, rows, columns, 2).reshape_as(target)
    if kind == "component_damping":
        return record.observation_normalized * torch.as_tensor(spec["value"]).reshape(1, 2)
    raise ValueError(f"unknown fixed predictor: {kind}")


def _case_metric(record: CaseRecord, candidate_id: str, prediction: torch.Tensor) -> dict[str, Any]:
    target = record.mismatch_normalized
    residual = target - prediction.reshape_as(target)
    target_norm = torch.linalg.vector_norm(target).clamp_min(1e-30)
    ratio = float(torch.linalg.vector_norm(residual) / target_norm)
    correction_norm = float(torch.linalg.vector_norm(prediction) / target_norm)
    cosine_denominator = torch.linalg.vector_norm(prediction) * target_norm
    cosine = 0.0 if float(cosine_denominator) <= 1e-30 else float(
        torch.sum(prediction * target) / cosine_denominator
    )
    return {
        "partition": record.partition,
        "base_seed": record.base_seed,
        "geometry_digest": record.case.inference.geometry.digest,
        "family": record.family,
        "case_id": record.case.inference.case_id,
        "candidate_id": candidate_id,
        "mismatch_relative_l2_to_continuous": float(
            torch.linalg.vector_norm(target)
            / torch.linalg.vector_norm(record.observation_normalized).clamp_min(1e-30)
        ),
        "residual_ratio_to_uncorrected_mismatch": ratio,
        "gain_vs_zero": 1.0 - ratio,
        "correction_norm_ratio_to_mismatch": correction_norm,
        "prediction_target_cosine": cosine,
        "warm_forward_calls": record.warm_forward_calls,
        "warm_adjoint_calls": record.warm_adjoint_calls,
        "visible_projection_forward_calls": record.visible_projection_forward_calls,
        "evaluator_truth_forward_calls": record.evaluator_truth_forward_calls,
        "warm_seconds": record.warm_seconds,
    }


def _cluster_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    keys = sorted({(row["partition"], row["candidate_id"], int(row["base_seed"])) for row in rows})
    for partition, candidate_id, base_seed in keys:
        selected = [
            row for row in rows
            if row["partition"] == partition
            and row["candidate_id"] == candidate_id
            and int(row["base_seed"]) == base_seed
        ]
        output.append(
            {
                "partition": partition,
                "candidate_id": candidate_id,
                "base_seed": base_seed,
                "geometry_digest": selected[0]["geometry_digest"],
                "family_count": len(selected),
                "mean_residual_ratio": float(np.mean([row["residual_ratio_to_uncorrected_mismatch"] for row in selected])),
                "mean_gain_vs_zero": float(np.mean([row["gain_vs_zero"] for row in selected])),
                "worst_family_gain_vs_zero": float(min(row["gain_vs_zero"] for row in selected)),
            }
        )
    return output


def _fit_ridge_candidates(
    records: list[CaseRecord], config: Mapping[str, Any]
) -> tuple[dict[str, StandardizedRidge], list[dict[str, Any]]]:
    selected: dict[str, StandardizedRidge] = {}
    calibration_rows: list[dict[str, Any]] = []
    for feature_set in config["ridge"]["feature_sets"]:
        names, fit_x, fit_y = _stack(records, "fit", str(feature_set))
        candidates: list[tuple[float, StandardizedRidge, float, float]] = []
        _, calibration_x, _ = _stack(records, "calibration", str(feature_set))
        calibration_records = [record for record in records if record.partition == "calibration"]
        offsets = np.cumsum([0] + [record.mismatch_normalized.numel() for record in calibration_records])
        for alpha in config["ridge"]["alphas"]:
            model = fit_standardized_ridge(
                fit_x, fit_y, feature_names=names, alpha=float(alpha)
            )
            predictions = model.predict(calibration_x)
            rows = [
                _case_metric(
                    record,
                    f"ridge_{feature_set}",
                    predictions[offsets[index] : offsets[index + 1]].reshape_as(record.mismatch_normalized),
                )
                for index, record in enumerate(calibration_records)
            ]
            clusters = _cluster_rows(rows)
            mean_gain = float(np.mean([row["mean_gain_vs_zero"] for row in clusters]))
            worst_gain = float(min(row["mean_gain_vs_zero"] for row in clusters))
            calibration_rows.append(
                {
                    "feature_set": str(feature_set),
                    "alpha": float(alpha),
                    "geometry_cluster_count": len(clusters),
                    "mean_geometry_cluster_gain_vs_zero": mean_gain,
                    "worst_geometry_cluster_gain_vs_zero": worst_gain,
                    "selected": False,
                }
            )
            candidates.append((float(alpha), model, mean_gain, worst_gain))
        alpha, model, _, _ = max(candidates, key=lambda value: (value[2], value[3], -value[0]))
        selected[str(feature_set)] = model
        for row in calibration_rows:
            if row["feature_set"] == feature_set and float(row["alpha"]) == alpha:
                row["selected"] = True
    return selected, calibration_rows


def _evaluate_deployable(
    records: list[CaseRecord],
    fixed: Mapping[str, Mapping[str, Any]],
    ridge: Mapping[str, StandardizedRidge],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.partition == "fit":
            continue
        for candidate_id, spec in fixed.items():
            rows.append(_case_metric(record, candidate_id, _predict_fixed(spec, record)))
        for feature_set, model in ridge.items():
            names, matrix = record.features[feature_set]
            if names != model.feature_names:
                raise RuntimeError("feature name contract drifted at evaluation")
            prediction = model.predict(matrix).reshape_as(record.mismatch_normalized)
            rows.append(_case_metric(record, f"ridge_{feature_set}", prediction))
    return rows


def _simple_baseline_from_calibration(
    cluster_rows: list[dict[str, Any]], fixed_ids: set[str]
) -> str:
    candidates = []
    for candidate_id in sorted(fixed_ids - {"zero"}):
        rows = [
            row for row in cluster_rows
            if row["partition"] == "calibration" and row["candidate_id"] == candidate_id
        ]
        candidates.append(
            (
                float(np.mean([row["mean_gain_vs_zero"] for row in rows])),
                float(min(row["mean_gain_vs_zero"] for row in rows)),
                candidate_id,
            )
        )
    return max(candidates)[2]


def _summaries(
    case_rows: list[dict[str, Any]],
    cluster_rows: list[dict[str, Any]],
    *,
    simple_baseline: str,
    ridge_models: Mapping[str, StandardizedRidge],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    gates = config["decision_gates"]
    candidate_ids = sorted({str(row["candidate_id"]) for row in case_rows})
    output: list[dict[str, Any]] = []
    baseline_clusters = {
        int(row["base_seed"]): row
        for row in cluster_rows
        if row["partition"] == "development" and row["candidate_id"] == simple_baseline
    }
    baseline_cases = {
        str(row["case_id"]): row
        for row in case_rows
        if row["partition"] == "development" and row["candidate_id"] == simple_baseline
    }
    for candidate_id in candidate_ids:
        dev_clusters = [
            row for row in cluster_rows
            if row["partition"] == "development" and row["candidate_id"] == candidate_id
        ]
        dev_cases = [
            row for row in case_rows
            if row["partition"] == "development" and row["candidate_id"] == candidate_id
        ]
        calibration_cases = [
            row for row in case_rows
            if row["partition"] == "calibration" and row["candidate_id"] == candidate_id
        ]
        relative_cluster_gains = [
            1.0 - float(row["mean_residual_ratio"]) / max(
                float(baseline_clusters[int(row["base_seed"])]["mean_residual_ratio"]), 1e-30
            )
            for row in dev_clusters
        ]
        family_relative: dict[str, float] = {}
        for family in config["families"]:
            gains = []
            for row in dev_cases:
                if row["family"] != family:
                    continue
                baseline = baseline_cases[str(row["case_id"])]
                gains.append(
                    1.0
                    - float(row["residual_ratio_to_uncorrected_mismatch"])
                    / max(float(baseline["residual_ratio_to_uncorrected_mismatch"]), 1e-30)
                )
            family_relative[str(family)] = float(np.mean(gains))
        case_harms = []
        for row in dev_cases:
            baseline = baseline_cases[str(row["case_id"])]
            relative_gain = 1.0 - float(row["residual_ratio_to_uncorrected_mismatch"]) / max(
                float(baseline["residual_ratio_to_uncorrected_mismatch"]), 1e-30
            )
            case_harms.append(
                relative_gain
                < -float(gates["visible_case_harm_over_simple_baseline_threshold"])
            )
        calibration_ratios = sorted(
            float(row["residual_ratio_to_uncorrected_mismatch"]) for row in calibration_cases
        )
        envelope_index = max(0, math.ceil(0.95 * len(calibration_ratios)) - 1)
        envelope_q95 = calibration_ratios[envelope_index]
        coverage = float(np.mean([
            float(row["residual_ratio_to_uncorrected_mismatch"]) <= envelope_q95
            for row in dev_cases
        ]))
        is_visible_ridge = candidate_id.startswith("ridge_")
        passed = bool(
            is_visible_ridge
            and float(np.mean(relative_cluster_gains))
            >= float(gates["visible_mean_gain_over_frozen_simple_baseline_minimum"])
            and min(relative_cluster_gains)
            >= float(gates["visible_worst_geometry_gain_over_frozen_simple_baseline_minimum"])
            and min(family_relative.values())
            >= float(gates["visible_each_family_gain_over_frozen_simple_baseline_minimum"])
            and float(np.mean(case_harms))
            <= float(gates["visible_case_harm_rate_maximum"])
        )
        model = ridge_models.get(candidate_id.removeprefix("ridge_"))
        output.append(
            {
                "candidate_id": candidate_id,
                "predictor_kind": "visible_ridge" if is_visible_ridge else "fixed_simple_control",
                "selected_alpha": None if model is None else model.alpha,
                "development_geometry_cluster_count": len(dev_clusters),
                "development_case_count": len(dev_cases),
                "development_mean_gain_vs_zero": float(np.mean([row["mean_gain_vs_zero"] for row in dev_clusters])),
                "development_worst_geometry_gain_vs_zero": float(min(row["mean_gain_vs_zero"] for row in dev_clusters)),
                "frozen_simple_baseline": simple_baseline,
                "development_mean_gain_over_simple_baseline": float(np.mean(relative_cluster_gains)),
                "development_worst_geometry_gain_over_simple_baseline": float(min(relative_cluster_gains)),
                "smooth_no_interface_gain_over_simple_baseline": family_relative.get("smooth_no_interface"),
                "single_interface_gain_over_simple_baseline": family_relative.get("single_interface"),
                "case_harm_rate_over_simple_baseline": float(np.mean(case_harms)),
                "calibration_residual_ratio_q95_descriptive": envelope_q95,
                "development_case_coverage_under_calibration_q95_descriptive": coverage,
                "forward_screen_passed": passed,
                "n1_5_b_field_and_h1_gates_not_yet_run": True,
            }
        )
    return output


def _pca_oracle_rows(records: list[CaseRecord], ranks: list[int]) -> list[dict[str, Any]]:
    fit = [record.mismatch_normalized.reshape(-1) for record in records if record.partition == "fit"]
    training = torch.stack(fit)
    maximum_rank = min(training.shape[0] - 1, training.shape[1])
    eligible_ranks = [rank for rank in ranks if 0 <= int(rank) <= maximum_rank]
    if not eligible_ranks:
        raise RuntimeError("no PCA oracle rank is eligible for the available fit cases")
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.partition != "development":
            continue
        target = record.mismatch_normalized.reshape(-1)
        for rank in eligible_ranks:
            prediction = pca_oracle_prediction(
                training_vectors=training, target_vector=target, rank=int(rank)
            )
            ratio = float(
                torch.linalg.vector_norm(target - prediction)
                / torch.linalg.vector_norm(target).clamp_min(1e-30)
            )
            rows.append(
                {
                    "base_seed": record.base_seed,
                    "geometry_digest": record.case.inference.geometry.digest,
                    "family": record.family,
                    "case_id": record.case.inference.case_id,
                    "rank": int(rank),
                    "residual_ratio_to_uncorrected_mismatch": ratio,
                    "gain_vs_zero": 1.0 - ratio,
                    "uses_fresh_exact_mismatch_coefficients": True,
                    "deployable": False,
                    "participates_in_gate": False,
                }
            )
    return rows


def _plot(
    summaries: list[dict[str, Any]], pca_rows: list[dict[str, Any]], output: Path
) -> None:
    ordered = sorted(
        summaries, key=lambda row: float(row["development_mean_gain_vs_zero"]), reverse=True
    )
    labels = [str(row["candidate_id"]).replace("ridge_", "R ") for row in ordered]
    y = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), constrained_layout=True)
    means = [float(row["development_mean_gain_vs_zero"]) for row in ordered]
    worst = [float(row["development_worst_geometry_gain_vs_zero"]) for row in ordered]
    axes[0].barh(y - 0.18, means, height=0.34, label="mean cluster", color="#1d6f78")
    axes[0].barh(y + 0.18, worst, height=0.34, label="worst cluster", color="#c05640")
    axes[0].axvline(0.0, color="#202020", linewidth=1)
    axes[0].set_yticks(y, labels, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("mismatch gain vs no correction")
    axes[0].legend()

    ridge = [row for row in ordered if row["predictor_kind"] == "visible_ridge"]
    ridge_labels = [str(row["candidate_id"]).replace("ridge_", "") for row in ridge]
    axes[1].barh(
        np.arange(len(ridge)),
        [float(row["development_mean_gain_over_simple_baseline"]) for row in ridge],
        color="#517a3a",
    )
    axes[1].axvline(0.10, color="#202020", linewidth=1, linestyle="--")
    axes[1].axvline(0.0, color="#202020", linewidth=1)
    axes[1].set_yticks(np.arange(len(ridge)), ridge_labels, fontsize=8)
    axes[1].set_xlabel("mean gain over frozen simple baseline")

    ranks = sorted({int(row["rank"]) for row in pca_rows})
    means_by_rank = [
        float(np.mean([
            row["residual_ratio_to_uncorrected_mismatch"]
            for row in pca_rows if int(row["rank"]) == rank
        ]))
        for rank in ranks
    ]
    axes[2].plot(ranks, means_by_rank, marker="o", color="#8a5a20")
    axes[2].set_xlabel("train-PCA oracle rank")
    axes[2].set_ylabel("mean residual ratio")
    axes[2].set_title("Evaluator-only ceiling (not deployable)")
    fig.suptitle("N1.5-A synthetic representation-mismatch headroom", fontsize=15)
    fig.savefig(output / "diagnostic.png", dpi=180)
    fig.savefig(output / "diagnostic.pdf")
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    config_path = args.config.resolve()
    config = _read_json(config_path)
    source = _read_json(ROOT / str(config["source_t0_config"]))
    _validate_config(config, source, args.seed_limit)
    output = args.output_dir.resolve()
    if output.exists():
        if not args.replace_output:
            raise FileExistsError(f"output already exists: {output}")
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    started = time.perf_counter()
    source_hashes = _source_manifest(config_path, config)
    records, manifest = _prepare_records(
        config, source, seed_limit=args.seed_limit
    )
    fixed = _fixed_predictors(records)
    ridge_models, calibration_rows = _fit_ridge_candidates(records, config)
    case_rows = _evaluate_deployable(records, fixed, ridge_models)
    cluster_rows = _cluster_rows(case_rows)
    simple_baseline = _simple_baseline_from_calibration(cluster_rows, set(fixed))
    summaries = _summaries(
        case_rows,
        cluster_rows,
        simple_baseline=simple_baseline,
        ridge_models=ridge_models,
        config=config,
    )
    pca_rows = _pca_oracle_rows(records, [int(value) for value in config["pca_oracle"]["ranks"]])
    routed = [row for row in summaries if bool(row["forward_screen_passed"])]
    status = "FORWARD_HEADROOM_ONLY_REQUIRES_N1_5_B" if routed else "NO_GO_VISIBLE_FORWARD_PREDICTOR"

    _write_csv(output / "case_metrics.csv", case_rows)
    _write_csv(output / "geometry_cluster_metrics.csv", cluster_rows)
    _write_csv(output / "calibration_alpha_rows.csv", calibration_rows)
    _write_csv(output / "candidate_summary.csv", summaries)
    _write_csv(output / "pca_oracle_rows.csv", pca_rows)
    _write_csv(output / "case_manifest.csv", manifest)
    model_payload = {
        feature_set: {
            "feature_names": list(model.feature_names),
            "feature_mean": model.feature_mean.tolist(),
            "feature_scale": model.feature_scale.tolist(),
            "coefficients": model.coefficients.tolist(),
            "intercept": float(model.intercept),
            "alpha": model.alpha,
        }
        for feature_set, model in ridge_models.items()
    }
    (output / "selected_ridge_models.json").write_text(
        json.dumps(model_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    summary = {
        "schema": REPORT_SCHEMA,
        "status": status,
        "evidence_level": config["evidence_level"],
        "fit_geometry_cluster_count": len({record.base_seed for record in records if record.partition == "fit"}),
        "calibration_geometry_cluster_count": len({record.base_seed for record in records if record.partition == "calibration"}),
        "development_geometry_cluster_count": len({record.base_seed for record in records if record.partition == "development"}),
        "field_family_count_per_cluster": len(config["families"]),
        "independent_unit": "base_seed_geometry_cluster",
        "ray_rows_are_not_independent_samples": True,
        "frozen_simple_baseline": simple_baseline,
        "forward_screen_routes": [row["candidate_id"] for row in routed],
        "n1_5_b_started": False,
        "pca_oracle_participates_in_gate": False,
        "mismatch_scope": "continuous analytic-gradient renderer minus voxel FD/trilinear forward projection",
        "excluded_physics": ["finite aperture", "depth of field", "ray bending", "calibration drift", "optical flow"],
        "opens_ood_fresh_or_final": False,
        "seed_limit": args.seed_limit,
        "runtime_seconds": time.perf_counter() - started,
        "candidate_summaries": summaries,
        "pca_oracle_summary": [
            {
                "rank": rank,
                "mean_residual_ratio": float(np.mean([
                    row["residual_ratio_to_uncorrected_mismatch"]
                    for row in pca_rows if int(row["rank"]) == rank
                ])),
                "worst_residual_ratio": float(max(
                    row["residual_ratio_to_uncorrected_mismatch"]
                    for row in pca_rows if int(row["rank"]) == rank
                )),
                "deployable": False,
            }
            for rank in sorted({int(row["rank"]) for row in pca_rows})
        ],
        "claim_boundary": config["claim_boundary"],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    provenance = {
        "schema": "jacru-n1-5-approximation-error-provenance-1.0",
        "git_commit_at_start": _git_commit(),
        "source_sha256": source_hashes,
        "config": config,
        "exact_cli": " ".join(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    _plot(summaries, pca_rows, output)
    readme = f"""# N1.5-A representation-mismatch headroom screen\n\nStatus: **{status}**.\n\nThis is an opened synthetic development diagnostic. It measures only the mismatch between the continuous analytic-gradient renderer and the voxel finite-difference/trilinear operator. It is not a complete optical camera model, real BOST evidence, or a confirmatory result.\n\n- Independent units: {summary['fit_geometry_cluster_count']} fit, {summary['calibration_geometry_cluster_count']} calibration, and {summary['development_geometry_cluster_count']} development geometry clusters. Two phantom families share each geometry and are not counted as independent rigs.\n- Sensor noise and camera bias are disabled to keep representation mismatch separate from sensor covariance.\n- Frozen simple comparator selected on calibration only: `{simple_baseline}`.\n- Deployable ridge inputs: geometry, measured observation, and/or a 12-pair CGLS warm projection. Exact mismatch, truth, clean labels, family labels, and development targets are absent from inference.\n- PCA rows use each fresh exact mismatch coefficient and are evaluator-only representational ceilings. They never enter the route gate.\n- N1.5-B field/H1 reconstruction gates have not been run.\n- OOD, fresh, final, experimental, finite-aperture, and ray-bending claims remain closed.\n"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    _write_checksums(output)
    print(json.dumps({
        "status": status,
        "simple_baseline": simple_baseline,
        "routes": summary["forward_screen_routes"],
        "output": str(output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
