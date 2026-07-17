#!/usr/bin/env python3
"""Run the opened-synthetic JACRU N1.1 flow-off covariance proximal ceiling.

This experiment retrains the frozen T0 proposal exactly as the opened M2 line
did, creates independent flow-off fit/threshold/audit repeats, and evaluates a
dense covariance-weighted anchored Tikhonov ceiling.  Candidate selection never
reads field truth or continuous-clean-target residuals.  Evaluator-only candidates are
explicitly labeled.  Dense ``A A^T`` assembly and factorization remain outside
the deployable budget, so even a passing result is mechanism evidence only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo_t16_operator.jacru_m2_exact_nullspace_oracle import (
    build_exact_dense_nullspace_projector,
)
from demo_t16_operator.jacru_n1_flowoff_covariance import (
    CameraRandomEffectCovariance,
    JACRUFlowOffCalibrationPayload,
    build_flowoff_calibration_payload,
    calibrate_discrepancy_threshold,
    dense_covariance_proximal_discrepancy,
    estimate_camera_random_effect_covariance,
    exact_camera_random_effect_covariance,
    isotropic_covariance_like,
    lock_coverage,
    whitened_quadratic,
)
from site_tools import run_jacru_m2_learned_residual_gate as m2
from site_tools import run_jacru_m2_2_exact_nullspace_oracle as m22


DEFAULT_CONFIG = (
    ROOT
    / "demo_t16_operator/configs/"
    "jacru_n1_1_flowoff_covariance_proximal_postopen_v1.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "demo_t16_operator/results/"
    "jacru_n1_1_flowoff_covariance_proximal_postopen_public"
)
REPORT_SCHEMA = "jacru-n1-1-flowoff-covariance-proximal-postopen-report-1.0"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed-limit", type=int)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _mean(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    if not materialized:
        raise ValueError("cannot average an empty sequence")
    return float(math.fsum(materialized) / len(materialized))


def _validate_sources(config: dict[str, Any]) -> dict[str, Path]:
    mapping = {
        "source_t0_config": ROOT / config["source_t0_config"],
        "source_t0_summary": ROOT / config["source_t0_results"] / "summary.json",
        "source_m2_7_config": ROOT / config["source_m2_7_config"],
        "source_m2_7_summary": ROOT / config["source_m2_7_results"] / "summary.json",
        "source_m2_8_config": ROOT / config["source_m2_8_config"],
        "source_m2_8_summary": ROOT / config["source_m2_8_results"] / "summary.json",
        "source_n1_0_config": ROOT / config["source_n1_0_config"],
        "source_n1_0_summary": ROOT / config["source_n1_0_results"] / "summary.json",
        "implementation_calibration_module": ROOT / config["implementation_calibration_module"],
        "implementation_dense_assembler": ROOT / config["implementation_dense_assembler"],
        "implementation_runner": ROOT / config["implementation_runner"],
    }
    for key, path in mapping.items():
        expected = str(config[f"{key}_sha256"])
        observed = _sha256(path)
        if observed != expected:
            raise RuntimeError(f"{key} hash drifted: {observed} != {expected}")
    n10 = _read_json(mapping["source_n1_0_summary"])
    if n10.get("status") != "N1_0_OBSERVABLE_DISCREPANCY_STOPPING_NO_GO":
        raise RuntimeError("N1.0 source status drifted")
    if not bool(
        n10.get("authorization", {}).get("continue_flow_off_covariance_research", False)
    ):
        raise RuntimeError("N1.0 did not authorize flow-off covariance research")
    return mapping


def _load_matched_baselines(config: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    path = ROOT / config["source_m2_7_results"] / "matched_baseline_rows.csv"
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    selected = [row for row in rows if int(row["projection_iterations"]) == 10]
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in selected:
        kind = str(row["baseline_kind"])
        if kind not in {"cgls_matched", "huber_pdhg_matched"}:
            continue
        key = (str(row["case_id"]), kind)
        if key in lookup:
            raise RuntimeError(f"duplicate matched baseline row: {key}")
        lookup[key] = row
    if len(lookup) != 60:
        raise RuntimeError("expected two K=10 matched baselines for each of 30 cases")
    return lookup


def _score_calibration_samples(
    samples_uv: torch.Tensor,
    *,
    mean_uv: torch.Tensor,
    covariance: torch.Tensor,
    quantile: float,
) -> tuple[float, torch.Tensor]:
    scores = torch.tensor(
        [
            whitened_quadratic(sample - mean_uv, covariance)
            for sample in samples_uv
        ],
        dtype=torch.float64,
    )
    threshold = float(torch.quantile(scores, float(quantile), interpolation="higher"))
    return threshold, scores


def _coverage(
    samples_uv: torch.Tensor,
    *,
    mean_uv: torch.Tensor,
    covariance: torch.Tensor,
    threshold: float,
) -> tuple[float, torch.Tensor]:
    scores = torch.tensor(
        [
            whitened_quadratic(sample - mean_uv, covariance)
            for sample in samples_uv
        ],
        dtype=torch.float64,
    )
    return float(torch.mean((scores <= float(threshold)).to(torch.float64))), scores


def _relative_frobenius(value: torch.Tensor, reference: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value - reference)) / max(
        float(torch.linalg.vector_norm(reference)), 1e-30
    )


def _build_calibration_cache(
    *,
    records: list[m2.PreparedRecord],
    source_config: dict[str, Any],
    config: dict[str, Any],
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    list[dict[str, Any]],
]:
    flow = config["flowoff_calibration"]
    fit_count = int(flow["fit_repeats"])
    threshold_count = int(flow["threshold_calibration_repeats"])
    audit_count = int(flow["audit_repeats"])
    quantile = float(flow["discrepancy_quantile"])
    fixture = source_config["fixture"]
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.split == "train":
            continue
        case = record.case
        clean = case.evaluation.clean_observations_uv
        signal_rms = float(torch.sqrt(torch.mean(clean.square())).clamp_min(1e-12))
        iid_std = float(fixture["noise_relative_std"]) * signal_rms
        bias_std = float(fixture["camera_bias_relative_std"]) * signal_rms
        camera_index = case.inference.geometry.camera_index
        exact_bias_mean = case.evaluation.camera_bias_uv[camera_index]
        for mode in ("paired_static", "unpaired_distribution"):
            payload = build_flowoff_calibration_payload(
                case_id=case.inference.case_id,
                geometry_digest=case.inference.geometry.digest,
                camera_index=camera_index,
                persistent_camera_bias_uv=case.evaluation.camera_bias_uv,
                iid_noise_std=iid_std,
                camera_bias_std=bias_std,
                mode=mode,
                fit_repeats=fit_count,
                selection_repeats=threshold_count,
                lock_repeats=audit_count,
                seed=int(flow["seed"]),
            )
            estimate = estimate_camera_random_effect_covariance(
                **payload.estimator_kwargs(),
                shrinkage=float(config["covariance_estimator"]["shrinkage"]),
                ridge_fraction=float(config["covariance_estimator"]["ridge_fraction"]),
            )
            if mode == "paired_static":
                candidate_mean = estimate.mean_uv
                exact_mean = exact_bias_mean
                exact_covariance = exact_camera_random_effect_covariance(
                    camera_index=camera_index,
                    iid_noise_std=iid_std,
                    camera_bias_std=0.0,
                )
            else:
                candidate_mean = torch.zeros_like(estimate.mean_uv)
                exact_mean = torch.zeros_like(estimate.mean_uv)
                exact_covariance = exact_camera_random_effect_covariance(
                    camera_index=camera_index,
                    iid_noise_std=iid_std,
                    camera_bias_std=bias_std,
                )
            empirical = calibrate_discrepancy_threshold(
                samples_uv=payload.selection_samples_uv,
                estimate=estimate,
                quantile=quantile,
                mean_uv=candidate_mean,
            )
            empirical_coverage, empirical_lock_scores = lock_coverage(
                samples_uv=payload.lock_samples_uv,
                estimate=estimate,
                threshold=empirical.threshold,
                mean_uv=candidate_mean,
            )
            exact_threshold, exact_selection_scores = _score_calibration_samples(
                payload.selection_samples_uv,
                mean_uv=exact_mean,
                covariance=exact_covariance,
                quantile=quantile,
            )
            exact_coverage, exact_lock_scores = _coverage(
                payload.lock_samples_uv,
                mean_uv=exact_mean,
                covariance=exact_covariance,
                threshold=exact_threshold,
            )
            mean_error = float(torch.linalg.vector_norm(candidate_mean - exact_mean))
            mean_error_relative_iid = mean_error / max(
                iid_std * math.sqrt(candidate_mean.numel()), 1e-30
            )
            covariance_error = _relative_frobenius(
                estimate.covariance, exact_covariance
            )
            cache[(case.inference.case_id, mode)] = {
                "payload": payload,
                "estimate": estimate,
                "candidate_mean": candidate_mean,
                "exact_mean": exact_mean,
                "exact_covariance": exact_covariance,
                "empirical_threshold": empirical.threshold,
                "exact_threshold": exact_threshold,
                "iid_std": iid_std,
                "bias_std": bias_std,
            }
            rows.append(
                {
                    "case_id": case.inference.case_id,
                    "split": record.split,
                    "family": record.family,
                    "base_seed": record.base_seed,
                    "geometry_digest": case.inference.geometry.digest,
                    "mode": mode,
                    "payload_digest": payload.payload_digest,
                    "fit_repeats": fit_count,
                    "threshold_calibration_repeats": threshold_count,
                    "audit_repeats": audit_count,
                    "discrepancy_quantile": quantile,
                    "iid_noise_std": iid_std,
                    "camera_bias_std": bias_std,
                    "estimated_condition_number": estimate.condition_number,
                    "estimated_minimum_eigenvalue": estimate.minimum_eigenvalue,
                    "estimated_maximum_eigenvalue": estimate.maximum_eigenvalue,
                    "mean_error_relative_iid": mean_error_relative_iid,
                    "covariance_relative_frobenius_error": covariance_error,
                    "empirical_threshold": empirical.threshold,
                    "empirical_selection_score_mean": empirical.score_mean,
                    "empirical_selection_score_maximum": empirical.score_maximum,
                    "empirical_audit_coverage": empirical_coverage,
                    "empirical_audit_score_mean": float(torch.mean(empirical_lock_scores)),
                    "exact_threshold": exact_threshold,
                    "exact_selection_score_mean": float(torch.mean(exact_selection_scores)),
                    "exact_audit_coverage": exact_coverage,
                    "exact_audit_score_mean": float(torch.mean(exact_lock_scores)),
                }
            )
    expected_rows = 2 * sum(record.split != "train" for record in records)
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"expected paired and unpaired calibration rows for each evaluation case: "
            f"{len(rows)} != {expected_rows}"
        )
    return cache, rows


def _candidate_components(
    *,
    candidate: dict[str, Any],
    calibration: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    mean_policy = str(candidate["mean_policy"])
    covariance_policy = str(candidate["proximal_covariance_policy"])
    selector_policy = str(candidate["selector_covariance_policy"])
    if mean_policy == "estimated_flowoff":
        mean = calibration["candidate_mean"]
    elif mean_policy == "zero":
        mean = torch.zeros_like(calibration["candidate_mean"])
    elif mean_policy == "exact_persistent_bias_oracle":
        mean = calibration["exact_mean"]
    else:
        raise ValueError(f"unsupported mean policy: {mean_policy}")
    if covariance_policy == "estimated_structured":
        proximal = calibration["estimate"].covariance
    elif covariance_policy == "estimated_isotropic":
        proximal = isotropic_covariance_like(calibration["estimate"].covariance)
    elif covariance_policy == "exact_generator_oracle":
        proximal = calibration["exact_covariance"]
    else:
        raise ValueError(f"unsupported proximal covariance policy: {covariance_policy}")
    if selector_policy == "estimated_structured":
        selector = calibration["estimate"].covariance
    elif selector_policy == "estimated_isotropic":
        selector = isotropic_covariance_like(calibration["estimate"].covariance)
    elif selector_policy == "exact_generator_oracle":
        selector = calibration["exact_covariance"]
    else:
        raise ValueError(f"unsupported selector covariance policy: {selector_policy}")
    payload: JACRUFlowOffCalibrationPayload = calibration["payload"]
    threshold, _ = _score_calibration_samples(
        payload.selection_samples_uv,
        mean_uv=mean,
        covariance=selector,
        quantile=float(candidate["discrepancy_quantile"]),
    )
    return mean, proximal, selector, threshold


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["candidate_id"]),
            str(row["method"]),
            int(row["model_seed"]),
            str(row["split"]),
        )
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = groups[key]
        finite_logs = [
            math.log10(float(row["alpha"]))
            for row in group
            if math.isfinite(float(row["alpha"])) and float(row["alpha"]) > 0.0
        ]
        output.append(
            {
                "candidate_id": key[0],
                "method": key[1],
                "model_seed": key[2],
                "split": key[3],
                "case_count": len(group),
                "field_gain_mean": _mean(row["field_gain_to_best_matched"] for row in group),
                "h1_gain_mean": _mean(row["h1_gain_to_best_matched"] for row in group),
                "clean_reprojection_ratio_to_base_mean": _mean(
                    row["clean_reprojection_ratio_to_base"] for row in group
                ),
                "clean_reprojection_ratio_to_base_maximum": max(
                    float(row["clean_reprojection_ratio_to_base"]) for row in group
                ),
                "measured_reprojection_ratio_to_cgls_mean": _mean(
                    row["measured_reprojection_ratio_to_cgls"] for row in group
                ),
                "field_harm_rate": _mean(row["field_harm"] for row in group),
                "worst_field_gain": min(float(row["field_gain_to_best_matched"]) for row in group),
                "target_crossing_rate": _mean(row["target_crossed"] for row in group),
                "raw_no_correction_rate": _mean(row["raw_no_correction"] for row in group),
                "log10_alpha_mean_finite": _mean(finite_logs) if finite_logs else "",
                "residual_closure_relative_error_maximum": max(
                    float(row["residual_closure_relative_error"]) for row in group
                ),
                "correction_norm_mean": _mean(row["correction_norm"] for row in group),
            }
        )
    return output


def _pooled_metrics(
    rows: list[dict[str, Any]],
    *,
    candidate_id: str,
    method: str,
    split: str,
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["candidate_id"] == candidate_id
        and row["method"] == method
        and row["split"] == split
    ]
    if not selected:
        raise RuntimeError("missing pooled candidate rows")
    seed_means = []
    for seed in sorted({int(row["model_seed"]) for row in selected}):
        seed_rows = [row for row in selected if int(row["model_seed"]) == seed]
        seed_means.append(_mean(row["field_gain_to_best_matched"] for row in seed_rows))
    return {
        "row_count": len(selected),
        "field_gain_mean": _mean(row["field_gain_to_best_matched"] for row in selected),
        "h1_gain_mean": _mean(row["h1_gain_to_best_matched"] for row in selected),
        "clean_reprojection_ratio_to_base_mean": _mean(
            row["clean_reprojection_ratio_to_base"] for row in selected
        ),
        "clean_reprojection_ratio_to_base_maximum": max(
            float(row["clean_reprojection_ratio_to_base"]) for row in selected
        ),
        "field_harm_rate": _mean(row["field_harm"] for row in selected),
        "worst_field_gain": min(float(row["field_gain_to_best_matched"]) for row in selected),
        "target_crossing_rate": _mean(row["target_crossed"] for row in selected),
        "residual_closure_relative_error_maximum": max(
            float(row["residual_closure_relative_error"]) for row in selected
        ),
        "all_model_seed_field_gain_means_positive": all(value > 0.0 for value in seed_means),
        "per_model_seed_field_gain_means": seed_means,
    }


def _calibration_decisions(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    gates = config["calibration_gates"]
    result: dict[str, Any] = {}
    target = float(config["flowoff_calibration"]["discrepancy_quantile"])
    for mode in ("paired_static", "unpaired_distribution"):
        selected = [row for row in rows if row["mode"] == mode]
        coverage = _mean(row["empirical_audit_coverage"] for row in selected)
        p90_error = float(
            np.quantile(
                [abs(float(row["empirical_audit_coverage"]) - target) for row in selected],
                0.9,
            )
        )
        max_condition = max(float(row["estimated_condition_number"]) for row in selected)
        checks = {
            "audit_coverage_mean_minimum": coverage >= float(gates["audit_coverage_mean_minimum"]),
            "audit_coverage_p90_error_maximum": p90_error <= float(gates["audit_coverage_p90_error_maximum"]),
            "condition_number_maximum": max_condition <= float(gates["condition_number_maximum"]),
            "covariance_spd": all(float(row["estimated_minimum_eigenvalue"]) > 0.0 for row in selected),
        }
        result[mode] = {
            "audit_coverage_mean": coverage,
            "audit_coverage_p90_absolute_error": p90_error,
            "condition_number_maximum": max_condition,
            "checks": checks,
            "passed": all(checks.values()),
        }
    return result


def _decisions(
    *,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    calibration_decisions: dict[str, Any],
) -> list[dict[str, Any]]:
    gates = config["decision_gates"]
    candidates = {str(value["id"]): value for value in config["candidates"]}
    output: list[dict[str, Any]] = []
    for candidate_id, candidate in candidates.items():
        for method in config["methods"]:
            development = _pooled_metrics(
                rows, candidate_id=candidate_id, method=str(method), split="development"
            )
            ood = _pooled_metrics(
                rows, candidate_id=candidate_id, method=str(method), split="ood"
            )
            mode = str(candidate["calibration_mode"])
            checks = {
                "calibration_valid": bool(calibration_decisions[mode]["passed"]),
                "development_field_gain": development["field_gain_mean"]
                >= float(gates["development_field_gain_minimum"]),
                "development_h1_gain": development["h1_gain_mean"]
                >= float(gates["development_h1_gain_minimum"]),
                "development_clean_mean": development[
                    "clean_reprojection_ratio_to_base_mean"
                ]
                <= float(gates["development_clean_reprojection_ratio_to_base_mean_maximum"]),
                "development_clean_worst": development[
                    "clean_reprojection_ratio_to_base_maximum"
                ]
                <= float(gates["development_clean_reprojection_ratio_to_base_worst_maximum"]),
                "development_harm": development["field_harm_rate"]
                <= float(gates["field_harm_rate_maximum"]),
                "development_worst": development["worst_field_gain"]
                >= float(gates["worst_field_gain_minimum"]),
                "ood_field_gain": ood["field_gain_mean"]
                >= float(gates["ood_field_gain_minimum"]),
                "ood_h1_gain": ood["h1_gain_mean"]
                >= float(gates["ood_h1_gain_minimum"]),
                "ood_clean_mean": ood["clean_reprojection_ratio_to_base_mean"]
                <= float(gates["ood_clean_reprojection_ratio_to_base_mean_maximum"]),
                "ood_clean_worst": ood["clean_reprojection_ratio_to_base_maximum"]
                <= float(gates["ood_clean_reprojection_ratio_to_base_worst_maximum"]),
                "ood_harm": ood["field_harm_rate"]
                <= float(gates["field_harm_rate_maximum"]),
                "ood_worst": ood["worst_field_gain"]
                >= float(gates["worst_field_gain_minimum"]),
                "all_seed_means_positive": development[
                    "all_model_seed_field_gain_means_positive"
                ]
                and ood["all_model_seed_field_gain_means_positive"],
                "target_crossing": min(
                    development["target_crossing_rate"], ood["target_crossing_rate"]
                )
                >= float(gates["minimum_target_crossing_rate"]),
                "residual_closure": max(
                    development["residual_closure_relative_error_maximum"],
                    ood["residual_closure_relative_error_maximum"],
                )
                <= float(gates["maximum_residual_closure_relative_error"]),
            }
            output.append(
                {
                    "candidate_id": candidate_id,
                    "method": str(method),
                    "uses_truth": bool(candidate["uses_truth"]),
                    "uses_exact_nuisance": bool(candidate["uses_exact_nuisance"]),
                    "dense_ceiling_only": True,
                    "calibration_mode": mode,
                    "development": development,
                    "ood": ood,
                    "checks": checks,
                    "passed": all(checks.values()),
                }
            )
    return output


def _plot(
    *,
    rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    output: Path,
) -> None:
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.2), constrained_layout=True)
    colors = {"jacru_m2": "#168b8c", "pooled_cnn": "#df7d42"}
    markers = {
        "paired_isotropic_sensor": "o",
        "paired_structured_sensor": "s",
        "unpaired_structured_sensor": "^",
        "paired_exact_mean_iid_sensor_oracle": "D",
        "unpaired_exact_covariance_sensor_oracle": "P",
        "paired_structured_truth_residual_oracle": "X",
    }
    for decision in decisions:
        dev = decision["development"]
        axes[0, 0].scatter(
            dev["clean_reprojection_ratio_to_base_mean"],
            dev["field_gain_mean"],
            color=colors[decision["method"]],
            marker=markers.get(decision["candidate_id"], "o"),
            s=65,
            alpha=0.85,
        )
        axes[0, 1].scatter(
            dev["field_harm_rate"],
            dev["worst_field_gain"],
            color=colors[decision["method"]],
            marker=markers.get(decision["candidate_id"], "o"),
            s=65,
            alpha=0.85,
        )
    axes[0, 0].axvline(1.10, color="#555555", linestyle="--", linewidth=1)
    axes[0, 0].axhline(0.05, color="#b45144", linestyle="--", linewidth=1)
    axes[0, 0].set_xlabel("development continuous-target residual / base")
    axes[0, 0].set_ylabel("development mean field gain")
    axes[0, 0].set_title("Anchored GLS ceiling: field vs continuous clean target")
    axes[0, 1].axvline(0.05, color="#555555", linestyle="--", linewidth=1)
    axes[0, 1].axhline(-0.05, color="#b45144", linestyle="--", linewidth=1)
    axes[0, 1].set_xlabel("development field harm rate")
    axes[0, 1].set_ylabel("development worst field gain")
    axes[0, 1].set_title("Tail gate")

    modes = ["paired_static", "unpaired_distribution"]
    coverages = [
        _mean(row["empirical_audit_coverage"] for row in calibration_rows if row["mode"] == mode)
        for mode in modes
    ]
    covariance_errors = [
        _mean(
            row["covariance_relative_frobenius_error"]
            for row in calibration_rows
            if row["mode"] == mode
        )
        for mode in modes
    ]
    x = np.arange(len(modes))
    axes[1, 0].bar(x - 0.18, coverages, width=0.36, color="#168b8c", label="audit coverage")
    axes[1, 0].bar(
        x + 0.18,
        covariance_errors,
        width=0.36,
        color="#df7d42",
        label="covariance relative error",
    )
    axes[1, 0].axhline(0.95, color="#555555", linestyle="--", linewidth=1)
    axes[1, 0].set_xticks(x, ["paired", "unpaired"])
    axes[1, 0].set_ylim(bottom=0.0)
    axes[1, 0].set_title("Independent flow-off audit")
    axes[1, 0].legend(frameon=False)

    check_names = list(decisions[0]["checks"])
    matrix = np.asarray(
        [[1.0 if decision["checks"][name] else 0.0 for name in check_names] for decision in decisions],
        dtype=np.float64,
    )
    axes[1, 1].imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="RdYlGn")
    axes[1, 1].set_xticks(range(len(check_names)), check_names, rotation=55, ha="right", fontsize=7)
    axes[1, 1].set_yticks(
        range(len(decisions)),
        [f"{d['method']} | {d['candidate_id']}" for d in decisions],
        fontsize=7,
    )
    axes[1, 1].set_title("Green single gates do not imply method authorization")
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            axes[1, 1].text(
                col_index,
                row_index,
                "PASS" if matrix[row_index, col_index] else "FAIL",
                ha="center",
                va="center",
                fontsize=5.5,
                color="white" if matrix[row_index, col_index] < 0.5 else "black",
            )
    fig.suptitle(
        "JACRU N1.1 flow-off covariance anchored-Tikhonov ceiling · opened synthetic only",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(output / "diagnostic.png", dpi=200)
    fig.savefig(output / "diagnostic.pdf")
    plt.close(fig)


def _write_checksums(output: Path) -> None:
    payloads = sorted(
        path for path in output.iterdir() if path.is_file() and path.name != "checksums.sha256"
    )
    lines = [f"{_sha256(path)}  {path.name}" for path in payloads]
    (output / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    config_path = args.config.resolve()
    config = _read_json(config_path)
    sources = _validate_sources(config)
    source_config = _read_json(sources["source_t0_config"])
    if args.seed_limit is not None:
        if args.seed_limit < 1:
            raise ValueError("seed-limit must be positive")
        source_config = json.loads(json.dumps(source_config))
        for split in source_config["splits"].values():
            split["base_seeds"] = split["base_seeds"][: args.seed_limit]
        source_config["training"]["model_seeds"] = source_config["training"][
            "model_seeds"
        ][: args.seed_limit]
    methods = [str(value) for value in config["methods"]]
    if not set(methods).issubset(set(source_config["methods"])):
        raise ValueError("N1.1 methods must be frozen T0 methods")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    fixture = m2._fixture_config(source_config)
    records = m2._prepare_records(source_config, fixture)
    matched = _load_matched_baselines(config)
    calibration_cache, calibration_rows = _build_calibration_cache(
        records=records, source_config=source_config, config=config
    )
    device = m2._choose_device(args.device or source_config["training"]["device"])
    trained: list[dict[str, Any]] = []
    for method in methods:
        for seed in source_config["training"]["model_seeds"]:
            trained.append(
                m2._train_one(
                    method=method,
                    seed=int(seed),
                    config=source_config,
                    records=records,
                    device=device,
                    epoch_override=args.epochs,
                )
            )

    projectors: dict[str, Any] = {}
    dense_setup_rows: list[dict[str, Any]] = []
    base_scores: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.split == "train":
            continue
        digest = record.case.inference.geometry.digest
        if digest not in projectors:
            operator = record.case.inference.operator
            dense_matrix, setup = m22._assemble_active_matrix_batched(
                operator,
                support=operator.support,
                batch_size=int(config["dense_ceiling"]["assembly_batch_size"]),
            )
            factor_started = time.perf_counter()
            projector = build_exact_dense_nullspace_projector(
                support=operator.support,
                dense_matrix=dense_matrix,
                rank_rtol=float(config["dense_ceiling"]["rank_relative_tolerance"]),
            )
            factor_seconds = time.perf_counter() - factor_started
            projectors[digest] = projector
            dense_setup_rows.append(
                {
                    "geometry_digest": digest,
                    "matrix_rows": int(dense_matrix.shape[0]),
                    "matrix_columns": int(dense_matrix.shape[1]),
                    "rank": int(projector.rank),
                    "rank_tolerance": float(projector.rank_tolerance),
                    "setup_forward_calls_batched": int(setup["setup_forward_calls"]),
                    "setup_forward_equivalents_unbatched": int(dense_matrix.shape[1] + 1),
                    "assembly_batch_size": int(config["dense_ceiling"]["assembly_batch_size"]),
                    "zero_forward_maximum_absolute": float(setup["zero_forward_maximum_absolute"]),
                    "factorization_seconds": factor_seconds,
                    "dense_setup_in_budget": False,
                    "status": "DENSE_TOY_ORACLE_SETUP_NOT_RECONSTRUCTION_BUDGET",
                }
            )
        base = record.batch.base_field[0, 0].to(record.case.inference.operator.support)
        base_scores[record.case.inference.case_id] = m2._score_prediction(
            record=record,
            method="prepared_cgls_base_12",
            model_seed=-1,
            prediction=base,
            gate=None,
            correction_rms=0.0,
            optimization_forward_calls=12,
            optimization_adjoint_calls=12,
            grouped_adjoint_calls=0,
            neural_inference_seconds=0.0,
        )

    rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    feature_f = int(config["matched_budget"]["learned_feature_preparation_forward_calls"])
    feature_a = int(config["matched_budget"]["learned_feature_preparation_adjoint_calls"])
    for run in trained:
        model = run["model"]
        model_device = next(model.parameters()).device
        for record in records:
            if record.split == "train":
                continue
            kwargs = m2._to_device(record.batch.model_kwargs(), model_device)
            m2._synchronize(model_device)
            inference_started = time.perf_counter()
            with torch.no_grad():
                prediction, gate = model(**kwargs, return_gate=True)
            m2._synchronize(model_device)
            inference_seconds = time.perf_counter() - inference_started
            operator = record.case.inference.operator
            initial = prediction[0, 0].detach().cpu().to(operator.support)
            raw_score = m2._score_prediction(
                record=record,
                method=str(run["method"]),
                model_seed=int(run["model_seed"]),
                prediction=initial,
                gate=float(gate[0, 0, 0, 0, 0].detach().cpu()),
                correction_rms=float(torch.sqrt(torch.mean((initial - record.batch.base_field[0, 0]).square()))),
                optimization_forward_calls=feature_f,
                optimization_adjoint_calls=feature_a,
                grouped_adjoint_calls=1,
                neural_inference_seconds=inference_seconds,
            )
            raw_reference = {"reference_kind": "raw_learned", **raw_score}
            reference_rows.append(raw_reference)
            projector = projectors[record.case.inference.geometry.digest]
            matrix = projector.dense_active_matrix
            support = projector.support_mask
            observation = record.case.inference.observations_uv[0].detach().cpu().to(torch.float64)
            truth = record.case.evaluation.truth_volume[0, 0].detach().cpu().to(torch.float64)
            truth_active = truth.masked_select(support)
            base_score = base_scores[record.case.inference.case_id]
            cgls = matched[(record.case.inference.case_id, "cgls_matched")]
            huber = matched[(record.case.inference.case_id, "huber_pdhg_matched")]
            best_field = min(float(cgls["field_relative_l2"]), float(huber["field_relative_l2"]))
            best_h1 = min(
                float(cgls["h1_seminorm_relative_error"]),
                float(huber["h1_seminorm_relative_error"]),
            )
            for candidate in config["candidates"]:
                mode = str(candidate["calibration_mode"])
                calibration = calibration_cache[(record.case.inference.case_id, mode)]
                mean_uv, proximal_covariance, selector_covariance, sensor_threshold = _candidate_components(
                    candidate=candidate, calibration=calibration
                )
                target = observation - mean_uv
                threshold = sensor_threshold
                truth_oracle_threshold = ""
                if candidate["threshold_policy"] == "truth_residual_oracle":
                    truth_residual = matrix @ truth_active - target.reshape(-1)
                    truth_oracle_threshold = whitened_quadratic(
                        truth_residual, selector_covariance
                    )
                    threshold = max(threshold, float(truth_oracle_threshold))
                elif candidate["threshold_policy"] != "empirical_sensor":
                    raise ValueError("unsupported threshold policy")
                result = dense_covariance_proximal_discrepancy(
                    initial_field=initial,
                    target_observation_uv=target,
                    dense_active_matrix=matrix,
                    support_mask=support,
                    proximal_covariance=proximal_covariance,
                    selector_covariance=selector_covariance,
                    discrepancy_threshold=threshold,
                    log10_alpha_bounds=tuple(float(v) for v in config["dense_ceiling"]["log10_alpha_bounds"]),
                    bisection_iterations=int(config["dense_ceiling"]["bisection_iterations"]),
                )
                score = m2._score_prediction(
                    record=record,
                    method=str(run["method"]),
                    model_seed=int(run["model_seed"]),
                    prediction=result.field,
                    gate=float(gate[0, 0, 0, 0, 0].detach().cpu()),
                    correction_rms=float(torch.sqrt(torch.mean((result.field - initial).square()))),
                    optimization_forward_calls=feature_f,
                    optimization_adjoint_calls=feature_a,
                    grouped_adjoint_calls=1,
                    neural_inference_seconds=inference_seconds,
                )
                field_gain = (best_field - float(score["field_relative_l2"])) / best_field
                h1_gain = (best_h1 - float(score["h1_seminorm_relative_error"])) / best_h1
                rows.append(
                    {
                        "candidate_id": candidate["id"],
                        "calibration_mode": mode,
                        "mean_policy": candidate["mean_policy"],
                        "proximal_covariance_policy": candidate["proximal_covariance_policy"],
                        "selector_covariance_policy": candidate["selector_covariance_policy"],
                        "threshold_policy": candidate["threshold_policy"],
                        "uses_truth": bool(candidate["uses_truth"]),
                        "uses_exact_nuisance": bool(candidate["uses_exact_nuisance"]),
                        "dense_ceiling_only": True,
                        "case_id": score["case_id"],
                        "split": score["split"],
                        "family": score["family"],
                        "base_seed": score["base_seed"],
                        "method": score["method"],
                        "model_seed": score["model_seed"],
                        "field_relative_l2": score["field_relative_l2"],
                        "h1_seminorm_relative_error": score["h1_seminorm_relative_error"],
                        "measured_reprojection_relative_l2": score["measured_reprojection_relative_l2"],
                        "clean_reprojection_relative_l2": score["clean_reprojection_relative_l2"],
                        "field_gain_to_best_matched": field_gain,
                        "h1_gain_to_best_matched": h1_gain,
                        "clean_reprojection_ratio_to_base": float(score["clean_reprojection_relative_l2"])
                        / max(float(base_score["clean_reprojection_relative_l2"]), 1e-30),
                        "measured_reprojection_ratio_to_cgls": float(score["measured_reprojection_relative_l2"])
                        / max(float(cgls["measured_reprojection_relative_l2"]), 1e-30),
                        "field_harm": field_gain < -float(config["decision_gates"]["field_harm_threshold_fraction"]),
                        "sensor_discrepancy_threshold": sensor_threshold,
                        "selected_discrepancy_threshold": threshold,
                        "truth_residual_oracle_threshold": truth_oracle_threshold,
                        "raw_discrepancy": result.raw_discrepancy,
                        "selected_discrepancy": result.selected_discrepancy,
                        "alpha": result.alpha,
                        "target_crossed": result.target_crossed,
                        "raw_no_correction": math.isinf(result.alpha),
                        "correction_norm": result.correction_norm,
                        "measurement_residual_norm": result.measurement_residual_norm,
                        "proximal_covariance_scale": result.proximal_covariance_scale,
                        "residual_closure_relative_error": result.residual_closure_relative_error,
                        "bisection_iterations": result.bisection_iterations,
                        "dense_matrix_rows": int(matrix.shape[0]),
                        "dense_matrix_columns": int(matrix.shape[1]),
                        "dense_setup_forward_equivalents": int(matrix.shape[1] + 1),
                        "dense_setup_in_budget": False,
                        "learned_feature_forward_calls": feature_f,
                        "learned_feature_adjoint_calls": feature_a,
                    }
                )
    aggregate_rows = _aggregate(rows)
    calibration_decisions = _calibration_decisions(calibration_rows, config)
    decisions = _decisions(
        rows=rows,
        config=config,
        calibration_decisions=calibration_decisions,
    )
    deployable_input_passes = [
        decision
        for decision in decisions
        if decision["passed"]
        and not decision["uses_truth"]
        and not decision["uses_exact_nuisance"]
    ]
    oracle_passes = [decision for decision in decisions if decision["passed"]]
    if deployable_input_passes:
        status = "N1_1_FLOWOFF_COVARIANCE_PROXIMAL_MECHANISM_SIGNAL_ONLY"
    elif oracle_passes:
        status = "N1_1_ORACLE_ONLY_COVARIANCE_PROXIMAL_NO_GO"
    else:
        status = "N1_1_FLOWOFF_COVARIANCE_PROXIMAL_NO_GO"
    authorization = {
        "claim_deployable_algorithm": False,
        "claim_method_superiority": False,
        "claim_real_bost_generalization": False,
        "open_fresh_or_final": False,
        "continue_matrix_free_covariance_proximal_research": bool(deployable_input_passes),
        "continue_model_mismatch_floor_research": True,
        "request_same_session_flowoff_from_lab": True,
    }
    summary = {
        "schema_version": REPORT_SCHEMA,
        "status": status,
        "evidence_level": config["evidence_level"],
        "source_config": str(config_path.relative_to(ROOT)),
        "source_config_sha256": _sha256(config_path),
        "source_hashes": {key: _sha256(path) for key, path in sources.items()},
        "runtime_seconds": time.perf_counter() - started,
        "device": str(device),
        "candidate_count": len(config["candidates"]),
        "calibration_row_count": len(calibration_rows),
        "metric_row_count": len(rows),
        "aggregate_row_count": len(aggregate_rows),
        "reference_row_count": len(reference_rows),
        "dense_setup_row_count": len(dense_setup_rows),
        "calibration_decisions": calibration_decisions,
        "decisions": decisions,
        "deployable_input_pass_count": len(deployable_input_passes),
        "oracle_pass_count": len(oracle_passes),
        "authorization": authorization,
        "claim_boundary": config["claim_boundary"],
    }
    _write_csv(output / "calibration_rows.csv", calibration_rows)
    _write_csv(output / "metric_rows.csv", rows)
    _write_csv(output / "aggregate_rows.csv", aggregate_rows)
    _write_csv(output / "reference_rows.csv", reference_rows)
    _write_csv(output / "dense_setup_rows.csv", dense_setup_rows)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot(
        rows=rows,
        calibration_rows=calibration_rows,
        decisions=decisions,
        output=output,
    )
    readme = f"""# JACRU N1.1 flow-off covariance proximal evidence packet

- Status: `{status}`
- Evidence: `{config['evidence_level']}`
- Flow-off rows: `{len(calibration_rows)}`
- Candidate rows: `{len(rows)}`
- Dense setup rows: `{len(dense_setup_rows)}`; evaluator ceiling only, with batched calls
  and unbatched forward-equivalents reported separately and excluded from deployment claims.
- Truth-aware and exact-nuisance candidates are labeled row by row.
- Raw observations, predictions, checkpoints, private PDFs, and lab data are not included.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    _write_checksums(output)
    print(json.dumps({"status": status, "rows": len(rows), "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
