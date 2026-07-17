#!/usr/bin/env python3
"""Audit observable first-crossing rules on the opened M2.7 trajectory.

N1.0 does not retrain a network or open a new split.  It reads the complete
K=0..10 exact-camera-block trajectory produced by M2.7, chooses the first
iterate satisfying one declared observable residual rule, and evaluates that
choice afterward.  Clean-renderer and field labels are never available to the
selector.  The exact camera-block setup remains an excluded toy oracle, so a
positive result could only authorize a deployable covariance/stopping study.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Iterable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT
    / "demo_t16_operator/configs/"
    "jacru_n1_0_observable_discrepancy_stopping_postopen_v1.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "demo_t16_operator/results/"
    "jacru_n1_0_observable_discrepancy_stopping_postopen_public"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected one JSON object")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"expected nonempty CSV: {path}")
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    if not materialized:
        raise ValueError("cannot average an empty collection")
    return float(math.fsum(materialized) / len(materialized))


def _validate_checksum_manifest(directory: Path) -> None:
    manifest = directory / "checksums.sha256"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("* ")
        path = directory / relative
        if _sha256(path) != expected:
            raise RuntimeError(f"checksum drifted: {path}")


def _validate_sources(config: dict[str, Any]) -> dict[str, Path]:
    sources = {
        "source_t0_config": ROOT / config["source_t0_config"],
        "source_m2_7_config": ROOT / config["source_m2_7_config"],
        "source_m2_7_summary": (
            ROOT / config["source_m2_7_results"] / "summary.json"
        ),
        "source_m2_8_config": ROOT / config["source_m2_8_config"],
        "source_m2_8_summary": (
            ROOT / config["source_m2_8_results"] / "summary.json"
        ),
    }
    for name, path in sources.items():
        expected = str(config[f"{name}_sha256"])
        observed = _sha256(path)
        if observed != expected:
            raise RuntimeError(f"{name} hash drift: {observed} != {expected}")
    m27 = _read_json(sources["source_m2_7_summary"])
    m28 = _read_json(sources["source_m2_8_summary"])
    if m27.get("status") != "M2_7_TARGET_NO_HARM_PARETO_ORACLE_NO_GO":
        raise RuntimeError("M2.7 source status drifted")
    if m28.get("status") != "M2_8_INTERPOLATION_CALIBRATION_ENVELOPE_NO_GO":
        raise RuntimeError("M2.8 source status drifted")
    _validate_checksum_manifest(ROOT / config["source_m2_7_results"])
    _validate_checksum_manifest(ROOT / config["source_m2_8_results"])
    return sources


def _group_trajectory_rows(
    rows: list[dict[str, str]],
    *,
    expected_iterations: list[int],
    expected_variant: str,
) -> dict[tuple[str, int, str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, int, str, str], list[dict[str, str]]] = {}
    for row in rows:
        if row["projection_variant"] != expected_variant:
            raise RuntimeError("M2.7 projection variant drifted")
        if float(row["damping_absolute"]) != 0.0:
            raise RuntimeError(
                "N1.0 system-residual stopping is valid only for zero damping"
            )
        if row["projection_target_mode"] != "affine_observation":
            raise RuntimeError("N1.0 requires the opened affine-observation path")
        if row["preconditioner_kind"] != "dense_exact_camera_block_jacobi_oracle":
            raise RuntimeError("N1.0 source preconditioner drifted")
        key = (
            row["method"],
            int(row["model_seed"]),
            row["split"],
            row["case_id"],
        )
        grouped.setdefault(key, []).append(row)
    for key, values in grouped.items():
        values.sort(key=lambda row: int(row["projection_iterations"]))
        observed = [int(row["projection_iterations"]) for row in values]
        if observed != expected_iterations:
            raise RuntimeError(f"incomplete M2.7 trajectory for {key}: {observed}")
    return grouped


def _base_anchor_lookup(
    rows: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    anchors = [row for row in rows if row["reference_kind"] == "base_anchor"]
    lookup = {row["case_id"]: row for row in anchors}
    if len(lookup) != len(anchors):
        raise RuntimeError("base-anchor rows must contain one row per case")
    return lookup


def _first_crossing(
    trajectory: list[dict[str, str]],
    *,
    observable: Callable[[dict[str, str]], float],
    threshold: float,
) -> tuple[dict[str, str] | None, bool]:
    """Select the first row using only one caller-supplied observable."""

    value = float(threshold)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("stopping threshold must be finite and nonnegative")
    for row in trajectory:
        observed = float(observable(row))
        if not math.isfinite(observed) or observed < 0.0:
            raise ValueError("stopping observable must be finite and nonnegative")
        if observed <= value:
            return row, True
    return None, False


def _candidate_specs(
    config: dict[str, Any], *, noise_floor: float
) -> list[dict[str, Any]]:
    families = config["observable_stopping_families"]
    specs: list[dict[str, Any]] = []
    for multiplier in families["simulator_noise_floor_multiple"]["multipliers"]:
        specs.append(
            {
                "candidate_id": f"noise_floor_x{float(multiplier):g}",
                "family": "simulator_noise_floor_multiple",
                "parameter": float(multiplier),
                "threshold": float(multiplier) * noise_floor,
                "comparator_only": False,
                "uses_simulator_nuisance_scale": True,
            }
        )
    for multiplier in families["base_anchor_residual_multiple"]["multipliers"]:
        specs.append(
            {
                "candidate_id": f"base_residual_x{float(multiplier):g}",
                "family": "base_anchor_residual_multiple",
                "parameter": float(multiplier),
                "threshold": None,
                "comparator_only": False,
                "uses_simulator_nuisance_scale": False,
            }
        )
    for fraction in families["initial_system_residual_fraction"][
        "maximum_fractions"
    ]:
        specs.append(
            {
                "candidate_id": f"system_fraction_{float(fraction):g}",
                "family": "initial_system_residual_fraction",
                "parameter": float(fraction),
                "threshold": float(fraction),
                "comparator_only": False,
                "uses_simulator_nuisance_scale": False,
            }
        )
    for iteration in config["fixed_iteration_comparators"]:
        specs.append(
            {
                "candidate_id": f"fixed_k{int(iteration)}",
                "family": "fixed_iteration_comparator",
                "parameter": int(iteration),
                "threshold": None,
                "comparator_only": True,
                "uses_simulator_nuisance_scale": False,
            }
        )
    return specs


def _best_h1_from_row(row: dict[str, str]) -> float:
    gain = float(row["h1_gain_to_best_matched_classical"])
    denominator = 1.0 - gain
    if denominator <= 1e-12:
        raise RuntimeError("cannot recover matched classical H1 reference")
    return float(row["h1_seminorm_relative_error"]) / denominator


def _materialize_selection(
    *,
    trajectory: list[dict[str, str]],
    base: dict[str, str],
    spec: dict[str, Any],
    maximum_iteration: int,
    harm_threshold: float,
) -> dict[str, Any]:
    family = str(spec["family"])
    if family == "fixed_iteration_comparator":
        selected = trajectory[int(spec["parameter"])]
        crossed = True
        threshold = None
        observable_value = float("nan")
    else:
        if family == "simulator_noise_floor_multiple":
            threshold = float(spec["threshold"])
            observable = lambda row: float(row["measured_reprojection_relative_l2"])
        elif family == "base_anchor_residual_multiple":
            threshold = float(spec["parameter"])
            base_residual = max(
                float(base["measured_reprojection_relative_l2"]), 1e-30
            )
            observable = lambda row: float(
                row["measured_reprojection_relative_l2"]
            ) / base_residual
        elif family == "initial_system_residual_fraction":
            threshold = float(spec["threshold"])
            observable = lambda row: float(row["system_residual_fraction"])
        else:
            raise ValueError(f"unsupported stopping family: {family}")
        selected, crossed = _first_crossing(
            trajectory,
            observable=observable,
            threshold=threshold,
        )
        observable_value = (
            float(observable(selected)) if selected is not None else float("nan")
        )

    attempted = selected if selected is not None else trajectory[-1]
    attempted_iteration = int(attempted["projection_iterations"])
    if selected is None:
        reference = trajectory[-1]
        field_error = float(base["field_relative_l2"])
        h1_error = float(base["h1_seminorm_relative_error"])
        measured_error = float(base["measured_reprojection_relative_l2"])
        clean_error = float(base["clean_reprojection_relative_l2"])
        best_field = min(
            float(reference["matched_cgls_field_relative_l2"]),
            float(reference["matched_huber_field_relative_l2"]),
        )
        best_h1 = _best_h1_from_row(reference)
        matched_cgls_residual = float(
            reference["measured_reprojection_relative_l2"]
        ) / max(float(reference["reprojection_ratio_to_matched_cgls"]), 1e-30)
        field_gain = (best_field - field_error) / max(best_field, 1e-30)
        h1_gain = (best_h1 - h1_error) / max(best_h1, 1e-30)
        reprojection_ratio = measured_error / max(matched_cgls_residual, 1e-30)
        closure_error = float(reference["projection_closure_relative_error"])
        returned_kind = "prepared_cgls_base_fallback"
        selected_iteration = -1
        forward_calls = 14 + maximum_iteration
        adjoint_calls = 13 + maximum_iteration
    else:
        field_error = float(selected["field_relative_l2"])
        h1_error = float(selected["h1_seminorm_relative_error"])
        measured_error = float(selected["measured_reprojection_relative_l2"])
        clean_error = float(selected["clean_reprojection_relative_l2"])
        field_gain = float(selected["field_gain_to_best_matched_classical"])
        h1_gain = float(selected["h1_gain_to_best_matched_classical"])
        reprojection_ratio = float(selected["reprojection_ratio_to_matched_cgls"])
        closure_error = float(selected["projection_closure_relative_error"])
        returned_kind = "selected_affine_pcg_iterate"
        selected_iteration = int(selected["projection_iterations"])
        forward_calls = int(selected["optimization_forward_calls"])
        adjoint_calls = int(selected["optimization_adjoint_calls"])

    base_clean = max(float(base["clean_reprojection_relative_l2"]), 1e-30)
    return {
        "candidate_id": spec["candidate_id"],
        "stopping_family": family,
        "stopping_parameter": spec["parameter"],
        "stopping_threshold": threshold,
        "comparator_only": bool(spec["comparator_only"]),
        "uses_simulator_nuisance_scale": bool(
            spec["uses_simulator_nuisance_scale"]
        ),
        "selection_uses_truth": False,
        "selection_uses_clean_renderer": False,
        "target_crossed": bool(crossed),
        "selected_observable_value": observable_value,
        "selected_iteration": selected_iteration,
        "attempted_iteration": attempted_iteration,
        "returned_field_kind": returned_kind,
        "case_id": trajectory[0]["case_id"],
        "split": trajectory[0]["split"],
        "family": trajectory[0]["family"],
        "base_seed": int(trajectory[0]["base_seed"]),
        "method": trajectory[0]["method"],
        "model_seed": int(trajectory[0]["model_seed"]),
        "field_relative_l2": field_error,
        "h1_seminorm_relative_error": h1_error,
        "measured_reprojection_relative_l2": measured_error,
        "clean_reprojection_relative_l2": clean_error,
        "clean_reprojection_ratio_to_base": clean_error / base_clean,
        "field_gain_to_best_matched_classical": field_gain,
        "h1_gain_to_best_matched_classical": h1_gain,
        "reprojection_ratio_to_matched_cgls": reprojection_ratio,
        "field_harm": field_gain < -float(harm_threshold),
        "projection_closure_relative_error": closure_error,
        "forward_calls": forward_calls,
        "adjoint_calls": adjoint_calls,
        "exact_camera_block_setup_forward_equivalents": 1001,
        "exact_camera_block_setup_in_budget": False,
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["candidate_id"]),
            str(row["method"]),
            int(row["model_seed"]),
            str(row["split"]),
        )
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (candidate, method, seed, split), values in sorted(grouped.items()):
        output.append(
            {
                "candidate_id": candidate,
                "stopping_family": values[0]["stopping_family"],
                "stopping_parameter": values[0]["stopping_parameter"],
                "comparator_only": values[0]["comparator_only"],
                "method": method,
                "model_seed": seed,
                "split": split,
                "case_count": len(values),
                "target_crossing_rate": _mean(row["target_crossed"] for row in values),
                "selected_iteration_mean": _mean(
                    max(int(row["selected_iteration"]), int(row["attempted_iteration"]))
                    for row in values
                ),
                "selected_iteration_maximum": max(
                    max(int(row["selected_iteration"]), int(row["attempted_iteration"]))
                    for row in values
                ),
                "field_gain_mean": _mean(
                    row["field_gain_to_best_matched_classical"] for row in values
                ),
                "h1_gain_mean": _mean(
                    row["h1_gain_to_best_matched_classical"] for row in values
                ),
                "clean_reprojection_ratio_to_base_mean": _mean(
                    row["clean_reprojection_ratio_to_base"] for row in values
                ),
                "clean_reprojection_ratio_to_base_maximum": max(
                    float(row["clean_reprojection_ratio_to_base"]) for row in values
                ),
                "reprojection_ratio_to_matched_cgls_mean": _mean(
                    row["reprojection_ratio_to_matched_cgls"] for row in values
                ),
                "field_harm_rate": _mean(row["field_harm"] for row in values),
                "worst_field_gain": min(
                    float(row["field_gain_to_best_matched_classical"])
                    for row in values
                ),
                "projection_closure_relative_error_maximum": max(
                    float(row["projection_closure_relative_error"])
                    for row in values
                ),
                "forward_calls_mean": _mean(row["forward_calls"] for row in values),
                "forward_calls_maximum": max(int(row["forward_calls"]) for row in values),
                "adjoint_calls_mean": _mean(row["adjoint_calls"] for row in values),
                "adjoint_calls_maximum": max(int(row["adjoint_calls"]) for row in values),
            }
        )
    return output


def _pooled_metrics(
    rows: list[dict[str, Any]], *, candidate_id: str, method: str, split: str
) -> dict[str, Any]:
    values = [
        row
        for row in rows
        if row["candidate_id"] == candidate_id
        and row["method"] == method
        and row["split"] == split
    ]
    if not values:
        raise RuntimeError("missing candidate rows")
    seed_means = []
    for seed in sorted({int(row["model_seed"]) for row in values}):
        seed_means.append(
            _mean(
                row["field_gain_to_best_matched_classical"]
                for row in values
                if int(row["model_seed"]) == seed
            )
        )
    return {
        "row_count": len(values),
        "target_crossing_rate": _mean(row["target_crossed"] for row in values),
        "selected_iteration_mean": _mean(
            max(int(row["selected_iteration"]), int(row["attempted_iteration"]))
            for row in values
        ),
        "field_gain_mean": _mean(
            row["field_gain_to_best_matched_classical"] for row in values
        ),
        "h1_gain_mean": _mean(
            row["h1_gain_to_best_matched_classical"] for row in values
        ),
        "clean_reprojection_ratio_to_base_mean": _mean(
            row["clean_reprojection_ratio_to_base"] for row in values
        ),
        "clean_reprojection_ratio_to_base_maximum": max(
            float(row["clean_reprojection_ratio_to_base"]) for row in values
        ),
        "reprojection_ratio_to_matched_cgls_mean": _mean(
            row["reprojection_ratio_to_matched_cgls"] for row in values
        ),
        "field_harm_rate": _mean(row["field_harm"] for row in values),
        "worst_field_gain": min(
            float(row["field_gain_to_best_matched_classical"]) for row in values
        ),
        "projection_closure_relative_error_maximum": max(
            float(row["projection_closure_relative_error"]) for row in values
        ),
        "forward_calls_maximum": max(int(row["forward_calls"]) for row in values),
        "adjoint_calls_maximum": max(int(row["adjoint_calls"]) for row in values),
        "per_model_seed_field_gain_means": seed_means,
    }


def _decisions(
    rows: list[dict[str, Any]], specs: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    gates = config["decision_gates"]
    decisions: dict[str, Any] = {}
    for method in config["methods"]:
        screened = []
        for spec in specs:
            if spec["comparator_only"]:
                continue
            development = _pooled_metrics(
                rows,
                candidate_id=spec["candidate_id"],
                method=str(method),
                split="development",
            )
            checks = {
                "field_gain": development["field_gain_mean"]
                >= float(gates["development_field_gain_minimum"]),
                "h1_gain": development["h1_gain_mean"]
                >= float(gates["development_h1_gain_minimum"]),
                "clean_reprojection_mean": development[
                    "clean_reprojection_ratio_to_base_mean"
                ]
                <= float(
                    gates[
                        "development_clean_reprojection_ratio_to_base_mean_maximum"
                    ]
                ),
                "clean_reprojection_worst": development[
                    "clean_reprojection_ratio_to_base_maximum"
                ]
                <= float(
                    gates[
                        "development_clean_reprojection_ratio_to_base_worst_maximum"
                    ]
                ),
                "harm_rate": development["field_harm_rate"]
                <= float(gates["field_harm_rate_maximum"]),
                "worst_field_gain": development["worst_field_gain"]
                >= float(gates["worst_field_gain_minimum"]),
                "target_crossing": development["target_crossing_rate"]
                >= float(gates["minimum_target_crossing_rate"]),
                "closure": development["projection_closure_relative_error_maximum"]
                <= float(gates["maximum_projection_closure_relative_error"]),
                "forward_budget": development["forward_calls_maximum"]
                <= int(gates["maximum_forward_calls"]),
                "adjoint_budget": development["adjoint_calls_maximum"]
                <= int(gates["maximum_adjoint_calls"]),
                "all_seed_means_positive": all(
                    value > 0.0
                    for value in development["per_model_seed_field_gain_means"]
                ),
            }
            screened.append(
                {
                    "candidate_id": spec["candidate_id"],
                    "stopping_family": spec["family"],
                    "stopping_parameter": spec["parameter"],
                    "development": development,
                    "development_checks": checks,
                    "development_eligible": all(checks.values()),
                }
            )
        eligible = [item for item in screened if item["development_eligible"]]
        eligible.sort(
            key=lambda item: (
                -float(item["development"]["field_gain_mean"]),
                float(item["development"]["selected_iteration_mean"]),
                str(item["candidate_id"]),
            )
        )
        if not eligible:
            decisions[str(method)] = {
                "screened_candidates": screened,
                "selection": None,
                "passed_opened_n1_0_gate": False,
            }
            continue
        selected = eligible[0]
        ood = _pooled_metrics(
            rows,
            candidate_id=selected["candidate_id"],
            method=str(method),
            split="ood",
        )
        ood_checks = {
            "field_gain": ood["field_gain_mean"]
            >= float(gates["ood_field_gain_minimum"]),
            "h1_gain": ood["h1_gain_mean"] >= float(gates["ood_h1_gain_minimum"]),
            "clean_reprojection_mean": ood[
                "clean_reprojection_ratio_to_base_mean"
            ]
            <= float(gates["ood_clean_reprojection_ratio_to_base_mean_maximum"]),
            "clean_reprojection_worst": ood[
                "clean_reprojection_ratio_to_base_maximum"
            ]
            <= float(gates["ood_clean_reprojection_ratio_to_base_worst_maximum"]),
            "harm_rate": ood["field_harm_rate"]
            <= float(gates["field_harm_rate_maximum"]),
            "worst_field_gain": ood["worst_field_gain"]
            >= float(gates["worst_field_gain_minimum"]),
            "target_crossing": ood["target_crossing_rate"]
            >= float(gates["minimum_target_crossing_rate"]),
            "closure": ood["projection_closure_relative_error_maximum"]
            <= float(gates["maximum_projection_closure_relative_error"]),
            "forward_budget": ood["forward_calls_maximum"]
            <= int(gates["maximum_forward_calls"]),
            "adjoint_budget": ood["adjoint_calls_maximum"]
            <= int(gates["maximum_adjoint_calls"]),
            "all_seed_means_positive": all(
                value > 0.0 for value in ood["per_model_seed_field_gain_means"]
            ),
        }
        decisions[str(method)] = {
            "screened_candidates": screened,
            "selection": {
                **selected,
                "ood": ood,
                "ood_checks": ood_checks,
                "passed_ood_gate": all(ood_checks.values()),
            },
            "passed_opened_n1_0_gate": all(ood_checks.values()),
        }
    return decisions


def _pareto_audit(
    rows: list[dict[str, Any]], specs: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    gates = config["decision_gates"]
    output: dict[str, Any] = {}
    for method in config["methods"]:
        candidates = []
        for spec in specs:
            if spec["comparator_only"]:
                continue
            metrics = _pooled_metrics(
                rows,
                candidate_id=spec["candidate_id"],
                method=str(method),
                split="development",
            )
            tail_safe = (
                metrics["field_harm_rate"]
                <= float(gates["field_harm_rate_maximum"])
                and metrics["worst_field_gain"]
                >= float(gates["worst_field_gain_minimum"])
            )
            renderer_safe = (
                metrics["clean_reprojection_ratio_to_base_mean"]
                <= float(
                    gates[
                        "development_clean_reprojection_ratio_to_base_mean_maximum"
                    ]
                )
                and metrics["clean_reprojection_ratio_to_base_maximum"]
                <= float(
                    gates[
                        "development_clean_reprojection_ratio_to_base_worst_maximum"
                    ]
                )
            )
            candidates.append(
                {
                    "candidate_id": spec["candidate_id"],
                    "stopping_family": spec["family"],
                    "tail_safe": tail_safe,
                    "renderer_safe": renderer_safe,
                    "joint_safe": tail_safe and renderer_safe,
                    "metrics": metrics,
                }
            )
        tail = [value for value in candidates if value["tail_safe"]]
        renderer = [value for value in candidates if value["renderer_safe"]]
        joint = [value for value in candidates if value["joint_safe"]]
        tail.sort(
            key=lambda value: (
                float(value["metrics"]["clean_reprojection_ratio_to_base_mean"]),
                str(value["candidate_id"]),
            )
        )
        renderer.sort(
            key=lambda value: (
                -float(value["metrics"]["worst_field_gain"]),
                float(value["metrics"]["clean_reprojection_ratio_to_base_mean"]),
                str(value["candidate_id"]),
            )
        )
        output[str(method)] = {
            "candidate_count": len(candidates),
            "tail_safe_count": len(tail),
            "renderer_safe_count": len(renderer),
            "joint_safe_count": len(joint),
            "best_renderer_consistency_among_tail_safe": tail[0] if tail else None,
            "best_field_tail_among_renderer_safe": renderer[0] if renderer else None,
        }
    return output


def _plot(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    decisions: dict[str, Any],
    config: dict[str, Any],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.4), constrained_layout=True)
    colors = {"jacru_m2": "#167c80", "pooled_cnn": "#d97941"}
    markers = {
        "simulator_noise_floor_multiple": "o",
        "base_anchor_residual_multiple": "s",
        "initial_system_residual_fraction": "^",
    }
    specs = {
        str(row["candidate_id"]): row
        for row in rows
        if row["split"] == "development" and not row["comparator_only"]
    }
    for method in config["methods"]:
        candidates = sorted(
            {
                str(row["candidate_id"])
                for row in rows
                if row["method"] == method
                and row["split"] == "development"
                and not row["comparator_only"]
            }
        )
        for candidate in candidates:
            metric = _pooled_metrics(
                rows,
                candidate_id=candidate,
                method=str(method),
                split="development",
            )
            family = str(specs[candidate]["stopping_family"])
            axes[0, 0].scatter(
                metric["clean_reprojection_ratio_to_base_mean"],
                metric["worst_field_gain"],
                s=42,
                color=colors[str(method)],
                marker=markers[family],
                alpha=0.78,
            )
            axes[0, 1].scatter(
                metric["selected_iteration_mean"],
                metric["field_gain_mean"],
                s=42,
                color=colors[str(method)],
                marker=markers[family],
                alpha=0.78,
            )
    axes[0, 0].axvline(
        float(
            config["decision_gates"][
                "development_clean_reprojection_ratio_to_base_mean_maximum"
            ]
        ),
        color="#555555",
        linestyle="--",
    )
    axes[0, 0].axhline(
        float(config["decision_gates"]["worst_field_gain_minimum"]),
        color="#a6473d",
        linestyle="--",
    )
    axes[0, 0].set_xlabel("mean clean-renderer residual / base anchor")
    axes[0, 0].set_ylabel("worst field gain")
    axes[0, 0].set_title("No observable rule enters the joint safe quadrant")
    axes[0, 1].set_xlabel("mean selected PCG iteration")
    axes[0, 1].set_ylabel("mean field gain vs matched classical")
    axes[0, 1].set_title("Stopping earlier preserves field mean but underfits renderer")
    method_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=colors[str(method)],
            label=str(method),
            markersize=7,
        )
        for method in config["methods"]
    ]
    family_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[family],
            linestyle="none",
            color="#555555",
            label=family.replace("_multiple", ""),
            markersize=7,
        )
        for family in markers
    ]
    axes[0, 1].legend(
        handles=method_handles + family_handles,
        fontsize=8,
        loc="lower left",
        frameon=True,
    )

    families = [
        "simulator_noise_floor_multiple",
        "base_anchor_residual_multiple",
        "initial_system_residual_fraction",
    ]
    for method_index, method in enumerate(config["methods"]):
        ax = axes[1, method_index]
        screen = decisions[str(method)]["screened_candidates"]
        labels = []
        matrix = []
        for family in families:
            family_rows = [row for row in screen if row["stopping_family"] == family]
            if not family_rows:
                continue
            best = max(
                family_rows,
                key=lambda row: (
                    sum(bool(value) for value in row["development_checks"].values()),
                    float(row["development"]["field_gain_mean"]),
                ),
            )
            labels.append(best["candidate_id"])
            matrix.append(
                [int(value) for value in best["development_checks"].values()]
            )
        check_labels = list(screen[0]["development_checks"])
        ax.imshow(np.asarray(matrix), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(check_labels)), check_labels, rotation=55, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_title(f"{method}: best rule per family still fails")
        for row_index, values in enumerate(matrix):
            for column_index, value in enumerate(values):
                ax.text(
                    column_index,
                    row_index,
                    "PASS" if value else "FAIL",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white",
                    fontweight="bold",
                )
    fig.suptitle(
        "JACRU N1.0 observable discrepancy stopping · opened synthetic only",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _write_checksums(output: Path, names: list[str]) -> None:
    lines = [f"{_sha256(output / name)}  {name}" for name in names]
    (output / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    config_path = args.config.resolve()
    config = _read_json(config_path)
    sources = _validate_sources(config)
    source_t0 = _read_json(sources["source_t0_config"])
    fixture = source_t0["fixture"]
    noise_floor = math.sqrt(
        float(fixture["noise_relative_std"]) ** 2
        + float(fixture["camera_bias_relative_std"]) ** 2
    )
    packet = ROOT / config["source_m2_7_results"]
    trajectory_rows = _read_csv(packet / "metric_rows.csv")
    reference_rows = _read_csv(packet / "reference_rows.csv")
    expected_iterations = [int(value) for value in config["trajectory"]["source_iterations"]]
    groups = _group_trajectory_rows(
        trajectory_rows,
        expected_iterations=expected_iterations,
        expected_variant=str(config["trajectory"]["projection_variant"]),
    )
    base_lookup = _base_anchor_lookup(reference_rows)
    specs = _candidate_specs(config, noise_floor=noise_floor)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for (_, _, _, case_id), trajectory in sorted(groups.items()):
        base = base_lookup[case_id]
        for spec in specs:
            rows.append(
                _materialize_selection(
                    trajectory=trajectory,
                    base=base,
                    spec=spec,
                    maximum_iteration=int(config["trajectory"]["maximum_iteration"]),
                    harm_threshold=float(
                        config["decision_gates"]["field_harm_threshold_fraction"]
                    ),
                )
            )
    aggregate = _aggregate(rows)
    decisions = _decisions(rows, specs, config)
    pareto_audit = _pareto_audit(rows, specs, config)
    passed = any(
        bool(value["passed_opened_n1_0_gate"]) for value in decisions.values()
    )
    status = (
        config["report_status"]["mechanism_signal"]
        if passed
        else config["report_status"]["no_go"]
    )
    summary = {
        "schema_version": config["report_schema_version"],
        "status": status,
        "evidence_level": config["evidence_level"],
        "source_config": str(config_path.relative_to(ROOT)),
        "source_config_sha256": _sha256(config_path),
        "source_hashes": {name: _sha256(path) for name, path in sources.items()},
        "simulator_relative_noise_floor": noise_floor,
        "trajectory_group_count": len(groups),
        "candidate_spec_count": len(specs),
        "row_count": len(rows),
        "aggregate_row_count": len(aggregate),
        "decisions": decisions,
        "pareto_audit": pareto_audit,
        "selector_observable_columns": [
            "measured_reprojection_relative_l2",
            "prepared_cgls_base_12_measured_reprojection_relative_l2",
            "system_residual_fraction"
        ],
        "authorization": {
            "continue_flow_off_covariance_research": not passed,
            "continue_heldout_fail_closed_research": not passed,
            "claim_deployable_algorithm": False,
            "claim_method_superiority": False,
            "claim_real_bost_generalization": False,
            "open_fresh_or_final": False,
        },
        "claim_boundary": config["claim_boundary"],
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_csv(output / "selected_rows.csv", rows)
    _write_csv(output / "aggregate_rows.csv", aggregate)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _plot(output / "diagnostic", rows=rows, decisions=decisions, config=config)
    readme = f"""# JACRU N1.0 observable discrepancy stopping

Status: `{status}`

This post-open ceiling reuses the complete M2.7 K=0..10 trajectory.  Every
selector reads only measured residuals, the prepared CGLS-12 residual, or the
PCG system-residual fraction.  Field truth and the independent clean renderer
are evaluation-only.  A missed threshold returns the already prepared CGLS-12
base after charging the full attempted trajectory budget.

The simulator noise floor (`{noise_floor:.8f}`) is derived from frozen nuisance
parameters, not from real flow-off repeats.  The exact camera-block setup is a
1001-forward-equivalent oracle excluded from the call budget.  Therefore this
packet cannot establish deployment, runtime superiority, fresh performance, or
real BOST generalization even if an opened candidate passes.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    _write_checksums(
        output,
        [
            "README.md",
            "aggregate_rows.csv",
            "diagnostic.pdf",
            "diagnostic.png",
            "selected_rows.csv",
            "summary.json",
        ],
    )
    print(json.dumps({"status": status, "rows": len(rows), "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
