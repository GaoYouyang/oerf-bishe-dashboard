#!/usr/bin/env python3
"""Audit fixed interpolation and a truth-oracle calibration ceiling after M2.7.

The candidate family is ``x(alpha)=x_net-alpha*(x_net-x_pcg)`` with one
globally frozen alpha.  A separate evaluator-only ceiling may choose alpha per
sample using truth, but only inside the interval that satisfies a per-sample
matched-CGLS reprojection gate.  That ceiling is impossible to deploy and can
only falsify the interpolation family or authorize later observable calibration
research.  It can never authorize a method or a fresh/final claim.
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

from demo_t16_operator.jacru_m2_matrix_free_projection import (
    matrix_free_measurement_projection_path,
)
from site_tools import run_jacru_m2_3_matrix_free_projection as m23
from site_tools import run_jacru_m2_learned_residual_gate as m2


DEFAULT_CONFIG = (
    ROOT
    / "demo_t16_operator/configs/"
    "jacru_m2_8_interpolation_calibration_ceiling_postopen_v1.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "demo_t16_operator/results/"
    "jacru_m2_8_interpolation_calibration_ceiling_postopen_public"
)


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
        raise ValueError("expected one JSON object")
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


def _mean(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    if not materialized:
        raise ValueError("cannot average an empty collection")
    return float(math.fsum(materialized) / len(materialized))


def _validate_sources(config: dict[str, Any]) -> dict[str, Path]:
    paths = {
        "source_t0_config": ROOT / config["source_t0_config"],
        "source_t0_summary": ROOT / config["source_t0_results"] / "summary.json",
        "source_m2_6_config": ROOT / config["source_m2_6_config"],
        "source_m2_6_summary": ROOT / config["source_m2_6_results"] / "summary.json",
        "source_m2_7_config": ROOT / config["source_m2_7_config"],
        "source_m2_7_summary": ROOT / config["source_m2_7_results"] / "summary.json",
    }
    for name, path in paths.items():
        expected = str(config[f"{name}_sha256"])
        observed = _sha256(path)
        if observed != expected:
            raise RuntimeError(f"{name} hash drift: {observed} != {expected}")
    m26 = _read_json(paths["source_m2_6_summary"])
    m27 = _read_json(paths["source_m2_7_summary"])
    if m26["status"] != "M2_6_CAMERA_BLOCK_PRECONDITIONER_ORACLE_NO_GO":
        raise RuntimeError("M2.6 source status drifted")
    if m27["status"] != "M2_7_TARGET_NO_HARM_PARETO_ORACLE_NO_GO":
        raise RuntimeError("M2.7 source status drifted")
    for decision in m27["decisions"].values():
        failed = {name for name, value in decision.get("checks", {}).items() if not value}
        if not {"development_harm_rate", "development_worst_case"}.issubset(failed):
            raise RuntimeError("M2.7 is not the declared target/no-harm failure")
    return paths


def _aggregate_fixed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str, int, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["method"]),
            int(row["model_seed"]),
            str(row["split"]),
            int(row["projection_iterations"]),
            float(row["interpolation_fraction"]),
        )
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (method, seed, split, iteration, fraction), values in sorted(grouped.items()):
        output.append(
            {
                "method": method,
                "model_seed": seed,
                "split": split,
                "projection_iterations": iteration,
                "interpolation_fraction": fraction,
                "case_count": len(values),
                "paired_call_budget": int(values[0]["paired_call_budget"]),
                "field_gain_mean": _mean(row["field_gain"] for row in values),
                "h1_gain_mean": _mean(row["h1_gain"] for row in values),
                "reprojection_ratio_mean": _mean(
                    row["reprojection_ratio_to_matched_cgls"] for row in values
                ),
                "reprojection_ratio_maximum": max(
                    float(row["reprojection_ratio_to_matched_cgls"])
                    for row in values
                ),
                "field_harm_rate": _mean(row["field_harm"] for row in values),
                "worst_field_gain": min(float(row["field_gain"]) for row in values),
            }
        )
    return output


def _candidate_metrics(
    rows: list[dict[str, Any]],
    *,
    method: str,
    split: str,
    iteration: int,
    fraction: float,
) -> dict[str, Any]:
    values = [
        row
        for row in rows
        if row["method"] == method
        and row["split"] == split
        and int(row["projection_iterations"]) == iteration
        and float(row["interpolation_fraction"]) == fraction
    ]
    if not values:
        raise RuntimeError("missing fixed-interpolation candidate rows")
    seed_means = []
    for seed in sorted({int(row["model_seed"]) for row in values}):
        seed_means.append(
            _mean(
                row["field_gain"]
                for row in values
                if int(row["model_seed"]) == seed
            )
        )
    budgets = {int(row["paired_call_budget"]) for row in values}
    if len(budgets) != 1:
        raise RuntimeError("fixed interpolation budget drift")
    return {
        "case_model_count": len(values),
        "paired_call_budget": budgets.pop(),
        "field_gain_mean": _mean(row["field_gain"] for row in values),
        "h1_gain_mean": _mean(row["h1_gain"] for row in values),
        "reprojection_ratio_mean": _mean(
            row["reprojection_ratio_to_matched_cgls"] for row in values
        ),
        "reprojection_ratio_maximum": max(
            float(row["reprojection_ratio_to_matched_cgls"]) for row in values
        ),
        "field_harm_rate": _mean(row["field_harm"] for row in values),
        "worst_field_gain": min(float(row["field_gain"]) for row in values),
        "per_model_seed_field_gain_means": seed_means,
    }


def _fixed_decisions(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    gates = config["decision_gates"]
    output: dict[str, Any] = {}
    for method in config["methods"]:
        candidates = []
        for iteration in config["projection"]["iterations"]:
            for fraction in config["interpolation"]["fixed_fractions"]:
                metrics = _candidate_metrics(
                    rows,
                    method=str(method),
                    split="development",
                    iteration=int(iteration),
                    fraction=float(fraction),
                )
                checks = {
                    "field_gain": metrics["field_gain_mean"]
                    >= float(gates["development_field_gain_minimum"]),
                    "h1_gain": metrics["h1_gain_mean"]
                    >= float(gates["development_h1_gain_minimum"]),
                    "mean_reprojection": metrics["reprojection_ratio_mean"]
                    <= float(gates["development_reprojection_ratio_maximum"]),
                    "worst_reprojection": metrics["reprojection_ratio_maximum"]
                    <= float(gates["development_worst_reprojection_ratio_maximum"]),
                    "harm_rate": metrics["field_harm_rate"]
                    <= float(gates["field_harm_rate_maximum"]),
                    "worst_field_gain": metrics["worst_field_gain"]
                    >= float(gates["worst_field_gain_minimum"]),
                    "all_seed_means_positive": all(
                        value > 0.0
                        for value in metrics["per_model_seed_field_gain_means"]
                    ),
                    "paired_budget": metrics["paired_call_budget"]
                    <= int(gates["maximum_paired_call_budget"]),
                }
                candidates.append(
                    {
                        "projection_iterations": int(iteration),
                        "interpolation_fraction": float(fraction),
                        "development": metrics,
                        "development_checks": checks,
                        "development_eligible": all(checks.values()),
                    }
                )
        eligible = [value for value in candidates if value["development_eligible"]]
        eligible.sort(
            key=lambda value: (
                -float(value["development"]["field_gain_mean"]),
                int(value["projection_iterations"]),
                -float(value["interpolation_fraction"]),
            )
        )
        if not eligible:
            output[str(method)] = {
                "screened_candidates": candidates,
                "selection": None,
                "passed_fixed_interpolation_gate": False,
            }
            continue
        chosen = eligible[0]
        ood = _candidate_metrics(
            rows,
            method=str(method),
            split="ood",
            iteration=int(chosen["projection_iterations"]),
            fraction=float(chosen["interpolation_fraction"]),
        )
        ood_checks = {
            "field_gain": ood["field_gain_mean"]
            >= float(gates["ood_field_gain_minimum"]),
            "h1_gain": ood["h1_gain_mean"] >= float(gates["ood_h1_gain_minimum"]),
            "mean_reprojection": ood["reprojection_ratio_mean"]
            <= float(gates["ood_reprojection_ratio_maximum"]),
            "worst_reprojection": ood["reprojection_ratio_maximum"]
            <= float(gates["ood_worst_reprojection_ratio_maximum"]),
            "harm_rate": ood["field_harm_rate"]
            <= float(gates["field_harm_rate_maximum"]),
            "worst_field_gain": ood["worst_field_gain"]
            >= float(gates["worst_field_gain_minimum"]),
            "all_seed_means_positive": all(
                value > 0.0 for value in ood["per_model_seed_field_gain_means"]
            ),
        }
        output[str(method)] = {
            "screened_candidates": candidates,
            "selection": {
                "projection_iterations": int(chosen["projection_iterations"]),
                "interpolation_fraction": float(chosen["interpolation_fraction"]),
                "used_ood_for_selection": False,
            },
            "development": chosen["development"],
            "ood": ood,
            "ood_checks": ood_checks,
            "passed_fixed_interpolation_gate": all(ood_checks.values()),
        }
    return output


def _ceiling_metrics(
    rows: list[dict[str, Any]], *, method: str, split: str, iteration: int
) -> dict[str, Any]:
    values = [
        row
        for row in rows
        if row["method"] == method
        and row["split"] == split
        and int(row["projection_iterations"]) == iteration
    ]
    if not values:
        raise RuntimeError("missing truth-oracle ceiling rows")
    feasible = [row for row in values if int(row["reprojection_feasible"]) == 1]
    budgets = {int(row["paired_call_budget"]) for row in values}
    if len(budgets) != 1:
        raise RuntimeError("truth ceiling budget drift")
    metrics: dict[str, Any] = {
        "case_model_count": len(values),
        "paired_call_budget": budgets.pop(),
        "reprojection_feasible_rate": len(feasible) / len(values),
    }
    if len(feasible) != len(values):
        metrics.update(
            {
                "field_gain_mean": None,
                "h1_gain_mean": None,
                "reprojection_ratio_mean": None,
                "field_harm_rate": None,
                "worst_field_gain": None,
            }
        )
        return metrics
    metrics.update(
        {
            "field_gain_mean": _mean(row["field_gain"] for row in feasible),
            "h1_gain_mean": _mean(row["h1_gain"] for row in feasible),
            "reprojection_ratio_mean": _mean(
                row["reprojection_ratio_to_matched_cgls"] for row in feasible
            ),
            "field_harm_rate": _mean(row["field_harm"] for row in feasible),
            "worst_field_gain": min(float(row["field_gain"]) for row in feasible),
            "alpha_mean": _mean(row["truth_oracle_alpha"] for row in feasible),
            "alpha_minimum": min(float(row["truth_oracle_alpha"]) for row in feasible),
            "alpha_maximum": max(float(row["truth_oracle_alpha"]) for row in feasible),
        }
    )
    return metrics


def _ceiling_decisions(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    gates = config["decision_gates"]
    output: dict[str, Any] = {}
    for method in config["methods"]:
        candidates = []
        for iteration in config["projection"]["iterations"]:
            metrics = _ceiling_metrics(
                rows,
                method=str(method),
                split="development",
                iteration=int(iteration),
            )
            checks = {
                "all_cases_reprojection_feasible": metrics[
                    "reprojection_feasible_rate"
                ]
                == 1.0,
                "paired_budget": metrics["paired_call_budget"]
                <= int(gates["maximum_paired_call_budget"]),
            }
            if metrics["field_gain_mean"] is not None:
                checks.update(
                    {
                        "field_gain": float(metrics["field_gain_mean"])
                        >= float(gates["development_field_gain_minimum"]),
                        "h1_gain": float(metrics["h1_gain_mean"])
                        >= float(gates["development_h1_gain_minimum"]),
                        "harm_rate": float(metrics["field_harm_rate"])
                        <= float(gates["field_harm_rate_maximum"]),
                        "worst_field_gain": float(metrics["worst_field_gain"])
                        >= float(gates["worst_field_gain_minimum"]),
                    }
                )
            candidates.append(
                {
                    "projection_iterations": int(iteration),
                    "development": metrics,
                    "development_checks": checks,
                    "development_eligible": all(checks.values()),
                }
            )
        eligible = [value for value in candidates if value["development_eligible"]]
        eligible.sort(
            key=lambda value: -float(value["development"]["field_gain_mean"])
        )
        if not eligible:
            output[str(method)] = {
                "screened_candidates": candidates,
                "selection": None,
                "passed_truth_oracle_ceiling": False,
            }
            continue
        chosen = eligible[0]
        ood = _ceiling_metrics(
            rows,
            method=str(method),
            split="ood",
            iteration=int(chosen["projection_iterations"]),
        )
        ood_checks = {
            "all_cases_reprojection_feasible": ood["reprojection_feasible_rate"]
            == 1.0,
            "field_gain": ood["field_gain_mean"] is not None
            and float(ood["field_gain_mean"]) >= float(gates["ood_field_gain_minimum"]),
            "h1_gain": ood["h1_gain_mean"] is not None
            and float(ood["h1_gain_mean"]) >= float(gates["ood_h1_gain_minimum"]),
            "harm_rate": ood["field_harm_rate"] is not None
            and float(ood["field_harm_rate"])
            <= float(gates["field_harm_rate_maximum"]),
            "worst_field_gain": ood["worst_field_gain"] is not None
            and float(ood["worst_field_gain"])
            >= float(gates["worst_field_gain_minimum"]),
        }
        output[str(method)] = {
            "screened_candidates": candidates,
            "selection": {
                "projection_iterations": int(chosen["projection_iterations"]),
                "truth_used_by_candidate": True,
                "used_ood_for_selection": False,
            },
            "development": chosen["development"],
            "ood": ood,
            "ood_checks": ood_checks,
            "passed_truth_oracle_ceiling": all(ood_checks.values()),
        }
    return output


def _plot(
    output: Path,
    *,
    fixed_aggregate: list[dict[str, Any]],
    ceiling_decisions: dict[str, Any],
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.0), constrained_layout=True)
    colors = {9: "#146c94", 10: "#d95f59"}
    styles = {"jacru_m2": "-", "pooled_cnn": "--"}
    labels = {"jacru_m2": "JACRU-M2", "pooled_cnn": "Pooled CNN"}
    for method in labels:
        for iteration in colors:
            rows = [
                row
                for row in fixed_aggregate
                if row["method"] == method
                and row["split"] == "development"
                and int(row["projection_iterations"]) == iteration
            ]
            grouped: dict[float, list[dict[str, Any]]] = {}
            for row in rows:
                grouped.setdefault(float(row["interpolation_fraction"]), []).append(row)
            xs = sorted(grouped)
            field = [_mean(row["field_gain_mean"] for row in grouped[x]) for x in xs]
            reprojection = [
                _mean(row["reprojection_ratio_mean"] for row in grouped[x]) for x in xs
            ]
            harm = [_mean(row["field_harm_rate"] for row in grouped[x]) for x in xs]
            label = f"{labels[method]} · K={iteration}"
            axes[0, 0].plot(xs, field, color=colors[iteration], linestyle=styles[method], marker="o", label=label)
            axes[0, 1].plot(xs, reprojection, color=colors[iteration], linestyle=styles[method], marker="o")
            axes[1, 0].plot(xs, harm, color=colors[iteration], linestyle=styles[method], marker="o")
    axes[0, 0].axhline(0.05, color="#222", linestyle=":")
    axes[0, 0].set_ylabel("field gain vs best matched classic")
    axes[0, 0].set_title("fixed global interpolation")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].set_yscale("log")
    axes[0, 1].axhline(1.1, color="#222", linestyle=":")
    axes[0, 1].set_ylabel("reprojection ratio to matched CGLS")
    axes[0, 1].set_title("data-consistency gate")
    axes[1, 0].axhline(0.05, color="#222", linestyle=":")
    axes[1, 0].set_ylabel("field harm rate")
    axes[1, 0].set_title("tail-risk gate")
    for axis in axes.reshape(-1)[:3]:
        axis.set_xlabel("interpolation fraction alpha")

    axis = axes[1, 1]
    x = np.arange(len(ceiling_decisions), dtype=np.float64)
    field = []
    worst = []
    tick_labels = []
    for method, decision in ceiling_decisions.items():
        tick_labels.append(labels[method])
        candidates = decision["screened_candidates"]
        best = max(
            candidates,
            key=lambda value: (
                -math.inf
                if value["development"]["field_gain_mean"] is None
                else float(value["development"]["field_gain_mean"])
            ),
        )
        field.append(float(best["development"]["field_gain_mean"] or 0.0))
        worst.append(float(best["development"]["worst_field_gain"] or 0.0))
    axis.bar(x - 0.18, field, 0.36, color="#146c94", label="truth-oracle mean")
    axis.bar(x + 0.18, worst, 0.36, color="#d95f59", label="truth-oracle worst")
    axis.axhline(0.0, color="#222", linewidth=1)
    axis.axhline(-0.05, color="#555", linestyle=":")
    axis.set_xticks(x, tick_labels)
    axis.set_ylabel("field gain")
    axis.set_title("impossible-to-deploy calibration ceiling")
    axis.legend(frameon=False, fontsize=8)
    fig.suptitle("M2.8 interpolation/calibration ceiling · opened synthetic T0", fontsize=15, fontweight="bold")
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


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

    iterations = [int(value) for value in config["projection"]["iterations"]]
    fractions = [float(value) for value in config["interpolation"]["fixed_fractions"]]
    if iterations != sorted(set(iterations)) or max(iterations) > 10:
        raise ValueError("M2.8 iterations must be unique and no greater than 10")
    if fractions != sorted(set(fractions)) or not all(0.0 <= x <= 1.0 for x in fractions):
        raise ValueError("fixed interpolation fractions must be unique, sorted, and in [0,1]")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    fixture = m2._fixture_config(source_config)
    records = m2._prepare_records(source_config, fixture)
    device = m2._choose_device(args.device or source_config["training"]["device"])
    trained: list[dict[str, Any]] = []
    for method in config["methods"]:
        for seed in source_config["training"]["model_seeds"]:
            trained.append(
                m2._train_one(
                    method=str(method),
                    seed=int(seed),
                    config=source_config,
                    records=records,
                    device=device,
                    epoch_override=args.epochs,
                )
            )

    norm_cache, projectors, oracle_setup = m23._build_norm_and_oracle_caches(
        records=records,
        source_config=source_config,
        config=config,
    )
    baseline_rows = m23._matched_baselines(
        records=records,
        source_config=source_config,
        config=config,
        norm_cache=norm_cache,
        iterations=iterations,
    )
    baseline_lookup = m23._baseline_lookup(baseline_rows)
    block_cache: dict[str, tuple[Any, dict[str, Any]]] = {}
    fixed_rows: list[dict[str, Any]] = []
    ceiling_rows: list[dict[str, Any]] = []
    maximum_iteration = max(iterations)
    feature_forward_calls = int(
        config["matched_budget"]["learned_feature_preparation_forward_calls"]
    )
    feature_adjoint_calls = int(
        config["matched_budget"]["learned_feature_preparation_adjoint_calls"]
    )
    harm_threshold = float(config["decision_gates"]["field_harm_threshold_fraction"])
    per_case_ratio = float(
        config["interpolation"]["truth_oracle_per_case_reprojection_ratio_maximum"]
    )

    for run in trained:
        model = run["model"]
        model_device = next(model.parameters()).device
        for record in records:
            if record.split == "train":
                continue
            kwargs = m2._to_device(record.batch.model_kwargs(), model_device)
            with torch.no_grad():
                prediction, _ = model(**kwargs, return_gate=True)
            operator = record.case.inference.operator
            initial = prediction[0, 0].detach().cpu().to(operator.support)
            base = record.batch.base_field[0, 0].to(operator.support)
            observation = record.case.inference.observations_uv[0].to(operator.support)
            truth = record.case.evaluation.truth_volume[0, 0].to(operator.support)
            forward, adjoint = m2._operator_maps(operator)
            digest = record.case.inference.geometry.digest
            if digest not in block_cache:
                block_cache[digest] = m23._dense_camera_block_preconditioner(
                    matrix=projectors[digest].dense_active_matrix,
                    camera_index=record.case.inference.geometry.camera_index,
                    measurement_shape=tuple(observation.shape),
                    damping=0.0,
                )
            preconditioner, metadata = block_cache[digest]
            operator.reset_call_counts()
            path = matrix_free_measurement_projection_path(
                reference_field=base,
                learned_field=initial,
                forward=forward,
                adjoint=adjoint,
                support=operator.support,
                snapshot_iterations=[0, *iterations],
                preconditioner_apply=preconditioner,
                preconditioner_name="dense_exact_camera_block_jacobi_oracle",
                target_observation=observation,
            )
            expected_calls = {
                "forward_calls": maximum_iteration + 1,
                "adjoint_calls": maximum_iteration,
            }
            if operator.call_report() != expected_calls:
                raise RuntimeError("M2.8 algorithm call ledger drift")
            initial_residual = path.system_residuals_by_iteration[0]
            observation_norm = float(torch.linalg.vector_norm(observation).clamp_min(1e-30))
            for iteration in iterations:
                removed = path.removed_corrections_by_iteration[iteration]
                final_residual = path.system_residuals_by_iteration[iteration]
                projected_measurement = initial_residual - final_residual
                matched_cgls = baseline_lookup[(record.case.inference.case_id, iteration, "cgls_matched")]
                matched_huber = baseline_lookup[(record.case.inference.case_id, iteration, "huber_pdhg_matched")]
                best_field = min(
                    float(matched_cgls["field_relative_l2"]),
                    float(matched_huber["field_relative_l2"]),
                )
                best_h1 = min(
                    float(matched_cgls["h1_seminorm_relative_error"]),
                    float(matched_huber["h1_seminorm_relative_error"]),
                )
                matched_reprojection = float(
                    matched_cgls["measured_reprojection_relative_l2"]
                )
                paired_budget = feature_forward_calls + iteration + 1
                for fraction in fractions:
                    field = (initial - fraction * removed) * operator.support
                    operator.reset_call_counts()
                    score = m2._score_prediction(
                        record=record,
                        method=str(run["method"]),
                        model_seed=int(run["model_seed"]),
                        prediction=field,
                        gate=None,
                        correction_rms=None,
                        optimization_forward_calls=paired_budget,
                        optimization_adjoint_calls=feature_adjoint_calls + iteration,
                        grouped_adjoint_calls=1,
                        neural_inference_seconds=0.0,
                    )
                    field_gain = (best_field - float(score["field_relative_l2"])) / max(best_field, 1e-30)
                    fixed_rows.append(
                        {
                            **score,
                            "projection_iterations": iteration,
                            "interpolation_fraction": fraction,
                            "paired_call_budget": paired_budget,
                            "field_gain": field_gain,
                            "h1_gain": (
                                best_h1 - float(score["h1_seminorm_relative_error"])
                            )
                            / max(best_h1, 1e-30),
                            "reprojection_ratio_to_matched_cgls": float(
                                score["measured_reprojection_relative_l2"]
                            )
                            / max(matched_reprojection, 1e-30),
                            "field_harm": int(
                                float(score["field_relative_l2"])
                                > best_field * (1.0 + harm_threshold)
                            ),
                            "truth_used_by_candidate": False,
                            "exact_camera_block_setup_forward_equivalents": int(
                                projectors[digest].dense_active_matrix.shape[1] + 1
                            ),
                            "preconditioner_block_count": int(metadata["block_count"]),
                        }
                    )

                threshold = per_case_ratio * matched_reprojection * observation_norm
                q = projected_measurement
                reprojection_interval = m23._convex_quadratic_feasible_interval(
                    quadratic=float(torch.sum(q * q)),
                    linear=-2.0 * float(torch.sum(initial_residual * q)),
                    constant=float(torch.sum(initial_residual * initial_residual))
                    - threshold * threshold,
                )
                common = {
                    "case_id": record.case.inference.case_id,
                    "split": record.split,
                    "family": record.family,
                    "base_seed": record.base_seed,
                    "method": str(run["method"]),
                    "model_seed": int(run["model_seed"]),
                    "projection_iterations": iteration,
                    "paired_call_budget": paired_budget,
                    "truth_used_by_candidate": True,
                    "candidate_deployable": False,
                    "per_case_reprojection_ratio_limit": per_case_ratio,
                    "exact_camera_block_setup_forward_equivalents": int(
                        projectors[digest].dense_active_matrix.shape[1] + 1
                    ),
                }
                if reprojection_interval is None:
                    ceiling_rows.append(
                        {
                            **common,
                            "reprojection_feasible": 0,
                            "feasible_alpha_lower": None,
                            "feasible_alpha_upper": None,
                            "truth_oracle_alpha": None,
                            "field_relative_l2": None,
                            "h1_seminorm_relative_error": None,
                            "measured_reprojection_relative_l2": None,
                            "clean_reprojection_relative_l2": None,
                            "field_gain": None,
                            "h1_gain": None,
                            "reprojection_ratio_to_matched_cgls": None,
                            "field_harm": None,
                        }
                    )
                    continue
                field_error = initial - truth
                alpha = m23._convex_quadratic_minimizer(
                    quadratic=float(torch.sum(removed * removed)),
                    linear=-2.0 * float(torch.sum(field_error * removed)),
                    interval=reprojection_interval,
                )
                oracle_field = (initial - alpha * removed) * operator.support
                operator.reset_call_counts()
                oracle_score = m2._score_prediction(
                    record=record,
                    method=str(run["method"]),
                    model_seed=int(run["model_seed"]),
                    prediction=oracle_field,
                    gate=None,
                    correction_rms=None,
                    optimization_forward_calls=paired_budget,
                    optimization_adjoint_calls=feature_adjoint_calls + iteration,
                    grouped_adjoint_calls=1,
                    neural_inference_seconds=0.0,
                )
                oracle_field_gain = (
                    best_field - float(oracle_score["field_relative_l2"])
                ) / max(best_field, 1e-30)
                ceiling_rows.append(
                    {
                        **common,
                        "reprojection_feasible": 1,
                        "feasible_alpha_lower": reprojection_interval[0],
                        "feasible_alpha_upper": reprojection_interval[1],
                        "truth_oracle_alpha": alpha,
                        "field_relative_l2": float(oracle_score["field_relative_l2"]),
                        "h1_seminorm_relative_error": float(
                            oracle_score["h1_seminorm_relative_error"]
                        ),
                        "measured_reprojection_relative_l2": float(
                            oracle_score["measured_reprojection_relative_l2"]
                        ),
                        "clean_reprojection_relative_l2": float(
                            oracle_score["clean_reprojection_relative_l2"]
                        ),
                        "field_gain": oracle_field_gain,
                        "h1_gain": (
                            best_h1
                            - float(oracle_score["h1_seminorm_relative_error"])
                        )
                        / max(best_h1, 1e-30),
                        "reprojection_ratio_to_matched_cgls": float(
                            oracle_score["measured_reprojection_relative_l2"]
                        )
                        / max(matched_reprojection, 1e-30),
                        "field_harm": int(
                            float(oracle_score["field_relative_l2"])
                            > best_field * (1.0 + harm_threshold)
                        ),
                    }
                )

    fixed_aggregate = _aggregate_fixed(fixed_rows)
    fixed_decisions = _fixed_decisions(fixed_rows, config)
    ceiling_decisions = _ceiling_decisions(ceiling_rows, config)
    fixed_pass = any(
        value["passed_fixed_interpolation_gate"]
        for value in fixed_decisions.values()
    )
    ceiling_pass = any(
        value["passed_truth_oracle_ceiling"]
        for value in ceiling_decisions.values()
    )
    statuses = config["report_status"]
    status = (
        statuses["fixed_signal"]
        if fixed_pass
        else statuses["truth_oracle_headroom"]
        if ceiling_pass
        else statuses["no_go"]
    )
    summary = {
        "schema_version": config["report_schema_version"],
        "status": status,
        "evidence_level": config["evidence_level"],
        "source_config_sha256": _sha256(config_path),
        **{name + "_sha256": _sha256(path) for name, path in sources.items()},
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
        "fixed_interpolation_row_count": len(fixed_rows),
        "truth_oracle_ceiling_row_count": len(ceiling_rows),
        "matched_baseline_row_count": len(baseline_rows),
        "training_runs": [
            {key: value for key, value in run.items() if key not in {"model", "history"}}
            for run in trained
        ],
        "retrospective_dense_oracle_setup": oracle_setup,
        "fixed_interpolation_decisions": fixed_decisions,
        "truth_oracle_ceiling_decisions": ceiling_decisions,
        "authorization": {
            "claim_deployable_algorithm": False,
            "claim_method_superiority": False,
            "claim_real_bost_generalization": False,
            "open_fresh_or_final": False,
            "continue_fixed_interpolation_research": bool(fixed_pass),
            "continue_observable_calibration_research": bool(
                ceiling_pass and not fixed_pass
            ),
            "continue_noise_aware_target_or_fail_closed_research": True,
        },
        "claim_boundary": config["claim_boundary"],
        "public_export_policy": {
            "contains_model_checkpoints": False,
            "contains_restricted_papers": False,
            "contains_private_experimental_arrays": False,
            "contains_truth_or_observation_arrays": False,
        },
    }
    _write_csv(output / "fixed_interpolation_rows.csv", fixed_rows)
    _write_csv(output / "fixed_interpolation_aggregate.csv", fixed_aggregate)
    _write_csv(output / "truth_oracle_ceiling_rows.csv", ceiling_rows)
    _write_csv(output / "matched_baseline_rows.csv", baseline_rows)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _plot(
        output / "diagnostic",
        fixed_aggregate=fixed_aggregate,
        ceiling_decisions=ceiling_decisions,
    )
    (output / "README.md").write_text(
        f"# JACRU-M2.8 interpolation/calibration ceiling\n\nStatus: `{status}`\n\n"
        "Fixed alpha is a deployable-form control but still uses an excluded exact "
        "camera-block setup. The per-sample truth-oracle alpha is evaluator-only and "
        "cannot be used by an algorithm. This opened synthetic packet never authorizes "
        "fresh/final, method superiority, runtime, or real-BOST claims.\n",
        encoding="utf-8",
    )
    artifacts = (
        "README.md",
        "diagnostic.pdf",
        "diagnostic.png",
        "fixed_interpolation_aggregate.csv",
        "fixed_interpolation_rows.csv",
        "matched_baseline_rows.csv",
        "summary.json",
        "truth_oracle_ceiling_rows.csv",
    )
    (output / "checksums.sha256").write_text(
        "\n".join(f"{_sha256(output / name)}  {name}" for name in artifacts) + "\n",
        encoding="ascii",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
