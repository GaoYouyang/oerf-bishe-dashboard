#!/usr/bin/env python3
"""Post-open M2.1 diagnostic for data consistency versus field accuracy.

This script deliberately reuses the already opened M2-T0 splits.  It retrains
the frozen JACRU-M2 and pooled-CNN arms, reproduces their zero-step scores, and
then applies two truth-free deterministic correction paths.  Its output may
motivate a future preregistration; it cannot authorize OOD/fresh opening or a
method-superiority claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo_t16_operator.jacru_m2_data_consistency import data_consistency_path
from site_tools import run_jacru_m2_learned_residual_gate as m2


DEFAULT_CONFIG = (
    ROOT
    / "demo_t16_operator"
    / "configs"
    / "jacru_m2_1_matched_data_consistency_postopen_v1_1.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "demo_t16_operator"
    / "results"
    / "jacru_m2_1_matched_data_consistency_postopen_public"
)
SCHEMA = "jacru-m2-1-matched-data-consistency-postopen-report-1.1"


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
        raise ValueError("config must be one JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_l2(value: torch.Tensor, reference: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(value - reference)
        / torch.linalg.vector_norm(reference).clamp_min(1e-30)
    )


def _source_metric_rows(path: Path) -> dict[tuple[str, int, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (row["method"], int(row["model_seed"]), row["case_id"]): row
        for row in rows
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _matched_baseline_rows(
    *,
    records: list[m2.PreparedRecord],
    source_config: dict[str, Any],
    diagnostic_config: dict[str, Any],
    norm_cache: dict[str, dict[str, Any]],
    steps: list[int],
) -> list[dict[str, Any]]:
    """Score three non-learned controls at every learned total-call budget."""

    rows: list[dict[str, Any]] = []
    base_iterations = int(source_config["physical_budget"]["cgls_base_iterations"])
    spacing_xyz = m2._spacing(m2._fixture_config(source_config))
    for record in records:
        if record.split == "train":
            continue
        case = record.case
        operator = case.inference.operator
        observation = case.inference.observations_uv[0]
        support = operator.support.detach().clone()
        forward, adjoint = m2._operator_maps(operator)
        norm = norm_cache[case.inference.geometry.digest]
        tau = float(diagnostic_config["step_safety_factor"]) / float(norm["bound"])
        base_field = record.batch.base_field[0, 0].to(operator.support)
        for step in steps:
            total_calls = base_iterations + 1 + int(step)
            for method in ("cgls_matched", "huber_pdhg_matched"):
                operator.reset_call_counts()
                if method == "cgls_matched":
                    result = m2.cgls_baseline(
                        observation,
                        forward=forward,
                        adjoint=adjoint,
                        support=support,
                        spacing_xyz=spacing_xyz,
                        iterations=total_calls,
                    )
                else:
                    result = m2.edge_preserving_pdhg_baseline(
                        observation,
                        forward=forward,
                        adjoint=adjoint,
                        support=support,
                        spacing_xyz=spacing_xyz,
                        iterations=total_calls,
                        regularization_weight=0.001,
                        data_norm_squared_bound=float(norm["bound"]),
                        penalty="huber",
                        huber_delta=0.08,
                        step_safety=0.98,
                    )
                if result.forward_calls != total_calls or result.adjoint_calls != total_calls:
                    raise RuntimeError(f"matched {method} call budget drifted")
                row = m2._score_prediction(
                    record=record,
                    method=method,
                    model_seed=-1,
                    prediction=result.field,
                    gate=None,
                    correction_rms=None,
                    optimization_forward_calls=total_calls,
                    optimization_adjoint_calls=total_calls,
                    grouped_adjoint_calls=0,
                    neural_inference_seconds=0.0,
                )
                row.update(
                    {
                        "matched_step": int(step),
                        "total_calls": total_calls,
                        "baseline_kind": method,
                        "dc_step_size": None,
                        "operator_norm_squared_bound": float(norm["bound"]),
                    }
                )
                rows.append(row)

            # CGLS-12 plus (step+1) ordinary Landweber pairs has exactly the
            # same total F/A count as learned feature preparation plus step
            # post-corrections.  It detects gains caused by extra iterations.
            landweber_steps = int(step) + 1
            operator.reset_call_counts()
            path = data_consistency_path(
                initial_field=base_field,
                observation=observation,
                forward=forward,
                adjoint=adjoint,
                support=support,
                step_size=tau,
                operator_norm_squared_bound=float(norm["bound"]),
                snapshot_steps=(0, landweber_steps),
                mode="measurement_pullback",
            )
            if operator.call_report() != {
                "forward_calls": landweber_steps,
                "adjoint_calls": landweber_steps,
            }:
                raise RuntimeError("matched base-only Landweber call budget drifted")
            current = path.fields_by_step[landweber_steps]
            row = m2._score_prediction(
                record=record,
                method="base_landweber_matched",
                model_seed=-1,
                prediction=current,
                gate=None,
                correction_rms=float(torch.sqrt(torch.mean((current - base_field).square()))),
                optimization_forward_calls=total_calls,
                optimization_adjoint_calls=total_calls,
                grouped_adjoint_calls=0,
                neural_inference_seconds=0.0,
            )
            row.update(
                {
                    "matched_step": int(step),
                    "total_calls": total_calls,
                    "baseline_kind": "base_landweber_matched",
                    "dc_step_size": tau,
                    "operator_norm_squared_bound": float(norm["bound"]),
                }
            )
            rows.append(row)
    return rows


def _aggregate_baselines(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row["method"]), str(row["split"]), int(row["matched_step"])),
            [],
        ).append(row)
    output: list[dict[str, Any]] = []
    for (method, split, step), values in sorted(grouped.items()):
        output.append(
            {
                "method": method,
                "split": split,
                "matched_step": step,
                "total_calls": int(values[0]["total_calls"]),
                "case_count": len(values),
                "field_relative_l2_mean": float(
                    np.mean([row["field_relative_l2"] for row in values])
                ),
                "h1_seminorm_relative_error_mean": float(
                    np.mean([row["h1_seminorm_relative_error"] for row in values])
                ),
                "measured_reprojection_relative_l2_mean": float(
                    np.mean([row["measured_reprojection_relative_l2"] for row in values])
                ),
            }
        )
    return output


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["method"]),
            int(row["model_seed"]),
            str(row["split"]),
            str(row["dc_mode"]),
            int(row["dc_steps"]),
        )
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (method, seed, split, mode, steps), values in sorted(grouped.items()):
        output.append(
            {
                "method": method,
                "model_seed": seed,
                "split": split,
                "dc_mode": mode,
                "dc_steps": steps,
                "case_count": len(values),
                "field_relative_l2_mean": float(
                    np.mean([row["field_relative_l2"] for row in values])
                ),
                "h1_seminorm_relative_error_mean": float(
                    np.mean([row["h1_seminorm_relative_error"] for row in values])
                ),
                "measured_reprojection_relative_l2_mean": float(
                    np.mean([row["measured_reprojection_relative_l2"] for row in values])
                ),
                "field_gain_to_best_matched_classical_mean": float(
                    np.mean(
                        [row["field_gain_to_best_matched_classical"] for row in values]
                    )
                ),
                "h1_gain_to_best_matched_classical_mean": float(
                    np.mean([row["h1_gain_to_best_matched_classical"] for row in values])
                ),
                "network_gain_to_base_landweber_mean": float(
                    np.mean([row["network_gain_to_base_landweber"] for row in values])
                ),
                "reprojection_ratio_to_matched_cgls_mean": float(
                    np.mean(
                        [row["reprojection_ratio_to_matched_cgls"] for row in values]
                    )
                ),
                "field_gain_to_best_matched_classical_minimum": float(
                    np.min(
                        [row["field_gain_to_best_matched_classical"] for row in values]
                    )
                ),
                "field_harm_rate": float(
                    np.mean([row["field_harm_to_best_matched_classical"] for row in values])
                ),
                "optimization_forward_calls": int(values[0]["optimization_forward_calls"]),
                "optimization_adjoint_calls": int(values[0]["optimization_adjoint_calls"]),
            }
        )
    return output


def _decisions(
    rows: list[dict[str, Any]],
    *,
    methods: list[str],
    modes: list[str],
    steps: list[int],
    gates: dict[str, Any],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for method in methods:
        for mode in modes:
            for step in steps:
                if step == 0 and mode != modes[0]:
                    continue
                key = f"{method}|{mode}|{step}"
                selected = [
                    row
                    for row in rows
                    if row["method"] == method
                    and row["dc_mode"] == mode
                    and int(row["dc_steps"]) == step
                ]
                diagnostics: dict[str, Any] = {}
                checks: dict[str, bool] = {}
                for split in ("development", "ood"):
                    values = [row for row in selected if row["split"] == split]
                    gains = np.asarray(
                        [row["field_gain_to_best_matched_classical"] for row in values],
                        dtype=np.float64,
                    )
                    h1_gains = np.asarray(
                        [row["h1_gain_to_best_matched_classical"] for row in values],
                        dtype=np.float64,
                    )
                    network_gains = np.asarray(
                        [row["network_gain_to_base_landweber"] for row in values],
                        dtype=np.float64,
                    )
                    ratios = np.asarray(
                        [row["reprojection_ratio_to_matched_cgls"] for row in values],
                        dtype=np.float64,
                    )
                    harm = gains < -float(gates["field_harm_threshold_fraction"])
                    diagnostics[f"{split}_field_gain_mean"] = float(np.mean(gains))
                    diagnostics[f"{split}_h1_gain_mean"] = float(np.mean(h1_gains))
                    diagnostics[f"{split}_network_gain_to_base_landweber_mean"] = float(
                        np.mean(network_gains)
                    )
                    diagnostics[f"{split}_reprojection_ratio_mean"] = float(np.mean(ratios))
                    diagnostics[f"{split}_field_harm_rate"] = float(np.mean(harm))
                    diagnostics[f"{split}_worst_field_gain"] = float(np.min(gains))
                    minimum_gain = float(
                        gates[
                            f"{split}_field_gain_to_best_matched_classical_minimum_fraction"
                        ]
                    )
                    minimum_h1_gain = float(
                        gates[
                            f"{split}_h1_gain_to_best_matched_classical_minimum_fraction"
                        ]
                    )
                    minimum_network_gain = float(
                        gates[
                            f"{split}_network_gain_to_base_landweber_minimum_fraction"
                        ]
                    )
                    maximum_ratio = float(
                        gates[f"{split}_reprojection_ratio_to_matched_cgls_maximum"]
                    )
                    checks[f"{split}_field_gain"] = float(np.mean(gains)) >= minimum_gain
                    checks[f"{split}_h1_gain"] = (
                        float(np.mean(h1_gains)) >= minimum_h1_gain
                    )
                    checks[f"{split}_network_marginal_gain"] = (
                        float(np.mean(network_gains)) >= minimum_network_gain
                    )
                    checks[f"{split}_reprojection"] = float(np.mean(ratios)) <= maximum_ratio
                    checks[f"{split}_harm"] = (
                        float(np.mean(harm)) <= float(gates["field_harm_rate_maximum"])
                    )
                    checks[f"{split}_worst_case"] = (
                        float(np.min(gains))
                        >= float(gates["worst_field_gain_minimum_fraction"])
                    )
                    seed_means = [
                        float(
                            np.mean(
                                [
                                    row["field_gain_to_best_matched_classical"]
                                    for row in values
                                    if int(row["model_seed"]) == seed
                                ]
                            )
                        )
                        for seed in sorted({int(row["model_seed"]) for row in values})
                    ]
                    diagnostics[f"{split}_per_seed_field_gain_means"] = seed_means
                    checks[f"{split}_all_seed_means_positive"] = (
                        not bool(gates["require_all_model_seed_mean_field_gains_positive"])
                        or all(value > 0.0 for value in seed_means)
                    )
                output[key] = {
                    "method": method,
                    "dc_mode": mode,
                    "dc_steps": step,
                    "checks": checks,
                    "diagnostics": diagnostics,
                    "passed_postopen_headroom_gate": all(checks.values()),
                }
    return output


def _plot(
    path: Path,
    aggregates: list[dict[str, Any]],
    *,
    methods: list[str],
    modes: list[str],
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), constrained_layout=True)
    colors = {"jacru_m2": "#146c94", "pooled_cnn": "#d95f59"}
    markers = {"measurement_pullback": "o", "base_nullspace_filter": "s"}
    labels = {
        "jacru_m2": "JACRU-M2",
        "pooled_cnn": "Pooled CNN",
        "measurement_pullback": "measured pullback",
        "base_nullspace_filter": "base-nullspace filter",
    }
    mean_rows: dict[tuple[str, str, str, int], dict[str, float]] = {}
    keys = {(r["method"], r["split"], r["dc_mode"], r["dc_steps"]) for r in aggregates}
    for key in keys:
        values = [
            row
            for row in aggregates
            if (row["method"], row["split"], row["dc_mode"], row["dc_steps"]) == key
        ]
        mean_rows[key] = {
            "field_gain": float(
                np.mean(
                    [row["field_gain_to_best_matched_classical_mean"] for row in values]
                )
            ),
            "reprojection_ratio": float(
                np.mean(
                    [row["reprojection_ratio_to_matched_cgls_mean"] for row in values]
                )
            ),
            "worst_gain": float(
                np.min(
                    [
                        row["field_gain_to_best_matched_classical_minimum"]
                        for row in values
                    ]
                )
            ),
            "calls": float(values[0]["optimization_forward_calls"]),
        }

    for column, split in enumerate(("development", "ood")):
        ax = axes[0, column]
        for method in methods:
            for mode in modes:
                values = sorted(
                    [
                        (steps, item)
                        for (m, s, mo, steps), item in mean_rows.items()
                        if m == method and s == split and mo == mode
                    ],
                    key=lambda value: value[0],
                )
                if mode != modes[0]:
                    values = [value for value in values if value[0] > 0]
                xs = [item["reprojection_ratio"] for _, item in values]
                ys = [100.0 * item["field_gain"] for _, item in values]
                ax.plot(
                    xs,
                    ys,
                    color=colors[method],
                    marker=markers[mode],
                    linewidth=1.8,
                    label=f"{labels[method]} · {labels[mode]}",
                )
                if mode == "measurement_pullback" and method == methods[0]:
                    for index, ((step, _), x, y) in enumerate(
                        zip(values, xs, ys, strict=True)
                    ):
                        offset = (5, 6 if index % 2 == 0 else -12)
                        ax.annotate(
                            str(step),
                            (x, y),
                            xytext=offset,
                            textcoords="offset points",
                            fontsize=8,
                            fontweight="bold",
                        )
        reprojection_gate = 1.10 if split == "development" else 1.15
        ax.axvspan(0.9, reprojection_gate, color="#2f9e73", alpha=0.08)
        ax.axvline(reprojection_gate, color="#222", linestyle="--", linewidth=1)
        ax.axhline(5.0 if split == "development" else 2.0, color="#222", linestyle=":", linewidth=1)
        ax.set_xscale("log")
        ax.set_xlabel("mean reprojection ratio to matched-call CGLS (log)")
        ax.set_ylabel("mean field gain to best matched classical (%)")
        split_label = "development" if split == "development" else "exploratory OOD"
        ax.set_title(f"{split_label}: truth gain vs measured consistency")

    for column, split in enumerate(("development", "ood")):
        ax = axes[1, column]
        for method in methods:
            for mode in modes:
                values = sorted(
                    [
                        (steps, item)
                        for (m, s, mo, steps), item in mean_rows.items()
                        if m == method and s == split and mo == mode
                    ],
                    key=lambda value: value[0],
                )
                if mode != modes[0]:
                    values = [value for value in values if value[0] > 0]
                ax.plot(
                    [step for step, _ in values],
                    [100.0 * item["worst_gain"] for _, item in values],
                    color=colors[method],
                    marker=markers[mode],
                    linewidth=1.8,
                    label=f"{labels[method]} · {labels[mode]}",
                )
        ax.axhline(-5.0, color="#222", linestyle="--", linewidth=1)
        ax.set_xlabel("additional A / A^T correction pairs")
        ax.set_ylabel("worst case field gain to best matched classical (%)")
        split_label = "development" if split == "development" else "exploratory OOD"
        ax.set_title(f"{split_label}: tail safety along the path")

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle(
        "M2.1 post-open data-consistency diagnostic\n"
        "numbers next to Pareto points are additional physical correction pairs",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    config_path = args.config.resolve()
    config = _read_json(config_path)
    source_config_path = ROOT / config["source_t0_config"]
    if _sha256(source_config_path) != config["source_t0_config_sha256"]:
        raise RuntimeError("source T0 config hash drifted")
    source_config = _read_json(source_config_path)
    methods = [str(value) for value in config["methods"]]
    modes = [str(value) for value in config["modes"]]
    steps = [int(value) for value in config["snapshot_steps"]]
    if 0 not in steps or any(value < 0 for value in steps):
        raise ValueError("snapshot_steps must include zero and remain non-negative")
    if not set(methods).issubset(set(source_config["methods"])):
        raise ValueError("post-open methods must be source T0 methods")
    if args.seed_limit is not None:
        if args.seed_limit < 1:
            raise ValueError("seed-limit must be positive")
        source_config = json.loads(json.dumps(source_config))
        for split in source_config["splits"].values():
            split["base_seeds"] = split["base_seeds"][: args.seed_limit]
        source_config["training"]["model_seeds"] = source_config["training"]["model_seeds"][: args.seed_limit]

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    fixture = m2._fixture_config(source_config)
    records = m2._prepare_records(source_config, fixture)
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

    source_result_dir = ROOT / config["source_t0_results"]
    source_rows = _source_metric_rows(source_result_dir / "metric_rows.csv")
    cgls_rows = {
        case_id: row
        for (method, seed, case_id), row in source_rows.items()
        if method == "cgls_13" and seed == -1
    }
    norm_cache: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.split == "train":
            continue
        digest = record.case.inference.geometry.digest
        if digest not in norm_cache:
            operator = record.case.inference.operator
            operator.reset_call_counts()
            norm_cache[digest] = m2._dense_norm_squared_bound(
                operator,
                batch_size=int(source_config["physical_budget"]["dense_norm_batch_size"]),
                safety_factor=float(
                    source_config["physical_budget"]["dense_norm_safety_factor"]
                ),
            )

    baseline_rows = _matched_baseline_rows(
        records=records,
        source_config=source_config,
        diagnostic_config=config,
        norm_cache=norm_cache,
        steps=steps,
    )
    baseline_lookup = {
        (row["case_id"], int(row["matched_step"]), row["method"]): row
        for row in baseline_rows
    }

    rows: list[dict[str, Any]] = []
    maximum_step = max(steps)
    base_iterations = int(source_config["physical_budget"]["cgls_base_iterations"])
    for trained_model in trained:
        model = trained_model["model"]
        model_device = next(model.parameters()).device
        for record in records:
            if record.split == "train":
                continue
            model_kwargs = m2._to_device(record.batch.model_kwargs(), model_device)
            with torch.no_grad():
                prediction, gate = model(**model_kwargs, return_gate=True)
            operator = record.case.inference.operator
            dtype = operator.support.dtype
            physical_device = operator.support.device
            # MPS cannot cast directly to float64.  Cross the device boundary
            # in float32 first, then adopt the physical CPU operator dtype.
            initial = prediction[0, 0].detach().cpu().to(
                device=physical_device,
                dtype=dtype,
            )
            base_field = record.batch.base_field[0, 0].to(
                device=physical_device, dtype=dtype
            )
            observation = record.case.inference.observations_uv[0].to(
                device=physical_device, dtype=dtype
            )
            support = operator.support.detach().clone()
            forward, adjoint = m2._operator_maps(operator)
            norm = norm_cache[record.case.inference.geometry.digest]
            tau = float(config["step_safety_factor"]) / float(norm["bound"])
            source_cgls = cgls_rows[record.case.inference.case_id]
            base_projection = forward(base_field)
            base_residual_norm = float(
                torch.linalg.vector_norm(observation - base_projection).clamp_min(1e-30)
            )
            for mode in modes:
                operator.reset_call_counts()
                path = data_consistency_path(
                    initial_field=initial,
                    base_field=base_field if mode == "base_nullspace_filter" else None,
                    observation=observation,
                    forward=forward,
                    adjoint=adjoint,
                    support=support,
                    step_size=tau,
                    operator_norm_squared_bound=float(norm["bound"]),
                    snapshot_steps=steps,
                    mode=mode,
                )
                calls = operator.call_report()
                if calls != {
                    "forward_calls": maximum_step,
                    "adjoint_calls": maximum_step,
                }:
                    raise RuntimeError(f"data-consistency call drift: {calls}")
                for step in steps:
                    if step == 0 and mode != modes[0]:
                        continue
                    current = path.fields_by_step[step]
                    correction = current - base_field
                    visible_correction_ratio = float(
                        torch.linalg.vector_norm(forward(correction)) / base_residual_norm
                    )
                    row = m2._score_prediction(
                        record=record,
                        method=str(trained_model["method"]),
                        model_seed=int(trained_model["model_seed"]),
                        prediction=current,
                        gate=float(gate[0, 0, 0, 0, 0].detach().cpu()),
                        correction_rms=float(torch.sqrt(torch.mean(correction.square()))),
                        optimization_forward_calls=base_iterations + 1 + step,
                        optimization_adjoint_calls=base_iterations + 1 + step,
                        grouped_adjoint_calls=1,
                        neural_inference_seconds=0.0,
                    )
                    matched_cgls = baseline_lookup[
                        (row["case_id"], int(step), "cgls_matched")
                    ]
                    matched_huber = baseline_lookup[
                        (row["case_id"], int(step), "huber_pdhg_matched")
                    ]
                    matched_landweber = baseline_lookup[
                        (row["case_id"], int(step), "base_landweber_matched")
                    ]
                    best_field_classical = min(
                        float(matched_cgls["field_relative_l2"]),
                        float(matched_huber["field_relative_l2"]),
                    )
                    best_h1_classical = min(
                        float(matched_cgls["h1_seminorm_relative_error"]),
                        float(matched_huber["h1_seminorm_relative_error"]),
                    )
                    field_value = float(row["field_relative_l2"])
                    h1_value = float(row["h1_seminorm_relative_error"])
                    field_source_cgls = float(source_cgls["field_relative_l2"])
                    reprojection_source_cgls = float(
                        source_cgls["measured_reprojection_relative_l2"]
                    )
                    row.update(
                        {
                            "dc_mode": mode,
                            "dc_steps": step,
                            "dc_step_size": tau,
                            "operator_norm_squared_bound": float(norm["bound"]),
                            "field_gain_to_source_cgls13": (
                                field_source_cgls - field_value
                            )
                            / field_source_cgls,
                            "reprojection_ratio_to_source_cgls13": float(
                                row["measured_reprojection_relative_l2"]
                            )
                            / max(reprojection_source_cgls, 1e-30),
                            "matched_cgls_field_relative_l2": float(
                                matched_cgls["field_relative_l2"]
                            ),
                            "matched_huber_field_relative_l2": float(
                                matched_huber["field_relative_l2"]
                            ),
                            "matched_base_landweber_field_relative_l2": float(
                                matched_landweber["field_relative_l2"]
                            ),
                            "field_gain_to_best_matched_classical": (
                                best_field_classical - field_value
                            )
                            / best_field_classical,
                            "h1_gain_to_best_matched_classical": (
                                best_h1_classical - h1_value
                            )
                            / best_h1_classical,
                            "network_gain_to_base_landweber": (
                                float(matched_landweber["field_relative_l2"])
                                - field_value
                            )
                            / float(matched_landweber["field_relative_l2"]),
                            "reprojection_ratio_to_matched_cgls": float(
                                row["measured_reprojection_relative_l2"]
                            )
                            / max(
                                float(matched_cgls["measured_reprojection_relative_l2"]),
                                1e-30,
                            ),
                            "observable_correction_to_base_residual_ratio": (
                                visible_correction_ratio
                            ),
                            "field_harm_to_best_matched_classical": int(
                                field_value
                                > best_field_classical
                                * (
                                    1.0
                                    + float(
                                        config["decision_gates"][
                                            "field_harm_threshold_fraction"
                                        ]
                                    )
                                )
                            ),
                        }
                    )
                    rows.append(row)

    reproduction_deltas: list[dict[str, Any]] = []
    for row in rows:
        if int(row["dc_steps"]) != 0:
            continue
        source = source_rows[(row["method"], int(row["model_seed"]), row["case_id"])]
        reproduction_deltas.append(
            {
                "method": row["method"],
                "model_seed": row["model_seed"],
                "case_id": row["case_id"],
                "field_absolute_delta": abs(
                    float(row["field_relative_l2"]) - float(source["field_relative_l2"])
                ),
                "reprojection_absolute_delta": abs(
                    float(row["measured_reprojection_relative_l2"])
                    - float(source["measured_reprojection_relative_l2"])
                ),
            }
        )
    reproduction = {
        "row_count": len(reproduction_deltas),
        "maximum_field_absolute_delta": max(
            row["field_absolute_delta"] for row in reproduction_deltas
        ),
        "maximum_reprojection_absolute_delta": max(
            row["reprojection_absolute_delta"] for row in reproduction_deltas
        ),
    }
    reproduction["passed_1e_6"] = bool(
        reproduction["maximum_field_absolute_delta"] <= 1e-6
        and reproduction["maximum_reprojection_absolute_delta"] <= 1e-6
    )
    if args.epochs is None and args.seed_limit is None and not reproduction["passed_1e_6"]:
        raise RuntimeError(f"zero-step source reproduction failed: {reproduction}")

    aggregates = _aggregate(rows)
    baseline_aggregates = _aggregate_baselines(baseline_rows)
    decisions = _decisions(
        rows,
        methods=methods,
        modes=modes,
        steps=steps,
        gates=config["decision_gates"],
    )
    any_headroom = any(
        value["passed_postopen_headroom_gate"] for value in decisions.values()
    )
    summary = {
        "schema_version": SCHEMA,
        "status": (
            "M2_1_POSTOPEN_HEADROOM_FOUND_NOT_CONFIRMATORY"
            if any_headroom
            else "M2_1_POSTOPEN_DATA_CONSISTENCY_NO_GO"
        ),
        "evidence_level": config["evidence_level"],
        "source_config_sha256": _sha256(config_path),
        "source_t0_config_sha256": _sha256(source_config_path),
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
        "metric_row_count": len(rows),
        "matched_baseline_metric_row_count": len(baseline_rows),
        "zero_step_source_reproduction": reproduction,
        "training_runs": [
            {
                key: value
                for key, value in run.items()
                if key not in {"model", "history"}
            }
            for run in trained
        ],
        "operator_norm_setup": norm_cache,
        "decisions": decisions,
        "aggregate": aggregates,
        "matched_baseline_aggregate": baseline_aggregates,
        "authorization": {
            "claim_method_superiority": False,
            "claim_real_bost_generalization": False,
            "claim_interface_detection": False,
            "open_fresh_or_final": False,
            "draft_new_preregistered_data_consistency_gate": any_headroom,
        },
        "claim_boundary": config["claim_boundary"],
        "public_export_policy": {
            "contains_model_checkpoints": False,
            "contains_restricted_papers": False,
            "contains_private_experimental_arrays": False,
        },
    }

    _write_csv(output / "metric_rows.csv", rows)
    _write_csv(output / "aggregate_rows.csv", aggregates)
    _write_csv(output / "matched_baseline_rows.csv", baseline_rows)
    _write_csv(output / "matched_baseline_aggregate_rows.csv", baseline_aggregates)
    _write_csv(output / "zero_step_reproduction.csv", reproduction_deltas)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _plot(output / "diagnostic", aggregates, methods=methods, modes=modes)
    readme = f"""# JACRU-M2.1 post-open data-consistency diagnostic

Status: `{summary['status']}`

This export reuses the opened synthetic M2-T0 train/development/exploratory-OOD
splits. It is diagnostic evidence, not confirmatory OOD/fresh evidence. Two
truth-free paths use one logical forward and one adjoint call per added step:
measured pullback minimizes `||Ax-y||^2`; base-nullspace filtering minimizes
`||A(x-x0)||^2`. A finite second path is only a near-null spectral filter, not
an exact null-space projection. Every learned point is compared with
`CGLS-(13+k)`, `Huber-PDHG-(13+k)`, and CGLS-12 plus `(k+1)` base-only
Landweber pairs at the same total logical F/A count. Dense numerical SVD setup
is reported separately and is not counted as reconstruction runtime or calls.

Zero-step reproduction passed: `{reproduction['passed_1e_6']}`. No checkpoint,
restricted paper, experimental array, or sealed split is exported. Even if a
post-open point meets the declared headroom gate, it only authorizes drafting a
new preregistered test with new data.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    artifact_names = (
        "README.md",
        "aggregate_rows.csv",
        "diagnostic.pdf",
        "diagnostic.png",
        "matched_baseline_aggregate_rows.csv",
        "matched_baseline_rows.csv",
        "metric_rows.csv",
        "summary.json",
        "zero_step_reproduction.csv",
    )
    checksums = "".join(f"{_sha256(output / name)}  {name}\n" for name in artifact_names)
    (output / "checksums.sha256").write_text(checksums, encoding="utf-8")
    print(json.dumps({"status": summary["status"], "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
