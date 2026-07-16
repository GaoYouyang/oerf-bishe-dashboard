#!/usr/bin/env python3
"""Train and audit camera-budget-matched direct 3D inverse operators."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import platform
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from .bost_physics import forward_volume
    from .data import BOSTDataset, generate_dataset, load_npz, split_indices
    from .direct_operator_data import (
        prepare_direct_operator_data,
        replace_lift_with_ridge,
        ridge_reconstruct,
    )
    from .models import make_model
    from .train_eval import (
        _gradient_relative,
        _masked_projection_relative,
        _relative_norm,
        collect_predictions,
        set_seed,
        train_model,
    )
except ImportError:
    from bost_physics import forward_volume
    from data import BOSTDataset, generate_dataset, load_npz, split_indices
    from direct_operator_data import prepare_direct_operator_data, replace_lift_with_ridge, ridge_reconstruct
    from models import make_model
    from train_eval import (
        _gradient_relative,
        _masked_projection_relative,
        _relative_norm,
        collect_predictions,
        set_seed,
        train_model,
    )


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "direct_operator_pilot.json"
METHODS = ["physics_lift", "ridge", "unet", "fno", "ridge_unet", "ridge_fno"]
NEURAL_METHODS = ["unet", "fno", "ridge_unet", "ridge_fno"]
LABELS = {
    "physics_lift": "train-calibrated physics lift",
    "ridge": "validation-tuned ridge",
    "unet": "FBP-lift residual 3D U-Net",
    "fno": "FBP-lift residual 3D FNO",
    "ridge_unet": "ridge-initialized residual 3D U-Net",
    "ridge_fno": "ridge-initialized residual 3D FNO",
}
METRICS = [
    "field_rel_l2",
    "gradient_rel_l2",
    "observed_reprojection_rel_l2",
    "audit_reprojection_rel_l2",
    "improvement_vs_classical_pct",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed-limit", type=int)
    parser.add_argument("--force-data", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_variant_indices(data: dict[str, np.ndarray]) -> np.ndarray:
    names = [str(value) for value in data["split_names"].tolist()]
    return np.flatnonzero(
        np.asarray([names[int(value)].startswith("test_") for value in data["split_id"]])
    )


def subset_budget_data(data: dict[str, np.ndarray], budget: int) -> dict[str, np.ndarray]:
    """Filter only variant-level arrays while preserving shared geometry metadata."""
    selected = np.flatnonzero(data["total_budget"] == int(budget))
    variant_count = len(data["field"])
    output = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray) and value.ndim > 0 and len(value) == variant_count:
            output[key] = value[selected]
        else:
            output[key] = value
    return output


def relative_field_error(prediction: np.ndarray, target: np.ndarray) -> float:
    return _relative_norm(prediction - target, target)


def tune_classical_baselines(
    data: dict[str, np.ndarray],
    budgets: list[int],
    ridge_grid: list[float],
) -> tuple[dict[int, float], dict[int, str], list[dict[str, object]]]:
    indices = split_indices(data)["val"]
    support = data["support"]
    operator = data["forward_matrix"]
    tuning_rows: list[dict[str, object]] = []
    selected_ridge: dict[int, float] = {}
    champions: dict[int, str] = {}
    for budget in budgets:
        subset = indices[data["total_budget"][indices] == int(budget)]
        physics_errors = [
            relative_field_error(data["lift"][index], data["field"][index])
            for index in subset
        ]
        physics_mean = float(np.mean(physics_errors))
        tuning_rows.append(
            {
                "total_budget": budget,
                "method": "physics_lift",
                "ridge_relative": "",
                "val_field_rel_l2": physics_mean,
                "selected": False,
            }
        )
        ridge_scores = []
        for ridge in ridge_grid:
            errors = []
            for index in subset:
                prediction = ridge_reconstruct(
                    data["observation"][index],
                    operator,
                    data["view_mask"][index],
                    ridge,
                    support,
                )
                errors.append(relative_field_error(prediction, data["field"][index]))
            score = float(np.mean(errors))
            ridge_scores.append((score, float(ridge)))
            tuning_rows.append(
                {
                    "total_budget": budget,
                    "method": "ridge",
                    "ridge_relative": ridge,
                    "val_field_rel_l2": score,
                    "selected": False,
                }
            )
        ridge_mean, ridge = min(ridge_scores)
        selected_ridge[budget] = ridge
        champions[budget] = "ridge" if ridge_mean <= physics_mean else "physics_lift"
        for row in tuning_rows:
            if int(row["total_budget"]) != budget:
                continue
            if row["method"] == champions[budget]:
                if row["method"] == "physics_lift" or math.isclose(
                    float(row["ridge_relative"]), ridge
                ):
                    row["selected"] = True
    return selected_ridge, champions, tuning_rows


def classical_predictions(
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    selected_ridge: dict[int, float],
) -> dict[str, np.ndarray]:
    ridge = []
    for index in indices:
        budget = int(data["total_budget"][index])
        ridge.append(
            ridge_reconstruct(
                data["observation"][index],
                data["forward_matrix"],
                data["view_mask"][index],
                selected_ridge[budget],
                data["support"],
            )
        )
    return {
        "physics_lift": data["lift"][indices].copy(),
        "ridge": np.stack(ridge),
    }


def prediction_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    clean: np.ndarray,
    observed_mask: np.ndarray,
    audit_mask: np.ndarray,
    operator: np.ndarray,
    baseline_error: float,
) -> dict[str, float]:
    projected = forward_volume(prediction, operator)
    field_error = relative_field_error(prediction, target)
    return {
        "field_rel_l2": field_error,
        "gradient_rel_l2": _gradient_relative(prediction, target),
        "observed_reprojection_rel_l2": _masked_projection_relative(
            projected, clean, observed_mask
        ),
        "audit_reprojection_rel_l2": _masked_projection_relative(
            projected, clean, audit_mask
        ),
        "improvement_vs_classical_pct": 100.0
        * (baseline_error - field_error)
        / (baseline_error + 1e-12),
    }


def evaluate_seed(
    seed: int,
    data: dict[str, np.ndarray],
    ridge_data: dict[str, np.ndarray],
    trained: dict[tuple[int, str], dict],
    dataset_config: dict,
    selected_ridge: dict[int, float],
    champions: dict[int, str],
    audit_query_index: int,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    indices = test_variant_indices(data)
    classical = classical_predictions(data, indices, selected_ridge)
    predictions: dict[str, dict[int, np.ndarray] | np.ndarray] = dict(classical)
    inference_ms: dict[str, float] = {"physics_lift": 0.0, "ridge": 0.0}
    for method in NEURAL_METHODS:
        mapped: dict[int, np.ndarray] = {}
        elapsed_values = []
        for budget in sorted({int(value) for value in data["total_budget"][indices]}):
            budget_indices = indices[data["total_budget"][indices] == budget]
            record = trained[(budget, method)]
            inference_data = ridge_data if method.startswith("ridge_") else data
            values, elapsed = collect_predictions(
                record["model"],
                BOSTDataset(inference_data, budget_indices),
                record["device"],
                int(dataset_config["training"]["batch_size"]),
            )
            mapped.update(
                {int(index): values[local_index, 0] for local_index, index in enumerate(budget_indices)}
            )
            elapsed_values.append(float(elapsed))
        predictions[method] = mapped
        inference_ms[method] = float(np.mean(elapsed_values))

    names = [str(value) for value in data["split_names"].tolist()]
    audit_mask = np.zeros(len(data["angles"]), dtype=np.float32)
    audit_mask[int(audit_query_index)] = 1.0
    rows: list[dict[str, object]] = []
    for local_index, variant_index in enumerate(indices):
        budget = int(data["total_budget"][variant_index])
        champion = champions[budget]
        baseline_prediction = classical[champion][local_index]
        baseline_error = relative_field_error(
            baseline_prediction, data["field"][variant_index]
        )
        for method in METHODS:
            prediction = (
                predictions[method][local_index]
                if method in {"physics_lift", "ridge"}
                else predictions[method][int(variant_index)]
            )
            row = {
                "model_seed": seed,
                "variant_index": int(variant_index),
                "source_index": int(data["source_index"][variant_index]),
                "sample_seed": int(data["sample_seed"][variant_index]),
                "source_split": names[int(data["split_id"][variant_index])],
                "family_id": int(data["family_id"][variant_index]),
                "noise_level": float(data["noise_level"][variant_index]),
                "total_budget": budget,
                "method": method,
                "classical_champion": champion,
                "ridge_relative": selected_ridge[budget],
                "inference_ms_per_sample": inference_ms[method],
            }
            row.update(
                prediction_metrics(
                    prediction,
                    data["field"][variant_index],
                    data["clean_observation"][variant_index],
                    data["view_mask"][variant_index],
                    audit_mask,
                    data["forward_matrix"],
                    baseline_error,
                )
            )
            rows.append(row)
    return rows, inference_ms


def collapse_model_seeds(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                int(row["source_index"]),
                str(row["source_split"]),
                int(row["total_budget"]),
                str(row["method"]),
            )
        ].append(row)
    output = []
    for (source_index, source_split, budget, method), subset in sorted(grouped.items()):
        first = subset[0]
        collapsed = {
            "source_index": source_index,
            "sample_seed": int(first["sample_seed"]),
            "source_split": source_split,
            "family_id": int(first["family_id"]),
            "noise_level": float(first["noise_level"]),
            "total_budget": budget,
            "method": method,
            "classical_champion": str(first["classical_champion"]),
            "ridge_relative": float(first["ridge_relative"]),
            "model_seed_count": len(subset),
        }
        for metric in METRICS:
            collapsed[metric] = float(np.mean([float(row[metric]) for row in subset]))
        output.append(collapsed)
    return output


def domain_equal_weights(rows: list[dict[str, object]]) -> np.ndarray:
    domains = sorted({str(row["source_split"]) for row in rows})
    counts = {
        domain: sum(str(row["source_split"]) == domain for row in rows)
        for domain in domains
    }
    return np.asarray(
        [1.0 / len(domains) / counts[str(row["source_split"])] for row in rows],
        dtype=np.float64,
    )


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    index = min(int(np.searchsorted(cumulative, quantile, side="left")), len(values) - 1)
    return float(values[order[index]])


def stratified_bootstrap_interval(
    rows: list[dict[str, object]],
    metric: str,
    rng: np.random.Generator,
    replicates: int,
) -> tuple[float, float]:
    domains = sorted({str(row["source_split"]) for row in rows})
    estimates = np.zeros(int(replicates), dtype=np.float64)
    for domain in domains:
        values = np.asarray(
            [float(row[metric]) for row in rows if row["source_split"] == domain],
            dtype=np.float64,
        )
        sampled = rng.integers(0, len(values), size=(int(replicates), len(values)))
        estimates += np.mean(values[sampled], axis=1) / len(domains)
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def summarize_clusters(
    rows: list[dict[str, object]],
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["total_budget"]), str(row["method"]))].append(row)
    rng = np.random.default_rng(int(bootstrap_seed))
    output = []
    for (budget, method), subset in sorted(grouped.items()):
        weights = domain_equal_weights(subset)
        gains = np.asarray(
            [float(row["improvement_vs_classical_pct"]) for row in subset], dtype=float
        )
        ci_low, ci_high = stratified_bootstrap_interval(
            subset,
            "improvement_vs_classical_pct",
            rng,
            bootstrap_replicates,
        )
        output.append(
            {
                "total_budget": budget,
                "method": method,
                "independent_field_count": len(subset),
                "model_seed_count": int(subset[0]["model_seed_count"]),
                "source_domain_count": len({str(row["source_split"]) for row in subset}),
                "classical_champion": str(subset[0]["classical_champion"]),
                "mean_field_rel_l2": float(
                    np.sum(weights * np.asarray([float(row["field_rel_l2"]) for row in subset]))
                ),
                "mean_audit_reprojection_rel_l2": float(
                    np.sum(
                        weights
                        * np.asarray(
                            [float(row["audit_reprojection_rel_l2"]) for row in subset]
                        )
                    )
                ),
                "mean_improvement_vs_classical_pct": float(np.sum(weights * gains)),
                "median_improvement_vs_classical_pct": weighted_quantile(gains, weights, 0.5),
                "p10_improvement_vs_classical_pct": weighted_quantile(gains, weights, 0.1),
                "harm_rate_gt_1pct_vs_classical": float(np.sum(weights * (gains < -1.0))),
                "improvement_ci95_cluster_low": ci_low,
                "improvement_ci95_cluster_high": ci_high,
            }
        )
    return output


def build_domain_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[
            (int(row["total_budget"]), str(row["method"]), str(row["source_split"]))
        ].append(row)
    return [
        {
            "total_budget": budget,
            "method": method,
            "source_split": split,
            "independent_field_count": len(subset),
            "mean_field_rel_l2": float(np.mean([float(row["field_rel_l2"]) for row in subset])),
            "mean_audit_reprojection_rel_l2": float(
                np.mean([float(row["audit_reprojection_rel_l2"]) for row in subset])
            ),
            "mean_improvement_vs_classical_pct": float(
                np.mean([float(row["improvement_vs_classical_pct"]) for row in subset])
            ),
        }
        for (budget, method, split), subset in sorted(grouped.items())
    ]


def summary_row(summary: list[dict[str, object]], budget: int, method: str) -> dict[str, object]:
    return next(
        row
        for row in summary
        if int(row["total_budget"]) == int(budget) and row["method"] == method
    )


def build_verdicts(
    summary: list[dict[str, object]],
    budgets: list[int],
    minimum_gain_pct: float,
    maximum_harm_rate: float,
) -> list[dict[str, object]]:
    rows = []
    for budget in budgets:
        for method in NEURAL_METHODS:
            item = summary_row(summary, budget, method)
            passed = (
                float(item["improvement_ci95_cluster_low"]) > float(minimum_gain_pct)
                and float(item["p10_improvement_vs_classical_pct"]) >= 0.0
                and float(item["harm_rate_gt_1pct_vs_classical"]) <= float(maximum_harm_rate)
            )
            rows.append(
                {
                    "total_budget": budget,
                    "method": method,
                    "minimum_gain_pct": minimum_gain_pct,
                    "maximum_harm_rate_gt_1pct": maximum_harm_rate,
                    "mean_gain_pct": item["mean_improvement_vs_classical_pct"],
                    "ci95_low": item["improvement_ci95_cluster_low"],
                    "ci95_high": item["improvement_ci95_cluster_high"],
                    "p10_gain_pct": item["p10_improvement_vs_classical_pct"],
                    "harm_rate_gt_1pct": item["harm_rate_gt_1pct_vs_classical"],
                    "development_gate_pass": bool(passed),
                }
            )
    return rows


def plot_improvement(summary: list[dict[str, object]], path: Path) -> None:
    budgets = sorted({int(row["total_budget"]) for row in summary})
    fig, ax = plt.subplots(figsize=(10.8, 5.3), constrained_layout=True)
    x = np.arange(len(budgets))
    width = 0.14
    colors = {
        "physics_lift": "#9aa1a5",
        "ridge": "#53666f",
        "unet": "#c98968",
        "fno": "#63a69d",
        "ridge_unet": "#aa5438",
        "ridge_fno": "#0f6e65",
    }
    for offset, method in enumerate(METHODS):
        values = [
            float(summary_row(summary, budget, method)["mean_improvement_vs_classical_pct"])
            for budget in budgets
        ]
        ax.bar(
            x + (offset - 2.5) * width,
            values,
            width=width,
            label=LABELS[method],
            color=colors[method],
        )
    ax.axhline(0.0, color="#343b3d", linewidth=1)
    ax.axhline(5.0, color="#ba6648", linewidth=1, linestyle="--", label="5% material gate")
    ax.set_xticks(x, [f"K={value}" for value in budgets])
    ax.set_ylabel("field improvement over validation-locked classical champion (%)")
    ax.set_title("Training-matched direct inverse-operator development audit")
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_domain_errors(domain_rows: list[dict[str, object]], path: Path) -> None:
    splits = sorted({str(row["source_split"]) for row in domain_rows})
    budgets = sorted({int(row["total_budget"]) for row in domain_rows})
    fig, axes = plt.subplots(1, len(budgets), figsize=(14.5, 4.8), sharey=True, constrained_layout=True)
    x = np.arange(len(splits))
    width = 0.14
    colors = ["#9aa1a5", "#53666f", "#c98968", "#63a69d", "#aa5438", "#0f6e65"]
    lookup = {
        (int(row["total_budget"]), str(row["method"]), str(row["source_split"])): row
        for row in domain_rows
    }
    for axis, budget in zip(np.atleast_1d(axes), budgets):
        for offset, (method, color) in enumerate(zip(METHODS, colors)):
            values = [
                float(lookup[(budget, method, split)]["mean_field_rel_l2"])
                for split in splits
            ]
            axis.bar(x + (offset - 2.5) * width, values, width=width, color=color, label=LABELS[method])
        axis.set_xticks(x, [value.replace("test_", "").replace("_", "\n") for value in splits])
        axis.set_title(f"K={budget}")
        axis.grid(True, axis="y", alpha=0.22)
    axes[0].set_ylabel("field relative L2, lower is better")
    axes[-1].legend(fontsize=7.5)
    fig.suptitle("Domain failures stay visible instead of being hidden by one mean")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_checksums(output_dir: Path, filenames: list[str]) -> None:
    lines = []
    for filename in filenames:
        digest = hashlib.sha256((output_dir / filename).read_bytes()).hexdigest()
        lines.append(f"{digest}  {filename}")
    (output_dir / "direct_operator_checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="ascii"
    )


def main() -> None:
    args = parse_args()
    experiment = read_json(args.config)
    dataset_path = args.config.parent / str(experiment["dataset_config"])
    dataset_config = read_json(dataset_path)
    if args.device is not None:
        dataset_config["training"]["device"] = args.device
    if args.epochs is not None:
        dataset_config["training"]["epochs"] = int(args.epochs)
        dataset_config["training"]["early_stop_patience"] = min(
            int(dataset_config["training"]["early_stop_patience"]), int(args.epochs)
        )
    output_dir = args.output_dir or ROOT / "results" / str(experiment["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = ROOT / "results" / str(experiment["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)

    base_dataset_path = ROOT / "results" / f"{dataset_config['name']}_dataset.npz"
    generate_dataset(dataset_config, base_dataset_path, force=args.force_data)
    base_data = load_npz(base_dataset_path)
    budgets = [int(value) for value in experiment["reconstruction_budgets"]]
    data = prepare_direct_operator_data(
        base_data,
        budgets,
        int(experiment["fixed_query_index"]),
        int(experiment["audit_query_index"]),
    )
    selected_ridge, champions, tuning_rows = tune_classical_baselines(
        data,
        budgets,
        [float(value) for value in experiment["ridge_relative_grid"]],
    )
    ridge_data = replace_lift_with_ridge(data, selected_ridge)
    write_csv(output_dir / "direct_operator_baseline_tuning.csv", tuning_rows)

    seeds = [int(value) for value in experiment["training_seeds"]]
    if args.seed_limit is not None:
        seeds = seeds[: int(args.seed_limit)]
    sample_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    for seed in seeds:
        set_seed(seed)
        run_config = copy.deepcopy(dataset_config)
        run_config["seed"] = seed
        seed_dir = work_dir / str(seed)
        seed_dir.mkdir(parents=True, exist_ok=True)
        trained = {}
        for budget in budgets:
            budget_inputs = {
                "fbp": subset_budget_data(data, budget),
                "ridge": subset_budget_data(ridge_data, budget),
            }
            budget_dir = seed_dir / f"K{budget}"
            budget_dir.mkdir(parents=True, exist_ok=True)
            for method in NEURAL_METHODS:
                backbone_name = method.removeprefix("ridge_")
                input_kind = "ridge" if method.startswith("ridge_") else "fbp"
                model_seed = seed + budget * 101 + NEURAL_METHODS.index(method) * 7
                set_seed(model_seed)
                budget_config = copy.deepcopy(run_config)
                budget_config["seed"] = model_seed
                model = make_model(
                    backbone_name,
                    budget_config["models"][backbone_name],
                    int(data["inputs"].shape[1]),
                    residual=True,
                )
                method_dir = budget_dir / method
                method_dir.mkdir(parents=True, exist_ok=True)
                record = train_model(
                    method,
                    model,
                    budget_inputs[input_kind],
                    budget_config,
                    method_dir,
                )
                trained[(budget, method)] = record
                training_rows.append(
                    {
                        "model_seed": seed,
                        "total_budget": budget,
                        "method": method,
                        "parameters": int(record["parameters"]),
                        "epochs_ran": int(record["epochs_ran"]),
                        "best_epoch": int(record["best_epoch"]),
                        "best_val_rel_l2": float(record["best_val_rel_l2"]),
                        "train_seconds": float(record["train_seconds"]),
                        "device": str(record["device"]),
                    }
                )
        rows, inference_ms = evaluate_seed(
            seed,
            data,
            ridge_data,
            trained,
            dataset_config,
            selected_ridge,
            champions,
            int(experiment["audit_query_index"]),
        )
        sample_rows.extend(rows)
        print(
            f"seed={seed}: rows={len(rows)}, unet={inference_ms['unet']:.3f} ms, "
            f"fno={inference_ms['fno']:.3f} ms, ridge-unet={inference_ms['ridge_unet']:.3f} ms, "
            f"ridge-fno={inference_ms['ridge_fno']:.3f} ms",
            flush=True,
        )

    cluster_rows = collapse_model_seeds(sample_rows)
    summary = summarize_clusters(
        cluster_rows,
        int(experiment["bootstrap_seed"]),
        int(experiment["bootstrap_replicates"]),
    )
    domain_rows = build_domain_rows(cluster_rows)
    verdicts = build_verdicts(
        summary,
        budgets,
        float(experiment["minimum_gain_pct"]),
        float(experiment["maximum_harm_rate_gt_1pct"]),
    )
    passed = sum(bool(row["development_gate_pass"]) for row in verdicts)
    status = (
        "DEVELOPMENT_PILOT_ALL_NEURAL_GATES_PASS"
        if passed == len(verdicts)
        else "DEVELOPMENT_PILOT_PARTIAL_NEURAL_GATES"
        if passed > 0
        else "DEVELOPMENT_PILOT_NO_NEURAL_GATE"
    )
    write_csv(output_dir / "direct_operator_training.csv", training_rows)
    write_csv(output_dir / "direct_operator_samples.csv", sample_rows)
    write_csv(output_dir / "direct_operator_clusters.csv", cluster_rows)
    write_csv(output_dir / "direct_operator_summary.csv", summary)
    write_csv(output_dir / "direct_operator_domains.csv", domain_rows)
    write_csv(output_dir / "direct_operator_verdicts.csv", verdicts)
    plot_improvement(summary, output_dir / "t16_direct_operator_improvement.png")
    plot_domain_errors(domain_rows, output_dir / "t16_direct_operator_domains.png")

    dashboard = {
        "experiment": experiment["name"],
        "scientific_status": status,
        "development_fields_only": bool(experiment["development_fields_only"]),
        "training_mask_matches_evaluation": True,
        "source_field_count": int(len(base_data["field"])),
        "independent_test_field_count": int(
            sum(
                int(spec["count"])
                for name, spec in dataset_config["splits"].items()
                if name.startswith("test_")
            )
        ),
        "model_seed_count": len(seeds),
        "reconstruction_budgets": budgets,
        "installed_camera_count": {str(value): value + 1 for value in budgets},
        "fixed_query_index": int(experiment["fixed_query_index"]),
        "audit_query_index": int(experiment["audit_query_index"]),
        "classical_champions": {str(key): value for key, value in champions.items()},
        "selected_ridge_relative": {str(key): value for key, value in selected_ridge.items()},
        "labels": LABELS,
        "summary": summary,
        "domains": domain_rows,
        "verdicts": verdicts,
    }
    (output_dir / "direct_operator_dashboard.json").write_text(
        json.dumps(dashboard, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    report = {
        "status": "completed_training_matched_direct_operator_development_pilot",
        "scientific_status": status,
        "experiment": experiment["name"],
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "requested_device": dataset_config["training"]["device"],
        },
        "protocol": {
            "reconstruction_budget": "K observed cameras enter both classical and neural reconstruction",
            "installed_camera_budget": "K plus one locked Q_audit camera",
            "training_mask_matches_evaluation": True,
            "q_audit_used_for_training_or_selection": False,
            "field_truth_used_for_training": True,
            "ridge_and_classical_champion_selected_on_validation_only": True,
            "test_unit": "three-dimensional source field after collapsing model seeds",
            "domain_weighting": "four test domains receive equal weight",
        },
        "dataset": {
            "source_fields": int(len(base_data["field"])),
            "budget_variants": int(len(data["field"])),
            "input_shape": list(data["inputs"].shape[1:]),
            "input_channels": [str(value) for value in data["input_channel_names"].tolist()],
            "ridge_residual_input_channels": [
                str(value) for value in ridge_data["input_channel_names"].tolist()
            ],
            "train_only_lift_calibration": [float(value) for value in data["calibration"]],
            "split_counts": {
                name: int(spec["count"]) for name, spec in dataset_config["splits"].items()
            },
        },
        "training": training_rows,
        "classical_champions": champions,
        "selected_ridge_relative": selected_ridge,
        "verdicts": verdicts,
        "claims_boundary": [
            "This is a development pilot on a linear 8x16x16 synthetic slice-stack forward model, not real OERF BOST evidence.",
            "Training and evaluation camera masks now match, but the fields and result have been inspected during development and are not a locked confirmatory test.",
            "Q_audit is an additional evaluation camera, so installed-camera count is K+1 even though reconstruction budget is K.",
            "The residual U-Net and FNO are tested from both an FBP-style lift and a validation-tuned ridge field; a ray-token or per-view encoder has not yet been implemented.",
            "A passing direct-operator gate only justifies testing operator warm-start NeRIF; it does not establish a publishable BOST method.",
            "Real-data evidence still requires OERF geometry, displacement/mask inputs, an accepted no-ground-truth audit, and a per-instance NeRIF baseline.",
        ],
    }
    (output_dir / "direct_operator_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_checksums(
        output_dir,
        [
            "direct_operator_baseline_tuning.csv",
            "direct_operator_training.csv",
            "direct_operator_samples.csv",
            "direct_operator_clusters.csv",
            "direct_operator_summary.csv",
            "direct_operator_domains.csv",
            "direct_operator_verdicts.csv",
            "direct_operator_dashboard.json",
            "direct_operator_report.json",
        ],
    )
    print(json.dumps({"status": status, "verdicts": verdicts}, indent=2, allow_nan=False))
    print(f"results: {output_dir}")


if __name__ == "__main__":
    main()
