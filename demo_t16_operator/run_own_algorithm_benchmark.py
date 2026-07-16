#!/usr/bin/env python3
"""Benchmark a provisional geometry-conditioned ray-set operator fairly."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
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
    from .data import BOSTDataset, generate_dataset, load_npz
    from .direct_operator_data import prepare_direct_operator_data, replace_lift_with_ridge
    from .models import count_parameters, make_model
    from .own_algorithm_data import append_ray_view_channels
    from .own_algorithm_models import GridDeepONetResidual, RaySetResidualOperator
    from .run_direct_operator_pilot import (
        METRICS,
        build_domain_rows,
        classical_predictions,
        collapse_model_seeds,
        domain_equal_weights,
        prediction_metrics,
        stratified_bootstrap_interval,
        subset_budget_data,
        summarize_clusters,
        test_variant_indices,
        tune_classical_baselines,
        weighted_quantile,
    )
    from .train_eval import collect_predictions, set_seed, train_model
except ImportError:
    from data import BOSTDataset, generate_dataset, load_npz
    from direct_operator_data import prepare_direct_operator_data, replace_lift_with_ridge
    from models import count_parameters, make_model
    from own_algorithm_data import append_ray_view_channels
    from own_algorithm_models import GridDeepONetResidual, RaySetResidualOperator
    from run_direct_operator_pilot import (
        METRICS,
        build_domain_rows,
        classical_predictions,
        collapse_model_seeds,
        domain_equal_weights,
        prediction_metrics,
        stratified_bootstrap_interval,
        subset_budget_data,
        summarize_clusters,
        test_variant_indices,
        tune_classical_baselines,
        weighted_quantile,
    )
    from train_eval import collect_predictions, set_seed, train_model


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "own_algorithm_benchmark.json"
METHODS = ["ridge", "ridge_unet_aug", "ridge_fno_aug", "ridge_deeponet", "ray_set_operator"]
NEURAL_METHODS = METHODS[1:]
BASELINE_METHODS = ["ridge_unet_aug", "ridge_fno_aug", "ridge_deeponet"]
LABELS = {
    "ridge": "validation-locked ridge",
    "ridge_unet_aug": "ridge residual 3D U-Net",
    "ridge_fno_aug": "ridge residual FNO",
    "ridge_deeponet": "ridge residual DeepONet",
    "ray_set_operator": "provisional geometry-conditioned ray-set operator",
}


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


def make_benchmark_model(
    method: str,
    config: dict,
    data: dict[str, np.ndarray],
) -> torch.nn.Module:
    names = [str(value) for value in data["input_channel_names"].tolist()]
    view_start = int(data["ray_view_channel_start"])
    view_count = int(data["ray_view_channel_count"])
    angle_sin_start = int(data["ray_angle_sin_channel_start"])
    angle_cos_start = int(data["ray_angle_cos_channel_start"])
    mask_start = names.index("camera_0_active")
    coordinates = tuple(names.index(axis) for axis in ("z", "y", "x"))
    if method == "ridge_unet_aug":
        return make_model(
            "unet",
            config["models"]["unet"],
            int(data["inputs"].shape[1]),
            residual=True,
        )
    if method == "ridge_fno_aug":
        return make_model(
            "fno",
            config["models"]["fno"],
            int(data["inputs"].shape[1]),
            residual=True,
        )
    if method == "ridge_deeponet":
        spec = config["own_models"]["deeponet"]
        return GridDeepONetResidual(
            view_channel_start=view_start,
            view_count=view_count,
            mask_channel_start=mask_start,
            angle_sin_channel_start=angle_sin_start,
            angle_cos_channel_start=angle_cos_start,
            coordinate_channels=coordinates,
            branch_hidden=int(spec["branch_hidden"]),
            trunk_hidden=int(spec["trunk_hidden"]),
            rank=int(spec["rank"]),
            pool_shape=tuple(int(value) for value in spec["pool_shape"]),
        )
    if method == "ray_set_operator":
        spec = config["own_models"]["ray_set_operator"]
        return RaySetResidualOperator(
            view_count,
            int(data["inputs"].shape[1]),
            view_channel_start=view_start,
            mask_channel_start=mask_start,
            angle_sin_channel_start=angle_sin_start,
            angle_cos_channel_start=angle_cos_start,
            coordinate_channels=coordinates,
            view_features=int(spec["view_features"]),
            hidden_channels=int(spec["hidden_channels"]),
            n_modes=tuple(int(value) for value in spec["n_modes"]),
            n_layers=int(spec["n_layers"]),
        )
    raise ValueError(f"unknown benchmark method: {method}")


def validation_locked_baselines(training_rows: list[dict[str, object]]) -> dict[int, str]:
    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in training_rows:
        if str(row["method"]) in BASELINE_METHODS:
            grouped[(int(row["total_budget"]), str(row["method"]))].append(
                float(row["best_val_rel_l2"])
            )
    budgets = sorted({budget for budget, _ in grouped})
    return {
        budget: min(
            BASELINE_METHODS,
            key=lambda method: float(np.mean(grouped[(budget, method)])),
        )
        for budget in budgets
    }


def evaluate_seed(
    seed: int,
    data: dict[str, np.ndarray],
    trained: dict[tuple[int, str], dict],
    dataset_config: dict,
    selected_ridge: dict[int, float],
    audit_query_index: int,
) -> list[dict[str, object]]:
    indices = test_variant_indices(data)
    classical = classical_predictions(data, indices, selected_ridge)
    mapped_predictions: dict[str, dict[int, np.ndarray]] = {}
    inference_ms = {"ridge": 0.0}
    for method in NEURAL_METHODS:
        mapped: dict[int, np.ndarray] = {}
        elapsed_values = []
        for budget in sorted({int(value) for value in data["total_budget"][indices]}):
            budget_indices = indices[data["total_budget"][indices] == budget]
            record = trained[(budget, method)]
            values, elapsed = collect_predictions(
                record["model"],
                BOSTDataset(data, budget_indices),
                record["device"],
                int(dataset_config["training"]["batch_size"]),
            )
            mapped.update(
                {int(index): values[local_index, 0] for local_index, index in enumerate(budget_indices)}
            )
            elapsed_values.append(float(elapsed))
        mapped_predictions[method] = mapped
        inference_ms[method] = float(np.mean(elapsed_values))

    split_names = [str(value) for value in data["split_names"].tolist()]
    audit_mask = np.zeros(len(data["angles"]), dtype=np.float32)
    audit_mask[int(audit_query_index)] = 1.0
    rows: list[dict[str, object]] = []
    for local_index, variant_index in enumerate(indices):
        budget = int(data["total_budget"][variant_index])
        ridge_prediction = classical["ridge"][local_index]
        ridge_error = float(
            np.linalg.norm((ridge_prediction - data["field"][variant_index]).reshape(-1))
            / (np.linalg.norm(data["field"][variant_index].reshape(-1)) + 1e-8)
        )
        for method in METHODS:
            prediction = ridge_prediction if method == "ridge" else mapped_predictions[method][int(variant_index)]
            row = {
                "model_seed": seed,
                "variant_index": int(variant_index),
                "source_index": int(data["source_index"][variant_index]),
                "sample_seed": int(data["sample_seed"][variant_index]),
                "source_split": split_names[int(data["split_id"][variant_index])],
                "family_id": int(data["family_id"][variant_index]),
                "noise_level": float(data["noise_level"][variant_index]),
                "total_budget": budget,
                "method": method,
                "classical_champion": "ridge",
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
                    ridge_error,
                )
            )
            rows.append(row)
    return rows


def build_pairwise_rows(
    clusters: list[dict[str, object]],
    locked_baselines: dict[int, str],
) -> list[dict[str, object]]:
    lookup = {
        (int(row["source_index"]), int(row["total_budget"]), str(row["method"])): row
        for row in clusters
    }
    rows = []
    for (source_index, budget, method), own in sorted(lookup.items()):
        if method != "ray_set_operator":
            continue
        baseline_method = locked_baselines[budget]
        baseline = lookup[(source_index, budget, baseline_method)]
        baseline_field = float(baseline["field_rel_l2"])
        own_field = float(own["field_rel_l2"])
        baseline_audit = float(baseline["audit_reprojection_rel_l2"])
        own_audit = float(own["audit_reprojection_rel_l2"])
        rows.append(
            {
                "source_index": source_index,
                "sample_seed": int(own["sample_seed"]),
                "source_split": str(own["source_split"]),
                "total_budget": budget,
                "locked_neural_baseline": baseline_method,
                "model_seed_count": int(own["model_seed_count"]),
                "baseline_field_rel_l2": baseline_field,
                "own_field_rel_l2": own_field,
                "field_superiority_pct": 100.0 * (baseline_field - own_field) / (baseline_field + 1e-12),
                "baseline_audit_rel_l2": baseline_audit,
                "own_audit_rel_l2": own_audit,
                "audit_superiority_pct": 100.0 * (baseline_audit - own_audit) / (baseline_audit + 1e-12),
            }
        )
    return rows


def summarize_pairwise(
    rows: list[dict[str, object]],
    bootstrap_seed: int,
    bootstrap_replicates: int,
    parameter_counts: dict[tuple[int, str], int],
    locked_baselines: dict[int, str],
    experiment: dict,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(int(bootstrap_seed))
    output = []
    for budget in sorted({int(row["total_budget"]) for row in rows}):
        subset = [row for row in rows if int(row["total_budget"]) == budget]
        weights = domain_equal_weights(subset)
        gains = np.asarray([float(row["field_superiority_pct"]) for row in subset])
        ci_low, ci_high = stratified_bootstrap_interval(
            subset,
            "field_superiority_pct",
            rng,
            int(bootstrap_replicates),
        )
        baseline = locked_baselines[budget]
        parameter_ratio = parameter_counts[(budget, "ray_set_operator")] / parameter_counts[(budget, baseline)]
        mean_gain = float(np.sum(weights * gains))
        p10 = weighted_quantile(gains, weights, 0.1)
        harm = float(np.sum(weights * (gains < -1.0)))
        mean_audit = float(
            np.sum(weights * np.asarray([float(row["audit_superiority_pct"]) for row in subset]))
        )
        passed = (
            ci_low > float(experiment["minimum_superiority_ci_low_pct"])
            and p10 >= float(experiment["minimum_p10_superiority_pct"])
            and harm <= float(experiment["maximum_harm_rate_gt_1pct"])
            and parameter_ratio <= float(experiment["maximum_parameter_ratio"])
            and mean_audit >= -1.0
        )
        output.append(
            {
                "total_budget": budget,
                "locked_neural_baseline": baseline,
                "independent_field_count": len(subset),
                "mean_field_superiority_pct": mean_gain,
                "field_superiority_ci95_low": ci_low,
                "field_superiority_ci95_high": ci_high,
                "p10_field_superiority_pct": p10,
                "harm_rate_gt_1pct": harm,
                "mean_audit_superiority_pct": mean_audit,
                "parameter_ratio_vs_locked_baseline": parameter_ratio,
                "development_superiority_gate_pass": bool(passed),
            }
        )
    return output


def plot_method_errors(summary: list[dict[str, object]], path: Path) -> None:
    budgets = sorted({int(row["total_budget"]) for row in summary})
    lookup = {(int(row["total_budget"]), str(row["method"])): row for row in summary}
    x = np.arange(len(budgets))
    width = 0.2
    colors = ["#596b72", "#8f6c55", "#477fa4", "#c47d4d", "#0f6f66"]
    fig, axis = plt.subplots(figsize=(10.5, 5.1), constrained_layout=True)
    for offset, (method, color) in enumerate(zip(METHODS, colors)):
        values = [float(lookup[(budget, method)]["mean_field_rel_l2"]) for budget in budgets]
        axis.bar(x + (offset - 1.5) * width, values, width=width, color=color, label=LABELS[method])
    axis.set_xticks(x, [f"K={budget}" for budget in budgets])
    axis.set_ylabel("domain-equal field relative L2")
    axis.set_title("v3b matched-input algorithm benchmark")
    axis.grid(True, axis="y", alpha=0.22)
    axis.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_pairwise(summary: list[dict[str, object]], path: Path) -> None:
    budgets = [int(row["total_budget"]) for row in summary]
    means = np.asarray([float(row["mean_field_superiority_pct"]) for row in summary])
    lows = np.asarray([float(row["field_superiority_ci95_low"]) for row in summary])
    highs = np.asarray([float(row["field_superiority_ci95_high"]) for row in summary])
    fig, axis = plt.subplots(figsize=(8.8, 4.9), constrained_layout=True)
    axis.errorbar(
        np.arange(len(budgets)),
        means,
        yerr=np.vstack([means - lows, highs - means]),
        fmt="o",
        color="#0f6f66",
        ecolor="#477fa4",
        capsize=5,
        linewidth=2,
    )
    axis.axhline(0.0, color="#343b3d", linewidth=1)
    axis.axhline(2.0, color="#a95040", linewidth=1, linestyle="--", label="2% CI-lower gate")
    axis.set_xticks(np.arange(len(budgets)), [f"K={budget}" for budget in budgets])
    axis.set_ylabel("ray-set field gain over validation-locked neural baseline (%)")
    axis.set_title("Own-method superiority is tested against a locked baseline")
    axis.grid(True, axis="y", alpha=0.22)
    axis.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_checksums(output_dir: Path, filenames: list[str]) -> None:
    lines = []
    for filename in filenames:
        digest = hashlib.sha256((output_dir / filename).read_bytes()).hexdigest()
        lines.append(f"{digest}  {filename}")
    (output_dir / "own_algorithm_checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="ascii"
    )


def main() -> None:
    args = parse_args()
    experiment = read_json(args.config)
    dataset_config = read_json(args.config.parent / str(experiment["dataset_config"]))
    dataset_config["own_models"] = copy.deepcopy(experiment["models"])
    if args.device is not None:
        dataset_config["training"]["device"] = args.device
    if args.epochs is not None:
        dataset_config["training"]["epochs"] = int(args.epochs)
        dataset_config["training"]["early_stop_patience"] = min(
            int(dataset_config["training"]["early_stop_patience"]), int(args.epochs)
        )
    output_dir = args.output_dir or ROOT / "results" / str(experiment["output_dir"])
    work_dir = ROOT / "results" / str(experiment["work_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    base_path = ROOT / "results" / f"{dataset_config['name']}_dataset.npz"
    generate_dataset(dataset_config, base_path, force=args.force_data)
    base_data = load_npz(base_path)
    budgets = [int(value) for value in experiment["reconstruction_budgets"]]
    direct_data = prepare_direct_operator_data(
        base_data,
        budgets,
        int(experiment["fixed_query_index"]),
        int(experiment["audit_query_index"]),
    )
    selected_ridge, champions, tuning_rows = tune_classical_baselines(
        direct_data,
        budgets,
        [float(value) for value in experiment["ridge_relative_grid"]],
    )
    if any(champions[budget] != "ridge" for budget in budgets):
        raise RuntimeError("v3b expects validation-locked ridge to be the classical anchor")
    ridge_data = replace_lift_with_ridge(direct_data, selected_ridge)
    data = append_ray_view_channels(ridge_data)
    write_csv(output_dir / "own_algorithm_baseline_tuning.csv", tuning_rows)

    seeds = [int(value) for value in experiment["training_seeds"]]
    if args.seed_limit is not None:
        seeds = seeds[: int(args.seed_limit)]
    sample_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    for seed in seeds:
        run_config = copy.deepcopy(dataset_config)
        run_config["seed"] = seed
        trained = {}
        for budget in budgets:
            budget_data = subset_budget_data(data, budget)
            for method in NEURAL_METHODS:
                model_seed = seed + budget * 101 + NEURAL_METHODS.index(method) * 7
                set_seed(model_seed)
                budget_config = copy.deepcopy(run_config)
                budget_config["seed"] = model_seed
                model = make_benchmark_model(method, budget_config, budget_data)
                method_dir = work_dir / str(seed) / f"K{budget}" / method
                method_dir.mkdir(parents=True, exist_ok=True)
                record = train_model(method, model, budget_data, budget_config, method_dir)
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
        rows = evaluate_seed(
            seed,
            data,
            trained,
            dataset_config,
            selected_ridge,
            int(experiment["audit_query_index"]),
        )
        sample_rows.extend(rows)
        print(f"seed={seed}: evaluated {len(rows)} method rows", flush=True)

    locked_baselines = validation_locked_baselines(training_rows)
    cluster_rows = collapse_model_seeds(sample_rows)
    summary = summarize_clusters(
        cluster_rows,
        int(experiment["bootstrap_seed"]),
        int(experiment["bootstrap_replicates"]),
    )
    domains = build_domain_rows(cluster_rows)
    pairwise = build_pairwise_rows(cluster_rows, locked_baselines)
    parameter_counts = {
        (budget, method): int(round(np.mean([
            int(row["parameters"])
            for row in training_rows
            if int(row["total_budget"]) == budget and str(row["method"]) == method
        ])))
        for budget in budgets
        for method in NEURAL_METHODS
    }
    superiority = summarize_pairwise(
        pairwise,
        int(experiment["bootstrap_seed"]) + 97,
        int(experiment["bootstrap_replicates"]),
        parameter_counts,
        locked_baselines,
        experiment,
    )
    pass_count = sum(bool(row["development_superiority_gate_pass"]) for row in superiority)
    status = (
        "PROVISIONAL_MODEL_BEATS_LOCKED_NEURAL_BASELINE_3_OF_3"
        if pass_count == len(budgets)
        else f"PROVISIONAL_MODEL_PARTIAL_SUPERIORITY_{pass_count}_OF_{len(budgets)}"
    )

    write_csv(output_dir / "own_algorithm_training.csv", training_rows)
    write_csv(output_dir / "own_algorithm_samples.csv", sample_rows)
    write_csv(output_dir / "own_algorithm_clusters.csv", cluster_rows)
    write_csv(output_dir / "own_algorithm_summary.csv", summary)
    write_csv(output_dir / "own_algorithm_domains.csv", domains)
    write_csv(output_dir / "own_algorithm_pairwise.csv", pairwise)
    write_csv(output_dir / "own_algorithm_superiority.csv", superiority)
    plot_method_errors(summary, output_dir / "t16_own_algorithm_errors.png")
    plot_pairwise(superiority, output_dir / "t16_own_algorithm_superiority.png")

    dashboard = {
        "experiment": experiment["name"],
        "scientific_status": status,
        "development_fields_only": bool(experiment["development_fields_only"]),
        "algorithm_name_status": "provisional_working_label_not_novelty_claim",
        "methods": METHODS,
        "labels": LABELS,
        "independent_test_field_count": 96,
        "model_seed_count": len(seeds),
        "reconstruction_budgets": budgets,
        "audit_query_index": int(experiment["audit_query_index"]),
        "q_audit_used_for_training_or_selection": False,
        "locked_neural_baselines": {str(key): value for key, value in locked_baselines.items()},
        "parameter_counts": {f"K{key[0]}:{key[1]}": value for key, value in parameter_counts.items()},
        "summary": summary,
        "domains": domains,
        "superiority": superiority,
    }
    (output_dir / "own_algorithm_dashboard.json").write_text(
        json.dumps(dashboard, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    report = {
        "status": "completed_provisional_own_algorithm_development_benchmark",
        "scientific_status": status,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "requested_device": dataset_config["training"]["device"],
        },
        "protocol": {
            "all_neural_methods_receive_identical_ray_backprojection_channels": True,
            "training_and_evaluation_masks_match": True,
            "neural_comparator_locked_by_validation_only": True,
            "q_audit_used_for_training_or_selection": False,
            "test_unit": "three-dimensional field after collapsing model seeds",
            "domain_weighting": "four test domains receive equal weight",
        },
        "input_channels": [str(value) for value in data["input_channel_names"].tolist()],
        "ray_view_scales": [float(value) for value in data["ray_view_scales"]],
        "training": training_rows,
        "locked_neural_baselines": locked_baselines,
        "superiority": superiority,
        "claims_boundary": [
            "The ray-set architecture is a provisional working hypothesis, not a novelty claim or final algorithm name.",
            "This benchmark reuses inspected 8x16x16 linear synthetic development fields and cannot be cited as confirmatory superiority.",
            "U-Net, DeepONet and FNO receive the same train-normalized per-view ray backprojections and ridge anchor as the proposed model.",
            "A positive result only justifies a new locked-field, variable-geometry, independent-forward benchmark plus NeRIF refinement.",
            "GRU-BOST, CGLS/TV/RBF, compact INR and real repeated acquisitions remain required publication baselines.",
        ],
    }
    (output_dir / "own_algorithm_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    files = [
        "own_algorithm_baseline_tuning.csv",
        "own_algorithm_training.csv",
        "own_algorithm_samples.csv",
        "own_algorithm_clusters.csv",
        "own_algorithm_summary.csv",
        "own_algorithm_domains.csv",
        "own_algorithm_pairwise.csv",
        "own_algorithm_superiority.csv",
        "own_algorithm_dashboard.json",
        "own_algorithm_report.json",
    ]
    write_checksums(output_dir, files)
    print(json.dumps({"status": status, "superiority": superiority}, indent=2, allow_nan=False))
    print(f"results: {output_dir}")


if __name__ == "__main__":
    main()
