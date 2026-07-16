#!/usr/bin/env python3
"""Build a validation-locked DeepONet/FNO error-compute frontier for T16."""

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
    from .data import BOSTDataset, generate_dataset, load_npz, split_indices
    from .direct_operator_data import prepare_direct_operator_data, replace_lift_with_ridge
    from .own_algorithm_data import append_ray_view_channels
    from .run_direct_operator_pilot import (
        domain_equal_weights,
        prediction_metrics,
        relative_field_error,
        stratified_bootstrap_interval,
        test_variant_indices,
        tune_classical_baselines,
        weighted_quantile,
    )
    from .run_own_algorithm_benchmark import make_benchmark_model
    from .run_v3c_k6_dev2_pilot import state_sha256, training_config
    from .run_v3d_fno_optimizer_audit import (
        batch_order_contract_sha256,
        build_pairwise_summary,
        build_strategy_summary,
        choose_validation_champion,
        collapse_test_rows,
        strategy_ids,
        summarize_seed_rows,
        summarize_strategies,
        summarize_test_rows,
        train_continuation_block,
    )
    from .run_v3d_fno_saturation_audit import cpu_state, read_csv, read_json, write_csv
    from .train_eval import choose_device, collect_predictions, set_seed, train_model
except ImportError:
    from data import BOSTDataset, generate_dataset, load_npz, split_indices
    from direct_operator_data import prepare_direct_operator_data, replace_lift_with_ridge
    from own_algorithm_data import append_ray_view_channels
    from run_direct_operator_pilot import (
        domain_equal_weights,
        prediction_metrics,
        relative_field_error,
        stratified_bootstrap_interval,
        test_variant_indices,
        tune_classical_baselines,
        weighted_quantile,
    )
    from run_own_algorithm_benchmark import make_benchmark_model
    from run_v3c_k6_dev2_pilot import state_sha256, training_config
    from run_v3d_fno_optimizer_audit import (
        batch_order_contract_sha256,
        build_pairwise_summary,
        build_strategy_summary,
        choose_validation_champion,
        collapse_test_rows,
        strategy_ids,
        summarize_seed_rows,
        summarize_strategies,
        summarize_test_rows,
        train_continuation_block,
    )
    from run_v3d_fno_saturation_audit import cpu_state, read_csv, read_json, write_csv
    from train_eval import choose_device, collect_predictions, set_seed, train_model


ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = ROOT / "configs"
DEFAULT_CONFIG = CONFIG_ROOT / "v3f_deeponet_frontier.json"
FNO_RESULTS = ROOT / "results" / "v3d_fno_optimizer_audit"
COMPUTE_RESULTS = ROOT / "results" / "v3e_compute_accounting"
CHECKSUM_FILES = [
    "v3f_deeponet_baseline_tuning.csv",
    "v3f_deeponet_lr_tuning.csv",
    "v3f_deeponet_history.csv",
    "v3f_deeponet_checkpoints.csv",
    "v3f_deeponet_validation_summary.csv",
    "v3f_deeponet_strategy_summary.csv",
    "v3f_deeponet_samples.csv",
    "v3f_deeponet_clusters.csv",
    "v3f_deeponet_test_summary.csv",
    "v3f_deeponet_pairwise_summary.csv",
    "v3f_deeponet_seed_summary.csv",
    "v3f_architecture_frontier.csv",
    "v3f_time_to_target.csv",
    "v3f_cross_architecture_pairwise.csv",
    "v3f_cross_architecture_domains.csv",
    "v3f_cross_architecture_seeds.csv",
    "v3f_selection_commit.json",
    "v3f_deeponet_frontier_dashboard.json",
    "v3f_deeponet_frontier_report.json",
    "t16_v3f_deeponet_fno_frontier.png",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--force-data", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def configured_model_data(
    dataset_config: dict, own_algorithm_config: dict
) -> dict:
    config = copy.deepcopy(dataset_config)
    config["own_models"] = copy.deepcopy(own_algorithm_config["models"])
    return config


def make_deeponet(config: dict, data: dict[str, np.ndarray]) -> torch.nn.Module:
    return make_benchmark_model("ridge_deeponet", config, data)


def select_global_learning_rate(
    rows: list[dict[str, object]], expected_seed_count: int
) -> tuple[float, list[dict[str, object]]]:
    grouped: dict[float, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[float(row["learning_rate"])].append(row)
    if any(len(values) != expected_seed_count for values in grouped.values()):
        raise ValueError("every learning rate must use the complete seed set")
    means = {
        learning_rate: float(
            np.mean([float(row["best_validation_rel_l2"]) for row in values])
        )
        for learning_rate, values in grouped.items()
    }
    champion = min(means, key=lambda value: (means[value], value))
    for row in rows:
        row["mean_validation_rel_l2_for_learning_rate"] = means[
            float(row["learning_rate"])
        ]
        row["global_validation_champion"] = (
            float(row["learning_rate"]) == champion
        )
    return float(champion), rows


def evaluate_deeponet_checkpoint(
    outer_seed: int,
    cumulative_epochs: int,
    state: dict[str, torch.Tensor],
    data: dict[str, np.ndarray],
    model_config: dict,
    ridge_relative: float,
    audit_query_index: int,
    device: torch.device,
) -> list[dict[str, object]]:
    model = make_deeponet(model_config, data)
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    indices = test_variant_indices(data)
    predictions, inference_ms = collect_predictions(
        model,
        BOSTDataset(data, indices),
        device,
        int(model_config["training"]["batch_size"]),
    )
    split_names = [str(value) for value in data["split_names"].tolist()]
    audit_mask = np.zeros(len(data["angles"]), dtype=np.float32)
    audit_mask[int(audit_query_index)] = 1.0
    rows = []
    for local_index, index in enumerate(indices):
        ridge_error = relative_field_error(data["lift"][index], data["field"][index])
        row = {
            "model_seed": int(outer_seed),
            "cumulative_epochs": int(cumulative_epochs),
            "variant_index": int(index),
            "source_index": int(data["source_index"][index]),
            "sample_seed": int(data["sample_seed"][index]),
            "source_split": split_names[int(data["split_id"][index])],
            "family_id": int(data["family_id"][index]),
            "noise_level": float(data["noise_level"][index]),
            "total_budget": int(data["total_budget"][index]),
            "ridge_relative": float(ridge_relative),
            "inference_ms_per_sample": float(inference_ms),
        }
        row.update(
            prediction_metrics(
                predictions[local_index, 0],
                data["field"][index],
                data["clean_observation"][index],
                data["view_mask"][index],
                audit_mask,
                data["forward_matrix"],
                ridge_error,
            )
        )
        rows.append(row)
    del model
    return rows


def cost_profile_lookup() -> dict[str, dict[str, object]]:
    return {
        str(row["method"]): row
        for row in read_csv(COMPUTE_RESULTS / "v3e_compute_profiles.csv")
    }


def build_frontier_rows(
    deep_validation: list[dict[str, object]],
    deep_champion: str,
    fno_validation: list[dict[str, object]],
    fno_champion: str,
    experiment: dict,
) -> list[dict[str, object]]:
    profiles = cost_profile_lookup()
    fixed = {int(value) for value in experiment["fixed_epoch_checkpoints"]}
    sources = [
        (
            "deeponet",
            "ridge_deeponet",
            deep_champion,
            deep_validation,
            "v3f validation-only DeepONet",
        ),
        (
            "fno",
            "ridge_fno_aug",
            fno_champion,
            fno_validation,
            "v3d validation-only FNO",
        ),
    ]
    output = []
    for architecture, method, strategy, rows, source in sources:
        profile = profiles[method]
        for row in rows:
            if str(row["strategy"]) != strategy:
                continue
            epoch = int(row["cumulative_epochs"])
            if epoch not in fixed:
                continue
            output.append(
                {
                    "architecture": architecture,
                    "method": method,
                    "strategy": strategy,
                    "cumulative_epochs": epoch,
                    "mean_validation_rel_l2": float(row["mean_validation_rel_l2"]),
                    "mean_cumulative_train_seconds": float(
                        row["mean_cumulative_train_seconds"]
                    ),
                    "mean_selected_checkpoint_epoch": float(
                        row["mean_selected_checkpoint_epoch"]
                    ),
                    "total_parameters": int(profile["total_parameters"]),
                    "forward_estimated_flops_v1": float(
                        profile["forward_estimated_flops_v1"]
                    ),
                    "inference_p50_ms": float(profile["inference_p50_ms"]),
                    "training_step_p50_ms": float(
                        profile["training_step_p50_ms"]
                    ),
                    "source": source,
                }
            )
    return sorted(output, key=lambda row: (str(row["architecture"]), int(row["cumulative_epochs"])))


def build_time_to_target_rows(
    frontier_source_rows: dict[str, list[dict[str, object]]],
    strategy_by_architecture: dict[str, str],
    experiment: dict,
) -> list[dict[str, object]]:
    output = []
    for architecture, rows in frontier_source_rows.items():
        strategy = strategy_by_architecture[architecture]
        subset = sorted(
            [row for row in rows if str(row["strategy"]) == strategy],
            key=lambda row: int(row["cumulative_epochs"]),
        )
        for target in experiment["time_to_target_validation_rel_l2"]:
            crossing = next(
                (
                    row
                    for row in subset
                    if float(row["mean_validation_rel_l2"]) <= float(target)
                ),
                None,
            )
            output.append(
                {
                    "architecture": architecture,
                    "strategy": strategy,
                    "target_validation_rel_l2": float(target),
                    "target_reached": crossing is not None,
                    "first_endpoint_epoch": (
                        None if crossing is None else int(crossing["cumulative_epochs"])
                    ),
                    "mean_cumulative_train_seconds": (
                        None
                        if crossing is None
                        else float(crossing["mean_cumulative_train_seconds"])
                    ),
                    "selection_scope": "validation_only",
                }
            )
    return output


def selected_architecture(
    deep_validation: list[dict[str, object]],
    deep_champion: str,
    fno_validation: list[dict[str, object]],
    fno_champion: str,
    max_epochs: int,
) -> tuple[str, dict[str, float]]:
    final = {}
    for architecture, strategy, rows in (
        ("deeponet", deep_champion, deep_validation),
        ("fno", fno_champion, fno_validation),
    ):
        row = next(
            value
            for value in rows
            if str(value["strategy"]) == strategy
            and int(value["cumulative_epochs"]) == int(max_epochs)
        )
        final[architecture] = float(row["mean_validation_rel_l2"])
    return min(final, key=lambda name: (final[name], name)), final


def build_dominance_diagnostic(
    deep_validation: list[dict[str, object]],
    deep_champion: str,
    fno_validation: list[dict[str, object]],
    fno_champion: str,
    experiment: dict,
) -> dict[str, object]:
    max_epochs = int(experiment["max_total_epochs"])
    fixed = {int(value) for value in experiment["fixed_epoch_checkpoints"]}
    deep_rows = sorted(
        [row for row in deep_validation if str(row["strategy"]) == deep_champion],
        key=lambda row: int(row["cumulative_epochs"]),
    )
    fno_rows = sorted(
        [row for row in fno_validation if str(row["strategy"]) == fno_champion],
        key=lambda row: int(row["cumulative_epochs"]),
    )
    deep_final = next(
        row for row in deep_rows if int(row["cumulative_epochs"]) == max_epochs
    )
    fno_final = next(
        row for row in fno_rows if int(row["cumulative_epochs"]) == max_epochs
    )
    first_fno_better_than_deep_final = next(
        row
        for row in fno_rows
        if float(row["mean_validation_rel_l2"])
        <= float(deep_final["mean_validation_rel_l2"])
    )
    dominated = []
    for deep_row in deep_rows:
        if int(deep_row["cumulative_epochs"]) not in fixed:
            continue
        witness = next(
            (
                row
                for row in fno_rows
                if float(row["mean_cumulative_train_seconds"])
                <= float(deep_row["mean_cumulative_train_seconds"])
                and float(row["mean_validation_rel_l2"])
                <= float(deep_row["mean_validation_rel_l2"])
            ),
            None,
        )
        dominated.append(witness is not None)
    profiles = cost_profile_lookup()
    deep_cost = profiles["ridge_deeponet"]
    fno_cost = profiles["ridge_fno_aug"]
    return {
        "deeponet_final_validation_rel_l2": float(
            deep_final["mean_validation_rel_l2"]
        ),
        "fno_final_validation_rel_l2": float(fno_final["mean_validation_rel_l2"]),
        "fno_final_validation_advantage_vs_deeponet_pct": 100.0
        * (
            float(deep_final["mean_validation_rel_l2"])
            - float(fno_final["mean_validation_rel_l2"])
        )
        / float(deep_final["mean_validation_rel_l2"]),
        "first_fno_endpoint_better_than_deeponet_final_epoch": int(
            first_fno_better_than_deep_final["cumulative_epochs"]
        ),
        "first_fno_endpoint_better_than_deeponet_final_seconds": float(
            first_fno_better_than_deep_final["mean_cumulative_train_seconds"]
        ),
        "deeponet_final_seconds": float(deep_final["mean_cumulative_train_seconds"]),
        "deeponet_final_to_fno_crossing_time_ratio": float(
            deep_final["mean_cumulative_train_seconds"]
        )
        / float(first_fno_better_than_deep_final["mean_cumulative_train_seconds"]),
        "observed_deeponet_fixed_checkpoint_count": len(dominated),
        "observed_pareto_dominated_deeponet_fixed_checkpoint_count": sum(dominated),
        "deeponet_forward_flops_v1_to_fno_ratio": float(
            deep_cost["forward_estimated_flops_v1"]
        )
        / float(fno_cost["forward_estimated_flops_v1"]),
        "deeponet_inference_p50_to_fno_ratio": float(deep_cost["inference_p50_ms"])
        / float(fno_cost["inference_p50_ms"]),
        "deeponet_training_step_p50_to_fno_ratio": float(
            deep_cost["training_step_p50_ms"]
        )
        / float(fno_cost["training_step_p50_ms"]),
        "scope": "observed synthetic development checkpoints only",
    }


def build_cross_architecture_rows(
    deep_clusters: list[dict[str, object]],
    deep_samples: list[dict[str, object]],
    deep_champion: str,
    fno_clusters: list[dict[str, object]],
    fno_samples: list[dict[str, object]],
    fno_champion: str,
    validation_winner: str,
    experiment: dict,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    cluster_sources = {
        "deeponet": [
            row for row in deep_clusters if str(row["strategy"]) == deep_champion
        ],
        "fno": [row for row in fno_clusters if str(row["strategy"]) == fno_champion],
    }
    sample_sources = {
        "deeponet": [
            row for row in deep_samples if str(row["strategy"]) == deep_champion
        ],
        "fno": [row for row in fno_samples if str(row["strategy"]) == fno_champion],
    }
    comparator = "fno" if validation_winner == "deeponet" else "deeponet"
    candidate_lookup = {
        int(row["source_index"]): row for row in cluster_sources[validation_winner]
    }
    comparator_lookup = {
        int(row["source_index"]): row for row in cluster_sources[comparator]
    }
    if set(candidate_lookup) != set(comparator_lookup):
        raise ValueError("cross-architecture dev2 fields do not align")
    augmented = []
    for source_index, candidate in sorted(candidate_lookup.items()):
        reference = comparator_lookup[source_index]
        augmented.append(
            {
                **candidate,
                "field_superiority_pct": 100.0
                * (float(reference["field_rel_l2"]) - float(candidate["field_rel_l2"]))
                / (float(reference["field_rel_l2"]) + 1e-12),
                "audit_superiority_pct": 100.0
                * (
                    float(reference["audit_reprojection_rel_l2"])
                    - float(candidate["audit_reprojection_rel_l2"])
                )
                / (float(reference["audit_reprojection_rel_l2"]) + 1e-12),
            }
        )
    weights = domain_equal_weights(augmented)
    field = np.asarray(
        [float(row["field_superiority_pct"]) for row in augmented], dtype=np.float64
    )
    audit = np.asarray(
        [float(row["audit_superiority_pct"]) for row in augmented], dtype=np.float64
    )
    rng = np.random.default_rng(int(experiment["bootstrap_seed"]))
    ci_low, ci_high = stratified_bootstrap_interval(
        augmented,
        "field_superiority_pct",
        rng,
        int(experiment["bootstrap_replicates"]),
    )
    domain_rows = []
    for domain in sorted({str(row["source_split"]) for row in augmented}):
        subset = [row for row in augmented if str(row["source_split"]) == domain]
        values = np.asarray(
            [float(row["field_superiority_pct"]) for row in subset], dtype=np.float64
        )
        domain_rows.append(
            {
                "candidate": validation_winner,
                "comparator": comparator,
                "source_split": domain,
                "independent_field_count": len(subset),
                "mean_field_superiority_pct": float(np.mean(values)),
                "p10_field_superiority_pct": float(np.quantile(values, 0.10)),
                "field_harm_rate_gt_1pct": float(np.mean(values < -1.0)),
            }
        )

    candidate_samples = {
        (int(row["model_seed"]), int(row["source_index"])): row
        for row in sample_sources[validation_winner]
    }
    comparator_samples = {
        (int(row["model_seed"]), int(row["source_index"])): row
        for row in sample_sources[comparator]
    }
    if set(candidate_samples) != set(comparator_samples):
        raise ValueError("cross-architecture seed-level fields do not align")
    seed_rows = []
    for seed in sorted({key[0] for key in candidate_samples}):
        values = []
        for key, candidate in candidate_samples.items():
            if key[0] != seed:
                continue
            reference = comparator_samples[key]
            values.append(
                {
                    **candidate,
                    "field_superiority_pct": 100.0
                    * (
                        float(reference["field_rel_l2"])
                        - float(candidate["field_rel_l2"])
                    )
                    / (float(reference["field_rel_l2"]) + 1e-12),
                }
            )
        seed_weights = domain_equal_weights(values)
        seed_rows.append(
            {
                "candidate": validation_winner,
                "comparator": comparator,
                "model_seed": seed,
                "independent_field_count": len(values),
                "domain_equal_mean_field_superiority_pct": float(
                    np.sum(
                        seed_weights
                        * np.asarray(
                            [float(row["field_superiority_pct"]) for row in values]
                        )
                    )
                ),
            }
        )

    gate = experiment["development_gate"]
    every_domain = all(
        float(row["mean_field_superiority_pct"]) >= 0.0 for row in domain_rows
    )
    every_seed = all(
        float(row["domain_equal_mean_field_superiority_pct"]) > 0.0
        for row in seed_rows
    )
    p10 = weighted_quantile(field, weights, 0.10)
    harm = float(np.sum(weights * (field < -1.0)))
    gate_pass = bool(
        ci_low > float(gate["minimum_field_superiority_ci95_low_pct"])
        and p10 >= float(gate["minimum_p10_field_superiority_pct"])
        and harm <= float(gate["maximum_field_harm_rate_gt_1pct"])
        and (not bool(gate["require_every_domain_mean_nonnegative"]) or every_domain)
        and (not bool(gate["require_every_seed_mean_positive"]) or every_seed)
    )
    pairwise = [
        {
            "selection_basis": "lowest mean validation relative L2 at fixed epoch 240",
            "candidate": validation_winner,
            "comparator": comparator,
            "independent_field_count": len(augmented),
            "model_seed_count": int(augmented[0]["model_seed_count"]),
            "mean_field_superiority_pct": float(np.sum(weights * field)),
            "field_superiority_ci95_low": ci_low,
            "field_superiority_ci95_high": ci_high,
            "p10_field_superiority_pct": p10,
            "field_harm_rate_gt_1pct": harm,
            "mean_audit_superiority_pct": float(np.sum(weights * audit)),
            "every_domain_mean_field_nonnegative": every_domain,
            "every_seed_mean_field_positive": every_seed,
            "development_gate_pass": gate_pass,
            "confirmatory_superiority_eligible": False,
        }
    ]
    return pairwise, domain_rows, seed_rows


def plot_results(
    deep_validation: list[dict[str, object]],
    deep_champion: str,
    fno_validation: list[dict[str, object]],
    fno_champion: str,
    frontier_rows: list[dict[str, object]],
    domain_rows: list[dict[str, object]],
    experiment: dict,
    output_path: Path,
) -> None:
    colors = {
        "restart_adam_restart_cosine": "#b15a4a",
        "carry_adam_restart_cosine": "#c6922d",
        "carry_adam_long_cosine": "#497d77",
    }
    labels = {str(row["id"]): str(row["label"]) for row in experiment["strategies"]}
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.8), constrained_layout=True)
    for strategy in strategy_ids(experiment):
        subset = sorted(
            [row for row in deep_validation if str(row["strategy"]) == strategy],
            key=lambda row: int(row["cumulative_epochs"]),
        )
        axes[0].plot(
            [int(row["cumulative_epochs"]) for row in subset],
            [float(row["mean_validation_rel_l2"]) for row in subset],
            color=colors[strategy],
            alpha=1.0 if strategy == deep_champion else 0.42,
            linewidth=2.2 if strategy == deep_champion else 1.0,
            label=f"DeepONet: {labels[strategy]}",
        )
    fno_curve = sorted(
        [row for row in fno_validation if str(row["strategy"]) == fno_champion],
        key=lambda row: int(row["cumulative_epochs"]),
    )
    axes[0].plot(
        [int(row["cumulative_epochs"]) for row in fno_curve],
        [float(row["mean_validation_rel_l2"]) for row in fno_curve],
        color="#315f7b",
        linewidth=2.4,
        linestyle="--",
        label="FNO validation champion",
    )
    axes[0].set_title("Validation-only learning curves")
    axes[0].set_xlabel("attempted epochs")
    axes[0].set_ylabel("prefix-best validation relative L2")
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=6.8)

    for architecture, color, marker in (
        ("deeponet", "#a65f4a", "o"),
        ("fno", "#315f7b", "s"),
    ):
        subset = sorted(
            [row for row in frontier_rows if str(row["architecture"]) == architecture],
            key=lambda row: int(row["cumulative_epochs"]),
        )
        axes[1].plot(
            [float(row["mean_cumulative_train_seconds"]) for row in subset],
            [float(row["mean_validation_rel_l2"]) for row in subset],
            marker=marker,
            color=color,
            linewidth=2.0,
            label=architecture,
        )
        for row in subset:
            axes[1].annotate(
                str(int(row["cumulative_epochs"])),
                (
                    float(row["mean_cumulative_train_seconds"]),
                    float(row["mean_validation_rel_l2"]),
                ),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7,
            )
    axes[1].set_title("Matched checkpoints, measured wall time")
    axes[1].set_xlabel("mean cumulative training seconds / seed")
    axes[1].set_ylabel("validation relative L2")
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=8)

    names = [str(row["source_split"]).replace("test_", "") for row in domain_rows]
    values = [float(row["mean_field_superiority_pct"]) for row in domain_rows]
    axes[2].bar(
        np.arange(len(names)),
        values,
        color=["#39827a" if value >= 0.0 else "#b24f46" for value in values],
    )
    axes[2].axhline(0.0, color="#273638", linewidth=1.0)
    axes[2].set_xticks(np.arange(len(names)), names, rotation=25, ha="right")
    axes[2].set_title("Post-selection dev2 domain diagnostic")
    axes[2].set_ylabel("validation winner superiority (%)")
    axes[2].grid(axis="y", alpha=0.2)
    fig.suptitle("T16 v3f DeepONet/FNO matched development frontier", fontweight="bold")
    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)


def build_provenance(
    config_path: Path, experiment: dict, dataset_path: Path
) -> dict[str, object]:
    return {
        "experiment_config_sha256": sha256_file(config_path),
        "dataset_config_sha256": sha256_file(
            CONFIG_ROOT / str(experiment["dataset_config"])
        ),
        "own_algorithm_config_sha256": sha256_file(
            CONFIG_ROOT / str(experiment["own_algorithm_config"])
        ),
        "training_script_sha256": sha256_file(Path(__file__).resolve()),
        "train_eval_script_sha256": sha256_file(ROOT / "train_eval.py"),
        "optimizer_audit_script_sha256": sha256_file(
            ROOT / "run_v3d_fno_optimizer_audit.py"
        ),
        "data_script_sha256": sha256_file(ROOT / "data.py"),
        "dataset_npz_sha256": sha256_file(dataset_path),
        "source_fno_dashboard_sha256": sha256_file(
            FNO_RESULTS / "v3d_optimizer_dashboard.json"
        ),
        "source_fno_validation_sha256": sha256_file(
            FNO_RESULTS / "v3d_optimizer_validation_summary.csv"
        ),
        "source_fno_clusters_sha256": sha256_file(
            FNO_RESULTS / "v3d_optimizer_clusters.csv"
        ),
        "source_compute_profiles_sha256": sha256_file(
            COMPUTE_RESULTS / "v3e_compute_profiles.csv"
        ),
        "dataset_npz_public": False,
        "checkpoint_weights_public": False,
    }


def write_output_checksums(output_dir: Path) -> None:
    lines = []
    for filename in CHECKSUM_FILES:
        digest = hashlib.sha256((output_dir / filename).read_bytes()).hexdigest()
        lines.append(f"{digest}  {filename}")
    (output_dir / "v3f_deeponet_frontier_checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="ascii"
    )


def analyze_and_write(
    experiment: dict,
    output_dir: Path,
    lr_rows: list[dict[str, object]],
    selected_lr: float,
    lr_screen_seconds: float,
    discarded_lr_seconds: float,
    checkpoint_rows: list[dict[str, object]],
    history_rows: list[dict[str, object]],
    sample_rows: list[dict[str, object]],
    environment: dict[str, object],
    provenance: dict[str, object],
) -> None:
    validation_rows, decisions = summarize_strategies(checkpoint_rows, experiment)
    deep_champion = choose_validation_champion(
        validation_rows, int(experiment["max_total_epochs"])
    )
    strategy_summary = build_strategy_summary(
        validation_rows, decisions, deep_champion, experiment
    )
    cluster_rows = collapse_test_rows(sample_rows)
    test_summary = summarize_test_rows(cluster_rows, experiment)
    pairwise_summary = build_pairwise_summary(
        cluster_rows, deep_champion, experiment
    )
    seed_summary = summarize_seed_rows(sample_rows)
    selection_commit = read_json(output_dir / "v3f_selection_commit.json")
    selection_commit_sha256 = str(selection_commit["selection_commit_sha256"])

    fno_dashboard = read_json(FNO_RESULTS / "v3d_optimizer_dashboard.json")
    fno_champion = str(fno_dashboard["validation_champion"])
    fno_validation = read_csv(FNO_RESULTS / "v3d_optimizer_validation_summary.csv")
    fno_clusters = read_csv(FNO_RESULTS / "v3d_optimizer_clusters.csv")
    fno_samples = read_csv(FNO_RESULTS / "v3d_optimizer_samples.csv")
    frontier_rows = build_frontier_rows(
        validation_rows,
        deep_champion,
        fno_validation,
        fno_champion,
        experiment,
    )
    time_rows = build_time_to_target_rows(
        {"deeponet": validation_rows, "fno": fno_validation},
        {"deeponet": deep_champion, "fno": fno_champion},
        experiment,
    )
    validation_winner, final_validation = selected_architecture(
        validation_rows,
        deep_champion,
        fno_validation,
        fno_champion,
        int(experiment["max_total_epochs"]),
    )
    dominance = build_dominance_diagnostic(
        validation_rows,
        deep_champion,
        fno_validation,
        fno_champion,
        experiment,
    )
    cross_pairwise, cross_domains, cross_seeds = build_cross_architecture_rows(
        cluster_rows,
        sample_rows,
        deep_champion,
        fno_clusters,
        fno_samples,
        fno_champion,
        validation_winner,
        experiment,
    )
    plot_results(
        validation_rows,
        deep_champion,
        fno_validation,
        fno_champion,
        frontier_rows,
        cross_domains,
        experiment,
        output_dir / "t16_v3f_deeponet_fno_frontier.png",
    )

    write_csv(output_dir / "v3f_deeponet_lr_tuning.csv", lr_rows)
    write_csv(output_dir / "v3f_deeponet_history.csv", history_rows)
    write_csv(output_dir / "v3f_deeponet_checkpoints.csv", checkpoint_rows)
    write_csv(output_dir / "v3f_deeponet_validation_summary.csv", validation_rows)
    write_csv(output_dir / "v3f_deeponet_strategy_summary.csv", strategy_summary)
    write_csv(output_dir / "v3f_deeponet_samples.csv", sample_rows)
    write_csv(output_dir / "v3f_deeponet_clusters.csv", cluster_rows)
    write_csv(output_dir / "v3f_deeponet_test_summary.csv", test_summary)
    write_csv(output_dir / "v3f_deeponet_pairwise_summary.csv", pairwise_summary)
    write_csv(output_dir / "v3f_deeponet_seed_summary.csv", seed_summary)
    write_csv(output_dir / "v3f_architecture_frontier.csv", frontier_rows)
    write_csv(output_dir / "v3f_time_to_target.csv", time_rows)
    write_csv(output_dir / "v3f_cross_architecture_pairwise.csv", cross_pairwise)
    write_csv(output_dir / "v3f_cross_architecture_domains.csv", cross_domains)
    write_csv(output_dir / "v3f_cross_architecture_seeds.csv", cross_seeds)

    status = "MATCHED_DEVELOPMENT_FRONTIER_COMPLETE_CONFIRMATORY_SUPERIORITY_LOCKED"
    dashboard = {
        "experiment": experiment["name"],
        "scientific_status": status,
        "development_fields_only": True,
        "blind_final_opened": False,
        "confirmatory_superiority_eligible": False,
        "matched_architectures": ["deeponet", "fno"],
        "fixed_epoch_checkpoints": experiment["fixed_epoch_checkpoints"],
        "model_seed_count": len(experiment["training_seeds"]),
        "independent_test_field_count": len(cluster_rows) // 4,
        "selected_base_learning_rate": selected_lr,
        "learning_rate_screen_total_seconds": lr_screen_seconds,
        "discarded_learning_rate_search_seconds": discarded_lr_seconds,
        "selection_commit_sha256": selection_commit_sha256,
        "deeponet_validation_champion": deep_champion,
        "fno_validation_champion": fno_champion,
        "validation_champion_architecture": validation_winner,
        "final_validation_rel_l2": final_validation,
        "dominance_diagnostic": dominance,
        "deeponet_champion_plateau_decision": decisions[deep_champion],
        "cross_architecture_pairwise": cross_pairwise[0],
        "cross_architecture_domains": cross_domains,
        "cross_architecture_seeds": cross_seeds,
        "development_gate": experiment["development_gate"],
        "frontier": frontier_rows,
        "time_to_target": time_rows,
        "provenance": provenance,
    }
    (output_dir / "v3f_deeponet_frontier_dashboard.json").write_text(
        json.dumps(dashboard, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    report = {
        "status": "completed",
        "scientific_status": status,
        "environment": environment,
        "provenance": provenance,
        "protocol": {
            "global_deeponet_learning_rate_selected_by_mean_validation_only": True,
            "discarded_learning_rate_search_cost_reported_separately": True,
            "same_dataset_budget_loss_seeds_and_batch_order_as_fno": True,
            "matched_attempted_epochs_not_equal_flops_or_equal_wall_time": True,
            "same_base_checkpoint_per_seed_across_deeponet_strategies": True,
            "same_block_batch_seed_across_strategies_and_architectures": True,
            "batch_order_contract_binds_actual_train_indices": True,
            "validation_metric_aggregated_per_sample": True,
            "base_optimizer_state_carried_into_continuation": False,
            "training_state_continues_from_raw_block_endpoint": True,
            "reported_checkpoint_is_validation_prefix_best": True,
            "architecture_selected_by_final_validation_only": True,
            "dev2_computed_after_all_deeponet_validation_decisions": True,
            "dev2_cannot_change_validation_winner": True,
            "blind_final_opened": False,
        },
        "selected_base_learning_rate": selected_lr,
        "learning_rate_screen_total_seconds": lr_screen_seconds,
        "discarded_learning_rate_search_seconds": discarded_lr_seconds,
        "selection_commit_sha256": selection_commit_sha256,
        "deeponet_validation_champion": deep_champion,
        "fno_validation_champion": fno_champion,
        "validation_champion_architecture": validation_winner,
        "final_validation_rel_l2": final_validation,
        "dominance_diagnostic": dominance,
        "cross_architecture_pairwise": cross_pairwise[0],
        "claims_boundary": [
            "This is a synthetic 8x16x16, K=6 development comparison, not real BOST evidence.",
            "Matched checkpoints mean the same attempted epochs, data, losses and batch-order contracts; architectures do not have equal FLOPs or equal wall time.",
            "DeepONet receives a declared validation-only learning-rate screen; discarded tuning cost is reported but excluded from its selected-run curve.",
            "The architecture winner is frozen by final mean validation error before dev2 and Q_audit are read.",
            "The v3c dev2 fields are reused development diagnostics already inspected in earlier project stages; they are not a fresh project-level audit set.",
            "The cluster-bootstrap interval resamples 128 fields after collapsing model seeds; three seed means are a separate directional diagnostic, not seed-level confidence inference.",
            "A development gate pass cannot open the blind final or support confirmatory superiority.",
            "The partial FLOPs-v1 values are inherited from v3e and exclude normalization, activation, pooling, indexing and most elementwise work.",
            "No acquisition-geometry mechanism, real-data transfer, NeRIF advantage or publication novelty follows from this comparison.",
        ],
    }
    (output_dir / "v3f_deeponet_frontier_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_output_checksums(output_dir)
    print(
        json.dumps(
            {
                "scientific_status": status,
                "selected_base_learning_rate": selected_lr,
                "deeponet_validation_champion": deep_champion,
                "validation_champion_architecture": validation_winner,
                "final_validation_rel_l2": final_validation,
                "development_gate_pass": cross_pairwise[0]["development_gate_pass"],
            },
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    experiment = read_json(args.config)
    dataset_config = read_json(CONFIG_ROOT / str(experiment["dataset_config"]))
    own_config = read_json(CONFIG_ROOT / str(experiment["own_algorithm_config"]))
    model_config = configured_model_data(dataset_config, own_config)
    device_name = args.device or str(dataset_config["training"]["device"])
    device = choose_device(device_name)
    output_dir = args.output_dir or ROOT / "results" / str(experiment["output_dir"])
    work_dir = args.work_dir or ROOT / "results" / str(experiment["work_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = ROOT / "results" / "t16_v3c_dev2_dataset.npz"
    if args.analysis_only:
        existing_dashboard = read_json(
            output_dir / "v3f_deeponet_frontier_dashboard.json"
        )
        existing_report = read_json(output_dir / "v3f_deeponet_frontier_report.json")
        analyze_and_write(
            experiment,
            output_dir,
            read_csv(output_dir / "v3f_deeponet_lr_tuning.csv"),
            float(existing_dashboard["selected_base_learning_rate"]),
            float(existing_dashboard["learning_rate_screen_total_seconds"]),
            float(existing_dashboard["discarded_learning_rate_search_seconds"]),
            read_csv(output_dir / "v3f_deeponet_checkpoints.csv"),
            read_csv(output_dir / "v3f_deeponet_history.csv"),
            read_csv(output_dir / "v3f_deeponet_samples.csv"),
            environment=existing_report["environment"],
            provenance=build_provenance(args.config, experiment, dataset_path),
        )
        return
    generate_dataset(dataset_config, dataset_path, force=bool(args.force_data))
    base_data = load_npz(dataset_path)
    budget = int(experiment["total_budget"])
    direct = prepare_direct_operator_data(
        base_data,
        [budget],
        int(experiment["fixed_query_index"]),
        int(experiment["audit_query_index"]),
    )
    selected_ridge, champions, baseline_rows = tune_classical_baselines(
        direct,
        [budget],
        [float(value) for value in experiment["ridge_relative_grid"]],
    )
    if champions[budget] != "ridge":
        raise RuntimeError("v3f requires validation-locked ridge")
    data = append_ray_view_channels(replace_lift_with_ridge(direct, selected_ridge))
    write_csv(output_dir / "v3f_deeponet_baseline_tuning.csv", baseline_rows)

    seeds = [int(value) for value in experiment["training_seeds"]]
    base_epochs = int(experiment["base_epochs"])
    block_epochs = int(experiment["continuation_block_epochs"])
    max_epochs = int(experiment["max_total_epochs"])
    continuation_epochs = max_epochs - base_epochs
    if continuation_epochs <= 0 or continuation_epochs % block_epochs:
        raise ValueError("max epochs must equal base plus complete continuation blocks")

    lr_rows: list[dict[str, object]] = []
    base_records: dict[tuple[float, int], dict[str, object]] = {}
    for learning_rate in [float(value) for value in experiment["base_learning_rate_grid"]]:
        for outer_seed in seeds:
            base_seed = outer_seed + 101
            set_seed(base_seed)
            model = make_deeponet(model_config, data)
            record = train_model(
                "base_deeponet",
                model,
                data,
                training_config(
                    model_config,
                    base_seed,
                    base_epochs,
                    device_name,
                    learning_rate=learning_rate,
                    fixed_epochs=True,
                ),
                work_dir / f"lr_{learning_rate:g}" / str(outer_seed),
            )
            if int(record["epochs_ran"]) != base_epochs:
                raise RuntimeError("DeepONet LR screen did not complete fixed epochs")
            base_records[(learning_rate, outer_seed)] = {
                "state": cpu_state(record["model"]),
                "best_validation_rel_l2": float(record["best_val_rel_l2"]),
                "best_epoch": int(record["best_epoch"]),
                "train_seconds": float(record["train_seconds"]),
                "history": copy.deepcopy(record["history"]),
            }
            lr_rows.append(
                {
                    "learning_rate": learning_rate,
                    "model_seed": outer_seed,
                    "training_seed": base_seed,
                    "attempted_epochs": base_epochs,
                    "best_checkpoint_epoch": int(record["best_epoch"]),
                    "best_validation_rel_l2": float(record["best_val_rel_l2"]),
                    "endpoint_validation_rel_l2": float(record["history"][-1]["val_rel_l2"]),
                    "train_seconds": float(record["train_seconds"]),
                    "selection_scope": "validation_only",
                }
            )
            del record, model
            if device.type == "mps":
                torch.mps.empty_cache()
    selected_lr, lr_rows = select_global_learning_rate(lr_rows, len(seeds))
    lr_screen_seconds = float(sum(float(row["train_seconds"]) for row in lr_rows))
    selected_base_seconds = float(
        sum(
            float(base_records[(selected_lr, seed)]["train_seconds"])
            for seed in seeds
        )
    )
    discarded_lr_seconds = lr_screen_seconds - selected_base_seconds
    print(
        f"DeepONet validation-only LR champion: {selected_lr:g}; "
        f"screen={lr_screen_seconds:.2f}s, discarded={discarded_lr_seconds:.2f}s",
        flush=True,
    )

    checkpoint_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    base_states: dict[int, dict[str, torch.Tensor]] = {}
    base_selected_epochs: dict[int, int] = {}
    final_states: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
    final_selected_epochs: dict[tuple[str, int], int] = {}
    strategy_lookup = {str(row["id"]): row for row in experiment["strategies"]}
    train_indices = split_indices(data)["train"]
    train_sample_count = len(train_indices)
    batch_size = int(model_config["training"]["batch_size"])

    for outer_seed in seeds:
        base_seed = outer_seed + 101
        base_record = base_records[(selected_lr, outer_seed)]
        base_state = copy.deepcopy(base_record["state"])
        base_states[outer_seed] = base_state
        base_validation = float(base_record["best_validation_rel_l2"])
        base_selected_epoch = int(base_record["best_epoch"])
        base_selected_epochs[outer_seed] = base_selected_epoch
        history_rows.extend(
            {
                "strategy": "shared_base",
                "model_seed": outer_seed,
                "block_index": 0,
                "cumulative_epoch": int(row["epoch"]),
                **{key: value for key, value in row.items() if key != "epoch"},
            }
            for row in base_record["history"]
        )
        for strategy in strategy_ids(experiment):
            spec = strategy_lookup[strategy]
            model = make_deeponet(model_config, data)
            model.load_state_dict(base_state, strict=True)
            model = model.to(device)
            selected_state = copy.deepcopy(base_state)
            selected_validation = base_validation
            selected_epoch = base_selected_epoch
            cumulative_seconds = float(base_record["train_seconds"])
            checkpoint_rows.append(
                {
                    "strategy": strategy,
                    "model_seed": outer_seed,
                    "cumulative_epochs": base_epochs,
                    "block_index": 0,
                    "block_seed": base_seed,
                    "batch_order_contract_sha256": batch_order_contract_sha256(
                        train_sample_count,
                        batch_size,
                        base_epochs,
                        base_seed,
                        sample_indices=train_indices,
                    ),
                    "endpoint_validation_rel_l2": float(
                        base_record["history"][-1]["val_rel_l2"]
                    ),
                    "candidate_best_val_rel_l2": base_validation,
                    "selected_validation_rel_l2": base_validation,
                    "relative_validation_improvement_pct": 0.0,
                    "retained_previous_checkpoint": False,
                    "selected_checkpoint_epoch": selected_epoch,
                    "best_epoch_in_block": base_selected_epoch,
                    "train_seconds_block": float(base_record["train_seconds"]),
                    "cumulative_train_seconds": cumulative_seconds,
                    "ending_learning_rate": 0.0,
                    "selected_checkpoint_sha256": state_sha256(selected_state),
                    "endpoint_checkpoint_sha256": "base_raw_endpoint_not_persisted",
                }
            )

            optimizer: torch.optim.Optimizer | None = None
            scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
            if bool(spec["carry_optimizer"]):
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=float(experiment["continuation_learning_rate"]),
                    weight_decay=float(model_config["training"]["weight_decay"]),
                )
                if str(spec["cosine_horizon"]) == "continuation":
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=continuation_epochs
                    )
            for block_index, cumulative_epoch in enumerate(
                range(base_epochs + block_epochs, max_epochs + 1, block_epochs),
                start=1,
            ):
                block_seed = outer_seed + 10_000 + block_index
                set_seed(block_seed)
                if not bool(spec["carry_optimizer"]):
                    optimizer = torch.optim.AdamW(
                        model.parameters(),
                        lr=float(experiment["continuation_learning_rate"]),
                        weight_decay=float(model_config["training"]["weight_decay"]),
                    )
                if str(spec["cosine_horizon"]) == "block":
                    assert optimizer is not None
                    for group in optimizer.param_groups:
                        group["lr"] = float(experiment["continuation_learning_rate"])
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=block_epochs
                    )
                assert optimizer is not None and scheduler is not None
                record = train_continuation_block(
                    model,
                    optimizer,
                    scheduler,
                    data,
                    model_config,
                    device,
                    block_seed,
                    block_epochs,
                )
                candidate_validation = float(record["best_validation_rel_l2"])
                retained = candidate_validation >= selected_validation - float(
                    experiment["checkpoint_acceptance_min_abs_improvement"]
                )
                previous_validation = selected_validation
                if not retained:
                    selected_validation = candidate_validation
                    selected_state = copy.deepcopy(record["best_state"])
                    selected_epoch = (
                        cumulative_epoch
                        - block_epochs
                        + int(record["best_epoch_in_block"])
                    )
                cumulative_seconds += float(record["train_seconds"])
                checkpoint_rows.append(
                    {
                        "strategy": strategy,
                        "model_seed": outer_seed,
                        "cumulative_epochs": cumulative_epoch,
                        "block_index": block_index,
                        "block_seed": block_seed,
                        "batch_order_contract_sha256": batch_order_contract_sha256(
                            train_sample_count,
                            batch_size,
                            block_epochs,
                            block_seed,
                            sample_indices=train_indices,
                        ),
                        "endpoint_validation_rel_l2": float(
                            record["endpoint_validation_rel_l2"]
                        ),
                        "candidate_best_val_rel_l2": candidate_validation,
                        "selected_validation_rel_l2": selected_validation,
                        "relative_validation_improvement_pct": 100.0
                        * (previous_validation - selected_validation)
                        / (previous_validation + 1e-12),
                        "retained_previous_checkpoint": retained,
                        "selected_checkpoint_epoch": selected_epoch,
                        "best_epoch_in_block": int(record["best_epoch_in_block"]),
                        "train_seconds_block": float(record["train_seconds"]),
                        "cumulative_train_seconds": cumulative_seconds,
                        "ending_learning_rate": float(record["ending_learning_rate"]),
                        "selected_checkpoint_sha256": state_sha256(selected_state),
                        "endpoint_checkpoint_sha256": state_sha256(
                            record["endpoint_state"]
                        ),
                    }
                )
                history_rows.extend(
                    {
                        "strategy": strategy,
                        "model_seed": outer_seed,
                        "block_index": block_index,
                        "cumulative_epoch": cumulative_epoch
                        - block_epochs
                        + int(row["epoch_in_block"]),
                        **{
                            key: value
                            for key, value in row.items()
                            if key != "epoch_in_block"
                        },
                    }
                    for row in record["history"]
                )
            final_states[(strategy, outer_seed)] = selected_state
            final_selected_epochs[(strategy, outer_seed)] = selected_epoch
            print(
                f"seed={outer_seed} DeepONet strategy={strategy}: through {max_epochs}",
                flush=True,
            )

    validation_rows, _ = summarize_strategies(checkpoint_rows, experiment)
    deep_champion = choose_validation_champion(validation_rows, max_epochs)
    print(f"DeepONet validation champion: {deep_champion}", flush=True)
    fno_dashboard = read_json(FNO_RESULTS / "v3d_optimizer_dashboard.json")
    selection_payload = {
        "commit_version": "v3f-selection-v1",
        "selection_scope": "validation_only",
        "validation_aggregation": "sample_weighted_field_mean",
        "selected_base_learning_rate": selected_lr,
        "deeponet_validation_champion": deep_champion,
        "deeponet_final_checkpoint_sha256_by_seed": {
            str(seed): state_sha256(final_states[(deep_champion, seed)])
            for seed in seeds
        },
        "deeponet_validation_summary": validation_rows,
        "fno_validation_champion": str(fno_dashboard["validation_champion"]),
        "fno_dashboard_sha256": sha256_file(
            FNO_RESULTS / "v3d_optimizer_dashboard.json"
        ),
        "fno_validation_summary_sha256": sha256_file(
            FNO_RESULTS / "v3d_optimizer_validation_summary.csv"
        ),
        "post_selection_dataset_role": (
            "reused synthetic development diagnostic; not a fresh project-level audit"
        ),
        "dev2_or_q_audit_metric_present_in_selection_payload": False,
    }
    selection_commit_sha256 = canonical_sha256(selection_payload)
    (output_dir / "v3f_selection_commit.json").write_text(
        json.dumps(
            {
                **selection_payload,
                "selection_commit_sha256": selection_commit_sha256,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    sample_rows: list[dict[str, object]] = []
    for outer_seed, state in sorted(base_states.items()):
        for row in evaluate_deeponet_checkpoint(
            outer_seed,
            base_epochs,
            state,
            data,
            model_config,
            selected_ridge[budget],
            int(experiment["audit_query_index"]),
            device,
        ):
            sample_rows.append(
                {
                    "strategy": "base_24",
                    "selection_commit_sha256": selection_commit_sha256,
                    "selected_checkpoint_epoch": base_selected_epochs[outer_seed],
                    **row,
                }
            )
    for (strategy, outer_seed), state in sorted(final_states.items()):
        for row in evaluate_deeponet_checkpoint(
            outer_seed,
            max_epochs,
            state,
            data,
            model_config,
            selected_ridge[budget],
            int(experiment["audit_query_index"]),
            device,
        ):
            sample_rows.append(
                {
                    "strategy": strategy,
                    "selection_commit_sha256": selection_commit_sha256,
                    "selected_checkpoint_epoch": final_selected_epochs[
                        (strategy, outer_seed)
                    ],
                    **row,
                }
            )

    analyze_and_write(
        experiment,
        output_dir,
        lr_rows,
        selected_lr,
        lr_screen_seconds,
        discarded_lr_seconds,
        checkpoint_rows,
        history_rows,
        sample_rows,
        environment={
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "requested_device": device_name,
            "runtime_device": str(device),
        },
        provenance=build_provenance(args.config, experiment, dataset_path),
    )
    print(f"results: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
