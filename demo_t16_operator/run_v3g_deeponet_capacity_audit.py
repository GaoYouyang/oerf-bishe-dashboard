#!/usr/bin/env python3
"""Audit bounded DeepONet capacity and learning-rate sensitivity for T16."""

from __future__ import annotations

import argparse
import copy
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
        stratified_bootstrap_interval,
        test_variant_indices,
        tune_classical_baselines,
        weighted_quantile,
    )
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
    from .run_v3f_deeponet_frontier import (
        canonical_sha256,
        configured_model_data,
        evaluate_deeponet_checkpoint,
        make_deeponet,
        sha256_file,
    )
    from .train_eval import choose_device, set_seed, train_model
except ImportError:
    from data import BOSTDataset, generate_dataset, load_npz, split_indices
    from direct_operator_data import prepare_direct_operator_data, replace_lift_with_ridge
    from own_algorithm_data import append_ray_view_channels
    from run_direct_operator_pilot import (
        domain_equal_weights,
        stratified_bootstrap_interval,
        test_variant_indices,
        tune_classical_baselines,
        weighted_quantile,
    )
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
    from run_v3f_deeponet_frontier import (
        canonical_sha256,
        configured_model_data,
        evaluate_deeponet_checkpoint,
        make_deeponet,
        sha256_file,
    )
    from train_eval import choose_device, set_seed, train_model


ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = ROOT / "configs"
DEFAULT_CONFIG = CONFIG_ROOT / "v3g_deeponet_capacity_audit.json"
V3F_RESULTS = ROOT / "results" / "v3f_deeponet_frontier"
FNO_RESULTS = ROOT / "results" / "v3d_fno_optimizer_audit"
CHECKSUM_FILES = [
    "v3g_deeponet_baseline_tuning.csv",
    "v3g_variant_manifest.csv",
    "v3g_screen.csv",
    "v3g_screen_summary.csv",
    "v3g_history.csv",
    "v3g_checkpoints.csv",
    "v3g_validation_summary.csv",
    "v3g_strategy_summary.csv",
    "v3g_samples.csv",
    "v3g_clusters.csv",
    "v3g_test_summary.csv",
    "v3g_pairwise_summary.csv",
    "v3g_seed_summary.csv",
    "v3g_validation_comparison.csv",
    "v3g_cross_baseline_pairwise.csv",
    "v3g_cross_baseline_domains.csv",
    "v3g_cross_baseline_seeds.csv",
    "v3g_selection_commit.json",
    "v3g_deeponet_capacity_dashboard.json",
    "v3g_deeponet_capacity_report.json",
    "t16_v3g_deeponet_capacity_audit.png",
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


def variant_model_config(base: dict, spec: dict) -> dict:
    config = copy.deepcopy(base)
    config["own_models"]["deeponet"] = {
        "branch_hidden": int(spec["branch_hidden"]),
        "trunk_hidden": int(spec["trunk_hidden"]),
        "rank": int(spec["rank"]),
        "pool_shape": [int(value) for value in spec["pool_shape"]],
    }
    return config


def parameter_count(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def build_variant_manifest(
    experiment: dict,
    base_model_config: dict,
    data: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    counts: dict[str, int] = {}
    for spec in experiment["variants"]:
        model = make_deeponet(variant_model_config(base_model_config, spec), data)
        counts[str(spec["id"])] = parameter_count(model)
        del model
    reference_id = str(experiment["parameter_cap_reference_variant"])
    if reference_id not in counts:
        raise ValueError("parameter-cap reference variant is missing")
    cap = int(np.floor(counts[reference_id] * float(experiment["parameter_cap_ratio"])))
    rows = []
    for order, spec in enumerate(experiment["variants"]):
        variant_id = str(spec["id"])
        declared_screen = bool(spec["screen"])
        within_cap = counts[variant_id] <= cap
        if declared_screen and not within_cap:
            raise ValueError(f"screen variant {variant_id} exceeds the parameter cap")
        if not declared_screen and within_cap:
            raise ValueError(f"excluded variant {variant_id} does not exceed the parameter cap")
        rows.append(
            {
                "pre_registered_order": order,
                "variant_id": variant_id,
                "label": str(spec["label"]),
                "screen": declared_screen,
                "branch_hidden": int(spec["branch_hidden"]),
                "trunk_hidden": int(spec["trunk_hidden"]),
                "rank": int(spec["rank"]),
                "pool_depth_bins": int(spec["pool_shape"][0]),
                "pool_height_bins": int(spec["pool_shape"][1]),
                "pool_width_bins": int(spec["pool_shape"][2]),
                "parameter_count": counts[variant_id],
                "reference_parameter_count": counts[reference_id],
                "parameter_cap": cap,
                "parameter_ratio_to_reference": counts[variant_id] / counts[reference_id],
                "within_parameter_cap": within_cap,
                "hypothesis": str(spec["hypothesis"]),
                "exclusion_reason": "" if declared_screen else "pre-registered parameter-cap exclusion",
            }
        )
    return rows


def select_screen_champion(
    rows: list[dict[str, object]], expected_seed_count: int
) -> tuple[str, float, list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[tuple[str, float], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["variant_id"]), float(row["learning_rate"]))].append(row)
    if any(len(values) != expected_seed_count for values in grouped.values()):
        raise ValueError("every architecture-learning-rate cell must use all seeds")
    summaries = []
    for (variant_id, learning_rate), values in sorted(grouped.items()):
        parameters = {int(row["parameter_count"]) for row in values}
        if len(parameters) != 1:
            raise ValueError("parameter count changed within a screen cell")
        errors = np.asarray(
            [float(row["best_validation_rel_l2"]) for row in values],
            dtype=np.float64,
        )
        summaries.append(
            {
                "variant_id": variant_id,
                "learning_rate": learning_rate,
                "model_seed_count": len(values),
                "parameter_count": parameters.pop(),
                "mean_best_validation_rel_l2": float(np.mean(errors)),
                "min_best_validation_rel_l2": float(np.min(errors)),
                "max_best_validation_rel_l2": float(np.max(errors)),
                "mean_endpoint_validation_rel_l2": float(
                    np.mean([float(row["endpoint_validation_rel_l2"]) for row in values])
                ),
                "total_train_seconds": float(
                    sum(float(row["train_seconds"]) for row in values)
                ),
                "selection_scope": "validation_only",
            }
        )
    champion = min(
        summaries,
        key=lambda row: (
            float(row["mean_best_validation_rel_l2"]),
            int(row["parameter_count"]),
            str(row["variant_id"]),
            float(row["learning_rate"]),
        ),
    )
    selected_variant = str(champion["variant_id"])
    selected_lr = float(champion["learning_rate"])
    for summary in summaries:
        summary["global_validation_champion"] = (
            str(summary["variant_id"]) == selected_variant
            and float(summary["learning_rate"]) == selected_lr
        )
    mean_lookup = {
        (str(row["variant_id"]), float(row["learning_rate"])): float(
            row["mean_best_validation_rel_l2"]
        )
        for row in summaries
    }
    for row in rows:
        key = (str(row["variant_id"]), float(row["learning_rate"]))
        row["mean_validation_rel_l2_for_cell"] = mean_lookup[key]
        row["global_validation_champion"] = (
            key == (selected_variant, selected_lr)
        )
    return selected_variant, selected_lr, rows, summaries


def build_validation_comparison(
    candidate_validation: list[dict[str, object]],
    candidate_champion: str,
    experiment: dict,
) -> list[dict[str, object]]:
    v3f_dashboard = read_json(V3F_RESULTS / "v3f_deeponet_frontier_dashboard.json")
    fno_dashboard = read_json(FNO_RESULTS / "v3d_optimizer_dashboard.json")
    sources = {
        "v3g_selected_deeponet": (
            candidate_validation,
            candidate_champion,
        ),
        "v3f_reference_deeponet": (
            read_csv(V3F_RESULTS / "v3f_deeponet_validation_summary.csv"),
            str(v3f_dashboard["deeponet_validation_champion"]),
        ),
        "fno": (
            read_csv(FNO_RESULTS / "v3d_optimizer_validation_summary.csv"),
            str(fno_dashboard["validation_champion"]),
        ),
    }
    epochs = [int(experiment["base_epochs"])] + [
        int(value) for value in experiment["fixed_epoch_checkpoints"]
    ]
    rows = []
    for cumulative_epochs in epochs:
        reference_error = next(
            float(row["mean_validation_rel_l2"])
            for row in sources["v3f_reference_deeponet"][0]
            if str(row["strategy"]) == sources["v3f_reference_deeponet"][1]
            and int(row["cumulative_epochs"]) == cumulative_epochs
        )
        for architecture, (values, strategy) in sources.items():
            row = next(
                item
                for item in values
                if str(item["strategy"]) == strategy
                and int(item["cumulative_epochs"]) == cumulative_epochs
            )
            error = float(row["mean_validation_rel_l2"])
            rows.append(
                {
                    "architecture": architecture,
                    "strategy": strategy,
                    "cumulative_epochs": cumulative_epochs,
                    "mean_validation_rel_l2": error,
                    "mean_cumulative_train_seconds": float(
                        row["mean_cumulative_train_seconds"]
                    ),
                    "relative_improvement_vs_v3f_reference_pct": 100.0
                    * (reference_error - error)
                    / reference_error,
                    "selection_scope": "validation_only",
                }
            )
    return rows


def compare_candidate_to_reference(
    candidate_name: str,
    candidate_clusters: list[dict[str, object]],
    candidate_samples: list[dict[str, object]],
    candidate_strategy: str,
    reference_name: str,
    reference_clusters: list[dict[str, object]],
    reference_samples: list[dict[str, object]],
    reference_strategy: str,
    experiment: dict,
    bootstrap_seed_offset: int,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    candidate_cluster_rows = [
        row for row in candidate_clusters if str(row["strategy"]) == candidate_strategy
    ]
    reference_cluster_rows = [
        row for row in reference_clusters if str(row["strategy"]) == reference_strategy
    ]
    candidate_lookup = {
        int(row["source_index"]): row for row in candidate_cluster_rows
    }
    reference_lookup = {
        int(row["source_index"]): row for row in reference_cluster_rows
    }
    if set(candidate_lookup) != set(reference_lookup):
        raise ValueError("candidate and reference development fields do not align")
    augmented = []
    for source_index, candidate in sorted(candidate_lookup.items()):
        reference = reference_lookup[source_index]
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
    rng = np.random.default_rng(
        int(experiment["bootstrap_seed"]) + int(bootstrap_seed_offset)
    )
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
                "candidate": candidate_name,
                "comparator": reference_name,
                "source_split": domain,
                "independent_field_count": len(subset),
                "mean_field_superiority_pct": float(np.mean(values)),
                "p10_field_superiority_pct": float(np.quantile(values, 0.10)),
                "field_harm_rate_gt_1pct": float(np.mean(values < -1.0)),
            }
        )

    candidate_sample_lookup = {
        (int(row["model_seed"]), int(row["source_index"])): row
        for row in candidate_samples
        if str(row["strategy"]) == candidate_strategy
    }
    reference_sample_lookup = {
        (int(row["model_seed"]), int(row["source_index"])): row
        for row in reference_samples
        if str(row["strategy"]) == reference_strategy
    }
    if set(candidate_sample_lookup) != set(reference_sample_lookup):
        raise ValueError("candidate and reference seed-level fields do not align")
    seed_rows = []
    for model_seed in sorted({key[0] for key in candidate_sample_lookup}):
        values = []
        for key, candidate in candidate_sample_lookup.items():
            if key[0] != model_seed:
                continue
            reference = reference_sample_lookup[key]
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
                "candidate": candidate_name,
                "comparator": reference_name,
                "model_seed": model_seed,
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
    p10 = weighted_quantile(field, weights, 0.10)
    harm = float(np.sum(weights * (field < -1.0)))
    every_domain = all(
        float(row["mean_field_superiority_pct"]) >= 0.0 for row in domain_rows
    )
    every_seed = all(
        float(row["domain_equal_mean_field_superiority_pct"]) > 0.0
        for row in seed_rows
    )
    gate_pass = bool(
        ci_low > float(gate["minimum_field_superiority_ci95_low_pct"])
        and p10 >= float(gate["minimum_p10_field_superiority_pct"])
        and harm <= float(gate["maximum_field_harm_rate_gt_1pct"])
        and (not bool(gate["require_every_domain_mean_nonnegative"]) or every_domain)
        and (not bool(gate["require_every_seed_mean_positive"]) or every_seed)
    )
    pairwise = {
        "selection_basis": "candidate frozen by validation before reused dev2 read",
        "candidate": candidate_name,
        "comparator": reference_name,
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
        "uncertainty_unit": "field cluster after model-seed collapse",
    }
    return pairwise, domain_rows, seed_rows


def plot_results(
    screen_summary: list[dict[str, object]],
    validation_comparison: list[dict[str, object]],
    pairwise: list[dict[str, object]],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    variants = []
    for row in screen_summary:
        variant_id = str(row["variant_id"])
        if variant_id not in variants:
            variants.append(variant_id)
    x = np.arange(len(variants))
    for learning_rate in sorted(
        {float(row["learning_rate"]) for row in screen_summary}
    ):
        lookup = {
            str(row["variant_id"]): float(row["mean_best_validation_rel_l2"])
            for row in screen_summary
            if float(row["learning_rate"]) == learning_rate
        }
        axes[0].plot(
            x,
            [lookup[name] for name in variants],
            marker="o",
            label=f"lr={learning_rate:g}",
        )
    axes[0].set_xticks(x, [name.replace("_", "\n") for name in variants], fontsize=7)
    axes[0].set_ylabel("mean best validation relative L2")
    axes[0].set_title("72-run validation-only screen")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    labels = {
        "v3g_selected_deeponet": "v3g selected DeepONet",
        "v3f_reference_deeponet": "v3f reference DeepONet",
        "fno": "FNO",
    }
    for architecture in labels:
        rows = sorted(
            [
                row
                for row in validation_comparison
                if str(row["architecture"]) == architecture
            ],
            key=lambda row: int(row["cumulative_epochs"]),
        )
        axes[1].plot(
            [int(row["cumulative_epochs"]) for row in rows],
            [float(row["mean_validation_rel_l2"]) for row in rows],
            marker="o",
            label=labels[architecture],
        )
    axes[1].set_xlabel("attempted epochs")
    axes[1].set_ylabel("mean prefix-best validation relative L2")
    axes[1].set_title("Long-horizon validation comparison")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    names = [str(row["comparator"]) for row in pairwise]
    means = [float(row["mean_field_superiority_pct"]) for row in pairwise]
    lows = [float(row["field_superiority_ci95_low"]) for row in pairwise]
    highs = [float(row["field_superiority_ci95_high"]) for row in pairwise]
    axes[2].bar(
        names,
        means,
        color=["#1f7a5a" if value >= 0 else "#b04a4a" for value in means],
        yerr=[
            [mean - low for mean, low in zip(means, lows)],
            [high - mean for mean, high in zip(means, highs)],
        ],
        capsize=4,
    )
    axes[2].axhline(0.0, color="black", linewidth=1)
    axes[2].set_ylabel("field superiority of v3g (%)")
    axes[2].set_title("Reused dev2 diagnostic\n(field-cluster 95% CI)")
    axes[2].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


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
        "v3f_frontier_script_sha256": sha256_file(
            ROOT / "run_v3f_deeponet_frontier.py"
        ),
        "dataset_npz_sha256": sha256_file(dataset_path),
        "source_v3f_dashboard_sha256": sha256_file(
            V3F_RESULTS / "v3f_deeponet_frontier_dashboard.json"
        ),
        "source_v3f_clusters_sha256": sha256_file(
            V3F_RESULTS / "v3f_deeponet_clusters.csv"
        ),
        "source_fno_dashboard_sha256": sha256_file(
            FNO_RESULTS / "v3d_optimizer_dashboard.json"
        ),
        "source_fno_clusters_sha256": sha256_file(
            FNO_RESULTS / "v3d_optimizer_clusters.csv"
        ),
        "dataset_npz_public": False,
        "checkpoint_weights_public": False,
    }


def write_output_checksums(output_dir: Path) -> None:
    lines = []
    for filename in CHECKSUM_FILES:
        digest = hashlib.sha256((output_dir / filename).read_bytes()).hexdigest()
        lines.append(f"{digest}  {filename}")
    (output_dir / "v3g_deeponet_capacity_checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="ascii"
    )


def analyze_and_write(
    experiment: dict,
    output_dir: Path,
    manifest_rows: list[dict[str, object]],
    screen_rows: list[dict[str, object]],
    screen_summary: list[dict[str, object]],
    selected_variant: str,
    selected_lr: float,
    screen_seconds: float,
    discarded_screen_seconds: float,
    checkpoint_rows: list[dict[str, object]],
    history_rows: list[dict[str, object]],
    sample_rows: list[dict[str, object]],
    environment: dict[str, object],
    provenance: dict[str, object],
) -> None:
    validation_rows, decisions = summarize_strategies(checkpoint_rows, experiment)
    candidate_champion = choose_validation_champion(
        validation_rows, int(experiment["max_total_epochs"])
    )
    strategy_summary = build_strategy_summary(
        validation_rows, decisions, candidate_champion, experiment
    )
    cluster_rows = collapse_test_rows(sample_rows)
    test_summary = summarize_test_rows(cluster_rows, experiment)
    pairwise_summary = build_pairwise_summary(
        cluster_rows, candidate_champion, experiment
    )
    seed_summary = summarize_seed_rows(sample_rows)
    selection_commit = read_json(output_dir / "v3g_selection_commit.json")
    selection_commit_sha256 = str(selection_commit["selection_commit_sha256"])

    v3f_dashboard = read_json(V3F_RESULTS / "v3f_deeponet_frontier_dashboard.json")
    fno_dashboard = read_json(FNO_RESULTS / "v3d_optimizer_dashboard.json")
    v3f_champion = str(v3f_dashboard["deeponet_validation_champion"])
    fno_champion = str(fno_dashboard["validation_champion"])
    v3f_clusters = read_csv(V3F_RESULTS / "v3f_deeponet_clusters.csv")
    v3f_samples = read_csv(V3F_RESULTS / "v3f_deeponet_samples.csv")
    fno_clusters = read_csv(FNO_RESULTS / "v3d_optimizer_clusters.csv")
    fno_samples = read_csv(FNO_RESULTS / "v3d_optimizer_samples.csv")

    comparison_rows = build_validation_comparison(
        validation_rows, candidate_champion, experiment
    )
    cross_pairwise = []
    cross_domains = []
    cross_seeds = []
    for offset, reference in enumerate(
        (
            ("v3f_reference_deeponet", v3f_clusters, v3f_samples, v3f_champion),
            ("fno", fno_clusters, fno_samples, fno_champion),
        ),
        start=1,
    ):
        pair, domains, seeds = compare_candidate_to_reference(
            "v3g_selected_deeponet",
            cluster_rows,
            sample_rows,
            candidate_champion,
            reference[0],
            reference[1],
            reference[2],
            reference[3],
            experiment,
            offset,
        )
        cross_pairwise.append(pair)
        cross_domains.extend(domains)
        cross_seeds.extend(seeds)

    plot_results(
        screen_summary,
        comparison_rows,
        cross_pairwise,
        output_dir / "t16_v3g_deeponet_capacity_audit.png",
    )
    write_csv(output_dir / "v3g_variant_manifest.csv", manifest_rows)
    write_csv(output_dir / "v3g_screen.csv", screen_rows)
    write_csv(output_dir / "v3g_screen_summary.csv", screen_summary)
    write_csv(output_dir / "v3g_history.csv", history_rows)
    write_csv(output_dir / "v3g_checkpoints.csv", checkpoint_rows)
    write_csv(output_dir / "v3g_validation_summary.csv", validation_rows)
    write_csv(output_dir / "v3g_strategy_summary.csv", strategy_summary)
    write_csv(output_dir / "v3g_samples.csv", sample_rows)
    write_csv(output_dir / "v3g_clusters.csv", cluster_rows)
    write_csv(output_dir / "v3g_test_summary.csv", test_summary)
    write_csv(output_dir / "v3g_pairwise_summary.csv", pairwise_summary)
    write_csv(output_dir / "v3g_seed_summary.csv", seed_summary)
    write_csv(output_dir / "v3g_validation_comparison.csv", comparison_rows)
    write_csv(output_dir / "v3g_cross_baseline_pairwise.csv", cross_pairwise)
    write_csv(output_dir / "v3g_cross_baseline_domains.csv", cross_domains)
    write_csv(output_dir / "v3g_cross_baseline_seeds.csv", cross_seeds)

    max_epochs = int(experiment["max_total_epochs"])
    final_validation = {
        str(row["architecture"]): float(row["mean_validation_rel_l2"])
        for row in comparison_rows
        if int(row["cumulative_epochs"]) == max_epochs
    }
    reference_error = final_validation["v3f_reference_deeponet"]
    fno_error = final_validation["fno"]
    candidate_error = final_validation["v3g_selected_deeponet"]
    gap_denominator = reference_error - fno_error
    fno_gap_closed_pct = (
        100.0 * (reference_error - candidate_error) / gap_denominator
        if abs(gap_denominator) > 1e-12
        else 0.0
    )
    reference_pair = next(
        row for row in cross_pairwise if str(row["comparator"]) == "v3f_reference_deeponet"
    )
    fno_pair = next(row for row in cross_pairwise if str(row["comparator"]) == "fno")
    status = "BOUNDED_DEEPONET_CAPACITY_AUDIT_COMPLETE"
    baseline_decision = {
        "selected_variant_improves_reference_validation": candidate_error < reference_error,
        "selected_variant_passes_reused_dev2_gate_vs_reference": bool(
            reference_pair["development_gate_pass"]
        ),
        "fno_remains_validation_better": fno_error < candidate_error,
        "fno_gap_closed_pct": fno_gap_closed_pct,
        "freeze_selected_deeponet_as_stronger_baseline": bool(
            candidate_error < reference_error
            and float(reference_pair["mean_field_superiority_pct"]) > 0.0
        ),
        "thesis_claim_eligible": False,
    }
    dashboard = {
        "experiment": experiment["name"],
        "scientific_status": status,
        "development_fields_only": True,
        "blind_final_opened": False,
        "confirmatory_superiority_eligible": False,
        "screen_cell_count": len(screen_summary),
        "screen_run_count": len(screen_rows),
        "screen_model_seed_count": len(experiment["training_seeds"]),
        "screen_summary": screen_summary,
        "screen_total_seconds": screen_seconds,
        "discarded_screen_seconds": discarded_screen_seconds,
        "selected_variant": selected_variant,
        "selected_learning_rate": selected_lr,
        "selected_variant_parameter_count": next(
            int(row["parameter_count"])
            for row in manifest_rows
            if str(row["variant_id"]) == selected_variant
        ),
        "selected_optimizer_strategy": candidate_champion,
        "selection_commit_sha256": selection_commit_sha256,
        "final_validation_rel_l2": final_validation,
        "cross_baseline_pairwise": cross_pairwise,
        "baseline_decision": baseline_decision,
        "validation_comparison": comparison_rows,
        "provenance": provenance,
    }
    (output_dir / "v3g_deeponet_capacity_dashboard.json").write_text(
        json.dumps(dashboard, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    report = {
        "status": "completed",
        "scientific_status": status,
        "environment": environment,
        "provenance": provenance,
        "protocol": {
            "pre_registered_bounded_variant_set": True,
            "capacity_cap_enforced_before_training": True,
            "architecture_and_learning_rate_selected_by_three_seed_mean_validation": True,
            "all_screen_cells_run_for_fixed_24_epochs": True,
            "screen_batch_order_contract_binds_actual_train_indices": True,
            "long_horizon_optimizer_selected_by_final_validation_only": True,
            "validation_metric_aggregated_per_sample": True,
            "reused_dev2_computed_after_selection_commit": True,
            "dev2_cannot_change_architecture_learning_rate_or_optimizer": True,
            "blind_final_opened": False,
        },
        "selected_variant": selected_variant,
        "selected_learning_rate": selected_lr,
        "selected_optimizer_strategy": candidate_champion,
        "screen_total_seconds": screen_seconds,
        "discarded_screen_seconds": discarded_screen_seconds,
        "selection_commit_sha256": selection_commit_sha256,
        "final_validation_rel_l2": final_validation,
        "cross_baseline_pairwise": cross_pairwise,
        "baseline_decision": baseline_decision,
        "claims_boundary": [
            "This is a synthetic 8x16x16, K=6 development audit, not real BOST evidence.",
            "The 72-run architecture-learning-rate screen is baseline due diligence, not a thesis contribution.",
            "The selected cell is determined by three-seed mean validation error with a pre-registered parameter cap.",
            "Only the selected screen cell receives the three-strategy 240-epoch continuation audit.",
            "The v3c dev2 fields were already inspected in earlier stages and are reused post-selection diagnostics, not a fresh audit set.",
            "Field-cluster bootstrap intervals collapse the three model seeds before resampling fields; seed means are listed separately.",
            "A better bounded DeepONet baseline strengthens future comparisons but cannot establish novelty or general DeepONet weakness.",
            "No blind-final, real-flow, geometry-transfer or publication claim is opened by this audit.",
        ],
    }
    (output_dir / "v3g_deeponet_capacity_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_output_checksums(output_dir)
    print(
        json.dumps(
            {
                "scientific_status": status,
                "selected_variant": selected_variant,
                "selected_learning_rate": selected_lr,
                "selected_optimizer_strategy": candidate_champion,
                "final_validation_rel_l2": final_validation,
                "baseline_decision": baseline_decision,
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
    base_model_config = configured_model_data(dataset_config, own_config)
    device_name = args.device or str(dataset_config["training"]["device"])
    device = choose_device(device_name)
    output_dir = args.output_dir or ROOT / "results" / str(experiment["output_dir"])
    work_dir = args.work_dir or ROOT / "results" / str(experiment["work_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = ROOT / "results" / "t16_v3c_dev2_dataset.npz"

    if args.analysis_only:
        dashboard = read_json(output_dir / "v3g_deeponet_capacity_dashboard.json")
        report = read_json(output_dir / "v3g_deeponet_capacity_report.json")
        analyze_and_write(
            experiment,
            output_dir,
            read_csv(output_dir / "v3g_variant_manifest.csv"),
            read_csv(output_dir / "v3g_screen.csv"),
            read_csv(output_dir / "v3g_screen_summary.csv"),
            str(dashboard["selected_variant"]),
            float(dashboard["selected_learning_rate"]),
            float(dashboard["screen_total_seconds"]),
            float(dashboard["discarded_screen_seconds"]),
            read_csv(output_dir / "v3g_checkpoints.csv"),
            read_csv(output_dir / "v3g_history.csv"),
            read_csv(output_dir / "v3g_samples.csv"),
            environment=report["environment"],
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
        raise RuntimeError("v3g requires validation-locked ridge")
    data = append_ray_view_channels(replace_lift_with_ridge(direct, selected_ridge))
    write_csv(output_dir / "v3g_deeponet_baseline_tuning.csv", baseline_rows)
    manifest_rows = build_variant_manifest(experiment, base_model_config, data)
    write_csv(output_dir / "v3g_variant_manifest.csv", manifest_rows)

    seeds = [int(value) for value in experiment["training_seeds"]]
    base_epochs = int(experiment["base_epochs"])
    block_epochs = int(experiment["continuation_block_epochs"])
    max_epochs = int(experiment["max_total_epochs"])
    continuation_epochs = max_epochs - base_epochs
    if continuation_epochs <= 0 or continuation_epochs % block_epochs:
        raise ValueError("max epochs must equal base plus complete continuation blocks")
    train_indices = split_indices(data)["train"]
    train_sample_count = len(train_indices)
    batch_size = int(base_model_config["training"]["batch_size"])
    spec_lookup = {str(spec["id"]): spec for spec in experiment["variants"]}
    parameter_lookup = {
        str(row["variant_id"]): int(row["parameter_count"])
        for row in manifest_rows
    }

    screen_rows: list[dict[str, object]] = []
    base_records: dict[tuple[str, float, int], dict[str, object]] = {}
    for spec in experiment["variants"]:
        if not bool(spec["screen"]):
            continue
        variant_id = str(spec["id"])
        model_config = variant_model_config(base_model_config, spec)
        for learning_rate in [
            float(value) for value in experiment["screen_learning_rates"]
        ]:
            for outer_seed in seeds:
                base_seed = outer_seed + 101
                set_seed(base_seed)
                model = make_deeponet(model_config, data)
                record = train_model(
                    f"screen_{variant_id}_{learning_rate:g}",
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
                    work_dir / variant_id / f"lr_{learning_rate:g}" / str(outer_seed),
                )
                if int(record["epochs_ran"]) != base_epochs:
                    raise RuntimeError("v3g screen did not complete fixed epochs")
                state = cpu_state(record["model"])
                base_records[(variant_id, learning_rate, outer_seed)] = {
                    "state": state,
                    "best_validation_rel_l2": float(record["best_val_rel_l2"]),
                    "best_epoch": int(record["best_epoch"]),
                    "train_seconds": float(record["train_seconds"]),
                    "history": copy.deepcopy(record["history"]),
                }
                screen_rows.append(
                    {
                        "variant_id": variant_id,
                        "learning_rate": learning_rate,
                        "model_seed": outer_seed,
                        "training_seed": base_seed,
                        "parameter_count": parameter_lookup[variant_id],
                        "attempted_epochs": base_epochs,
                        "best_checkpoint_epoch": int(record["best_epoch"]),
                        "best_validation_rel_l2": float(record["best_val_rel_l2"]),
                        "endpoint_validation_rel_l2": float(
                            record["history"][-1]["val_rel_l2"]
                        ),
                        "train_seconds": float(record["train_seconds"]),
                        "batch_order_contract_sha256": batch_order_contract_sha256(
                            train_sample_count,
                            batch_size,
                            base_epochs,
                            base_seed,
                            sample_indices=train_indices,
                        ),
                        "selected_checkpoint_sha256": state_sha256(state),
                        "selection_scope": "validation_only",
                    }
                )
                del record, model
                if device.type == "mps":
                    torch.mps.empty_cache()
    selected_variant, selected_lr, screen_rows, screen_summary = select_screen_champion(
        screen_rows, len(seeds)
    )
    screen_seconds = float(sum(float(row["train_seconds"]) for row in screen_rows))
    selected_base_seconds = float(
        sum(
            float(base_records[(selected_variant, selected_lr, seed)]["train_seconds"])
            for seed in seeds
        )
    )
    discarded_screen_seconds = screen_seconds - selected_base_seconds
    write_csv(output_dir / "v3g_screen.csv", screen_rows)
    write_csv(output_dir / "v3g_screen_summary.csv", screen_summary)
    print(
        f"v3g screen champion: {selected_variant} at lr={selected_lr:g}; "
        f"screen={screen_seconds:.2f}s, discarded={discarded_screen_seconds:.2f}s",
        flush=True,
    )

    selected_model_config = variant_model_config(
        base_model_config, spec_lookup[selected_variant]
    )
    checkpoint_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    base_states: dict[int, dict[str, torch.Tensor]] = {}
    base_selected_epochs: dict[int, int] = {}
    final_states: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
    final_selected_epochs: dict[tuple[str, int], int] = {}
    strategy_lookup = {str(row["id"]): row for row in experiment["strategies"]}

    for outer_seed in seeds:
        base_seed = outer_seed + 101
        base_record = base_records[(selected_variant, selected_lr, outer_seed)]
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
            strategy_spec = strategy_lookup[strategy]
            model = make_deeponet(selected_model_config, data)
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
            if bool(strategy_spec["carry_optimizer"]):
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=float(experiment["continuation_learning_rate"]),
                    weight_decay=float(selected_model_config["training"]["weight_decay"]),
                )
                if str(strategy_spec["cosine_horizon"]) == "continuation":
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=continuation_epochs
                    )
            for block_index, cumulative_epoch in enumerate(
                range(base_epochs + block_epochs, max_epochs + 1, block_epochs),
                start=1,
            ):
                block_seed = outer_seed + 10_000 + block_index
                set_seed(block_seed)
                if not bool(strategy_spec["carry_optimizer"]):
                    optimizer = torch.optim.AdamW(
                        model.parameters(),
                        lr=float(experiment["continuation_learning_rate"]),
                        weight_decay=float(
                            selected_model_config["training"]["weight_decay"]
                        ),
                    )
                if str(strategy_spec["cosine_horizon"]) == "block":
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
                    selected_model_config,
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
                f"seed={outer_seed} v3g strategy={strategy}: through {max_epochs}",
                flush=True,
            )
            del model
            if device.type == "mps":
                torch.mps.empty_cache()

    validation_rows, _ = summarize_strategies(checkpoint_rows, experiment)
    candidate_champion = choose_validation_champion(validation_rows, max_epochs)
    v3f_dashboard = read_json(V3F_RESULTS / "v3f_deeponet_frontier_dashboard.json")
    fno_dashboard = read_json(FNO_RESULTS / "v3d_optimizer_dashboard.json")
    selection_payload = {
        "commit_version": "v3g-selection-v1",
        "selection_scope": "validation_only",
        "validation_aggregation": "sample_weighted_field_mean",
        "parameter_cap_reference_variant": experiment[
            "parameter_cap_reference_variant"
        ],
        "parameter_cap_ratio": experiment["parameter_cap_ratio"],
        "screen_definition_sha256": canonical_sha256(
            {
                "variants": experiment["variants"],
                "learning_rates": experiment["screen_learning_rates"],
                "training_seeds": seeds,
                "base_epochs": base_epochs,
                "tie_break": experiment["screen_tie_break"],
            }
        ),
        "selected_variant": selected_variant,
        "selected_learning_rate": selected_lr,
        "selected_optimizer_strategy": candidate_champion,
        "selected_final_checkpoint_sha256_by_seed": {
            str(seed): state_sha256(final_states[(candidate_champion, seed)])
            for seed in seeds
        },
        "screen_summary": screen_summary,
        "long_horizon_validation_summary": validation_rows,
        "v3f_reference_dashboard_sha256": sha256_file(
            V3F_RESULTS / "v3f_deeponet_frontier_dashboard.json"
        ),
        "v3f_reference_champion": str(
            v3f_dashboard["deeponet_validation_champion"]
        ),
        "fno_dashboard_sha256": sha256_file(
            FNO_RESULTS / "v3d_optimizer_dashboard.json"
        ),
        "fno_validation_champion": str(fno_dashboard["validation_champion"]),
        "post_selection_dataset_role": (
            "reused synthetic development diagnostic; not a fresh project-level audit"
        ),
        "dev2_or_q_audit_metric_present_in_selection_payload": False,
    }
    selection_commit_sha256 = canonical_sha256(selection_payload)
    (output_dir / "v3g_selection_commit.json").write_text(
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
    print(
        f"v3g validation champion frozen: {candidate_champion}; "
        f"commit={selection_commit_sha256[:12]}",
        flush=True,
    )

    sample_rows: list[dict[str, object]] = []
    for outer_seed, state in sorted(base_states.items()):
        for row in evaluate_deeponet_checkpoint(
            outer_seed,
            base_epochs,
            state,
            data,
            selected_model_config,
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
            selected_model_config,
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
        manifest_rows,
        screen_rows,
        screen_summary,
        selected_variant,
        selected_lr,
        screen_seconds,
        discarded_screen_seconds,
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
