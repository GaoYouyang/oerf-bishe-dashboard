#!/usr/bin/env python3
"""Audit whether the K=6 FNO reaches a validation plateau by 96 epochs."""

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
from torch.utils.data import DataLoader

try:
    from .data import BOSTDataset, generate_dataset, load_npz, split_indices
    from .direct_operator_data import prepare_direct_operator_data, replace_lift_with_ridge
    from .models import make_model
    from .own_algorithm_data import append_ray_view_channels
    from .run_direct_operator_pilot import (
        domain_equal_weights,
        prediction_metrics,
        relative_field_error,
        stratified_bootstrap_interval,
        test_variant_indices,
        tune_classical_baselines,
        write_checksums,
    )
    from .run_v3c_k6_dev2_pilot import state_sha256, training_config
    from .train_eval import (
        batch_relative_l2,
        choose_device,
        collect_predictions,
        sample_weighted_mean,
        set_seed,
        train_model,
    )
except ImportError:
    from data import BOSTDataset, generate_dataset, load_npz, split_indices
    from direct_operator_data import prepare_direct_operator_data, replace_lift_with_ridge
    from models import make_model
    from own_algorithm_data import append_ray_view_channels
    from run_direct_operator_pilot import (
        domain_equal_weights,
        prediction_metrics,
        relative_field_error,
        stratified_bootstrap_interval,
        test_variant_indices,
        tune_classical_baselines,
        write_checksums,
    )
    from run_v3c_k6_dev2_pilot import state_sha256, training_config
    from train_eval import (
        batch_relative_l2,
        choose_device,
        collect_predictions,
        sample_weighted_mean,
        set_seed,
        train_model,
    )


ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = ROOT / "configs"
DEFAULT_CONFIG = CONFIG_ROOT / "v3d_fno_saturation_audit.json"
CHECKSUM_FILES = [
    "v3d_fno_baseline_tuning.csv",
    "v3d_fno_history.csv",
    "v3d_fno_checkpoints.csv",
    "v3d_fno_validation_summary.csv",
    "v3d_fno_samples.csv",
    "v3d_fno_clusters.csv",
    "v3d_fno_test_summary.csv",
    "v3d_fno_seed_summary.csv",
    "v3d_fno_saturation_dashboard.json",
    "v3d_fno_saturation_report.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--seed-limit", type=int)
    parser.add_argument("--max-total-epochs", type=int)
    parser.add_argument("--force-data", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_csv_scalar(value: str) -> object:
    if value == "True":
        return True
    if value == "False":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def read_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {key: parse_csv_scalar(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if torch.is_tensor(value)
    }


def write_output_checksums(output_dir: Path) -> None:
    write_checksums(output_dir, CHECKSUM_FILES)
    source = output_dir / "direct_operator_checksums.sha256"
    target = output_dir / "v3d_fno_checksums.sha256"
    target.write_text(source.read_text(encoding="ascii"), encoding="ascii")
    source.unlink()


def evaluate_validation(
    model: torch.nn.Module,
    data: dict[str, np.ndarray],
    dataset_config: dict,
    device: torch.device,
) -> float:
    indices = split_indices(data)["val"]
    loader = DataLoader(
        BOSTDataset(data, indices),
        batch_size=int(dataset_config["training"]["batch_size"]),
    )
    model = model.to(device)
    model.eval()
    values = []
    batch_sizes = []
    with torch.no_grad():
        for batch in loader:
            prediction = model(batch["x"].to(device))
            values.append(
                float(batch_relative_l2(prediction, batch["field"].to(device)).cpu())
            )
            batch_sizes.append(int(batch["x"].shape[0]))
    return sample_weighted_mean(values, batch_sizes)


def relative_improvement_pct(previous: float, current: float) -> float:
    return 100.0 * (float(previous) - float(current)) / (float(previous) + 1e-12)


def summarize_validation(
    checkpoint_rows: list[dict[str, object]],
    rule: dict,
    base_epochs: int,
) -> list[dict[str, object]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in checkpoint_rows:
        grouped[int(row["cumulative_epochs"])].append(row)
    output = []
    for epoch, subset in sorted(grouped.items()):
        validation = np.asarray(
            [float(row["selected_validation_rel_l2"]) for row in subset],
            dtype=np.float64,
        )
        improvements = np.asarray(
            [float(row["relative_validation_improvement_pct"]) for row in subset],
            dtype=np.float64,
        )
        is_continuation = epoch > int(base_epochs)
        plateau_block = bool(
            is_continuation
            and float(np.mean(improvements))
            <= float(rule["maximum_mean_relative_improvement_pct_per_block"])
            and float(np.max(improvements))
            <= float(rule["maximum_seed_relative_improvement_pct_per_block"])
        )
        output.append(
            {
                "cumulative_epochs": epoch,
                "model_seed_count": len(subset),
                "mean_validation_rel_l2": float(np.mean(validation)),
                "min_validation_rel_l2": float(np.min(validation)),
                "max_validation_rel_l2": float(np.max(validation)),
                "mean_relative_validation_improvement_pct": float(
                    np.mean(improvements)
                ),
                "max_seed_relative_validation_improvement_pct": float(
                    np.max(improvements)
                ),
                "retained_previous_checkpoint_count": sum(
                    bool(row["retained_previous_checkpoint"]) for row in subset
                ),
                "mean_cumulative_train_seconds": float(
                    np.mean([float(row["cumulative_train_seconds"]) for row in subset])
                ),
                "plateau_block": plateau_block,
            }
        )
    return output


def select_validation_plateau(
    validation_summary: list[dict[str, object]],
    rule: dict,
    base_epochs: int,
) -> dict[str, object]:
    continuation = [
        row
        for row in validation_summary
        if int(row["cumulative_epochs"]) > int(base_epochs)
    ]
    trailing = 0
    for row in reversed(continuation):
        if bool(row["plateau_block"]):
            trailing += 1
        else:
            break
    required = int(rule["required_consecutive_final_plateau_blocks"])
    reached = trailing >= required
    onset_epoch = None
    if reached:
        onset_index = len(continuation) - trailing + required - 1
        onset_epoch = int(continuation[onset_index]["cumulative_epochs"])
    return {
        "plateau_reached": reached,
        "required_consecutive_final_plateau_blocks": required,
        "observed_consecutive_final_plateau_blocks": trailing,
        "plateau_onset_endpoint_epoch": onset_epoch,
        "reference_checkpoint_epoch": (
            int(validation_summary[-1]["cumulative_epochs"]) if reached else None
        ),
        "selection_metric": str(rule["metric"]),
        "selection_uses_validation_only": bool(rule["selection_uses_validation_only"]),
        "test_and_q_audit_locked_out_of_selection": bool(
            rule["test_and_q_audit_locked_out_of_selection"]
        ),
    }


def evaluate_checkpoint(
    outer_seed: int,
    cumulative_epochs: int,
    state: dict[str, torch.Tensor],
    data: dict[str, np.ndarray],
    dataset_config: dict,
    ridge_relative: float,
    audit_query_index: int,
    device: torch.device,
) -> list[dict[str, object]]:
    model = make_model(
        "fno",
        dataset_config["models"]["fno"],
        int(data["inputs"].shape[1]),
        residual=True,
    )
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    indices = test_variant_indices(data)
    predictions, inference_ms = collect_predictions(
        model,
        BOSTDataset(data, indices),
        device,
        int(dataset_config["training"]["batch_size"]),
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
    return rows


def collapse_model_seeds(
    sample_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in sample_rows:
        grouped[
            (
                int(row["cumulative_epochs"]),
                int(row["source_index"]),
                str(row["source_split"]),
            )
        ].append(row)
    output = []
    for (epoch, source_index, split), subset in sorted(grouped.items()):
        first = subset[0]
        output.append(
            {
                "cumulative_epochs": epoch,
                "source_index": source_index,
                "sample_seed": int(first["sample_seed"]),
                "source_split": split,
                "family_id": int(first["family_id"]),
                "noise_level": float(first["noise_level"]),
                "model_seed_count": len(subset),
                "field_rel_l2": float(
                    np.mean([float(row["field_rel_l2"]) for row in subset])
                ),
                "audit_reprojection_rel_l2": float(
                    np.mean(
                        [float(row["audit_reprojection_rel_l2"]) for row in subset]
                    )
                ),
            }
        )
    return output


def summarize_test(
    cluster_rows: list[dict[str, object]],
    experiment: dict,
) -> list[dict[str, object]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in cluster_rows:
        grouped[int(row["cumulative_epochs"])].append(row)
    base_epoch = int(experiment["base_epochs"])
    base = {
        int(row["source_index"]): float(row["field_rel_l2"])
        for row in grouped[base_epoch]
    }
    rng = np.random.default_rng(int(experiment["bootstrap_seed"]))
    output = []
    for epoch, subset in sorted(grouped.items()):
        augmented = []
        for row in subset:
            current = float(row["field_rel_l2"])
            reference = base[int(row["source_index"])]
            augmented.append(
                {
                    **row,
                    "field_superiority_vs_epoch24_pct": relative_improvement_pct(
                        reference, current
                    ),
                }
            )
        weights = domain_equal_weights(augmented)
        superiority = np.asarray(
            [float(row["field_superiority_vs_epoch24_pct"]) for row in augmented],
            dtype=np.float64,
        )
        ci_low, ci_high = stratified_bootstrap_interval(
            augmented,
            "field_superiority_vs_epoch24_pct",
            rng,
            int(experiment["bootstrap_replicates"]),
        )
        output.append(
            {
                "cumulative_epochs": epoch,
                "independent_field_count": len(augmented),
                "model_seed_count": int(augmented[0]["model_seed_count"]),
                "mean_field_rel_l2": float(
                    np.sum(
                        weights
                        * np.asarray(
                            [float(row["field_rel_l2"]) for row in augmented]
                        )
                    )
                ),
                "mean_clean_audit_rel_l2": float(
                    np.sum(
                        weights
                        * np.asarray(
                            [
                                float(row["audit_reprojection_rel_l2"])
                                for row in augmented
                            ]
                        )
                    )
                ),
                "mean_field_superiority_vs_epoch24_pct": float(
                    np.sum(weights * superiority)
                ),
                "field_superiority_ci95_low": ci_low,
                "field_superiority_ci95_high": ci_high,
                "every_domain_mean_nonnegative_vs_epoch24": all(
                    np.mean(
                        [
                            float(row["field_superiority_vs_epoch24_pct"])
                            for row in augmented
                            if str(row["source_split"]) == domain
                        ]
                    )
                    >= 0.0
                    for domain in sorted(
                        {str(row["source_split"]) for row in augmented}
                    )
                ),
            }
        )
    return output


def summarize_seed_test(
    sample_rows: list[dict[str, object]],
    base_epochs: int,
) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for row in sample_rows:
        grouped[(int(row["model_seed"]), int(row["cumulative_epochs"]))].append(row)
    base = {
        (seed, int(row["source_index"])): float(row["field_rel_l2"])
        for (seed, epoch), subset in grouped.items()
        if epoch == int(base_epochs)
        for row in subset
    }
    output = []
    for (seed, epoch), subset in sorted(grouped.items()):
        weights = domain_equal_weights(subset)
        superiority = np.asarray(
            [
                relative_improvement_pct(
                    base[(seed, int(row["source_index"]))],
                    float(row["field_rel_l2"]),
                )
                for row in subset
            ],
            dtype=np.float64,
        )
        output.append(
            {
                "model_seed": seed,
                "cumulative_epochs": epoch,
                "independent_field_count": len(subset),
                "domain_equal_mean_field_superiority_vs_epoch24_pct": float(
                    np.sum(weights * superiority)
                ),
            }
        )
    return output


def plot_results(
    checkpoint_rows: list[dict[str, object]],
    validation_summary: list[dict[str, object]],
    test_summary: list[dict[str, object]],
    plateau: dict[str, object],
    experiment: dict,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))
    seeds = sorted({int(row["model_seed"]) for row in checkpoint_rows})
    for seed in seeds:
        subset = sorted(
            [row for row in checkpoint_rows if int(row["model_seed"]) == seed],
            key=lambda row: int(row["cumulative_epochs"]),
        )
        axes[0].plot(
            [int(row["cumulative_epochs"]) for row in subset],
            [float(row["selected_validation_rel_l2"]) for row in subset],
            marker="o",
            linewidth=1.6,
            label=str(seed),
        )
    if plateau["plateau_reached"]:
        axes[0].axvline(
            int(plateau["plateau_onset_endpoint_epoch"]),
            color="#b04a3f",
            linestyle="--",
            linewidth=1.3,
            label="plateau gate",
        )
    axes[0].set_title("Validation-only checkpoint trajectory")
    axes[0].set_xlabel("cumulative epochs")
    axes[0].set_ylabel("validation relative L2")
    axes[0].grid(alpha=0.22)
    axes[0].legend(fontsize=7)

    continuation = [
        row
        for row in validation_summary
        if int(row["cumulative_epochs"]) > int(experiment["base_epochs"])
    ]
    epochs = [int(row["cumulative_epochs"]) for row in continuation]
    axes[1].plot(
        epochs,
        [float(row["mean_relative_validation_improvement_pct"]) for row in continuation],
        marker="o",
        color="#247b76",
        label="mean seed improvement",
    )
    axes[1].plot(
        epochs,
        [
            float(row["max_seed_relative_validation_improvement_pct"])
            for row in continuation
        ],
        marker="s",
        color="#d49b35",
        label="max seed improvement",
    )
    rule = experiment["validation_plateau_rule"]
    axes[1].axhline(
        float(rule["maximum_mean_relative_improvement_pct_per_block"]),
        color="#247b76",
        linestyle=":",
        linewidth=1.2,
    )
    axes[1].axhline(
        float(rule["maximum_seed_relative_improvement_pct_per_block"]),
        color="#d49b35",
        linestyle=":",
        linewidth=1.2,
    )
    axes[1].set_title("Validation gain per 12-epoch block")
    axes[1].set_xlabel("cumulative epochs")
    axes[1].set_ylabel("relative improvement (%)")
    axes[1].grid(alpha=0.22)
    axes[1].legend(fontsize=7)

    axes[2].plot(
        [int(row["cumulative_epochs"]) for row in test_summary],
        [float(row["mean_field_rel_l2"]) for row in test_summary],
        marker="o",
        color="#1f6f78",
        label="dev2 field L2",
    )
    axes[2].plot(
        [int(row["cumulative_epochs"]) for row in test_summary],
        [float(row["mean_clean_audit_rel_l2"]) for row in test_summary],
        marker="s",
        color="#b04a3f",
        label="clean Q_audit L2",
    )
    axes[2].set_title("Dev2 diagnostics (not used for selection)")
    axes[2].set_xlabel("cumulative epochs")
    axes[2].set_ylabel("domain-equal relative L2")
    axes[2].grid(alpha=0.22)
    axes[2].legend(fontsize=7)

    fig.suptitle(
        "T16 v3d K=6 FNO validation-plateau audit",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def analyze_and_write(
    experiment: dict,
    output_dir: Path,
    checkpoint_rows: list[dict[str, object]],
    sample_rows: list[dict[str, object]],
    history_rows: list[dict[str, object]],
    full_protocol: bool,
    environment: dict[str, object],
) -> None:
    validation_summary = summarize_validation(
        checkpoint_rows,
        experiment["validation_plateau_rule"],
        int(experiment["base_epochs"]),
    )
    plateau = select_validation_plateau(
        validation_summary,
        experiment["validation_plateau_rule"],
        int(experiment["base_epochs"]),
    )
    cluster_rows = collapse_model_seeds(sample_rows)
    test_summary = summarize_test(cluster_rows, experiment)
    seed_summary = summarize_seed_test(sample_rows, int(experiment["base_epochs"]))
    status = (
        "V3D_FNO_VALIDATION_PLATEAU_REACHED"
        if full_protocol and bool(plateau["plateau_reached"])
        else (
            "V3D_FNO_VALIDATION_PLATEAU_NOT_REACHED_BY_MAX_EPOCH"
            if full_protocol
            else "V3D_FNO_SATURATION_SMOKE_ONLY"
        )
    )
    plot_results(
        checkpoint_rows,
        validation_summary,
        test_summary,
        plateau,
        experiment,
        output_dir / "t16_v3d_fno_saturation.png",
    )
    write_csv(output_dir / "v3d_fno_history.csv", history_rows)
    write_csv(output_dir / "v3d_fno_checkpoints.csv", checkpoint_rows)
    write_csv(output_dir / "v3d_fno_validation_summary.csv", validation_summary)
    write_csv(output_dir / "v3d_fno_samples.csv", sample_rows)
    write_csv(output_dir / "v3d_fno_clusters.csv", cluster_rows)
    write_csv(output_dir / "v3d_fno_test_summary.csv", test_summary)
    write_csv(output_dir / "v3d_fno_seed_summary.csv", seed_summary)
    dashboard = {
        "experiment": experiment["name"],
        "scientific_status": status,
        "full_protocol": full_protocol,
        "development_fields_only": True,
        "blind_final_opened": False,
        "total_budget": int(experiment["total_budget"]),
        "independent_test_field_count": 128,
        "model_seed_count": len({int(row["model_seed"]) for row in checkpoint_rows}),
        "base_epochs": int(experiment["base_epochs"]),
        "continuation_block_epochs": int(experiment["continuation_block_epochs"]),
        "max_total_epochs": max(
            int(row["cumulative_epochs"]) for row in checkpoint_rows
        ),
        "validation_plateau_rule": experiment["validation_plateau_rule"],
        "plateau_decision": plateau,
        "validation_summary": validation_summary,
        "test_summary": test_summary,
        "seed_summary": seed_summary,
        "checkpoint_rows": checkpoint_rows,
    }
    (output_dir / "v3d_fno_saturation_dashboard.json").write_text(
        json.dumps(dashboard, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report = {
        "status": "completed" if full_protocol else "smoke_only",
        "scientific_status": status,
        "environment": environment,
        "protocol": {
            "same_k6_inputs_and_ridge_anchor_as_v3c": True,
            "base_schedule_matches_v3c": True,
            "continuation_uses_locked_12_epoch_blocks": True,
            "optimizer_and_scheduler_restarted_for_each_continuation_block": True,
            "previous_checkpoint_retained_when_block_does_not_improve_validation": True,
            "plateau_selection_uses_validation_only": True,
            "validation_metric_aggregated_per_sample": True,
            "test_or_q_audit_used_for_plateau_selection": False,
            "test_unit": "128 three-dimensional dev2 fields after collapsing model seeds",
            "blind_final_opened": False,
        },
        "plateau_decision": plateau,
        "validation_summary": validation_summary,
        "test_summary": test_summary,
        "claims_boundary": [
            "Cumulative epoch labels join validation-selected checkpoints across independently restarted 12-epoch AdamW/cosine blocks; optimizer state is not carried across blocks.",
            "Plateau means the preregistered validation rule was met under this optimizer and block schedule; it is not a global optimization proof.",
            "Dev2 field and clean Q_audit curves are reused development diagnostics computed after this run's validation-only plateau decision; the fields were already inspected in earlier project stages.",
            "The inspected linear synthetic dev2 fields cannot establish real-BOST superiority.",
            "No blind-final seed, data, checkpoint or restricted material is stored or opened.",
        ],
    }
    (output_dir / "v3d_fno_saturation_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_output_checksums(output_dir)
    print(json.dumps({"scientific_status": status, "plateau": plateau}, indent=2))
    print(f"results: {output_dir}")


def reanalyze_existing(experiment: dict, output_dir: Path) -> None:
    checkpoint_rows = read_csv(output_dir / "v3d_fno_checkpoints.csv")
    sample_rows = read_csv(output_dir / "v3d_fno_samples.csv")
    history_rows = read_csv(output_dir / "v3d_fno_history.csv")
    report_path = output_dir / "v3d_fno_saturation_report.json"
    dashboard_path = output_dir / "v3d_fno_saturation_dashboard.json"
    existing_report = read_json(report_path) if report_path.exists() else {}
    existing_dashboard = read_json(dashboard_path) if dashboard_path.exists() else {}
    analyze_and_write(
        experiment,
        output_dir,
        checkpoint_rows,
        sample_rows,
        history_rows,
        full_protocol=bool(existing_dashboard.get("full_protocol", False)),
        environment=existing_report.get("environment", {"mode": "analysis_only"}),
    )


def main() -> None:
    args = parse_args()
    experiment = read_json(args.config)
    dataset_config = read_json(CONFIG_ROOT / experiment["dataset_config"])
    device_name = args.device or str(dataset_config["training"]["device"])
    device = choose_device(device_name)
    seeds = [int(value) for value in experiment["training_seeds"]]
    if args.seed_limit is not None:
        seeds = seeds[: int(args.seed_limit)]
    max_total_epochs = int(args.max_total_epochs or experiment["max_total_epochs"])
    base_epochs = int(experiment["base_epochs"])
    block_epochs = int(experiment["continuation_block_epochs"])
    if max_total_epochs < base_epochs or (max_total_epochs - base_epochs) % block_epochs:
        raise ValueError("max total epochs must equal base + N * continuation block")
    output_dir = args.output_dir or ROOT / "results" / experiment["output_dir"]
    work_dir = args.work_dir or ROOT / "results" / experiment["work_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    if args.analysis_only:
        reanalyze_existing(experiment, output_dir)
        return

    dataset_path = ROOT / "results" / "t16_v3c_dev2_dataset.npz"
    generate_dataset(dataset_config, dataset_path, force=bool(args.force_data))
    base_data = load_npz(dataset_path)
    budget = int(experiment["total_budget"])
    direct = prepare_direct_operator_data(
        base_data,
        [budget],
        int(experiment["fixed_query_index"]),
        int(experiment["audit_query_index"]),
    )
    selected_ridge, champions, tuning_rows = tune_classical_baselines(
        direct,
        [budget],
        [float(value) for value in experiment["ridge_relative_grid"]],
    )
    if champions[budget] != "ridge":
        raise RuntimeError("v3d saturation audit requires validation-locked ridge")
    data = append_ray_view_channels(replace_lift_with_ridge(direct, selected_ridge))
    write_csv(output_dir / "v3d_fno_baseline_tuning.csv", tuning_rows)

    checkpoint_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    checkpoint_states: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
    for outer_seed in seeds:
        base_seed = outer_seed + 101
        set_seed(base_seed)
        model = make_model(
            "fno",
            dataset_config["models"]["fno"],
            int(data["inputs"].shape[1]),
            residual=True,
        )
        base_config = training_config(
            dataset_config,
            base_seed,
            base_epochs,
            device_name,
            fixed_epochs=False,
        )
        base_record = train_model(
            "base_fno",
            model,
            data,
            base_config,
            work_dir / str(outer_seed) / "epoch_024",
        )
        if int(base_record["epochs_ran"]) != base_epochs:
            raise RuntimeError("base FNO stopped before the locked 24-epoch horizon")
        model = base_record["model"]
        selected_val = evaluate_validation(model, data, dataset_config, device)
        cumulative_seconds = float(base_record["train_seconds"])
        state = cpu_state(model)
        checkpoint_states[(outer_seed, base_epochs)] = state
        checkpoint_rows.append(
            {
                "model_seed": outer_seed,
                "cumulative_epochs": base_epochs,
                "phase": "base",
                "block_index": 0,
                "candidate_best_val_rel_l2": float(base_record["best_val_rel_l2"]),
                "selected_validation_rel_l2": selected_val,
                "relative_validation_improvement_pct": 0.0,
                "retained_previous_checkpoint": False,
                "epochs_ran_in_block": int(base_record["epochs_ran"]),
                "best_epoch_in_block": int(base_record["best_epoch"]),
                "train_seconds_block": float(base_record["train_seconds"]),
                "cumulative_train_seconds": cumulative_seconds,
                "checkpoint_sha256": state_sha256(state),
            }
        )
        history_rows.extend(
            {
                "model_seed": outer_seed,
                "phase": "base",
                "block_index": 0,
                "cumulative_epoch": int(row["epoch"]),
                **{key: value for key, value in row.items() if key != "epoch"},
            }
            for row in base_record["history"]
        )

        block_index = 0
        for cumulative_epoch in range(
            base_epochs + block_epochs,
            max_total_epochs + 1,
            block_epochs,
        ):
            block_index += 1
            previous_state = cpu_state(model)
            previous_val = evaluate_validation(model, data, dataset_config, device)
            continuation_model = make_model(
                "fno",
                dataset_config["models"]["fno"],
                int(data["inputs"].shape[1]),
                residual=True,
            )
            continuation_model.load_state_dict(previous_state, strict=True)
            continuation_seed = outer_seed + 10_000 + block_index
            set_seed(continuation_seed)
            block_config = training_config(
                dataset_config,
                continuation_seed,
                block_epochs,
                device_name,
                learning_rate=float(experiment["continuation_learning_rate"]),
                fixed_epochs=True,
            )
            record = train_model(
                f"continued_fno_{cumulative_epoch}",
                continuation_model,
                data,
                block_config,
                work_dir / str(outer_seed) / f"epoch_{cumulative_epoch:03d}",
            )
            if int(record["epochs_ran"]) != block_epochs:
                raise RuntimeError("continuation block did not receive locked epochs")
            candidate_val = evaluate_validation(
                record["model"], data, dataset_config, device
            )
            required_improvement = float(
                experiment["checkpoint_acceptance_min_abs_improvement"]
            )
            retained = candidate_val >= previous_val - required_improvement
            if retained:
                record["model"].load_state_dict(previous_state, strict=True)
                selected_val = previous_val
                best_epoch = 0
            else:
                selected_val = candidate_val
                best_epoch = int(record["best_epoch"])
            model = record["model"]
            cumulative_seconds += float(record["train_seconds"])
            state = cpu_state(model)
            checkpoint_states[(outer_seed, cumulative_epoch)] = state
            checkpoint_rows.append(
                {
                    "model_seed": outer_seed,
                    "cumulative_epochs": cumulative_epoch,
                    "phase": "continuation",
                    "block_index": block_index,
                    "candidate_best_val_rel_l2": candidate_val,
                    "selected_validation_rel_l2": selected_val,
                    "relative_validation_improvement_pct": relative_improvement_pct(
                        previous_val, selected_val
                    ),
                    "retained_previous_checkpoint": retained,
                    "epochs_ran_in_block": int(record["epochs_ran"]),
                    "best_epoch_in_block": best_epoch,
                    "train_seconds_block": float(record["train_seconds"]),
                    "cumulative_train_seconds": cumulative_seconds,
                    "checkpoint_sha256": state_sha256(state),
                }
            )
            history_rows.extend(
                {
                    "model_seed": outer_seed,
                    "phase": "continuation",
                    "block_index": block_index,
                    "cumulative_epoch": cumulative_epoch
                    - block_epochs
                    + int(row["epoch"]),
                    **{key: value for key, value in row.items() if key != "epoch"},
                }
                for row in record["history"]
            )
        print(f"seed={outer_seed}: trained through epoch {max_total_epochs}", flush=True)

    validation_summary = summarize_validation(
        checkpoint_rows,
        experiment["validation_plateau_rule"],
        base_epochs,
    )
    plateau = select_validation_plateau(
        validation_summary,
        experiment["validation_plateau_rule"],
        base_epochs,
    )
    print(json.dumps({"validation_only_plateau": plateau}, indent=2), flush=True)

    sample_rows: list[dict[str, object]] = []
    for (outer_seed, cumulative_epoch), state in sorted(checkpoint_states.items()):
        sample_rows.extend(
            evaluate_checkpoint(
                outer_seed,
                cumulative_epoch,
                state,
                data,
                dataset_config,
                selected_ridge[budget],
                int(experiment["audit_query_index"]),
                device,
            )
        )
    full_protocol = (
        len(seeds) == len(experiment["training_seeds"])
        and max_total_epochs == int(experiment["max_total_epochs"])
    )
    analyze_and_write(
        experiment,
        output_dir,
        checkpoint_rows,
        sample_rows,
        history_rows,
        full_protocol,
        environment={
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "requested_device": device_name,
            "runtime_device": str(device),
        },
    )


if __name__ == "__main__":
    main()
