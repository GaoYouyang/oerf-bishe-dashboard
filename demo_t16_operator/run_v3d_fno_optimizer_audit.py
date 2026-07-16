#!/usr/bin/env python3
"""Compare three validation-only FNO continuation protocols through 240 epochs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional
from torch.utils.data import DataLoader

try:
    from .data import BOSTDataset, generate_dataset, load_npz, split_indices
    from .direct_operator_data import prepare_direct_operator_data, replace_lift_with_ridge
    from .models import make_model
    from .own_algorithm_data import append_ray_view_channels
    from .run_direct_operator_pilot import (
        domain_equal_weights,
        stratified_bootstrap_interval,
        tune_classical_baselines,
        write_checksums,
    )
    from .run_v3c_k6_dev2_pilot import state_sha256, training_config
    from .run_v3d_fno_saturation_audit import (
        cpu_state,
        evaluate_checkpoint,
        read_csv,
        read_json,
        relative_improvement_pct,
        select_validation_plateau,
        summarize_validation,
        write_csv,
    )
    from .train_eval import (
        batch_relative_l2,
        choose_device,
        gradient_mse,
        masked_relative_projection_loss,
        project_torch,
        sample_weighted_mean,
        set_seed,
        synchronize,
        train_model,
    )
except ImportError:
    from data import BOSTDataset, generate_dataset, load_npz, split_indices
    from direct_operator_data import prepare_direct_operator_data, replace_lift_with_ridge
    from models import make_model
    from own_algorithm_data import append_ray_view_channels
    from run_direct_operator_pilot import (
        domain_equal_weights,
        stratified_bootstrap_interval,
        tune_classical_baselines,
        write_checksums,
    )
    from run_v3c_k6_dev2_pilot import state_sha256, training_config
    from run_v3d_fno_saturation_audit import (
        cpu_state,
        evaluate_checkpoint,
        read_csv,
        read_json,
        relative_improvement_pct,
        select_validation_plateau,
        summarize_validation,
        write_csv,
    )
    from train_eval import (
        batch_relative_l2,
        choose_device,
        gradient_mse,
        masked_relative_projection_loss,
        project_torch,
        sample_weighted_mean,
        set_seed,
        synchronize,
        train_model,
    )


ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = ROOT / "configs"
DEFAULT_CONFIG = CONFIG_ROOT / "v3d_fno_optimizer_audit.json"
CHECKSUM_FILES = [
    "v3d_optimizer_baseline_tuning.csv",
    "v3d_optimizer_history.csv",
    "v3d_optimizer_checkpoints.csv",
    "v3d_optimizer_validation_summary.csv",
    "v3d_optimizer_strategy_summary.csv",
    "v3d_optimizer_samples.csv",
    "v3d_optimizer_clusters.csv",
    "v3d_optimizer_test_summary.csv",
    "v3d_optimizer_pairwise_summary.csv",
    "v3d_optimizer_seed_summary.csv",
    "v3d_optimizer_dashboard.json",
    "v3d_optimizer_report.json",
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


def strategy_ids(experiment: dict) -> list[str]:
    ids = [str(row["id"]) for row in experiment["strategies"]]
    if len(ids) != len(set(ids)):
        raise ValueError("strategy ids must be unique")
    return ids


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def batch_order_contract_sha256(
    sample_count: int,
    batch_size: int,
    epochs: int,
    seed: int,
    sample_indices: np.ndarray | list[int] | None = None,
) -> str:
    indices = np.asarray(
        np.arange(int(sample_count), dtype=np.int64)
        if sample_indices is None
        else sample_indices,
        dtype=np.int64,
    )
    if indices.shape != (int(sample_count),) or len(np.unique(indices)) != len(indices):
        raise ValueError("sample indices must be a unique vector matching sample_count")
    generator = torch.Generator().manual_seed(int(seed))
    digest = hashlib.sha256()
    digest.update(
        f"samples={sample_count};batch={batch_size};epochs={epochs};seed={seed}".encode(
            "ascii"
        )
    )
    digest.update(indices.astype("<i8", copy=False).tobytes())
    for _ in range(int(epochs)):
        order = torch.randperm(int(sample_count), generator=generator, dtype=torch.int64)
        digest.update(indices[order.numpy()].astype("<i8", copy=False).tobytes())
    return digest.hexdigest()


def build_provenance(
    experiment_config_path: Path,
    experiment: dict,
    dataset_path: Path,
) -> dict[str, object]:
    return {
        "experiment_config_sha256": sha256_file(experiment_config_path),
        "dataset_config_sha256": sha256_file(
            CONFIG_ROOT / str(experiment["dataset_config"])
        ),
        "training_script_sha256": sha256_file(Path(__file__).resolve()),
        "train_eval_script_sha256": sha256_file(ROOT / "train_eval.py"),
        "data_script_sha256": sha256_file(ROOT / "data.py"),
        "dataset_npz_sha256": sha256_file(dataset_path),
        "dataset_npz_public": False,
        "checkpoint_weights_public": False,
    }


def train_continuation_block(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    data: dict[str, np.ndarray],
    dataset_config: dict,
    device: torch.device,
    block_seed: int,
    block_epochs: int,
) -> dict[str, object]:
    indices = split_indices(data)
    generator = torch.Generator().manual_seed(int(block_seed))
    train_loader = DataLoader(
        BOSTDataset(data, indices["train"]),
        batch_size=int(dataset_config["training"]["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        BOSTDataset(data, indices["val"]),
        batch_size=int(dataset_config["training"]["batch_size"]),
    )
    model = model.to(device)
    operator = torch.from_numpy(data["forward_matrix"]).to(device)
    training = dataset_config["training"]
    history: list[dict[str, object]] = []
    best_validation = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    synchronize(device)
    start = time.perf_counter()
    for epoch in range(1, int(block_epochs) + 1):
        model.train()
        running = {
            "total": 0.0,
            "field": 0.0,
            "gradient": 0.0,
            "projection": 0.0,
            "boundary": 0.0,
        }
        sample_count = 0
        for batch in train_loader:
            inputs = batch["x"].to(device)
            target = batch["field"].to(device)
            observed = batch["observation"].to(device)
            view_mask = batch["view_mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)
            projected = project_torch(prediction, operator)
            field_loss = functional.mse_loss(prediction, target)
            gradient_loss = gradient_mse(prediction, target)
            projection_loss = masked_relative_projection_loss(
                projected, observed, view_mask
            )
            outside = (inputs[:, 1:2] < 0.02).to(prediction.dtype)
            boundary_loss = torch.mean((prediction * outside) ** 2)
            total = (
                field_loss
                + float(training["lambda_gradient"]) * gradient_loss
                + float(training["lambda_reprojection"]) * projection_loss
                + float(training["lambda_boundary"]) * boundary_loss
            )
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            batch_size = int(inputs.shape[0])
            sample_count += batch_size
            for key, value in (
                ("total", total),
                ("field", field_loss),
                ("gradient", gradient_loss),
                ("projection", projection_loss),
                ("boundary", boundary_loss),
            ):
                running[key] += float(value.detach().cpu()) * batch_size

        model.eval()
        validation_values = []
        validation_batch_sizes = []
        with torch.no_grad():
            for batch in val_loader:
                prediction = model(batch["x"].to(device))
                validation_values.append(
                    float(
                        batch_relative_l2(
                            prediction, batch["field"].to(device)
                        ).cpu()
                    )
                )
                validation_batch_sizes.append(int(batch["x"].shape[0]))
        validation = sample_weighted_mean(
            validation_values, validation_batch_sizes
        )
        history.append(
            {
                "epoch_in_block": epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "val_rel_l2": validation,
                **{
                    f"train_{key}": value / max(sample_count, 1)
                    for key, value in running.items()
                },
            }
        )
        if validation < best_validation - 1e-5:
            best_validation = validation
            best_epoch = epoch
            best_state = cpu_state(model)
        scheduler.step()
    synchronize(device)
    elapsed = time.perf_counter() - start
    if best_state is None:
        raise RuntimeError("continuation block produced no validation checkpoint")
    return {
        "history": history,
        "best_validation_rel_l2": best_validation,
        "best_epoch_in_block": best_epoch,
        "best_state": best_state,
        "endpoint_validation_rel_l2": float(history[-1]["val_rel_l2"]),
        "endpoint_state": cpu_state(model),
        "train_seconds": elapsed,
        "ending_learning_rate": float(optimizer.param_groups[0]["lr"]),
    }


def summarize_strategies(
    checkpoint_rows: list[dict[str, object]], experiment: dict
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    validation_rows: list[dict[str, object]] = []
    decisions: dict[str, dict[str, object]] = {}
    for strategy in strategy_ids(experiment):
        subset = [row for row in checkpoint_rows if row["strategy"] == strategy]
        summary = summarize_validation(
            subset,
            experiment["validation_plateau_rule"],
            int(experiment["base_epochs"]),
        )
        raw_by_epoch: dict[int, list[float]] = defaultdict(list)
        selected_epoch_by_endpoint: dict[int, list[int]] = defaultdict(list)
        for row in subset:
            endpoint = int(row["cumulative_epochs"])
            raw_by_epoch[endpoint].append(float(row["endpoint_validation_rel_l2"]))
            selected_epoch_by_endpoint[endpoint].append(
                int(row["selected_checkpoint_epoch"])
            )
        for row in summary:
            endpoint = int(row["cumulative_epochs"])
            row["strategy"] = strategy
            row["mean_endpoint_validation_rel_l2"] = float(
                np.mean(raw_by_epoch[endpoint])
            )
            row["mean_selected_checkpoint_epoch"] = float(
                np.mean(selected_epoch_by_endpoint[endpoint])
            )
            validation_rows.append(row)
        decisions[strategy] = select_validation_plateau(
            summary,
            experiment["validation_plateau_rule"],
            int(experiment["base_epochs"]),
        )
    return validation_rows, decisions


def choose_validation_champion(
    validation_rows: list[dict[str, object]], max_total_epochs: int
) -> str:
    final_rows = [
        row
        for row in validation_rows
        if int(row["cumulative_epochs"]) == int(max_total_epochs)
    ]
    if not final_rows:
        raise ValueError("no final validation rows")
    return str(min(final_rows, key=lambda row: float(row["mean_validation_rel_l2"]))["strategy"])


def build_strategy_summary(
    validation_rows: list[dict[str, object]],
    decisions: dict[str, dict[str, object]],
    champion: str,
    experiment: dict,
) -> list[dict[str, object]]:
    labels = {str(row["id"]): str(row["label"]) for row in experiment["strategies"]}
    output = []
    max_epoch = int(experiment["max_total_epochs"])
    for strategy in strategy_ids(experiment):
        final = next(
            row
            for row in validation_rows
            if row["strategy"] == strategy
            and int(row["cumulative_epochs"]) == max_epoch
        )
        decision = decisions[strategy]
        output.append(
            {
                "strategy": strategy,
                "label": labels[strategy],
                "validation_champion": strategy == champion,
                "plateau_reached": bool(decision["plateau_reached"]),
                "plateau_onset_endpoint_epoch": decision[
                    "plateau_onset_endpoint_epoch"
                ],
                "observed_consecutive_final_plateau_blocks": int(
                    decision["observed_consecutive_final_plateau_blocks"]
                ),
                "final_mean_selected_validation_rel_l2": float(
                    final["mean_validation_rel_l2"]
                ),
                "final_mean_endpoint_validation_rel_l2": float(
                    final["mean_endpoint_validation_rel_l2"]
                ),
                "last_block_mean_improvement_pct": float(
                    final["mean_relative_validation_improvement_pct"]
                ),
                "last_block_max_seed_improvement_pct": float(
                    final["max_seed_relative_validation_improvement_pct"]
                ),
                "mean_cumulative_train_seconds": float(
                    final["mean_cumulative_train_seconds"]
                ),
            }
        )
    return output


def collapse_test_rows(
    sample_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in sample_rows:
        grouped[
            (
                str(row["strategy"]),
                int(row["source_index"]),
                str(row["source_split"]),
            )
        ].append(row)
    output = []
    for (strategy, source_index, split), subset in sorted(grouped.items()):
        first = subset[0]
        output.append(
            {
                "strategy": strategy,
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


def summarize_test_rows(
    cluster_rows: list[dict[str, object]], experiment: dict
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in cluster_rows:
        grouped[str(row["strategy"])].append(row)
    base = {
        int(row["source_index"]): float(row["field_rel_l2"])
        for row in grouped["base_24"]
    }
    rng = np.random.default_rng(int(experiment["bootstrap_seed"]))
    output = []
    for strategy, subset in sorted(grouped.items()):
        augmented = []
        for row in subset:
            current = float(row["field_rel_l2"])
            reference = base[int(row["source_index"])]
            augmented.append(
                {
                    **row,
                    "field_superiority_vs_base24_pct": relative_improvement_pct(
                        reference, current
                    ),
                }
            )
        weights = domain_equal_weights(augmented)
        superiority = np.asarray(
            [float(row["field_superiority_vs_base24_pct"]) for row in augmented],
            dtype=np.float64,
        )
        ci_low, ci_high = stratified_bootstrap_interval(
            augmented,
            "field_superiority_vs_base24_pct",
            rng,
            int(experiment["bootstrap_replicates"]),
        )
        output.append(
            {
                "strategy": strategy,
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
                "mean_field_superiority_vs_base24_pct": float(
                    np.sum(weights * superiority)
                ),
                "field_superiority_ci95_low": ci_low,
                "field_superiority_ci95_high": ci_high,
                "every_domain_mean_nonnegative_vs_base24": all(
                    np.mean(
                        [
                            float(row["field_superiority_vs_base24_pct"])
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


def summarize_seed_rows(
    sample_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in sample_rows:
        grouped[(str(row["strategy"]), int(row["model_seed"]))].append(row)
    base = {
        (seed, int(row["source_index"])): float(row["field_rel_l2"])
        for (strategy, seed), subset in grouped.items()
        if strategy == "base_24"
        for row in subset
    }
    output = []
    for (strategy, seed), subset in sorted(grouped.items()):
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
                "strategy": strategy,
                "model_seed": seed,
                "independent_field_count": len(subset),
                "domain_equal_mean_field_superiority_vs_base24_pct": float(
                    np.sum(weights * superiority)
                ),
            }
        )
    return output


def build_pairwise_summary(
    cluster_rows: list[dict[str, object]],
    champion: str,
    experiment: dict,
) -> list[dict[str, object]]:
    grouped: dict[str, dict[int, dict[str, object]]] = defaultdict(dict)
    for row in cluster_rows:
        grouped[str(row["strategy"])][int(row["source_index"])] = row
    rng = np.random.default_rng(int(experiment["bootstrap_seed"]) + 1)
    output = []
    for comparator in strategy_ids(experiment):
        if comparator == champion:
            continue
        augmented = []
        for source_index, champion_row in sorted(grouped[champion].items()):
            comparator_row = grouped[comparator][source_index]
            augmented.append(
                {
                    **champion_row,
                    "field_superiority_pct": relative_improvement_pct(
                        float(comparator_row["field_rel_l2"]),
                        float(champion_row["field_rel_l2"]),
                    ),
                    "audit_superiority_pct": relative_improvement_pct(
                        float(comparator_row["audit_reprojection_rel_l2"]),
                        float(champion_row["audit_reprojection_rel_l2"]),
                    ),
                }
            )
        weights = domain_equal_weights(augmented)
        field_values = np.asarray(
            [float(row["field_superiority_pct"]) for row in augmented],
            dtype=np.float64,
        )
        audit_values = np.asarray(
            [float(row["audit_superiority_pct"]) for row in augmented],
            dtype=np.float64,
        )
        ci_low, ci_high = stratified_bootstrap_interval(
            augmented,
            "field_superiority_pct",
            rng,
            int(experiment["bootstrap_replicates"]),
        )
        domain_means = {
            domain: float(
                np.mean(
                    [
                        float(row["field_superiority_pct"])
                        for row in augmented
                        if str(row["source_split"]) == domain
                    ]
                )
            )
            for domain in sorted({str(row["source_split"]) for row in augmented})
        }
        worst_domain = min(domain_means, key=domain_means.get)
        output.append(
            {
                "candidate": champion,
                "comparator": comparator,
                "independent_field_count": len(augmented),
                "model_seed_count": int(augmented[0]["model_seed_count"]),
                "mean_field_superiority_pct": float(
                    np.sum(weights * field_values)
                ),
                "field_superiority_ci95_low": ci_low,
                "field_superiority_ci95_high": ci_high,
                "p10_field_superiority_pct": float(np.quantile(field_values, 0.10)),
                "field_harm_rate_gt_1pct": float(np.mean(field_values < -1.0)),
                "mean_audit_superiority_pct": float(
                    np.sum(weights * audit_values)
                ),
                "worst_domain": worst_domain,
                "worst_domain_mean_field_superiority_pct": domain_means[worst_domain],
                "every_domain_mean_field_nonnegative": all(
                    value >= 0.0 for value in domain_means.values()
                ),
            }
        )
    return output


def plot_results(
    validation_rows: list[dict[str, object]],
    strategy_summary: list[dict[str, object]],
    test_summary: list[dict[str, object]],
    experiment: dict,
    output_path: Path,
) -> None:
    colors = {
        "restart_adam_restart_cosine": "#b04a3f",
        "carry_adam_restart_cosine": "#d49b35",
        "carry_adam_long_cosine": "#247b76",
    }
    labels = {str(row["id"]): str(row["label"]) for row in experiment["strategies"]}
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6))
    for strategy in strategy_ids(experiment):
        subset = sorted(
            [row for row in validation_rows if row["strategy"] == strategy],
            key=lambda row: int(row["cumulative_epochs"]),
        )
        axes[0].plot(
            [int(row["cumulative_epochs"]) for row in subset],
            [float(row["mean_validation_rel_l2"]) for row in subset],
            marker="o",
            markersize=3,
            linewidth=1.6,
            color=colors[strategy],
            label=labels[strategy],
        )
        axes[1].plot(
            [int(row["cumulative_epochs"]) for row in subset[1:]],
            [float(row["mean_relative_validation_improvement_pct"]) for row in subset[1:]],
            marker="o",
            markersize=3,
            linewidth=1.4,
            color=colors[strategy],
            label=labels[strategy],
        )
    axes[0].set_title("Prefix-best validation trajectory")
    axes[0].set_xlabel("attempted cumulative epochs")
    axes[0].set_ylabel("mean validation relative L2")
    axes[0].grid(alpha=0.22)
    axes[0].legend(fontsize=7)
    axes[1].axhline(
        float(
            experiment["validation_plateau_rule"][
                "maximum_mean_relative_improvement_pct_per_block"
            ]
        ),
        color="#303a3c",
        linestyle=":",
        linewidth=1.2,
        label="mean plateau threshold",
    )
    axes[1].set_title("Validation gain per 12-epoch block")
    axes[1].set_xlabel("attempted cumulative epochs")
    axes[1].set_ylabel("relative improvement (%)")
    axes[1].grid(alpha=0.22)
    axes[1].legend(fontsize=7)

    test_lookup = {str(row["strategy"]): row for row in test_summary}
    strategies = strategy_ids(experiment)
    x = np.arange(len(strategies))
    field = [float(test_lookup[name]["mean_field_rel_l2"]) for name in strategies]
    audit = [float(test_lookup[name]["mean_clean_audit_rel_l2"]) for name in strategies]
    width = 0.34
    axes[2].bar(x - width / 2, field, width, label="dev2 field L2", color="#247b76")
    axes[2].bar(x + width / 2, audit, width, label="clean Q_audit L2", color="#d49b35")
    axes[2].set_xticks(x, ["restart\nboth", "carry Adam\nrestart cosine", "carry Adam\nlong cosine"], fontsize=7)
    axes[2].set_title("Post-selection dev2 diagnostics")
    axes[2].set_ylabel("domain-equal relative L2")
    axes[2].grid(axis="y", alpha=0.22)
    axes[2].legend(fontsize=7)
    fig.suptitle("T16 v3d FNO optimizer-protocol audit", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_output_checksums(output_dir: Path) -> None:
    write_checksums(output_dir, CHECKSUM_FILES)
    source = output_dir / "direct_operator_checksums.sha256"
    target = output_dir / "v3d_optimizer_checksums.sha256"
    target.write_text(source.read_text(encoding="ascii"), encoding="ascii")
    source.unlink()


def analyze_and_write(
    experiment: dict,
    output_dir: Path,
    checkpoint_rows: list[dict[str, object]],
    history_rows: list[dict[str, object]],
    sample_rows: list[dict[str, object]],
    full_protocol: bool,
    environment: dict[str, object],
    provenance: dict[str, object],
) -> None:
    validation_rows, decisions = summarize_strategies(checkpoint_rows, experiment)
    champion = choose_validation_champion(
        validation_rows, int(experiment["max_total_epochs"])
    )
    strategy_summary = build_strategy_summary(
        validation_rows, decisions, champion, experiment
    )
    cluster_rows = collapse_test_rows(sample_rows)
    test_summary = summarize_test_rows(cluster_rows, experiment)
    pairwise_summary = build_pairwise_summary(cluster_rows, champion, experiment)
    seed_summary = summarize_seed_rows(sample_rows)
    champion_plateau = bool(decisions[champion]["plateau_reached"])
    status = (
        "V3D_FNO_VALIDATION_CHAMPION_PLATEAU_REACHED"
        if full_protocol and champion_plateau
        else (
            "V3D_FNO_VALIDATION_CHAMPION_NOT_PLATEAUED_BY_MAX_EPOCH"
            if full_protocol
            else "V3D_FNO_OPTIMIZER_AUDIT_SMOKE_ONLY"
        )
    )
    plot_results(
        validation_rows,
        strategy_summary,
        test_summary,
        experiment,
        output_dir / "t16_v3d_fno_optimizer_audit.png",
    )
    write_csv(output_dir / "v3d_optimizer_history.csv", history_rows)
    write_csv(output_dir / "v3d_optimizer_checkpoints.csv", checkpoint_rows)
    write_csv(output_dir / "v3d_optimizer_validation_summary.csv", validation_rows)
    write_csv(output_dir / "v3d_optimizer_strategy_summary.csv", strategy_summary)
    write_csv(output_dir / "v3d_optimizer_samples.csv", sample_rows)
    write_csv(output_dir / "v3d_optimizer_clusters.csv", cluster_rows)
    write_csv(output_dir / "v3d_optimizer_test_summary.csv", test_summary)
    write_csv(output_dir / "v3d_optimizer_pairwise_summary.csv", pairwise_summary)
    write_csv(output_dir / "v3d_optimizer_seed_summary.csv", seed_summary)
    dashboard = {
        "experiment": experiment["name"],
        "scientific_status": status,
        "full_protocol": full_protocol,
        "development_fields_only": True,
        "blind_final_opened": False,
        "geometry_gate_resolved": False,
        "base_epochs": int(experiment["base_epochs"]),
        "continuation_block_epochs": int(experiment["continuation_block_epochs"]),
        "max_total_epochs": int(experiment["max_total_epochs"]),
        "model_seed_count": len({int(row["model_seed"]) for row in checkpoint_rows}),
        "independent_test_field_count": 128,
        "strategy_ids": strategy_ids(experiment),
        "validation_champion": champion,
        "champion_plateau_decision": decisions[champion],
        "validation_plateau_rule": experiment["validation_plateau_rule"],
        "strategy_summary": strategy_summary,
        "validation_summary": validation_rows,
        "test_summary": test_summary,
        "pairwise_summary": pairwise_summary,
        "seed_summary": seed_summary,
        "provenance": provenance,
    }
    (output_dir / "v3d_optimizer_dashboard.json").write_text(
        json.dumps(dashboard, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report = {
        "status": "completed" if full_protocol else "smoke_only",
        "scientific_status": status,
        "environment": environment,
        "provenance": provenance,
        "protocol": {
            "same_base_checkpoint_per_seed_across_strategies": True,
            "same_block_batch_seed_across_strategies": True,
            "batch_order_contract_hash_recorded_per_block": True,
            "batch_order_contract_binds_actual_train_indices": True,
            "validation_metric_aggregated_per_sample": True,
            "base_optimizer_state_carried_into_continuation": False,
            "carry_adam_means_across_continuation_blocks_only": True,
            "training_state_continues_from_raw_block_endpoint": True,
            "reported_checkpoint_is_validation_prefix_best": True,
            "dev2_computed_after_all_validation_decisions": True,
            "test_or_q_audit_used_for_strategy_or_checkpoint_selection": False,
            "blind_final_opened": False,
            "test_unit": "128 three-dimensional dev2 fields after collapsing model seeds",
        },
        "validation_champion": champion,
        "champion_plateau_decision": decisions[champion],
        "strategy_summary": strategy_summary,
        "test_summary": test_summary,
        "pairwise_summary": pairwise_summary,
        "claims_boundary": [
            "Carry Adam means carrying moments across continuation blocks after a newly initialized continuation optimizer; base-training optimizer moments are not restored.",
            "The current budget is matched attempted epochs within one FNO architecture, not a cross-architecture equal-FLOPs guarantee.",
            "The three branches isolate optimizer-state carry and cosine horizon only within this fixed K=6 synthetic setup.",
            "Attempted cumulative epochs count compute; validation prefix-best checkpoints do not feed back into the raw continuation state.",
            "A plateau is an operational result under the locked optimizer protocols, not a global optimization proof.",
            "Dev2 diagnostics were computed only after strategy and checkpoint selection and cannot open the blind final.",
            "No real BOST superiority, acquisition-geometry benefit or paper-level novelty follows from this audit.",
        ],
    }
    (output_dir / "v3d_optimizer_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_output_checksums(output_dir)
    print(
        json.dumps(
            {
                "scientific_status": status,
                "validation_champion": champion,
                "champion_plateau": decisions[champion],
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"results: {output_dir}", flush=True)


def reanalyze_existing(
    experiment: dict,
    experiment_config_path: Path,
    output_dir: Path,
) -> None:
    report_path = output_dir / "v3d_optimizer_report.json"
    dashboard_path = output_dir / "v3d_optimizer_dashboard.json"
    existing_report = read_json(report_path) if report_path.exists() else {}
    existing_dashboard = read_json(dashboard_path) if dashboard_path.exists() else {}
    dataset_path = ROOT / "results" / "t16_v3c_dev2_dataset.npz"
    checkpoint_rows = read_csv(output_dir / "v3d_optimizer_checkpoints.csv")
    dataset_config = read_json(CONFIG_ROOT / experiment["dataset_config"])
    train_sample_count = int(dataset_config["splits"]["train"]["count"])
    batch_size = int(dataset_config["training"]["batch_size"])
    for row in checkpoint_rows:
        block_index = int(row["block_index"])
        row["block_seed"] = (
            int(row["model_seed"]) + 101
            if block_index == 0
            else int(row["model_seed"]) + 10_000 + block_index
        )
        row["batch_order_contract_sha256"] = batch_order_contract_sha256(
            train_sample_count,
            batch_size,
            (
                int(experiment["base_epochs"])
                if block_index == 0
                else int(experiment["continuation_block_epochs"])
            ),
            int(row["block_seed"]),
        )
    analyze_and_write(
        experiment,
        output_dir,
        checkpoint_rows,
        read_csv(output_dir / "v3d_optimizer_history.csv"),
        read_csv(output_dir / "v3d_optimizer_samples.csv"),
        bool(existing_dashboard.get("full_protocol", False)),
        existing_report.get("environment", {"mode": "analysis_only"}),
        build_provenance(experiment_config_path, experiment, dataset_path),
    )


def main() -> None:
    args = parse_args()
    experiment = read_json(args.config)
    configured_max_total_epochs = int(experiment["max_total_epochs"])
    configured_seed_count = len(experiment["training_seeds"])
    dataset_config = read_json(CONFIG_ROOT / experiment["dataset_config"])
    device_name = args.device or str(dataset_config["training"]["device"])
    device = choose_device(device_name)
    seeds = [int(value) for value in experiment["training_seeds"]]
    if args.seed_limit is not None:
        seeds = seeds[: int(args.seed_limit)]
    max_total_epochs = int(args.max_total_epochs or experiment["max_total_epochs"])
    if max_total_epochs != configured_max_total_epochs:
        experiment = copy.deepcopy(experiment)
        experiment["max_total_epochs"] = max_total_epochs
    base_epochs = int(experiment["base_epochs"])
    block_epochs = int(experiment["continuation_block_epochs"])
    if max_total_epochs < base_epochs or (max_total_epochs - base_epochs) % block_epochs:
        raise ValueError("max total epochs must equal base + N * continuation block")
    output_dir = args.output_dir or ROOT / "results" / experiment["output_dir"]
    work_dir = args.work_dir or ROOT / "results" / experiment["work_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    if args.analysis_only:
        reanalyze_existing(experiment, args.config, output_dir)
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
        raise RuntimeError("optimizer audit requires validation-locked ridge")
    data = append_ray_view_channels(replace_lift_with_ridge(direct, selected_ridge))
    write_csv(output_dir / "v3d_optimizer_baseline_tuning.csv", tuning_rows)

    checkpoint_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    base_states: dict[int, dict[str, torch.Tensor]] = {}
    base_selected_epochs: dict[int, int] = {}
    final_states: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
    final_selected_epochs: dict[tuple[str, int], int] = {}
    continuation_epochs = max_total_epochs - base_epochs
    strategy_lookup = {str(row["id"]): row for row in experiment["strategies"]}

    for outer_seed in seeds:
        base_seed = outer_seed + 101
        set_seed(base_seed)
        base_model = make_model(
            "fno",
            dataset_config["models"]["fno"],
            int(data["inputs"].shape[1]),
            residual=True,
        )
        base_record = train_model(
            "base_fno",
            base_model,
            data,
            training_config(
                dataset_config,
                base_seed,
                base_epochs,
                device_name,
                fixed_epochs=False,
            ),
            work_dir / str(outer_seed) / "base",
        )
        if int(base_record["epochs_ran"]) != base_epochs:
            raise RuntimeError("shared base stopped before 24 epochs")
        base_state = cpu_state(base_record["model"])
        base_states[outer_seed] = base_state
        base_validation = float(base_record["best_val_rel_l2"])
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
        train_indices = split_indices(data)["train"]
        train_sample_count = len(train_indices)
        batch_size = int(dataset_config["training"]["batch_size"])

        for strategy in strategy_ids(experiment):
            spec = strategy_lookup[strategy]
            model = make_model(
                "fno",
                dataset_config["models"]["fno"],
                int(data["inputs"].shape[1]),
                residual=True,
            )
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
                    "endpoint_validation_rel_l2": base_validation,
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
                    "endpoint_checkpoint_sha256": state_sha256(base_state),
                }
            )

            optimizer: torch.optim.Optimizer | None = None
            scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
            if bool(spec["carry_optimizer"]):
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=float(experiment["continuation_learning_rate"]),
                    weight_decay=float(dataset_config["training"]["weight_decay"]),
                )
                if str(spec["cosine_horizon"]) == "continuation":
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=continuation_epochs
                    )

            for block_index, cumulative_epoch in enumerate(
                range(base_epochs + block_epochs, max_total_epochs + 1, block_epochs),
                start=1,
            ):
                block_seed = outer_seed + 10_000 + block_index
                set_seed(block_seed)
                if not bool(spec["carry_optimizer"]):
                    optimizer = torch.optim.AdamW(
                        model.parameters(),
                        lr=float(experiment["continuation_learning_rate"]),
                        weight_decay=float(dataset_config["training"]["weight_decay"]),
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
                    dataset_config,
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
                        "relative_validation_improvement_pct": relative_improvement_pct(
                            previous_validation, selected_validation
                        ),
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
                f"seed={outer_seed} strategy={strategy}: attempted through {max_total_epochs}",
                flush=True,
            )

    validation_rows, decisions = summarize_strategies(checkpoint_rows, experiment)
    champion = choose_validation_champion(validation_rows, max_total_epochs)
    print(
        json.dumps(
            {
                "validation_champion": champion,
                "decisions": decisions,
            },
            indent=2,
        ),
        flush=True,
    )

    sample_rows: list[dict[str, object]] = []
    for outer_seed, state in sorted(base_states.items()):
        for row in evaluate_checkpoint(
            outer_seed,
            base_epochs,
            state,
            data,
            dataset_config,
            selected_ridge[budget],
            int(experiment["audit_query_index"]),
            device,
        ):
            sample_rows.append(
                {
                    "strategy": "base_24",
                    "selected_checkpoint_epoch": base_selected_epochs[outer_seed],
                    **row,
                }
            )
    for (strategy, outer_seed), state in sorted(final_states.items()):
        for row in evaluate_checkpoint(
            outer_seed,
            max_total_epochs,
            state,
            data,
            dataset_config,
            selected_ridge[budget],
            int(experiment["audit_query_index"]),
            device,
        ):
            sample_rows.append(
                {
                    "strategy": strategy,
                    "selected_checkpoint_epoch": final_selected_epochs[
                        (strategy, outer_seed)
                    ],
                    **row,
                }
            )

    full_protocol = (
        len(seeds) == configured_seed_count
        and max_total_epochs == configured_max_total_epochs
    )
    analyze_and_write(
        experiment,
        output_dir,
        checkpoint_rows,
        history_rows,
        sample_rows,
        full_protocol,
        environment={
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "requested_device": device_name,
            "runtime_device": str(device),
        },
        provenance=build_provenance(args.config, experiment, dataset_path),
    )


if __name__ == "__main__":
    main()
