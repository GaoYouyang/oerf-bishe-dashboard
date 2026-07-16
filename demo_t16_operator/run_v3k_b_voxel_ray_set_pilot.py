#!/usr/bin/env python3
"""Run the v3k-B voxel-local ray-set mechanism pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

try:
    from .counterfactual_geometry import (
        CounterfactualGeometryDataset,
        CounterfactualInputFactory,
        build_pair_schedule,
        ray_angle_pairing_derangement_rows,
        ray_set_components_for_pairs,
        schedule_balance,
    )
    from .models import make_model
    from .own_algorithm_models import VoxelRaySetResidualOperator
    from .run_v3j_gc_sro_functional_pilot import parameter_count
    from .run_v3k_a_counterfactual_supervision import (
        bootstrap_interval,
        components_for_batch,
        field_relative_l2,
        load_private_dataset,
        metric_rows,
        precompute_base_predictions,
        read_json,
        sha256,
        source_cluster_mean,
        write_csv,
    )
    from .train_eval import (
        choose_device,
        gradient_mse,
        masked_relative_projection_loss,
        project_torch,
        set_seed,
        synchronize,
    )
except ImportError:
    from counterfactual_geometry import (
        CounterfactualGeometryDataset,
        CounterfactualInputFactory,
        build_pair_schedule,
        ray_angle_pairing_derangement_rows,
        ray_set_components_for_pairs,
        schedule_balance,
    )
    from models import make_model
    from own_algorithm_models import VoxelRaySetResidualOperator
    from run_v3j_gc_sro_functional_pilot import parameter_count
    from run_v3k_a_counterfactual_supervision import (
        bootstrap_interval,
        components_for_batch,
        field_relative_l2,
        load_private_dataset,
        metric_rows,
        precompute_base_predictions,
        read_json,
        sha256,
        source_cluster_mean,
        write_csv,
    )
    from train_eval import (
        choose_device,
        gradient_mse,
        masked_relative_projection_loss,
        project_torch,
        set_seed,
        synchronize,
    )


ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = ROOT / "configs"
DEFAULT_CONFIG = CONFIG_ROOT / "v3k_b_voxel_ray_set_pilot.json"
PUBLIC_FILES = [
    "v3k_b_pair_manifest.csv",
    "v3k_b_ray_angle_pairing_derangement.csv",
    "v3k_b_training_history.csv",
    "v3k_b_sample_metrics.csv",
    "v3k_b_split_summary.csv",
    "v3k_b_pairwise_mechanism.csv",
    "v3k_b_same_model_ray_swap.csv",
    "v3k_b_voxel_ray_set_dashboard.json",
    "v3k_b_voxel_ray_set_report.json",
    "t16_v3k_b_voxel_ray_set_pilot.png",
]
LABELS = {
    "locked_fno": "Locked fixed-layout FNO",
    "pooled_static": "Pooled local evidence",
    "geometry_only": "Geometry-only local set",
    "shuffled_pairing": "Shuffled ray-angle pairing",
    "correct_ray_set": "Correct local ray set",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--analysis-only", action="store_true")
    return parser.parse_args()


def build_model(
    config: dict,
    dataset_config: dict,
    data: dict[str, np.ndarray],
    checkpoint: dict[str, torch.Tensor],
    seed: int,
    method: str,
) -> VoxelRaySetResidualOperator:
    base = make_model(
        "fno", dataset_config["models"]["fno"], int(data["inputs"].shape[1]), residual=True
    )
    base.load_state_dict(checkpoint, strict=True)
    set_seed(int(seed))
    names = [str(value) for value in data["input_channel_names"].tolist()]
    mode = {
        "pooled_static": "pooled_static",
        "geometry_only": "geometry_only",
        "shuffled_pairing": "ray_set",
        "correct_ray_set": "ray_set",
    }[method]
    model_config = config["voxel_ray_set"]
    return VoxelRaySetResidualOperator(
        base_operator=base,
        view_count=int(data["ray_view_channel_count"]),
        mask_channel_start=names.index("camera_0_active"),
        ray_channel_start=int(data["ray_view_channel_start"]),
        angle_sin_channel_start=int(data["ray_angle_sin_channel_start"]),
        angle_cos_channel_start=int(data["ray_angle_cos_channel_start"]),
        coordinate_channels=tuple(names.index(axis) for axis in ("z", "y", "x")),
        token_hidden=int(model_config["token_hidden"]),
        latent_features=int(model_config["latent_features"]),
        adapter_hidden=int(model_config["adapter_hidden"]),
        spectral_modes=tuple(int(value) for value in model_config["spectral_modes"]),
        maximum_correction_scale=float(model_config["maximum_correction_scale"]),
        acquisition_mode=mode,
        freeze_base=bool(model_config["freeze_base"]),
    )


def selected_components(
    method: str,
    wrong_components: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    indices: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if method != "shuffled_pairing":
        return None
    return components_for_batch(wrong_components, indices, device)


def train_model(
    config: dict,
    dataset_config: dict,
    data: dict[str, np.ndarray],
    checkpoint: dict[str, torch.Tensor],
    train_dataset: CounterfactualGeometryDataset,
    val_dataset: CounterfactualGeometryDataset,
    train_base: np.ndarray,
    val_base: np.ndarray,
    train_wrong: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    val_wrong: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    method: str,
    seed: int,
    device: torch.device,
    work_dir: Path,
) -> tuple[torch.nn.Module, list[dict[str, object]], dict[str, object]]:
    training = config["training"]
    model = build_model(config, dataset_config, data, checkpoint, seed, method).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(training["epochs"])
    )
    generator = torch.Generator().manual_seed(int(seed))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=int(training["batch_size"]), shuffle=False
    )
    operator = torch.from_numpy(data["forward_matrix"]).to(device)
    best_val = math.inf
    best_epoch = -1
    best_state = None
    history: list[dict[str, object]] = []
    start = time.perf_counter()
    for epoch in range(int(training["epochs"])):
        model.train()
        total_sum = 0.0
        seen = 0
        for batch in train_loader:
            indices = batch["index"]
            x = batch["x"].to(device)
            target = batch["field"].to(device)
            base = torch.from_numpy(train_base[indices.numpy()]).to(device)
            acquisition = selected_components(method, train_wrong, indices, device)
            optimizer.zero_grad(set_to_none=True)
            correction, _ = model.correction(
                x, base_prediction=base, acquisition_components=acquisition
            )
            prediction = base + correction
            field_loss = functional.mse_loss(prediction, target)
            grad_loss = gradient_mse(prediction, target)
            reprojection = masked_relative_projection_loss(
                project_torch(prediction, operator),
                batch["observation"].to(device),
                batch["view_mask"].to(device),
            )
            outside = (x[:, 1:2] < 0.02).to(prediction.dtype)
            boundary = torch.mean((prediction * outside) ** 2)
            loss = (
                field_loss
                + float(training["lambda_gradient"]) * grad_loss
                + float(training["lambda_reprojection"]) * reprojection
                + float(training["lambda_boundary"]) * boundary
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainable, float(training["gradient_clip_norm"])
            )
            optimizer.step()
            total_sum += float(loss.detach()) * len(indices)
            seen += len(indices)

        model.eval()
        val_values: list[float] = []
        val_sources: list[int] = []
        with torch.no_grad():
            for batch in val_loader:
                indices = batch["index"]
                x = batch["x"].to(device)
                target = batch["field"].to(device)
                base = torch.from_numpy(val_base[indices.numpy()]).to(device)
                acquisition = selected_components(method, val_wrong, indices, device)
                correction, _ = model.correction(
                    x, base_prediction=base, acquisition_components=acquisition
                )
                error = field_relative_l2(base + correction, target)
                val_values.extend(error.cpu().tolist())
                val_sources.extend(batch["source_index"].cpu().tolist())
        val_mean = source_cluster_mean(val_values, val_sources)
        history.append(
            {
                "training_arm": str(config["training_arm"]),
                "method": method,
                "model_seed": int(seed),
                "epoch": epoch + 1,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train_total_loss": total_sum / max(seen, 1),
                "validation_field_cluster_rel_l2": val_mean,
            }
        )
        if val_mean < best_val:
            best_val = val_mean
            best_epoch = epoch + 1
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
                if torch.is_tensor(value)
            }
        scheduler.step()
    synchronize(device)
    elapsed = time.perf_counter() - start
    if best_state is None:
        raise RuntimeError("v3k-B did not produce a validation-selected checkpoint")
    model.load_state_dict(best_state)
    model.to("cpu")
    checkpoint_path = work_dir / str(seed) / "checkpoints" / f"{method}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, checkpoint_path)
    record = {
        "training_arm": str(config["training_arm"]),
        "method": method,
        "model_seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_validation_field_cluster_rel_l2": best_val,
        "train_seconds": elapsed,
        "training_pair_rows": len(train_dataset),
        "independent_training_fields": len(
            {int(row["source_index"]) for row in train_dataset.pairs}
        ),
        "total_parameters": parameter_count(model),
        "trainable_parameters": parameter_count(model, trainable_only=True),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_public": False,
    }
    return model, history, record


def predict_dataset(
    model: torch.nn.Module,
    dataset: CounterfactualGeometryDataset,
    base_predictions: np.ndarray,
    wrong_components: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    method: str,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False)
    model = model.to(device).eval()
    output = []
    with torch.no_grad():
        for batch in loader:
            indices = batch["index"]
            x = batch["x"].to(device)
            base = torch.from_numpy(base_predictions[indices.numpy()]).to(device)
            acquisition = selected_components(method, wrong_components, indices, device)
            correction, _ = model.correction(
                x, base_prediction=base, acquisition_components=acquisition
            )
            output.append((base + correction).cpu().numpy())
    model.to("cpu")
    return np.concatenate(output).astype(np.float32)


def field_seed_errors(
    sample_rows: list[dict[str, object]],
) -> dict[tuple[str, int, str, int], float]:
    grouped: dict[tuple[str, int, str, int], list[float]] = defaultdict(list)
    for row in sample_rows:
        key = (
            str(row["method"]),
            int(row["model_seed"]),
            str(row["source_split"]),
            int(row["source_index"]),
        )
        grouped[key].append(float(row["field_rel_l2"]))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def summarize_results(
    sample_rows: list[dict[str, object]], config: dict
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    errors = field_seed_errors(sample_rows)
    seeds = [int(value) for value in config["model_seeds"]]
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in sample_rows:
        grouped[(str(row["method"]), str(row["source_split"]))].append(row)
    split_summary = []
    for (method, split), rows in sorted(grouped.items()):
        sources = sorted({int(row["source_index"]) for row in rows})
        collapsed = [
            np.mean([errors[(method, seed, split, source)] for seed in seeds])
            for source in sources
        ]
        split_summary.append(
            {
                "training_arm": str(config["training_arm"]),
                "method": method,
                "method_label": LABELS[method],
                "source_split": split,
                "sample_metric_rows": len(rows),
                "independent_field_count": len(sources),
                "layouts_per_field": len(rows) // (len(sources) * len(seeds)),
                "model_seed_count": len(seeds),
                "mean_field_rel_l2": float(np.mean(collapsed)),
                "median_field_rel_l2": float(np.median(collapsed)),
                "p90_field_rel_l2": float(np.quantile(collapsed, 0.9)),
                "mean_audit_reprojection_rel_l2": float(
                    np.mean([float(row["audit_reprojection_rel_l2"]) for row in rows])
                ),
            }
        )

    splits = sorted({str(row["source_split"]) for row in sample_rows})
    comparators = [
        "locked_fno",
        "pooled_static",
        "geometry_only",
        "shuffled_pairing",
    ]
    gate = config["mechanism_gate"]
    pairwise = []
    for split in splits:
        sources = sorted(
            {
                int(row["source_index"])
                for row in sample_rows
                if str(row["source_split"]) == split
            }
        )
        for comparator in comparators:
            field_collapsed = []
            seed_means = []
            all_gains = []
            for source in sources:
                gains = []
                for seed in seeds:
                    candidate = errors[("correct_ray_set", seed, split, source)]
                    control = errors[(comparator, seed, split, source)]
                    gain = 100.0 * (control - candidate) / max(control, 1e-12)
                    gains.append(gain)
                    all_gains.append(gain)
                field_collapsed.append(float(np.mean(gains)))
            for seed in seeds:
                seed_means.append(
                    float(
                        np.mean(
                            [
                                100.0
                                * (
                                    errors[(comparator, seed, split, source)]
                                    - errors[("correct_ray_set", seed, split, source)]
                                )
                                / max(errors[(comparator, seed, split, source)], 1e-12)
                                for source in sources
                            ]
                        )
                    )
                )
            ci_low, ci_high = bootstrap_interval(
                np.asarray(field_collapsed),
                int(gate["bootstrap_seed"]) + len(pairwise),
                int(gate["bootstrap_replicates"]),
            )
            pairwise.append(
                {
                    "training_arm": str(config["training_arm"]),
                    "source_split": split,
                    "candidate": "correct_ray_set",
                    "comparator": comparator,
                    "independent_field_count": len(sources),
                    "model_seed_count": len(seeds),
                    "mean_field_gain_pct": float(np.mean(field_collapsed)),
                    "field_cluster_ci95_low_pct": ci_low,
                    "field_cluster_ci95_high_pct": ci_high,
                    "median_field_gain_pct": float(np.median(field_collapsed)),
                    "p10_field_gain_pct": float(np.quantile(field_collapsed, 0.1)),
                    "harm_rate_gt_1pct": float(np.mean(np.asarray(all_gains) < -1.0)),
                    "positive_seed_count": int(sum(value > 0.0 for value in seed_means)),
                    "seed_mean_gains_pct": ";".join(
                        f"{value:.8f}" for value in seed_means
                    ),
                }
            )
    return split_summary, pairwise


def same_model_swap_audit(
    trained: dict[tuple[str, int], torch.nn.Module],
    val_dataset: CounterfactualGeometryDataset,
    val_base: np.ndarray,
    correct_components: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    wrong_components: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    config: dict,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    seeds = [int(value) for value in config["model_seeds"]]
    rows: list[dict[str, object]] = []
    loader = DataLoader(
        val_dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False
    )
    for seed in seeds:
        model = trained[("correct_ray_set", seed)].to(device).eval()
        with torch.no_grad():
            for batch in loader:
                indices = batch["index"]
                selected = indices.numpy()
                x = batch["x"].to(device)
                target = batch["field"].to(device)
                base = torch.from_numpy(val_base[selected]).to(device)
                correct = components_for_batch(correct_components, indices, device)
                wrong = components_for_batch(wrong_components, indices, device)
                correct_correction, correct_attention = model.correction(
                    x, base_prediction=base, acquisition_components=correct
                )
                wrong_correction, wrong_attention = model.correction(
                    x, base_prediction=base, acquisition_components=wrong
                )
                correct_error = field_relative_l2(base + correct_correction, target)
                wrong_error = field_relative_l2(base + wrong_correction, target)
                gains = 100.0 * (wrong_error - correct_error) / wrong_error.clamp_min(
                    1e-12
                )
                correction_change = 100.0 * torch.linalg.vector_norm(
                    (correct_correction - wrong_correction).flatten(start_dim=1), dim=1
                ) / torch.linalg.vector_norm(
                    correct_correction.flatten(start_dim=1), dim=1
                ).clamp_min(1e-12)
                attention_change = torch.mean(
                    torch.abs(correct_attention - wrong_attention), dim=(1, 2, 3, 4)
                )
                ray_change = 100.0 * torch.linalg.vector_norm(
                    (correct[3] - wrong[3]).flatten(start_dim=1), dim=1
                ) / torch.linalg.vector_norm(
                    correct[3].flatten(start_dim=1), dim=1
                ).clamp_min(1e-12)
                for local, pair_index in enumerate(selected):
                    pair = val_dataset.pairs[int(pair_index)]
                    rows.append(
                        {
                            "training_arm": str(config["training_arm"]),
                            "model_seed": seed,
                            "pair_index": int(pair_index),
                            "source_index": int(pair["source_index"]),
                            "geometry_id": str(pair["geometry_id"]),
                            "shuffled_pairing_rule": "cyclic angle reassignment within unchanged active ray set",
                            "ray_input_swap_relative_pct": float(ray_change[local]),
                            "attention_swap_mean_l1": float(attention_change[local]),
                            "correction_swap_relative_pct": float(
                                correction_change[local]
                            ),
                            "correct_ray_set_field_rel_l2": float(
                                correct_error[local]
                            ),
                            "wrong_ray_set_field_rel_l2": float(wrong_error[local]),
                            "correct_ray_set_field_gain_pct": float(gains[local]),
                        }
                    )
        model.to("cpu")

    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["model_seed"]), int(row["source_index"]))].append(
            float(row["correct_ray_set_field_gain_pct"])
        )
    sources = sorted({source for _, source in grouped})
    field_collapsed = [
        float(np.mean([np.mean(grouped[(seed, source)]) for seed in seeds]))
        for source in sources
    ]
    seed_means = [
        float(np.mean([np.mean(grouped[(seed, source)]) for source in sources]))
        for seed in seeds
    ]
    gate = config["mechanism_gate"]
    ci_low, ci_high = bootstrap_interval(
        np.asarray(field_collapsed),
        int(gate["bootstrap_seed"]) + 20_000,
        int(gate["bootstrap_replicates"]),
    )
    summary = {
        "validation_pair_rows": len(rows),
        "independent_field_count": len(sources),
        "model_seed_count": len(seeds),
        "mean_ray_input_swap_relative_pct": float(
            np.mean([float(row["ray_input_swap_relative_pct"]) for row in rows])
        ),
        "mean_attention_swap_l1": float(
            np.mean([float(row["attention_swap_mean_l1"]) for row in rows])
        ),
        "mean_correction_swap_relative_pct": float(
            np.mean([float(row["correction_swap_relative_pct"]) for row in rows])
        ),
        "mean_correct_ray_set_field_gain_pct": float(np.mean(field_collapsed)),
        "field_cluster_ci95_low_pct": ci_low,
        "field_cluster_ci95_high_pct": ci_high,
        "p10_correct_ray_set_field_gain_pct": float(
            np.quantile(field_collapsed, 0.1)
        ),
        "positive_seed_count": int(sum(value > 0.0 for value in seed_means)),
        "seed_mean_gains_pct": ";".join(f"{value:.8f}" for value in seed_means),
    }
    return rows, summary


def find_row(rows: list[dict[str, object]], **criteria: object) -> dict[str, object]:
    matched = [
        row
        for row in rows
        if all(str(row[key]) == str(value) for key, value in criteria.items())
    ]
    if len(matched) != 1:
        raise ValueError(f"expected one row for {criteria}, found {len(matched)}")
    return matched[0]


def gate_result(
    config: dict,
    pairwise: list[dict[str, object]],
    swap: dict[str, object],
    v3k_a_dashboard: dict,
) -> tuple[dict[str, bool], str, float]:
    gate = config["mechanism_gate"]
    delta = float(gate["minimum_mean_gain_pct"])
    ci_floor = float(gate["minimum_cluster_ci95_low_pct"])
    required_seeds = int(gate["required_positive_seed_count"])
    primary = str(gate["primary_split"])
    held_out = str(gate["held_out_geometry_split"])
    joint = str(gate["joint_ood_split"])
    val_shuffled = find_row(
        pairwise, source_split=primary, comparator="shuffled_pairing"
    )
    val_static = find_row(pairwise, source_split=primary, comparator="pooled_static")
    val_geometry = find_row(
        pairwise, source_split=primary, comparator="geometry_only"
    )
    held_out_shuffled = find_row(
        pairwise, source_split=held_out, comparator="shuffled_pairing"
    )
    joint_shuffled = find_row(
        pairwise, source_split=joint, comparator="shuffled_pairing"
    )
    global_row = find_row(
        v3k_a_dashboard["pairwise_mechanism"],
        training_arm="m4_counterfactual",
        source_split=primary,
        comparator="shuffled_geometry",
    )
    gain_over_global = float(val_shuffled["mean_field_gain_pct"]) - float(
        global_row["mean_field_gain_pct"]
    )

    def gain_pass(row: dict[str, object]) -> bool:
        return (
            float(row["mean_field_gain_pct"]) >= delta
            and float(row["field_cluster_ci95_low_pct"]) > ci_floor
            and int(row["positive_seed_count"]) >= required_seeds
        )

    checks = {
        "correct_vs_shuffled_validation": gain_pass(val_shuffled),
        "correct_vs_pooled_static_validation": gain_pass(val_static),
        "correct_vs_geometry_only_validation": gain_pass(val_geometry),
        "gain_over_v3k_a_global_modulation": gain_over_global
        >= float(gate["minimum_gain_over_v3k_a_correct_vs_shuffled_pct"]),
        "correct_vs_shuffled_geometry_held_out": gain_pass(held_out_shuffled),
        "joint_ood_no_material_harm": float(joint_shuffled["mean_field_gain_pct"])
        >= float(gate["minimum_joint_ood_gain_pct"]),
        "same_model_ray_swap_propagates": (
            float(swap["mean_correct_ray_set_field_gain_pct"])
            >= float(gate["minimum_swap_field_gain_pct"])
            and float(swap["field_cluster_ci95_low_pct"]) > ci_floor
            and int(swap["positive_seed_count"]) >= required_seeds
            and float(swap["mean_correction_swap_relative_pct"])
            >= float(gate["minimum_correction_swap_relative_pct"])
            and float(swap["mean_attention_swap_l1"])
            >= float(gate["minimum_attention_swap_l1"])
        ),
    }
    if all(checks.values()):
        status = "VOXEL_RAY_SET_MECHANISM_GATE_PASS_MATCHED_BASELINES_REQUIRED"
    elif (
        float(val_shuffled["field_cluster_ci95_high_pct"]) < 0.0
        or float(swap["field_cluster_ci95_high_pct"]) < 0.0
    ):
        status = "VOXEL_RAY_ANGLE_PAIRING_HARM_STOP_ATTENTION_SCALING"
    else:
        upper_bounds = [
            float(val_shuffled["field_cluster_ci95_high_pct"]),
            float(val_static["field_cluster_ci95_high_pct"]),
            float(val_geometry["field_cluster_ci95_high_pct"]),
            float(swap["field_cluster_ci95_high_pct"]),
        ]
        if all(value < delta for value in upper_bounds):
            status = "VOXEL_RAY_SET_MECHANISM_FAIL_STOP_ATTENTION_SCALING"
        else:
            status = "VOXEL_RAY_SET_MECHANISM_INCONCLUSIVE_DO_NOT_SCALE"
    return checks, status, gain_over_global


def plot_results(
    path: Path,
    history: list[dict[str, object]],
    pairwise: list[dict[str, object]],
    swap: dict[str, object],
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4))
    colors = {
        "pooled_static": "#7c8792",
        "geometry_only": "#d58b37",
        "shuffled_pairing": "#c45c66",
        "correct_ray_set": "#247d72",
    }
    for method, color in colors.items():
        epochs = sorted(
            {int(row["epoch"]) for row in history if row["method"] == method}
        )
        values = [
            np.mean(
                [
                    float(row["validation_field_cluster_rel_l2"])
                    for row in history
                    if row["method"] == method and int(row["epoch"]) == epoch
                ]
            )
            for epoch in epochs
        ]
        axes[0].plot(epochs, values, label=LABELS[method], color=color)
    axes[0].set_title("Validation learning curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Field relative L2")
    axes[0].legend(fontsize=7)

    validation = [
        row
        for row in pairwise
        if row["source_split"] == "val" and row["comparator"] != "locked_fno"
    ]
    x = np.arange(len(validation))
    means = np.asarray([float(row["mean_field_gain_pct"]) for row in validation])
    lows = np.asarray([float(row["field_cluster_ci95_low_pct"]) for row in validation])
    highs = np.asarray([float(row["field_cluster_ci95_high_pct"]) for row in validation])
    axes[1].errorbar(
        x, means, yerr=np.vstack([means - lows, highs - means]), fmt="o", color="#247d72"
    )
    axes[1].axhline(0.25, color="#b44b4b", linestyle="--", linewidth=1)
    axes[1].axhline(0.0, color="#333333", linewidth=0.8)
    axes[1].set_xticks(x, [str(row["comparator"]) for row in validation], rotation=25, ha="right")
    axes[1].set_title("Correct ray-set gain")
    axes[1].set_ylabel("Gain (%)")

    domains = [
        row for row in pairwise if row["comparator"] == "shuffled_pairing"
    ]
    axes[2].bar(
        np.arange(len(domains)),
        [float(row["mean_field_gain_pct"]) for row in domains],
        color="#5d8f88",
    )
    axes[2].axhline(0.25, color="#b44b4b", linestyle="--", linewidth=1)
    axes[2].axhline(0.0, color="#333333", linewidth=0.8)
    axes[2].set_xticks(
        np.arange(len(domains)),
        [str(row["source_split"]) for row in domains],
        rotation=25,
        ha="right",
    )
    axes[2].set_title("Geometry-domain replication")
    axes[2].set_ylabel("Correct vs shuffled gain (%)")

    swap_values = [
        float(swap["mean_attention_swap_l1"]),
        float(swap["mean_correction_swap_relative_pct"]),
        float(swap["mean_correct_ray_set_field_gain_pct"]),
    ]
    axes[3].bar(
        np.arange(3), swap_values, color=["#7c8792", "#d58b37", "#247d72"]
    )
    axes[3].axhline(0.25, color="#b44b4b", linestyle="--", linewidth=1)
    axes[3].axhline(0.0, color="#333333", linewidth=0.8)
    axes[3].set_yscale("symlog", linthresh=0.01)
    for index, value in enumerate(swap_values):
        axes[3].text(
            index,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
        )
    axes[3].set_xticks(
        np.arange(3), ["attention L1", "correction %", "field gain %"], rotation=20
    )
    axes[3].set_title("Same-model ray-set swap")
    fig.suptitle("v3k-B voxel-local ray-set mechanism pilot", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_checksums(output_dir: Path) -> None:
    rows = []
    for name in PUBLIC_FILES:
        rows.append(f"{sha256(output_dir / name)}  {name}")
    (output_dir / "v3k_b_voxel_ray_set_checksums.sha256").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )


def mechanism_diagnosis(
    pairwise: list[dict[str, object]], swap: dict[str, object]
) -> dict[str, object]:
    geometry = find_row(
        pairwise, source_split="val", comparator="geometry_only"
    )
    pooled = find_row(pairwise, source_split="val", comparator="pooled_static")
    pairing = find_row(
        pairwise, source_split="val", comparator="shuffled_pairing"
    )
    return {
        "local_ray_values_help_vs_geometry_only": float(
            geometry["field_cluster_ci95_low_pct"]
        )
        > 0.0,
        "correct_vs_geometry_only_gain_pct": float(geometry["mean_field_gain_pct"]),
        "correct_vs_pooled_static_gain_pct": float(pooled["mean_field_gain_pct"]),
        "correct_pairing_harms_vs_shuffled_pairing": float(
            pairing["field_cluster_ci95_high_pct"]
        )
        < 0.0,
        "correct_vs_shuffled_pairing_gain_pct": float(
            pairing["mean_field_gain_pct"]
        ),
        "same_model_correct_pairing_harms": float(
            swap["field_cluster_ci95_high_pct"]
        )
        < 0.0,
        "same_model_correct_pairing_gain_pct": float(
            swap["mean_correct_ray_set_field_gain_pct"]
        ),
        "interpretation": "voxel-local ray evidence is useful, but scalar sin/cos pairing drives a reproducibly misaligned correction",
    }


def refresh_analysis_only(config: dict, v3k_a: dict) -> None:
    output_dir = ROOT / "results" / str(config["output_dir"])
    dashboard_path = output_dir / PUBLIC_FILES[7]
    report_path = output_dir / PUBLIC_FILES[8]
    dashboard = read_json(dashboard_path)
    report = read_json(report_path)
    checks, status, gain_over_global = gate_result(
        config,
        dashboard["pairwise_mechanism"],
        dashboard["same_model_ray_set_swap"],
        v3k_a,
    )
    mechanism_pass = all(checks.values())
    dashboard["scientific_status"] = status
    dashboard["voxel_ray_set_mechanism_gate_pass"] = mechanism_pass
    dashboard["gate_checks"] = checks
    dashboard["gain_over_v3k_a_correct_vs_shuffled_pct"] = gain_over_global
    dashboard["mechanism_diagnosis"] = mechanism_diagnosis(
        dashboard["pairwise_mechanism"], dashboard["same_model_ray_set_swap"]
    )
    dashboard["next_decision"] = {
        "if_pass": "train matched variable-geometry FNO and VIDON baselines before any superiority claim",
        "current_decision": "stop attention width/head scaling; replace scalar angle pairing with operator-derived local calibration features only under a new preregistration",
        "superiority_training_authorized": mechanism_pass,
        "blind_final_opened": False,
    }
    with (output_dir / PUBLIC_FILES[2]).open(newline="", encoding="utf-8") as handle:
        history = list(csv.DictReader(handle))
    plot_results(
        output_dir / PUBLIC_FILES[-1],
        history,
        dashboard["pairwise_mechanism"],
        dashboard["same_model_ray_set_swap"],
    )
    report["status"] = status
    report["dashboard"] = dashboard
    dashboard_path.write_text(
        json.dumps(dashboard, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_checksums(output_dir)
    print(json.dumps({"status": status, "gate_checks": checks}, indent=2))


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    if args.device:
        config["training"]["device"] = args.device
    if args.epochs:
        config["training"]["epochs"] = int(args.epochs)
    dataset_config = read_json(CONFIG_ROOT / str(config["dataset_config"]))
    private_path = ROOT / "results" / str(config["private_dataset_npz"])
    data = load_private_dataset(private_path)
    v3k_a_path = ROOT / "results" / str(config["v3k_a_dashboard"])
    v3k_a = read_json(v3k_a_path)
    if v3k_a["counterfactual_data_mechanism_gate_pass"]:
        raise RuntimeError("v3k-B is authorized only after the global mechanism fails")
    if args.analysis_only:
        refresh_analysis_only(config, v3k_a)
        return
    checkpoint_path = ROOT / "results" / str(config["base_checkpoint"])
    base_hash_before = sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    device = choose_device(str(config["training"]["device"]))
    design = config["pair_design"]
    factory = CounterfactualInputFactory(data, float(design["ridge_relative"]))
    pairing_rows = ray_angle_pairing_derangement_rows(factory)
    repeats = int(design["repeats_per_source"])
    assignment_seed = int(design["assignment_seed"])
    stride = int(design["counterfactual_stride"])
    train_pairs = build_pair_schedule(
        data,
        "train",
        str(design["train_geometry_partition"]),
        str(config["training_arm"]),
        repeats,
        assignment_seed,
        stride,
    )
    val_pairs = build_pair_schedule(
        data,
        "val",
        str(design["evaluation_geometry_partition_by_split"]["val"]),
        "evaluation",
        repeats,
        assignment_seed,
        stride,
    )
    train_dataset = CounterfactualGeometryDataset(factory, train_pairs)
    val_dataset = CounterfactualGeometryDataset(factory, val_pairs)
    schedule_audit = {
        "training_m4_counterfactual": schedule_balance(train_pairs),
        "shared_validation": schedule_balance(val_pairs),
    }
    pair_manifest = [
        {**row, "schedule_role": "training"} for row in train_pairs
    ] + [{**row, "schedule_role": "checkpoint_selection"} for row in val_pairs]

    base_model = make_model(
        "fno", dataset_config["models"]["fno"], int(data["inputs"].shape[1]), residual=True
    )
    base_model.load_state_dict(checkpoint, strict=True)
    batch_size = int(config["training"]["batch_size"])
    train_base = precompute_base_predictions(
        base_model, train_dataset, device, batch_size
    )
    val_base = precompute_base_predictions(base_model, val_dataset, device, batch_size)
    train_wrong = ray_set_components_for_pairs(
        factory, train_pairs, shuffle_angle_pairing=True
    )
    val_correct = ray_set_components_for_pairs(factory, val_pairs)
    val_wrong = ray_set_components_for_pairs(
        factory, val_pairs, shuffle_angle_pairing=True
    )

    work_dir = ROOT / "results" / str(config["work_dir"])
    output_dir = ROOT / "results" / str(config["output_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    histories: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    trained: dict[tuple[str, int], torch.nn.Module] = {}
    for seed in [int(value) for value in config["model_seeds"]]:
        for method in [str(value) for value in config["methods"]]:
            model, history, record = train_model(
                config,
                dataset_config,
                data,
                checkpoint,
                train_dataset,
                val_dataset,
                train_base,
                val_base,
                train_wrong,
                val_wrong,
                method,
                seed,
                device,
                work_dir,
            )
            histories.extend(history)
            records.append(record)
            trained[(method, seed)] = model
            print(
                json.dumps(
                    {
                        "method": method,
                        "seed": seed,
                        "best_epoch": record["best_epoch"],
                        "best_val": record["best_validation_field_cluster_rel_l2"],
                        "seconds": record["train_seconds"],
                    }
                )
            )

    evaluation_datasets = {"val": val_dataset}
    evaluation_bases = {"val": val_base}
    evaluation_wrong = {"val": val_wrong}
    for split, partition in design["evaluation_geometry_partition_by_split"].items():
        split = str(split)
        if split == "val":
            continue
        pairs = build_pair_schedule(
            data,
            split,
            str(partition),
            "evaluation",
            repeats,
            assignment_seed,
            stride,
        )
        pair_manifest.extend({**row, "schedule_role": "development_audit"} for row in pairs)
        dataset = CounterfactualGeometryDataset(factory, pairs)
        evaluation_datasets[split] = dataset
        evaluation_bases[split] = precompute_base_predictions(
            base_model, dataset, device, batch_size
        )
        evaluation_wrong[split] = ray_set_components_for_pairs(
            factory, pairs, shuffle_angle_pairing=True
        )
        schedule_audit[f"shared_{split}"] = schedule_balance(pairs)

    sample_rows: list[dict[str, object]] = []
    seeds = [int(value) for value in config["model_seeds"]]
    for split, dataset in evaluation_datasets.items():
        base_predictions = evaluation_bases[split]
        wrong = evaluation_wrong[split]
        for seed in seeds:
            sample_rows.extend(
                metric_rows(
                    base_predictions,
                    dataset,
                    str(config["training_arm"]),
                    "locked_fno",
                    seed,
                )
            )
            for method in [str(value) for value in config["methods"]]:
                predictions = predict_dataset(
                    trained[(method, seed)],
                    dataset,
                    base_predictions,
                    wrong,
                    method,
                    device,
                    batch_size,
                )
                sample_rows.extend(
                    metric_rows(
                        predictions,
                        dataset,
                        str(config["training_arm"]),
                        method,
                        seed,
                    )
                )

    split_summary, pairwise = summarize_results(sample_rows, config)
    swap_rows, swap_summary = same_model_swap_audit(
        trained,
        val_dataset,
        val_base,
        val_correct,
        val_wrong,
        config,
        device,
    )
    gate_checks, status, gain_over_global = gate_result(
        config, pairwise, swap_summary, v3k_a
    )
    mechanism_pass = all(gate_checks.values())
    plot_results(output_dir / PUBLIC_FILES[-1], histories, pairwise, swap_summary)
    write_csv(output_dir / PUBLIC_FILES[0], pair_manifest)
    write_csv(output_dir / PUBLIC_FILES[1], pairing_rows)
    write_csv(output_dir / PUBLIC_FILES[2], histories)
    write_csv(output_dir / PUBLIC_FILES[3], sample_rows)
    write_csv(output_dir / PUBLIC_FILES[4], split_summary)
    write_csv(output_dir / PUBLIC_FILES[5], pairwise)
    write_csv(output_dir / PUBLIC_FILES[6], swap_rows)

    trainable_counts = sorted({int(row["trainable_parameters"]) for row in records})
    dashboard = {
        "experiment": config["name"],
        "scientific_status": status,
        "voxel_ray_set_mechanism_gate_pass": mechanism_pass,
        "development_only": True,
        "training_arm": config["training_arm"],
        "methods": ["locked_fno", *config["methods"]],
        "model_seeds": seeds,
        "training_epochs": int(config["training"]["epochs"]),
        "training_records": records,
        "schedule_audit": schedule_audit,
        "engineering_contract": {
            "voxel_local_camera_tokens": [
                "normalized_backprojection",
                "camera_mask",
                "sin_theta",
                "cos_theta",
            ],
            "shared_token_encoder": True,
            "masked_query_attention": True,
            "joint_camera_permutation_invariant": True,
            "zero_initialized_support_limited_head": True,
            "base_checkpoint_frozen": True,
            "audit_camera_zero_in_all_ray_sets": True,
            "same_fields_rows_optimizer_steps_across_methods": True,
        },
        "parameter_contract": {
            "trainable_parameters": trainable_counts,
            "v3k_a_trainable_parameters": 1023,
            "ratio_vs_v3k_a": trainable_counts[0] / 1023.0,
            "within_preregistered_upper_ratio": trainable_counts[0] / 1023.0
            <= float(
                config["voxel_ray_set"][
                    "maximum_trainable_parameter_ratio_vs_v3k_a"
                ]
            ),
        },
        "gate_thresholds": config["mechanism_gate"],
        "gate_checks": gate_checks,
        "gain_over_v3k_a_correct_vs_shuffled_pct": gain_over_global,
        "split_summary": split_summary,
        "pairwise_mechanism": pairwise,
        "same_model_ray_set_swap": swap_summary,
        "mechanism_diagnosis": mechanism_diagnosis(pairwise, swap_summary),
        "next_decision": {
            "if_pass": "train matched variable-geometry FNO and VIDON baselines before any superiority claim",
            "if_clear_fail": "stop attention width/head scaling and replace scalar angle pairing with operator-derived local calibration features only under a new preregistration",
            "if_inconclusive": "do not scale; add independent fields only under a new preregistration",
            "superiority_training_authorized": mechanism_pass,
            "blind_final_opened": False,
        },
        "claims_boundary": config["claims_boundary"],
    }
    (output_dir / PUBLIC_FILES[7]).write_text(
        json.dumps(dashboard, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report = {
        "status": status,
        "dashboard": dashboard,
        "protocol": {
            "candidate": "voxel-local active-camera token aggregation conditioned on the frozen FNO query",
            "controls": "same architecture, active rays, camera budget, and parameters with pooled-static, geometry-only, or within-set ray-angle pairing shuffle",
            "selection": "best epoch by source-field mean over all four unseen validation layouts",
            "statistical_unit": "source field after collapsing layouts and three model seeds",
            "test_read_timing": "development test metrics computed after all 12 checkpoints were selected",
            "minimum_meaningful_gain_pct": config["mechanism_gate"][
                "minimum_mean_gain_pct"
            ],
        },
        "provenance": {
            "config_sha256": sha256(args.config),
            "private_dataset_sha256": sha256(private_path),
            "base_checkpoint_sha256_before": base_hash_before,
            "base_checkpoint_sha256_after": sha256(checkpoint_path),
            "base_checkpoint_drift": int(base_hash_before != sha256(checkpoint_path)),
            "v3k_a_dashboard_sha256": sha256(v3k_a_path),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "training_seconds_total": float(
                sum(float(row["train_seconds"]) for row in records)
            ),
        },
        "public_assets": PUBLIC_FILES,
        "private_assets": {
            "checkpoint_count": len(records),
            "work_dir": str(config["work_dir"]),
            "published": False,
        },
    }
    (output_dir / PUBLIC_FILES[8]).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_checksums(output_dir)
    print(json.dumps({"status": status, "gate_checks": gate_checks}, indent=2))


if __name__ == "__main__":
    main()
