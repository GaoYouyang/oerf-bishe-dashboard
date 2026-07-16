#!/usr/bin/env python3
"""Audit geometry identifiability and verify the GC-SRO v0 engineering contract."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

try:
    from .data import generate_dataset, load_npz, split_indices
    from .direct_operator_data import prepare_direct_operator_data, replace_lift_with_ridge
    from .models import make_model
    from .own_algorithm_data import append_ray_view_channels
    from .own_algorithm_models import GeometryConditionedSpectralResidualOperator
    from .run_v3d_fno_saturation_audit import read_json, write_csv
    from .run_v3f_deeponet_frontier import sha256_file
    from .train_eval import set_seed
    from .variable_geometry import (
        assign_geometry_partitions,
        build_geometry_manifest,
        evaluate_geometry_ridge,
        geometry_entropy_bits,
        geometry_id,
        mean_pairwise_jaccard_distance,
        partition_counts,
        reference_mask_id,
        summarize_field_geometry_spread,
        summarize_geometry_errors,
    )
except ImportError:
    from data import generate_dataset, load_npz, split_indices
    from direct_operator_data import prepare_direct_operator_data, replace_lift_with_ridge
    from models import make_model
    from own_algorithm_data import append_ray_view_channels
    from own_algorithm_models import GeometryConditionedSpectralResidualOperator
    from run_v3d_fno_saturation_audit import read_json, write_csv
    from run_v3f_deeponet_frontier import sha256_file
    from train_eval import set_seed
    from variable_geometry import (
        assign_geometry_partitions,
        build_geometry_manifest,
        evaluate_geometry_ridge,
        geometry_entropy_bits,
        geometry_id,
        mean_pairwise_jaccard_distance,
        partition_counts,
        reference_mask_id,
        summarize_field_geometry_spread,
        summarize_geometry_errors,
    )


ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = ROOT / "configs"
DEFAULT_CONFIG = CONFIG_ROOT / "v3h_gc_sro_geometry_gate.json"
CHECKSUM_FILES = [
    "v3h_geometry_manifest.csv",
    "v3h_geometry_field_errors.csv",
    "v3h_geometry_summary.csv",
    "v3h_field_geometry_spread.csv",
    "v3h_geometry_gate.csv",
    "v3h_gc_sro_control_contract.csv",
    "v3h_gc_sro_geometry_dashboard.json",
    "v3h_gc_sro_geometry_report.json",
    "t16_v3h_gc_sro_geometry_gate.png",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--force-data", action="store_true")
    return parser.parse_args()


def tensor_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
        if torch.is_tensor(value)
    }


def parameter_count(module: torch.nn.Module, trainable_only: bool = False) -> int:
    return int(
        sum(
            parameter.numel()
            for parameter in module.parameters()
            if not trainable_only or parameter.requires_grad
        )
    )


def mean_pairwise_distance(values: torch.Tensor) -> float:
    if values.shape[0] < 2:
        return 0.0
    return float(torch.pdist(values.detach().cpu()).mean())


def make_variable_descriptor_batch(
    template: torch.Tensor,
    manifest_rows: list[dict[str, object]],
    masks: dict[str, np.ndarray],
    angles_degrees: np.ndarray,
    mask_channel_start: int,
    angle_sin_channel_start: int,
    angle_cos_channel_start: int,
) -> torch.Tensor:
    batch = template[:1].repeat(len(manifest_rows), 1, 1, 1, 1)
    radians = np.deg2rad(np.asarray(angles_degrees, dtype=np.float32))
    for row_index, row in enumerate(manifest_rows):
        mask = masks[str(row["geometry_id"])]
        for camera_index, active in enumerate(mask):
            batch[row_index, mask_channel_start + camera_index] = float(active)
            batch[row_index, angle_sin_channel_start + camera_index] = float(
                active * np.sin(radians[camera_index])
            )
            batch[row_index, angle_cos_channel_start + camera_index] = float(
                active * np.cos(radians[camera_index])
            )
    return batch


def build_gc_sro_gate(
    experiment: dict,
    dataset_config: dict,
    ray_data: dict[str, np.ndarray],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    gc_config = experiment["gc_sro"]
    names = [str(value) for value in ray_data["input_channel_names"].tolist()]
    base = make_model(
        "fno",
        dataset_config["models"]["fno"],
        int(ray_data["inputs"].shape[1]),
        residual=True,
    )
    checkpoint_path = ROOT / "results" / str(gc_config["base_checkpoint"])
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    base.load_state_dict(checkpoint, strict=True)
    set_seed(int(gc_config["model_seed"]))
    model = GeometryConditionedSpectralResidualOperator(
        base_operator=base,
        view_count=int(ray_data["ray_view_channel_count"]),
        mask_channel_start=names.index("camera_0_active"),
        angle_sin_channel_start=int(ray_data["ray_angle_sin_channel_start"]),
        angle_cos_channel_start=int(ray_data["ray_angle_cos_channel_start"]),
        coordinate_channels=tuple(names.index(axis) for axis in ("z", "y", "x")),
        descriptor_hidden=int(gc_config["descriptor_hidden"]),
        descriptor_embedding=int(gc_config["descriptor_embedding"]),
        adapter_hidden=int(gc_config["adapter_hidden"]),
        spectral_modes=tuple(int(value) for value in gc_config["spectral_modes"]),
        maximum_correction_scale=float(gc_config["maximum_correction_scale"]),
        descriptor_mode="geometry",
        freeze_base=bool(gc_config["freeze_base"]),
    )

    exactness_indices = []
    for split_values in split_indices(ray_data).values():
        exactness_indices.extend(
            int(value)
            for value in split_values[: int(gc_config["exactness_samples_per_split"])]
        )
    exactness_indices = np.asarray(exactness_indices, dtype=np.int64)
    x = torch.from_numpy(ray_data["inputs"][exactness_indices])
    target = torch.from_numpy(ray_data["field"][exactness_indices, None])
    model.eval()
    base.eval()
    mode_differences = {}
    with torch.no_grad():
        base_prediction = base(x)
        for mode in gc_config["descriptor_modes"]:
            model.set_descriptor_mode(str(mode))
            mode_differences[str(mode)] = float(
                torch.max(torch.abs(model(x) - base_prediction))
            )
    initial_head_weight = float(torch.max(torch.abs(model.head.weight.detach())))
    initial_head_bias = float(torch.max(torch.abs(model.head.bias.detach())))
    model.set_descriptor_mode("geometry")

    current_embedding, _ = model.descriptor_embedding(x, mode="geometry")
    current_shuffled_embedding, _ = model.descriptor_embedding(x, mode="shuffled")
    current_shuffle_distance = float(
        torch.linalg.vector_norm(
            (current_embedding - current_shuffled_embedding).detach()
        )
    )

    base_state = tensor_state(base)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(gc_config["protocol_learning_rate"]),
        weight_decay=0.0,
    )
    model.train()
    head_gradient_first = 0.0
    conditioner_gradient_last = 0.0
    for step in range(int(gc_config["optimizer_steps_checked"])):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(x) - target) ** 2)
        loss.backward()
        if step == 0:
            head_gradient_first = float(
                torch.linalg.vector_norm(model.head.weight.grad).detach()
            )
        conditioner_gradients = [
            parameter.grad.reshape(-1)
            for parameter in model.conditioner.parameters()
            if parameter.grad is not None
        ]
        conditioner_gradient_last = float(
            torch.linalg.vector_norm(torch.cat(conditioner_gradients)).detach()
        )
        optimizer.step()
    model.eval()
    with torch.no_grad():
        trained_correction, _ = model.correction(x)
    base_drift = max(
        float(torch.max(torch.abs(value.detach().cpu() - base_state[name])))
        for name, value in base.state_dict().items()
        if torch.is_tensor(value)
    )

    base_parameters = parameter_count(base)
    total_parameters = parameter_count(model)
    trainable_parameters = parameter_count(model, trainable_only=True)
    adapter_parameters = total_parameters - base_parameters
    control_rows = []
    information = {
        "geometry": "active camera mask + correct sin/cos angle set",
        "mask_only": "K-cardinality only after angle removal; all fixed-K layouts collapse",
        "static": "K fraction only; camera identity and angle removed",
        "shuffled": "mask + angle descriptor from another batch sample",
    }
    for mode in gc_config["descriptor_modes"]:
        control_rows.append(
            {
                "descriptor_mode": str(mode),
                "descriptor_information": information[str(mode)],
                "combined_total_parameters": total_parameters,
                "base_fno_parameters": base_parameters,
                "adapter_parameters": adapter_parameters,
                "trainable_parameters": trainable_parameters,
                "parameter_matched_across_descriptor_modes": True,
                "initial_max_abs_difference_vs_base_fno": mode_differences[str(mode)],
                "current_fixed_k6_sample_varying_descriptor": False,
                "variable_mask_protocol_sample_varying_descriptor": str(mode)
                in {"geometry", "shuffled"},
            }
        )

    gate = {
        "base_checkpoint_path_public": False,
        "base_checkpoint_sha256": sha256_file(checkpoint_path),
        "stratified_exactness_sample_count": int(len(exactness_indices)),
        "maximum_abs_initial_difference_vs_base_fno": max(mode_differences.values()),
        "maximum_abs_initial_head_weight": initial_head_weight,
        "maximum_abs_initial_head_bias": initial_head_bias,
        "current_fixed_protocol_geometry_vs_shuffled_embedding_l2": current_shuffle_distance,
        "head_gradient_norm_after_first_backward": head_gradient_first,
        "conditioner_gradient_norm_after_last_backward": conditioner_gradient_last,
        "optimizer_steps_checked": int(gc_config["optimizer_steps_checked"]),
        "correction_l2_after_checked_steps": float(
            torch.linalg.vector_norm(trained_correction)
        ),
        "maximum_frozen_base_parameter_drift": base_drift,
        "base_fno_parameters": base_parameters,
        "adapter_parameters": adapter_parameters,
        "trainable_parameters": trainable_parameters,
        "combined_total_parameters": total_parameters,
        "base_frozen": all(
            not parameter.requires_grad for parameter in base.parameters()
        ),
        "descriptor_mode_parameter_counts_identical": len(
            {int(row["combined_total_parameters"]) for row in control_rows}
        )
        == 1,
    }
    gate["engineering_gate_pass"] = bool(
        gate["maximum_abs_initial_difference_vs_base_fno"] == 0.0
        and gate["maximum_frozen_base_parameter_drift"] == 0.0
        and gate["head_gradient_norm_after_first_backward"] > 0.0
        and gate["conditioner_gradient_norm_after_last_backward"] > 0.0
        and gate["correction_l2_after_checked_steps"] > 0.0
        and gate["base_frozen"]
        and gate["descriptor_mode_parameter_counts_identical"]
    )
    return gate, control_rows


def add_variable_descriptor_checks(
    gc_gate: dict[str, object],
    experiment: dict,
    dataset_config: dict,
    ray_data: dict[str, np.ndarray],
    manifest_rows: list[dict[str, object]],
    masks: dict[str, np.ndarray],
) -> None:
    gc_config = experiment["gc_sro"]
    names = [str(value) for value in ray_data["input_channel_names"].tolist()]
    base = make_model(
        "fno",
        dataset_config["models"]["fno"],
        int(ray_data["inputs"].shape[1]),
        residual=True,
    )
    set_seed(int(gc_config["model_seed"]))
    model = GeometryConditionedSpectralResidualOperator(
        base_operator=base,
        view_count=int(ray_data["ray_view_channel_count"]),
        mask_channel_start=names.index("camera_0_active"),
        angle_sin_channel_start=int(ray_data["ray_angle_sin_channel_start"]),
        angle_cos_channel_start=int(ray_data["ray_angle_cos_channel_start"]),
        coordinate_channels=tuple(names.index(axis) for axis in ("z", "y", "x")),
        descriptor_hidden=int(gc_config["descriptor_hidden"]),
        descriptor_embedding=int(gc_config["descriptor_embedding"]),
        adapter_hidden=int(gc_config["adapter_hidden"]),
        spectral_modes=tuple(int(value) for value in gc_config["spectral_modes"]),
        freeze_base=True,
    )
    template = torch.from_numpy(ray_data["inputs"][:1])
    variable = make_variable_descriptor_batch(
        template,
        manifest_rows,
        masks,
        ray_data["angles"],
        names.index("camera_0_active"),
        int(ray_data["ray_angle_sin_channel_start"]),
        int(ray_data["ray_angle_cos_channel_start"]),
    )
    model.eval()
    with torch.no_grad():
        geometry_embedding, _ = model.descriptor_embedding(variable, mode="geometry")
        mask_embedding, _ = model.descriptor_embedding(variable, mode="mask_only")
        static_embedding, _ = model.descriptor_embedding(variable, mode="static")
        shuffled_embedding, _ = model.descriptor_embedding(variable, mode="shuffled")
        masks_tensor, sin_tensor, cos_tensor = model.conditioner.components(variable)
        permutation = torch.tensor([8, 2, 5, 0, 7, 1, 6, 4, 3])
        permuted_embedding, _ = model.conditioner.encode_components(
            masks_tensor[:, permutation],
            sin_tensor[:, permutation],
            cos_tensor[:, permutation],
            mode="geometry",
        )
    gc_gate.update(
        {
            "variable_geometry_mean_pairwise_embedding_l2": mean_pairwise_distance(
                geometry_embedding
            ),
            "variable_mask_only_mean_pairwise_embedding_l2": mean_pairwise_distance(
                mask_embedding
            ),
            "variable_static_mean_pairwise_embedding_l2": mean_pairwise_distance(
                static_embedding
            ),
            "variable_geometry_vs_shuffled_embedding_l2": float(
                torch.linalg.vector_norm(geometry_embedding - shuffled_embedding)
            ),
            "variable_geometry_vs_mask_only_embedding_l2": float(
                torch.linalg.vector_norm(geometry_embedding - mask_embedding)
            ),
            "maximum_joint_camera_permutation_embedding_difference": float(
                torch.max(torch.abs(geometry_embedding - permuted_embedding))
            ),
        }
    )
    gc_gate["engineering_gate_pass"] = bool(
        gc_gate["engineering_gate_pass"]
        and gc_gate["variable_geometry_mean_pairwise_embedding_l2"] > 0.0
        and gc_gate["variable_mask_only_mean_pairwise_embedding_l2"] < 1e-7
        and gc_gate["variable_geometry_vs_shuffled_embedding_l2"] > 0.0
        and gc_gate["variable_static_mean_pairwise_embedding_l2"] < 1e-7
        and gc_gate["maximum_joint_camera_permutation_embedding_difference"] < 1e-6
    )


def geometry_diagnostics(
    experiment: dict,
    current_view_masks: np.ndarray,
    manifest_rows: list[dict[str, object]],
    masks: dict[str, np.ndarray],
    geometry_summary: list[dict[str, object]],
    field_spread: list[dict[str, object]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    gate_config = experiment["geometry_gate"]
    current_unique = len(np.unique(np.asarray(current_view_masks), axis=0))
    variable_masks = [masks[str(row["geometry_id"])] for row in manifest_rows]
    variable_unique = len({geometry_id(mask) for mask in variable_masks})
    ordered = sorted(variable_masks, key=geometry_id)
    shuffled = ordered[1:] + ordered[:1]
    changed_fraction = float(
        np.mean([geometry_id(left) != geometry_id(right) for left, right in zip(ordered, shuffled)])
    )
    summary_errors = np.asarray(
        [float(row["mean_field_rel_l2"]) for row in geometry_summary]
    )
    conditions = np.asarray(
        [float(row["operator_condition_nonzero"]) for row in geometry_summary]
    )
    gaps = np.asarray([float(row["maximum_gap_degrees"]) for row in geometry_summary])
    resultants = np.asarray(
        [float(row["first_angular_resultant"]) for row in geometry_summary]
    )
    spreads = np.asarray(
        [float(row["best_to_worst_spread_pct"]) for row in field_spread]
    )
    reference_id = reference_mask_id(experiment["reference_mask"])
    reference_row = next(
        row for row in geometry_summary if str(row["geometry_id"]) == reference_id
    )
    reference_rank = 1 + sum(
        float(row["mean_field_rel_l2"]) < float(reference_row["mean_field_rel_l2"])
        for row in geometry_summary
    )
    error_range_pct = float(
        100.0 * (np.max(summary_errors) - np.min(summary_errors)) / np.mean(summary_errors)
    )
    condition_cv = float(np.std(conditions) / np.mean(conditions))
    median_spread = float(np.median(spreads))
    current_identifiable = bool(
        current_unique >= 2
        if bool(gate_config["current_protocol_must_have_at_least_two_masks"])
        else True
    )
    variable_ready = bool(
        variable_unique >= int(gate_config["minimum_variable_unique_masks"])
        and changed_fraction
        >= float(gate_config["minimum_changed_fraction_after_shuffle"])
        and error_range_pct
        >= float(gate_config["minimum_mean_ridge_error_range_pct"])
        and median_spread
        >= float(gate_config["minimum_median_field_best_worst_spread_pct"])
        and condition_cv >= float(gate_config["minimum_operator_condition_cv"])
    )
    diagnostics = {
        "current_unique_geometry_masks": current_unique,
        "current_geometry_entropy_bits": geometry_entropy_bits(current_view_masks),
        "current_geometry_conditioning_identifiable": current_identifiable,
        "variable_unique_geometry_masks": variable_unique,
        "variable_geometry_entropy_bits_if_balanced": geometry_entropy_bits(
            np.stack(variable_masks)
        ),
        "variable_mean_pairwise_jaccard_distance": mean_pairwise_jaccard_distance(
            variable_masks
        ),
        "variable_changed_fraction_after_deterministic_shuffle": changed_fraction,
        "mean_ridge_error_range_pct_across_geometries": error_range_pct,
        "minimum_geometry_mean_field_rel_l2": float(np.min(summary_errors)),
        "maximum_geometry_mean_field_rel_l2": float(np.max(summary_errors)),
        "median_field_best_worst_spread_pct": median_spread,
        "p10_field_best_worst_spread_pct": float(np.quantile(spreads, 0.10)),
        "p90_field_best_worst_spread_pct": float(np.quantile(spreads, 0.90)),
        "operator_condition_cv": condition_cv,
        "operator_condition_min": float(np.min(conditions)),
        "operator_condition_max": float(np.max(conditions)),
        "spearman_max_gap_vs_mean_field_error": float(
            spearmanr(gaps, summary_errors).statistic
        ),
        "spearman_condition_vs_mean_field_error": float(
            spearmanr(conditions, summary_errors).statistic
        ),
        "spearman_angular_resultant_vs_mean_field_error": float(
            spearmanr(resultants, summary_errors).statistic
        ),
        "reference_geometry_id": reference_id,
        "reference_geometry_mean_field_rel_l2": float(
            reference_row["mean_field_rel_l2"]
        ),
        "reference_geometry_rank_of_28": int(reference_rank),
        "variable_geometry_protocol_ready": variable_ready,
        "geometry_claim_allowed": False,
    }
    gate_rows = [
        {
            "protocol": "current_fixed_k6",
            "unique_masks": current_unique,
            "geometry_entropy_bits": diagnostics["current_geometry_entropy_bits"],
            "shuffle_changed_fraction": 0.0,
            "mean_ridge_error_range_pct": 0.0,
            "median_field_best_worst_spread_pct": 0.0,
            "operator_condition_cv": 0.0,
            "geometry_conditioning_identifiable": current_identifiable,
            "superiority_training_authorized": False,
            "reason": "all 328 development variants share one camera mask",
        },
        {
            "protocol": "v3h_variable_geometry_development",
            "unique_masks": variable_unique,
            "geometry_entropy_bits": diagnostics[
                "variable_geometry_entropy_bits_if_balanced"
            ],
            "shuffle_changed_fraction": changed_fraction,
            "mean_ridge_error_range_pct": error_range_pct,
            "median_field_best_worst_spread_pct": median_spread,
            "operator_condition_cv": condition_cv,
            "geometry_conditioning_identifiable": variable_ready,
            "superiority_training_authorized": False,
            "reason": "functional pilot allowed after dataset builder; no superiority or blind claim",
        },
    ]
    return diagnostics, gate_rows


def plot_results(
    manifest_rows: list[dict[str, object]],
    masks: dict[str, np.ndarray],
    geometry_summary: list[dict[str, object]],
    field_spread: list[dict[str, object]],
    angles: np.ndarray,
    output_path: Path,
) -> None:
    partition_order = {"train": 0, "validation": 1, "geometry_ood": 2, "stress": 3}
    ordered_manifest = sorted(
        manifest_rows,
        key=lambda row: (
            partition_order[str(row["partition"])],
            float(row["maximum_gap_degrees"]),
            str(row["geometry_id"]),
        ),
    )
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    matrix = np.stack([masks[str(row["geometry_id"])] for row in ordered_manifest])
    axes[0].imshow(matrix, aspect="auto", cmap="YlGn", vmin=0.0, vmax=1.0)
    axes[0].set_xticks(range(len(angles)), [f"{float(value):g}°" for value in angles])
    axes[0].set_yticks(
        range(len(ordered_manifest)),
        [f"{row['partition'][:3]} · {row['mask_bits']}" for row in ordered_manifest],
        fontsize=6,
    )
    axes[0].set_xlabel("canonical camera angle; 60° is always audit-only")
    axes[0].set_title("28 legal K=6 development geometries")

    colors = {
        "train": "#287a67",
        "validation": "#315f93",
        "geometry_ood": "#9a6b17",
        "stress": "#a94f3e",
    }
    for partition in partition_order:
        rows = [row for row in geometry_summary if str(row["partition"]) == partition]
        axes[1].scatter(
            [float(row["operator_condition_nonzero"]) for row in rows],
            [float(row["mean_field_rel_l2"]) for row in rows],
            s=[30.0 + float(row["maximum_gap_degrees"]) for row in rows],
            color=colors[partition],
            alpha=0.82,
            label=partition,
        )
    reference = next(
        row for row in geometry_summary if bool(row["reference_fixed_k6_geometry"])
    )
    axes[1].scatter(
        [float(reference["operator_condition_nonzero"])],
        [float(reference["mean_field_rel_l2"])],
        marker="*",
        s=240,
        color="black",
        label="current fixed K=6",
    )
    axes[1].set_xlabel("nonzero operator condition number")
    axes[1].set_ylabel("mean validation ridge field relative L2")
    axes[1].set_title("Camera layout changes inverse difficulty")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    best = [float(row["best_field_rel_l2"]) for row in field_spread]
    fixed = [float(row["reference_field_rel_l2"]) for row in field_spread]
    worst = [float(row["worst_field_rel_l2"]) for row in field_spread]
    axes[2].boxplot(
        [best, fixed, worst],
        tick_labels=["best of 28", "current fixed", "worst of 28"],
        showfliers=False,
    )
    for left, middle, right in zip(best, fixed, worst):
        axes[2].plot([1, 2, 3], [left, middle, right], color="#8da3a0", alpha=0.12)
    axes[2].set_ylabel("per-field ridge relative L2")
    axes[2].set_title("40 validation fields; geometry-only spread")
    axes[2].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_checksums(output_dir: Path) -> None:
    lines = []
    for filename in CHECKSUM_FILES:
        digest = hashlib.sha256((output_dir / filename).read_bytes()).hexdigest()
        lines.append(f"{digest}  {filename}")
    (output_dir / "v3h_gc_sro_geometry_checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="ascii"
    )


def main() -> None:
    args = parse_args()
    experiment = read_json(args.config)
    dataset_config_path = CONFIG_ROOT / str(experiment["dataset_config"])
    dataset_config = read_json(dataset_config_path)
    dataset_path = ROOT / "results" / str(experiment["dataset_npz"])
    generate_dataset(dataset_config, dataset_path, force=bool(args.force_data))
    data = load_npz(dataset_path)
    output_dir = args.output_dir or ROOT / "results" / str(experiment["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_id = reference_mask_id(experiment["reference_mask"])
    manifest_rows, masks = build_geometry_manifest(
        data["angles"],
        data["forward_matrix"],
        int(experiment["total_budget"]),
        int(experiment["audit_query_index"]),
        float(experiment["geometry_period_degrees"]),
    )
    manifest_rows = assign_geometry_partitions(
        manifest_rows,
        reference_id,
        int(experiment["partition_seed"]),
        {key: int(value) for key, value in experiment["partition_counts"].items()},
    )
    split_names = [str(value) for value in data["split_names"].tolist()]
    evaluation_split_id = split_names.index(str(experiment["geometry_evaluation_split"]))
    evaluation_indices = np.flatnonzero(data["split_id"] == evaluation_split_id)
    sample_rows = evaluate_geometry_ridge(
        data,
        evaluation_indices,
        manifest_rows,
        masks,
        float(experiment["ridge_relative"]),
        int(experiment["total_budget"]),
        int(experiment["audit_query_index"]),
    )
    geometry_summary = summarize_geometry_errors(sample_rows, manifest_rows)
    field_spread = summarize_field_geometry_spread(sample_rows, reference_id)

    current_direct = prepare_direct_operator_data(
        data,
        [int(experiment["total_budget"])],
        fixed_query_index=4,
        audit_query_index=int(experiment["audit_query_index"]),
    )
    diagnostics, geometry_gate_rows = geometry_diagnostics(
        experiment,
        current_direct["view_mask"],
        manifest_rows,
        masks,
        geometry_summary,
        field_spread,
    )
    ridge_data = replace_lift_with_ridge(
        current_direct,
        {int(experiment["total_budget"]): float(experiment["ridge_relative"])},
    )
    ray_data = append_ray_view_channels(ridge_data)
    gc_gate, control_rows = build_gc_sro_gate(experiment, dataset_config, ray_data)
    add_variable_descriptor_checks(
        gc_gate,
        experiment,
        dataset_config,
        ray_data,
        manifest_rows,
        masks,
    )

    status = (
        "CURRENT_FIXED_GEOMETRY_FAIL_VARIABLE_PROTOCOL_READY_GC_SRO_ENGINEERING_PASS"
        if not diagnostics["current_geometry_conditioning_identifiable"]
        and diagnostics["variable_geometry_protocol_ready"]
        and gc_gate["engineering_gate_pass"]
        else "V3H_GEOMETRY_OR_ENGINEERING_GATE_FAIL"
    )
    training_decision = {
        "train_gc_sro_on_current_fixed_k6": False,
        "reason_current": (
            "correct, shuffled and constant acquisition geometry are not sample-identifiable "
            "when every field uses one fixed mask"
        ),
        "build_variable_geometry_functional_pilot": bool(
            diagnostics["variable_geometry_protocol_ready"]
            and gc_gate["engineering_gate_pass"]
        ),
        "functional_pilot_controls": [
            "locked_fno",
            "parameter_matched_wider_fno",
            "static_spectral_adapter",
            "mask_only_gc_sro",
            "shuffled_geometry_gc_sro",
            "correct_geometry_gc_sro",
        ],
        "superiority_training_authorized": False,
        "blind_final_opened": False,
        "real_geometry_required_before_publication_claim": True,
    }
    provenance = {
        "experiment_config_sha256": sha256_file(args.config),
        "dataset_config_sha256": sha256_file(dataset_config_path),
        "dataset_npz_sha256": sha256_file(dataset_path),
        "training_script_sha256": sha256_file(Path(__file__).resolve()),
        "variable_geometry_script_sha256": sha256_file(ROOT / "variable_geometry.py"),
        "model_script_sha256": sha256_file(ROOT / "own_algorithm_models.py"),
        "base_checkpoint_sha256": gc_gate["base_checkpoint_sha256"],
        "dataset_npz_public": False,
        "checkpoint_weights_public": False,
    }
    dashboard = {
        "experiment": experiment["name"],
        "scientific_status": status,
        "development_only": True,
        "superiority_tested": False,
        "blind_final_opened": False,
        "geometry_diagnostics": diagnostics,
        "geometry_partition_counts": partition_counts(manifest_rows),
        "geometry_evaluation_field_count": int(len(evaluation_indices)),
        "geometry_sample_metric_row_count": len(sample_rows),
        "gc_sro_engineering_gate": gc_gate,
        "control_contract": control_rows,
        "training_decision": training_decision,
        "provenance": provenance,
    }
    report = {
        "status": "completed",
        "scientific_status": status,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "protocol": {
            "audit_camera_reserved_in_all_28_masks": True,
            "partition_uses_no_neural_model_or_gc_sro_metric": True,
            "stress_masks_selected_by_geometry_and_operator_descriptors_only": True,
            "ridge_relative_inherited_from_v3f_validation": float(
                experiment["ridge_relative"]
            ),
            "same_full_view_noise_realization_across_masks_per_field": True,
            "gc_sro_descriptor_controls_share_parameters": True,
            "gc_sro_starts_exactly_at_local_locked_fno": True,
            "current_fixed_geometry_not_used_for_superiority_claim": True,
        },
        "geometry_diagnostics": diagnostics,
        "gc_sro_engineering_gate": gc_gate,
        "training_decision": training_decision,
        "provenance": provenance,
        "claims_boundary": [
            "The current K=6 development data use one fixed camera mask, so acquisition-geometry benefit is not identifiable there.",
            "The 28 masks vary missing-camera layout over one canonical nine-angle linear projector; they do not represent real intrinsics, extrinsics, calibration drift or nonlinear rays.",
            "Ridge difficulty is measured on 40 already-inspected validation fields and is a development diagnostic, not a blind test.",
            "GC-SRO v0 passes only zero-init, frozen-base, gradient-flow, permutation and matched-control engineering checks.",
            "No GC-SRO reconstruction superiority, novelty, real-flow transfer, NeRIF acceleration or publication claim was tested.",
            "The current fixed mask ranks within the 28-mask validation audit only; this does not establish an optimal experimental camera design.",
        ],
    }

    write_csv(output_dir / "v3h_geometry_manifest.csv", manifest_rows)
    write_csv(output_dir / "v3h_geometry_field_errors.csv", sample_rows)
    write_csv(output_dir / "v3h_geometry_summary.csv", geometry_summary)
    write_csv(output_dir / "v3h_field_geometry_spread.csv", field_spread)
    write_csv(output_dir / "v3h_geometry_gate.csv", geometry_gate_rows)
    write_csv(output_dir / "v3h_gc_sro_control_contract.csv", control_rows)
    plot_results(
        manifest_rows,
        masks,
        geometry_summary,
        field_spread,
        data["angles"],
        output_dir / "t16_v3h_gc_sro_geometry_gate.png",
    )
    (output_dir / "v3h_gc_sro_geometry_dashboard.json").write_text(
        json.dumps(dashboard, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_dir / "v3h_gc_sro_geometry_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_checksums(output_dir)
    print(
        json.dumps(
            {
                "scientific_status": status,
                "current_unique_masks": diagnostics["current_unique_geometry_masks"],
                "variable_unique_masks": diagnostics["variable_unique_geometry_masks"],
                "mean_ridge_error_range_pct": diagnostics[
                    "mean_ridge_error_range_pct_across_geometries"
                ],
                "median_field_best_worst_spread_pct": diagnostics[
                    "median_field_best_worst_spread_pct"
                ],
                "reference_geometry_rank_of_28": diagnostics[
                    "reference_geometry_rank_of_28"
                ],
                "gc_sro_engineering_gate_pass": gc_gate["engineering_gate_pass"],
                "build_variable_geometry_functional_pilot": training_decision[
                    "build_variable_geometry_functional_pilot"
                ],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
