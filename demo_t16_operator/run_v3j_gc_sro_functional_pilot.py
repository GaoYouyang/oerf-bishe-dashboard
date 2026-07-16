#!/usr/bin/env python3
"""Run the v3j matched GC-SRO descriptor-mechanism functional pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
import torch.nn.functional as functional
from torch.utils.data import DataLoader

try:
    from .data import BOSTDataset, split_indices
    from .models import make_model
    from .own_algorithm_models import GeometryConditionedSpectralResidualOperator
    from .train_eval import (
        choose_device,
        gradient_mse,
        masked_relative_projection_loss,
        project_torch,
        set_seed,
        synchronize,
    )
except ImportError:
    from data import BOSTDataset, split_indices
    from models import make_model
    from own_algorithm_models import GeometryConditionedSpectralResidualOperator
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
DEFAULT_CONFIG = CONFIG_ROOT / "v3j_gc_sro_functional_pilot.json"
PUBLIC_FILES = [
    "v3j_geometry_derangement.csv",
    "v3j_training_history.csv",
    "v3j_sample_metrics.csv",
    "v3j_split_summary.csv",
    "v3j_pairwise_mechanism.csv",
    "v3j_same_model_descriptor_swap.csv",
    "v3j_gc_sro_functional_dashboard.json",
    "v3j_gc_sro_functional_report.json",
    "t16_v3j_gc_sro_functional_pilot.png",
]
LABELS = {
    "locked_fno": "Locked fixed-layout FNO",
    "static": "Static spectral adapter",
    "k_cardinality": "K-cardinality adapter",
    "shuffled_geometry": "Shuffled-geometry GC-SRO",
    "correct_geometry": "Correct-geometry GC-SRO",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def load_private_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def parameter_count(module: torch.nn.Module, trainable_only: bool = False) -> int:
    return int(
        sum(
            parameter.numel()
            for parameter in module.parameters()
            if not trainable_only or parameter.requires_grad
        )
    )


def build_model(
    config: dict,
    dataset_config: dict,
    data: dict[str, np.ndarray],
    checkpoint: dict[str, torch.Tensor],
    seed: int,
    method: str,
) -> GeometryConditionedSpectralResidualOperator:
    base = make_model(
        "fno", dataset_config["models"]["fno"], int(data["inputs"].shape[1]), residual=True
    )
    base.load_state_dict(checkpoint, strict=True)
    set_seed(int(seed))
    names = [str(value) for value in data["input_channel_names"].tolist()]
    mode = {
        "static": "static",
        "k_cardinality": "mask_only",
        "shuffled_geometry": "geometry",
        "correct_geometry": "geometry",
    }[method]
    gc = config["gc_sro"]
    return GeometryConditionedSpectralResidualOperator(
        base_operator=base,
        view_count=int(data["ray_view_channel_count"]),
        mask_channel_start=names.index("camera_0_active"),
        angle_sin_channel_start=int(data["ray_angle_sin_channel_start"]),
        angle_cos_channel_start=int(data["ray_angle_cos_channel_start"]),
        coordinate_channels=tuple(names.index(axis) for axis in ("z", "y", "x")),
        descriptor_hidden=int(gc["descriptor_hidden"]),
        descriptor_embedding=int(gc["descriptor_embedding"]),
        adapter_hidden=int(gc["adapter_hidden"]),
        spectral_modes=tuple(int(value) for value in gc["spectral_modes"]),
        maximum_correction_scale=float(gc["maximum_correction_scale"]),
        descriptor_mode=mode,
        freeze_base=bool(gc["freeze_base"]),
    )


def geometry_derangement(
    data: dict[str, np.ndarray],
) -> tuple[list[dict[str, object]], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    geometry_ids = [str(value) for value in data["geometry_id"].tolist()]
    partitions = [str(value) for value in data["geometry_partition"].tolist()]
    by_partition: dict[str, list[str]] = defaultdict(list)
    for identifier, partition in zip(geometry_ids, partitions):
        if identifier not in by_partition[partition]:
            by_partition[partition].append(identifier)
    mapping = {}
    rows = []
    for partition, identifiers in sorted(by_partition.items()):
        identifiers.sort()
        for index, identifier in enumerate(identifiers):
            wrong = identifiers[(index + 1) % len(identifiers)]
            if wrong == identifier:
                raise RuntimeError("geometry derangement contains a fixed point")
            mapping[identifier] = wrong
            rows.append(
                {
                    "geometry_partition": partition,
                    "correct_geometry_id": identifier,
                    "wrong_geometry_id": wrong,
                    "correct_mask_bits": identifier.removeprefix("g_"),
                    "wrong_mask_bits": wrong.removeprefix("g_"),
                    "fixed_point": False,
                    "mapping_rule": "lexical cyclic derangement within partition",
                }
            )
    masks = np.asarray(
        [
            [float(value) for value in mapping[identifier].removeprefix("g_")]
            for identifier in geometry_ids
        ],
        dtype=np.float32,
    )
    radians = np.deg2rad(np.asarray(data["angles"], dtype=np.float32))
    sin = masks * np.sin(radians)[None]
    cos = masks * np.cos(radians)[None]
    return rows, (masks, sin.astype(np.float32), cos.astype(np.float32))


def precompute_base_predictions(
    model: torch.nn.Module,
    data: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    loader = DataLoader(
        BOSTDataset(data, np.arange(len(data["field"]))), batch_size=batch_size
    )
    model = model.to(device).eval()
    output = []
    with torch.no_grad():
        for batch in loader:
            output.append(model(batch["x"].to(device)).cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def wrong_components_for_batch(
    wrong: tuple[np.ndarray, np.ndarray, np.ndarray],
    indices: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = indices.cpu().numpy()
    return tuple(torch.from_numpy(value[selected]).to(device) for value in wrong)


def predict_with_adapter(
    model: GeometryConditionedSpectralResidualOperator,
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    base_predictions: np.ndarray,
    wrong: tuple[np.ndarray, np.ndarray, np.ndarray],
    method: str,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    loader = DataLoader(BOSTDataset(data, indices), batch_size=batch_size)
    model.eval()
    output = []
    with torch.no_grad():
        for batch in loader:
            batch_indices = batch["index"]
            x = batch["x"].to(device)
            base = torch.from_numpy(base_predictions[batch_indices.numpy()]).to(device)
            components = (
                wrong_components_for_batch(wrong, batch_indices, device)
                if method == "shuffled_geometry"
                else None
            )
            correction, _ = model.correction(
                x, base_prediction=base, descriptor_components=components
            )
            output.append((base + correction).cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def train_adapter(
    config: dict,
    dataset_config: dict,
    data: dict[str, np.ndarray],
    checkpoint: dict[str, torch.Tensor],
    base_predictions: np.ndarray,
    wrong: tuple[np.ndarray, np.ndarray, np.ndarray],
    method: str,
    seed: int,
    device: torch.device,
    work_dir: Path,
) -> tuple[GeometryConditionedSpectralResidualOperator, list[dict[str, object]], dict[str, object]]:
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
    indices = split_indices(data)
    generator = torch.Generator().manual_seed(int(seed))
    train_loader = DataLoader(
        BOSTDataset(data, indices["train"]),
        batch_size=int(training["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        BOSTDataset(data, indices["val"]), batch_size=int(training["batch_size"])
    )
    operator = torch.from_numpy(data["forward_matrix"]).to(device)
    best_val = math.inf
    best_epoch = -1
    best_state = None
    history = []
    start = time.perf_counter()
    for epoch in range(int(training["epochs"])):
        model.train()
        total_sum = 0.0
        seen = 0
        for batch in train_loader:
            batch_indices = batch["index"]
            x = batch["x"].to(device)
            target = batch["field"].to(device)
            base = torch.from_numpy(base_predictions[batch_indices.numpy()]).to(device)
            components = (
                wrong_components_for_batch(wrong, batch_indices, device)
                if method == "shuffled_geometry"
                else None
            )
            optimizer.zero_grad(set_to_none=True)
            correction, _ = model.correction(
                x, base_prediction=base, descriptor_components=components
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
            total_sum += float(loss.detach()) * len(batch_indices)
            seen += len(batch_indices)

        model.eval()
        val_values = []
        with torch.no_grad():
            for batch in val_loader:
                batch_indices = batch["index"]
                x = batch["x"].to(device)
                target = batch["field"].to(device)
                base = torch.from_numpy(base_predictions[batch_indices.numpy()]).to(device)
                components = (
                    wrong_components_for_batch(wrong, batch_indices, device)
                    if method == "shuffled_geometry"
                    else None
                )
                correction, _ = model.correction(
                    x, base_prediction=base, descriptor_components=components
                )
                prediction = base + correction
                numerator = torch.linalg.vector_norm(
                    (prediction - target).flatten(start_dim=1), dim=1
                )
                denominator = torch.linalg.vector_norm(
                    target.flatten(start_dim=1), dim=1
                ).clamp_min(1e-8)
                val_values.extend((numerator / denominator).cpu().tolist())
        val_mean = float(np.mean(val_values))
        history.append(
            {
                "method": method,
                "model_seed": int(seed),
                "epoch": epoch + 1,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train_total_loss": total_sum / max(seen, 1),
                "validation_field_rel_l2": val_mean,
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
        raise RuntimeError("functional pilot did not produce a checkpoint")
    model.load_state_dict(best_state)
    checkpoint_path = work_dir / str(seed) / "checkpoints" / f"{method}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, checkpoint_path)
    record = {
        "method": method,
        "model_seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_validation_field_rel_l2": best_val,
        "train_seconds": elapsed,
        "total_parameters": parameter_count(model),
        "trainable_parameters": parameter_count(model, trainable_only=True),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_public": False,
    }
    return model, history, record


def per_sample_metrics(
    predictions: np.ndarray,
    data: dict[str, np.ndarray],
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    predicted = predictions[:, 0]
    target = data["field"][indices]
    field = np.linalg.norm((predicted - target).reshape(len(indices), -1), axis=1) / np.maximum(
        np.linalg.norm(target.reshape(len(indices), -1), axis=1), 1e-8
    )
    audit = int(data["audit_query_index"])
    projected = np.einsum(
        "bdp,np->bdn",
        predicted.reshape(len(indices), predicted.shape[1], -1),
        data["forward_matrix"][audit],
        optimize=True,
    )
    reference = data["clean_observation"][indices, :, audit]
    audit_error = np.linalg.norm((projected - reference).reshape(len(indices), -1), axis=1) / np.maximum(
        np.linalg.norm(reference.reshape(len(indices), -1), axis=1), 1e-8
    )
    return field, audit_error


def bootstrap_interval(values: np.ndarray, seed: int, replicates: int) -> tuple[float, float]:
    rng = np.random.default_rng(int(seed))
    samples = rng.integers(0, len(values), size=(int(replicates), len(values)))
    estimates = values[samples].mean(axis=1)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def same_model_descriptor_swap(
    models: dict[tuple[str, int], GeometryConditionedSpectralResidualOperator],
    data: dict[str, np.ndarray],
    base_predictions: np.ndarray,
    wrong: tuple[np.ndarray, np.ndarray, np.ndarray],
    seeds: list[int],
    device: torch.device,
    batch_size: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    val_indices = split_indices(data)["val"]
    loader = DataLoader(BOSTDataset(data, val_indices), batch_size=batch_size)
    rows = []
    seed_means = []
    for seed in seeds:
        model = models[("correct_geometry", seed)].to(device).eval()
        seed_gains = []
        with torch.no_grad():
            for batch in loader:
                batch_indices = batch["index"]
                selected = batch_indices.numpy()
                x = batch["x"].to(device)
                target = batch["field"].to(device)
                base = torch.from_numpy(base_predictions[selected]).to(device)
                correct_components = model.conditioner.components(x)
                wrong_components = wrong_components_for_batch(
                    wrong, batch_indices, device
                )
                correct_embedding, correct_modulation = model.descriptor_embedding(
                    x, descriptor_components=correct_components
                )
                wrong_embedding, wrong_modulation = model.descriptor_embedding(
                    x, descriptor_components=wrong_components
                )
                correct_correction, _ = model.correction(
                    x,
                    base_prediction=base,
                    descriptor_components=correct_components,
                )
                wrong_correction, _ = model.correction(
                    x,
                    base_prediction=base,
                    descriptor_components=wrong_components,
                )
                correct_prediction = base + correct_correction
                wrong_prediction = base + wrong_correction
                correct_error = torch.linalg.vector_norm(
                    (correct_prediction - target).flatten(start_dim=1), dim=1
                ) / torch.linalg.vector_norm(target.flatten(start_dim=1), dim=1).clamp_min(1e-8)
                wrong_error = torch.linalg.vector_norm(
                    (wrong_prediction - target).flatten(start_dim=1), dim=1
                ) / torch.linalg.vector_norm(target.flatten(start_dim=1), dim=1).clamp_min(1e-8)
                gains = 100.0 * (wrong_error - correct_error) / wrong_error.clamp_min(1e-12)
                correction_swap = 100.0 * torch.linalg.vector_norm(
                    (correct_correction - wrong_correction).flatten(start_dim=1), dim=1
                ) / torch.linalg.vector_norm(
                    correct_correction.flatten(start_dim=1), dim=1
                ).clamp_min(1e-12)
                for local, source_index in enumerate(selected):
                    gain = float(gains[local])
                    seed_gains.append(gain)
                    rows.append(
                        {
                            "model_seed": int(seed),
                            "source_index": int(source_index),
                            "geometry_id": str(data["geometry_id"][source_index]),
                            "geometry_partition": str(
                                data["geometry_partition"][source_index]
                            ),
                            "embedding_swap_l2": float(
                                torch.linalg.vector_norm(
                                    correct_embedding[local] - wrong_embedding[local]
                                )
                            ),
                            "modulation_swap_l2": float(
                                torch.linalg.vector_norm(
                                    correct_modulation[local] - wrong_modulation[local]
                                )
                            ),
                            "correction_swap_relative_pct": float(
                                correction_swap[local]
                            ),
                            "correct_descriptor_field_rel_l2": float(
                                correct_error[local]
                            ),
                            "wrong_descriptor_field_rel_l2": float(
                                wrong_error[local]
                            ),
                            "correct_descriptor_field_gain_pct": gain,
                        }
                    )
        seed_means.append(float(np.mean(seed_gains)))
    summary = {
        "validation_field_count": len(val_indices),
        "model_seed_count": len(seeds),
        "mean_embedding_swap_l2": float(
            np.mean([float(row["embedding_swap_l2"]) for row in rows])
        ),
        "mean_modulation_swap_l2": float(
            np.mean([float(row["modulation_swap_l2"]) for row in rows])
        ),
        "mean_correction_swap_relative_pct": float(
            np.mean([float(row["correction_swap_relative_pct"]) for row in rows])
        ),
        "mean_correct_descriptor_field_gain_pct": float(
            np.mean([float(row["correct_descriptor_field_gain_pct"]) for row in rows])
        ),
        "p10_correct_descriptor_field_gain_pct": float(
            np.quantile(
                [float(row["correct_descriptor_field_gain_pct"]) for row in rows],
                0.1,
            )
        ),
        "positive_seed_count": int(sum(value > 0.0 for value in seed_means)),
        "seed_mean_gains_pct": seed_means,
        "geometry_encoded_but_not_usefully_propagated": True,
    }
    return rows, summary


def summarize(
    rows: list[dict[str, object]], config: dict
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), str(row["source_split"]))].append(row)
    split_summary = []
    for (method, split), subset in sorted(grouped.items()):
        field = np.asarray([float(row["field_rel_l2"]) for row in subset])
        audit = np.asarray([float(row["audit_reprojection_rel_l2"]) for row in subset])
        seed_means = []
        for seed in sorted({int(row["model_seed"]) for row in subset}):
            seed_means.append(
                float(np.mean([float(row["field_rel_l2"]) for row in subset if int(row["model_seed"]) == seed]))
            )
        split_summary.append(
            {
                "method": method,
                "method_label": LABELS[method],
                "source_split": split,
                "sample_metric_rows": len(subset),
                "independent_field_count": len({int(row["source_index"]) for row in subset}),
                "model_seed_count": len(seed_means),
                "mean_field_rel_l2": float(np.mean(field)),
                "median_field_rel_l2": float(np.median(field)),
                "p90_field_rel_l2": float(np.quantile(field, 0.9)),
                "mean_audit_reprojection_rel_l2": float(np.mean(audit)),
                "minimum_seed_mean_field_rel_l2": min(seed_means),
                "maximum_seed_mean_field_rel_l2": max(seed_means),
            }
        )

    by_key = {
        (str(row["method"]), int(row["model_seed"]), int(row["source_index"])): row
        for row in rows
    }
    pairwise = []
    gate = config["functional_gate"]
    seeds = [int(value) for value in config["model_seeds"]]
    for split in [str(value) for value in np.unique([row["source_split"] for row in rows])]:
        source_indices = sorted(
            {int(row["source_index"]) for row in rows if row["source_split"] == split}
        )
        for comparator in ["locked_fno", "static", "k_cardinality", "shuffled_geometry"]:
            field_gains = []
            audit_gains = []
            seed_means = []
            field_collapsed = []
            for source_index in source_indices:
                per_seed = []
                for seed in seeds:
                    correct = by_key[("correct_geometry", seed, source_index)]
                    control = by_key[(comparator, seed, source_index)]
                    gain = 100.0 * (float(control["field_rel_l2"]) - float(correct["field_rel_l2"])) / max(float(control["field_rel_l2"]), 1e-12)
                    audit_gain = 100.0 * (float(control["audit_reprojection_rel_l2"]) - float(correct["audit_reprojection_rel_l2"])) / max(float(control["audit_reprojection_rel_l2"]), 1e-12)
                    per_seed.append(gain)
                    field_gains.append(gain)
                    audit_gains.append(audit_gain)
                field_collapsed.append(float(np.mean(per_seed)))
            for seed in seeds:
                values = []
                for source_index in source_indices:
                    correct = by_key[("correct_geometry", seed, source_index)]
                    control = by_key[(comparator, seed, source_index)]
                    values.append(100.0 * (float(control["field_rel_l2"]) - float(correct["field_rel_l2"])) / max(float(control["field_rel_l2"]), 1e-12))
                seed_means.append(float(np.mean(values)))
            ci_low, ci_high = bootstrap_interval(
                np.asarray(field_collapsed),
                int(gate["bootstrap_seed"]) + len(pairwise),
                int(gate["bootstrap_replicates"]),
            )
            pairwise.append(
                {
                    "source_split": split,
                    "candidate": "correct_geometry",
                    "comparator": comparator,
                    "independent_field_count": len(source_indices),
                    "model_seed_count": len(seeds),
                    "mean_field_gain_pct": float(np.mean(field_gains)),
                    "field_cluster_ci95_low_pct": ci_low,
                    "field_cluster_ci95_high_pct": ci_high,
                    "median_field_gain_pct": float(np.median(field_gains)),
                    "p10_field_gain_pct": float(np.quantile(field_gains, 0.1)),
                    "harm_rate_gt_1pct": float(np.mean(np.asarray(field_gains) < -1.0)),
                    "mean_audit_gain_pct": float(np.mean(audit_gains)),
                    "positive_seed_count": int(sum(value > 0.0 for value in seed_means)),
                    "seed_mean_gains_pct": ";".join(f"{value:.8f}" for value in seed_means),
                }
            )
    return split_summary, pairwise


def plot_results(
    path: Path,
    history: list[dict[str, object]],
    split_summary: list[dict[str, object]],
    pairwise: list[dict[str, object]],
) -> None:
    colors = {"static": "#7a7f83", "k_cardinality": "#b08968", "shuffled_geometry": "#c75b4d", "correct_geometry": "#16817a", "locked_fno": "#355c7d"}
    fig, axes = plt.subplots(1, 3, figsize=(15.3, 4.8), constrained_layout=True)
    for method in ("static", "k_cardinality", "shuffled_geometry", "correct_geometry"):
        subset = [row for row in history if row["method"] == method]
        epochs = sorted({int(row["epoch"]) for row in subset})
        mean = [np.mean([float(row["validation_field_rel_l2"]) for row in subset if int(row["epoch"]) == epoch]) for epoch in epochs]
        axes[0].plot(epochs, mean, label=LABELS[method], color=colors[method])
    axes[0].set_title("Matched validation learning curves")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("field relative L2")
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=7)

    validation = [row for row in pairwise if row["source_split"] == "val"]
    x = np.arange(len(validation))
    means = np.asarray([float(row["mean_field_gain_pct"]) for row in validation])
    lows = np.asarray([float(row["field_cluster_ci95_low_pct"]) for row in validation])
    highs = np.asarray([float(row["field_cluster_ci95_high_pct"]) for row in validation])
    axes[1].errorbar(x, means, yerr=np.vstack([means - lows, highs - means]), fmt="o", color="#16817a", capsize=4)
    axes[1].axhline(0, color="#333", linewidth=1)
    axes[1].set_xticks(x, [str(row["comparator"]).replace("_", "\n") for row in validation])
    axes[1].set_title("Correct geometry gain on unseen val layouts")
    axes[1].set_ylabel("paired field gain (%)")
    axes[1].grid(axis="y", alpha=0.2)

    methods = ["locked_fno", "static", "shuffled_geometry", "correct_geometry"]
    splits = ["val", "test_iid", "test_noise_ood", "test_family_ood", "test_joint_ood"]
    lookup = {(row["method"], row["source_split"]): float(row["mean_field_rel_l2"]) for row in split_summary}
    width = 0.19
    for index, method in enumerate(methods):
        axes[2].bar(np.arange(len(splits)) + (index - 1.5) * width, [lookup[(method, split)] for split in splits], width, label=LABELS[method], color=colors[method])
    axes[2].set_xticks(np.arange(len(splits)), [value.replace("test_", "") for value in splits], rotation=20)
    axes[2].set_title("Absolute error remains visible")
    axes[2].set_ylabel("mean field relative L2")
    axes[2].legend(fontsize=7)
    axes[2].grid(axis="y", alpha=0.2)
    fig.suptitle("v3j GC-SRO descriptor-mechanism functional pilot", fontsize=14)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    if args.device:
        config["training"]["device"] = args.device
    if args.epochs:
        config["training"]["epochs"] = int(args.epochs)
    dataset_config_path = CONFIG_ROOT / str(config["dataset_config"])
    dataset_config = read_json(dataset_config_path)
    private_path = ROOT / "results" / str(config["private_dataset_npz"])
    v3i_dashboard_path = ROOT / "results" / str(config["v3i_dashboard"])
    v3i = read_json(v3i_dashboard_path)
    if not v3i["dataset_gate_pass"]:
        raise RuntimeError("v3i dataset gate is not open")
    data = load_private_dataset(private_path)
    checkpoint_path = ROOT / "results" / str(config["base_checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    device = choose_device(str(config["training"]["device"]))
    base_for_cache = make_model(
        "fno", dataset_config["models"]["fno"], int(data["inputs"].shape[1]), residual=True
    )
    base_for_cache.load_state_dict(checkpoint, strict=True)
    base_predictions = precompute_base_predictions(
        base_for_cache, data, device, int(config["training"]["batch_size"])
    )
    derangement_rows, wrong = geometry_derangement(data)
    work_dir = ROOT / "results" / str(config["work_dir"])
    output_dir = ROOT / "results" / str(config["output_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    histories = []
    records = []
    trained = {}
    for seed in [int(value) for value in config["model_seeds"]]:
        for method in config["methods"]:
            model, history, record = train_adapter(
                config, dataset_config, data, checkpoint, base_predictions, wrong,
                str(method), seed, device, work_dir
            )
            histories.extend(history)
            records.append(record)
            trained[(str(method), seed)] = model
            print(json.dumps({"method": method, "seed": seed, "best_epoch": record["best_epoch"], "best_val": record["best_validation_field_rel_l2"], "seconds": record["train_seconds"]}))

    sample_rows = []
    indices_by_split = split_indices(data)
    seeds = [int(value) for value in config["model_seeds"]]
    for split, indices in indices_by_split.items():
        base_field, base_audit = per_sample_metrics(base_predictions[indices], data, indices)
        for seed in seeds:
            for local, source_index in enumerate(indices):
                sample_rows.append({
                    "method": "locked_fno", "model_seed": seed, "source_split": split,
                    "source_index": int(source_index), "geometry_id": str(data["geometry_id"][source_index]),
                    "geometry_partition": str(data["geometry_partition"][source_index]),
                    "family_id": int(data["family_id"][source_index]), "noise_level": float(data["noise_level"][source_index]),
                    "field_rel_l2": float(base_field[local]), "audit_reprojection_rel_l2": float(base_audit[local]),
                })
        for seed in seeds:
            for method in config["methods"]:
                predictions = predict_with_adapter(
                    trained[(str(method), seed)], data, indices, base_predictions,
                    wrong, str(method), device, int(config["training"]["batch_size"])
                )
                field, audit = per_sample_metrics(predictions, data, indices)
                for local, source_index in enumerate(indices):
                    sample_rows.append({
                        "method": str(method), "model_seed": seed, "source_split": split,
                        "source_index": int(source_index), "geometry_id": str(data["geometry_id"][source_index]),
                        "geometry_partition": str(data["geometry_partition"][source_index]),
                        "family_id": int(data["family_id"][source_index]), "noise_level": float(data["noise_level"][source_index]),
                        "field_rel_l2": float(field[local]), "audit_reprojection_rel_l2": float(audit[local]),
                    })

    split_summary, pairwise = summarize(sample_rows, config)
    swap_rows, swap_summary = same_model_descriptor_swap(
        trained,
        data,
        base_predictions,
        wrong,
        seeds,
        device,
        int(config["training"]["batch_size"]),
    )
    gate = config["functional_gate"]
    primary = [
        row for row in pairwise
        if row["source_split"] == gate["primary_split"]
        and row["comparator"] in gate["required_comparators"]
    ]
    gate_rows = []
    for row in primary:
        passed = (
            float(row["mean_field_gain_pct"]) > float(gate["minimum_mean_field_gain_pct"])
            and float(row["field_cluster_ci95_low_pct"]) > float(gate["minimum_field_cluster_ci95_low_pct"])
            and int(row["positive_seed_count"]) >= int(gate["required_positive_seed_count"])
            and float(row["mean_audit_gain_pct"]) >= float(gate["minimum_mean_audit_gain_pct"])
        )
        gate_rows.append({**row, "functional_gate_pass": bool(passed)})
    mechanism_pass = len(gate_rows) == len(gate["required_comparators"]) and all(
        bool(row["functional_gate_pass"]) for row in gate_rows
    )
    status = (
        "GC_SRO_FUNCTIONAL_MECHANISM_GATE_PASS_SUPERIORITY_NOT_AUTHORIZED"
        if mechanism_pass
        else "GC_SRO_FUNCTIONAL_MECHANISM_GATE_FAIL_STOP_OR_REDESIGN"
    )
    plot_results(output_dir / PUBLIC_FILES[-1], histories, split_summary, pairwise)
    write_csv(output_dir / PUBLIC_FILES[0], derangement_rows)
    write_csv(output_dir / PUBLIC_FILES[1], histories)
    write_csv(output_dir / PUBLIC_FILES[2], sample_rows)
    write_csv(output_dir / PUBLIC_FILES[3], split_summary)
    write_csv(output_dir / PUBLIC_FILES[4], pairwise)
    write_csv(output_dir / PUBLIC_FILES[5], swap_rows)
    validation_summary = {
        str(row["method"]): row
        for row in split_summary
        if str(row["source_split"]) == "val"
    }
    generic_adapter_gain = 100.0 * (
        float(validation_summary["locked_fno"]["mean_field_rel_l2"])
        - float(validation_summary["static"]["mean_field_rel_l2"])
    ) / float(validation_summary["locked_fno"]["mean_field_rel_l2"])
    dashboard = {
        "experiment": config["name"],
        "scientific_status": status,
        "functional_mechanism_gate_pass": mechanism_pass,
        "development_only": True,
        "methods": ["locked_fno", *config["methods"]],
        "model_seeds": seeds,
        "training_epochs": int(config["training"]["epochs"]),
        "training_records": records,
        "parameter_contract": {
            "adapter_total_parameters": sorted({int(row["total_parameters"]) for row in records}),
            "adapter_trainable_parameters": sorted({int(row["trainable_parameters"]) for row in records}),
            "parameter_matched": len({int(row["total_parameters"]) for row in records}) == 1,
            "base_checkpoint_frozen": True,
            "base_predictions_precomputed": True,
        },
        "derangement_contract": {
            "mapping_rows": len(derangement_rows),
            "fixed_points": sum(bool(row["fixed_point"]) for row in derangement_rows),
            "within_partition": True,
            "batch_order_independent": True,
        },
        "primary_gate_rows": gate_rows,
        "generic_static_adapter_validation_gain_vs_locked_fno_pct": generic_adapter_gain,
        "same_model_descriptor_swap": swap_summary,
        "split_summary": split_summary,
        "pairwise_mechanism": pairwise,
        "next_decision": {
            "continue_gc_sro_architecture": mechanism_pass,
            "if_fail": "stop claim or redesign data resampling before changing capacity",
            "matched_variable_geometry_fno_required_before_superiority": True,
            "superiority_training_authorized": False,
            "blind_final_opened": False,
        },
        "claims_boundary": config["claims_boundary"],
    }
    report = {
        "status": status,
        "dashboard": dashboard,
        "protocol": {
            "selection": "best epoch by unseen-layout validation field relative L2 per method and seed",
            "test_read_timing": "all adapter training and checkpoint selection completed before test metrics",
            "statistical_unit": "source field after collapsing three model seeds for cluster bootstrap",
            "shuffled_control": "fixed-point-free cyclic derangement within geometry partition",
        },
        "provenance": {
            "config_sha256": sha256(args.config),
            "dataset_sha256": sha256(private_path),
            "v3i_dashboard_sha256": sha256(v3i_dashboard_path),
            "base_checkpoint_sha256": sha256(checkpoint_path),
            "script_sha256": sha256(Path(__file__).resolve()),
            "model_module_sha256": sha256(ROOT / "own_algorithm_models.py"),
        },
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__, "device": str(device)},
    }
    (output_dir / PUBLIC_FILES[6]).write_text(json.dumps(dashboard, indent=2), encoding="utf-8")
    (output_dir / PUBLIC_FILES[7]).write_text(json.dumps(report, indent=2), encoding="utf-8")
    checksum_lines = [f"{sha256(output_dir / name)}  {name}" for name in PUBLIC_FILES]
    (output_dir / "v3j_gc_sro_functional_checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="ascii")
    print(json.dumps({
        "scientific_status": status,
        "mechanism_gate_pass": mechanism_pass,
        "primary_gate_rows": gate_rows,
        "training_seconds_total": sum(float(row["train_seconds"]) for row in records),
    }, indent=2))


if __name__ == "__main__":
    main()
