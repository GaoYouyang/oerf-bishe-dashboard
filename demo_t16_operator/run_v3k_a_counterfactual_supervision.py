#!/usr/bin/env python3
"""Run the v3k-A equal-exposure same-field/multi-geometry mechanism audit."""

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
    from .counterfactual_geometry import (
        CounterfactualGeometryDataset,
        CounterfactualInputFactory,
        build_pair_schedule,
        descriptor_components_for_pairs,
        geometry_derangement_map,
        schedule_balance,
    )
    from .models import make_model
    from .run_v3j_gc_sro_functional_pilot import build_model, parameter_count
    from .train_eval import (
        choose_device,
        gradient_mse,
        masked_relative_projection_loss,
        project_torch,
        synchronize,
    )
except ImportError:
    from counterfactual_geometry import (
        CounterfactualGeometryDataset,
        CounterfactualInputFactory,
        build_pair_schedule,
        descriptor_components_for_pairs,
        geometry_derangement_map,
        schedule_balance,
    )
    from models import make_model
    from run_v3j_gc_sro_functional_pilot import build_model, parameter_count
    from train_eval import (
        choose_device,
        gradient_mse,
        masked_relative_projection_loss,
        project_torch,
        synchronize,
    )


ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = ROOT / "configs"
DEFAULT_CONFIG = CONFIG_ROOT / "v3k_a_counterfactual_supervision.json"
PUBLIC_FILES = [
    "v3k_a_pair_manifest.csv",
    "v3k_a_geometry_derangement.csv",
    "v3k_a_training_history.csv",
    "v3k_a_sample_metrics.csv",
    "v3k_a_split_summary.csv",
    "v3k_a_pairwise_mechanism.csv",
    "v3k_a_exposure_interaction.csv",
    "v3k_a_descriptor_swap.csv",
    "v3k_a_counterfactual_dashboard.json",
    "v3k_a_counterfactual_report.json",
    "t16_v3k_a_counterfactual_supervision.png",
]
LABELS = {
    "locked_fno": "Locked fixed-layout FNO",
    "static": "Static spectral adapter",
    "k_cardinality": "K-cardinality adapter",
    "shuffled_geometry": "Shuffled-geometry GC-SRO",
    "correct_geometry": "Correct-geometry GC-SRO",
}
ARM_LABELS = {
    "m1_repeat": "M1 repeat",
    "m4_counterfactual": "M4 counterfactual",
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
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def load_private_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def components_for_batch(
    components: tuple[np.ndarray, np.ndarray, np.ndarray],
    indices: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = indices.cpu().numpy()
    return tuple(torch.from_numpy(value[selected]).to(device) for value in components)


def precompute_base_predictions(
    model: torch.nn.Module,
    dataset: CounterfactualGeometryDataset,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False)
    model = model.to(device).eval()
    output = []
    with torch.no_grad():
        for batch in loader:
            output.append(model(batch["x"].to(device)).cpu().numpy())
    model.to("cpu")
    return np.concatenate(output).astype(np.float32)


def field_relative_l2(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    numerator = torch.linalg.vector_norm(
        (prediction - target).flatten(start_dim=1), dim=1
    )
    denominator = torch.linalg.vector_norm(
        target.flatten(start_dim=1), dim=1
    ).clamp_min(1e-8)
    return numerator / denominator


def source_cluster_mean(values: list[float], sources: list[int]) -> float:
    grouped: dict[int, list[float]] = defaultdict(list)
    for value, source in zip(values, sources):
        grouped[int(source)].append(float(value))
    return float(np.mean([np.mean(subset) for subset in grouped.values()]))


def train_adapter(
    config: dict,
    dataset_config: dict,
    data: dict[str, np.ndarray],
    checkpoint: dict[str, torch.Tensor],
    train_dataset: CounterfactualGeometryDataset,
    val_dataset: CounterfactualGeometryDataset,
    train_base: np.ndarray,
    val_base: np.ndarray,
    train_wrong: tuple[np.ndarray, np.ndarray, np.ndarray],
    val_wrong: tuple[np.ndarray, np.ndarray, np.ndarray],
    arm: str,
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
            pair_indices = batch["index"]
            x = batch["x"].to(device)
            target = batch["field"].to(device)
            base = torch.from_numpy(train_base[pair_indices.numpy()]).to(device)
            descriptor = (
                components_for_batch(train_wrong, pair_indices, device)
                if method == "shuffled_geometry"
                else None
            )
            optimizer.zero_grad(set_to_none=True)
            correction, _ = model.correction(
                x, base_prediction=base, descriptor_components=descriptor
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
            total_sum += float(loss.detach()) * len(pair_indices)
            seen += len(pair_indices)

        model.eval()
        val_values: list[float] = []
        val_sources: list[int] = []
        with torch.no_grad():
            for batch in val_loader:
                pair_indices = batch["index"]
                x = batch["x"].to(device)
                target = batch["field"].to(device)
                base = torch.from_numpy(val_base[pair_indices.numpy()]).to(device)
                descriptor = (
                    components_for_batch(val_wrong, pair_indices, device)
                    if method == "shuffled_geometry"
                    else None
                )
                correction, _ = model.correction(
                    x, base_prediction=base, descriptor_components=descriptor
                )
                error = field_relative_l2(base + correction, target)
                val_values.extend(error.cpu().tolist())
                val_sources.extend(batch["source_index"].cpu().tolist())
        val_mean = source_cluster_mean(val_values, val_sources)
        history.append(
            {
                "training_arm": arm,
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
        raise RuntimeError("v3k-A did not produce a validation-selected checkpoint")
    model.load_state_dict(best_state)
    model.to("cpu")
    checkpoint_path = work_dir / arm / str(seed) / "checkpoints" / f"{method}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, checkpoint_path)
    record = {
        "training_arm": arm,
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
    wrong_components: tuple[np.ndarray, np.ndarray, np.ndarray],
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
            descriptor = (
                components_for_batch(wrong_components, indices, device)
                if method == "shuffled_geometry"
                else None
            )
            correction, _ = model.correction(
                x, base_prediction=base, descriptor_components=descriptor
            )
            output.append((base + correction).cpu().numpy())
    model.to("cpu")
    return np.concatenate(output).astype(np.float32)


def metric_rows(
    predictions: np.ndarray,
    dataset: CounterfactualGeometryDataset,
    training_arm: str,
    method: str,
    seed: int,
) -> list[dict[str, object]]:
    data = dataset.data
    predicted = predictions[:, 0]
    sources = np.asarray(
        [int(row["source_index"]) for row in dataset.pairs], dtype=np.int64
    )
    target = data["field"][sources]
    field = np.linalg.norm(
        (predicted - target).reshape(len(dataset), -1), axis=1
    ) / np.maximum(np.linalg.norm(target.reshape(len(dataset), -1), axis=1), 1e-8)
    audit = int(data["audit_query_index"])
    projected = np.einsum(
        "bdp,np->bdn",
        predicted.reshape(len(dataset), predicted.shape[1], -1),
        data["forward_matrix"][audit],
        optimize=True,
    )
    reference = data["clean_observation"][sources, :, audit]
    audit_error = np.linalg.norm(
        (projected - reference).reshape(len(dataset), -1), axis=1
    ) / np.maximum(np.linalg.norm(reference.reshape(len(dataset), -1), axis=1), 1e-8)
    rows = []
    for index, pair in enumerate(dataset.pairs):
        source = int(pair["source_index"])
        rows.append(
            {
                "training_arm": training_arm,
                "method": method,
                "model_seed": int(seed),
                "source_split": str(pair["source_split"]),
                "pair_index": int(pair["pair_index"]),
                "source_index": source,
                "sample_seed": int(data["sample_seed"][source]),
                "geometry_id": str(pair["geometry_id"]),
                "geometry_partition": str(pair["geometry_partition"]),
                "family_id": int(data["family_id"][source]),
                "noise_level": float(data["noise_level"][source]),
                "field_rel_l2": float(field[index]),
                "audit_reprojection_rel_l2": float(audit_error[index]),
            }
        )
    return rows


def bootstrap_interval(
    values: np.ndarray, seed: int, replicates: int
) -> tuple[float, float]:
    rng = np.random.default_rng(int(seed))
    sampled = rng.integers(0, len(values), size=(int(replicates), len(values)))
    estimates = values[sampled].mean(axis=1)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def _field_seed_errors(
    sample_rows: list[dict[str, object]],
) -> dict[tuple[str, str, int, str, int], float]:
    grouped: dict[tuple[str, str, int, str, int], list[float]] = defaultdict(list)
    for row in sample_rows:
        key = (
            str(row["training_arm"]),
            str(row["method"]),
            int(row["model_seed"]),
            str(row["source_split"]),
            int(row["source_index"]),
        )
        grouped[key].append(float(row["field_rel_l2"]))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def summarize_results(
    sample_rows: list[dict[str, object]], config: dict
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    field_seed = _field_seed_errors(sample_rows)
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in sample_rows:
        grouped[
            (
                str(row["training_arm"]),
                str(row["method"]),
                str(row["source_split"]),
            )
        ].append(row)
    split_summary = []
    for (arm, method, split), rows in sorted(grouped.items()):
        independent_fields = sorted({int(row["source_index"]) for row in rows})
        seeds = sorted({int(row["model_seed"]) for row in rows})
        collapsed = [
            np.mean([field_seed[(arm, method, seed, split, source)] for seed in seeds])
            for source in independent_fields
        ]
        split_summary.append(
            {
                "training_arm": arm,
                "method": method,
                "method_label": LABELS[method],
                "source_split": split,
                "sample_metric_rows": len(rows),
                "independent_field_count": len(independent_fields),
                "layouts_per_field": len(rows) // (len(independent_fields) * len(seeds)),
                "model_seed_count": len(seeds),
                "mean_field_rel_l2": float(np.mean(collapsed)),
                "median_field_rel_l2": float(np.median(collapsed)),
                "p90_field_rel_l2": float(np.quantile(collapsed, 0.9)),
                "mean_audit_reprojection_rel_l2": float(
                    np.mean([float(row["audit_reprojection_rel_l2"]) for row in rows])
                ),
            }
        )

    arms = [str(value) for value in config["training_arms"]]
    seeds = [int(value) for value in config["model_seeds"]]
    splits = sorted({str(row["source_split"]) for row in sample_rows})
    comparators = ["locked_fno", "static", "k_cardinality", "shuffled_geometry"]
    bootstrap_seed = int(config["mechanism_gate"]["bootstrap_seed"])
    replicates = int(config["mechanism_gate"]["bootstrap_replicates"])
    pairwise = []
    field_gain_lookup: dict[tuple[str, str, str, int, int], float] = {}
    for arm in arms:
        for split in splits:
            sources = sorted(
                {
                    int(row["source_index"])
                    for row in sample_rows
                    if str(row["training_arm"]) == arm
                    and str(row["source_split"]) == split
                }
            )
            for comparator in comparators:
                seed_means = []
                field_collapsed = []
                all_gains = []
                for source in sources:
                    source_gains = []
                    for seed in seeds:
                        correct = field_seed[
                            (arm, "correct_geometry", seed, split, source)
                        ]
                        control = field_seed[(arm, comparator, seed, split, source)]
                        gain = 100.0 * (control - correct) / max(control, 1e-12)
                        source_gains.append(gain)
                        all_gains.append(gain)
                        field_gain_lookup[(arm, split, comparator, seed, source)] = gain
                    field_collapsed.append(float(np.mean(source_gains)))
                for seed in seeds:
                    seed_means.append(
                        float(
                            np.mean(
                                [
                                    field_gain_lookup[
                                        (arm, split, comparator, seed, source)
                                    ]
                                    for source in sources
                                ]
                            )
                        )
                    )
                ci_low, ci_high = bootstrap_interval(
                    np.asarray(field_collapsed),
                    bootstrap_seed + len(pairwise),
                    replicates,
                )
                pairwise.append(
                    {
                        "training_arm": arm,
                        "source_split": split,
                        "candidate": "correct_geometry",
                        "comparator": comparator,
                        "independent_field_count": len(sources),
                        "model_seed_count": len(seeds),
                        "mean_field_gain_pct": float(np.mean(field_collapsed)),
                        "field_cluster_ci95_low_pct": ci_low,
                        "field_cluster_ci95_high_pct": ci_high,
                        "median_field_gain_pct": float(np.median(field_collapsed)),
                        "p10_field_gain_pct": float(np.quantile(field_collapsed, 0.1)),
                        "harm_rate_gt_1pct": float(
                            np.mean(np.asarray(all_gains) < -1.0)
                        ),
                        "positive_seed_count": int(
                            sum(value > 0.0 for value in seed_means)
                        ),
                        "seed_mean_gains_pct": ";".join(
                            f"{value:.8f}" for value in seed_means
                        ),
                    }
                )

    interactions = []
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
            for source in sources:
                differences = [
                    field_gain_lookup[
                        ("m4_counterfactual", split, comparator, seed, source)
                    ]
                    - field_gain_lookup[("m1_repeat", split, comparator, seed, source)]
                    for seed in seeds
                ]
                field_collapsed.append(float(np.mean(differences)))
            for seed in seeds:
                seed_means.append(
                    float(
                        np.mean(
                            [
                                field_gain_lookup[
                                    (
                                        "m4_counterfactual",
                                        split,
                                        comparator,
                                        seed,
                                        source,
                                    )
                                ]
                                - field_gain_lookup[
                                    ("m1_repeat", split, comparator, seed, source)
                                ]
                                for source in sources
                            ]
                        )
                    )
                )
            ci_low, ci_high = bootstrap_interval(
                np.asarray(field_collapsed),
                bootstrap_seed + 10_000 + len(interactions),
                replicates,
            )
            interactions.append(
                {
                    "source_split": split,
                    "candidate": "M4_minus_M1_correct_control_gain",
                    "comparator": comparator,
                    "independent_field_count": len(sources),
                    "model_seed_count": len(seeds),
                    "mean_interaction_gain_pct": float(np.mean(field_collapsed)),
                    "field_cluster_ci95_low_pct": ci_low,
                    "field_cluster_ci95_high_pct": ci_high,
                    "positive_seed_count": int(sum(value > 0.0 for value in seed_means)),
                    "seed_mean_interactions_pct": ";".join(
                        f"{value:.8f}" for value in seed_means
                    ),
                }
            )
    return split_summary, pairwise, interactions


def descriptor_swap_audit(
    trained: dict[tuple[str, str, int], torch.nn.Module],
    val_dataset: CounterfactualGeometryDataset,
    val_base: np.ndarray,
    correct_components: tuple[np.ndarray, np.ndarray, np.ndarray],
    wrong_components: tuple[np.ndarray, np.ndarray, np.ndarray],
    geometry_mapping: dict[str, str],
    config: dict,
    device: torch.device,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    batch_size = int(config["training"]["batch_size"])
    seeds = [int(value) for value in config["model_seeds"]]
    rows: list[dict[str, object]] = []
    for arm in [str(value) for value in config["training_arms"]]:
        for seed in seeds:
            model = trained[(arm, "correct_geometry", seed)].to(device).eval()
            loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            with torch.no_grad():
                for batch in loader:
                    indices = batch["index"]
                    selected = indices.numpy()
                    x = batch["x"].to(device)
                    target = batch["field"].to(device)
                    base = torch.from_numpy(val_base[selected]).to(device)
                    correct = components_for_batch(correct_components, indices, device)
                    wrong = components_for_batch(wrong_components, indices, device)
                    correct_embedding, correct_modulation = model.descriptor_embedding(
                        x, descriptor_components=correct
                    )
                    wrong_embedding, wrong_modulation = model.descriptor_embedding(
                        x, descriptor_components=wrong
                    )
                    correct_correction, _ = model.correction(
                        x, base_prediction=base, descriptor_components=correct
                    )
                    wrong_correction, _ = model.correction(
                        x, base_prediction=base, descriptor_components=wrong
                    )
                    correct_error = field_relative_l2(
                        base + correct_correction, target
                    )
                    wrong_error = field_relative_l2(base + wrong_correction, target)
                    gains = 100.0 * (wrong_error - correct_error) / wrong_error.clamp_min(
                        1e-12
                    )
                    correction_change = 100.0 * torch.linalg.vector_norm(
                        (correct_correction - wrong_correction).flatten(start_dim=1),
                        dim=1,
                    ) / torch.linalg.vector_norm(
                        correct_correction.flatten(start_dim=1), dim=1
                    ).clamp_min(1e-12)
                    for local, pair_index in enumerate(selected):
                        pair = val_dataset.pairs[int(pair_index)]
                        rows.append(
                            {
                                "training_arm": arm,
                                "model_seed": seed,
                                "pair_index": int(pair_index),
                                "source_index": int(pair["source_index"]),
                                "geometry_id": str(pair["geometry_id"]),
                                "wrong_geometry_id": str(
                                    geometry_mapping[str(pair["geometry_id"])]
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
                                    correction_change[local]
                                ),
                                "correct_descriptor_field_rel_l2": float(
                                    correct_error[local]
                                ),
                                "wrong_descriptor_field_rel_l2": float(wrong_error[local]),
                                "correct_descriptor_field_gain_pct": float(gains[local]),
                            }
                        )
            model.to("cpu")

    gate = config["mechanism_gate"]
    summaries = []
    for arm in [str(value) for value in config["training_arms"]]:
        subset = [row for row in rows if str(row["training_arm"]) == arm]
        grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
        for row in subset:
            grouped[(int(row["model_seed"]), int(row["source_index"]))].append(
                float(row["correct_descriptor_field_gain_pct"])
            )
        sources = sorted({source for _, source in grouped})
        field_collapsed = [
            float(
                np.mean(
                    [
                        np.mean(grouped[(seed, source)])
                        for seed in seeds
                    ]
                )
            )
            for source in sources
        ]
        seed_means = [
            float(np.mean([np.mean(grouped[(seed, source)]) for source in sources]))
            for seed in seeds
        ]
        ci_low, ci_high = bootstrap_interval(
            np.asarray(field_collapsed),
            int(gate["bootstrap_seed"]) + 20_000 + len(summaries),
            int(gate["bootstrap_replicates"]),
        )
        summaries.append(
            {
                "training_arm": arm,
                "validation_pair_rows": len(subset),
                "independent_field_count": len(sources),
                "model_seed_count": len(seeds),
                "mean_embedding_swap_l2": float(
                    np.mean([float(row["embedding_swap_l2"]) for row in subset])
                ),
                "mean_modulation_swap_l2": float(
                    np.mean([float(row["modulation_swap_l2"]) for row in subset])
                ),
                "mean_correction_swap_relative_pct": float(
                    np.mean(
                        [float(row["correction_swap_relative_pct"]) for row in subset]
                    )
                ),
                "mean_correct_descriptor_field_gain_pct": float(
                    np.mean(field_collapsed)
                ),
                "field_cluster_ci95_low_pct": ci_low,
                "field_cluster_ci95_high_pct": ci_high,
                "p10_correct_descriptor_field_gain_pct": float(
                    np.quantile(field_collapsed, 0.1)
                ),
                "positive_seed_count": int(sum(value > 0.0 for value in seed_means)),
                "seed_mean_gains_pct": ";".join(
                    f"{value:.8f}" for value in seed_means
                ),
            }
        )
    return rows, summaries


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
    interactions: list[dict[str, object]],
    swap_summary: list[dict[str, object]],
) -> tuple[dict[str, bool], str]:
    gate = config["mechanism_gate"]
    delta = float(gate["minimum_mean_gain_pct"])
    ci_floor = float(gate["minimum_cluster_ci95_low_pct"])
    required_seeds = int(gate["required_positive_seed_count"])
    primary = str(gate["primary_split"])
    held_out = str(gate["held_out_geometry_split"])
    joint = str(gate["joint_ood_split"])
    val_shuffled = find_row(
        pairwise,
        training_arm="m4_counterfactual",
        source_split=primary,
        comparator="shuffled_geometry",
    )
    val_static = find_row(
        pairwise,
        training_arm="m4_counterfactual",
        source_split=primary,
        comparator="static",
    )
    held_out_shuffled = find_row(
        pairwise,
        training_arm="m4_counterfactual",
        source_split=held_out,
        comparator="shuffled_geometry",
    )
    joint_shuffled = find_row(
        pairwise,
        training_arm="m4_counterfactual",
        source_split=joint,
        comparator="shuffled_geometry",
    )
    interaction = find_row(
        interactions, source_split=primary, comparator="shuffled_geometry"
    )
    swap = find_row(swap_summary, training_arm="m4_counterfactual")

    def gain_pass(row: dict[str, object], minimum: float = delta) -> bool:
        return (
            float(row["mean_field_gain_pct"]) >= minimum
            and float(row["field_cluster_ci95_low_pct"]) > ci_floor
            and int(row["positive_seed_count"]) >= required_seeds
        )

    checks = {
        "m4_correct_vs_shuffled_validation": gain_pass(val_shuffled),
        "m4_correct_vs_static_validation": gain_pass(val_static),
        "m4_minus_m1_shuffled_interaction": (
            float(interaction["mean_interaction_gain_pct"])
            >= float(gate["minimum_interaction_gain_pct"])
            and float(interaction["field_cluster_ci95_low_pct"]) > ci_floor
            and int(interaction["positive_seed_count"]) >= required_seeds
        ),
        "m4_correct_vs_shuffled_geometry_held_out": gain_pass(held_out_shuffled),
        "joint_ood_no_material_harm": float(joint_shuffled["mean_field_gain_pct"])
        >= float(gate["minimum_joint_ood_gain_pct"]),
        "same_model_descriptor_swap_propagates": (
            float(swap["mean_correct_descriptor_field_gain_pct"])
            >= float(gate["minimum_swap_field_gain_pct"])
            and float(swap["field_cluster_ci95_low_pct"]) > ci_floor
            and int(swap["positive_seed_count"]) >= required_seeds
            and float(swap["mean_correction_swap_relative_pct"])
            >= float(gate["minimum_correction_swap_relative_pct"])
        ),
    }
    if all(checks.values()):
        status = "COUNTERFACTUAL_DATA_MECHANISM_GATE_PASS_CONFIRMATION_NOT_AUTHORIZED"
    else:
        key_upper_bounds = [
            float(val_shuffled["field_cluster_ci95_high_pct"]),
            float(val_static["field_cluster_ci95_high_pct"]),
            float(interaction["field_cluster_ci95_high_pct"]),
            float(swap["field_cluster_ci95_high_pct"]),
        ]
        if all(value < delta for value in key_upper_bounds):
            status = "GLOBAL_GEOMETRY_MODULATION_MECHANISM_FAIL_STOP_CAPACITY_SEARCH"
        else:
            status = "COUNTERFACTUAL_MECHANISM_INCONCLUSIVE_DO_NOT_SCALE"
    return checks, status


def plot_results(
    path: Path,
    history: list[dict[str, object]],
    pairwise: list[dict[str, object]],
    interactions: list[dict[str, object]],
    swap_summary: list[dict[str, object]],
) -> None:
    colors = {"m1_repeat": "#7a7f83", "m4_counterfactual": "#16817a"}
    fig, axes = plt.subplots(1, 4, figsize=(18.2, 4.8), constrained_layout=True)
    for arm in ("m1_repeat", "m4_counterfactual"):
        for method, style in (("shuffled_geometry", "--"), ("correct_geometry", "-")):
            subset = [
                row
                for row in history
                if row["training_arm"] == arm and row["method"] == method
            ]
            epochs = sorted({int(row["epoch"]) for row in subset})
            mean = [
                np.mean(
                    [
                        float(row["validation_field_cluster_rel_l2"])
                        for row in subset
                        if int(row["epoch"]) == epoch
                    ]
                )
                for epoch in epochs
            ]
            axes[0].plot(
                epochs,
                mean,
                linestyle=style,
                color=colors[arm],
                label=f"{ARM_LABELS[arm]} / {method.replace('_geometry', '')}",
            )
    axes[0].set_title("Matched validation curves")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("field-cluster relative L2")
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=7)

    validation = [
        row
        for row in pairwise
        if row["source_split"] == "val"
        and row["comparator"] in {"static", "shuffled_geometry"}
    ]
    x = np.arange(len(validation))
    means = np.asarray([float(row["mean_field_gain_pct"]) for row in validation])
    lows = np.asarray(
        [float(row["field_cluster_ci95_low_pct"]) for row in validation]
    )
    highs = np.asarray(
        [float(row["field_cluster_ci95_high_pct"]) for row in validation]
    )
    axes[1].errorbar(
        x,
        means,
        yerr=np.vstack([means - lows, highs - means]),
        fmt="o",
        color="#16817a",
        capsize=4,
    )
    axes[1].axhline(0.0, color="#333", linewidth=1)
    axes[1].axhline(0.25, color="#c9823b", linewidth=1, linestyle=":")
    axes[1].set_xticks(
        x,
        [
            f"{str(row['training_arm']).split('_')[0]}\n{str(row['comparator']).replace('_geometry', '')}"
            for row in validation
        ],
    )
    axes[1].set_title("Correct descriptor gain")
    axes[1].set_ylabel("paired field gain (%)")
    axes[1].grid(axis="y", alpha=0.2)

    interaction = [
        row for row in interactions if row["comparator"] == "shuffled_geometry"
    ]
    axes[2].bar(
        np.arange(len(interaction)),
        [float(row["mean_interaction_gain_pct"]) for row in interaction],
        color=["#16817a" if row["source_split"] == "val" else "#6c757d" for row in interaction],
    )
    axes[2].axhline(0.0, color="#333", linewidth=1)
    axes[2].axhline(0.25, color="#c9823b", linewidth=1, linestyle=":")
    axes[2].set_xticks(
        np.arange(len(interaction)),
        [str(row["source_split"]).replace("test_", "") for row in interaction],
        rotation=25,
    )
    axes[2].set_title("M4 minus M1 mechanism gain")
    axes[2].set_ylabel("interaction gain (%)")
    axes[2].grid(axis="y", alpha=0.2)

    arms = [str(row["training_arm"]) for row in swap_summary]
    correction = [float(row["mean_correction_swap_relative_pct"]) for row in swap_summary]
    field_gain = [float(row["mean_correct_descriptor_field_gain_pct"]) for row in swap_summary]
    positions = np.arange(len(arms))
    axes[3].bar(positions - 0.17, correction, 0.34, label="correction change", color="#355c7d")
    axes[3].bar(positions + 0.17, field_gain, 0.34, label="field gain", color="#16817a")
    axes[3].axhline(0.0, color="#333", linewidth=1)
    axes[3].set_xticks(positions, [ARM_LABELS[arm] for arm in arms])
    axes[3].set_title("Same-model descriptor swap")
    axes[3].set_ylabel("relative change / gain (%)")
    axes[3].legend(fontsize=7)
    axes[3].grid(axis="y", alpha=0.2)
    fig.suptitle("v3k-A equal-exposure counterfactual supervision audit", fontsize=14)
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
    data = load_private_dataset(private_path)
    v3j_path = ROOT / "results" / str(config["v3j_dashboard"])
    v3j = read_json(v3j_path)
    if v3j["functional_mechanism_gate_pass"]:
        raise RuntimeError("v3k-A is authorized only after the v3j mechanism failure")
    checkpoint_path = ROOT / "results" / str(config["base_checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    device = choose_device(str(config["training"]["device"]))
    design = config["pair_design"]
    factory = CounterfactualInputFactory(data, float(design["ridge_relative"]))
    mapping, derangement_rows = geometry_derangement_map(factory.catalog)
    repeats = int(design["repeats_per_source"])
    assignment_seed = int(design["assignment_seed"])
    stride = int(design["counterfactual_stride"])

    train_pairs = {
        arm: build_pair_schedule(
            data,
            "train",
            str(design["train_geometry_partition"]),
            arm,
            repeats,
            assignment_seed,
            stride,
        )
        for arm in config["training_arms"]
    }
    val_pairs = build_pair_schedule(
        data,
        "val",
        str(design["evaluation_geometry_partition_by_split"]["val"]),
        "evaluation",
        repeats,
        assignment_seed,
        stride,
    )
    schedule_audit = {
        arm: schedule_balance(rows) for arm, rows in train_pairs.items()
    }
    schedule_audit["shared_validation"] = schedule_balance(val_pairs)
    train_source_sets = [
        {int(row["source_index"]) for row in train_pairs[arm]}
        for arm in config["training_arms"]
    ]
    if len({frozenset(values) for values in train_source_sets}) != 1:
        raise RuntimeError("M1 and M4 do not use the same training fields")
    if schedule_audit["m1_repeat"]["row_count"] != schedule_audit["m4_counterfactual"]["row_count"]:
        raise RuntimeError("M1 and M4 exposure rows are not matched")

    pair_manifest: list[dict[str, object]] = []
    for arm, rows in train_pairs.items():
        pair_manifest.extend({**row, "schedule_role": "training"} for row in rows)
    pair_manifest.extend({**row, "schedule_role": "checkpoint_selection"} for row in val_pairs)
    train_datasets = {
        arm: CounterfactualGeometryDataset(factory, rows)
        for arm, rows in train_pairs.items()
    }
    val_dataset = CounterfactualGeometryDataset(factory, val_pairs)

    base_for_cache = make_model(
        "fno", dataset_config["models"]["fno"], int(data["inputs"].shape[1]), residual=True
    )
    base_for_cache.load_state_dict(checkpoint, strict=True)
    train_base = {
        arm: precompute_base_predictions(
            base_for_cache,
            dataset,
            device,
            int(config["training"]["batch_size"]),
        )
        for arm, dataset in train_datasets.items()
    }
    val_base = precompute_base_predictions(
        base_for_cache,
        val_dataset,
        device,
        int(config["training"]["batch_size"]),
    )
    train_wrong = {
        arm: descriptor_components_for_pairs(factory, dataset.pairs, mapping)
        for arm, dataset in train_datasets.items()
    }
    val_correct = descriptor_components_for_pairs(factory, val_dataset.pairs)
    val_wrong = descriptor_components_for_pairs(factory, val_dataset.pairs, mapping)
    work_dir = ROOT / "results" / str(config["work_dir"])
    output_dir = ROOT / "results" / str(config["output_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    histories: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    trained: dict[tuple[str, str, int], torch.nn.Module] = {}
    for arm in [str(value) for value in config["training_arms"]]:
        for seed in [int(value) for value in config["model_seeds"]]:
            for method in [str(value) for value in config["methods"]]:
                model, history, record = train_adapter(
                    config,
                    dataset_config,
                    data,
                    checkpoint,
                    train_datasets[arm],
                    val_dataset,
                    train_base[arm],
                    val_base,
                    train_wrong[arm],
                    val_wrong,
                    arm,
                    method,
                    seed,
                    device,
                    work_dir,
                )
                histories.extend(history)
                records.append(record)
                trained[(arm, method, seed)] = model
                print(
                    json.dumps(
                        {
                            "arm": arm,
                            "method": method,
                            "seed": seed,
                            "best_epoch": record["best_epoch"],
                            "best_val": record[
                                "best_validation_field_cluster_rel_l2"
                            ],
                            "seconds": record["train_seconds"],
                        }
                    )
                )

    sample_rows: list[dict[str, object]] = []
    evaluation_specs = design["evaluation_geometry_partition_by_split"]
    evaluation_datasets: dict[str, CounterfactualGeometryDataset] = {"val": val_dataset}
    evaluation_bases: dict[str, np.ndarray] = {"val": val_base}
    evaluation_wrong: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        "val": val_wrong
    }
    for split, partition in evaluation_specs.items():
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
            base_for_cache,
            dataset,
            device,
            int(config["training"]["batch_size"]),
        )
        evaluation_wrong[split] = descriptor_components_for_pairs(
            factory, dataset.pairs, mapping
        )
        schedule_audit[f"shared_{split}"] = schedule_balance(pairs)

    seeds = [int(value) for value in config["model_seeds"]]
    for split, dataset in evaluation_datasets.items():
        base_predictions = evaluation_bases[split]
        wrong = evaluation_wrong[split]
        for arm in [str(value) for value in config["training_arms"]]:
            for seed in seeds:
                sample_rows.extend(
                    metric_rows(
                        base_predictions,
                        dataset,
                        arm,
                        "locked_fno",
                        seed,
                    )
                )
                for method in [str(value) for value in config["methods"]]:
                    predictions = predict_dataset(
                        trained[(arm, method, seed)],
                        dataset,
                        base_predictions,
                        wrong,
                        method,
                        device,
                        int(config["training"]["batch_size"]),
                    )
                    sample_rows.extend(
                        metric_rows(predictions, dataset, arm, method, seed)
                    )

    split_summary, pairwise, interactions = summarize_results(sample_rows, config)
    swap_rows, swap_summary = descriptor_swap_audit(
        trained,
        val_dataset,
        val_base,
        val_correct,
        val_wrong,
        mapping,
        config,
        device,
    )
    gate_checks, status = gate_result(config, pairwise, interactions, swap_summary)
    mechanism_pass = all(gate_checks.values())
    plot_results(
        output_dir / PUBLIC_FILES[-1],
        histories,
        pairwise,
        interactions,
        swap_summary,
    )
    write_csv(output_dir / PUBLIC_FILES[0], pair_manifest)
    write_csv(output_dir / PUBLIC_FILES[1], derangement_rows)
    write_csv(output_dir / PUBLIC_FILES[2], histories)
    write_csv(output_dir / PUBLIC_FILES[3], sample_rows)
    write_csv(output_dir / PUBLIC_FILES[4], split_summary)
    write_csv(output_dir / PUBLIC_FILES[5], pairwise)
    write_csv(output_dir / PUBLIC_FILES[6], interactions)
    write_csv(output_dir / PUBLIC_FILES[7], swap_rows)

    dashboard = {
        "experiment": config["name"],
        "scientific_status": status,
        "counterfactual_data_mechanism_gate_pass": mechanism_pass,
        "development_only": True,
        "training_arms": config["training_arms"],
        "methods": ["locked_fno", *config["methods"]],
        "model_seeds": seeds,
        "training_epochs": int(config["training"]["epochs"]),
        "training_records": records,
        "equal_exposure_contract": {
            "same_training_fields": True,
            "same_rows_per_arm": True,
            "same_optimizer_steps": True,
            "same_model_initialization_seed": True,
            "same_parameter_count": True,
            "same_validation_fields_and_all_four_validation_layouts": True,
            "m1_unique_layouts_per_field": schedule_audit["m1_repeat"][
                "minimum_unique_geometries_per_source"
            ],
            "m4_unique_layouts_per_field": schedule_audit["m4_counterfactual"][
                "minimum_unique_geometries_per_source"
            ],
            "rows_per_training_arm": schedule_audit["m1_repeat"]["row_count"],
            "independent_training_fields": schedule_audit["m1_repeat"][
                "source_count"
            ],
            "frozen_v3i_ray_scales_reused": True,
            "shared_full_view_noise": True,
        },
        "schedule_audit": schedule_audit,
        "parameter_contract": {
            "adapter_total_parameters": sorted(
                {int(row["total_parameters"]) for row in records}
            ),
            "adapter_trainable_parameters": sorted(
                {int(row["trainable_parameters"]) for row in records}
            ),
            "parameter_matched": len(
                {int(row["total_parameters"]) for row in records}
            )
            == 1,
            "base_checkpoint_frozen": True,
            "base_predictions_geometry_specific": True,
        },
        "gate_thresholds": config["mechanism_gate"],
        "gate_checks": gate_checks,
        "split_summary": split_summary,
        "pairwise_mechanism": pairwise,
        "exposure_interactions": interactions,
        "same_model_descriptor_swap": swap_summary,
        "next_decision": {
            "continue_global_geometry_modulation": mechanism_pass,
            "if_clear_fail": "stop capacity search and move to voxel-level ray-set conditioning",
            "if_inconclusive": "do not scale width; add independent fields only under a new preregistration",
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
            "causal_contrast": "M4 distinct layouts versus M1 one layout repeated at identical exposure",
            "selection": "best epoch by source-field-cluster mean over all four unseen validation layouts",
            "test_read_timing": "development test metrics computed only after all 24 checkpoints were selected",
            "statistical_unit": "source field after collapsing layouts and three model seeds",
            "shuffled_control": "fixed-point-free cyclic derangement within frozen geometry partition",
            "minimum_meaningful_gain_pct": config["mechanism_gate"][
                "minimum_mean_gain_pct"
            ],
        },
        "provenance": {
            "config_sha256": sha256(args.config),
            "dataset_sha256": sha256(private_path),
            "v3j_dashboard_sha256": sha256(v3j_path),
            "base_checkpoint_sha256": sha256(checkpoint_path),
            "script_sha256": sha256(Path(__file__).resolve()),
            "counterfactual_module_sha256": sha256(
                ROOT / "counterfactual_geometry.py"
            ),
            "model_module_sha256": sha256(ROOT / "own_algorithm_models.py"),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
        },
    }
    (output_dir / PUBLIC_FILES[8]).write_text(
        json.dumps(dashboard, indent=2), encoding="utf-8"
    )
    (output_dir / PUBLIC_FILES[9]).write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    checksum_lines = [
        f"{sha256(output_dir / name)}  {name}" for name in PUBLIC_FILES
    ]
    (output_dir / "v3k_a_counterfactual_checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="ascii"
    )
    print(
        json.dumps(
            {
                "scientific_status": status,
                "mechanism_gate_pass": mechanism_pass,
                "gate_checks": gate_checks,
                "training_runs": len(records),
                "training_seconds_total": sum(
                    float(row["train_seconds"]) for row in records
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
