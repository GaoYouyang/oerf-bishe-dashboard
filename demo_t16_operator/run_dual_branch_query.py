#!/usr/bin/env python3
"""Train a dual-head FNO whose router is supervised by withheld camera views."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset

try:
    from .bost_physics import baseline_lift, forward_volume, make_grid_3d
    from .data import BOSTDataset, generate_dataset, load_npz, split_indices
    from .models import count_parameters, make_dual_branch_model
    from .run_ablations import METRICS, SPLIT_ORDER, read_json, t_interval, write_csv
    from .train_eval import (
        _centroid,
        _gradient_relative,
        _masked_projection_relative,
        _relative_norm,
        batch_relative_l2,
        choose_device,
        gradient_mse,
        masked_relative_projection_loss,
        project_torch,
        set_seed,
    )
except ImportError:
    from bost_physics import baseline_lift, forward_volume, make_grid_3d
    from data import BOSTDataset, generate_dataset, load_npz, split_indices
    from models import count_parameters, make_dual_branch_model
    from run_ablations import METRICS, SPLIT_ORDER, read_json, t_interval, write_csv
    from train_eval import (
        _centroid,
        _gradient_relative,
        _masked_projection_relative,
        _relative_norm,
        batch_relative_l2,
        choose_device,
        gradient_mse,
        masked_relative_projection_loss,
        project_torch,
        set_seed,
    )


ROOT = Path(__file__).resolve().parent
METHODS = ["residual_head", "absolute_head", "uniform_dual", "support_fit_mix", "query_router"]
LABELS = {
    "residual_head": "Residual head",
    "absolute_head": "Absolute head",
    "uniform_dual": "Uniform dual",
    "support_fit_mix": "Support-fit mix",
    "query_router": "Query-view router",
}
COLORS = {
    "residual_head": "#16806a",
    "absolute_head": "#b56a2b",
    "uniform_dual": "#7257a3",
    "support_fit_mix": "#b44f5f",
    "query_router": "#2e6f9e",
}


def write_checksum_manifest(output_dir: Path, filenames: list[str]) -> None:
    lines = []
    for filename in filenames:
        digest = hashlib.sha256((output_dir / filename).read_bytes()).hexdigest()
        lines.append(f"{digest}  {filename}")
    (output_dir / "dual_checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="ascii"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "dual_branch_query.json")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--force-data", action="store_true")
    return parser.parse_args()


def maximum_angular_gap(mask: np.ndarray, angles: np.ndarray) -> float:
    selected = np.sort(angles[np.asarray(mask) > 0.5].astype(float))
    if len(selected) < 2:
        return 1.0
    wrapped = np.concatenate([selected, [selected[0] + 180.0]])
    return float(np.diff(wrapped).max() / 180.0)


class SupportQueryDataset(Dataset):
    """Fixed view-dropout variants; identical across optimization seeds."""

    def __init__(
        self,
        data: dict[str, np.ndarray],
        indices: np.ndarray,
        support_specs: list[int | str],
        variants_per_sample: int,
        augmentation_seed: int,
    ):
        self.records: list[dict[str, np.ndarray | int | float]] = []
        angles = data["angles"].astype(np.float32)
        scale, offset = [float(value) for value in data["calibration"]]
        n = int(data["field"].shape[-1])
        for index in np.asarray(indices, dtype=int):
            observed_indices = np.flatnonzero(data["view_mask"][index] > 0.5)
            if len(observed_indices) < 2:
                raise ValueError("Support/query splitting requires at least two observed views")
            for variant in range(int(variants_per_sample)):
                spec = support_specs[variant % len(support_specs)]
                requested = len(observed_indices) - 1 if spec == "all_but_one" else int(spec)
                support_count = min(max(requested, 1), len(observed_indices) - 1)
                rng = np.random.default_rng(int(augmentation_seed) + int(index) * 1009 + variant * 9176)
                support_indices = np.sort(rng.choice(observed_indices, size=support_count, replace=False))
                query_indices = np.setdiff1d(observed_indices, support_indices)
                support_mask = np.zeros_like(data["view_mask"][index], dtype=np.float32)
                query_mask = np.zeros_like(support_mask)
                support_mask[support_indices] = 1.0
                query_mask[query_indices] = 1.0
                raw_lift = baseline_lift(data["observation"][index][:, support_indices], angles[support_indices], n)
                calibrated_lift = (scale * raw_lift + offset).astype(np.float32)
                inputs = data["inputs"][index].copy()
                inputs[0] = calibrated_lift
                inputs[2].fill(support_count / len(support_mask))
                self.records.append(
                    {
                        "index": int(index),
                        "x": inputs,
                        "field": data["field"][index][None],
                        "observation": data["observation"][index],
                        "support_mask": support_mask,
                        "query_mask": query_mask,
                        "max_gap": maximum_angular_gap(support_mask, angles),
                    }
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        record = self.records[item]
        return {
            "index": torch.tensor(int(record["index"]), dtype=torch.long),
            "x": torch.from_numpy(np.asarray(record["x"])),
            "field": torch.from_numpy(np.asarray(record["field"])),
            "observation": torch.from_numpy(np.asarray(record["observation"])),
            "support_mask": torch.from_numpy(np.asarray(record["support_mask"])),
            "query_mask": torch.from_numpy(np.asarray(record["query_mask"])),
            "max_gap": torch.tensor(float(record["max_gap"]), dtype=torch.float32),
        }


def per_sample_projection_error(
    projected: torch.Tensor,
    observed: torch.Tensor,
    view_mask: torch.Tensor,
) -> torch.Tensor:
    mask = view_mask[:, None, :, None]
    numerator = torch.sum(((projected - observed) * mask) ** 2, dim=(1, 2, 3))
    denominator = torch.sum((observed * mask) ** 2, dim=(1, 2, 3)).clamp_min(1e-8)
    return numerator / denominator


def branch_disagreement(residual: torch.Tensor, absolute: torch.Tensor) -> torch.Tensor:
    difference = torch.linalg.vector_norm((residual - absolute).flatten(start_dim=1), dim=1)
    scale = 0.5 * (
        torch.linalg.vector_norm(residual.flatten(start_dim=1), dim=1)
        + torch.linalg.vector_norm(absolute.flatten(start_dim=1), dim=1)
    )
    return difference / scale.clamp_min(1e-8)


def closed_form_support_weight(
    projected_residual: torch.Tensor,
    projected_absolute: torch.Tensor,
    observed: torch.Tensor,
    support_mask: torch.Tensor,
) -> torch.Tensor:
    """Least-squares residual-head weight using only inference-visible views."""
    mask = support_mask[:, None, :, None]
    direction = (projected_residual - projected_absolute) * mask
    target = (observed - projected_absolute) * mask
    numerator = torch.sum(direction * target, dim=(1, 2, 3))
    denominator = torch.sum(direction**2, dim=(1, 2, 3)).clamp_min(1e-8)
    return (numerator / denominator).clamp(0.0, 1.0)[:, None, None, None, None]


def router_features(
    x: torch.Tensor,
    residual: torch.Tensor,
    absolute: torch.Tensor,
    projected_residual: torch.Tensor,
    projected_absolute: torch.Tensor,
    observed: torch.Tensor,
    support_mask: torch.Tensor,
    max_gap: torch.Tensor,
) -> torch.Tensor:
    residual_error = torch.sqrt(per_sample_projection_error(projected_residual, observed, support_mask).clamp_min(0.0))
    absolute_error = torch.sqrt(per_sample_projection_error(projected_absolute, observed, support_mask).clamp_min(0.0))
    return torch.stack(
        [
            x[:, 2].mean(dim=(1, 2, 3)),
            x[:, 3].mean(dim=(1, 2, 3)),
            max_gap,
            residual_error,
            absolute_error,
            branch_disagreement(residual, absolute),
        ],
        dim=1,
    )


def mask_gap_batch(view_mask: torch.Tensor, angles: np.ndarray, device: torch.device) -> torch.Tensor:
    values = [maximum_angular_gap(mask, angles) for mask in view_mask.detach().cpu().numpy()]
    return torch.tensor(values, dtype=torch.float32, device=device)


def routed_predictions(
    model: torch.nn.Module,
    x: torch.Tensor,
    observed: torch.Tensor,
    support_mask: torch.Tensor,
    operator: torch.Tensor,
    angles: np.ndarray,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    residual, absolute = model.experts(x)
    projected_residual = project_torch(residual, operator)
    projected_absolute = project_torch(absolute, operator)
    max_gap = mask_gap_batch(support_mask, angles, x.device)
    features = router_features(
        x,
        residual,
        absolute,
        projected_residual,
        projected_absolute,
        observed,
        support_mask,
        max_gap,
    )
    weight = model.route(features.detach())
    support_weight = closed_form_support_weight(
        projected_residual,
        projected_absolute,
        observed,
        support_mask,
    )
    predictions = {
        "residual_head": residual,
        "absolute_head": absolute,
        "uniform_dual": model.combine(residual, absolute, torch.full_like(weight, 0.5)),
        "support_fit_mix": model.combine(residual, absolute, support_weight),
        "query_router": model.combine(residual, absolute, weight),
    }
    return predictions, weight, support_weight, features


def train_dual_model(
    model: torch.nn.Module,
    train_dataset: SupportQueryDataset,
    data: dict[str, np.ndarray],
    config: dict,
    experiment_config: dict,
    seed: int,
    results_dir: Path,
) -> dict:
    training = config["training"]
    device = choose_device(str(training["device"]))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    val_indices = split_indices(data)["val"]
    val_loader = DataLoader(BOSTDataset(data, val_indices), batch_size=int(training["batch_size"]))
    model = model.to(device)
    operator = torch.from_numpy(data["forward_matrix"]).to(device)
    angles = data["angles"].astype(np.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(training["epochs"]), 1))
    best_val = math.inf
    best_epoch = -1
    best_state = None
    stale_epochs = 0
    history = []
    start_time = time.perf_counter()

    for epoch in range(int(training["epochs"])):
        model.train()
        running = {
            key: 0.0
            for key in (
                "total",
                "field",
                "gradient",
                "support",
                "query",
                "router",
                "boundary",
                "weight",
                "target_weight",
                "target_weight_sq",
                "query_head_gap",
            )
        }
        sample_count = 0
        for batch in train_loader:
            x = batch["x"].to(device)
            target = batch["field"].to(device)
            observed = batch["observation"].to(device)
            support_mask = batch["support_mask"].to(device)
            query_mask = batch["query_mask"].to(device)
            max_gap = batch["max_gap"].to(device)
            optimizer.zero_grad(set_to_none=True)
            residual, absolute = model.experts(x)
            projected_residual = project_torch(residual, operator)
            projected_absolute = project_torch(absolute, operator)
            features = router_features(
                x,
                residual,
                absolute,
                projected_residual,
                projected_absolute,
                observed,
                support_mask,
                max_gap,
            )
            weight = model.route(features.detach())
            mixture = model.combine(residual, absolute, weight)
            projected_mixture = project_torch(mixture, operator)

            field_loss = 0.5 * (functional.mse_loss(residual, target) + functional.mse_loss(absolute, target))
            grad_loss = 0.5 * (gradient_mse(residual, target) + gradient_mse(absolute, target))
            support_loss = 0.5 * (
                masked_relative_projection_loss(projected_residual, observed, support_mask)
                + masked_relative_projection_loss(projected_absolute, observed, support_mask)
            )
            residual_query = per_sample_projection_error(projected_residual, observed, query_mask)
            absolute_query = per_sample_projection_error(projected_absolute, observed, query_mask)
            query_loss = per_sample_projection_error(projected_mixture, observed, query_mask).mean()
            temperature = float(experiment_config["router_temperature"])
            target_weight = torch.sigmoid((absolute_query.detach() - residual_query.detach()) / temperature)
            router_loss = functional.binary_cross_entropy(weight.flatten(), target_weight)
            outside = (x[:, 1:2] < 0.02).to(residual.dtype)
            boundary_loss = 0.5 * (
                torch.mean((residual * outside) ** 2) + torch.mean((absolute * outside) ** 2)
            )
            total = (
                field_loss
                + float(training["lambda_gradient"]) * grad_loss
                + float(training["lambda_reprojection"]) * support_loss
                + float(training["lambda_boundary"]) * boundary_loss
                + float(experiment_config["lambda_query"]) * query_loss
                + float(experiment_config["lambda_router"]) * router_loss
            )
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            batch_size = x.shape[0]
            sample_count += batch_size
            values = {
                "total": total,
                "field": field_loss,
                "gradient": grad_loss,
                "support": support_loss,
                "query": query_loss,
                "router": router_loss,
                "boundary": boundary_loss,
                "weight": weight.mean(),
                "target_weight": target_weight.mean(),
                "target_weight_sq": torch.mean(target_weight**2),
                "query_head_gap": torch.mean(torch.abs(residual_query - absolute_query)),
            }
            for key, value in values.items():
                running[key] += float(value.detach().cpu()) * batch_size

        model.eval()
        validation_values = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                predictions, _, _, _ = routed_predictions(
                    model,
                    x,
                    batch["observation"].to(device),
                    batch["view_mask"].to(device),
                    operator,
                    angles,
                )
                validation_values.append(float(batch_relative_l2(predictions["query_router"], batch["field"].to(device)).cpu()))
        val_rel_l2 = float(np.mean(validation_values))
        train_averages = {key: value / max(sample_count, 1) for key, value in running.items()}
        target_variance = max(
            train_averages["target_weight_sq"] - train_averages["target_weight"] ** 2,
            0.0,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "val_rel_l2": val_rel_l2,
                **{f"train_{key}": value for key, value in train_averages.items()},
                "train_target_weight_std": math.sqrt(target_variance),
            }
        )
        scheduler.step()
        if val_rel_l2 < best_val - 1e-5:
            best_val = val_rel_l2
            best_epoch = epoch + 1
            stale_epochs = 0
            best_state = copy.deepcopy(model.state_dict())
            best_state.pop("_metadata", None)
        else:
            stale_epochs += 1
        if stale_epochs >= int(training["early_stop_patience"]):
            break

    train_seconds = time.perf_counter() - start_time
    if best_state is None:
        raise RuntimeError("No dual-branch checkpoint was produced")
    model.load_state_dict(best_state)
    results_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, results_dir / "dual_branch.pt")
    with (results_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)
    return {
        "model": model,
        "device": device,
        "parameters": count_parameters(model),
        "best_val_rel_l2": best_val,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "train_seconds": train_seconds,
        "history": history,
    }


def evaluate_seed(
    record: dict,
    data: dict[str, np.ndarray],
    config: dict,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    model = record["model"]
    device = record["device"]
    operator_torch = torch.from_numpy(data["forward_matrix"]).to(device)
    operator_numpy = data["forward_matrix"]
    angles = data["angles"].astype(np.float32)
    n = int(config["grid_size"])
    depth = int(config["depth"])
    grids = make_grid_3d(n, depth)
    run_rows = []
    sample_rows = []
    model.eval()

    for split, indices in split_indices(data).items():
        if split not in SPLIT_ORDER:
            continue
        loader = DataLoader(BOSTDataset(data, indices), batch_size=int(config["training"]["batch_size"]))
        prediction_lists = {method: [] for method in METHODS}
        query_weight_list = []
        support_weight_list = []
        feature_list = []
        index_list = []
        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(device)
                predictions, weight, support_weight, features = routed_predictions(
                    model,
                    x,
                    batch["observation"].to(device),
                    batch["view_mask"].to(device),
                    operator_torch,
                    angles,
                )
                for method, prediction in predictions.items():
                    prediction_lists[method].append(prediction[:, 0].cpu().numpy())
                query_weight_list.append(weight.flatten().cpu().numpy())
                support_weight_list.append(support_weight.flatten().cpu().numpy())
                feature_list.append(features.cpu().numpy())
                index_list.append(batch["index"].cpu().numpy())

        predictions_numpy = {method: np.concatenate(values) for method, values in prediction_lists.items()}
        query_weights = np.concatenate(query_weight_list)
        support_weights = np.concatenate(support_weight_list)
        method_weights = {
            "residual_head": np.ones_like(query_weights),
            "absolute_head": np.zeros_like(query_weights),
            "uniform_dual": np.full_like(query_weights, 0.5),
            "support_fit_mix": support_weights,
            "query_router": query_weights,
        }
        features = np.concatenate(feature_list)
        global_indices = np.concatenate(index_list).astype(int)
        for method in METHODS:
            method_values = []
            for local_index, global_index in enumerate(global_indices):
                prediction = predictions_numpy[method][local_index]
                target = data["field"][global_index]
                projection = forward_volume(prediction, operator_numpy)
                clean = data["clean_observation"][global_index]
                observed_mask = data["view_mask"][global_index]
                heldout_mask = 1.0 - observed_mask
                positive = np.clip(prediction, 0.0, None)
                row = {
                    "seed": seed,
                    "method": method,
                    "split": split,
                    "sample_index": int(global_index),
                    "family_id": int(data["family_id"][global_index]),
                    "view_count": int(data["view_count"][global_index]),
                    "noise_level": float(data["noise_level"][global_index]),
                    "router_weight": float(method_weights[method][local_index]),
                    "max_angular_gap": float(features[local_index, 2]),
                    "residual_support_reprojection": float(features[local_index, 3]),
                    "absolute_support_reprojection": float(features[local_index, 4]),
                    "branch_disagreement": float(features[local_index, 5]),
                    "rel_l2": _relative_norm(prediction - target, target),
                    "gradient_rel_l2": _gradient_relative(prediction, target),
                    "observed_reprojection_rel_l2": _masked_projection_relative(projection, clean, observed_mask),
                    "heldout_reprojection_rel_l2": _masked_projection_relative(projection, clean, heldout_mask),
                    "mass_relative_error": float(abs(positive.sum() - target.sum()) / (target.sum() + 1e-8)),
                    "centroid_error": float(np.linalg.norm(_centroid(positive, grids) - _centroid(target, grids))),
                }
                sample_rows.append(row)
                method_values.append(row)
            run_rows.append(
                {
                    "seed": seed,
                    "experiment": method,
                    "split": split,
                    "parameters": int(record["parameters"]),
                    "train_seconds": float(record["train_seconds"]),
                    "best_epoch": int(record["best_epoch"]),
                    "best_val_rel_l2": float(record["best_val_rel_l2"]),
                    "router_weight_mean": float(method_weights[method].mean()),
                    **{
                        aggregate_name: float(np.mean([row[public_name.replace("field_rel_l2", "rel_l2")] for row in method_values]))
                        for public_name, aggregate_name in METRICS.items()
                    },
                }
            )
    return run_rows, sample_rows


def summarize_runs(run_rows: list[dict]) -> list[dict]:
    rows = []
    for method in METHODS:
        for split in SPLIT_ORDER:
            selected = [row for row in run_rows if row["experiment"] == method and row["split"] == split]
            summary = {
                "experiment": method,
                "split": split,
                "seed_count": len(selected),
                "parameters_mean": float(np.mean([row["parameters"] for row in selected])),
                "train_seconds_mean": float(np.mean([row["train_seconds"] for row in selected])),
            }
            for public_name, aggregate_name in METRICS.items():
                mean, seed_std, seed_ci = t_interval(float(row[aggregate_name]) for row in selected)
                summary[f"{public_name}_mean"] = mean
                summary[f"{public_name}_seed_std"] = seed_std
                summary[f"{public_name}_seed_ci95_t"] = seed_ci
            weight_mean, weight_std, weight_ci = t_interval(float(row["router_weight_mean"]) for row in selected)
            summary["router_weight_mean"] = weight_mean
            summary["router_weight_seed_std"] = weight_std
            summary["router_weight_seed_ci95_t"] = weight_ci
            rows.append(summary)
    return rows


def oracle_audit(sample_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    lookup = {
        (int(row["seed"]), str(row["split"]), int(row["sample_index"]), str(row["method"])): row
        for row in sample_rows
    }
    keys = sorted({(int(row["seed"]), str(row["split"]), int(row["sample_index"])) for row in sample_rows})
    regret_rows = []
    selection_rows = []
    for candidate in ("uniform_dual", "support_fit_mix", "query_router"):
        for metric in ("rel_l2", "heldout_reprojection_rel_l2"):
            values = []
            oracle_or_better = 0
            for seed, split, sample_index in keys:
                residual = float(lookup[(seed, split, sample_index, "residual_head")][metric])
                absolute = float(lookup[(seed, split, sample_index, "absolute_head")][metric])
                candidate_value = float(lookup[(seed, split, sample_index, candidate)][metric])
                oracle = min(residual, absolute)
                values.append((candidate_value - oracle) / max(oracle, 1e-8))
                oracle_or_better += int(candidate_value <= oracle + 1e-10)
            mean, cell_std, cell_ci = t_interval(values)
            regret_rows.append(
                {
                    "candidate": candidate,
                    "metric": metric,
                    "cell_count": len(values),
                    "mean_relative_regret": mean,
                    "cell_std": cell_std,
                    "normal_style_ci95": cell_ci,
                    "p95_relative_regret": float(np.percentile(values, 95)),
                    "oracle_or_better_fraction": oracle_or_better / len(values),
                }
            )

    for metric in ("rel_l2", "heldout_reprojection_rel_l2"):
        for split in [*SPLIT_ORDER, "all"]:
            selected_keys = keys if split == "all" else [key for key in keys if key[1] == split]
            correct = 0
            weights = []
            advantages = []
            for seed, split_name, sample_index in selected_keys:
                residual = float(lookup[(seed, split_name, sample_index, "residual_head")][metric])
                absolute = float(lookup[(seed, split_name, sample_index, "absolute_head")][metric])
                weight = float(lookup[(seed, split_name, sample_index, "query_router")]["router_weight"])
                residual_better = residual <= absolute
                correct += int((weight >= 0.5) == residual_better)
                weights.append(weight)
                advantages.append(absolute - residual)
            correlation = float(np.corrcoef(weights, advantages)[0, 1]) if np.std(weights) > 0 and np.std(advantages) > 0 else math.nan
            selection_rows.append(
                {
                    "metric": metric,
                    "split": split,
                    "cell_count": len(selected_keys),
                    "selection_accuracy": correct / len(selected_keys),
                    "weight_advantage_pearson": correlation,
                    "router_weight_mean": float(np.mean(weights)),
                }
            )
    return regret_rows, selection_rows


def _average_ranks(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    start = 0
    while start < len(array):
        stop = start + 1
        while stop < len(array) and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _correlation(left: list[float], right: list[float], rank: bool = False) -> float | None:
    left_array = _average_ranks(left) if rank else np.asarray(left, dtype=float)
    right_array = _average_ranks(right) if rank else np.asarray(right, dtype=float)
    if np.std(left_array) == 0.0 or np.std(right_array) == 0.0:
        return None
    return float(np.corrcoef(left_array, right_array)[0, 1])


def feature_alignment_audit(sample_rows: list[dict]) -> list[dict]:
    """Retrospective audit of which observable signals predict the better head."""
    lookup = {
        (int(row["seed"]), str(row["split"]), int(row["sample_index"]), str(row["method"])): row
        for row in sample_rows
    }
    keys = sorted({(int(row["seed"]), str(row["split"]), int(row["sample_index"])) for row in sample_rows})
    audit_rows = []
    for target_metric in ("rel_l2", "heldout_reprojection_rel_l2"):
        for split in [*SPLIT_ORDER, "all"]:
            selected_keys = keys if split == "all" else [key for key in keys if key[1] == split]
            target_advantage = []
            feature_values = {
                "support_reprojection_advantage": [],
                "heldout_reprojection_advantage": [],
                "branch_disagreement": [],
                "view_count": [],
                "noise_level": [],
                "max_angular_gap": [],
            }
            for seed, split_name, sample_index in selected_keys:
                residual = lookup[(seed, split_name, sample_index, "residual_head")]
                absolute = lookup[(seed, split_name, sample_index, "absolute_head")]
                target_advantage.append(float(absolute[target_metric]) - float(residual[target_metric]))
                feature_values["support_reprojection_advantage"].append(
                    float(absolute["absolute_support_reprojection"])
                    - float(residual["residual_support_reprojection"])
                )
                feature_values["heldout_reprojection_advantage"].append(
                    float(absolute["heldout_reprojection_rel_l2"])
                    - float(residual["heldout_reprojection_rel_l2"])
                )
                feature_values["branch_disagreement"].append(float(residual["branch_disagreement"]))
                feature_values["view_count"].append(float(residual["view_count"]))
                feature_values["noise_level"].append(float(residual["noise_level"]))
                feature_values["max_angular_gap"].append(float(residual["max_angular_gap"]))
            for feature, values in feature_values.items():
                directional = feature.endswith("_advantage")
                audit_rows.append(
                    {
                        "target_metric": target_metric,
                        "split": split,
                        "feature": feature,
                        "cell_count": len(selected_keys),
                        "pearson": _correlation(values, target_advantage),
                        "spearman": _correlation(values, target_advantage, rank=True),
                        "sign_selection_accuracy": (
                            float(np.mean((np.asarray(values) >= 0.0) == (np.asarray(target_advantage) >= 0.0)))
                            if directional
                            else None
                        ),
                    }
                )
    return audit_rows


def plot_field_heatmap(summary_rows: list[dict], output_dir: Path) -> None:
    lookup = {(row["experiment"], row["split"]): row for row in summary_rows}
    values = np.asarray([[lookup[(method, split)]["field_rel_l2_mean"] for split in SPLIT_ORDER] for method in METHODS])
    fig, axis = plt.subplots(figsize=(10.8, 4.8), constrained_layout=True)
    image = axis.imshow(values, cmap="YlGnBu", aspect="auto")
    axis.set_xticks(np.arange(len(SPLIT_ORDER)), [split.replace("test_", "").replace("_", "\n") for split in SPLIT_ORDER])
    axis.set_yticks(np.arange(len(METHODS)), [LABELS[method] for method in METHODS])
    axis.set_title("Dual-branch field error across five evaluation domains")
    threshold = 0.62 * float(values.max())
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            axis.text(column_index, row_index, f"{value:.3f}", ha="center", va="center", color="white" if value > threshold else "#17212b")
    fig.colorbar(image, ax=axis, label="Field relative L2")
    fig.savefig(output_dir / "t16_dual_field_heatmap.png", dpi=180)
    plt.close(fig)


def plot_router_weights(summary_rows: list[dict], output_dir: Path) -> None:
    lookup = {(row["experiment"], row["split"]): row for row in summary_rows}
    fig, axis = plt.subplots(figsize=(9.6, 4.4), constrained_layout=True)
    x = np.arange(len(SPLIT_ORDER))
    for method, label in (("support_fit_mix", "Closed-form support fit"), ("query_router", "Query-trained router")):
        values = [lookup[(method, split)]["router_weight_mean"] for split in SPLIT_ORDER]
        errors = [lookup[(method, split)]["router_weight_seed_ci95_t"] for split in SPLIT_ORDER]
        axis.errorbar(
            x,
            values,
            yerr=errors,
            marker="o",
            linewidth=2.0,
            capsize=4,
            color=COLORS[method],
            label=label,
        )
    axis.axhline(0.5, color="#31363b", linestyle="--", linewidth=1.0, label="Equal mixture")
    axis.set_xticks(x, [split.replace("test_", "").replace("_", "\n") for split in SPLIT_ORDER])
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Residual-head weight")
    axis.set_title("Inference-visible routing: analytic support fit vs learned query router")
    axis.grid(alpha=0.22)
    axis.legend()
    fig.savefig(output_dir / "t16_dual_router_weights.png", dpi=180)
    plt.close(fig)


def plot_oracle_regret(regret_rows: list[dict], output_dir: Path) -> None:
    lookup = {(row["candidate"], row["metric"]): row for row in regret_rows}
    candidates = ["uniform_dual", "support_fit_mix", "query_router"]
    x = np.arange(len(candidates))
    width = 0.34
    fig, axis = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    for offset, (metric, label, color) in enumerate(
        (("rel_l2", "Field", "#16806a"), ("heldout_reprojection_rel_l2", "Held-out", "#2e6f9e"))
    ):
        values = [100.0 * lookup[(candidate, metric)]["mean_relative_regret"] for candidate in candidates]
        axis.bar(x + (offset - 0.5) * width, values, width=width, color=color, alpha=0.86, label=label)
    axis.set_xticks(x, [LABELS[candidate] for candidate in candidates])
    axis.set_ylabel("Mean sample regret to best head (%)")
    axis.set_title("Which no-GT mixture exploits dual-head complementarity?")
    axis.grid(axis="y", alpha=0.22)
    axis.legend()
    fig.savefig(output_dir / "t16_dual_oracle_regret.png", dpi=180)
    plt.close(fig)


def plot_selection(sample_rows: list[dict], output_dir: Path) -> None:
    lookup = {
        (int(row["seed"]), str(row["split"]), int(row["sample_index"]), str(row["method"])): row
        for row in sample_rows
    }
    keys = sorted({(int(row["seed"]), str(row["split"]), int(row["sample_index"])) for row in sample_rows})
    split_colors = dict(zip(SPLIT_ORDER, ["#16806a", "#b56a2b", "#2e6f9e", "#b44f5f", "#7257a3"]))
    fig, axis = plt.subplots(figsize=(9.0, 5.2), constrained_layout=True)
    for split in SPLIT_ORDER:
        selected = [key for key in keys if key[1] == split]
        advantage = [
            float(lookup[(seed, split_name, index, "absolute_head")]["rel_l2"])
            - float(lookup[(seed, split_name, index, "residual_head")]["rel_l2"])
            for seed, split_name, index in selected
        ]
        weights = [float(lookup[(seed, split_name, index, "query_router")]["router_weight"]) for seed, split_name, index in selected]
        axis.scatter(advantage, weights, s=22, alpha=0.58, color=split_colors[split], label=split.replace("test_", ""))
    axis.axvline(0.0, color="#31363b", linestyle="--", linewidth=1.0)
    axis.axhline(0.5, color="#31363b", linestyle=":", linewidth=1.0)
    axis.set_xlabel("Absolute field error - Residual field error (positive: Residual better)")
    axis.set_ylabel("Router residual weight")
    axis.set_title("Does no-GT routing track the field-wise winning head?")
    axis.grid(alpha=0.2)
    axis.legend(ncol=3, fontsize=8.5)
    fig.savefig(output_dir / "t16_dual_selection_scatter.png", dpi=180)
    plt.close(fig)


def plot_feature_alignment(feature_rows: list[dict], output_dir: Path) -> None:
    selected = [
        row
        for row in feature_rows
        if row["target_metric"] == "rel_l2" and row["split"] == "all"
    ]
    labels = {
        "support_reprojection_advantage": "Support\nreprojection advantage",
        "heldout_reprojection_advantage": "Held-out\nadvantage (audit)",
        "branch_disagreement": "Branch\ndisagreement",
        "view_count": "View\ncount",
        "noise_level": "Noise\nmetadata",
        "max_angular_gap": "Angular\ngap",
    }
    lookup = {row["feature"]: row for row in selected}
    features = list(labels)
    x = np.arange(len(features))
    width = 0.36
    pearson = [float(lookup[feature]["pearson"] or 0.0) for feature in features]
    spearman = [float(lookup[feature]["spearman"] or 0.0) for feature in features]
    fig, axis = plt.subplots(figsize=(10.2, 4.8), constrained_layout=True)
    axis.bar(x - width / 2, pearson, width=width, color="#2e6f9e", label="Pearson")
    axis.bar(x + width / 2, spearman, width=width, color="#16806a", label="Spearman")
    axis.axhline(0.0, color="#31363b", linewidth=1.0)
    axis.set_xticks(x, [labels[feature] for feature in features])
    axis.set_ylim(-0.4, 0.85)
    axis.set_ylabel("Correlation with Absolute - Residual field error")
    axis.set_title("Which inference-visible signals identify the better field expert?")
    axis.grid(axis="y", alpha=0.22)
    axis.legend()
    fig.savefig(output_dir / "t16_dual_feature_alignment.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    experiment_config = read_json(args.config)
    dataset_config = read_json(args.config.parent / str(experiment_config["dataset_config"]))
    if args.device is not None:
        dataset_config["training"]["device"] = args.device
    if args.epochs is not None:
        dataset_config["training"]["epochs"] = args.epochs
    results_root = ROOT / "results"
    output_dir = results_root / str(experiment_config.get("output_dir", "dual_branch_query"))
    work_root = results_root / str(experiment_config.get("work_dir", "dual_work"))
    output_dir.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    dataset_path = results_root / f"{dataset_config['name']}_dataset.npz"
    generate_dataset(dataset_config, dataset_path, force=args.force_data)
    data = load_npz(dataset_path)
    indices = split_indices(data)
    train_dataset = SupportQueryDataset(
        data,
        indices["train"],
        support_specs=list(experiment_config["support_view_counts"]),
        variants_per_sample=int(experiment_config["support_variants_per_sample"]),
        augmentation_seed=int(experiment_config["augmentation_seed"]),
    )
    seeds = [int(seed) for seed in experiment_config["training_seeds"]]
    all_run_rows = []
    all_sample_rows = []
    seed_metadata = []
    for seed in seeds:
        set_seed(seed)
        model = make_dual_branch_model(
            dataset_config["models"]["fno"],
            in_channels=int(data["inputs"].shape[1]),
            router_features=len(experiment_config["router_features"]),
            router_hidden=int(experiment_config["router_hidden"]),
            expert_sharing=str(experiment_config.get("expert_sharing", "shared")),
        )
        record = train_dual_model(
            model,
            train_dataset,
            data,
            dataset_config,
            experiment_config,
            seed,
            work_root / str(seed),
        )
        run_rows, sample_rows = evaluate_seed(record, data, dataset_config, seed)
        all_run_rows.extend(run_rows)
        all_sample_rows.extend(sample_rows)
        seed_metadata.append(
            {
                "seed": seed,
                "parameters": int(record["parameters"]),
                "best_epoch": int(record["best_epoch"]),
                "best_val_rel_l2": float(record["best_val_rel_l2"]),
                "epochs_ran": int(record["epochs_ran"]),
                "train_seconds": float(record["train_seconds"]),
                "final_train_router_weight": float(record["history"][-1]["train_weight"]),
                "final_train_target_weight": float(record["history"][-1]["train_target_weight"]),
                "final_train_target_weight_std": float(record["history"][-1]["train_target_weight_std"]),
                "final_train_query_head_gap": float(record["history"][-1]["train_query_head_gap"]),
            }
        )
        print(
            f"seed={seed}: params={record['parameters']:,}, best val={record['best_val_rel_l2']:.4f}, "
            f"epoch={record['best_epoch']}, seconds={record['train_seconds']:.1f}"
        )

    summary_rows = summarize_runs(all_run_rows)
    regret_rows, selection_rows = oracle_audit(all_sample_rows)
    feature_rows = feature_alignment_audit(all_sample_rows)
    write_csv(output_dir / "dual_runs.csv", all_run_rows)
    write_csv(output_dir / "dual_summary.csv", summary_rows)
    write_csv(output_dir / "dual_sample_metrics.csv", all_sample_rows)
    write_csv(output_dir / "dual_oracle_regret.csv", regret_rows)
    write_csv(output_dir / "dual_selection_audit.csv", selection_rows)
    write_csv(output_dir / "dual_feature_alignment.csv", feature_rows)
    plot_field_heatmap(summary_rows, output_dir)
    plot_router_weights(summary_rows, output_dir)
    plot_oracle_regret(regret_rows, output_dir)
    plot_selection(all_sample_rows, output_dir)
    plot_feature_alignment(feature_rows, output_dir)

    regret_lookup = {(row["candidate"], row["metric"]): row for row in regret_rows}
    selection_lookup = {(row["metric"], row["split"]): row for row in selection_rows}
    feature_lookup = {
        (row["target_metric"], row["split"], row["feature"]): row
        for row in feature_rows
    }

    report = {
        "status": "completed_dual_branch_support_fit_prototype",
        "name": experiment_config["name"],
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device_request": dataset_config["training"]["device"],
        },
        "dataset": {
            "samples": int(len(data["field"])),
            "shape": list(data["field"].shape[1:]),
            "base_train_samples": int(len(indices["train"])),
            "fixed_support_query_variants": int(len(train_dataset)),
            "augmentation_fixed_across_optimization_seeds": True,
        },
        "support_query_contract": {
            "support": "Only support views construct the physics lift and router support residuals.",
            "query": "Query views come from the originally observed camera set and are excluded from the model input.",
            "router_target": "Soft preference derived only from noisy query-view projection error; no field truth enters router features or target.",
            "test_time": "The router uses view/noise metadata, angular gap, support reprojection of both heads, and branch disagreement.",
        },
        "analytic_support_contract": {
            "formula": "clip(<A_S(x_res-x_abs), y_S-A_S(x_abs)> / ||A_S(x_res-x_abs)||^2, 0, 1)",
            "inputs": "Only the two expert predictions, the known forward operator, and inference-visible support observations.",
            "role": "Mandatory no-ground-truth baseline for any learned router or correction.",
        },
        "model": {
            "architecture": (
                "Two independent one-output FNO experts with residual and absolute parameterizations, analytic support-fit mixture, and a six-feature MLP query router."
                if str(experiment_config.get("expert_sharing", "shared")) == "independent"
                else "One almost fully shared two-output FNO, residual and absolute outputs, analytic support-fit mixture, and a six-feature MLP query router."
            ),
            "expert_sharing": str(experiment_config.get("expert_sharing", "shared")),
            "router_features": experiment_config["router_features"],
            "seed_runs": seed_metadata,
        },
        "summary": summary_rows,
        "oracle_regret": regret_rows,
        "selection_audit": selection_rows,
        "feature_alignment": feature_rows,
        "key_findings": {
            "support_fit_field_regret": regret_lookup[("support_fit_mix", "rel_l2")]["mean_relative_regret"],
            "support_fit_field_oracle_or_better_fraction": regret_lookup[("support_fit_mix", "rel_l2")]["oracle_or_better_fraction"],
            "uniform_field_regret": regret_lookup[("uniform_dual", "rel_l2")]["mean_relative_regret"],
            "query_router_field_regret": regret_lookup[("query_router", "rel_l2")]["mean_relative_regret"],
            "query_router_field_selection_accuracy": selection_lookup[("rel_l2", "all")]["selection_accuracy"],
            "query_router_weight_field_advantage_pearson": selection_lookup[("rel_l2", "all")]["weight_advantage_pearson"],
            "support_advantage_field_spearman": feature_lookup[("rel_l2", "all", "support_reprojection_advantage")]["spearman"],
            "interpretation": "The analytic physics-visible mixture is the strongest no-GT route; the learned query router collapses near equal weighting.",
        },
        "recommended_next_test": [
            "Train independent or low-sharing residual and absolute experts with parameter-matched ensemble controls.",
            "Anchor routing at the closed-form support weight and learn only a bounded correction.",
            "Use ||A_S(x_res-x_abs)||^2 as an identifiability score for loss weighting and abstention.",
            "Test a support-nullspace field correction supervised by query views before scaling the router.",
        ],
        "claims_boundary": [
            "The three-view evaluation split is no longer view-count OOD because three-view support variants are introduced during training.",
            "The query cameras are synthetic training supervision; the deployed router does not receive them.",
            (
                "The experts are independent and therefore use roughly twice the operator capacity; matched-capacity single/ensemble controls remain required."
                if str(experiment_config.get("expert_sharing", "shared")) == "independent"
                else "The two heads share almost all parameters, so this is not a full independent mixture-of-experts."
            ),
            "The analytic support fit optimizes only the line segment between the two experts and can still fit support-view noise.",
            "Its closed form relies on the current linear synthetic forward operator; nonlinear ray bending needs line search or local linearization.",
            "Selection accuracy against field truth is retrospective audit only.",
            "The linear synthetic forward model is not the full OERF BOST ray geometry.",
            "Three optimization seeds are a stability screen rather than a population confidence interval.",
        ],
    }
    with (output_dir / "dual_report.json").open("w") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    write_checksum_manifest(
        output_dir,
        [
            "dual_runs.csv",
            "dual_summary.csv",
            "dual_sample_metrics.csv",
            "dual_oracle_regret.csv",
            "dual_selection_audit.csv",
            "dual_feature_alignment.csv",
            "dual_report.json",
        ],
    )

    lookup = {(row["experiment"], row["split"]): row for row in summary_rows}
    print("T16 dual-branch query prototype complete")
    for method in METHODS:
        row = lookup[(method, "test_iid")]
        print(
            f"{method}: field={row['field_rel_l2_mean']:.4f}, "
            f"heldout={row['heldout_reprojection_rel_l2_mean']:.4f}, "
            f"router_weight={row['router_weight_mean']:.3f}"
        )
    print(f"results: {output_dir}")


if __name__ == "__main__":
    main()
