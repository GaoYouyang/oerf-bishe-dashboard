"""Training, physical losses, evaluation metrics, and plots for T16."""

from __future__ import annotations

import csv
import copy
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

try:
    from .bost_physics import forward_volume, make_grid_3d
    from .data import BOSTDataset, split_indices
    from .models import count_parameters
except ImportError:
    from bost_physics import forward_volume, make_grid_3d
    from data import BOSTDataset, split_indices
    from models import count_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def project_torch(volume: torch.Tensor, operator: torch.Tensor) -> torch.Tensor:
    flat = volume[:, 0].flatten(start_dim=2)
    return torch.einsum("vnp,bdp->bdvn", operator, flat)


def gradient_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    losses = []
    for dimension in (2, 3, 4):
        losses.append(functional.mse_loss(torch.diff(prediction, dim=dimension), torch.diff(target, dim=dimension)))
    return torch.stack(losses).mean()


def masked_relative_projection_loss(
    projected: torch.Tensor,
    observed: torch.Tensor,
    view_mask: torch.Tensor,
) -> torch.Tensor:
    mask = view_mask[:, None, :, None]
    numerator = torch.sum(((projected - observed) * mask) ** 2)
    denominator = torch.sum((observed * mask) ** 2).clamp_min(1e-8)
    return numerator / denominator


def batch_relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    error = (prediction - target).flatten(start_dim=1)
    truth = target.flatten(start_dim=1)
    return torch.mean(torch.linalg.vector_norm(error, dim=1) / torch.linalg.vector_norm(truth, dim=1).clamp_min(1e-8))


def sample_weighted_mean(batch_means: list[float], batch_sizes: list[int]) -> float:
    if not batch_means or len(batch_means) != len(batch_sizes):
        raise ValueError("batch means and sizes must be non-empty and aligned")
    sample_count = sum(int(value) for value in batch_sizes)
    if sample_count <= 0:
        raise ValueError("validation sample count must be positive")
    return float(
        sum(float(mean) * int(size) for mean, size in zip(batch_means, batch_sizes))
        / sample_count
    )


def train_model(
    name: str,
    model: torch.nn.Module,
    data: dict[str, np.ndarray],
    config: dict,
    results_dir: Path,
) -> dict:
    training = config["training"]
    device = choose_device(str(training["device"]))
    indices = split_indices(data)
    generator = torch.Generator().manual_seed(int(config["seed"]))
    train_loader = DataLoader(
        BOSTDataset(data, indices["train"]),
        batch_size=int(training["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(BOSTDataset(data, indices["val"]), batch_size=int(training["batch_size"]))
    model = model.to(device)
    operator = torch.from_numpy(data["forward_matrix"]).to(device)
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
    checkpoint_dir = results_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{name}.pt"
    start_time = time.perf_counter()

    for epoch in range(int(training["epochs"])):
        model.train()
        running = {"total": 0.0, "field": 0.0, "gradient": 0.0, "projection": 0.0, "boundary": 0.0}
        sample_count = 0
        for batch in train_loader:
            x = batch["x"].to(device)
            target = batch["field"].to(device)
            observed = batch["observation"].to(device)
            view_mask = batch["view_mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            projected = project_torch(prediction, operator)
            field_loss = functional.mse_loss(prediction, target)
            grad_loss = gradient_mse(prediction, target)
            projection_loss = masked_relative_projection_loss(projected, observed, view_mask)
            outside = (x[:, 1:2] < 0.02).to(prediction.dtype)
            boundary_loss = torch.mean((prediction * outside) ** 2)
            total = (
                field_loss
                + float(training["lambda_gradient"]) * grad_loss
                + float(training["lambda_reprojection"]) * projection_loss
                + float(training["lambda_boundary"]) * boundary_loss
            )
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            batch_size = x.shape[0]
            sample_count += batch_size
            for key, value in (
                ("total", total),
                ("field", field_loss),
                ("gradient", grad_loss),
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
                validation_values.append(float(batch_relative_l2(prediction, batch["field"].to(device)).cpu()))
                validation_batch_sizes.append(int(batch["x"].shape[0]))
        val_rel_l2 = sample_weighted_mean(validation_values, validation_batch_sizes)
        row = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "val_rel_l2": val_rel_l2,
            **{f"train_{key}": value / max(sample_count, 1) for key, value in running.items()},
        }
        history.append(row)
        scheduler.step()

        if val_rel_l2 < best_val - 1e-5:
            best_val = val_rel_l2
            best_epoch = epoch + 1
            stale_epochs = 0
            best_state = copy.deepcopy(model.state_dict())
            best_state.pop("_metadata", None)
            torch.save(best_state, checkpoint_path)
        else:
            stale_epochs += 1
        if stale_epochs >= int(training["early_stop_patience"]):
            break

    train_seconds = time.perf_counter() - start_time
    if best_state is None:
        raise RuntimeError(f"No valid checkpoint was produced for {name}")
    model.load_state_dict(best_state)
    with (results_dir / f"history_{name}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)
    return {
        "name": name,
        "model": model,
        "device": device,
        "history": history,
        "best_val_rel_l2": best_val,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "train_seconds": train_seconds,
        "parameters": count_parameters(model),
    }


def _relative_norm(error: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(error.reshape(-1)) / (np.linalg.norm(reference.reshape(-1)) + 1e-8))


def _gradient_relative(prediction: np.ndarray, target: np.ndarray) -> float:
    error_parts = []
    target_parts = []
    for axis in (0, 1, 2):
        error_parts.append(np.diff(prediction - target, axis=axis).reshape(-1))
        target_parts.append(np.diff(target, axis=axis).reshape(-1))
    return _relative_norm(np.concatenate(error_parts), np.concatenate(target_parts))


def _masked_projection_relative(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    expanded = mask[None, :, None]
    return _relative_norm((prediction - target) * expanded, target * expanded)


def _centroid(field: np.ndarray, grids: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    positive = np.clip(field, 0.0, None)
    mass = float(positive.sum()) + 1e-8
    return np.asarray([float((positive * grid).sum() / mass) for grid in grids])


def collect_predictions(
    model: torch.nn.Module | None,
    dataset: BOSTDataset,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    if model is None:
        return dataset.data["lift"][dataset.indices][:, None].copy(), 0.0
    loader = DataLoader(dataset, batch_size=batch_size)
    model.eval()
    predictions = []
    elapsed = 0.0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            x = batch["x"].to(device)
            if batch_index == 0:
                _ = model(x)
                synchronize(device)
            start = time.perf_counter()
            prediction = model(x)
            synchronize(device)
            elapsed += time.perf_counter() - start
            predictions.append(prediction.cpu().numpy())
    return np.concatenate(predictions, axis=0), 1000.0 * elapsed / max(len(dataset), 1)


def evaluate_methods(
    data: dict[str, np.ndarray],
    trained: dict[str, dict],
    config: dict,
    results_dir: Path,
) -> tuple[list[dict], list[dict], dict]:
    indices_by_split = split_indices(data)
    operator = data["forward_matrix"]
    n = int(config["grid_size"])
    depth = int(config["depth"])
    grids = make_grid_3d(n, depth)
    methods = {"physics_lift": None, **trained}
    aggregate_rows = []
    sample_rows = []
    prediction_cache: dict[str, dict[str, np.ndarray]] = {}

    for method_name, record in methods.items():
        prediction_cache[method_name] = {}
        model = None if record is None else record["model"]
        device = torch.device("cpu") if record is None else record["device"]
        parameters = 0 if record is None else int(record["parameters"])
        train_seconds = 0.0 if record is None else float(record["train_seconds"])
        for split_name, indices in indices_by_split.items():
            if split_name in {"train", "val"}:
                continue
            dataset = BOSTDataset(data, indices)
            predictions, inference_ms = collect_predictions(
                model,
                dataset,
                device,
                int(config["training"]["batch_size"]),
            )
            predictions = predictions[:, 0]
            prediction_cache[method_name][split_name] = predictions
            values = []
            for local_index, global_index in enumerate(indices):
                prediction = predictions[local_index]
                target = data["field"][global_index]
                projection = forward_volume(prediction, operator)
                clean = data["clean_observation"][global_index]
                observed_mask = data["view_mask"][global_index]
                heldout_mask = 1.0 - observed_mask
                positive_prediction = np.clip(prediction, 0.0, None)
                row = {
                    "method": method_name,
                    "split": split_name,
                    "sample_index": int(global_index),
                    "family_id": int(data["family_id"][global_index]),
                    "view_count": int(data["view_count"][global_index]),
                    "noise_level": float(data["noise_level"][global_index]),
                    "rel_l2": _relative_norm(prediction - target, target),
                    "gradient_rel_l2": _gradient_relative(prediction, target),
                    "observed_reprojection_rel_l2": _masked_projection_relative(projection, clean, observed_mask),
                    "heldout_reprojection_rel_l2": _masked_projection_relative(projection, clean, heldout_mask),
                    "mass_relative_error": float(abs(positive_prediction.sum() - target.sum()) / (target.sum() + 1e-8)),
                    "centroid_error": float(np.linalg.norm(_centroid(positive_prediction, grids) - _centroid(target, grids))),
                }
                row["squared_rel_l2"] = row["rel_l2"] ** 2
                sample_rows.append(row)
                values.append(row)

            metric_names = [
                "rel_l2",
                "squared_rel_l2",
                "gradient_rel_l2",
                "observed_reprojection_rel_l2",
                "heldout_reprojection_rel_l2",
                "mass_relative_error",
                "centroid_error",
            ]
            aggregate = {
                "method": method_name,
                "split": split_name,
                "n": len(values),
                "parameters": parameters,
                "train_seconds": train_seconds,
                "inference_ms_per_sample": inference_ms,
            }
            for metric in metric_names:
                metric_values = np.asarray([row[metric] for row in values], dtype=float)
                aggregate[f"{metric}_mean"] = float(metric_values.mean())
                aggregate[f"{metric}_std"] = float(metric_values.std(ddof=1)) if len(metric_values) > 1 else 0.0
                aggregate[f"{metric}_ci95"] = float(1.96 * aggregate[f"{metric}_std"] / math.sqrt(max(len(metric_values), 1)))
            aggregate_rows.append(aggregate)

    with (results_dir / "sample_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(sample_rows)
    with (results_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(aggregate_rows)
    return aggregate_rows, sample_rows, prediction_cache


def plot_training(trained: dict[str, dict], results_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for name, record in trained.items():
        epochs = [row["epoch"] for row in record["history"]]
        axes[0].plot(epochs, [row["train_total"] for row in record["history"]], marker="o", label=name.upper())
        axes[1].plot(epochs, [row["val_rel_l2"] for row in record["history"]], marker="o", label=name.upper())
    axes[0].set_title("Training objective")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Weighted loss")
    axes[1].set_title("Validation relative L2")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Relative L2")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.savefig(results_dir / "t16_training_curves.png", dpi=180)
    plt.close(fig)


def plot_split_metrics(rows: list[dict], results_dir: Path) -> None:
    split_order = ["test_iid", "test_view_ood", "test_noise_ood", "test_joint_ood", "test_family_ood"]
    methods = ["physics_lift", "unet", "fno"]
    colors = {"physics_lift": "#8b9494", "unet": "#2e6f9e", "fno": "#16806a"}
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.0), constrained_layout=True)
    x = np.arange(len(split_order))
    width = 0.24
    lookup = {(row["method"], row["split"]): row for row in rows}
    for offset, method in enumerate(methods):
        positions = x + (offset - 1) * width
        for axis, metric, title in (
            (axes[0], "rel_l2_mean", "Field-space reconstruction"),
            (axes[1], "heldout_reprojection_rel_l2_mean", "Held-out-view reprojection"),
        ):
            values = [lookup[(method, split)][metric] for split in split_order]
            ci_metric = metric.replace("_mean", "_ci95")
            errors = [lookup[(method, split)][ci_metric] for split in split_order]
            axis.bar(
                positions,
                values,
                yerr=errors,
                capsize=2.5,
                width=width,
                label=method.replace("_", " "),
                color=colors[method],
            )
            axis.set_title(title)
            axis.set_ylabel("Relative L2, lower is better")
            axis.grid(axis="y", alpha=0.22)
    for axis in axes:
        axis.set_xticks(x, [value.replace("test_", "").replace("_", "\n") for value in split_order])
        axis.legend(ncol=3)
    fig.savefig(results_dir / "t16_split_metrics.png", dpi=180)
    plt.close(fig)


def plot_reconstruction_panel(
    data: dict[str, np.ndarray],
    predictions: dict[str, dict[str, np.ndarray]],
    results_dir: Path,
) -> None:
    indices = split_indices(data)["test_family_ood"]
    global_index = int(indices[0])
    z_index = data["field"].shape[1] // 2
    target = data["field"][global_index, z_index]
    panels = [("GT thin-front OOD", target, "viridis")]
    for method, title in (("physics_lift", "Physics lift"), ("unet", "3D U-Net"), ("fno", "Residual 3D FNO")):
        prediction = predictions[method]["test_family_ood"][0, z_index]
        panels.append((title, prediction, "viridis"))
    fno_prediction = predictions["fno"]["test_family_ood"][0, z_index]
    panels.append(("FNO absolute error", np.abs(fno_prediction - target), "magma"))
    common_min = min(float(panel[1].min()) for panel in panels[:4])
    common_max = max(float(panel[1].max()) for panel in panels[:4])
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4), constrained_layout=True)
    for panel_index, (axis, (title, image, cmap)) in enumerate(zip(axes, panels)):
        limits = {"vmin": common_min, "vmax": common_max} if panel_index < 4 else {}
        view = axis.imshow(image, origin="lower", cmap=cmap, **limits)
        axis.set_title(title, fontsize=9.5)
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(view, ax=axis, fraction=0.046, pad=0.03)
    fig.savefig(results_dir / "t16_reconstruction_panel.png", dpi=180)
    plt.close(fig)


def plot_accuracy_cost(rows: list[dict], results_dir: Path) -> None:
    iid = {row["method"]: row for row in rows if row["split"] == "test_iid"}
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    colors = {"physics_lift": "#8b9494", "unet": "#2e6f9e", "fno": "#16806a"}
    for method in ("unet", "fno"):
        row = iid[method]
        x = float(row["inference_ms_per_sample"])
        size = 90 + 35 * math.log10(max(float(row["parameters"]), 10))
        ax.scatter(x, row["rel_l2_mean"], s=size, color=colors[method], alpha=0.85)
        ax.annotate(method.replace("_", " "), (x, row["rel_l2_mean"]), xytext=(6, 5), textcoords="offset points")
    ax.axhline(
        iid["physics_lift"]["rel_l2_mean"],
        color=colors["physics_lift"],
        linestyle="--",
        label="physics lift error (lift runtime excluded)",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Neural forward inference ms/sample (log scale)")
    ax.set_ylabel("IID field relative L2")
    ax.set_title("Accuracy-cost checkpoint")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(results_dir / "t16_accuracy_cost.png", dpi=180)
    plt.close(fig)


def write_report(
    config: dict,
    data: dict[str, np.ndarray],
    trained: dict[str, dict],
    rows: list[dict],
    sample_rows: list[dict],
    results_dir: Path,
) -> dict:
    try:
        import neuralop

        neuralop_version = getattr(neuralop, "__version__", "unknown")
    except ImportError:
        neuralop_version = "missing"
    paired = []
    indexed = {
        (row["method"], row["split"], row["sample_index"]): row
        for row in sample_rows
    }
    comparison_metrics = [
        "rel_l2",
        "gradient_rel_l2",
        "heldout_reprojection_rel_l2",
        "mass_relative_error",
        "centroid_error",
    ]
    for split_name in sorted({row["split"] for row in sample_rows}):
        sample_ids = sorted({row["sample_index"] for row in sample_rows if row["split"] == split_name})
        for baseline in ("physics_lift", "unet"):
            comparison = {"split": split_name, "challenger": "fno", "baseline": baseline}
            for metric in comparison_metrics:
                fno_values = np.asarray([indexed[("fno", split_name, sample_id)][metric] for sample_id in sample_ids])
                base_values = np.asarray([indexed[(baseline, split_name, sample_id)][metric] for sample_id in sample_ids])
                delta = fno_values - base_values
                comparison[metric] = {
                    "mean_delta": float(delta.mean()),
                    "delta_ci95": float(1.96 * delta.std(ddof=1) / math.sqrt(len(delta))) if len(delta) > 1 else 0.0,
                    "mean_relative_gain": float(np.mean((base_values - fno_values) / np.maximum(base_values, 1e-8))),
                    "win_rate": float(np.mean(fno_values < base_values)),
                }
            fno_field = np.asarray([indexed[("fno", split_name, sample_id)]["rel_l2"] for sample_id in sample_ids])
            base_field = np.asarray([indexed[(baseline, split_name, sample_id)]["rel_l2"] for sample_id in sample_ids])
            fno_heldout = np.asarray(
                [indexed[("fno", split_name, sample_id)]["heldout_reprojection_rel_l2"] for sample_id in sample_ids]
            )
            base_heldout = np.asarray(
                [indexed[(baseline, split_name, sample_id)]["heldout_reprojection_rel_l2"] for sample_id in sample_ids]
            )
            comparison["cross_metric_audit"] = {
                "field_better_heldout_worse_count": int(np.sum((fno_field < base_field) & (fno_heldout > base_heldout))),
                "field_worse_heldout_better_count": int(np.sum((fno_field > base_field) & (fno_heldout < base_heldout))),
                "sample_count": len(sample_ids),
            }
            paired.append(comparison)

    report = {
        "status": "completed_smoke_benchmark",
        "config_name": config["name"],
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "neuraloperator": neuralop_version,
            "mps_available": bool(torch.backends.mps.is_available()),
        },
        "dataset": {
            "samples": int(len(data["field"])),
            "shape": list(data["field"].shape[1:]),
            "split_counts": {name: int(len(values)) for name, values in split_indices(data).items()},
            "train_only_global_calibration": {
                "scale": float(data["calibration"][0]),
                "offset": float(data["calibration"][1]),
            },
            "input_channels": ["calibrated_lift", "support", "view_fraction", "noise", "z", "y", "x"],
        },
        "models": {
            name: {
                "parameters": int(record["parameters"]),
                "epochs_ran": int(record["epochs_ran"]),
                "best_epoch": int(record["best_epoch"]),
                "best_val_rel_l2": float(record["best_val_rel_l2"]),
                "train_seconds": float(record["train_seconds"]),
                "device": str(record["device"]),
            }
            for name, record in trained.items()
        },
        "metrics": rows,
        "paired_comparisons": paired,
        "claims_boundary": [
            "Synthetic closure test only; no claim of real OERF data performance.",
            "The train-only affine calibration is shared across all samples; no per-sample truth alignment is used.",
            "The linear stack model is a controlled BOST surrogate, not the full nonlinear ray model in NeRIF.",
            "Resolution transfer and real-data transfer remain untested in this smoke configuration.",
        ],
    }
    with (results_dir / "run_report.json").open("w") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    return report
