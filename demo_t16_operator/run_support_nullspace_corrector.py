#!/usr/bin/env python3
"""Train matched free and support-nullspace 3D neural-operator correctors."""

from __future__ import annotations

import argparse
import copy
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
    from .bost_physics import forward_volume
    from .data import BOSTDataset, generate_dataset, load_npz, split_indices
    from .models import count_parameters, make_dual_branch_model, make_model
    from .run_ablations import SPLIT_ORDER, t_interval
    from .run_dual_branch_query import SupportQueryDataset, closed_form_support_weight
    from .run_nullspace_identifiability_audit import (
        nullspace_decomposition,
        project_to_nullspace,
    )
    from .train_eval import (
        _gradient_relative,
        _masked_projection_relative,
        _relative_norm,
        batch_relative_l2,
        gradient_mse,
        masked_relative_projection_loss,
        project_torch,
        set_seed,
    )
except ImportError:
    from bost_physics import forward_volume
    from data import BOSTDataset, generate_dataset, load_npz, split_indices
    from models import count_parameters, make_dual_branch_model, make_model
    from run_ablations import SPLIT_ORDER, t_interval
    from run_dual_branch_query import SupportQueryDataset, closed_form_support_weight
    from run_nullspace_identifiability_audit import nullspace_decomposition, project_to_nullspace
    from train_eval import (
        _gradient_relative,
        _masked_projection_relative,
        _relative_norm,
        batch_relative_l2,
        gradient_mse,
        masked_relative_projection_loss,
        project_torch,
        set_seed,
    )


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "support_nullspace_corrector.json"
DEFAULT_RESULTS = ROOT / "results" / "support_nullspace_corrector"
DEFAULT_WORK = ROOT / "results" / "support_nullspace_corrector_work"
METHODS = [
    "support_fit_base",
    "free_correction",
    "nullspace_correction",
    "oracle_null_upper_bound",
]
LABELS = {
    "support_fit_base": "independent support fit",
    "free_correction": "free FNO correction",
    "nullspace_correction": "support-nullspace FNO",
    "oracle_null_upper_bound": "truth-oracle null upper bound",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class TorchSupportNullProjector:
    """Exact SVD nullspace projection for each declared support-view mask."""

    def __init__(self, operator: np.ndarray):
        self.operator = np.asarray(operator)
        self.numpy_cache: dict[tuple[int, ...], np.ndarray] = {}
        self.torch_cache: dict[tuple[tuple[int, ...], str], torch.Tensor] = {}

    @staticmethod
    def key(mask: torch.Tensor | np.ndarray) -> tuple[int, ...]:
        values = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else mask
        return tuple(int(value > 0.5) for value in np.asarray(values).tolist())

    def numpy_basis(self, mask: torch.Tensor | np.ndarray) -> np.ndarray:
        key = self.key(mask)
        if key not in self.numpy_cache:
            decomposition = nullspace_decomposition(
                self.operator,
                np.asarray(key, dtype=np.float32),
            )
            self.numpy_cache[key] = np.asarray(decomposition["null_basis"], dtype=np.float64)
        return self.numpy_cache[key]

    def torch_basis(self, mask: torch.Tensor, device: torch.device) -> torch.Tensor:
        key = self.key(mask)
        cache_key = (key, str(device))
        if cache_key not in self.torch_cache:
            self.torch_cache[cache_key] = torch.from_numpy(self.numpy_basis(mask)).to(device=device)
        return self.torch_cache[cache_key]

    def project(self, correction: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        projected = []
        for index in range(correction.shape[0]):
            basis = self.torch_basis(masks[index], correction.device)
            flat = correction[index, 0].flatten(start_dim=1).to(torch.float64)
            value = (flat @ basis.T) @ basis
            projected.append(value.reshape_as(correction[index, 0]).to(correction.dtype))
        return torch.stack(projected, dim=0)[:, None]


def bounded_correction(
    correction: torch.Tensor,
    base: torch.Tensor,
    cap_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    correction_norm = torch.linalg.vector_norm(correction.flatten(start_dim=1), dim=1)
    base_norm = torch.linalg.vector_norm(base.flatten(start_dim=1), dim=1).clamp_min(1e-8)
    maximum = float(cap_ratio) * base_norm
    scale = torch.minimum(torch.ones_like(maximum), maximum / correction_norm.clamp_min(1e-8))
    return correction * scale[:, None, None, None, None], scale


def load_base_model(
    base_config: dict,
    dataset_config: dict,
    data: dict[str, np.ndarray],
    seed: int,
    device: torch.device,
) -> torch.nn.Module:
    model = make_dual_branch_model(
        dataset_config["models"]["fno"],
        in_channels=int(data["inputs"].shape[1]),
        router_features=len(base_config["router_features"]),
        router_hidden=int(base_config["router_hidden"]),
        expert_sharing="independent",
    )
    checkpoint = ROOT / "results" / str(base_config["work_dir"]) / str(seed) / "dual_branch.pt"
    if not checkpoint.exists():
        raise SystemExit(f"missing independent base checkpoint: {checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def support_fit_state(
    base_model: torch.nn.Module,
    x: torch.Tensor,
    observation: torch.Tensor,
    support_mask: torch.Tensor,
    operator: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        residual, absolute = base_model.experts(x)
        projected_residual = project_torch(residual, operator)
        projected_absolute = project_torch(absolute, operator)
        weight = closed_form_support_weight(
            projected_residual,
            projected_absolute,
            observation,
            support_mask,
        )
        base = base_model.combine(residual, absolute, weight)
    return base, residual, absolute, weight


def corrector_input(
    x: torch.Tensor,
    base: torch.Tensor,
    residual: torch.Tensor,
    absolute: torch.Tensor,
) -> torch.Tensor:
    return torch.cat([x, base, residual - absolute], dim=1)


def corrected_prediction(
    mode: str,
    model: torch.nn.Module,
    inputs: torch.Tensor,
    base: torch.Tensor,
    support_mask: torch.Tensor,
    projector: TorchSupportNullProjector,
    cap_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = model(inputs)
    constrained = projector.project(raw, support_mask) if mode == "nullspace_correction" else raw
    correction, scale = bounded_correction(constrained, base, cap_ratio)
    return base + correction, correction, scale


def train_corrector_pair(
    base_model: torch.nn.Module,
    train_dataset: SupportQueryDataset,
    data: dict[str, np.ndarray],
    dataset_config: dict,
    experiment_config: dict,
    seed: int,
    work_dir: Path,
) -> dict[str, dict[str, object]]:
    training = experiment_config["training"]
    device = torch.device(str(training["device"]))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        BOSTDataset(data, split_indices(data)["val"]),
        batch_size=int(training["batch_size"]),
    )
    operator = torch.from_numpy(data["forward_matrix"]).to(device)
    projector = TorchSupportNullProjector(data["forward_matrix"])
    modes = [str(value) for value in experiment_config["correction_modes"]]
    template = make_model(
        "fno",
        experiment_config["model"],
        in_channels=int(data["inputs"].shape[1]) + 2,
        residual=False,
    ).to(device)
    models = {mode: copy.deepcopy(template) for mode in modes}
    optimizers = {
        mode: torch.optim.AdamW(
            model.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        for mode, model in models.items()
    }
    schedulers = {
        mode: torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizers[mode], T_max=max(int(training["epochs"]), 1)
        )
        for mode in modes
    }
    best = {mode: {"value": math.inf, "epoch": -1, "state": None} for mode in modes}
    stale = {mode: 0 for mode in modes}
    histories: dict[str, list[dict[str, object]]] = {mode: [] for mode in modes}
    started = time.perf_counter()

    for epoch in range(int(training["epochs"])):
        running = {
            mode: defaultdict(float)
            for mode in modes
        }
        sample_count = 0
        for model in models.values():
            model.train()
        for batch in train_loader:
            x = batch["x"].to(device)
            target = batch["field"].to(device)
            observation = batch["observation"].to(device)
            support_mask = batch["support_mask"].to(device)
            query_mask = batch["query_mask"].to(device)
            base, residual, absolute, _ = support_fit_state(
                base_model, x, observation, support_mask, operator
            )
            inputs = corrector_input(x, base, residual, absolute).detach()
            batch_size = x.shape[0]
            sample_count += batch_size

            for mode, model in models.items():
                optimizers[mode].zero_grad(set_to_none=True)
                prediction, correction, scale = corrected_prediction(
                    mode,
                    model,
                    inputs,
                    base,
                    support_mask,
                    projector,
                    float(experiment_config["correction_cap_ratio"]),
                )
                projected = project_torch(prediction, operator)
                field_loss = functional.mse_loss(prediction, target)
                gradient_loss = gradient_mse(prediction, target)
                support_loss = masked_relative_projection_loss(
                    projected, observation, support_mask
                )
                query_loss = masked_relative_projection_loss(
                    projected, observation, query_mask
                )
                outside = (x[:, 1:2] < 0.02).to(prediction.dtype)
                boundary_loss = torch.mean((prediction * outside) ** 2)
                relative_correction = torch.linalg.vector_norm(
                    correction.flatten(start_dim=1), dim=1
                ) / torch.linalg.vector_norm(base.flatten(start_dim=1), dim=1).clamp_min(1e-8)
                correction_loss = torch.mean(relative_correction**2)
                total = (
                    field_loss
                    + float(training["lambda_gradient"]) * gradient_loss
                    + float(training["lambda_support"]) * support_loss
                    + float(training["lambda_query"]) * query_loss
                    + float(training["lambda_boundary"]) * boundary_loss
                    + float(training["lambda_correction_norm"]) * correction_loss
                )
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizers[mode].step()
                for key, value in {
                    "total": total,
                    "field": field_loss,
                    "gradient": gradient_loss,
                    "support": support_loss,
                    "query": query_loss,
                    "boundary": boundary_loss,
                    "correction_norm": correction_loss,
                    "cap_scale": scale.mean(),
                }.items():
                    running[mode][key] += float(value.detach().cpu()) * batch_size

        validation: dict[str, list[float]] = {mode: [] for mode in modes}
        for model in models.values():
            model.eval()
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                target = batch["field"].to(device)
                observation = batch["observation"].to(device)
                mask = batch["view_mask"].to(device)
                base, residual, absolute, _ = support_fit_state(
                    base_model, x, observation, mask, operator
                )
                inputs = corrector_input(x, base, residual, absolute)
                for mode, model in models.items():
                    prediction, _, _ = corrected_prediction(
                        mode,
                        model,
                        inputs,
                        base,
                        mask,
                        projector,
                        float(experiment_config["correction_cap_ratio"]),
                    )
                    validation[mode].append(float(batch_relative_l2(prediction, target).cpu()))

        for mode in modes:
            val_rel_l2 = float(np.mean(validation[mode]))
            histories[mode].append(
                {
                    "epoch": epoch + 1,
                    "learning_rate": optimizers[mode].param_groups[0]["lr"],
                    "val_rel_l2": val_rel_l2,
                    **{
                        f"train_{key}": value / max(sample_count, 1)
                        for key, value in running[mode].items()
                    },
                }
            )
            schedulers[mode].step()
            if val_rel_l2 < float(best[mode]["value"]) - 1e-5:
                state = copy.deepcopy(models[mode].state_dict())
                state.pop("_metadata", None)
                best[mode] = {
                    "value": val_rel_l2,
                    "epoch": epoch + 1,
                    "state": state,
                }
                stale[mode] = 0
            else:
                stale[mode] += 1
        if all(stale[mode] >= int(training["early_stop_patience"]) for mode in modes):
            break

    elapsed = time.perf_counter() - started
    work_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, object]] = {}
    for mode in modes:
        if best[mode]["state"] is None:
            raise RuntimeError(f"no checkpoint produced for {mode}")
        models[mode].load_state_dict(best[mode]["state"])
        models[mode].eval()
        torch.save(best[mode]["state"], work_dir / f"{mode}.pt")
        history_path = work_dir / f"history_{mode}.csv"
        with history_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(histories[mode][0].keys()),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(histories[mode])
        records[mode] = {
            "model": models[mode],
            "parameters": count_parameters(models[mode]),
            "best_val_rel_l2": float(best[mode]["value"]),
            "best_epoch": int(best[mode]["epoch"]),
            "epochs_ran": len(histories[mode]),
            "pair_train_seconds": elapsed,
            "history": histories[mode],
        }
    return records


def sample_metric(
    method: str,
    prediction: np.ndarray,
    base: np.ndarray,
    target: np.ndarray,
    clean: np.ndarray,
    mask: np.ndarray,
    operator: np.ndarray,
) -> dict[str, float | str]:
    projection = forward_volume(prediction, operator)
    correction_projection = forward_volume(prediction - base, operator)
    query_mask = 1.0 - mask
    base_error = _relative_norm(base - target, target)
    field_error = _relative_norm(prediction - target, target)
    correction_norm = _relative_norm(prediction - base, base)
    return {
        "method": method,
        "rel_l2": field_error,
        "gradient_rel_l2": _gradient_relative(prediction, target),
        "support_reprojection_rel_l2": _masked_projection_relative(
            projection, clean, mask
        ),
        "heldout_reprojection_rel_l2": _masked_projection_relative(
            projection, clean, query_mask
        ),
        "support_correction_leakage": _relative_norm(
            correction_projection * mask[None, :, None],
            clean * mask[None, :, None],
        ),
        "correction_norm_ratio": correction_norm,
        "field_improvement_vs_base_pct": 100.0
        * (base_error - field_error)
        / (base_error + 1e-12),
    }


def evaluate_seed(
    seed: int,
    base_model: torch.nn.Module,
    records: dict[str, dict[str, object]],
    data: dict[str, np.ndarray],
    dataset_config: dict,
    experiment_config: dict,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    device = torch.device(str(experiment_config["training"]["device"]))
    operator_torch = torch.from_numpy(data["forward_matrix"]).to(device)
    operator_numpy = np.asarray(data["forward_matrix"])
    projector = TorchSupportNullProjector(operator_numpy)
    sample_rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []
    modes = [str(value) for value in experiment_config["correction_modes"]]

    for split, indices in split_indices(data).items():
        if split not in SPLIT_ORDER:
            continue
        loader = DataLoader(
            BOSTDataset(data, indices),
            batch_size=int(experiment_config["training"]["batch_size"]),
        )
        split_rows: list[dict[str, object]] = []
        local_offset = 0
        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(device)
                observation = batch["observation"].to(device)
                mask = batch["view_mask"].to(device)
                base, residual, absolute, weight = support_fit_state(
                    base_model, x, observation, mask, operator_torch
                )
                inputs = corrector_input(x, base, residual, absolute)
                prediction_tensors: dict[str, torch.Tensor] = {
                    "support_fit_base": base,
                }
                for mode in modes:
                    prediction_tensors[mode] = corrected_prediction(
                        mode,
                        records[mode]["model"],
                        inputs,
                        base,
                        mask,
                        projector,
                        float(experiment_config["correction_cap_ratio"]),
                    )[0]
                base_numpy = base[:, 0].cpu().numpy()
                for batch_index in range(x.shape[0]):
                    sample_index = int(indices[local_offset + batch_index])
                    target = np.asarray(data["field"][sample_index])
                    clean = np.asarray(data["clean_observation"][sample_index])
                    mask_numpy = np.asarray(data["view_mask"][sample_index])
                    basis = projector.numpy_basis(mask_numpy)
                    oracle = base_numpy[batch_index] + project_to_nullspace(
                        target - base_numpy[batch_index], basis
                    )
                    predictions = {
                        method: tensor[batch_index, 0].cpu().numpy()
                        for method, tensor in prediction_tensors.items()
                    }
                    predictions["oracle_null_upper_bound"] = oracle
                    for method, prediction in predictions.items():
                        row: dict[str, object] = {
                            "seed": seed,
                            "split": split,
                            "sample_index": sample_index,
                            "sample_seed": int(data["sample_seed"][sample_index]),
                            "family_id": int(data["family_id"][sample_index]),
                            "view_count": int(data["view_count"][sample_index]),
                            "noise_level": float(data["noise_level"][sample_index]),
                            "support_fit_residual_weight": float(
                                weight[batch_index, 0, 0, 0, 0].cpu()
                            ),
                        }
                        row.update(
                            sample_metric(
                                method,
                                np.asarray(prediction, dtype=np.float64),
                                np.asarray(base_numpy[batch_index], dtype=np.float64),
                                np.asarray(target, dtype=np.float64),
                                np.asarray(clean, dtype=np.float64),
                                np.asarray(mask_numpy, dtype=np.float64),
                                np.asarray(operator_numpy, dtype=np.float64),
                            )
                        )
                        split_rows.append(row)
                        sample_rows.append(row)
                local_offset += x.shape[0]

        for method in METHODS:
            subset = [row for row in split_rows if row["method"] == method]
            run_rows.append(
                {
                    "seed": seed,
                    "split": split,
                    "method": method,
                    "sample_count": len(subset),
                    "parameters": 0
                    if method in {"support_fit_base", "oracle_null_upper_bound"}
                    else int(records[method]["parameters"]),
                    **{
                        f"{field}_mean": float(np.mean([float(row[field]) for row in subset]))
                        for field in [
                            "rel_l2",
                            "gradient_rel_l2",
                            "support_reprojection_rel_l2",
                            "heldout_reprojection_rel_l2",
                            "support_correction_leakage",
                            "correction_norm_ratio",
                            "field_improvement_vs_base_pct",
                        ]
                    },
                }
            )
    return run_rows, sample_rows


def summarize_runs(run_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    metrics = [
        "rel_l2_mean",
        "gradient_rel_l2_mean",
        "support_reprojection_rel_l2_mean",
        "heldout_reprojection_rel_l2_mean",
        "support_correction_leakage_mean",
        "correction_norm_ratio_mean",
        "field_improvement_vs_base_pct_mean",
    ]
    for method in METHODS:
        for split in SPLIT_ORDER:
            subset = [
                row
                for row in run_rows
                if row["method"] == method and row["split"] == split
            ]
            row: dict[str, object] = {
                "method": method,
                "split": split,
                "seed_count": len(subset),
                "parameters": int(np.mean([int(value["parameters"]) for value in subset])),
            }
            for metric in metrics:
                mean, standard_deviation, interval = t_interval(
                    float(value[metric]) for value in subset
                )
                row[metric] = mean
                row[f"{metric}_seed_std"] = standard_deviation
                row[f"{metric}_seed_ci95_t"] = interval
            output.append(row)
    return output


def plot_field_heatmap(summary: list[dict[str, object]], path: Path) -> None:
    matrix = np.asarray(
        [
            [
                float(
                    next(
                        row["rel_l2_mean"]
                        for row in summary
                        if row["method"] == method and row["split"] == split
                    )
                )
                for split in SPLIT_ORDER
            ]
            for method in METHODS
        ]
    )
    fig, ax = plt.subplots(figsize=(10.6, 4.6), constrained_layout=True)
    image = ax.imshow(matrix, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(
        np.arange(len(SPLIT_ORDER)),
        [split.replace("test_", "").replace("_", "\n") for split in SPLIT_ORDER],
    )
    ax.set_yticks(np.arange(len(METHODS)), [LABELS[method] for method in METHODS])
    ax.set_title("Can a learned support-nullspace operator capture oracle headroom?")
    threshold = 0.62 * float(matrix.max())
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            ax.text(
                column_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if value > threshold else "#263238",
                fontsize=9,
            )
    fig.colorbar(image, ax=ax, label="field relative L2")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_improvement(summary: list[dict[str, object]], path: Path) -> None:
    x = np.arange(len(SPLIT_ORDER))
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 4.8), constrained_layout=True)
    for ax, (method, color) in zip(
        axes,
        (("free_correction", "#315f8d"), ("nullspace_correction", "#196e63")),
    ):
        values = [
            float(
                next(
                    row["field_improvement_vs_base_pct_mean"]
                    for row in summary
                    if row["method"] == method and row["split"] == split
                )
            )
            for split in SPLIT_ORDER
        ]
        errors = [
            float(
                next(
                    row["field_improvement_vs_base_pct_mean_seed_ci95_t"]
                    for row in summary
                    if row["method"] == method and row["split"] == split
                )
            )
            for split in SPLIT_ORDER
        ]
        ax.bar(
            x,
            values,
            width=0.62,
            yerr=errors,
            capsize=3,
            color=color,
        )
        ax.axhline(0.0, color="#555555", linewidth=1)
        ax.set_xticks(
            x,
            [split.replace("test_", "").replace("_", "\n") for split in SPLIT_ORDER],
        )
        ax.set_ylabel("field improvement over support fit (%)")
        ax.set_title(LABELS[method])
        ax.grid(True, axis="y", alpha=0.24)
    fig.suptitle(
        "Matched correctors on separate scales: stability is not the same as net improvement",
        fontsize=13.5,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_leakage(sample_rows: list[dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.1), constrained_layout=True)
    colors = {"free_correction": "#315f8d", "nullspace_correction": "#196e63"}
    for ax, method in zip(axes, colors):
        subset = [row for row in sample_rows if row["method"] == method]
        ax.scatter(
            [float(row["support_correction_leakage"]) for row in subset],
            [float(row["field_improvement_vs_base_pct"]) for row in subset],
            s=20,
            alpha=0.38,
            color=colors[method],
        )
        ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1)
        ax.set_xscale("symlog", linthresh=1e-8)
        ax.set_xlabel("support projection change")
        ax.set_ylabel("field improvement over support fit (%)")
        ax.set_title(LABELS[method])
        ax.grid(True, alpha=0.22)
    fig.suptitle(
        "Hard nullspace projection trades free correction for explicit support consistency",
        fontsize=13.5,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_checksums(output_dir: Path, filenames: list[str]) -> None:
    lines = []
    for filename in filenames:
        digest = hashlib.sha256((output_dir / filename).read_bytes()).hexdigest()
        lines.append(f"{digest}  {filename}")
    (output_dir / "support_nullspace_checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="ascii"
    )


def main() -> None:
    args = parse_args()
    experiment_config = read_json(args.config)
    base_config_path = args.config.parent / str(
        experiment_config["base_experiment_config"]
    )
    base_config = read_json(base_config_path)
    dataset_config = read_json(base_config_path.parent / str(base_config["dataset_config"]))
    if args.epochs is not None:
        experiment_config["training"]["epochs"] = args.epochs
    if args.device is not None:
        experiment_config["training"]["device"] = args.device
    device = torch.device(str(experiment_config["training"]["device"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = ROOT / "results" / f"{dataset_config['name']}_dataset.npz"
    generate_dataset(dataset_config, dataset_path)
    data = load_npz(dataset_path)
    indices = split_indices(data)
    train_dataset = SupportQueryDataset(
        data,
        indices["train"],
        support_specs=list(base_config["support_view_counts"]),
        variants_per_sample=int(base_config["support_variants_per_sample"]),
        augmentation_seed=int(base_config["augmentation_seed"]),
    )
    all_run_rows: list[dict[str, object]] = []
    all_sample_rows: list[dict[str, object]] = []
    training_records = []

    for seed in [int(value) for value in base_config["training_seeds"]]:
        set_seed(seed + 50_000)
        base_model = load_base_model(base_config, dataset_config, data, seed, device)
        records = train_corrector_pair(
            base_model,
            train_dataset,
            data,
            dataset_config,
            experiment_config,
            seed,
            args.work_dir / str(seed),
        )
        run_rows, sample_rows = evaluate_seed(
            seed,
            base_model,
            records,
            data,
            dataset_config,
            experiment_config,
        )
        all_run_rows.extend(run_rows)
        all_sample_rows.extend(sample_rows)
        training_records.append(
            {
                "seed": seed,
                "modes": {
                    mode: {
                        "parameters": int(record["parameters"]),
                        "best_val_rel_l2": float(record["best_val_rel_l2"]),
                        "best_epoch": int(record["best_epoch"]),
                        "epochs_ran": int(record["epochs_ran"]),
                        "pair_train_seconds": float(record["pair_train_seconds"]),
                    }
                    for mode, record in records.items()
                },
            }
        )
        print(
            f"seed={seed}: "
            + ", ".join(
                f"{mode} val={float(record['best_val_rel_l2']):.4f} epoch={int(record['best_epoch'])}"
                for mode, record in records.items()
            ),
            flush=True,
        )

    summary = summarize_runs(all_run_rows)
    write_csv(args.output_dir / "support_nullspace_runs.csv", all_run_rows)
    write_csv(args.output_dir / "support_nullspace_samples.csv", all_sample_rows)
    write_csv(args.output_dir / "support_nullspace_summary.csv", summary)
    plot_field_heatmap(summary, args.output_dir / "t16_support_nullspace_field_heatmap.png")
    plot_improvement(summary, args.output_dir / "t16_support_nullspace_improvement.png")
    plot_leakage(all_sample_rows, args.output_dir / "t16_support_nullspace_leakage.png")

    sample_lookup = {
        (
            int(row["seed"]),
            str(row["split"]),
            int(row["sample_index"]),
            str(row["method"]),
        ): row
        for row in all_sample_rows
    }
    keys = sorted(
        {
            (int(row["seed"]), str(row["split"]), int(row["sample_index"]))
            for row in all_sample_rows
        }
    )
    paired = {}
    for method in ["free_correction", "nullspace_correction"]:
        improvements = []
        heldout_improvements = []
        oracle_capture = []
        better_than_base = 0
        for seed, split, sample_index in keys:
            base = sample_lookup[(seed, split, sample_index, "support_fit_base")]
            candidate = sample_lookup[(seed, split, sample_index, method)]
            oracle = sample_lookup[(seed, split, sample_index, "oracle_null_upper_bound")]
            base_error = float(base["rel_l2"])
            candidate_error = float(candidate["rel_l2"])
            oracle_error = float(oracle["rel_l2"])
            improvements.append(100.0 * (base_error - candidate_error) / (base_error + 1e-12))
            heldout_improvements.append(
                100.0
                * (
                    float(base["heldout_reprojection_rel_l2"])
                    - float(candidate["heldout_reprojection_rel_l2"])
                )
                / (float(base["heldout_reprojection_rel_l2"]) + 1e-12)
            )
            oracle_capture.append(
                (base_error - candidate_error) / max(base_error - oracle_error, 1e-8)
            )
            better_than_base += int(candidate_error < base_error)
        paired[method] = {
            "sample_seed_cells": len(keys),
            "mean_field_improvement_pct": float(np.mean(improvements)),
            "p10_field_improvement_pct": float(np.percentile(improvements, 10)),
            "field_better_than_base_fraction": better_than_base / len(keys),
            "mean_heldout_improvement_pct": float(np.mean(heldout_improvements)),
            "mean_oracle_headroom_capture_fraction": float(np.mean(oracle_capture)),
        }

    null_rows = [
        row for row in all_sample_rows if row["method"] == "nullspace_correction"
    ]
    report = {
        "status": "completed_matched_free_vs_support_nullspace_corrector",
        "experiment": experiment_config["name"],
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
        },
        "dataset": {
            "samples": int(len(data["field"])),
            "shape": list(data["field"].shape[1:]),
            "base_train_samples": int(len(indices["train"])),
            "fixed_support_query_variants": int(len(train_dataset)),
            "test_sample_seed_cells": len(keys),
        },
        "model": {
            "base": "two frozen independent residual/absolute FNO experts plus analytic support-fit mixture",
            "corrector_input_channels": int(data["inputs"].shape[1]) + 2,
            "corrector_config": experiment_config["model"],
            "correction_cap_ratio": experiment_config["correction_cap_ratio"],
            "free_and_null_correctors_parameter_matched": True,
            "free_and_null_correctors_identical_initialization_within_seed": True,
            "training_records": training_records,
        },
        "summary": summary,
        "paired_findings": paired,
        "key_findings": {
            "null_mean_field_improvement_pct": paired["nullspace_correction"]["mean_field_improvement_pct"],
            "null_field_better_than_base_fraction": paired["nullspace_correction"]["field_better_than_base_fraction"],
            "null_mean_heldout_improvement_pct": paired["nullspace_correction"]["mean_heldout_improvement_pct"],
            "null_mean_oracle_headroom_capture_fraction": paired["nullspace_correction"]["mean_oracle_headroom_capture_fraction"],
            "free_mean_field_improvement_pct": paired["free_correction"]["mean_field_improvement_pct"],
            "free_field_better_than_base_fraction": paired["free_correction"]["field_better_than_base_fraction"],
            "maximum_null_support_correction_leakage": float(
                np.max([float(row["support_correction_leakage"]) for row in null_rows])
            ),
        },
        "decision_rule": {
            "promote_nullspace_model_if": "it improves field and held-out metrics across seeds/splits while preserving support projections and captures nontrivial oracle headroom",
            "retain_as_negative_control_if": "the exact constraint protects support data but prevents learning useful correction or fails family/geometry OOD",
        },
        "claims_boundary": [
            "Both learned correctors use synthetic field truth; query views are additional training supervision, not inference inputs.",
            "The exact nullspace guarantee applies to the declared linear synthetic support matrix and numerical precision only.",
            "The independent base doubles operator capacity and has not yet been matched against a single equally wide model or independent ensemble.",
            "The correction network is an undergraduate-scale FNO on 8x16x16 fields, not evidence of OERF-resolution performance.",
            "Three optimization seeds are a stability screen, not a population confidence interval.",
            "Family, geometry, resolution, nonlinear-ray, and real-data transfer remain separate tests.",
        ],
    }
    report_path = args.output_dir / "support_nullspace_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_checksums(
        args.output_dir,
        [
            "support_nullspace_runs.csv",
            "support_nullspace_samples.csv",
            "support_nullspace_summary.csv",
            "support_nullspace_report.json",
        ],
    )
    print(json.dumps(report["key_findings"], indent=2))
    print(f"results: {args.output_dir}")


if __name__ == "__main__":
    main()
