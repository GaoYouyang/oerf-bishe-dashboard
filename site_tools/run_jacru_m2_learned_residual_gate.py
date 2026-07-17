#!/usr/bin/env python3
"""Train and falsify CGLS-based learned residuals on the JACRU T0 fixture.

This is an opened synthetic train/development/OOD screen.  It deliberately
does not construct, read, or score a fresh/final split.  Continuous analytic
gradients generate observations while a separate voxel FD/trilinear operator
prepares CGLS bases and data-residual lifts.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo_t16_operator.analytic_bost_phantoms import analytic_phantom_grid
from demo_t16_operator.interface_baselines import (
    cgls_baseline,
    edge_preserving_pdhg_baseline,
)
from demo_t16_operator.jacru_m2_learned_residual import (
    JACRUM2Batch,
    JACRUM2LearnedResidual,
    prepare_jacru_m2_batch,
)
from demo_t16_operator.jacru_synthetic_fixture import (
    JACRUSyntheticCase,
    JACRUSyntheticFixtureConfig,
    build_jacru_synthetic_case,
)
from demo_t16_operator.psu_b0_streaming_operator import zero_outer_boundary_support
from demo_t16_operator.spatial_reconstruction_metrics import synthetic_field_metrics


DEFAULT_CONFIG = (
    ROOT
    / "demo_t16_operator"
    / "configs"
    / "jacru_m2_learned_residual_t0_v1.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "demo_t16_operator"
    / "results"
    / "jacru_m2_learned_residual_t0_public"
)
REPORT_SCHEMA = "jacru-m2-learned-residual-t0-report-1.0"
CLASSICAL_METHODS = ("cgls_13", "huber_pdhg_13")


@dataclass(frozen=True)
class PreparedRecord:
    split: str
    family: str
    base_seed: int
    case: JACRUSyntheticCase
    batch: JACRUM2Batch
    preparation_calls: dict[str, int]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed-limit", type=int)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("config must contain one JSON object")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _choose_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return device
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _fixture_config(config: dict[str, Any]) -> JACRUSyntheticFixtureConfig:
    values = config["fixture"]
    return JACRUSyntheticFixtureConfig(
        grid_shape=tuple(int(value) for value in values["grid_shape"]),
        detector_shape=tuple(int(value) for value in values["detector_shape"]),
        samples_per_ray=int(values["samples_per_ray"]),
        noise_relative_std=float(values["noise_relative_std"]),
        camera_bias_relative_std=float(values["camera_bias_relative_std"]),
        enable_noise=bool(values["enable_noise"]),
        enable_camera_bias=bool(values["enable_camera_bias"]),
    )


def _spacing(config: JACRUSyntheticFixtureConfig) -> tuple[float, float, float]:
    return tuple(
        (config.domain_maximum_xyz[index] - config.domain_minimum_xyz[index])
        / (config.grid_shape[2 - index] - 1)
        for index in range(3)
    )


def _prepare_records(
    config: dict[str, Any],
    fixture: JACRUSyntheticFixtureConfig,
) -> list[PreparedRecord]:
    base_iterations = int(config["physical_budget"]["cgls_base_iterations"])
    expected_calls = base_iterations + 1
    support = zero_outer_boundary_support(fixture.grid_shape, dtype=torch.float64)
    records: list[PreparedRecord] = []
    seen_case_ids: set[str] = set()
    for split in ("train", "development", "ood"):
        split_config = config["splits"][split]
        for seed in split_config["base_seeds"]:
            for family in split_config["families"]:
                case = build_jacru_synthetic_case(
                    family=str(family),
                    split=split,
                    base_seed=int(seed),
                    config=fixture,
                )
                if case.inference.case_id in seen_case_ids:
                    raise RuntimeError("case IDs must be disjoint across all T0 splits")
                seen_case_ids.add(case.inference.case_id)
                operator = case.inference.operator
                operator.reset_call_counts()
                batch = prepare_jacru_m2_batch(
                    case.inference,
                    support=support,
                    cgls_iterations=base_iterations,
                    model_dtype=torch.float32,
                    model_device="cpu",
                )
                calls = operator.call_report()
                if calls != {
                    "forward_calls": expected_calls,
                    "adjoint_calls": expected_calls,
                }:
                    raise RuntimeError(
                        f"feature preparation call contract failed for {case.inference.case_id}: {calls}"
                    )
                records.append(
                    PreparedRecord(
                        split=split,
                        family=str(family),
                        base_seed=int(seed),
                        case=case,
                        batch=batch,
                        preparation_calls=calls,
                    )
                )
    return records


def _stack_batches(records: Iterable[PreparedRecord]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    values = list(records)
    if not values:
        raise ValueError("cannot stack an empty record list")
    kwargs = {
        name: torch.cat([getattr(record.batch, name) for record in values], dim=0)
        for name in (
            "base_field",
            "support",
            "adjoint_lifted_data_residual",
            "camera_pose_features",
            "camera_mask",
        )
    }
    truth = torch.cat(
        [record.case.evaluation.truth_volume.to(torch.float32) for record in values],
        dim=0,
    )
    return kwargs, truth


def _to_device(values: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device) for name, tensor in values.items()}


def _relative_field_loss(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    numerator = (prediction - truth).square().flatten(start_dim=1).sum(dim=1)
    denominator = truth.square().flatten(start_dim=1).sum(dim=1).clamp_min(1e-8)
    return torch.mean(numerator / denominator)


def _relative_gradient_loss(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    numerator = prediction.new_zeros(prediction.shape[0])
    denominator = prediction.new_zeros(prediction.shape[0])
    for axis in (2, 3, 4):
        error = torch.diff(prediction - truth, dim=axis).flatten(start_dim=1)
        target = torch.diff(truth, dim=axis).flatten(start_dim=1)
        numerator = numerator + error.square().sum(dim=1)
        denominator = denominator + target.square().sum(dim=1)
    return torch.mean(numerator / denominator.clamp_min(1e-8))


def _relative_l2_tensor(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    error = (prediction - truth).flatten(start_dim=1)
    target = truth.flatten(start_dim=1)
    return torch.linalg.vector_norm(error, dim=1) / torch.linalg.vector_norm(
        target, dim=1
    ).clamp_min(1e-8)


def _augment_camera_set(
    kwargs: dict[str, torch.Tensor],
    *,
    generator: torch.Generator,
    dropout_probability: float,
) -> dict[str, torch.Tensor]:
    augmented = dict(kwargs)
    lifts = kwargs["adjoint_lifted_data_residual"]
    poses = kwargs["camera_pose_features"]
    masks = kwargs["camera_mask"].clone()
    view_count = lifts.shape[1]
    permutation = torch.randperm(view_count, generator=generator)
    augmented["adjoint_lifted_data_residual"] = lifts[:, permutation]
    augmented["camera_pose_features"] = poses[:, permutation]
    masks = masks[:, permutation]
    if dropout_probability > 0.0:
        random_values = torch.rand(masks.shape, generator=generator)
        drop = (random_values < float(dropout_probability)) & (masks > 0.5)
        masks[drop] = 0.0
        empty = masks.sum(dim=1) < 0.5
        if bool(torch.any(empty)):
            masks[empty, 0] = 1.0
    augmented["camera_mask"] = masks
    return augmented


def _build_model(method: str, config: dict[str, Any]) -> torch.nn.Module:
    spec = config["models"][method]
    if method == "jacru_m2":
        return JACRUM2LearnedResidual(
            set_channels=int(spec["set_channels"]),
            hidden_channels=int(spec["hidden_channels"]),
            gate_hidden=int(spec["gate_hidden"]),
            maximum_residual_magnitude=float(spec["maximum_residual_magnitude"]),
        )
    from demo_t16_operator.jacru_m2_comparators import (
        FixedGridDeepONetResidualComparator,
        NeuralOpFNOResidualComparator,
        PooledCNN3DResidualComparator,
    )

    if method == "pooled_cnn":
        return PooledCNN3DResidualComparator(
            hidden_channels=int(spec["hidden_channels"]),
            maximum_residual_magnitude=float(spec["maximum_residual_magnitude"]),
        )
    if method == "grid_deeponet":
        return FixedGridDeepONetResidualComparator(
            grid_shape=tuple(int(value) for value in config["fixture"]["grid_shape"]),
            pool_shape=(3, 3, 3),
            branch_hidden=int(spec["branch_hidden"]),
            trunk_hidden=int(spec["trunk_hidden"]),
            rank=int(spec["rank"]),
            maximum_residual_magnitude=float(spec["maximum_residual_magnitude"]),
        )
    if method == "pooled_fno":
        return NeuralOpFNOResidualComparator(
            hidden_channels=int(spec["hidden_channels"]),
            n_modes=tuple(int(value) for value in spec["n_modes"]),
            n_layers=int(spec["n_layers"]),
            maximum_residual_magnitude=float(spec["maximum_residual_magnitude"]),
        )
    raise ValueError(f"unknown learned method: {method}")


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _train_one(
    *,
    method: str,
    seed: int,
    config: dict[str, Any],
    records: list[PreparedRecord],
    device: torch.device,
    epoch_override: int | None,
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model = _build_model(method, config).to(device)
    training = config["training"]
    epochs = int(training["epochs"] if epoch_override is None else epoch_override)
    if epochs < 1:
        raise ValueError("epochs must be positive")
    train_records = [record for record in records if record.split == "train"]
    dev_records = [record for record in records if record.split == "development"]
    train_kwargs, train_truth = _stack_batches(train_records)
    dev_kwargs, dev_truth = _stack_batches(dev_records)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    generator = torch.Generator().manual_seed(int(seed) * 1009 + 7)
    best_value = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_records), generator=generator)
        totals = {"total": 0.0, "field": 0.0, "h1": 0.0, "correction": 0.0, "gate": 0.0}
        sample_count = 0
        batch_size = int(training["batch_size"])
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            batch_kwargs = {name: value[indices] for name, value in train_kwargs.items()}
            batch_kwargs = _augment_camera_set(
                batch_kwargs,
                generator=generator,
                dropout_probability=float(training["camera_dropout_probability"]),
            )
            target = train_truth[indices]
            batch_kwargs = _to_device(batch_kwargs, device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction, gate = model(**batch_kwargs, return_gate=True)
            field_loss = _relative_field_loss(prediction, target)
            h1_loss = _relative_gradient_loss(prediction, target)
            correction_loss = torch.mean(
                (prediction - batch_kwargs["base_field"]).square()
            )
            gate_loss = torch.mean(gate.square())
            loss = (
                field_loss
                + float(training["lambda_h1"]) * h1_loss
                + float(training["lambda_correction"]) * correction_loss
                + float(training["lambda_gate"]) * gate_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(training["gradient_clip_norm"]),
            )
            optimizer.step()
            count = int(len(indices))
            sample_count += count
            for name, value in (
                ("total", loss),
                ("field", field_loss),
                ("h1", h1_loss),
                ("correction", correction_loss),
                ("gate", gate_loss),
            ):
                totals[name] += float(value.detach().cpu()) * count

        model.eval()
        with torch.no_grad():
            dev_prediction = model(**_to_device(dev_kwargs, device))
            dev_value = float(
                torch.mean(_relative_l2_tensor(dev_prediction, dev_truth.to(device))).cpu()
            )
        scheduler.step()
        row = {
            "method": method,
            "model_seed": int(seed),
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "development_field_relative_l2": dev_value,
            **{
                f"train_{name}": value / max(sample_count, 1)
                for name, value in totals.items()
            },
        }
        history.append(row)
        if dev_value < best_value - 1e-6:
            best_value = dev_value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_state.pop("_metadata", None)
            stale = 0
        else:
            stale += 1
        if (
            epoch >= int(training["minimum_epoch"])
            and stale >= int(training["early_stop_patience"])
        ):
            break

    _synchronize(device)
    elapsed = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("training produced no finite development checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return {
        "model": model,
        "method": method,
        "model_seed": int(seed),
        "parameters": _parameter_count(model),
        "best_epoch": best_epoch,
        "best_development_field_relative_l2": best_value,
        "epochs_ran": len(history),
        "train_seconds": elapsed,
        "history": history,
        "device": str(device),
    }


def _operator_maps(operator):
    def forward(field: torch.Tensor) -> torch.Tensor:
        return operator(field[None, None])[0]

    def adjoint(observation: torch.Tensor) -> torch.Tensor:
        return operator.adjoint(observation[None])[0, 0]

    return forward, adjoint


@torch.no_grad()
def _dense_norm_squared_bound(
    operator,
    *,
    batch_size: int,
    safety_factor: float,
) -> dict[str, Any]:
    voxel_count = int(np.prod(operator.grid_shape))
    rows: list[torch.Tensor] = []
    start_calls = int(operator.forward_calls)
    for start in range(0, voxel_count, int(batch_size)):
        stop = min(start + int(batch_size), voxel_count)
        indices = torch.arange(start, stop, dtype=torch.int64, device=operator.support.device)
        basis = torch.zeros(
            (stop - start, voxel_count),
            dtype=operator.support.dtype,
            device=operator.support.device,
        )
        basis[torch.arange(stop - start, device=basis.device), indices] = 1.0
        rows.append(
            operator.forward(basis.reshape(stop - start, 1, *operator.grid_shape))
            .reshape(stop - start, -1)
            .cpu()
        )
    matrix = torch.cat(rows, dim=0)
    largest = float(torch.linalg.svdvals(matrix).max())
    estimate = largest**2
    return {
        "matrix_shape": [int(matrix.shape[1]), int(matrix.shape[0])],
        "spectral_norm_squared": estimate,
        "bound": float(safety_factor) * estimate,
        "safety_factor": float(safety_factor),
        "setup_forward_calls": int(operator.forward_calls) - start_calls,
        "status": "DENSE_NUMERICAL_SVD_TIMES_SAFETY_FACTOR_NOT_INTERVAL_CERTIFIED",
    }


def _relative_l2(value: torch.Tensor, reference: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(value - reference)
        / torch.linalg.vector_norm(reference).clamp_min(1e-30)
    )


def _score_prediction(
    *,
    record: PreparedRecord,
    method: str,
    model_seed: int,
    prediction: torch.Tensor,
    gate: float | None,
    correction_rms: float | None,
    optimization_forward_calls: int,
    optimization_adjoint_calls: int,
    grouped_adjoint_calls: int,
    neural_inference_seconds: float,
) -> dict[str, Any]:
    case = record.case
    operator = case.inference.operator
    prediction = prediction.to(dtype=torch.float64, device=operator.support.device)
    projected = operator(prediction[None, None])[0]
    evaluation = case.evaluation
    truth_evaluation = analytic_phantom_grid(
        evaluation.phantom_spec,
        grid_shape=tuple(int(value) for value in prediction.shape),
        dtype=torch.float64,
    )
    metrics = synthetic_field_metrics(
        prediction.detach().cpu().numpy(),
        evaluation.truth_volume[0, 0].cpu().numpy(),
        analytic_truth_gradient_xyz=truth_evaluation.gradient_xyz.cpu().numpy(),
        spacing_xyz=operator.spacing_xyz,
    )
    return {
        "case_id": case.inference.case_id,
        "split": record.split,
        "family": record.family,
        "base_seed": record.base_seed,
        "method": method,
        "model_seed": int(model_seed),
        **metrics,
        "measured_reprojection_relative_l2": _relative_l2(
            projected, case.inference.observations_uv[0]
        ),
        "clean_reprojection_relative_l2": _relative_l2(
            projected, evaluation.clean_observations_uv[0]
        ),
        "gate": gate,
        "correction_rms": correction_rms,
        "optimization_forward_calls": int(optimization_forward_calls),
        "optimization_adjoint_calls": int(optimization_adjoint_calls),
        "grouped_adjoint_calls": int(grouped_adjoint_calls),
        "evaluation_forward_calls": 1,
        "neural_inference_seconds": float(neural_inference_seconds),
    }


def _evaluate_classical(
    *,
    records: list[PreparedRecord],
    config: dict[str, Any],
    fixture: JACRUSyntheticFixtureConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    norm_cache: dict[str, dict[str, Any]] = {}
    budget = config["physical_budget"]
    iterations = int(budget["classical_comparator_iterations"])
    spacing_xyz = _spacing(fixture)
    for record in records:
        if record.split == "train":
            continue
        case = record.case
        operator = case.inference.operator
        observation = case.inference.observations_uv[0]
        support = operator.support.detach().clone()
        forward, adjoint = _operator_maps(operator)

        operator.reset_call_counts()
        cgls = cgls_baseline(
            observation,
            forward=forward,
            adjoint=adjoint,
            support=support,
            spacing_xyz=spacing_xyz,
            iterations=iterations,
        )
        if cgls.forward_calls != iterations or cgls.adjoint_calls != iterations:
            raise RuntimeError("CGLS comparison budget drifted")
        rows.append(
            _score_prediction(
                record=record,
                method="cgls_13",
                model_seed=-1,
                prediction=cgls.field,
                gate=None,
                correction_rms=None,
                optimization_forward_calls=iterations,
                optimization_adjoint_calls=iterations,
                grouped_adjoint_calls=0,
                neural_inference_seconds=0.0,
            )
        )

        digest = case.inference.geometry.digest
        if digest not in norm_cache:
            operator.reset_call_counts()
            norm_cache[digest] = _dense_norm_squared_bound(
                operator,
                batch_size=int(budget["dense_norm_batch_size"]),
                safety_factor=float(budget["dense_norm_safety_factor"]),
            )
        operator.reset_call_counts()
        huber = edge_preserving_pdhg_baseline(
            observation,
            forward=forward,
            adjoint=adjoint,
            support=support,
            spacing_xyz=spacing_xyz,
            iterations=iterations,
            regularization_weight=0.001,
            data_norm_squared_bound=float(norm_cache[digest]["bound"]),
            penalty="huber",
            huber_delta=0.08,
            step_safety=0.98,
        )
        if huber.forward_calls != iterations or huber.adjoint_calls != iterations:
            raise RuntimeError("Huber-PDHG comparison budget drifted")
        rows.append(
            _score_prediction(
                record=record,
                method="huber_pdhg_13",
                model_seed=-1,
                prediction=huber.field,
                gate=None,
                correction_rms=None,
                optimization_forward_calls=iterations,
                optimization_adjoint_calls=iterations,
                grouped_adjoint_calls=0,
                neural_inference_seconds=0.0,
            )
        )
    return rows, norm_cache


def _evaluate_learned(
    *,
    trained: list[dict[str, Any]],
    records: list[PreparedRecord],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base_iterations = int(config["physical_budget"]["cgls_base_iterations"])
    for trained_model in trained:
        model = trained_model["model"]
        device = next(model.parameters()).device
        for record in records:
            if record.split == "train":
                continue
            kwargs = _to_device(record.batch.model_kwargs(), device)
            _synchronize(device)
            started = time.perf_counter()
            with torch.no_grad():
                prediction, gate = model(**kwargs, return_gate=True)
            _synchronize(device)
            elapsed = time.perf_counter() - started
            prediction_cpu = prediction[0, 0].detach().cpu()
            correction = prediction_cpu - record.batch.base_field[0, 0]
            rows.append(
                _score_prediction(
                    record=record,
                    method=str(trained_model["method"]),
                    model_seed=int(trained_model["model_seed"]),
                    prediction=prediction_cpu,
                    gate=float(gate[0, 0, 0, 0, 0].detach().cpu()),
                    correction_rms=float(torch.sqrt(torch.mean(correction.square()))),
                    optimization_forward_calls=base_iterations + 1,
                    optimization_adjoint_calls=base_iterations + 1,
                    grouped_adjoint_calls=1,
                    neural_inference_seconds=elapsed,
                )
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["method"]), int(row["model_seed"]), str(row["split"]))
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (method, model_seed, split), values in sorted(grouped.items()):
        output.append(
            {
                "method": method,
                "model_seed": model_seed,
                "split": split,
                "case_count": len(values),
                "field_relative_l2_mean": float(np.mean([row["field_relative_l2"] for row in values])),
                "field_relative_l2_maximum": float(np.max([row["field_relative_l2"] for row in values])),
                "h1_seminorm_relative_error_mean": float(
                    np.mean([row["h1_seminorm_relative_error"] for row in values])
                ),
                "measured_reprojection_relative_l2_mean": float(
                    np.mean([row["measured_reprojection_relative_l2"] for row in values])
                ),
                "gate_mean": (
                    None
                    if values[0]["gate"] is None
                    else float(np.mean([row["gate"] for row in values]))
                ),
                "correction_rms_mean": (
                    None
                    if values[0]["correction_rms"] is None
                    else float(np.mean([row["correction_rms"] for row in values]))
                ),
                "neural_inference_seconds_mean": float(
                    np.mean([row["neural_inference_seconds"] for row in values])
                ),
            }
        )
    return output


def _method_decisions(
    rows: list[dict[str, Any]],
    methods: list[str],
    gates: dict[str, Any],
) -> dict[str, Any]:
    classical = [row for row in rows if row["method"] in CLASSICAL_METHODS]
    learned = [row for row in rows if row["method"] not in CLASSICAL_METHODS]
    classical_lookup = {
        (row["case_id"], row["method"]): row for row in classical
    }
    best_by_split: dict[str, str] = {}
    for split in ("development", "ood"):
        best_by_split[split] = min(
            CLASSICAL_METHODS,
            key=lambda method: float(
                np.mean(
                    [
                        row["field_relative_l2"]
                        for row in classical
                        if row["split"] == split and row["method"] == method
                    ]
                )
            ),
        )
    decisions: dict[str, Any] = {}
    for method in methods:
        method_rows = [row for row in learned if row["method"] == method]
        diagnostics: dict[str, Any] = {}
        checks: dict[str, bool] = {}
        seed_gains: dict[str, list[float]] = {"development": [], "ood": []}
        for split in ("development", "ood"):
            baseline_method = best_by_split[split]
            split_rows = [row for row in method_rows if row["split"] == split]
            paired = []
            for row in split_rows:
                baseline = classical_lookup[(row["case_id"], baseline_method)]
                field_gain = 1.0 - float(row["field_relative_l2"]) / float(
                    baseline["field_relative_l2"]
                )
                h1_gain = 1.0 - float(row["h1_seminorm_relative_error"]) / float(
                    baseline["h1_seminorm_relative_error"]
                )
                cgls = classical_lookup[(row["case_id"], "cgls_13")]
                reprojection_ratio = float(row["measured_reprojection_relative_l2"]) / max(
                    float(cgls["measured_reprojection_relative_l2"]), 1e-30
                )
                paired.append((field_gain, h1_gain, reprojection_ratio))
            field_values = [value[0] for value in paired]
            h1_values = [value[1] for value in paired]
            reprojection_values = [value[2] for value in paired]
            harm_threshold = float(gates["field_harm_threshold_fraction"])
            diagnostics[f"{split}_field_gain_mean"] = float(np.mean(field_values))
            diagnostics[f"{split}_h1_gain_mean"] = float(np.mean(h1_values))
            diagnostics[f"{split}_reprojection_ratio_mean"] = float(
                np.mean(reprojection_values)
            )
            diagnostics[f"{split}_field_harm_rate"] = float(
                np.mean(np.asarray(field_values) < -harm_threshold)
            )
            diagnostics[f"{split}_worst_field_gain"] = float(np.min(field_values))
            for seed in sorted({int(row["model_seed"]) for row in split_rows}):
                seed_values = [
                    field_values[index]
                    for index, row in enumerate(split_rows)
                    if int(row["model_seed"]) == seed
                ]
                seed_gains[split].append(float(np.mean(seed_values)))
        checks = {
            "development_field_gain": diagnostics["development_field_gain_mean"]
            >= float(gates["development_field_gain_over_best_classical_minimum_fraction"]),
            "development_h1_gain": diagnostics["development_h1_gain_mean"]
            >= float(gates["development_h1_gain_over_best_classical_minimum_fraction"]),
            "ood_field_gain": diagnostics["ood_field_gain_mean"]
            >= float(gates["ood_field_gain_over_best_classical_minimum_fraction"]),
            "ood_h1_gain": diagnostics["ood_h1_gain_mean"]
            >= float(gates["ood_h1_gain_over_best_classical_minimum_fraction"]),
            "development_reprojection": diagnostics["development_reprojection_ratio_mean"]
            <= float(gates["development_reprojection_ratio_to_cgls_maximum"]),
            "ood_reprojection": diagnostics["ood_reprojection_ratio_mean"]
            <= float(gates["ood_reprojection_ratio_to_cgls_maximum"]),
            "development_harm": diagnostics["development_field_harm_rate"]
            <= float(gates["field_harm_rate_maximum"]),
            "ood_harm": diagnostics["ood_field_harm_rate"]
            <= float(gates["field_harm_rate_maximum"]),
            "worst_case": min(
                diagnostics["development_worst_field_gain"],
                diagnostics["ood_worst_field_gain"],
            )
            >= float(gates["worst_field_gain_minimum_fraction"]),
            "all_seed_means_positive": all(
                gain > 0.0 for values in seed_gains.values() for gain in values
            ),
        }
        decisions[method] = {
            "passed": all(checks.values()),
            "best_classical_by_split": best_by_split,
            "checks": checks,
            "diagnostics": diagnostics,
            "per_seed_field_gain_means": seed_gains,
        }
    return decisions


def _plot(
    output: Path,
    rows: list[dict[str, Any]],
    aggregate: list[dict[str, Any]],
    methods: list[str],
) -> None:
    labels = {
        "cgls_13": "CGLS-13",
        "huber_pdhg_13": "Huber-PDHG-13",
        "jacru_m2": "JACRU-M2",
        "pooled_cnn": "Pooled CNN",
        "grid_deeponet": "DeepONet",
        "pooled_fno": "FNO",
    }
    plotted = list(CLASSICAL_METHODS) + methods
    colors = ["#315f93", "#39724f", "#a34f43", "#8a651b", "#6f5a92", "#177b7f"]
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)
    x = np.arange(len(plotted))
    width = 0.36
    for metric, axis, title in (
        ("field_relative_l2_mean", axes[0, 0], "Field relative-L2"),
        ("h1_seminorm_relative_error_mean", axes[0, 1], "H1 seminorm relative error"),
    ):
        for offset, split in ((-width / 2, "development"), (width / 2, "ood")):
            values = []
            for method in plotted:
                candidates = [
                    row[metric]
                    for row in aggregate
                    if row["method"] == method and row["split"] == split
                ]
                values.append(float(np.mean(candidates)))
            axis.bar(x + offset, values, width, label=split)
        axis.set_xticks(x, [labels.get(method, method) for method in plotted], rotation=25, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)

    primary = methods[0]
    classical_lookup = {
        (row["case_id"], row["method"]): row
        for row in rows
        if row["method"] in CLASSICAL_METHODS
    }
    candidate_rows = [row for row in rows if row["method"] == primary]
    family_names = ("smooth_no_interface", "single_interface", "two_interface")
    data = []
    positions = []
    tick_labels = []
    position = 1
    for split in ("development", "ood"):
        for family in family_names:
            selected = [
                row for row in candidate_rows if row["split"] == split and row["family"] == family
            ]
            if not selected:
                continue
            gains = []
            for row in selected:
                baseline = min(
                    (classical_lookup[(row["case_id"], method)] for method in CLASSICAL_METHODS),
                    key=lambda value: float(value["field_relative_l2"]),
                )
                gains.append(
                    100.0
                    * (1.0 - float(row["field_relative_l2"]) / float(baseline["field_relative_l2"]))
                )
            data.append(gains)
            positions.append(position)
            tick_labels.append(f"{split[:3]}\n{family.replace('_interface', '').replace('_', ' ')}")
            position += 1
    axes[1, 0].boxplot(data, positions=positions, widths=0.65, showfliers=True)
    axes[1, 0].axhline(0.0, color="#a34f43", linewidth=1)
    axes[1, 0].set_xticks(positions, tick_labels, rotation=20, ha="right")
    axes[1, 0].set_ylabel("Paired field gain over per-case best classical (%)")
    axes[1, 0].set_title(f"{labels.get(primary, primary)} paired gains")
    axes[1, 0].grid(axis="y", alpha=0.25)

    for method, color in zip(methods, colors[2:], strict=False):
        selected = [row for row in rows if row["method"] == method]
        axes[1, 1].scatter(
            [row["measured_reprojection_relative_l2"] for row in selected],
            [row["field_relative_l2"] for row in selected],
            s=20,
            alpha=0.6,
            label=labels.get(method, method),
            color=color,
        )
    axes[1, 1].set_xlabel("Measured reprojection relative-L2")
    axes[1, 1].set_ylabel("Field relative-L2")
    axes[1, 1].set_title("Field / data-consistency Pareto")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(frameon=False)

    figure.suptitle("JACRU M2 T0: opened synthetic development/OOD screen", fontsize=15)
    figure.savefig(output / "diagnostic.png", dpi=180)
    figure.savefig(output / "diagnostic.pdf")
    plt.close(figure)


def _write_checksums(output: Path) -> None:
    paths = sorted(
        path
        for path in output.iterdir()
        if path.is_file() and path.name != "checksums.sha256"
    )
    text = "".join(f"{_sha256(path)}  {path.name}\n" for path in paths)
    (output / "checksums.sha256").write_text(text, encoding="ascii")


def main() -> None:
    args = _parse_args()
    config_path = args.config.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = _read_json(config_path)
    if bool(config["claim_boundary"]["opens_fresh_or_final"]):
        raise ValueError("T0 must not open fresh or final data")
    methods = [str(value) for value in (args.methods or config["methods"])]
    unknown = set(methods).difference(config["methods"])
    if unknown:
        raise ValueError(f"methods are absent from the frozen config: {sorted(unknown)}")
    model_seeds = [int(value) for value in config["training"]["model_seeds"]]
    if args.seed_limit is not None:
        model_seeds = model_seeds[: int(args.seed_limit)]
    if not model_seeds:
        raise ValueError("at least one model seed is required")
    requested_device = str(args.device or config["training"]["device"])
    device = _choose_device(requested_device)
    fixture = _fixture_config(config)
    started = time.perf_counter()
    records = _prepare_records(config, fixture)

    trained: list[dict[str, Any]] = []
    for method in methods:
        for seed in model_seeds:
            trained.append(
                _train_one(
                    method=method,
                    seed=seed,
                    config=config,
                    records=records,
                    device=device,
                    epoch_override=args.epochs,
                )
            )
    classical_rows, norm_cache = _evaluate_classical(
        records=records,
        config=config,
        fixture=fixture,
    )
    learned_rows = _evaluate_learned(trained=trained, records=records, config=config)
    rows = classical_rows + learned_rows
    aggregate = _aggregate(rows)
    decisions = _method_decisions(rows, methods, config["decision_gates"])
    primary = methods[0]
    passed = bool(decisions[primary]["passed"])

    expected_eval_cases = sum(
        len(config["splits"][split]["base_seeds"])
        * len(config["splits"][split]["families"])
        for split in ("development", "ood")
    )
    expected_rows = 2 * expected_eval_cases + len(methods) * len(model_seeds) * expected_eval_cases
    if len(rows) != expected_rows:
        raise RuntimeError(f"metric row count drifted: {len(rows)} != {expected_rows}")
    history_rows = [row for item in trained for row in item["history"]]
    _write_csv(output / "metric_rows.csv", rows)
    _write_csv(output / "aggregate_rows.csv", aggregate)
    _write_csv(output / "training_history.csv", history_rows)
    _plot(output, rows, aggregate, methods)

    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "M2_T0_DEVELOPMENT_OOD_PASS_FOR_LARGER_PREREGISTERED_GATE"
        if passed
        else "M2_T0_NO_GO_OR_REVISE",
        "evidence_level": config["evidence_level"],
        "source_config_sha256": _sha256(config_path),
        "device": str(device),
        "fixture": fixture.manifest(),
        "split_case_counts": {
            split: sum(record.split == split for record in records)
            for split in ("train", "development", "ood")
        },
        "case_manifest": [
            {
                "case_id": record.case.inference.case_id,
                "split": record.split,
                "family": record.family,
                "base_seed": record.base_seed,
                "geometry_digest": record.case.inference.geometry.digest,
                "observation_digest": record.case.inference.observation_digest,
            }
            for record in records
        ],
        "physical_budget": config["physical_budget"],
        "norm_setup": norm_cache,
        "training_runs": [
            {
                key: value
                for key, value in item.items()
                if key not in {"model", "history"}
            }
            for item in trained
        ],
        "metric_row_count": len(rows),
        "aggregate": aggregate,
        "method_decisions": decisions,
        "primary_method": primary,
        "primary_passed": passed,
        "authorization": {
            "continue_to_larger_preregistered_synthetic_gate": passed,
            "claim_neural_operator_superiority": False,
            "claim_interface_detection": False,
            "claim_real_bost_generalization": False,
            "open_fresh_or_final": False,
        },
        "claim_boundary": config["claim_boundary"],
        "elapsed_seconds": time.perf_counter() - started,
        "public_export_policy": {
            "contains_truth_observation_or_geometry_arrays": False,
            "contains_model_checkpoints": False,
            "contains_aggregate_and_per_case_metrics": True,
        },
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        "# JACRU M2 T0 learned-residual screen\n\n"
        "Opened independent-renderer synthetic train/development/OOD evidence only. "
        "The package contains no raw truth, observations, geometry arrays, model checkpoints, "
        "fresh/final cases, experimental reconstruction, or method-superiority authorization.\n",
        encoding="utf-8",
    )
    _write_checksums(output)
    print(json.dumps({
        "status": report["status"],
        "primary_method": primary,
        "primary_passed": passed,
        "metric_rows": len(rows),
        "elapsed_seconds": report["elapsed_seconds"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
