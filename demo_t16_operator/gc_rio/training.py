"""Training and validation utilities for the v5h GC-RIO mechanism gate.

Only noisy target residual labels are allowed for fitting and checkpoint
selection. Truth fields and clean target residuals do not appear in any public
function signature in this module.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .model import (
    GeometryConditionedResidualInverseOperator,
    ResidualInverseOutput,
    adjoint_fisher_statistics,
    physics_decode,
    source_data_consistency_step,
)


PREDICTOR_TENSOR_KEYS = (
    "source_operator",
    "target_operator",
    "source_residual",
    "source_sigma",
    "target_sigma",
    "base_field",
    "analytic_correction",
    "support",
)
FORBIDDEN_FIT_KEYS = frozenset(
    {
        "target_observation",
        "truth_field",
        "clean_target_residual",
        "clean_observation",
        "true_sigma",
    }
)


@dataclass(frozen=True)
class ObjectiveWeights:
    target: float = 1.0
    source_consistency: float = 0.08
    learned_increment: float = 0.002


@dataclass(frozen=True)
class ObjectiveTerms:
    total: torch.Tensor
    target: torch.Tensor
    source_consistency: torch.Tensor
    learned_increment: torch.Tensor


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    best_validation_metric: float
    checkpoint_path: Path
    history: tuple[dict[str, float | int], ...]


def predictor_tensors(
    batch: Mapping[str, Any], device: torch.device | str
) -> dict[str, torch.Tensor]:
    """Convert the predictor-only contract and reject label leakage."""

    leaked = FORBIDDEN_FIT_KEYS & batch.keys()
    if leaked:
        raise ValueError(f"forbidden predictor keys: {sorted(leaked)}")
    missing = set(PREDICTOR_TENSOR_KEYS) - batch.keys()
    if missing:
        raise ValueError(f"missing predictor keys: {sorted(missing)}")
    return {
        key: torch.as_tensor(batch[key], dtype=torch.float32, device=device)
        for key in PREDICTOR_TENSOR_KEYS
    }


def target_supervision(
    rows: Sequence[Mapping[str, Any]], indices: Sequence[int], device: torch.device | str
) -> torch.Tensor:
    """Read only the noisy target residual label required for supervised fit."""

    values = [np.asarray(rows[int(index)]["target_residual_label"]) for index in indices]
    return torch.as_tensor(np.stack(values), dtype=torch.float32, device=device)


def _expanded_target_sigma(
    sigma: torch.Tensor, prediction: torch.Tensor
) -> torch.Tensor:
    values = sigma
    if values.ndim == 1:
        values = values[:, None]
    try:
        return torch.broadcast_to(values, prediction.shape)
    except RuntimeError as error:
        raise ValueError("target_sigma cannot broadcast to target prediction") from error


def training_objective(
    output: ResidualInverseOutput,
    target_residual_label: torch.Tensor,
    target_sigma: torch.Tensor,
    weights: ObjectiveWeights,
) -> ObjectiveTerms:
    """Compute the fit objective without accepting truth or clean signals."""

    if target_residual_label.shape != output.target_residual_prediction.shape:
        raise ValueError("target residual label shape disagrees with prediction")
    sigma = _expanded_target_sigma(target_sigma, output.target_residual_prediction)
    if torch.any(sigma <= 0.0) or not torch.all(torch.isfinite(sigma)):
        raise ValueError("target_sigma must be finite and positive")
    target = torch.mean(
        ((output.target_residual_prediction - target_residual_label) / sigma).square()
    )
    source = torch.mean(output.source_data_consistency_rms.square())
    increment = torch.mean(output.learned_increment.square())
    total = (
        float(weights.target) * target
        + float(weights.source_consistency) * source
        + float(weights.learned_increment) * increment
    )
    return ObjectiveTerms(total, target, source, increment)


def analytic_dc_prediction(
    tensors: Mapping[str, torch.Tensor], *, ridge_lambda: float, step_fraction: float
) -> torch.Tensor:
    """Declared non-learning source-residual data-consistency baseline."""

    _, source_fisher, _ = adjoint_fisher_statistics(
        tensors["source_operator"],
        tensors["target_operator"],
        tensors["source_residual"],
        tensors["source_sigma"],
        tensors["target_sigma"],
    )
    correction = source_data_consistency_step(
        tensors["analytic_correction"],
        tensors["source_operator"],
        tensors["source_residual"],
        tensors["source_sigma"],
        source_fisher,
        tensors["support"],
        ridge_lambda=float(ridge_lambda),
        step_fraction=float(step_fraction),
    )
    return physics_decode(tensors["target_operator"], correction)


def row_whitened_rmse(
    prediction: np.ndarray, label: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    """Per-row target residual RMSE in estimated-noise units."""

    predicted = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(label, dtype=np.float64)
    scale = np.asarray(sigma, dtype=np.float64)
    if predicted.shape != target.shape or predicted.ndim != 2:
        raise ValueError("prediction and label must be matching [row,measurement]")
    if scale.ndim == 1:
        scale = scale[:, None]
    if np.any(scale <= 0.0) or not np.all(np.isfinite(scale)):
        raise ValueError("sigma must be finite and positive")
    return np.sqrt(np.mean(((predicted - target) / scale) ** 2, axis=1))


def cluster_mean_metric(
    row_metric: np.ndarray, rig_ids: Sequence[str], families: Sequence[str]
) -> tuple[float, tuple[dict[str, float | str | int], ...]]:
    """Average rig-family cell means so repeated rows are not independent units."""

    values = np.asarray(row_metric, dtype=np.float64)
    if not (len(values) == len(rig_ids) == len(families)):
        raise ValueError("cluster metadata length mismatch")
    cells: list[dict[str, float | str | int]] = []
    keys = sorted(set(zip(map(str, rig_ids), map(str, families), strict=True)))
    for rig_id, family in keys:
        mask = np.asarray(
            [str(rig) == rig_id and str(item) == family for rig, item in zip(rig_ids, families, strict=True)]
        )
        cells.append(
            {
                "rig_id": rig_id,
                "family": family,
                "row_count": int(mask.sum()),
                "mean_whitened_rmse": float(np.mean(values[mask])),
            }
        )
    return float(np.mean([float(cell["mean_whitened_rmse"]) for cell in cells])), tuple(cells)


def _batches(
    indices: Sequence[int], batch_size: int, rng: np.random.Generator | None
) -> list[np.ndarray]:
    values = np.asarray(indices, dtype=np.int64).copy()
    if rng is not None:
        rng.shuffle(values)
    return [values[start : start + int(batch_size)] for start in range(0, len(values), int(batch_size))]


def predict_rows(
    model: GeometryConditionedResidualInverseOperator,
    bundle: Any,
    indices: Sequence[int],
    *,
    batch_size: int,
    device: torch.device | str,
    conditioning_permutation: Sequence[int] | None = None,
    conditioning_source_permutation: Sequence[int] | None = None,
) -> np.ndarray:
    """Predict without asking the data bundle for truth-bearing scoring batches."""

    model.eval()
    permutation = None
    if conditioning_permutation is not None:
        permutation = np.asarray(conditioning_permutation, dtype=np.int64)
        if permutation.shape != (len(indices),):
            raise ValueError("conditioning permutation length mismatch")
    source_permutation = None
    if conditioning_source_permutation is not None:
        source_permutation = np.asarray(
            conditioning_source_permutation, dtype=np.int64
        )
        if source_permutation.shape != (len(indices),):
            raise ValueError("conditioning source permutation length mismatch")
    if permutation is not None and source_permutation is not None:
        raise ValueError("only one conditioning permutation may be active")
    predictions: list[np.ndarray] = []
    offset = 0
    with torch.no_grad():
        for batch_indices in _batches(indices, batch_size, None):
            raw = bundle.predictor_batch(batch_indices)
            tensors = predictor_tensors(raw, device)
            query = None
            source_query = None
            if permutation is not None:
                count = len(batch_indices)
                query_indices = np.asarray(indices, dtype=np.int64)[
                    permutation[offset : offset + count]
                ]
                query_raw = bundle.predictor_batch(query_indices)
                query = torch.as_tensor(
                    query_raw["target_operator"], dtype=torch.float32, device=device
                )
                offset += count
            if source_permutation is not None:
                count = len(batch_indices)
                query_indices = np.asarray(indices, dtype=np.int64)[
                    source_permutation[offset : offset + count]
                ]
                query_raw = bundle.predictor_batch(query_indices)
                source_query = torch.as_tensor(
                    query_raw["source_operator"], dtype=torch.float32, device=device
                )
                offset += count
            if source_query is None:
                output = model(**tensors, conditioning_target_operator=query)
            else:
                output = model(
                    **tensors, conditioning_source_operator=source_query
                )
            predictions.append(output.target_residual_prediction.cpu().numpy())
    return np.concatenate(predictions, axis=0)


def evaluate_validation(
    model: GeometryConditionedResidualInverseOperator,
    bundle: Any,
    indices: Sequence[int],
    *,
    batch_size: int,
    device: torch.device | str,
) -> tuple[float, tuple[dict[str, float | str | int], ...]]:
    prediction = predict_rows(
        model, bundle, indices, batch_size=batch_size, device=device
    )
    labels = np.stack([bundle.rows[int(i)]["target_residual_label"] for i in indices])
    sigma = np.asarray([bundle.rows[int(i)]["target_sigma"] for i in indices])
    metric = row_whitened_rmse(prediction, labels, sigma)
    return cluster_mean_metric(
        metric,
        [bundle.rows[int(i)]["rig_id"] for i in indices],
        [bundle.rows[int(i)]["family"] for i in indices],
    )


def train_validation_model(
    model: GeometryConditionedResidualInverseOperator,
    bundle: Any,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    *,
    training_config: Mapping[str, Any],
    model_seed: int,
    checkpoint_path: Path,
    device: torch.device | str = "cpu",
) -> TrainingResult:
    """Train with train labels and select a prefix-best validation checkpoint."""

    if not train_indices or not validation_indices:
        raise ValueError("train and validation indices must be nonempty")
    train_splits = {bundle.rows[int(i)]["split"] for i in train_indices}
    validation_splits = {bundle.rows[int(i)]["split"] for i in validation_indices}
    if train_splits != {"train"} or validation_splits != {"validation"}:
        raise ValueError("training can only use train and validation split rows")
    torch.manual_seed(int(model_seed))
    np_rng = np.random.default_rng(int(model_seed))
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    weights = ObjectiveWeights(
        target=float(training_config["target_loss_weight"]),
        source_consistency=float(training_config["source_consistency_weight"]),
        learned_increment=float(training_config["increment_weight"]),
    )
    batch_size = int(training_config["batch_size"])
    best_metric = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    stale_epochs = 0
    for epoch in range(1, int(training_config["epochs"]) + 1):
        model.train()
        totals: list[float] = []
        target_terms: list[float] = []
        for batch_indices in _batches(train_indices, batch_size, np_rng):
            raw = bundle.predictor_batch(batch_indices)
            tensors = predictor_tensors(raw, device)
            label = target_supervision(bundle.rows, batch_indices, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(**tensors)
            terms = training_objective(output, label, tensors["target_sigma"], weights)
            if not torch.isfinite(terms.total):
                raise RuntimeError("nonfinite training objective")
            terms.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            totals.append(float(terms.total.detach().cpu()))
            target_terms.append(float(terms.target.detach().cpu()))
        validation_metric, _ = evaluate_validation(
            model,
            bundle,
            validation_indices,
            batch_size=batch_size,
            device=device,
        )
        history.append(
            {
                "epoch": epoch,
                "train_total": float(np.mean(totals)),
                "train_target": float(np.mean(target_terms)),
                "validation_cluster_whitened_rmse": validation_metric,
            }
        )
        if validation_metric < best_metric - 1e-8:
            best_metric = validation_metric
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= int(training_config["early_stopping_patience"]):
            break
    if best_state is None:
        raise RuntimeError("training produced no finite validation checkpoint")
    model.load_state_dict(copy.deepcopy(best_state), strict=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, checkpoint_path)
    return TrainingResult(
        best_epoch=best_epoch,
        best_validation_metric=best_metric,
        checkpoint_path=checkpoint_path,
        history=tuple(history),
    )
