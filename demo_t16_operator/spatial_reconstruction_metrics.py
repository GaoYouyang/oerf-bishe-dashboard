"""Spatial metrics for synthetic 3D BOST reconstruction benchmarks.

Truth-field and interface metrics are valid only when a synthetic reference is
available.  They must not be computed for experimental PSU or OERF fields that
have no volumetric ground truth.  Distances are reported in the same physical
units as ``spacing_xyz``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt


SPATIAL_METRIC_SCHEMA = "spatial-bost-reconstruction-metrics-1.0"


def _finite_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _spacing_xyz(value: Any) -> tuple[float, float, float]:
    spacing = np.asarray(value, dtype=np.float64).reshape(-1)
    if spacing.shape != (3,) or np.any(~np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("spacing_xyz must contain three finite positive values")
    return tuple(float(item) for item in spacing)


def scalar_grid_gradient(field: Any, *, spacing_xyz: Any) -> np.ndarray:
    """Return a second-order numerical gradient with final component order xyz."""

    values = _finite_array(field, name="field")
    if values.ndim != 3 or any(size < 3 for size in values.shape):
        raise ValueError("field must have shape [z,y,x] with dimensions at least three")
    dx, dy, dz = _spacing_xyz(spacing_xyz)
    gradient_z, gradient_y, gradient_x = np.gradient(
        values,
        dz,
        dy,
        dx,
        edge_order=2,
    )
    return np.stack((gradient_x, gradient_y, gradient_z), axis=-1)


def synthetic_field_metrics(
    prediction: Any,
    truth: Any,
    *,
    analytic_truth_gradient_xyz: Any,
    spacing_xyz: Any,
) -> dict[str, float]:
    """Field and H1 diagnostics against an analytic synthetic reference."""

    predicted = _finite_array(prediction, name="prediction")
    target = _finite_array(truth, name="truth")
    truth_gradient = _finite_array(
        analytic_truth_gradient_xyz,
        name="analytic_truth_gradient_xyz",
    )
    if predicted.shape != target.shape or predicted.ndim != 3:
        raise ValueError("prediction and truth must have the same [z,y,x] shape")
    if truth_gradient.shape != (*target.shape, 3):
        raise ValueError("analytic truth gradient must have shape [z,y,x,3]")
    predicted_gradient = scalar_grid_gradient(predicted, spacing_xyz=spacing_xyz)
    difference = predicted - target
    gradient_difference = predicted_gradient - truth_gradient
    truth_norm = max(float(np.linalg.norm(target)), np.finfo(np.float64).tiny)
    gradient_norm = max(
        float(np.linalg.norm(truth_gradient)),
        np.finfo(np.float64).tiny,
    )
    dynamic_range = max(
        float(np.max(target) - np.min(target)),
        np.finfo(np.float64).tiny,
    )
    rmse = float(np.sqrt(np.mean(np.square(difference))))
    return {
        "field_relative_l2": float(np.linalg.norm(difference) / truth_norm),
        "field_rmse": rmse,
        "field_nrmse_dynamic_range": rmse / dynamic_range,
        "field_mean_bias": float(np.mean(difference)),
        "h1_seminorm_relative_error": float(
            np.linalg.norm(gradient_difference) / gradient_norm
        ),
    }


def interface_surface_from_level_set(level_set: Any) -> np.ndarray:
    """Mark six-connected sign transitions without treating the box wall as a front."""

    level = _finite_array(level_set, name="level_set")
    if level.ndim != 3 or any(size < 2 for size in level.shape):
        raise ValueError("level_set must have shape [z,y,x]")
    inside = level >= 0.0
    surface = np.zeros_like(inside, dtype=np.bool_)
    for axis in range(3):
        low = [slice(None)] * 3
        high = [slice(None)] * 3
        low[axis] = slice(0, -1)
        high[axis] = slice(1, None)
        low_tuple = tuple(low)
        high_tuple = tuple(high)
        transition = inside[low_tuple] != inside[high_tuple]
        surface[low_tuple] |= transition
        surface[high_tuple] |= transition
    return surface


def surface_distance_metrics(
    predicted_surface: Any,
    truth_surface: Any,
    *,
    spacing_xyz: Any,
    tolerance_distances: tuple[float, ...],
) -> dict[str, float]:
    """Compute symmetric distances and bidirectional tolerance surface-F1."""

    predicted = np.asarray(predicted_surface, dtype=np.bool_)
    truth = np.asarray(truth_surface, dtype=np.bool_)
    if predicted.shape != truth.shape or predicted.ndim != 3:
        raise ValueError("surface masks must share one [z,y,x] shape")
    if not np.any(predicted) or not np.any(truth):
        raise ValueError("both surface masks must contain at least one voxel")
    dx, dy, dz = _spacing_xyz(spacing_xyz)
    tolerances = tuple(float(value) for value in tolerance_distances)
    if not tolerances or any(not np.isfinite(value) or value < 0.0 for value in tolerances):
        raise ValueError("tolerance distances must be finite and nonnegative")
    if len(set(tolerances)) != len(tolerances):
        raise ValueError("tolerance distances must be unique")

    distance_to_truth = distance_transform_edt(~truth, sampling=(dz, dy, dx))
    distance_to_prediction = distance_transform_edt(~predicted, sampling=(dz, dy, dx))
    predicted_to_truth = distance_to_truth[predicted]
    truth_to_predicted = distance_to_prediction[truth]
    pooled = np.concatenate((predicted_to_truth, truth_to_predicted))
    metrics = {
        "surface_assd": float(
            0.5 * (np.mean(predicted_to_truth) + np.mean(truth_to_predicted))
        ),
        "surface_hd95": float(np.quantile(pooled, 0.95, method="higher")),
        "surface_predicted_count": float(np.count_nonzero(predicted)),
        "surface_truth_count": float(np.count_nonzero(truth)),
    }
    for tolerance in tolerances:
        precision = float(np.mean(predicted_to_truth <= tolerance))
        recall = float(np.mean(truth_to_predicted <= tolerance))
        denominator = precision + recall
        f1 = 0.0 if denominator == 0.0 else 2.0 * precision * recall / denominator
        label = format(tolerance, ".12g").replace("-", "m").replace(".", "p")
        metrics[f"surface_precision_at_{label}"] = precision
        metrics[f"surface_recall_at_{label}"] = recall
        metrics[f"surface_f1_at_{label}"] = f1
    return metrics


def level_set_surface_metrics(
    predicted_level_set: Any,
    truth_level_set: Any,
    *,
    spacing_xyz: Any,
    tolerance_distances: tuple[float, ...],
) -> dict[str, float]:
    return surface_distance_metrics(
        interface_surface_from_level_set(predicted_level_set),
        interface_surface_from_level_set(truth_level_set),
        spacing_xyz=spacing_xyz,
        tolerance_distances=tolerance_distances,
    )


def normal_angle_metrics(
    predicted_gradient_xyz: Any,
    truth_gradient_xyz: Any,
    *,
    evaluation_mask: Any,
    minimum_gradient_norm: float = 1e-12,
) -> dict[str, float]:
    """Report signed and orientation-invariant normal-angle errors in degrees."""

    predicted = _finite_array(predicted_gradient_xyz, name="predicted_gradient_xyz")
    truth = _finite_array(truth_gradient_xyz, name="truth_gradient_xyz")
    mask = np.asarray(evaluation_mask, dtype=np.bool_)
    if predicted.shape != truth.shape or predicted.ndim != 4 or predicted.shape[-1] != 3:
        raise ValueError("gradient arrays must share shape [z,y,x,3]")
    if mask.shape != predicted.shape[:-1]:
        raise ValueError("evaluation_mask must match the gradient grid")
    if not np.isfinite(minimum_gradient_norm) or minimum_gradient_norm <= 0.0:
        raise ValueError("minimum_gradient_norm must be finite and positive")
    predicted_norm = np.linalg.norm(predicted, axis=-1)
    truth_norm = np.linalg.norm(truth, axis=-1)
    active = mask & (predicted_norm >= minimum_gradient_norm) & (
        truth_norm >= minimum_gradient_norm
    )
    if not np.any(active):
        raise ValueError("no valid gradient normals remain in the evaluation mask")
    dot = np.sum(predicted[active] * truth[active], axis=-1)
    cosine = dot / (predicted_norm[active] * truth_norm[active])
    cosine = np.clip(cosine, -1.0, 1.0)
    signed_angle = np.degrees(np.arccos(cosine))
    unoriented_angle = np.degrees(np.arccos(np.abs(cosine)))
    return {
        "normal_angle_median_degrees": float(np.median(signed_angle)),
        "normal_angle_p95_degrees": float(
            np.quantile(signed_angle, 0.95, method="higher")
        ),
        "normal_angle_unoriented_median_degrees": float(
            np.median(unoriented_angle)
        ),
        "normal_angle_unoriented_p95_degrees": float(
            np.quantile(unoriented_angle, 0.95, method="higher")
        ),
        "normal_angle_valid_voxels": float(np.count_nonzero(active)),
    }
