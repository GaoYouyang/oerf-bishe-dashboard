"""Multi-interface scoring contract for synthetic phase-field BOST results.

The contract accepts zero, one, or two independently declared level-set
volumes.  Surface-bearing predictions are matched one-to-one to truth
interfaces with a global Hungarian assignment.  Unmatched truth and predicted
surfaces receive explicit miss and false-positive penalties; pairwise metrics
are never reduced by selecting only the easiest interface.

Truth level sets are synthetic references.  These metrics must not be reported
for experimental data without volumetric interface ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from demo_t16_operator.spatial_reconstruction_metrics import (
    interface_surface_from_level_set,
    normal_angle_metrics,
    scalar_grid_gradient,
    surface_distance_metrics,
)


MULTI_INTERFACE_METRIC_SCHEMA = "phase-field-bost-multi-interface-metrics-1.0"
_MAX_INTERFACES = 2


@dataclass(frozen=True)
class _ExtractedInterface:
    declared_index: int
    level_set: np.ndarray
    surface: np.ndarray
    surface_points_xyz: np.ndarray
    gradient_xyz: np.ndarray


def _finite_level_sets(value: Any, *, name: str) -> tuple[list[np.ndarray], tuple[int, int, int] | None]:
    if value is None:
        return [], None
    if isinstance(value, (list, tuple)):
        arrays = [np.asarray(item, dtype=np.float64) for item in value]
        if len(arrays) > _MAX_INTERFACES:
            raise ValueError(f"{name} supports at most {_MAX_INTERFACES} interfaces")
        shape: tuple[int, int, int] | None = None
        for index, array in enumerate(arrays):
            if array.ndim != 3 or any(size < 2 for size in array.shape):
                raise ValueError(f"{name}[{index}] must have shape [z,y,x]")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name}[{index}] must contain only finite values")
            if shape is None:
                shape = tuple(int(size) for size in array.shape)
            elif array.shape != shape:
                raise ValueError(f"all {name} volumes must share one [z,y,x] shape")
        return arrays, shape

    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if array.ndim == 3:
        if any(size < 2 for size in array.shape):
            raise ValueError(f"{name} must have shape [z,y,x] or [z,y,x,k]")
        return [array], tuple(int(size) for size in array.shape)
    if array.ndim != 4 or any(size < 2 for size in array.shape[:3]):
        raise ValueError(f"{name} must have shape [z,y,x] or [z,y,x,k]")
    if array.shape[-1] > _MAX_INTERFACES:
        raise ValueError(f"{name} supports at most {_MAX_INTERFACES} interfaces")
    shape = tuple(int(size) for size in array.shape[:3])
    return [array[..., index] for index in range(array.shape[-1])], shape


def _spacing_xyz(value: Any) -> tuple[float, float, float]:
    spacing = np.asarray(value, dtype=np.float64).reshape(-1)
    if spacing.shape != (3,) or np.any(~np.isfinite(spacing)) or np.any(spacing <= 0.0):
        raise ValueError("spacing_xyz must contain three finite positive values")
    return tuple(float(item) for item in spacing)


def _finite_gradients(
    value: Any,
    *,
    name: str,
    level_sets: Sequence[np.ndarray],
    spacing_xyz: tuple[float, float, float],
) -> list[np.ndarray]:
    if value is None:
        return [scalar_grid_gradient(level, spacing_xyz=spacing_xyz) for level in level_sets]
    if isinstance(value, (list, tuple)):
        gradients = [np.asarray(item, dtype=np.float64) for item in value]
    else:
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 4 and array.shape[-1] == 3 and len(level_sets) == 1:
            gradients = [array]
        elif array.ndim == 5 and array.shape[-1] == 3:
            gradients = [array[..., index, :] for index in range(array.shape[-2])]
        else:
            raise ValueError(f"{name} must have shape [z,y,x,3] or [z,y,x,k,3]")
    if len(gradients) != len(level_sets):
        raise ValueError(f"{name} must provide one gradient volume per level set")
    for index, (gradient, level) in enumerate(zip(gradients, level_sets, strict=True)):
        if gradient.shape != (*level.shape, 3):
            raise ValueError(f"{name}[{index}] must have shape [z,y,x,3]")
        if not np.all(np.isfinite(gradient)):
            raise ValueError(f"{name}[{index}] must contain only finite values")
    return gradients


def _extract_interfaces(
    level_sets: Sequence[np.ndarray],
    gradients: Sequence[np.ndarray],
    *,
    spacing_xyz: tuple[float, float, float],
) -> tuple[list[_ExtractedInterface], list[int]]:
    dx, dy, dz = spacing_xyz
    extracted: list[_ExtractedInterface] = []
    degenerate_indices: list[int] = []
    for index, (level, gradient) in enumerate(zip(level_sets, gradients, strict=True)):
        surface = interface_surface_from_level_set(level)
        coordinates_zyx = np.argwhere(surface)
        if coordinates_zyx.size == 0:
            degenerate_indices.append(index)
            continue
        points_xyz = coordinates_zyx[:, ::-1].astype(np.float64)
        points_xyz *= np.array([dx, dy, dz], dtype=np.float64)
        extracted.append(
            _ExtractedInterface(
                declared_index=index,
                level_set=level,
                surface=surface,
                surface_points_xyz=points_xyz,
                gradient_xyz=gradient,
            )
        )
    return extracted, degenerate_indices


def _metric_label(distance: float) -> str:
    return format(float(distance), ".12g").replace("-", "m").replace(".", "p")


def _pair_metrics(
    predicted: _ExtractedInterface,
    truth: _ExtractedInterface,
    *,
    spacing_xyz: tuple[float, float, float],
    domain_diagonal: float,
    minimum_gradient_norm: float,
) -> dict[str, Any]:
    dx = spacing_xyz[0]
    tolerances = (dx, 2.0 * dx)
    distances = surface_distance_metrics(
        predicted.surface,
        truth.surface,
        spacing_xyz=spacing_xyz,
        tolerance_distances=tolerances,
    )
    normal_mask = predicted.surface | truth.surface
    try:
        angles = normal_angle_metrics(
            predicted.gradient_xyz,
            truth.gradient_xyz,
            evaluation_mask=normal_mask,
            minimum_gradient_norm=minimum_gradient_norm,
        )
        normal_available = True
    except ValueError as exc:
        angles = {
            "normal_angle_median_degrees": None,
            "normal_angle_p95_degrees": None,
            "normal_angle_unoriented_median_degrees": None,
            "normal_angle_unoriented_p95_degrees": None,
            "normal_angle_valid_voxels": 0.0,
        }
        normal_available = False
        normal_unavailable_reason = str(exc)
    f1_1dx = float(distances[f"surface_f1_at_{_metric_label(dx)}"])
    f1_2dx = float(distances[f"surface_f1_at_{_metric_label(2.0 * dx)}"])
    normalized_assd = min(float(distances["surface_assd"]) / domain_diagonal, 1.0)
    assignment_cost = 0.5 * normalized_assd + 0.25 * (1.0 - f1_1dx) + 0.25 * (
        1.0 - f1_2dx
    )
    row: dict[str, Any] = {
        "predicted_index": predicted.declared_index,
        "truth_index": truth.declared_index,
        "predicted_surface_point_count": int(predicted.surface_points_xyz.shape[0]),
        "truth_surface_point_count": int(truth.surface_points_xyz.shape[0]),
        "assignment_cost": float(assignment_cost),
        "surface_assd": float(distances["surface_assd"]),
        "surface_hd95": float(distances["surface_hd95"]),
        "surface_f1_at_1dx": f1_1dx,
        "surface_f1_at_2dx": f1_2dx,
        "normal_metrics_available": normal_available,
        "normal_angle_median_degrees": angles["normal_angle_median_degrees"],
        "normal_angle_p95_degrees": angles["normal_angle_p95_degrees"],
        "normal_angle_unoriented_median_degrees": angles[
            "normal_angle_unoriented_median_degrees"
        ],
        "normal_angle_unoriented_p95_degrees": angles[
            "normal_angle_unoriented_p95_degrees"
        ],
        "normal_angle_valid_voxels": int(angles["normal_angle_valid_voxels"]),
    }
    if not normal_available:
        row["normal_metrics_unavailable_reason"] = normal_unavailable_reason
    return row


def _mean(rows: Sequence[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return None if not values else float(np.mean(values))


def multi_interface_level_set_metrics(
    predicted_level_sets: Any,
    truth_level_sets: Any,
    *,
    spacing_xyz: Any,
    predicted_level_set_gradients_xyz: Any | None = None,
    truth_level_set_gradients_xyz: Any | None = None,
    minimum_gradient_norm: float = 1e-12,
) -> dict[str, Any]:
    """Score zero to two phase-field interfaces without best-interface selection.

    Arrays use ``[z,y,x,k]`` for level sets and ``[z,y,x,k,3]`` for optional
    xyz gradients.  A single ``[z,y,x]`` level set is also accepted.  Passing
    ``None``, an empty sequence, or an array with ``k=0`` declares no interface.

    Assignment minimizes a bounded combination of normalized ASSD and
    ``1-F1`` at one and two x-voxel spacings.  Aggregate metrics divide by the
    larger detected cardinality and assign every unmatched surface the worst
    in-domain distance/F1/normal penalty.  A truth declaration with no sign
    transition invalidates aggregate scoring instead of becoming a silent
    zero-interface reference.
    """

    spacing = _spacing_xyz(spacing_xyz)
    if not np.isfinite(minimum_gradient_norm) or minimum_gradient_norm <= 0.0:
        raise ValueError("minimum_gradient_norm must be finite and positive")
    predicted_levels, predicted_shape = _finite_level_sets(
        predicted_level_sets,
        name="predicted_level_sets",
    )
    truth_levels, truth_shape = _finite_level_sets(
        truth_level_sets,
        name="truth_level_sets",
    )
    if predicted_shape is not None and truth_shape is not None and predicted_shape != truth_shape:
        raise ValueError("predicted and truth level sets must share one [z,y,x] shape")
    grid_shape = truth_shape if truth_shape is not None else predicted_shape

    predicted_gradients = _finite_gradients(
        predicted_level_set_gradients_xyz,
        name="predicted_level_set_gradients_xyz",
        level_sets=predicted_levels,
        spacing_xyz=spacing,
    )
    truth_gradients = _finite_gradients(
        truth_level_set_gradients_xyz,
        name="truth_level_set_gradients_xyz",
        level_sets=truth_levels,
        spacing_xyz=spacing,
    )
    predicted, degenerate_predicted = _extract_interfaces(
        predicted_levels,
        predicted_gradients,
        spacing_xyz=spacing,
    )
    truth, degenerate_truth = _extract_interfaces(
        truth_levels,
        truth_gradients,
        spacing_xyz=spacing,
    )

    if grid_shape is None:
        domain_diagonal = 0.0
    else:
        nz, ny, nx = grid_shape
        dx, dy, dz = spacing
        domain_diagonal = float(
            np.linalg.norm(((nx - 1) * dx, (ny - 1) * dy, (nz - 1) * dz))
        )
    safe_diagonal = max(domain_diagonal, np.finfo(np.float64).tiny)

    pair_table: list[list[dict[str, Any]]] = []
    if predicted and truth:
        for predicted_interface in predicted:
            pair_table.append(
                [
                    _pair_metrics(
                        predicted_interface,
                        truth_interface,
                        spacing_xyz=spacing,
                        domain_diagonal=safe_diagonal,
                        minimum_gradient_norm=float(minimum_gradient_norm),
                    )
                    for truth_interface in truth
                ]
            )
        cost = np.array(
            [[row["assignment_cost"] for row in pair_rows] for pair_rows in pair_table],
            dtype=np.float64,
        )
        predicted_assignment, truth_assignment = linear_sum_assignment(cost)
        matches = [
            pair_table[int(predicted_index)][int(truth_index)]
            for predicted_index, truth_index in zip(
                predicted_assignment,
                truth_assignment,
                strict=True,
            )
        ]
        matches.sort(key=lambda row: (row["truth_index"], row["predicted_index"]))
    else:
        matches = []

    matched_truth = {int(row["truth_index"]) for row in matches}
    matched_predicted = {int(row["predicted_index"]) for row in matches}
    unmatched_truth = [item.declared_index for item in truth if item.declared_index not in matched_truth]
    unmatched_predicted = [
        item.declared_index for item in predicted if item.declared_index not in matched_predicted
    ]
    missed_count = len(unmatched_truth)
    false_positive_count = len(unmatched_predicted)
    denominator = max(len(truth), len(predicted), 1)
    score_valid = not degenerate_truth

    matched_assd = _mean(matches, "surface_assd")
    matched_hd95 = _mean(matches, "surface_hd95")
    matched_f1_1dx = _mean(matches, "surface_f1_at_1dx")
    matched_f1_2dx = _mean(matches, "surface_f1_at_2dx")
    matched_normal_median = _mean(matches, "normal_angle_median_degrees")
    matched_normal_p95 = _mean(matches, "normal_angle_p95_degrees")
    matched_unoriented_median = _mean(matches, "normal_angle_unoriented_median_degrees")
    matched_unoriented_p95 = _mean(matches, "normal_angle_unoriented_p95_degrees")
    unavailable_normals = sum(not row["normal_metrics_available"] for row in matches)
    unmatched_count = missed_count + false_positive_count

    if not score_valid:
        aggregate = {
            "penalized_surface_assd": None,
            "penalized_surface_hd95": None,
            "penalized_surface_f1_at_1dx": None,
            "penalized_surface_f1_at_2dx": None,
            "penalized_normal_angle_median_degrees": None,
            "penalized_normal_angle_p95_degrees": None,
            "penalized_normal_angle_unoriented_median_degrees": None,
            "penalized_normal_angle_unoriented_p95_degrees": None,
        }
    else:
        clean_negative = not truth and not predicted
        aggregate = {
            "penalized_surface_assd": float(
                (sum(float(row["surface_assd"]) for row in matches) + unmatched_count * domain_diagonal)
                / denominator
            ),
            "penalized_surface_hd95": float(
                (sum(float(row["surface_hd95"]) for row in matches) + unmatched_count * domain_diagonal)
                / denominator
            ),
            "penalized_surface_f1_at_1dx": float(
                1.0
                if clean_negative
                else sum(float(row["surface_f1_at_1dx"]) for row in matches) / denominator
            ),
            "penalized_surface_f1_at_2dx": float(
                1.0
                if clean_negative
                else sum(float(row["surface_f1_at_2dx"]) for row in matches) / denominator
            ),
            "penalized_normal_angle_median_degrees": float(
                (
                    sum(
                        180.0
                        if row["normal_angle_median_degrees"] is None
                        else float(row["normal_angle_median_degrees"])
                        for row in matches
                    )
                    + unmatched_count * 180.0
                )
                / denominator
            ),
            "penalized_normal_angle_p95_degrees": float(
                (
                    sum(
                        180.0
                        if row["normal_angle_p95_degrees"] is None
                        else float(row["normal_angle_p95_degrees"])
                        for row in matches
                    )
                    + unmatched_count * 180.0
                )
                / denominator
            ),
            "penalized_normal_angle_unoriented_median_degrees": float(
                (
                    sum(
                        90.0
                        if row["normal_angle_unoriented_median_degrees"] is None
                        else float(row["normal_angle_unoriented_median_degrees"])
                        for row in matches
                    )
                    + unmatched_count * 90.0
                )
                / denominator
            ),
            "penalized_normal_angle_unoriented_p95_degrees": float(
                (
                    sum(
                        90.0
                        if row["normal_angle_unoriented_p95_degrees"] is None
                        else float(row["normal_angle_unoriented_p95_degrees"])
                        for row in matches
                    )
                    + unmatched_count * 90.0
                )
                / denominator
            ),
        }

    if not truth and not predicted:
        detection_precision = detection_recall = detection_f1 = 1.0
    else:
        detection_precision = len(matches) / max(len(predicted), 1)
        detection_recall = len(matches) / max(len(truth), 1)
        total = detection_precision + detection_recall
        detection_f1 = 0.0 if total == 0.0 else 2.0 * detection_precision * detection_recall / total

    result: dict[str, Any] = {
        "schema": MULTI_INTERFACE_METRIC_SCHEMA,
        "score_valid": score_valid,
        "status": "VALID" if score_valid else "INVALID_DEGENERATE_TRUTH_SURFACE",
        "grid_shape_zyx": None if grid_shape is None else list(grid_shape),
        "spacing_xyz": list(spacing),
        "voxel_tolerance_1dx": spacing[0],
        "voxel_tolerance_2dx": 2.0 * spacing[0],
        "domain_diagonal": domain_diagonal,
        "truth_declared_count": len(truth_levels),
        "predicted_declared_count": len(predicted_levels),
        "truth_surface_count": len(truth),
        "predicted_surface_count": len(predicted),
        "degenerate_truth_count": len(degenerate_truth),
        "degenerate_predicted_count": len(degenerate_predicted),
        "degenerate_truth_indices": degenerate_truth,
        "degenerate_predicted_indices": degenerate_predicted,
        "matched_count": len(matches),
        "missed_truth_count": missed_count,
        "false_positive_count": false_positive_count,
        "unmatched_truth_indices": unmatched_truth,
        "unmatched_predicted_indices": unmatched_predicted,
        "cardinality_error_count": unmatched_count,
        "cardinality_penalty": float(unmatched_count / denominator),
        "interface_detection_precision": float(detection_precision),
        "interface_detection_recall": float(detection_recall),
        "interface_detection_f1": float(detection_f1),
        "normal_unavailable_match_count": unavailable_normals,
        "matched_surface_assd_mean": matched_assd,
        "matched_surface_hd95_mean": matched_hd95,
        "matched_surface_f1_at_1dx_mean": matched_f1_1dx,
        "matched_surface_f1_at_2dx_mean": matched_f1_2dx,
        "matched_normal_angle_median_mean_degrees": matched_normal_median,
        "matched_normal_angle_p95_mean_degrees": matched_normal_p95,
        "matched_normal_angle_unoriented_median_mean_degrees": matched_unoriented_median,
        "matched_normal_angle_unoriented_p95_mean_degrees": matched_unoriented_p95,
        "assignment_cost_total": float(sum(float(row["assignment_cost"]) for row in matches)),
        "matches": matches,
        **aggregate,
    }
    return result


__all__ = ["MULTI_INTERFACE_METRIC_SCHEMA", "multi_interface_level_set_metrics"]
