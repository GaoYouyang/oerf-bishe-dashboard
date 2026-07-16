#!/usr/bin/env python3
"""Post-open camera-local kernel operator-correction screening."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from demo_t16_operator.gc_rio.protocol import sha256_json
    from demo_t16_operator.independent_reaction_bost import make_reaction_field
    from demo_t16_operator.run_v5s_dco_low_rank_screening import (
        _build_operator_pair,
        _gradient_cosine,
        fit_ridge,
        build_rig_manifest,
    )
    from demo_t16_operator.run_v5t_camera_local_tangent_diagnosis import (
        _renderer_parameters,
        truth_parameter_vector,
    )
    from demo_t16_operator.run_v5u_calibrated_renderer_residual_screening import (
        _feature_matrix as _rig_feature_matrix,
        _low_fidelity_operator_at_calibrated_geometry,
    )
else:
    from .gc_rio.protocol import sha256_json
    from .independent_reaction_bost import make_reaction_field
    from .run_v5s_dco_low_rank_screening import (
        _build_operator_pair,
        _gradient_cosine,
        fit_ridge,
        build_rig_manifest,
    )
    from .run_v5t_camera_local_tangent_diagnosis import (
        _renderer_parameters,
        truth_parameter_vector,
    )
    from .run_v5u_calibrated_renderer_residual_screening import (
        _feature_matrix as _rig_feature_matrix,
        _low_fidelity_operator_at_calibrated_geometry,
    )


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "v5v_camera_local_kernel_correction.json"
OUTPUT_DIR = ROOT / "results" / "v5v_camera_local_kernel_correction"
V5S_REPORT = ROOT / "results" / "v5s_dco_low_rank_screening" / "report.json"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_checksums(output: Path, names: Sequence[str]) -> None:
    lines = [f"{_file_sha256(output / name)}  {name}" for name in names]
    (output / "checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="ascii"
    )


def kernel_offsets(radius: int) -> list[tuple[int, int]]:
    value = int(radius)
    if value < 0:
        raise ValueError("kernel radius must be non-negative")
    return [
        (depth_offset, detector_offset)
        for depth_offset in range(-value, value + 1)
        for detector_offset in range(-value, value + 1)
    ]


def identity_kernel(offsets: Sequence[tuple[int, int]]) -> np.ndarray:
    values = np.zeros(len(offsets), dtype=np.float64)
    values[list(offsets).index((0, 0))] = 1.0
    return values


def _shift_to_output(
    values: np.ndarray, depth_offset: int, detector_offset: int
) -> np.ndarray:
    """Map source neighbors to each output cell with zero boundary padding."""

    source = np.asarray(values, dtype=np.float64)
    if source.ndim < 2:
        raise ValueError("measurement grid must have depth and detector axes")
    output = np.zeros_like(source)
    depth = source.shape[0]
    detector = source.shape[1]
    output_depth_start = max(0, -depth_offset)
    output_depth_stop = min(depth, depth - depth_offset)
    output_detector_start = max(0, -detector_offset)
    output_detector_stop = min(detector, detector - detector_offset)
    if (
        output_depth_start >= output_depth_stop
        or output_detector_start >= output_detector_stop
    ):
        return output
    source_depth = slice(
        output_depth_start + depth_offset, output_depth_stop + depth_offset
    )
    source_detector = slice(
        output_detector_start + detector_offset,
        output_detector_stop + detector_offset,
    )
    output[
        output_depth_start:output_depth_stop,
        output_detector_start:output_detector_stop,
        ...,
    ] = source[source_depth, source_detector, ...]
    return output


def apply_camera_kernels(
    values: np.ndarray,
    kernels: np.ndarray,
    offsets: Sequence[tuple[int, int]],
) -> np.ndarray:
    """Apply one local measurement kernel per camera/view."""

    source = np.asarray(values, dtype=np.float64)
    coefficients = np.asarray(kernels, dtype=np.float64)
    if source.ndim < 3 or coefficients.shape != (source.shape[0], len(offsets)):
        raise ValueError("camera kernel dimensions do not match the measurement grid")
    output = np.zeros_like(source)
    for view_index in range(source.shape[0]):
        for coefficient, (depth_offset, detector_offset) in zip(
            coefficients[view_index], offsets, strict=True
        ):
            output[view_index] += coefficient * _shift_to_output(
                source[view_index], depth_offset, detector_offset
            )
    return output


def apply_camera_kernels_transpose(
    values: np.ndarray,
    kernels: np.ndarray,
    offsets: Sequence[tuple[int, int]],
) -> np.ndarray:
    """Apply the exact transpose of :func:`apply_camera_kernels`."""

    source = np.asarray(values, dtype=np.float64)
    coefficients = np.asarray(kernels, dtype=np.float64)
    if source.ndim < 3 or coefficients.shape != (source.shape[0], len(offsets)):
        raise ValueError("camera kernel dimensions do not match the measurement grid")
    output = np.zeros_like(source)
    for view_index in range(source.shape[0]):
        for coefficient, (depth_offset, detector_offset) in zip(
            coefficients[view_index], offsets, strict=True
        ):
            shifted = _shift_to_output(
                source[view_index], -depth_offset, -detector_offset
            )
            output[view_index] += coefficient * shifted
    return output


def fit_camera_kernels(
    low_operator: np.ndarray,
    high_operator: np.ndarray,
    views: int,
    depth: int,
    detector: int,
    offsets: Sequence[tuple[int, int]],
    relative_ridge: float,
) -> np.ndarray:
    low = np.asarray(low_operator, dtype=np.float64).reshape(
        views, depth, detector, -1
    )
    high = np.asarray(high_operator, dtype=np.float64).reshape(
        views, depth, detector, -1
    )
    prior = identity_kernel(offsets)
    kernels = []
    for view_index in range(views):
        design = np.column_stack(
            [
                _shift_to_output(low[view_index], *offset).reshape(-1)
                for offset in offsets
            ]
        )
        target = high[view_index].reshape(-1)
        gram = design.T @ design
        penalty = float(relative_ridge) * max(
            float(np.trace(gram)) / len(gram), 1e-15
        )
        coefficients = np.linalg.solve(
            gram + penalty * np.eye(len(gram)),
            design.T @ target + penalty * prior,
        )
        kernels.append(coefficients)
    return np.stack(kernels)


def apply_kernels_to_operator(
    operator: np.ndarray,
    kernels: np.ndarray,
    views: int,
    depth: int,
    detector: int,
    offsets: Sequence[tuple[int, int]],
) -> np.ndarray:
    values = np.asarray(operator, dtype=np.float64).reshape(
        views, depth, detector, -1
    )
    return apply_camera_kernels(values, kernels, offsets).reshape(
        views * depth * detector, -1
    )


def view_geometry_features(vector: np.ndarray, view_index: int, views: int) -> np.ndarray:
    parameters = _renderer_parameters(np.asarray(vector, dtype=np.float64), views)
    angle = float(parameters["angles"][view_index])
    return np.asarray(
        [
            np.sin(np.deg2rad(angle)),
            np.cos(np.deg2rad(angle)),
            float(parameters["aperture_radius"]),
            float(parameters["cone_u"]),
            float(parameters["cone_z"]),
            float(parameters["bend"]),
            float(view_index) / max(views - 1, 1),
        ],
        dtype=np.float64,
    )


def view_feature_matrix(
    vectors: Sequence[np.ndarray],
    views: int,
    *,
    center: np.ndarray | None = None,
    scale: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.stack(
        [
            view_geometry_features(vector, view_index, views)
            for vector in vectors
            for view_index in range(views)
        ]
    )
    if center is None:
        center = np.mean(raw, axis=0)
    if scale is None:
        scale = np.std(raw, axis=0)
    scale = np.where(scale > 1e-10, scale, 1.0)
    standardized = (raw - center) / scale
    design = np.concatenate(
        [np.ones((len(raw), 1)), standardized, np.square(standardized)], axis=1
    )
    return design, center, scale


def _relative_residual_errors(
    prediction: np.ndarray, truth: np.ndarray, low: np.ndarray
) -> np.ndarray:
    return np.linalg.norm(prediction - truth, axis=(1, 2)) / np.maximum(
        np.linalg.norm(truth - low, axis=(1, 2)), 1e-15
    )


def run() -> dict[str, Any]:
    diagnosis_config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    source_config_path = ROOT / str(diagnosis_config["source_config"])
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    if not bool(diagnosis_config["design_lock_construction_forbidden"]):
        raise ValueError("v5v must forbid design-lock construction")
    v5s_report = json.loads(V5S_REPORT.read_text(encoding="utf-8"))
    manifest = build_rig_manifest(source_config)
    if sha256_json(manifest) != v5s_report["manifest_sha256_before_operator_build"]:
        raise RuntimeError("v5s manifest reproduction failed")
    views = int(source_config["views"])
    depth = int(source_config["depth"])
    detector = int(source_config["grid_size"])

    low_all = []
    truth_all = []
    calibrated_vectors = []
    for row in manifest:
        _, truth, timing = _build_operator_pair(row, source_config)
        calibrated = truth_parameter_vector(row)
        low = _low_fidelity_operator_at_calibrated_geometry(
            calibrated, source_config, timing["normalization_scale"]
        )
        low_all.append(low)
        truth_all.append(truth)
        calibrated_vectors.append(calibrated)
    low_all = np.stack(low_all)
    truth_all = np.stack(truth_all)
    train_mask = np.asarray([row["split"] == "train" for row in manifest])
    development_mask = ~train_mask
    train_low = low_all[train_mask]
    train_truth = truth_all[train_mask]
    development_low = low_all[development_mask]
    development_truth = truth_all[development_mask]
    train_vectors = [
        vector for vector, selected in zip(calibrated_vectors, train_mask, strict=True) if selected
    ]
    development_vectors = [
        vector
        for vector, selected in zip(calibrated_vectors, development_mask, strict=True)
        if selected
    ]
    development_rows = [row for row in manifest if row["split"] == "development"]
    train_view_design, view_center, view_scale = view_feature_matrix(
        train_vectors, views
    )
    development_view_design, _, _ = view_feature_matrix(
        development_vectors, views, center=view_center, scale=view_scale
    )

    sweep_rows = []
    best: dict[str, Any] | None = None
    kernel_cache: dict[tuple[int, float], tuple[np.ndarray, np.ndarray]] = {}
    for radius in diagnosis_config["kernel_radii"]:
        offsets = kernel_offsets(int(radius))
        prior = identity_kernel(offsets)
        for kernel_ridge in diagnosis_config["kernel_fit_ridge_grid"]:
            train_kernels = np.stack(
                [
                    fit_camera_kernels(
                        low,
                        truth,
                        views,
                        depth,
                        detector,
                        offsets,
                        float(kernel_ridge),
                    )
                    for low, truth in zip(train_low, train_truth, strict=True)
                ]
            )
            development_oracle = np.stack(
                [
                    fit_camera_kernels(
                        low,
                        truth,
                        views,
                        depth,
                        detector,
                        offsets,
                        float(kernel_ridge),
                    )
                    for low, truth in zip(
                        development_low, development_truth, strict=True
                    )
                ]
            )
            kernel_cache[(int(radius), float(kernel_ridge))] = (
                train_kernels,
                development_oracle,
            )
            flattened_targets = (train_kernels - prior[None, None]).reshape(
                len(train_kernels) * views, -1
            )
            for geometry_ridge in diagnosis_config["geometry_ridge_grid"]:
                coefficients = fit_ridge(
                    train_view_design, flattened_targets, float(geometry_ridge)
                )
                predicted = (
                    development_view_design @ coefficients
                ).reshape(len(development_rows), views, -1) + prior[None, None]
                predicted_operators = np.stack(
                    [
                        apply_kernels_to_operator(
                            low, kernels, views, depth, detector, offsets
                        )
                        for low, kernels in zip(
                            development_low, predicted, strict=True
                        )
                    ]
                )
                errors = _relative_residual_errors(
                    predicted_operators, development_truth, development_low
                )
                record = {
                    "kernel_radius": int(radius),
                    "kernel_coefficients_per_view": len(offsets),
                    "kernel_fit_ridge": float(kernel_ridge),
                    "geometry_ridge": float(geometry_ridge),
                    "development_mean_relative_residual_error": float(
                        np.mean(errors)
                    ),
                    "development_worst_relative_residual_error": float(
                        np.max(errors)
                    ),
                }
                sweep_rows.append(record)
                if best is None or (
                    record["development_mean_relative_residual_error"],
                    len(offsets),
                ) < (
                    best["development_mean_relative_residual_error"],
                    best["kernel_coefficients_per_view"],
                ):
                    best = {
                        **record,
                        "coefficients": coefficients,
                        "predicted_kernels": predicted,
                    }
    assert best is not None
    selected_radius = int(best["kernel_radius"])
    selected_kernel_ridge = float(best["kernel_fit_ridge"])
    offsets = kernel_offsets(selected_radius)
    prior = identity_kernel(offsets)
    train_kernels, development_oracle = kernel_cache[
        (selected_radius, selected_kernel_ridge)
    ]
    predicted_kernels = best["predicted_kernels"]
    predicted_operators = np.stack(
        [
            apply_kernels_to_operator(low, kernels, views, depth, detector, offsets)
            for low, kernels in zip(
                development_low, predicted_kernels, strict=True
            )
        ]
    )
    oracle_operators = np.stack(
        [
            apply_kernels_to_operator(low, kernels, views, depth, detector, offsets)
            for low, kernels in zip(
                development_low, development_oracle, strict=True
            )
        ]
    )
    mean_by_view = np.mean(train_kernels, axis=0)
    mean_operators = np.stack(
        [
            apply_kernels_to_operator(low, mean_by_view, views, depth, detector, offsets)
            for low in development_low
        ]
    )
    standardized_train = train_view_design[:, 1:8].reshape(len(train_vectors), views, -1)
    standardized_development = development_view_design[:, 1:8].reshape(
        len(development_vectors), views, -1
    )
    nearest_kernels = []
    for development_features in standardized_development:
        distance = np.linalg.norm(
            standardized_train - development_features[None], axis=(1, 2)
        )
        nearest_kernels.append(train_kernels[int(np.argmin(distance))])
    nearest_kernels = np.stack(nearest_kernels)
    nearest_operators = np.stack(
        [
            apply_kernels_to_operator(low, kernels, views, depth, detector, offsets)
            for low, kernels in zip(
                development_low, nearest_kernels, strict=True
            )
        ]
    )

    train_rig_design, rig_center, rig_scale = _rig_feature_matrix(
        train_vectors, views
    )
    development_rig_design, _, _ = _rig_feature_matrix(
        development_vectors, views, center=rig_center, scale=rig_scale
    )
    train_residual = train_truth - train_low
    full_ridge_best: dict[str, Any] | None = None
    for ridge in diagnosis_config["geometry_ridge_grid"]:
        coefficients = fit_ridge(
            train_rig_design,
            train_residual.reshape(len(train_residual), -1),
            float(ridge),
        )
        correction = (development_rig_design @ coefficients).reshape(
            development_truth.shape
        )
        operators = development_low + correction
        error = float(
            np.mean(
                _relative_residual_errors(
                    operators, development_truth, development_low
                )
            )
        )
        if full_ridge_best is None or error < full_ridge_best["error"]:
            full_ridge_best = {
                "error": error,
                "ridge": float(ridge),
                "operators": operators,
                "coefficients": coefficients,
            }
    assert full_ridge_best is not None
    methods = {
        "calibrated_low_renderer": development_low,
        "per_view_mean_kernel": mean_operators,
        "nearest_rig_kernel": nearest_operators,
        "full_matrix_geometry_ridge": full_ridge_best["operators"],
        "camera_local_kernel_geometry_ridge": predicted_operators,
        "oracle_camera_local_kernel": oracle_operators,
    }

    probe_rng = np.random.default_rng(int(source_config["seed"]) + 3991)
    probe_fields = []
    for family in source_config["probe_families"]:
        for _ in range(int(source_config["probes_per_family"])):
            probe_fields.append(
                make_reaction_field(
                    str(family),
                    detector,
                    depth,
                    probe_rng,
                ).reshape(-1)
            )
    metric_rows = []
    summary = {}
    for method, operators in methods.items():
        residual_errors = _relative_residual_errors(
            operators, development_truth, development_low
        )
        operator_errors = np.linalg.norm(
            operators - development_truth, axis=(1, 2)
        ) / np.maximum(np.linalg.norm(development_truth, axis=(1, 2)), 1e-15)
        method_forward = []
        method_gradient = []
        for rig_index, (operator, truth) in enumerate(
            zip(operators, development_truth, strict=True)
        ):
            forward = []
            gradient = []
            for field in probe_fields:
                truth_measurement = truth @ field
                forward.append(
                    np.linalg.norm(operator @ field - truth_measurement)
                    / max(np.linalg.norm(truth_measurement), 1e-15)
                )
                gradient.append(_gradient_cosine(operator, truth, field))
            method_forward.extend(forward)
            method_gradient.extend(gradient)
            metric_rows.append(
                {
                    "rig_id": development_rows[rig_index]["rig_id"],
                    "method": method,
                    "relative_renderer_residual_error": float(
                        residual_errors[rig_index]
                    ),
                    "relative_operator_error": float(operator_errors[rig_index]),
                    "mean_probe_forward_relative_error": float(np.mean(forward)),
                    "mean_probe_gradient_cosine": float(np.mean(gradient)),
                    "worst_probe_gradient_cosine": float(np.min(gradient)),
                }
            )
        summary[method] = {
            "mean_relative_renderer_residual_error": float(
                np.mean(residual_errors)
            ),
            "worst_relative_renderer_residual_error": float(
                np.max(residual_errors)
            ),
            "mean_relative_operator_error": float(np.mean(operator_errors)),
            "mean_probe_forward_relative_error": float(np.mean(method_forward)),
            "mean_probe_gradient_cosine": float(np.mean(method_gradient)),
            "worst_probe_gradient_cosine": float(np.min(method_gradient)),
        }

    candidate_error = summary["camera_local_kernel_geometry_ridge"][
        "mean_relative_renderer_residual_error"
    ]
    full_error = summary["full_matrix_geometry_ridge"][
        "mean_relative_renderer_residual_error"
    ]
    improvement = 1.0 - candidate_error / max(full_error, 1e-15)
    rule = diagnosis_config["diagnostic_reference_rule"]
    oracle_pass = summary["oracle_camera_local_kernel"][
        "mean_relative_renderer_residual_error"
    ] <= float(rule["maximum_oracle_kernel_relative_residual_error"])
    predictor_pass = improvement >= float(
        rule["minimum_prediction_improvement_over_full_matrix_ridge"]
    )

    adjoint_rng = np.random.default_rng(int(source_config["seed"]) + 4811)
    measurement_x = adjoint_rng.normal(size=(views, depth, detector))
    measurement_y = adjoint_rng.normal(size=(views, depth, detector))
    kernel_for_test = predicted_kernels[0]
    left = float(
        np.vdot(
            apply_camera_kernels(measurement_x, kernel_for_test, offsets),
            measurement_y,
        )
    )
    right = float(
        np.vdot(
            measurement_x,
            apply_camera_kernels_transpose(
                measurement_y, kernel_for_test, offsets
            ),
        )
    )
    adjoint_relative_defect = abs(left - right) / max(abs(left), abs(right), 1e-15)

    kernel_rows = []
    for split, rows, kernels in (
        ("train_oracle", [row for row in manifest if row["split"] == "train"], train_kernels),
        ("development_oracle", development_rows, development_oracle),
        ("development_prediction", development_rows, predicted_kernels),
    ):
        for rig_row, rig_kernels in zip(rows, kernels, strict=True):
            for view_index, view_kernel in enumerate(rig_kernels):
                for coefficient, (depth_offset, detector_offset) in zip(
                    view_kernel, offsets, strict=True
                ):
                    kernel_rows.append(
                        {
                            "split_role": split,
                            "rig_id": rig_row["rig_id"],
                            "view_index": view_index,
                            "depth_offset": depth_offset,
                            "detector_offset": detector_offset,
                            "coefficient": float(coefficient),
                        }
                    )
    report = {
        "schema": diagnosis_config["schema"],
        "evidence_label": diagnosis_config["evidence_label"],
        "claim_ceiling": diagnosis_config["claim_ceiling"],
        "config_sha256": sha256_json(diagnosis_config),
        "source_config_sha256": sha256_json(source_config),
        "reproduced_v5s_manifest_sha256": sha256_json(manifest),
        "source_provenance": {
            "runner": _file_sha256(Path(__file__).resolve()),
            "diagnosis_config": _file_sha256(CONFIG_PATH),
            "source_config": _file_sha256(source_config_path),
        },
        "oracle_firewall": {
            "truth_calibrated_geometry_used": True,
            "complete_training_operators_used_to_fit_kernel_targets": True,
            "limited_probe_estimator_trained": False,
            "development_truth_used_for_oracle_kernel_only": True,
        },
        "sample_accounting": {
            "train_rigs": int(np.sum(train_mask)),
            "development_rigs": int(np.sum(development_mask)),
            "design_lock_rigs": 0,
            "views_per_rig": views,
            "probe_fields": len(probe_fields),
        },
        "selected_development_hyperparameters": {
            "kernel_radius": selected_radius,
            "kernel_coefficients_per_view": len(offsets),
            "kernel_fit_ridge": selected_kernel_ridge,
            "geometry_ridge": float(best["geometry_ridge"]),
            "full_matrix_ridge": float(full_ridge_best["ridge"]),
        },
        "model_size": {
            "camera_kernel_coefficients_per_rig": views * len(offsets),
            "kernel_geometry_predictor_coefficients": int(
                best["coefficients"].size
            ),
            "full_matrix_geometry_predictor_coefficients": int(
                full_ridge_best["coefficients"].size
            ),
        },
        "development_summary": summary,
        "candidate_relative_improvement_over_full_matrix_ridge": improvement,
        "kernel_measurement_adjoint_relative_defect": adjoint_relative_defect,
        "diagnostic_reference_rule": rule,
        "decision": (
            "CAMERA_LOCAL_KERNEL_SIGNAL_POSTOPEN"
            if oracle_pass and predictor_pass
            else (
                "KERNEL_REPRESENTATION_SIGNAL_PREDICTOR_NO_GO_POSTOPEN"
                if oracle_pass
                else "CAMERA_LOCAL_KERNEL_REPRESENTATION_NO_GO_POSTOPEN"
            )
        ),
        "limitations": [
            "All rigs and mismatch rules were already opened in v5s-v5u.",
            "Truth-side geometry and complete training operators are supplied.",
            "Development truth selects kernel size and regularization.",
            "The local kernel is shift-invariant within each small camera image.",
            "No limited calibration probes, inverse reconstruction, nonlinear ray field dependence, or real data is evaluated.",
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_DIR / "sweep.csv", sweep_rows)
    _write_csv(OUTPUT_DIR / "development_rig_metrics.csv", metric_rows)
    _write_csv(OUTPUT_DIR / "selected_kernels.csv", kernel_rows)
    _write_json(OUTPUT_DIR / "report.json", report)
    _write_checksums(
        OUTPUT_DIR,
        ["sweep.csv", "development_rig_metrics.csv", "selected_kernels.csv", "report.json"],
    )
    return report


def main() -> None:
    report = run()
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "selected": report["selected_development_hyperparameters"],
                "model_size": report["model_size"],
                "candidate_improvement": report[
                    "candidate_relative_improvement_over_full_matrix_ridge"
                ],
                "adjoint_defect": report[
                    "kernel_measurement_adjoint_relative_defect"
                ],
                "development_summary": report["development_summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
