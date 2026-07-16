#!/usr/bin/env python3
"""Develop a ray-conditioned local voxel-kernel correction on opened rigs."""

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
        truth_parameter_vector,
    )
    from demo_t16_operator.run_v5u_calibrated_renderer_residual_screening import (
        _feature_matrix as _rig_feature_matrix,
    )
    from demo_t16_operator.run_v5w_clean_aperture_kernel_screening import (
        _right_kernel_basis,
        _thin_high_fidelity_operator,
        voxel_kernel_offsets,
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
    from .run_v5t_camera_local_tangent_diagnosis import truth_parameter_vector
    from .run_v5u_calibrated_renderer_residual_screening import (
        _feature_matrix as _rig_feature_matrix,
    )
    from .run_v5w_clean_aperture_kernel_screening import (
        _right_kernel_basis,
        _thin_high_fidelity_operator,
        voxel_kernel_offsets,
    )


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "v5x_ray_conditioned_voxel_kernel.json"
OUTPUT_DIR = ROOT / "results" / "v5x_ray_conditioned_voxel_kernel"
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


def ray_features(
    vector: np.ndarray,
    view_index: int,
    depth_index: int,
    detector_index: int,
    views: int,
    depth: int,
    detector: int,
) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64)
    angle = float(values[view_index])
    radius, cone_u, cone_z, bend = values[views:]
    detector_u = float(np.linspace(-0.82, 0.82, detector)[detector_index])
    detector_z = float(np.linspace(-0.82, 0.82, depth)[depth_index])
    sin_angle = np.sin(np.deg2rad(angle))
    cos_angle = np.cos(np.deg2rad(angle))
    return np.asarray(
        [
            sin_angle,
            cos_angle,
            detector_u,
            detector_z,
            radius,
            cone_u,
            cone_z,
            bend,
            detector_u * cone_u,
            detector_z * cone_z,
            detector_u * radius,
            detector_z * radius,
            sin_angle * detector_u,
            cos_angle * detector_u,
            detector_u**2,
            detector_z**2,
        ],
        dtype=np.float64,
    )


def ray_feature_matrix(
    vectors: Sequence[np.ndarray],
    views: int,
    depth: int,
    detector: int,
    *,
    center: np.ndarray | None = None,
    scale: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.stack(
        [
            ray_features(
                vector,
                view_index,
                depth_index,
                detector_index,
                views,
                depth,
                detector,
            )
            for vector in vectors
            for view_index in range(views)
            for depth_index in range(depth)
            for detector_index in range(detector)
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


def fit_rowwise_voxel_kernels(
    thin_operator: np.ndarray,
    finite_operator: np.ndarray,
    views: int,
    depth: int,
    detector: int,
    offsets: Sequence[tuple[int, int, int]],
    relative_ridge: float,
) -> np.ndarray:
    thin = np.asarray(thin_operator, dtype=np.float64).reshape(
        views, depth * detector, -1
    )
    finite = np.asarray(finite_operator, dtype=np.float64).reshape(
        views, depth * detector, -1
    )
    prior = np.zeros(len(offsets), dtype=np.float64)
    prior[list(offsets).index((0, 0, 0))] = 1.0
    output = np.empty((views, depth, detector, len(offsets)), dtype=np.float64)
    for view_index in range(views):
        basis = _right_kernel_basis(thin[view_index], depth, detector, offsets)
        for row_index in range(depth * detector):
            design = np.column_stack(
                [value[row_index].reshape(-1) for value in basis]
            )
            target = finite[view_index, row_index].reshape(-1)
            gram = design.T @ design
            penalty = float(relative_ridge) * max(
                float(np.trace(gram)) / len(gram), 1e-15
            )
            output[
                view_index, row_index // detector, row_index % detector
            ] = np.linalg.solve(
                gram + penalty * np.eye(len(gram)),
                design.T @ target + penalty * prior,
            )
    return output


def apply_rowwise_voxel_kernels_to_operator(
    operator: np.ndarray,
    kernels: np.ndarray,
    views: int,
    depth: int,
    detector: int,
    offsets: Sequence[tuple[int, int, int]],
) -> np.ndarray:
    values = np.asarray(operator, dtype=np.float64).reshape(
        views, depth * detector, -1
    )
    coefficients = np.asarray(kernels, dtype=np.float64).reshape(
        views, depth * detector, len(offsets)
    )
    output = np.empty_like(values)
    for view_index in range(views):
        basis = _right_kernel_basis(values[view_index], depth, detector, offsets)
        for row_index in range(depth * detector):
            output[view_index, row_index] = sum(
                coefficient * value[row_index]
                for coefficient, value in zip(
                    coefficients[view_index, row_index], basis, strict=True
                )
            )
    return output.reshape(views * depth * detector, -1)


def _relative_aperture_errors(
    prediction: np.ndarray, finite: np.ndarray, thin: np.ndarray
) -> np.ndarray:
    return np.linalg.norm(prediction - finite, axis=(1, 2)) / np.maximum(
        np.linalg.norm(finite - thin, axis=(1, 2)), 1e-15
    )


def run() -> dict[str, Any]:
    diagnosis_config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    source_config_path = ROOT / str(diagnosis_config["source_config"])
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    if not bool(diagnosis_config["design_lock_construction_forbidden"]):
        raise ValueError("v5x must forbid design-lock construction")
    v5s_report = json.loads(V5S_REPORT.read_text(encoding="utf-8"))
    manifest = build_rig_manifest(source_config)
    if sha256_json(manifest) != v5s_report["manifest_sha256_before_operator_build"]:
        raise RuntimeError("v5s manifest reproduction failed")
    views = int(source_config["views"])
    depth = int(source_config["depth"])
    detector = int(source_config["grid_size"])
    offsets = voxel_kernel_offsets(int(diagnosis_config["voxel_kernel_radius"]))
    prior = np.zeros(len(offsets), dtype=np.float64)
    prior[list(offsets).index((0, 0, 0))] = 1.0

    thin_all = []
    finite_all = []
    calibrated_vectors = []
    for row in manifest:
        _, finite, timing = _build_operator_pair(row, source_config)
        calibrated = truth_parameter_vector(row)
        thin = _thin_high_fidelity_operator(
            calibrated, source_config, timing["normalization_scale"]
        )
        thin_all.append(thin)
        finite_all.append(finite)
        calibrated_vectors.append(calibrated)
    thin_all = np.stack(thin_all)
    finite_all = np.stack(finite_all)
    train_mask = np.asarray([row["split"] == "train" for row in manifest])
    development_mask = ~train_mask
    train_thin = thin_all[train_mask]
    train_finite = finite_all[train_mask]
    development_thin = thin_all[development_mask]
    development_finite = finite_all[development_mask]
    train_vectors = [
        vector for vector, selected in zip(calibrated_vectors, train_mask, strict=True) if selected
    ]
    development_vectors = [
        vector
        for vector, selected in zip(calibrated_vectors, development_mask, strict=True)
        if selected
    ]
    development_rows = [row for row in manifest if row["split"] == "development"]
    train_ray_design, ray_center, ray_scale = ray_feature_matrix(
        train_vectors, views, depth, detector
    )
    development_ray_design, _, _ = ray_feature_matrix(
        development_vectors,
        views,
        depth,
        detector,
        center=ray_center,
        scale=ray_scale,
    )

    sweep_rows = []
    best_prediction: dict[str, Any] | None = None
    best_oracle: dict[str, Any] | None = None
    kernel_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for kernel_ridge in diagnosis_config["row_kernel_fit_ridge_grid"]:
        train_kernels = np.stack(
            [
                fit_rowwise_voxel_kernels(
                    thin,
                    finite,
                    views,
                    depth,
                    detector,
                    offsets,
                    float(kernel_ridge),
                )
                for thin, finite in zip(train_thin, train_finite, strict=True)
            ]
        )
        development_oracle = np.stack(
            [
                fit_rowwise_voxel_kernels(
                    thin,
                    finite,
                    views,
                    depth,
                    detector,
                    offsets,
                    float(kernel_ridge),
                )
                for thin, finite in zip(
                    development_thin, development_finite, strict=True
                )
            ]
        )
        kernel_cache[float(kernel_ridge)] = (train_kernels, development_oracle)
        oracle_operators = np.stack(
            [
                apply_rowwise_voxel_kernels_to_operator(
                    thin, kernels, views, depth, detector, offsets
                )
                for thin, kernels in zip(
                    development_thin, development_oracle, strict=True
                )
            ]
        )
        oracle_errors = _relative_aperture_errors(
            oracle_operators, development_finite, development_thin
        )
        if best_oracle is None or float(np.mean(oracle_errors)) < best_oracle[
            "mean_error"
        ]:
            best_oracle = {
                "row_kernel_fit_ridge": float(kernel_ridge),
                "mean_error": float(np.mean(oracle_errors)),
                "worst_error": float(np.max(oracle_errors)),
                "operators": oracle_operators,
                "kernels": development_oracle,
            }
        targets = (train_kernels - prior[None, None, None, None]).reshape(
            len(train_ray_design), len(offsets)
        )
        for predictor_ridge in diagnosis_config["ray_predictor_ridge_grid"]:
            coefficients = fit_ridge(
                train_ray_design, targets, float(predictor_ridge)
            )
            predicted_kernels = (
                development_ray_design @ coefficients
            ).reshape(
                len(development_vectors), views, depth, detector, len(offsets)
            ) + prior[None, None, None, None]
            operators = np.stack(
                [
                    apply_rowwise_voxel_kernels_to_operator(
                        thin, kernels, views, depth, detector, offsets
                    )
                    for thin, kernels in zip(
                        development_thin, predicted_kernels, strict=True
                    )
                ]
            )
            errors = _relative_aperture_errors(
                operators, development_finite, development_thin
            )
            record = {
                "row_kernel_fit_ridge": float(kernel_ridge),
                "ray_predictor_ridge": float(predictor_ridge),
                "development_mean_relative_aperture_error": float(
                    np.mean(errors)
                ),
                "development_worst_relative_aperture_error": float(
                    np.max(errors)
                ),
                "oracle_mean_relative_aperture_error": float(
                    np.mean(oracle_errors)
                ),
                "oracle_worst_relative_aperture_error": float(
                    np.max(oracle_errors)
                ),
            }
            sweep_rows.append(record)
            if best_prediction is None or record[
                "development_mean_relative_aperture_error"
            ] < best_prediction["mean_error"]:
                best_prediction = {
                    "row_kernel_fit_ridge": float(kernel_ridge),
                    "ray_predictor_ridge": float(predictor_ridge),
                    "mean_error": record[
                        "development_mean_relative_aperture_error"
                    ],
                    "worst_error": record[
                        "development_worst_relative_aperture_error"
                    ],
                    "operators": operators,
                    "kernels": predicted_kernels,
                    "coefficients": coefficients,
                }
    assert best_prediction is not None and best_oracle is not None

    train_design, rig_center, rig_scale = _rig_feature_matrix(train_vectors, views)
    development_design, _, _ = _rig_feature_matrix(
        development_vectors, views, center=rig_center, scale=rig_scale
    )
    train_residual = train_finite - train_thin
    full_ridge_best: dict[str, Any] | None = None
    for ridge in diagnosis_config["ray_predictor_ridge_grid"]:
        coefficients = fit_ridge(
            train_design,
            train_residual.reshape(len(train_residual), -1),
            float(ridge),
        )
        operators = development_thin + (
            development_design @ coefficients
        ).reshape(development_finite.shape)
        errors = _relative_aperture_errors(
            operators, development_finite, development_thin
        )
        if full_ridge_best is None or float(np.mean(errors)) < full_ridge_best[
            "mean_error"
        ]:
            full_ridge_best = {
                "ridge": float(ridge),
                "mean_error": float(np.mean(errors)),
                "worst_error": float(np.max(errors)),
                "operators": operators,
                "coefficients": coefficients,
            }
    assert full_ridge_best is not None
    methods = {
        "thin_ray_high_fidelity": development_thin,
        "full_matrix_geometry_ridge": full_ridge_best["operators"],
        "ray_conditioned_voxel_kernel": best_prediction["operators"],
        "oracle_rowwise_voxel_kernel": best_oracle["operators"],
    }

    probe_rng = np.random.default_rng(int(source_config["seed"]) + 5991)
    probe_fields = []
    for family in source_config["probe_families"]:
        for _ in range(int(source_config["probes_per_family"])):
            probe_fields.append(
                make_reaction_field(
                    str(family), detector, depth, probe_rng
                ).reshape(-1)
            )
    metric_rows = []
    summary = {}
    for method, operators in methods.items():
        aperture_errors = _relative_aperture_errors(
            operators, development_finite, development_thin
        )
        operator_errors = np.linalg.norm(
            operators - development_finite, axis=(1, 2)
        ) / np.maximum(np.linalg.norm(development_finite, axis=(1, 2)), 1e-15)
        forward_all = []
        gradient_all = []
        for rig_index, (operator, truth) in enumerate(
            zip(operators, development_finite, strict=True)
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
            forward_all.extend(forward)
            gradient_all.extend(gradient)
            metric_rows.append(
                {
                    "rig_id": development_rows[rig_index]["rig_id"],
                    "method": method,
                    "relative_aperture_residual_error": float(
                        aperture_errors[rig_index]
                    ),
                    "relative_operator_error": float(operator_errors[rig_index]),
                    "mean_probe_forward_relative_error": float(np.mean(forward)),
                    "mean_probe_gradient_cosine": float(np.mean(gradient)),
                    "worst_probe_gradient_cosine": float(np.min(gradient)),
                }
            )
        summary[method] = {
            "mean_relative_aperture_residual_error": float(
                np.mean(aperture_errors)
            ),
            "worst_relative_aperture_residual_error": float(
                np.max(aperture_errors)
            ),
            "mean_relative_operator_error": float(np.mean(operator_errors)),
            "mean_probe_forward_relative_error": float(np.mean(forward_all)),
            "mean_probe_gradient_cosine": float(np.mean(gradient_all)),
            "worst_probe_gradient_cosine": float(np.min(gradient_all)),
        }

    improvement = 1.0 - best_prediction["mean_error"] / max(
        full_ridge_best["mean_error"], 1e-15
    )
    worst_ratio = best_prediction["worst_error"] / max(
        full_ridge_best["worst_error"], 1e-15
    )
    rule = diagnosis_config["development_reference_rule"]
    reference_pass = (
        best_oracle["mean_error"]
        <= float(rule["maximum_oracle_relative_aperture_error"])
        and improvement
        >= float(rule["minimum_prediction_improvement_over_full_matrix_ridge"])
        and worst_ratio
        <= float(rule["maximum_worst_rig_ratio_to_full_matrix_ridge"])
    )

    coefficient_rows = []
    for feature_index, row in enumerate(best_prediction["coefficients"]):
        for kernel_index, value in enumerate(row):
            depth_offset, y_offset, x_offset = offsets[kernel_index]
            coefficient_rows.append(
                {
                    "feature_index": feature_index,
                    "kernel_index": kernel_index,
                    "depth_offset": depth_offset,
                    "y_offset": y_offset,
                    "x_offset": x_offset,
                    "coefficient": float(value),
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
        "factor_isolation": {
            "same_truth_geometry_and_path_sampling": True,
            "only_declared_difference": "aperture radius zero versus truth radius",
            "truth_calibrated_geometry_used": True,
        },
        "selected_development_hyperparameters": {
            "voxel_kernel_radius": int(diagnosis_config["voxel_kernel_radius"]),
            "kernel_coefficients_per_ray": len(offsets),
            "row_kernel_fit_ridge": best_prediction["row_kernel_fit_ridge"],
            "ray_predictor_ridge": best_prediction["ray_predictor_ridge"],
            "full_matrix_ridge": full_ridge_best["ridge"],
        },
        "model_size": {
            "ray_conditioned_predictor_coefficients": int(
                best_prediction["coefficients"].size
            ),
            "full_matrix_predictor_coefficients": int(
                full_ridge_best["coefficients"].size
            ),
            "compression_ratio_vs_full_matrix_predictor": float(
                full_ridge_best["coefficients"].size
                / best_prediction["coefficients"].size
            ),
        },
        "best_oracle": {
            "row_kernel_fit_ridge": best_oracle["row_kernel_fit_ridge"],
            "mean_relative_aperture_error": best_oracle["mean_error"],
            "worst_relative_aperture_error": best_oracle["worst_error"],
        },
        "full_matrix_baseline": {
            "mean_relative_aperture_error": full_ridge_best["mean_error"],
            "worst_relative_aperture_error": full_ridge_best["worst_error"],
        },
        "prediction_improvement_over_full_matrix_ridge": improvement,
        "prediction_worst_rig_ratio_to_full_matrix_ridge": worst_ratio,
        "development_summary": summary,
        "development_reference_rule": rule,
        "decision": (
            "RAY_CONDITIONED_KERNEL_DEVELOPMENT_SIGNAL_FREEZE_FRESH_NEXT"
            if reference_pass
            else "RAY_CONDITIONED_KERNEL_DEVELOPMENT_NO_GO"
        ),
        "sample_accounting": {
            "train_rigs": int(np.sum(train_mask)),
            "development_rigs": int(np.sum(development_mask)),
            "design_lock_rigs": 0,
            "views_per_rig": views,
            "rays_per_rig": views * depth * detector,
            "probe_fields": len(probe_fields),
        },
        "limitations": [
            "The hypothesis and features were designed after opening v5w.",
            "Truth geometry and complete train/development operators are used.",
            "Development truth selects regularization.",
            "The renderer is a prescribed linear weak-BOST surrogate.",
            "No fresh rigs, limited probes, inverse reconstruction, nonlinear field-dependent rays, or real data are tested.",
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_DIR / "sweep.csv", sweep_rows)
    _write_csv(OUTPUT_DIR / "development_rig_metrics.csv", metric_rows)
    _write_csv(OUTPUT_DIR / "selected_predictor_coefficients.csv", coefficient_rows)
    _write_json(OUTPUT_DIR / "report.json", report)
    _write_checksums(
        OUTPUT_DIR,
        [
            "sweep.csv",
            "development_rig_metrics.csv",
            "selected_predictor_coefficients.csv",
            "report.json",
        ],
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
                "best_oracle": report["best_oracle"],
                "full_matrix": report["full_matrix_baseline"],
                "improvement": report[
                    "prediction_improvement_over_full_matrix_ridge"
                ],
                "worst_ratio": report[
                    "prediction_worst_rig_ratio_to_full_matrix_ridge"
                ],
                "development_summary": report["development_summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
