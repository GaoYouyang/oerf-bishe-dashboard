#!/usr/bin/env python3
"""Factor-isolated finite-aperture kernel structure screening."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from demo_t16_operator.finite_aperture_bost import (
        build_finite_aperture_operator_bank,
    )
    from demo_t16_operator.gc_rio.data import _flatten_operator
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
    )
    from demo_t16_operator.run_v5v_camera_local_kernel_correction import (
        apply_kernels_to_operator,
        fit_camera_kernels,
        identity_kernel,
        kernel_offsets,
        view_feature_matrix,
    )
else:
    from .finite_aperture_bost import build_finite_aperture_operator_bank
    from .gc_rio.data import _flatten_operator
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
    )
    from .run_v5v_camera_local_kernel_correction import (
        apply_kernels_to_operator,
        fit_camera_kernels,
        identity_kernel,
        kernel_offsets,
        view_feature_matrix,
    )


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "v5w_clean_aperture_kernel_screening.json"
OUTPUT_DIR = ROOT / "results" / "v5w_clean_aperture_kernel_screening"
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


def _thin_high_fidelity_operator(
    vector: np.ndarray,
    source_config: Mapping[str, Any],
    normalization_scale: float,
) -> np.ndarray:
    views = int(source_config["views"])
    parameters = _renderer_parameters(vector, views)
    renderer = source_config["renderer"]
    n = int(source_config["grid_size"])
    depth = int(source_config["depth"])
    operator = build_finite_aperture_operator_bank(
        n,
        depth,
        parameters["angles"],
        [0.0],
        aperture_samples=int(renderer["truth_aperture_samples"]),
        path_samples=int(renderer["truth_path_samples"]),
        cone_u=float(parameters["cone_u"]),
        cone_z=float(parameters["cone_z"]),
        bend=float(parameters["bend"]),
        normalization_scale=float(normalization_scale),
    )[0]
    return _flatten_operator(operator).reshape(-1, n * n * depth).astype(np.float64)


def voxel_kernel_offsets(radius: int) -> list[tuple[int, int, int]]:
    value = int(radius)
    if value < 0:
        raise ValueError("voxel kernel radius must be non-negative")
    return [
        (depth_offset, y_offset, x_offset)
        for depth_offset in range(-value, value + 1)
        for y_offset in range(-value, value + 1)
        for x_offset in range(-value, value + 1)
    ]


def _shift_volume_to_output(
    values: np.ndarray, offset: tuple[int, int, int]
) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 3:
        raise ValueError("voxel volume must be three-dimensional")
    output = np.zeros_like(source)
    output_slices = []
    source_slices = []
    for size, delta in zip(source.shape, offset, strict=True):
        output_start = max(0, -delta)
        output_stop = min(size, size - delta)
        if output_start >= output_stop:
            return output
        output_slices.append(slice(output_start, output_stop))
        source_slices.append(slice(output_start + delta, output_stop + delta))
    output[tuple(output_slices)] = source[tuple(source_slices)]
    return output


def apply_voxel_kernel(
    field: np.ndarray,
    coefficients: np.ndarray,
    offsets: Sequence[tuple[int, int, int]],
) -> np.ndarray:
    values = np.asarray(field, dtype=np.float64)
    kernel = np.asarray(coefficients, dtype=np.float64)
    if kernel.shape != (len(offsets),):
        raise ValueError("voxel kernel dimensions do not match offsets")
    output = np.zeros_like(values)
    for coefficient, offset in zip(kernel, offsets, strict=True):
        output += coefficient * _shift_volume_to_output(values, offset)
    return output


def _right_kernel_basis(
    operator_view: np.ndarray,
    depth: int,
    detector: int,
    offsets: Sequence[tuple[int, int, int]],
) -> list[np.ndarray]:
    rows = np.asarray(operator_view, dtype=np.float64).reshape(-1, depth, detector, detector)
    basis = []
    for offset in offsets:
        transformed = np.stack(
            [
                _shift_volume_to_output(row, tuple(-value for value in offset))
                for row in rows
            ]
        )
        basis.append(transformed.reshape(operator_view.shape))
    return basis


def fit_camera_voxel_kernels(
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
    kernels = []
    for view_index in range(views):
        design = np.column_stack(
            [
                value.reshape(-1)
                for value in _right_kernel_basis(
                    thin[view_index], depth, detector, offsets
                )
            ]
        )
        target = finite[view_index].reshape(-1)
        gram = design.T @ design
        penalty = float(relative_ridge) * max(
            float(np.trace(gram)) / len(gram), 1e-15
        )
        kernels.append(
            np.linalg.solve(
                gram + penalty * np.eye(len(gram)),
                design.T @ target + penalty * prior,
            )
        )
    return np.stack(kernels)


def apply_voxel_kernels_to_operator(
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
    output = []
    for view_index in range(views):
        basis = _right_kernel_basis(values[view_index], depth, detector, offsets)
        output.append(
            sum(
                coefficient * value
                for coefficient, value in zip(
                    kernels[view_index], basis, strict=True
                )
            )
        )
    return np.stack(output).reshape(views * depth * detector, -1)


def _relative_aperture_errors(
    prediction: np.ndarray, finite: np.ndarray, thin: np.ndarray
) -> np.ndarray:
    return np.linalg.norm(prediction - finite, axis=(1, 2)) / np.maximum(
        np.linalg.norm(finite - thin, axis=(1, 2)), 1e-15
    )


def _screen_family(
    *,
    family: str,
    radii: Sequence[int],
    diagnosis_config: Mapping[str, Any],
    train_thin: np.ndarray,
    train_finite: np.ndarray,
    development_thin: np.ndarray,
    development_finite: np.ndarray,
    train_view_design: np.ndarray,
    development_view_design: np.ndarray,
    views: int,
    depth: int,
    detector: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    sweep = []
    best_prediction: dict[str, Any] | None = None
    best_oracle: dict[str, Any] | None = None
    for radius in radii:
        if family == "left_measurement":
            offsets: Sequence[tuple[int, ...]] = kernel_offsets(int(radius))
            fit_function: Callable[..., np.ndarray] = fit_camera_kernels
            apply_function: Callable[..., np.ndarray] = apply_kernels_to_operator
            prior = identity_kernel(offsets)
        elif family == "right_voxel":
            offsets = voxel_kernel_offsets(int(radius))
            fit_function = fit_camera_voxel_kernels
            apply_function = apply_voxel_kernels_to_operator
            prior = np.zeros(len(offsets), dtype=np.float64)
            prior[list(offsets).index((0, 0, 0))] = 1.0
        else:
            raise ValueError(f"unknown kernel family: {family}")
        for kernel_ridge in diagnosis_config["kernel_fit_ridge_grid"]:
            train_kernels = np.stack(
                [
                    fit_function(
                        thin,
                        finite,
                        views,
                        depth,
                        detector,
                        offsets,
                        float(kernel_ridge),
                    )
                    for thin, finite in zip(
                        train_thin, train_finite, strict=True
                    )
                ]
            )
            development_oracle = np.stack(
                [
                    fit_function(
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
            oracle_operators = np.stack(
                [
                    apply_function(
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
            oracle_record = {
                "family": family,
                "kernel_radius": int(radius),
                "kernel_coefficients_per_view": len(offsets),
                "kernel_fit_ridge": float(kernel_ridge),
                "mean_error": float(np.mean(oracle_errors)),
                "worst_error": float(np.max(oracle_errors)),
                "operators": oracle_operators,
                "kernels": development_oracle,
                "offsets": offsets,
            }
            if best_oracle is None or oracle_record["mean_error"] < best_oracle[
                "mean_error"
            ]:
                best_oracle = oracle_record
            targets = (train_kernels - prior[None, None]).reshape(
                len(train_kernels) * views, -1
            )
            for geometry_ridge in diagnosis_config["geometry_ridge_grid"]:
                coefficients = fit_ridge(
                    train_view_design, targets, float(geometry_ridge)
                )
                predicted_kernels = (
                    development_view_design @ coefficients
                ).reshape(len(development_thin), views, -1) + prior[None, None]
                predicted_operators = np.stack(
                    [
                        apply_function(
                            thin, kernels, views, depth, detector, offsets
                        )
                        for thin, kernels in zip(
                            development_thin, predicted_kernels, strict=True
                        )
                    ]
                )
                errors = _relative_aperture_errors(
                    predicted_operators, development_finite, development_thin
                )
                record = {
                    "family": family,
                    "kernel_radius": int(radius),
                    "kernel_coefficients_per_view": len(offsets),
                    "kernel_fit_ridge": float(kernel_ridge),
                    "geometry_ridge": float(geometry_ridge),
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
                sweep.append(record)
                if best_prediction is None or record[
                    "development_mean_relative_aperture_error"
                ] < best_prediction["mean_error"]:
                    best_prediction = {
                        "mean_error": record[
                            "development_mean_relative_aperture_error"
                        ],
                        "worst_error": record[
                            "development_worst_relative_aperture_error"
                        ],
                        "operators": predicted_operators,
                        "kernels": predicted_kernels,
                        "coefficients": coefficients,
                        "offsets": offsets,
                        **{
                            key: record[key]
                            for key in (
                                "family",
                                "kernel_radius",
                                "kernel_coefficients_per_view",
                                "kernel_fit_ridge",
                                "geometry_ridge",
                            )
                        },
                    }
    assert best_prediction is not None and best_oracle is not None
    return sweep, best_prediction, best_oracle


def run() -> dict[str, Any]:
    diagnosis_config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    source_config_path = ROOT / str(diagnosis_config["source_config"])
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    if not bool(diagnosis_config["design_lock_construction_forbidden"]):
        raise ValueError("v5w must forbid design-lock construction")
    v5s_report = json.loads(V5S_REPORT.read_text(encoding="utf-8"))
    manifest = build_rig_manifest(source_config)
    if sha256_json(manifest) != v5s_report["manifest_sha256_before_operator_build"]:
        raise RuntimeError("v5s manifest reproduction failed")
    views = int(source_config["views"])
    depth = int(source_config["depth"])
    detector = int(source_config["grid_size"])

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
    train_view_design, center, scale = view_feature_matrix(train_vectors, views)
    development_view_design, _, _ = view_feature_matrix(
        development_vectors, views, center=center, scale=scale
    )

    left_sweep, left_prediction, left_oracle = _screen_family(
        family="left_measurement",
        radii=diagnosis_config["left_measurement_kernel_radii"],
        diagnosis_config=diagnosis_config,
        train_thin=train_thin,
        train_finite=train_finite,
        development_thin=development_thin,
        development_finite=development_finite,
        train_view_design=train_view_design,
        development_view_design=development_view_design,
        views=views,
        depth=depth,
        detector=detector,
    )
    right_sweep, right_prediction, right_oracle = _screen_family(
        family="right_voxel",
        radii=diagnosis_config["right_voxel_kernel_radii"],
        diagnosis_config=diagnosis_config,
        train_thin=train_thin,
        train_finite=train_finite,
        development_thin=development_thin,
        development_finite=development_finite,
        train_view_design=train_view_design,
        development_view_design=development_view_design,
        views=views,
        depth=depth,
        detector=detector,
    )

    train_design, rig_center, rig_scale = _rig_feature_matrix(train_vectors, views)
    development_design, _, _ = _rig_feature_matrix(
        development_vectors, views, center=rig_center, scale=rig_scale
    )
    train_residual = train_finite - train_thin
    full_ridge_best: dict[str, Any] | None = None
    for ridge in diagnosis_config["geometry_ridge_grid"]:
        coefficients = fit_ridge(
            train_design,
            train_residual.reshape(len(train_residual), -1),
            float(ridge),
        )
        operators = development_thin + (
            development_design @ coefficients
        ).reshape(development_finite.shape)
        error = float(
            np.mean(
                _relative_aperture_errors(
                    operators, development_finite, development_thin
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
        "thin_ray_high_fidelity": development_thin,
        "full_matrix_geometry_ridge": full_ridge_best["operators"],
        "left_measurement_kernel_geometry_ridge": left_prediction["operators"],
        "right_voxel_kernel_geometry_ridge": right_prediction["operators"],
        "oracle_left_measurement_kernel": left_oracle["operators"],
        "oracle_right_voxel_kernel": right_oracle["operators"],
    }

    probe_rng = np.random.default_rng(int(source_config["seed"]) + 4991)
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

    best_prediction = min(
        (left_prediction, right_prediction), key=lambda value: value["mean_error"]
    )
    best_oracle = min(
        (left_oracle, right_oracle), key=lambda value: value["mean_error"]
    )
    improvement = 1.0 - best_prediction["mean_error"] / max(
        full_ridge_best["error"], 1e-15
    )
    rule = diagnosis_config["diagnostic_reference_rule"]
    oracle_pass = best_oracle["mean_error"] <= float(
        rule["maximum_oracle_relative_aperture_residual_error"]
    )
    predictor_pass = improvement >= float(
        rule["minimum_prediction_improvement_over_full_matrix_ridge"]
    )

    adjoint_rng = np.random.default_rng(int(source_config["seed"]) + 5881)
    field = adjoint_rng.normal(size=(depth, detector, detector))
    selected_right_offsets = right_prediction["offsets"]
    selected_right_kernel = right_prediction["kernels"][0, 0]
    right_operator = apply_voxel_kernels_to_operator(
        development_thin[0],
        right_prediction["kernels"][0],
        views,
        depth,
        detector,
        selected_right_offsets,
    ).reshape(views, depth * detector, -1)[0]
    direct_measurement = right_operator @ field.reshape(-1)
    composed_measurement = development_thin[0].reshape(
        views, depth * detector, -1
    )[0] @ apply_voxel_kernel(
        field, selected_right_kernel, selected_right_offsets
    ).reshape(-1)
    right_composition_relative_defect = float(
        np.linalg.norm(direct_measurement - composed_measurement)
        / max(np.linalg.norm(composed_measurement), 1e-15)
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
            "finite_aperture_renderer": _file_sha256(
                ROOT / "finite_aperture_bost.py"
            ),
        },
        "factor_isolation": {
            "thin_and_finite_share_truth_angles": True,
            "thin_and_finite_share_cone_and_bend": True,
            "thin_and_finite_share_truth_path_samples": True,
            "only_declared_difference": "aperture radius zero versus truth radius",
            "truth_calibrated_geometry_used": True,
        },
        "sample_accounting": {
            "train_rigs": int(np.sum(train_mask)),
            "development_rigs": int(np.sum(development_mask)),
            "design_lock_rigs": 0,
            "views_per_rig": views,
            "probe_fields": len(probe_fields),
        },
        "best_left_prediction": {
            key: value
            for key, value in left_prediction.items()
            if key not in {"operators", "kernels", "coefficients", "offsets"}
        },
        "best_right_prediction": {
            key: value
            for key, value in right_prediction.items()
            if key not in {"operators", "kernels", "coefficients", "offsets"}
        },
        "best_left_oracle": {
            key: value
            for key, value in left_oracle.items()
            if key not in {"operators", "kernels", "offsets"}
        },
        "best_right_oracle": {
            key: value
            for key, value in right_oracle.items()
            if key not in {"operators", "kernels", "offsets"}
        },
        "full_matrix_geometry_ridge": {
            "relative_aperture_error": float(full_ridge_best["error"]),
            "ridge": float(full_ridge_best["ridge"]),
            "predictor_coefficients": int(full_ridge_best["coefficients"].size),
        },
        "best_predicted_kernel_family": best_prediction["family"],
        "best_oracle_kernel_family": best_oracle["family"],
        "best_prediction_improvement_over_full_matrix_ridge": improvement,
        "right_kernel_composition_relative_defect": right_composition_relative_defect,
        "development_summary": summary,
        "diagnostic_reference_rule": rule,
        "decision": (
            "FACTOR_ISOLATED_APERTURE_KERNEL_SIGNAL_POSTOPEN"
            if oracle_pass and predictor_pass
            else (
                "APERTURE_KERNEL_REPRESENTATION_SIGNAL_PREDICTOR_NO_GO_POSTOPEN"
                if oracle_pass
                else "APERTURE_KERNEL_REPRESENTATION_NO_GO_POSTOPEN"
            )
        ),
        "limitations": [
            "The factor-isolated task was designed after opening v5s-v5v results.",
            "Truth geometry and complete operator matrices are supplied.",
            "Development truth selects kernel family, size, and regularization.",
            "The renderer remains a prescribed linear weak-BOST surrogate.",
            "No calibration probe learner, inverse reconstruction, nonlinear field-dependent rays, or real data is tested.",
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_DIR / "sweep.csv", left_sweep + right_sweep)
    _write_csv(OUTPUT_DIR / "development_rig_metrics.csv", metric_rows)
    _write_json(OUTPUT_DIR / "report.json", report)
    _write_checksums(
        OUTPUT_DIR, ["sweep.csv", "development_rig_metrics.csv", "report.json"]
    )
    return report


def main() -> None:
    report = run()
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "best_left_prediction": report["best_left_prediction"],
                "best_right_prediction": report["best_right_prediction"],
                "best_left_oracle": report["best_left_oracle"],
                "best_right_oracle": report["best_right_oracle"],
                "full_matrix": report["full_matrix_geometry_ridge"],
                "best_prediction_improvement": report[
                    "best_prediction_improvement_over_full_matrix_ridge"
                ],
                "right_composition_defect": report[
                    "right_kernel_composition_relative_defect"
                ],
                "development_summary": report["development_summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
