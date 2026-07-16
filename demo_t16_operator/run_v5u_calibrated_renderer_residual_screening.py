#!/usr/bin/env python3
"""Post-open screening of renderer residuals after oracle geometry calibration."""

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
    from demo_t16_operator.finite_aperture_bost import (
        build_finite_aperture_operator_bank,
    )
    from demo_t16_operator.gc_rio.data import _flatten_operator
    from demo_t16_operator.gc_rio.protocol import sha256_json
    from demo_t16_operator.independent_reaction_bost import make_reaction_field
    from demo_t16_operator.run_v5s_dco_low_rank_screening import (
        _build_operator_pair,
        _gradient_cosine,
        _nearest_predictions,
        fit_ridge,
        hosvd_bases,
        project_cores,
        reconstruct_operators,
        build_rig_manifest,
    )
    from demo_t16_operator.run_v5t_camera_local_tangent_diagnosis import (
        _renderer_parameters,
        truth_parameter_vector,
    )
else:
    from .finite_aperture_bost import build_finite_aperture_operator_bank
    from .gc_rio.data import _flatten_operator
    from .gc_rio.protocol import sha256_json
    from .independent_reaction_bost import make_reaction_field
    from .run_v5s_dco_low_rank_screening import (
        _build_operator_pair,
        _gradient_cosine,
        _nearest_predictions,
        fit_ridge,
        hosvd_bases,
        project_cores,
        reconstruct_operators,
        build_rig_manifest,
    )
    from .run_v5t_camera_local_tangent_diagnosis import (
        _renderer_parameters,
        truth_parameter_vector,
    )


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "v5u_calibrated_renderer_residual_screening.json"
OUTPUT_DIR = ROOT / "results" / "v5u_calibrated_renderer_residual_screening"
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


def calibrated_geometry_features(vector: np.ndarray, views: int) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64)
    parameters = _renderer_parameters(values, int(views))
    angles = np.asarray(parameters["angles"], dtype=np.float64)
    return np.asarray(
        [
            *np.sin(np.deg2rad(angles)),
            *np.cos(np.deg2rad(angles)),
            float(parameters["aperture_radius"]),
            float(parameters["cone_u"]),
            float(parameters["cone_z"]),
            float(parameters["bend"]),
            float(np.mean(np.diff(angles))),
            float(np.std(np.diff(angles))),
        ],
        dtype=np.float64,
    )


def _feature_matrix(
    vectors: Sequence[np.ndarray],
    views: int,
    *,
    center: np.ndarray | None = None,
    scale: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.stack([calibrated_geometry_features(row, views) for row in vectors])
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


def _low_fidelity_operator_at_calibrated_geometry(
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
        [parameters["aperture_radius"]],
        aperture_samples=int(renderer["nominal_aperture_samples"]),
        path_samples=int(renderer["nominal_path_samples"]),
        cone_u=float(parameters["cone_u"]),
        cone_z=float(parameters["cone_z"]),
        bend=float(parameters["bend"]),
        normalization_scale=float(normalization_scale),
    )[0]
    return _flatten_operator(operator).reshape(-1, n * n * depth).astype(np.float64)


def _relative_errors(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.linalg.norm(prediction - truth, axis=(1, 2)) / np.maximum(
        np.linalg.norm(truth, axis=(1, 2)), 1e-15
    )


def run() -> dict[str, Any]:
    diagnosis_config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    source_config_path = ROOT / str(diagnosis_config["source_config"])
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    if not bool(diagnosis_config["design_lock_construction_forbidden"]):
        raise ValueError("v5u must forbid design-lock construction")
    v5s_report = json.loads(V5S_REPORT.read_text(encoding="utf-8"))
    manifest = build_rig_manifest(source_config)
    if sha256_json(manifest) != v5s_report["manifest_sha256_before_operator_build"]:
        raise RuntimeError("v5s manifest reproduction failed")

    low_nominal_all = []
    low_calibrated_all = []
    truth_all = []
    calibrated_vectors = []
    for row in manifest:
        low_nominal, truth, timing = _build_operator_pair(row, source_config)
        calibrated = truth_parameter_vector(row)
        low_calibrated = _low_fidelity_operator_at_calibrated_geometry(
            calibrated, source_config, timing["normalization_scale"]
        )
        low_nominal_all.append(low_nominal)
        low_calibrated_all.append(low_calibrated)
        truth_all.append(truth)
        calibrated_vectors.append(calibrated)
    low_nominal_all = np.stack(low_nominal_all)
    low_calibrated_all = np.stack(low_calibrated_all)
    truth_all = np.stack(truth_all)
    residual_all = truth_all - low_calibrated_all
    raw_discrepancy_all = truth_all - low_nominal_all
    train_mask = np.asarray([row["split"] == "train" for row in manifest])
    development_mask = ~train_mask
    train_residual = residual_all[train_mask]
    development_residual = residual_all[development_mask]
    development_truth = truth_all[development_mask]
    development_low = low_calibrated_all[development_mask]
    train_vectors = [
        vector for vector, selected in zip(calibrated_vectors, train_mask, strict=True) if selected
    ]
    development_vectors = [
        vector
        for vector, selected in zip(calibrated_vectors, development_mask, strict=True)
        if selected
    ]
    development_rows = [row for row in manifest if row["split"] == "development"]
    train_design, center, scale = _feature_matrix(
        train_vectors, int(source_config["views"])
    )
    development_design, _, _ = _feature_matrix(
        development_vectors,
        int(source_config["views"]),
        center=center,
        scale=scale,
    )
    u_full, v_full, measurement_singular, voxel_singular = hosvd_bases(
        train_residual
    )

    sweep_rows = []
    best: dict[str, Any] | None = None
    for rank_m in diagnosis_config["rank_grid"]:
        for rank_v in diagnosis_config["rank_grid"]:
            u = u_full[:, : int(rank_m)]
            v = v_full[:, : int(rank_v)]
            train_cores = project_cores(train_residual, u, v)
            for ridge in diagnosis_config["ridge_grid"]:
                coefficients = fit_ridge(
                    train_design,
                    train_cores.reshape(len(train_cores), -1),
                    float(ridge),
                )
                predicted_cores = (development_design @ coefficients).reshape(
                    len(development_rows), int(rank_m), int(rank_v)
                )
                prediction = reconstruct_operators(predicted_cores, u, v)
                errors = _relative_errors(prediction, development_residual)
                record = {
                    "rank_measurement": int(rank_m),
                    "rank_voxel": int(rank_v),
                    "relative_ridge": float(ridge),
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
                    int(rank_m) * int(rank_v),
                ) < (
                    best["development_mean_relative_residual_error"],
                    best["rank_measurement"] * best["rank_voxel"],
                ):
                    best = {**record, "coefficients": coefficients}
    assert best is not None
    rank_m = int(best["rank_measurement"])
    rank_v = int(best["rank_voxel"])
    selected_u = u_full[:, :rank_m]
    selected_v = v_full[:, :rank_v]
    predicted_cores = (development_design @ best["coefficients"]).reshape(
        len(development_rows), rank_m, rank_v
    )
    hosvd_prediction = reconstruct_operators(
        predicted_cores, selected_u, selected_v
    )
    mean_prediction = np.broadcast_to(
        np.mean(train_residual, axis=0), development_residual.shape
    ).copy()
    nearest_prediction = _nearest_predictions(
        train_design, development_design, train_residual
    )
    full_ridge_best: dict[str, Any] | None = None
    for ridge in diagnosis_config["ridge_grid"]:
        coefficients = fit_ridge(
            train_design,
            train_residual.reshape(len(train_residual), -1),
            float(ridge),
        )
        prediction = (development_design @ coefficients).reshape(
            development_residual.shape
        )
        error = float(np.mean(_relative_errors(prediction, development_residual)))
        if full_ridge_best is None or error < full_ridge_best["error"]:
            full_ridge_best = {
                "error": error,
                "ridge": float(ridge),
                "prediction": prediction,
            }
    assert full_ridge_best is not None
    oracle_cores = project_cores(development_residual, selected_u, selected_v)
    oracle_prediction = reconstruct_operators(
        oracle_cores, selected_u, selected_v
    )
    predictions = {
        "zero_calibrated_low_renderer": np.zeros_like(development_residual),
        "mean_renderer_residual": mean_prediction,
        "nearest_calibrated_geometry": nearest_prediction,
        "full_matrix_geometry_ridge": full_ridge_best["prediction"],
        "cal_hosvd_ridge": hosvd_prediction,
        "oracle_shared_subspace": oracle_prediction,
    }

    probe_rng = np.random.default_rng(int(source_config["seed"]) + 2991)
    probe_fields = []
    for family in source_config["probe_families"]:
        for _ in range(int(source_config["probes_per_family"])):
            probe_fields.append(
                make_reaction_field(
                    str(family),
                    int(source_config["grid_size"]),
                    int(source_config["depth"]),
                    probe_rng,
                ).reshape(-1)
            )
    metric_rows = []
    summary = {}
    for method, correction in predictions.items():
        corrected = development_low + correction
        residual_error = _relative_errors(correction, development_residual)
        operator_error = np.linalg.norm(
            corrected - development_truth, axis=(1, 2)
        ) / np.maximum(np.linalg.norm(development_truth, axis=(1, 2)), 1e-15)
        forward_by_rig = []
        gradient_by_rig = []
        for rig_index, (operator, truth) in enumerate(
            zip(corrected, development_truth, strict=True)
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
            forward_by_rig.append(float(np.mean(forward)))
            gradient_by_rig.append(float(np.mean(gradient)))
            metric_rows.append(
                {
                    "rig_id": development_rows[rig_index]["rig_id"],
                    "method": method,
                    "relative_renderer_residual_error": float(
                        residual_error[rig_index]
                    ),
                    "relative_operator_error": float(operator_error[rig_index]),
                    "mean_probe_forward_relative_error": float(np.mean(forward)),
                    "mean_probe_gradient_cosine": float(np.mean(gradient)),
                    "worst_probe_gradient_cosine": float(np.min(gradient)),
                }
            )
        summary[method] = {
            "mean_relative_renderer_residual_error": float(
                np.mean(residual_error)
            ),
            "worst_relative_renderer_residual_error": float(
                np.max(residual_error)
            ),
            "mean_relative_operator_error": float(np.mean(operator_error)),
            "mean_probe_forward_relative_error": float(np.mean(forward_by_rig)),
            "mean_probe_gradient_cosine": float(np.mean(gradient_by_rig)),
            "worst_probe_gradient_cosine": float(
                np.min(
                    [row["worst_probe_gradient_cosine"] for row in metric_rows if row["method"] == method]
                )
            ),
        }

    full_error = summary["full_matrix_geometry_ridge"][
        "mean_relative_renderer_residual_error"
    ]
    candidate_error = summary["cal_hosvd_ridge"][
        "mean_relative_renderer_residual_error"
    ]
    relative_improvement = 1.0 - candidate_error / max(full_error, 1e-15)
    rule = diagnosis_config["diagnostic_reference_rule"]
    reference_pass = (
        summary["oracle_shared_subspace"][
            "mean_relative_renderer_residual_error"
        ]
        <= float(rule["maximum_oracle_shared_subspace_error"])
        and relative_improvement
        >= float(rule["minimum_relative_improvement_over_full_matrix_ridge"])
    )
    raw_norms = np.linalg.norm(raw_discrepancy_all[development_mask], axis=(1, 2))
    calibrated_norms = np.linalg.norm(development_residual, axis=(1, 2))
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
        "oracle_firewall": {
            "truth_calibrated_geometry_used": True,
            "calibration_estimator_trained": False,
            "interpretation": (
                "This diagnostic conditions both low and high renderers on the "
                "same truth-side geometry to isolate renderer fidelity residuals."
            ),
        },
        "sample_accounting": {
            "train_rigs": int(np.sum(train_mask)),
            "development_rigs": int(np.sum(development_mask)),
            "design_lock_rigs": 0,
            "probe_fields": len(probe_fields),
        },
        "geometry_alignment_effect": {
            "mean_calibrated_residual_to_raw_discrepancy_norm_ratio": float(
                np.mean(calibrated_norms / np.maximum(raw_norms, 1e-15))
            ),
            "mean_fraction_raw_discrepancy_norm_removed": float(
                np.mean(1.0 - calibrated_norms / np.maximum(raw_norms, 1e-15))
            ),
        },
        "selected_development_hyperparameters": {
            "rank_measurement": rank_m,
            "rank_voxel": rank_v,
            "relative_ridge": float(best["relative_ridge"]),
            "full_matrix_ridge": float(full_ridge_best["ridge"]),
        },
        "singular_energy": {
            "measurement_first_4": float(
                np.sum(np.square(measurement_singular[:4]))
                / np.sum(np.square(measurement_singular))
            ),
            "measurement_first_16": float(
                np.sum(np.square(measurement_singular[:16]))
                / np.sum(np.square(measurement_singular))
            ),
            "voxel_first_4": float(
                np.sum(np.square(voxel_singular[:4]))
                / np.sum(np.square(voxel_singular))
            ),
            "voxel_first_16": float(
                np.sum(np.square(voxel_singular[:16]))
                / np.sum(np.square(voxel_singular))
            ),
        },
        "raw_v5s_singular_energy": v5s_report["singular_energy"],
        "development_summary": summary,
        "cal_hosvd_relative_improvement_over_full_matrix_ridge": relative_improvement,
        "diagnostic_reference_rule": rule,
        "decision": (
            "CALIBRATED_RENDERER_LOW_RANK_SIGNAL_POSTOPEN"
            if reference_pass
            else "CALIBRATED_RENDERER_LOW_RANK_NO_GO_POSTOPEN"
        ),
        "limitations": [
            "The same rigs and mismatch generator were already opened in v5s/v5t.",
            "Truth-side geometry is supplied; no calibration inference is tested.",
            "Training uses complete residual matrices rather than limited probes.",
            "The renderer is a prescribed linear weak-BOST surrogate.",
            "No inverse reconstruction or real data is evaluated.",
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_DIR / "sweep.csv", sweep_rows)
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
                "alignment": report["geometry_alignment_effect"],
                "selected": report["selected_development_hyperparameters"],
                "singular_energy": report["singular_energy"],
                "relative_improvement": report[
                    "cal_hosvd_relative_improvement_over_full_matrix_ridge"
                ],
                "development_summary": report["development_summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
