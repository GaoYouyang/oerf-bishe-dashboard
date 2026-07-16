#!/usr/bin/env python3
"""Directly train a compact ray-conditioned local operator correction."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from demo_t16_operator.gc_rio.protocol import sha256_json
    from demo_t16_operator.independent_reaction_bost import make_reaction_field
    from demo_t16_operator.release_provenance import (
        relative_file_hashes,
        runtime_environment,
    )
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
    from demo_t16_operator.run_v5x_ray_conditioned_voxel_kernel import (
        ray_feature_matrix,
    )
else:
    from .gc_rio.protocol import sha256_json
    from .independent_reaction_bost import make_reaction_field
    from .release_provenance import relative_file_hashes, runtime_environment
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
    from .run_v5x_ray_conditioned_voxel_kernel import ray_feature_matrix


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "v5y_direct_ray_conditioned_kernel.json"
OUTPUT_DIR = ROOT / "results" / "v5y_direct_ray_conditioned_kernel"
V5S_REPORT = ROOT / "results" / "v5s_dco_low_rank_screening" / "report.json"
V5X_REPORT = ROOT / "results" / "v5x_ray_conditioned_voxel_kernel" / "report.json"


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


def build_ray_basis_and_targets(
    thin_operators: np.ndarray,
    finite_operators: np.ndarray,
    views: int,
    depth: int,
    detector: int,
    offsets: Sequence[tuple[int, int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    thin_all = np.asarray(thin_operators, dtype=np.float64)
    finite_all = np.asarray(finite_operators, dtype=np.float64)
    basis_rows = []
    target_rows = []
    for thin, finite in zip(thin_all, finite_all, strict=True):
        thin_views = thin.reshape(views, depth * detector, -1)
        finite_views = finite.reshape(views, depth * detector, -1)
        for view_index in range(views):
            basis = _right_kernel_basis(
                thin_views[view_index], depth, detector, offsets
            )
            stacked = np.stack(basis, axis=-1)
            basis_rows.append(stacked)
            target_rows.append(finite_views[view_index] - thin_views[view_index])
    return (
        np.concatenate(basis_rows, axis=0).astype(np.float32),
        np.concatenate(target_rows, axis=0).astype(np.float32),
    )


def _device_from_config(value: str) -> torch.device:
    if value == "mps_if_available" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_direct_model(
    features: np.ndarray,
    basis: np.ndarray,
    targets: np.ndarray,
    *,
    steps: int,
    batch_rays: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    gradient_clip_norm: float | None = None,
    cosine_eta_fraction: float | None = None,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    feature_values = np.asarray(features, dtype=np.float32)
    basis_values = np.asarray(basis, dtype=np.float32)
    target_values = np.asarray(targets, dtype=np.float32)
    if feature_values.shape[0] != basis_values.shape[0] or basis_values.shape[
        :2
    ] != target_values.shape:
        raise ValueError("ray features, bases, and targets are not aligned")
    torch.manual_seed(int(seed))
    parameter = torch.nn.Parameter(
        torch.zeros(
            (feature_values.shape[1], basis_values.shape[2]),
            dtype=torch.float32,
            device=device,
        )
    )
    optimizer = torch.optim.Adam([parameter], lr=float(learning_rate))
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(steps),
            eta_min=float(learning_rate) * float(cosine_eta_fraction),
        )
        if cosine_eta_fraction is not None
        else None
    )
    rng = np.random.default_rng(int(seed))
    residual_scale = max(float(np.sqrt(np.mean(np.square(target_values)))), 1e-8)
    history = []
    for step in range(1, int(steps) + 1):
        indices = rng.integers(0, len(feature_values), size=int(batch_rays))
        feature_batch = torch.as_tensor(feature_values[indices], device=device)
        basis_batch = torch.as_tensor(basis_values[indices], device=device)
        target_batch = torch.as_tensor(target_values[indices], device=device)
        kernel_delta = feature_batch @ parameter
        prediction = torch.einsum("bpk,bk->bp", basis_batch, kernel_delta)
        data_loss = torch.mean(torch.square((prediction - target_batch) / residual_scale))
        regularizer = float(weight_decay) * torch.mean(torch.square(parameter))
        loss = data_loss + regularizer
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(parameter, float(gradient_clip_norm))
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if step == 1 or step % 250 == 0 or step == int(steps):
            history.append(
                {
                    "step": float(step),
                    "batch_data_loss": float(data_loss.detach().cpu()),
                    "regularizer": float(regularizer.detach().cpu()),
                    "total_loss": float(loss.detach().cpu()),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )
    return parameter.detach().cpu().numpy().astype(np.float64), history


def predict_ray_residuals(
    features: np.ndarray,
    basis: np.ndarray,
    coefficients: np.ndarray,
    *,
    device: torch.device,
    batch_rays: int = 256,
) -> np.ndarray:
    feature_values = np.asarray(features, dtype=np.float32)
    basis_values = np.asarray(basis, dtype=np.float32)
    parameter = torch.as_tensor(
        np.asarray(coefficients, dtype=np.float32), device=device
    )
    output = []
    with torch.no_grad():
        for start in range(0, len(feature_values), int(batch_rays)):
            stop = min(start + int(batch_rays), len(feature_values))
            feature_batch = torch.as_tensor(feature_values[start:stop], device=device)
            basis_batch = torch.as_tensor(basis_values[start:stop], device=device)
            kernel_delta = feature_batch @ parameter
            prediction = torch.einsum("bpk,bk->bp", basis_batch, kernel_delta)
            output.append(prediction.cpu().numpy())
    return np.concatenate(output, axis=0).astype(np.float64)


def _relative_aperture_errors(
    prediction: np.ndarray, finite: np.ndarray, thin: np.ndarray
) -> np.ndarray:
    return np.linalg.norm(prediction - finite, axis=(1, 2)) / np.maximum(
        np.linalg.norm(finite - thin, axis=(1, 2)), 1e-15
    )


def run(
    config_path: Path = CONFIG_PATH,
    output_dir: Path = OUTPUT_DIR,
    entrypoint_path: Path | None = None,
) -> dict[str, Any]:
    diagnosis_config = json.loads(config_path.read_text(encoding="utf-8"))
    source_config_path = ROOT / str(diagnosis_config["source_config"])
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    if not bool(diagnosis_config["design_lock_construction_forbidden"]):
        raise ValueError("v5y must forbid design-lock construction")
    v5s_report = json.loads(V5S_REPORT.read_text(encoding="utf-8"))
    v5x_report = json.loads(V5X_REPORT.read_text(encoding="utf-8"))
    manifest = build_rig_manifest(source_config)
    if sha256_json(manifest) != v5s_report["manifest_sha256_before_operator_build"]:
        raise RuntimeError("v5s manifest reproduction failed")
    views = int(source_config["views"])
    depth = int(source_config["depth"])
    detector = int(source_config["grid_size"])
    rays_per_rig = views * depth * detector
    offsets = voxel_kernel_offsets(int(diagnosis_config["voxel_kernel_radius"]))

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
    internal_fit_rigs = int(diagnosis_config["internal_fit_rigs"])
    if not 1 <= internal_fit_rigs < len(train_vectors):
        raise ValueError("internal_fit_rigs must leave at least one selection rig")

    train_basis, train_targets = build_ray_basis_and_targets(
        train_thin, train_finite, views, depth, detector, offsets
    )
    development_basis, development_targets = build_ray_basis_and_targets(
        development_thin,
        development_finite,
        views,
        depth,
        detector,
        offsets,
    )
    fit_vectors = train_vectors[:internal_fit_rigs]
    selection_vectors = train_vectors[internal_fit_rigs:]
    fit_features, fit_center, fit_scale = ray_feature_matrix(
        fit_vectors, views, depth, detector
    )
    selection_features, _, _ = ray_feature_matrix(
        selection_vectors,
        views,
        depth,
        detector,
        center=fit_center,
        scale=fit_scale,
    )
    fit_stop = internal_fit_rigs * rays_per_rig
    fit_basis = train_basis[:fit_stop]
    fit_targets = train_targets[:fit_stop]
    selection_basis = train_basis[fit_stop:]
    selection_targets = train_targets[fit_stop:]
    optimizer_config = diagnosis_config["optimizer"]
    device = _device_from_config(str(optimizer_config["device"]))

    selection_rows = []
    best_selection: dict[str, Any] | None = None
    for index, weight_decay in enumerate(optimizer_config["weight_decay_grid"]):
        coefficients, history = train_direct_model(
            fit_features,
            fit_basis,
            fit_targets,
            steps=int(optimizer_config["steps"]),
            batch_rays=int(optimizer_config["batch_rays"]),
            learning_rate=float(optimizer_config["learning_rate"]),
            weight_decay=float(weight_decay),
            seed=int(optimizer_config["seed"]) + index,
            device=device,
            gradient_clip_norm=optimizer_config.get("gradient_clip_norm"),
            cosine_eta_fraction=optimizer_config.get("cosine_eta_fraction"),
        )
        prediction = predict_ray_residuals(
            selection_features,
            selection_basis,
            coefficients,
            device=device,
            batch_rays=int(optimizer_config["batch_rays"]),
        )
        error = float(
            np.linalg.norm(prediction - selection_targets)
            / max(np.linalg.norm(selection_targets), 1e-15)
        )
        record = {
            "weight_decay": float(weight_decay),
            "selection_relative_aperture_error": error,
            "final_batch_data_loss": history[-1]["batch_data_loss"],
            "final_total_loss": history[-1]["total_loss"],
        }
        selection_rows.append(record)
        if best_selection is None or error < best_selection["error"]:
            best_selection = {
                "weight_decay": float(weight_decay),
                "error": error,
                "history": history,
            }
    assert best_selection is not None

    all_train_features, all_center, all_scale = ray_feature_matrix(
        train_vectors, views, depth, detector
    )
    development_features, _, _ = ray_feature_matrix(
        development_vectors,
        views,
        depth,
        detector,
        center=all_center,
        scale=all_scale,
    )
    coefficients, refit_history = train_direct_model(
        all_train_features,
        train_basis,
        train_targets,
        steps=int(optimizer_config["steps"]),
        batch_rays=int(optimizer_config["batch_rays"]),
        learning_rate=float(optimizer_config["learning_rate"]),
        weight_decay=float(best_selection["weight_decay"]),
        seed=int(optimizer_config["seed"]) + 91,
        device=device,
        gradient_clip_norm=optimizer_config.get("gradient_clip_norm"),
        cosine_eta_fraction=optimizer_config.get("cosine_eta_fraction"),
    )
    predicted_residual_rows = predict_ray_residuals(
        development_features,
        development_basis,
        coefficients,
        device=device,
        batch_rays=int(optimizer_config["batch_rays"]),
    )
    direct_operators = development_thin + predicted_residual_rows.reshape(
        development_finite.shape
    )

    train_design, rig_center, rig_scale = _rig_feature_matrix(train_vectors, views)
    development_design, _, _ = _rig_feature_matrix(
        development_vectors, views, center=rig_center, scale=rig_scale
    )
    train_residual = train_finite - train_thin
    full_ridge_best: dict[str, Any] | None = None
    for ridge in diagnosis_config["full_matrix_ridge_grid"]:
        full_coefficients = fit_ridge(
            train_design,
            train_residual.reshape(len(train_residual), -1),
            float(ridge),
        )
        operators = development_thin + (
            development_design @ full_coefficients
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
                "coefficients": full_coefficients,
            }
    assert full_ridge_best is not None
    methods = {
        "thin_ray_high_fidelity": development_thin,
        "full_matrix_geometry_ridge": full_ridge_best["operators"],
        "direct_ray_conditioned_kernel": direct_operators,
    }

    probe_rng = np.random.default_rng(int(source_config["seed"]) + 6991)
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

    candidate = summary["direct_ray_conditioned_kernel"]
    baseline = summary["full_matrix_geometry_ridge"]
    improvement = 1.0 - candidate[
        "mean_relative_aperture_residual_error"
    ] / max(baseline["mean_relative_aperture_residual_error"], 1e-15)
    worst_ratio = candidate["worst_relative_aperture_residual_error"] / max(
        baseline["worst_relative_aperture_residual_error"], 1e-15
    )
    rule = diagnosis_config["development_reference_rule"]
    v5x_oracle_error = float(
        v5x_report["best_oracle"]["mean_relative_aperture_error"]
    )
    reference_pass = (
        v5x_oracle_error
        <= float(rule["maximum_v5x_oracle_relative_aperture_error"])
        and improvement
        >= float(rule["minimum_prediction_improvement_over_full_matrix_ridge"])
        and worst_ratio
        <= float(rule["maximum_worst_rig_ratio_to_full_matrix_ridge"])
    )

    dot_rng = np.random.default_rng(int(source_config["seed"]) + 7881)
    field = dot_rng.normal(size=direct_operators.shape[-1])
    measurement = dot_rng.normal(size=direct_operators.shape[1])
    dot_left = float((direct_operators[0] @ field) @ measurement)
    dot_right = float(field @ (direct_operators[0].T @ measurement))
    dot_defect = abs(dot_left - dot_right) / max(
        abs(dot_left), abs(dot_right), 1e-15
    )

    history_rows = [
        {
            "phase": "refit_all_30",
            "weight_decay": best_selection["weight_decay"],
            **row,
        }
        for row in refit_history
    ]
    model_path = output_dir / "direct_ray_kernel_model.npz"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_path,
        coefficients=coefficients.astype(np.float32),
        feature_center=all_center.astype(np.float32),
        feature_scale=all_scale.astype(np.float32),
        offsets=np.asarray(offsets, dtype=np.int32),
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
            "entrypoint": _file_sha256(
                (entrypoint_path or Path(__file__)).resolve()
            ),
            "diagnosis_config": _file_sha256(config_path),
            "source_config": _file_sha256(source_config_path),
            "torch_version": torch.__version__,
            "direct_dependency_sha256": relative_file_hashes(
                ROOT,
                [
                    Path(__file__),
                    entrypoint_path or Path(__file__),
                    config_path,
                    source_config_path,
                    V5S_REPORT,
                    V5X_REPORT,
                    ROOT / "release_provenance.py",
                    ROOT / "gc_rio" / "protocol.py",
                    ROOT / "independent_reaction_bost.py",
                    ROOT / "run_v5s_dco_low_rank_screening.py",
                    ROOT / "run_v5t_camera_local_tangent_diagnosis.py",
                    ROOT / "run_v5u_calibrated_renderer_residual_screening.py",
                    ROOT / "run_v5w_clean_aperture_kernel_screening.py",
                    ROOT / "run_v5x_ray_conditioned_voxel_kernel.py",
                ],
            ),
            "runtime_environment": runtime_environment(
                device=str(device), torch_version=torch.__version__
            ),
            "determinism_boundary": (
                "The stored MPS reference is artifact- and tolerance-validated; "
                "bitwise equality across devices or Torch builds is not claimed."
            ),
        },
        "training_protocol": {
            "device": str(device),
            "internal_fit_rigs": internal_fit_rigs,
            "internal_selection_rigs": len(train_vectors) - internal_fit_rigs,
            "selected_weight_decay": best_selection["weight_decay"],
            "selection_relative_aperture_error": best_selection["error"],
            "refit_rigs": len(train_vectors),
            "complete_operator_rows_used": True,
            "truth_calibrated_geometry_used": True,
        },
        "model_size": {
            "ray_feature_dimension": int(coefficients.shape[0]),
            "kernel_coefficients_per_ray": int(coefficients.shape[1]),
            "direct_model_coefficients": int(coefficients.size),
            "full_matrix_predictor_coefficients": int(
                full_ridge_best["coefficients"].size
            ),
            "compression_ratio_vs_full_matrix_predictor": float(
                full_ridge_best["coefficients"].size / coefficients.size
            ),
        },
        "development_summary": summary,
        "v5x_oracle_mean_relative_aperture_error": v5x_oracle_error,
        "prediction_improvement_over_full_matrix_ridge": improvement,
        "prediction_worst_rig_ratio_to_full_matrix_ridge": worst_ratio,
        "materialized_operator_dot_product_relative_defect": dot_defect,
        "development_reference_rule": rule,
        "decision": (
            "DIRECT_RAY_KERNEL_DEVELOPMENT_SIGNAL_PREREGISTER_FRESH"
            if reference_pass
            else "DIRECT_RAY_KERNEL_DEVELOPMENT_NO_GO"
        ),
        "sample_accounting": {
            "train_rigs": len(train_vectors),
            "development_rigs": len(development_vectors),
            "design_lock_rigs": 0,
            "rays_per_rig": rays_per_rig,
            "train_operator_rows": len(train_basis),
            "development_operator_rows": len(development_basis),
            "probe_fields": len(probe_fields),
        },
        "limitations": [
            "The direct objective was designed after opening v5x.",
            "All development rigs were previously opened in v5s-v5x.",
            "Truth geometry and complete high-fidelity operator rows are used.",
            "The finite-aperture renderer is a prescribed linear weak-BOST surrogate.",
            "MPS execution is not claimed bitwise deterministic.",
            "No limited calibration probes, fresh rigs, inverse reconstruction, nonlinear field-dependent rays, or real data are tested.",
        ],
    }
    _write_csv(output_dir / "selection_sweep.csv", selection_rows)
    _write_csv(output_dir / "training_history.csv", history_rows)
    _write_csv(output_dir / "development_rig_metrics.csv", metric_rows)
    _write_json(output_dir / "report.json", report)
    _write_checksums(
        output_dir,
        [
            "selection_sweep.csv",
            "training_history.csv",
            "development_rig_metrics.csv",
            "direct_ray_kernel_model.npz",
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
                "training": report["training_protocol"],
                "model_size": report["model_size"],
                "improvement": report[
                    "prediction_improvement_over_full_matrix_ridge"
                ],
                "worst_ratio": report[
                    "prediction_worst_rig_ratio_to_full_matrix_ridge"
                ],
                "dot_defect": report[
                    "materialized_operator_dot_product_relative_defect"
                ],
                "development_summary": report["development_summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
