#!/usr/bin/env python3
"""Develop a compact nonlinear ray-kernel hypernetwork on opened rigs."""

from __future__ import annotations

import copy
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
        _thin_high_fidelity_operator,
        voxel_kernel_offsets,
    )
    from demo_t16_operator.run_v5x_ray_conditioned_voxel_kernel import (
        ray_feature_matrix,
    )
    from demo_t16_operator.run_v5y_direct_ray_conditioned_kernel import (
        _device_from_config,
        build_ray_basis_and_targets,
        predict_ray_residuals,
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
        _thin_high_fidelity_operator,
        voxel_kernel_offsets,
    )
    from .run_v5x_ray_conditioned_voxel_kernel import ray_feature_matrix
    from .run_v5y_direct_ray_conditioned_kernel import (
        _device_from_config,
        build_ray_basis_and_targets,
        predict_ray_residuals,
    )


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "v6a_ray_kernel_hypernetwork_development.json"
OUTPUT_DIR = ROOT / "results" / "v6a_ray_kernel_hypernetwork_development"
V5S_REPORT = ROOT / "results" / "v5s_dco_low_rank_screening" / "report.json"
V5X_REPORT = ROOT / "results" / "v5x_ray_conditioned_voxel_kernel" / "report.json"
V5Z_REPORT = ROOT / "results" / "v5z_stabilized_direct_ray_kernel" / "report.json"
V5Z_MODEL = (
    ROOT
    / "results"
    / "v5z_stabilized_direct_ray_kernel"
    / "direct_ray_kernel_model.npz"
)


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


class RayKernelHypernetwork(torch.nn.Module):
    def __init__(
        self, feature_dimension: int, hidden_width: int, hidden_layers: int, kernels: int
    ) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        current = int(feature_dimension)
        for _ in range(int(hidden_layers)):
            layers.extend(
                [torch.nn.Linear(current, int(hidden_width)), torch.nn.SiLU()]
            )
            current = int(hidden_width)
        output = torch.nn.Linear(current, int(kernels))
        torch.nn.init.zeros_(output.weight)
        torch.nn.init.zeros_(output.bias)
        layers.append(output)
        self.network = torch.nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def predict_hypernetwork_residuals(
    model: RayKernelHypernetwork,
    features: np.ndarray,
    basis: np.ndarray,
    *,
    device: torch.device,
    batch_rays: int,
) -> np.ndarray:
    feature_values = np.asarray(features, dtype=np.float32)
    basis_values = np.asarray(basis, dtype=np.float32)
    output = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(feature_values), int(batch_rays)):
            stop = min(start + int(batch_rays), len(feature_values))
            feature_batch = torch.as_tensor(feature_values[start:stop], device=device)
            basis_batch = torch.as_tensor(basis_values[start:stop], device=device)
            kernels = model(feature_batch)
            prediction = torch.einsum("bpk,bk->bp", basis_batch, kernels)
            output.append(prediction.cpu().numpy())
    return np.concatenate(output, axis=0).astype(np.float64)


def train_hypernetwork(
    fit_features: np.ndarray,
    fit_basis: np.ndarray,
    fit_targets: np.ndarray,
    *,
    selection_features: np.ndarray | None,
    selection_basis: np.ndarray | None,
    selection_targets: np.ndarray | None,
    hidden_width: int,
    hidden_layers: int,
    maximum_steps: int,
    selection_interval: int,
    batch_rays: int,
    learning_rate: float,
    gradient_clip_norm: float,
    cosine_eta_fraction: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> tuple[RayKernelHypernetwork, int, list[dict[str, float]]]:
    features = np.asarray(fit_features, dtype=np.float32)
    basis = np.asarray(fit_basis, dtype=np.float32)
    targets = np.asarray(fit_targets, dtype=np.float32)
    torch.manual_seed(int(seed))
    model = RayKernelHypernetwork(
        features.shape[1], hidden_width, hidden_layers, basis.shape[2]
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(maximum_steps),
        eta_min=float(learning_rate) * float(cosine_eta_fraction),
    )
    rng = np.random.default_rng(int(seed))
    residual_scale = max(float(np.sqrt(np.mean(np.square(targets)))), 1e-8)
    best_state = copy.deepcopy(model.state_dict())
    best_step = int(maximum_steps)
    best_selection = float("inf")
    history = []
    for step in range(1, int(maximum_steps) + 1):
        indices = rng.integers(0, len(features), size=int(batch_rays))
        feature_batch = torch.as_tensor(features[indices], device=device)
        basis_batch = torch.as_tensor(basis[indices], device=device)
        target_batch = torch.as_tensor(targets[indices], device=device)
        kernels = model(feature_batch)
        prediction = torch.einsum("bpk,bk->bp", basis_batch, kernels)
        data_loss = torch.mean(torch.square((prediction - target_batch) / residual_scale))
        regularizer = float(weight_decay) * sum(
            torch.mean(torch.square(parameter)) for parameter in model.parameters()
        )
        loss = data_loss + regularizer
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip_norm))
        optimizer.step()
        scheduler.step()
        if step == 1 or step % int(selection_interval) == 0 or step == int(maximum_steps):
            selection_error = float("nan")
            if (
                selection_features is not None
                and selection_basis is not None
                and selection_targets is not None
            ):
                selection_prediction = predict_hypernetwork_residuals(
                    model,
                    selection_features,
                    selection_basis,
                    device=device,
                    batch_rays=batch_rays,
                )
                selection_error = float(
                    np.linalg.norm(selection_prediction - selection_targets)
                    / max(np.linalg.norm(selection_targets), 1e-15)
                )
                if selection_error < best_selection:
                    best_selection = selection_error
                    best_step = step
                    best_state = copy.deepcopy(model.state_dict())
            history.append(
                {
                    "step": float(step),
                    "batch_data_loss": float(data_loss.detach().cpu()),
                    "regularizer": float(regularizer.detach().cpu()),
                    "total_loss": float(loss.detach().cpu()),
                    "selection_relative_aperture_error": selection_error,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )
    if selection_features is not None:
        model.load_state_dict(best_state)
    return model, best_step, history


def _relative_aperture_errors(
    prediction: np.ndarray, finite: np.ndarray, thin: np.ndarray
) -> np.ndarray:
    return np.linalg.norm(prediction - finite, axis=(1, 2)) / np.maximum(
        np.linalg.norm(finite - thin, axis=(1, 2)), 1e-15
    )


def run() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    source_config_path = ROOT / str(config["source_config"])
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    if not bool(config["design_lock_construction_forbidden"]):
        raise ValueError("v6a must forbid design-lock construction")
    v5s_report = json.loads(V5S_REPORT.read_text(encoding="utf-8"))
    v5x_report = json.loads(V5X_REPORT.read_text(encoding="utf-8"))
    v5z_report = json.loads(V5Z_REPORT.read_text(encoding="utf-8"))
    manifest = build_rig_manifest(source_config)
    if sha256_json(manifest) != v5s_report["manifest_sha256_before_operator_build"]:
        raise RuntimeError("v5s manifest reproduction failed")
    views = int(source_config["views"])
    depth = int(source_config["depth"])
    detector = int(source_config["grid_size"])
    rays_per_rig = views * depth * detector
    offsets = voxel_kernel_offsets(int(config["voxel_kernel_radius"]))

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
    fit_rigs = int(config["internal_fit_rigs"])
    train_basis, train_targets = build_ray_basis_and_targets(
        train_thin, train_finite, views, depth, detector, offsets
    )
    development_basis, _ = build_ray_basis_and_targets(
        development_thin,
        development_finite,
        views,
        depth,
        detector,
        offsets,
    )
    fit_vectors = train_vectors[:fit_rigs]
    selection_vectors = train_vectors[fit_rigs:]
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
    fit_stop = fit_rigs * rays_per_rig
    fit_basis = train_basis[:fit_stop]
    fit_targets = train_targets[:fit_stop]
    selection_basis = train_basis[fit_stop:]
    selection_targets = train_targets[fit_stop:]
    optimizer_config = config["optimizer"]
    device = _device_from_config(str(optimizer_config["device"]))

    selection_histories = []
    best_steps = []
    for seed in optimizer_config["seeds"]:
        _, best_step, history = train_hypernetwork(
            fit_features,
            fit_basis,
            fit_targets,
            selection_features=selection_features,
            selection_basis=selection_basis,
            selection_targets=selection_targets,
            hidden_width=int(config["hidden_width"]),
            hidden_layers=int(config["hidden_layers"]),
            maximum_steps=int(optimizer_config["maximum_steps"]),
            selection_interval=int(optimizer_config["selection_interval"]),
            batch_rays=int(optimizer_config["batch_rays"]),
            learning_rate=float(optimizer_config["learning_rate"]),
            gradient_clip_norm=float(optimizer_config["gradient_clip_norm"]),
            cosine_eta_fraction=float(optimizer_config["cosine_eta_fraction"]),
            weight_decay=float(optimizer_config["weight_decay"]),
            seed=int(seed),
            device=device,
        )
        best_steps.append(best_step)
        selection_histories.extend(
            [{"phase": "internal_selection", "seed": int(seed), **row} for row in history]
        )
    refit_steps = int(np.median(best_steps))
    all_train_features, center, scale = ray_feature_matrix(
        train_vectors, views, depth, detector
    )
    development_features, _, _ = ray_feature_matrix(
        development_vectors,
        views,
        depth,
        detector,
        center=center,
        scale=scale,
    )
    seed_predictions = []
    seed_models = []
    refit_histories = []
    for seed in optimizer_config["seeds"]:
        model, _, history = train_hypernetwork(
            all_train_features,
            train_basis,
            train_targets,
            selection_features=None,
            selection_basis=None,
            selection_targets=None,
            hidden_width=int(config["hidden_width"]),
            hidden_layers=int(config["hidden_layers"]),
            maximum_steps=refit_steps,
            selection_interval=int(optimizer_config["selection_interval"]),
            batch_rays=int(optimizer_config["batch_rays"]),
            learning_rate=float(optimizer_config["learning_rate"]),
            gradient_clip_norm=float(optimizer_config["gradient_clip_norm"]),
            cosine_eta_fraction=float(optimizer_config["cosine_eta_fraction"]),
            weight_decay=float(optimizer_config["weight_decay"]),
            seed=int(seed) + 1000,
            device=device,
        )
        prediction = predict_hypernetwork_residuals(
            model,
            development_features,
            development_basis,
            device=device,
            batch_rays=int(optimizer_config["batch_rays"]),
        )
        seed_predictions.append(prediction.reshape(development_finite.shape))
        seed_models.append(model)
        refit_histories.extend(
            [{"phase": "refit_all_30", "seed": int(seed), **row} for row in history]
        )
    hyper_operators_by_seed = [
        development_thin + prediction for prediction in seed_predictions
    ]
    ensemble_operators = development_thin + np.mean(seed_predictions, axis=0)

    stable_linear_model = np.load(V5Z_MODEL)
    linear_coefficients = np.asarray(
        stable_linear_model["coefficients"], dtype=np.float64
    )
    linear_residual = predict_ray_residuals(
        development_features,
        development_basis,
        linear_coefficients,
        device=device,
        batch_rays=int(optimizer_config["batch_rays"]),
    ).reshape(development_finite.shape)
    linear_operators = development_thin + linear_residual

    train_design, rig_center, rig_scale = _rig_feature_matrix(train_vectors, views)
    development_design, _, _ = _rig_feature_matrix(
        development_vectors, views, center=rig_center, scale=rig_scale
    )
    train_residual = train_finite - train_thin
    full_ridge_best: dict[str, Any] | None = None
    for ridge in config["full_matrix_ridge_grid"]:
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
        "stable_linear_ray_kernel": linear_operators,
        "hypernetwork_ensemble": ensemble_operators,
    }
    for seed, operators in zip(
        optimizer_config["seeds"], hyper_operators_by_seed, strict=True
    ):
        methods[f"hypernetwork_seed_{seed}"] = operators

    probe_rng = np.random.default_rng(int(source_config["seed"]) + 7991)
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

    candidate = summary["hypernetwork_ensemble"]
    baseline = summary["full_matrix_geometry_ridge"]
    improvement = 1.0 - candidate[
        "mean_relative_aperture_residual_error"
    ] / max(baseline["mean_relative_aperture_residual_error"], 1e-15)
    worst_ratio = candidate["worst_relative_aperture_residual_error"] / max(
        baseline["worst_relative_aperture_residual_error"], 1e-15
    )
    baseline_by_rig = {
        row["rig_id"]: row["relative_aperture_residual_error"]
        for row in metric_rows
        if row["method"] == "full_matrix_geometry_ridge"
    }
    candidate_by_rig = {
        row["rig_id"]: row["relative_aperture_residual_error"]
        for row in metric_rows
        if row["method"] == "hypernetwork_ensemble"
    }
    positive_rig_fraction = float(
        np.mean(
            [
                candidate_by_rig[rig] < baseline_by_rig[rig]
                for rig in sorted(baseline_by_rig)
            ]
        )
    )
    paired_rig_relative_changes = [
        candidate_by_rig[rig] / max(baseline_by_rig[rig], 1e-15) - 1.0
        for rig in sorted(baseline_by_rig)
    ]
    maximum_paired_rig_degradation = float(max(paired_rig_relative_changes))
    paired_rig_harm_fraction = float(
        np.mean([change > 0.0 for change in paired_rig_relative_changes])
    )
    v5x_oracle = float(v5x_report["best_oracle"]["mean_relative_aperture_error"])
    rule = config["development_reference_rule"]
    reference_pass = (
        v5x_oracle
        <= float(rule["maximum_v5x_oracle_relative_aperture_error"])
        and improvement
        >= float(rule["minimum_ensemble_improvement_over_full_matrix_ridge"])
        and worst_ratio
        <= float(rule["maximum_ensemble_worst_rig_ratio_to_full_matrix_ridge"])
        and positive_rig_fraction
        >= float(rule["minimum_positive_rig_fraction_vs_full_matrix_ridge"])
    )

    state_arrays = {}
    for model_index, model in enumerate(seed_models):
        for name, tensor in model.state_dict().items():
            state_arrays[f"model_{model_index}_{name.replace('.', '__')}"] = (
                tensor.detach().cpu().numpy()
            )
    model_path = OUTPUT_DIR / "ray_kernel_hypernetwork_models.npz"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_path,
        **state_arrays,
        feature_center=center.astype(np.float32),
        feature_scale=scale.astype(np.float32),
        offsets=np.asarray(offsets, dtype=np.int32),
        seeds=np.asarray(optimizer_config["seeds"], dtype=np.int64),
    )
    parameter_count = sum(parameter.numel() for parameter in seed_models[0].parameters())
    report = {
        "schema": config["schema"],
        "evidence_label": config["evidence_label"],
        "claim_ceiling": config["claim_ceiling"],
        "config_sha256": sha256_json(config),
        "source_config_sha256": sha256_json(source_config),
        "reproduced_v5s_manifest_sha256": sha256_json(manifest),
        "source_provenance": {
            "runner": _file_sha256(Path(__file__).resolve()),
            "config": _file_sha256(CONFIG_PATH),
            "source_config": _file_sha256(source_config_path),
            "torch_version": torch.__version__,
            "direct_dependency_sha256": relative_file_hashes(
                ROOT,
                [
                    Path(__file__),
                    CONFIG_PATH,
                    source_config_path,
                    V5S_REPORT,
                    V5X_REPORT,
                    V5Z_REPORT,
                    V5Z_MODEL,
                    ROOT / "release_provenance.py",
                    ROOT / "gc_rio" / "protocol.py",
                    ROOT / "independent_reaction_bost.py",
                    ROOT / "run_v5s_dco_low_rank_screening.py",
                    ROOT / "run_v5t_camera_local_tangent_diagnosis.py",
                    ROOT / "run_v5u_calibrated_renderer_residual_screening.py",
                    ROOT / "run_v5w_clean_aperture_kernel_screening.py",
                    ROOT / "run_v5x_ray_conditioned_voxel_kernel.py",
                    ROOT / "run_v5y_direct_ray_conditioned_kernel.py",
                    ROOT / "run_v5z_stabilized_direct_ray_kernel.py",
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
            "seeds": optimizer_config["seeds"],
            "internal_fit_rigs": fit_rigs,
            "internal_selection_rigs": len(train_vectors) - fit_rigs,
            "best_steps_by_seed": best_steps,
            "fixed_refit_steps": refit_steps,
            "refit_rigs": len(train_vectors),
            "complete_operator_rows_used": True,
            "truth_calibrated_geometry_used": True,
        },
        "model_size": {
            "parameters_per_seed": int(parameter_count),
            "ensemble_parameters": int(parameter_count * len(seed_models)),
            "full_matrix_predictor_coefficients": int(
                full_ridge_best["coefficients"].size
            ),
            "single_model_compression_vs_full_matrix": float(
                full_ridge_best["coefficients"].size / parameter_count
            ),
        },
        "development_summary": summary,
        "stable_linear_reference_from_v5z": v5z_report[
            "development_summary"
        ]["direct_ray_conditioned_kernel"],
        "v5x_oracle_mean_relative_aperture_error": v5x_oracle,
        "ensemble_improvement_over_full_matrix_ridge": improvement,
        "ensemble_worst_rig_ratio_to_full_matrix_ridge": worst_ratio,
        "ensemble_positive_rig_fraction_vs_full_matrix_ridge": positive_rig_fraction,
        "ensemble_maximum_paired_rig_degradation_vs_full_matrix_ridge": (
            maximum_paired_rig_degradation
        ),
        "ensemble_paired_rig_harm_fraction_vs_full_matrix_ridge": (
            paired_rig_harm_fraction
        ),
        "development_reference_rule": rule,
        "decision": (
            "RAY_KERNEL_HYPERNET_DEVELOPMENT_SIGNAL_PREREGISTER_FRESH"
            if reference_pass
            else "RAY_KERNEL_HYPERNET_DEVELOPMENT_NO_GO_STOP_CAPACITY_ESCALATION"
        ),
        "sample_accounting": {
            "train_rigs": len(train_vectors),
            "development_rigs": len(development_vectors),
            "design_lock_rigs": 0,
            "rays_per_rig": rays_per_rig,
            "probe_fields": len(probe_fields),
        },
        "limitations": [
            "The nonlinear hypernetwork was designed after opening v5z.",
            "All development rigs were already opened in v5s-v5z.",
            "Truth geometry and complete high-fidelity operator rows are used.",
            "MPS execution is not claimed bitwise deterministic.",
            "No limited probes, fresh rigs, inverse reconstruction, nonlinear field-dependent rays, or real data are tested.",
        ],
    }
    _write_csv(
        OUTPUT_DIR / "training_history.csv", selection_histories + refit_histories
    )
    _write_csv(OUTPUT_DIR / "development_rig_metrics.csv", metric_rows)
    _write_json(OUTPUT_DIR / "report.json", report)
    _write_checksums(
        OUTPUT_DIR,
        [
            "training_history.csv",
            "development_rig_metrics.csv",
            "ray_kernel_hypernetwork_models.npz",
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
                    "ensemble_improvement_over_full_matrix_ridge"
                ],
                "worst_ratio": report[
                    "ensemble_worst_rig_ratio_to_full_matrix_ridge"
                ],
                "positive_rig_fraction": report[
                    "ensemble_positive_rig_fraction_vs_full_matrix_ridge"
                ],
                "development_summary": report["development_summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
