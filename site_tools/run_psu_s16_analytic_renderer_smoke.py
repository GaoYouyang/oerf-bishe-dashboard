#!/usr/bin/env python3
"""Run an independent-renderer synthetic smoke on real PSU support geometry."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
from pathlib import Path
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

import demo_t16_operator.analytic_bost_phantoms as analytic_module
from demo_t16_operator.analytic_bost_phantoms import (
    ANALYTIC_PHANTOM_SCHEMA,
    ANALYTIC_RENDERER_SCHEMA,
    analytic_phantom_grid,
    make_analytic_phantom,
    render_analytic_bost,
)
from demo_t16_operator.psu_b0_spectral_preconditioner import (
    weighted_cgls_reconstruction,
)
from demo_t16_operator.psu_b0_streaming_operator import zero_outer_boundary_support
from demo_t16_operator.spatial_reconstruction_metrics import (
    interface_surface_from_level_set,
    normal_angle_metrics,
    scalar_grid_gradient,
    surface_distance_metrics,
    synthetic_field_metrics,
)
from site_tools.run_psu_b0_real_interface_audit import (
    _make_operator,
    load_real_support_geometry,
)


REPORT_SCHEMA = "psu-s16-analytic-renderer-smoke-report-1.0"
PUBLIC_STATUS = "E1_SYNTHETIC_INTERFACE_SMOKE_NOT_SUPERIORITY_EVIDENCE"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_l2(value: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(reference).clamp_min(1e-30)
    return float(torch.linalg.vector_norm(value - reference) / denominator)


def _renderer_independence_audit() -> dict[str, Any]:
    source = inspect.getsource(analytic_module)
    forbidden = ("PSUB0VoxelGradientOperator", "finite_difference_gradient")
    present = [token for token in forbidden if token in source]
    return {
        "forbidden_inverse_chain_tokens": list(forbidden),
        "present_tokens": present,
        "passed": not present,
    }


def _render_batch(
    specs,
    geometry: dict[str, np.ndarray],
) -> torch.Tensor:
    return torch.stack(
        [
            render_analytic_bost(
                spec,
                sample_points_xyz=geometry["sample_points"],
                projection_u_xyz=geometry["projection_u"],
                projection_v_xyz=geometry["projection_v"],
                line_length=geometry["line_length"],
                system_constant=geometry["system_constant"],
            )
            for spec in specs
        ],
        dim=0,
    )


def _noisy_observations(
    clean: torch.Tensor,
    *,
    view_count: int,
    rays_per_view: int,
    relative_noise: float,
    seeds: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(clean) != len(seeds):
        raise ValueError("one noise seed is required per phantom")
    view_values = clean.reshape(len(clean), view_count, rays_per_view, 2)
    view_rms = torch.sqrt(torch.mean(view_values.square(), dim=(2, 3)))
    global_floor = torch.sqrt(torch.mean(clean.square(), dim=(1, 2))).clamp_min(1e-30)
    sigma = torch.maximum(
        float(relative_noise) * view_rms,
        (1e-6 * global_floor)[:, None],
    )
    rows = []
    for index, seed in enumerate(seeds):
        generator = torch.Generator().manual_seed(int(seed))
        noise = torch.randn(
            clean[index].shape,
            generator=generator,
            dtype=clean.dtype,
        )
        expanded = sigma[index].repeat_interleave(rays_per_view)[:, None]
        rows.append(clean[index] + noise * expanded)
    return torch.stack(rows), sigma


def _front_metrics(
    *,
    prediction: np.ndarray,
    predicted_gradient: np.ndarray,
    truth_gradient: np.ndarray,
    truth_level_sets: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    tolerance_voxels: list[float],
    band_voxels: float,
) -> tuple[dict[str, float | None], str]:
    if truth_level_sets.shape[-1] != 1:
        return {
            "surface_assd": None,
            "surface_hd95": None,
            "surface_f1_at_1_voxel": None,
            "surface_f1_at_2_voxels": None,
            "normal_angle_median_degrees": None,
            "normal_angle_p95_degrees": None,
        }, "NOT_APPLICABLE_REQUIRES_EXACTLY_ONE_DECLARED_INTERFACE"
    truth_level = truth_level_sets[..., 0]
    predicted_surface = interface_surface_from_level_set(prediction)
    truth_surface = interface_surface_from_level_set(truth_level)
    dx = float(spacing_xyz[0])
    tolerances = tuple(float(value) * dx for value in tolerance_voxels)
    try:
        distances = surface_distance_metrics(
            predicted_surface,
            truth_surface,
            spacing_xyz=spacing_xyz,
            tolerance_distances=tolerances,
        )
    except ValueError as exc:
        return {
            "surface_assd": None,
            "surface_hd95": None,
            "surface_f1_at_1_voxel": None,
            "surface_f1_at_2_voxels": None,
            "normal_angle_median_degrees": None,
            "normal_angle_p95_degrees": None,
        }, f"UNAVAILABLE_{str(exc).upper().replace(' ', '_')}"
    normalized_spacing = 2.0 / (prediction.shape[-1] - 1)
    band = np.abs(truth_level) <= float(band_voxels) * normalized_spacing
    angles = normal_angle_metrics(
        predicted_gradient,
        truth_gradient,
        evaluation_mask=band,
    )
    labels = [format(value, ".12g").replace(".", "p") for value in tolerances]
    return {
        "surface_assd": distances["surface_assd"],
        "surface_hd95": distances["surface_hd95"],
        "surface_f1_at_1_voxel": distances[f"surface_f1_at_{labels[0]}"],
        "surface_f1_at_2_voxels": distances[f"surface_f1_at_{labels[1]}"],
        "normal_angle_median_degrees": angles["normal_angle_median_degrees"],
        "normal_angle_p95_degrees": angles["normal_angle_p95_degrees"],
    }, "AVAILABLE_SYNTHETIC_SINGLE_INTERFACE_ONLY"


def _plot(
    *,
    truths: list[np.ndarray],
    predictions: list[np.ndarray],
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        len(rows),
        4,
        figsize=(13.6, 3.1 * len(rows)),
        constrained_layout=True,
    )
    for index, row in enumerate(rows):
        truth = truths[index]
        prediction = predictions[index]
        error = prediction - truth
        center = truth.shape[0] // 2
        limit = max(float(np.max(np.abs(truth))), float(np.max(np.abs(prediction))), 1e-12)
        error_limit = max(float(np.max(np.abs(error))), 1e-12)
        axes[index, 0].imshow(
            truth[center], cmap="RdBu_r", vmin=-limit, vmax=limit, origin="lower"
        )
        axes[index, 1].imshow(
            prediction[center], cmap="RdBu_r", vmin=-limit, vmax=limit, origin="lower"
        )
        axes[index, 2].imshow(
            error[center],
            cmap="coolwarm",
            vmin=-error_limit,
            vmax=error_limit,
            origin="lower",
        )
        truth_gradient = np.gradient(truth)
        prediction_gradient = np.gradient(prediction)
        truth_magnitude = np.sqrt(sum(component**2 for component in truth_gradient))
        prediction_magnitude = np.sqrt(
            sum(component**2 for component in prediction_gradient)
        )
        axes[index, 3].plot(
            truth_magnitude[center, center],
            label="truth grid profile",
            color="#1b4965",
            linewidth=2.0,
        )
        axes[index, 3].plot(
            prediction_magnitude[center, center],
            label="CGLS",
            color="#d1495b",
            linewidth=1.8,
        )
        axes[index, 3].grid(alpha=0.25)
        axes[index, 3].set_ylim(bottom=0.0)
        if index == 0:
            axes[index, 3].legend(frameon=False, fontsize=8)
        axes[index, 0].set_ylabel(row["family"].replace("_", "\n"), fontsize=9)
        axes[index, 1].set_title(
            f"CGLS-12 | field L2 {row['field_relative_l2']:.3f}\n"
            f"H1 {row['h1_seminorm_relative_error']:.3f}",
            fontsize=9,
        )
        axes[index, 2].set_title(
            f"error | reprojection {row['support_reprojection_relative_l2']:.3f}",
            fontsize=9,
        )
        for column in range(3):
            axes[index, column].set_xticks([])
            axes[index, column].set_yticks([])
    axes[0, 0].set_title("analytic truth slice", fontsize=10)
    axes[0, 2].set_title("signed error slice", fontsize=10)
    axes[0, 3].set_title("center-line gradient magnitude", fontsize=10)
    figure.suptitle(
        "PSU-S16 independent analytic renderer smoke\n"
        "real nine-view geometry, synthetic morphology only, no superiority claim",
        fontsize=13,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    figure.savefig(output_path.with_suffix(".pdf"))
    plt.close(figure)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_checksums(output_dir: Path) -> None:
    paths = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "checksums.sha256"
    )
    lines = [f"{_sha256(path)}  {path.name}" for path in paths]
    (output_dir / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="ascii")


def run_smoke(
    *,
    config_path: Path,
    view_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = _load_json(config_path)
    geometry_config = config["geometry"]
    grid_size = int(geometry_config["grid_size"])
    view_count = int(geometry_config["view_count"])
    rays_per_view = int(geometry_config["rays_per_view"])
    bounds = geometry_config["bounds_m"]
    minimum = tuple(float(value) for value in bounds["minimum_xyz"])
    maximum = tuple(float(value) for value in bounds["maximum_xyz"])
    truth_geometry, _ = load_real_support_geometry(
        view_root,
        rays_per_view=rays_per_view,
        sample_count=int(geometry_config["truth_renderer_qmc_samples"]),
        lower=minimum,
        upper=maximum,
    )
    audit_geometry, _ = load_real_support_geometry(
        view_root,
        rays_per_view=rays_per_view,
        sample_count=int(geometry_config["renderer_audit_qmc_samples"]),
        lower=minimum,
        upper=maximum,
    )
    inverse_geometry, _ = load_real_support_geometry(
        view_root,
        rays_per_view=rays_per_view,
        sample_count=int(geometry_config["inverse_operator_qmc_samples"]),
        lower=minimum,
        upper=maximum,
    )
    specs = [
        make_analytic_phantom(
            family=row["family"],
            seed=int(row["seed"]),
            domain_minimum_xyz=minimum,
            domain_maximum_xyz=maximum,
        )
        for row in config["phantoms"]
    ]
    clean_truth = _render_batch(specs, truth_geometry).to(torch.float64)
    clean_audit = _render_batch(specs, audit_geometry).to(torch.float64)
    qmc_discrepancy = [
        _relative_l2(clean_audit[index], clean_truth[index])
        for index in range(len(specs))
    ]
    observations, sigma = _noisy_observations(
        clean_truth,
        view_count=view_count,
        rays_per_view=rays_per_view,
        relative_noise=float(config["noise"]["relative_to_clean_view_rms"]),
        seeds=[int(value) for value in config["noise"]["seeds"]],
    )

    operator = _make_operator(
        inverse_geometry,
        grid_size=grid_size,
        dtype=torch.float32,
    )
    support = zero_outer_boundary_support(
        (grid_size,) * 3,
        dtype=torch.float32,
    )
    operator.support.copy_(support)
    adjoint_relative_error = operator.adjoint_relative_error(seed=1709)
    operator.reset_call_counts()
    solver_started = time.perf_counter()
    result = weighted_cgls_reconstruction(
        operator,
        observations.to(torch.float32),
        sigma_by_view=sigma.to(torch.float32),
        view_mask=torch.ones((len(specs), view_count), dtype=torch.float32),
        rays_per_view=rays_per_view,
        stages=int(config["solver"]["fixed_stages"]),
    )
    solver_seconds = time.perf_counter() - solver_started
    predictions = result.volume.detach().cpu().numpy()[:, 0]
    spacing_xyz = tuple(
        (maximum[index] - minimum[index]) / (grid_size - 1) for index in range(3)
    )
    truth_arrays: list[np.ndarray] = []
    metric_rows: list[dict[str, Any]] = []
    finite_outputs = bool(np.all(np.isfinite(predictions)))
    for index, spec in enumerate(specs):
        truth_evaluation = analytic_phantom_grid(
            spec,
            grid_shape=(grid_size,) * 3,
            dtype=torch.float64,
        )
        truth = truth_evaluation.field.detach().cpu().numpy()
        truth_gradient = truth_evaluation.gradient_xyz.detach().cpu().numpy()
        truth_levels = truth_evaluation.level_sets.detach().cpu().numpy()
        prediction = predictions[index]
        predicted_gradient = scalar_grid_gradient(
            prediction,
            spacing_xyz=spacing_xyz,
        )
        field_metrics = synthetic_field_metrics(
            prediction,
            truth,
            analytic_truth_gradient_xyz=truth_gradient,
            spacing_xyz=spacing_xyz,
        )
        front_metrics, front_status = _front_metrics(
            prediction=prediction,
            predicted_gradient=predicted_gradient,
            truth_gradient=truth_gradient,
            truth_level_sets=truth_levels,
            spacing_xyz=spacing_xyz,
            tolerance_voxels=config["metrics"]["surface_tolerance_voxels"],
            band_voxels=float(config["metrics"]["front_band_half_width_voxels"]),
        )
        support_prediction = operator.forward(result.volume[index : index + 1]).detach()
        support_reprojection = _relative_l2(
            support_prediction[0].to(torch.float64),
            clean_truth[index],
        )
        truth_arrays.append(truth)
        metric_rows.append(
            {
                "family": spec.family,
                "seed": spec.seed,
                "qmc32_vs_qmc64_relative_l2": qmc_discrepancy[index],
                **field_metrics,
                **front_metrics,
                "front_metric_status": front_status,
                "support_reprojection_relative_l2": support_reprojection,
            }
        )
    evaluation_forward_calls = len(specs)
    optimization_calls = {
        "forward_calls": result.forward_calls,
        "adjoint_calls": result.adjoint_calls,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "metric_rows.csv", metric_rows)
    _plot(
        truths=truth_arrays,
        predictions=[value for value in predictions],
        rows=metric_rows,
        output_path=output_dir / "diagnostic.png",
    )
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": PUBLIC_STATUS,
        "evidence_level": config["evidence_level"],
        "source_config_sha256": _sha256(config_path),
        "interfaces": {
            "analytic_phantom_schema": ANALYTIC_PHANTOM_SCHEMA,
            "analytic_renderer_schema": ANALYTIC_RENDERER_SCHEMA,
            "renderer_independence_audit": _renderer_independence_audit(),
        },
        "geometry": {
            "view_count": view_count,
            "rays_per_view": rays_per_view,
            "ray_count": view_count * rays_per_view,
            "grid_size": grid_size,
            "truth_renderer_qmc_samples": geometry_config[
                "truth_renderer_qmc_samples"
            ],
            "renderer_audit_qmc_samples": geometry_config[
                "renderer_audit_qmc_samples"
            ],
            "inverse_operator_qmc_samples": geometry_config[
                "inverse_operator_qmc_samples"
            ],
            "uses_real_psu_support_geometry": True,
            "contains_public_ray_coordinates": False,
        },
        "operator_audit": {
            "adjoint_relative_error": adjoint_relative_error,
            "adjoint_gate_maximum": 2e-5,
            "adjoint_gate_passed": adjoint_relative_error <= 2e-5,
        },
        "solver": {
            "method": config["solver"]["method"],
            "fixed_stages": config["solver"]["fixed_stages"],
            "optimization_calls": optimization_calls,
            "evaluation_forward_calls": evaluation_forward_calls,
            "solver_seconds": solver_seconds,
        },
        "metric_rows": metric_rows,
        "aggregate_diagnostics": {
            "all_outputs_finite": finite_outputs,
            "qmc32_vs_qmc64_relative_l2_maximum": max(qmc_discrepancy),
            "field_relative_l2_mean": float(
                np.mean([row["field_relative_l2"] for row in metric_rows])
            ),
            "h1_seminorm_relative_error_mean": float(
                np.mean([row["h1_seminorm_relative_error"] for row in metric_rows])
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": config["claim_boundary"],
        "public_export_policy": {
            "contains_local_paths": False,
            "contains_ray_coordinates_or_indices": False,
            "contains_measurement_or_volume_arrays": False,
            "contains_private_manifest_hashes": False,
            "contains_only_synthetic_field_slices_and_aggregate_metrics": True,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "# PSU-S16 independent-renderer smoke\n\n"
        "This is an E1 synthetic interface check on real PSU support geometry. "
        "Analytic gradients generate the observations; the inverse uses the "
        "voxel-gradient operator. It is not experimental reconstruction, CFD "
        "validation, fresh confirmation, or evidence that a learned operator is "
        "better than a baseline.\n",
        encoding="ascii",
    )
    _write_checksums(output_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run_smoke(
        config_path=args.config.resolve(),
        view_root=args.view_root.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(report["aggregate_diagnostics"], indent=2))


if __name__ == "__main__":
    main()
