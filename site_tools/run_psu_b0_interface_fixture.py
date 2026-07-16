#!/usr/bin/env python3
"""Run a synthetic exact-adjoint reconstruction fixture for the PSU B0 interface."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from demo_t16_operator.psu_b0_reconstruction_interface import (
    INTERFACE_SCHEMA,
    PSUB0VoxelGradientOperator,
    build_trilinear_stencil,
    project_dirichlet_gauge,
)
from site_tools.psu_bost_aperture_domain import (
    deterministic_paired_uniform_aperture_samples,
    generate_aperture_sample_points,
)
from site_tools.psu_bost_forward_geometry import intersect_forward_ray_box


REPORT_SCHEMA = "psu-b0-reconstruction-interface-fixture-1.0"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parallel_ray_geometry(
    detector_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if detector_count < 3:
        raise ValueError("detector_count must be at least three")
    detector = np.linspace(-0.84, 0.84, detector_count)
    origins: list[list[float]] = []
    directions: list[list[float]] = []
    projection_u: list[list[float]] = []
    projection_v: list[list[float]] = []
    for first in detector:
        for second in detector:
            origins.append([-1.4, first, second])
            directions.append([1.0, 0.0, 0.0])
            projection_u.append([0.0, 1.0, 0.0])
            projection_v.append([0.0, 0.0, 1.0])

            origins.append([first, -1.4, second])
            directions.append([0.0, 1.0, 0.0])
            projection_u.append([1.0, 0.0, 0.0])
            projection_v.append([0.0, 0.0, 1.0])

            origins.append([first, second, -1.4])
            directions.append([0.0, 0.0, 1.0])
            projection_u.append([1.0, 0.0, 0.0])
            projection_v.append([0.0, 1.0, 0.0])
    return (
        np.asarray(origins, dtype=np.float64),
        np.asarray(directions, dtype=np.float64),
        np.asarray(projection_u, dtype=np.float64),
        np.asarray(projection_v, dtype=np.float64),
    )


def _build_operator(
    *,
    grid_size: int,
    detector_count: int,
    sample_count: int,
    dtype: torch.dtype,
) -> tuple[PSUB0VoxelGradientOperator, dict[str, Any]]:
    origins, directions, projection_u, projection_v = _parallel_ray_geometry(
        detector_count
    )
    lower = np.array([-1.0, -1.0, -1.0])
    upper = np.array([1.0, 1.0, 1.0])
    intersections = intersect_forward_ray_box(
        origins,
        directions,
        lower,
        upper,
        layout="rows",
    )
    start = origins + intersections["enter"][:, None] * intersections["direction_unit"]
    stop = origins + intersections["exit"][:, None] * intersections["direction_unit"]
    design = deterministic_paired_uniform_aperture_samples(sample_count)
    sample_points = generate_aperture_sample_points(
        start,
        stop,
        projection_u,
        projection_v,
        np.full(len(start), 0.055),
        np.full(len(start), 0.055),
        design["longitudinal_fractions"],
        design["unit_disk_offsets"],
    )
    stencil = build_trilinear_stencil(
        sample_points,
        grid_shape=(grid_size, grid_size, grid_size),
        grid_minimum_xyz=lower,
        grid_maximum_xyz=upper,
        dtype=dtype,
    )
    operator = PSUB0VoxelGradientOperator(
        stencil=stencil,
        projection_u_xyz=projection_u,
        projection_v_xyz=projection_v,
        line_length=intersections["length"],
        system_constant=np.ones(len(start)),
        grid_minimum_xyz=lower,
        grid_maximum_xyz=upper,
        dtype=dtype,
    )
    geometry = {
        "ray_count": int(operator.ray_count),
        "sample_count_per_ray": int(operator.sample_count),
        "valid_sample_count": int(torch.count_nonzero(operator.sample_valid)),
        "total_sample_count": int(operator.sample_valid.numel()),
        "valid_sample_fraction": float(operator.sample_valid.to(torch.float64).mean()),
        "detector_count_per_axis_pair": detector_count,
        "view_direction_count": 3,
    }
    return operator, geometry


def _truth_field(grid_size: int, dtype: torch.dtype) -> torch.Tensor:
    axis = torch.linspace(-1.0, 1.0, grid_size, dtype=dtype)
    zz, yy, xx = torch.meshgrid(axis, axis, axis, indexing="ij")
    positive = 0.9 * torch.exp(
        -0.5
        * (
            ((xx + 0.22) / 0.24) ** 2
            + ((yy - 0.12) / 0.31) ** 2
            + ((zz + 0.08) / 0.27) ** 2
        )
    )
    negative = -0.52 * torch.exp(
        -0.5
        * (
            ((xx - 0.28) / 0.20) ** 2
            + ((yy + 0.18) / 0.26) ** 2
            + ((zz - 0.16) / 0.22) ** 2
        )
    )
    front = 0.22 * torch.tanh(
        (xx - 0.12 * torch.sin(2.4 * yy + 1.7 * zz)) / 0.10
    )
    window = (1.0 - xx.square()) * (1.0 - yy.square()) * (1.0 - zz.square())
    field = (positive + negative + front) * window.clamp_min(0.0)
    return project_dirichlet_gauge(field.reshape(1, 1, grid_size, grid_size, grid_size))


def _relative_l2(error: torch.Tensor, reference: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(error)
    denominator = torch.linalg.vector_norm(reference).clamp_min(1e-18)
    return float(numerator / denominator)


def _plot_fixture(
    *,
    truth: np.ndarray,
    reconstruction: np.ndarray,
    residual_history: list[float],
    output_stem: Path,
) -> list[dict[str, Any]]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    center = truth.shape[0] // 2
    error = reconstruction - truth
    value_limit = float(max(np.max(np.abs(truth)), np.max(np.abs(reconstruction))))
    error_limit = float(np.max(np.abs(error)))
    figure, axes = plt.subplots(1, 4, figsize=(13.2, 3.35), constrained_layout=True)
    image0 = axes[0].imshow(
        truth[center],
        cmap="RdBu_r",
        vmin=-value_limit,
        vmax=value_limit,
        origin="lower",
    )
    axes[0].set_title("Truth perturbation")
    axes[1].imshow(
        reconstruction[center],
        cmap="RdBu_r",
        vmin=-value_limit,
        vmax=value_limit,
        origin="lower",
    )
    axes[1].set_title("Landweber reconstruction")
    image2 = axes[2].imshow(
        error[center],
        cmap="coolwarm",
        vmin=-error_limit,
        vmax=error_limit,
        origin="lower",
    )
    axes[2].set_title("Signed error")
    axes[3].semilogy(
        np.arange(len(residual_history)),
        residual_history,
        color="#15616d",
        linewidth=2.2,
    )
    axes[3].set_title("Support residual")
    axes[3].set_xlabel("Fixed iteration")
    axes[3].set_ylabel("Relative L2")
    axes[3].grid(alpha=0.25)
    for axis in axes[:3]:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.colorbar(image0, ax=axes[:2], shrink=0.78, label="Perturbation")
    figure.colorbar(image2, ax=axes[2], shrink=0.78, label="Error")
    figure.suptitle(
        "PSU B0 interface fixture: exact discrete adjoint, not an experimental result",
        fontsize=12,
    )
    records = []
    for suffix in ("png", "pdf", "svg"):
        path = output_stem.with_suffix(f".{suffix}")
        figure.savefig(path, dpi=220 if suffix == "png" else None)
        records.append(
            {
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    plt.close(figure)
    return records


def run_fixture(
    *,
    grid_size: int = 12,
    detector_count: int = 7,
    sample_count: int = 16,
    power_iterations: int = 18,
    reconstruction_iterations: int = 60,
    step_fraction: float = 1.35,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    if not 0.0 < step_fraction < 2.0:
        raise ValueError("step_fraction must lie in (0,2)")
    dtype = torch.float64
    operator, geometry = _build_operator(
        grid_size=grid_size,
        detector_count=detector_count,
        sample_count=sample_count,
        dtype=dtype,
    )
    truth = _truth_field(grid_size, dtype)
    operator.reset_call_counts()
    observation = operator.forward(truth)
    generation_calls = operator.call_report()
    operator.reset_call_counts()

    adjoint_error = operator.adjoint_relative_error(seed=211)
    operator.reset_call_counts()
    spectral_started = time.perf_counter()
    lipschitz = operator.estimate_lipschitz(
        power_iterations=power_iterations,
        boundary_width=1,
        seed=307,
    )
    spectral_seconds = time.perf_counter() - spectral_started
    spectral_calls = operator.call_report()
    operator.reset_call_counts()

    current = torch.zeros_like(truth)
    residual_history: list[float] = []
    optimization_started = time.perf_counter()
    for _ in range(reconstruction_iterations):
        residual = observation - operator.forward(current)
        residual_history.append(_relative_l2(residual, observation))
        gradient = operator.adjoint(residual)
        current = project_dirichlet_gauge(
            current + (step_fraction / lipschitz) * gradient,
            support=operator.support,
            boundary_width=1,
        )
    final_prediction = operator.forward(current)
    optimization_seconds = time.perf_counter() - optimization_started
    optimization_calls = operator.call_report()
    final_residual = _relative_l2(final_prediction - observation, observation)
    field_relative_l2 = _relative_l2(current - truth, truth)

    gates = {
        "adjoint_dot_product": adjoint_error <= 1e-11,
        "fixed_iteration_residual_decreased": final_residual
        < residual_history[0],
        "finite_outputs": all(
            np.isfinite(value)
            for value in (
                adjoint_error,
                lipschitz,
                final_residual,
                field_relative_l2,
            )
        ),
        "logical_optimization_calls": optimization_calls
        == {
            "forward_calls": reconstruction_iterations + 1,
            "adjoint_calls": reconstruction_iterations,
        },
    }
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "status": (
            "B0_RECONSTRUCTION_INTERFACE_FIXTURE_PASS"
            if all(gates.values())
            else "B0_RECONSTRUCTION_INTERFACE_FIXTURE_FAIL"
        ),
        "evidence_scope": (
            "SELF_GENERATED_LINEAR_FIXTURE_EXACT_DISCRETE_ADJOINT_AND_FIXED_ITERATION_"
            "RECONSTRUCTION_NO_PSU_MEASUREMENT_NO_EXPERIMENTAL_OR_SUPERIORITY_CLAIM"
        ),
        "interface_schema_version": INTERFACE_SCHEMA,
        "configuration": {
            "grid_shape_zyx": [grid_size, grid_size, grid_size],
            "detector_count": detector_count,
            "sample_count": sample_count,
            "power_iterations": power_iterations,
            "reconstruction_iterations": reconstruction_iterations,
            "normalized_step_fraction": step_fraction,
            "dtype": "float64",
            "device": "cpu",
            "gauge": "zero_outer_voxel_boundary",
        },
        "geometry": geometry,
        "metrics": {
            "adjoint_relative_dot_error": adjoint_error,
            "estimated_normal_operator_lipschitz": lipschitz,
            "initial_measurement_relative_l2": residual_history[0],
            "final_measurement_relative_l2": final_residual,
            "field_relative_l2_fixture_truth_only": field_relative_l2,
        },
        "calls": {
            "self_generation": generation_calls,
            "spectral_estimation": spectral_calls,
            "optimization_and_final_evaluation": optimization_calls,
        },
        "runtime_seconds": {
            "spectral_estimation": spectral_seconds,
            "optimization_and_final_evaluation": optimization_seconds,
        },
        "gates": gates,
        "claim_boundary": {
            "psu_experimental_data_used": False,
            "field_l2_transferable_to_psu": False,
            "algorithm_superiority": False,
            "purpose": "freeze and falsify the reconstruction interface before development data access",
        },
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        arrays_path = output_dir / "fixture_arrays.npz"
        np.savez_compressed(
            arrays_path,
            truth=truth[0, 0].numpy(),
            reconstruction=current[0, 0].numpy(),
            residual_history=np.asarray(residual_history),
        )
        report["arrays"] = {
            "filename": arrays_path.name,
            "bytes": arrays_path.stat().st_size,
            "sha256": _sha256(arrays_path),
        }
        report["figures"] = _plot_fixture(
            truth=truth[0, 0].numpy(),
            reconstruction=current[0, 0].numpy(),
            residual_history=residual_history + [final_residual],
            output_stem=output_dir / "psu_b0_interface_fixture",
        )
        report_path = output_dir / "report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=12)
    parser.add_argument("--detector-count", type=int, default=7)
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--power-iterations", type=int, default=18)
    parser.add_argument("--reconstruction-iterations", type=int, default=60)
    parser.add_argument("--step-fraction", type=float, default=1.35)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report = run_fixture(
        grid_size=args.grid_size,
        detector_count=args.detector_count,
        sample_count=args.sample_count,
        power_iterations=args.power_iterations,
        reconstruction_iterations=args.reconstruction_iterations,
        step_fraction=args.step_fraction,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"].endswith("_PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
