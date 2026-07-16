#!/usr/bin/env python3
"""Run a fixed-budget streamed B0 CGLS baseline on real PSU support rays."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
from pathlib import Path
import resource
import time
from typing import Any

import numpy as np
import torch

from demo_t16_operator.psu_b0_streaming_operator import (
    PSUB0StreamingOperator,
    cgls_solve,
    zero_outer_boundary_support,
)
from site_tools.psu_b0_compact_cache import PSUCompactCachedRayStore
from site_tools.psu_b0_real_support_store import PSURealSupportRayStore


PRIVATE_SCHEMA = "psu-b0-streaming-baseline-private-report-1.0"
PUBLIC_SCHEMA = "psu-b0-streaming-baseline-public-summary-1.0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _max_rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if platform.system() == "Darwin" else raw * 1024


def _dtype(name: str) -> torch.dtype:
    values = {
        "float32": torch.float32,
        "float64": torch.float64,
    }
    try:
        return values[str(name)]
    except KeyError as exc:
        raise ValueError("dtype must be float32 or float64") from exc


def _view_offsets(store: Any) -> list[tuple[int, int, int]]:
    rows = []
    start = 0
    for view in store.views:
        stop = start + view.ray_count
        rows.append((view.view_id, start, stop))
        start = stop
    return rows


def _relative_l2(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(numerator)
        / torch.linalg.vector_norm(denominator).clamp_min(1e-24)
    )


def _field_diagnostics(volume: torch.Tensor) -> dict[str, float]:
    values = volume.detach().cpu()[0, 0]
    dx = values[:, :, 1:] - values[:, :, :-1]
    dy = values[:, 1:, :] - values[:, :-1, :]
    dz = values[1:, :, :] - values[:-1, :, :]
    boundary = torch.cat(
        (
            values[0].reshape(-1),
            values[-1].reshape(-1),
            values[:, 0, :].reshape(-1),
            values[:, -1, :].reshape(-1),
            values[:, :, 0].reshape(-1),
            values[:, :, -1].reshape(-1),
        )
    )
    return {
        "minimum": float(torch.min(values)),
        "maximum": float(torch.max(values)),
        "mean": float(torch.mean(values)),
        "l2": float(torch.linalg.vector_norm(values)),
        "max_abs_outer_boundary": float(torch.max(torch.abs(boundary))),
        "anisotropic_total_variation": float(
            torch.sum(torch.abs(dx))
            + torch.sum(torch.abs(dy))
            + torch.sum(torch.abs(dz))
        ),
    }


def _public_call_records(
    records: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    allowed = {
        "operation",
        "wall_seconds",
        "ray_count",
        "chunk_count",
        "b0_hit_count",
        "max_rss_bytes_after_call",
    }
    return [
        {key: value for key, value in row.items() if key in allowed}
        for row in records
    ]


def build_public_summary(private: dict[str, Any]) -> dict[str, Any]:
    """Strip local paths, hashes, absolute data values, and private volume data."""

    return {
        "schema_version": PUBLIC_SCHEMA,
        "status": private["status"],
        "evidence_scope": private["evidence_scope"],
        "dataset": copy.deepcopy(private["dataset"]),
        "configuration": copy.deepcopy(private["configuration"]),
        "selection": copy.deepcopy(private["selection"]),
        "interface_profile": copy.deepcopy(private["interface_profile"]),
        "optimization": copy.deepcopy(private["optimization"]),
        "evaluation": copy.deepcopy(private["evaluation"]),
        "resource_gate": copy.deepcopy(private["resource_gate"]),
        "gates": copy.deepcopy(private["gates"]),
        "claim_boundary": copy.deepcopy(private["claim_boundary"]),
        "public_export_policy": {
            "contains_local_paths": False,
            "contains_measurement_values": False,
            "contains_reconstruction_voxels": False,
            "contains_private_file_hashes": False,
            "contains_development_or_final_audit_values": False,
        },
    }


def run_baseline(
    *,
    view_root: Path | None,
    grid_size: int,
    sample_count: int,
    chunk_rays: int,
    rays_per_view: int | None,
    iterations: int,
    dtype_name: str,
    device_name: str,
    dot_seed: int,
    dot_dual: str,
    dot_threshold: float,
    recurrence_threshold: float,
    local_pair_seconds_maximum: float,
    local_rss_bytes_maximum: int,
    config_path: Path | None = None,
    cache_root: Path | None = None,
    torch_threads: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any], torch.Tensor]:
    if torch_threads is not None:
        if int(torch_threads) < 1:
            raise ValueError("torch_threads must be positive")
        torch.set_num_threads(int(torch_threads))
    dtype = _dtype(dtype_name)
    device = torch.device(device_name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")
    shape = (int(grid_size),) * 3
    if cache_root is None:
        if view_root is None:
            raise ValueError("view_root is required when cache_root is absent")
        store = PSURealSupportRayStore(
            view_root,
            rays_per_view=rays_per_view,
            sample_count=sample_count,
            chunk_rays=chunk_rays,
        )
        ray_store_mode = "source_geometry_rebuilt_per_logical_call"
    else:
        store = PSUCompactCachedRayStore(cache_root)
        if store.grid_shape != shape:
            raise ValueError("compact cache grid does not match grid_size")
        if store.sample_count != int(sample_count):
            raise ValueError("compact cache sample count does not match request")
        if store.chunk_rays != int(chunk_rays):
            raise ValueError("compact cache chunk size does not match request")
        if rays_per_view is not None:
            expected = int(rays_per_view) * len(store.views)
            if store.ray_count != expected:
                raise ValueError("compact cache ray selection does not match request")
        ray_store_mode = "private_compact_stencil_cache"
    support = zero_outer_boundary_support(shape, dtype=dtype)
    operator = PSUB0StreamingOperator(
        ray_store=store,
        grid_shape=shape,
        grid_minimum_xyz=(-0.11, -0.11, -0.11),
        grid_maximum_xyz=(0.11, 0.11, 0.11),
        support=support,
        dtype=dtype,
    ).to(device)

    load_started = time.perf_counter()
    observation = operator.load_observations()
    observation_load_seconds = time.perf_counter() - load_started
    if not torch.all(torch.isfinite(observation)):
        raise ValueError("support observations contain non-finite values")

    probe = torch.randn(
        (1, 1, *shape),
        generator=torch.Generator().manual_seed(int(dot_seed)),
        dtype=dtype,
    ).to(device)
    probe = probe * operator.support
    if dot_dual != "deterministic_random_vector":
        raise ValueError("dot_dual must be deterministic_random_vector")
    dual = torch.randn(
        (1, store.ray_count, 2),
        generator=torch.Generator().manual_seed(int(dot_seed) + 1),
        dtype=dtype,
    ).to(device)
    projected_probe = operator.forward(probe)
    backprojected_dual = operator.adjoint(dual)
    lhs = torch.sum(projected_probe.to(torch.float64) * dual.to(torch.float64))
    rhs = torch.sum(probe.to(torch.float64) * backprojected_dual.to(torch.float64))
    dot_denominator = torch.maximum(torch.abs(lhs), torch.abs(rhs)).clamp_min(1e-24)
    dot_error = float(torch.abs(lhs - rhs) / dot_denominator)
    norm_product_denominator = (
        torch.linalg.vector_norm(projected_probe.to(torch.float64))
        * torch.linalg.vector_norm(dual.to(torch.float64))
        + torch.linalg.vector_norm(probe.to(torch.float64))
        * torch.linalg.vector_norm(backprojected_dual.to(torch.float64))
    ).clamp_min(1e-24)
    norm_product_defect = float(
        torch.abs(lhs - rhs) / norm_product_denominator
    )
    inner_cosine_scale = float(
        torch.maximum(torch.abs(lhs), torch.abs(rhs))
        / (0.5 * norm_product_denominator)
    )
    interface_calls = _public_call_records(operator.call_report()["records"])
    pair_seconds = float(sum(row["wall_seconds"] for row in interface_calls))

    operator.reset_call_counts()
    solve_started = time.perf_counter()
    result = cgls_solve(
        operator,
        observation,
        iterations=iterations,
    )
    solve_seconds = time.perf_counter() - solve_started
    optimization_report = operator.call_report()
    optimization_calls = _public_call_records(optimization_report["records"])

    operator.reset_call_counts()
    direct_prediction = operator.forward(result.volume)
    evaluation_calls = _public_call_records(operator.call_report()["records"])
    direct_residual = observation - direct_prediction
    recurrence_difference = _relative_l2(
        direct_residual - result.residual,
        direct_residual,
    )
    pooled_relative = _relative_l2(direct_residual, observation)
    per_view = []
    for view_id, start, stop in _view_offsets(store):
        per_view.append(
            {
                "view_id_zero_based": int(view_id),
                "ray_count": int(stop - start),
                "relative_measurement_l2": _relative_l2(
                    direct_residual[:, start:stop],
                    observation[:, start:stop],
                ),
            }
        )

    max_rss = _max_rss_bytes()
    field = _field_diagnostics(result.volume)
    all_hits = all(
        int(row["b0_hit_count"]) == int(row["ray_count"])
        for row in interface_calls + optimization_calls + evaluation_calls
    )
    finite_history = all(
        np.isfinite(float(value))
        for row in result.history
        for value in row.values()
        if isinstance(value, (float, int)) and not isinstance(value, bool)
    )
    residual_decreased = pooled_relative < 1.0
    gates = {
        "nine_support_views_present": len(store.views) == 9,
        "all_selected_active_rays_hit_b0": all_hits,
        "full_stream_float_dot_threshold": dot_error <= float(dot_threshold),
        "cgls_no_breakdown": not result.breakdown,
        "cgls_history_finite": finite_history,
        "direct_residual_lower_than_zero_start": residual_decreased,
        "direct_vs_recurrence_residual_threshold": recurrence_difference
        <= float(recurrence_threshold),
        "development_rotation_40_not_accessed": True,
        "final_audit_not_accessed": True,
    }
    local_pair_pass = pair_seconds <= float(local_pair_seconds_maximum)
    local_memory_pass = max_rss <= int(local_rss_bytes_maximum)
    resource_gate = {
        "full_forward_adjoint_pair_wall_seconds": pair_seconds,
        "pair_wall_seconds_local_maximum": float(local_pair_seconds_maximum),
        "pair_wall_time_pass": local_pair_pass,
        "process_max_rss_bytes": int(max_rss),
        "process_max_rss_bytes_local_maximum": int(local_rss_bytes_maximum),
        "memory_pass": local_memory_pass,
        "server_required_for_current_16_cubed_gate": not (
            local_pair_pass and local_memory_pass
        ),
    }
    configuration: dict[str, Any] = {
        "grid_shape_zyx": list(shape),
        "dtype": dtype_name,
        "device": device_name,
        "finite_aperture_sample_count": int(sample_count),
        "chunk_rays": int(chunk_rays),
        "cgls_fixed_iterations": int(iterations),
        "gauge": "zero_one_voxel_outer_boundary",
        "positivity": False,
        "logical_call": (
            "one_complete_deterministic_traversal_of_all_selected_support_chunks"
        ),
        "ray_store_mode": ray_store_mode,
        "torch_threads": int(torch.get_num_threads()),
    }
    if config_path is not None:
        configuration["config_filename"] = config_path.name
        configuration["config_sha256_private_only"] = _sha256(config_path)

    private = {
        "schema_version": PRIVATE_SCHEMA,
        "status": (
            "REAL_SUPPORT_B0_CGLS_FIXED_BUDGET_COMPLETE_NO_FIELD_TRUTH"
            if all(gates.values())
            else "REAL_SUPPORT_B0_CGLS_GATE_FAIL"
        ),
        "evidence_scope": (
            "REAL_PSU_NINE_SUPPORT_VIEW_STREAMED_B0_FLOAT_DOT_AND_FIXED_BUDGET_"
            "CGLS_SUPPORT_FIT_NO_DEVELOPMENT_NO_FINAL_AUDIT_NO_FIELD_TRUTH_NO_"
            "SUPERIORITY"
        ),
        "dataset": {
            "name": (
                "Open-source BOS tomography dataset of high-speed flow over "
                "a flight body"
            ),
            "doi": "10.26208/1VE2-5C19",
            "support_view_count": 9,
        },
        "configuration": configuration,
        "selection": store.selection_summary(),
        "interface_profile": {
            "dot_dual": dot_dual,
            "dot_seed": int(dot_seed),
            "relative_dot_error": dot_error,
            "relative_dot_error_maximum": float(dot_threshold),
            "norm_product_defect": norm_product_defect,
            "inner_cosine_scale": inner_cosine_scale,
            "observation_load_seconds": float(observation_load_seconds),
            "logical_calls": interface_calls,
        },
        "optimization": {
            "solver": "CGLS",
            "start": "zero_field",
            "fixed_iteration_budget": int(iterations),
            "iteration_budget_selected_on_support_or_development": False,
            "wall_seconds": float(solve_seconds),
            "logical_forward_calls": int(
                optimization_report["forward_calls"]
            ),
            "logical_adjoint_calls": int(
                optimization_report["adjoint_calls"]
            ),
            "logical_calls": optimization_calls,
            "history": result.history,
            "breakdown": bool(result.breakdown),
        },
        "evaluation": {
            "direct_support_relative_measurement_l2": pooled_relative,
            "direct_vs_recurrence_residual_relative_difference": (
                recurrence_difference
            ),
            "per_view_support_relative_measurement_l2": per_view,
            "evaluation_logical_calls": evaluation_calls,
            "field_diagnostics_private": field,
        },
        "resource_gate": resource_gate,
        "host": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "torch_num_threads": int(torch.get_num_threads()),
        },
        "gates": gates,
        "claim_boundary": {
            "real_psu_measurement_values_used_for_support_fit": True,
            "support_fit_is_heldout_generalization": False,
            "support_fit_is_unique_three_dimensional_truth": False,
            "experimental_field_l2_available": False,
            "development_rotation_40_opened": False,
            "final_audit_opened": False,
            "algorithm_superiority": False,
        },
    }
    public = build_public_summary(private)
    public["configuration"].pop("config_sha256_private_only", None)
    public["evaluation"].pop("field_diagnostics_private", None)
    return private, public, result.volume.detach().cpu()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--view-root", type=Path)
    source.add_argument("--cache-root", type=Path)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--chunk-rays", type=int, default=32768)
    parser.add_argument(
        "--rays-per-view",
        type=int,
        default=0,
        help="0 means every corrected active row",
    )
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--dot-seed", type=int, default=20260717)
    parser.add_argument(
        "--dot-dual",
        choices=["deterministic_random_vector"],
        default="deterministic_random_vector",
    )
    parser.add_argument("--dot-threshold", type=float, default=1e-11)
    parser.add_argument("--recurrence-threshold", type=float, default=1e-4)
    parser.add_argument("--local-pair-seconds-maximum", type=float, default=600.0)
    parser.add_argument(
        "--local-rss-bytes-maximum",
        type=int,
        default=24 * 1024**3,
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--public-output", type=Path)
    parser.add_argument("--private-volume", type=Path)
    args = parser.parse_args()
    private, public, volume = run_baseline(
        view_root=args.view_root,
        grid_size=args.grid_size,
        sample_count=args.sample_count,
        chunk_rays=args.chunk_rays,
        rays_per_view=(
            None if int(args.rays_per_view) == 0 else int(args.rays_per_view)
        ),
        iterations=args.iterations,
        dtype_name=args.dtype,
        device_name=args.device,
        dot_seed=args.dot_seed,
        dot_dual=args.dot_dual,
        dot_threshold=args.dot_threshold,
        recurrence_threshold=args.recurrence_threshold,
        local_pair_seconds_maximum=args.local_pair_seconds_maximum,
        local_rss_bytes_maximum=args.local_rss_bytes_maximum,
        config_path=args.config,
        cache_root=args.cache_root,
        torch_threads=args.torch_threads,
    )
    if args.private_volume is not None:
        args.private_volume.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.private_volume, volume.numpy(), allow_pickle=False)
        private["private_volume"] = {
            "filename": args.private_volume.name,
            "sha256": _sha256(args.private_volume),
            "shape": list(volume.shape),
            "dtype": str(volume.numpy().dtype),
        }
    if args.private_output is not None:
        args.private_output.parent.mkdir(parents=True, exist_ok=True)
        args.private_output.write_text(
            json.dumps(private, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.public_output is not None:
        args.public_output.parent.mkdir(parents=True, exist_ok=True)
        args.public_output.write_text(
            json.dumps(public, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(public, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if public["status"].endswith("NO_FIELD_TRUTH") else 1


if __name__ == "__main__":
    raise SystemExit(main())
