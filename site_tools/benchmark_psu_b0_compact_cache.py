#!/usr/bin/env python3
"""Benchmark direct geometry against the private PSU compact stencil cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import platform
from statistics import median
from typing import Any

import torch

from demo_t16_operator.psu_b0_streaming_operator import (
    PSUB0StreamingOperator,
    zero_outer_boundary_support,
)
from site_tools.psu_b0_compact_cache import PSUCompactCachedRayStore
from site_tools.psu_b0_real_support_store import PSURealSupportRayStore


BENCHMARK_SCHEMA = "psu-b0-compact-cache-public-benchmark-1.0"


def _max_rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if platform.system() == "Darwin" else raw * 1024


def _relative_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(left - right)
        / torch.linalg.vector_norm(left).clamp_min(1e-24)
    )


def _dot_error(
    volume: torch.Tensor,
    dual: torch.Tensor,
    projected: torch.Tensor,
    backprojected: torch.Tensor,
) -> float:
    lhs = torch.sum(projected.to(torch.float64) * dual.to(torch.float64))
    rhs = torch.sum(volume.to(torch.float64) * backprojected.to(torch.float64))
    denominator = torch.maximum(torch.abs(lhs), torch.abs(rhs)).clamp_min(1e-24)
    return float(torch.abs(lhs - rhs) / denominator)


@torch.no_grad()
def benchmark_compact_cache(
    *,
    direct_store: PSURealSupportRayStore,
    cached_store: PSUCompactCachedRayStore,
    grid_shape: tuple[int, int, int],
    grid_minimum_xyz: tuple[float, float, float],
    grid_maximum_xyz: tuple[float, float, float],
    seed: int,
    repeats: int,
    torch_threads: int,
    forward_difference_maximum: float,
    adjoint_difference_maximum: float,
    dot_error_maximum: float,
    speedup_minimum: float,
    rss_bytes_maximum: int,
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if torch_threads < 1:
        raise ValueError("torch_threads must be positive")
    if direct_store.ray_count != cached_store.ray_count:
        raise ValueError("direct and cached stores must contain the same rays")
    if direct_store.sample_count != cached_store.sample_count:
        raise ValueError("direct and cached stores must use the same aperture samples")
    shape = tuple(int(value) for value in grid_shape)
    if cached_store.grid_shape != shape:
        raise ValueError("cached grid does not match benchmark grid")
    torch.set_num_threads(int(torch_threads))
    support = zero_outer_boundary_support(
        shape,
        dtype=torch.float64,
    )
    operators = {
        "direct": PSUB0StreamingOperator(
            ray_store=direct_store,
            grid_shape=shape,
            grid_minimum_xyz=grid_minimum_xyz,
            grid_maximum_xyz=grid_maximum_xyz,
            support=support,
            dtype=torch.float64,
        ),
        "cached": PSUB0StreamingOperator(
            ray_store=cached_store,
            grid_shape=shape,
            grid_minimum_xyz=grid_minimum_xyz,
            grid_maximum_xyz=grid_maximum_xyz,
            support=support,
            dtype=torch.float64,
        ),
    }
    generator = torch.Generator().manual_seed(int(seed))
    volume = torch.randn(
        (1, 1, *shape),
        generator=generator,
        dtype=torch.float64,
    )
    volume = volume * support
    dual = torch.randn(
        (1, direct_store.ray_count, 2),
        generator=generator,
        dtype=torch.float64,
    )
    timings: dict[str, list[float]] = {"direct": [], "cached": []}
    first_outputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    execution_order: list[str] = []
    for repeat in range(int(repeats)):
        order = ("direct", "cached") if repeat % 2 == 0 else ("cached", "direct")
        for mode in order:
            operator = operators[mode]
            operator.reset_call_counts()
            projected = operator(volume)
            backprojected = operator.adjoint(dual)
            records = operator.call_report()["records"]
            if len(records) != 2:
                raise RuntimeError("each benchmark pair must record one F and one A^T")
            pair_seconds = float(sum(float(row["wall_seconds"]) for row in records))
            timings[mode].append(pair_seconds)
            execution_order.append(mode)
            if mode not in first_outputs:
                first_outputs[mode] = (
                    projected.detach().clone(),
                    backprojected.detach().clone(),
                )
    direct_projected, direct_backprojected = first_outputs["direct"]
    cached_projected, cached_backprojected = first_outputs["cached"]
    forward_difference = _relative_difference(
        direct_projected,
        cached_projected,
    )
    adjoint_difference = _relative_difference(
        direct_backprojected,
        cached_backprojected,
    )
    direct_dot_error = _dot_error(
        volume,
        dual,
        direct_projected,
        direct_backprojected,
    )
    cached_dot_error = _dot_error(
        volume,
        dual,
        cached_projected,
        cached_backprojected,
    )
    direct_median = float(median(timings["direct"]))
    cached_median = float(median(timings["cached"]))
    speedup = direct_median / max(cached_median, 1e-24)
    rss_bytes = _max_rss_bytes()
    gates = {
        "forward_relative_difference": forward_difference
        <= float(forward_difference_maximum),
        "adjoint_relative_difference": adjoint_difference
        <= float(adjoint_difference_maximum),
        "cached_relative_dot_error": cached_dot_error <= float(dot_error_maximum),
        "median_pair_speedup": speedup >= float(speedup_minimum),
        "process_max_rss": rss_bytes <= int(rss_bytes_maximum),
        "development_rotation_40_not_accessed": True,
        "final_audit_not_accessed": True,
    }
    cache_manifest = cached_store.manifest
    return {
        "schema_version": BENCHMARK_SCHEMA,
        "status": (
            "COMPACT_CACHE_NUMERICAL_AND_SPEED_GATE_PASS"
            if all(gates.values())
            else "COMPACT_CACHE_BENCHMARK_GATE_FAIL"
        ),
        "evidence_scope": (
            "SAME_SESSION_SUPPORT_ONLY_DIRECT_VS_PRIVATE_COMPACT_CACHE_"
            "NO_DEVELOPMENT_NO_FINAL_AUDIT"
        ),
        "configuration": {
            "grid_shape_zyx": list(shape),
            "ray_count": int(direct_store.ray_count),
            "sample_count": int(direct_store.sample_count),
            "chunk_rays": int(direct_store.chunk_rays),
            "dtype": "float64",
            "device": "cpu",
            "torch_threads": int(torch.get_num_threads()),
            "seed": int(seed),
            "pair_repeats_per_mode": int(repeats),
            "execution_order": execution_order,
        },
        "cache": {
            "schema_version": cache_manifest["schema_version"],
            "fraction_dtype": cache_manifest["fraction_dtype"],
            "build_wall_seconds": cache_manifest["build_wall_seconds"],
            "total_cache_bytes": sum(
                int(record["nbytes"])
                for record in cache_manifest["arrays"].values()
            ),
        },
        "numerical_equivalence": {
            "forward_relative_difference": forward_difference,
            "forward_relative_difference_maximum": float(
                forward_difference_maximum
            ),
            "adjoint_relative_difference": adjoint_difference,
            "adjoint_relative_difference_maximum": float(
                adjoint_difference_maximum
            ),
            "direct_relative_dot_error": direct_dot_error,
            "cached_relative_dot_error": cached_dot_error,
            "cached_relative_dot_error_maximum": float(dot_error_maximum),
        },
        "performance": {
            "direct_pair_seconds": timings["direct"],
            "cached_pair_seconds": timings["cached"],
            "direct_median_pair_seconds": direct_median,
            "cached_median_pair_seconds": cached_median,
            "median_pair_speedup": speedup,
            "median_pair_speedup_minimum": float(speedup_minimum),
            "process_max_rss_bytes": int(rss_bytes),
            "process_max_rss_bytes_maximum": int(rss_bytes_maximum),
        },
        "gates": gates,
        "claim_boundary": {
            "same_session_timing_is_universal_hardware_speedup": False,
            "cache_changes_the_declared_discrete_operator": False,
            "support_fit_is_heldout_generalization": False,
            "field_truth_available": False,
            "algorithm_superiority": False,
            "private_cache_publicly_uploaded": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--chunk-rays", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--forward-difference-maximum", type=float, default=1e-14)
    parser.add_argument("--adjoint-difference-maximum", type=float, default=1e-14)
    parser.add_argument("--dot-error-maximum", type=float, default=1e-11)
    parser.add_argument("--speedup-minimum", type=float, default=1.5)
    parser.add_argument(
        "--rss-bytes-maximum",
        type=int,
        default=24 * 1024**3,
    )
    parser.add_argument("--verify-cache-hashes", action="store_true")
    args = parser.parse_args()
    direct = PSURealSupportRayStore(
        args.view_root,
        rays_per_view=None,
        sample_count=int(args.sample_count),
        chunk_rays=int(args.chunk_rays),
    )
    cached = PSUCompactCachedRayStore(
        args.cache_root,
        verify_hashes=bool(args.verify_cache_hashes),
    )
    report = benchmark_compact_cache(
        direct_store=direct,
        cached_store=cached,
        grid_shape=(int(args.grid_size),) * 3,
        grid_minimum_xyz=(-0.11, -0.11, -0.11),
        grid_maximum_xyz=(0.11, 0.11, 0.11),
        seed=int(args.seed),
        repeats=int(args.repeats),
        torch_threads=int(args.torch_threads),
        forward_difference_maximum=float(args.forward_difference_maximum),
        adjoint_difference_maximum=float(args.adjoint_difference_maximum),
        dot_error_maximum=float(args.dot_error_maximum),
        speedup_minimum=float(args.speedup_minimum),
        rss_bytes_maximum=int(args.rss_bytes_maximum),
    )
    args.public_output.parent.mkdir(parents=True, exist_ok=True)
    args.public_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"].endswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
