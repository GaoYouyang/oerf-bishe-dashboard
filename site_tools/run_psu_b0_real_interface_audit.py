#!/usr/bin/env python3
"""Audit the B0 reconstruction interface on a sealed subset of real PSU support rays."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from demo_t16_operator.psu_b0_reconstruction_interface import (
    PSUB0VoxelGradientOperator,
    build_trilinear_stencil,
)
from site_tools.psu_bost_aperture_domain import (
    deterministic_paired_uniform_aperture_samples,
    generate_aperture_sample_points,
)
from site_tools.psu_bost_forward_geometry import intersect_forward_ray_box


PRIVATE_SCHEMA = "psu-b0-real-support-interface-audit-1.0"
PUBLIC_SCHEMA = "psu-b0-real-support-interface-public-summary-1.0"
VIEW_BUNDLE_STATUS = "VIEW_SHARD_BUNDLE_TRANSCODED_AND_SOURCE_STREAMS_VERIFIED"
MASK_STATUS = "CORRECTED_VIEW_MASK_SHARDS_BUILT_MECHANICAL_CONTRACT_PASS"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def deterministic_quantile_indices(indices: np.ndarray, count: int) -> np.ndarray:
    """Select ordered active rows without using measurement magnitude."""

    values = np.asarray(indices, dtype=np.int64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("indices must be a nonempty one-dimensional array")
    if values.size > 1 and np.any(values[1:] <= values[:-1]):
        raise ValueError("indices must be strictly increasing")
    if count < 1 or count > values.size:
        raise ValueError("count must lie between one and the number of indices")
    positions = np.floor(
        (np.arange(count, dtype=np.float64) + 0.5) * values.size / count
    ).astype(np.int64)
    selected = values[positions]
    if np.unique(selected).size != count:
        raise RuntimeError("quantile selection produced duplicate rows")
    return selected


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _load_selected_view(
    view_dir: Path,
    *,
    rays_per_view: int,
    lower: np.ndarray,
    upper: np.ndarray,
    design: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    bundle_dir = view_dir / "bundle"
    mask_dir = view_dir / "corrected_masks"
    bundle_manifest_path = bundle_dir / "view_bundle_manifest.json"
    mask_manifest_path = mask_dir / "corrected_view_masks_manifest.json"
    bundle_manifest = _load_json(bundle_manifest_path)
    mask_manifest = _load_json(mask_manifest_path)
    if bundle_manifest.get("status") != VIEW_BUNDLE_STATUS:
        raise ValueError(f"unverified view bundle: {view_dir.name}")
    if mask_manifest.get("status") != MASK_STATUS:
        raise ValueError(f"unverified corrected masks: {view_dir.name}")

    active = np.load(
        mask_dir / "amask_all_zero_based.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    selected = deterministic_quantile_indices(active, rays_per_view)
    vector_names = ("c", "v", "Ruvecs", "Rvvecs", "Rxvecs", "Ryvecs")
    scalar_names = ("Rapvec", "Dfvec", "Csys_all")
    fields = {
        name: np.load(
            bundle_dir / f"{name}.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        for name in vector_names + scalar_names
    }
    origins = np.asarray(fields["c"][selected], dtype=np.float64)
    directions = np.asarray(fields["v"][selected], dtype=np.float64)
    b0 = intersect_forward_ray_box(
        origins,
        directions,
        lower,
        upper,
        layout="rows",
    )
    if not np.all(b0["hit"]):
        raise ValueError(
            f"selected active rows contain {np.count_nonzero(~b0['hit'])} B0 misses"
        )
    start = origins + b0["enter"][:, None] * b0["direction_unit"]
    stop = origins + b0["exit"][:, None] * b0["direction_unit"]
    original_enter = b0["enter"] / b0["direction_norm"]
    original_exit = b0["exit"] / b0["direction_norm"]
    aperture = np.asarray(fields["Rapvec"][selected, 0], dtype=np.float64)
    focal_distance = np.asarray(fields["Dfvec"][selected, 0], dtype=np.float64)
    inner_radius = aperture * (1.0 - original_enter / focal_distance)
    outer_radius = aperture * (1.0 - original_exit / focal_distance)
    projection_u = np.asarray(fields["Ruvecs"][selected], dtype=np.float64)
    projection_v = np.asarray(fields["Rvvecs"][selected], dtype=np.float64)
    aperture_x = np.asarray(fields["Rxvecs"][selected], dtype=np.float64)
    aperture_y = np.asarray(fields["Ryvecs"][selected], dtype=np.float64)
    sample_points = generate_aperture_sample_points(
        start,
        stop,
        aperture_x,
        aperture_y,
        inner_radius,
        outer_radius,
        design["longitudinal_fractions"],
        design["unit_disk_offsets"],
    )
    values = {
        "sample_points": sample_points,
        "projection_u": projection_u,
        "projection_v": projection_v,
        "line_length": np.asarray(b0["length"], dtype=np.float64),
        "system_constant": np.asarray(
            fields["Csys_all"][selected, 0],
            dtype=np.float64,
        ),
    }
    view_id = int(bundle_manifest["view"]["view_id_zero_based"])
    provenance = {
        "view_id_zero_based": view_id,
        "active_row_count": int(active.size),
        "selected_ray_count": int(selected.size),
        "selected_index_sha256": _array_sha256(selected),
        "bundle_manifest_sha256": _sha256(bundle_manifest_path),
        "mask_manifest_sha256": _sha256(mask_manifest_path),
        "b0_hit_count": int(np.count_nonzero(b0["hit"])),
        "direction_norm_minimum": float(np.min(b0["direction_norm"])),
        "direction_norm_maximum": float(np.max(b0["direction_norm"])),
        "line_length_minimum_m": float(np.min(b0["length"])),
        "line_length_maximum_m": float(np.max(b0["length"])),
    }
    return values, provenance


def load_real_support_geometry(
    view_root: Path,
    *,
    rays_per_view: int,
    sample_count: int,
    lower: tuple[float, float, float] = (-0.11, -0.11, -0.11),
    upper: tuple[float, float, float] = (0.11, 0.11, 0.11),
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    view_dirs = sorted(path for path in view_root.glob("view_*") if path.is_dir())
    if len(view_dirs) != 9:
        raise ValueError(f"expected exactly nine support view directories, got {len(view_dirs)}")
    lower_array = np.asarray(lower, dtype=np.float64)
    upper_array = np.asarray(upper, dtype=np.float64)
    design = deterministic_paired_uniform_aperture_samples(sample_count)
    rows = [
        _load_selected_view(
            view_dir,
            rays_per_view=rays_per_view,
            lower=lower_array,
            upper=upper_array,
            design=design,
        )
        for view_dir in view_dirs
    ]
    combined = {
        name: np.concatenate([values[name] for values, _ in rows], axis=0)
        for name in (
            "sample_points",
            "projection_u",
            "projection_v",
            "line_length",
            "system_constant",
        )
    }
    return combined, [provenance for _, provenance in rows]


def _make_operator(
    geometry: dict[str, np.ndarray],
    *,
    grid_size: int,
    dtype: torch.dtype,
) -> PSUB0VoxelGradientOperator:
    minimum = (-0.11, -0.11, -0.11)
    maximum = (0.11, 0.11, 0.11)
    stencil = build_trilinear_stencil(
        geometry["sample_points"],
        grid_shape=(grid_size, grid_size, grid_size),
        grid_minimum_xyz=minimum,
        grid_maximum_xyz=maximum,
        dtype=dtype,
    )
    return PSUB0VoxelGradientOperator(
        stencil=stencil,
        projection_u_xyz=geometry["projection_u"],
        projection_v_xyz=geometry["projection_v"],
        line_length=geometry["line_length"],
        system_constant=geometry["system_constant"],
        grid_minimum_xyz=minimum,
        grid_maximum_xyz=maximum,
        dtype=dtype,
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _profile(
    operator: PSUB0VoxelGradientOperator,
    *,
    device_name: str,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    device = torch.device(device_name)
    operator = operator.to(device)
    generator = torch.Generator().manual_seed(int(seed))
    volume = torch.randn(
        (1, 1, *operator.grid_shape),
        generator=generator,
        dtype=operator.support.dtype,
    ).to(device)
    with torch.no_grad():
        for _ in range(2):
            projected = operator.forward(volume)
            operator.adjoint(projected)
        _synchronize(device)
        operator.reset_call_counts()
        forward_seconds = []
        adjoint_seconds = []
        finite = True
        for _ in range(repeats):
            started = time.perf_counter()
            projected = operator.forward(volume)
            _synchronize(device)
            forward_seconds.append(time.perf_counter() - started)
            started = time.perf_counter()
            reconstructed = operator.adjoint(projected)
            _synchronize(device)
            adjoint_seconds.append(time.perf_counter() - started)
            finite &= bool(torch.all(torch.isfinite(projected)))
            finite &= bool(torch.all(torch.isfinite(reconstructed)))
    memory: dict[str, int | None] = {
        "mps_current_allocated_bytes": None,
        "mps_driver_allocated_bytes": None,
    }
    if device.type == "mps":
        current = getattr(torch.mps, "current_allocated_memory", None)
        driver = getattr(torch.mps, "driver_allocated_memory", None)
        memory["mps_current_allocated_bytes"] = int(current()) if current else None
        memory["mps_driver_allocated_bytes"] = int(driver()) if driver else None
    return {
        "device": device_name,
        "dtype": str(operator.support.dtype).removeprefix("torch."),
        "repeats": repeats,
        "forward_seconds_median": statistics.median(forward_seconds),
        "forward_seconds_minimum": min(forward_seconds),
        "adjoint_seconds_median": statistics.median(adjoint_seconds),
        "adjoint_seconds_minimum": min(adjoint_seconds),
        "finite_outputs": finite,
        "logical_calls": operator.call_report(),
        **memory,
    }


def _max_rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if platform.system() == "Darwin" else raw * 1024


def build_public_summary(private_report: dict[str, Any]) -> dict[str, Any]:
    """Export aggregate evidence without paths, ray indices, or private hashes."""

    return {
        "schema_version": PUBLIC_SCHEMA,
        "status": private_report["status"],
        "evidence_scope": private_report["evidence_scope"],
        "dataset": private_report["dataset"],
        "configuration": private_report["configuration"],
        "aggregate_geometry": private_report["aggregate_geometry"],
        "grid_profiles": private_report["grid_profiles"],
        "gates": private_report["gates"],
        "claim_boundary": private_report["claim_boundary"],
        "public_export_policy": {
            "contains_local_paths": False,
            "contains_ray_indices_or_coordinates": False,
            "contains_measurement_values": False,
            "contains_private_manifest_or_selection_hashes": False,
        },
    }


def run_audit(
    *,
    view_root: Path,
    rays_per_view: int = 256,
    sample_count: int = 16,
    grid_sizes: tuple[int, ...] = (16, 32),
    repeats: int = 5,
    interface_config: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    geometry, provenance = load_real_support_geometry(
        view_root,
        rays_per_view=rays_per_view,
        sample_count=sample_count,
    )
    profiles = []
    all_dot_pass = True
    all_finite = True
    mps_available = torch.backends.mps.is_available()
    for grid_size in grid_sizes:
        cpu_operator = _make_operator(
            geometry,
            grid_size=grid_size,
            dtype=torch.float64,
        )
        cpu_dot = cpu_operator.adjoint_relative_error(seed=grid_size + 401)
        cpu_profile = _profile(
            cpu_operator,
            device_name="cpu",
            repeats=repeats,
            seed=grid_size + 503,
        )
        row: dict[str, Any] = {
            "grid_shape_zyx": [grid_size, grid_size, grid_size],
            "voxel_count": grid_size**3,
            "cpu_float64_adjoint_relative_error": cpu_dot,
            "cpu_profile": cpu_profile,
            "mps_float32_adjoint_relative_error": None,
            "mps_profile": None,
        }
        all_dot_pass &= cpu_dot <= 1e-11
        all_finite &= bool(cpu_profile["finite_outputs"])
        if mps_available:
            mps_operator = _make_operator(
                geometry,
                grid_size=grid_size,
                dtype=torch.float32,
            ).to("mps")
            mps_dot = mps_operator.adjoint_relative_error(seed=grid_size + 607)
            mps_profile = _profile(
                mps_operator,
                device_name="mps",
                repeats=repeats,
                seed=grid_size + 709,
            )
            row["mps_float32_adjoint_relative_error"] = mps_dot
            row["mps_profile"] = mps_profile
            all_dot_pass &= mps_dot <= 2e-5
            all_finite &= bool(mps_profile["finite_outputs"])
        profiles.append(row)

    valid = np.all(
        (geometry["sample_points"] >= -0.11)
        & (geometry["sample_points"] <= 0.11),
        axis=-1,
    )
    configuration: dict[str, Any] = {
        "rays_per_view": rays_per_view,
        "view_count": 9,
        "total_ray_count": int(geometry["line_length"].size),
        "sample_count_per_ray": sample_count,
        "grid_sizes": list(grid_sizes),
        "profile_repeats": repeats,
        "selection": "ordered_active_mask_quantile_without_measurement_magnitude",
        "domain": "B0_FORWARD_BOX",
        "sample_policy": "fixed_denominator_B0_indicator",
        "mps_available": mps_available,
    }
    if interface_config is not None:
        configuration["interface_config_filename"] = interface_config.name
        configuration["interface_config_sha256"] = _sha256(interface_config)
    aggregate = {
        "selected_b0_hit_count": sum(row["b0_hit_count"] for row in provenance),
        "selected_ray_count": int(geometry["line_length"].size),
        "total_sample_count": int(valid.size),
        "valid_sample_count": int(np.count_nonzero(valid)),
        "valid_sample_fraction": float(np.mean(valid)),
        "direction_norm_minimum": min(
            row["direction_norm_minimum"] for row in provenance
        ),
        "direction_norm_maximum": max(
            row["direction_norm_maximum"] for row in provenance
        ),
        "line_length_minimum_m": min(
            row["line_length_minimum_m"] for row in provenance
        ),
        "line_length_maximum_m": max(
            row["line_length_maximum_m"] for row in provenance
        ),
    }
    gates = {
        "nine_support_views_present": len(provenance) == 9,
        "all_selected_active_rays_hit_b0": aggregate["selected_b0_hit_count"]
        == aggregate["selected_ray_count"],
        "cpu_float64_and_mps_float32_adjoint_thresholds": all_dot_pass,
        "all_profile_outputs_finite": all_finite,
        "development_rotation_40_not_accessed": True,
        "final_audit_not_accessed": True,
    }
    private = {
        "schema_version": PRIVATE_SCHEMA,
        "status": (
            "REAL_SUPPORT_GEOMETRY_INTERFACE_PASS_NO_RECONSTRUCTION"
            if all(gates.values())
            else "REAL_SUPPORT_GEOMETRY_INTERFACE_FAIL"
        ),
        "evidence_scope": (
            "REAL_PSU_NINE_SUPPORT_VIEW_GEOMETRY_DETERMINISTIC_ACTIVE_SUBSET_"
            "B0_QMC_FORWARD_ADJOINT_AND_RESOURCE_PROFILE_NO_INVERSE_NO_DEVELOPMENT_"
            "NO_FINAL_AUDIT_NO_SUPERIORITY"
        ),
        "dataset": {
            "name": "Open-source BOS tomography dataset of high-speed flow over a flight body",
            "doi": "10.26208/1VE2-5C19",
            "support_view_count": 9,
        },
        "configuration": configuration,
        "aggregate_geometry": aggregate,
        "grid_profiles": profiles,
        "host": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "process_max_rss_bytes_at_report": _max_rss_bytes(),
        },
        "private_view_provenance": provenance,
        "gates": gates,
        "claim_boundary": {
            "real_psu_geometry_used": True,
            "real_psu_measurement_values_used_for_training": False,
            "inverse_reconstruction_run": False,
            "development_rotation_40_opened": False,
            "final_audit_opened": False,
            "algorithm_superiority": False,
            "field_l2_available": False,
        },
    }
    return private, build_public_summary(private)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--rays-per-view", type=int, default=256)
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--grid-sizes", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--interface-config", type=Path)
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--public-output", type=Path)
    args = parser.parse_args()
    private, public = run_audit(
        view_root=args.view_root,
        rays_per_view=args.rays_per_view,
        sample_count=args.sample_count,
        grid_sizes=tuple(args.grid_sizes),
        repeats=args.repeats,
        interface_config=args.interface_config,
    )
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
    return 0 if public["status"].startswith("REAL_SUPPORT_GEOMETRY_INTERFACE_PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
