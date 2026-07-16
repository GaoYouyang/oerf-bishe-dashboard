#!/usr/bin/env python3
"""Run the B0/B1 fixed-domain geometry census on PSU BOST view shards.

B0 is the forward-ray intersection with the reconstruction box. B1 is the
intersection of that same box with a normalized one-nappe cone. Unlike the
author-compatible A0/A1 paths, B1 never falls back to another spatial domain.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from .psu_bost_forward_geometry import (
        CONTRACT_VERSION as GEOMETRY_CONTRACT_VERSION,
        intersect_forward_ray_box,
        intersect_forward_ray_box_cone,
    )
except ImportError:  # Direct script execution.
    from psu_bost_forward_geometry import (  # type: ignore[no-redef]
        CONTRACT_VERSION as GEOMETRY_CONTRACT_VERSION,
        intersect_forward_ray_box,
        intersect_forward_ray_box_cone,
    )


REPORT_SCHEMA = "psu-bost-fixed-domain-geometry-audit-1.0"
AGGREGATE_SCHEMA = "psu-bost-fixed-domain-all-view-audit-1.0"
VIEW_BUNDLE_STATUS = "VIEW_SHARD_BUNDLE_TRANSCODED_AND_SOURCE_STREAMS_VERIFIED"
MASK_STATUS = "CORRECTED_VIEW_MASK_SHARDS_BUILT_MECHANICAL_CONTRACT_PASS"
SETUP_STATUS_PREFIX = "STREAMED_SETUP_"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _open_view_contract(
    view_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    bundle_dir = view_dir / "bundle"
    setup_dir = view_dir / "setup"
    mask_dir = view_dir / "corrected_masks"
    bundle_manifest_path = bundle_dir / "view_bundle_manifest.json"
    setup_manifest_path = setup_dir / "streamed_setup_manifest.json"
    mask_manifest_path = mask_dir / "corrected_view_masks_manifest.json"

    bundle_manifest = _load_json(bundle_manifest_path)
    setup_manifest = _load_json(setup_manifest_path)
    mask_manifest = _load_json(mask_manifest_path)
    if bundle_manifest.get("status") != VIEW_BUNDLE_STATUS:
        raise ValueError(f"view bundle is not verified: {view_dir}")
    if not str(setup_manifest.get("status", "")).startswith(SETUP_STATUS_PREFIX):
        raise ValueError(f"streamed setup status is missing: {view_dir}")
    if mask_manifest.get("status") != MASK_STATUS:
        raise ValueError(f"corrected masks are not verified: {view_dir}")

    arrays = {
        "origin": np.load(bundle_dir / "c.npy", mmap_mode="r", allow_pickle=False),
        "direction": np.load(bundle_dir / "v.npy", mmap_mode="r", allow_pickle=False),
        "author_cam_data": np.load(
            setup_dir / "cam_data.npy", mmap_mode="r", allow_pickle=False
        ),
        "author_geometry_flags": np.load(
            setup_dir / "geometry_flags.npy", mmap_mode="r", allow_pickle=False
        ),
    }
    rows = int(arrays["origin"].shape[0])
    if arrays["origin"].shape != (rows, 3):
        raise ValueError("origin must have shape (N, 3)")
    if arrays["direction"].shape != (rows, 3):
        raise ValueError("direction must have shape (N, 3)")
    if arrays["author_cam_data"].shape != (rows, 18):
        raise ValueError("author cam_data must have shape (N, 18)")
    if arrays["author_geometry_flags"].shape != (rows,):
        raise ValueError("author geometry_flags must have shape (N,)")

    masks = {
        name: np.load(
            mask_dir / f"{name}_zero_based.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        for name in ("amask_all", "imask_all")
    }
    for name, indices in masks.items():
        if indices.ndim != 1 or indices.dtype != np.int64:
            raise ValueError(f"{name} must be a one-dimensional int64 array")
        if indices.size and (indices[0] < 0 or indices[-1] >= rows):
            raise ValueError(f"{name} contains an out-of-range index")
        if indices.size > 1 and np.any(indices[1:] <= indices[:-1]):
            raise ValueError(f"{name} must be strictly increasing")

    metadata = {
        "view_id_zero_based": int(bundle_manifest["view"]["view_id_zero_based"]),
        "rows": rows,
        "bundle_manifest_sha256": _sha256_file(bundle_manifest_path),
        "setup_manifest_sha256": _sha256_file(setup_manifest_path),
        "mask_manifest_sha256": _sha256_file(mask_manifest_path),
        "bundle_status": bundle_manifest["status"],
        "setup_status": setup_manifest["status"],
        "mask_status": mask_manifest["status"],
    }
    return arrays, masks, metadata


def _empty_mask_accumulator(count: int) -> dict[str, float | int]:
    return {
        "count": count,
        "author_nonzero_count": 0,
        "b0_hit_count": 0,
        "b1_hit_count": 0,
        "b1_removed_from_b0_count": 0,
        "author_length_sum_m": 0.0,
        "b0_length_sum_m": 0.0,
        "b1_length_sum_m": 0.0,
    }


def _safe_fraction(numerator: float | int, denominator: float | int) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _finalize_mask_accumulator(value: Mapping[str, float | int]) -> dict[str, Any]:
    count = int(value["count"])
    b0_length = float(value["b0_length_sum_m"])
    return {
        **value,
        "author_nonzero_fraction": _safe_fraction(
            value["author_nonzero_count"], count
        ),
        "b0_hit_fraction": _safe_fraction(value["b0_hit_count"], count),
        "b1_hit_fraction": _safe_fraction(value["b1_hit_count"], count),
        "b1_removed_from_b0_fraction": _safe_fraction(
            value["b1_removed_from_b0_count"], count
        ),
        "b1_path_fraction_of_b0": _safe_fraction(
            value["b1_length_sum_m"], b0_length
        ),
    }


def audit_fixed_domain_view(
    *,
    view_dir: Path,
    chunk_rows: int = 65_536,
    outer_minimum: tuple[float, float, float] = (-0.110, -0.110, -0.110),
    outer_maximum: tuple[float, float, float] = (0.110, 0.110, 0.110),
    cone_vertex: tuple[float, float, float] = (0.060, 0.015, 0.0),
    cone_axis: tuple[float, float, float] = (-1.0, -0.1, 0.0),
    cone_angle_degrees: float = 25.0,
) -> dict[str, Any]:
    """Audit one real view without loading its full ray table into memory."""

    if chunk_rows < 2:
        raise ValueError("chunk_rows must be at least two")
    if not math.isfinite(cone_angle_degrees) or not 0.0 < cone_angle_degrees < 90.0:
        raise ValueError("cone_angle_degrees must be strictly between 0 and 90")
    lower = np.asarray(outer_minimum, dtype=np.float64)
    upper = np.asarray(outer_maximum, dtype=np.float64)
    vertex = np.asarray(cone_vertex, dtype=np.float64)
    axis = np.asarray(cone_axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    theta = math.radians(cone_angle_degrees)

    arrays, masks, metadata = _open_view_contract(view_dir)
    rows = int(metadata["rows"])
    counts = {
        "ray_count": 0,
        "author_selected_nonzero_count": 0,
        "author_full_box_zero_flag_count": 0,
        "b0_hit_count": 0,
        "b0_zero_length_count": 0,
        "b1_hit_count": 0,
        "b1_zero_length_count": 0,
        "b1_removed_from_b0_count": 0,
        "b0_endpoint_box_violation_count": 0,
        "b1_endpoint_box_violation_count": 0,
        "b1_endpoint_nappe_violation_count": 0,
        "b1_endpoint_cone_radial_violation_count": 0,
        "b1_midpoint_cone_violation_count": 0,
        "b1_hit_without_b0_hit_count": 0,
        "b1_length_exceeds_b0_count": 0,
        "nonfinite_output_count": 0,
        "b0_point_touch_count": 0,
        "b1_point_touch_count": 0,
        "b1_nappe_rejected_double_cone_count": 0,
        "b1_nappe_rejected_with_b0_hit_count": 0,
    }
    sums = {
        "author_selected_length_sum_m": 0.0,
        "b0_length_sum_m": 0.0,
        "b1_length_sum_m": 0.0,
    }
    mask_stats = {
        name: _empty_mask_accumulator(int(indices.size))
        for name, indices in masks.items()
    }
    tolerance = 2e-9
    started = time.perf_counter()

    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        origin = np.asarray(arrays["origin"][start:stop], dtype=np.float64)
        direction = np.asarray(arrays["direction"][start:stop], dtype=np.float64)
        author_length = np.asarray(
            arrays["author_cam_data"][start:stop, 0], dtype=np.float64
        )
        author_flags = np.asarray(
            arrays["author_geometry_flags"][start:stop], dtype=np.uint8
        )

        b0 = intersect_forward_ray_box(origin, direction, lower, upper, layout="rows")
        b1 = intersect_forward_ray_box_cone(
            origin,
            direction,
            lower,
            upper,
            vertex,
            axis,
            theta,
            layout="rows",
        )
        b0_hit = np.asarray(b0["hit"], dtype=bool)
        b1_hit = np.asarray(b1["hit"], dtype=bool)
        b0_length = np.asarray(b0["length"], dtype=np.float64)
        b1_length = np.asarray(b1["length"], dtype=np.float64)
        b1_removed = b0_hit & ~b1_hit

        b0_start = origin + b0["enter"][:, None] * b0["direction_unit"]
        b0_end = origin + b0["exit"][:, None] * b0["direction_unit"]
        b1_start = origin + b1["enter"][:, None] * b1["direction_unit"]
        b1_end = origin + b1["exit"][:, None] * b1["direction_unit"]
        b0_inside = (
            np.all(b0_start >= lower[None, :] - tolerance, axis=1)
            & np.all(b0_start <= upper[None, :] + tolerance, axis=1)
            & np.all(b0_end >= lower[None, :] - tolerance, axis=1)
            & np.all(b0_end <= upper[None, :] + tolerance, axis=1)
        )
        b1_inside = (
            np.all(b1_start >= lower[None, :] - tolerance, axis=1)
            & np.all(b1_start <= upper[None, :] + tolerance, axis=1)
            & np.all(b1_end >= lower[None, :] - tolerance, axis=1)
            & np.all(b1_end <= upper[None, :] + tolerance, axis=1)
        )
        b1_alpha_start = (b1_start - vertex[None, :]) @ axis
        b1_alpha_end = (b1_end - vertex[None, :]) @ axis
        b1_nappe_inside = (b1_alpha_start >= -tolerance) & (
            b1_alpha_end >= -tolerance
        )
        b1_midpoint = 0.5 * (b1_start + b1_end)

        def cone_membership(points: np.ndarray) -> np.ndarray:
            displacement = points - vertex[None, :]
            alpha = displacement @ axis
            perpendicular = displacement - alpha[:, None] * axis[None, :]
            radial = np.linalg.norm(perpendicular, axis=1)
            return (alpha >= -tolerance) & (
                radial <= np.maximum(alpha, 0.0) * math.tan(theta) + tolerance
            )

        b1_start_cone_inside = cone_membership(b1_start)
        b1_end_cone_inside = cone_membership(b1_end)
        b1_midpoint_cone_inside = cone_membership(b1_midpoint)

        counts["ray_count"] += stop - start
        counts["author_selected_nonzero_count"] += int(
            np.count_nonzero(author_length > 0.0)
        )
        counts["author_full_box_zero_flag_count"] += int(
            np.count_nonzero(author_flags & np.uint8(1))
        )
        counts["b0_hit_count"] += int(np.count_nonzero(b0_hit))
        counts["b1_hit_count"] += int(np.count_nonzero(b1_hit))
        counts["b1_removed_from_b0_count"] += int(np.count_nonzero(b1_removed))
        counts["b0_endpoint_box_violation_count"] += int(
            np.count_nonzero(b0_hit & ~b0_inside)
        )
        counts["b1_endpoint_box_violation_count"] += int(
            np.count_nonzero(b1_hit & ~b1_inside)
        )
        counts["b1_endpoint_nappe_violation_count"] += int(
            np.count_nonzero(b1_hit & ~b1_nappe_inside)
        )
        counts["b1_endpoint_cone_radial_violation_count"] += int(
            np.count_nonzero(
                b1_hit & ~(b1_start_cone_inside & b1_end_cone_inside)
            )
        )
        counts["b1_midpoint_cone_violation_count"] += int(
            np.count_nonzero(b1_hit & ~b1_midpoint_cone_inside)
        )
        counts["b1_hit_without_b0_hit_count"] += int(
            np.count_nonzero(b1_hit & ~b0_hit)
        )
        counts["b1_length_exceeds_b0_count"] += int(
            np.count_nonzero(b1_length > b0_length + tolerance)
        )
        counts["nonfinite_output_count"] += sum(
            int(np.count_nonzero(~np.isfinite(value)))
            for value in (
                b0_length,
                b1_length,
                b0_start[b0_hit],
                b0_end[b0_hit],
                b1_start[b1_hit],
                b1_end[b1_hit],
            )
        )
        counts["b0_point_touch_count"] += int(
            np.count_nonzero(b0["forward_point_touch"])
        )
        counts["b1_point_touch_count"] += int(np.count_nonzero(b1["point_touch"]))
        counts["b1_nappe_rejected_double_cone_count"] += int(
            np.count_nonzero(b1["nappe_rejected"])
        )
        counts["b1_nappe_rejected_with_b0_hit_count"] += int(
            np.count_nonzero(b1["nappe_rejected"] & b0_hit)
        )
        sums["author_selected_length_sum_m"] += float(author_length.sum())
        sums["b0_length_sum_m"] += float(b0_length.sum())
        sums["b1_length_sum_m"] += float(b1_length.sum())

        for name, indices in masks.items():
            left = int(np.searchsorted(indices, start, side="left"))
            right = int(np.searchsorted(indices, stop, side="left"))
            local = np.asarray(indices[left:right] - start, dtype=np.int64)
            if not local.size:
                continue
            accumulator = mask_stats[name]
            accumulator["author_nonzero_count"] += int(
                np.count_nonzero(author_length[local] > 0.0)
            )
            accumulator["b0_hit_count"] += int(np.count_nonzero(b0_hit[local]))
            accumulator["b1_hit_count"] += int(np.count_nonzero(b1_hit[local]))
            accumulator["b1_removed_from_b0_count"] += int(
                np.count_nonzero(b1_removed[local])
            )
            accumulator["author_length_sum_m"] += float(author_length[local].sum())
            accumulator["b0_length_sum_m"] += float(b0_length[local].sum())
            accumulator["b1_length_sum_m"] += float(b1_length[local].sum())

    counts["b0_zero_length_count"] = rows - counts["b0_hit_count"]
    counts["b1_zero_length_count"] = rows - counts["b1_hit_count"]
    invariants_pass = all(
        counts[key] == 0
        for key in (
            "b0_endpoint_box_violation_count",
            "b1_endpoint_box_violation_count",
            "b1_endpoint_nappe_violation_count",
            "b1_endpoint_cone_radial_violation_count",
            "b1_midpoint_cone_violation_count",
            "b1_hit_without_b0_hit_count",
            "b1_length_exceeds_b0_count",
            "nonfinite_output_count",
        )
    ) and counts["ray_count"] == rows
    author_length_sum = sums["author_selected_length_sum_m"]
    b0_length_sum = sums["b0_length_sum_m"]
    b1_length_sum = sums["b1_length_sum_m"]

    return {
        "schema_version": REPORT_SCHEMA,
        "status": (
            "B0_B1_FIXED_DOMAIN_ANALYTIC_CONTRACT_PASS_B2_REQUIRED"
            if invariants_pass
            else "B0_B1_FIXED_DOMAIN_ANALYTIC_CONTRACT_INVALID"
        ),
        "evidence_scope": "REAL_ONE_VIEW_B0_FORWARD_BOX_AND_B1_ONE_NAPPE_CONE_BOX_RAY_CENSUS_NO_TENSORFLOW_NO_RECONSTRUCTION",
        "view_id_zero_based": metadata["view_id_zero_based"],
        "source": {
            "bundle_manifest_sha256": metadata["bundle_manifest_sha256"],
            "setup_manifest_sha256": metadata["setup_manifest_sha256"],
            "corrected_mask_manifest_sha256": metadata["mask_manifest_sha256"],
            "geometry_implementation_filename": Path(__file__).with_name(
                "psu_bost_forward_geometry.py"
            ).name,
            "geometry_implementation_sha256": _sha256_file(
                Path(__file__).with_name("psu_bost_forward_geometry.py")
            ),
            "audit_implementation_sha256": _sha256_file(Path(__file__)),
        },
        "configuration": {
            "rows": rows,
            "chunk_rows": chunk_rows,
            "outer_minimum_m": list(outer_minimum),
            "outer_maximum_m": list(outer_maximum),
            "cone_vertex_m": list(cone_vertex),
            "cone_axis_normalized": axis.tolist(),
            "cone_angle_degrees": cone_angle_degrees,
            "geometry_contract_version": GEOMETRY_CONTRACT_VERSION,
            "b0_policy": "FORWARD_NORMALIZED_RAY_INTERSECT_CLOSED_AXIS_ALIGNED_BOX",
            "b1_policy": "B0_INTERSECT_NORMALIZED_ONE_NAPPE_CONE_NO_FALLBACK",
        },
        "counts": counts,
        "path_length": {
            **sums,
            "b1_fraction_of_b0": _safe_fraction(b1_length_sum, b0_length_sum),
            "b1_fraction_of_author_selected": _safe_fraction(
                b1_length_sum, author_length_sum
            ),
            "b0_fraction_of_author_selected": _safe_fraction(
                b0_length_sum, author_length_sum
            ),
        },
        "mask_conditioned": {
            name: _finalize_mask_accumulator(value)
            for name, value in mask_stats.items()
        },
        "runtime_observation": {
            "wall_seconds": time.perf_counter() - started,
            "scope": "CACHED_LOCAL_DIAGNOSTIC_NOT_A_SPEED_BENCHMARK",
        },
        "decision": {
            "analytic_domain_invariants_pass": invariants_pass,
            "declared_computational_domain_mechanically_enforced": invariants_pass,
            "physical_spatial_domain_validated": False,
            "cone_parameter_physical_semantics_confirmed": False,
            "finite_aperture_sample_support_audited": False,
            "training_ready": "NO",
            "algorithm_superiority_claim": "LOCKED",
            "next_gate": "B2_FINITE_APERTURE_DOMAIN_INDICATOR_AND_B3_GEOMETRY_SAFE_MASK",
        },
        "limitations": [
            "B1 reuses the released cone vertex, axis, and angle only as a computational-domain hypothesis; their physical meaning is not independently confirmed",
            "centerline domain validity does not prove that finite-aperture samples stay inside the same domain",
            "the active and inactive masks are diagnostic labels rather than density or refractive-index ground truth",
            "no held-out camera, TensorFlow NIRT, neural field, inverse reconstruction, or superiority comparison is run",
        ],
        "upstream_view_contract": {
            "bundle_status": metadata["bundle_status"],
            "setup_status": metadata["setup_status"],
            "mask_status": metadata["mask_status"],
        },
    }


def aggregate_fixed_domain_views(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("at least one view record is required")
    view_ids = [int(record["view_id_zero_based"]) for record in records]
    if view_ids != list(range(len(records))):
        raise ValueError("view ids must be the ordered contiguous range from zero")
    invalid = [
        view_id
        for view_id, record in zip(view_ids, records)
        if record["status"] != "B0_B1_FIXED_DOMAIN_ANALYTIC_CONTRACT_PASS_B2_REQUIRED"
    ]

    count_keys = records[0]["counts"].keys()
    pooled_counts = {
        key: sum(int(record["counts"][key]) for record in records)
        for key in count_keys
    }
    length_keys = (
        "author_selected_length_sum_m",
        "b0_length_sum_m",
        "b1_length_sum_m",
    )
    pooled_lengths = {
        key: sum(float(record["path_length"][key]) for record in records)
        for key in length_keys
    }
    pooled_masks: dict[str, dict[str, Any]] = {}
    for mask_name in ("amask_all", "imask_all"):
        accumulator = _empty_mask_accumulator(0)
        for record in records:
            view_mask = record["mask_conditioned"][mask_name]
            for key in accumulator:
                accumulator[key] += view_mask[key]
        pooled_masks[mask_name] = _finalize_mask_accumulator(accumulator)
    b0_length = pooled_lengths["b0_length_sum_m"]
    b1_length = pooled_lengths["b1_length_sum_m"]
    author_length = pooled_lengths["author_selected_length_sum_m"]
    active_b1_zero_views = [
        view_id
        for view_id, record in zip(view_ids, records)
        if int(record["mask_conditioned"]["amask_all"]["b1_hit_count"])
        < int(record["mask_conditioned"]["amask_all"]["count"])
    ]
    upstream_author_setup_no_go_views = [
        view_id
        for view_id, record in zip(view_ids, records)
        if "NO_GO" in str(record["upstream_view_contract"]["setup_status"])
    ]
    return {
        "schema_version": AGGREGATE_SCHEMA,
        "execution_status": "COMPLETE",
        "scientific_verdict": (
            "INVALID"
            if invalid
            else "MECHANICAL_PASS_PHYSICAL_CONE_SEMANTICS_AND_FINITE_APERTURE_UNCONFIRMED"
        ),
        "status": (
            "B0_B1_ALL_VIEW_ANALYTIC_CONTRACT_INVALID"
            if invalid
            else "B0_B1_ALL_VIEW_ANALYTIC_CONTRACT_PASS_B2_REQUIRED"
        ),
        "view_count": len(records),
        "views": list(records),
        "aggregate": {
            "invalid_view_ids": invalid,
            "active_mask_b1_zero_view_ids": active_b1_zero_views,
            "upstream_author_setup_no_go_view_ids": upstream_author_setup_no_go_views,
            "counts": pooled_counts,
            "path_length": {
                **pooled_lengths,
                "b1_fraction_of_b0": _safe_fraction(b1_length, b0_length),
                "b1_fraction_of_author_selected": _safe_fraction(
                    b1_length, author_length
                ),
                "b0_fraction_of_author_selected": _safe_fraction(
                    b0_length, author_length
                ),
            },
            "mask_conditioned": pooled_masks,
        },
        "decision": {
            "analytic_domain_invariants_pass": not invalid,
            "declared_computational_domain_mechanically_enforced": not invalid,
            "physical_spatial_domain_validated": False,
            "author_mixed_domain_length_comparison_is_context_only": True,
            "cone_parameter_physical_semantics_confirmed": False,
            "finite_aperture_sample_support_audited": False,
            "training_ready": "NO",
            "algorithm_superiority_claim": "LOCKED",
            "next_gate": "B2_FINITE_APERTURE_DOMAIN_INDICATOR_AND_B3_GEOMETRY_SAFE_MASK",
        },
        "limitations": [
            "B0/B1 are deterministic geometry baselines and do not measure reconstruction quality",
            "the released 25 degree cone is treated as a computational sampling hull, not a shock or Mach angle",
            "B2 finite-aperture support and held-out reprojection are required before inverse or neural-operator comparison",
            "no statistical uncertainty interval applies to this exhaustive ray census",
        ],
    }


def write_metrics_csv(path: Path, report: Mapping[str, Any]) -> None:
    fieldnames = [
        "view_id_zero_based",
        "ray_count",
        "b0_hit_fraction",
        "b1_hit_fraction",
        "b1_removed_from_b0_fraction",
        "b1_path_fraction_of_b0",
        "active_b1_hit_fraction",
        "inactive_b1_hit_fraction",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    with partial.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for view in report["views"]:
            counts = view["counts"]
            ray_count = int(counts["ray_count"])
            writer.writerow(
                {
                    "view_id_zero_based": view["view_id_zero_based"],
                    "ray_count": ray_count,
                    "b0_hit_fraction": _safe_fraction(counts["b0_hit_count"], ray_count),
                    "b1_hit_fraction": _safe_fraction(counts["b1_hit_count"], ray_count),
                    "b1_removed_from_b0_fraction": _safe_fraction(
                        counts["b1_removed_from_b0_count"], ray_count
                    ),
                    "b1_path_fraction_of_b0": view["path_length"][
                        "b1_fraction_of_b0"
                    ],
                    "active_b1_hit_fraction": view["mask_conditioned"][
                        "amask_all"
                    ]["b1_hit_fraction"],
                    "inactive_b1_hit_fraction": view["mask_conditioned"][
                        "imask_all"
                    ]["b1_hit_fraction"],
                }
            )
    os.replace(partial, path)


def run_all_view_fixed_domain_audit(
    *,
    audit_root: Path,
    output_path: Path,
    csv_output_path: Path,
    view_count: int,
    chunk_rows: int = 65_536,
) -> dict[str, Any]:
    records = [
        audit_fixed_domain_view(
            view_dir=audit_root / f"view_{view_id:02d}",
            chunk_rows=chunk_rows,
        )
        for view_id in range(view_count)
    ]
    report = aggregate_fixed_domain_views(records)
    _atomic_json(output_path, report)
    write_metrics_csv(csv_output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--view-count", type=int, required=True)
    parser.add_argument("--chunk-rows", type=int, default=65_536)
    args = parser.parse_args()
    report = run_all_view_fixed_domain_audit(
        audit_root=args.audit_root,
        output_path=args.output,
        csv_output_path=args.csv_output,
        view_count=args.view_count,
        chunk_rows=args.chunk_rows,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["scientific_verdict"] != "INVALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())
