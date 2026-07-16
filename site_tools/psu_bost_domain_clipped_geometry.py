#!/usr/bin/env python3
"""Audit the A1 author-compatible clipped-hybrid ablation on PSU BOST rays."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from .official_nirt_geometry_fixture import load_numpy_geometry_functions
    from .psu_bost_mat_stream import _sha256_file
    from .psu_bost_streamed_setup import SCALAR_FIELDS, VECTOR_FIELDS
except ImportError:  # Direct script execution.
    from official_nirt_geometry_fixture import (  # type: ignore[no-redef]
        load_numpy_geometry_functions,
    )
    from psu_bost_mat_stream import _sha256_file  # type: ignore[no-redef]
    from psu_bost_streamed_setup import (  # type: ignore[no-redef]
        SCALAR_FIELDS,
        VECTOR_FIELDS,
    )


REPORT_SCHEMA = "psu-bost-author-compatible-clipped-hybrid-audit-1.0"
VIEW_BUNDLE_STATUS = "VIEW_SHARD_BUNDLE_TRANSCODED_AND_SOURCE_STREAMS_VERIFIED"
MASK_STATUS = "CORRECTED_VIEW_MASK_SHARDS_BUILT_MECHANICAL_CONTRACT_PASS"


def _as_1d(value: np.ndarray | Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def clip_author_intervals_to_forward_box(
    *,
    full_first: np.ndarray | Sequence[float],
    full_second: np.ndarray | Sequence[float],
    full_length: np.ndarray | Sequence[float],
    cone_first: np.ndarray | Sequence[float],
    cone_second: np.ndarray | Sequence[float],
    cone_length: np.ndarray | Sequence[float],
    tolerance: float = 1e-12,
) -> dict[str, np.ndarray]:
    """Clip author cone intervals to the forward-facing reconstruction box.

    The author's fallback is preserved: a zero cone length uses the box interval.
    A nonzero cone interval that has no forward box overlap becomes an explicit
    zero-length row rather than sampling an undefined field outside the domain.
    """

    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and nonnegative")
    arrays = {
        "full_first": _as_1d(full_first, "full_first"),
        "full_second": _as_1d(full_second, "full_second"),
        "full_length": _as_1d(full_length, "full_length"),
        "cone_first": np.asarray(cone_first, dtype=np.float64).reshape(-1),
        "cone_second": np.asarray(cone_second, dtype=np.float64).reshape(-1),
        "cone_length": _as_1d(cone_length, "cone_length"),
    }
    sizes = {array.size for array in arrays.values()}
    if len(sizes) != 1:
        raise ValueError("all interval arrays must have the same size")

    box_low_raw = np.minimum(arrays["full_first"], arrays["full_second"])
    box_high = np.maximum(arrays["full_first"], arrays["full_second"])
    box_low = np.maximum(box_low_raw, 0.0)
    box_valid = (
        (arrays["full_length"] > tolerance)
        & (box_high > box_low + tolerance)
        & np.isfinite(box_low)
        & np.isfinite(box_high)
    )

    cone_low_raw = np.minimum(arrays["cone_first"], arrays["cone_second"])
    cone_high = np.maximum(arrays["cone_first"], arrays["cone_second"])
    cone_low = np.maximum(cone_low_raw, 0.0)
    cone_valid = (
        (arrays["cone_length"] > tolerance)
        & np.isfinite(cone_low)
        & np.isfinite(cone_high)
        & (cone_high > cone_low + tolerance)
    )
    fallback = ~cone_valid

    overlap_low = np.maximum(cone_low, box_low)
    overlap_high = np.minimum(cone_high, box_high)
    cone_overlap_valid = (
        cone_valid & box_valid & (overlap_high > overlap_low + tolerance)
    )
    selected_valid = np.where(cone_valid, cone_overlap_valid, box_valid)
    selected_low = np.where(cone_valid, overlap_low, box_low)
    selected_high = np.where(cone_valid, overlap_high, box_high)
    selected_low = np.where(selected_valid, selected_low, 0.0)
    selected_high = np.where(selected_valid, selected_high, 0.0)
    selected_length = np.where(selected_valid, selected_high - selected_low, 0.0)

    author_low = np.where(fallback, box_low_raw, cone_low_raw)
    author_high = np.where(fallback, box_high, cone_high)
    author_length = np.where(
        fallback, arrays["full_length"], arrays["cone_length"]
    )
    changed = (
        np.abs(selected_low - author_low) > tolerance
    ) | (np.abs(selected_high - author_high) > tolerance)

    return {
        "enter": selected_low,
        "exit": selected_high,
        "length": selected_length,
        "valid": selected_valid,
        "box_valid": box_valid,
        "cone_valid": cone_valid,
        "fallback": fallback,
        "cone_overlap_valid": cone_overlap_valid,
        "cone_shortened": cone_overlap_valid
        & (selected_length < arrays["cone_length"] - tolerance),
        "cone_zeroed_for_no_box_overlap": cone_valid & ~cone_overlap_valid,
        "forward_box_shortened": fallback
        & box_valid
        & (selected_length < arrays["full_length"] - tolerance),
        "changed_from_author": changed,
        "author_length": author_length,
    }


def _open_view(view_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    bundle_dir = view_dir / "bundle"
    manifest_path = bundle_dir / "view_bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != VIEW_BUNDLE_STATUS:
        raise ValueError(f"view bundle is not verified: {view_dir}")
    fields = {
        name: np.load(bundle_dir / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in VECTOR_FIELDS + SCALAR_FIELDS
    }
    rows = {int(value.shape[0]) for value in fields.values()}
    if len(rows) != 1:
        raise ValueError(f"view field row counts disagree: {view_dir}")
    return fields, manifest


def _open_masks(view_dir: Path, rows: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    mask_dir = view_dir / "corrected_masks"
    manifest_path = mask_dir / "corrected_view_masks_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != MASK_STATUS:
        raise ValueError(f"corrected masks are not verified: {view_dir}")
    masks = {
        variable: np.load(
            mask_dir / f"{variable}_zero_based.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        for variable in ("amask_all", "imask_all")
    }
    for variable, indices in masks.items():
        if indices.dtype != np.int64 or indices.ndim != 1:
            raise ValueError(f"{variable} must be a one-dimensional int64 array")
        if indices.size and (indices[0] < 0 or indices[-1] >= rows):
            raise ValueError(f"{variable} contains an out-of-range index")
    return masks, manifest


def _empty_mask_accumulator(total: int) -> dict[str, float | int]:
    return {
        "count": total,
        "changed_count": 0,
        "zeroed_count": 0,
        "shortened_count": 0,
        "author_length_sum_m": 0.0,
        "clipped_length_sum_m": 0.0,
    }


def _finalize_mask_accumulator(accumulator: Mapping[str, float | int]) -> dict[str, Any]:
    count = int(accumulator["count"])
    author_sum = float(accumulator["author_length_sum_m"])
    clipped_sum = float(accumulator["clipped_length_sum_m"])
    return {
        **accumulator,
        "changed_fraction": int(accumulator["changed_count"]) / count
        if count
        else None,
        "zeroed_fraction": int(accumulator["zeroed_count"]) / count
        if count
        else None,
        "shortened_fraction": int(accumulator["shortened_count"]) / count
        if count
        else None,
        "path_length_retained_fraction": clipped_sum / author_sum
        if author_sum
        else None,
    }


def audit_domain_clipped_view(
    *,
    view_dir: Path,
    geometry_source: Path,
    chunk_rows: int = 65_536,
    outer_minimum: tuple[float, float, float] = (-0.110, -0.110, -0.110),
    outer_maximum: tuple[float, float, float] = (0.110, 0.110, 0.110),
    cone_vertex: tuple[float, float, float] = (0.060, 0.015, 0.0),
    cone_axis: tuple[float, float, float] = (-1.0, -0.1, 0.0),
    cone_angle_degrees: float = 25.0,
) -> dict[str, Any]:
    if chunk_rows < 2:
        raise ValueError("chunk_rows must be at least two")
    fields, bundle_manifest = _open_view(view_dir)
    rows = int(fields["c"].shape[0])
    masks, mask_manifest = _open_masks(view_dir, rows)
    functions = load_numpy_geometry_functions(geometry_source)
    box = functions["rayBoxIntersection"]
    cone = functions["rayConeIntersection"]
    lower = np.asarray(outer_minimum, dtype=np.float64)[:, None]
    upper = np.asarray(outer_maximum, dtype=np.float64)[:, None]
    vertex = np.asarray(cone_vertex, dtype=np.float64)[:, None]
    axis = np.asarray(cone_axis, dtype=np.float64)[:, None]
    axis = axis / np.linalg.norm(axis)
    angle = np.radians(cone_angle_degrees)

    counts = {
        "ray_count": 0,
        "author_nonzero_count": 0,
        "clipped_nonzero_count": 0,
        "changed_from_author_count": 0,
        "cone_shortened_count": 0,
        "cone_zeroed_for_no_box_overlap_count": 0,
        "forward_box_shortened_count": 0,
        "clipped_endpoint_box_violation_count": 0,
        "clipped_length_exceeds_author_count": 0,
        "nonfinite_output_count": 0,
        "negative_clipped_inner_aperture_radius_count": 0,
        "negative_clipped_outer_aperture_radius_count": 0,
    }
    sums = {"author_length_sum_m": 0.0, "clipped_length_sum_m": 0.0}
    mask_stats = {
        variable: _empty_mask_accumulator(int(indices.size))
        for variable, indices in masks.items()
    }
    started = time.perf_counter()
    tolerance = 1e-10

    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        origins = np.asarray(fields["c"][start:stop], dtype=np.float64).T
        directions = np.asarray(fields["v"][start:stop], dtype=np.float64).T
        full_first, full_second, full_length = box(origins, directions, lower, upper)
        cone_first, cone_second, cone_length = cone(
            origins, directions, vertex, axis, angle
        )
        clipped = clip_author_intervals_to_forward_box(
            full_first=full_first,
            full_second=full_second,
            full_length=full_length,
            cone_first=cone_first,
            cone_second=cone_second,
            cone_length=cone_length,
        )
        enter = clipped["enter"]
        exit_ = clipped["exit"]
        length = clipped["length"]
        author_length = clipped["author_length"]
        valid = clipped["valid"]
        intersection = origins + enter[None, :] * directions
        endpoint = origins + exit_[None, :] * directions
        inside = (
            np.all(intersection >= lower - tolerance, axis=0)
            & np.all(intersection <= upper + tolerance, axis=0)
            & np.all(endpoint >= lower - tolerance, axis=0)
            & np.all(endpoint <= upper + tolerance, axis=0)
        )
        box_violations = valid & ~inside

        rap = np.asarray(fields["Rapvec"][start:stop, 0], dtype=np.float64)
        df = np.asarray(fields["Dfvec"][start:stop, 0], dtype=np.float64)
        inner = rap * (1.0 - enter / df)
        outer = rap * (1.0 - exit_ / df)

        counts["ray_count"] += stop - start
        counts["author_nonzero_count"] += int(np.count_nonzero(author_length > 0))
        counts["clipped_nonzero_count"] += int(np.count_nonzero(length > 0))
        counts["changed_from_author_count"] += int(
            np.count_nonzero(clipped["changed_from_author"])
        )
        counts["cone_shortened_count"] += int(
            np.count_nonzero(clipped["cone_shortened"])
        )
        counts["cone_zeroed_for_no_box_overlap_count"] += int(
            np.count_nonzero(clipped["cone_zeroed_for_no_box_overlap"])
        )
        counts["forward_box_shortened_count"] += int(
            np.count_nonzero(clipped["forward_box_shortened"])
        )
        counts["clipped_endpoint_box_violation_count"] += int(
            np.count_nonzero(box_violations)
        )
        counts["clipped_length_exceeds_author_count"] += int(
            np.count_nonzero(length > author_length + tolerance)
        )
        counts["nonfinite_output_count"] += int(
            np.count_nonzero(~np.isfinite(length))
            + np.count_nonzero(~np.isfinite(intersection[:, valid]))
            + np.count_nonzero(~np.isfinite(endpoint[:, valid]))
            + np.count_nonzero(~np.isfinite(inner[valid]))
            + np.count_nonzero(~np.isfinite(outer[valid]))
        )
        counts["negative_clipped_inner_aperture_radius_count"] += int(
            np.count_nonzero(inner[valid] < 0)
        )
        counts["negative_clipped_outer_aperture_radius_count"] += int(
            np.count_nonzero(outer[valid] < 0)
        )
        sums["author_length_sum_m"] += float(author_length.sum())
        sums["clipped_length_sum_m"] += float(length.sum())

        changed = clipped["changed_from_author"]
        zeroed = (author_length > 0) & (length == 0)
        shortened = (length > 0) & (length < author_length - tolerance)
        for variable, indices in masks.items():
            left = int(np.searchsorted(indices, start, side="left"))
            right = int(np.searchsorted(indices, stop, side="left"))
            local = np.asarray(indices[left:right] - start, dtype=np.int64)
            if not local.size:
                continue
            accumulator = mask_stats[variable]
            accumulator["changed_count"] += int(np.count_nonzero(changed[local]))
            accumulator["zeroed_count"] += int(np.count_nonzero(zeroed[local]))
            accumulator["shortened_count"] += int(
                np.count_nonzero(shortened[local])
            )
            accumulator["author_length_sum_m"] += float(author_length[local].sum())
            accumulator["clipped_length_sum_m"] += float(length[local].sum())

    author_sum = sums["author_length_sum_m"]
    clipped_sum = sums["clipped_length_sum_m"]
    domain_contract_pass = all(
        (
            counts["ray_count"] == rows,
            counts["clipped_endpoint_box_violation_count"] == 0,
            counts["clipped_length_exceeds_author_count"] == 0,
            counts["nonfinite_output_count"] == 0,
            counts["negative_clipped_inner_aperture_radius_count"] == 0,
            counts["negative_clipped_outer_aperture_radius_count"] == 0,
        )
    )
    zero_rows = rows - counts["clipped_nonzero_count"]
    view_id = int(bundle_manifest["view"]["view_id_zero_based"])
    return {
        "schema_version": REPORT_SCHEMA,
        "status": (
            "AUTHOR_COMPATIBLE_CLIPPED_HYBRID_CONTRACT_PASS_MASK_FILTER_REQUIRED"
            if domain_contract_pass and zero_rows
            else "AUTHOR_COMPATIBLE_CLIPPED_HYBRID_CONTRACT_PASS"
            if domain_contract_pass
            else "AUTHOR_COMPATIBLE_CLIPPED_HYBRID_INVALID"
        ),
        "evidence_scope": "REAL_ONE_VIEW_A1_AUTHOR_DOUBLE_CONE_INTERVAL_CLIPPED_TO_FORWARD_BOX_WITH_AUTHOR_FALLBACK_NO_TENSORFLOW_NO_RECONSTRUCTION",
        "view_id_zero_based": view_id,
        "source": {
            "view_bundle_manifest_sha256": hashlib.sha256(
                (view_dir / "bundle" / "view_bundle_manifest.json").read_bytes()
            ).hexdigest(),
            "corrected_mask_manifest_sha256": hashlib.sha256(
                (
                    view_dir
                    / "corrected_masks"
                    / "corrected_view_masks_manifest.json"
                ).read_bytes()
            ).hexdigest(),
            "geometry_source_filename": geometry_source.name,
            "geometry_source_sha256": _sha256_file(geometry_source),
            "author_source_modified": False,
        },
        "configuration": {
            "rows": rows,
            "chunk_rows": chunk_rows,
            "outer_minimum_m": list(outer_minimum),
            "outer_maximum_m": list(outer_maximum),
            "cone_vertex_m": list(cone_vertex),
            "cone_axis_normalized": axis[:, 0].tolist(),
            "cone_angle_degrees": cone_angle_degrees,
            "policy": "AUTHOR_CONE_INTERVAL_INTERSECT_FORWARD_BOX_AUTHOR_ZERO_CONE_FALLBACK_TO_FORWARD_BOX",
        },
        "counts": {**counts, "clipped_zero_length_count": zero_rows},
        "path_length": {
            **sums,
            "removed_length_sum_m": author_sum - clipped_sum,
            "retained_fraction": clipped_sum / author_sum if author_sum else None,
            "removed_fraction": 1.0 - clipped_sum / author_sum
            if author_sum
            else None,
        },
        "mask_conditioned": {
            variable: _finalize_mask_accumulator(accumulator)
            for variable, accumulator in mask_stats.items()
        },
        "runtime_observation": {
            "wall_seconds": time.perf_counter() - started,
            "scope": "CACHED_LOCAL_DIAGNOSTIC_NOT_A_SPEED_BENCHMARK",
        },
        "decision": {
            "positive_segments_inside_forward_box": domain_contract_pass,
            "geometry_safe_zero_row_filter_required": bool(zero_rows),
            "fixed_spatial_domain_established": False,
            "training_ready": "NO",
            "algorithm_superiority_claim": "LOCKED",
            "next_gate": "GEOMETRY_SAFE_MASK_ABLATION_AND_HELD_OUT_REPROJECTION",
        },
        "limitations": [
            "A1 is an author-compatibility ablation, not the defensible B0 box or B1 one-nappe cone-box fixed-domain baseline",
            "the ablation preserves the author cone roots, double-cone primitive, and cone-miss-to-box fallback",
            "zero-length rows must be filtered explicitly and are not valid line-of-sight samples",
            "mask-conditioned counts are diagnostic labels rather than density ground truth",
            "no TensorFlow loss, neural field, held-out reprojection, inverse reconstruction, or superiority comparison is run",
        ],
        "upstream_view_contract": {
            "bundle_status": bundle_manifest["status"],
            "mask_status": mask_manifest["status"],
        },
    }


def aggregate_domain_clipped_views(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("at least one view record is required")
    view_ids = [int(record["view_id_zero_based"]) for record in records]
    if view_ids != list(range(len(records))):
        raise ValueError("view ids must be the ordered contiguous range from zero")
    invalid = [
        view_id
        for view_id, record in zip(view_ids, records)
        if not str(record["status"]).startswith(
            "AUTHOR_COMPATIBLE_CLIPPED_HYBRID_CONTRACT_PASS"
        )
    ]
    filter_views = [
        view_id
        for view_id, record in zip(view_ids, records)
        if record["decision"]["geometry_safe_zero_row_filter_required"]
    ]
    total_author = sum(float(record["path_length"]["author_length_sum_m"]) for record in records)
    total_clipped = sum(float(record["path_length"]["clipped_length_sum_m"]) for record in records)
    return {
        "schema_version": "psu-bost-author-compatible-clipped-hybrid-all-view-audit-1.0",
        "execution_status": "COMPLETE",
        "scientific_verdict": (
            "INVALID" if invalid else "AUTHOR_COMPATIBILITY_ABLATION_ONLY"
        ),
        "status": (
            "AUTHOR_COMPATIBLE_CLIPPED_HYBRID_ALL_VIEW_INVALID"
            if invalid
            else "AUTHOR_COMPATIBLE_CLIPPED_HYBRID_ALL_VIEW_CONTRACT_PASS_MASK_FILTER_REQUIRED"
            if filter_views
            else "AUTHOR_COMPATIBLE_CLIPPED_HYBRID_ALL_VIEW_CONTRACT_PASS"
        ),
        "view_count": len(records),
        "views": list(records),
        "aggregate": {
            "invalid_view_ids": invalid,
            "zero_row_filter_required_view_ids": filter_views,
            "author_length_sum_m": total_author,
            "clipped_length_sum_m": total_clipped,
            "removed_length_sum_m": total_author - total_clipped,
            "retained_fraction": total_clipped / total_author if total_author else None,
            "removed_fraction": 1.0 - total_clipped / total_author
            if total_author
            else None,
            "changed_ray_count": sum(
                int(record["counts"]["changed_from_author_count"])
                for record in records
            ),
            "clipped_zero_length_count": sum(
                int(record["counts"]["clipped_zero_length_count"])
                for record in records
            ),
        },
        "decision": {
            "domain_clipping_mechanically_valid": not invalid,
            "fixed_spatial_domain_established": False,
            "training_ready": "NO",
            "algorithm_superiority_claim": "LOCKED",
            "next_gate": "GEOMETRY_SAFE_MASK_ABLATION_AND_HELD_OUT_REPROJECTION",
        },
        "limitations": [
            "A1 only isolates clipping while preserving the author's hybrid domain and fallback semantics",
            "domain consistency is a necessary forward-model contract, not evidence that reconstruction improves",
            "the baseline has not been compared against NeRIF, NIRT, DeepONet, FNO, or classical tomography",
            "the all-view aggregate contains deterministic ray censuses and no statistical uncertainty estimate",
        ],
    }


def run_all_view_domain_clipped_audit(
    *,
    audit_root: Path,
    geometry_source: Path,
    output_path: Path,
    view_count: int,
    chunk_rows: int = 65_536,
) -> dict[str, Any]:
    records = [
        audit_domain_clipped_view(
            view_dir=audit_root / f"view_{view_id:02d}",
            geometry_source=geometry_source,
            chunk_rows=chunk_rows,
        )
        for view_id in range(view_count)
    ]
    report = aggregate_domain_clipped_views(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(f".{output_path.name}.partial")
    partial.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, output_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--geometry-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--view-count", type=int, required=True)
    parser.add_argument("--chunk-rows", type=int, default=65_536)
    args = parser.parse_args()
    report = run_all_view_domain_clipped_audit(
        audit_root=args.audit_root,
        geometry_source=args.geometry_source,
        output_path=args.output,
        view_count=args.view_count,
        chunk_rows=args.chunk_rows,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["scientific_verdict"] != "INVALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())
