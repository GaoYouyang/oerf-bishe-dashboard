#!/usr/bin/env python3
"""Audit the private PSU support-view identity and two-grid cache contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from site_tools.psu_b0_compact_cache import PSUCompactCachedRayStore


PRIVATE_SCHEMA = "psu-support-rotation-loro-preflight-private-1.0"
PUBLIC_SCHEMA = "psu-support-rotation-loro-preflight-public-1.0"
ROTATIONS = (0, 50, 90)
CAMERAS = (2, 3, 4)
MEASUREMENTS_PER_VIEW = 2160 * 2560
COMMON_CACHE_ARRAYS = (
    "observations_uv",
    "projection_uv_xyz",
    "ray_scale",
    "valid",
)


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_view_mapping() -> list[dict[str, int]]:
    """Return the source-order mapping implied by the two official MATLAB loops."""

    return [
        {
            "view_id_zero_based": view_id,
            "rotation_degrees": rotation,
            "camera_id": camera,
        }
        for view_id, (rotation, camera) in enumerate(
            (rotation, camera)
            for rotation in ROTATIONS
            for camera in CAMERAS
        )
    ]


def audit_source_scripts(script_root: Path) -> dict[str, Any]:
    auto_path = script_root / "AEDC_pprocess_auto.m"
    per_rotation_path = script_root / "AEDC_pprocess.m"
    auto = auto_path.read_text(encoding="utf-8")
    per_rotation = per_rotation_path.read_text(encoding="utf-8")
    gates = {
        "outer_rotation_order_is_0_50_90": auto.count("ang_list = [0 50 90]") >= 2,
        "outer_loop_iterates_rotation_order": "for j = ang_list" in auto,
        "positive_rotations_append_after_rotation_zero": (
            "if j == 0" in auto
            and "if j > 0" in auto
            and "c = [c tmpfile.c]" in auto
            and "epsu_all = [epsu_all tmpfile.epsu_all]" in auto
        ),
        "inner_camera_order_is_2_3_4": "cincl = 2:4" in per_rotation,
        "inner_loop_iterates_camera_order": "for q = cincl" in per_rotation,
        "camera_fields_append_in_loop_order": (
            "c = [c camc .* ones(3,camn)]" in per_rotation
            and "epsu_all = [epsu_all epsu(:)']" in per_rotation
        ),
    }
    return {
        "source_files": {
            "rotation_outer_loop": {
                "filename": auto_path.name,
                "sha256": _sha256(auto_path),
            },
            "camera_inner_loop": {
                "filename": per_rotation_path.name,
                "sha256": _sha256(per_rotation_path),
            },
        },
        "gates": gates,
        "mapping": expected_view_mapping(),
    }


def audit_view_manifests(view_root: Path) -> dict[str, Any]:
    rows: list[dict[str, int]] = []
    manifest_hashes: dict[str, str] = {}
    for view_id in range(9):
        path = view_root / f"view_{view_id:02d}" / "bundle" / "view_bundle_manifest.json"
        manifest = _json_object(path)
        view = manifest.get("view")
        if not isinstance(view, dict):
            raise ValueError(f"view metadata missing: {path}")
        rows.append(
            {
                "view_id_zero_based": int(view["view_id_zero_based"]),
                "measurement_start": int(view["measurement_start"]),
                "measurement_stop": int(view["measurement_stop"]),
                "measurement_count": int(view["measurement_count"]),
            }
        )
        manifest_hashes[f"view_{view_id:02d}"] = _sha256(path)
    expected_rows = [
        {
            "view_id_zero_based": view_id,
            "measurement_start": view_id * MEASUREMENTS_PER_VIEW,
            "measurement_stop": (view_id + 1) * MEASUREMENTS_PER_VIEW,
            "measurement_count": MEASUREMENTS_PER_VIEW,
        }
        for view_id in range(9)
    ]
    return {
        "rows": rows,
        "manifest_sha256_private_only": manifest_hashes,
        "gates": {
            "nine_ordered_contiguous_view_blocks": rows == expected_rows,
            "source_mat_filename_consistent": all(
                _json_object(
                    view_root
                    / f"view_{view_id:02d}"
                    / "bundle"
                    / "view_bundle_manifest.json"
                ).get("source", {}).get("filename")
                == "HSOF_9CAM_RT.mat"
                for view_id in range(9)
            ),
        },
    }


def _total_cache_bytes(manifest: dict[str, Any]) -> int:
    arrays = manifest.get("arrays")
    if not isinstance(arrays, dict):
        raise ValueError("cache array records are missing")
    return sum(int(record["nbytes"]) for record in arrays.values())


def cross_grid_coordinate_diagnostics(
    cache16: PSUCompactCachedRayStore,
    cache32: PSUCompactCachedRayStore,
) -> dict[str, float | int]:
    """Compare normalized Cartesian sample coordinates without materializing them."""

    chunks16 = cache16.manifest["chunks"]
    chunks32 = cache32.manifest["chunks"]
    if chunks16 != chunks32:
        raise ValueError("cache chunk partitions differ across grids")
    if not np.array_equal(cache16.valid, cache32.valid):
        raise ValueError("cache valid-sample masks differ across grids")

    shape16_xyz = tuple(reversed(cache16.grid_shape))
    shape32_xyz = tuple(reversed(cache32.grid_shape))
    maximum = 0.0
    squared_sum = 0.0
    coordinate_count = 0
    valid_sample_count = 0
    for row in chunks16:
        start = int(row["start_index"])
        stop = int(row["stop_index"])
        valid = np.asarray(cache16.valid[start:stop], dtype=np.bool_)
        valid_sample_count += int(np.count_nonzero(valid))
        base16 = np.asarray(cache16.base_indices[start:stop], dtype=np.int64)
        base32 = np.asarray(cache32.base_indices[start:stop], dtype=np.int64)
        fractions16 = np.asarray(cache16.fractions_xyz[start:stop], dtype=np.float64)
        fractions32 = np.asarray(cache32.fractions_xyz[start:stop], dtype=np.float64)
        lower16 = (
            base16 % shape16_xyz[0],
            (base16 // shape16_xyz[0]) % shape16_xyz[1],
            base16 // (shape16_xyz[0] * shape16_xyz[1]),
        )
        lower32 = (
            base32 % shape32_xyz[0],
            (base32 // shape32_xyz[0]) % shape32_xyz[1],
            base32 // (shape32_xyz[0] * shape32_xyz[1]),
        )
        for axis in range(3):
            normalized16 = (
                lower16[axis] + fractions16[..., axis]
            ) / float(shape16_xyz[axis] - 1)
            normalized32 = (
                lower32[axis] + fractions32[..., axis]
            ) / float(shape32_xyz[axis] - 1)
            difference = np.abs(normalized16[valid] - normalized32[valid])
            if difference.size:
                maximum = max(maximum, float(np.max(difference)))
                squared_sum += float(np.sum(difference * difference, dtype=np.float64))
                coordinate_count += int(difference.size)
    if coordinate_count < 1:
        raise ValueError("cache contains no valid sample coordinates")
    physical_span = max(
        upper - lower
        for lower, upper in zip(
            cache16.grid_minimum_xyz,
            cache16.grid_maximum_xyz,
            strict=True,
        )
    )
    return {
        "valid_sample_count": valid_sample_count,
        "coordinate_component_count": coordinate_count,
        "normalized_coordinate_max_abs": maximum,
        "normalized_coordinate_rms": (squared_sum / coordinate_count) ** 0.5,
        "physical_coordinate_max_abs_m": maximum * physical_span,
    }


def audit_caches(cache16_root: Path, cache32_root: Path) -> dict[str, Any]:
    cache16 = PSUCompactCachedRayStore(cache16_root, verify_hashes=True)
    cache32 = PSUCompactCachedRayStore(cache32_root, verify_hashes=True)
    manifest16 = cache16.manifest
    manifest32 = cache32.manifest
    common_hash_equal = {
        name: manifest16["arrays"][name]["sha256"]
        == manifest32["arrays"][name]["sha256"]
        for name in COMMON_CACHE_ARRAYS
    }
    grid_hash_distinct = {
        name: manifest16["arrays"][name]["sha256"]
        != manifest32["arrays"][name]["sha256"]
        for name in ("base_indices", "fractions_xyz")
    }
    coordinate_diagnostics = cross_grid_coordinate_diagnostics(cache16, cache32)
    return {
        "cache_manifest_sha256_private_only": {
            "16_cubed": _sha256(cache16_root / "manifest.json"),
            "32_cubed": _sha256(cache32_root / "manifest.json"),
        },
        "cache_rows": [
            {
                "grid_shape_zyx": list(cache.grid_shape),
                "ray_count": int(cache.ray_count),
                "sample_count": int(cache.sample_count),
                "chunk_count": len(cache.manifest["chunks"]),
                "total_cache_bytes": _total_cache_bytes(cache.manifest),
                "build_wall_seconds": float(cache.manifest["build_wall_seconds"]),
            }
            for cache in (cache16, cache32)
        ],
        "selection": cache16.selection_summary(),
        "gates": {
            "all_declared_array_hashes_verified": True,
            "grid_shapes_are_16_and_32": (
                tuple(cache16.grid_shape) == (16, 16, 16)
                and tuple(cache32.grid_shape) == (32, 32, 32)
            ),
            "source_selection_equal": (
                manifest16["source_selection"] == manifest32["source_selection"]
            ),
            "common_physical_ray_arrays_hash_equal": all(common_hash_equal.values()),
            "grid_dependent_arrays_hash_distinct": all(grid_hash_distinct.values()),
            "all_valid_aperture_sample_coordinates_match_across_grids": (
                coordinate_diagnostics["normalized_coordinate_max_abs"] <= 1e-12
            ),
            "nine_views_and_all_active_rays_present": (
                len(cache16.views) == len(cache32.views) == 9
                and cache16.ray_count == cache32.ray_count == 10_628_822
            ),
            "no_development_or_final_data_in_cache": (
                manifest16["claim_boundary"]["development_rotation_40_opened"] is False
                and manifest16["claim_boundary"]["final_audit_opened"] is False
                and manifest32["claim_boundary"]["development_rotation_40_opened"] is False
                and manifest32["claim_boundary"]["final_audit_opened"] is False
            ),
        },
        "diagnostics": {
            "common_array_hash_equal": common_hash_equal,
            "grid_dependent_array_hash_distinct": grid_hash_distinct,
            "cross_grid_coordinate_equivalence": coordinate_diagnostics,
        },
    }


def build_public_summary(private: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": PUBLIC_SCHEMA,
        "status": private["status"],
        "evidence_scope": private["evidence_scope"],
        "view_mapping": private["source_script_audit"]["mapping"],
        "source_script_gates": private["source_script_audit"]["gates"],
        "view_manifest_gates": private["view_manifest_audit"]["gates"],
        "cache_gates": private["cache_audit"]["gates"],
        "cache_rows": private["cache_audit"]["cache_rows"],
        "selection": private["cache_audit"]["selection"],
        "diagnostics": private["cache_audit"]["diagnostics"],
        "claim_boundary": {
            "support_identity_cross_bound_by_preflight_to_official_source_order_and_contiguous_blocks": True,
            "compact_cache_manifest_self_contains_camera_rotation_identity": False,
            "private_cache_array_hashes_verified": True,
            "development_rotation_40_opened": False,
            "final_audit_opened": False,
            "cgls_loro_scores_generated": False,
            "field_truth_available": False,
            "algorithm_superiority": False,
        },
        "public_export_policy": {
            "contains_local_paths": False,
            "contains_measurement_values": False,
            "contains_reconstruction_voxels": False,
            "contains_private_file_hashes": False,
            "contains_development_or_final_audit_values": False,
        },
    }


def run_preflight(
    *,
    script_root: Path,
    view_root: Path,
    cache16_root: Path,
    cache32_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = audit_source_scripts(script_root)
    views = audit_view_manifests(view_root)
    caches = audit_caches(cache16_root, cache32_root)
    all_gates = [
        *source["gates"].values(),
        *views["gates"].values(),
        *caches["gates"].values(),
    ]
    passed = all(value is True for value in all_gates)
    private = {
        "schema_version": PRIVATE_SCHEMA,
        "status": (
            "SUPPORT_ROTATION_LORO_PREFLIGHT_PASS"
            if passed
            else "SUPPORT_ROTATION_LORO_PREFLIGHT_FAIL"
        ),
        "evidence_scope": (
            "PRIVATE_SUPPORT_VIEW_IDENTITY_AND_16_32_COMPACT_CACHE_INTEGRITY_"
            "ONLY_NO_CGLS_LORO_SCORE_NO_DEVELOPMENT_NO_FINAL_AUDIT"
        ),
        "source_script_audit": source,
        "view_manifest_audit": views,
        "cache_audit": caches,
        "all_gates_pass": passed,
    }
    return private, build_public_summary(private)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script-root", type=Path, required=True)
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--cache16-root", type=Path, required=True)
    parser.add_argument("--cache32-root", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    args = parser.parse_args()
    private, public = run_preflight(
        script_root=args.script_root,
        view_root=args.view_root,
        cache16_root=args.cache16_root,
        cache32_root=args.cache32_root,
    )
    args.private_output.parent.mkdir(parents=True, exist_ok=True)
    args.public_output.parent.mkdir(parents=True, exist_ok=True)
    args.private_output.write_text(
        json.dumps(private, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.public_output.write_text(
        json.dumps(public, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(public, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if private["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
