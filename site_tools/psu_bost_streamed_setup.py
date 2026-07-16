#!/usr/bin/env python3
"""Assemble one-view PSU NIRT setup arrays from bounded-memory numeric shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .official_nirt_geometry_fixture import load_numpy_geometry_functions
    from .psu_bost_mat_stream import _sha256_file
except ImportError:  # Direct script execution.
    from official_nirt_geometry_fixture import (  # type: ignore[no-redef]
        load_numpy_geometry_functions,
    )
    from psu_bost_mat_stream import _sha256_file  # type: ignore[no-redef]


VECTOR_FIELDS = ("c", "v", "Ruvecs", "Rvvecs", "Rxvecs", "Ryvecs")
SCALAR_FIELDS = ("epsu_all", "epsv_all", "Csys_all", "Rapvec", "Dfvec")
OUTPUT_SHAPES = {
    "b_data": 4,
    "cam_data": 18,
    "ipf": 3,
    "epf": 3,
}
FLAG_FULL_BOX_ZERO = 1
FLAG_CONE_FALLBACK = 2
FLAG_FINAL_ZERO = 4
FLAG_CONE_EXTENDS_OUTSIDE_BOX = 8


def _open_bundle(bundle_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest_path = bundle_dir / "view_bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status")
        != "VIEW_SHARD_BUNDLE_TRANSCODED_AND_SOURCE_STREAMS_VERIFIED"
    ):
        raise ValueError("view bundle has not passed source-stream verification")
    arrays = {
        name: np.load(bundle_dir / f"{name}.npy", mmap_mode="r")
        for name in VECTOR_FIELDS + SCALAR_FIELDS
    }
    row_counts = {int(array.shape[0]) for array in arrays.values()}
    if len(row_counts) != 1:
        raise ValueError(f"view bundle row counts disagree: {sorted(row_counts)}")
    for name in VECTOR_FIELDS:
        if arrays[name].shape[1:] != (3,):
            raise ValueError(f"{name} must have three columns")
    for name in SCALAR_FIELDS:
        if arrays[name].shape[1:] != (1,):
            raise ValueError(f"{name} must have one column")
    return arrays, manifest


def _open_partial_outputs(output_dir: Path, rows: int):
    partials: dict[str, Path] = {}
    arrays: dict[str, np.memmap] = {}
    for name, columns in OUTPUT_SHAPES.items():
        final = output_dir / f"{name}.npy"
        partial = final.with_name(f".{final.name}.partial.npy")
        partial.unlink(missing_ok=True)
        partials[name] = partial
        arrays[name] = np.lib.format.open_memmap(
            partial, mode="w+", dtype=np.float32, shape=(rows, columns)
        )
    return arrays, partials


def _finite_count(*arrays: np.ndarray) -> int:
    return sum(int(np.count_nonzero(~np.isfinite(array))) for array in arrays)


def assemble_streamed_setup(
    *,
    view_bundle_dir: Path,
    geometry_source: Path,
    output_dir: Path,
    corrected_mask_dir: Path | None = None,
    chunk_rows: int = 65_536,
    outer_minimum: tuple[float, float, float] = (-0.110, -0.110, -0.110),
    outer_maximum: tuple[float, float, float] = (0.110, 0.110, 0.110),
    cone_vertex: tuple[float, float, float] = (0.060, 0.015, 0.0),
    cone_axis: tuple[float, float, float] = (-1.0, -0.1, 0.0),
    cone_angle_degrees: float = 25.0,
) -> dict[str, Any]:
    if chunk_rows < 2:
        raise ValueError("chunk_rows must be at least two for stable author-function shapes")
    fields, bundle_manifest = _open_bundle(view_bundle_dir)
    rows = int(fields["c"].shape[0])
    functions = load_numpy_geometry_functions(geometry_source)
    box = functions["rayBoxIntersection"]
    cone = functions["rayConeIntersection"]
    lower = np.asarray(outer_minimum, dtype=np.float64)[:, None]
    upper = np.asarray(outer_maximum, dtype=np.float64)[:, None]
    vertex = np.asarray(cone_vertex, dtype=np.float64)[:, None]
    axis = np.asarray(cone_axis, dtype=np.float64)[:, None]
    axis = axis / np.linalg.norm(axis)
    angle = np.radians(cone_angle_degrees)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs, partials = _open_partial_outputs(output_dir, rows)
    flags_partial = output_dir / ".geometry_flags.npy.partial.npy"
    flags_partial.unlink(missing_ok=True)
    geometry_flags = np.lib.format.open_memmap(
        flags_partial, mode="w+", dtype=np.uint8, shape=(rows,)
    )

    counters = {
        "ray_count": 0,
        "full_box_zero_length_count": 0,
        "full_box_behind_origin_count": 0,
        "cone_zero_length_fallback_count": 0,
        "cone_root_label_inversion_count": 0,
        "full_box_miss_but_cone_nonzero_count": 0,
        "final_zero_length_count": 0,
        "cone_segment_no_box_overlap_count": 0,
        "cone_segment_partial_outside_box_count": 0,
        "negative_inner_aperture_radius_count": 0,
        "negative_outer_aperture_radius_count": 0,
        "nonfinite_input_count": 0,
        "nonfinite_output_count": 0,
        "runtime_warning_count": 0,
    }
    cone_length_sum = 0.0
    cone_box_overlap_sum = 0.0
    extrema = {
        "line_length_minimum_m": float("inf"),
        "line_length_maximum_m": float("-inf"),
        "inner_aperture_radius_minimum_m": float("inf"),
        "inner_aperture_radius_maximum_m": float("-inf"),
        "outer_aperture_radius_minimum_m": float("inf"),
        "outer_aperture_radius_maximum_m": float("-inf"),
    }
    start_time = time.perf_counter()
    try:
        for start in range(0, rows, chunk_rows):
            stop = min(start + chunk_rows, rows)
            origins = np.asarray(fields["c"][start:stop], dtype=np.float64).T
            directions = np.asarray(fields["v"][start:stop], dtype=np.float64).T
            counters["nonfinite_input_count"] += _finite_count(origins, directions)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                full_enter, full_exit, full_length = box(
                    origins, directions, lower, upper
                )
                cone_first, cone_second, cone_length = cone(
                    origins, directions, vertex, axis, angle
                )
            counters["runtime_warning_count"] += len(caught)
            full_enter = np.asarray(full_enter).reshape(-1)
            full_exit = np.asarray(full_exit).reshape(-1)
            full_length = np.asarray(full_length).reshape(-1)
            cone_first = np.asarray(cone_first).reshape(-1)
            cone_second = np.asarray(cone_second).reshape(-1)
            cone_length = np.asarray(cone_length).reshape(-1)
            fallback = cone_length == 0
            enter = np.where(fallback, full_enter, cone_first)
            exit_ = np.where(fallback, full_exit, cone_second)
            length = np.where(fallback, full_length, cone_length)
            full_low = np.minimum(full_enter, full_exit)
            full_high = np.maximum(full_enter, full_exit)
            cone_low = np.minimum(cone_first, cone_second)
            cone_high = np.maximum(cone_first, cone_second)
            cone_active = ~fallback
            cone_overlap = np.maximum(
                0.0,
                np.minimum(full_high, cone_high) - np.maximum(full_low, cone_low),
            )
            cone_overlap = np.where(
                cone_active & (full_length > 0), cone_overlap, 0.0
            )
            cone_outside = cone_active & (
                cone_overlap < cone_length - 1e-12
            )
            intersection = origins + enter[None, :] * directions
            endpoint = origins + exit_[None, :] * directions

            rap = np.asarray(fields["Rapvec"][start:stop, 0], dtype=np.float64)
            df = np.asarray(fields["Dfvec"][start:stop, 0], dtype=np.float64)
            inner = rap * (1.0 - enter / df)
            outer = rap * (1.0 - exit_ / df)

            count = stop - start
            b_data = np.zeros((count, 4), dtype=np.float32)
            b_data[:, 0] = fields["epsu_all"][start:stop, 0]
            b_data[:, 1] = fields["epsv_all"][start:stop, 0]
            cam_data = np.zeros((count, 18), dtype=np.float32)
            cam_data[:, 0] = length
            cam_data[:, 1:4] = fields["Ruvecs"][start:stop]
            cam_data[:, 4:7] = fields["Rvvecs"][start:stop]
            cam_data[:, 9:12] = fields["Rxvecs"][start:stop]
            cam_data[:, 12:15] = fields["Ryvecs"][start:stop]
            cam_data[:, 15] = inner
            cam_data[:, 16] = outer
            cam_data[:, 17] = fields["Csys_all"][start:stop, 0]
            ipf = np.asarray(intersection.T, dtype=np.float32)
            epf = np.asarray(endpoint.T, dtype=np.float32)

            outputs["b_data"][start:stop] = b_data
            outputs["cam_data"][start:stop] = cam_data
            outputs["ipf"][start:stop] = ipf
            outputs["epf"][start:stop] = epf
            flags = np.zeros(count, dtype=np.uint8)
            flags[full_length == 0] |= FLAG_FULL_BOX_ZERO
            flags[fallback] |= FLAG_CONE_FALLBACK
            flags[length == 0] |= FLAG_FINAL_ZERO
            flags[cone_outside] |= FLAG_CONE_EXTENDS_OUTSIDE_BOX
            geometry_flags[start:stop] = flags

            counters["ray_count"] += count
            counters["full_box_zero_length_count"] += int(
                np.count_nonzero(full_length == 0)
            )
            counters["full_box_behind_origin_count"] += int(
                np.count_nonzero(full_exit <= 0)
            )
            counters["cone_zero_length_fallback_count"] += int(
                np.count_nonzero(fallback)
            )
            counters["cone_root_label_inversion_count"] += int(
                np.count_nonzero(
                    (~fallback) & np.isfinite(cone_first) & (cone_first > cone_second)
                )
            )
            counters["full_box_miss_but_cone_nonzero_count"] += int(
                np.count_nonzero((full_length == 0) & cone_active)
            )
            counters["final_zero_length_count"] += int(np.count_nonzero(length == 0))
            counters["cone_segment_no_box_overlap_count"] += int(
                np.count_nonzero(cone_active & (cone_overlap == 0))
            )
            counters["cone_segment_partial_outside_box_count"] += int(
                np.count_nonzero(cone_outside)
            )
            cone_length_sum += float(cone_length[cone_active].sum())
            cone_box_overlap_sum += float(cone_overlap[cone_active].sum())
            counters["negative_inner_aperture_radius_count"] += int(
                np.count_nonzero(inner < 0)
            )
            counters["negative_outer_aperture_radius_count"] += int(
                np.count_nonzero(outer < 0)
            )
            counters["nonfinite_output_count"] += _finite_count(
                b_data, cam_data, ipf, epf
            )
            extrema["line_length_minimum_m"] = min(
                extrema["line_length_minimum_m"], float(np.min(length))
            )
            extrema["line_length_maximum_m"] = max(
                extrema["line_length_maximum_m"], float(np.max(length))
            )
            extrema["inner_aperture_radius_minimum_m"] = min(
                extrema["inner_aperture_radius_minimum_m"], float(np.min(inner))
            )
            extrema["inner_aperture_radius_maximum_m"] = max(
                extrema["inner_aperture_radius_maximum_m"], float(np.max(inner))
            )
            extrema["outer_aperture_radius_minimum_m"] = min(
                extrema["outer_aperture_radius_minimum_m"], float(np.min(outer))
            )
            extrema["outer_aperture_radius_maximum_m"] = max(
                extrema["outer_aperture_radius_maximum_m"], float(np.max(outer))
            )
        for array in outputs.values():
            array.flush()
        geometry_flags.flush()
    except Exception:
        for partial in partials.values():
            partial.unlink(missing_ok=True)
        flags_partial.unlink(missing_ok=True)
        raise
    del outputs
    del geometry_flags
    output_records = []
    for name, partial in partials.items():
        final = output_dir / f"{name}.npy"
        os.replace(partial, final)
        output_records.append(
            {
                "name": name,
                "filename": final.name,
                "shape": [rows, OUTPUT_SHAPES[name]],
                "dtype": "float32",
                "bytes": final.stat().st_size,
                "sha256": _sha256_file(final),
            }
        )
    flags_final = output_dir / "geometry_flags.npy"
    os.replace(flags_partial, flags_final)
    output_records.append(
        {
            "name": "geometry_flags",
            "filename": flags_final.name,
            "shape": [rows],
            "dtype": "uint8",
            "bytes": flags_final.stat().st_size,
            "sha256": _sha256_file(flags_final),
            "bit_contract": {
                "1": "full box length is zero",
                "2": "cone length is zero and setup falls back to box",
                "4": "final selected line length is zero",
                "8": "selected cone segment extends outside the outer box",
            },
        }
    )

    mask_intersection: dict[str, Any] | None = None
    if corrected_mask_dir is not None:
        flags = np.load(flags_final, mmap_mode="r")
        mask_intersection = {}
        for variable in ("amask_all", "imask_all"):
            indices = np.load(
                corrected_mask_dir / f"{variable}_zero_based.npy", mmap_mode="r"
            )
            selected_flags = flags[indices]
            unsafe = (
                selected_flags
                & (FLAG_FINAL_ZERO | FLAG_CONE_EXTENDS_OUTSIDE_BOX)
            ) != 0
            mask_intersection[variable] = {
                "count": int(indices.size),
                "full_box_zero_count": int(
                    np.count_nonzero(selected_flags & FLAG_FULL_BOX_ZERO)
                ),
                "cone_fallback_count": int(
                    np.count_nonzero(selected_flags & FLAG_CONE_FALLBACK)
                ),
                "final_zero_length_count": int(
                    np.count_nonzero(selected_flags & FLAG_FINAL_ZERO)
                ),
                "cone_extends_outside_box_count": int(
                    np.count_nonzero(
                        selected_flags & FLAG_CONE_EXTENDS_OUTSIDE_BOX
                    )
                ),
                "unsafe_geometry_union_count": int(np.count_nonzero(unsafe)),
                "unsafe_geometry_union_fraction": float(np.mean(unsafe))
                if unsafe.size
                else None,
            }

    counters["cone_segment_length_sum_m"] = cone_length_sum
    counters["cone_box_overlap_length_sum_m"] = cone_box_overlap_sum
    counters["cone_length_weighted_outside_box_fraction"] = (
        1.0 - cone_box_overlap_sum / cone_length_sum if cone_length_sum else None
    )

    mechanical_pass = all(
        (
            counters["ray_count"] == rows,
            counters["nonfinite_input_count"] == 0,
            counters["nonfinite_output_count"] == 0,
            counters["full_box_zero_length_count"] == 0,
            counters["full_box_behind_origin_count"] == 0,
            counters["full_box_miss_but_cone_nonzero_count"] == 0,
            counters["final_zero_length_count"] == 0,
            counters["cone_segment_partial_outside_box_count"] == 0,
            counters["negative_inner_aperture_radius_count"] == 0,
            counters["negative_outer_aperture_radius_count"] == 0,
        )
    )
    report = {
        "schema_version": "psu-bost-streamed-setup-1.0",
        "status": "STREAMED_SETUP_MECHANICAL_CONTRACT_PASS"
        if mechanical_pass
        else "STREAMED_SETUP_DIAGNOSTIC_NO_GO",
        "evidence_scope": "ONE_REAL_VIEW_AUTHOR_GEOMETRY_FORMULAS_STREAMED_ASSEMBLY_NO_TENSORFLOW_NO_NIRT",
        "source": {
            "view_bundle_manifest_sha256": hashlib.sha256(
                (view_bundle_dir / "view_bundle_manifest.json").read_bytes()
            ).hexdigest(),
            "geometry_source_filename": geometry_source.name,
            "geometry_source_sha256": hashlib.sha256(
                geometry_source.read_bytes()
            ).hexdigest(),
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
            "view_id_zero_based": bundle_manifest["view"]["view_id_zero_based"],
        },
        "diagnostics": {**counters, **extrema},
        "corrected_mask_intersection": mask_intersection,
        "outputs": output_records,
        "runtime_observation": {
            "wall_seconds": time.perf_counter() - start_time,
            "scope": "CACHED_LOCAL_RUN_NOT_A_SPEED_BENCHMARK",
        },
        "decision": {
            "streamed_setup_mechanically_valid": mechanical_pass,
            "eager_author_setup_equivalence": "NOT_DIRECTLY_COMPARED",
            "full_nirt_reconstruction": "NOT_UNLOCKED",
            "next_gate": "MASKED_MINIBATCH_LOS_SURROGATE_AND_TENSORFLOW_NETWORK_SMOKE",
        },
        "limitations": [
            "float64 source scalars are cast to float32 because the author pipeline later converts tensors to float32",
            "author box/cone limitations are preserved rather than silently corrected",
            "the run covers one selected view only and does not prove multi-view alignment",
            "no random ray sampling, autodiff density gradient, loss, training, or reconstruction is executed",
        ],
    }
    (output_dir / "streamed_setup_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view-bundle-dir", type=Path, required=True)
    parser.add_argument("--geometry-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--corrected-mask-dir", type=Path)
    parser.add_argument("--chunk-rows", type=int, default=65_536)
    args = parser.parse_args()
    report = assemble_streamed_setup(
        view_bundle_dir=args.view_bundle_dir,
        geometry_source=args.geometry_source,
        output_dir=args.output_dir,
        corrected_mask_dir=args.corrected_mask_dir,
        chunk_rows=args.chunk_rows,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"].endswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
