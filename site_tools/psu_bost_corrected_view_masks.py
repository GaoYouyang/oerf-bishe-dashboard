#!/usr/bin/env python3
"""Build explicit zero-based mask shards and audit their displacement semantics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .psu_bost_mat_stream import (
        _sha256_file,
        open_measurement_stream,
        view_measurement_range,
    )
except ImportError:  # Direct script execution.
    from psu_bost_mat_stream import (  # type: ignore[no-redef]
        _sha256_file,
        open_measurement_stream,
        view_measurement_range,
    )


MASK_VARIABLES = ("amask_all", "imask_all")


def _iter_corrected_local_indices(
    *,
    mat_path: Path,
    variable: str,
    view_start: int,
    view_stop: int,
    chunk_measurements: int,
):
    stream = open_measurement_stream(
        mat_path,
        variable,
        chunk_measurements=chunk_measurements,
        cast_dtype="int64",
    )
    for chunk in stream:
        global_zero_based = chunk.values[:, 0] - 1
        selected = global_zero_based[
            (global_zero_based >= view_start) & (global_zero_based < view_stop)
        ]
        if selected.size:
            yield selected - view_start
    if not stream.audit.complete:
        raise RuntimeError(f"{variable} stream integrity was not verified")


def _build_mask_shard(
    *,
    mat_path: Path,
    variable: str,
    output_path: Path,
    view_start: int,
    view_stop: int,
    chunk_measurements: int,
) -> dict[str, Any]:
    count = sum(
        int(values.size)
        for values in _iter_corrected_local_indices(
            mat_path=mat_path,
            variable=variable,
            view_start=view_start,
            view_stop=view_stop,
            chunk_measurements=chunk_measurements,
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(f".{output_path.name}.partial.npy")
    partial_path.unlink(missing_ok=True)
    mapped = np.lib.format.open_memmap(
        partial_path, mode="w+", dtype=np.int64, shape=(count,)
    )
    cursor = 0
    try:
        for values in _iter_corrected_local_indices(
            mat_path=mat_path,
            variable=variable,
            view_start=view_start,
            view_stop=view_stop,
            chunk_measurements=chunk_measurements,
        ):
            mapped[cursor : cursor + values.size] = values
            cursor += int(values.size)
        mapped.flush()
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise
    del mapped
    if cursor != count:
        partial_path.unlink(missing_ok=True)
        raise RuntimeError(f"mask shard count changed between passes: {cursor} != {count}")
    os.replace(partial_path, output_path)
    values = np.load(output_path, mmap_mode="r")
    strictly_increasing = bool(values.size < 2 or np.all(np.diff(values) > 0))
    return {
        "variable": variable,
        "count": count,
        "minimum_local_index": int(values.min()) if values.size else None,
        "maximum_local_index": int(values.max()) if values.size else None,
        "strictly_increasing": strictly_increasing,
        "all_local_indices_in_range": bool(
            values.size == 0
            or (int(values.min()) >= 0 and int(values.max()) < view_stop - view_start)
        ),
        "filename": output_path.name,
        "bytes": output_path.stat().st_size,
        "sha256": _sha256_file(output_path),
    }


def _displacement_statistics(
    *,
    indices_path: Path,
    u: np.ndarray,
    v: np.ndarray,
    chunk_size: int = 262_144,
) -> dict[str, Any]:
    indices = np.load(indices_path, mmap_mode="r")
    count = 0
    magnitude_sum = 0.0
    magnitude_square_sum = 0.0
    maximum = 0.0
    finite = True
    shifted_valid = 0
    shifted_changed = 0
    shifted_square_difference = 0.0
    boundary_crossings = 0
    for start in range(0, int(indices.size), chunk_size):
        local = np.asarray(indices[start : start + chunk_size])
        uu = np.asarray(u[local], dtype=np.float64)
        vv = np.asarray(v[local], dtype=np.float64)
        magnitude = np.hypot(uu, vv)
        count += int(local.size)
        magnitude_sum += float(magnitude.sum())
        magnitude_square_sum += float(np.square(magnitude).sum())
        maximum = max(maximum, float(magnitude.max(initial=0.0)))
        finite = finite and bool(np.all(np.isfinite(magnitude)))

        shifted = local + 1
        valid = shifted < u.shape[0]
        boundary_crossings += int(np.count_nonzero(~valid))
        if np.any(valid):
            delta_u = np.asarray(u[shifted[valid]], dtype=np.float64) - uu[valid]
            delta_v = np.asarray(v[shifted[valid]], dtype=np.float64) - vv[valid]
            delta_square = np.square(delta_u) + np.square(delta_v)
            shifted_valid += int(delta_square.size)
            shifted_changed += int(np.count_nonzero(delta_square > 1e-14))
            shifted_square_difference += float(delta_square.sum())
    return {
        "count": count,
        "all_displacements_finite": finite,
        "mean_magnitude_pixels": magnitude_sum / count if count else None,
        "rms_magnitude_pixels": (magnitude_square_sum / count) ** 0.5
        if count
        else None,
        "maximum_magnitude_pixels": maximum if count else None,
        "uncorrected_plus_one_comparison": {
            "valid_comparison_count": shifted_valid,
            "changed_vector_count_at_1e_7_pixels": shifted_changed,
            "changed_fraction": shifted_changed / shifted_valid
            if shifted_valid
            else None,
            "vector_rmse_pixels": (shifted_square_difference / shifted_valid) ** 0.5
            if shifted_valid
            else None,
            "view_boundary_crossing_count": boundary_crossings,
        },
    }


def build_corrected_view_masks(
    *,
    mat_path: Path,
    view_bundle_dir: Path,
    output_dir: Path,
    view_id: int,
    image_height: int,
    image_width: int,
    view_count: int,
    chunk_measurements: int = 262_144,
) -> dict[str, Any]:
    start, stop = view_measurement_range(
        view_id=view_id,
        image_height=image_height,
        image_width=image_width,
        view_count=view_count,
    )
    u = np.load(view_bundle_dir / "epsu_all.npy", mmap_mode="r")[:, 0]
    v = np.load(view_bundle_dir / "epsv_all.npy", mmap_mode="r")[:, 0]
    expected_shape = (stop - start,)
    if u.shape != expected_shape or v.shape != expected_shape:
        raise ValueError(
            f"deflection shards have shapes {u.shape}/{v.shape}, expected {expected_shape}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_reports = [
        _build_mask_shard(
            mat_path=mat_path,
            variable=variable,
            output_path=output_dir / f"{variable}_zero_based.npy",
            view_start=start,
            view_stop=stop,
            chunk_measurements=chunk_measurements,
        )
        for variable in MASK_VARIABLES
    ]
    displacement = {
        item["variable"]: _displacement_statistics(
            indices_path=output_dir / item["filename"], u=u, v=v
        )
        for item in mask_reports
    }
    active_rms = displacement["amask_all"]["rms_magnitude_pixels"]
    inactive_rms = displacement["imask_all"]["rms_magnitude_pixels"]
    report = {
        "schema_version": "psu-bost-corrected-view-masks-1.0",
        "status": "CORRECTED_VIEW_MASK_SHARDS_BUILT_MECHANICAL_CONTRACT_PASS",
        "evidence_scope": "EXPLICIT_ONE_TO_ZERO_BASE_ADAPTER_AND_REAL_DEFLECTION_ROWS_NO_TENSORFLOW_NO_NIRT",
        "source": {"filename": mat_path.name},
        "view": {
            "view_id_zero_based": view_id,
            "image_height": image_height,
            "image_width": image_width,
            "measurement_start": start,
            "measurement_stop": stop,
            "measurement_count": stop - start,
        },
        "correction": {
            "formula": "python_zero_based_global_index = matlab_find_index - 1",
            "local_formula": "view_local_index = python_zero_based_global_index - view_start",
            "author_source_modified": False,
        },
        "mask_shards": mask_reports,
        "deflection_semantics": displacement,
        "diagnostic": {
            "active_to_inactive_rms_magnitude_ratio": active_rms / inactive_rms
            if active_rms is not None and inactive_rms
            else None,
            "interpretation": "DESCRIPTIVE_ONLY_MASK_SEPARATION_IS_NOT_GROUND_TRUTH",
        },
        "decision": {
            "corrected_indices_mechanically_valid": all(
                item["all_local_indices_in_range"] and item["strictly_increasing"]
                for item in mask_reports
            ),
            "uncorrected_official_gather": "NO_GO",
            "physical_mask_semantics": "REVIEW_REQUIRED",
            "next_gate": "STREAMED_CAM_DATA_ASSEMBLY_WITH_CORRECTED_MASKS",
        },
        "limitations": [
            "active/inactive displacement statistics are descriptive and do not establish correct physical labels",
            "the one-pixel shift effect is measured on this selected view only",
            "no TensorFlow gather, boundary loss, LoS integration, or reconstruction is run",
        ],
    }
    (output_dir / "corrected_view_masks_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat", type=Path, required=True)
    parser.add_argument("--view-bundle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--view-id", type=int, required=True)
    parser.add_argument("--image-height", type=int, required=True)
    parser.add_argument("--image-width", type=int, required=True)
    parser.add_argument("--view-count", type=int, required=True)
    args = parser.parse_args()
    report = build_corrected_view_masks(
        mat_path=args.mat,
        view_bundle_dir=args.view_bundle_dir,
        output_dir=args.output_dir,
        view_id=args.view_id,
        image_height=args.image_height,
        image_width=args.image_width,
        view_count=args.view_count,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
