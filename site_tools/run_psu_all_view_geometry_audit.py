#!/usr/bin/env python3
"""Run the streamed PSU geometry and mask audit independently across all views."""

from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from .build_psu_view_shards import DEFAULT_VARIABLES, build_view_bundle
    from .psu_bost_corrected_view_masks import build_corrected_view_masks
    from .psu_bost_streamed_setup import assemble_streamed_setup
except ImportError:  # Direct script execution.
    from build_psu_view_shards import (  # type: ignore[no-redef]
        DEFAULT_VARIABLES,
        build_view_bundle,
    )
    from psu_bost_corrected_view_masks import (  # type: ignore[no-redef]
        build_corrected_view_masks,
    )
    from psu_bost_streamed_setup import (  # type: ignore[no-redef]
        assemble_streamed_setup,
    )


METRICS = (
    "full_box_zero_fraction",
    "box_miss_but_cone_nonzero_fraction",
    "final_zero_length_fraction",
    "cone_outside_ray_fraction",
    "cone_length_weighted_outside_box_fraction",
    "active_unsafe_geometry_fraction",
    "inactive_unsafe_geometry_fraction",
    "active_rms_magnitude_pixels",
    "inactive_rms_magnitude_pixels",
    "active_to_inactive_rms_ratio",
)

BUNDLE_SCHEMA = "psu-bost-view-shard-bundle-1.0"
BUNDLE_STATUS = "VIEW_SHARD_BUNDLE_TRANSCODED_AND_SOURCE_STREAMS_VERIFIED"
MASK_SCHEMA = "psu-bost-corrected-view-masks-1.0"
MASK_STATUS = "CORRECTED_VIEW_MASK_SHARDS_BUILT_MECHANICAL_CONTRACT_PASS"
SETUP_SCHEMA = "psu-bost-streamed-setup-1.0"


def _sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _validate_npy_artifact(
    *,
    directory: Path,
    record: dict[str, Any],
    filename_key: str,
    bytes_key: str,
    sha256_key: str,
    shape_key: str,
    dtype_key: str,
    verify_sha256: bool,
) -> None:
    path = directory / str(record[filename_key])
    if not path.is_file() or path.stat().st_size != int(record[bytes_key]):
        raise ValueError(f"artifact size contract failed: {path}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if list(array.shape) != list(record[shape_key]):
        raise ValueError(f"artifact shape contract failed: {path}")
    if str(array.dtype) != str(record[dtype_key]):
        raise ValueError(f"artifact dtype contract failed: {path}")
    del array
    if verify_sha256 and _sha256_file(path) != str(record[sha256_key]):
        raise ValueError(f"artifact SHA-256 contract failed: {path}")


def _load_completed_view(
    *,
    view_dir: Path,
    mat_path: Path,
    geometry_source: Path,
    view_id: int,
    image_height: int,
    image_width: int,
    view_count: int,
    verify_sha256: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    """Return validated reports for a complete view, or None for a rebuild."""

    bundle_dir = view_dir / "bundle"
    masks_dir = view_dir / "corrected_masks"
    setup_dir = view_dir / "setup"
    try:
        bundle_path = bundle_dir / "view_bundle_manifest.json"
        mask_path = masks_dir / "corrected_view_masks_manifest.json"
        setup_path = setup_dir / "streamed_setup_manifest.json"
        bundle = _read_json_object(bundle_path)
        masks = _read_json_object(mask_path)
        setup = _read_json_object(setup_path)

        measurement_count = image_height * image_width
        expected_view = {
            "view_id_zero_based": view_id,
            "image_height": image_height,
            "image_width": image_width,
            "measurement_start": view_id * measurement_count,
            "measurement_stop": (view_id + 1) * measurement_count,
            "measurement_count": measurement_count,
        }

        if bundle.get("schema_version") != BUNDLE_SCHEMA:
            raise ValueError("bundle schema mismatch")
        if bundle.get("status") != BUNDLE_STATUS:
            raise ValueError("bundle status mismatch")
        if bundle.get("source", {}).get("filename") != mat_path.name:
            raise ValueError("bundle source mismatch")
        bundle_view = bundle.get("view", {})
        if any(bundle_view.get(key) != value for key, value in expected_view.items()):
            raise ValueError("bundle view contract mismatch")
        if bundle_view.get("view_count") != view_count:
            raise ValueError("bundle view-count mismatch")
        variables = bundle.get("variables", [])
        if [item.get("name") for item in variables] != list(DEFAULT_VARIABLES):
            raise ValueError("bundle variable contract mismatch")
        if not bundle.get("aggregate", {}).get("all_source_streams_verified"):
            raise ValueError("bundle source streams were not fully verified")
        for item in variables:
            artifact = dict(item)
            artifact["filename"] = f"{item['name']}.npy"
            _validate_npy_artifact(
                directory=bundle_dir,
                record=artifact,
                filename_key="filename",
                bytes_key="shard_bytes",
                sha256_key="shard_sha256",
                shape_key="shard_shape",
                dtype_key="shard_dtype",
                verify_sha256=verify_sha256,
            )
            summary = _read_json_object(bundle_dir / f"{item['name']}.summary.json")
            if summary.get("source", {}).get("variable") != item["name"]:
                raise ValueError(f"bundle summary variable mismatch: {item['name']}")
            if summary.get("output", {}).get("sha256") != item["shard_sha256"]:
                raise ValueError(f"bundle summary hash mismatch: {item['name']}")
            if not summary.get("stream_audit", {}).get("matrix_stream_verified"):
                raise ValueError(f"bundle summary stream incomplete: {item['name']}")

        if masks.get("schema_version") != MASK_SCHEMA or masks.get("status") != MASK_STATUS:
            raise ValueError("mask schema or status mismatch")
        if masks.get("source", {}).get("filename") != mat_path.name:
            raise ValueError("mask source mismatch")
        mask_view = masks.get("view", {})
        if any(mask_view.get(key) != value for key, value in expected_view.items()):
            raise ValueError("mask view contract mismatch")
        mask_records = masks.get("mask_shards", [])
        if [item.get("variable") for item in mask_records] != [
            "amask_all",
            "imask_all",
        ]:
            raise ValueError("mask variable contract mismatch")
        for item in mask_records:
            path = masks_dir / str(item["filename"])
            if not path.is_file() or path.stat().st_size != int(item["bytes"]):
                raise ValueError(f"mask size contract failed: {path}")
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            if values.shape != (int(item["count"]),) or values.dtype != np.int64:
                raise ValueError(f"mask array contract failed: {path}")
            del values
            if verify_sha256 and _sha256_file(path) != str(item["sha256"]):
                raise ValueError(f"mask SHA-256 contract failed: {path}")

        if setup.get("schema_version") != SETUP_SCHEMA:
            raise ValueError("setup schema mismatch")
        if setup.get("status") not in {
            "STREAMED_SETUP_MECHANICAL_CONTRACT_PASS",
            "STREAMED_SETUP_DIAGNOSTIC_NO_GO",
        }:
            raise ValueError("setup status mismatch")
        configuration = setup.get("configuration", {})
        if configuration.get("view_id_zero_based") != view_id:
            raise ValueError("setup view mismatch")
        if configuration.get("rows") != measurement_count:
            raise ValueError("setup row-count mismatch")
        setup_source = setup.get("source", {})
        if setup_source.get("geometry_source_filename") != geometry_source.name:
            raise ValueError("geometry source filename mismatch")
        if setup_source.get("geometry_source_sha256") != _sha256_file(geometry_source):
            raise ValueError("geometry source SHA-256 mismatch")
        if setup_source.get("view_bundle_manifest_sha256") != _sha256_file(bundle_path):
            raise ValueError("setup-to-bundle manifest binding mismatch")
        setup_outputs = setup.get("outputs", [])
        if [item.get("name") for item in setup_outputs] != [
            "b_data",
            "cam_data",
            "ipf",
            "epf",
            "geometry_flags",
        ]:
            raise ValueError("setup output contract mismatch")
        for item in setup_outputs:
            _validate_npy_artifact(
                directory=setup_dir,
                record=item,
                filename_key="filename",
                bytes_key="bytes",
                sha256_key="sha256",
                shape_key="shape",
                dtype_key="dtype",
                verify_sha256=verify_sha256,
            )
        if not {"amask_all", "imask_all"}.issubset(
            setup.get("corrected_mask_intersection", {})
        ):
            raise ValueError("setup corrected-mask intersection missing")
    except (KeyError, OSError, TypeError, ValueError):
        return None
    return bundle, masks, setup


def _remove_partial_npy_files(view_dir: Path) -> list[str]:
    removed: list[str] = []
    for child in ("bundle", "corrected_masks", "setup"):
        directory = view_dir / child
        if not directory.is_dir():
            continue
        for path in directory.glob(".*.partial.npy"):
            if path.is_file():
                path.unlink()
                removed.append(str(path.relative_to(view_dir)))
    return removed


def _atomic_write_text(path: Path, text: str) -> None:
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(text, encoding="utf-8")
    os.replace(partial, path)


@contextlib.contextmanager
def _exclusive_run_lock(output_root: Path):
    lock_path = output_root / ".all_view_geometry_audit.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another audit process holds {lock_path}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _build_run_contract(
    *,
    mat_path: Path,
    geometry_source: Path,
    image_height: int,
    image_width: int,
    view_count: int,
    chunk_measurements: int,
) -> tuple[dict[str, Any], str]:
    validator_path = Path(__file__)
    generator_paths = {
        "view_bundle_builder": Path(build_view_bundle.__code__.co_filename),
        "corrected_mask_builder": Path(
            build_corrected_view_masks.__code__.co_filename
        ),
        "streamed_setup_builder": Path(assemble_streamed_setup.__code__.co_filename),
    }
    contract = {
        "schema_version": "psu-bost-all-view-run-contract-1.0",
        "source": {
            "mat_filename": mat_path.name,
            "mat_sha256": _sha256_file(mat_path),
            "geometry_source_filename": geometry_source.name,
            "geometry_source_sha256": _sha256_file(geometry_source),
        },
        "configuration": {
            "image_height": image_height,
            "image_width": image_width,
            "view_count": view_count,
            "chunk_measurements": chunk_measurements,
            "variables": list(DEFAULT_VARIABLES),
        },
        "code_provenance": {
            "validator_current_sha256": _sha256_file(validator_path),
            "artifact_generators_current_snapshot_sha256": {
                name: _sha256_file(path)
                for name, path in sorted(generator_paths.items())
            },
            "artifact_generator_source_binding_at_initial_generation": (
                "NOT_RECORDED_NUMERIC_ARTIFACTS_REVALIDATED_BY_FILE_HASH"
            ),
        },
    }
    canonical = json.dumps(
        contract, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    return contract, hashlib.sha256(canonical).hexdigest()


def _view_record(
    *,
    view_id: int,
    bundle: dict[str, Any],
    masks: dict[str, Any],
    setup: dict[str, Any],
) -> dict[str, Any]:
    diagnostics = setup["diagnostics"]
    intersections = setup["corrected_mask_intersection"]
    rays = int(diagnostics["ray_count"])
    active = masks["deflection_semantics"]["amask_all"]
    inactive = masks["deflection_semantics"]["imask_all"]
    record = {
        "view_id_zero_based": view_id,
        "measurement_count": rays,
        "bundle_status": bundle["status"],
        "mask_status": masks["status"],
        "setup_status": setup["status"],
        "full_box_zero_count": diagnostics["full_box_zero_length_count"],
        "full_box_zero_fraction": diagnostics["full_box_zero_length_count"] / rays,
        "box_miss_but_cone_nonzero_count": diagnostics[
            "full_box_miss_but_cone_nonzero_count"
        ],
        "box_miss_but_cone_nonzero_fraction": diagnostics[
            "full_box_miss_but_cone_nonzero_count"
        ]
        / rays,
        "final_zero_length_count": diagnostics["final_zero_length_count"],
        "final_zero_length_fraction": diagnostics["final_zero_length_count"] / rays,
        "cone_outside_ray_count": diagnostics[
            "cone_segment_partial_outside_box_count"
        ],
        "cone_outside_ray_fraction": diagnostics[
            "cone_segment_partial_outside_box_count"
        ]
        / rays,
        "cone_no_box_overlap_count": diagnostics[
            "cone_segment_no_box_overlap_count"
        ],
        "cone_length_weighted_outside_box_fraction": diagnostics[
            "cone_length_weighted_outside_box_fraction"
        ],
        "cone_segment_length_sum_m": diagnostics["cone_segment_length_sum_m"],
        "cone_box_overlap_length_sum_m": diagnostics[
            "cone_box_overlap_length_sum_m"
        ],
        "active_count": intersections["amask_all"]["count"],
        "active_unsafe_geometry_count": intersections["amask_all"][
            "unsafe_geometry_union_count"
        ],
        "active_unsafe_geometry_fraction": intersections["amask_all"][
            "unsafe_geometry_union_fraction"
        ],
        "inactive_count": intersections["imask_all"]["count"],
        "inactive_unsafe_geometry_count": intersections["imask_all"][
            "unsafe_geometry_union_count"
        ],
        "inactive_unsafe_geometry_fraction": intersections["imask_all"][
            "unsafe_geometry_union_fraction"
        ],
        "active_rms_magnitude_pixels": active["rms_magnitude_pixels"],
        "inactive_rms_magnitude_pixels": inactive["rms_magnitude_pixels"],
        "active_to_inactive_rms_ratio": masks["diagnostic"][
            "active_to_inactive_rms_magnitude_ratio"
        ],
        "active_shift_vector_rmse_pixels": active[
            "uncorrected_plus_one_comparison"
        ]["vector_rmse_pixels"],
        "inactive_shift_vector_rmse_pixels": inactive[
            "uncorrected_plus_one_comparison"
        ]["vector_rmse_pixels"],
    }
    return record


def aggregate_view_records(
    records: Sequence[dict[str, Any]], *, expected_view_count: int | None = None
) -> dict[str, Any]:
    if not records:
        raise ValueError("at least one view record is required")
    ids = [int(item["view_id_zero_based"]) for item in records]
    if len(ids) != len(set(ids)):
        raise ValueError("view ids must be unique")
    if expected_view_count is not None:
        if expected_view_count < 1:
            raise ValueError("expected_view_count must be positive")
        if ids != list(range(expected_view_count)):
            raise ValueError(
                "view ids must exactly match the ordered range "
                f"0..{expected_view_count - 1}"
            )
    for item in records:
        if int(item["measurement_count"]) <= 0:
            raise ValueError("every view must contain at least one measurement")
        if item["bundle_status"] != BUNDLE_STATUS:
            raise ValueError("every view bundle must pass source-stream verification")
        if item["mask_status"] != MASK_STATUS:
            raise ValueError("every corrected mask must pass its mechanical contract")
        if item["setup_status"] not in {
            "STREAMED_SETUP_MECHANICAL_CONTRACT_PASS",
            "STREAMED_SETUP_DIAGNOSTIC_NO_GO",
        }:
            raise ValueError("every setup status must be a reviewed contract value")
    metric_summary = {}
    for metric in METRICS:
        values = [float(item[metric]) for item in records]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError(f"metric {metric!r} must be finite and nonnegative")
        if metric.endswith("_fraction") and any(value > 1 for value in values):
            raise ValueError(f"fraction metric {metric!r} must not exceed one")
        maximum_index = max(range(len(values)), key=values.__getitem__)
        minimum_index = min(range(len(values)), key=values.__getitem__)
        metric_summary[metric] = {
            "minimum": values[minimum_index],
            "minimum_view_id": ids[minimum_index],
            "mean": sum(values) / len(values),
            "maximum": values[maximum_index],
            "maximum_view_id": ids[maximum_index],
        }
    no_go_views = [
        item["view_id_zero_based"]
        for item in records
        if item["setup_status"] != "STREAMED_SETUP_MECHANICAL_CONTRACT_PASS"
    ]
    total_rays = sum(int(item["measurement_count"]) for item in records)
    total_cone_length = sum(
        float(item["cone_segment_length_sum_m"]) for item in records
    )
    total_cone_overlap = sum(
        float(item["cone_box_overlap_length_sum_m"]) for item in records
    )
    return {
        "schema_version": "psu-bost-all-view-geometry-audit-1.0",
        "status": (
            "ALL_VIEW_GEOMETRY_AUDIT_NO_GO"
            if no_go_views
            else "ALL_VIEW_GEOMETRY_MECHANICAL_CONTRACT_PASS"
        ),
        "evidence_scope": "PER_VIEW_REAL_NUMERIC_STREAMS_MASK_BASE_CORRECTION_AND_AUTHOR_GEOMETRY_NO_TENSORFLOW_NO_NIRT",
        "view_count": len(records),
        "views": list(records),
        "metric_summary": metric_summary,
        "pooled_geometry": {
            "ray_count": total_rays,
            "full_box_zero_count": sum(
                int(item["full_box_zero_count"]) for item in records
            ),
            "box_miss_but_cone_nonzero_count": sum(
                int(item["box_miss_but_cone_nonzero_count"])
                for item in records
            ),
            "final_zero_length_count": sum(
                int(item["final_zero_length_count"]) for item in records
            ),
            "cone_segment_length_sum_m": total_cone_length,
            "cone_box_overlap_length_sum_m": total_cone_overlap,
            "cone_outside_length_sum_m": total_cone_length - total_cone_overlap,
            "cone_length_weighted_outside_box_fraction": (
                1.0 - total_cone_overlap / total_cone_length
                if total_cone_length
                else None
            ),
        },
        "prevalence": {
            "views_with_full_box_zero_rays": sum(
                item["full_box_zero_count"] > 0 for item in records
            ),
            "views_with_cone_outside_box_rays": sum(
                item["cone_outside_ray_count"] > 0 for item in records
            ),
            "views_with_active_unsafe_geometry": sum(
                item["active_unsafe_geometry_count"] > 0 for item in records
            ),
            "views_with_inactive_unsafe_geometry": sum(
                item["inactive_unsafe_geometry_count"] > 0 for item in records
            ),
            "setup_no_go_view_ids": no_go_views,
        },
        "decision": {
            "geometry_problem_is_single_view_artifact": "NO"
            if len(no_go_views) > 1
            else "UNRESOLVED",
            "official_setup_ready_for_training": "NO_GO"
            if no_go_views
            else "MECHANICAL_ONLY",
            "algorithm_success_claim": "LOCKED",
            "next_gate": "DOMAIN_CLIPPED_GEOMETRY_BASELINE_AND_GEOMETRY_SAFE_MASK_ABLATION",
        },
        "limitations": [
            "cross-view prevalence establishes a repeatable computational-geometry mechanism, not reconstruction superiority",
            "active and inactive masks are diagnostic labels rather than density ground truth",
            "the audit preserves the author cone/box formulas and does not yet test a corrected forward model",
            "no held-out reprojection, inverse field metric, neural baseline, or runtime comparison is available",
        ],
    }


def _write_csv(path: Path, records: Sequence[dict[str, Any]]) -> None:
    fieldnames = list(records[0])
    partial = path.with_name(f".{path.name}.partial")
    with partial.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    os.replace(partial, path)


def _run_all_view_audit_locked(
    *,
    mat_path: Path,
    geometry_source: Path,
    output_root: Path,
    image_height: int,
    image_width: int,
    view_count: int,
    chunk_measurements: int = 65_536,
    resume: bool = False,
    verify_resume_sha256: bool = True,
    overwrite: bool = False,
    repair_in_place: bool = False,
    refresh_aggregate: bool = False,
) -> dict[str, Any]:
    report_path = output_root / "all_view_geometry_audit.json"
    if report_path.exists() and not resume and not overwrite:
        raise FileExistsError(
            f"completed aggregate exists at {report_path}; use --resume or "
            "explicit --overwrite"
        )
    run_contract, run_contract_sha256 = _build_run_contract(
        mat_path=mat_path,
        geometry_source=geometry_source,
        image_height=image_height,
        image_width=image_width,
        view_count=view_count,
        chunk_measurements=chunk_measurements,
    )
    records: list[dict[str, Any]] = []
    reused_views: list[int] = []
    rebuilt_views: list[int] = []
    removed_partial_files: list[str] = []
    started = time.perf_counter()
    for view_id in range(view_count):
        view_dir = output_root / f"view_{view_id:02d}"
        bundle_dir = view_dir / "bundle"
        masks_dir = view_dir / "corrected_masks"
        setup_dir = view_dir / "setup"
        completed = None
        if resume:
            completed = _load_completed_view(
                view_dir=view_dir,
                mat_path=mat_path,
                geometry_source=geometry_source,
                view_id=view_id,
                image_height=image_height,
                image_width=image_width,
                view_count=view_count,
                verify_sha256=verify_resume_sha256,
            )
        if completed is not None:
            bundle, masks, setup = completed
            reused_views.append(view_id)
        else:
            existing_files = list(view_dir.rglob("*")) if view_dir.exists() else []
            if resume and existing_files and not repair_in_place:
                raise RuntimeError(
                    f"view {view_id} is incomplete or does not match the run contract; "
                    "refusing in-place repair without --repair-in-place"
                )
            removed_partial_files.extend(
                f"view_{view_id:02d}/{path}"
                for path in _remove_partial_npy_files(view_dir)
            )
            bundle = build_view_bundle(
                mat_path=mat_path,
                output_dir=bundle_dir,
                view_id=view_id,
                image_height=image_height,
                image_width=image_width,
                view_count=view_count,
                chunk_measurements=chunk_measurements,
            )
            masks = build_corrected_view_masks(
                mat_path=mat_path,
                view_bundle_dir=bundle_dir,
                output_dir=masks_dir,
                view_id=view_id,
                image_height=image_height,
                image_width=image_width,
                view_count=view_count,
            )
            setup = assemble_streamed_setup(
                view_bundle_dir=bundle_dir,
                geometry_source=geometry_source,
                output_dir=setup_dir,
                corrected_mask_dir=masks_dir,
                chunk_rows=chunk_measurements,
            )
            rebuilt_views.append(view_id)
        records.append(
            _view_record(view_id=view_id, bundle=bundle, masks=masks, setup=setup)
        )

    if resume and not rebuilt_views and report_path.exists():
        existing_report = _read_json_object(report_path)
        same_contract = (
            existing_report.get("run_contract_sha256") == run_contract_sha256
        )
        same_records = existing_report.get("views") == records
        complete = existing_report.get("execution_status") == "COMPLETE"
        if same_contract and same_records and complete and not refresh_aggregate:
            return existing_report
        if not refresh_aggregate:
            raise RuntimeError(
                "all views validate, but the existing aggregate is stale or uses a "
                "different validation contract; use --refresh-aggregate to replace "
                "only the aggregate JSON/CSV"
            )

    report = aggregate_view_records(records, expected_view_count=view_count)
    report["execution_status"] = "COMPLETE"
    report["scientific_verdict"] = (
        "NO_GO"
        if report["status"] == "ALL_VIEW_GEOMETRY_AUDIT_NO_GO"
        else "MECHANICAL_CONTRACT_PASS"
    )
    report["runtime_observation"] = {
        "wall_seconds": time.perf_counter() - started,
        "scope": "CACHED_LOCAL_SSD_RUN_NOT_A_SPEED_BENCHMARK",
    }
    report["source"] = run_contract["source"]
    report["run_contract"] = run_contract
    report["run_contract_sha256"] = run_contract_sha256
    report["resume_audit"] = {
        "requested": resume,
        "sha256_verified": bool(resume and verify_resume_sha256),
        "reused_view_ids": reused_views,
        "rebuilt_view_ids": rebuilt_views,
        "removed_partial_files": removed_partial_files,
    }
    _atomic_write_text(
        output_root / "all_view_geometry_audit.json",
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _write_csv(output_root / "all_view_geometry_metrics.csv", records)
    return report


def run_all_view_audit(
    *,
    mat_path: Path,
    geometry_source: Path,
    output_root: Path,
    image_height: int,
    image_width: int,
    view_count: int,
    chunk_measurements: int = 65_536,
    resume: bool = False,
    verify_resume_sha256: bool = True,
    overwrite: bool = False,
    repair_in_place: bool = False,
    refresh_aggregate: bool = False,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    with _exclusive_run_lock(output_root):
        return _run_all_view_audit_locked(
            mat_path=mat_path,
            geometry_source=geometry_source,
            output_root=output_root,
            image_height=image_height,
            image_width=image_width,
            view_count=view_count,
            chunk_measurements=chunk_measurements,
            resume=resume,
            verify_resume_sha256=verify_resume_sha256,
            overwrite=overwrite,
            repair_in_place=repair_in_place,
            refresh_aggregate=refresh_aggregate,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat", type=Path, required=True)
    parser.add_argument("--geometry-source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--image-height", type=int, required=True)
    parser.add_argument("--image-width", type=int, required=True)
    parser.add_argument("--view-count", type=int, required=True)
    parser.add_argument("--chunk-measurements", type=int, default=65_536)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse only views whose manifests, array contracts, and hashes validate",
    )
    parser.add_argument(
        "--skip-resume-sha256",
        action="store_true",
        help="resume with schema/shape/size checks only; weaker and recorded in the report",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="explicitly allow a non-resume run to overwrite a completed output root",
    )
    parser.add_argument(
        "--repair-in-place",
        action="store_true",
        help="explicitly allow resume to rebuild an invalid or incomplete view in place",
    )
    parser.add_argument(
        "--refresh-aggregate",
        action="store_true",
        help="replace only stale aggregate JSON/CSV after all view artifacts validate",
    )
    parser.add_argument(
        "--fail-on-no-go",
        action="store_true",
        help="return exit code 2 when execution completes but the scientific verdict is NO_GO",
    )
    args = parser.parse_args()
    report = run_all_view_audit(
        mat_path=args.mat,
        geometry_source=args.geometry_source,
        output_root=args.output_root,
        image_height=args.image_height,
        image_width=args.image_width,
        view_count=args.view_count,
        chunk_measurements=args.chunk_measurements,
        resume=args.resume,
        verify_resume_sha256=not args.skip_resume_sha256,
        overwrite=args.overwrite,
        repair_in_place=args.repair_in_place,
        refresh_aggregate=args.refresh_aggregate,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if args.fail_on_no_go and report["scientific_verdict"] == "NO_GO":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
