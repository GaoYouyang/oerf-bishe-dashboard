#!/usr/bin/env python3
"""Audit the seven-camera PSU rotation-40 cell payload and build private shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.io import loadmat, whosmat


ACCESS_SCHEMA = "psu-rotation40-development-access-private-1.0"
ACCESS_STATUS = "ROTATION40_DEVELOPMENT_MEMBER_EXTRACTED_AND_VERIFIED"
REPORT_SCHEMA = "psu-rotation40-cell-payload-private-1.0"
PUBLIC_SCHEMA = "psu-rotation40-cell-payload-public-1.0"
REPORT_STATUS = "ROTATION40_SEVEN_CAMERA_CELL_PAYLOAD_AUDITED_AND_PRIVATE_SHARDS_BUILT"
PUBLIC_STATUS = "ROTATION40_REAL_DISPLACEMENT_PAYLOAD_READY_GEOMETRY_AND_REPROJECTION_UNAVAILABLE"
REQUIRED_VARIABLES = (
    "typevector_free",
    "typevector_new",
    "u_new",
    "v_new",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def _save_npy_atomic(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    with partial.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
    os.replace(partial, path)


def _cells(value: np.ndarray, name: str) -> list[np.ndarray]:
    if value.shape != (1, 7) or value.dtype != object:
        raise ValueError(f"{name} must be a 1x7 MATLAB cell array")
    return [np.asarray(item) for item in value.ravel(order="C")]


def _binary_mask(values: np.ndarray, name: str, shape: tuple[int, int]) -> np.ndarray:
    if values.shape != shape or values.dtype.kind not in "biuf":
        raise ValueError(f"{name} has an unsupported shape or dtype")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.all((values == 0) | (values == 1)):
        raise ValueError(f"{name} is not binary")
    return np.asarray(values, dtype=bool)


def _finite_field(values: np.ndarray, name: str, shape: tuple[int, int]) -> np.ndarray:
    if values.shape != shape or values.dtype.kind not in "fc":
        raise ValueError(f"{name} has an unsupported shape or dtype")
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def _region_metrics(u: np.ndarray, v: np.ndarray, mask: np.ndarray) -> dict[str, float | int | None]:
    count = int(np.count_nonzero(mask))
    if count == 0:
        return {
            "pixel_count": 0,
            "u_mean_px": None,
            "v_mean_px": None,
            "u_rms_px": None,
            "v_rms_px": None,
            "vector_rms_px": None,
            "magnitude_p95_px": None,
        }
    selected_u = u[mask]
    selected_v = v[mask]
    magnitude = np.hypot(selected_u, selected_v)
    return {
        "pixel_count": count,
        "u_mean_px": float(np.mean(selected_u)),
        "v_mean_px": float(np.mean(selected_v)),
        "u_rms_px": _rms(selected_u),
        "v_rms_px": _rms(selected_v),
        "vector_rms_px": _rms(magnitude),
        "magnitude_p95_px": float(np.quantile(magnitude, 0.95)),
    }


def audit_rotation40_payload(
    *,
    mat_path: Path,
    access_report_path: Path,
    private_shard_root: Path,
    public_summary_path: Path | None = None,
    expected_shape: tuple[int, int] = (2160, 2560),
    shard_camera_ids: Sequence[int] = (2, 3, 4),
) -> tuple[dict[str, Any], dict[str, Any]]:
    mat_path = mat_path.resolve()
    access_report_path = access_report_path.resolve()
    private_shard_root = private_shard_root.resolve()
    access = _read_json(access_report_path)
    if access.get("schema_version") != ACCESS_SCHEMA or access.get("status") != ACCESS_STATUS:
        raise ValueError("rotation-40 access report is absent or unverified")
    dataset = access.get("dataset")
    if not isinstance(dataset, Mapping) or int(dataset.get("rotation_degrees", -1)) != 40:
        raise ValueError("access report does not authorize rotation 40")
    mat_sha = _sha256(mat_path)
    if dataset.get("extracted_sha256") != mat_sha:
        raise ValueError("rotation-40 MAT checksum differs from the access report")
    if int(dataset.get("member_uncompressed_bytes", -1)) != mat_path.stat().st_size:
        raise ValueError("rotation-40 MAT byte count differs from the access report")

    inventory = whosmat(mat_path)
    inventory_names = [name for name, _, _ in inventory]
    if inventory_names != list(REQUIRED_VARIABLES):
        raise ValueError("rotation-40 MAT top-level variable inventory changed")
    if any(shape != (1, 7) or kind != "cell" for _, shape, kind in inventory):
        raise ValueError("rotation-40 MAT variables must all be 1x7 cells")
    payload = loadmat(
        mat_path,
        variable_names=list(REQUIRED_VARIABLES),
        squeeze_me=False,
        struct_as_record=False,
    )
    free_cells = _cells(payload["typevector_free"], "typevector_free")
    active_cells = _cells(payload["typevector_new"], "typevector_new")
    u_cells = _cells(payload["u_new"], "u_new")
    v_cells = _cells(payload["v_new"], "v_new")
    requested = {int(camera_id) for camera_id in shard_camera_ids}
    if not requested or not requested.issubset(set(range(1, 8))):
        raise ValueError("shard camera ids must be a nonempty subset of 1..7")

    camera_rows: list[dict[str, Any]] = []
    shard_manifests: list[dict[str, Any]] = []
    for camera_id in range(1, 8):
        index = camera_id - 1
        free_source = _binary_mask(free_cells[index], f"camera {camera_id} typevector_free", expected_shape)
        active_source = _binary_mask(active_cells[index], f"camera {camera_id} typevector_new", expected_shape)
        u = _finite_field(u_cells[index], f"camera {camera_id} u_new", expected_shape)
        raw_v = _finite_field(v_cells[index], f"camera {camera_id} v_new", expected_shape)
        v = -raw_v
        ambient = ~free_source
        active = active_source & ~ambient
        excluded = ~(active | ambient)
        if np.any(active & ambient) or np.any(active & excluded) or np.any(ambient & excluded):
            raise RuntimeError("derived masks overlap")
        if not np.all(active | ambient | excluded):
            raise RuntimeError("derived masks do not partition the detector")
        total = int(np.prod(expected_shape))
        row = {
            "camera_id": camera_id,
            "image_shape_hw": list(expected_shape),
            "pixel_count": total,
            "active": _region_metrics(u, v, active),
            "ambient": _region_metrics(u, v, ambient),
            "excluded_pixel_count": int(np.count_nonzero(excluded)),
            "source_mask_one_fraction": {
                "typevector_free": float(np.mean(free_source)),
                "typevector_new": float(np.mean(active_source)),
            },
        }
        camera_rows.append(row)
        if camera_id not in requested:
            continue
        camera_dir = private_shard_root / f"camera_{camera_id:02d}"
        # MATLAB ``epsu(:)'`` and the calibration ray columns use column-major
        # pixel order. Flatten each detector plane explicitly before pairing
        # components so observation and geometry rows cannot be silently
        # transposed by NumPy's C-order default.
        measured = np.column_stack(
            (u.reshape(-1, order="F"), v.reshape(-1, order="F"))
        ).astype(np.float32)
        active_flat = np.ascontiguousarray(active.reshape(-1, order="F"), dtype=bool)
        ambient_flat = np.ascontiguousarray(ambient.reshape(-1, order="F"), dtype=bool)
        excluded_flat = np.ascontiguousarray(excluded.reshape(-1, order="F"), dtype=bool)
        files = {
            "measured_uv_px.npy": measured,
            "active_mask.npy": active_flat,
            "ambient_mask.npy": ambient_flat,
            "excluded_mask.npy": excluded_flat,
        }
        for filename, values in files.items():
            _save_npy_atomic(camera_dir / filename, values)
        manifest = {
            "schema_version": "psu-rotation40-camera-shard-1.0",
            "status": "ROTATION40_CAMERA_DISPLACEMENT_AND_MASK_SHARD_VERIFIED",
            "source_split": "ROTATION_40_DEVELOPMENT",
            "camera_id": camera_id,
            "rotation_degrees": 40,
            "image_shape_hw": list(expected_shape),
            "row_order": "MATLAB_COLUMN_MAJOR_MATCHING_AUTHOR_EPSU_COLON",
            "component_contract": "u_new_and_negative_v_new_pixels",
            "source_mat_sha256": mat_sha,
            "files": {
                filename: {
                    "shape": list(values.shape),
                    "dtype": str(values.dtype),
                    "sha256": _sha256(camera_dir / filename),
                }
                for filename, values in files.items()
            },
            "claim_boundary": {
                "contains_real_measurement_values": True,
                "contains_geometry": False,
                "contains_field_truth": False,
                "publishable": False,
            },
        }
        _write_json_atomic(camera_dir / "shard_manifest.json", manifest)
        shard_manifests.append(manifest)

    private_report = {
        "schema_version": REPORT_SCHEMA,
        "status": REPORT_STATUS,
        "evidence_scope": "REAL_ROTATION40_DISPLACEMENT_AND_AUTHOR_MASK_SEMANTICS_NO_GEOMETRY_NO_REPROJECTION_NO_FIELD_TRUTH",
        "source": {
            "filename": mat_path.name,
            "sha256": mat_sha,
            "bytes": mat_path.stat().st_size,
            "access_report_sha256": _sha256(access_report_path),
        },
        "camera_count": 7,
        "sharded_camera_ids": sorted(requested),
        "author_transform": {
            "u_px": "u_new",
            "v_px": "negative_v_new",
            "ambient_mask": "one_minus_typevector_free",
            "active_mask": "clip(typevector_new_minus_ambient,0,1)",
            "excluded_mask": "complement_of_active_union_ambient",
        },
        "camera_rows": camera_rows,
        "shard_manifests": shard_manifests,
        "claim_boundary": {
            "real_measurement_values_interpreted": True,
            "rotation40_is_development_only": True,
            "camera_geometry_available": False,
            "reprojection_scored": False,
            "experimental_field_truth_available": False,
            "algorithm_superiority": False,
            "final_rotations_opened": False,
        },
    }
    _write_json_atomic(private_shard_root / "payload_private_report.json", private_report)
    public_summary = {
        "schema_version": PUBLIC_SCHEMA,
        "status": PUBLIC_STATUS,
        "evidence_scope": private_report["evidence_scope"],
        "dataset": {
            "doi": "10.26208/1VE2-5C19",
            "rotation_degrees": 40,
            "camera_count": 7,
            "image_shape_hw": list(expected_shape),
            "pixel_count_per_camera": int(np.prod(expected_shape)),
        },
        "author_transform": dict(private_report["author_transform"]),
        "row_order": "MATLAB_COLUMN_MAJOR_MATCHING_AUTHOR_EPSU_COLON",
        "camera_rows": camera_rows,
        "private_sharded_camera_ids": sorted(requested),
        "claim_boundary": dict(private_report["claim_boundary"]),
        "missing_for_reprojection": [
            "rotation40_camera_extrinsics",
            "rotation40_background_extrinsics",
            "rotation40_per_pixel_ray_directions",
            "camera_system_constants_bound_to_the_same_rows",
        ],
        "public_export_policy": {
            "contains_local_paths": False,
            "contains_measurement_arrays": False,
            "contains_masks": False,
            "contains_only_aggregate_camera_statistics": True,
        },
    }
    if public_summary_path is not None:
        _write_json_atomic(public_summary_path.resolve(), public_summary)
    return private_report, public_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat", type=Path, required=True)
    parser.add_argument("--access-report", type=Path, required=True)
    parser.add_argument("--private-shards", type=Path, required=True)
    parser.add_argument("--public-summary", type=Path)
    args = parser.parse_args()
    _, public = audit_rotation40_payload(
        mat_path=args.mat,
        access_report_path=args.access_report,
        private_shard_root=args.private_shards,
        public_summary_path=args.public_summary,
    )
    print(json.dumps(public, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
