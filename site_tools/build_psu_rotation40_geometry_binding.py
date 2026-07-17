#!/usr/bin/env python3
"""Bind PSU rotation-40 observations to verified per-pixel geometry rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.io import loadmat


CONFIG_SCHEMA = "psu-rotation40-geometry-binding-config-1.0"
CONFIG_STATUS = "FROZEN_BEFORE_ROTATION40_GEOMETRY_BINDING"
OBSERVATION_STATUS = "ROTATION40_CAMERA_DISPLACEMENT_AND_MASK_SHARD_VERIFIED"
SUPPORT_STATUS = "VIEW_SHARD_BUNDLE_TRANSCODED_AND_SOURCE_STREAMS_VERIFIED"
PRIVATE_SCHEMA = "psu-rotation40-geometry-binding-private-1.0"
PUBLIC_SCHEMA = "psu-rotation40-geometry-binding-public-1.0"
PRIVATE_STATUS = "ROTATION40_ACTIVE_ROW_GEOMETRY_AND_OBSERVATIONS_BOUND_PRIVATE"
PUBLIC_STATUS = "ROTATION40_GEOMETRY_ROW_BINDING_VERIFIED_DEVELOPMENT_REPROJECTION_READY"
VECTOR_FIELDS = ("c", "v", "Ruvecs", "Rvvecs", "Rxvecs", "Ryvecs")
SCALAR_FIELDS = ("Rapvec", "Dfvec", "Csys_all")


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


def _rotation_x(degrees: float) -> np.ndarray:
    angle = np.deg2rad(float(degrees))
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=np.float64,
    )


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA or config.get("status") != CONFIG_STATUS:
        raise ValueError("rotation-40 geometry config is absent or not frozen")
    dataset = config.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("dataset contract is absent")
    if int(dataset.get("rotation_degrees", -1)) != 40:
        raise ValueError("only rotation 40 is authorized")
    if dataset.get("camera_ids") != [2, 3, 4]:
        raise ValueError("geometry binding must remain cameras 2/3/4")
    if dataset.get("row_order") != "MATLAB_COLUMN_MAJOR_MATCHING_AUTHOR_EPSU_COLON":
        raise ValueError("the frozen MATLAB row-order contract changed")
    source = config.get("geometry_source")
    if not isinstance(source, Mapping) or int(source.get("bytes", 0)) < 1:
        raise ValueError("geometry source contract is absent")
    sha = source.get("sha256")
    if not isinstance(sha, str) or len(sha) != 64:
        raise ValueError("geometry source SHA-256 is invalid")
    mapping = config.get("camera_mapping")
    if not isinstance(mapping, Mapping) or set(mapping) != {"2", "3", "4"}:
        raise ValueError("camera mapping must contain exactly cameras 2/3/4")
    expected = {
        "2": (0, 3, 6),
        "3": (1, 4, 7),
        "4": (2, 5, 8),
    }
    for camera, ids in expected.items():
        row = mapping[camera]
        actual = (
            int(row.get("support_zero_view_id", -1)),
            int(row.get("support_50_view_id", -1)),
            int(row.get("support_90_view_id", -1)),
        )
        if actual != ids:
            raise ValueError(f"camera {camera} support-view mapping changed")
    firewall = config.get("claim_firewall")
    if not isinstance(firewall, Mapping) or any(
        firewall.get(key) is not False
        for key in (
            "final_rotations_opened",
            "experimental_field_truth_available",
            "reprojection_scored_by_this_stage",
            "algorithm_superiority",
            "publish_raw_geometry_or_measurements",
        )
    ):
        raise ValueError("claim firewall is incomplete")


def _max_transform_error(
    source: np.ndarray,
    target: np.ndarray,
    transform: np.ndarray,
    *,
    chunk_rows: int,
) -> tuple[float, float]:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("vector fields must have matching (N,3) shapes")
    maximum = 0.0
    square_sum = 0.0
    count = 0
    for start in range(0, source.shape[0], chunk_rows):
        stop = min(start + chunk_rows, source.shape[0])
        left = np.asarray(source[start:stop], dtype=np.float64) @ transform
        right = np.asarray(target[start:stop], dtype=np.float64)
        difference = left - right
        maximum = max(maximum, float(np.max(np.abs(difference))))
        square_sum += float(np.sum(np.square(difference)))
        count += int(difference.size)
    return maximum, float(np.sqrt(square_sum / max(count, 1)))


def _max_scalar_error(first: np.ndarray, second: np.ndarray, *, chunk_rows: int) -> float:
    if first.shape != second.shape or first.ndim != 2 or first.shape[1] != 1:
        raise ValueError("scalar fields must have matching (N,1) shapes")
    maximum = 0.0
    for start in range(0, first.shape[0], chunk_rows):
        stop = min(start + chunk_rows, first.shape[0])
        difference = np.asarray(first[start:stop], dtype=np.float64) - np.asarray(
            second[start:stop], dtype=np.float64
        )
        maximum = max(maximum, float(np.max(np.abs(difference))))
    return maximum


def _array_manifest(path: Path, values: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "sha256": _sha256(path),
    }


def build_rotation40_geometry_binding(
    *,
    config_path: Path,
    geometry_mat_path: Path,
    support_view_root: Path,
    observation_root: Path,
    output_root: Path,
    public_summary_path: Path | None = None,
    chunk_rows: int = 250_000,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = config_path.resolve()
    geometry_mat_path = geometry_mat_path.resolve()
    support_view_root = support_view_root.resolve()
    observation_root = observation_root.resolve()
    output_root = output_root.resolve()
    config = _read_json(config_path)
    validate_config(config)
    source_contract = config["geometry_source"]
    if geometry_mat_path.stat().st_size != int(source_contract["bytes"]):
        raise ValueError("geometry MAT byte count differs from the frozen contract")
    geometry_sha = _sha256(geometry_mat_path)
    if geometry_sha != source_contract["sha256"]:
        raise ValueError("geometry MAT checksum differs from the frozen contract")

    payload = loadmat(
        geometry_mat_path,
        variable_names=list(source_contract["required_variables"]),
        squeeze_me=False,
        struct_as_record=False,
    )
    angle = int(np.asarray(payload["modelAngle"]).reshape(-1)[0])
    if angle != 40:
        raise ValueError("geometry MAT is not rotation 40")
    homogeneous = np.asarray(payload["Arotcam"], dtype=np.float64)
    if homogeneous.shape != (4, 4):
        raise ValueError("Arotcam must be a 4x4 homogeneous transform")
    transform = homogeneous[:3, :3]
    tolerance = config["tolerances"]
    orthogonality = float(np.max(np.abs(transform.T @ transform - np.eye(3))))
    if orthogonality > float(tolerance["rotation_matrix_orthogonality_max_abs"]):
        raise ValueError("Arotcam rotation block is not orthogonal")
    if not np.allclose(homogeneous[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12, rtol=0.0):
        raise ValueError("Arotcam homogeneous row changed")
    if float(np.linalg.det(transform)) <= 0.0:
        raise ValueError("Arotcam is not a proper rotation")
    ray_cells = payload["ray_dir"]
    if ray_cells.shape != (7, 1) or ray_cells.dtype != object:
        raise ValueError("ray_dir must be a 7x1 MATLAB cell array")

    camera_rows: list[dict[str, Any]] = []
    private_manifests: list[dict[str, Any]] = []
    for camera_id in config["dataset"]["camera_ids"]:
        mapping = config["camera_mapping"][str(camera_id)]
        view_zero = int(mapping["support_zero_view_id"])
        view_50 = int(mapping["support_50_view_id"])
        view_90 = int(mapping["support_90_view_id"])
        support_dirs = {
            angle_key: support_view_root / f"view_{view_id:02d}" / "bundle"
            for angle_key, view_id in ((0, view_zero), (50, view_50), (90, view_90))
        }
        support_arrays: dict[int, dict[str, np.ndarray]] = {}
        for support_angle, directory in support_dirs.items():
            manifest = _read_json(directory / "view_bundle_manifest.json")
            if manifest.get("status") != SUPPORT_STATUS:
                raise ValueError(f"camera {camera_id} support view {support_angle} is unverified")
            support_arrays[support_angle] = {
                field: np.load(directory / f"{field}.npy", mmap_mode="r", allow_pickle=False)
                for field in (*VECTOR_FIELDS, *SCALAR_FIELDS)
            }
        row_count = int(support_arrays[0]["v"].shape[0])
        for fields in support_arrays.values():
            if any(array.shape[0] != row_count for array in fields.values()):
                raise ValueError("support-view row counts differ")

        known_rotation_errors: dict[str, dict[str, float]] = {}
        for known_angle in (50, 90):
            rotation = _rotation_x(known_angle)
            for field in ("c", "v", "Ruvecs", "Rvvecs"):
                maximum, rms = _max_transform_error(
                    support_arrays[0][field],
                    support_arrays[known_angle][field],
                    rotation,
                    chunk_rows=chunk_rows,
                )
                known_rotation_errors[f"rotation_{known_angle}_{field}"] = {
                    "max_abs": maximum,
                    "rms": rms,
                }
                if maximum > float(tolerance["known_rotation_row_max_abs"]):
                    raise ValueError(f"known rotation audit failed for camera {camera_id}/{field}")
        scalar_errors: dict[str, float] = {}
        for field in SCALAR_FIELDS:
            for known_angle in (50, 90):
                maximum = _max_scalar_error(
                    support_arrays[0][field],
                    support_arrays[known_angle][field],
                    chunk_rows=chunk_rows,
                )
                scalar_errors[f"rotation_{known_angle}_{field}"] = maximum
                if maximum > float(tolerance["scalar_invariance_max_abs"]):
                    raise ValueError(f"scalar invariance failed for camera {camera_id}/{field}")

        official_v = np.asarray(ray_cells[camera_id - 1, 0], dtype=np.float64).T
        if official_v.shape != (row_count, 3) or not np.all(np.isfinite(official_v)):
            raise ValueError(f"camera {camera_id} official ray_dir has an invalid shape")
        ray_maximum, ray_rms = _max_transform_error(
            support_arrays[0]["v"], official_v, transform, chunk_rows=chunk_rows
        )
        if ray_maximum > float(tolerance["official_rotation40_ray_max_abs"]):
            raise ValueError(f"camera {camera_id} official rotation-40 ray rows do not bind")

        observation_dir = observation_root / f"camera_{camera_id:02d}"
        observation_manifest = _read_json(observation_dir / "shard_manifest.json")
        if observation_manifest.get("status") != OBSERVATION_STATUS:
            raise ValueError(f"camera {camera_id} observation shard is unverified")
        if observation_manifest.get("row_order") != config["dataset"]["row_order"]:
            raise ValueError(f"camera {camera_id} observation row order differs from geometry")
        measured = np.load(observation_dir / "measured_uv_px.npy", mmap_mode="r", allow_pickle=False)
        active_mask = np.load(observation_dir / "active_mask.npy", mmap_mode="r", allow_pickle=False)
        if measured.shape != (row_count, 2) or active_mask.shape != (row_count,):
            raise ValueError(f"camera {camera_id} observation row count differs from geometry")
        active_indices = np.flatnonzero(np.asarray(active_mask, dtype=bool)).astype(np.int64)
        if active_indices.size == 0:
            raise ValueError(f"camera {camera_id} has no active rows")

        camera_dir = output_root / f"camera_{camera_id:02d}"
        arrays: dict[str, np.ndarray] = {
            "active_indices": np.ascontiguousarray(active_indices),
            "measured_uv_px": np.ascontiguousarray(measured[active_indices], dtype=np.float32),
            "v": np.ascontiguousarray(official_v[active_indices], dtype=np.float32),
        }
        for field in ("c", "Ruvecs", "Rvvecs", "Rxvecs", "Ryvecs"):
            arrays[field] = np.ascontiguousarray(
                np.asarray(support_arrays[0][field][active_indices], dtype=np.float64) @ transform,
                dtype=np.float32,
            )
        for field in SCALAR_FIELDS:
            arrays[field] = np.ascontiguousarray(
                support_arrays[0][field][active_indices], dtype=np.float32
            )
        for filename, values in arrays.items():
            _save_npy_atomic(camera_dir / f"{filename}.npy", values)
        file_manifest = {
            f"{filename}.npy": _array_manifest(camera_dir / f"{filename}.npy", values)
            for filename, values in arrays.items()
        }
        manifest = {
            "schema_version": "psu-rotation40-active-geometry-camera-1.0",
            "status": PRIVATE_STATUS,
            "camera_id": camera_id,
            "rotation_degrees": 40,
            "row_order": config["dataset"]["row_order"],
            "full_detector_row_count": row_count,
            "active_row_count": int(active_indices.size),
            "geometry_source_sha256": geometry_sha,
            "observation_manifest_sha256": _sha256(observation_dir / "shard_manifest.json"),
            "support_zero_manifest_sha256": _sha256(
                support_dirs[0] / "view_bundle_manifest.json"
            ),
            "official_ray_binding": {"max_abs": ray_maximum, "rms": ray_rms},
            "known_rotation_errors": known_rotation_errors,
            "scalar_invariance_errors": scalar_errors,
            "files": file_manifest,
            "publishable": False,
        }
        _write_json_atomic(camera_dir / "geometry_manifest.json", manifest)
        private_manifests.append(manifest)
        camera_rows.append(
            {
                "camera_id": camera_id,
                "full_detector_row_count": row_count,
                "active_row_count": int(active_indices.size),
                "official_ray_binding_max_abs": ray_maximum,
                "official_ray_binding_rms": ray_rms,
                "known_rotation_max_abs": max(
                    row["max_abs"] for row in known_rotation_errors.values()
                ),
                "scalar_invariance_max_abs": max(scalar_errors.values()),
            }
        )

    private_report = {
        "schema_version": PRIVATE_SCHEMA,
        "status": PRIVATE_STATUS,
        "config_sha256": _sha256(config_path),
        "geometry_source": {
            "bytes": geometry_mat_path.stat().st_size,
            "sha256": geometry_sha,
            "model_angle_degrees": angle,
        },
        "rotation_matrix": transform.tolist(),
        "rotation_matrix_orthogonality_max_abs": orthogonality,
        "camera_rows": camera_rows,
        "camera_manifests": private_manifests,
        "claim_boundary": dict(config["claim_firewall"]),
    }
    _write_json_atomic(output_root / "geometry_binding_private_report.json", private_report)
    public_summary = {
        "schema_version": PUBLIC_SCHEMA,
        "status": PUBLIC_STATUS,
        "evidence_scope": "REAL_ROTATION40_OBSERVATION_TO_GEOMETRY_ROW_BINDING_NO_REPROJECTION_SCORE_NO_FIELD_TRUTH",
        "dataset": {
            "doi": config["dataset"]["doi"],
            "rotation_degrees": 40,
            "camera_ids": [2, 3, 4],
            "row_order": config["dataset"]["row_order"],
        },
        "geometry_source": {
            "bytes": geometry_mat_path.stat().st_size,
            "sha256": geometry_sha,
            "model_angle_degrees": angle,
        },
        "rotation_matrix": transform.tolist(),
        "rotation_matrix_orthogonality_max_abs": orthogonality,
        "camera_rows": camera_rows,
        "claim_boundary": {
            "development_only": True,
            "camera_geometry_available": True,
            "reprojection_scored": False,
            "experimental_field_truth_available": False,
            "algorithm_superiority": False,
            "final_rotations_opened": False,
        },
        "public_export_policy": {
            "contains_local_paths": False,
            "contains_geometry_arrays": False,
            "contains_measurement_arrays": False,
            "contains_masks": False,
            "contains_only_hashes_counts_and_error_aggregates": True,
        },
    }
    if public_summary_path is not None:
        _write_json_atomic(public_summary_path.resolve(), public_summary)
    return private_report, public_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--geometry-mat", type=Path, required=True)
    parser.add_argument("--support-view-root", type=Path, required=True)
    parser.add_argument("--observation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--public-summary", type=Path)
    args = parser.parse_args()
    _, public = build_rotation40_geometry_binding(
        config_path=args.config,
        geometry_mat_path=args.geometry_mat,
        support_view_root=args.support_view_root,
        observation_root=args.observation_root,
        output_root=args.output_root,
        public_summary_path=args.public_summary,
    )
    print(json.dumps(public, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
