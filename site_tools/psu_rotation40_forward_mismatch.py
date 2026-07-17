#!/usr/bin/env python3
"""Fail-closed rotation-held-out H2 author-array versus B0 diagnostic.

The real author bundle is deliberately an external input.  This runner never
pretends to construct it: it verifies the private provenance graph first, then
recomputes only the B0 fixed-domain forward sensitivity probe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Keep both ``python -m`` and direct CLI execution rooted at the repository.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo_t16_operator.psu_b0_reconstruction_interface import (
    PSUB0VoxelGradientOperator,
    build_trilinear_stencil,
)
from site_tools.psu_b0_real_support_store import (
    MASK_STATUS,
    VIEW_BUNDLE_STATUS,
    deterministic_quantile_indices,
)
from site_tools.psu_bost_aperture_domain import (
    deterministic_paired_uniform_aperture_samples,
    generate_aperture_sample_points,
)
from site_tools.psu_bost_forward_geometry import intersect_forward_ray_box


PREREG_SCHEMA = "psu-rotation40-forward-mismatch-preregistration-1.1"
REPORT_SCHEMA = "psu-rotation40-forward-mismatch-private-report-1.1"
REPORT_STATUS = "H2_FORWARD_MISMATCH_DIAGNOSTIC_COMPLETE_B0_SENSITIVITY_ONLY_NO_FIELD_TRUTH_NO_SUPERIORITY"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
VECTOR_FIELDS = ("c", "v", "Ruvecs", "Rvvecs", "Rxvecs", "Ryvecs")
SCALAR_FIELDS = ("epsu_all", "epsv_all", "Csys_all", "Rapvec", "Dfvec")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required private input is absent: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path.name}")
    return value


def _load_array(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"required private input is absent: {path.name}")
    return np.load(path, mmap_mode="r", allow_pickle=False)


def _finite_array(
    values: np.ndarray,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} has shape {array.shape}, expected {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _require_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _safe_path(root: Path, relative: Any, name: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"{name} must be a relative private path")
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"{name} escapes the private bundle root")
    return candidate


def _validate_preregistration(config: dict[str, Any]) -> None:
    if config.get("schema_version") != PREREG_SCHEMA:
        raise ValueError("unexpected preregistration schema")
    if config.get("status") != "UNCONSTRUCTED_FROZEN_DEVELOPMENT_ONLY_H2_DIAGNOSTIC":
        raise ValueError("H2 preregistration must remain explicitly UNCONSTRUCTED")
    partition = config.get("development_partition", {})
    if (
        partition.get("rotation_degrees") != 40
        or partition.get("camera_ids") != [2, 3, 4]
        or partition.get("support_camera_ids") != [2, 3, 4]
        or partition.get("support_rotation_degrees") != [0, 50, 90]
        or partition.get("held_out_unit") != "rotation_run_not_camera"
        or partition.get("camera_held_out") is not False
    ):
        raise ValueError("H2 scope must be rotation-held-out cameras 2/3/4 only")
    bounded = config.get("bounded_execution", {})
    if int(bounded.get("rays_per_camera", 0)) < 1:
        raise ValueError("rays_per_camera must be positive")
    if [int(value) for value in bounded.get("qmc_sample_counts", [])] != [8, 16, 32]:
        raise ValueError("H2 scope must use B0 sensitivity probes 8/16/32 exactly")
    if bounded.get("qmc_design") != "deterministic_b0_sensitivity_probe_not_paired_convergence":
        raise ValueError("QMC schedule must not be labeled paired convergence")
    if bounded.get("qmc_interpretation") != "B0_SENSITIVITY_ONLY_AUTHOR_ARRAY_IS_SINGLE_FIXED_ARTIFACT":
        raise ValueError("QMC interpretation must be B0 sensitivity only")
    if config.get("frozen_field", {}).get("grid_shape_zyx") != [32, 32, 32]:
        raise ValueError("H2 scope must use a frozen 32^3 field")
    boundary = config.get("claim_boundary", {})
    if (
        boundary.get("experimental_field_truth")
        or boundary.get("diagnostic_is_algorithm_superiority")
        or boundary.get("camera_held_out") is not False
        or boundary.get("qmc_8_16_32_is_paired_convergence") is not False
        or boundary.get("rotation_40_is_development_only") is not True
    ):
        raise ValueError("H2 claim boundary is not fail-closed")
    generator = config.get("generator", {})
    files = generator.get("source_files")
    hashes = generator.get("source_sha256")
    if not isinstance(files, list) or not files or not isinstance(hashes, dict) or set(files) != set(hashes):
        raise ValueError("generator source hashes are incomplete")
    for path in files:
        _require_sha(hashes.get(path), f"generator.source_sha256[{path}]")


def _source_fingerprint(
    repo_root: Path,
    config: dict[str, Any],
) -> tuple[dict[str, str], str, str]:
    generator = config["generator"]
    actual: dict[str, str] = {}
    for relative in generator["source_files"]:
        path = _safe_path(repo_root, relative, "generator source filename")
        if not path.is_file():
            raise FileNotFoundError(f"generator source is absent: {relative}")
        digest = _sha256(path)
        expected = generator["source_sha256"][relative]
        if digest != expected:
            raise ValueError(f"generator source checksum mismatch: {relative}")
        actual[relative] = digest
    canonical = "".join(f"{path}\0{actual[path]}\n" for path in sorted(actual))
    fingerprint = hashlib.sha256(canonical.encode("ascii")).hexdigest()
    return actual, fingerprint, str(generator["version"])


def _require_provenance(
    manifest: dict[str, Any],
    *,
    location: str,
    geometry_hash: str,
    generator_fingerprint: str,
    generator_version: str,
) -> None:
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{location} provenance is absent")
    if provenance.get("geometry_sha256") != geometry_hash:
        raise ValueError(f"{location} geometry provenance is inconsistent")
    if provenance.get("generator_source_fingerprint") != generator_fingerprint:
        raise ValueError(f"{location} generator source provenance is inconsistent")
    if provenance.get("generator_version") != generator_version:
        raise ValueError(f"{location} generator version is inconsistent")
    if provenance.get("source_split") not in {
        "SUPPORT_ONLY_FROZEN_BEFORE_ROTATION_40_ACCESS",
        "ROTATION_40_DEVELOPMENT",
    }:
        raise ValueError(f"{location} source split is not declared")


def _load_geometry_provenance(
    root: Path,
    config: dict[str, Any],
    generator_fingerprint: str,
    generator_version: str,
) -> dict[str, str]:
    contract = config["geometry_provenance"]
    manifest_path = _safe_path(root, contract["required_manifest_filename"], "geometry manifest filename")
    manifest = _load_json(manifest_path)
    if manifest.get("status") != contract["required_manifest_status"]:
        raise ValueError("geometry provenance is not frozen")
    source_filename = manifest.get("source_filename")
    if source_filename != contract["required_source_filename"]:
        raise ValueError("geometry source filename differs from the preregistration")
    source_path = _safe_path(root, source_filename, "geometry source filename")
    if not source_path.is_file():
        raise FileNotFoundError(f"required private input is absent: {source_path.name}")
    geometry_hash = _sha256(source_path)
    if manifest.get("source_sha256") != geometry_hash:
        raise ValueError("geometry source checksum does not match its manifest")
    _require_sha(geometry_hash, "geometry source checksum")
    if manifest.get("source_split") != contract["required_source_split"]:
        raise ValueError("geometry source split is not rotation-40 development")
    if manifest.get("frozen_before_rotation40_access") is not True:
        raise ValueError("geometry provenance was not frozen before rotation-40 access")
    if manifest.get("generator_source_fingerprint") != generator_fingerprint:
        raise ValueError("geometry generator source provenance is inconsistent")
    if manifest.get("generator_version") != generator_version:
        raise ValueError("geometry generator version is inconsistent")
    return {
        "source_sha256": geometry_hash,
        "manifest_sha256": _sha256(manifest_path),
    }


def _load_frozen_field(
    root: Path,
    config: dict[str, Any],
    geometry_hash: str,
    generator_fingerprint: str,
    generator_version: str,
) -> tuple[np.ndarray, dict[str, str]]:
    field_config = config["frozen_field"]
    field_path = _safe_path(root, field_config["required_filename"], "field filename")
    manifest_path = _safe_path(root, field_config["required_manifest_filename"], "field manifest filename")
    manifest = _load_json(manifest_path)
    field = _finite_array(
        _load_array(field_path),
        name="frozen support field",
        shape=tuple(int(value) for value in field_config["grid_shape_zyx"]),
    )
    field_hash = _sha256(field_path)
    if manifest.get("status") != field_config["required_manifest_status"]:
        raise ValueError("frozen field manifest status is not approved")
    if manifest.get("grid_shape_zyx") != field_config["grid_shape_zyx"]:
        raise ValueError("frozen field manifest grid differs from preregistration")
    if manifest.get("bounds_m") != field_config["bounds_m"]:
        raise ValueError("frozen field bounds differ from preregistration")
    if manifest.get("field_sha256") != field_hash:
        raise ValueError("frozen field checksum does not match its manifest")
    if manifest.get("source_split") != field_config["required_source_split"]:
        raise ValueError("frozen field is not support-only")
    if manifest.get("frozen_before_rotation40_access") is not True:
        raise ValueError("frozen field was not frozen before rotation-40 access")
    if manifest.get("geometry_sha256") != geometry_hash:
        raise ValueError("frozen field geometry provenance is inconsistent")
    if manifest.get("generator_source_fingerprint") != generator_fingerprint:
        raise ValueError("frozen field generator source provenance is inconsistent")
    if manifest.get("generator_version") != generator_version:
        raise ValueError("frozen field generator version is inconsistent")
    return field, {
        "field_sha256": field_hash,
        "manifest_sha256": _sha256(manifest_path),
    }


def _metric_block(
    measurement: np.ndarray,
    author_exact: np.ndarray,
    fixed_domain: np.ndarray,
) -> dict[str, float | int]:
    if measurement.shape != author_exact.shape or measurement.shape != fixed_domain.shape:
        raise ValueError("measurement, author, and B0 predictions must have identical shapes")
    author_residual = measurement - author_exact
    fixed_residual = measurement - fixed_domain
    mismatch = author_exact - fixed_domain
    count = int(measurement.size)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            values: dict[str, float | int] = {
                "measurement_squared_l2": float(np.sum(measurement * measurement)),
                "author_exact_residual_squared_l2": float(np.sum(author_residual * author_residual)),
                "fixed_domain_residual_squared_l2": float(np.sum(fixed_residual * fixed_residual)),
                "forward_mismatch_squared_l2": float(np.sum(mismatch * mismatch)),
                "component_count": count,
            }
            denominator = max(float(values["measurement_squared_l2"]), 1e-30)
            for name in ("author_exact_residual", "fixed_domain_residual", "forward_mismatch"):
                squared = float(values[f"{name}_squared_l2"])
                values[f"{name}_l2"] = float(np.sqrt(squared))
                values[f"{name}_rms"] = float(np.sqrt(squared / count))
                values[f"{name}_relative_l2"] = float(np.sqrt(squared / denominator))
            values["measurement_l2"] = float(np.sqrt(float(values["measurement_squared_l2"])))
    except FloatingPointError as exc:
        raise ValueError("metric computation overflowed or produced an invalid value") from exc
    if any(not np.isfinite(float(value)) for key, value in values.items() if key != "component_count"):
        raise ValueError("metric computation produced a non-finite value")
    return values


def _domain_block(valid: np.ndarray, centerline_hit: np.ndarray) -> dict[str, int | float]:
    per_ray = np.count_nonzero(valid, axis=1)
    sample_count = int(valid.shape[1])
    valid_count = int(np.count_nonzero(valid))
    ray_count = int(valid.shape[0])
    hit_count = int(np.count_nonzero(centerline_hit))
    return {
        "ray_count": ray_count,
        "qmc_sample_count": sample_count,
        "total_sample_count": int(valid.size),
        "valid_sample_count": valid_count,
        "valid_sample_fraction": float(valid_count / valid.size),
        "empty_ray_count": int(np.count_nonzero(per_ray == 0)),
        "partial_ray_count": int(np.count_nonzero((per_ray > 0) & (per_ray < sample_count))),
        "full_ray_count": int(np.count_nonzero(per_ray == sample_count)),
        "centerline_ray_count": ray_count,
        "centerline_hit_count": hit_count,
        "centerline_miss_count": ray_count - hit_count,
        "centerline_hit_fraction": float(hit_count / ray_count),
    }


def _verify_bundle_arrays(
    bundle_root: Path,
    bundle: dict[str, Any],
    measurement_count: int,
    camera_id: int,
) -> dict[str, np.ndarray]:
    variables = bundle.get("variables")
    if not isinstance(variables, list) or len(variables) != len(VECTOR_FIELDS) + len(SCALAR_FIELDS):
        raise ValueError(f"camera {camera_id} bundle variable manifest is incomplete")
    by_name = {item.get("name"): item for item in variables if isinstance(item, dict)}
    expected_names = set(VECTOR_FIELDS + SCALAR_FIELDS)
    if set(by_name) != expected_names:
        raise ValueError(f"camera {camera_id} bundle variable names differ from the contract")
    arrays: dict[str, np.ndarray] = {}
    for name in VECTOR_FIELDS + SCALAR_FIELDS:
        array_path = bundle_root / f"{name}.npy"
        array = _load_array(array_path)
        expected_shape = (measurement_count, 3) if name in VECTOR_FIELDS else (measurement_count, 1)
        if array.shape != expected_shape or array.dtype.kind not in "fc":
            raise ValueError(f"camera {camera_id} {name} has an unsupported shape or dtype")
        _finite_array(array, name=f"camera {camera_id} {name}", shape=expected_shape)
        item = by_name[name]
        if item.get("shard_shape") != list(expected_shape) or item.get("shard_dtype") != array.dtype.name:
            raise ValueError(f"camera {camera_id} {name} manifest shape or dtype differs")
        if item.get("shard_sha256") != _sha256(array_path):
            raise ValueError(f"camera {camera_id} {name} checksum differs from its manifest")
        arrays[name] = array
    return arrays


def _load_camera(
    *,
    root: Path,
    camera_id: int,
    config: dict[str, Any],
    field_hash: str,
    geometry_hash: str,
    generator_fingerprint: str,
    generator_version: str,
) -> dict[str, Any]:
    contract = config["private_bundle_contract"]
    camera_root = _safe_path(
        root,
        str(contract["camera_directory_pattern"]).format(camera_id=camera_id),
        "camera directory",
    )
    identity_path = camera_root / str(contract["required_camera_identity_filename"])
    identity = _load_json(identity_path)
    identity_hash = _sha256(identity_path)
    if identity.get("status") != contract["camera_identity_status"]:
        raise ValueError(f"camera {camera_id} identity is not frozen")
    if int(identity.get("camera_id", -1)) != camera_id or int(identity.get("rotation_degrees", -1)) != 40:
        raise ValueError(f"camera {camera_id} is not the required rotation-40 partition")
    if identity.get("measurement_count", -1) < 1:
        raise ValueError(f"camera {camera_id} identity has no measurements")
    _require_provenance(identity, location=f"camera {camera_id} identity", geometry_hash=geometry_hash, generator_fingerprint=generator_fingerprint, generator_version=generator_version)
    if identity.get("provenance", {}).get("source_split") != "ROTATION_40_DEVELOPMENT":
        raise ValueError(f"camera {camera_id} identity is not rotation-40 development data")

    bundle_root = camera_root / "bundle"
    mask_root = camera_root / "corrected_masks"
    bundle_path = bundle_root / "view_bundle_manifest.json"
    mask_path = mask_root / "corrected_view_masks_manifest.json"
    bundle = _load_json(bundle_path)
    masks = _load_json(mask_path)
    bundle_hash = _sha256(bundle_path)
    mask_manifest_hash = _sha256(mask_path)
    if bundle.get("status") != VIEW_BUNDLE_STATUS or masks.get("status") != MASK_STATUS:
        raise ValueError(f"camera {camera_id} bundle or mask contract is unverified")
    _require_provenance(bundle, location=f"camera {camera_id} bundle", geometry_hash=geometry_hash, generator_fingerprint=generator_fingerprint, generator_version=generator_version)
    _require_provenance(masks, location=f"camera {camera_id} mask manifest", geometry_hash=geometry_hash, generator_fingerprint=generator_fingerprint, generator_version=generator_version)
    bundle_view = bundle.get("view", {})
    mask_view = masks.get("view", {})
    measurement_count = int(bundle_view.get("measurement_count", -1))
    if (
        bundle_view.get("camera_id") != camera_id
        or bundle_view.get("rotation_degrees") != 40
        or mask_view.get("camera_id") != camera_id
        or mask_view.get("rotation_degrees") != 40
        or int(mask_view.get("measurement_count", -1)) != measurement_count
        or int(identity.get("measurement_count", -1)) != measurement_count
    ):
        raise ValueError(f"camera {camera_id} identity, bundle, and mask rows are inconsistent")
    if identity.get("bundle_manifest_sha256") != bundle_hash or identity.get("mask_manifest_sha256") != mask_manifest_hash:
        raise ValueError(f"camera {camera_id} identity does not bind bundle and mask manifests")
    if masks.get("provenance", {}).get("bundle_manifest_sha256") != bundle_hash:
        raise ValueError(f"camera {camera_id} mask manifest does not bind its bundle")

    arrays = _verify_bundle_arrays(bundle_root, bundle, measurement_count, camera_id)
    mask_path_array = mask_root / str(contract["required_mask_filename"])
    active_raw = _load_array(mask_path_array)
    if active_raw.dtype.kind not in "iu":
        raise ValueError(f"camera {camera_id} active mask must have an integer dtype")
    active = np.asarray(active_raw, dtype=np.int64)
    mask_shards = masks.get("mask_shards")
    mask_entry = next((item for item in mask_shards or [] if isinstance(item, dict) and item.get("variable") == contract["required_mask_variable"]), None)
    if not isinstance(mask_entry, dict) or mask_entry.get("filename") != contract["required_mask_filename"]:
        raise ValueError(f"camera {camera_id} mask manifest does not identify its active mask")
    mask_hash = _sha256(mask_path_array)
    if mask_entry.get("sha256") != mask_hash or masks.get("provenance", {}).get("mask_sha256") != mask_hash:
        raise ValueError(f"camera {camera_id} active mask checksum differs from its manifest")
    if active.ndim != 1 or active.size == 0:
        raise ValueError(f"camera {camera_id} active mask is invalid")
    if active[0] < 0:
        raise ValueError(f"camera {camera_id} active mask contains a negative index")
    if active[-1] >= measurement_count:
        raise ValueError(f"camera {camera_id} active mask indexes exceed bundle rows")
    if active.size > 1 and np.any(active[1:] <= active[:-1]):
        raise ValueError(f"camera {camera_id} active mask is not strictly increasing")
    rays_per_camera = int(config["bounded_execution"]["rays_per_camera"])
    selected = deterministic_quantile_indices(active, rays_per_camera)
    if selected.size != rays_per_camera or np.any(selected < 0) or np.any(selected >= measurement_count):
        raise ValueError(f"camera {camera_id} selected ray index is negative or out of range")

    author_manifest_path = camera_root / str(contract["required_author_manifest_filename"])
    author_manifest = _load_json(author_manifest_path)
    author_manifest_hash = _sha256(author_manifest_path)
    author_path = camera_root / str(contract["required_author_forward_filename"])
    author_all_raw = _load_array(author_path)
    author_all = _finite_array(author_all_raw, name=f"camera {camera_id} author exact forward", shape=(measurement_count, 2))
    author_hash = _sha256(author_path)
    author_source_filename = author_manifest.get("author_source_filename")
    if author_source_filename != contract["required_author_source_filename"]:
        raise ValueError(f"camera {camera_id} author source filename differs from the contract")
    author_source_path = _safe_path(root, author_source_filename, "author source filename")
    if not author_source_path.is_file():
        raise FileNotFoundError(f"required private input is absent: {author_source_path.name}")
    author_source_hash = _sha256(author_source_path)
    if author_manifest.get("status") != contract["author_manifest_status"]:
        raise ValueError(f"camera {camera_id} author forward is not approved")
    expected_author = {
        "author_array_sha256": author_hash,
        "field_sha256": field_hash,
        "geometry_sha256": geometry_hash,
        "camera_identity_sha256": identity_hash,
        "bundle_manifest_sha256": bundle_hash,
        "mask_manifest_sha256": mask_manifest_hash,
        "mask_sha256": mask_hash,
        "author_source_sha256": author_source_hash,
        "generator_source_fingerprint": generator_fingerprint,
        "generator_version": generator_version,
    }
    for key, expected in expected_author.items():
        if author_manifest.get(key) != expected:
            raise ValueError(f"camera {camera_id} author provenance mismatch: {key}")
    if int(author_manifest.get("measurement_count", -1)) != measurement_count:
        raise ValueError(f"camera {camera_id} author forward row count is inconsistent")
    if author_manifest.get("row_order") != contract["row_order"]:
        raise ValueError(f"camera {camera_id} author forward row order is unsupported")

    return {
        "camera_id": camera_id,
        "selected": selected,
        "arrays": arrays,
        "author_exact": author_all[selected],
        "measurement": np.column_stack((arrays["epsu_all"][selected, 0], arrays["epsv_all"][selected, 0])),
        "provenance": {
            "camera_identity_sha256": identity_hash,
            "bundle_manifest_sha256": bundle_hash,
            "mask_manifest_sha256": mask_manifest_hash,
            "mask_sha256": mask_hash,
            "author_manifest_sha256": author_manifest_hash,
            "author_forward_sha256": author_hash,
            "author_source_sha256": author_source_hash,
            "geometry_sha256": geometry_hash,
            "field_sha256": field_hash,
            "generator_source_fingerprint": generator_fingerprint,
            "generator_version": generator_version,
        },
    }


def _fixed_domain_forward(
    *,
    camera: dict[str, Any],
    field: np.ndarray,
    sample_count: int,
    bounds: dict[str, list[float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = camera["selected"]
    arrays = camera["arrays"]
    origins = np.asarray(arrays["c"][selected], dtype=np.float64)
    directions = np.asarray(arrays["v"][selected], dtype=np.float64)
    lower = np.asarray(bounds["minimum"], dtype=np.float64)
    upper = np.asarray(bounds["maximum"], dtype=np.float64)
    box = intersect_forward_ray_box(origins, directions, lower, upper, layout="rows")
    design = deterministic_paired_uniform_aperture_samples(sample_count)
    start = origins + box["enter"][:, None] * box["direction_unit"]
    stop = origins + box["exit"][:, None] * box["direction_unit"]
    original_enter = box["enter"] / box["direction_norm"]
    original_exit = box["exit"] / box["direction_norm"]
    aperture = np.asarray(arrays["Rapvec"][selected, 0], dtype=np.float64)
    focal_distance = np.asarray(arrays["Dfvec"][selected, 0], dtype=np.float64)
    if np.any(focal_distance == 0.0):
        raise ValueError(f"camera {camera['camera_id']} has zero focal distance")
    sample_points = generate_aperture_sample_points(
        start,
        stop,
        np.asarray(arrays["Rxvecs"][selected], dtype=np.float64),
        np.asarray(arrays["Ryvecs"][selected], dtype=np.float64),
        aperture * (1.0 - original_enter / focal_distance),
        aperture * (1.0 - original_exit / focal_distance),
        design["longitudinal_fractions"],
        design["unit_disk_offsets"],
    )
    stencil = build_trilinear_stencil(
        sample_points,
        grid_shape=(32, 32, 32),
        grid_minimum_xyz=lower,
        grid_maximum_xyz=upper,
        dtype=torch.float64,
    )
    operator = PSUB0VoxelGradientOperator(
        stencil=stencil,
        projection_u_xyz=np.asarray(arrays["Ruvecs"][selected], dtype=np.float64),
        projection_v_xyz=np.asarray(arrays["Rvvecs"][selected], dtype=np.float64),
        line_length=np.asarray(box["length"], dtype=np.float64),
        system_constant=np.asarray(arrays["Csys_all"][selected, 0], dtype=np.float64),
        grid_minimum_xyz=lower,
        grid_maximum_xyz=upper,
        dtype=torch.float64,
    )
    volume = torch.from_numpy(np.array(field, dtype=np.float64, copy=True, order="C"))[None, None]
    prediction = operator(volume).detach().cpu().numpy()[0]
    return prediction, stencil.valid.detach().cpu().numpy(), np.asarray(box["hit"], dtype=bool)


def _pooled(records: list[dict[str, Any]], sample_count: int) -> dict[str, Any]:
    keys = ("measurement", "author_exact_residual", "fixed_domain_residual", "forward_mismatch")
    sums = {f"{key}_squared_l2": float(sum(row["metrics"][f"{key}_squared_l2"] for row in records)) for key in keys}
    component_count = int(sum(row["metrics"]["component_count"] for row in records))
    result: dict[str, Any] = {"qmc_sample_count": sample_count, "camera_count": len(records), "component_count": component_count, **sums}
    denominator = max(sums["measurement_squared_l2"], 1e-30)
    for key in keys:
        squared = sums[f"{key}_squared_l2"]
        result[f"{key}_l2"] = float(np.sqrt(squared))
    for key in keys[1:]:
        squared = sums[f"{key}_squared_l2"]
        result[f"{key}_rms"] = float(np.sqrt(squared / component_count))
        result[f"{key}_relative_l2"] = float(np.sqrt(squared / denominator))
    domain_keys = (
        "total_sample_count", "valid_sample_count", "empty_ray_count", "partial_ray_count", "full_ray_count",
        "centerline_ray_count", "centerline_hit_count", "centerline_miss_count",
    )
    domain = {key: int(sum(row["domain"][key] for row in records)) for key in domain_keys}
    domain["valid_sample_fraction"] = float(domain["valid_sample_count"] / domain["total_sample_count"])
    domain["centerline_hit_fraction"] = float(domain["centerline_hit_count"] / domain["centerline_ray_count"])
    result["domain"] = domain
    return result


def run_diagnostic(*, config_path: Path, private_bundle_root: Path, private_output: Path) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    if private_output.resolve().is_relative_to(repo_root):
        raise ValueError("private output must be outside the repository")
    config = _load_json(config_path)
    _validate_preregistration(config)
    source_hashes, generator_fingerprint, generator_version = _source_fingerprint(repo_root, config)
    geometry = _load_geometry_provenance(private_bundle_root, config, generator_fingerprint, generator_version)
    field, field_provenance = _load_frozen_field(private_bundle_root, config, geometry["source_sha256"], generator_fingerprint, generator_version)
    cameras = [
        _load_camera(
            root=private_bundle_root,
            camera_id=int(camera_id),
            config=config,
            field_hash=field_provenance["field_sha256"],
            geometry_hash=geometry["source_sha256"],
            generator_fingerprint=generator_fingerprint,
            generator_version=generator_version,
        )
        for camera_id in config["development_partition"]["camera_ids"]
    ]
    records: list[dict[str, Any]] = []
    pooled: list[dict[str, Any]] = []
    for sample_count in config["bounded_execution"]["qmc_sample_counts"]:
        qmc_records = []
        for camera in cameras:
            fixed_domain, valid, centerline_hit = _fixed_domain_forward(camera=camera, field=field, sample_count=int(sample_count), bounds=config["frozen_field"]["bounds_m"])
            record = {
                "camera_id": int(camera["camera_id"]),
                "qmc_sample_count": int(sample_count),
                "metrics": _metric_block(camera["measurement"], camera["author_exact"], fixed_domain),
                "domain": _domain_block(valid, centerline_hit),
                "provenance": camera["provenance"],
            }
            records.append(record)
            qmc_records.append(record)
        pooled.append(_pooled(qmc_records, int(sample_count)))
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": REPORT_STATUS,
        "evidence_scope": "DEVELOPMENT_ONLY_ROTATION_40_AUTHOR_ARRAY_VERSUS_B0_FIXED_DOMAIN_FINITE_APERTURE_FORWARD_ON_FROZEN_32CUBED_FIELD",
        "configuration": {
            "preregistration_sha256": _sha256(config_path),
            "rotation_degrees": 40,
            "camera_ids": [2, 3, 4],
            "rays_per_camera": int(config["bounded_execution"]["rays_per_camera"]),
            "qmc_sample_counts": [8, 16, 32],
            "qmc_interpretation": config["bounded_execution"]["qmc_interpretation"],
            "field_grid_shape_zyx": [32, 32, 32],
            "b0_sample_policy": config["bounded_execution"]["fixed_denominator"],
        },
        "provenance": {
            "generator_version": generator_version,
            "generator_source_sha256": source_hashes,
            "generator_source_fingerprint": generator_fingerprint,
            "field_sha256": field_provenance["field_sha256"],
            "field_manifest_sha256": field_provenance["manifest_sha256"],
            "geometry_sha256": geometry["source_sha256"],
            "geometry_manifest_sha256": geometry["manifest_sha256"],
            "field_frozen_before_rotation40_access": True,
            "geometry_frozen_before_rotation40_access": True,
            "camera_bindings": {str(camera["camera_id"]): camera["provenance"] for camera in cameras},
        },
        "domain_and_ood_accounting": {
            "rotation_40_is_disjoint_from_support_rotations": True,
            "support_rotation_degrees": [0, 50, 90],
            "camera_ids_are_shared_with_support": True,
            "camera_held_out": False,
            "held_out_unit": "rotation_run_not_camera",
            "author_domain_membership_not_inferred": True,
            "b0_domain_indicator_accounted_per_qmc_sample": True,
            "centerline_misses_accounted": True,
        },
        "per_camera": records,
        "pooled": pooled,
        "claim_boundary": config["claim_boundary"],
        "private_data_policy": {
            "contains_private_paths": False,
            "contains_source_paths": False,
            "contains_field_values": False,
            "contains_measurement_values": False,
            "contains_prediction_values": False,
            "contains_ray_indices": False,
        },
    }
    private_output.parent.mkdir(parents=True, exist_ok=True)
    private_output.write_text(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--private-bundle-root", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    args = parser.parse_args()
    report = run_diagnostic(config_path=args.config, private_bundle_root=args.private_bundle_root, private_output=args.private_output)
    print(json.dumps({"status": report["status"], "qmc_interpretation": report["configuration"]["qmc_interpretation"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
