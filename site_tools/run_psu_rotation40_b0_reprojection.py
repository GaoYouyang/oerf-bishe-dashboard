#!/usr/bin/env python3
"""Score the frozen support B0 on all rotation-40 active development rays."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from demo_t16_operator.psu_b0_streaming_operator import (
    PSUB0StreamingOperator,
    zero_outer_boundary_support,
)
from site_tools.psu_rotation40_active_store import PSURotation40ActiveRayStore


CONFIG_SCHEMA = "psu-rotation40-b0-reprojection-config-1.0"
CONFIG_STATUS = "FROZEN_AFTER_NON_TUNING_3K_INTERFACE_SMOKE_BEFORE_ALL_ACTIVE_SCORE"
PUBLIC_SCHEMA = "psu-rotation40-b0-reprojection-public-1.0"
PUBLIC_STATUS = "ROTATION40_FROZEN_SUPPORT_B0_REAL_REPROJECTION_SCORED_NO_FIELD_TRUTH"


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


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA or config.get("status") != CONFIG_STATUS:
        raise ValueError("rotation-40 B0 reprojection config is absent or not frozen")
    dataset = config.get("dataset")
    if not isinstance(dataset, Mapping) or dataset.get("camera_ids") != [2, 3, 4]:
        raise ValueError("B0 reprojection must remain cameras 2/3/4")
    if dataset.get("selection_mode") != "all_rotation40_active_rows":
        raise ValueError("full active-row scoring cannot be changed after the smoke")
    forward = config.get("forward")
    expected = {
        "grid_shape_zyx": [32, 32, 32],
        "grid_minimum_xyz_m": [-0.11, -0.11, -0.11],
        "grid_maximum_xyz_m": [0.11, 0.11, 0.11],
        "gauge": "zero_one_voxel_outer_boundary",
        "finite_aperture_sample_count": 16,
        "chunk_rays": 32768,
        "dtype": "float64",
        "device": "cpu",
        "torch_threads": 8,
    }
    if not isinstance(forward, Mapping) or any(forward.get(key) != value for key, value in expected.items()):
        raise ValueError("frozen B0 forward configuration changed")
    firewall = config.get("claim_firewall")
    if not isinstance(firewall, Mapping) or any(value is not False for value in firewall.values()):
        raise ValueError("B0 claim firewall is incomplete")


def metric_row(measured: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    measured = np.asarray(measured, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if measured.shape != predicted.shape or measured.ndim != 2 or measured.shape[1] != 2:
        raise ValueError("measured and predicted arrays must match with shape (N,2)")
    if measured.shape[0] < 1 or not np.all(np.isfinite(measured)) or not np.all(np.isfinite(predicted)):
        raise ValueError("metric arrays must be nonempty and finite")
    residual = predicted - measured
    measured_norm = float(np.linalg.norm(measured))
    if measured_norm <= 0.0:
        raise ValueError("measured vector norm must be positive")
    residual_magnitude = np.linalg.norm(residual, axis=1)
    return {
        "ray_count": int(measured.shape[0]),
        "vector_relative_l2": float(np.linalg.norm(residual) / measured_norm),
        "component_rmse_px": float(np.sqrt(np.mean(np.square(residual)))),
        "component_mae_px": float(np.mean(np.abs(residual))),
        "residual_magnitude_p95_px": float(np.quantile(residual_magnitude, 0.95)),
        "measured_vector_rms_px": float(np.sqrt(np.mean(np.sum(np.square(measured), axis=1)))),
        "predicted_vector_rms_px": float(np.sqrt(np.mean(np.sum(np.square(predicted), axis=1)))),
    }


def run_b0_reprojection(
    *,
    config_path: Path,
    geometry_root: Path,
    geometry_private_report_path: Path,
    geometry_public_summary_path: Path,
    volume_path: Path,
    public_summary_path: Path | None = None,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = _read_json(config_path)
    validate_config(config)
    if _sha256(geometry_private_report_path) != config["geometry_binding"]["private_report_sha256"]:
        raise ValueError("geometry private report checksum differs from the frozen contract")
    if _sha256(geometry_public_summary_path) != config["geometry_binding"]["public_summary_sha256"]:
        raise ValueError("geometry public summary checksum differs from the frozen contract")
    if _sha256(volume_path) != config["frozen_support_volume"]["sha256"]:
        raise ValueError("frozen support volume checksum differs from the contract")
    volume_numpy = np.load(volume_path, allow_pickle=False)
    if list(volume_numpy.shape) != config["frozen_support_volume"]["shape"] or str(volume_numpy.dtype) != "float64":
        raise ValueError("frozen support volume shape or dtype changed")

    forward = config["forward"]
    torch.set_num_threads(int(forward["torch_threads"]))
    store = PSURotation40ActiveRayStore(
        geometry_root,
        rays_per_camera=None,
        sample_count=int(forward["finite_aperture_sample_count"]),
        chunk_rays=int(forward["chunk_rays"]),
        grid_minimum_xyz=tuple(forward["grid_minimum_xyz_m"]),
        grid_maximum_xyz=tuple(forward["grid_maximum_xyz_m"]),
    )
    dtype = torch.float64
    support = zero_outer_boundary_support(
        tuple(forward["grid_shape_zyx"]), dtype=dtype
    )
    operator = PSUB0StreamingOperator(
        ray_store=store,
        grid_shape=tuple(forward["grid_shape_zyx"]),
        grid_minimum_xyz=tuple(forward["grid_minimum_xyz_m"]),
        grid_maximum_xyz=tuple(forward["grid_maximum_xyz_m"]),
        support=support,
        dtype=dtype,
    )
    measured_tensor = operator.load_observations()
    volume = torch.from_numpy(volume_numpy).to(dtype=dtype)
    with torch.no_grad():
        predicted_tensor = operator(volume)
    measured = measured_tensor[0].cpu().numpy()
    predicted = predicted_tensor[0].cpu().numpy()
    aggregate = metric_row(measured, predicted)
    per_camera = []
    start = 0
    for camera in store.cameras:
        stop = start + camera.ray_count
        row = metric_row(measured[start:stop], predicted[start:stop])
        row["camera_id"] = int(camera.camera_id)
        per_camera.append(row)
        start = stop
    call_report = operator.call_report()
    hit_count = int(sum(record["b0_hit_count"] for record in call_report["records"]))
    report = {
        "schema_version": PUBLIC_SCHEMA,
        "status": PUBLIC_STATUS,
        "evidence_scope": "FROZEN_SUPPORT_FIELD_TO_REAL_ROTATION40_DEVELOPMENT_REPROJECTION_NO_FIELD_TRUTH_NO_CANDIDATE_COMPARISON",
        "config_sha256": _sha256(config_path),
        "dataset": {
            "doi": config["dataset"]["doi"],
            "rotation_degrees": 40,
            "camera_ids": [2, 3, 4],
        },
        "selection": store.selection_summary(),
        "forward": dict(forward),
        "aggregate": aggregate,
        "per_camera": per_camera,
        "runtime": call_report,
        "gates": {
            "all_selected_active_rays_hit_b0": hit_count == store.ray_count,
            "prediction_finite": bool(np.all(np.isfinite(predicted))),
            "single_logical_forward_call": call_report["forward_calls"] == 1,
            "final_rotations_unopened": True,
        },
        "claim_boundary": {
            "development_rotation40_scored": True,
            "frozen_b0_baseline_only": True,
            "candidate_compared": False,
            "algorithm_superiority": False,
            "experimental_field_truth_available": False,
            "field_relative_l2_available": False,
            "final_rotations_opened": False,
        },
        "public_export_policy": {
            "contains_predictions": False,
            "contains_measurements": False,
            "contains_geometry_arrays": False,
            "contains_only_aggregate_metrics": True,
        },
    }
    if not all(report["gates"].values()):
        raise RuntimeError("rotation-40 B0 reprojection gate failed")
    if public_summary_path is not None:
        _write_json_atomic(public_summary_path.resolve(), report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--geometry-private-report", type=Path, required=True)
    parser.add_argument("--geometry-public-summary", type=Path, required=True)
    parser.add_argument("--volume", type=Path, required=True)
    parser.add_argument("--public-summary", type=Path)
    args = parser.parse_args()
    report = run_b0_reprojection(
        config_path=args.config,
        geometry_root=args.geometry_root,
        geometry_private_report_path=args.geometry_private_report,
        geometry_public_summary_path=args.geometry_public_summary,
        volume_path=args.volume,
        public_summary_path=args.public_summary,
    )
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
