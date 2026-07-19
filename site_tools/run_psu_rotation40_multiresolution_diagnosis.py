#!/usr/bin/env python3
"""Separate forward-grid effects from frozen reconstruction-field differences.

This is a post-open mechanism diagnostic on the already opened PSU rotation-40
development block. It never reads final rotations and cannot authorize a field,
generalization, algorithm-superiority, or paper claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional

from demo_t16_operator.psu_b0_streaming_operator import (
    PSUB0StreamingOperator,
    zero_outer_boundary_support,
)
from site_tools.psu_rotation40_active_store import PSURotation40ActiveRayStore
from site_tools.run_psu_rotation40_b0_reprojection import metric_row
from site_tools.run_psu_rotation40_resolution_transfer import (
    validate_config as validate_source_config,
    validate_public_report,
    verify_private_inputs,
)


CONFIG_SCHEMA = "psu-rotation40-multiresolution-diagnosis-development-1.0"
CONFIG_STATUS = "FROZEN_POSTOPEN_DIAGNOSTIC_BEFORE_DERIVED_FORWARD_EVALUATION"
REPORT_SCHEMA = "psu-rotation40-multiresolution-diagnosis-public-report-1.0"


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


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git(*args: str, binary: bool = False, check: bool = True) -> str | bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=check,
        capture_output=True,
    )
    if binary:
        return result.stdout
    return result.stdout.decode("utf-8").strip()


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("multiresolution diagnosis schema changed")
    if config.get("status") != CONFIG_STATUS:
        raise ValueError("diagnosis was not frozen before derived forwards")
    if config.get("scope") != (
        "ALREADY_OPENED_ROTATION40_DEVELOPMENT_MECHANISM_DIAGNOSIS_"
        "NO_FINAL_ROTATION_NO_FIELD_TRUTH"
    ):
        raise ValueError("diagnostic evidence scope changed")
    expected_source = {
        "config_path": "demo_t16_operator/configs/psu_rotation40_resolution_transfer_preregistered_v1.json",
        "config_sha256": "9b23fc69951b2ad5c5b7f736738d1fea5937b7da152be4beb0f22b0e45ab2889",
        "result_path": "demo_t16_operator/results/psu_rotation40_resolution_transfer_public_v1/summary.json",
        "result_sha256": "34f67c29a3dade7a25b609622472ddd14735f2d8e24fc7ce6aeed3d272eab439",
        "result_commit": "21c6e4a5f57fbed73d216f6d7328c41639554d29",
    }
    if config.get("source_resolution_protocol") != expected_source:
        raise ValueError("source resolution evidence binding changed")
    expected_dataset = {
        "doi": "10.26208/1VE2-5C19",
        "rotation_degrees": 40,
        "camera_ids": [2, 3, 4],
        "held_out_unit": "ROTATION_RUN_NOT_CAMERA",
        "independent_rotation_block_count": 1,
        "camera_rows_are_not_independent_repeats": True,
        "selection_mode": "all_rotation40_active_rows",
    }
    if config.get("dataset") != expected_dataset:
        raise ValueError("diagnostic dataset contract changed")
    expected_protocol_files = [
        "demo_t16_operator/configs/psu_rotation40_multiresolution_diagnosis_development_v1.json",
        "docs/psu_rotation40_multiresolution_diagnosis_protocol_2026-07-19.md",
        "site_tools/run_psu_rotation40_multiresolution_diagnosis.py",
        "site_tools/test_run_psu_rotation40_multiresolution_diagnosis.py",
        "site_tools/run_psu_rotation40_resolution_transfer.py",
        "site_tools/run_psu_rotation40_b0_reprojection.py",
        "site_tools/psu_rotation40_active_store.py",
        "site_tools/psu_b0_real_support_store.py",
        "site_tools/psu_bost_aperture_domain.py",
        "site_tools/psu_bost_forward_geometry.py",
        "demo_t16_operator/psu_b0_streaming_operator.py",
        "demo_t16_operator/psu_b0_reconstruction_interface.py",
    ]
    if config.get("protocol_files") != expected_protocol_files:
        raise ValueError("protocol or transitive dependency file set changed")
    expected_transforms = {
        "volume_layout": "BATCH_CHANNEL_Z_Y_X",
        "resize_mode": "trilinear",
        "align_corners": True,
        "low_shape_zyx": [16, 16, 16],
        "high_shape_zyx": [32, 32, 32],
        "apply_zero_outer_boundary_after_every_resize": True,
        "fixed_line_alphas": [0.0, 0.25, 0.5, 0.75, 1.0],
        "line_definition": "U_X16_PLUS_ALPHA_TIMES_X32_MINUS_U_X16",
        "direct_midpoint_linearity_check_alpha": 0.5,
        "no_alpha_is_selected_as_an_algorithm": True,
        "no_measurement_fitted_candidate_is_exported": True,
    }
    if config.get("transforms") != expected_transforms:
        raise ValueError("resize or fixed-line diagnostic changed")
    if config.get("comparisons") != [
        "A16_X16",
        "A32_U_X16",
        "A16_D_X32",
        "A32_U_D_X32",
        "A32_X32",
        "A32_LINE_U_X16_TO_X32",
    ]:
        raise ValueError("diagnostic comparison set changed")
    expected_gates = {
        "native16_vs_prolonged16_absolute_relative_l2_gap_max": 0.01,
        "native32_harm_vs_prolonged16_absolute_relative_l2_min": 0.01,
        "require_all_camera_fine_correction_cosines_negative": True,
        "linearity_max_abs_tolerance": 1e-10,
        "thresholds_are_mechanism_screens_not_physical_significance_bounds": True,
    }
    if config.get("diagnostic_gates") != expected_gates:
        raise ValueError("post-open diagnostic gates changed")
    expected_forward = {
        "grid_minimum_xyz_m": [-0.11, -0.11, -0.11],
        "grid_maximum_xyz_m": [0.11, 0.11, 0.11],
        "gauge": "zero_one_grid_node_outer_boundary",
        "finite_aperture_sample_count": 16,
        "chunk_rays": 32768,
        "dtype": "float64",
        "device": "cpu",
        "torch_threads": 8,
    }
    if config.get("forward") != expected_forward:
        raise ValueError("forward contract changed")
    if config.get("formal_output") != (
        "demo_t16_operator/results/psu_rotation40_multiresolution_diagnosis_public_v1"
    ):
        raise ValueError("formal output changed")
    expected_firewall = {
        "postopen_development_only": True,
        "algorithm_superiority": False,
        "experimental_field_truth_available": False,
        "field_relative_l2_available": False,
        "final_rotations_opened": False,
        "cross_session_generalization": False,
        "independent_camera_replication": False,
        "practical_significance_established": False,
        "causal_mechanism_proved": False,
        "neural_operator_trained": False,
        "publish_predictions_or_measurement_arrays": False,
        "paper_claim_authorized": False,
    }
    if config.get("claim_firewall") != expected_firewall:
        raise ValueError("claim firewall changed")


def verify_protocol_commit(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    protocol_commit: str,
    output_dir: Path,
) -> dict[str, str]:
    if not re.fullmatch(r"[0-9a-f]{40}", protocol_commit):
        raise ValueError("protocol commit must be a full Git SHA")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol_commit, "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if ancestor.returncode != 0:
        raise ValueError("protocol commit is not an ancestor of HEAD")
    source_commit = str(config["source_resolution_protocol"]["result_commit"])
    source_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, protocol_commit],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if source_ancestor.returncode != 0:
        raise ValueError("source result commit is not an ancestor of the protocol")
    digests: dict[str, str] = {}
    for relative in config["protocol_files"]:
        current = (ROOT / str(relative)).resolve()
        if ROOT not in current.parents or not current.is_file():
            raise ValueError(f"protocol file missing or escapes repository: {relative}")
        current_digest = _sha256(current)
        frozen = _git("show", f"{protocol_commit}:{relative}", binary=True)
        if hashlib.sha256(frozen).hexdigest() != current_digest:
            raise ValueError(f"protocol file changed after freeze: {relative}")
        digests[str(relative)] = current_digest
    relative_config = config_path.resolve().relative_to(ROOT).as_posix()
    if digests.get(relative_config) != _sha256(config_path):
        raise ValueError("config is not included in the protocol file set")
    dirty = _git("status", "--porcelain", "--", *config["protocol_files"])
    if dirty:
        raise ValueError("protocol files contain uncommitted changes")
    output_relative = output_dir.resolve().relative_to(ROOT).as_posix()
    tracked_at_protocol = subprocess.run(
        ["git", "cat-file", "-e", f"{protocol_commit}:{output_relative}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if tracked_at_protocol.returncode == 0:
        raise ValueError("formal output already existed at the protocol commit")
    if output_dir.exists():
        raise FileExistsError(f"formal output already exists: {output_dir}")
    return digests


def resize_volume(volume: torch.Tensor, shape_zyx: tuple[int, int, int]) -> torch.Tensor:
    if volume.ndim != 5 or volume.shape[:2] != (1, 1):
        raise ValueError("volume must have shape [1,1,z,y,x]")
    resized = functional.interpolate(
        volume,
        size=shape_zyx,
        mode="trilinear",
        align_corners=True,
    )
    support = zero_outer_boundary_support(shape_zyx, dtype=resized.dtype)
    return resized * support[None, None]


def correction_alignment(
    measured: np.ndarray,
    baseline: np.ndarray,
    corrected: np.ndarray,
) -> dict[str, float]:
    for name, value in (
        ("measured", measured),
        ("baseline", baseline),
        ("corrected", corrected),
    ):
        if value.ndim != 2 or value.shape[1] != 2 or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be a finite [ray,2] array")
    if measured.shape != baseline.shape or measured.shape != corrected.shape:
        raise ValueError("alignment arrays must share shape")
    residual = measured - baseline
    correction = corrected - baseline
    residual_norm = float(np.linalg.norm(residual))
    correction_norm = float(np.linalg.norm(correction))
    dot = float(np.vdot(residual.reshape(-1), correction.reshape(-1)).real)
    denominator = residual_norm * correction_norm
    cosine = dot / denominator if denominator > 0.0 else float("nan")
    correction_squared = correction_norm * correction_norm
    alpha_star = dot / correction_squared if correction_squared > 0.0 else float("nan")
    return {
        "residual_l2": residual_norm,
        "correction_l2": correction_norm,
        "residual_correction_dot": dot,
        "residual_correction_cosine": float(cosine),
        "unconstrained_least_squares_alpha_diagnostic_only": float(alpha_star),
        "clipped_zero_one_alpha_diagnostic_only": float(
            np.clip(alpha_star, 0.0, 1.0)
        ),
    }


def mechanism_decision(
    *,
    native16_relative_l2: float,
    prolonged16_relative_l2: float,
    native32_relative_l2: float,
    camera_correction_cosines: list[float],
    grid_gap_max: float,
    harm_min: float,
) -> dict[str, Any]:
    grid_gap = abs(float(native16_relative_l2) - float(prolonged16_relative_l2))
    high_harm = float(native32_relative_l2) - float(prolonged16_relative_l2)
    all_negative = len(camera_correction_cosines) == 3 and all(
        float(value) < 0.0 for value in camera_correction_cosines
    )
    if grid_gap <= grid_gap_max and high_harm >= harm_min and all_negative:
        status = "OPENED_BLOCK_FIELD_CORRECTION_ANTI_ALIGNED_GRID_FORWARD_GAP_SMALL"
    elif grid_gap > grid_gap_max:
        status = "OPENED_BLOCK_FORWARD_GRID_CHANGE_MATERIAL_MECHANISM_UNRESOLVED"
    elif high_harm >= harm_min:
        status = "OPENED_BLOCK_FIELD_DIFFERENCE_HARM_WITH_MIXED_CAMERA_ALIGNMENT"
    else:
        status = "OPENED_BLOCK_NO_CLEAR_MULTRESOLUTION_MECHANISM_SEPARATION"
    return {
        "machine_diagnosis": status,
        "native16_vs_prolonged16_absolute_relative_l2_gap": grid_gap,
        "native32_harm_vs_prolonged16_absolute_relative_l2": high_harm,
        "all_three_camera_fine_correction_cosines_negative": all_negative,
        "grid_gap_screen_max": float(grid_gap_max),
        "native32_harm_screen_min": float(harm_min),
        "mechanism_screen_only": True,
        "causal_mechanism_proved": False,
    }


def _candidate_metric(
    candidate_id: str,
    construction: str,
    measured: np.ndarray,
    predicted: np.ndarray,
    store: PSURotation40ActiveRayStore,
    *,
    direct_forward_calls: int,
) -> dict[str, Any]:
    aggregate = metric_row(measured, predicted)
    per_camera = []
    start = 0
    for camera in store.cameras:
        stop = start + camera.ray_count
        row = metric_row(measured[start:stop], predicted[start:stop])
        row["camera_id"] = int(camera.camera_id)
        per_camera.append(row)
        start = stop
    return {
        "candidate_id": candidate_id,
        "construction": construction,
        "aggregate": aggregate,
        "per_camera": per_camera,
        "direct_forward_calls": int(direct_forward_calls),
        "postopen_candidate_selection": False,
    }


def _predict(
    operator: PSUB0StreamingOperator,
    volume: torch.Tensor,
) -> tuple[np.ndarray, dict[str, Any]]:
    operator.reset_call_counts()
    started = time.perf_counter()
    with torch.no_grad():
        predicted = operator(volume)
    elapsed = float(time.perf_counter() - started)
    values = predicted[0].cpu().numpy()
    calls = operator.call_report()
    if calls["forward_calls"] != 1 or calls["adjoint_calls"] != 0:
        raise RuntimeError("a diagnostic field must use exactly one direct forward")
    if not np.all(np.isfinite(values)):
        raise RuntimeError("diagnostic prediction is non-finite")
    hit_count = int(sum(row["b0_hit_count"] for row in calls["records"]))
    if hit_count != operator.ray_count:
        raise RuntimeError("not every selected active ray hit B0")
    return values, {
        "logical_forward_calls": 1,
        "logical_adjoint_calls": 0,
        "wall_seconds": elapsed,
        "all_selected_active_rays_hit_b0": True,
    }


def _field_norm(volume: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(volume).item())


def _gradient_norm(
    volume: torch.Tensor,
    *,
    grid_minimum_xyz: list[float],
    grid_maximum_xyz: list[float],
) -> float:
    nz, ny, nx = (int(value) for value in volume.shape[-3:])
    spacing_zyx = (
        (grid_maximum_xyz[2] - grid_minimum_xyz[2]) / (nz - 1),
        (grid_maximum_xyz[1] - grid_minimum_xyz[1]) / (ny - 1),
        (grid_maximum_xyz[0] - grid_minimum_xyz[0]) / (nx - 1),
    )
    squared = torch.zeros((), dtype=volume.dtype)
    for dimension, spacing in zip((2, 3, 4), spacing_zyx, strict=True):
        difference = torch.diff(volume, dim=dimension) / float(spacing)
        squared = squared + torch.sum(difference * difference)
    return float(torch.sqrt(squared).item())


def _write_csv(path: Path, candidates: list[dict[str, Any]]) -> None:
    fieldnames = [
        "candidate_id",
        "construction",
        "scope",
        "camera_id",
        "ray_count",
        "vector_relative_l2",
        "component_rmse_px",
        "component_mae_px",
        "residual_magnitude_p95_px",
        "measured_vector_rms_px",
        "predicted_vector_rms_px",
        "direct_forward_calls",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for candidate in candidates:
            shared = {
                "candidate_id": candidate["candidate_id"],
                "construction": candidate["construction"],
                "direct_forward_calls": candidate["direct_forward_calls"],
            }
            writer.writerow(
                {
                    **shared,
                    "scope": "pooled",
                    "camera_id": "",
                    **candidate["aggregate"],
                }
            )
            for camera in candidate["per_camera"]:
                row = dict(camera)
                camera_id = row.pop("camera_id")
                writer.writerow(
                    {**shared, "scope": "camera", "camera_id": camera_id, **row}
                )


def _plot(report: Mapping[str, Any], png: Path, pdf: Path) -> None:
    candidates = report["candidates"]
    direct_ids = [
        "native_16",
        "prolong_16_to_32",
        "restrict_32_to_16",
        "roundtrip_32_via_16",
        "native_32",
    ]
    direct = {row["candidate_id"]: row for row in candidates}
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), constrained_layout=True)
    colors = ["#2f6f8f", "#2f806d", "#c28a2c", "#b45863", "#4d4d4d"]
    direct_values = [direct[key]["aggregate"]["vector_relative_l2"] for key in direct_ids]
    axes[0, 0].bar(range(len(direct_ids)), direct_values, color=colors)
    axes[0, 0].set_xticks(
        range(len(direct_ids)),
        ["A16 x16", "A32 Ux16", "A16 Dx32", "A32 UDx32", "A32 x32"],
        rotation=18,
        ha="right",
    )
    axes[0, 0].set_ylabel("vector relative L2")
    axes[0, 0].set_title("Grid/field decomposition on rotation 40")
    axes[0, 0].grid(axis="y", alpha=0.2)

    line = sorted(
        (row for row in candidates if row["candidate_id"].startswith("line_alpha_")),
        key=lambda row: float(row["line_alpha"]),
    )
    alphas = [float(row["line_alpha"]) for row in line]
    axes[0, 1].plot(
        alphas,
        [row["aggregate"]["vector_relative_l2"] for row in line],
        "o-",
        color="#2f6f8f",
        label="pooled",
    )
    for camera_id, color in zip((2, 3, 4), ("#2f806d", "#c28a2c", "#b45863"), strict=True):
        axes[0, 1].plot(
            alphas,
            [
                next(
                    camera["vector_relative_l2"]
                    for camera in row["per_camera"]
                    if camera["camera_id"] == camera_id
                )
                for row in line
            ],
            "o--",
            color=color,
            label=f"camera {camera_id}",
        )
    axes[0, 1].set_xlabel("fixed alpha in Ux16 + alpha(x32-Ux16)")
    axes[0, 1].set_ylabel("vector relative L2")
    axes[0, 1].set_title("Diagnostic path; no alpha is selected")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.2)

    rows = report["correction_alignment"]["per_camera"]
    axes[1, 0].bar(
        [f"cam {row['camera_id']}" for row in rows],
        [row["residual_correction_cosine"] for row in rows],
        color=["#2f806d", "#c28a2c", "#b45863"],
    )
    axes[1, 0].axhline(0.0, color="#333333", linewidth=1)
    axes[1, 0].set_ylim(-1.0, 1.0)
    axes[1, 0].set_ylabel("cos(residual at Ux16, x32-Ux16 prediction)")
    axes[1, 0].set_title("Does the frozen fine-field correction point toward data?")
    axes[1, 0].grid(axis="y", alpha=0.2)

    axes[1, 1].axis("off")
    decision = report["mechanism_decision"]
    pooled = report["correction_alignment"]["pooled"]
    lines = [
        "POST-OPEN MECHANISM DIAGNOSIS",
        "",
        f"Decision: {decision['machine_diagnosis']}",
        "A16 x16 vs A32 Ux16 gap: "
        f"{decision['native16_vs_prolonged16_absolute_relative_l2_gap']:.6f}",
        "A32 x32 harm vs A32 Ux16: "
        f"{decision['native32_harm_vs_prolonged16_absolute_relative_l2']:+.6f}",
        f"pooled correction cosine: {pooled['residual_correction_cosine']:+.6f}",
        "diagnostic alpha*: "
        f"{pooled['unconstrained_least_squares_alpha_diagnostic_only']:+.6f}",
        "",
        "One opened rotation block; same cameras.",
        "No field truth, final rotation, causal proof,",
        "algorithm selection, or paper claim.",
    ]
    axes[1, 1].text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
    )
    fig.suptitle("What caused the 32-cubed rotation-transfer reversal?", fontsize=14)
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)


def _write_checksums(output_dir: Path) -> None:
    paths = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "checksums.sha256"
    )
    (output_dir / "checksums.sha256").write_text(
        "\n".join(f"{_sha256(path)}  {path.name}" for path in paths) + "\n",
        encoding="ascii",
    )


def run_diagnosis(
    *,
    config_path: Path,
    protocol_commit: str,
    private_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    config = _read_json(config_path)
    validate_config(config)
    expected_output = (ROOT / str(config["formal_output"])).resolve()
    if output_dir != expected_output:
        raise ValueError("output directory differs from the frozen config")
    protocol_digests = verify_protocol_commit(
        config=config,
        config_path=config_path,
        protocol_commit=protocol_commit,
        output_dir=output_dir,
    )
    source = config["source_resolution_protocol"]
    source_config_path = (ROOT / str(source["config_path"])).resolve()
    source_result_path = (ROOT / str(source["result_path"])).resolve()
    if _sha256(source_config_path) != source["config_sha256"]:
        raise ValueError("source resolution config checksum changed")
    if _sha256(source_result_path) != source["result_sha256"]:
        raise ValueError("source resolution result checksum changed")
    source_config = _read_json(source_config_path)
    validate_source_config(source_config)
    source_result = _read_json(source_result_path)
    if source_result.get("status") != (
        "SUPPORT_RESOLUTION_GAIN_DID_NOT_CLEAR_NUMERICAL_TRANSFER_GATE_NO_GO"
    ):
        raise ValueError("source resolution result is not the frozen NO-GO")

    private_root = private_root.resolve()
    geometry_root = private_root / "rotation40_geometry_binding_v1"
    payload_root = private_root / "rotation40_development_h2_v1"
    volume_paths = {
        "support_cgls4_16cubed": private_root
        / "b0_streaming_baseline_v1/reconstruction_16cubed.npy",
        "support_cgls4_32cubed": private_root
        / "b0_cached_reference_v1/reconstruction_32cubed_cached.npy",
    }
    private_reports = {
        "support_cgls4_16cubed": private_root
        / "b0_streaming_baseline_v1/private_report.json",
        "support_cgls4_32cubed": private_root
        / "b0_cached_reference_v1/private_report.json",
    }
    public_reports = {
        "support_cgls4_16cubed": ROOT
        / "docs/psu_b0_streaming_baseline_public_summary.json",
        "support_cgls4_32cubed": ROOT
        / "docs/psu_b0_cached_reference_public_summary.json",
    }
    verify_private_inputs(
        config=source_config,
        geometry_root=geometry_root,
        geometry_private_report_path=geometry_root
        / "geometry_binding_private_report.json",
        geometry_public_summary_path=ROOT
        / "docs/psu_rotation40_geometry_binding_public_summary.json",
        payload_root=payload_root,
        payload_private_report_path=payload_root / "payload_private_report.json",
        payload_public_summary_path=ROOT
        / "docs/psu_rotation40_cell_payload_public_summary.json",
        resolution_public_summary_path=ROOT
        / "docs/psu_b0_streaming_resolution_public_summary.json",
        volume_paths=volume_paths,
        candidate_private_report_paths=private_reports,
        candidate_public_summary_paths=public_reports,
    )

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
    operator16 = PSUB0StreamingOperator(
        ray_store=store,
        grid_shape=(16, 16, 16),
        grid_minimum_xyz=tuple(forward["grid_minimum_xyz_m"]),
        grid_maximum_xyz=tuple(forward["grid_maximum_xyz_m"]),
        support=zero_outer_boundary_support((16, 16, 16), dtype=torch.float64),
        dtype=torch.float64,
    )
    operator32 = PSUB0StreamingOperator(
        ray_store=store,
        grid_shape=(32, 32, 32),
        grid_minimum_xyz=tuple(forward["grid_minimum_xyz_m"]),
        grid_maximum_xyz=tuple(forward["grid_maximum_xyz_m"]),
        support=zero_outer_boundary_support((32, 32, 32), dtype=torch.float64),
        dtype=torch.float64,
    )
    measured = operator16.load_observations()[0].cpu().numpy()
    x16 = torch.from_numpy(np.load(volume_paths["support_cgls4_16cubed"], allow_pickle=False))
    x32 = torch.from_numpy(np.load(volume_paths["support_cgls4_32cubed"], allow_pickle=False))
    if x16.shape != (1, 1, 16, 16, 16) or x32.shape != (1, 1, 32, 32, 32):
        raise ValueError("frozen volume shape changed")
    x16 = x16.to(dtype=torch.float64)
    x32 = x32.to(dtype=torch.float64)
    ux16 = resize_volume(x16, (32, 32, 32))
    dx32 = resize_volume(x32, (16, 16, 16))
    udx32 = resize_volume(dx32, (32, 32, 32))

    fields = {
        "native_16": (operator16, x16, "A16_X16"),
        "prolong_16_to_32": (operator32, ux16, "A32_U_X16"),
        "restrict_32_to_16": (operator16, dx32, "A16_D_X32"),
        "roundtrip_32_via_16": (operator32, udx32, "A32_U_D_X32"),
        "native_32": (operator32, x32, "A32_X32"),
    }
    predictions: dict[str, np.ndarray] = {}
    runtimes: dict[str, Any] = {}
    candidates: list[dict[str, Any]] = []
    for candidate_id, (operator, field, construction) in fields.items():
        predicted, runtime = _predict(operator, field)
        predictions[candidate_id] = predicted
        runtimes[candidate_id] = runtime
        candidates.append(
            _candidate_metric(
                candidate_id,
                construction,
                measured,
                predicted,
                store,
                direct_forward_calls=1,
            )
        )

    correction = predictions["native_32"] - predictions["prolong_16_to_32"]
    line_candidates = []
    for alpha in config["transforms"]["fixed_line_alphas"]:
        predicted = predictions["prolong_16_to_32"] + float(alpha) * correction
        row = _candidate_metric(
            f"line_alpha_{float(alpha):.2f}",
            "A32_LINE_U_X16_TO_X32",
            measured,
            predicted,
            store,
            direct_forward_calls=0,
        )
        row["line_alpha"] = float(alpha)
        line_candidates.append(row)
    candidates.extend(line_candidates)

    midpoint = ux16 + 0.5 * (x32 - ux16)
    midpoint_direct, midpoint_runtime = _predict(operator32, midpoint)
    midpoint_derived = predictions["prolong_16_to_32"] + 0.5 * correction
    linearity_max_abs = float(np.max(np.abs(midpoint_direct - midpoint_derived)))
    tolerance = float(config["diagnostic_gates"]["linearity_max_abs_tolerance"])
    if linearity_max_abs > tolerance:
        raise RuntimeError("direct midpoint did not match the linear derived prediction")

    pooled_alignment = correction_alignment(
        measured,
        predictions["prolong_16_to_32"],
        predictions["native_32"],
    )
    camera_alignment = []
    start = 0
    for camera in store.cameras:
        stop = start + camera.ray_count
        row = correction_alignment(
            measured[start:stop],
            predictions["prolong_16_to_32"][start:stop],
            predictions["native_32"][start:stop],
        )
        row["camera_id"] = int(camera.camera_id)
        camera_alignment.append(row)
        start = stop

    candidate_by_id = {row["candidate_id"]: row for row in candidates}
    decision = mechanism_decision(
        native16_relative_l2=candidate_by_id["native_16"]["aggregate"]["vector_relative_l2"],
        prolonged16_relative_l2=candidate_by_id["prolong_16_to_32"]["aggregate"]["vector_relative_l2"],
        native32_relative_l2=candidate_by_id["native_32"]["aggregate"]["vector_relative_l2"],
        camera_correction_cosines=[
            float(row["residual_correction_cosine"]) for row in camera_alignment
        ],
        grid_gap_max=float(
            config["diagnostic_gates"][
                "native16_vs_prolonged16_absolute_relative_l2_gap_max"
            ]
        ),
        harm_min=float(
            config["diagnostic_gates"][
                "native32_harm_vs_prolonged16_absolute_relative_l2_min"
            ]
        ),
    )

    native_prediction_gap = predictions["prolong_16_to_32"] - predictions["native_16"]
    measured_norm = float(np.linalg.norm(measured))
    native16_prediction_norm = float(np.linalg.norm(predictions["native_16"]))
    x32_minus_ux16 = x32 - ux16
    x32_minus_udx32 = x32 - udx32
    field_diagnostics = {
        "x16_l2": _field_norm(x16),
        "ux16_l2": _field_norm(ux16),
        "x32_l2": _field_norm(x32),
        "udx32_l2": _field_norm(udx32),
        "x32_minus_ux16_l2": _field_norm(x32_minus_ux16),
        "x32_minus_ux16_over_x32_l2": _field_norm(x32_minus_ux16)
        / max(_field_norm(x32), np.finfo(np.float64).tiny),
        "x32_minus_udx32_l2": _field_norm(x32_minus_udx32),
        "x32_minus_udx32_over_x32_l2": _field_norm(x32_minus_udx32)
        / max(_field_norm(x32), np.finfo(np.float64).tiny),
        "ux16_gradient_l2_physical_spacing": _gradient_norm(
            ux16,
            grid_minimum_xyz=forward["grid_minimum_xyz_m"],
            grid_maximum_xyz=forward["grid_maximum_xyz_m"],
        ),
        "x32_gradient_l2_physical_spacing": _gradient_norm(
            x32,
            grid_minimum_xyz=forward["grid_minimum_xyz_m"],
            grid_maximum_xyz=forward["grid_maximum_xyz_m"],
        ),
        "native16_vs_prolonged16_prediction_difference_over_measured_l2": float(
            np.linalg.norm(native_prediction_gap)
            / max(measured_norm, np.finfo(np.float64).tiny)
        ),
        "native16_vs_prolonged16_prediction_difference_over_native16_prediction_l2": float(
            np.linalg.norm(native_prediction_gap)
            / max(native16_prediction_norm, np.finfo(np.float64).tiny)
        ),
    }

    report = {
        "schema_version": REPORT_SCHEMA,
        "status": decision["machine_diagnosis"],
        "evidence_scope": config["scope"],
        "protocol_commit": protocol_commit,
        "config_sha256": _sha256(config_path),
        "protocol_file_sha256": protocol_digests,
        "source_resolution_protocol": dict(config["source_resolution_protocol"]),
        "dataset": dict(config["dataset"]),
        "selection": store.selection_summary(),
        "forward": dict(forward),
        "transforms": dict(config["transforms"]),
        "candidates": candidates,
        "direct_forward_runtime": {**runtimes, "midpoint_linearity_check": midpoint_runtime},
        "linearity_check": {
            "alpha": 0.5,
            "max_abs_direct_minus_derived": linearity_max_abs,
            "tolerance": tolerance,
            "passed": True,
        },
        "correction_alignment": {
            "definition": "A32_X32_MINUS_A32_U_X16_AGAINST_Y_MINUS_A32_U_X16",
            "pooled": pooled_alignment,
            "per_camera": camera_alignment,
            "least_squares_alpha_is_diagnostic_only_not_a_candidate": True,
        },
        "field_diagnostics": field_diagnostics,
        "mechanism_decision": decision,
        "claim_boundary": dict(config["claim_firewall"]),
        "public_export_policy": {
            "contains_predictions": False,
            "contains_measurements": False,
            "contains_geometry_arrays": False,
            "contains_volumes": False,
            "contains_only_aggregate_metrics": True,
        },
    }
    private_tokens = (str(private_root), str(geometry_root), str(payload_root))
    validate_public_report(report, forbidden_path_tokens=private_tokens)
    output_dir.mkdir(parents=True)
    _write_json(output_dir / "summary.json", report)
    _write_csv(output_dir / "comparison_rows.csv", candidates)
    _plot(report, output_dir / "diagnostic.png", output_dir / "diagnostic.pdf")
    (output_dir / "README.md").write_text(
        "# PSU rotation-40 multiresolution mechanism diagnosis\n\n"
        f"Machine diagnosis: `{decision['machine_diagnosis']}`.\n\n"
        "This is a post-open aggregate-only diagnostic on one already opened rotation "
        "block. It contains no measurements, predictions, geometry arrays, private "
        "volumes, final rotations, volumetric truth, selected alpha, causal proof, "
        "algorithm-superiority claim, or paper claim.\n",
        encoding="utf-8",
    )
    _write_checksums(output_dir)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol-commit", required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run_diagnosis(
        config_path=args.config,
        protocol_commit=args.protocol_commit,
        private_root=args.private_root,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "mechanism_decision": report["mechanism_decision"],
                "correction_alignment": report["correction_alignment"],
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
