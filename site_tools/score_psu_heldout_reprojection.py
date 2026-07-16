#!/usr/bin/env python3
"""Score one frozen candidate on the PSU primary held-out rotation audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from site_tools.validate_psu_heldout_camera_protocol import (
    EXPECTED_PRIMARY_AUDIT_ROTATIONS,
    EXPECTED_SUPPORT_CAMERAS,
    validate_protocol,
)


BUNDLE_SCHEMA = "psu-heldout-reprojection-bundle-1.0"
BUNDLE_STATUS = "FROZEN_BEFORE_SINGLE_FINAL_AUDIT"
REPORT_SCHEMA = "psu-heldout-reprojection-score-1.0"
REQUIRED_ARRAYS = (
    "measured_uv_px",
    "candidate_uv_px",
    "baseline_uv_px",
    "active_mask",
    "ambient_mask",
    "front_band_mask",
)


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be an object")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON: {path}") from exc
    return dict(_mapping(value, str(path)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any, location: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{location} must be a lowercase SHA-256 digest")
    return text


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def _score_pixels(
    measured: np.ndarray,
    predicted: np.ndarray,
    mask: np.ndarray,
    *,
    location: str,
) -> dict[str, float | int]:
    selected_measured = measured[mask]
    selected_error = predicted[mask] - selected_measured
    error_norm = np.linalg.norm(selected_error, axis=1)
    signal_norm = float(np.linalg.norm(selected_measured))
    if signal_norm <= 0.0:
        raise ValueError(f"{location} measured signal norm must be positive")
    return {
        "pixel_count": int(mask.sum()),
        "vector_relative_l2": float(np.linalg.norm(selected_error) / signal_norm),
        "vector_rmse_px": float(np.sqrt(np.mean(np.square(error_norm)))),
        "vector_mae_px": float(np.mean(error_norm)),
        "vector_error_p95_px": float(np.quantile(error_norm, 0.95)),
    }


def _predicted_rms(predicted: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.square(predicted[mask]), axis=1))))


def _load_view(
    *,
    manifest_dir: Path,
    record: Mapping[str, Any],
    index: int,
) -> dict[str, Any]:
    camera_id = int(record.get("camera_id", -1))
    rotation = int(record.get("rotation_degrees", -1))
    relative = Path(str(record.get("npz", "")))
    if relative.is_absolute():
        raise ValueError(f"views[{index}].npz must be relative")
    path = (manifest_dir / relative).resolve()
    try:
        path.relative_to(manifest_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"views[{index}].npz escapes the bundle directory") from exc
    if not path.is_file():
        raise ValueError(f"views[{index}].npz is missing")
    if _sha256(path) != _digest(record.get("sha256"), f"views[{index}].sha256"):
        raise ValueError(f"views[{index}] SHA-256 mismatch")

    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = [name for name in REQUIRED_ARRAYS if name not in archive.files]
            if missing:
                raise ValueError(f"views[{index}] missing arrays: {missing}")
            arrays = {name: np.asarray(archive[name]) for name in REQUIRED_ARRAYS}
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot load safe NPZ for views[{index}]") from exc

    measured = np.asarray(arrays["measured_uv_px"], dtype=np.float64)
    candidate = np.asarray(arrays["candidate_uv_px"], dtype=np.float64)
    baseline = np.asarray(arrays["baseline_uv_px"], dtype=np.float64)
    active = np.asarray(arrays["active_mask"])
    ambient = np.asarray(arrays["ambient_mask"])
    front_band = np.asarray(arrays["front_band_mask"])
    if measured.ndim != 2 or measured.shape[1] != 2:
        raise ValueError(f"views[{index}] measured_uv_px must have shape (N, 2)")
    if candidate.shape != measured.shape or baseline.shape != measured.shape:
        raise ValueError(f"views[{index}] predictions must match measured shape")
    if (
        active.shape != (measured.shape[0],)
        or ambient.shape != active.shape
        or front_band.shape != active.shape
    ):
        raise ValueError(f"views[{index}] masks must have shape (N,)")
    if (
        active.dtype != np.bool_
        or ambient.dtype != np.bool_
        or front_band.dtype != np.bool_
    ):
        raise ValueError(f"views[{index}] masks must be boolean")
    if not active.any() or not ambient.any() or not front_band.any():
        raise ValueError(
            f"views[{index}] active, ambient, and front-band masks must be nonempty"
        )
    if np.any(active & ambient):
        raise ValueError(f"views[{index}] active and ambient masks must be disjoint")
    if np.any(front_band & ~active):
        raise ValueError(f"views[{index}] front-band mask must be a subset of active")
    if not all(np.isfinite(value).all() for value in (measured, candidate, baseline)):
        raise ValueError(f"views[{index}] UV arrays must be finite")
    return {
        "camera_id": camera_id,
        "rotation_degrees": rotation,
        "measured": measured,
        "candidate": candidate,
        "baseline": baseline,
        "active": active,
        "ambient": ambient,
        "front_band": front_band,
    }


def _paired_sign_pvalue(wins: int, blocks: int) -> float:
    return sum(math.comb(blocks, index) for index in range(wins, blocks + 1)) / (
        2**blocks
    )


def score_bundle(protocol_path: Path, manifest_path: Path) -> dict[str, Any]:
    protocol = _read_json(protocol_path)
    protocol_summary = validate_protocol(protocol)
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != BUNDLE_SCHEMA:
        raise ValueError("unsupported held-out reprojection bundle schema")
    if manifest.get("registration_status") != BUNDLE_STATUS:
        raise ValueError("bundle was not frozen before the single final audit")
    if manifest.get("split_id") != "audit_rotation_same_cameras":
        raise ValueError("this scorer only accepts the primary held-out rotation split")
    if _digest(manifest.get("protocol_sha256"), "protocol_sha256") != _sha256(
        protocol_path
    ):
        raise ValueError("bundle is not bound to the supplied frozen protocol")

    methods = {}
    for method_name in ("candidate", "baseline"):
        method = _mapping(manifest.get(method_name), method_name)
        methods[method_name] = {
            "id": str(method.get("id", "")),
            "checkpoint_sha256": _digest(
                method.get("checkpoint_sha256"),
                f"{method_name}.checkpoint_sha256",
            ),
            "config_sha256": _digest(
                method.get("config_sha256"),
                f"{method_name}.config_sha256",
            ),
        }
        if not methods[method_name]["id"]:
            raise ValueError(f"{method_name}.id must be nonempty")

    raw_views = manifest.get("views")
    if not isinstance(raw_views, list):
        raise ValueError("views must be an array")
    views = [
        _load_view(
            manifest_dir=manifest_path.parent,
            record=_mapping(value, f"views[{index}]"),
            index=index,
        )
        for index, value in enumerate(raw_views)
    ]
    expected_pairs = {
        (rotation, camera)
        for rotation in EXPECTED_PRIMARY_AUDIT_ROTATIONS
        for camera in EXPECTED_SUPPORT_CAMERAS
    }
    actual_pairs = {
        (view["rotation_degrees"], view["camera_id"]) for view in views
    }
    if len(actual_pairs) != len(views):
        raise ValueError("held-out view pairs must be unique")
    if actual_pairs != expected_pairs:
        raise ValueError("bundle must contain exactly the frozen 18 primary audit views")

    per_view = []
    blocks = []
    for rotation in EXPECTED_PRIMARY_AUDIT_ROTATIONS:
        block_views = [
            view for view in views if view["rotation_degrees"] == rotation
        ]
        measured_active = np.concatenate(
            [view["measured"][view["active"]] for view in block_views]
        )
        candidate_active = np.concatenate(
            [view["candidate"][view["active"]] for view in block_views]
        )
        baseline_active = np.concatenate(
            [view["baseline"][view["active"]] for view in block_views]
        )
        block_mask = np.ones(measured_active.shape[0], dtype=bool)
        candidate_metrics = _score_pixels(
            measured_active,
            candidate_active,
            block_mask,
            location=f"rotation {rotation} candidate",
        )
        baseline_metrics = _score_pixels(
            measured_active,
            baseline_active,
            block_mask,
            location=f"rotation {rotation} baseline",
        )
        blocks.append(
            {
                "rotation_degrees": rotation,
                "camera_ids": list(EXPECTED_SUPPORT_CAMERAS),
                "candidate": candidate_metrics,
                "baseline": baseline_metrics,
                "candidate_lower_primary_endpoint": (
                    candidate_metrics["vector_relative_l2"]
                    < baseline_metrics["vector_relative_l2"]
                ),
                "active_rmse_reduction_px": (
                    baseline_metrics["vector_rmse_px"]
                    - candidate_metrics["vector_rmse_px"]
                ),
            }
        )

    for view in sorted(
        views, key=lambda value: (value["rotation_degrees"], value["camera_id"])
    ):
        per_view.append(
            {
                "rotation_degrees": view["rotation_degrees"],
                "camera_id": view["camera_id"],
                "active": {
                    "candidate": _score_pixels(
                        view["measured"],
                        view["candidate"],
                        view["active"],
                        location="per-view candidate",
                    ),
                    "baseline": _score_pixels(
                        view["measured"],
                        view["baseline"],
                        view["active"],
                        location="per-view baseline",
                    ),
                },
                "ambient_predicted_deflection_rms_px": {
                    "candidate": _predicted_rms(
                        view["candidate"], view["ambient"]
                    ),
                    "baseline": _predicted_rms(
                        view["baseline"], view["ambient"]
                    ),
                },
                "front_band": {
                    "candidate": _score_pixels(
                        view["measured"],
                        view["candidate"],
                        view["front_band"],
                        location="per-view front-band candidate",
                    ),
                    "baseline": _score_pixels(
                        view["measured"],
                        view["baseline"],
                        view["front_band"],
                        location="per-view front-band baseline",
                    ),
                },
            }
        )

    wins = sum(block["candidate_lower_primary_endpoint"] for block in blocks)
    sign_pvalue = _paired_sign_pvalue(wins, len(blocks))
    reductions = [block["active_rmse_reduction_px"] for block in blocks]
    repeatability_raw = manifest.get("development_repeatability_floor_px")
    repeatability_floor = (
        None if repeatability_raw is None else float(repeatability_raw)
    )
    if repeatability_floor is not None and (
        not math.isfinite(repeatability_floor) or repeatability_floor < 0.0
    ):
        raise ValueError("development_repeatability_floor_px must be non-negative")
    practical_gate = (
        None
        if repeatability_floor is None
        else float(np.median(reductions)) > repeatability_floor
    )
    tail_gate = (
        None
        if repeatability_floor is None
        else all(
            block["candidate"]["vector_error_p95_px"]
            <= block["baseline"]["vector_error_p95_px"] + repeatability_floor
            for block in blocks
        )
    )
    candidate_ambient_values = np.concatenate(
        [view["candidate"][view["ambient"]] for view in views]
    )
    baseline_ambient_values = np.concatenate(
        [view["baseline"][view["ambient"]] for view in views]
    )
    candidate_ambient = float(
        np.sqrt(np.mean(np.sum(np.square(candidate_ambient_values), axis=1)))
    )
    baseline_ambient = float(
        np.sqrt(np.mean(np.sum(np.square(baseline_ambient_values), axis=1)))
    )
    ambient_gate = candidate_ambient <= baseline_ambient
    calibration_gate = manifest.get("calibration_perturbation_gate_passed")
    if calibration_gate not in (True, False, None):
        raise ValueError("calibration_perturbation_gate_passed must be boolean or null")

    primary_gate = wins == len(blocks) and math.isclose(
        sign_pvalue, 1 / 64, rel_tol=0.0, abs_tol=1e-15
    )
    gate_values = (
        primary_gate,
        practical_gate is True,
        tail_gate is True,
        ambient_gate,
        calibration_gate is True,
    )
    passed = all(gate_values)
    lock_reasons = []
    if not primary_gate:
        lock_reasons.append("PRIMARY_SIX_BLOCK_SIGN_GATE_FAILED")
    if repeatability_floor is None:
        lock_reasons.append("FLOW_OFF_REPEATABILITY_FLOOR_MISSING")
    elif not practical_gate:
        lock_reasons.append("PRACTICAL_SIGNIFICANCE_GATE_FAILED")
    if tail_gate is False:
        lock_reasons.append("TAIL_GATE_FAILED")
    if not ambient_gate:
        lock_reasons.append("AMBIENT_LEAKAGE_GATE_FAILED")
    if calibration_gate is not True:
        lock_reasons.append("CALIBRATION_PERTURBATION_GATE_NOT_PASSED")

    return {
        "schema_version": REPORT_SCHEMA,
        "status": (
            "PRIMARY_HELDOUT_GATE_PASS_IMAGE_SPACE_ONLY"
            if passed
            else "PRIMARY_HELDOUT_GATE_NOT_PASSED_OR_CLAIM_LOCKED"
        ),
        "protocol": {
            "schema_version": protocol_summary["schema_version"],
            "view_count": 18,
            "independent_experimental_unit": "MODEL_ROTATION_RUN_BLOCK",
            "pixel_independence_assumption": False,
        },
        "methods": methods,
        "per_rotation_block": blocks,
        "per_view": per_view,
        "aggregate": {
            "candidate_lower_block_count": wins,
            "block_count": len(blocks),
            "one_sided_exact_sign_pvalue": sign_pvalue,
            "median_active_rmse_reduction_px": float(np.median(reductions)),
            "development_repeatability_floor_px": repeatability_floor,
            "ambient_predicted_deflection_rms_px": {
                "candidate": candidate_ambient,
                "baseline": baseline_ambient,
            },
        },
        "gates": {
            "primary_all_six_blocks_lower": primary_gate,
            "practical_reduction_exceeds_repeatability": practical_gate,
            "tail_no_worse_than_repeatability_margin": tail_gate,
            "ambient_no_increase": ambient_gate,
            "calibration_perturbation_passed": calibration_gate,
            "all_image_space_gates_pass": passed,
        },
        "claim": {
            "authorized": (
                "HELD_OUT_IMAGE_SPACE_CONSISTENCY_ONLY"
                if passed
                else "NO_SUPERIORITY_CLAIM"
            ),
            "lock_reasons": lock_reasons,
            "field_l2_available": False,
            "unique_3d_truth_established": False,
            "pixel_bootstrap_permitted": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("bundle_manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = score_bundle(args.protocol, args.bundle_manifest)
    _atomic_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
