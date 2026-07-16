#!/usr/bin/env python3
"""Audit predeclared B1 cone-parameter sensitivity on PSU BOST ray shards.

The audit compares one-nappe ``box intersect cone`` supports against a frozen
released-parameter reference. It reports aggregate and per-view effect sizes
only. It does not select cone parameters or run a reconstruction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from .psu_bost_forward_geometry import (
        CONTRACT_VERSION as GEOMETRY_CONTRACT_VERSION,
        intersect_forward_ray_box,
        intersect_forward_ray_box_cone,
    )
    from .run_psu_fixed_domain_geometry_audit import (
        _atomic_json,
        _open_view_contract,
        _sha256_file,
        _safe_fraction,
    )
except ImportError:  # Direct script execution.
    from psu_bost_forward_geometry import (  # type: ignore[no-redef]
        CONTRACT_VERSION as GEOMETRY_CONTRACT_VERSION,
        intersect_forward_ray_box,
        intersect_forward_ray_box_cone,
    )
    from run_psu_fixed_domain_geometry_audit import (  # type: ignore[no-redef]
        _atomic_json,
        _open_view_contract,
        _sha256_file,
        _safe_fraction,
    )


CONFIG_SCHEMA = "psu-b1-parameter-sensitivity-preregistration-1.0"
REPORT_SCHEMA = "psu-b1-parameter-sensitivity-view-audit-1.0"
AGGREGATE_SCHEMA = "psu-b1-parameter-sensitivity-all-view-audit-1.0"
FROZEN_STATUS = "FROZEN_BEFORE_REAL_SCORING"
SCOPE_NAMES = ("all", "active", "inactive")
SUM_FIELDS = (
    "ray_count",
    "b0_hit_count",
    "baseline_hit_count",
    "candidate_hit_count",
    "shared_hit_count",
    "union_hit_count",
    "gained_hit_count",
    "lost_hit_count",
    "hit_disagreement_count",
    "changed_interval_count",
    "shared_interval_changed_count",
    "shared_disjoint_interval_count",
    "candidate_removed_from_b0_count",
    "candidate_hit_without_b0_count",
    "candidate_length_exceeds_b0_count",
    "nonfinite_output_count",
    "b0_length_sum_m",
    "baseline_length_sum_m",
    "candidate_length_sum_m",
    "interval_overlap_length_sum_m",
    "interval_union_length_sum_m",
    "path_length_l1_delta_sum_m",
    "path_length_l2_delta_sum_m2",
    "shared_endpoint_l1_delta_sum_m",
    "shared_midpoint_abs_delta_sum_m",
)
MAX_FIELDS = ("maximum_abs_endpoint_delta_m",)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _vector3(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite length-three vector")
    return array


def load_frozen_config(path: Path) -> dict[str, Any]:
    config = _load_json(path)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unexpected sensitivity config schema")
    if config.get("registration_status") != FROZEN_STATUS:
        raise ValueError("sensitivity config must be frozen before scoring")
    if not str(config.get("selection_policy", "")).startswith(
        "NO_PARAMETER_SELECTION"
    ):
        raise ValueError("config must lock data-derived parameter selection")

    box = config.get("outer_box")
    if not isinstance(box, dict):
        raise ValueError("outer_box must be an object")
    lower = _vector3(box.get("minimum_m"), name="outer_box.minimum_m")
    upper = _vector3(box.get("maximum_m"), name="outer_box.maximum_m")
    if np.any(upper <= lower):
        raise ValueError("outer box maximum must exceed minimum")

    variants = config.get("variants")
    if not isinstance(variants, list) or len(variants) < 2:
        raise ValueError("at least two sensitivity variants are required")
    ids: list[str] = []
    for index, variant in enumerate(variants):
        if not isinstance(variant, dict):
            raise ValueError(f"variant {index} must be an object")
        variant_id = str(variant.get("id", ""))
        if not variant_id:
            raise ValueError(f"variant {index} has no id")
        ids.append(variant_id)
        _vector3(variant.get("cone_vertex_m"), name=f"{variant_id}.cone_vertex_m")
        axis = _vector3(variant.get("cone_axis"), name=f"{variant_id}.cone_axis")
        if float(np.linalg.norm(axis)) <= 0.0:
            raise ValueError(f"{variant_id}.cone_axis must be nonzero")
        angle = float(variant.get("cone_angle_degrees", math.nan))
        if not math.isfinite(angle) or not 0.0 < angle < 90.0:
            raise ValueError(f"{variant_id}.cone_angle_degrees must be in (0, 90)")
    if len(set(ids)) != len(ids):
        raise ValueError("variant ids must be unique")
    baseline_id = str(config.get("baseline_variant_id", ""))
    if baseline_id not in ids:
        raise ValueError("baseline_variant_id must name one frozen variant")
    return config


def _normalized_variant(variant: Mapping[str, Any]) -> dict[str, Any]:
    axis = _vector3(variant["cone_axis"], name=f"{variant['id']}.cone_axis")
    axis /= np.linalg.norm(axis)
    return {
        "id": str(variant["id"]),
        "family": str(variant.get("family", "unspecified")),
        "role": str(variant.get("role", "")),
        "cone_vertex_m": _vector3(
            variant["cone_vertex_m"], name=f"{variant['id']}.cone_vertex_m"
        ),
        "cone_axis_normalized": axis,
        "cone_angle_degrees": float(variant["cone_angle_degrees"]),
        "cone_angle_radians": math.radians(float(variant["cone_angle_degrees"])),
    }


def _empty_scope() -> dict[str, float | int]:
    return {**{name: 0 for name in SUM_FIELDS}, **{name: 0.0 for name in MAX_FIELDS}}


def _take(value: np.ndarray, local: np.ndarray | None) -> np.ndarray:
    return value if local is None else value[local]


def _update_scope(
    accumulator: dict[str, float | int],
    *,
    local: np.ndarray | None,
    b0: Mapping[str, np.ndarray],
    baseline: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    tolerance: float,
) -> None:
    b0_hit = _take(np.asarray(b0["hit"], dtype=bool), local)
    b0_length = _take(np.asarray(b0["length"], dtype=np.float64), local)
    baseline_hit = _take(np.asarray(baseline["hit"], dtype=bool), local)
    baseline_length = _take(
        np.asarray(baseline["length"], dtype=np.float64), local
    )
    baseline_enter = _take(np.asarray(baseline["enter"], dtype=np.float64), local)
    baseline_exit = _take(np.asarray(baseline["exit"], dtype=np.float64), local)
    candidate_hit = _take(np.asarray(candidate["hit"], dtype=bool), local)
    candidate_length = _take(
        np.asarray(candidate["length"], dtype=np.float64), local
    )
    candidate_enter = _take(np.asarray(candidate["enter"], dtype=np.float64), local)
    candidate_exit = _take(np.asarray(candidate["exit"], dtype=np.float64), local)

    shared = baseline_hit & candidate_hit
    union = baseline_hit | candidate_hit
    gained = candidate_hit & ~baseline_hit
    lost = baseline_hit & ~candidate_hit
    overlap = np.zeros_like(candidate_length)
    overlap[shared] = np.maximum(
        0.0,
        np.minimum(baseline_exit[shared], candidate_exit[shared])
        - np.maximum(baseline_enter[shared], candidate_enter[shared]),
    )
    union_length = baseline_length + candidate_length - overlap
    length_delta = candidate_length - baseline_length
    endpoint_enter_delta = np.zeros_like(candidate_length)
    endpoint_exit_delta = np.zeros_like(candidate_length)
    endpoint_enter_delta[shared] = np.abs(
        candidate_enter[shared] - baseline_enter[shared]
    )
    endpoint_exit_delta[shared] = np.abs(
        candidate_exit[shared] - baseline_exit[shared]
    )
    endpoint_l1 = endpoint_enter_delta + endpoint_exit_delta
    midpoint_delta = np.zeros_like(candidate_length)
    midpoint_delta[shared] = 0.5 * np.abs(
        candidate_enter[shared]
        + candidate_exit[shared]
        - baseline_enter[shared]
        - baseline_exit[shared]
    )
    shared_changed = shared & (
        (endpoint_enter_delta > tolerance) | (endpoint_exit_delta > tolerance)
    )
    interval_changed = gained | lost | shared_changed
    nonfinite = (
        ~np.isfinite(candidate_length)
        | (candidate_hit & ~np.isfinite(candidate_enter))
        | (candidate_hit & ~np.isfinite(candidate_exit))
    )

    accumulator["ray_count"] += int(candidate_hit.size)
    accumulator["b0_hit_count"] += int(np.count_nonzero(b0_hit))
    accumulator["baseline_hit_count"] += int(np.count_nonzero(baseline_hit))
    accumulator["candidate_hit_count"] += int(np.count_nonzero(candidate_hit))
    accumulator["shared_hit_count"] += int(np.count_nonzero(shared))
    accumulator["union_hit_count"] += int(np.count_nonzero(union))
    accumulator["gained_hit_count"] += int(np.count_nonzero(gained))
    accumulator["lost_hit_count"] += int(np.count_nonzero(lost))
    accumulator["hit_disagreement_count"] += int(np.count_nonzero(gained | lost))
    accumulator["changed_interval_count"] += int(np.count_nonzero(interval_changed))
    accumulator["shared_interval_changed_count"] += int(
        np.count_nonzero(shared_changed)
    )
    accumulator["shared_disjoint_interval_count"] += int(
        np.count_nonzero(shared & (overlap <= tolerance))
    )
    accumulator["candidate_removed_from_b0_count"] += int(
        np.count_nonzero(b0_hit & ~candidate_hit)
    )
    accumulator["candidate_hit_without_b0_count"] += int(
        np.count_nonzero(candidate_hit & ~b0_hit)
    )
    accumulator["candidate_length_exceeds_b0_count"] += int(
        np.count_nonzero(candidate_length > b0_length + tolerance)
    )
    accumulator["nonfinite_output_count"] += int(np.count_nonzero(nonfinite))
    accumulator["b0_length_sum_m"] += float(b0_length.sum())
    accumulator["baseline_length_sum_m"] += float(baseline_length.sum())
    accumulator["candidate_length_sum_m"] += float(candidate_length.sum())
    accumulator["interval_overlap_length_sum_m"] += float(overlap.sum())
    accumulator["interval_union_length_sum_m"] += float(union_length.sum())
    accumulator["path_length_l1_delta_sum_m"] += float(
        np.abs(length_delta).sum()
    )
    accumulator["path_length_l2_delta_sum_m2"] += float(
        np.square(length_delta).sum()
    )
    accumulator["shared_endpoint_l1_delta_sum_m"] += float(endpoint_l1.sum())
    accumulator["shared_midpoint_abs_delta_sum_m"] += float(midpoint_delta.sum())
    if candidate_hit.size:
        accumulator["maximum_abs_endpoint_delta_m"] = max(
            float(accumulator["maximum_abs_endpoint_delta_m"]),
            float(max(endpoint_enter_delta.max(), endpoint_exit_delta.max())),
        )


def _finalize_scope(raw: Mapping[str, float | int]) -> dict[str, Any]:
    ray_count = int(raw["ray_count"])
    shared = int(raw["shared_hit_count"])
    baseline_length = float(raw["baseline_length_sum_m"])
    b0_length = float(raw["b0_length_sum_m"])
    candidate_length = float(raw["candidate_length_sum_m"])
    return {
        **raw,
        "b0_hit_fraction": _safe_fraction(raw["b0_hit_count"], ray_count),
        "baseline_hit_fraction": _safe_fraction(
            raw["baseline_hit_count"], ray_count
        ),
        "candidate_hit_fraction": _safe_fraction(
            raw["candidate_hit_count"], ray_count
        ),
        "gained_hit_fraction": _safe_fraction(raw["gained_hit_count"], ray_count),
        "lost_hit_fraction": _safe_fraction(raw["lost_hit_count"], ray_count),
        "hit_disagreement_fraction": _safe_fraction(
            raw["hit_disagreement_count"], ray_count
        ),
        "changed_interval_fraction": _safe_fraction(
            raw["changed_interval_count"], ray_count
        ),
        "shared_interval_changed_fraction": _safe_fraction(
            raw["shared_interval_changed_count"], shared
        ),
        "hit_set_jaccard": _safe_fraction(
            raw["shared_hit_count"], raw["union_hit_count"]
        ),
        "ray_support_length_iou": _safe_fraction(
            raw["interval_overlap_length_sum_m"],
            raw["interval_union_length_sum_m"],
        ),
        "candidate_path_fraction_of_b0": _safe_fraction(
            candidate_length, b0_length
        ),
        "candidate_path_fraction_of_baseline": _safe_fraction(
            candidate_length, baseline_length
        ),
        "candidate_path_relative_signed_change_from_baseline": _safe_fraction(
            candidate_length - baseline_length, baseline_length
        ),
        "path_length_l1_fraction_of_baseline": _safe_fraction(
            raw["path_length_l1_delta_sum_m"], baseline_length
        ),
        "path_length_rmse_m": (
            math.sqrt(float(raw["path_length_l2_delta_sum_m2"]) / ray_count)
            if ray_count
            else None
        ),
        "shared_mean_endpoint_l1_delta_m": _safe_fraction(
            raw["shared_endpoint_l1_delta_sum_m"], shared
        ),
        "shared_mean_midpoint_abs_delta_m": _safe_fraction(
            raw["shared_midpoint_abs_delta_sum_m"], shared
        ),
    }


def _scope_local_indices(
    indices: np.ndarray, *, start: int, stop: int
) -> np.ndarray:
    left = int(np.searchsorted(indices, start, side="left"))
    right = int(np.searchsorted(indices, stop, side="left"))
    return np.asarray(indices[left:right] - start, dtype=np.int64)


def audit_parameter_sensitivity_view(
    *,
    view_dir: Path,
    config: Mapping[str, Any],
    config_path: Path,
    chunk_rows: int = 65_536,
) -> dict[str, Any]:
    if chunk_rows < 2:
        raise ValueError("chunk_rows must be at least two")
    lower = _vector3(
        config["outer_box"]["minimum_m"], name="outer_box.minimum_m"
    )
    upper = _vector3(
        config["outer_box"]["maximum_m"], name="outer_box.maximum_m"
    )
    variants = [_normalized_variant(value) for value in config["variants"]]
    baseline_id = str(config["baseline_variant_id"])
    baseline_variant = next(
        variant for variant in variants if variant["id"] == baseline_id
    )

    arrays, masks, metadata = _open_view_contract(view_dir)
    rows = int(metadata["rows"])
    tolerance = 2e-9
    accumulators = {
        variant["id"]: {scope: _empty_scope() for scope in SCOPE_NAMES}
        for variant in variants
    }
    started = time.perf_counter()

    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        origin = np.asarray(arrays["origin"][start:stop], dtype=np.float64)
        direction = np.asarray(arrays["direction"][start:stop], dtype=np.float64)
        b0 = intersect_forward_ray_box(
            origin, direction, lower, upper, layout="rows"
        )
        baseline = intersect_forward_ray_box_cone(
            origin,
            direction,
            lower,
            upper,
            baseline_variant["cone_vertex_m"],
            baseline_variant["cone_axis_normalized"],
            baseline_variant["cone_angle_radians"],
            layout="rows",
        )
        scope_indices = {
            "all": None,
            "active": _scope_local_indices(
                masks["amask_all"], start=start, stop=stop
            ),
            "inactive": _scope_local_indices(
                masks["imask_all"], start=start, stop=stop
            ),
        }
        for variant in variants:
            candidate = (
                baseline
                if variant["id"] == baseline_id
                else intersect_forward_ray_box_cone(
                    origin,
                    direction,
                    lower,
                    upper,
                    variant["cone_vertex_m"],
                    variant["cone_axis_normalized"],
                    variant["cone_angle_radians"],
                    layout="rows",
                )
            )
            for scope, local in scope_indices.items():
                _update_scope(
                    accumulators[variant["id"]][scope],
                    local=local,
                    b0=b0,
                    baseline=baseline,
                    candidate=candidate,
                    tolerance=tolerance,
                )

    expected_counts = {
        "all": rows,
        "active": int(masks["amask_all"].size),
        "inactive": int(masks["imask_all"].size),
    }
    variant_reports: list[dict[str, Any]] = []
    invalid_variants: list[str] = []
    for variant in variants:
        scopes = {
            scope: _finalize_scope(accumulators[variant["id"]][scope])
            for scope in SCOPE_NAMES
        }
        if any(
            int(scopes[scope]["ray_count"]) != expected_counts[scope]
            for scope in SCOPE_NAMES
        ):
            raise RuntimeError("scope accounting did not cover the expected rows")
        invariant_failures = sum(
            int(scopes["all"][name])
            for name in (
                "candidate_hit_without_b0_count",
                "candidate_length_exceeds_b0_count",
                "nonfinite_output_count",
            )
        )
        if invariant_failures:
            invalid_variants.append(variant["id"])
        variant_reports.append(
            {
                "id": variant["id"],
                "family": variant["family"],
                "role": variant["role"],
                "configuration": {
                    "cone_vertex_m": variant["cone_vertex_m"].tolist(),
                    "cone_axis_normalized": variant[
                        "cone_axis_normalized"
                    ].tolist(),
                    "cone_angle_degrees": variant["cone_angle_degrees"],
                },
                "mechanical_invariants_pass": invariant_failures == 0,
                "scopes": scopes,
            }
        )

    return {
        "schema_version": REPORT_SCHEMA,
        "status": (
            "B1_PARAMETER_SENSITIVITY_VIEW_INVALID"
            if invalid_variants
            else "B1_PARAMETER_SENSITIVITY_VIEW_MECHANICAL_PASS_SELECTION_LOCKED"
        ),
        "evidence_scope": "REAL_ONE_VIEW_PREDECLARED_B1_SUPPORT_SENSITIVITY_NO_RECONSTRUCTION_NO_PARAMETER_SELECTION",
        "view_id_zero_based": metadata["view_id_zero_based"],
        "source": {
            "frozen_config_filename": config_path.name,
            "frozen_config_sha256": _sha256_file(config_path),
            "geometry_implementation_sha256": _sha256_file(
                Path(__file__).with_name("psu_bost_forward_geometry.py")
            ),
            "audit_implementation_sha256": _sha256_file(Path(__file__)),
            "bundle_manifest_sha256": metadata["bundle_manifest_sha256"],
            "setup_manifest_sha256": metadata["setup_manifest_sha256"],
            "corrected_mask_manifest_sha256": metadata["mask_manifest_sha256"],
        },
        "configuration": {
            "rows": rows,
            "chunk_rows": chunk_rows,
            "baseline_variant_id": baseline_id,
            "outer_minimum_m": lower.tolist(),
            "outer_maximum_m": upper.tolist(),
            "variant_count": len(variants),
            "geometry_contract_version": GEOMETRY_CONTRACT_VERSION,
        },
        "invalid_variant_ids": invalid_variants,
        "variants": variant_reports,
        "decision": {
            "mechanical_invariants_pass": not invalid_variants,
            "physical_cone_parameter_selection_validated": False,
            "data_derived_parameter_selection_permitted": False,
            "training_ready": "NO",
            "algorithm_superiority_claim": "LOCKED",
        },
        "limitations": [
            "the audit measures computational-support dependence and does not identify a physical flow boundary",
            "the 5 mm shifts are coarse stress tests rather than measured calibration uncertainty",
            "the axis sign flip is a semantic falsifier rather than a local perturbation",
            "no finite-aperture rerendering, held-out camera, inverse reconstruction, or neural model is scored",
        ],
        "runtime_observation": {
            "wall_seconds": time.perf_counter() - started,
            "scope": "CACHED_LOCAL_DIAGNOSTIC_NOT_A_SPEED_BENCHMARK",
        },
    }


def _pool_scopes(scopes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    raw: dict[str, float | int] = {}
    for name in SUM_FIELDS:
        raw[name] = sum(scope[name] for scope in scopes)
    for name in MAX_FIELDS:
        raw[name] = max(float(scope[name]) for scope in scopes)
    return _finalize_scope(raw)


def _family_envelope(
    variants: Sequence[Mapping[str, Any]], *, family: str
) -> dict[str, Any] | None:
    selected = [variant for variant in variants if variant["family"] == family]
    if not selected:
        return None
    return {
        "variant_ids": [variant["id"] for variant in selected],
        "active_candidate_hit_fraction_range": [
            min(
                float(variant["scopes"]["active"]["candidate_hit_fraction"])
                for variant in selected
            ),
            max(
                float(variant["scopes"]["active"]["candidate_hit_fraction"])
                for variant in selected
            ),
        ],
        "active_changed_interval_fraction_range": [
            min(
                float(variant["scopes"]["active"]["changed_interval_fraction"])
                for variant in selected
            ),
            max(
                float(variant["scopes"]["active"]["changed_interval_fraction"])
                for variant in selected
            ),
        ],
        "all_candidate_path_fraction_of_b0_range": [
            min(
                float(variant["scopes"]["all"]["candidate_path_fraction_of_b0"])
                for variant in selected
            ),
            max(
                float(variant["scopes"]["all"]["candidate_path_fraction_of_b0"])
                for variant in selected
            ),
        ],
        "all_ray_support_length_iou_range": [
            min(
                float(variant["scopes"]["all"]["ray_support_length_iou"])
                for variant in selected
            ),
            max(
                float(variant["scopes"]["all"]["ray_support_length_iou"])
                for variant in selected
            ),
        ],
    }


def aggregate_parameter_sensitivity_views(
    records: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    config_path: Path,
) -> dict[str, Any]:
    if not records:
        raise ValueError("at least one view record is required")
    view_ids = [int(record["view_id_zero_based"]) for record in records]
    if view_ids != list(range(len(records))):
        raise ValueError("view ids must be the ordered contiguous range from zero")
    expected_ids = [str(variant["id"]) for variant in config["variants"]]
    for record in records:
        if [variant["id"] for variant in record["variants"]] != expected_ids:
            raise ValueError("variant order does not match the frozen config")

    pooled_variants: list[dict[str, Any]] = []
    invalid_variants: list[str] = []
    for index, variant_config in enumerate(config["variants"]):
        view_variants = [record["variants"][index] for record in records]
        scopes = {
            scope: _pool_scopes(
                [variant["scopes"][scope] for variant in view_variants]
            )
            for scope in SCOPE_NAMES
        }
        mechanical_pass = all(
            bool(variant["mechanical_invariants_pass"]) for variant in view_variants
        )
        if not mechanical_pass:
            invalid_variants.append(str(variant_config["id"]))
        pooled_variants.append(
            {
                "id": str(variant_config["id"]),
                "family": str(variant_config.get("family", "unspecified")),
                "role": str(variant_config.get("role", "")),
                "configuration": view_variants[0]["configuration"],
                "mechanical_invariants_pass": mechanical_pass,
                "scopes": scopes,
            }
        )

    baseline_id = str(config["baseline_variant_id"])
    nonreference = [
        variant for variant in pooled_variants if variant["id"] != baseline_id
    ]
    largest_active_change = max(
        nonreference,
        key=lambda value: float(
            value["scopes"]["active"]["changed_interval_fraction"]
        ),
    )
    lowest_support_iou = min(
        nonreference,
        key=lambda value: float(value["scopes"]["all"]["ray_support_length_iou"]),
    )
    axis_flip = next(
        (
            variant
            for variant in pooled_variants
            if variant["id"] == "axis_sign_flip"
        ),
        None,
    )
    family_envelopes = {
        family: envelope
        for family in ("angle", "vertex", "axis_semantics")
        if (
            envelope := _family_envelope(pooled_variants, family=family)
        )
        is not None
    }
    return {
        "schema_version": AGGREGATE_SCHEMA,
        "execution_status": "COMPLETE",
        "status": (
            "B1_PARAMETER_SENSITIVITY_ALL_VIEW_INVALID"
            if invalid_variants
            else "B1_PARAMETER_SENSITIVITY_QUANTIFIED_PHYSICAL_SELECTION_LOCKED"
        ),
        "scientific_verdict": (
            "INVALID"
            if invalid_variants
            else "MECHANICAL_PASS_PARAMETER_DEPENDENCE_QUANTIFIED_PHYSICAL_SELECTION_REQUIRED"
        ),
        "evidence_scope": "REAL_NINE_VIEW_PREDECLARED_B1_SUPPORT_SENSITIVITY_NO_RECONSTRUCTION_NO_PARAMETER_SELECTION",
        "view_count": len(records),
        "source": {
            "frozen_config_filename": config_path.name,
            "frozen_config_sha256": _sha256_file(config_path),
            "audit_implementation_sha256": _sha256_file(Path(__file__)),
        },
        "configuration": {
            "registration_status": config["registration_status"],
            "frozen_at_utc": config["frozen_at_utc"],
            "baseline_variant_id": baseline_id,
            "variant_count": len(expected_ids),
            "variant_ids": expected_ids,
            "selection_policy": config["selection_policy"],
        },
        "invalid_variant_ids": invalid_variants,
        "views": list(records),
        "aggregate": {
            "variants": pooled_variants,
            "family_envelopes": family_envelopes,
            "largest_active_interval_change": {
                "variant_id": largest_active_change["id"],
                "changed_interval_fraction": largest_active_change["scopes"][
                    "active"
                ]["changed_interval_fraction"],
            },
            "lowest_all_ray_support_iou": {
                "variant_id": lowest_support_iou["id"],
                "ray_support_length_iou": lowest_support_iou["scopes"]["all"][
                    "ray_support_length_iou"
                ],
            },
            "axis_sign_flip": (
                {
                    "active_hit_disagreement_count": axis_flip["scopes"]["active"][
                        "hit_disagreement_count"
                    ],
                    "active_changed_interval_count": axis_flip["scopes"]["active"][
                        "changed_interval_count"
                    ],
                    "active_ray_support_length_iou": axis_flip["scopes"]["active"][
                        "ray_support_length_iou"
                    ],
                    "all_ray_support_length_iou": axis_flip["scopes"]["all"][
                        "ray_support_length_iou"
                    ],
                }
                if axis_flip is not None
                else None
            ),
        },
        "decision": {
            "mechanical_invariants_pass": not invalid_variants,
            "released_parameters_physically_validated": False,
            "parameter_optimization_run": False,
            "selected_variant_id": None,
            "least_assumptive_reference": "B0_FORWARD_BOX_WITH_FIXED_DENOMINATOR_APERTURE_INDICATOR",
            "next_required_evidence": [
                "rig metadata defining cone axis sign, vertex, and angle",
                "frozen held-out camera reprojection",
                "flow-off or calibration-target uncertainty",
            ],
            "training_ready": "NO",
            "algorithm_superiority_claim": "LOCKED",
        },
        "limitations": list(config["interpretation_rules"])
        + [
            "the ray table contains no independent three-dimensional density truth",
            "aggregate support sensitivity does not establish reconstruction benefit",
        ],
    }


def write_metrics_csv(path: Path, report: Mapping[str, Any]) -> None:
    fieldnames = [
        "level",
        "view_id_zero_based",
        "variant_id",
        "family",
        "scope",
        "ray_count",
        "candidate_hit_fraction",
        "changed_interval_fraction",
        "hit_disagreement_fraction",
        "ray_support_length_iou",
        "candidate_path_fraction_of_b0",
        "candidate_path_relative_signed_change_from_baseline",
        "gained_hit_count",
        "lost_hit_count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    with partial.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()

        def emit(
            *, level: str, view_id: int | str, variant: Mapping[str, Any]
        ) -> None:
            for scope_name in SCOPE_NAMES:
                scope = variant["scopes"][scope_name]
                writer.writerow(
                    {
                        "level": level,
                        "view_id_zero_based": view_id,
                        "variant_id": variant["id"],
                        "family": variant["family"],
                        "scope": scope_name,
                        "ray_count": scope["ray_count"],
                        "candidate_hit_fraction": scope["candidate_hit_fraction"],
                        "changed_interval_fraction": scope[
                            "changed_interval_fraction"
                        ],
                        "hit_disagreement_fraction": scope[
                            "hit_disagreement_fraction"
                        ],
                        "ray_support_length_iou": scope[
                            "ray_support_length_iou"
                        ],
                        "candidate_path_fraction_of_b0": scope[
                            "candidate_path_fraction_of_b0"
                        ],
                        "candidate_path_relative_signed_change_from_baseline": scope[
                            "candidate_path_relative_signed_change_from_baseline"
                        ],
                        "gained_hit_count": scope["gained_hit_count"],
                        "lost_hit_count": scope["lost_hit_count"],
                    }
                )

        for view in report["views"]:
            for variant in view["variants"]:
                emit(
                    level="view",
                    view_id=int(view["view_id_zero_based"]),
                    variant=variant,
                )
        for variant in report["aggregate"]["variants"]:
            emit(level="pooled", view_id="ALL", variant=variant)
    os.replace(partial, path)


def run_all_view_parameter_sensitivity(
    *,
    audit_root: Path,
    config_path: Path,
    output_path: Path,
    csv_output_path: Path,
    view_count: int,
    chunk_rows: int = 65_536,
) -> dict[str, Any]:
    config = load_frozen_config(config_path)
    records = [
        audit_parameter_sensitivity_view(
            view_dir=audit_root / f"view_{view_id:02d}",
            config=config,
            config_path=config_path,
            chunk_rows=chunk_rows,
        )
        for view_id in range(view_count)
    ]
    report = aggregate_parameter_sensitivity_views(
        records, config=config, config_path=config_path
    )
    _atomic_json(output_path, report)
    write_metrics_csv(csv_output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--view-count", type=int, required=True)
    parser.add_argument("--chunk-rows", type=int, default=65_536)
    args = parser.parse_args()
    report = run_all_view_parameter_sensitivity(
        audit_root=args.audit_root,
        config_path=args.config,
        output_path=args.output,
        csv_output_path=args.csv_output,
        view_count=args.view_count,
        chunk_rows=args.chunk_rows,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "scientific_verdict": report["scientific_verdict"],
                "aggregate": report["aggregate"],
                "decision": report["decision"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["scientific_verdict"] != "INVALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())
