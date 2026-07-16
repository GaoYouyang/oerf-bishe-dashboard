#!/usr/bin/env python3
"""Build a strict public summary of the PSU B1 parameter-sensitivity audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping


REPORT_SCHEMA = "psu-b1-parameter-sensitivity-all-view-audit-1.0"
REPORT_STATUS = "B1_PARAMETER_SENSITIVITY_QUANTIFIED_PHYSICAL_SELECTION_LOCKED"
CONFIG_SCHEMA = "psu-b1-parameter-sensitivity-preregistration-1.0"
CONFIG_STATUS = "FROZEN_BEFORE_REAL_SCORING"
PUBLIC_SCHEMA = "psu-b1-parameter-sensitivity-public-summary-1.0"
PUBLIC_STATUS = "B1_PARAMETER_DEPENDENCE_QUANTIFIED_PHYSICAL_SELECTION_REQUIRED"
SCOPE_NAMES = ("all", "active", "inactive")
PUBLIC_SCOPE_FIELDS = (
    "ray_count",
    "candidate_hit_count",
    "candidate_hit_fraction",
    "gained_hit_count",
    "lost_hit_count",
    "hit_disagreement_count",
    "hit_disagreement_fraction",
    "changed_interval_count",
    "changed_interval_fraction",
    "ray_support_length_iou",
    "candidate_path_fraction_of_b0",
    "candidate_path_relative_signed_change_from_baseline",
)


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be an object")
    return value


def _array(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{location} must be an array")
    return value


def _integer(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{location} must be a non-negative integer")
    return value


def _number(value: Any, location: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{location} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{location} must be finite")
    return result


def _fraction(value: Any, location: str) -> float:
    result = _number(value, location)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{location} must be in [0, 1]")
    return result


def _read(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON: {path}") from exc
    return _mapping(value, str(path))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def _public_scope(scope: Mapping[str, Any], location: str) -> dict[str, Any]:
    value: dict[str, Any] = {}
    integer_fields = {
        "ray_count",
        "candidate_hit_count",
        "gained_hit_count",
        "lost_hit_count",
        "hit_disagreement_count",
        "changed_interval_count",
    }
    fraction_fields = {
        "candidate_hit_fraction",
        "hit_disagreement_fraction",
        "changed_interval_fraction",
        "ray_support_length_iou",
        "candidate_path_fraction_of_b0",
    }
    for field in PUBLIC_SCOPE_FIELDS:
        if field not in scope:
            raise ValueError(f"{location} missing {field}")
        if field in integer_fields:
            value[field] = _integer(scope[field], f"{location}.{field}")
        elif field in fraction_fields:
            value[field] = _fraction(scope[field], f"{location}.{field}")
        else:
            value[field] = _number(scope[field], f"{location}.{field}")
    if value["candidate_hit_count"] > value["ray_count"]:
        raise ValueError(f"{location} candidate hits exceed rows")
    if value["gained_hit_count"] + value["lost_hit_count"] != value[
        "hit_disagreement_count"
    ]:
        raise ValueError(f"{location} hit disagreement does not reconcile")
    return value


def _public_variant(
    variant: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    location: str,
) -> dict[str, Any]:
    variant_id = str(variant.get("id", ""))
    if variant_id != str(expected.get("id", "")):
        raise ValueError(f"{location} id does not match frozen config")
    if variant.get("mechanical_invariants_pass") is not True:
        raise ValueError(f"{location} failed mechanical invariants")
    configuration = _mapping(variant.get("configuration"), f"{location}.configuration")
    expected_vertex = [
        _number(value, f"{location}.expected_vertex")
        for value in _array(expected.get("cone_vertex_m"), f"{location}.expected_vertex")
    ]
    actual_vertex = [
        _number(value, f"{location}.actual_vertex")
        for value in _array(
            configuration.get("cone_vertex_m"), f"{location}.actual_vertex"
        )
    ]
    if len(expected_vertex) != 3 or actual_vertex != expected_vertex:
        raise ValueError(f"{location} vertex differs from frozen config")
    expected_axis = [
        _number(value, f"{location}.expected_axis")
        for value in _array(expected.get("cone_axis"), f"{location}.expected_axis")
    ]
    norm = math.sqrt(sum(value * value for value in expected_axis))
    expected_axis = [value / norm for value in expected_axis]
    actual_axis = [
        _number(value, f"{location}.actual_axis")
        for value in _array(
            configuration.get("cone_axis_normalized"), f"{location}.actual_axis"
        )
    ]
    if len(actual_axis) != 3 or any(
        not math.isclose(actual, expected_value, rel_tol=1e-12, abs_tol=1e-12)
        for actual, expected_value in zip(actual_axis, expected_axis)
    ):
        raise ValueError(f"{location} axis differs from frozen config")
    angle = _number(
        configuration.get("cone_angle_degrees"), f"{location}.angle"
    )
    expected_angle = _number(
        expected.get("cone_angle_degrees"), f"{location}.expected_angle"
    )
    if not math.isclose(angle, expected_angle, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{location} angle differs from frozen config")
    scopes = _mapping(variant.get("scopes"), f"{location}.scopes")
    return {
        "id": variant_id,
        "family": str(expected.get("family", "unspecified")),
        "role": str(expected.get("role", "")),
        "configuration": {
            "cone_vertex_m": actual_vertex,
            "cone_axis_normalized": actual_axis,
            "cone_angle_degrees": angle,
        },
        "scopes": {
            scope_name: _public_scope(
                _mapping(scopes.get(scope_name), f"{location}.{scope_name}"),
                f"{location}.{scope_name}",
            )
            for scope_name in SCOPE_NAMES
        },
    }


def build_public_summary(
    report_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    report = _read(report_path)
    config = _read(config_path)
    if report.get("schema_version") != REPORT_SCHEMA:
        raise ValueError("unsupported sensitivity report schema")
    if report.get("status") != REPORT_STATUS:
        raise ValueError("sensitivity report is not a locked complete audit")
    if report.get("scientific_verdict") != (
        "MECHANICAL_PASS_PARAMETER_DEPENDENCE_QUANTIFIED_PHYSICAL_SELECTION_REQUIRED"
    ):
        raise ValueError("unexpected scientific verdict")
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unsupported frozen config schema")
    if config.get("registration_status") != CONFIG_STATUS:
        raise ValueError("config was not frozen before real scoring")
    if _mapping(report.get("source"), "report.source").get(
        "frozen_config_sha256"
    ) != _sha256(config_path):
        raise ValueError("report is not bound to the supplied frozen config")

    config_variants = [
        _mapping(value, f"config.variants[{index}]")
        for index, value in enumerate(
            _array(config.get("variants"), "config.variants")
        )
    ]
    aggregate = _mapping(report.get("aggregate"), "report.aggregate")
    aggregate_variants_raw = _array(
        aggregate.get("variants"), "report.aggregate.variants"
    )
    aggregate_variants = [
        _public_variant(
            _mapping(value, f"report.aggregate.variants[{index}]"),
            expected=config_variants[index],
            location=f"report.aggregate.variants[{index}]",
        )
        for index, value in enumerate(aggregate_variants_raw)
    ]
    if len(aggregate_variants) != len(config_variants):
        raise ValueError("aggregate variant count differs from frozen config")

    views_raw = _array(report.get("views"), "report.views")
    views = []
    for view_index, raw_view in enumerate(views_raw):
        view = _mapping(raw_view, f"report.views[{view_index}]")
        view_id = _integer(
            view.get("view_id_zero_based"),
            f"report.views[{view_index}].view_id",
        )
        if view_id != view_index:
            raise ValueError("view ids must be ordered and contiguous")
        raw_variants = _array(
            view.get("variants"), f"report.views[{view_index}].variants"
        )
        if len(raw_variants) != len(config_variants):
            raise ValueError("view variant count differs from frozen config")
        public_variants = []
        for variant_index, raw_variant in enumerate(raw_variants):
            value = _public_variant(
                _mapping(
                    raw_variant,
                    f"report.views[{view_index}].variants[{variant_index}]",
                ),
                expected=config_variants[variant_index],
                location=(
                    f"report.views[{view_index}].variants[{variant_index}]"
                ),
            )
            public_variants.append(
                {
                    "id": value["id"],
                    "active": value["scopes"]["active"],
                    "all": value["scopes"]["all"],
                }
            )
        views.append(
            {
                "view_id_zero_based": view_id,
                "variants": public_variants,
            }
        )
    if len(views) != 9:
        raise ValueError("real PSU public sensitivity summary requires nine views")

    by_id = {variant["id"]: variant for variant in aggregate_variants}
    baseline_id = str(config.get("baseline_variant_id", ""))
    if baseline_id not in by_id or "axis_sign_flip" not in by_id:
        raise ValueError("baseline and axis-sign variants are required")
    nonreference = [
        variant for variant in aggregate_variants if variant["id"] != baseline_id
    ]
    vertex_variants = [
        variant for variant in nonreference if variant["family"] == "vertex"
    ]
    angle_variants = [
        variant for variant in nonreference if variant["family"] == "angle"
    ]
    axis = by_id["axis_sign_flip"]["scopes"]["active"]
    worst_vertex = min(
        vertex_variants,
        key=lambda value: value["scopes"]["active"]["candidate_hit_fraction"],
    )
    narrowest_angle = min(
        angle_variants,
        key=lambda value: value["configuration"]["cone_angle_degrees"],
    )
    return {
        "schema_version": PUBLIC_SCHEMA,
        "status": PUBLIC_STATUS,
        "evidence_scope": "REAL_NINE_VIEW_PREDECLARED_B1_SUPPORT_SENSITIVITY_NO_RECONSTRUCTION_NO_PARAMETER_SELECTION",
        "preregistration": {
            "registration_status": config["registration_status"],
            "frozen_at_utc": config["frozen_at_utc"],
            "baseline_variant_id": baseline_id,
            "variant_ids": [variant["id"] for variant in config_variants],
            "selection_policy": config["selection_policy"],
        },
        "view_count": len(views),
        "variant_count": len(aggregate_variants),
        "aggregate_variants": aggregate_variants,
        "per_view": views,
        "headline_metrics": {
            "axis_sign_flip_active_lost_hit_count": axis["lost_hit_count"],
            "axis_sign_flip_active_hit_fraction": axis["candidate_hit_fraction"],
            "axis_sign_flip_active_ray_support_length_iou": axis[
                "ray_support_length_iou"
            ],
            "narrowest_angle_variant_id": narrowest_angle["id"],
            "narrowest_angle_active_hit_fraction": narrowest_angle["scopes"][
                "active"
            ]["candidate_hit_fraction"],
            "worst_vertex_variant_id": worst_vertex["id"],
            "worst_vertex_active_hit_fraction": worst_vertex["scopes"]["active"][
                "candidate_hit_fraction"
            ],
            "vertex_all_ray_support_length_iou_range": [
                min(
                    variant["scopes"]["all"]["ray_support_length_iou"]
                    for variant in vertex_variants
                ),
                max(
                    variant["scopes"]["all"]["ray_support_length_iou"]
                    for variant in vertex_variants
                ),
            ],
        },
        "decision": {
            "mechanical_invariants_pass": True,
            "parameter_dependence_quantified": True,
            "released_parameters_physically_validated": False,
            "parameter_optimization_run": False,
            "selected_variant_id": None,
            "least_assumptive_reference": "B0_FORWARD_BOX_WITH_FIXED_DENOMINATOR_APERTURE_INDICATOR",
            "next_gate": "FROZEN_70_VIEW_HELD_OUT_REPROJECTION_AND_FLOW_OFF_REPEATABILITY",
            "training_ready": "NO",
            "algorithm_superiority_claim": "LOCKED",
        },
        "limitations": list(config["interpretation_rules"])
        + [
            "all changed-interval fractions use a two-nanometer numerical tolerance and are effect diagnostics, not calibration uncertainty estimates",
            "the real ray table has no independent three-dimensional density truth",
            "no finite-aperture rerendering, held-out view, inverse reconstruction, or neural model is scored",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = build_public_summary(args.report, args.config)
    _atomic_json(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
