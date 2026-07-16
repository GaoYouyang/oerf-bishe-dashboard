#!/usr/bin/env python3
"""Export a strict public summary from the private PSU fixed-domain audit."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


PRIVATE_SCHEMA_VERSION = "psu-bost-fixed-domain-all-view-audit-1.0"
PRIVATE_VIEW_SCHEMA_VERSION = "psu-bost-fixed-domain-geometry-audit-1.0"
PUBLIC_SCHEMA_VERSION = "psu-bost-fixed-domain-geometry-public-summary-1.0"

EXECUTION_STATUS = "COMPLETE"
REPORT_STATUS = "B0_B1_ALL_VIEW_ANALYTIC_CONTRACT_PASS_B2_REQUIRED"
VIEW_STATUS = "B0_B1_FIXED_DOMAIN_ANALYTIC_CONTRACT_PASS_B2_REQUIRED"
SCIENTIFIC_VERDICT = (
    "MECHANICAL_PASS_PHYSICAL_CONE_SEMANTICS_AND_FINITE_APERTURE_UNCONFIRMED"
)
NEXT_GATE = "B2_FINITE_APERTURE_DOMAIN_INDICATOR_AND_B3_GEOMETRY_SAFE_MASK"
VIEW_EVIDENCE_SCOPE = (
    "REAL_ONE_VIEW_B0_FORWARD_BOX_AND_B1_ONE_NAPPE_CONE_BOX_RAY_CENSUS_"
    "NO_TENSORFLOW_NO_RECONSTRUCTION"
)

BUNDLE_STATUS = "VIEW_SHARD_BUNDLE_TRANSCODED_AND_SOURCE_STREAMS_VERIFIED"
MASK_STATUS = "CORRECTED_VIEW_MASK_SHARDS_BUILT_MECHANICAL_CONTRACT_PASS"
SETUP_STATUSES = {
    "STREAMED_SETUP_MECHANICAL_CONTRACT_PASS",
    "STREAMED_SETUP_DIAGNOSTIC_NO_GO",
}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

COUNT_FIELDS = (
    "ray_count",
    "author_selected_nonzero_count",
    "author_full_box_zero_flag_count",
    "b0_hit_count",
    "b0_zero_length_count",
    "b1_hit_count",
    "b1_zero_length_count",
    "b1_removed_from_b0_count",
    "b0_endpoint_box_violation_count",
    "b1_endpoint_box_violation_count",
    "b1_endpoint_nappe_violation_count",
    "b1_endpoint_cone_radial_violation_count",
    "b1_midpoint_cone_violation_count",
    "b1_hit_without_b0_hit_count",
    "b1_length_exceeds_b0_count",
    "nonfinite_output_count",
    "b0_point_touch_count",
    "b1_point_touch_count",
    "b1_nappe_rejected_double_cone_count",
    "b1_nappe_rejected_with_b0_hit_count",
)
PATH_SUM_FIELDS = (
    "author_selected_length_sum_m",
    "b0_length_sum_m",
    "b1_length_sum_m",
)
PATH_FRACTION_FIELDS = (
    "b1_fraction_of_b0",
    "b1_fraction_of_author_selected",
    "b0_fraction_of_author_selected",
)
MASK_COUNT_FIELDS = (
    "count",
    "author_nonzero_count",
    "b0_hit_count",
    "b1_hit_count",
    "b1_removed_from_b0_count",
)
MASK_SUM_FIELDS = (
    "author_length_sum_m",
    "b0_length_sum_m",
    "b1_length_sum_m",
)
MASK_FRACTION_FIELDS = (
    "author_nonzero_fraction",
    "b0_hit_fraction",
    "b1_hit_fraction",
    "b1_removed_from_b0_fraction",
    "b1_path_fraction_of_b0",
)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return value


def _sequence(value: Any, path: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be an array")
    return value


def _required(value: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in value:
        raise ValueError(f"{path}.{key} is required")
    return value[key]


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a nonempty string")
    if any(character in value for character in "\r\n\t"):
        raise ValueError(f"{path} must be single-line text")
    return value


def _expected_text(value: Any, expected: str, path: str) -> str:
    text = _text(value, path)
    if text != expected:
        raise ValueError(f"{path} has an unreviewed value: {text}")
    return text


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a boolean")
    return value


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{path} must be an integer >= {minimum}")
    return value


def _number(value: Any, path: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{path} must be a finite number >= {minimum}")
    return result


def _fraction(value: Any, path: str) -> float:
    result = _number(value, path)
    if result > 1.0:
        raise ValueError(f"{path} must be <= 1")
    return result


def _filename(value: Any, path: str) -> str:
    result = _text(value, path)
    if Path(result).name != result or "/" in result or "\\" in result:
        raise ValueError(f"{path} must be a basename without directories")
    return result


def _sha256(value: Any, path: str) -> str:
    result = _text(value, path)
    if not _SHA256_PATTERN.fullmatch(result):
        raise ValueError(f"{path} must be a lowercase SHA-256 digest")
    return result


def _same_number(actual: float, expected: float, path: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError(f"{path} is inconsistent")


def _ratio(numerator: int | float, denominator: int | float, path: str) -> float:
    if denominator == 0:
        raise ValueError(f"{path} denominator must be nonzero")
    return float(numerator) / float(denominator)


def _validate_json_tree(value: Any, path: str = "report") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string key")
            _validate_json_tree(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_tree(item, f"{path}[{index}]")
    elif value is None or isinstance(value, (str, bool, int)):
        return
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
    else:
        raise ValueError(f"{path} contains a non-JSON value")


def _validate_decision(value: Any, path: str) -> None:
    source = _mapping(value, path)
    expected_booleans = {
        "analytic_domain_invariants_pass": True,
        "declared_computational_domain_mechanically_enforced": True,
        "physical_spatial_domain_validated": False,
        "cone_parameter_physical_semantics_confirmed": False,
        "finite_aperture_sample_support_audited": False,
    }
    for field, expected in expected_booleans.items():
        actual = _boolean(_required(source, field, path), f"{path}.{field}")
        if actual is not expected:
            raise ValueError(f"{path}.{field} conflicts with the public claim boundary")
    _expected_text(_required(source, "training_ready", path), "NO", f"{path}.training_ready")
    _expected_text(
        _required(source, "algorithm_superiority_claim", path),
        "LOCKED",
        f"{path}.algorithm_superiority_claim",
    )
    _expected_text(_required(source, "next_gate", path), NEXT_GATE, f"{path}.next_gate")
    if "author_mixed_domain_length_comparison_is_context_only" in source:
        if not _boolean(
            source["author_mixed_domain_length_comparison_is_context_only"],
            f"{path}.author_mixed_domain_length_comparison_is_context_only",
        ):
            raise ValueError(
                f"{path}.author_mixed_domain_length_comparison_is_context_only "
                "must be true"
            )


def _copy_counts(value: Any, path: str) -> dict[str, int]:
    source = _mapping(value, path)
    return {
        field: _integer(
            _required(source, field, path),
            f"{path}.{field}",
            minimum=1 if field == "ray_count" else 0,
        )
        for field in COUNT_FIELDS
    }


def _copy_path_length(value: Any, path: str) -> dict[str, float]:
    source = _mapping(value, path)
    result = {
        field: _number(_required(source, field, path), f"{path}.{field}")
        for field in PATH_SUM_FIELDS
    }
    result.update(
        {
            field: _fraction(_required(source, field, path), f"{path}.{field}")
            if field == "b1_fraction_of_b0"
            else _number(_required(source, field, path), f"{path}.{field}")
            for field in PATH_FRACTION_FIELDS
        }
    )
    if result["b1_length_sum_m"] > result["b0_length_sum_m"]:
        raise ValueError(f"{path}.b1_length_sum_m cannot exceed b0_length_sum_m")
    _same_number(
        result["b1_fraction_of_b0"],
        _ratio(
            result["b1_length_sum_m"],
            result["b0_length_sum_m"],
            f"{path}.b1_fraction_of_b0",
        ),
        f"{path}.b1_fraction_of_b0",
    )
    _same_number(
        result["b1_fraction_of_author_selected"],
        _ratio(
            result["b1_length_sum_m"],
            result["author_selected_length_sum_m"],
            f"{path}.b1_fraction_of_author_selected",
        ),
        f"{path}.b1_fraction_of_author_selected",
    )
    _same_number(
        result["b0_fraction_of_author_selected"],
        _ratio(
            result["b0_length_sum_m"],
            result["author_selected_length_sum_m"],
            f"{path}.b0_fraction_of_author_selected",
        ),
        f"{path}.b0_fraction_of_author_selected",
    )
    return result


def _copy_mask(value: Any, path: str) -> dict[str, Any]:
    source = _mapping(value, path)
    result: dict[str, Any] = {
        field: _integer(
            _required(source, field, path),
            f"{path}.{field}",
            minimum=1 if field == "count" else 0,
        )
        for field in MASK_COUNT_FIELDS
    }
    result.update(
        {
            field: _number(_required(source, field, path), f"{path}.{field}")
            for field in MASK_SUM_FIELDS
        }
    )
    result.update(
        {
            field: _fraction(_required(source, field, path), f"{path}.{field}")
            for field in MASK_FRACTION_FIELDS
        }
    )
    count = result["count"]
    for field in ("author_nonzero_count", "b0_hit_count", "b1_hit_count"):
        if result[field] > count:
            raise ValueError(f"{path}.{field} cannot exceed count")
    if result["b1_hit_count"] > result["b0_hit_count"]:
        raise ValueError(f"{path}.b1_hit_count cannot exceed b0_hit_count")
    if result["b1_removed_from_b0_count"] != (
        result["b0_hit_count"] - result["b1_hit_count"]
    ):
        raise ValueError(f"{path}.b1_removed_from_b0_count is inconsistent")
    for count_field, fraction_field in (
        ("author_nonzero_count", "author_nonzero_fraction"),
        ("b0_hit_count", "b0_hit_fraction"),
        ("b1_hit_count", "b1_hit_fraction"),
        ("b1_removed_from_b0_count", "b1_removed_from_b0_fraction"),
    ):
        _same_number(
            result[fraction_field],
            _ratio(result[count_field], count, f"{path}.{fraction_field}"),
            f"{path}.{fraction_field}",
        )
    if result["b1_length_sum_m"] > result["b0_length_sum_m"]:
        raise ValueError(f"{path}.b1_length_sum_m cannot exceed b0_length_sum_m")
    _same_number(
        result["b1_path_fraction_of_b0"],
        _ratio(
            result["b1_length_sum_m"],
            result["b0_length_sum_m"],
            f"{path}.b1_path_fraction_of_b0",
        ),
        f"{path}.b1_path_fraction_of_b0",
    )
    return result


def _validate_configuration(value: Any, counts: Mapping[str, int], path: str) -> None:
    source = _mapping(value, path)
    rows = _integer(_required(source, "rows", path), f"{path}.rows", minimum=1)
    if rows != counts["ray_count"]:
        raise ValueError(f"{path}.rows conflicts with counts.ray_count")
    _integer(_required(source, "chunk_rows", path), f"{path}.chunk_rows", minimum=2)
    for field in (
        "outer_minimum_m",
        "outer_maximum_m",
        "cone_vertex_m",
        "cone_axis_normalized",
    ):
        vector = _sequence(_required(source, field, path), f"{path}.{field}")
        if len(vector) != 3:
            raise ValueError(f"{path}.{field} must contain three values")
        for index, item in enumerate(vector):
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError(f"{path}.{field}[{index}] must be a finite number")
            if not math.isfinite(float(item)):
                raise ValueError(f"{path}.{field}[{index}] must be a finite number")
    angle = _number(
        _required(source, "cone_angle_degrees", path),
        f"{path}.cone_angle_degrees",
    )
    if not 0.0 < angle < 90.0:
        raise ValueError(f"{path}.cone_angle_degrees must be between 0 and 90")
    _expected_text(
        _required(source, "geometry_contract_version", path),
        "psu-bost-forward-geometry-1.0",
        f"{path}.geometry_contract_version",
    )
    _expected_text(
        _required(source, "b0_policy", path),
        "FORWARD_NORMALIZED_RAY_INTERSECT_CLOSED_AXIS_ALIGNED_BOX",
        f"{path}.b0_policy",
    )
    _expected_text(
        _required(source, "b1_policy", path),
        "B0_INTERSECT_NORMALIZED_ONE_NAPPE_CONE_NO_FALLBACK",
        f"{path}.b1_policy",
    )


def _validate_private_source(value: Any, path: str) -> tuple[str, str, str]:
    source = _mapping(value, path)
    for field in (
        "bundle_manifest_sha256",
        "setup_manifest_sha256",
        "corrected_mask_manifest_sha256",
    ):
        _sha256(_required(source, field, path), f"{path}.{field}")
    geometry_filename = _filename(
        _required(source, "geometry_implementation_filename", path),
        f"{path}.geometry_implementation_filename",
    )
    if geometry_filename != "psu_bost_forward_geometry.py":
        raise ValueError(f"{path}.geometry_implementation_filename is unreviewed")
    geometry_sha256 = _sha256(
        _required(source, "geometry_implementation_sha256", path),
        f"{path}.geometry_implementation_sha256",
    )
    audit_sha256 = _sha256(
        _required(source, "audit_implementation_sha256", path),
        f"{path}.audit_implementation_sha256",
    )
    return geometry_filename, geometry_sha256, audit_sha256


def _validate_view_counts(counts: Mapping[str, int], path: str) -> None:
    ray_count = counts["ray_count"]
    if counts["b0_hit_count"] + counts["b0_zero_length_count"] != ray_count:
        raise ValueError(f"{path} B0 hit and zero-length counts are inconsistent")
    if counts["b1_hit_count"] + counts["b1_zero_length_count"] != ray_count:
        raise ValueError(f"{path} B1 hit and zero-length counts are inconsistent")
    if counts["b1_hit_count"] + counts["b1_removed_from_b0_count"] != counts[
        "b0_hit_count"
    ]:
        raise ValueError(f"{path}.b1_removed_from_b0_count is inconsistent")
    required_zero_fields = (
        "b0_endpoint_box_violation_count",
        "b1_endpoint_box_violation_count",
        "b1_endpoint_nappe_violation_count",
        "b1_endpoint_cone_radial_violation_count",
        "b1_midpoint_cone_violation_count",
        "b1_hit_without_b0_hit_count",
        "b1_length_exceeds_b0_count",
        "nonfinite_output_count",
        "b0_point_touch_count",
        "b1_point_touch_count",
    )
    for field in required_zero_fields:
        if counts[field] != 0:
            raise ValueError(f"{path}.{field} must be zero for a passing audit")


def _copy_view(value: Any, index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    path = f"report.views[{index}]"
    source = _mapping(value, path)
    _expected_text(
        _required(source, "schema_version", path),
        PRIVATE_VIEW_SCHEMA_VERSION,
        f"{path}.schema_version",
    )
    _expected_text(_required(source, "status", path), VIEW_STATUS, f"{path}.status")
    _expected_text(
        _required(source, "evidence_scope", path),
        VIEW_EVIDENCE_SCOPE,
        f"{path}.evidence_scope",
    )
    view_id = _integer(
        _required(source, "view_id_zero_based", path),
        f"{path}.view_id_zero_based",
    )
    counts = _copy_counts(_required(source, "counts", path), f"{path}.counts")
    _validate_view_counts(counts, f"{path}.counts")
    path_length = _copy_path_length(
        _required(source, "path_length", path),
        f"{path}.path_length",
    )
    masks = _mapping(
        _required(source, "mask_conditioned", path),
        f"{path}.mask_conditioned",
    )
    active = _copy_mask(
        _required(masks, "amask_all", f"{path}.mask_conditioned"),
        f"{path}.mask_conditioned.amask_all",
    )
    inactive = _copy_mask(
        _required(masks, "imask_all", f"{path}.mask_conditioned"),
        f"{path}.mask_conditioned.imask_all",
    )
    _validate_decision(_required(source, "decision", path), f"{path}.decision")
    upstream = _mapping(
        _required(source, "upstream_view_contract", path),
        f"{path}.upstream_view_contract",
    )
    _expected_text(
        _required(upstream, "bundle_status", f"{path}.upstream_view_contract"),
        BUNDLE_STATUS,
        f"{path}.upstream_view_contract.bundle_status",
    )
    _expected_text(
        _required(upstream, "mask_status", f"{path}.upstream_view_contract"),
        MASK_STATUS,
        f"{path}.upstream_view_contract.mask_status",
    )
    setup_status = _text(
        _required(upstream, "setup_status", f"{path}.upstream_view_contract"),
        f"{path}.upstream_view_contract.setup_status",
    )
    if setup_status not in SETUP_STATUSES:
        raise ValueError(
            f"{path}.upstream_view_contract.setup_status has an unreviewed value"
        )
    _validate_configuration(
        _required(source, "configuration", path),
        counts,
        f"{path}.configuration",
    )
    code_identity = _validate_private_source(
        _required(source, "source", path),
        f"{path}.source",
    )

    public = {
        "view_id_zero_based": view_id,
        "counts": {
            "ray_count": counts["ray_count"],
            "b0_hit_count": counts["b0_hit_count"],
            "b1_hit_count": counts["b1_hit_count"],
            "b1_removed_from_b0_count": counts["b1_removed_from_b0_count"],
        },
        "hit_fractions": {
            "b0_hit_fraction": _ratio(
                counts["b0_hit_count"],
                counts["ray_count"],
                f"{path}.counts.b0_hit_count",
            ),
            "b1_hit_fraction": _ratio(
                counts["b1_hit_count"],
                counts["ray_count"],
                f"{path}.counts.b1_hit_count",
            ),
            "b1_removed_from_b0_fraction": _ratio(
                counts["b1_removed_from_b0_count"],
                counts["ray_count"],
                f"{path}.counts.b1_removed_from_b0_count",
            ),
        },
        "path_length": {
            "b0_length_sum_m": path_length["b0_length_sum_m"],
            "b1_length_sum_m": path_length["b1_length_sum_m"],
            "b1_fraction_of_b0": path_length["b1_fraction_of_b0"],
        },
        "mask_conditioned": {
            "active_b1_hit_fraction": active["b1_hit_fraction"],
            "inactive_b1_hit_fraction": inactive["b1_hit_fraction"],
        },
    }
    private = {
        "counts": counts,
        "path_length": path_length,
        "active": active,
        "inactive": inactive,
        "code_identity": code_identity,
        "setup_status": setup_status,
    }
    return public, private


def _copy_aggregate(
    value: Any,
    private_views: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    path = "report.aggregate"
    source = _mapping(value, path)
    counts = _copy_counts(_required(source, "counts", path), f"{path}.counts")
    path_length = _copy_path_length(
        _required(source, "path_length", path),
        f"{path}.path_length",
    )
    expected_counts = {
        field: sum(int(view["counts"][field]) for view in private_views)
        for field in COUNT_FIELDS
    }
    for field, expected in expected_counts.items():
        if counts[field] != expected:
            raise ValueError(f"{path}.counts.{field} conflicts with per-view data")
    for field in PATH_SUM_FIELDS:
        expected = sum(float(view["path_length"][field]) for view in private_views)
        _same_number(path_length[field], expected, f"{path}.path_length.{field}")

    invalid_ids = [
        _integer(item, f"{path}.invalid_view_ids[{index}]")
        for index, item in enumerate(
            _sequence(_required(source, "invalid_view_ids", path), f"{path}.invalid_view_ids")
        )
    ]
    if invalid_ids:
        raise ValueError(f"{path}.invalid_view_ids must be empty for a passing audit")
    active_shortfall_ids = [
        _integer(item, f"{path}.active_mask_b1_zero_view_ids[{index}]")
        for index, item in enumerate(
            _sequence(
                _required(source, "active_mask_b1_zero_view_ids", path),
                f"{path}.active_mask_b1_zero_view_ids",
            )
        )
    ]
    expected_shortfall_ids = [
        index
        for index, view in enumerate(private_views)
        if view["active"]["b1_hit_count"] < view["active"]["count"]
    ]
    if active_shortfall_ids != expected_shortfall_ids:
        raise ValueError(
            f"{path}.active_mask_b1_zero_view_ids conflicts with per-view masks"
        )
    upstream_no_go_ids = [
        _integer(item, f"{path}.upstream_author_setup_no_go_view_ids[{index}]")
        for index, item in enumerate(
            _sequence(
                _required(source, "upstream_author_setup_no_go_view_ids", path),
                f"{path}.upstream_author_setup_no_go_view_ids",
            )
        )
    ]
    expected_upstream_no_go_ids = [
        index
        for index, view in enumerate(private_views)
        if view["setup_status"] == "STREAMED_SETUP_DIAGNOSTIC_NO_GO"
    ]
    if upstream_no_go_ids != expected_upstream_no_go_ids:
        raise ValueError(
            f"{path}.upstream_author_setup_no_go_view_ids conflicts with views"
        )

    raw_masks = _mapping(
        _required(source, "mask_conditioned", path),
        f"{path}.mask_conditioned",
    )
    aggregate_masks = {
        public_name: _copy_mask(
            _required(raw_masks, private_name, f"{path}.mask_conditioned"),
            f"{path}.mask_conditioned.{private_name}",
        )
        for public_name, private_name in (
            ("active", "amask_all"),
            ("inactive", "imask_all"),
        )
    }
    for public_name in ("active", "inactive"):
        aggregate_mask = aggregate_masks[public_name]
        for field in MASK_COUNT_FIELDS:
            expected = sum(
                int(view[public_name][field]) for view in private_views
            )
            if aggregate_mask[field] != expected:
                raise ValueError(
                    f"{path}.mask_conditioned.{public_name}.{field} "
                    "conflicts with per-view data"
                )
        for field in MASK_SUM_FIELDS:
            expected = sum(
                float(view[public_name][field]) for view in private_views
            )
            _same_number(
                aggregate_mask[field],
                expected,
                f"{path}.mask_conditioned.{public_name}.{field}",
            )

    return {
        "counts": {
            "ray_count": counts["ray_count"],
            "b0_hit_count": counts["b0_hit_count"],
            "b1_hit_count": counts["b1_hit_count"],
            "b1_removed_from_b0_count": counts["b1_removed_from_b0_count"],
        },
        "hit_fractions": {
            "b0_hit_fraction": _ratio(
                counts["b0_hit_count"], counts["ray_count"], f"{path}.counts"
            ),
            "b1_hit_fraction": _ratio(
                counts["b1_hit_count"], counts["ray_count"], f"{path}.counts"
            ),
            "b1_removed_from_b0_fraction": _ratio(
                counts["b1_removed_from_b0_count"],
                counts["ray_count"],
                f"{path}.counts",
            ),
        },
        "path_length": {
            field: path_length[field] for field in PATH_SUM_FIELDS + PATH_FRACTION_FIELDS
        },
        "mask_conditioned": {
            public_name: {
                "count": aggregate_masks[public_name]["count"],
                "b0_hit_count": aggregate_masks[public_name]["b0_hit_count"],
                "b1_hit_count": aggregate_masks[public_name]["b1_hit_count"],
                "b0_hit_fraction": aggregate_masks[public_name]["b0_hit_fraction"],
                "b1_hit_fraction": aggregate_masks[public_name]["b1_hit_fraction"],
                "b1_path_fraction_of_b0": aggregate_masks[public_name][
                    "b1_path_fraction_of_b0"
                ],
            }
            for public_name in ("active", "inactive")
        },
        "active_mask_b1_shortfall_view_count": len(active_shortfall_ids),
        "upstream_author_setup_no_go_view_count": len(upstream_no_go_ids),
    }


def build_public_summary(private_report: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the private audit and rebuild only approved public summaries."""
    _validate_json_tree(private_report)
    source = _mapping(private_report, "report")
    _expected_text(
        _required(source, "schema_version", "report"),
        PRIVATE_SCHEMA_VERSION,
        "report.schema_version",
    )
    execution_status = _expected_text(
        _required(source, "execution_status", "report"),
        EXECUTION_STATUS,
        "report.execution_status",
    )
    scientific_verdict = _expected_text(
        _required(source, "scientific_verdict", "report"),
        SCIENTIFIC_VERDICT,
        "report.scientific_verdict",
    )
    status = _expected_text(
        _required(source, "status", "report"),
        REPORT_STATUS,
        "report.status",
    )
    view_count = _integer(
        _required(source, "view_count", "report"),
        "report.view_count",
        minimum=1,
    )
    raw_views = _sequence(_required(source, "views", "report"), "report.views")
    if len(raw_views) != view_count:
        raise ValueError("report.view_count must equal the number of views")

    public_views: list[dict[str, Any]] = []
    private_views: list[dict[str, Any]] = []
    for index, value in enumerate(raw_views):
        public_view, private_view = _copy_view(value, index)
        public_views.append(public_view)
        private_views.append(private_view)
    view_ids = [view["view_id_zero_based"] for view in public_views]
    if view_ids != list(range(view_count)):
        raise ValueError("view ids must be the ordered contiguous range from zero")
    code_identity = private_views[0]["code_identity"]
    if any(view["code_identity"] != code_identity for view in private_views[1:]):
        raise ValueError("geometry and audit source identity must match across views")

    _validate_decision(_required(source, "decision", "report"), "report.decision")
    aggregate = _copy_aggregate(
        _required(source, "aggregate", "report"),
        private_views,
    )
    report = {
        "schema_version": PUBLIC_SCHEMA_VERSION,
        "source_schema_version": PRIVATE_SCHEMA_VERSION,
        "source_view_schema_version": PRIVATE_VIEW_SCHEMA_VERSION,
        "execution_status": execution_status,
        "scientific_verdict": scientific_verdict,
        "status": status,
        "view_count": view_count,
        "aggregate": aggregate,
        "views": public_views,
        "claim_boundary": {
            "supported_claim": "B0_B1_DECLARED_COMPUTATIONAL_DOMAIN_MECHANICALLY_ENFORCED",
            "b0_b1_declared_computational_domain_mechanically_enforced": True,
            "physical_spatial_domain_validated": False,
            "cone_physical_semantics": "UNCONFIRMED",
            "b2_finite_aperture_support": "MISSING",
            "held_out_reprojection": "MISSING",
            "training": "LOCKED",
            "algorithm_superiority": "LOCKED",
        },
        "public_export_policy": {
            "strict_field_allowlist": True,
            "aggregate_and_per_view_summary_only": True,
            "contains_local_timing": False,
            "contains_private_or_local_paths": False,
            "contains_raw_arrays": False,
            "contains_raw_index_lists": False,
            "contains_source_code_snippets": False,
            "contains_private_hashes": False,
            "contains_configuration_vectors": False,
        },
    }
    _validate_json_tree(report, "public_summary")
    return report


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is not allowed: {value}")


def load_private_report(path: Path) -> Mapping[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_nonstandard_json_constant,
    )
    return _mapping(value, "report")


def write_json_atomic(report: Mapping[str, Any], output_path: Path) -> None:
    payload = (
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def export_public_summary(input_path: Path, output_path: Path) -> dict[str, Any]:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output paths must differ")
    report = build_public_summary(load_private_report(input_path))
    write_json_atomic(report, output_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        "--private-report",
        "--fixed-domain-audit",
        dest="input_path",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_public_summary(args.input_path, args.output)
    print(f"wrote public fixed-domain summary: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
