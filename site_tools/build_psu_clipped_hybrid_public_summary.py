#!/usr/bin/env python3
"""Export a strict public summary from the private PSU clipped-hybrid audit."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


PRIVATE_SCHEMA_VERSION = "psu-bost-author-compatible-clipped-hybrid-all-view-audit-1.0"
PRIVATE_VIEW_SCHEMA_VERSION = "psu-bost-author-compatible-clipped-hybrid-audit-1.0"
PUBLIC_SCHEMA_VERSION = "psu-bost-author-compatible-clipped-hybrid-public-summary-1.0"
EVIDENCE_SCOPE = (
    "REAL_ONE_VIEW_A1_AUTHOR_DOUBLE_CONE_INTERVAL_CLIPPED_TO_FORWARD_BOX_"
    "WITH_AUTHOR_FALLBACK_NO_TENSORFLOW_NO_RECONSTRUCTION"
)

EXECUTION_STATUS = "COMPLETE"
SCIENTIFIC_VERDICTS = {
    "INVALID",
    "AUTHOR_COMPATIBILITY_ABLATION_ONLY",
}
REPORT_STATUSES = {
    "AUTHOR_COMPATIBLE_CLIPPED_HYBRID_ALL_VIEW_INVALID",
    "AUTHOR_COMPATIBLE_CLIPPED_HYBRID_ALL_VIEW_CONTRACT_PASS_MASK_FILTER_REQUIRED",
    "AUTHOR_COMPATIBLE_CLIPPED_HYBRID_ALL_VIEW_CONTRACT_PASS",
}
VIEW_STATUSES = {
    "AUTHOR_COMPATIBLE_CLIPPED_HYBRID_INVALID",
    "AUTHOR_COMPATIBLE_CLIPPED_HYBRID_CONTRACT_PASS_MASK_FILTER_REQUIRED",
    "AUTHOR_COMPATIBLE_CLIPPED_HYBRID_CONTRACT_PASS",
}
VIEW_BUNDLE_STATUS = "VIEW_SHARD_BUNDLE_TRANSCODED_AND_SOURCE_STREAMS_VERIFIED"
MASK_STATUS = "CORRECTED_VIEW_MASK_SHARDS_BUILT_MECHANICAL_CONTRACT_PASS"
CONFIGURATION_POLICY = (
    "AUTHOR_CONE_INTERVAL_INTERSECT_FORWARD_BOX_"
    "AUTHOR_ZERO_CONE_FALLBACK_TO_FORWARD_BOX"
)
RUNTIME_SCOPE = "CACHED_LOCAL_DIAGNOSTIC_NOT_A_SPEED_BENCHMARK"
NEXT_GATE = "GEOMETRY_SAFE_MASK_ABLATION_AND_HELD_OUT_REPROJECTION"

VIEW_COUNT_FIELDS = (
    "ray_count",
    "author_nonzero_count",
    "clipped_nonzero_count",
    "changed_from_author_count",
    "cone_shortened_count",
    "cone_zeroed_for_no_box_overlap_count",
    "forward_box_shortened_count",
    "clipped_endpoint_box_violation_count",
    "clipped_length_exceeds_author_count",
    "nonfinite_output_count",
    "negative_clipped_inner_aperture_radius_count",
    "negative_clipped_outer_aperture_radius_count",
    "clipped_zero_length_count",
)
VIEW_SAFETY_FAILURE_FIELDS = (
    "clipped_endpoint_box_violation_count",
    "clipped_length_exceeds_author_count",
    "nonfinite_output_count",
    "negative_clipped_inner_aperture_radius_count",
    "negative_clipped_outer_aperture_radius_count",
)
PATH_LENGTH_FIELDS = (
    "author_length_sum_m",
    "clipped_length_sum_m",
    "removed_length_sum_m",
)
PATH_FRACTION_FIELDS = ("retained_fraction", "removed_fraction")
MASK_NAMES = ("amask_all", "imask_all")
MASK_COUNT_FIELDS = (
    "count",
    "changed_count",
    "zeroed_count",
    "shortened_count",
)
MASK_LENGTH_FIELDS = (
    "author_length_sum_m",
    "clipped_length_sum_m",
)
MASK_FRACTION_FIELDS = (
    "changed_fraction",
    "zeroed_fraction",
    "shortened_fraction",
    "path_length_retained_fraction",
)
AGGREGATE_COUNT_FIELDS = (
    "changed_ray_count",
    "clipped_zero_length_count",
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_PRIVATE_PATH_PATTERNS = (
    re.compile(
        r"(?i)(?:^|[\s\"'(])(?:/(?:Users|home|private|tmp|Volumes)/"
        r"|/var/folders/|~[/\\]|\.\.?[/\\])"
    ),
    re.compile(r"\b[A-Za-z]:[\\/]"),
    re.compile(
        r"(?i)\b(?:file://|private_library(?:[/\\]|$)|"
        r"localhost(?::\d+)?|127\.0\.0\.1)"
    ),
)
_SOURCE_SNIPPET_PATTERNS = (
    re.compile(r"\bdef\s+[A-Za-z_]\w*\s*\("),
    re.compile(r"\bclass\s+[A-Za-z_]\w*\s*[:(]"),
    re.compile(r"\bfrom\s+\S+\s+import\s+"),
    re.compile(r"(?:^|[;\s])import\s+[A-Za-z_]\w*"),
)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return value


def _sequence(value: Any, path: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be an array")
    return value


def _required(mapping: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{path}.{key} is required")
    return mapping[key]


def _safe_text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a nonempty string")
    if any(character in value for character in ("\n", "\r", "\t")):
        raise ValueError(f"{path} must be single-line text")
    if any(pattern.search(value) for pattern in _PRIVATE_PATH_PATTERNS):
        raise ValueError(f"{path} contains a private or local path")
    if any(pattern.search(value) for pattern in _SOURCE_SNIPPET_PATTERNS):
        raise ValueError(f"{path} contains a source-code snippet")
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a boolean")
    return value


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{path} must be an integer >= {minimum}")
    return value


def _number(
    value: Any,
    path: str,
    *,
    minimum: float | None = 0.0,
    maximum: float | None = None,
) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{path} must be a finite number")
    if minimum is not None and numeric < minimum:
        raise ValueError(f"{path} must be >= {minimum}")
    if maximum is not None and numeric > maximum:
        raise ValueError(f"{path} must be <= {maximum}")
    return value


def _fraction(value: Any, path: str) -> int | float | None:
    if value is None:
        return None
    return _number(value, path, maximum=1.0)


def _status(
    mapping: Mapping[str, Any],
    key: str,
    path: str,
    allowed: set[str],
) -> str:
    value = _safe_text(_required(mapping, key, path), f"{path}.{key}")
    if value not in allowed:
        raise ValueError(f"{path}.{key} has an unreviewed status: {value}")
    return value


def _sha256(value: Any, path: str) -> str:
    text = _safe_text(value, path)
    if not _SHA256_PATTERN.fullmatch(text):
        raise ValueError(f"{path} must be a lowercase SHA-256 digest")
    return text


def _filename(value: Any, path: str) -> str:
    text = _safe_text(value, path)
    if text in {".", ".."} or Path(text).name != text or "/" in text or "\\" in text:
        raise ValueError(f"{path} must be a basename without directories")
    return text


def _close(left: int | float, right: int | float) -> bool:
    return math.isclose(
        float(left),
        float(right),
        rel_tol=1e-10,
        abs_tol=1e-8,
    )


def _require_close(
    actual: int | float,
    expected: int | float,
    path: str,
) -> None:
    if not _close(actual, expected):
        raise ValueError(f"{path} is inconsistent with the source aggregates")


def _validate_ratio(
    actual: int | float | None,
    numerator: int | float,
    denominator: int | float,
    path: str,
) -> None:
    if float(denominator) == 0.0:
        if actual is not None:
            raise ValueError(f"{path} must be null when its denominator is zero")
        return
    if actual is None:
        raise ValueError(f"{path} must be numeric when its denominator is nonzero")
    _require_close(actual, float(numerator) / float(denominator), path)


def _copy_limitations(value: Any, path: str) -> list[str]:
    source = _sequence(value, path)
    if not source:
        raise ValueError(f"{path} must not be empty")
    return [_safe_text(item, f"{path}[{index}]") for index, item in enumerate(source)]


def _vector3(value: Any, path: str) -> tuple[float, float, float]:
    source = _sequence(value, path)
    if len(source) != 3:
        raise ValueError(f"{path} must contain exactly three numbers")
    return tuple(
        float(_number(item, f"{path}[{index}]", minimum=None))
        for index, item in enumerate(source)
    )


def _validate_configuration(value: Any, path: str) -> int:
    source = _mapping(value, path)
    rows = _integer(_required(source, "rows", path), f"{path}.rows", minimum=1)
    _integer(
        _required(source, "chunk_rows", path),
        f"{path}.chunk_rows",
        minimum=2,
    )
    lower = _vector3(
        _required(source, "outer_minimum_m", path),
        f"{path}.outer_minimum_m",
    )
    upper = _vector3(
        _required(source, "outer_maximum_m", path),
        f"{path}.outer_maximum_m",
    )
    if any(low >= high for low, high in zip(lower, upper)):
        raise ValueError(f"{path} outer bounds must satisfy minimum < maximum")
    _vector3(_required(source, "cone_vertex_m", path), f"{path}.cone_vertex_m")
    axis = _vector3(
        _required(source, "cone_axis_normalized", path),
        f"{path}.cone_axis_normalized",
    )
    if not math.isclose(
        math.sqrt(sum(component * component for component in axis)),
        1.0,
        rel_tol=1e-10,
        abs_tol=1e-10,
    ):
        raise ValueError(f"{path}.cone_axis_normalized must have unit length")
    angle = _number(
        _required(source, "cone_angle_degrees", path),
        f"{path}.cone_angle_degrees",
        maximum=180.0,
    )
    if float(angle) <= 0.0:
        raise ValueError(f"{path}.cone_angle_degrees must be > 0")
    policy = _safe_text(_required(source, "policy", path), f"{path}.policy")
    if policy != CONFIGURATION_POLICY:
        raise ValueError(f"{path}.policy changed; review before public export")
    return rows


def _validate_runtime(value: Any, path: str) -> None:
    source = _mapping(value, path)
    _number(_required(source, "wall_seconds", path), f"{path}.wall_seconds")
    scope = _safe_text(_required(source, "scope", path), f"{path}.scope")
    if scope != RUNTIME_SCOPE:
        raise ValueError(f"{path}.scope changed; review before public export")


def _copy_source(value: Any, path: str) -> dict[str, Any]:
    source = _mapping(value, path)
    _sha256(
        _required(source, "view_bundle_manifest_sha256", path),
        f"{path}.view_bundle_manifest_sha256",
    )
    _sha256(
        _required(source, "corrected_mask_manifest_sha256", path),
        f"{path}.corrected_mask_manifest_sha256",
    )
    author_source_modified = _boolean(
        _required(source, "author_source_modified", path),
        f"{path}.author_source_modified",
    )
    if author_source_modified:
        raise ValueError(f"{path}.author_source_modified must remain false")
    return {
        "geometry_source_filename": _filename(
            _required(source, "geometry_source_filename", path),
            f"{path}.geometry_source_filename",
        ),
        "geometry_source_sha256": _sha256(
            _required(source, "geometry_source_sha256", path),
            f"{path}.geometry_source_sha256",
        ),
        "author_source_modified": author_source_modified,
    }


def _copy_upstream_contract(value: Any, path: str) -> dict[str, str]:
    source = _mapping(value, path)
    bundle_status = _safe_text(
        _required(source, "bundle_status", path),
        f"{path}.bundle_status",
    )
    mask_status = _safe_text(
        _required(source, "mask_status", path),
        f"{path}.mask_status",
    )
    if bundle_status != VIEW_BUNDLE_STATUS:
        raise ValueError(f"{path}.bundle_status is not verified")
    if mask_status != MASK_STATUS:
        raise ValueError(f"{path}.mask_status is not verified")
    return {
        "bundle_status": bundle_status,
        "mask_status": mask_status,
    }


def _copy_counts(value: Any, path: str, rows: int) -> dict[str, int]:
    source = _mapping(value, path)
    result = {
        field: _integer(_required(source, field, path), f"{path}.{field}")
        for field in VIEW_COUNT_FIELDS
    }
    bounded_fields = (
        "ray_count",
        "author_nonzero_count",
        "clipped_nonzero_count",
        "changed_from_author_count",
        "cone_shortened_count",
        "cone_zeroed_for_no_box_overlap_count",
        "forward_box_shortened_count",
        "clipped_length_exceeds_author_count",
        "negative_clipped_inner_aperture_radius_count",
        "negative_clipped_outer_aperture_radius_count",
        "clipped_zero_length_count",
    )
    for field in bounded_fields:
        if result[field] > rows:
            raise ValueError(f"{path}.{field} cannot exceed configuration.rows")
    if result["author_nonzero_count"] < result["clipped_nonzero_count"]:
        raise ValueError(f"{path}.clipped_nonzero_count exceeds author count")
    if result["clipped_nonzero_count"] + result["clipped_zero_length_count"] != rows:
        raise ValueError(
            f"{path}.clipped_zero_length_count conflicts with clipped count"
        )
    if (
        result["cone_zeroed_for_no_box_overlap_count"]
        > result["clipped_zero_length_count"]
    ):
        raise ValueError(
            f"{path}.cone_zeroed_for_no_box_overlap_count exceeds zero rows"
        )
    for field in (
        "cone_shortened_count",
        "cone_zeroed_for_no_box_overlap_count",
        "forward_box_shortened_count",
    ):
        if result[field] > result["changed_from_author_count"]:
            raise ValueError(f"{path}.{field} exceeds changed_from_author_count")
    return result


def _copy_path_length(value: Any, path: str) -> dict[str, Any]:
    source = _mapping(value, path)
    result: dict[str, Any] = {
        field: _number(_required(source, field, path), f"{path}.{field}")
        for field in PATH_LENGTH_FIELDS
    }
    result.update(
        {
            field: _fraction(_required(source, field, path), f"{path}.{field}")
            for field in PATH_FRACTION_FIELDS
        }
    )
    author = result["author_length_sum_m"]
    clipped = result["clipped_length_sum_m"]
    removed = result["removed_length_sum_m"]
    if float(clipped) > float(author) and not _close(clipped, author):
        raise ValueError(f"{path}.clipped_length_sum_m exceeds author length")
    _require_close(
        removed, float(author) - float(clipped), f"{path}.removed_length_sum_m"
    )
    _validate_ratio(
        result["retained_fraction"],
        clipped,
        author,
        f"{path}.retained_fraction",
    )
    _validate_ratio(
        result["removed_fraction"],
        removed,
        author,
        f"{path}.removed_fraction",
    )
    return result


def _copy_mask_statistics(
    value: Any,
    path: str,
    *,
    rows: int,
    view_counts: Mapping[str, int],
    path_length: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    source = _mapping(value, path)
    result: dict[str, dict[str, Any]] = {}
    for mask_name in MASK_NAMES:
        mask_path = f"{path}.{mask_name}"
        item = _mapping(_required(source, mask_name, path), mask_path)
        copied: dict[str, Any] = {
            field: _integer(
                _required(item, field, mask_path),
                f"{mask_path}.{field}",
            )
            for field in MASK_COUNT_FIELDS
        }
        copied.update(
            {
                field: _number(
                    _required(item, field, mask_path),
                    f"{mask_path}.{field}",
                )
                for field in MASK_LENGTH_FIELDS
            }
        )
        copied.update(
            {
                field: _fraction(
                    _required(item, field, mask_path),
                    f"{mask_path}.{field}",
                )
                for field in MASK_FRACTION_FIELDS
            }
        )

        count = copied["count"]
        if count > rows:
            raise ValueError(f"{mask_path}.count cannot exceed configuration.rows")
        for field in ("changed_count", "zeroed_count", "shortened_count"):
            if copied[field] > count:
                raise ValueError(f"{mask_path}.{field} cannot exceed count")
        if copied["zeroed_count"] + copied["shortened_count"] > copied["changed_count"]:
            raise ValueError(
                f"{mask_path} zeroed and shortened counts exceed changed_count"
            )
        if copied["changed_count"] > view_counts["changed_from_author_count"]:
            raise ValueError(f"{mask_path}.changed_count exceeds the view total")
        if copied["zeroed_count"] > view_counts["clipped_zero_length_count"]:
            raise ValueError(f"{mask_path}.zeroed_count exceeds the view total")

        author = copied["author_length_sum_m"]
        clipped = copied["clipped_length_sum_m"]
        if float(clipped) > float(author) and not _close(clipped, author):
            raise ValueError(f"{mask_path}.clipped_length_sum_m exceeds author length")
        if float(author) > float(path_length["author_length_sum_m"]) and not _close(
            author, path_length["author_length_sum_m"]
        ):
            raise ValueError(f"{mask_path}.author_length_sum_m exceeds the view total")
        if float(clipped) > float(path_length["clipped_length_sum_m"]) and not _close(
            clipped, path_length["clipped_length_sum_m"]
        ):
            raise ValueError(f"{mask_path}.clipped_length_sum_m exceeds the view total")

        _validate_ratio(
            copied["changed_fraction"],
            copied["changed_count"],
            count,
            f"{mask_path}.changed_fraction",
        )
        _validate_ratio(
            copied["zeroed_fraction"],
            copied["zeroed_count"],
            count,
            f"{mask_path}.zeroed_fraction",
        )
        _validate_ratio(
            copied["shortened_fraction"],
            copied["shortened_count"],
            count,
            f"{mask_path}.shortened_fraction",
        )
        _validate_ratio(
            copied["path_length_retained_fraction"],
            clipped,
            author,
            f"{mask_path}.path_length_retained_fraction",
        )
        result[mask_name] = copied
    return result


def _copy_view_decision(value: Any, path: str) -> dict[str, Any]:
    source = _mapping(value, path)
    result = {
        "positive_segments_inside_forward_box": _boolean(
            _required(source, "positive_segments_inside_forward_box", path),
            f"{path}.positive_segments_inside_forward_box",
        ),
        "geometry_safe_zero_row_filter_required": _boolean(
            _required(
                source,
                "geometry_safe_zero_row_filter_required",
                path,
            ),
            f"{path}.geometry_safe_zero_row_filter_required",
        ),
        "fixed_spatial_domain_established": _boolean(
            _required(source, "fixed_spatial_domain_established", path),
            f"{path}.fixed_spatial_domain_established",
        ),
        "training_ready": _safe_text(
            _required(source, "training_ready", path),
            f"{path}.training_ready",
        ),
        "algorithm_superiority_claim": _safe_text(
            _required(source, "algorithm_superiority_claim", path),
            f"{path}.algorithm_superiority_claim",
        ),
        "next_gate": _safe_text(
            _required(source, "next_gate", path),
            f"{path}.next_gate",
        ),
    }
    if result["fixed_spatial_domain_established"]:
        raise ValueError(f"{path}.fixed_spatial_domain_established must remain false")
    if result["training_ready"] != "NO":
        raise ValueError(f"{path}.training_ready must remain NO")
    if result["algorithm_superiority_claim"] != "LOCKED":
        raise ValueError(f"{path}.algorithm_superiority_claim must remain LOCKED")
    if result["next_gate"] != NEXT_GATE:
        raise ValueError(f"{path}.next_gate changed; review before public export")
    return result


def _copy_view(record: Any, index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    path = f"report.views[{index}]"
    source = _mapping(record, path)
    schema_version = _safe_text(
        _required(source, "schema_version", path),
        f"{path}.schema_version",
    )
    if schema_version != PRIVATE_VIEW_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported {path}.schema_version {schema_version!r}; "
            f"expected {PRIVATE_VIEW_SCHEMA_VERSION!r}"
        )
    status = _status(source, "status", path, VIEW_STATUSES)
    evidence_scope = _safe_text(
        _required(source, "evidence_scope", path),
        f"{path}.evidence_scope",
    )
    if evidence_scope != EVIDENCE_SCOPE:
        raise ValueError(f"{path}.evidence_scope changed; review before public export")

    view_id = _integer(
        _required(source, "view_id_zero_based", path),
        f"{path}.view_id_zero_based",
    )
    provenance = _copy_source(_required(source, "source", path), f"{path}.source")
    rows = _validate_configuration(
        _required(source, "configuration", path),
        f"{path}.configuration",
    )
    counts = _copy_counts(_required(source, "counts", path), f"{path}.counts", rows)
    path_length = _copy_path_length(
        _required(source, "path_length", path),
        f"{path}.path_length",
    )
    mask_conditioned = _copy_mask_statistics(
        _required(source, "mask_conditioned", path),
        f"{path}.mask_conditioned",
        rows=rows,
        view_counts=counts,
        path_length=path_length,
    )
    _validate_runtime(
        _required(source, "runtime_observation", path),
        f"{path}.runtime_observation",
    )
    decision = _copy_view_decision(
        _required(source, "decision", path),
        f"{path}.decision",
    )
    limitations = _copy_limitations(
        _required(source, "limitations", path),
        f"{path}.limitations",
    )
    upstream_contract = _copy_upstream_contract(
        _required(source, "upstream_view_contract", path),
        f"{path}.upstream_view_contract",
    )

    mechanically_valid = counts["ray_count"] == rows and all(
        counts[field] == 0 for field in VIEW_SAFETY_FAILURE_FIELDS
    )
    filter_required = counts["clipped_zero_length_count"] > 0
    expected_status = (
        "AUTHOR_COMPATIBLE_CLIPPED_HYBRID_INVALID"
        if not mechanically_valid
        else "AUTHOR_COMPATIBLE_CLIPPED_HYBRID_CONTRACT_PASS_MASK_FILTER_REQUIRED"
        if filter_required
        else "AUTHOR_COMPATIBLE_CLIPPED_HYBRID_CONTRACT_PASS"
    )
    if status != expected_status:
        raise ValueError(f"{path}.status conflicts with counts and configuration")
    if decision["positive_segments_inside_forward_box"] != mechanically_valid:
        raise ValueError(
            f"{path}.decision.positive_segments_inside_forward_box "
            "conflicts with safety counts"
        )
    if decision["geometry_safe_zero_row_filter_required"] != filter_required:
        raise ValueError(
            f"{path}.decision.geometry_safe_zero_row_filter_required "
            "conflicts with clipped_zero_length_count"
        )

    public_view = {
        "view_id_zero_based": view_id,
        "status": status,
        "upstream_view_contract": upstream_contract,
        "counts": counts,
        "path_length": path_length,
        "mask_conditioned": mask_conditioned,
        "decision": decision,
        "limitations": limitations,
    }
    return public_view, provenance


def _copy_private_id_list(
    value: Any,
    path: str,
    view_ids: set[int],
) -> list[int]:
    source = _sequence(value, path)
    result = [_integer(item, f"{path}[{index}]") for index, item in enumerate(source)]
    if result != sorted(set(result)):
        raise ValueError(f"{path} must contain unique sorted view ids")
    if not set(result) <= view_ids:
        raise ValueError(f"{path} contains an unknown view id")
    return result


def _copy_aggregate(
    value: Any,
    views: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    path = "report.aggregate"
    source = _mapping(value, path)
    view_ids = {int(view["view_id_zero_based"]) for view in views}
    invalid_ids = _copy_private_id_list(
        _required(source, "invalid_view_ids", path),
        f"{path}.invalid_view_ids",
        view_ids,
    )
    filter_ids = _copy_private_id_list(
        _required(source, "zero_row_filter_required_view_ids", path),
        f"{path}.zero_row_filter_required_view_ids",
        view_ids,
    )
    expected_invalid_ids = [
        int(view["view_id_zero_based"])
        for view in views
        if view["status"] == "AUTHOR_COMPATIBLE_CLIPPED_HYBRID_INVALID"
    ]
    expected_filter_ids = [
        int(view["view_id_zero_based"])
        for view in views
        if view["decision"]["geometry_safe_zero_row_filter_required"]
    ]
    if invalid_ids != expected_invalid_ids:
        raise ValueError(f"{path}.invalid_view_ids conflicts with per-view status")
    if filter_ids != expected_filter_ids:
        raise ValueError(
            f"{path}.zero_row_filter_required_view_ids "
            "conflicts with per-view decisions"
        )

    result: dict[str, Any] = {
        "invalid_view_count": len(invalid_ids),
        "zero_row_filter_required_view_count": len(filter_ids),
    }
    result.update(
        {
            field: _number(_required(source, field, path), f"{path}.{field}")
            for field in PATH_LENGTH_FIELDS
        }
    )
    result.update(
        {
            field: _fraction(_required(source, field, path), f"{path}.{field}")
            for field in PATH_FRACTION_FIELDS
        }
    )
    result.update(
        {
            field: _integer(_required(source, field, path), f"{path}.{field}")
            for field in AGGREGATE_COUNT_FIELDS
        }
    )

    expected_author = sum(
        float(view["path_length"]["author_length_sum_m"]) for view in views
    )
    expected_clipped = sum(
        float(view["path_length"]["clipped_length_sum_m"]) for view in views
    )
    expected_removed = expected_author - expected_clipped
    _require_close(
        result["author_length_sum_m"],
        expected_author,
        f"{path}.author_length_sum_m",
    )
    _require_close(
        result["clipped_length_sum_m"],
        expected_clipped,
        f"{path}.clipped_length_sum_m",
    )
    _require_close(
        result["removed_length_sum_m"],
        expected_removed,
        f"{path}.removed_length_sum_m",
    )
    _validate_ratio(
        result["retained_fraction"],
        result["clipped_length_sum_m"],
        result["author_length_sum_m"],
        f"{path}.retained_fraction",
    )
    _validate_ratio(
        result["removed_fraction"],
        result["removed_length_sum_m"],
        result["author_length_sum_m"],
        f"{path}.removed_fraction",
    )
    expected_counts = {
        "changed_ray_count": sum(
            int(view["counts"]["changed_from_author_count"]) for view in views
        ),
        "clipped_zero_length_count": sum(
            int(view["counts"]["clipped_zero_length_count"]) for view in views
        ),
    }
    for field, expected in expected_counts.items():
        if result[field] != expected:
            raise ValueError(f"{path}.{field} conflicts with per-view counts")
    return result


def _copy_report_decision(value: Any) -> dict[str, Any]:
    path = "report.decision"
    source = _mapping(value, path)
    result = {
        "domain_clipping_mechanically_valid": _boolean(
            _required(source, "domain_clipping_mechanically_valid", path),
            f"{path}.domain_clipping_mechanically_valid",
        ),
        "fixed_spatial_domain_established": _boolean(
            _required(source, "fixed_spatial_domain_established", path),
            f"{path}.fixed_spatial_domain_established",
        ),
        "training_ready": _safe_text(
            _required(source, "training_ready", path),
            f"{path}.training_ready",
        ),
        "algorithm_superiority_claim": _safe_text(
            _required(source, "algorithm_superiority_claim", path),
            f"{path}.algorithm_superiority_claim",
        ),
        "next_gate": _safe_text(
            _required(source, "next_gate", path),
            f"{path}.next_gate",
        ),
    }
    if result["fixed_spatial_domain_established"]:
        raise ValueError(f"{path}.fixed_spatial_domain_established must remain false")
    if result["training_ready"] != "NO":
        raise ValueError(f"{path}.training_ready must remain NO")
    if result["algorithm_superiority_claim"] != "LOCKED":
        raise ValueError(f"{path}.algorithm_superiority_claim must remain LOCKED")
    if result["next_gate"] != NEXT_GATE:
        raise ValueError(f"{path}.next_gate changed; review before public export")
    return result


def _validate_report_consistency(report: Mapping[str, Any]) -> None:
    invalid_count = int(report["aggregate"]["invalid_view_count"])
    filter_count = int(report["aggregate"]["zero_row_filter_required_view_count"])
    expected_status = (
        "AUTHOR_COMPATIBLE_CLIPPED_HYBRID_ALL_VIEW_INVALID"
        if invalid_count
        else "AUTHOR_COMPATIBLE_CLIPPED_HYBRID_ALL_VIEW_CONTRACT_PASS_MASK_FILTER_REQUIRED"
        if filter_count
        else "AUTHOR_COMPATIBLE_CLIPPED_HYBRID_ALL_VIEW_CONTRACT_PASS"
    )
    if report["status"] != expected_status:
        raise ValueError("report.status conflicts with per-view statuses")
    expected_verdict = (
        "INVALID" if invalid_count else "AUTHOR_COMPATIBILITY_ABLATION_ONLY"
    )
    if report["scientific_verdict"] != expected_verdict:
        raise ValueError("report.scientific_verdict conflicts with per-view statuses")
    if report["decision"]["domain_clipping_mechanically_valid"] != (invalid_count == 0):
        raise ValueError(
            "report.decision.domain_clipping_mechanically_valid "
            "conflicts with per-view statuses"
        )


def _validate_public_payload(value: Any, path: str = "public_summary") -> None:
    """Defend the allowlist against unsafe future additions."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string key")
            _validate_public_payload(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_public_payload(item, f"{path}[{index}]")
    elif isinstance(value, str):
        _safe_text(value, path)
    elif value is not None and not isinstance(value, (bool, int, float)):
        raise ValueError(f"{path} contains a non-JSON value")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains a non-finite number")


def build_public_summary(private_report: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the private audit and rebuild it from the public allowlist."""
    source = _mapping(private_report, "report")
    schema_version = _safe_text(
        _required(source, "schema_version", "report"),
        "report.schema_version",
    )
    if schema_version != PRIVATE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported private schema_version {schema_version!r}; "
            f"expected {PRIVATE_SCHEMA_VERSION!r}"
        )
    execution_status = _safe_text(
        _required(source, "execution_status", "report"),
        "report.execution_status",
    )
    if execution_status != EXECUTION_STATUS:
        raise ValueError("report.execution_status must be COMPLETE")
    scientific_verdict = _status(
        source,
        "scientific_verdict",
        "report",
        SCIENTIFIC_VERDICTS,
    )
    status = _status(source, "status", "report", REPORT_STATUSES)
    view_count = _integer(
        _required(source, "view_count", "report"),
        "report.view_count",
        minimum=1,
    )
    raw_views = _sequence(_required(source, "views", "report"), "report.views")
    if len(raw_views) != view_count:
        raise ValueError("report.view_count must equal the number of views")

    views: list[dict[str, Any]] = []
    provenance_records: list[dict[str, Any]] = []
    for index, item in enumerate(raw_views):
        view, provenance = _copy_view(item, index)
        views.append(view)
        provenance_records.append(provenance)
    view_ids = [int(view["view_id_zero_based"]) for view in views]
    if view_ids != list(range(view_count)):
        raise ValueError("view ids must be the ordered contiguous range from zero")
    provenance = provenance_records[0]
    if any(item != provenance for item in provenance_records[1:]):
        raise ValueError("geometry source provenance must match across all views")

    report = {
        "schema_version": PUBLIC_SCHEMA_VERSION,
        "source_schema_version": PRIVATE_SCHEMA_VERSION,
        "source_view_schema_version": PRIVATE_VIEW_SCHEMA_VERSION,
        "execution_status": execution_status,
        "scientific_verdict": scientific_verdict,
        "status": status,
        "evidence_scope": EVIDENCE_SCOPE,
        "view_count": view_count,
        "views": views,
        "aggregate": _copy_aggregate(
            _required(source, "aggregate", "report"),
            views,
        ),
        "decision": _copy_report_decision(_required(source, "decision", "report")),
        "limitations": _copy_limitations(
            _required(source, "limitations", "report"),
            "report.limitations",
        ),
        "provenance": provenance,
        "claim_boundary": {
            "supported_claim": "A1_AUTHOR_COMPATIBILITY_ABLATION_ONLY",
            "fixed_spatial_domain_established": False,
            "reconstruction_established": False,
            "algorithm_superiority_established": False,
        },
        "public_export_policy": {
            "strict_field_allowlist": True,
            "aggregate_and_per_view_summary_only": True,
            "contains_private_or_local_paths": False,
            "contains_local_timing": False,
            "contains_source_code_snippets": False,
            "contains_raw_arrays": False,
            "contains_raw_index_lists": False,
            "contains_per_artifact_private_hashes": False,
        },
    }
    _validate_report_consistency(report)
    _validate_public_payload(report)
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
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
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
        "--clipped-hybrid-audit",
        dest="input_path",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_public_summary(args.input_path, args.output)
    print(f"wrote public clipped-hybrid summary: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
