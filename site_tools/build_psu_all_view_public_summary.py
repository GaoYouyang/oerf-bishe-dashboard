#!/usr/bin/env python3
"""Export a strict aggregate-only public summary from a private PSU all-view audit."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


PRIVATE_SCHEMA_VERSION = "psu-bost-all-view-geometry-audit-1.0"
PUBLIC_SCHEMA_VERSION = "psu-bost-all-view-geometry-public-summary-1.1"
EVIDENCE_SCOPE = (
    "PER_VIEW_REAL_NUMERIC_STREAMS_MASK_BASE_CORRECTION_AND_AUTHOR_GEOMETRY_"
    "NO_TENSORFLOW_NO_NIRT"
)

AUDIT_STATUSES = {
    "ALL_VIEW_GEOMETRY_AUDIT_NO_GO",
    "ALL_VIEW_GEOMETRY_MECHANICAL_CONTRACT_PASS",
}
BUNDLE_STATUSES = {
    "VIEW_SHARD_BUNDLE_TRANSCODED_AND_SOURCE_STREAMS_VERIFIED",
}
MASK_STATUSES = {
    "CORRECTED_VIEW_MASK_SHARDS_BUILT_MECHANICAL_CONTRACT_PASS",
}
SETUP_PASS_STATUS = "STREAMED_SETUP_MECHANICAL_CONTRACT_PASS"
SETUP_STATUSES = {
    SETUP_PASS_STATUS,
    "STREAMED_SETUP_DIAGNOSTIC_NO_GO",
}

VIEW_STATUS_FIELDS = ("bundle_status", "mask_status", "setup_status")
VIEW_COUNT_FIELDS = (
    "measurement_count",
    "full_box_zero_count",
    "box_miss_but_cone_nonzero_count",
    "final_zero_length_count",
    "cone_outside_ray_count",
    "cone_no_box_overlap_count",
    "active_count",
    "active_unsafe_geometry_count",
    "inactive_count",
    "inactive_unsafe_geometry_count",
)
VIEW_FRACTION_FIELDS = (
    "full_box_zero_fraction",
    "box_miss_but_cone_nonzero_fraction",
    "final_zero_length_fraction",
    "cone_outside_ray_fraction",
    "cone_length_weighted_outside_box_fraction",
    "active_unsafe_geometry_fraction",
    "inactive_unsafe_geometry_fraction",
)
VIEW_NONNEGATIVE_METRIC_FIELDS = (
    "active_rms_magnitude_pixels",
    "inactive_rms_magnitude_pixels",
    "active_to_inactive_rms_ratio",
    "active_shift_vector_rmse_pixels",
    "inactive_shift_vector_rmse_pixels",
)
SUMMARY_METRICS = (
    "full_box_zero_fraction",
    "box_miss_but_cone_nonzero_fraction",
    "final_zero_length_fraction",
    "cone_outside_ray_fraction",
    "cone_length_weighted_outside_box_fraction",
    "active_unsafe_geometry_fraction",
    "inactive_unsafe_geometry_fraction",
    "active_rms_magnitude_pixels",
    "inactive_rms_magnitude_pixels",
    "active_to_inactive_rms_ratio",
)
SUMMARY_VALUE_FIELDS = ("minimum", "mean", "maximum")
SUMMARY_VIEW_FIELDS = ("minimum_view_id", "maximum_view_id")
PREVALENCE_COUNT_FIELDS = (
    "views_with_full_box_zero_rays",
    "views_with_cone_outside_box_rays",
    "views_with_active_unsafe_geometry",
    "views_with_inactive_unsafe_geometry",
)
POOLED_COUNT_FIELDS = (
    "ray_count",
    "full_box_zero_count",
    "box_miss_but_cone_nonzero_count",
    "final_zero_length_count",
)
POOLED_LENGTH_FIELDS = (
    "cone_segment_length_sum_m",
    "cone_box_overlap_length_sum_m",
    "cone_outside_length_sum_m",
)
DECISION_FIELDS = (
    "geometry_problem_is_single_view_artifact",
    "official_setup_ready_for_training",
    "algorithm_success_claim",
    "next_gate",
)

_PRIVATE_PATH_PATTERNS = (
    re.compile(r"(?:^|[\s\"'(])/(?:Users|home|private|tmp|Volumes|var/folders)/"),
    re.compile(r"\b[A-Za-z]:[\\/]"),
    re.compile(r"(?i)\b(?:file://|private_library(?:[/\\]|$)|localhost(?::\d+)?|127\.0\.0\.1)"),
)
_SOURCE_SNIPPET_PATTERNS = (
    re.compile(r"\bdef\s+[A-Za-z_]\w*\s*\("),
    re.compile(r"\bclass\s+[A-Za-z_]\w*\s*[:(]"),
    re.compile(r"\bfrom\s+\S+\s+import\s+"),
    re.compile(r"(?:^|[;\s])import\s+[A-Za-z_]\w*"),
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


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
    if "\n" in value or "\r" in value or "\t" in value:
        raise ValueError(f"{path} must be single-line text")
    if any(pattern.search(value) for pattern in _PRIVATE_PATH_PATTERNS):
        raise ValueError(f"{path} contains a private or local path")
    if any(pattern.search(value) for pattern in _SOURCE_SNIPPET_PATTERNS):
        raise ValueError(f"{path} contains a source-code snippet")
    return value


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{path} must be an integer >= {minimum}")
    return value


def _number(
    value: Any,
    path: str,
    *,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < minimum:
        raise ValueError(f"{path} must be a finite number >= {minimum}")
    if maximum is not None and numeric > maximum:
        raise ValueError(f"{path} must be <= {maximum}")
    return value


def _status(
    mapping: Mapping[str, Any], key: str, path: str, allowed: set[str]
) -> str:
    value = _safe_text(_required(mapping, key, path), f"{path}.{key}")
    if value not in allowed:
        raise ValueError(f"{path}.{key} has an unreviewed status: {value}")
    return value


def _copy_view(record: Any, index: int) -> dict[str, Any]:
    path = f"views[{index}]"
    source = _mapping(record, path)
    view_id = _integer(
        _required(source, "view_id_zero_based", path),
        f"{path}.view_id_zero_based",
    )
    result: dict[str, Any] = {"view_id_zero_based": view_id}
    result["bundle_status"] = _status(source, "bundle_status", path, BUNDLE_STATUSES)
    result["mask_status"] = _status(source, "mask_status", path, MASK_STATUSES)
    result["setup_status"] = _status(source, "setup_status", path, SETUP_STATUSES)
    for field in VIEW_COUNT_FIELDS:
        minimum = 1 if field == "measurement_count" else 0
        result[field] = _integer(
            _required(source, field, path), f"{path}.{field}", minimum=minimum
        )
    for field in VIEW_FRACTION_FIELDS:
        result[field] = _number(
            _required(source, field, path),
            f"{path}.{field}",
            maximum=1.0,
        )
    for field in VIEW_NONNEGATIVE_METRIC_FIELDS:
        result[field] = _number(_required(source, field, path), f"{path}.{field}")
    return result


def _copy_metric_summary(value: Any, view_ids: set[int]) -> dict[str, Any]:
    source = _mapping(value, "metric_summary")
    result: dict[str, Any] = {}
    for metric in SUMMARY_METRICS:
        path = f"metric_summary.{metric}"
        item = _mapping(_required(source, metric, "metric_summary"), path)
        maximum = 1.0 if metric in VIEW_FRACTION_FIELDS else None
        copied = {
            field: _number(
                _required(item, field, path),
                f"{path}.{field}",
                maximum=maximum,
            )
            for field in SUMMARY_VALUE_FIELDS
        }
        for field in SUMMARY_VIEW_FIELDS:
            view_id = _integer(_required(item, field, path), f"{path}.{field}")
            if view_id not in view_ids:
                raise ValueError(f"{path}.{field} does not identify an exported view")
            copied[field] = view_id
        if not copied["minimum"] <= copied["mean"] <= copied["maximum"]:
            raise ValueError(f"{path} must satisfy minimum <= mean <= maximum")
        result[metric] = copied
    return result


def _copy_prevalence(
    value: Any, views: Sequence[Mapping[str, Any]], view_ids: set[int]
) -> dict[str, int]:
    source = _mapping(value, "prevalence")
    result = {
        field: _integer(
            _required(source, field, "prevalence"), f"prevalence.{field}"
        )
        for field in PREVALENCE_COUNT_FIELDS
    }
    for field, count in result.items():
        if count > len(views):
            raise ValueError(f"prevalence.{field} cannot exceed view_count")

    raw_no_go_ids = _sequence(
        _required(source, "setup_no_go_view_ids", "prevalence"),
        "prevalence.setup_no_go_view_ids",
    )
    no_go_ids = [
        _integer(item, f"prevalence.setup_no_go_view_ids[{index}]")
        for index, item in enumerate(raw_no_go_ids)
    ]
    if len(no_go_ids) != len(set(no_go_ids)) or not set(no_go_ids) <= view_ids:
        raise ValueError("prevalence.setup_no_go_view_ids must be unique exported views")
    derived_no_go_ids = {
        int(view["view_id_zero_based"])
        for view in views
        if view["setup_status"] != SETUP_PASS_STATUS
    }
    if set(no_go_ids) != derived_no_go_ids:
        raise ValueError("prevalence.setup_no_go_view_ids conflicts with view statuses")
    result["setup_no_go_view_count"] = len(no_go_ids)
    return result


def _copy_pooled_geometry(value: Any, views: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    source = _mapping(value, "pooled_geometry")
    result = {
        field: _integer(
            _required(source, field, "pooled_geometry"), f"pooled_geometry.{field}"
        )
        for field in POOLED_COUNT_FIELDS
    }
    result.update(
        {
            field: _number(
                _required(source, field, "pooled_geometry"),
                f"pooled_geometry.{field}",
            )
            for field in POOLED_LENGTH_FIELDS
        }
    )
    result["cone_length_weighted_outside_box_fraction"] = _number(
        _required(
            source,
            "cone_length_weighted_outside_box_fraction",
            "pooled_geometry",
        ),
        "pooled_geometry.cone_length_weighted_outside_box_fraction",
        maximum=1.0,
    )
    expected_counts = {
        "ray_count": sum(int(view["measurement_count"]) for view in views),
        "full_box_zero_count": sum(int(view["full_box_zero_count"]) for view in views),
        "box_miss_but_cone_nonzero_count": sum(
            int(view["box_miss_but_cone_nonzero_count"]) for view in views
        ),
        "final_zero_length_count": sum(
            int(view["final_zero_length_count"]) for view in views
        ),
    }
    for field, expected in expected_counts.items():
        if result[field] != expected:
            raise ValueError(f"pooled_geometry.{field} conflicts with per-view data")
    segment = float(result["cone_segment_length_sum_m"])
    overlap = float(result["cone_box_overlap_length_sum_m"])
    outside = float(result["cone_outside_length_sum_m"])
    if overlap > segment or not math.isclose(
        outside, segment - overlap, rel_tol=1e-10, abs_tol=1e-9
    ):
        raise ValueError("pooled_geometry path-length fields are inconsistent")
    expected_fraction = 1.0 - overlap / segment if segment else 0.0
    if not math.isclose(
        float(result["cone_length_weighted_outside_box_fraction"]),
        expected_fraction,
        rel_tol=1e-10,
        abs_tol=1e-12,
    ):
        raise ValueError("pooled_geometry outside-box fraction is inconsistent")
    return result


def _copy_decision(value: Any) -> dict[str, str]:
    source = _mapping(value, "decision")
    return {
        field: _safe_text(
            _required(source, field, "decision"), f"decision.{field}"
        )
        for field in DECISION_FIELDS
    }


def _copy_limitations(value: Any) -> list[str]:
    source = _sequence(value, "limitations")
    if not source:
        raise ValueError("limitations must not be empty")
    return [
        _safe_text(item, f"limitations[{index}]")
        for index, item in enumerate(source)
    ]


def _sha256(value: Any, path: str) -> str:
    text = _safe_text(value, path)
    if not _SHA256_PATTERN.fullmatch(text):
        raise ValueError(f"{path} must be a lowercase SHA-256 digest")
    return text


def _filename(value: Any, path: str) -> str:
    text = _safe_text(value, path)
    if Path(text).name != text or "/" in text or "\\" in text:
        raise ValueError(f"{path} must be a basename without directories")
    return text


def _copy_provenance(source: Mapping[str, Any]) -> dict[str, str]:
    private_source = _mapping(
        _required(source, "source", "report"), "report.source"
    )
    run_contract = _mapping(
        _required(source, "run_contract", "report"), "report.run_contract"
    )
    code_provenance = _mapping(
        _required(run_contract, "code_provenance", "report.run_contract"),
        "report.run_contract.code_provenance",
    )
    binding = _safe_text(
        _required(
            code_provenance,
            "artifact_generator_source_binding_at_initial_generation",
            "report.run_contract.code_provenance",
        ),
        "report.run_contract.code_provenance."
        "artifact_generator_source_binding_at_initial_generation",
    )
    if binding != "NOT_RECORDED_NUMERIC_ARTIFACTS_REVALIDATED_BY_FILE_HASH":
        raise ValueError("unreviewed artifact-generator source-binding status")
    return {
        "mat_filename": _filename(
            _required(private_source, "mat_filename", "report.source"),
            "report.source.mat_filename",
        ),
        "mat_sha256": _sha256(
            _required(private_source, "mat_sha256", "report.source"),
            "report.source.mat_sha256",
        ),
        "geometry_source_filename": _filename(
            _required(
                private_source, "geometry_source_filename", "report.source"
            ),
            "report.source.geometry_source_filename",
        ),
        "geometry_source_sha256": _sha256(
            _required(private_source, "geometry_source_sha256", "report.source"),
            "report.source.geometry_source_sha256",
        ),
        "run_contract_sha256": _sha256(
            _required(source, "run_contract_sha256", "report"),
            "report.run_contract_sha256",
        ),
        "artifact_generator_source_binding_at_initial_generation": binding,
    }


def _validate_consistency(report: Mapping[str, Any]) -> None:
    views = report["views"]
    no_go_count = report["prevalence"]["setup_no_go_view_count"]
    expected_status = (
        "ALL_VIEW_GEOMETRY_AUDIT_NO_GO"
        if no_go_count
        else "ALL_VIEW_GEOMETRY_MECHANICAL_CONTRACT_PASS"
    )
    if report["status"] != expected_status:
        raise ValueError("status conflicts with per-view setup statuses")

    expected_prevalence = {
        "views_with_full_box_zero_rays": sum(
            view["full_box_zero_count"] > 0 for view in views
        ),
        "views_with_cone_outside_box_rays": sum(
            view["cone_outside_ray_count"] > 0 for view in views
        ),
        "views_with_active_unsafe_geometry": sum(
            view["active_unsafe_geometry_count"] > 0 for view in views
        ),
        "views_with_inactive_unsafe_geometry": sum(
            view["inactive_unsafe_geometry_count"] > 0 for view in views
        ),
    }
    for field, expected in expected_prevalence.items():
        if report["prevalence"][field] != expected:
            raise ValueError(f"prevalence.{field} conflicts with per-view counts")

    decision = report["decision"]
    if decision["algorithm_success_claim"] != "LOCKED":
        raise ValueError("decision.algorithm_success_claim must remain LOCKED")
    expected_training = "NO_GO" if no_go_count else "MECHANICAL_ONLY"
    if decision["official_setup_ready_for_training"] != expected_training:
        raise ValueError(
            "decision.official_setup_ready_for_training conflicts with audit status"
        )


def _validate_public_payload(value: Any, path: str = "public_summary") -> None:
    """Reject unsafe text even if a future edit accidentally expands an allowlist."""
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
    """Validate a private report and rebuild it from the public field allowlist."""
    source = _mapping(private_report, "report")
    schema_version = _safe_text(
        _required(source, "schema_version", "report"), "report.schema_version"
    )
    if schema_version != PRIVATE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported private schema_version {schema_version!r}; "
            f"expected {PRIVATE_SCHEMA_VERSION!r}"
        )
    status = _status(source, "status", "report", AUDIT_STATUSES)
    execution_status = _safe_text(
        _required(source, "execution_status", "report"), "report.execution_status"
    )
    if execution_status != "COMPLETE":
        raise ValueError("report.execution_status must be COMPLETE")
    scientific_verdict = _safe_text(
        _required(source, "scientific_verdict", "report"),
        "report.scientific_verdict",
    )
    expected_verdict = (
        "NO_GO"
        if status == "ALL_VIEW_GEOMETRY_AUDIT_NO_GO"
        else "MECHANICAL_CONTRACT_PASS"
    )
    if scientific_verdict != expected_verdict:
        raise ValueError("scientific_verdict conflicts with status")
    evidence_scope = _safe_text(
        _required(source, "evidence_scope", "report"), "report.evidence_scope"
    )
    if evidence_scope != EVIDENCE_SCOPE:
        raise ValueError("evidence_scope changed; review before public export")

    view_count = _integer(
        _required(source, "view_count", "report"), "report.view_count", minimum=1
    )
    raw_views = _sequence(_required(source, "views", "report"), "report.views")
    if len(raw_views) != view_count:
        raise ValueError("report.view_count must equal the number of views")
    views = [_copy_view(item, index) for index, item in enumerate(raw_views)]
    view_ids = {int(view["view_id_zero_based"]) for view in views}
    if len(view_ids) != view_count or view_ids != set(range(view_count)):
        raise ValueError("view ids must be unique and contiguous from zero")

    report = {
        "schema_version": PUBLIC_SCHEMA_VERSION,
        "source_schema_version": PRIVATE_SCHEMA_VERSION,
        "execution_status": execution_status,
        "scientific_verdict": scientific_verdict,
        "status": status,
        "evidence_scope": evidence_scope,
        "view_count": view_count,
        "views": views,
        "metric_summary": _copy_metric_summary(
            _required(source, "metric_summary", "report"), view_ids
        ),
        "pooled_geometry": _copy_pooled_geometry(
            _required(source, "pooled_geometry", "report"), views
        ),
        "prevalence": _copy_prevalence(
            _required(source, "prevalence", "report"), views, view_ids
        ),
        "decision": _copy_decision(_required(source, "decision", "report")),
        "limitations": _copy_limitations(
            _required(source, "limitations", "report")
        ),
        "provenance": _copy_provenance(source),
        "claim_boundary": {
            "supported_claim": (
                "REAL_NUMERIC_PER_VIEW_AND_CROSS_VIEW_AGGREGATE_GEOMETRY_"
                "DIAGNOSTICS_ONLY"
            ),
            "official_nirt_reconstruction_executed": False,
            "corrected_forward_model_validated": False,
            "held_out_reprojection_validated": False,
            "three_dimensional_ground_truth_validated": False,
            "training_readiness_established": False,
            "algorithm_superiority_established": False,
            "runtime_or_speed_comparison_established": False,
        },
        "public_export_policy": {
            "aggregate_only": True,
            "contains_private_or_local_paths": False,
            "contains_raw_array_values": False,
            "contains_raw_index_lists": False,
            "contains_source_code_snippets": False,
            "contains_local_timing": False,
        },
    }
    _validate_consistency(report)
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
        "--all-view-audit",
        dest="input_path",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_public_summary(args.input_path, args.output)
    print(f"wrote public aggregate summary: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
