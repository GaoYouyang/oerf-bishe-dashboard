#!/usr/bin/env python3
"""Build the publication figure for the PSU B0/B1 fixed-domain audit."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402


REPORT_SCHEMA = "psu-bost-fixed-domain-all-view-audit-1.0"
VIEW_SCHEMA = "psu-bost-fixed-domain-geometry-audit-1.0"
MANIFEST_SCHEMA = "psu-bost-fixed-domain-geometry-figure-1.0"
ROOT_STATUS = "B0_B1_ALL_VIEW_ANALYTIC_CONTRACT_PASS_B2_REQUIRED"
VIEW_STATUS = "B0_B1_FIXED_DOMAIN_ANALYTIC_CONTRACT_PASS_B2_REQUIRED"
SCIENTIFIC_VERDICT = (
    "MECHANICAL_PASS_PHYSICAL_CONE_SEMANTICS_AND_FINITE_APERTURE_UNCONFIRMED"
)
EVIDENCE_SCOPE = (
    "REAL_ONE_VIEW_B0_FORWARD_BOX_AND_B1_ONE_NAPPE_CONE_BOX_RAY_CENSUS_"
    "NO_TENSORFLOW_NO_RECONSTRUCTION"
)
B0_POLICY = "FORWARD_NORMALIZED_RAY_INTERSECT_CLOSED_AXIS_ALIGNED_BOX"
B1_POLICY = "B0_INTERSECT_NORMALIZED_ONE_NAPPE_CONE_NO_FALLBACK"
GEOMETRY_CONTRACT = "psu-bost-forward-geometry-1.0"
NEXT_GATE = "B2_FINITE_APERTURE_DOMAIN_INDICATOR_AND_B3_GEOMETRY_SAFE_MASK"
DEFAULT_OUTPUT_STEM = "psu_fixed_domain_geometry_audit_figure"
FIGURE_SIZE_INCHES = (11.0, 7.4)
PNG_DPI = 300
CAPTION = (
    "Deterministic centerline geometry census; no error bars. "
    "Not a reconstruction result and not physical validation of the released "
    "25-degree cone."
)
PANEL_METRICS = {
    "A": ("b0_hit_fraction", "b1_hit_fraction"),
    "B": ("b1_path_fraction_of_b0",),
    "C": ("active_b1_hit_fraction", "inactive_b1_hit_fraction"),
    "D": (
        "pooled_b0_hit_fraction_of_all_rays",
        "pooled_b1_hit_fraction_of_all_rays",
        "pooled_b1_removed_fraction_of_b0_hits",
    ),
}

ROOT_LIMITATIONS = (
    "B0/B1 are deterministic geometry baselines and do not measure "
    "reconstruction quality",
    "the released 25 degree cone is treated as a computational sampling hull, "
    "not a shock or Mach angle",
    "B2 finite-aperture support and held-out reprojection are required before "
    "inverse or neural-operator comparison",
    "no statistical uncertainty interval applies to this exhaustive ray census",
)
VIEW_LIMITATIONS = (
    "B1 reuses the released cone vertex, axis, and angle only as a "
    "computational-domain hypothesis; their physical meaning is not "
    "independently confirmed",
    "centerline domain validity does not prove that finite-aperture samples "
    "stay inside the same domain",
    "the active and inactive masks are diagnostic labels rather than density "
    "or refractive-index ground truth",
    "no held-out camera, TensorFlow NIRT, neural field, inverse reconstruction, "
    "or superiority comparison is run",
)

_COUNT_FIELDS = (
    "author_full_box_zero_flag_count",
    "author_selected_nonzero_count",
    "b0_endpoint_box_violation_count",
    "b0_hit_count",
    "b0_point_touch_count",
    "b0_zero_length_count",
    "b1_endpoint_box_violation_count",
    "b1_endpoint_nappe_violation_count",
    "b1_endpoint_cone_radial_violation_count",
    "b1_midpoint_cone_violation_count",
    "b1_hit_count",
    "b1_hit_without_b0_hit_count",
    "b1_length_exceeds_b0_count",
    "b1_nappe_rejected_double_cone_count",
    "b1_nappe_rejected_with_b0_hit_count",
    "b1_point_touch_count",
    "b1_removed_from_b0_count",
    "b1_zero_length_count",
    "nonfinite_output_count",
    "ray_count",
)
_ZERO_CONTRACT_COUNT_FIELDS = (
    "b0_endpoint_box_violation_count",
    "b1_endpoint_box_violation_count",
    "b1_endpoint_nappe_violation_count",
    "b1_endpoint_cone_radial_violation_count",
    "b1_midpoint_cone_violation_count",
    "b1_hit_without_b0_hit_count",
    "b1_length_exceeds_b0_count",
    "nonfinite_output_count",
)
_MASK_NAMES = ("amask_all", "imask_all")
_OUTPUT_STEM_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

# Okabe-Ito colors plus a neutral publication band.
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#5F6368",
    "band": "#ECEFF1",
    "ink": "#202124",
    "grid": "#D9DDE1",
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read report JSON {path}: {exc}") from exc
    try:
        report = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"report JSON is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(report, dict):
        raise ValueError("report JSON schema error: root must be an object")
    return report, payload


def _required(source: Mapping[str, Any], key: str, location: str) -> Any:
    if key not in source:
        raise ValueError(f"{location} missing required field: {key}")
    return source[key]


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be an object")
    return value


def _array(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{location} must be an array")
    return value


def _integer(
    value: Any,
    location: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{location} must be an integer")
    if value < minimum:
        raise ValueError(f"{location} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{location} must be <= {maximum}")
    return value


def _number(
    value: Any,
    location: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{location} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{location} must be a finite number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{location} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{location} must be <= {maximum}")
    return result


def _fraction(value: Any, location: str) -> float:
    return _number(value, location, minimum=0.0, maximum=1.0)


def _expect(value: Any, expected: Any, location: str) -> None:
    if value != expected:
        raise ValueError(f"{location} must be {expected!r}")


def _expect_close(
    actual: float,
    expected: float,
    location: str,
    *,
    rel_tol: float = 1e-10,
    abs_tol: float = 1e-10,
) -> None:
    if not math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol):
        raise ValueError(
            f"{location} is inconsistent with source counts or path lengths "
            f"({actual!r} != {expected!r})"
        )


def _validate_sha256(value: Any, location: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{location} must be a lowercase SHA-256 digest")
    return value


def _validate_limitations(value: Any, expected: Sequence[str], location: str) -> None:
    limitations = _array(value, location)
    if limitations != list(expected):
        raise ValueError(
            f"{location} does not preserve the reviewed publication claim boundary"
        )


def _validate_decision(value: Any, location: str) -> None:
    decision = _mapping(value, location)
    for key, expected in (
        ("analytic_domain_invariants_pass", True),
        ("declared_computational_domain_mechanically_enforced", True),
        ("physical_spatial_domain_validated", False),
        ("cone_parameter_physical_semantics_confirmed", False),
        ("finite_aperture_sample_support_audited", False),
        ("training_ready", "NO"),
        ("algorithm_superiority_claim", "LOCKED"),
        ("next_gate", NEXT_GATE),
    ):
        _expect(_required(decision, key, location), expected, f"{location}.{key}")
    if "author_mixed_domain_length_comparison_is_context_only" in decision:
        _expect(
            decision["author_mixed_domain_length_comparison_is_context_only"],
            True,
            f"{location}.author_mixed_domain_length_comparison_is_context_only",
        )


def _validate_vector(value: Any, location: str) -> tuple[float, float, float]:
    vector = _array(value, location)
    if len(vector) != 3:
        raise ValueError(f"{location} must contain three coordinates")
    return tuple(
        _number(component, f"{location}[{index}]")
        for index, component in enumerate(vector)
    )


def _validate_configuration(
    value: Any, *, location: str, ray_count: int
) -> tuple[Any, ...]:
    configuration = _mapping(value, location)
    rows = _integer(
        _required(configuration, "rows", location),
        f"{location}.rows",
        minimum=1,
    )
    if rows != ray_count:
        raise ValueError(f"{location}.rows must equal counts.ray_count")
    _integer(
        _required(configuration, "chunk_rows", location),
        f"{location}.chunk_rows",
        minimum=2,
    )
    _expect(
        _required(configuration, "b0_policy", location),
        B0_POLICY,
        f"{location}.b0_policy",
    )
    _expect(
        _required(configuration, "b1_policy", location),
        B1_POLICY,
        f"{location}.b1_policy",
    )
    _expect(
        _required(configuration, "geometry_contract_version", location),
        GEOMETRY_CONTRACT,
        f"{location}.geometry_contract_version",
    )
    lower = _validate_vector(
        _required(configuration, "outer_minimum_m", location),
        f"{location}.outer_minimum_m",
    )
    upper = _validate_vector(
        _required(configuration, "outer_maximum_m", location),
        f"{location}.outer_maximum_m",
    )
    if any(low >= high for low, high in zip(lower, upper)):
        raise ValueError(f"{location} outer bounds must be strictly ordered")
    vertex = _validate_vector(
        _required(configuration, "cone_vertex_m", location),
        f"{location}.cone_vertex_m",
    )
    axis = _validate_vector(
        _required(configuration, "cone_axis_normalized", location),
        f"{location}.cone_axis_normalized",
    )
    _expect_close(
        math.sqrt(sum(component * component for component in axis)),
        1.0,
        f"{location}.cone_axis_normalized norm",
        abs_tol=1e-12,
    )
    angle = _number(
        _required(configuration, "cone_angle_degrees", location),
        f"{location}.cone_angle_degrees",
    )
    if not 0.0 < angle < 90.0:
        raise ValueError(f"{location}.cone_angle_degrees must be within (0, 90)")
    return (lower, upper, vertex, axis, angle)


def _validate_counts(value: Any, *, location: str) -> dict[str, int]:
    source = _mapping(value, location)
    counts = {
        name: _integer(_required(source, name, location), f"{location}.{name}")
        for name in _COUNT_FIELDS
    }
    ray_count = counts["ray_count"]
    if ray_count < 1:
        raise ValueError(f"{location}.ray_count must be positive")
    for name, count in counts.items():
        if name != "ray_count" and count > ray_count:
            raise ValueError(f"{location}.{name} cannot exceed ray_count")
    if counts["b0_hit_count"] + counts["b0_zero_length_count"] != ray_count:
        raise ValueError(
            f"{location}.b0_hit_count + b0_zero_length_count must equal ray_count"
        )
    if counts["b1_hit_count"] + counts["b1_zero_length_count"] != ray_count:
        raise ValueError(
            f"{location}.b1_hit_count + b1_zero_length_count must equal ray_count"
        )
    if (
        counts["b1_hit_count"] + counts["b1_removed_from_b0_count"]
        != counts["b0_hit_count"]
    ):
        raise ValueError(
            f"{location}.b1_removed_from_b0_count conflicts with B0/B1 hit counts"
        )
    if counts["author_full_box_zero_flag_count"] != counts["b0_zero_length_count"]:
        raise ValueError(
            f"{location}.author_full_box_zero_flag_count conflicts with B0 misses"
        )
    for name in _ZERO_CONTRACT_COUNT_FIELDS:
        if counts[name] != 0:
            raise ValueError(
                f"{location}.{name} must be zero for a contract-pass report"
            )
    return counts


def _validate_path_length(value: Any, *, location: str) -> dict[str, float]:
    source = _mapping(value, location)
    author = _number(
        _required(source, "author_selected_length_sum_m", location),
        f"{location}.author_selected_length_sum_m",
        minimum=0.0,
    )
    b0 = _number(
        _required(source, "b0_length_sum_m", location),
        f"{location}.b0_length_sum_m",
        minimum=0.0,
    )
    b1 = _number(
        _required(source, "b1_length_sum_m", location),
        f"{location}.b1_length_sum_m",
        minimum=0.0,
    )
    if author <= 0.0 or b0 <= 0.0:
        raise ValueError(f"{location} author and B0 length sums must be positive")
    if b1 > b0 + 1e-8:
        raise ValueError(f"{location}.b1_length_sum_m cannot exceed B0 length")
    b1_of_b0 = _fraction(
        _required(source, "b1_fraction_of_b0", location),
        f"{location}.b1_fraction_of_b0",
    )
    b1_of_author = _number(
        _required(source, "b1_fraction_of_author_selected", location),
        f"{location}.b1_fraction_of_author_selected",
        minimum=0.0,
    )
    b0_of_author = _number(
        _required(source, "b0_fraction_of_author_selected", location),
        f"{location}.b0_fraction_of_author_selected",
        minimum=0.0,
    )
    _expect_close(b1_of_b0, b1 / b0, f"{location}.b1_fraction_of_b0")
    _expect_close(
        b1_of_author,
        b1 / author,
        f"{location}.b1_fraction_of_author_selected",
    )
    _expect_close(
        b0_of_author,
        b0 / author,
        f"{location}.b0_fraction_of_author_selected",
    )
    return {
        "author_selected_length_sum_m": author,
        "b0_length_sum_m": b0,
        "b1_length_sum_m": b1,
        "b1_fraction_of_b0": b1_of_b0,
        "b1_fraction_of_author_selected": b1_of_author,
        "b0_fraction_of_author_selected": b0_of_author,
    }


def _validate_mask(value: Any, *, location: str) -> dict[str, float | int]:
    source = _mapping(value, location)
    count = _integer(
        _required(source, "count", location),
        f"{location}.count",
        minimum=1,
    )
    count_fields = (
        "author_nonzero_count",
        "b0_hit_count",
        "b1_hit_count",
        "b1_removed_from_b0_count",
    )
    counts = {
        name: _integer(
            _required(source, name, location),
            f"{location}.{name}",
            maximum=count,
        )
        for name in count_fields
    }
    if (
        counts["b1_hit_count"] + counts["b1_removed_from_b0_count"]
        != counts["b0_hit_count"]
    ):
        raise ValueError(
            f"{location}.b1_removed_from_b0_count conflicts with mask B0/B1 hits"
        )
    fractions = {
        name: _fraction(_required(source, name, location), f"{location}.{name}")
        for name in (
            "author_nonzero_fraction",
            "b0_hit_fraction",
            "b1_hit_fraction",
            "b1_removed_from_b0_fraction",
            "b1_path_fraction_of_b0",
        )
    }
    for count_name, fraction_name in (
        ("author_nonzero_count", "author_nonzero_fraction"),
        ("b0_hit_count", "b0_hit_fraction"),
        ("b1_hit_count", "b1_hit_fraction"),
        ("b1_removed_from_b0_count", "b1_removed_from_b0_fraction"),
    ):
        _expect_close(
            fractions[fraction_name],
            counts[count_name] / count,
            f"{location}.{fraction_name}",
        )
    lengths = {
        name: _number(
            _required(source, name, location),
            f"{location}.{name}",
            minimum=0.0,
        )
        for name in (
            "author_length_sum_m",
            "b0_length_sum_m",
            "b1_length_sum_m",
        )
    }
    if lengths["b0_length_sum_m"] <= 0.0:
        raise ValueError(f"{location}.b0_length_sum_m must be positive")
    if lengths["b1_length_sum_m"] > lengths["b0_length_sum_m"] + 1e-8:
        raise ValueError(f"{location}.b1_length_sum_m cannot exceed B0 length")
    _expect_close(
        fractions["b1_path_fraction_of_b0"],
        lengths["b1_length_sum_m"] / lengths["b0_length_sum_m"],
        f"{location}.b1_path_fraction_of_b0",
    )
    return {"count": count, **counts, **fractions, **lengths}


def _validate_source(value: Any, *, location: str) -> None:
    source = _mapping(value, location)
    for name in (
        "audit_implementation_sha256",
        "bundle_manifest_sha256",
        "corrected_mask_manifest_sha256",
        "geometry_implementation_sha256",
        "setup_manifest_sha256",
    ):
        _validate_sha256(_required(source, name, location), f"{location}.{name}")
    filename = _required(source, "geometry_implementation_filename", location)
    _expect(
        filename,
        "psu_bost_forward_geometry.py",
        f"{location}.geometry_implementation_filename",
    )


def _validate_upstream(value: Any, *, location: str) -> None:
    upstream = _mapping(value, location)
    _expect(
        _required(upstream, "bundle_status", location),
        "VIEW_SHARD_BUNDLE_TRANSCODED_AND_SOURCE_STREAMS_VERIFIED",
        f"{location}.bundle_status",
    )
    _expect(
        _required(upstream, "mask_status", location),
        "CORRECTED_VIEW_MASK_SHARDS_BUILT_MECHANICAL_CONTRACT_PASS",
        f"{location}.mask_status",
    )
    setup_status = _required(upstream, "setup_status", location)
    if not isinstance(setup_status, str) or not setup_status.startswith(
        "STREAMED_SETUP_"
    ):
        raise ValueError(f"{location}.setup_status must start with 'STREAMED_SETUP_'")


def _validate_view(
    value: Any, *, index: int
) -> tuple[dict[str, float | int], tuple[Any, ...], dict[str, int], dict[str, float]]:
    location = f"report.views[{index}]"
    source = _mapping(value, location)
    _expect(
        _required(source, "schema_version", location),
        VIEW_SCHEMA,
        f"{location}.schema_version",
    )
    _expect(_required(source, "status", location), VIEW_STATUS, f"{location}.status")
    _expect(
        _required(source, "evidence_scope", location),
        EVIDENCE_SCOPE,
        f"{location}.evidence_scope",
    )
    view_id = _integer(
        _required(source, "view_id_zero_based", location),
        f"{location}.view_id_zero_based",
    )
    counts = _validate_counts(
        _required(source, "counts", location),
        location=f"{location}.counts",
    )
    configuration = _validate_configuration(
        _required(source, "configuration", location),
        location=f"{location}.configuration",
        ray_count=counts["ray_count"],
    )
    path = _validate_path_length(
        _required(source, "path_length", location),
        location=f"{location}.path_length",
    )
    masks_source = _mapping(
        _required(source, "mask_conditioned", location),
        f"{location}.mask_conditioned",
    )
    masks = {
        name: _validate_mask(
            _required(masks_source, name, f"{location}.mask_conditioned"),
            location=f"{location}.mask_conditioned.{name}",
        )
        for name in _MASK_NAMES
    }
    _validate_source(
        _required(source, "source", location),
        location=f"{location}.source",
    )
    _validate_upstream(
        _required(source, "upstream_view_contract", location),
        location=f"{location}.upstream_view_contract",
    )
    _validate_decision(
        _required(source, "decision", location),
        f"{location}.decision",
    )
    _validate_limitations(
        _required(source, "limitations", location),
        VIEW_LIMITATIONS,
        f"{location}.limitations",
    )
    ray_count = counts["ray_count"]
    record: dict[str, float | int] = {
        "view_id_zero_based": view_id,
        "ray_count": ray_count,
        "b0_hit_fraction": counts["b0_hit_count"] / ray_count,
        "b1_hit_fraction": counts["b1_hit_count"] / ray_count,
        "b1_path_fraction_of_b0": path["b1_fraction_of_b0"],
        "active_b1_hit_fraction": float(masks["amask_all"]["b1_hit_fraction"]),
        "inactive_b1_hit_fraction": float(masks["imask_all"]["b1_hit_fraction"]),
        "b1_removed_fraction_of_b0": (
            counts["b1_removed_from_b0_count"] / counts["b0_hit_count"]
        ),
        "b0_miss_count": counts["b0_zero_length_count"],
        "active_b1_miss_count": (
            int(masks["amask_all"]["count"]) - int(masks["amask_all"]["b1_hit_count"])
        ),
    }
    return record, configuration, counts, path


def load_plot_records(
    report_json: Path,
) -> tuple[list[dict[str, float | int]], dict[str, Any]]:
    """Load and validate all numeric inputs and publication claim boundaries."""

    report_json = Path(report_json)
    report, report_bytes = _read_json(report_json)
    _expect(
        _required(report, "schema_version", "report"),
        REPORT_SCHEMA,
        "report.schema_version",
    )
    _expect(
        _required(report, "execution_status", "report"),
        "COMPLETE",
        "report.execution_status",
    )
    _expect(
        _required(report, "scientific_verdict", "report"),
        SCIENTIFIC_VERDICT,
        "report.scientific_verdict",
    )
    _expect(_required(report, "status", "report"), ROOT_STATUS, "report.status")
    view_count = _integer(
        _required(report, "view_count", "report"),
        "report.view_count",
        minimum=1,
    )
    raw_views = _array(_required(report, "views", "report"), "report.views")
    if len(raw_views) != view_count:
        raise ValueError("report.view_count does not match report.views")

    validated = [
        _validate_view(value, index=index) for index, value in enumerate(raw_views)
    ]
    records = [item[0] for item in validated]
    configurations = [item[1] for item in validated]
    per_view_counts = [item[2] for item in validated]
    per_view_paths = [item[3] for item in validated]
    view_ids = [int(record["view_id_zero_based"]) for record in records]
    if view_ids != list(range(view_count)):
        raise ValueError(
            "report.views view ids must be the ordered contiguous range from zero"
        )
    if any(configuration != configurations[0] for configuration in configurations[1:]):
        raise ValueError("report.views geometry configurations must agree across views")

    aggregate = _mapping(_required(report, "aggregate", "report"), "report.aggregate")
    invalid_view_ids = _array(
        _required(aggregate, "invalid_view_ids", "report.aggregate"),
        "report.aggregate.invalid_view_ids",
    )
    if invalid_view_ids:
        raise ValueError(
            "report.aggregate.invalid_view_ids must be empty for publication"
        )
    active_miss_ids = [
        int(record["view_id_zero_based"])
        for record in records
        if int(record["active_b1_miss_count"]) > 0
    ]
    aggregate_active_miss_ids = [
        _integer(
            value,
            f"report.aggregate.active_mask_b1_zero_view_ids[{index}]",
        )
        for index, value in enumerate(
            _array(
                _required(
                    aggregate,
                    "active_mask_b1_zero_view_ids",
                    "report.aggregate",
                ),
                "report.aggregate.active_mask_b1_zero_view_ids",
            )
        )
    ]
    if aggregate_active_miss_ids != active_miss_ids:
        raise ValueError(
            "report.aggregate.active_mask_b1_zero_view_ids conflicts with "
            "per-view active B1 misses"
        )

    aggregate_counts = _validate_counts(
        _required(aggregate, "counts", "report.aggregate"),
        location="report.aggregate.counts",
    )
    for name in _COUNT_FIELDS:
        expected = sum(counts[name] for counts in per_view_counts)
        if aggregate_counts[name] != expected:
            raise ValueError(
                f"report.aggregate.counts.{name} conflicts with per-view counts"
            )
    aggregate_path = _validate_path_length(
        _required(aggregate, "path_length", "report.aggregate"),
        location="report.aggregate.path_length",
    )
    for name in (
        "author_selected_length_sum_m",
        "b0_length_sum_m",
        "b1_length_sum_m",
    ):
        _expect_close(
            aggregate_path[name],
            sum(path[name] for path in per_view_paths),
            f"report.aggregate.path_length.{name}",
            abs_tol=1e-7,
        )

    _validate_decision(
        _required(report, "decision", "report"),
        "report.decision",
    )
    _validate_limitations(
        _required(report, "limitations", "report"),
        ROOT_LIMITATIONS,
        "report.limitations",
    )

    b0_miss_ids = [
        int(record["view_id_zero_based"])
        for record in records
        if int(record["b0_miss_count"]) > 0
    ]
    pooled = {
        "pooled_b0_hit_fraction_of_all_rays": (
            aggregate_counts["b0_hit_count"] / aggregate_counts["ray_count"]
        ),
        "pooled_b1_hit_fraction_of_all_rays": (
            aggregate_counts["b1_hit_count"] / aggregate_counts["ray_count"]
        ),
        "pooled_b1_removed_fraction_of_b0_hits": (
            aggregate_counts["b1_removed_from_b0_count"]
            / aggregate_counts["b0_hit_count"]
        ),
    }
    source_sha256 = _sha256_bytes(report_bytes)
    provenance = {
        "source_sha256": source_sha256,
        "source": {
            "filename": report_json.name,
            "sha256": source_sha256,
            "schema_version": REPORT_SCHEMA,
            "status": ROOT_STATUS,
        },
        "b0_miss_view_ids_zero_based": b0_miss_ids,
        "active_b1_miss_view_ids_zero_based": active_miss_ids,
        "pooled_metrics": pooled,
        "cone_angle_degrees": configurations[0][-1],
    }
    return records, provenance


def _percentage_formatter(value: float, _position: int) -> str:
    if abs(value) >= 10.0:
        return f"{value:.0f}"
    if abs(value) >= 1.0:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _style_view_axis(ax: Axes, view_ids: Sequence[int]) -> None:
    positions = list(range(len(view_ids)))
    ax.set_xticks(positions, [str(view_id) for view_id in view_ids])
    ax.set_xlim(-0.55, len(view_ids) - 0.45)
    ax.set_xlabel("View (zero-based)")
    ax.tick_params(axis="both", which="major", length=3.5, width=0.7, pad=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.65)


def _style_percentage_axis(ax: Axes, *, upper: float = 105.0) -> None:
    ax.set_ylim(0.0, upper)
    ax.yaxis.set_major_locator(
        MaxNLocator(nbins=6, min_n_ticks=4, steps=[1.0, 2.0, 2.5, 5.0, 10.0])
    )
    ax.yaxis.set_major_formatter(FuncFormatter(_percentage_formatter))


def _panel_heading(ax: Axes, letter: str, title: str) -> None:
    ax.text(
        0.0,
        1.045,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12.0,
        fontweight="bold",
        color=COLORS["ink"],
        clip_on=False,
    )
    ax.text(
        0.062,
        1.045,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["ink"],
        clip_on=False,
    )


def _render_figure(
    records: Sequence[dict[str, float | int]],
    provenance: Mapping[str, Any],
) -> Figure:
    view_ids = [int(record["view_id_zero_based"]) for record in records]
    x = list(range(len(records)))
    percentages = {
        metric: [100.0 * float(record[metric]) for record in records]
        for metric in (
            "b0_hit_fraction",
            "b1_hit_fraction",
            "b1_path_fraction_of_b0",
            "active_b1_hit_fraction",
            "inactive_b1_hit_fraction",
        )
    }
    b0_miss_ids = list(provenance["b0_miss_view_ids_zero_based"])
    active_miss_ids = list(provenance["active_b1_miss_view_ids_zero_based"])
    positions_by_id = {view_id: index for index, view_id in enumerate(view_ids)}
    b0_miss_positions = [positions_by_id[view_id] for view_id in b0_miss_ids]

    figure, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE_INCHES, squeeze=True)
    ax_a, ax_b, ax_c, ax_d = axes.flat
    figure.patch.set_facecolor("white")
    figure.subplots_adjust(
        left=0.078,
        right=0.985,
        bottom=0.165,
        top=0.95,
        wspace=0.235,
        hspace=0.40,
    )

    for axis in (ax_a, ax_b, ax_c):
        for position in b0_miss_positions:
            axis.axvspan(
                position - 0.48,
                position + 0.48,
                color=COLORS["band"],
                linewidth=0,
                zorder=0,
            )

    width = 0.34
    for offset, metric, label, color, hatch in (
        (-width / 2.0, "b0_hit_fraction", "B0 box", COLORS["blue"], "///"),
        (
            width / 2.0,
            "b1_hit_fraction",
            "B1 box + one-nappe cone",
            COLORS["orange"],
            "\\\\",
        ),
    ):
        ax_a.bar(
            [position + offset for position in x],
            percentages[metric],
            width=width,
            label=label,
            color=color,
            edgecolor=COLORS["ink"],
            linewidth=0.4,
            hatch=hatch,
            zorder=2,
        )
    _style_view_axis(ax_a, view_ids)
    _style_percentage_axis(ax_a)
    ax_a.set_ylabel("Centerline rays hit (%)")
    ax_a.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        frameon=True,
        framealpha=0.92,
        edgecolor="none",
        handlelength=1.8,
    )
    _panel_heading(ax_a, "A", "Per-view B0 and B1 hit fractions")

    b_values = percentages["b1_path_fraction_of_b0"]
    ax_b.bar(
        x,
        b_values,
        width=0.64,
        color=COLORS["green"],
        edgecolor=COLORS["ink"],
        linewidth=0.42,
        hatch="..",
        zorder=2,
    )
    _style_view_axis(ax_b, view_ids)
    b_upper = max(20.0, math.ceil(max(b_values) / 5.0) * 5.0)
    _style_percentage_axis(ax_b, upper=b_upper)
    ax_b.set_ylabel("B1 path length / B0 (%)")
    _panel_heading(ax_b, "B", "Per-view B1 path fraction of B0")

    for metric, label, color, marker, linestyle in (
        (
            "active_b1_hit_fraction",
            "Active mask",
            COLORS["vermillion"],
            "o",
            "-",
        ),
        (
            "inactive_b1_hit_fraction",
            "Inactive mask",
            COLORS["blue"],
            "s",
            "--",
        ),
    ):
        ax_c.plot(
            x,
            percentages[metric],
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.8,
            markersize=5.2,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.65,
            zorder=3,
        )
    if active_miss_ids:
        marker_positions = [positions_by_id[view_id] for view_id in active_miss_ids]
        marker_values = [
            percentages["active_b1_hit_fraction"][position]
            for position in marker_positions
        ]
        ax_c.scatter(
            marker_positions,
            marker_values,
            marker="*",
            s=112,
            facecolor="white",
            edgecolor=COLORS["vermillion"],
            linewidth=1.25,
            label="Active B1 misses present",
            zorder=5,
        )
    _style_view_axis(ax_c, view_ids)
    _style_percentage_axis(ax_c)
    ax_c.set_ylabel("Mask centerlines hit by B1 (%)")
    ax_c.legend(loc="center right", frameon=False, handlelength=2.2)
    _panel_heading(ax_c, "C", "Active vs inactive B1 hit fraction")

    pooled = provenance["pooled_metrics"]
    d_labels = (
        "B0 hits / all rays",
        "B1 hits / all rays",
        "B0 hits removed by B1 / B0 hits",
    )
    d_metrics = PANEL_METRICS["D"]
    d_values = [100.0 * float(pooled[metric]) for metric in d_metrics]
    d_colors = (COLORS["blue"], COLORS["orange"], COLORS["purple"])
    d_hatches = ("///", "\\\\", "..")
    y = [2, 1, 0]
    bars = ax_d.barh(
        y,
        d_values,
        height=0.58,
        color=d_colors,
        edgecolor=COLORS["ink"],
        linewidth=0.42,
        zorder=2,
    )
    for bar, hatch, value in zip(bars, d_hatches, d_values):
        bar.set_hatch(hatch)
        ax_d.text(
            min(value + 1.4, 98.5),
            bar.get_y() + bar.get_height() / 2.0,
            f"{value:.1f}%",
            ha="left" if value < 96.0 else "right",
            va="center",
            fontsize=8.2,
            color=COLORS["ink"],
        )
    ax_d.set_yticks(y, d_labels)
    ax_d.set_xlim(0.0, 105.0)
    ax_d.set_xlabel("Pooled centerline census (%)")
    ax_d.xaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
    ax_d.xaxis.set_major_formatter(FuncFormatter(_percentage_formatter))
    ax_d.tick_params(axis="both", which="major", length=3.5, width=0.7, pad=3)
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)
    ax_d.spines["left"].set_linewidth(0.8)
    ax_d.spines["bottom"].set_linewidth(0.8)
    ax_d.set_axisbelow(True)
    ax_d.grid(axis="x", color=COLORS["grid"], linewidth=0.65)
    _panel_heading(ax_d, "D", "Pooled B0/B1 domain comparison")

    band_ids = ", ".join(str(view_id) for view_id in b0_miss_ids) or "none"
    active_ids = ", ".join(str(view_id) for view_id in active_miss_ids) or "none"
    figure.text(
        0.5,
        0.064,
        CAPTION,
        ha="center",
        va="bottom",
        fontsize=8.0,
        color=COLORS["ink"],
    )
    figure.text(
        0.5,
        0.027,
        f"Gray bands: views with B0 centerline misses ({band_ids}). "
        f"Star marker: active-mask B1 misses ({active_ids}).",
        ha="center",
        va="bottom",
        fontsize=8.0,
        color=COLORS["gray"],
    )
    return figure


def _save_staged_outputs(
    figure: Figure, stage_dir: Path, output_stem: str
) -> dict[str, Path]:
    paths = {
        "png": stage_dir / f"{output_stem}.png",
        "svg": stage_dir / f"{output_stem}.svg",
        "pdf": stage_dir / f"{output_stem}.pdf",
    }
    fixed_time = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
    title = "PSU B0/B1 fixed-domain centerline geometry census"
    description = (
        "Four-panel deterministic centerline geometry census; no error bars; "
        "not reconstruction or physical cone validation"
    )
    figure.savefig(
        paths["png"],
        format="png",
        dpi=PNG_DPI,
        facecolor="white",
        edgecolor="none",
        metadata={
            "Title": title,
            "Author": "OERF PSU BOST audit",
            "Description": description,
            "Software": "site_tools/plot_psu_fixed_domain_geometry_audit.py",
        },
    )
    figure.savefig(
        paths["svg"],
        format="svg",
        facecolor="white",
        edgecolor="none",
        metadata={
            "Title": title,
            "Creator": "site_tools/plot_psu_fixed_domain_geometry_audit.py",
            "Description": description,
            "Date": "1970-01-01T00:00:00Z",
        },
    )
    figure.savefig(
        paths["pdf"],
        format="pdf",
        facecolor="white",
        edgecolor="none",
        metadata={
            "Title": title,
            "Author": "OERF PSU BOST audit",
            "Subject": description,
            "Creator": "site_tools/plot_psu_fixed_domain_geometry_audit.py",
            "Producer": "Matplotlib",
            "CreationDate": fixed_time,
            "ModDate": fixed_time,
        },
    )
    return paths


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def build_fixed_domain_geometry_audit_figure(
    report_json: Path,
    output_dir: Path,
    *,
    output_stem: str = DEFAULT_OUTPUT_STEM,
) -> dict[str, Any]:
    """Build deterministic PNG, SVG, PDF, and SHA-256 manifest outputs."""

    if not _OUTPUT_STEM_PATTERN.fullmatch(output_stem):
        raise ValueError(
            "output_stem must contain only ASCII letters, digits, dot, "
            "underscore, or hyphen"
        )
    records, provenance = load_plot_records(Path(report_json))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rc = {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9.2,
        "axes.labelsize": 9.2,
        "axes.labelcolor": COLORS["ink"],
        "axes.edgecolor": COLORS["ink"],
        "text.color": COLORS["ink"],
        "xtick.color": COLORS["ink"],
        "ytick.color": COLORS["ink"],
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "legend.fontsize": 8.0,
        "hatch.linewidth": 0.72,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": MANIFEST_SCHEMA,
    }

    manifest_name = f"{output_stem}_manifest.json"
    with tempfile.TemporaryDirectory(
        prefix=f".{output_stem}.", dir=output_dir
    ) as temporary_directory:
        stage_dir = Path(temporary_directory)
        with matplotlib.rc_context(rc):
            figure = _render_figure(records, provenance)
            try:
                staged_outputs = _save_staged_outputs(figure, stage_dir, output_stem)
            finally:
                plt.close(figure)

        expected_width = round(FIGURE_SIZE_INCHES[0] * PNG_DPI)
        expected_height = round(FIGURE_SIZE_INCHES[1] * PNG_DPI)
        outputs: dict[str, dict[str, Any]] = {}
        for output_format, path in staged_outputs.items():
            _fsync_file(path)
            details: dict[str, Any] = {
                "filename": path.name,
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            if output_format == "png":
                details.update(
                    {
                        "dpi": PNG_DPI,
                        "width_pixels": expected_width,
                        "height_pixels": expected_height,
                    }
                )
            outputs[output_format] = details

        manifest: dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA,
            "status": "FIGURE_BUILD_COMPLETE",
            "hash_algorithm": "sha256",
            **provenance,
            "caption": CAPTION,
            "claim_boundary": {
                "census": "deterministic exhaustive centerline geometry",
                "error_bars": "not applicable",
                "reconstruction_result": False,
                "physical_validation_of_cone_angle": False,
                "finite_aperture_support_audited": False,
                "algorithm_superiority_claim": "LOCKED",
                "training_ready": "NO",
                "next_gate": NEXT_GATE,
            },
            "data_contract": {
                "view_count": len(records),
                "view_ids_zero_based": [
                    int(record["view_id_zero_based"]) for record in records
                ],
                "gray_band_rule": "counts.b0_zero_length_count > 0",
                "active_miss_marker_rule": (
                    "mask_conditioned.amask_all.b1_hit_count "
                    "< mask_conditioned.amask_all.count"
                ),
                "numeric_source": (
                    "all plotted values are derived from the validated report JSON; "
                    "no dataset values or view ids are embedded in the plotting source"
                ),
                "panel_formulas": {
                    "A": {
                        "b0_hit_fraction": "counts.b0_hit_count / counts.ray_count",
                        "b1_hit_fraction": "counts.b1_hit_count / counts.ray_count",
                    },
                    "B": {
                        "b1_path_fraction_of_b0": (
                            "path_length.b1_length_sum_m / path_length.b0_length_sum_m"
                        )
                    },
                    "C": {
                        "active_b1_hit_fraction": (
                            "mask_conditioned.amask_all.b1_hit_count / "
                            "mask_conditioned.amask_all.count"
                        ),
                        "inactive_b1_hit_fraction": (
                            "mask_conditioned.imask_all.b1_hit_count / "
                            "mask_conditioned.imask_all.count"
                        ),
                    },
                    "D": {
                        "pooled_b0_hit_fraction_of_all_rays": (
                            "aggregate.counts.b0_hit_count / aggregate.counts.ray_count"
                        ),
                        "pooled_b1_hit_fraction_of_all_rays": (
                            "aggregate.counts.b1_hit_count / aggregate.counts.ray_count"
                        ),
                        "pooled_b1_removed_fraction_of_b0_hits": (
                            "aggregate.counts.b1_removed_from_b0_count / "
                            "aggregate.counts.b0_hit_count"
                        ),
                    },
                },
            },
            "rendering": {
                "layout": "2x2",
                "figure_size_inches": list(FIGURE_SIZE_INCHES),
                "png_dpi": PNG_DPI,
                "background": "#FFFFFF",
                "font_family": "DejaVu Sans",
                "color_palette": {
                    name: COLORS[name]
                    for name in (
                        "blue",
                        "orange",
                        "green",
                        "vermillion",
                        "purple",
                        "gray",
                    )
                },
                "redundant_encodings": ["hatches", "markers", "line_styles"],
                "matplotlib_version": matplotlib.__version__,
            },
            "panels": {key: list(metrics) for key, metrics in PANEL_METRICS.items()},
            "outputs": outputs,
        }
        manifest_path = stage_dir / manifest_name
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _fsync_file(manifest_path)

        for output_format in ("png", "svg", "pdf"):
            staged = staged_outputs[output_format]
            os.replace(staged, output_dir / staged.name)
        os.replace(manifest_path, output_dir / manifest_name)

    return manifest


build_figure = build_fixed_domain_geometry_audit_figure
build_fixed_domain_geometry_figure = build_fixed_domain_geometry_audit_figure
plot_psu_fixed_domain_geometry_audit = build_fixed_domain_geometry_audit_figure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-json",
        "--audit-json",
        dest="report_json",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--output-stem",
        "--output-prefix",
        default=DEFAULT_OUTPUT_STEM,
    )
    args = parser.parse_args()
    manifest = build_fixed_domain_geometry_audit_figure(
        args.report_json,
        args.output_dir,
        output_stem=args.output_stem,
    )
    print(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
