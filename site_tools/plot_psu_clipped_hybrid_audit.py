#!/usr/bin/env python3
"""Build the publication figure for the PSU A1 clipped-hybrid audit."""

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


REPORT_SCHEMA = "psu-bost-author-compatible-clipped-hybrid-all-view-audit-1.0"
VIEW_SCHEMA = "psu-bost-author-compatible-clipped-hybrid-audit-1.0"
MANIFEST_SCHEMA = "psu-bost-author-compatible-clipped-hybrid-figure-1.0"
DEFAULT_OUTPUT_STEM = "psu_clipped_hybrid_audit_figure"
FIGURE_SIZE_INCHES = (11.0, 7.4)
PNG_DPI = 300

ROOT_PASS_STATUS = "AUTHOR_COMPATIBLE_CLIPPED_HYBRID_ALL_VIEW_CONTRACT_PASS"
ROOT_FILTER_STATUS = (
    "AUTHOR_COMPATIBLE_CLIPPED_HYBRID_ALL_VIEW_CONTRACT_PASS_MASK_FILTER_REQUIRED"
)
VIEW_PASS_STATUS = "AUTHOR_COMPATIBLE_CLIPPED_HYBRID_CONTRACT_PASS"
VIEW_FILTER_STATUS = (
    "AUTHOR_COMPATIBLE_CLIPPED_HYBRID_CONTRACT_PASS_MASK_FILTER_REQUIRED"
)
AUTHOR_POLICY = (
    "AUTHOR_CONE_INTERVAL_INTERSECT_FORWARD_BOX_"
    "AUTHOR_ZERO_CONE_FALLBACK_TO_FORWARD_BOX"
)
CAPTION = (
    "A1 is an author-compatibility clipping ablation that preserves the author's "
    "double-cone primitive and cone-miss-to-box fallback; it is not a fixed-domain "
    "or reconstruction result."
)
GRAY_BAND_CAPTION = (
    "Gray bands mark views with nonzero changes. Deterministic ray census; "
    "no statistical error bars."
)

PANEL_METRICS = {
    "A": ("path_removed_fraction",),
    "B": (
        "changed_ray_fraction",
        "shortened_ray_fraction",
        "zeroed_ray_fraction",
    ),
    "C": (
        "active_mask_changed_fraction",
        "inactive_mask_changed_fraction",
    ),
    "D": (
        "active_path_retained_fraction",
        "inactive_path_retained_fraction",
    ),
}

_OUTPUT_STEM_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ROOT_STATUSES = {ROOT_PASS_STATUS, ROOT_FILTER_STATUS}
_VIEW_STATUSES = {VIEW_PASS_STATUS, VIEW_FILTER_STATUS}
_UPSTREAM_BUNDLE_STATUS = "VIEW_SHARD_BUNDLE_TRANSCODED_AND_SOURCE_STREAMS_VERIFIED"
_UPSTREAM_MASK_STATUS = "CORRECTED_VIEW_MASK_SHARDS_BUILT_MECHANICAL_CONTRACT_PASS"
_COUNT_FIELDS = (
    "ray_count",
    "author_nonzero_count",
    "clipped_nonzero_count",
    "changed_from_author_count",
    "cone_shortened_count",
    "cone_zeroed_for_no_box_overlap_count",
    "forward_box_shortened_count",
    "clipped_zero_length_count",
    "clipped_endpoint_box_violation_count",
    "clipped_length_exceeds_author_count",
    "nonfinite_output_count",
    "negative_clipped_inner_aperture_radius_count",
    "negative_clipped_outer_aperture_radius_count",
)
_ZERO_CONTRACT_COUNT_FIELDS = (
    "clipped_endpoint_box_violation_count",
    "clipped_length_exceeds_author_count",
    "nonfinite_output_count",
    "negative_clipped_inner_aperture_radius_count",
    "negative_clipped_outer_aperture_radius_count",
)
_MASK_NAMES = ("amask_all", "imask_all")

# Okabe-Ito colors. Hatches, markers, and line styles keep color from being the
# only visual cue.
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "ink": "#202124",
    "gray": "#5F6368",
    "band": "#ECEFF1",
    "grid": "#D9DDE1",
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be an object")
    return value


def _array(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{location} must be an array")
    return value


def _required(source: Mapping[str, Any], key: str, location: str) -> Any:
    if key not in source:
        raise ValueError(f"{location}.{key} is required")
    return source[key]


def _integer(value: Any, location: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{location} must be an integer >= {minimum}")
    return value


def _number(
    value: Any,
    location: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{location} must be a finite number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{location} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{location} must be <= {maximum}")
    return result


def _fraction(value: Any, location: str) -> float:
    return _number(value, location, minimum=0.0, maximum=1.0)


def _text(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{location} must be a nonempty string")
    return value


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
    text = _text(value, location)
    if not _SHA256_PATTERN.fullmatch(text):
        raise ValueError(f"{location} must be a lowercase SHA-256 digest")
    return text


def _validate_mask(value: Any, *, location: str) -> dict[str, float | int]:
    source = _mapping(value, location)
    count = _integer(
        _required(source, "count", location), f"{location}.count", minimum=1
    )
    counts = {
        name: _integer(
            _required(source, name, location),
            f"{location}.{name}",
        )
        for name in ("changed_count", "shortened_count", "zeroed_count")
    }
    for name, count_value in counts.items():
        if count_value > count:
            raise ValueError(f"{location}.{name} cannot exceed {location}.count")
    if counts["shortened_count"] + counts["zeroed_count"] > counts["changed_count"]:
        raise ValueError(
            f"{location} shortened_count + zeroed_count cannot exceed changed_count"
        )

    fractions = {
        name: _fraction(
            _required(source, name, location),
            f"{location}.{name}",
        )
        for name in ("changed_fraction", "shortened_fraction", "zeroed_fraction")
    }
    for count_name, fraction_name in (
        ("changed_count", "changed_fraction"),
        ("shortened_count", "shortened_fraction"),
        ("zeroed_count", "zeroed_fraction"),
    ):
        _expect_close(
            fractions[fraction_name],
            counts[count_name] / count,
            f"{location}.{fraction_name}",
        )

    author_length = _number(
        _required(source, "author_length_sum_m", location),
        f"{location}.author_length_sum_m",
        minimum=0.0,
    )
    clipped_length = _number(
        _required(source, "clipped_length_sum_m", location),
        f"{location}.clipped_length_sum_m",
        minimum=0.0,
    )
    if author_length <= 0.0:
        raise ValueError(f"{location}.author_length_sum_m must be positive")
    if clipped_length > author_length + 1e-8:
        raise ValueError(f"{location}.clipped_length_sum_m cannot exceed author length")
    retained = _fraction(
        _required(source, "path_length_retained_fraction", location),
        f"{location}.path_length_retained_fraction",
    )
    _expect_close(
        retained,
        clipped_length / author_length,
        f"{location}.path_length_retained_fraction",
        abs_tol=1e-12,
    )
    return {
        "count": count,
        **counts,
        **fractions,
        "author_length_sum_m": author_length,
        "clipped_length_sum_m": clipped_length,
        "path_length_retained_fraction": retained,
    }


def _validate_view(value: Any, *, index: int) -> dict[str, float | int | str]:
    location = f"report.views[{index}]"
    source = _mapping(value, location)
    _expect(
        _required(source, "schema_version", location),
        VIEW_SCHEMA,
        f"{location}.schema_version",
    )
    status = _text(_required(source, "status", location), f"{location}.status")
    if status not in _VIEW_STATUSES:
        raise ValueError(f"{location}.status has an unreviewed status: {status}")
    view_id = _integer(
        _required(source, "view_id_zero_based", location),
        f"{location}.view_id_zero_based",
    )

    configuration = _mapping(
        _required(source, "configuration", location),
        f"{location}.configuration",
    )
    rows = _integer(
        _required(configuration, "rows", f"{location}.configuration"),
        f"{location}.configuration.rows",
        minimum=1,
    )
    _expect(
        _required(configuration, "policy", f"{location}.configuration"),
        AUTHOR_POLICY,
        f"{location}.configuration.policy",
    )

    counts_source = _mapping(
        _required(source, "counts", location), f"{location}.counts"
    )
    counts = {
        name: _integer(
            _required(counts_source, name, f"{location}.counts"),
            f"{location}.counts.{name}",
        )
        for name in _COUNT_FIELDS
    }
    ray_count = counts["ray_count"]
    if ray_count < 1:
        raise ValueError(f"{location}.counts.ray_count must be positive")
    if rows != ray_count:
        raise ValueError(f"{location}.configuration.rows must equal counts.ray_count")
    for name in (
        "author_nonzero_count",
        "clipped_nonzero_count",
        "changed_from_author_count",
        "cone_shortened_count",
        "cone_zeroed_for_no_box_overlap_count",
        "forward_box_shortened_count",
        "clipped_zero_length_count",
    ):
        if counts[name] > ray_count:
            raise ValueError(f"{location}.counts.{name} cannot exceed ray_count")
    if counts["clipped_nonzero_count"] > counts["author_nonzero_count"]:
        raise ValueError(
            f"{location}.counts.clipped_nonzero_count cannot exceed "
            "author_nonzero_count"
        )
    if counts["clipped_zero_length_count"] != (
        ray_count - counts["clipped_nonzero_count"]
    ):
        raise ValueError(
            f"{location}.counts.clipped_zero_length_count conflicts with "
            "ray_count - clipped_nonzero_count"
        )
    shortened_count = (
        counts["cone_shortened_count"] + counts["forward_box_shortened_count"]
    )
    if (
        shortened_count + counts["cone_zeroed_for_no_box_overlap_count"]
        > counts["changed_from_author_count"]
    ):
        raise ValueError(
            f"{location}.counts shortened and zeroed changes exceed "
            "changed_from_author_count"
        )
    for name in _ZERO_CONTRACT_COUNT_FIELDS:
        if counts[name] != 0:
            raise ValueError(
                f"{location}.counts.{name} must be zero for a contract-pass report"
            )

    path_source = _mapping(
        _required(source, "path_length", location), f"{location}.path_length"
    )
    author_length = _number(
        _required(path_source, "author_length_sum_m", f"{location}.path_length"),
        f"{location}.path_length.author_length_sum_m",
        minimum=0.0,
    )
    clipped_length = _number(
        _required(path_source, "clipped_length_sum_m", f"{location}.path_length"),
        f"{location}.path_length.clipped_length_sum_m",
        minimum=0.0,
    )
    removed_length = _number(
        _required(path_source, "removed_length_sum_m", f"{location}.path_length"),
        f"{location}.path_length.removed_length_sum_m",
        minimum=0.0,
    )
    if author_length <= 0.0:
        raise ValueError(f"{location}.path_length.author_length_sum_m must be positive")
    if clipped_length > author_length + 1e-8:
        raise ValueError(
            f"{location}.path_length.clipped_length_sum_m cannot exceed author length"
        )
    _expect_close(
        removed_length,
        author_length - clipped_length,
        f"{location}.path_length.removed_length_sum_m",
        abs_tol=1e-8,
    )
    retained_fraction = _fraction(
        _required(path_source, "retained_fraction", f"{location}.path_length"),
        f"{location}.path_length.retained_fraction",
    )
    removed_fraction = _fraction(
        _required(path_source, "removed_fraction", f"{location}.path_length"),
        f"{location}.path_length.removed_fraction",
    )
    _expect_close(
        retained_fraction,
        clipped_length / author_length,
        f"{location}.path_length.retained_fraction",
        abs_tol=1e-12,
    )
    _expect_close(
        removed_fraction,
        removed_length / author_length,
        f"{location}.path_length.removed_fraction",
        abs_tol=1e-12,
    )
    _expect_close(
        retained_fraction + removed_fraction,
        1.0,
        f"{location}.path_length retained + removed fractions",
        abs_tol=1e-12,
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

    report_source = _mapping(
        _required(source, "source", location), f"{location}.source"
    )
    _expect(
        _required(report_source, "author_source_modified", f"{location}.source"),
        False,
        f"{location}.source.author_source_modified",
    )
    for key in (
        "view_bundle_manifest_sha256",
        "corrected_mask_manifest_sha256",
        "geometry_source_sha256",
    ):
        _validate_sha256(
            _required(report_source, key, f"{location}.source"),
            f"{location}.source.{key}",
        )

    upstream = _mapping(
        _required(source, "upstream_view_contract", location),
        f"{location}.upstream_view_contract",
    )
    _expect(
        _required(upstream, "bundle_status", f"{location}.upstream_view_contract"),
        _UPSTREAM_BUNDLE_STATUS,
        f"{location}.upstream_view_contract.bundle_status",
    )
    _expect(
        _required(upstream, "mask_status", f"{location}.upstream_view_contract"),
        _UPSTREAM_MASK_STATUS,
        f"{location}.upstream_view_contract.mask_status",
    )

    decision = _mapping(_required(source, "decision", location), f"{location}.decision")
    _expect(
        _required(
            decision, "positive_segments_inside_forward_box", f"{location}.decision"
        ),
        True,
        f"{location}.decision.positive_segments_inside_forward_box",
    )
    _expect(
        _required(decision, "fixed_spatial_domain_established", f"{location}.decision"),
        False,
        f"{location}.decision.fixed_spatial_domain_established",
    )
    _expect(
        _required(decision, "training_ready", f"{location}.decision"),
        "NO",
        f"{location}.decision.training_ready",
    )
    _expect(
        _required(decision, "algorithm_superiority_claim", f"{location}.decision"),
        "LOCKED",
        f"{location}.decision.algorithm_superiority_claim",
    )
    filter_required = counts["clipped_zero_length_count"] > 0
    _expect(
        _required(
            decision,
            "geometry_safe_zero_row_filter_required",
            f"{location}.decision",
        ),
        filter_required,
        f"{location}.decision.geometry_safe_zero_row_filter_required",
    )
    expected_status = VIEW_FILTER_STATUS if filter_required else VIEW_PASS_STATUS
    if status != expected_status:
        raise ValueError(f"{location}.status conflicts with clipped_zero_length_count")

    return {
        "view_id_zero_based": view_id,
        "status": status,
        "ray_count": ray_count,
        "changed_from_author_count": counts["changed_from_author_count"],
        "clipped_zero_length_count": counts["clipped_zero_length_count"],
        "author_length_sum_m": author_length,
        "clipped_length_sum_m": clipped_length,
        "path_removed_fraction": removed_fraction,
        "changed_ray_fraction": counts["changed_from_author_count"] / ray_count,
        "shortened_ray_fraction": shortened_count / ray_count,
        "zeroed_ray_fraction": counts["clipped_zero_length_count"] / ray_count,
        "active_mask_changed_fraction": float(masks["amask_all"]["changed_fraction"]),
        "inactive_mask_changed_fraction": float(masks["imask_all"]["changed_fraction"]),
        "active_path_retained_fraction": float(
            masks["amask_all"]["path_length_retained_fraction"]
        ),
        "inactive_path_retained_fraction": float(
            masks["imask_all"]["path_length_retained_fraction"]
        ),
    }


def load_plot_records(
    report_json: Path,
) -> tuple[list[dict[str, float | int | str]], dict[str, Any]]:
    """Load and validate every numeric and claim-boundary input to the figure."""

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
        "AUTHOR_COMPATIBILITY_ABLATION_ONLY",
        "report.scientific_verdict",
    )
    status = _text(_required(report, "status", "report"), "report.status")
    if status not in _ROOT_STATUSES:
        raise ValueError(f"report.status has an unreviewed status: {status}")
    view_count = _integer(
        _required(report, "view_count", "report"),
        "report.view_count",
        minimum=1,
    )
    raw_views = _array(_required(report, "views", "report"), "report.views")
    if len(raw_views) != view_count:
        raise ValueError("report.view_count does not match report.views")
    records = [
        _validate_view(value, index=index) for index, value in enumerate(raw_views)
    ]
    view_ids = [int(record["view_id_zero_based"]) for record in records]
    if view_ids != list(range(view_count)):
        raise ValueError(
            "report.views view ids must be the ordered contiguous range from zero"
        )

    changed_view_ids = [
        int(record["view_id_zero_based"])
        for record in records
        if int(record["changed_from_author_count"]) > 0
    ]
    filter_view_ids = [
        int(record["view_id_zero_based"])
        for record in records
        if int(record["clipped_zero_length_count"]) > 0
    ]
    expected_status = ROOT_FILTER_STATUS if filter_view_ids else ROOT_PASS_STATUS
    if status != expected_status:
        raise ValueError(
            "report.status conflicts with per-view clipped zero-length counts"
        )

    aggregate = _mapping(_required(report, "aggregate", "report"), "report.aggregate")
    invalid_view_ids = _array(
        _required(aggregate, "invalid_view_ids", "report.aggregate"),
        "report.aggregate.invalid_view_ids",
    )
    if invalid_view_ids:
        raise ValueError(
            "report.aggregate.invalid_view_ids must be empty for publication"
        )
    aggregate_filter_ids = [
        _integer(value, f"report.aggregate.zero_row_filter_required_view_ids[{index}]")
        for index, value in enumerate(
            _array(
                _required(
                    aggregate,
                    "zero_row_filter_required_view_ids",
                    "report.aggregate",
                ),
                "report.aggregate.zero_row_filter_required_view_ids",
            )
        )
    ]
    if aggregate_filter_ids != filter_view_ids:
        raise ValueError(
            "report.aggregate.zero_row_filter_required_view_ids conflicts "
            "with per-view counts"
        )

    aggregate_author = _number(
        _required(aggregate, "author_length_sum_m", "report.aggregate"),
        "report.aggregate.author_length_sum_m",
        minimum=0.0,
    )
    aggregate_clipped = _number(
        _required(aggregate, "clipped_length_sum_m", "report.aggregate"),
        "report.aggregate.clipped_length_sum_m",
        minimum=0.0,
    )
    aggregate_removed = _number(
        _required(aggregate, "removed_length_sum_m", "report.aggregate"),
        "report.aggregate.removed_length_sum_m",
        minimum=0.0,
    )
    expected_author = sum(float(record["author_length_sum_m"]) for record in records)
    expected_clipped = sum(float(record["clipped_length_sum_m"]) for record in records)
    _expect_close(
        aggregate_author,
        expected_author,
        "report.aggregate.author_length_sum_m",
        abs_tol=1e-7,
    )
    _expect_close(
        aggregate_clipped,
        expected_clipped,
        "report.aggregate.clipped_length_sum_m",
        abs_tol=1e-7,
    )
    _expect_close(
        aggregate_removed,
        aggregate_author - aggregate_clipped,
        "report.aggregate.removed_length_sum_m",
        abs_tol=1e-7,
    )
    if aggregate_author <= 0.0:
        raise ValueError("report.aggregate.author_length_sum_m must be positive")
    aggregate_retained_fraction = _fraction(
        _required(aggregate, "retained_fraction", "report.aggregate"),
        "report.aggregate.retained_fraction",
    )
    aggregate_removed_fraction = _fraction(
        _required(aggregate, "removed_fraction", "report.aggregate"),
        "report.aggregate.removed_fraction",
    )
    _expect_close(
        aggregate_retained_fraction,
        aggregate_clipped / aggregate_author,
        "report.aggregate.retained_fraction",
        abs_tol=1e-12,
    )
    _expect_close(
        aggregate_removed_fraction,
        aggregate_removed / aggregate_author,
        "report.aggregate.removed_fraction",
        abs_tol=1e-12,
    )
    expected_changed = sum(
        int(record["changed_from_author_count"]) for record in records
    )
    expected_zeroed = sum(
        int(record["clipped_zero_length_count"]) for record in records
    )
    if (
        _integer(
            _required(aggregate, "changed_ray_count", "report.aggregate"),
            "report.aggregate.changed_ray_count",
        )
        != expected_changed
    ):
        raise ValueError(
            "report.aggregate.changed_ray_count conflicts with per-view counts"
        )
    if (
        _integer(
            _required(aggregate, "clipped_zero_length_count", "report.aggregate"),
            "report.aggregate.clipped_zero_length_count",
        )
        != expected_zeroed
    ):
        raise ValueError(
            "report.aggregate.clipped_zero_length_count conflicts with per-view counts"
        )

    decision = _mapping(_required(report, "decision", "report"), "report.decision")
    for key, expected in (
        ("domain_clipping_mechanically_valid", True),
        ("fixed_spatial_domain_established", False),
        ("training_ready", "NO"),
        ("algorithm_superiority_claim", "LOCKED"),
    ):
        _expect(
            _required(decision, key, "report.decision"),
            expected,
            f"report.decision.{key}",
        )

    source_sha256 = _sha256_bytes(report_bytes)
    provenance = {
        "source_sha256": source_sha256,
        "source": {
            "filename": report_json.name,
            "sha256": source_sha256,
            "schema_version": REPORT_SCHEMA,
            "status": status,
        },
        "changed_view_ids_zero_based": changed_view_ids,
    }
    return records, provenance


def _tick_positions(view_count: int, maximum_ticks: int = 12) -> list[int]:
    if view_count <= maximum_ticks:
        return list(range(view_count))
    return sorted(
        {
            round(index * (view_count - 1) / (maximum_ticks - 1))
            for index in range(maximum_ticks)
        }
    )


def _nice_upper(maximum: float, *, fallback: float = 1.0) -> float:
    if maximum <= 0.0:
        return fallback
    target = maximum * 1.12
    magnitude = 10.0 ** math.floor(math.log10(target))
    for multiplier in (1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0):
        candidate = multiplier * magnitude
        if candidate >= target:
            return candidate
    raise AssertionError("nice-axis bound search failed")


def _trimmed_decimal(value: float, decimals: int) -> str:
    if decimals == 0:
        return f"{value:.0f}"
    return f"{value:.{decimals}f}".rstrip("0").rstrip(".")


def _percentage_formatter(span: float) -> FuncFormatter:
    if span <= 0.25:
        decimals = 3
    elif span <= 2.5:
        decimals = 2
    elif span <= 25.0:
        decimals = 1
    else:
        decimals = 0
    return FuncFormatter(lambda value, _position: _trimmed_decimal(value, decimals))


def _style_axis(ax: Axes, view_ids: Sequence[int]) -> None:
    tick_positions = _tick_positions(len(view_ids))
    ax.set_xticks(tick_positions, [str(view_ids[index]) for index in tick_positions])
    ax.set_xlim(-0.55, len(view_ids) - 0.45)
    ax.set_xlabel("View (zero-based)")
    ax.tick_params(axis="both", which="major", length=3.5, width=0.7, pad=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.65)


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


def _set_zero_percentage_axis(ax: Axes, series: Sequence[Sequence[float]]) -> None:
    maximum = max((max(values, default=0.0) for values in series), default=0.0)
    upper = _nice_upper(maximum)
    ax.set_ylim(0.0, upper)
    ax.yaxis.set_major_locator(
        MaxNLocator(nbins=6, min_n_ticks=4, steps=[1.0, 2.0, 2.5, 5.0, 10.0])
    )
    ax.yaxis.set_major_formatter(_percentage_formatter(upper))


def _set_retained_percentage_axis(ax: Axes, series: Sequence[Sequence[float]]) -> None:
    values = [value for values in series for value in values]
    minimum = min(values, default=100.0)
    gap = max(0.0, 100.0 - minimum)
    padding = max(0.02, gap * 0.18)
    lower = max(0.0, minimum - padding)
    upper = 100.0 + max(0.01, (100.0 - lower) * 0.025)
    ax.set_ylim(lower, upper)
    ax.yaxis.set_major_locator(
        MaxNLocator(nbins=6, min_n_ticks=4, steps=[1.0, 2.0, 2.5, 5.0, 10.0])
    )
    ax.yaxis.set_major_formatter(_percentage_formatter(upper - lower))


def _render_figure(
    records: Sequence[dict[str, float | int | str]],
) -> Figure:
    view_ids = [int(record["view_id_zero_based"]) for record in records]
    x = list(range(len(records)))
    percentages = {
        metric: [100.0 * float(record[metric]) for record in records]
        for metrics in PANEL_METRICS.values()
        for metric in metrics
    }
    changed_positions = [
        index
        for index, record in enumerate(records)
        if int(record["changed_from_author_count"]) > 0
    ]

    figure, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE_INCHES, squeeze=True)
    ax_a, ax_b, ax_c, ax_d = axes.flat
    figure.patch.set_facecolor("white")
    figure.subplots_adjust(
        left=0.078,
        right=0.985,
        bottom=0.155,
        top=0.952,
        wspace=0.235,
        hspace=0.40,
    )

    for axis in (ax_a, ax_b, ax_c, ax_d):
        for position in changed_positions:
            axis.axvspan(
                position - 0.48,
                position + 0.48,
                color=COLORS["band"],
                alpha=1.0,
                linewidth=0,
                zorder=0,
            )

    a_values = percentages["path_removed_fraction"]
    ax_a.bar(
        x,
        a_values,
        width=0.66,
        color=COLORS["blue"],
        edgecolor=COLORS["ink"],
        linewidth=0.45,
        hatch="///",
        zorder=2,
    )
    _style_axis(ax_a, view_ids)
    _set_zero_percentage_axis(ax_a, [a_values])
    ax_a.set_ylabel("Path length removed (%)")
    _panel_heading(ax_a, "A", "Per-view path length removed")

    b_specs = (
        (
            "changed_ray_fraction",
            "Changed",
            COLORS["blue"],
            "///",
        ),
        (
            "shortened_ray_fraction",
            "Shortened",
            COLORS["orange"],
            "\\\\",
        ),
        (
            "zeroed_ray_fraction",
            "Zero-length after clipping",
            COLORS["green"],
            "..",
        ),
    )
    width = 0.24
    b_series: list[list[float]] = []
    for offset, (metric, label, color, hatch) in zip((-width, 0.0, width), b_specs):
        values = percentages[metric]
        b_series.append(values)
        ax_b.bar(
            [position + offset for position in x],
            values,
            width=width,
            label=label,
            color=color,
            edgecolor=COLORS["ink"],
            linewidth=0.4,
            hatch=hatch,
            zorder=2,
        )
    _style_axis(ax_b, view_ids)
    _set_zero_percentage_axis(ax_b, b_series)
    ax_b.set_ylabel("Rays (%)")
    ax_b.legend(
        loc="upper left",
        frameon=False,
        handlelength=1.8,
        borderaxespad=0.45,
    )
    _panel_heading(ax_b, "B", "Changed, shortened, and zeroed rays")

    c_specs = (
        (
            "active_mask_changed_fraction",
            "Active mask",
            COLORS["vermillion"],
            "o",
            "-",
        ),
        (
            "inactive_mask_changed_fraction",
            "Inactive mask",
            COLORS["blue"],
            "s",
            "--",
        ),
    )
    c_series: list[list[float]] = []
    for metric, label, color, marker, linestyle in c_specs:
        values = percentages[metric]
        c_series.append(values)
        ax_c.plot(
            x,
            values,
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
    _style_axis(ax_c, view_ids)
    _set_zero_percentage_axis(ax_c, c_series)
    ax_c.set_ylabel("Mask samples changed (%)")
    ax_c.legend(
        loc="upper right",
        frameon=False,
        ncol=2,
        handlelength=2.2,
    )
    _panel_heading(ax_c, "C", "Active vs inactive mask changes")

    d_specs = (
        (
            "active_path_retained_fraction",
            "Active mask",
            COLORS["vermillion"],
            "^",
            "-",
        ),
        (
            "inactive_path_retained_fraction",
            "Inactive mask",
            COLORS["blue"],
            "D",
            "--",
        ),
    )
    d_series: list[list[float]] = []
    for metric, label, color, marker, linestyle in d_specs:
        values = percentages[metric]
        d_series.append(values)
        ax_d.plot(
            x,
            values,
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
    _style_axis(ax_d, view_ids)
    _set_retained_percentage_axis(ax_d, d_series)
    ax_d.set_ylabel("Path length retained (%)")
    ax_d.legend(
        loc="lower left",
        frameon=False,
        ncol=2,
        handlelength=2.2,
    )
    _panel_heading(ax_d, "D", "Active/inactive path length retained")

    figure.text(
        0.5,
        0.053,
        CAPTION,
        ha="center",
        va="bottom",
        fontsize=8.0,
        color=COLORS["ink"],
    )
    figure.text(
        0.5,
        0.025,
        GRAY_BAND_CAPTION,
        ha="center",
        va="bottom",
        fontsize=8.0,
        color=COLORS["gray"],
    )
    return figure


def _save_staged_outputs(
    figure: Figure, stage_dir: Path, output_stem: str
) -> dict[str, Path]:
    output_paths = {
        "png": stage_dir / f"{output_stem}.png",
        "svg": stage_dir / f"{output_stem}.svg",
        "pdf": stage_dir / f"{output_stem}.pdf",
    }
    fixed_time = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
    title = "PSU A1 author-compatible clipped-hybrid audit"
    description = (
        "Four-panel deterministic audit of path clipping and mask-conditioned "
        "effects across PSU BOST views"
    )
    figure.savefig(
        output_paths["png"],
        format="png",
        dpi=PNG_DPI,
        facecolor="white",
        edgecolor="none",
        metadata={
            "Title": title,
            "Author": "OERF PSU BOST audit",
            "Description": description,
            "Software": "site_tools/plot_psu_clipped_hybrid_audit.py",
        },
    )
    figure.savefig(
        output_paths["svg"],
        format="svg",
        facecolor="white",
        edgecolor="none",
        metadata={
            "Title": title,
            "Creator": "site_tools/plot_psu_clipped_hybrid_audit.py",
            "Description": description,
            "Date": "1970-01-01T00:00:00Z",
        },
    )
    figure.savefig(
        output_paths["pdf"],
        format="pdf",
        facecolor="white",
        edgecolor="none",
        metadata={
            "Title": title,
            "Author": "OERF PSU BOST audit",
            "Subject": description,
            "Creator": "site_tools/plot_psu_clipped_hybrid_audit.py",
            "Producer": "Matplotlib",
            "CreationDate": fixed_time,
            "ModDate": fixed_time,
        },
    )
    return output_paths


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def build_clipped_hybrid_audit_figure(
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
            figure = _render_figure(records)
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
            "gray_band_caption": GRAY_BAND_CAPTION,
            "data_contract": {
                "view_count": len(records),
                "view_ids_zero_based": [
                    int(record["view_id_zero_based"]) for record in records
                ],
                "gray_band_rule": "counts.changed_from_author_count > 0",
                "numeric_source": (
                    "all plotted values are derived from the validated report JSON; "
                    "no dataset values are embedded in the plotting source"
                ),
                "panel_formulas": {
                    "A": {
                        "path_removed_fraction": "path_length.removed_fraction",
                    },
                    "B": {
                        "changed_ray_fraction": (
                            "counts.changed_from_author_count / counts.ray_count"
                        ),
                        "shortened_ray_fraction": (
                            "(counts.cone_shortened_count + "
                            "counts.forward_box_shortened_count) / counts.ray_count"
                        ),
                        "zeroed_ray_fraction": (
                            "counts.clipped_zero_length_count / counts.ray_count"
                        ),
                    },
                    "C": {
                        "active_mask_changed_fraction": (
                            "mask_conditioned.amask_all.changed_fraction"
                        ),
                        "inactive_mask_changed_fraction": (
                            "mask_conditioned.imask_all.changed_fraction"
                        ),
                    },
                    "D": {
                        "active_path_retained_fraction": (
                            "mask_conditioned.amask_all."
                            "path_length_retained_fraction"
                        ),
                        "inactive_path_retained_fraction": (
                            "mask_conditioned.imask_all."
                            "path_length_retained_fraction"
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
            "panels": {key: list(value) for key, value in PANEL_METRICS.items()},
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


build_figure = build_clipped_hybrid_audit_figure
build_clipped_hybrid_figure = build_clipped_hybrid_audit_figure
plot_psu_clipped_hybrid_audit = build_clipped_hybrid_audit_figure


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
    manifest = build_clipped_hybrid_audit_figure(
        args.report_json,
        args.output_dir,
        output_stem=args.output_stem,
    )
    print(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
