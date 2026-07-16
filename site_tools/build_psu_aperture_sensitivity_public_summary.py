#!/usr/bin/env python3
"""Export strict public B2 aperture support sensitivity from QMC 8/16/32."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


PRIVATE_SCHEMA_VERSION = "psu-bost-aperture-domain-all-view-audit-1.0"
PRIVATE_VIEW_SCHEMA_VERSION = "psu-bost-aperture-domain-audit-1.0"
PUBLIC_SCHEMA_VERSION = "psu-bost-aperture-sensitivity-public-summary-1.0"

EXECUTION_STATUS = "COMPLETE"
REPORT_STATUS = "B2_ALL_VIEW_APERTURE_DOMAIN_AUDIT_PASS_B3_REQUIRED"
VIEW_STATUS = "B2_DETERMINISTIC_APERTURE_DOMAIN_AUDIT_PASS_B3_REQUIRED"
SCIENTIFIC_VERDICT = "DISCRETE_APERTURE_SUPPORT_AUDIT_COMPLETE_B3_AND_HELD_OUT_REQUIRED"
PUBLIC_STATUS = "B2_DISCRETE_APERTURE_SUPPORT_SENSITIVITY_COMPLETE"
PUBLIC_VERDICT = "DISCRETE_DETERMINISTIC_APERTURE_SUPPORT_SENSITIVITY_ONLY"
EVIDENCE_SCOPE = (
    "REAL_ONE_VIEW_DETERMINISTIC_PAIRED_LOW_DISCREPANCY_APERTURE_SUPPORT_"
    "AUDIT_NO_TENSORFLOW_NO_RECONSTRUCTION"
)
NEXT_GATE = (
    "QMC_SAMPLE_COUNT_SENSITIVITY_THEN_B3_GEOMETRY_SAFE_MASK_AND_HELD_OUT_REPROJECTION"
)
NORMALIZATION_POLICY = "FIXED_ORIGINAL_SAMPLE_COUNT_NO_SURVIVOR_RENORMALIZATION"
SAMPLE_DESIGN = "PAIRED_HAMMERSLEY_UNIFORM_PATH_AND_DISK_INTERIOR"
FORWARD_CONTRACT_VERSION = "psu-bost-forward-geometry-1.0"
APERTURE_CONTRACT_VERSION = "psu-bost-aperture-domain-1.0"

EXPECTED_SAMPLE_COUNTS = (8, 16, 32)
EXPECTED_VIEW_IDS = tuple(range(9))
DOMAINS = ("B0", "B1")
MASKS = {"active": "amask_all", "inactive": "imask_all"}
CONTEXTS = ("all", "active", "inactive")

SOURCE_FIELDS = (
    "aperture_geometry_sha256",
    "audit_implementation_sha256",
    "bundle_manifest_sha256",
    "corrected_mask_manifest_sha256",
    "forward_geometry_sha256",
    "setup_manifest_sha256",
)
SUPPORT_COUNT_FIELDS = (
    "ray_count",
    "centerline_hit_count",
    "centerline_miss_count",
    "eligible_sample_count",
    "in_domain_sample_count",
    "out_of_domain_sample_count",
    "box_out_sample_count",
    "cone_only_out_sample_count",
    "all_samples_in_domain_ray_count",
    "any_sample_out_of_domain_ray_count",
    "empty_sample_support_ray_count",
)
QUANTILE_FIELDS = {
    "retained_fraction_minimum": 0.0,
    "retained_fraction_p10_nearest_rank": 0.10,
    "retained_fraction_median_nearest_rank": 0.50,
    "retained_fraction_p90_nearest_rank": 0.90,
}
DIAGNOSTIC_FIELDS = (
    "basis_validation_failure_count",
    "negative_radius_count",
    "nonfinite_radius_count",
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_PRIVATE_TEXT_PATTERNS = (
    re.compile(r"(?:^|[\s\"'])/(?:Users|Volumes|private|tmp)/", re.IGNORECASE),
    re.compile(r"file://", re.IGNORECASE),
    re.compile(r"private_library", re.IGNORECASE),
    re.compile(r"(?:^|\s)(?:def|class|import|from)\s+[A-Za-z_]", re.IGNORECASE),
)
_FORBIDDEN_PUBLIC_KEYS = {
    "configuration",
    "runtime_observation",
    "source",
    "unit_disk_offsets",
    "longitudinal_fractions",
    "invalid_view_ids",
}


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
    result = _text(value, path)
    if result != expected:
        raise ValueError(f"{path} has an unreviewed value: {result}")
    return result


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a boolean")
    return value


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{path} must be an integer >= {minimum}")
    return value


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be a finite number")
    return result


def _fraction(value: Any, path: str) -> float:
    result = _number(value, path)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{path} must lie in [0, 1]")
    return result


def _optional_fraction(value: Any, path: str) -> float | None:
    if value is None:
        return None
    return _fraction(value, path)


def _same_number(actual: float, expected: float, path: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{path} is inconsistent")


def _validate_fraction(
    actual: Any,
    numerator: int,
    denominator: int,
    path: str,
) -> float | None:
    result = _optional_fraction(actual, path)
    if denominator == 0:
        if result is not None:
            raise ValueError(f"{path} must be null when its denominator is zero")
        return None
    if result is None:
        raise ValueError(f"{path} must be numeric when its denominator is nonzero")
    _same_number(result, numerator / denominator, path)
    return result


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


def _sha256(value: Any, path: str) -> str:
    result = _text(value, path)
    if not _SHA256_PATTERN.fullmatch(result):
        raise ValueError(f"{path} must be a lowercase SHA-256 digest")
    return result


def _vector(
    value: Any,
    path: str,
    *,
    length: int,
) -> tuple[float, ...]:
    items = _sequence(value, path)
    if len(items) != length:
        raise ValueError(f"{path} must contain exactly {length} values")
    return tuple(_number(item, f"{path}[{index}]") for index, item in enumerate(items))


def _nearest_rank_fraction(
    histogram: Sequence[int],
    quantile: float,
) -> float | None:
    total = sum(histogram)
    if total == 0:
        return None
    rank = 1 if quantile == 0.0 else math.ceil(quantile * total)
    cumulative = 0
    denominator = len(histogram) - 1
    for retained, count in enumerate(histogram):
        cumulative += count
        if cumulative >= rank:
            return retained / denominator
    raise AssertionError("histogram rank was not found")


def _validate_decision(value: Any, path: str) -> None:
    source = _mapping(value, path)
    expected_booleans = {
        "continuous_aperture_containment_proved": False,
        "discrete_aperture_support_audited": True,
        "fixed_denominator_indicator_implemented": True,
        "geometry_safe_mask_built": False,
    }
    for field, expected in expected_booleans.items():
        actual = _boolean(_required(source, field, path), f"{path}.{field}")
        if actual is not expected:
            raise ValueError(f"{path}.{field} conflicts with the claim boundary")
    _expected_text(
        _required(source, "training_ready", path), "NO", f"{path}.training_ready"
    )
    _expected_text(
        _required(source, "algorithm_superiority_claim", path),
        "LOCKED",
        f"{path}.algorithm_superiority_claim",
    )
    _expected_text(_required(source, "next_gate", path), NEXT_GATE, f"{path}.next_gate")


def _validate_source(value: Any, path: str) -> dict[str, str]:
    source = _mapping(value, path)
    return {
        field: _sha256(_required(source, field, path), f"{path}.{field}")
        for field in SOURCE_FIELDS
    }


def _validate_configuration(
    value: Any,
    sample_count: int,
    path: str,
) -> dict[str, Any]:
    source = _mapping(value, path)
    configured_count = _integer(
        _required(source, "sample_count_per_centerline_hit", path),
        f"{path}.sample_count_per_centerline_hit",
        minimum=1,
    )
    if configured_count != sample_count:
        raise ValueError(
            f"{path}.sample_count_per_centerline_hit conflicts with the report"
        )
    _integer(_required(source, "chunk_rows", path), f"{path}.chunk_rows", minimum=1)
    _expected_text(
        _required(source, "sample_design", path),
        SAMPLE_DESIGN,
        f"{path}.sample_design",
    )
    _expected_text(
        _required(source, "normalization_policy", path),
        NORMALIZATION_POLICY,
        f"{path}.normalization_policy",
    )

    fractions = _sequence(
        _required(source, "longitudinal_fractions", path),
        f"{path}.longitudinal_fractions",
    )
    if len(fractions) != sample_count:
        raise ValueError(
            f"{path}.longitudinal_fractions must match the fixed sample count"
        )
    for index, item in enumerate(fractions):
        actual = _fraction(item, f"{path}.longitudinal_fractions[{index}]")
        _same_number(
            actual,
            (index + 0.5) / sample_count,
            f"{path}.longitudinal_fractions[{index}]",
        )

    offsets = _sequence(
        _required(source, "unit_disk_offsets", path),
        f"{path}.unit_disk_offsets",
    )
    if len(offsets) != sample_count:
        raise ValueError(f"{path}.unit_disk_offsets must match the fixed sample count")
    for index, item in enumerate(offsets):
        x, y = _vector(item, f"{path}.unit_disk_offsets[{index}]", length=2)
        if x * x + y * y > 1.0 + 1e-12:
            raise ValueError(f"{path}.unit_disk_offsets[{index}] lies outside the disk")

    compatible = {
        "rows": _integer(_required(source, "rows", path), f"{path}.rows", minimum=1),
        "outer_minimum_m": _vector(
            _required(source, "outer_minimum_m", path),
            f"{path}.outer_minimum_m",
            length=3,
        ),
        "outer_maximum_m": _vector(
            _required(source, "outer_maximum_m", path),
            f"{path}.outer_maximum_m",
            length=3,
        ),
        "cone_vertex_m": _vector(
            _required(source, "cone_vertex_m", path),
            f"{path}.cone_vertex_m",
            length=3,
        ),
        "cone_axis_normalized": _vector(
            _required(source, "cone_axis_normalized", path),
            f"{path}.cone_axis_normalized",
            length=3,
        ),
        "cone_angle_degrees": _number(
            _required(source, "cone_angle_degrees", path),
            f"{path}.cone_angle_degrees",
        ),
        "forward_contract_version": _expected_text(
            _required(source, "forward_contract_version", path),
            FORWARD_CONTRACT_VERSION,
            f"{path}.forward_contract_version",
        ),
        "aperture_contract_version": _expected_text(
            _required(source, "aperture_contract_version", path),
            APERTURE_CONTRACT_VERSION,
            f"{path}.aperture_contract_version",
        ),
        "normalization_policy": NORMALIZATION_POLICY,
    }
    if any(
        lower >= upper
        for lower, upper in zip(
            compatible["outer_minimum_m"],
            compatible["outer_maximum_m"],
        )
    ):
        raise ValueError(f"{path} outer bounds must be ordered")
    if not 0.0 < compatible["cone_angle_degrees"] < 90.0:
        raise ValueError(f"{path}.cone_angle_degrees must lie in (0, 90)")
    axis_norm = math.sqrt(
        sum(item * item for item in compatible["cone_axis_normalized"])
    )
    _same_number(axis_norm, 1.0, f"{path}.cone_axis_normalized")
    return compatible


def _validate_diagnostics(value: Any, path: str) -> None:
    source = _mapping(value, path)
    for domain in DOMAINS:
        domain_source = _mapping(_required(source, domain, path), f"{path}.{domain}")
        for field in DIAGNOSTIC_FIELDS:
            count = _integer(
                _required(domain_source, field, f"{path}.{domain}"),
                f"{path}.{domain}.{field}",
            )
            if count != 0:
                raise ValueError(f"{path}.{domain}.{field} must be zero")


def _validate_upstream(value: Any, path: str) -> dict[str, str]:
    source = _mapping(value, path)
    bundle_status = _expected_text(
        _required(source, "bundle_status", path),
        "VIEW_SHARD_BUNDLE_TRANSCODED_AND_SOURCE_STREAMS_VERIFIED",
        f"{path}.bundle_status",
    )
    mask_status = _expected_text(
        _required(source, "mask_status", path),
        "CORRECTED_VIEW_MASK_SHARDS_BUILT_MECHANICAL_CONTRACT_PASS",
        f"{path}.mask_status",
    )
    setup_status = _text(
        _required(source, "setup_status", path),
        f"{path}.setup_status",
    )
    if setup_status not in {
        "STREAMED_SETUP_MECHANICAL_CONTRACT_PASS",
        "STREAMED_SETUP_DIAGNOSTIC_NO_GO",
    }:
        raise ValueError(f"{path}.setup_status has an unreviewed value: {setup_status}")
    return {
        "bundle_status": bundle_status,
        "mask_status": mask_status,
        "setup_status": setup_status,
    }


def _validate_support(
    value: Any,
    sample_count: int,
    path: str,
    *,
    domain: str,
) -> dict[str, Any]:
    source = _mapping(value, path)
    counts = {
        field: _integer(_required(source, field, path), f"{path}.{field}")
        for field in SUPPORT_COUNT_FIELDS
    }
    histogram_source = _sequence(
        _required(source, "retained_sample_count_histogram", path),
        f"{path}.retained_sample_count_histogram",
    )
    if len(histogram_source) != sample_count + 1:
        raise ValueError(
            f"{path}.retained_sample_count_histogram must have sample_count + 1 bins"
        )
    histogram = [
        _integer(item, f"{path}.retained_sample_count_histogram[{index}]")
        for index, item in enumerate(histogram_source)
    ]

    if (
        counts["centerline_hit_count"] + counts["centerline_miss_count"]
        != counts["ray_count"]
    ):
        raise ValueError(f"{path} centerline counts do not sum to ray_count")
    if counts["eligible_sample_count"] != counts["centerline_hit_count"] * sample_count:
        raise ValueError(f"{path}.eligible_sample_count violates the fixed denominator")
    if (
        counts["in_domain_sample_count"] + counts["out_of_domain_sample_count"]
        != counts["eligible_sample_count"]
    ):
        raise ValueError(f"{path} sample counts do not sum to eligible_sample_count")
    if (
        counts["box_out_sample_count"] + counts["cone_only_out_sample_count"]
        != counts["out_of_domain_sample_count"]
    ):
        raise ValueError(f"{path} OOD sample counts do not reconcile")
    if domain == "B0" and counts["cone_only_out_sample_count"] != 0:
        raise ValueError(f"{path}.cone_only_out_sample_count must be zero for B0")
    if (
        counts["all_samples_in_domain_ray_count"]
        + counts["any_sample_out_of_domain_ray_count"]
        != counts["centerline_hit_count"]
    ):
        raise ValueError(f"{path} support ray counts do not sum to centerline hits")
    if sum(histogram) != counts["centerline_hit_count"]:
        raise ValueError(f"{path} histogram sum does not match centerline hits")
    if (
        sum(index * count for index, count in enumerate(histogram))
        != counts["in_domain_sample_count"]
    ):
        raise ValueError(
            f"{path} histogram weighted sum does not match retained samples"
        )
    if histogram[0] != counts["empty_sample_support_ray_count"]:
        raise ValueError(f"{path} histogram zero bin does not match empty support")
    if histogram[-1] != counts["all_samples_in_domain_ray_count"]:
        raise ValueError(f"{path} histogram full bin does not match full support")
    if sum(histogram[:-1]) != counts["any_sample_out_of_domain_ray_count"]:
        raise ValueError(f"{path} histogram OOD bins do not match any-OOD rays")

    fractions = {
        "centerline_hit_fraction": _validate_fraction(
            _required(source, "centerline_hit_fraction", path),
            counts["centerline_hit_count"],
            counts["ray_count"],
            f"{path}.centerline_hit_fraction",
        ),
        "fixed_denominator_sample_retained_fraction": _validate_fraction(
            _required(source, "fixed_denominator_sample_retained_fraction", path),
            counts["in_domain_sample_count"],
            counts["eligible_sample_count"],
            f"{path}.fixed_denominator_sample_retained_fraction",
        ),
        "any_sample_out_of_domain_ray_fraction_of_hits": _validate_fraction(
            _required(source, "any_sample_out_of_domain_ray_fraction_of_hits", path),
            counts["any_sample_out_of_domain_ray_count"],
            counts["centerline_hit_count"],
            f"{path}.any_sample_out_of_domain_ray_fraction_of_hits",
        ),
        "empty_sample_support_ray_fraction_of_hits": _validate_fraction(
            _required(source, "empty_sample_support_ray_fraction_of_hits", path),
            counts["empty_sample_support_ray_count"],
            counts["centerline_hit_count"],
            f"{path}.empty_sample_support_ray_fraction_of_hits",
        ),
    }
    for field, quantile in QUANTILE_FIELDS.items():
        actual = _optional_fraction(_required(source, field, path), f"{path}.{field}")
        expected = _nearest_rank_fraction(histogram, quantile)
        if actual is None or expected is None:
            if actual is not expected:
                raise ValueError(f"{path}.{field} is inconsistent")
        else:
            _same_number(actual, expected, f"{path}.{field}")

    return {
        **counts,
        **fractions,
        "retained_sample_count_histogram": histogram,
    }


def _support_summary(value: Mapping[str, Any]) -> dict[str, int | float | None]:
    return {
        "retained_sample_fraction": value["fixed_denominator_sample_retained_fraction"],
        "any_ood_ray_count": value["any_sample_out_of_domain_ray_count"],
        "any_ood_ray_fraction_of_centerline_hits": value[
            "any_sample_out_of_domain_ray_fraction_of_hits"
        ],
        "empty_support_ray_count": value["empty_sample_support_ray_count"],
    }


def _validate_view(
    value: Any,
    sample_count: int,
    expected_view_id: int,
    path: str,
) -> dict[str, Any]:
    source = _mapping(value, path)
    _expected_text(
        _required(source, "schema_version", path),
        PRIVATE_VIEW_SCHEMA_VERSION,
        f"{path}.schema_version",
    )
    _expected_text(_required(source, "status", path), VIEW_STATUS, f"{path}.status")
    _expected_text(
        _required(source, "evidence_scope", path),
        EVIDENCE_SCOPE,
        f"{path}.evidence_scope",
    )
    view_id = _integer(
        _required(source, "view_id_zero_based", path),
        f"{path}.view_id_zero_based",
    )
    if view_id != expected_view_id:
        raise ValueError(
            f"{path}.view_id_zero_based must equal the exact expected view id "
            f"{expected_view_id}"
        )
    _validate_decision(_required(source, "decision", path), f"{path}.decision")
    _validate_diagnostics(
        _required(source, "diagnostics", path),
        f"{path}.diagnostics",
    )

    domains_source = _mapping(_required(source, "domains", path), f"{path}.domains")
    masks_source = _mapping(
        _required(source, "mask_conditioned", path),
        f"{path}.mask_conditioned",
    )
    supports: dict[str, dict[str, dict[str, Any]]] = {}
    for domain in DOMAINS:
        supports[domain] = {
            "all": _validate_support(
                _required(domains_source, domain, f"{path}.domains"),
                sample_count,
                f"{path}.domains.{domain}",
                domain=domain,
            )
        }
        for public_name, private_name in MASKS.items():
            mask_source = _mapping(
                _required(masks_source, private_name, f"{path}.mask_conditioned"),
                f"{path}.mask_conditioned.{private_name}",
            )
            supports[domain][public_name] = _validate_support(
                _required(
                    mask_source,
                    domain,
                    f"{path}.mask_conditioned.{private_name}",
                ),
                sample_count,
                f"{path}.mask_conditioned.{private_name}.{domain}",
                domain=domain,
            )

    return {
        "view_id_zero_based": view_id,
        "source": _validate_source(_required(source, "source", path), f"{path}.source"),
        "configuration": _validate_configuration(
            _required(source, "configuration", path),
            sample_count,
            f"{path}.configuration",
        ),
        "upstream_view_contract": _validate_upstream(
            _required(source, "upstream_view_contract", path),
            f"{path}.upstream_view_contract",
        ),
        "supports": supports,
    }


def _sum_view_support(
    views: Sequence[Mapping[str, Any]],
    domain: str,
    context: str,
) -> dict[str, Any]:
    result = {
        field: sum(int(view["supports"][domain][context][field]) for view in views)
        for field in SUPPORT_COUNT_FIELDS
    }
    histogram_length = len(
        views[0]["supports"][domain][context]["retained_sample_count_histogram"]
    )
    result["retained_sample_count_histogram"] = [
        sum(
            int(
                view["supports"][domain][context]["retained_sample_count_histogram"][
                    index
                ]
            )
            for view in views
        )
        for index in range(histogram_length)
    ]
    return result


def _validate_aggregate_support(
    aggregate: Mapping[str, Any],
    views: Sequence[Mapping[str, Any]],
    sample_count: int,
    domain: str,
    context: str,
    path: str,
) -> dict[str, Any]:
    validated = _validate_support(
        aggregate,
        sample_count,
        path,
        domain=domain,
    )
    pooled = _sum_view_support(views, domain, context)
    for field in SUPPORT_COUNT_FIELDS:
        if validated[field] != pooled[field]:
            raise ValueError(f"{path}.{field} conflicts with per-view data")
    if (
        validated["retained_sample_count_histogram"]
        != pooled["retained_sample_count_histogram"]
    ):
        raise ValueError(f"{path}.retained_sample_count_histogram conflicts with views")
    return validated


def _public_support_tree(
    supports: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, dict[str, dict[str, int | float | None]]]:
    return {
        domain: {
            context: _support_summary(supports[domain][context]) for context in CONTEXTS
        }
        for domain in DOMAINS
    }


def _validate_private_report(
    value: Mapping[str, Any],
    sample_count: int,
    path: str,
) -> dict[str, Any]:
    _validate_json_tree(value, path)
    _expected_text(
        _required(value, "schema_version", path),
        PRIVATE_SCHEMA_VERSION,
        f"{path}.schema_version",
    )
    _expected_text(
        _required(value, "execution_status", path),
        EXECUTION_STATUS,
        f"{path}.execution_status",
    )
    _expected_text(
        _required(value, "scientific_verdict", path),
        SCIENTIFIC_VERDICT,
        f"{path}.scientific_verdict",
    )
    _expected_text(_required(value, "status", path), REPORT_STATUS, f"{path}.status")
    reported_sample_count = _integer(
        _required(value, "sample_count_per_centerline_hit", path),
        f"{path}.sample_count_per_centerline_hit",
        minimum=1,
    )
    if reported_sample_count != sample_count:
        raise ValueError(
            f"{path}.sample_count_per_centerline_hit must equal {sample_count}"
        )
    view_count = _integer(
        _required(value, "view_count", path),
        f"{path}.view_count",
        minimum=1,
    )
    if view_count != len(EXPECTED_VIEW_IDS):
        raise ValueError(f"{path}.view_count must equal {len(EXPECTED_VIEW_IDS)}")
    _validate_decision(_required(value, "decision", path), f"{path}.decision")

    raw_views = _sequence(_required(value, "views", path), f"{path}.views")
    if len(raw_views) != view_count:
        raise ValueError(f"{path}.view_count must equal the number of views")
    views = [
        _validate_view(
            item,
            sample_count,
            expected_view_id,
            f"{path}.views[{index}]",
        )
        for index, (item, expected_view_id) in enumerate(
            zip(raw_views, EXPECTED_VIEW_IDS)
        )
    ]

    aggregate_source = _mapping(
        _required(value, "aggregate", path),
        f"{path}.aggregate",
    )
    invalid_ids = _sequence(
        _required(aggregate_source, "invalid_view_ids", f"{path}.aggregate"),
        f"{path}.aggregate.invalid_view_ids",
    )
    if invalid_ids:
        raise ValueError(f"{path}.aggregate.invalid_view_ids must be empty")
    domains_source = _mapping(
        _required(aggregate_source, "domains", f"{path}.aggregate"),
        f"{path}.aggregate.domains",
    )
    masks_source = _mapping(
        _required(aggregate_source, "mask_conditioned", f"{path}.aggregate"),
        f"{path}.aggregate.mask_conditioned",
    )
    aggregate_supports: dict[str, dict[str, dict[str, Any]]] = {}
    for domain in DOMAINS:
        aggregate_supports[domain] = {
            "all": _validate_aggregate_support(
                _mapping(
                    _required(domains_source, domain, f"{path}.aggregate.domains"),
                    f"{path}.aggregate.domains.{domain}",
                ),
                views,
                sample_count,
                domain,
                "all",
                f"{path}.aggregate.domains.{domain}",
            )
        }
        for public_name, private_name in MASKS.items():
            mask_source = _mapping(
                _required(
                    masks_source,
                    private_name,
                    f"{path}.aggregate.mask_conditioned",
                ),
                f"{path}.aggregate.mask_conditioned.{private_name}",
            )
            aggregate_supports[domain][public_name] = _validate_aggregate_support(
                _mapping(
                    _required(
                        mask_source,
                        domain,
                        f"{path}.aggregate.mask_conditioned.{private_name}",
                    ),
                    f"{path}.aggregate.mask_conditioned.{private_name}.{domain}",
                ),
                views,
                sample_count,
                domain,
                public_name,
                f"{path}.aggregate.mask_conditioned.{private_name}.{domain}",
            )

    return {
        "sample_count_per_centerline_hit": sample_count,
        "views": views,
        "aggregate_supports": aggregate_supports,
    }


def _validate_cross_report_compatibility(
    reports: Sequence[Mapping[str, Any]],
) -> None:
    baseline = reports[0]
    for report in reports[1:]:
        for view_id in EXPECTED_VIEW_IDS:
            baseline_view = baseline["views"][view_id]
            candidate_view = report["views"][view_id]
            for field in ("source", "configuration", "upstream_view_contract"):
                if candidate_view[field] != baseline_view[field]:
                    raise ValueError(
                        f"view {view_id} {field} is incompatible across sample counts"
                    )
            for domain in DOMAINS:
                for context in CONTEXTS:
                    baseline_support = baseline_view["supports"][domain][context]
                    candidate_support = candidate_view["supports"][domain][context]
                    for field in (
                        "ray_count",
                        "centerline_hit_count",
                        "centerline_miss_count",
                    ):
                        if candidate_support[field] != baseline_support[field]:
                            raise ValueError(
                                f"view {view_id} {domain}/{context} {field} "
                                "is incompatible across sample counts"
                            )


def _validate_public_payload(value: Mapping[str, Any]) -> None:
    _validate_json_tree(value, "public_summary")

    def walk(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if key in _FORBIDDEN_PUBLIC_KEYS or key.endswith("_sha256"):
                    raise ValueError(f"{path}.{key} is forbidden in the public payload")
                walk(child, f"{path}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                walk(child, f"{path}[{index}]")
        elif isinstance(item, str):
            if any(pattern.search(item) for pattern in _PRIVATE_TEXT_PATTERNS):
                raise ValueError(f"{path} contains private or executable text")

    walk(value, "public_summary")


def build_public_summary(
    qmc8_report: Mapping[str, Any],
    qmc16_report: Mapping[str, Any],
    qmc32_report: Mapping[str, Any],
) -> dict[str, Any]:
    private_reports = (qmc8_report, qmc16_report, qmc32_report)
    validated = [
        _validate_private_report(
            _mapping(report, f"qmc{sample_count}_report"),
            sample_count,
            f"qmc{sample_count}_report",
        )
        for report, sample_count in zip(private_reports, EXPECTED_SAMPLE_COUNTS)
    ]
    _validate_cross_report_compatibility(validated)

    support_sensitivity = []
    for report in validated:
        support_sensitivity.append(
            {
                "sample_count_per_centerline_hit": report[
                    "sample_count_per_centerline_hit"
                ],
                "aggregate": _public_support_tree(report["aggregate_supports"]),
                "views": [
                    {
                        "view_id_zero_based": view["view_id_zero_based"],
                        "support": _public_support_tree(view["supports"]),
                    }
                    for view in report["views"]
                ],
            }
        )

    public = {
        "schema_version": PUBLIC_SCHEMA_VERSION,
        "source_schema_version": PRIVATE_SCHEMA_VERSION,
        "source_view_schema_version": PRIVATE_VIEW_SCHEMA_VERSION,
        "execution_status": EXECUTION_STATUS,
        "scientific_verdict": PUBLIC_VERDICT,
        "status": PUBLIC_STATUS,
        "view_count": len(EXPECTED_VIEW_IDS),
        "sample_counts_per_centerline_hit": list(EXPECTED_SAMPLE_COUNTS),
        "fixed_denominator_policy": NORMALIZATION_POLICY,
        "support_sensitivity": support_sensitivity,
        "claim_boundary": {
            "supported_claim": (
                "DISCRETE_DETERMINISTIC_APERTURE_SUPPORT_SAMPLE_COUNT_SENSITIVITY_ONLY"
            ),
            "continuous_aperture_containment": "UNCONFIRMED",
            "zero_extension": "UNCONFIRMED",
            "b1_physical_cone_semantics": "UNCONFIRMED",
            "any_ood_ray_count_is_sample_resolution_sensitive": True,
            "b3_geometry_safe_mask": "LOCKED",
            "held_out_reprojection": "LOCKED",
            "training": "LOCKED",
            "algorithm_superiority": "LOCKED",
        },
        "public_export_policy": {
            "strict_field_allowlist": True,
            "aggregate_and_per_view_support_sensitivity_only": True,
            "contains_runtime": False,
            "contains_paths": False,
            "contains_raw_sample_design_arrays": False,
            "contains_private_hashes_or_indices": False,
            "contains_private_provenance": False,
        },
    }
    _validate_public_payload(public)
    return public


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is not allowed: {value}")


def load_private_report(path: Path) -> Mapping[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_nonstandard_json_constant,
    )
    return _mapping(value, "report")


def write_json_atomic(report: Mapping[str, Any], output_path: Path) -> None:
    _validate_public_payload(report)
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


def export_public_summary(
    qmc8_path: Path,
    qmc16_path: Path,
    qmc32_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    input_paths = (qmc8_path, qmc16_path, qmc32_path)
    resolved_inputs = [path.resolve() for path in input_paths]
    if len(set(resolved_inputs)) != len(resolved_inputs):
        raise ValueError("QMC input paths must be distinct")
    if output_path.resolve() in resolved_inputs:
        raise ValueError("input and output paths must differ")
    report = build_public_summary(*(load_private_report(path) for path in input_paths))
    write_json_atomic(report, output_path)
    return report


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    private_root = (
        repo_root / "private_library" / "external_datasets" / "psu_bost_flight_body"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qmc8",
        type=Path,
        default=private_root / "aperture_domain_qmc8_audit.json",
    )
    parser.add_argument(
        "--qmc16",
        type=Path,
        default=private_root / "aperture_domain_qmc16_audit.json",
    )
    parser.add_argument(
        "--qmc32",
        type=Path,
        default=private_root / "aperture_domain_qmc32_audit.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_public_summary(args.qmc8, args.qmc16, args.qmc32, args.output)
    print(f"wrote public aperture sensitivity summary: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
