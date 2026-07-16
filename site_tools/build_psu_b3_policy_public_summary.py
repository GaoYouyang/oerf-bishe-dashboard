#!/usr/bin/env python3
"""Build a strict public B3 mask-policy sensitivity summary from PSU B2 audits."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


SOURCE_SCHEMA = "psu-bost-aperture-domain-all-view-audit-1.0"
SOURCE_VIEW_SCHEMA = "psu-bost-aperture-domain-audit-1.0"
SOURCE_STATUS = "B2_ALL_VIEW_APERTURE_DOMAIN_AUDIT_PASS_B3_REQUIRED"
SOURCE_VIEW_STATUS = "B2_DETERMINISTIC_APERTURE_DOMAIN_AUDIT_PASS_B3_REQUIRED"
PUBLIC_SCHEMA = "psu-bost-b3-policy-public-summary-1.0"
EXPECTED_SAMPLE_COUNTS = (8, 16, 32)
DOMAINS = ("B0", "B1")
MASK_KEYS = {
    "all": None,
    "active": "amask_all",
    "inactive": "imask_all",
}
SUPPORT_FLOORS = (0.875, 0.9375)
POLICY_ORDER = (
    "indicator_keep",
    "drop_empty",
    "support_floor_0.875",
    "support_floor_0.9375",
    "drop_any_out",
)


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be an object")
    return value


def _array(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{location} must be an array")
    return value


def _integer(value: Any, location: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{location} must be an integer >= {minimum}")
    return value


def _fraction(value: Any, location: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{location} must be a finite fraction")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location} must be a finite fraction") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{location} must be a finite fraction in [0, 1]")
    return result


def _required(source: Mapping[str, Any], key: str, location: str) -> Any:
    if key not in source:
        raise ValueError(f"{location} missing required field: {key}")
    return source[key]


def _read_report(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON report: {path}") from exc
    return _mapping(value, str(path))


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def _support_record(
    report: Mapping[str, Any],
    *,
    domain: str,
    category: str,
    location: str,
) -> Mapping[str, Any]:
    aggregate = _mapping(
        _required(report, "aggregate", location), f"{location}.aggregate"
    )
    mask_key = MASK_KEYS[category]
    if mask_key is None:
        domains = _mapping(
            _required(aggregate, "domains", f"{location}.aggregate"),
            f"{location}.aggregate.domains",
        )
    else:
        conditioned = _mapping(
            _required(aggregate, "mask_conditioned", f"{location}.aggregate"),
            f"{location}.aggregate.mask_conditioned",
        )
        mask_record = _mapping(
            _required(conditioned, mask_key, f"{location}.aggregate.mask_conditioned"),
            f"{location}.aggregate.mask_conditioned.{mask_key}",
        )
        domains = mask_record
    return _mapping(
        _required(domains, domain, f"{location}.{category}"),
        f"{location}.{category}.{domain}",
    )


def _view_support_record(
    view: Mapping[str, Any],
    *,
    domain: str,
    category: str,
    location: str,
) -> Mapping[str, Any]:
    mask_key = MASK_KEYS[category]
    if mask_key is None:
        domains = _mapping(_required(view, "domains", location), f"{location}.domains")
    else:
        conditioned = _mapping(
            _required(view, "mask_conditioned", location),
            f"{location}.mask_conditioned",
        )
        domains = _mapping(
            _required(conditioned, mask_key, f"{location}.mask_conditioned"),
            f"{location}.mask_conditioned.{mask_key}",
        )
    return _mapping(
        _required(domains, domain, f"{location}.{category}"),
        f"{location}.{category}.{domain}",
    )


def _validate_histogram(
    record: Mapping[str, Any],
    *,
    sample_count: int,
    location: str,
) -> tuple[list[int], int, int, int, float]:
    hit_count = _integer(
        _required(record, "centerline_hit_count", location),
        f"{location}.centerline_hit_count",
    )
    miss_count = _integer(
        _required(record, "centerline_miss_count", location),
        f"{location}.centerline_miss_count",
    )
    ray_count = _integer(
        _required(record, "ray_count", location),
        f"{location}.ray_count",
    )
    if hit_count + miss_count != ray_count:
        raise ValueError(f"{location} hit/miss counts do not reconcile")
    histogram_raw = _array(
        _required(record, "retained_sample_count_histogram", location),
        f"{location}.retained_sample_count_histogram",
    )
    if len(histogram_raw) != sample_count + 1:
        raise ValueError(f"{location} histogram length must equal sample_count + 1")
    histogram = [
        _integer(value, f"{location}.histogram[{index}]")
        for index, value in enumerate(histogram_raw)
    ]
    if sum(histogram) != hit_count:
        raise ValueError(f"{location} histogram does not sum to centerline hits")
    retained_fraction = _fraction(
        _required(record, "fixed_denominator_sample_retained_fraction", location),
        f"{location}.fixed_denominator_sample_retained_fraction",
    )
    retained_samples = sum(index * count for index, count in enumerate(histogram))
    expected_fraction = (
        retained_samples / (hit_count * sample_count) if hit_count else 0.0
    )
    if not math.isclose(
        retained_fraction,
        expected_fraction,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{location} fixed-denominator fraction is inconsistent")
    return histogram, hit_count, miss_count, ray_count, retained_fraction


def _policy_thresholds(sample_count: int) -> dict[str, int]:
    return {
        "indicator_keep": 0,
        "drop_empty": 1,
        "support_floor_0.875": math.ceil(SUPPORT_FLOORS[0] * sample_count),
        "support_floor_0.9375": math.ceil(SUPPORT_FLOORS[1] * sample_count),
        "drop_any_out": sample_count,
    }


def _policy_summary(
    *,
    histogram: Sequence[int],
    hit_count: int,
    miss_count: int,
    ray_count: int,
    sample_count: int,
) -> dict[str, Any]:
    thresholds = _policy_thresholds(sample_count)
    policies: dict[str, Any] = {}
    for name in POLICY_ORDER:
        threshold = thresholds[name]
        kept = int(sum(histogram[threshold:]))
        excluded_hits = hit_count - kept
        excluded_empty = int(histogram[0]) if threshold >= 1 else 0
        excluded_partial = excluded_hits - excluded_empty
        policies[name] = {
            "minimum_retained_sample_count": threshold,
            "kept_count": kept,
            "excluded_count": ray_count - kept,
            "excluded_centerline_miss_count": miss_count,
            "excluded_empty_count": excluded_empty,
            "excluded_partial_or_support_floor_count": excluded_partial,
            "kept_fraction_of_all_rows": kept / ray_count if ray_count else None,
            "kept_fraction_of_centerline_hits": kept / hit_count if hit_count else None,
        }
    return policies


def _extract_category(
    record: Mapping[str, Any],
    *,
    sample_count: int,
    location: str,
) -> dict[str, Any]:
    histogram, hit_count, miss_count, ray_count, retained_fraction = (
        _validate_histogram(
            record,
            sample_count=sample_count,
            location=location,
        )
    )
    return {
        "ray_count": ray_count,
        "centerline_hit_count": hit_count,
        "centerline_miss_count": miss_count,
        "b2_fixed_denominator_retained_sample_fraction": retained_fraction,
        "policies": _policy_summary(
            histogram=histogram,
            hit_count=hit_count,
            miss_count=miss_count,
            ray_count=ray_count,
            sample_count=sample_count,
        ),
    }


def _validate_report(
    report: Mapping[str, Any],
    *,
    expected_sample_count: int,
    expected_view_ids: list[int] | None,
    location: str,
) -> tuple[list[Mapping[str, Any]], list[int]]:
    if _required(report, "schema_version", location) != SOURCE_SCHEMA:
        raise ValueError(f"{location} has an unsupported schema")
    if _required(report, "status", location) != SOURCE_STATUS:
        raise ValueError(f"{location} is not a passing B2 report")
    sample_count = _integer(
        _required(report, "sample_count_per_centerline_hit", location),
        f"{location}.sample_count_per_centerline_hit",
        minimum=1,
    )
    if sample_count != expected_sample_count:
        raise ValueError(
            f"{location} sample count must be {expected_sample_count}, got {sample_count}"
        )
    views_raw = _array(_required(report, "views", location), f"{location}.views")
    views = [
        _mapping(value, f"{location}.views[{index}]")
        for index, value in enumerate(views_raw)
    ]
    view_ids = [
        _integer(
            _required(view, "view_id_zero_based", f"{location}.views[{index}]"),
            f"{location}.views[{index}].view_id_zero_based",
        )
        for index, view in enumerate(views)
    ]
    if view_ids != list(range(len(views))):
        raise ValueError(f"{location} view ids must be ordered and contiguous")
    if expected_view_ids is not None and view_ids != expected_view_ids:
        raise ValueError("all QMC reports must contain the same ordered views")
    for index, view in enumerate(views):
        view_location = f"{location}.views[{index}]"
        if _required(view, "schema_version", view_location) != SOURCE_VIEW_SCHEMA:
            raise ValueError(f"{view_location} has an unsupported schema")
        if _required(view, "status", view_location) != SOURCE_VIEW_STATUS:
            raise ValueError(f"{view_location} is not a passing B2 view")
    return views, view_ids


def _range(values: Sequence[float]) -> dict[str, float]:
    return {
        "minimum": min(values),
        "maximum": max(values),
        "range": max(values) - min(values),
    }


def build_public_summary(report_paths: Sequence[Path]) -> dict[str, Any]:
    if len(report_paths) != len(EXPECTED_SAMPLE_COUNTS):
        raise ValueError("exactly three QMC reports are required")
    reports = [_read_report(path) for path in report_paths]
    sensitivity: list[dict[str, Any]] = []
    expected_view_ids: list[int] | None = None

    for report, sample_count, path in zip(
        reports,
        EXPECTED_SAMPLE_COUNTS,
        report_paths,
    ):
        views, view_ids = _validate_report(
            report,
            expected_sample_count=sample_count,
            expected_view_ids=expected_view_ids,
            location=path.name,
        )
        expected_view_ids = view_ids
        categories = {
            domain: {
                category: _extract_category(
                    _support_record(
                        report,
                        domain=domain,
                        category=category,
                        location=path.name,
                    ),
                    sample_count=sample_count,
                    location=f"{path.name}.{domain}.{category}",
                )
                for category in MASK_KEYS
            }
            for domain in DOMAINS
        }
        active_b1_views = []
        for view_id, view in zip(view_ids, views):
            active_b1_views.append(
                {
                    "view_id_zero_based": view_id,
                    **_extract_category(
                        _view_support_record(
                            view,
                            domain="B1",
                            category="active",
                            location=f"{path.name}.view_{view_id}",
                        ),
                        sample_count=sample_count,
                        location=f"{path.name}.view_{view_id}.B1.active",
                    ),
                }
            )
        sensitivity.append(
            {
                "sample_count_per_centerline_hit": sample_count,
                "domains": categories,
                "active_b1_per_view": active_b1_views,
            }
        )

    active_b1_weight = [
        item["domains"]["B1"]["active"]["b2_fixed_denominator_retained_sample_fraction"]
        for item in sensitivity
    ]
    active_b1_policy_ranges = {}
    for policy in POLICY_ORDER:
        values = [
            item["domains"]["B1"]["active"]["policies"][policy][
                "kept_fraction_of_centerline_hits"
            ]
            for item in sensitivity
        ]
        active_b1_policy_ranges[policy] = _range(values)

    return {
        "schema_version": PUBLIC_SCHEMA,
        "status": "B3_POLICY_SENSITIVITY_COMPLETE_HELD_OUT_SELECTION_REQUIRED",
        "execution_status": "COMPLETE",
        "scientific_verdict": (
            "MASK_POLICY_SENSITIVITY_ONLY_NO_RECONSTRUCTION_OR_POLICY_SUPERIORITY"
        ),
        "view_count": len(expected_view_ids or []),
        "sample_counts_per_centerline_hit": list(EXPECTED_SAMPLE_COUNTS),
        "policy_order": list(POLICY_ORDER),
        "policy_contract": {
            "indicator_keep": "keep every centerline hit; B2 zero-extends out-of-domain samples under the original fixed denominator",
            "drop_empty": "drop centerline hits with zero retained aperture samples",
            "support_floor_0.875": "keep when retained count is at least ceil(0.875 * S); threshold predeclared",
            "support_floor_0.9375": "keep when retained count is at least ceil(0.9375 * S); threshold predeclared",
            "drop_any_out": "keep only when all S aperture samples are in domain",
            "subset_only_no_active_to_inactive_relabel": True,
            "survivor_renormalization": False,
        },
        "sensitivity": sensitivity,
        "headline_metrics": {
            "active_b1_fixed_denominator_retained_sample_fraction": {
                **_range(active_b1_weight),
                "values_by_sample_count": dict(
                    zip(map(str, EXPECTED_SAMPLE_COUNTS), active_b1_weight)
                ),
            },
            "active_b1_policy_kept_fraction_ranges": active_b1_policy_ranges,
            "qmc_designs_are_nested": False,
            "drop_any_out_is_sample_design_sensitive": True,
            "support_floor_is_a_diagnostic_family_not_a_selected_hyperparameter": True,
        },
        "decision": {
            "default_b3_policy_selected": False,
            "least_assumptive_reference": "B0_INDICATOR_KEEP",
            "drop_empty_is_numerical_cleanup_not_a_physical_domain_claim": True,
            "b1_physical_support_semantics_confirmed": False,
            "held_out_reprojection_required_for_policy_selection": True,
            "training_ready": "NO",
            "algorithm_superiority_claim": "LOCKED",
            "next_gate": (
                "HELD_OUT_CAMERA_REPROJECTION_AND_FLOW_OFF_UNCERTAINTY_BEFORE_"
                "INVERSE_OR_NEURAL_OPERATOR_COMPARISON"
            ),
        },
        "limitations": [
            "the 8, 16, and 32 point deterministic designs are not nested and do not define confidence intervals",
            "discrete sample support does not prove continuous aperture containment",
            "support floors are predeclared diagnostics and are not selected from these results",
            "B1 remains an unconfirmed one-nappe computational support hypothesis",
            "no held-out camera, density truth, inverse reconstruction, training, or superiority comparison is included",
        ],
        "public_export_policy": {
            "strict_field_allowlist": True,
            "contains_paths": False,
            "contains_private_hashes_or_indices": False,
            "contains_private_provenance": False,
            "contains_raw_histograms_or_sample_arrays": False,
            "contains_runtime": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qmc8", type=Path, required=True)
    parser.add_argument("--qmc16", type=Path, required=True)
    parser.add_argument("--qmc32", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = build_public_summary((args.qmc8, args.qmc16, args.qmc32))
    _atomic_json(args.output, summary)
    print(f"wrote public B3 policy summary: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
