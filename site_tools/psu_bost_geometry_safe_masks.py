"""Source-independent B3 geometry-safe mask policy primitives.

B2 applies the sample-level domain indicator while retaining the original
fixed sample-count denominator.  B3 is a separate row-selection decision:
these helpers decide which centerline-hit rows remain eligible without
relabeling excluded active rows as inactive.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


CONTRACT_VERSION = "psu-bost-geometry-safe-masks-1.0"
FIXED_DENOMINATOR_POLICY = (
    "B2_INDICATOR_WITH_FIXED_ORIGINAL_SAMPLE_COUNT_NO_SURVIVOR_RENORMALIZATION"
)

POLICY_DEFINITIONS: dict[str, dict[str, str]] = {
    "indicator_keep": {
        "selection": "keep every centerline hit, including empty and partial support",
        "b2_interaction": (
            "sample-level domain indicators suppress out-of-domain contributions"
        ),
        "denominator": "the original fixed sample_count remains unchanged",
    },
    "drop_empty": {
        "selection": (
            "keep centerline hits with at least one retained in-domain sample"
        ),
        "exclusion": "exclude centerline misses and zero-retained-sample rows",
    },
    "drop_any_out": {
        "selection": (
            "keep centerline hits only when all fixed S samples are in domain"
        ),
        "exclusion": "exclude centerline misses, empty rows, and partial-support rows",
    },
    "support_floor": {
        "selection": (
            "keep centerline hits whose retained count is at least "
            "ceil(predeclared_floor * S)"
        ),
        "declaration": (
            "the finite floor is passed explicitly and is not tuned or selected"
        ),
    },
}

STANDARD_POLICIES = ("indicator_keep", "drop_empty", "drop_any_out")


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _validated_inputs(
    centerline_hit: Any,
    retained_sample_count: Any,
    sample_count: Any,
) -> tuple[np.ndarray, np.ndarray, int]:
    hit = np.asarray(centerline_hit)
    if hit.ndim != 1:
        raise ValueError("centerline_hit must have shape (N,)")
    if hit.dtype != np.dtype(np.bool_):
        raise ValueError("centerline_hit must contain boolean values")

    retained = np.asarray(retained_sample_count)
    if retained.ndim != 1:
        raise ValueError("retained_sample_count must have shape (N,)")
    if retained.shape != hit.shape:
        raise ValueError(
            "centerline_hit and retained_sample_count must have the same shape"
        )
    if not np.issubdtype(retained.dtype, np.integer) or np.issubdtype(
        retained.dtype, np.bool_
    ):
        raise ValueError("retained_sample_count must contain integer values")

    fixed_count = _positive_integer(sample_count, "sample_count")
    retained_int64 = np.ascontiguousarray(retained, dtype=np.int64)
    if np.any((retained_int64 < 0) | (retained_int64 > fixed_count)):
        raise ValueError(
            "retained_sample_count values must lie in the closed interval "
            "[0, sample_count]"
        )
    return (
        np.ascontiguousarray(hit, dtype=np.bool_),
        retained_int64,
        fixed_count,
    )


def _validated_support_floor(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("support_floor must be a finite float in [0, 1]")
    try:
        floor = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("support_floor must be a finite float in [0, 1]") from exc
    if not np.isfinite(floor) or floor < 0.0 or floor > 1.0:
        raise ValueError("support_floor must be a finite float in [0, 1]")
    return floor


def _validated_support_floors(values: Any) -> tuple[float, ...]:
    if values is None:
        return ()
    try:
        array = np.asarray(values)
    except ValueError as exc:
        raise ValueError("support_floors must be one-dimensional") from exc
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise ValueError("support_floors must be one-dimensional")
    floors = tuple(_validated_support_floor(value) for value in array)
    if len(set(floors)) != len(floors):
        raise ValueError("support_floors must not contain duplicate declarations")
    return floors


def minimum_retained_sample_count(
    policy: str,
    sample_count: Any,
    *,
    support_floor: Any | None = None,
) -> int:
    """Return the exact retained-sample threshold for one declared policy."""

    fixed_count = _positive_integer(sample_count, "sample_count")
    if policy == "indicator_keep":
        threshold = 0
    elif policy == "drop_empty":
        threshold = 1
    elif policy == "drop_any_out":
        threshold = fixed_count
    elif policy == "support_floor":
        if support_floor is None:
            raise ValueError("support_floor is required for the support_floor policy")
        floor = _validated_support_floor(support_floor)
        threshold = math.ceil(floor * fixed_count)
    else:
        allowed = ", ".join((*STANDARD_POLICIES, "support_floor"))
        raise ValueError(f"policy must be one of: {allowed}")

    if policy != "support_floor" and support_floor is not None:
        raise ValueError("support_floor is only valid for the support_floor policy")
    return threshold


def geometry_safe_keep_mask(
    centerline_hit: Any,
    retained_sample_count: Any,
    sample_count: Any,
    *,
    policy: str,
    support_floor: Any | None = None,
) -> np.ndarray:
    """Return a boolean B3 keep mask for one explicit policy."""

    hit, retained, fixed_count = _validated_inputs(
        centerline_hit,
        retained_sample_count,
        sample_count,
    )
    threshold = minimum_retained_sample_count(
        policy,
        fixed_count,
        support_floor=support_floor,
    )
    return np.ascontiguousarray(hit & (retained >= threshold), dtype=np.bool_)


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _policy_name(policy: str, support_floor: float | None) -> str:
    if policy != "support_floor":
        return policy
    assert support_floor is not None
    return f"support_floor_{support_floor}"


def _policy_report(
    *,
    hit: np.ndarray,
    retained: np.ndarray,
    sample_count: int,
    policy: str,
    support_floor: float | None,
) -> dict[str, Any]:
    threshold = minimum_retained_sample_count(
        policy,
        sample_count,
        support_floor=support_floor,
    )
    keep = np.ascontiguousarray(hit & (retained >= threshold), dtype=np.bool_)
    centerline_miss = ~hit
    excluded = ~keep
    excluded_centerline_miss = excluded & centerline_miss
    excluded_empty = excluded & hit & (retained == 0)
    excluded_partial = excluded & hit & (retained > 0)

    ray_count = int(hit.size)
    hit_count = int(np.count_nonzero(hit))
    excluded_count = int(np.count_nonzero(excluded))
    counts = {
        "ray_count": ray_count,
        "centerline_hit_count": hit_count,
        "centerline_miss_count": int(np.count_nonzero(centerline_miss)),
        "centerline_hit_empty_count": int(np.count_nonzero(hit & (retained == 0))),
        "centerline_hit_partial_count": int(
            np.count_nonzero(hit & (retained > 0) & (retained < sample_count))
        ),
        "centerline_hit_full_count": int(
            np.count_nonzero(hit & (retained == sample_count))
        ),
        "kept_count": int(np.count_nonzero(keep)),
        "excluded_count": excluded_count,
        "excluded_centerline_miss_count": int(
            np.count_nonzero(excluded_centerline_miss)
        ),
        "excluded_empty_count": int(np.count_nonzero(excluded_empty)),
        "excluded_partial_or_support_floor_count": int(
            np.count_nonzero(excluded_partial)
        ),
    }
    if (
        counts["excluded_centerline_miss_count"]
        + counts["excluded_empty_count"]
        + counts["excluded_partial_or_support_floor_count"]
        != excluded_count
    ):
        raise RuntimeError("B3 exclusion categories must be disjoint and exhaustive")

    return {
        "name": _policy_name(policy, support_floor),
        "policy": policy,
        "definition": dict(POLICY_DEFINITIONS[policy]),
        "support_floor": support_floor,
        "minimum_retained_sample_count": threshold,
        "sample_count": sample_count,
        "keep_mask": keep,
        "counts": counts,
        "fractions": {
            "kept_of_all_rays": _fraction(counts["kept_count"], ray_count),
            "kept_of_centerline_hits": _fraction(counts["kept_count"], hit_count),
            "excluded_of_all_rays": _fraction(excluded_count, ray_count),
            "excluded_centerline_miss_of_all_rays": _fraction(
                counts["excluded_centerline_miss_count"], ray_count
            ),
            "excluded_empty_of_all_rays": _fraction(
                counts["excluded_empty_count"], ray_count
            ),
            "excluded_partial_or_support_floor_of_all_rays": _fraction(
                counts["excluded_partial_or_support_floor_count"], ray_count
            ),
        },
        "normalization_policy": FIXED_DENOMINATOR_POLICY,
        "subset_only_no_active_to_inactive_relabel": True,
    }


def _validated_original_mask_indices(value: Any, ray_count: int) -> np.ndarray:
    indices = np.asarray(value)
    if indices.ndim != 1:
        raise ValueError("original_mask_indices must have shape (M,)")
    if indices.size == 0:
        return np.empty(0, dtype=np.int64)
    if not np.issubdtype(indices.dtype, np.integer) or np.issubdtype(
        indices.dtype, np.bool_
    ):
        raise ValueError("original_mask_indices must contain integer values")
    result = np.ascontiguousarray(indices, dtype=np.int64)
    if result.size > 1 and np.any(np.diff(result) <= 0):
        raise ValueError(
            "original_mask_indices must be sorted and contain unique indices"
        )
    if np.any((result < 0) | (result >= ray_count)):
        raise ValueError("original_mask_indices must lie in the ray index range")
    return result


def subset_original_mask_indices(
    original_mask_indices: Any,
    centerline_hit: Any,
    retained_sample_count: Any,
    sample_count: Any,
    *,
    policy: str,
    support_floor: Any | None = None,
) -> dict[str, Any]:
    """Apply one B3 policy as a subset-only operation on sorted mask indices."""

    hit, retained, fixed_count = _validated_inputs(
        centerline_hit,
        retained_sample_count,
        sample_count,
    )
    indices = _validated_original_mask_indices(original_mask_indices, int(hit.size))
    threshold = minimum_retained_sample_count(
        policy,
        fixed_count,
        support_floor=support_floor,
    )
    declared_floor = (
        _validated_support_floor(support_floor) if policy == "support_floor" else None
    )

    indexed_hit = hit[indices]
    indexed_retained = retained[indices]
    indexed_keep = indexed_hit & (indexed_retained >= threshold)
    kept = np.ascontiguousarray(indices[indexed_keep], dtype=np.int64)
    excluded = np.ascontiguousarray(indices[~indexed_keep], dtype=np.int64)
    centerline_miss = np.ascontiguousarray(
        indices[~indexed_hit],
        dtype=np.int64,
    )
    empty = np.ascontiguousarray(
        indices[indexed_hit & (indexed_retained == 0) & ~indexed_keep],
        dtype=np.int64,
    )
    partial = np.ascontiguousarray(
        indices[indexed_hit & (indexed_retained > 0) & ~indexed_keep],
        dtype=np.int64,
    )

    original_count = int(indices.size)
    excluded_count = int(excluded.size)
    counts = {
        "original_count": original_count,
        "kept_count": int(kept.size),
        "excluded_count": excluded_count,
        "excluded_centerline_miss_count": int(centerline_miss.size),
        "excluded_empty_count": int(empty.size),
        "excluded_partial_or_support_floor_count": int(partial.size),
    }
    if (
        counts["excluded_centerline_miss_count"]
        + counts["excluded_empty_count"]
        + counts["excluded_partial_or_support_floor_count"]
        != excluded_count
    ):
        raise RuntimeError("mask exclusion categories must be disjoint and exhaustive")

    return {
        "policy": policy,
        "policy_name": _policy_name(policy, declared_floor),
        "support_floor": declared_floor,
        "minimum_retained_sample_count": threshold,
        "sample_count": fixed_count,
        "original_indices": indices.copy(),
        "kept_indices": kept,
        "excluded_indices": excluded,
        "exclusions": {
            "centerline_miss_indices": centerline_miss,
            "empty_indices": empty,
            "partial_or_support_floor_indices": partial,
        },
        "counts": counts,
        "fractions": {
            "kept_of_original": _fraction(counts["kept_count"], original_count),
            "excluded_of_original": _fraction(excluded_count, original_count),
            "excluded_centerline_miss_of_original": _fraction(
                counts["excluded_centerline_miss_count"], original_count
            ),
            "excluded_empty_of_original": _fraction(
                counts["excluded_empty_count"], original_count
            ),
            "excluded_partial_or_support_floor_of_original": _fraction(
                counts["excluded_partial_or_support_floor_count"], original_count
            ),
        },
        "semantics": {
            "operation": "SUBSET_ONLY",
            "preserves_sorted_int64_indices": True,
            "excluded_active_indices_relabelled_as_inactive": False,
        },
    }


def subset_mask_indices(
    original_mask_indices: Any,
    centerline_hit: Any,
    retained_sample_count: Any,
    sample_count: Any,
    *,
    policy: str,
    support_floor: Any | None = None,
) -> dict[str, Any]:
    """Compatibility name for :func:`subset_original_mask_indices`."""

    return subset_original_mask_indices(
        original_mask_indices,
        centerline_hit,
        retained_sample_count,
        sample_count,
        policy=policy,
        support_floor=support_floor,
    )


def build_geometry_safe_masks(
    centerline_hit: Any,
    retained_sample_count: Any,
    sample_count: Any,
    *,
    support_floors: Any = (),
    original_mask_indices: Any | None = None,
) -> dict[str, Any]:
    """Build the fixed B3 policy family and optional original-mask subsets."""

    hit, retained, fixed_count = _validated_inputs(
        centerline_hit,
        retained_sample_count,
        sample_count,
    )
    floors = _validated_support_floors(support_floors)
    declarations = [(policy, None) for policy in STANDARD_POLICIES] + [
        ("support_floor", floor) for floor in floors
    ]
    policy_reports = [
        _policy_report(
            hit=hit,
            retained=retained,
            sample_count=fixed_count,
            policy=policy,
            support_floor=floor,
        )
        for policy, floor in declarations
    ]
    policies = {report["name"]: report for report in policy_reports}

    subsets = None
    if original_mask_indices is not None:
        subsets = {
            report["name"]: subset_original_mask_indices(
                original_mask_indices,
                hit,
                retained,
                fixed_count,
                policy=report["policy"],
                support_floor=report["support_floor"],
            )
            for report in policy_reports
        }

    return {
        "contract_version": CONTRACT_VERSION,
        "source_independent": True,
        "sample_count": fixed_count,
        "ray_count": int(hit.size),
        "policy_order": tuple(report["name"] for report in policy_reports),
        "policy_definitions": {
            name: dict(definition) for name, definition in POLICY_DEFINITIONS.items()
        },
        "support_floor_declarations": [
            {
                "support_floor": floor,
                "minimum_retained_sample_count": math.ceil(floor * fixed_count),
                "conversion": "ceil(support_floor * sample_count)",
                "tuned_or_selected": False,
            }
            for floor in floors
        ],
        "policies": policies,
        "masks": {name: report["keep_mask"] for name, report in policies.items()},
        "mask_subsets": subsets,
        "normalization_policy": FIXED_DENOMINATOR_POLICY,
        "semantics": {
            "b2_and_b3_are_separate": True,
            "active_to_inactive_relabel": False,
            "original_sample_denominator_changed": False,
        },
    }
