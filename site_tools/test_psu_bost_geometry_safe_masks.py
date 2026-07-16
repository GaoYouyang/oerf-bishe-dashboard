from __future__ import annotations

import numpy as np
import pytest

from site_tools.psu_bost_geometry_safe_masks import (
    FIXED_DENOMINATOR_POLICY,
    build_geometry_safe_masks,
    geometry_safe_keep_mask,
    minimum_retained_sample_count,
    subset_original_mask_indices,
)


HIT = np.array([True, True, True, True, False, True])
RETAINED = np.array([16, 15, 14, 0, 16, 1], dtype=np.int64)


def test_fixed_policies_separate_indicator_empty_and_any_out_behavior() -> None:
    report = build_geometry_safe_masks(
        HIT,
        RETAINED,
        16,
        support_floors=(0.875, 0.9375),
    )

    np.testing.assert_array_equal(
        report["policies"]["indicator_keep"]["keep_mask"],
        [True, True, True, True, False, True],
    )
    np.testing.assert_array_equal(
        report["policies"]["drop_empty"]["keep_mask"],
        [True, True, True, False, False, True],
    )
    np.testing.assert_array_equal(
        report["policies"]["drop_any_out"]["keep_mask"],
        [True, False, False, False, False, False],
    )
    np.testing.assert_array_equal(
        report["policies"]["support_floor_0.875"]["keep_mask"],
        [True, True, True, False, False, False],
    )
    np.testing.assert_array_equal(
        report["policies"]["support_floor_0.9375"]["keep_mask"],
        [True, True, False, False, False, False],
    )
    assert (
        report["masks"]["drop_empty"] is report["policies"]["drop_empty"]["keep_mask"]
    )

    indicator = report["policies"]["indicator_keep"]
    assert indicator["counts"]["kept_count"] == 5
    assert indicator["counts"]["excluded_centerline_miss_count"] == 1
    assert indicator["counts"]["excluded_empty_count"] == 0
    assert indicator["counts"]["excluded_partial_or_support_floor_count"] == 0
    assert indicator["fractions"]["kept_of_all_rays"] == pytest.approx(5 / 6)
    assert indicator["normalization_policy"] == FIXED_DENOMINATOR_POLICY
    assert (
        "original fixed sample_count"
        in (report["policy_definitions"]["indicator_keep"]["denominator"])
    )

    strict = report["policies"]["drop_any_out"]
    assert strict["counts"]["centerline_hit_empty_count"] == 1
    assert strict["counts"]["centerline_hit_partial_count"] == 3
    assert strict["counts"]["centerline_hit_full_count"] == 1
    assert strict["counts"]["excluded_count"] == 5
    assert strict["counts"]["excluded_centerline_miss_count"] == 1
    assert strict["counts"]["excluded_empty_count"] == 1
    assert strict["counts"]["excluded_partial_or_support_floor_count"] == 3
    assert strict["fractions"]["kept_of_centerline_hits"] == pytest.approx(1 / 5)


def test_centerline_miss_is_never_kept_even_with_full_sample_count() -> None:
    for policy, support_floor in (
        ("indicator_keep", None),
        ("drop_empty", None),
        ("drop_any_out", None),
        ("support_floor", 0.5),
    ):
        keep = geometry_safe_keep_mask(
            [False],
            np.array([16], dtype=np.int64),
            16,
            policy=policy,
            support_floor=support_floor,
        )
        np.testing.assert_array_equal(keep, [False])


def test_support_floors_use_ceil_and_remain_sample_count_sensitive() -> None:
    assert minimum_retained_sample_count("support_floor", 8, support_floor=0.875) == 7
    assert minimum_retained_sample_count("support_floor", 16, support_floor=0.875) == 14
    assert (
        minimum_retained_sample_count("support_floor", 16, support_floor=0.9375) == 15
    )

    eight = geometry_safe_keep_mask(
        [True, True],
        np.array([6, 7]),
        8,
        policy="support_floor",
        support_floor=0.875,
    )
    sixteen = geometry_safe_keep_mask(
        [True, True],
        np.array([13, 14]),
        16,
        policy="support_floor",
        support_floor=0.875,
    )
    np.testing.assert_array_equal(eight, [False, True])
    np.testing.assert_array_equal(sixteen, [False, True])

    report = build_geometry_safe_masks(
        [True],
        np.array([15]),
        16,
        support_floors=0.9375,
    )
    assert report["support_floor_declarations"] == [
        {
            "support_floor": 0.9375,
            "minimum_retained_sample_count": 15,
            "conversion": "ceil(support_floor * sample_count)",
            "tuned_or_selected": False,
        }
    ]


def test_mask_subset_preserves_sorted_int64_and_reports_exclusions() -> None:
    original_active = np.array([0, 2, 3, 4, 5], dtype=np.int32)
    result = subset_original_mask_indices(
        original_active,
        HIT,
        RETAINED,
        16,
        policy="support_floor",
        support_floor=0.875,
    )

    np.testing.assert_array_equal(result["kept_indices"], [0, 2])
    np.testing.assert_array_equal(result["excluded_indices"], [3, 4, 5])
    np.testing.assert_array_equal(result["exclusions"]["centerline_miss_indices"], [4])
    np.testing.assert_array_equal(result["exclusions"]["empty_indices"], [3])
    np.testing.assert_array_equal(
        result["exclusions"]["partial_or_support_floor_indices"], [5]
    )
    assert result["kept_indices"].dtype == np.int64
    assert result["excluded_indices"].dtype == np.int64
    assert result["counts"] == {
        "original_count": 5,
        "kept_count": 2,
        "excluded_count": 3,
        "excluded_centerline_miss_count": 1,
        "excluded_empty_count": 1,
        "excluded_partial_or_support_floor_count": 1,
    }
    assert result["fractions"]["kept_of_original"] == pytest.approx(0.4)
    assert result["semantics"]["operation"] == "SUBSET_ONLY"
    assert result["semantics"]["excluded_active_indices_relabelled_as_inactive"] is (
        False
    )
    assert "inactive_indices" not in result


def test_all_policy_report_can_subset_without_relabeling() -> None:
    report = build_geometry_safe_masks(
        HIT,
        RETAINED,
        16,
        support_floors=(0.875,),
        original_mask_indices=[0, 1, 2, 3, 4, 5],
    )

    subsets = report["mask_subsets"]
    assert subsets is not None
    np.testing.assert_array_equal(
        subsets["indicator_keep"]["kept_indices"], [0, 1, 2, 3, 5]
    )
    np.testing.assert_array_equal(subsets["drop_empty"]["kept_indices"], [0, 1, 2, 5])
    np.testing.assert_array_equal(subsets["drop_any_out"]["kept_indices"], [0])
    np.testing.assert_array_equal(
        subsets["support_floor_0.875"]["kept_indices"], [0, 1, 2]
    )
    assert report["semantics"]["active_to_inactive_relabel"] is False


@pytest.mark.parametrize(
    ("centerline_hit", "retained", "sample_count", "match"),
    [
        ([[True]], [1], 1, "shape"),
        ([1], [1], 1, "boolean"),
        ([True], [[1]], 1, "shape"),
        ([True, False], [1], 1, "same shape"),
        ([True], [1.0], 1, "integer"),
        ([True], [-1], 1, "closed interval"),
        ([True], [2], 1, "closed interval"),
        ([True], [1], 0, "positive integer"),
        ([True], [1], True, "positive integer"),
    ],
)
def test_invalid_core_inputs_are_rejected(
    centerline_hit: object,
    retained: object,
    sample_count: object,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        build_geometry_safe_masks(
            centerline_hit,
            np.asarray(retained),
            sample_count,
        )


@pytest.mark.parametrize("floor", [np.nan, np.inf, -0.01, 1.01, True])
def test_invalid_support_floors_are_rejected(floor: object) -> None:
    with pytest.raises(ValueError, match="finite float"):
        minimum_retained_sample_count(
            "support_floor",
            16,
            support_floor=floor,
        )


def test_policy_and_floor_declarations_are_strict() -> None:
    with pytest.raises(ValueError, match="required"):
        minimum_retained_sample_count("support_floor", 16)
    with pytest.raises(ValueError, match="only valid"):
        minimum_retained_sample_count(
            "drop_empty",
            16,
            support_floor=0.5,
        )
    with pytest.raises(ValueError, match="policy must be"):
        minimum_retained_sample_count("selected_best", 16)
    with pytest.raises(ValueError, match="duplicate"):
        build_geometry_safe_masks(
            [True],
            np.array([1]),
            1,
            support_floors=(0.875, 0.875),
        )
    with pytest.raises(ValueError, match="one-dimensional"):
        build_geometry_safe_masks(
            [True],
            np.array([1]),
            1,
            support_floors=[[0.875]],
        )


@pytest.mark.parametrize(
    ("indices", "match"),
    [
        ([[0]], "shape"),
        ([0.0], "integer"),
        ([1, 0], "sorted"),
        ([0, 0], "sorted"),
        ([-1], "ray index range"),
        ([6], "ray index range"),
    ],
)
def test_invalid_original_masks_are_rejected(indices: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        subset_original_mask_indices(
            indices,
            HIT,
            RETAINED,
            16,
            policy="drop_empty",
        )


def test_empty_original_mask_is_a_valid_int64_subset() -> None:
    result = subset_original_mask_indices(
        [],
        HIT,
        RETAINED,
        16,
        policy="drop_empty",
    )
    assert result["kept_indices"].dtype == np.int64
    assert result["excluded_indices"].dtype == np.int64
    assert result["kept_indices"].size == 0
    assert result["excluded_indices"].size == 0
    assert result["counts"]["original_count"] == 0
    assert result["fractions"]["kept_of_original"] is None
