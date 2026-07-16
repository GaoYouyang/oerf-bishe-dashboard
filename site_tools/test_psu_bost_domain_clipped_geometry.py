from __future__ import annotations

import numpy as np
import pytest

from site_tools.psu_bost_domain_clipped_geometry import (
    aggregate_domain_clipped_views,
    clip_author_intervals_to_forward_box,
)


def test_domain_clipping_preserves_fallback_and_clips_nonzero_cone() -> None:
    result = clip_author_intervals_to_forward_box(
        full_first=[1.0, 1.0, 1.0, -3.0, -3.0],
        full_second=[3.0, 3.0, 3.0, -1.0, 3.0],
        full_length=[2.0, 2.0, 2.0, 2.0, 6.0],
        cone_first=[1.5, 0.0, 4.0, np.nan, np.nan],
        cone_second=[2.5, 4.0, 5.0, np.nan, np.nan],
        cone_length=[1.0, 4.0, 1.0, 0.0, 0.0],
    )

    np.testing.assert_allclose(result["enter"], [1.5, 1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(result["exit"], [2.5, 3.0, 0.0, 0.0, 3.0])
    np.testing.assert_allclose(result["length"], [1.0, 2.0, 0.0, 0.0, 3.0])
    assert result["fallback"].tolist() == [False, False, False, True, True]
    assert result["cone_shortened"].tolist() == [False, True, False, False, False]
    assert result["cone_zeroed_for_no_box_overlap"].tolist() == [
        False,
        False,
        True,
        False,
        False,
    ]
    assert result["forward_box_shortened"].tolist() == [
        False,
        False,
        False,
        False,
        True,
    ]


def test_domain_clipping_sorts_swapped_roots_and_rejects_bad_shapes() -> None:
    result = clip_author_intervals_to_forward_box(
        full_first=[3.0],
        full_second=[1.0],
        full_length=[2.0],
        cone_first=[2.5],
        cone_second=[1.5],
        cone_length=[1.0],
    )
    assert result["enter"][0] == pytest.approx(1.5)
    assert result["exit"][0] == pytest.approx(2.5)
    assert result["length"][0] == pytest.approx(1.0)

    with pytest.raises(ValueError, match="same size"):
        clip_author_intervals_to_forward_box(
            full_first=[1.0, 2.0],
            full_second=[3.0],
            full_length=[2.0],
            cone_first=[np.nan],
            cone_second=[np.nan],
            cone_length=[0.0],
        )


def _view(view_id: int, *, invalid: bool = False, zero_rows: int = 1) -> dict:
    return {
        "view_id_zero_based": view_id,
        "status": (
            "AUTHOR_COMPATIBLE_CLIPPED_HYBRID_INVALID"
            if invalid
            else "AUTHOR_COMPATIBLE_CLIPPED_HYBRID_CONTRACT_PASS_MASK_FILTER_REQUIRED"
        ),
        "counts": {
            "changed_from_author_count": 4,
            "clipped_zero_length_count": zero_rows,
        },
        "path_length": {
            "author_length_sum_m": 10.0,
            "clipped_length_sum_m": 8.0,
        },
        "decision": {"geometry_safe_zero_row_filter_required": bool(zero_rows)},
    }


def test_all_view_aggregate_keeps_mechanical_and_scientific_verdicts_separate() -> None:
    report = aggregate_domain_clipped_views([_view(0), _view(1, zero_rows=0)])
    assert report["scientific_verdict"] == "AUTHOR_COMPATIBILITY_ABLATION_ONLY"
    assert report["status"].endswith("MASK_FILTER_REQUIRED")
    assert report["aggregate"]["zero_row_filter_required_view_ids"] == [0]
    assert report["aggregate"]["removed_fraction"] == pytest.approx(0.2)
    assert report["decision"]["algorithm_superiority_claim"] == "LOCKED"


def test_all_view_aggregate_rejects_noncontiguous_views() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        aggregate_domain_clipped_views([_view(1)])
