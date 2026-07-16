from __future__ import annotations

import numpy as np

from site_tools.audit_psu_b0_real_detector_feature_domain import (
    deterministic_camera_subset_masks,
    robust_domain_comparison,
)


def test_camera_subset_masks_cover_six_through_nine_views() -> None:
    masks = deterministic_camera_subset_masks(
        view_count=9,
        minimum_active=6,
        maximum_active=9,
    )
    assert masks.shape == (130, 9)
    assert set(np.sum(masks, axis=1).astype(int)) == {6, 7, 8, 9}
    assert len({tuple(row) for row in masks}) == 130


def test_robust_domain_comparison_flags_shifted_candidates() -> None:
    rng = np.random.default_rng(10)
    reference = rng.normal(size=(60, 4))
    candidate = rng.normal(loc=5.0, size=(8, 4))
    summary = robust_domain_comparison(
        reference,
        candidate,
        feature_names=("a", "b", "c", "d"),
    )
    assert summary["informative_feature_count"] == 4
    assert summary[
        "candidate_rows_outside_any_train_95pct_feature_envelope"
    ] == 1.0
    assert summary["robust_center_distance"]["median"] > 2.0
