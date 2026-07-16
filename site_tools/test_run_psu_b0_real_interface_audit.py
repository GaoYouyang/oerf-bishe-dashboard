from __future__ import annotations

import numpy as np
import pytest

from site_tools.run_psu_b0_real_interface_audit import (
    PUBLIC_SCHEMA,
    build_public_summary,
    detector_xy_from_matlab_zero_based_indices,
    deterministic_quantile_indices,
)


def test_quantile_selection_is_ordered_unique_and_magnitude_free() -> None:
    active = np.arange(10, 110, 3, dtype=np.int64)
    selected = deterministic_quantile_indices(active, 7)
    assert selected.shape == (7,)
    assert np.all(selected[1:] > selected[:-1])
    assert set(selected).issubset(set(active))
    with pytest.raises(ValueError, match="strictly increasing"):
        deterministic_quantile_indices(active[::-1], 3)


def test_detector_coordinates_follow_matlab_column_major_order() -> None:
    coordinates = detector_xy_from_matlab_zero_based_indices(
        np.asarray([0, 2, 3, 4, 11]),
        image_height=3,
        image_width=4,
    )
    scale = 3.0
    expected = np.asarray(
        [
            [-1.5 / scale, -1.0 / scale],
            [-1.5 / scale, 1.0 / scale],
            [-0.5 / scale, -1.0 / scale],
            [-0.5 / scale, 0.0],
            [1.5 / scale, 1.0 / scale],
        ]
    )
    assert np.allclose(coordinates, expected)


def test_public_summary_strips_private_provenance() -> None:
    private = {
        "status": "REAL_SUPPORT_GEOMETRY_INTERFACE_PASS_NO_RECONSTRUCTION",
        "evidence_scope": "fixture",
        "dataset": {"doi": "10.26208/1VE2-5C19"},
        "configuration": {"total_ray_count": 9},
        "aggregate_geometry": {"selected_ray_count": 9},
        "grid_profiles": [],
        "gates": {"pass": True},
        "claim_boundary": {"algorithm_superiority": False},
        "private_view_provenance": [{"selected_index_sha256": "secret"}],
        "host": {"platform": "private"},
    }
    public = build_public_summary(private)
    assert public["schema_version"] == PUBLIC_SCHEMA
    assert "private_view_provenance" not in public
    assert "host" not in public
    assert "secret" not in str(public)
