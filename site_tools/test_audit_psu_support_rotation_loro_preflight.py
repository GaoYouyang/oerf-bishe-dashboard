from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from site_tools.audit_psu_support_rotation_loro_preflight import (
    PUBLIC_SCHEMA,
    build_public_summary,
    cross_grid_coordinate_diagnostics,
    expected_view_mapping,
)


def test_expected_mapping_is_rotation_outer_camera_inner() -> None:
    assert expected_view_mapping() == [
        {"view_id_zero_based": 0, "rotation_degrees": 0, "camera_id": 2},
        {"view_id_zero_based": 1, "rotation_degrees": 0, "camera_id": 3},
        {"view_id_zero_based": 2, "rotation_degrees": 0, "camera_id": 4},
        {"view_id_zero_based": 3, "rotation_degrees": 50, "camera_id": 2},
        {"view_id_zero_based": 4, "rotation_degrees": 50, "camera_id": 3},
        {"view_id_zero_based": 5, "rotation_degrees": 50, "camera_id": 4},
        {"view_id_zero_based": 6, "rotation_degrees": 90, "camera_id": 2},
        {"view_id_zero_based": 7, "rotation_degrees": 90, "camera_id": 3},
        {"view_id_zero_based": 8, "rotation_degrees": 90, "camera_id": 4},
    ]


def test_public_summary_strips_private_hashes_and_paths() -> None:
    private = {
        "status": "SUPPORT_ROTATION_LORO_PREFLIGHT_PASS",
        "evidence_scope": "fixture",
        "source_script_audit": {
            "mapping": expected_view_mapping(),
            "gates": {"source": True},
            "source_files": {"secret": {"sha256": "a" * 64}},
        },
        "view_manifest_audit": {
            "gates": {"views": True},
            "manifest_sha256_private_only": {"view_00": "b" * 64},
        },
        "cache_audit": {
            "gates": {"cache": True},
            "cache_rows": [{"grid_shape_zyx": [16, 16, 16]}],
            "selection": {"selection_mode": "all_active_rows"},
            "diagnostics": {"common_array_hash_equal": {"valid": True}},
            "cache_manifest_sha256_private_only": {"16_cubed": "c" * 64},
        },
    }
    public = build_public_summary(private)
    encoded = json.dumps(public, sort_keys=True)
    assert public["schema_version"] == PUBLIC_SCHEMA
    assert "sha256" not in encoded
    assert "private_library" not in encoded
    assert public["claim_boundary"]["cgls_loro_scores_generated"] is False
    assert (
        public["claim_boundary"][
            "compact_cache_manifest_self_contains_camera_rotation_identity"
        ]
        is False
    )


def test_cross_grid_coordinates_reconstruct_the_same_points() -> None:
    valid = np.array([[True, False]], dtype=np.bool_)
    chunks = [
        {
            "chunk_index": 0,
            "start_index": 0,
            "stop_index": 1,
            "view_id": 0,
            "b0_hit_count": 1,
        }
    ]
    cache16 = SimpleNamespace(
        manifest={"chunks": chunks},
        valid=valid,
        grid_shape=(4, 4, 4),
        grid_minimum_xyz=(-1.0, -1.0, -1.0),
        grid_maximum_xyz=(1.0, 1.0, 1.0),
        base_indices=np.array([[20, 0]], dtype=np.uint16),
        fractions_xyz=np.array([[[0.6, 0.2, 0.8], [0.0, 0.0, 0.0]]]),
    )
    cache32 = SimpleNamespace(
        manifest={"chunks": chunks},
        valid=valid.copy(),
        grid_shape=(6, 6, 6),
        grid_minimum_xyz=(-1.0, -1.0, -1.0),
        grid_maximum_xyz=(1.0, 1.0, 1.0),
        base_indices=np.array([[121, 0]], dtype=np.uint16),
        fractions_xyz=np.zeros((1, 2, 3), dtype=np.float64),
    )
    report = cross_grid_coordinate_diagnostics(cache16, cache32)
    assert report["valid_sample_count"] == 1
    assert report["coordinate_component_count"] == 3
    assert report["normalized_coordinate_max_abs"] < 1e-15
    assert report["physical_coordinate_max_abs_m"] < 2e-15
