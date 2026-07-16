import json

import pytest

from site_tools.build_psu_public_summary import build_public_summary


def _loader():
    return {
        "status": "LOADER_NUMERIC_CONTRACT_CONFORMANT",
        "source_snapshot": {"file": "sample.mat", "file_size_bytes": 10},
        "configuration": {"views": 9, "grid_shape": [2, 2, 2]},
        "checks": {"a": True, "b": True},
        "coordinate_contract": [
            {
                "field": "X",
                "axis": 0,
                "dimension": 2,
                "lower_cell_center_m": -0.5,
                "upper_cell_center_m": 0.5,
                "spacing_m": 1.0,
                "inferred_cell_centered_extent_m": 2.0,
                "separable_on_landmarks": True,
                "centered": True,
                "extent_matches_official": True,
                "private_detail": "/private/raw/data.mat",
            }
        ],
        "ray_contract": {"sampled_measurements": 2},
    }


def _preflight():
    return {
        "status": "FULL_AUTHOR_NIRT_NO_GO_CURRENT_ENVIRONMENT",
        "dependencies_available": {
            "numpy": {"importable": True, "version": "2"},
            "tensorflow": {"importable": False, "error": "private path"},
        },
        "blocker_count": 1,
        "static_hazards": [
            {"code": "GPU_DEVICE_FORCED", "severity": "blocker", "evidence": "secret"}
        ],
        "memory_floor": {
            "known_persistent_floor_bytes": 100,
            "known_persistent_floor_gib": 0.1,
            "host_physical_memory_bytes": 1000,
        },
        "decision": {"safe_next_gate": "TINY_FIXTURE"},
    }


def test_public_summary_contains_only_approved_aggregate_fields() -> None:
    result = build_public_summary(_loader(), _preflight(), source_sha256="a" * 64)
    payload = json.dumps(result)
    assert result["status"] == "LOADER_NUMERIC_CONTRACT_CONFORMANT"
    assert result["official_nirt_preflight"]["blocker_codes"] == [
        "GPU_DEVICE_FORCED"
    ]
    assert "/Users/" not in payload
    assert "private_library" not in payload
    assert "private_detail" not in payload
    assert '"samples"' not in payload
    assert result["public_export_policy"]["contains_raw_arrays"] is False


def test_public_summary_rejects_failed_contract() -> None:
    loader = _loader()
    loader["checks"]["b"] = False
    with pytest.raises(ValueError, match="not all loader checks passed"):
        build_public_summary(loader, _preflight(), source_sha256="a" * 64)


def test_public_summary_rejects_unreviewed_preflight_change() -> None:
    preflight = _preflight()
    preflight["status"] = "FULL_AUTHOR_NIRT_READY"
    with pytest.raises(ValueError, match="preflight status changed"):
        build_public_summary(_loader(), preflight, source_sha256="a" * 64)
