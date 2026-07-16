from __future__ import annotations

import json

import numpy as np

from site_tools.plot_psu_b0_interface_audit import (
    FIGURE_SCHEMA,
    FIXTURE_SCHEMA,
    PUBLIC_SCHEMA,
    build_figure,
)


def test_builds_interface_gate_figure_from_allowlisted_inputs(tmp_path) -> None:
    public = {
        "schema_version": PUBLIC_SCHEMA,
        "aggregate_geometry": {
            "selected_ray_count": 18,
            "total_sample_count": 288,
        },
        "grid_profiles": [
            {
                "grid_shape_zyx": [16, 16, 16],
                "cpu_float64_adjoint_relative_error": 1e-15,
                "mps_float32_adjoint_relative_error": 1e-7,
                "cpu_profile": {
                    "forward_seconds_median": 0.001,
                    "adjoint_seconds_median": 0.002,
                },
                "mps_profile": {
                    "forward_seconds_median": 0.0008,
                    "adjoint_seconds_median": 0.0012,
                    "mps_current_allocated_bytes": 4 * 1024**2,
                    "mps_driver_allocated_bytes": 40 * 1024**2,
                },
            },
            {
                "grid_shape_zyx": [32, 32, 32],
                "cpu_float64_adjoint_relative_error": 2e-15,
                "mps_float32_adjoint_relative_error": 2e-7,
                "cpu_profile": {
                    "forward_seconds_median": 0.002,
                    "adjoint_seconds_median": 0.003,
                },
                "mps_profile": {
                    "forward_seconds_median": 0.001,
                    "adjoint_seconds_median": 0.0015,
                    "mps_current_allocated_bytes": 8 * 1024**2,
                    "mps_driver_allocated_bytes": 42 * 1024**2,
                },
            },
        ],
    }
    fixture = {
        "schema_version": FIXTURE_SCHEMA,
        "metrics": {
            "final_measurement_relative_l2": 0.01,
            "field_relative_l2_fixture_truth_only": 0.4,
        },
    }
    public_path = tmp_path / "public.json"
    fixture_path = tmp_path / "fixture.json"
    arrays_path = tmp_path / "arrays.npz"
    public_path.write_text(json.dumps(public), encoding="utf-8")
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    np.savez_compressed(arrays_path, residual_history=np.geomspace(1.0, 0.02, 12))
    manifest = build_figure(
        public_summary_path=public_path,
        fixture_report_path=fixture_path,
        fixture_arrays_path=arrays_path,
        output_stem=tmp_path / "figure",
    )
    assert manifest["schema_version"] == FIGURE_SCHEMA
    assert len(manifest["outputs"]) == 3
    assert all((tmp_path / row["filename"]).is_file() for row in manifest["outputs"])
