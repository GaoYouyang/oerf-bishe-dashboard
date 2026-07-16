from __future__ import annotations

from site_tools.run_psu_b0_interface_fixture import REPORT_SCHEMA, run_fixture


def test_tiny_b0_reconstruction_interface_fixture_passes() -> None:
    report = run_fixture(
        grid_size=7,
        detector_count=3,
        sample_count=4,
        power_iterations=4,
        reconstruction_iterations=5,
        step_fraction=1.0,
    )
    assert report["schema_version"] == REPORT_SCHEMA
    assert report["status"] == "B0_RECONSTRUCTION_INTERFACE_FIXTURE_PASS"
    assert report["metrics"]["adjoint_relative_dot_error"] < 1e-11
    assert report["calls"]["optimization_and_final_evaluation"] == {
        "forward_calls": 6,
        "adjoint_calls": 5,
    }
