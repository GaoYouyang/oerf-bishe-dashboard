from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from site_tools.run_psu_rotation40_resolution_transfer import (
    compare_resolution_metrics,
    validate_config,
    validate_public_report,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "demo_t16_operator/configs/psu_rotation40_resolution_transfer_preregistered_v1.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _candidate(pooled: float, cameras: tuple[float, float, float]) -> dict:
    return {
        "aggregate": {"vector_relative_l2": pooled},
        "per_camera": [
            {"camera_id": camera_id, "vector_relative_l2": value}
            for camera_id, value in zip((2, 3, 4), cameras, strict=True)
        ],
    }


def test_preregistered_config_is_valid() -> None:
    validate_config(_config())


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("status",), "POSTOPEN"),
        (("dataset", "camera_ids"), [2, 3]),
        (("dataset", "held_out_unit"), "CAMERA"),
        (("dataset", "support_rotation_degrees"), [0, 90]),
        (("input_bindings", "rotation40_payload_private_report_sha256"), "0" * 64),
        (("candidates", 0, "grid_shape_zyx"), [32, 32, 32]),
        (("candidates", 0, "volume_sha256"), "0" * 64),
        (("candidates", 0, "support_relative_l2"), 0.1),
        (("candidates", 0, "support_fit_role"), "POSTOPEN"),
        (("candidates", 0, "free_interior_value_count"), 4096),
        (("candidates", 0, "private_report_sha256"), "0" * 64),
        (("geometry_binding", "public_summary_sha256"), "0" * 64),
        (("support_reconstruction_contract", "fixed_iterations"), 5),
        (("forward", "finite_aperture_sample_count"), 8),
        (("metrics",), ["vector_relative_l2"]),
        (("decision", "minimum_predeclared_numerical_absolute_improvement"), 0.0),
        (("decision", "no_amplitude_rescaling"), False),
        (("formal_output",), "demo_t16_operator/results/changed"),
        (("attested_files", "runner"), "changed.py"),
        (("claim_firewall", "algorithm_superiority"), True),
    ],
)
def test_config_drift_fails_closed(path: tuple, value: object) -> None:
    config = copy.deepcopy(_config())
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_numerical_all_camera_transfer_passes() -> None:
    result = compare_resolution_metrics(
        _candidate(0.96, (0.82, 0.98, 0.99)),
        _candidate(0.94, (0.80, 0.97, 0.98)),
        minimum_predeclared_numerical_absolute_improvement=0.01,
        camera_nonworse_tolerance=0.0,
    )
    assert result["machine_decision"] == "RESOLUTION_TRANSFER_SIGNAL_PASS_NO_FIELD_TRUTH"
    assert result["predeclared_numerical_pooled_improvement"]
    assert result["all_three_cameras_nonworse"]
    assert result["equal_camera_macro_absolute_improvement_16_minus_32"] > 0
    assert result["worst_camera_absolute_improvement_16_minus_32"] > 0
    assert not result["practical_significance_established"]


def test_pooled_gain_with_camera_harm_is_no_go() -> None:
    result = compare_resolution_metrics(
        _candidate(0.96, (0.82, 0.98, 0.99)),
        _candidate(0.94, (0.83, 0.95, 0.96)),
        minimum_predeclared_numerical_absolute_improvement=0.01,
        camera_nonworse_tolerance=0.0,
    )
    assert result["machine_decision"] == "POOLED_TRANSFER_WITH_CAMERA_HARM_NO_GO"
    assert result["predeclared_numerical_pooled_improvement"]
    assert not result["all_three_cameras_nonworse"]


def test_subthreshold_pooled_gain_is_no_go() -> None:
    result = compare_resolution_metrics(
        _candidate(0.960, (0.82, 0.98, 0.99)),
        _candidate(0.955, (0.81, 0.97, 0.98)),
        minimum_predeclared_numerical_absolute_improvement=0.01,
        camera_nonworse_tolerance=0.0,
    )
    assert (
        result["machine_decision"]
        == "SUPPORT_RESOLUTION_GAIN_DID_NOT_CLEAR_NUMERICAL_TRANSFER_GATE_NO_GO"
    )
    assert not result["predeclared_numerical_pooled_improvement"]


def test_missing_or_duplicate_camera_fails() -> None:
    bad = _candidate(0.95, (0.80, 0.90, 1.00))
    bad["per_camera"][2]["camera_id"] = 3
    with pytest.raises(ValueError, match="cameras 2, 3, and 4"):
        compare_resolution_metrics(
            _candidate(0.96, (0.82, 0.98, 0.99)),
            bad,
            minimum_predeclared_numerical_absolute_improvement=0.01,
            camera_nonworse_tolerance=0.0,
        )


def test_runner_has_no_measurement_fitted_scale_or_search_loop() -> None:
    source = (ROOT / "site_tools/run_psu_rotation40_resolution_transfer.py").read_text(
        encoding="utf-8"
    )
    forbidden = (
        "polyfit(",
        "curve_fit(",
        "least_squares(",
        "grid_search",
        "best_scale",
        "optimize.minimize",
    )
    assert all(token not in source for token in forbidden)
    assert "verify_attestation(" in source
    assert "verify_private_inputs(" in source
    assert '"--attestation"' in source


def _public_report() -> dict:
    return {
        "schema_version": "test",
        "status": "test",
        "public_export_policy": {
            "contains_predictions": False,
            "contains_measurements": False,
            "contains_geometry_arrays": False,
            "contains_volumes": False,
            "contains_only_aggregate_metrics": True,
        },
        "candidates": [
            {
                "candidate_id": "support_cgls4_16cubed",
                "aggregate": {"vector_relative_l2": 0.9},
                "per_camera": [{"camera_id": 2, "vector_relative_l2": 0.9}],
            }
        ],
    }


def test_public_report_accepts_aggregate_metrics() -> None:
    validate_public_report(_public_report())


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("measurement_values", [1.0, 2.0]),
        ("predictions", [[0.1, 0.2]]),
        ("volume_values", [0.0]),
        ("private_paths", ["hidden.npy"]),
    ],
)
def test_public_report_rejects_private_payload_keys(key: str, value: object) -> None:
    report = _public_report()
    report[key] = value
    with pytest.raises(ValueError, match="forbidden payload key"):
        validate_public_report(report)


def test_public_report_rejects_private_path_string() -> None:
    report = _public_report()
    report["note"] = "/Users/example/private.npy"
    with pytest.raises(ValueError, match="private path"):
        validate_public_report(report)


def test_preregistration_names_rotation_not_camera_holdout() -> None:
    note = (
        ROOT / "docs/psu_rotation40_resolution_transfer_prereg_2026-07-19.md"
    ).read_text(encoding="utf-8")
    assert "ROTATION_RUN_NOT_CAMERA" in note
    assert "同一组三台物理相机" in note
    assert "未参与 support 重建的 rotation-40 相机" not in note
