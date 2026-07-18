from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Callable

import pytest

from site_tools.validate_oerf_n2_real_bost_contract import (
    FIXTURE_STATUS,
    NOT_READY_STATUS,
    PLACEHOLDER_STATUS,
    READY_STATUS,
    compute_split_digest,
    validate_contract,
    write_report,
)


ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER = ROOT / "data_templates/oerf_n2_lab_intake.placeholder.json"
SCHEMA = ROOT / "data_templates/oerf_n2_real_bost_contract.schema.json"
VALIDATOR = ROOT / "site_tools/validate_oerf_n2_real_bost_contract.py"


def _sensor(sensor_id: str, calibration: str, f_number: float) -> dict[str, object]:
    return {
        "sensor_id": sensor_id,
        "optical_channel_id": "optical_channel_main",
        "image_shape": [128, 160],
        "pixel_pitch_m": 6.5e-6,
        "focal_length_m": 0.05,
        "f_number": f_number,
        "intrinsics_path": f"geometry/{sensor_id}/intrinsics.json",
        "extrinsics_path": f"geometry/{sensor_id}/extrinsics.json",
        "ray_bundle_path": None,
        "distortion_model": "brown_conrady",
        "calibration_id": calibration,
        "calibration_version": "v1",
        "calibration_reprojection_rmse_px": 0.18,
    }


def _view(
    view_id: str,
    sensor_id: str,
    role: str,
    condition_id: str,
    pattern_id: str,
) -> dict[str, object]:
    return {
        "view_id": view_id,
        "sensor_id": sensor_id,
        "role": role,
        "run_id": "run_01",
        "session_id": "session_01",
        "condition_id": condition_id,
        "geometry_id": "geometry_01",
        "reference_path": f"raw/{view_id}/reference.tif",
        "flow_on_path": f"raw/{view_id}/flow_on.tif",
        "displacement_path": f"displacement/{view_id}.npy",
        "mask_path": f"masks/{view_id}.npy",
        "confidence_path": f"confidence/{view_id}.npy",
        "timestamps_path": None,
        "observation_units": "pixel_displacement",
        "component_order": "uv",
        "displacement_method": "horn_schunck_v1",
        "background_pattern_id": pattern_id,
    }


def complete_contract() -> dict[str, object]:
    contract: dict[str, object] = {
        "schema_version": "oerf-n2-real-bost-contract-1.0",
        "record_kind": "CONTRACT_TEST_FIXTURE",
        "identity": {
            "dataset_id": "private_dataset_alpha",
            "case_id": "private_case_alpha",
            "data_origin": "synthetic_contract_test",
            "run_id": "run_01",
            "session_id": "session_01",
            "condition_id": "aperture_pair",
            "geometry_id": "geometry_01",
            "independent_unit": "run",
            "field_is_time_resolved": False,
            "acquisition_time_utc": "2026-07-18T00:00:00Z",
        },
        "provenance": {
            "evidence_class": "synthetic_fixture",
            "manifest_path": "manifest/files.sha256",
            "manifest_digest_sha256": "b" * 64,
            "manifest_entry_count": 24,
            "source_evidence_path": "manifest/fixture_source.json",
            "verified_at_utc": "2026-07-18T00:05:00Z",
        },
        "field_domain": {
            "parameterization": "refractive_index",
            "units": "dimensionless",
            "axis_order": "zyx",
            "grid_shape": [16, 16, 16],
            "bounds_m": [[-0.02, 0.02], [-0.02, 0.02], [-0.02, 0.02]],
            "support_path": "geometry/support.npy",
            "truth_available": False,
            "truth_path": None,
            "truth_provenance": "none",
        },
        "sensors": [
            _sensor("sensor_f4", "cal_f4", 4.0),
            _sensor("sensor_f22", "cal_f22", 22.0),
            _sensor("sensor_audit", "cal_audit", 22.0),
        ],
        "views": [
            _view("view_train_f4", "sensor_f4", "reconstruction", "condition_f4", "pattern_a"),
            _view("view_train_f22", "sensor_f22", "reconstruction", "condition_f22", "pattern_b"),
            _view("view_audit", "sensor_audit", "audit_locked", "condition_f22", "pattern_b"),
        ],
        "forward_model": {
            "regime": "cone_ray",
            "interface": "forward_adjoint",
            "implementation_id": "cone_forward_v1",
            "deterministic": True,
            "aperture_model": "cone_multi_ray",
            "ray_bending": False,
            "optical_flow_inside_forward": False,
            "normalization_id": "pixel_displacement_v1",
            "can_apply_forward": True,
            "can_apply_adjoint": True,
            "can_jvp": False,
            "can_vjp": False,
            "can_export_tiny_matrix": True,
            "row_layout_documented": True,
        },
        "operator_audit": {
            "dot_product": {"required": True, "relative_error": 2e-9},
            "finite_difference": {"required": False, "relative_error": None},
            "unit_scale_passed": True,
            "support_mask_passed": True,
        },
        "physical_mismatch": {
            "primary": "finite_aperture",
            "secondary": [],
            "frozen_before_audit": True,
            "evidence": {
                "f_number_levels": [4.0, 22.0],
                "calibration_versions": ["v1"],
                "displacement_methods": ["horn_schunck_v1"],
                "discretization_levels": ["16cubed"],
                "condition_evidence": [
                    {
                        "condition_id": "condition_f4",
                        "session_id": "session_01",
                        "geometry_id": "geometry_01",
                        "sensor_id": "sensor_f4",
                        "f_number": 4.0,
                        "flow_off_repeat_count": 50,
                        "flow_off_manifest_path": "flow_off/condition_f4.sha256",
                        "background_pattern_ids": ["pattern_a"],
                    },
                    {
                        "condition_id": "condition_f22",
                        "session_id": "session_01",
                        "geometry_id": "geometry_01",
                        "sensor_id": "sensor_f22",
                        "f_number": 22.0,
                        "flow_off_repeat_count": 52,
                        "flow_off_manifest_path": "flow_off/condition_f22.sha256",
                        "background_pattern_ids": ["pattern_b"],
                    },
                ],
                "independent_background_pattern_ids": ["pattern_a", "pattern_b"],
                "aperture_pairing_policy": "same_optical_channel_same_geometry",
                "repeated_session_ids": ["session_01"],
                "paired_condition_ids": ["condition_f4", "condition_f22"],
                "high_fidelity_forward_available": True,
                "notes": "Contract-only fixture; it is not a measurement result.",
            },
        },
        "split_contract": {
            "split_unit": "view",
            "training_unit_ids": ["view_train_f4", "view_train_f22"],
            "tuning_unit_ids": [],
            "validation_unit_ids": [],
            "audit_unit_ids": ["view_audit"],
            "audit_locked": True,
            "audit_opened": False,
            "frozen_at_utc": "2026-07-18T00:10:00Z",
            "split_digest_sha256": None,
            "random_frame_split_permitted": False,
        },
        "endpoints": {
            "primary": {
                "name": "heldout_reprojection_relative_l2",
                "direction": "lower_is_better",
                "truth_required": False,
            },
            "secondary": [],
            "heldout_mask_policy": "fixed_before_reconstruction",
            "external_reference_path": None,
            "external_reference_provenance": "none",
            "cost_ledger": [
                "forward_calls",
                "adjoint_calls",
                "ray_samples",
                "wall_time",
                "peak_memory",
            ],
        },
        "permissions": {
            "local_storage": True,
            "local_training": True,
            "group_meeting": True,
            "thesis_text": True,
            "thesis_figures": True,
            "public_metadata": True,
            "public_derived_metrics": True,
            "public_raw_data": False,
            "redistribution_basis": "none",
        },
        "claim_boundary": {
            "real_field_truth_claim_allowed": False,
            "heldout_reprojection_is_unique_3d_truth": False,
            "audit_may_select_model": False,
            "audit_may_select_stopping": False,
            "synthetic_and_experimental_metrics_kept_separate": True,
        },
    }
    split = contract["split_contract"]
    assert isinstance(split, dict)
    split["split_digest_sha256"] = compute_split_digest(split)
    return contract


def complete_real_record() -> dict[str, object]:
    contract = copy.deepcopy(complete_contract())
    contract["record_kind"] = "DATASET_RECORD"
    contract["identity"]["data_origin"] = "public_real"
    contract["provenance"].update(
        evidence_class="open_dataset_manifest",
        source_evidence_path="manifest/open_dataset_landing_page.txt",
    )
    return contract


def _write_contract(path: Path, contract: dict[str, object]) -> None:
    path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")


def _run_cli(path: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VALIDATOR), str(path), *extra],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_schema_and_placeholder_are_machine_readable() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    placeholder = json.loads(PLACEHOLDER.read_text(encoding="utf-8"))
    assert schema["$schema"].endswith("2020-12/schema")
    assert placeholder["schema_version"] == "oerf-n2-real-bost-contract-1.0"
    assert "provenance" in placeholder


def test_placeholder_fails_closed_with_actionable_seven_gate_report() -> None:
    report = validate_contract(json.loads(PLACEHOLDER.read_text(encoding="utf-8")))
    assert report["status"] == PLACEHOLDER_STATUS
    assert report["passed_gate_count"] == 0
    assert report["required_gate_count"] == 7
    assert len(report["next_actions"]) == 7
    assert not any(report["authorization"].values())


def test_complete_real_contract_only_authorizes_preregistration_with_audit_sealed() -> None:
    report = validate_contract(complete_real_record())
    assert report["status"] == READY_STATUS
    assert report["passed_gate_count"] == 7
    assert report["authorization"]["may_preregister_n2_experiment"] is True
    assert report["authorization"]["may_train_on_non_audit_units_after_preregistration"] is True
    assert report["authorization"]["may_open_locked_audit"] is False
    assert report["authorization"]["may_claim_algorithm_success"] is False
    assert report["authorization"]["may_claim_real_bost_improvement"] is False


def test_complete_contract_fixture_cannot_impersonate_real_data() -> None:
    report = validate_contract(complete_contract())
    assert report["status"] == FIXTURE_STATUS
    assert report["passed_gate_count"] == 7
    assert report["authorization"]["may_preregister_n2_experiment"] is False


def test_simple_record_relabel_is_rejected_without_real_provenance() -> None:
    contract = complete_contract()
    contract["record_kind"] = "DATASET_RECORD"
    contract["identity"]["data_origin"] = "public_real"
    with pytest.raises(ValueError, match="E_RECORD_PROVENANCE_MISMATCH"):
        validate_contract(contract)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["identity"].pop("acquisition_time_utc"),
        lambda value: value["identity"].__setitem__("dataset_id", 123),
        lambda value: value["field_domain"].__setitem__("parameterization", "outside_schema"),
        lambda value: value.__setitem__("unexpected_private_field", "not accepted"),
    ],
)
def test_schema_violations_fail_closed(mutate: Callable[[dict[str, object]], object]) -> None:
    contract = complete_contract()
    mutate(contract)
    with pytest.raises(ValueError, match="E_SCHEMA_VIOLATION"):
        validate_contract(contract)


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_nonfinite_numbers_are_rejected_before_threshold_checks(bad_value: float) -> None:
    contract = complete_contract()
    contract["operator_audit"]["dot_product"]["relative_error"] = bad_value
    with pytest.raises(ValueError, match="E_NONFINITE_NUMBER"):
        validate_contract(contract)


def test_negative_relative_error_is_rejected_by_schema() -> None:
    contract = complete_contract()
    contract["operator_audit"]["dot_product"]["relative_error"] = -1e-9
    with pytest.raises(ValueError, match="E_SCHEMA_VIOLATION"):
        validate_contract(contract)


def test_split_overlap_is_rejected_without_echoing_private_unit() -> None:
    contract = complete_contract()
    contract["split_contract"]["audit_unit_ids"] = ["view_train_f4"]
    with pytest.raises(ValueError, match="E_SPLIT_OVERLAP") as captured:
        validate_contract(contract)
    assert "view_train_f4" not in str(captured.value)


def test_session_split_cannot_hide_audit_view_in_training_unit() -> None:
    contract = complete_contract()
    contract["views"][0]["session_id"] = "private_train_session_a"
    contract["views"][1]["session_id"] = "private_train_session_b"
    contract["views"][2]["session_id"] = "private_train_session_a"
    contract["split_contract"].update(
        split_unit="session",
        training_unit_ids=["private_train_session_a", "private_train_session_b"],
        tuning_unit_ids=[],
        validation_unit_ids=[],
        audit_unit_ids=["private_audit_session"],
    )
    with pytest.raises(ValueError, match="E_SPLIT_ROLE_MISMATCH") as captured:
        validate_contract(contract)
    assert "private_train_session_a" not in str(captured.value)


def test_audit_opening_removes_authorization() -> None:
    contract = complete_real_record()
    contract["split_contract"]["audit_opened"] = True
    contract["split_contract"]["split_digest_sha256"] = compute_split_digest(
        contract["split_contract"]
    )
    report = validate_contract(contract)
    assert report["status"] == NOT_READY_STATUS
    split_gate = next(row for row in report["gates"] if row["gate"] == "independent_split_lock")
    assert split_gate["passed"] is False
    assert report["authorization"]["may_preregister_n2_experiment"] is False


def test_split_digest_must_bind_current_membership_and_policy() -> None:
    contract = complete_real_record()
    contract["split_contract"]["split_digest_sha256"] = "c" * 64
    report = validate_contract(contract)
    gate = next(row for row in report["gates"] if row["gate"] == "independent_split_lock")
    assert gate["passed"] is False
    assert "digest" in gate["detail"]


def test_finite_aperture_claim_requires_observed_paired_f_numbers() -> None:
    contract = complete_real_record()
    for sensor in contract["sensors"]:
        sensor["f_number"] = 22.0
    for row in contract["physical_mismatch"]["evidence"]["condition_evidence"]:
        row["f_number"] = 22.0
    report = validate_contract(contract)
    gate = next(row for row in report["gates"] if row["gate"] == "physical_mismatch_evidence")
    assert gate["passed"] is False
    assert report["status"] == NOT_READY_STATUS


def test_flow_off_threshold_is_checked_per_fixed_condition() -> None:
    contract = complete_real_record()
    contract["physical_mismatch"]["evidence"]["condition_evidence"][1][
        "flow_off_repeat_count"
    ] = 49
    report = validate_contract(contract)
    gate = next(row for row in report["gates"] if row["gate"] == "physical_mismatch_evidence")
    assert gate["passed"] is False
    assert report["inventory"]["minimum_flow_off_repeat_count_per_fixed_condition"] == 49


def test_field_l2_cannot_be_primary_without_independent_truth() -> None:
    contract = complete_real_record()
    contract["endpoints"]["primary"] = {
        "name": "field_relative_l2",
        "direction": "lower_is_better",
        "truth_required": True,
    }
    report = validate_contract(contract)
    gate = next(row for row in report["gates"] if row["gate"] == "endpoint_legality")
    assert gate["passed"] is False
    assert "独立真值" in gate["detail"]


def test_public_raw_data_requires_redistribution_basis() -> None:
    contract = complete_real_record()
    contract["permissions"]["public_raw_data"] = True
    with pytest.raises(ValueError, match="E_RAW_REDISTRIBUTION_PERMISSION"):
        validate_contract(contract)


def test_parent_traversal_path_is_rejected() -> None:
    contract = complete_contract()
    contract["views"][0]["reference_path"] = "../private/reference.tif"
    with pytest.raises(ValueError, match="E_PATH_ESCAPES_PRIVATE_ROOT"):
        validate_contract(contract)


def test_public_report_redacts_private_paths_ids_and_permission_values(tmp_path: Path) -> None:
    contract = complete_real_record()
    report = validate_contract(contract)
    output = tmp_path / "public_summary.json"
    write_report(output, report)
    rendered = output.read_text(encoding="utf-8")
    for private_value in (
        "private_dataset_alpha",
        "private_case_alpha",
        "view_train_f4",
        "sensor_f4",
        "raw/view_train_f4/reference.tif",
        "manifest/open_dataset_landing_page.txt",
    ):
        assert private_value not in rendered
    reloaded = json.loads(rendered)
    assert reloaded["privacy"]["source_paths_emitted"] is False
    assert reloaded["privacy"]["raw_dataset_or_case_ids_emitted"] is False


def test_cli_error_stream_does_not_echo_private_ids(tmp_path: Path) -> None:
    contract = complete_contract()
    contract["split_contract"]["audit_unit_ids"] = ["view_train_f4"]
    path = tmp_path / "contract.json"
    _write_contract(path, contract)
    completed = _run_cli(path)
    combined = completed.stdout + completed.stderr
    assert completed.returncode == 2
    assert "E_SPLIT_OVERLAP" in combined
    for private_value in ("private_dataset_alpha", "private_case_alpha", "view_train_f4"):
        assert private_value not in combined


def test_cli_uses_fail_closed_exit_codes(tmp_path: Path) -> None:
    placeholder_path = tmp_path / "placeholder.json"
    fixture_path = tmp_path / "fixture.json"
    real_path = tmp_path / "real.json"
    placeholder_path.write_text(PLACEHOLDER.read_text(encoding="utf-8"), encoding="utf-8")
    _write_contract(fixture_path, complete_contract())
    _write_contract(real_path, complete_real_record())
    assert _run_cli(placeholder_path).returncode == 2
    assert _run_cli(fixture_path).returncode == 3
    assert _run_cli(fixture_path, "--allow-fixture").returncode == 0
    assert _run_cli(real_path).returncode == 0


def test_cli_rejects_nonstandard_json_nan(tmp_path: Path) -> None:
    contract = complete_contract()
    contract["operator_audit"]["dot_product"]["relative_error"] = math.nan
    path = tmp_path / "nan.json"
    path.write_text(json.dumps(contract, allow_nan=True), encoding="utf-8")
    completed = _run_cli(path)
    assert completed.returncode == 2
    assert "E_JSON_NONFINITE" in completed.stderr


def test_atomic_writer_ignores_predictable_partial_symlink(tmp_path: Path) -> None:
    victim = tmp_path / "victim.txt"
    victim.write_text("unchanged\n", encoding="utf-8")
    output = tmp_path / "summary.json"
    predictable = tmp_path / f".{output.name}.partial"
    predictable.symlink_to(victim)
    write_report(output, validate_contract(complete_contract()))
    assert victim.read_text(encoding="utf-8") == "unchanged\n"
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == FIXTURE_STATUS


def test_non_linear_jvp_contract_needs_finite_difference_audit() -> None:
    contract = complete_real_record()
    contract["forward_model"].update(
        regime="curved_ray",
        interface="jvp_vjp",
        can_apply_forward=False,
        can_apply_adjoint=False,
        can_jvp=True,
        can_vjp=True,
    )
    contract["physical_mismatch"]["primary"] = "ray_bending"
    contract["physical_mismatch"]["evidence"]["aperture_pairing_policy"] = "not_applicable"
    report = validate_contract(contract)
    gate = next(row for row in report["gates"] if row["gate"] == "operator_and_adjoint")
    assert gate["passed"] is False
    assert "有限差分" in gate["detail"]
