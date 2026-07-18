#!/usr/bin/env python3
"""Build a privacy-safe PSU-to-N2 evidence rehearsal.

This report does not fabricate a complete N2 dataset record. It records which
contract fields are supported, absent, locally verifiable, or forbidden to
infer from the tracked public PSU summaries. All research authorizations remain
false until an actual N2 contract passes the separate fail-closed validator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

try:
    from site_tools.validate_oerf_n2_real_bost_contract import write_report
except ModuleNotFoundError:  # direct execution from site_tools/
    from validate_oerf_n2_real_bost_contract import write_report


ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA = "psu-n2-contract-rehearsal-public-1.0"
STATUS = "PUBLIC_PSU_INTERFACE_REHEARSAL_ONLY_N2_BLOCKED"
DECISION = "GO_INTERFACE_REHEARSAL_STOP_N2_ALGORITHM_CLAIMS"
ALLOWED_FIELD_STATUSES = {
    "PUBLIC_SUPPORTED",
    "PUBLIC_NEGATIVE",
    "LOCAL_VERIFICATION_REQUIRED",
    "MISSING",
    "FORBIDDEN_TO_INFER",
}
SOURCE_PATHS = {
    "heldout_protocol": ROOT / "docs/psu_heldout_camera_protocol_public_summary.json",
    "flowoff_inventory": ROOT / "docs/psu_flowoff_repeat_inventory_public_summary.json",
    "cell_payload": ROOT / "docs/psu_rotation40_cell_payload_public_summary.json",
    "geometry_binding": ROOT / "docs/psu_rotation40_geometry_binding_public_summary.json",
    "operator_interface": ROOT / "docs/psu_b0_reconstruction_interface_public_summary.json",
    "primary_source_facts": ROOT / "docs/psu_primary_source_fact_audit_2026-07-18.json",
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path.name}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _field(
    status: str,
    reason: str,
    evidence_refs: list[str],
    public_value: Any | None = None,
) -> dict[str, Any]:
    _require(status in ALLOWED_FIELD_STATUSES, f"unsupported field status: {status}")
    row: dict[str, Any] = {
        "status": status,
        "reason": reason,
        "evidence_refs": evidence_refs,
    }
    if public_value is not None:
        row["public_value"] = public_value
    return row


def _source_state(document: Mapping[str, Any]) -> str:
    for key in ("status", "execution_status"):
        value = document.get(key)
        if isinstance(value, str) and value:
            return value
    return "PUBLIC_SUMMARY_WITHOUT_TOP_LEVEL_STATUS"


def build_report(
    documents: Mapping[str, Mapping[str, Any]],
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    heldout = documents["heldout_protocol"]
    flowoff = documents["flowoff_inventory"]
    payload = documents["cell_payload"]
    geometry = documents["geometry_binding"]
    interface = documents["operator_interface"]
    primary = documents["primary_source_facts"]

    doi = str(heldout["dataset"]["doi"])
    for name, document in {
        "cell_payload": payload,
        "geometry_binding": geometry,
        "operator_interface": interface,
    }.items():
        _require(str(document["dataset"]["doi"]) == doi, f"dataset DOI mismatch: {name}")
    _require(str(primary["source"]["doi"]) == doi, "primary-source DOI mismatch")

    assessment = flowoff["temporal_repeat_assessment"]
    repeat_count = int(assessment["independent_temporal_flowoff_frames_available_per_condition"])
    _require(repeat_count == 0, "public flow-off inventory changed; re-audit before N2 rehearsal")
    _require(
        assessment["public_archive_authorizes_temporal_covariance_estimation"] is False,
        "public archive unexpectedly authorizes temporal covariance; re-audit required",
    )
    _require(
        heldout["claim_boundary"]["field_l2_available"] is False,
        "public field-truth boundary changed; re-audit required",
    )
    _require(
        heldout["claim_boundary"]["held_out_reprojection_is_unique_3d_truth"] is False,
        "reprojection truth boundary changed; re-audit required",
    )
    _require(
        payload["dataset"]["image_shape_hw"] == [2160, 2560],
        "public detector shape changed; re-audit required",
    )
    _require(int(heldout["dataset"]["view_count"]) == 70, "public view count changed")
    _require(int(heldout["dataset"]["rotation_run_count"]) == 10, "rotation count changed")
    primary_facts = primary["facts"]
    _require(
        int(primary_facts["acquisition"]["reported_flow_off_frames_per_test"]) == 2000,
        "primary-source flow-off acquisition count changed",
    )
    _require(
        primary_facts["optics"]["clean_same_channel_same_geometry_f_number_pair_reported"]
        is False,
        "primary source now reports a clean aperture pair; re-audit required",
    )

    cpu_errors = [
        float(row["cpu_float64_adjoint_relative_error"])
        for row in interface["grid_profiles"]
    ]
    mps_errors = [
        float(row["mps_float32_adjoint_relative_error"])
        for row in interface["grid_profiles"]
    ]
    operator_rehearsal_pass = (
        bool(interface["gates"]["cpu_float64_and_mps_float32_adjoint_thresholds"])
        and max(cpu_errors) <= 1e-6
        and max(mps_errors) <= 1e-6
    )
    _require(operator_rehearsal_pass, "tracked B0 operator interface no longer passes its audit")

    partition_counts = {
        str(row["id"]): int(row["view_count"])
        for row in heldout["partition"]
    }
    _require(sum(partition_counts.values()) == 70, "held-out partition no longer covers 70 views")

    field_audit = {
        "identity.dataset_id": _field(
            "PUBLIC_SUPPORTED",
            "DOI and dataset title are present in the tracked public protocol.",
            ["heldout_protocol"],
            public_value=f"doi:{doi}",
        ),
        "identity.session_and_acquisition_time": _field(
            "MISSING",
            "The public summaries do not bind a verifiable acquisition timestamp or session identity.",
            ["heldout_protocol"],
        ),
        "field_domain.parameterization_units_and_support": _field(
            "FORBIDDEN_TO_INFER",
            "The project B0 computational box is not evidence for the measured field variable, units, or physical support.",
            ["operator_interface"],
        ),
        "field_domain.truth_available": _field(
            "PUBLIC_NEGATIVE",
            "The experimental archive has no independent three-dimensional field truth.",
            ["heldout_protocol", "cell_payload"],
            public_value=False,
        ),
        "sensors.image_shape": _field(
            "PUBLIC_SUPPORTED",
            "All seven rotation-40 camera payloads report one common detector shape.",
            ["cell_payload"],
            public_value=[2160, 2560],
        ),
        "sensors.optics_and_calibration_rmse": _field(
            "LOCAL_VERIFICATION_REQUIRED",
            "The paper reports lens groups, f-numbers, pixel pitch, and aggregate calibration RMSE, but it does not bind every N2 sensor alias to an optical channel, calibration version, and timestamp.",
            ["primary_source_facts", "geometry_binding", "cell_payload"],
            public_value={
                "pixel_pitch_m": float(primary_facts["acquisition"]["pixel_pitch_m"]),
                "lens_group_count": len(primary_facts["optics"]["lens_groups"]),
                "reported_f_numbers": sorted(
                    {float(row["f_number"]) for row in primary_facts["optics"]["lens_groups"]}
                ),
                "camera_reprojection_rmse_px": float(
                    primary_facts["calibration"]["camera_reprojection_rmse_px"]
                ),
                "background_reprojection_rmse_px": float(
                    primary_facts["calibration"]["background_reprojection_rmse_px"]
                ),
            },
        ),
        "views.reference_flowon_displacement_mask": _field(
            "LOCAL_VERIFICATION_REQUIRED",
            "The archive inventory lists reference, deflected, displacement, and mask products, but a per-view manifest must bind exact files and checksums locally.",
            ["cell_payload", "heldout_protocol"],
        ),
        "views.confidence_and_timestamps": _field(
            "MISSING",
            "The tracked public summaries do not bind a per-view confidence product or acquisition timestamps.",
            ["cell_payload"],
        ),
        "forward_model.B0_forward_adjoint": _field(
            "PUBLIC_SUPPORTED",
            "A deterministic nine-support-view B0 forward/adjoint interface passed tracked CPU64 and MPS32 dot-product audits; this is interface evidence only.",
            ["operator_interface"],
            public_value={
                "support_view_count": int(interface["configuration"]["view_count"]),
                "grid_sizes": list(interface["configuration"]["grid_sizes"]),
                "cpu64_max_dot_error": max(cpu_errors),
                "mps32_max_dot_error": max(mps_errors),
            },
        ),
        "forward_model.published_cone_ray_sampling": _field(
            "PUBLIC_SUPPORTED",
            "The paper reports a cone-ray data operator with 8.5% coefficient of variation and 8000 points per pixel; this is a cost motivation, not surrogate evidence.",
            ["primary_source_facts"],
            public_value={
                "coefficient_of_variation": float(
                    primary_facts["nirt_forward"]["reported_coefficient_of_variation"]
                ),
                "points_per_pixel": int(
                    primary_facts["nirt_forward"]["reported_points_per_pixel"]
                ),
            },
        ),
        "physical_mismatch.primary": _field(
            "FORBIDDEN_TO_INFER",
            "Finite aperture, calibration drift, displacement error, and discretization cannot be ranked causally; the reported f/22 and f/32 groups also change lens and camera placement.",
            ["primary_source_facts", "flowoff_inventory", "geometry_binding", "operator_interface"],
            public_value="unknown",
        ),
        "physical_mismatch.flow_off_repeats": _field(
            "PUBLIC_NEGATIVE",
            "The paper reports 2000 acquired frames per test, but the tracked archive inventory exposes zero independent temporal repeats per fixed condition; distinct rotations are distinct geometries.",
            ["primary_source_facts", "flowoff_inventory"],
            public_value={
                "paper_reported_acquired_flow_off_frames_per_test": int(
                    primary_facts["acquisition"]["reported_flow_off_frames_per_test"]
                ),
                "independent_repeats_per_fixed_condition": repeat_count,
                "temporal_covariance_authorized": False,
            },
        ),
        "split_contract.policy": _field(
            "PUBLIC_SUPPORTED",
            "The public protocol separates three support rotations, one development rotation, and six sealed audit rotations.",
            ["heldout_protocol"],
            public_value=partition_counts,
        ),
        "split_contract.materialized_digest": _field(
            "LOCAL_VERIFICATION_REQUIRED",
            "A policy summary is not the actual member manifest or recomputable split digest used by an N2 run.",
            ["heldout_protocol"],
        ),
        "endpoints.heldout_reprojection": _field(
            "PUBLIC_SUPPORTED",
            "Rotation-block active-vector relative L2 is legal image-space evidence when masks stay frozen; it is not unique 3D truth.",
            ["heldout_protocol"],
            public_value="rotation_block_active_vector_relative_l2",
        ),
        "permissions": _field(
            "FORBIDDEN_TO_INFER",
            "Public availability alone does not fill project-specific storage, training, thesis, figure, or redistribution permissions.",
            ["heldout_protocol"],
        ),
    }

    status_counts = {status: 0 for status in sorted(ALLOWED_FIELD_STATUSES)}
    for row in field_audit.values():
        status_counts[str(row["status"])] += 1
    forbidden_present = status_counts["FORBIDDEN_TO_INFER"] > 0
    authorization = {
        "may_create_n2_dataset_record": False,
        "may_preregister_n2_experiment": False,
        "may_train_n2_model": False,
        "may_open_locked_audit": False,
        "may_claim_algorithm_success": False,
        "may_claim_real_bost_improvement": False,
        "may_publish_raw_data": False,
    }
    _require(forbidden_present, "rehearsal unexpectedly contains no forbidden inference")
    _require(not any(authorization.values()), "blocked rehearsal cannot authorize any action")

    gate_results = [
        {
            "gate": "identity_and_units",
            "n2_gate_passed": False,
            "rehearsal_state": "BLOCKED",
            "reason": "Acquisition/session identity and measured field semantics are missing.",
        },
        {
            "gate": "observation_and_geometry",
            "n2_gate_passed": False,
            "rehearsal_state": "PARTIAL",
            "reason": "Detector shape and selected geometry are known, but per-view confidence and calibration RMSE are not bound.",
        },
        {
            "gate": "operator_and_adjoint",
            "n2_gate_passed": False,
            "rehearsal_state": "INTERFACE_EVIDENCE_ONLY",
            "reason": "The B0 nine-view operator audit passed, but no complete N2 dataset record binds it to every view and condition.",
        },
        {
            "gate": "physical_mismatch_evidence",
            "n2_gate_passed": False,
            "rehearsal_state": "BLOCKED",
            "reason": "The primary mismatch is unknown and independent flow-off repeats per fixed condition are zero.",
        },
        {
            "gate": "independent_split_lock",
            "n2_gate_passed": False,
            "rehearsal_state": "POLICY_ONLY",
            "reason": "A 70-view policy exists, but an N2 member manifest and recomputable split digest are not materialized here.",
        },
        {
            "gate": "endpoint_legality",
            "n2_gate_passed": False,
            "rehearsal_state": "LEGAL_FORM_ONLY",
            "reason": "Held-out reprojection is a legal form, but the complete fixed-mask audit contract is not yet bound.",
        },
        {
            "gate": "permissions_and_claims",
            "n2_gate_passed": False,
            "rehearsal_state": "BLOCKED",
            "reason": "Project-specific storage, training, thesis, and redistribution permissions were not inferred.",
        },
    ]

    return {
        "schema_version": REPORT_SCHEMA,
        "status": STATUS,
        "decision": DECISION,
        "evidence_scope": "PUBLIC_METADATA_AND_LOCAL_INTERFACE_REHEARSAL_NO_N2_DATASET_RECORD_NO_MODEL_RESULT",
        "snapshot_date": "2026-07-18",
        "source_snapshot": [
            {
                "ref": ref,
                "sha256": source_hashes[ref],
                "schema_version": str(documents[ref]["schema_version"]),
                "status": _source_state(documents[ref]),
            }
            for ref in sorted(documents)
        ],
        "dataset_public_facts": {
            "doi": doi,
            "name": str(heldout["dataset"]["name"]),
            "camera_count": int(heldout["dataset"]["camera_count"]),
            "rotation_run_count": int(heldout["dataset"]["rotation_run_count"]),
            "view_count": int(heldout["dataset"]["view_count"]),
            "image_shape_hw": list(payload["dataset"]["image_shape_hw"]),
            "independent_experimental_unit": str(
                heldout["dataset"]["independent_experimental_unit"]
            ),
            "paper_reported_flow_off_frames_per_test": int(
                primary_facts["acquisition"]["reported_flow_off_frames_per_test"]
            ),
            "archive_exposed_independent_flow_off_repeats_per_condition": repeat_count,
        },
        "field_audit": field_audit,
        "field_status_counts": status_counts,
        "gate_results": gate_results,
        "operator_interface_rehearsal": {
            "passed_its_own_interface_audit": operator_rehearsal_pass,
            "n2_operator_gate_passed": False,
            "reason": "An audited support operator is necessary but not sufficient for a bound N2 dataset record.",
        },
        "stop_rules": [
            "Do not rerun or enlarge v5y/v6a RayKernel models on opened synthetic rigs.",
            "Do not call 70 camera-rotation views 70 independent temporal repeats.",
            "Do not freeze finite aperture as primary without paired real f-number or focus conditions on one optical channel and geometry.",
            "Do not treat the paper's cross-camera f/22 versus f/32 setup as a clean aperture intervention because lens and placement are confounded.",
            "Do not report field L2, three-dimensional truth, real improvement, or generalization from held-out reprojection alone.",
        ],
        "advisor_requests": [
            {
                "item": "one complete real session manifest",
                "purpose": "bind run/session/condition/geometry identity, timestamps, units, and provenance",
            },
            {
                "item": "at least 50 unaveraged flow-off frames per fixed condition",
                "purpose": "estimate detector noise and slow drift under the current engineering gate",
            },
            {
                "item": "camera calibration, pixel/ray mapping, masks, and confidence",
                "purpose": "bind observations to the physical forward model and catch row-order errors",
            },
            {
                "item": "paired f-number or focus conditions on the same optical channel and geometry",
                "purpose": "identify finite-aperture mismatch instead of assigning all residual error to aperture",
            },
            {
                "item": "callable forward/adjoint or JVP/VJP plus a tiny reference case",
                "purpose": "audit adjointness, units, support, and matched solver cost",
            },
            {
                "item": "permanent camera, session, or geometry audit manifest with SHA-256",
                "purpose": "test independent generalization without renaming opened views as fresh",
            },
            {
                "item": "an independent physical endpoint when available",
                "purpose": "separate field or interface accuracy from image-space consistency",
            },
            {
                "item": "separate storage, training, thesis, figure, and public-release permissions",
                "purpose": "keep laboratory data and calibration material out of the public repository",
            },
        ],
        "authorization": authorization,
        "privacy": {
            "emits_absolute_paths": False,
            "emits_local_usernames": False,
            "emits_raw_arrays": False,
            "emits_private_dataset_or_case_ids": False,
            "emits_permission_values": False,
            "emits_only_public_aggregate_facts_and_hashes": True,
        },
        "claim_boundary": [
            "This is a field-level contract rehearsal, not an N2 dataset record.",
            "The passed B0 operator interface is not a reconstruction or model result.",
            "The public PSU archive has no independent three-dimensional field truth.",
            "Held-out reprojection measures image-space consistency, not unique recovery of the true field.",
            "No algorithm, real-data improvement, generalization, or publication claim is authorized.",
        ],
    }


def build_from_tracked_sources() -> dict[str, Any]:
    documents = {ref: _load(path) for ref, path in SOURCE_PATHS.items()}
    source_hashes = {ref: _sha256(path) for ref, path in SOURCE_PATHS.items()}
    return build_report(documents, source_hashes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs/psu_n2_contract_rehearsal_public_summary.json",
    )
    args = parser.parse_args()
    report = build_from_tracked_sources()
    write_report(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
