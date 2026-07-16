from __future__ import annotations

import json
from pathlib import Path

import pytest

from site_tools import build_psu_clipped_hybrid_public_summary as exporter


REAL_PRIVATE_REPORT = (
    Path(__file__).resolve().parents[1]
    / "private_library"
    / "external_datasets"
    / "psu_bost_flight_body"
    / "domain_clipped_author_compatibility_audit.json"
)


def _mask_statistics(
    *,
    count: int,
    changed: int,
    zeroed: int,
    shortened: int,
    author_length: float,
    clipped_length: float,
) -> dict:
    return {
        "count": count,
        "changed_count": changed,
        "zeroed_count": zeroed,
        "shortened_count": shortened,
        "author_length_sum_m": author_length,
        "clipped_length_sum_m": clipped_length,
        "changed_fraction": changed / count if count else None,
        "zeroed_fraction": zeroed / count if count else None,
        "shortened_fraction": shortened / count if count else None,
        "path_length_retained_fraction": (
            clipped_length / author_length if author_length else None
        ),
    }


def _view(
    view_id: int,
    *,
    filter_required: bool = False,
    invalid: bool = False,
) -> dict:
    rows = 100
    zero_count = 5 if filter_required else 0
    changed_count = 10 if filter_required else 0
    author_length = 10.0 + 2.0 * view_id
    clipped_length = author_length - (1.0 if filter_required else 0.0)
    mechanically_valid = not invalid
    status = (
        "AUTHOR_COMPATIBLE_CLIPPED_HYBRID_INVALID"
        if invalid
        else "AUTHOR_COMPATIBLE_CLIPPED_HYBRID_CONTRACT_PASS_MASK_FILTER_REQUIRED"
        if filter_required
        else "AUTHOR_COMPATIBLE_CLIPPED_HYBRID_CONTRACT_PASS"
    )
    mask_conditioned = {
        "amask_all": _mask_statistics(
            count=40,
            changed=4 if filter_required else 0,
            zeroed=2 if filter_required else 0,
            shortened=2 if filter_required else 0,
            author_length=4.0,
            clipped_length=3.6 if filter_required else 4.0,
        ),
        "imask_all": _mask_statistics(
            count=50,
            changed=5 if filter_required else 0,
            zeroed=3 if filter_required else 0,
            shortened=2 if filter_required else 0,
            author_length=5.0,
            clipped_length=4.5 if filter_required else 5.0,
        ),
    }
    return {
        "schema_version": exporter.PRIVATE_VIEW_SCHEMA_VERSION,
        "status": status,
        "evidence_scope": exporter.EVIDENCE_SCOPE,
        "view_id_zero_based": view_id,
        "source": {
            "view_bundle_manifest_sha256": f"{view_id + 1:x}" * 64,
            "corrected_mask_manifest_sha256": f"{view_id + 3:x}" * 64,
            "geometry_source_filename": "meas.py",
            "geometry_source_sha256": "a" * 64,
            "author_source_modified": False,
        },
        "configuration": {
            "rows": rows,
            "chunk_rows": 32,
            "outer_minimum_m": [-0.11, -0.11, -0.11],
            "outer_maximum_m": [0.11, 0.11, 0.11],
            "cone_vertex_m": [0.06, 0.015, 0.0],
            "cone_axis_normalized": [1.0, 0.0, 0.0],
            "cone_angle_degrees": 25.0,
            "policy": exporter.CONFIGURATION_POLICY,
        },
        "counts": {
            "ray_count": rows,
            "author_nonzero_count": rows,
            "clipped_nonzero_count": rows - zero_count,
            "changed_from_author_count": changed_count,
            "cone_shortened_count": 5 if filter_required else 0,
            "cone_zeroed_for_no_box_overlap_count": (5 if filter_required else 0),
            "forward_box_shortened_count": 0,
            "clipped_endpoint_box_violation_count": 1 if invalid else 0,
            "clipped_length_exceeds_author_count": 0,
            "nonfinite_output_count": 0,
            "negative_clipped_inner_aperture_radius_count": 0,
            "negative_clipped_outer_aperture_radius_count": 0,
            "clipped_zero_length_count": zero_count,
        },
        "path_length": {
            "author_length_sum_m": author_length,
            "clipped_length_sum_m": clipped_length,
            "removed_length_sum_m": author_length - clipped_length,
            "retained_fraction": clipped_length / author_length,
            "removed_fraction": 1.0 - clipped_length / author_length,
        },
        "mask_conditioned": mask_conditioned,
        "runtime_observation": {
            "wall_seconds": 1.5 + view_id,
            "scope": exporter.RUNTIME_SCOPE,
        },
        "decision": {
            "positive_segments_inside_forward_box": mechanically_valid,
            "geometry_safe_zero_row_filter_required": filter_required,
            "fixed_spatial_domain_established": False,
            "training_ready": "NO",
            "algorithm_superiority_claim": "LOCKED",
            "next_gate": exporter.NEXT_GATE,
        },
        "limitations": [
            "A1 remains an author-compatibility ablation",
            "no reconstruction or superiority comparison is run",
        ],
        "upstream_view_contract": {
            "bundle_status": exporter.VIEW_BUNDLE_STATUS,
            "mask_status": exporter.MASK_STATUS,
        },
    }


def _private_report(*, invalid: bool = False) -> dict:
    views = (
        [_view(0, invalid=True), _view(1)]
        if invalid
        else [_view(0, filter_required=True), _view(1)]
    )
    invalid_ids = [
        view["view_id_zero_based"]
        for view in views
        if view["status"] == "AUTHOR_COMPATIBLE_CLIPPED_HYBRID_INVALID"
    ]
    filter_ids = [
        view["view_id_zero_based"]
        for view in views
        if view["decision"]["geometry_safe_zero_row_filter_required"]
    ]
    author_length = sum(view["path_length"]["author_length_sum_m"] for view in views)
    clipped_length = sum(view["path_length"]["clipped_length_sum_m"] for view in views)
    removed_length = author_length - clipped_length
    return {
        "schema_version": exporter.PRIVATE_SCHEMA_VERSION,
        "execution_status": exporter.EXECUTION_STATUS,
        "scientific_verdict": (
            "INVALID" if invalid_ids else "AUTHOR_COMPATIBILITY_ABLATION_ONLY"
        ),
        "status": (
            "AUTHOR_COMPATIBLE_CLIPPED_HYBRID_ALL_VIEW_INVALID"
            if invalid_ids
            else "AUTHOR_COMPATIBLE_CLIPPED_HYBRID_ALL_VIEW_CONTRACT_PASS_MASK_FILTER_REQUIRED"
            if filter_ids
            else "AUTHOR_COMPATIBLE_CLIPPED_HYBRID_ALL_VIEW_CONTRACT_PASS"
        ),
        "view_count": len(views),
        "views": views,
        "aggregate": {
            "invalid_view_ids": invalid_ids,
            "zero_row_filter_required_view_ids": filter_ids,
            "author_length_sum_m": author_length,
            "clipped_length_sum_m": clipped_length,
            "removed_length_sum_m": removed_length,
            "retained_fraction": clipped_length / author_length,
            "removed_fraction": removed_length / author_length,
            "changed_ray_count": sum(
                view["counts"]["changed_from_author_count"] for view in views
            ),
            "clipped_zero_length_count": sum(
                view["counts"]["clipped_zero_length_count"] for view in views
            ),
        },
        "decision": {
            "domain_clipping_mechanically_valid": not invalid_ids,
            "fixed_spatial_domain_established": False,
            "training_ready": "NO",
            "algorithm_superiority_claim": "LOCKED",
            "next_gate": exporter.NEXT_GATE,
        },
        "limitations": [
            "A1 only isolates clipping within the author hybrid geometry",
            "domain consistency is not reconstruction evidence",
        ],
    }


def _add_private_leaks(report: dict) -> None:
    report["local_report_path"] = (
        "/Users/alice/private/domain_clipped_author_compatibility_audit.json"
    )
    report["raw_ray_array"] = [987654321.125]
    report["raw_view_indices"] = [0, 1]
    report["source_snippet"] = "def private_formula(): return secret"
    report["views"][0]["configuration"]["raw_box_intersections"] = [123456789.125]
    report["views"][0]["source"]["local_manifest_path"] = (
        "/Volumes/private/view_00/manifest.json"
    )
    report["views"][0]["runtime_observation"]["machine_note"] = (
        "file:///Users/alice/run.log"
    )
    report["views"][0]["counts"]["changed_ray_indices"] = [2, 4, 8]
    report["aggregate"]["private_invalid_view_ids"] = [7, 9]
    report["decision"]["source_code"] = "import private_module"


def test_public_summary_is_a_strict_allowlist_and_drops_private_fields() -> None:
    private = _private_report()
    private_manifest_hashes = {
        view["source"]["view_bundle_manifest_sha256"] for view in private["views"]
    } | {view["source"]["corrected_mask_manifest_sha256"] for view in private["views"]}
    _add_private_leaks(private)

    result = exporter.build_public_summary(private)
    payload = json.dumps(result, sort_keys=True)

    assert set(result) == {
        "schema_version",
        "source_schema_version",
        "source_view_schema_version",
        "execution_status",
        "scientific_verdict",
        "status",
        "evidence_scope",
        "view_count",
        "views",
        "aggregate",
        "decision",
        "limitations",
        "provenance",
        "claim_boundary",
        "public_export_policy",
    }
    assert set(result["views"][0]) == {
        "view_id_zero_based",
        "status",
        "upstream_view_contract",
        "counts",
        "path_length",
        "mask_conditioned",
        "decision",
        "limitations",
    }
    assert set(result["aggregate"]) == {
        "invalid_view_count",
        "zero_row_filter_required_view_count",
        *exporter.PATH_LENGTH_FIELDS,
        *exporter.PATH_FRACTION_FIELDS,
        *exporter.AGGREGATE_COUNT_FIELDS,
    }
    assert result["aggregate"]["invalid_view_count"] == 0
    assert result["aggregate"]["zero_row_filter_required_view_count"] == 1
    assert "invalid_view_ids" not in result["aggregate"]
    assert "zero_row_filter_required_view_ids" not in result["aggregate"]
    assert result["views"][0]["counts"] == {
        field: private["views"][0]["counts"][field]
        for field in exporter.VIEW_COUNT_FIELDS
    }
    assert result["views"][0]["path_length"] == private["views"][0]["path_length"]
    assert (
        result["views"][0]["mask_conditioned"]
        == private["views"][0]["mask_conditioned"]
    )
    assert result["provenance"] == {
        "geometry_source_filename": "meas.py",
        "geometry_source_sha256": "a" * 64,
        "author_source_modified": False,
    }
    assert result["claim_boundary"] == {
        "supported_claim": "A1_AUTHOR_COMPATIBILITY_ABLATION_ONLY",
        "fixed_spatial_domain_established": False,
        "reconstruction_established": False,
        "algorithm_superiority_established": False,
    }
    assert result["public_export_policy"]["strict_field_allowlist"] is True

    for secret in (
        "/Users/",
        "/Volumes/",
        "file://",
        "987654321.125",
        "123456789.125",
        "private_formula",
        "private_module",
        "runtime_observation",
        "wall_seconds",
        "configuration",
        "raw_ray_array",
        "raw_view_indices",
        "changed_ray_indices",
        "private_invalid_view_ids",
        "view_bundle_manifest_sha256",
        "corrected_mask_manifest_sha256",
    ):
        assert secret not in payload
    for private_hash in private_manifest_hashes:
        assert private_hash not in payload


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda report: report["limitations"].append(
                "debug copy at /Users/alice/private.json"
            ),
            "private or local path",
        ),
        (
            lambda report: report["views"][0]["limitations"].append(
                "def leaked_source(): return 1"
            ),
            "source-code snippet",
        ),
        (
            lambda report: report["views"][0]["source"].update(
                geometry_source_filename="private_library/meas.py"
            ),
            "private or local path",
        ),
    ],
)
def test_public_summary_rejects_leaks_in_preserved_fields(mutation, message) -> None:
    private = _private_report()
    mutation(private)
    with pytest.raises(ValueError, match=message):
        exporter.build_public_summary(private)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda report: report.update(schema_version="wrong-schema"),
            "unsupported private schema_version",
        ),
        (lambda report: report.pop("views"), "report.views is required"),
        (lambda report: report.update(view_count=3), "view_count must equal"),
        (
            lambda report: report["views"][1].update(view_id_zero_based=0),
            "ordered contiguous range",
        ),
        (
            lambda report: report["views"][0].update(schema_version="wrong-view"),
            "unsupported report.views\\[0\\].schema_version",
        ),
        (
            lambda report: report["views"][0].update(
                status="AUTHOR_COMPATIBLE_CLIPPED_HYBRID_CONTRACT_PASS"
            ),
            "status conflicts",
        ),
        (
            lambda report: report["views"][0]["path_length"].update(
                retained_fraction=0.5
            ),
            "retained_fraction is inconsistent",
        ),
        (
            lambda report: report["views"][0]["mask_conditioned"]["amask_all"].update(
                changed_fraction=0.5
            ),
            "changed_fraction is inconsistent",
        ),
        (
            lambda report: report["aggregate"].update(changed_ray_count=999),
            "changed_ray_count conflicts",
        ),
        (
            lambda report: report["aggregate"].update(
                zero_row_filter_required_view_ids=[]
            ),
            "conflicts with per-view decisions",
        ),
        (
            lambda report: report["decision"].update(
                fixed_spatial_domain_established=True
            ),
            "must remain false",
        ),
        (
            lambda report: report.update(
                status="AUTHOR_COMPATIBLE_CLIPPED_HYBRID_ALL_VIEW_CONTRACT_PASS"
            ),
            "status conflicts",
        ),
    ],
)
def test_public_summary_rejects_malformed_or_inconsistent_input(
    mutation, message
) -> None:
    private = _private_report()
    mutation(private)
    with pytest.raises(ValueError, match=message):
        exporter.build_public_summary(private)


def test_well_formed_invalid_audit_remains_exportable_negative_evidence() -> None:
    result = exporter.build_public_summary(_private_report(invalid=True))

    assert result["scientific_verdict"] == "INVALID"
    assert result["status"] == "AUTHOR_COMPATIBLE_CLIPPED_HYBRID_ALL_VIEW_INVALID"
    assert result["aggregate"]["invalid_view_count"] == 1
    assert result["decision"]["domain_clipping_mechanically_valid"] is False
    assert result["claim_boundary"]["reconstruction_established"] is False


def test_export_is_atomic_and_malformed_json_does_not_replace_output(
    tmp_path, monkeypatch
) -> None:
    input_path = tmp_path / "private" / "clipped_hybrid_audit.json"
    output_path = tmp_path / "docs" / "clipped_hybrid_public.json"
    input_path.parent.mkdir()
    input_path.write_text(json.dumps(_private_report()), encoding="utf-8")

    replace_calls = []
    real_replace = exporter.os.replace

    def record_replace(source, destination):
        replace_calls.append((source, destination))
        return real_replace(source, destination)

    monkeypatch.setattr(exporter.os, "replace", record_replace)
    result = exporter.export_public_summary(input_path, output_path)

    assert json.loads(output_path.read_text(encoding="utf-8")) == result
    assert len(replace_calls) == 1
    assert replace_calls[0][0].parent == output_path.parent
    assert replace_calls[0][1] == output_path
    assert list(output_path.parent.glob(f".{output_path.name}.*.tmp")) == []

    original = output_path.read_bytes()
    input_path.write_text("{malformed", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        exporter.export_public_summary(input_path, output_path)
    assert output_path.read_bytes() == original
    assert len(replace_calls) == 1


def test_atomic_write_cleans_up_temporary_file_on_replace_failure(
    tmp_path, monkeypatch
) -> None:
    output_path = tmp_path / "public.json"
    output_path.write_text('{"old": true}\n', encoding="utf-8")
    original = output_path.read_bytes()

    def fail_replace(source, destination):
        raise OSError("replace failed")

    monkeypatch.setattr(exporter.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        exporter.write_json_atomic({"new": True}, output_path)

    assert output_path.read_bytes() == original
    assert list(tmp_path.glob(f".{output_path.name}.*.tmp")) == []


@pytest.mark.skipif(
    not REAL_PRIVATE_REPORT.exists(),
    reason="private PSU audit is not present in this checkout",
)
def test_real_private_report_matches_the_public_export_contract() -> None:
    private = exporter.load_private_report(REAL_PRIVATE_REPORT)
    result = exporter.build_public_summary(private)
    payload = json.dumps(result, sort_keys=True)

    assert result["view_count"] == 9
    assert result["aggregate"]["invalid_view_count"] == 0
    assert result["aggregate"]["zero_row_filter_required_view_count"] == 3
    assert (
        result["claim_boundary"]["supported_claim"]
        == "A1_AUTHOR_COMPATIBILITY_ABLATION_ONLY"
    )
    assert result["claim_boundary"]["fixed_spatial_domain_established"] is False
    assert result["claim_boundary"]["reconstruction_established"] is False
    assert result["claim_boundary"]["algorithm_superiority_established"] is False
    assert "runtime_observation" not in payload
    assert "invalid_view_ids" not in payload
    assert "zero_row_filter_required_view_ids" not in payload
    for view in private["views"]:
        assert view["source"]["view_bundle_manifest_sha256"] not in payload
        assert view["source"]["corrected_mask_manifest_sha256"] not in payload
