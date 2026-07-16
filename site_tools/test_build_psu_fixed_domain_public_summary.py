from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from site_tools import build_psu_fixed_domain_public_summary as exporter


REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_PRIVATE_REPORT = (
    REPO_ROOT
    / "private_library"
    / "external_datasets"
    / "psu_bost_flight_body"
    / "fixed_domain_geometry_audit.json"
)


def _mask(
    *,
    count: int,
    b0_hits: int,
    b1_hits: int,
    b0_length: float,
    b1_length: float,
) -> dict:
    return {
        "count": count,
        "author_nonzero_count": count,
        "b0_hit_count": b0_hits,
        "b1_hit_count": b1_hits,
        "b1_removed_from_b0_count": b0_hits - b1_hits,
        "author_length_sum_m": b1_length,
        "b0_length_sum_m": b0_length,
        "b1_length_sum_m": b1_length,
        "author_nonzero_fraction": 1.0,
        "b0_hit_fraction": b0_hits / count,
        "b1_hit_fraction": b1_hits / count,
        "b1_removed_from_b0_fraction": (b0_hits - b1_hits) / count,
        "b1_path_fraction_of_b0": b1_length / b0_length,
    }


def _view(view_id: int, *, active_shortfall: bool = False) -> dict:
    ray_count = 100
    b0_hits = 90
    b1_hits = 60
    author_length = 80.0
    b0_length = 120.0
    b1_length = 30.0
    active = _mask(
        count=40,
        b0_hits=40,
        b1_hits=39 if active_shortfall else 40,
        b0_length=50.0,
        b1_length=20.0,
    )
    inactive = _mask(
        count=50,
        b0_hits=45,
        b1_hits=20,
        b0_length=60.0,
        b1_length=5.0,
    )
    return {
        "schema_version": exporter.PRIVATE_VIEW_SCHEMA_VERSION,
        "status": exporter.VIEW_STATUS,
        "evidence_scope": exporter.VIEW_EVIDENCE_SCOPE,
        "view_id_zero_based": view_id,
        "source": {
            "bundle_manifest_sha256": "1" * 64,
            "setup_manifest_sha256": "2" * 64,
            "corrected_mask_manifest_sha256": "3" * 64,
            "geometry_implementation_filename": "psu_bost_forward_geometry.py",
            "geometry_implementation_sha256": "4" * 64,
            "audit_implementation_sha256": "5" * 64,
        },
        "configuration": {
            "rows": ray_count,
            "chunk_rows": 64,
            "outer_minimum_m": [-0.11, -0.11, -0.11],
            "outer_maximum_m": [0.11, 0.11, 0.11],
            "cone_vertex_m": [0.06, 0.015, 0.0],
            "cone_axis_normalized": [-0.995, -0.0995, 0.0],
            "cone_angle_degrees": 25.0,
            "geometry_contract_version": "psu-bost-forward-geometry-1.0",
            "b0_policy": (
                "FORWARD_NORMALIZED_RAY_INTERSECT_CLOSED_AXIS_ALIGNED_BOX"
            ),
            "b1_policy": "B0_INTERSECT_NORMALIZED_ONE_NAPPE_CONE_NO_FALLBACK",
        },
        "counts": {
            "ray_count": ray_count,
            "author_selected_nonzero_count": 95,
            "author_full_box_zero_flag_count": 10,
            "b0_hit_count": b0_hits,
            "b0_zero_length_count": ray_count - b0_hits,
            "b1_hit_count": b1_hits,
            "b1_zero_length_count": ray_count - b1_hits,
            "b1_removed_from_b0_count": b0_hits - b1_hits,
            "b0_endpoint_box_violation_count": 0,
            "b1_endpoint_box_violation_count": 0,
            "b1_endpoint_nappe_violation_count": 0,
            "b1_endpoint_cone_radial_violation_count": 0,
            "b1_midpoint_cone_violation_count": 0,
            "b1_hit_without_b0_hit_count": 0,
            "b1_length_exceeds_b0_count": 0,
            "nonfinite_output_count": 0,
            "b0_point_touch_count": 0,
            "b1_point_touch_count": 0,
            "b1_nappe_rejected_double_cone_count": 2,
            "b1_nappe_rejected_with_b0_hit_count": 2,
        },
        "path_length": {
            "author_selected_length_sum_m": author_length,
            "b0_length_sum_m": b0_length,
            "b1_length_sum_m": b1_length,
            "b1_fraction_of_b0": b1_length / b0_length,
            "b1_fraction_of_author_selected": b1_length / author_length,
            "b0_fraction_of_author_selected": b0_length / author_length,
        },
        "mask_conditioned": {
            "amask_all": active,
            "imask_all": inactive,
        },
        "runtime_observation": {
            "wall_seconds": 12345.6789,
            "scope": "CACHED_LOCAL_DIAGNOSTIC_NOT_A_SPEED_BENCHMARK",
        },
        "decision": {
            "analytic_domain_invariants_pass": True,
            "declared_computational_domain_mechanically_enforced": True,
            "physical_spatial_domain_validated": False,
            "cone_parameter_physical_semantics_confirmed": False,
            "finite_aperture_sample_support_audited": False,
            "training_ready": "NO",
            "algorithm_superiority_claim": "LOCKED",
            "next_gate": exporter.NEXT_GATE,
        },
        "limitations": ["private diagnostic detail"],
        "upstream_view_contract": {
            "bundle_status": exporter.BUNDLE_STATUS,
            "setup_status": "STREAMED_SETUP_MECHANICAL_CONTRACT_PASS",
            "mask_status": exporter.MASK_STATUS,
        },
    }


def _private_report() -> dict:
    views = [_view(0, active_shortfall=True), _view(1)]
    counts = {
        field: sum(view["counts"][field] for view in views)
        for field in exporter.COUNT_FIELDS
    }
    path_sums = {
        field: sum(view["path_length"][field] for view in views)
        for field in exporter.PATH_SUM_FIELDS
    }
    aggregate_masks = {}
    for mask_name in ("amask_all", "imask_all"):
        pooled = {
            field: sum(
                view["mask_conditioned"][mask_name][field] for view in views
            )
            for field in exporter.MASK_COUNT_FIELDS + exporter.MASK_SUM_FIELDS
        }
        pooled.update(
            {
                "author_nonzero_fraction": (
                    pooled["author_nonzero_count"] / pooled["count"]
                ),
                "b0_hit_fraction": pooled["b0_hit_count"] / pooled["count"],
                "b1_hit_fraction": pooled["b1_hit_count"] / pooled["count"],
                "b1_removed_from_b0_fraction": (
                    pooled["b1_removed_from_b0_count"] / pooled["count"]
                ),
                "b1_path_fraction_of_b0": (
                    pooled["b1_length_sum_m"] / pooled["b0_length_sum_m"]
                ),
            }
        )
        aggregate_masks[mask_name] = pooled
    return {
        "schema_version": exporter.PRIVATE_SCHEMA_VERSION,
        "execution_status": exporter.EXECUTION_STATUS,
        "scientific_verdict": exporter.SCIENTIFIC_VERDICT,
        "status": exporter.REPORT_STATUS,
        "view_count": len(views),
        "views": views,
        "aggregate": {
            "invalid_view_ids": [],
            "active_mask_b1_zero_view_ids": [0],
            "upstream_author_setup_no_go_view_ids": [],
            "counts": counts,
            "mask_conditioned": aggregate_masks,
            "path_length": {
                **path_sums,
                "b1_fraction_of_b0": (
                    path_sums["b1_length_sum_m"] / path_sums["b0_length_sum_m"]
                ),
                "b1_fraction_of_author_selected": (
                    path_sums["b1_length_sum_m"]
                    / path_sums["author_selected_length_sum_m"]
                ),
                "b0_fraction_of_author_selected": (
                    path_sums["b0_length_sum_m"]
                    / path_sums["author_selected_length_sum_m"]
                ),
            },
        },
        "decision": {
            **copy.deepcopy(views[0]["decision"]),
            "author_mixed_domain_length_comparison_is_context_only": True,
        },
        "limitations": ["aggregate private diagnostic detail"],
    }


def _add_private_leaks(report: dict) -> None:
    report["local_report_path"] = (
        "/Users/alice/private_library/fixed_domain_geometry_audit.json"
    )
    report["raw_ray_indices"] = [123456789, 987654321]
    report["source_snippet"] = "def private_formula(): return secret"
    report["views"][0]["source"]["local_manifest_path"] = (
        "/Volumes/private/view_00/manifest.json"
    )
    report["views"][0]["runtime_observation"]["private_log"] = (
        "file:///Users/alice/audit.log"
    )
    report["views"][0]["configuration"]["private_vector"] = [123456789.25]
    report["views"][0]["counts"]["raw_hit_indices"] = [1, 2, 3]


def test_public_summary_is_strict_allowlist_with_required_claim_boundary() -> None:
    private = _private_report()
    private_hashes = {
        value
        for view in private["views"]
        for key, value in view["source"].items()
        if key.endswith("sha256")
    }
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
        "view_count",
        "aggregate",
        "views",
        "claim_boundary",
        "public_export_policy",
    }
    assert result["execution_status"] == "COMPLETE"
    assert result["scientific_verdict"] == exporter.SCIENTIFIC_VERDICT
    assert result["aggregate"]["counts"] == {
        "ray_count": 200,
        "b0_hit_count": 180,
        "b1_hit_count": 120,
        "b1_removed_from_b0_count": 60,
    }
    assert result["aggregate"]["mask_conditioned"]["active"] == {
        "count": 80,
        "b0_hit_count": 80,
        "b1_hit_count": 79,
        "b0_hit_fraction": 1.0,
        "b1_hit_fraction": 79 / 80,
        "b1_path_fraction_of_b0": 0.4,
    }
    assert result["views"][0]["hit_fractions"]["b0_hit_fraction"] == 0.9
    assert result["views"][0]["path_length"]["b1_fraction_of_b0"] == 0.25
    assert result["views"][0]["mask_conditioned"] == {
        "active_b1_hit_fraction": 39 / 40,
        "inactive_b1_hit_fraction": 0.4,
    }
    assert result["claim_boundary"] == {
        "supported_claim": "B0_B1_DECLARED_COMPUTATIONAL_DOMAIN_MECHANICALLY_ENFORCED",
        "b0_b1_declared_computational_domain_mechanically_enforced": True,
        "physical_spatial_domain_validated": False,
        "cone_physical_semantics": "UNCONFIRMED",
        "b2_finite_aperture_support": "MISSING",
        "held_out_reprojection": "MISSING",
        "training": "LOCKED",
        "algorithm_superiority": "LOCKED",
    }

    for secret in (
        "/Users/",
        "/Volumes/",
        "file://",
        "private_library",
        "12345.6789",
        "123456789",
        "987654321",
        "private_formula",
        "runtime_observation",
        "raw_ray_indices",
        "raw_hit_indices",
        "source_snippet",
        "private_vector",
        "sha256",
    ):
        assert secret not in payload
    for private_hash in private_hashes:
        assert private_hash not in payload


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda report: report.update(schema_version="wrong-schema"),
            "report.schema_version has an unreviewed value",
        ),
        (lambda report: report.pop("views"), "report.views is required"),
        (lambda report: report.update(view_count=3), "view_count must equal"),
        (
            lambda report: report["views"][1].update(view_id_zero_based=0),
            "ordered contiguous range",
        ),
        (
            lambda report: report["views"][0]["decision"].update(
                cone_parameter_physical_semantics_confirmed=True
            ),
            "conflicts with the public claim boundary",
        ),
        (
            lambda report: report["views"][0]["counts"].update(
                b1_removed_from_b0_count=29
            ),
            "b1_removed_from_b0_count is inconsistent",
        ),
        (
            lambda report: report["views"][0]["source"].update(
                geometry_implementation_filename="/Users/alice/private.py"
            ),
            "basename without directories",
        ),
        (
            lambda report: report["views"][0]["mask_conditioned"][
                "amask_all"
            ].update(b1_hit_fraction=0.5),
            "b1_hit_fraction is inconsistent",
        ),
    ],
)
def test_public_summary_rejects_malformed_or_inconsistent_views(
    mutation, message
) -> None:
    private = _private_report()
    mutation(private)
    with pytest.raises(ValueError, match=message):
        exporter.build_public_summary(private)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda report: report["aggregate"]["counts"].update(ray_count=201),
            "ray_count conflicts with per-view data",
        ),
        (
            lambda report: report["aggregate"]["path_length"].update(
                b1_length_sum_m=61.0
            ),
            "b1_fraction_of_b0 is inconsistent",
        ),
        (
            lambda report: report["aggregate"].update(
                active_mask_b1_zero_view_ids=[]
            ),
            "conflicts with per-view masks",
        ),
        (
            lambda report: report["aggregate"]["mask_conditioned"][
                "amask_all"
            ].update(b1_hit_count=78),
            "b1_removed_from_b0_count is inconsistent",
        ),
        (
            lambda report: report["aggregate"]["mask_conditioned"][
                "imask_all"
            ].update(b0_length_sum_m=121.0),
            "b1_path_fraction_of_b0 is inconsistent",
        ),
    ],
)
def test_public_summary_rejects_inconsistent_aggregates(mutation, message) -> None:
    private = _private_report()
    mutation(private)
    with pytest.raises(ValueError, match=message):
        exporter.build_public_summary(private)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda report: report["views"][0]["path_length"].update(
            b0_length_sum_m=float("nan")
        ),
        lambda report: report["views"][0]["runtime_observation"].update(
            wall_seconds=float("inf")
        ),
        lambda report: report["views"][0]["configuration"].update(
            cone_axis_normalized=[0.0, float("-inf"), 0.0]
        ),
    ],
)
def test_public_summary_rejects_nonfinite_values_anywhere(mutation) -> None:
    private = _private_report()
    mutation(private)
    with pytest.raises(ValueError, match="non-finite|finite number"):
        exporter.build_public_summary(private)


def test_loader_rejects_nonstandard_json_constants(tmp_path: Path) -> None:
    input_path = tmp_path / "private.json"
    input_path.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-standard JSON constant"):
        exporter.load_private_report(input_path)


def test_export_is_atomic_and_invalid_input_does_not_replace_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "private" / "fixed_domain_geometry_audit.json"
    output_path = tmp_path / "docs" / "fixed_domain_public_summary.json"
    input_path.parent.mkdir()
    input_path.write_text(json.dumps(_private_report()), encoding="utf-8")

    replace_calls = []
    real_replace = exporter.os.replace

    def record_replace(source: Path, destination: Path) -> None:
        replace_calls.append((source, destination))
        real_replace(source, destination)

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


def test_atomic_write_cleans_up_after_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_path = tmp_path / "public.json"
    output_path.write_text('{"old": true}\n', encoding="utf-8")
    original = output_path.read_bytes()

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(exporter.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        exporter.write_json_atomic({"new": True}, output_path)

    assert output_path.read_bytes() == original
    assert list(tmp_path.glob(f".{output_path.name}.*.tmp")) == []


@pytest.mark.skipif(
    not REAL_PRIVATE_REPORT.exists(),
    reason="private PSU fixed-domain audit is not present in this checkout",
)
def test_real_private_report_matches_public_contract() -> None:
    private = exporter.load_private_report(REAL_PRIVATE_REPORT)
    result = exporter.build_public_summary(private)
    payload = json.dumps(result, sort_keys=True)

    assert result["view_count"] == 9
    assert result["aggregate"]["counts"]["ray_count"] == 49_766_400
    assert result["aggregate"]["counts"]["b0_hit_count"] == 49_515_803
    assert result["aggregate"]["counts"]["b1_hit_count"] == 22_924_319
    assert result["aggregate"]["active_mask_b1_shortfall_view_count"] == 1
    assert result["views"][0]["mask_conditioned"][
        "active_b1_hit_fraction"
    ] == pytest.approx(0.9986679112651291)
    assert result["views"][1]["mask_conditioned"]["active_b1_hit_fraction"] == 1.0
    assert result["claim_boundary"]["cone_physical_semantics"] == "UNCONFIRMED"
    assert result["claim_boundary"][
        "b0_b1_declared_computational_domain_mechanically_enforced"
    ] is True
    assert result["claim_boundary"]["physical_spatial_domain_validated"] is False
    assert result["claim_boundary"]["b2_finite_aperture_support"] == "MISSING"
    assert result["claim_boundary"]["held_out_reprojection"] == "MISSING"
    assert result["claim_boundary"]["training"] == "LOCKED"
    assert result["claim_boundary"]["algorithm_superiority"] == "LOCKED"
    for forbidden in (
        "runtime_observation",
        "invalid_view_ids",
        "active_mask_b1_zero_view_ids",
        "sha256",
        "private_library",
    ):
        assert forbidden not in payload
    assert all("configuration" not in view for view in result["views"])
    assert all("source" not in view for view in result["views"])
