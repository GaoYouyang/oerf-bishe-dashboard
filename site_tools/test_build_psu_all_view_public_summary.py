from __future__ import annotations

import json

import pytest

from site_tools import build_psu_all_view_public_summary as exporter


def _view(view_id: int, *, no_go: bool) -> dict:
    unsafe_count = 2 if no_go else 0
    return {
        "view_id_zero_based": view_id,
        "measurement_count": 100,
        "bundle_status": "VIEW_SHARD_BUNDLE_TRANSCODED_AND_SOURCE_STREAMS_VERIFIED",
        "mask_status": "CORRECTED_VIEW_MASK_SHARDS_BUILT_MECHANICAL_CONTRACT_PASS",
        "setup_status": (
            "STREAMED_SETUP_DIAGNOSTIC_NO_GO"
            if no_go
            else "STREAMED_SETUP_MECHANICAL_CONTRACT_PASS"
        ),
        "full_box_zero_count": unsafe_count,
        "full_box_zero_fraction": unsafe_count / 100,
        "box_miss_but_cone_nonzero_count": unsafe_count,
        "box_miss_but_cone_nonzero_fraction": unsafe_count / 100,
        "final_zero_length_count": unsafe_count,
        "final_zero_length_fraction": unsafe_count / 100,
        "cone_outside_ray_count": unsafe_count,
        "cone_outside_ray_fraction": unsafe_count / 100,
        "cone_no_box_overlap_count": unsafe_count,
        "cone_length_weighted_outside_box_fraction": unsafe_count / 100,
        "active_count": 40,
        "active_unsafe_geometry_count": 0,
        "active_unsafe_geometry_fraction": 0.0,
        "inactive_count": 50,
        "inactive_unsafe_geometry_count": unsafe_count,
        "inactive_unsafe_geometry_fraction": unsafe_count / 50,
        "active_rms_magnitude_pixels": 0.2 + view_id * 0.1,
        "inactive_rms_magnitude_pixels": 0.1 + view_id * 0.1,
        "active_to_inactive_rms_ratio": 2.0,
        "active_shift_vector_rmse_pixels": 0.01,
        "inactive_shift_vector_rmse_pixels": 0.02,
    }


def _private_report() -> dict:
    views = [_view(0, no_go=True), _view(1, no_go=False)]
    metric_summary = {}
    for metric in exporter.SUMMARY_METRICS:
        values = [view[metric] for view in views]
        metric_summary[metric] = {
            "minimum": min(values),
            "minimum_view_id": values.index(min(values)),
            "mean": sum(values) / len(values),
            "maximum": max(values),
            "maximum_view_id": values.index(max(values)),
        }
    return {
        "schema_version": exporter.PRIVATE_SCHEMA_VERSION,
        "execution_status": "COMPLETE",
        "scientific_verdict": "NO_GO",
        "status": "ALL_VIEW_GEOMETRY_AUDIT_NO_GO",
        "evidence_scope": exporter.EVIDENCE_SCOPE,
        "view_count": 2,
        "views": views,
        "metric_summary": metric_summary,
        "pooled_geometry": {
            "ray_count": 200,
            "full_box_zero_count": 2,
            "box_miss_but_cone_nonzero_count": 2,
            "final_zero_length_count": 2,
            "cone_segment_length_sum_m": 20.0,
            "cone_box_overlap_length_sum_m": 19.6,
            "cone_outside_length_sum_m": 0.4,
            "cone_length_weighted_outside_box_fraction": 0.02,
        },
        "prevalence": {
            "views_with_full_box_zero_rays": 1,
            "views_with_cone_outside_box_rays": 1,
            "views_with_active_unsafe_geometry": 0,
            "views_with_inactive_unsafe_geometry": 1,
            "setup_no_go_view_ids": [0],
        },
        "decision": {
            "geometry_problem_is_single_view_artifact": "UNRESOLVED",
            "official_setup_ready_for_training": "NO_GO",
            "algorithm_success_claim": "LOCKED",
            "next_gate": (
                "DOMAIN_CLIPPED_GEOMETRY_BASELINE_AND_GEOMETRY_SAFE_MASK_ABLATION"
            ),
        },
        "limitations": [
            "cross-view aggregates are geometry diagnostics, not reconstruction evidence"
        ],
        "source": {
            "mat_filename": "fixture.mat",
            "mat_sha256": "1" * 64,
            "geometry_source_filename": "meas.py",
            "geometry_source_sha256": "2" * 64,
        },
        "run_contract_sha256": "3" * 64,
        "run_contract": {
            "code_provenance": {
                "artifact_generator_source_binding_at_initial_generation": (
                    "NOT_RECORDED_NUMERIC_ARTIFACTS_REVALIDATED_BY_FILE_HASH"
                )
            }
        },
    }


def _add_private_fields(report: dict) -> None:
    report["source"]["mat_path"] = "/Users/alice/private/HSOF_9CAM_RT.mat"
    report["runtime_observation"] = {
        "wall_seconds": 314159.265,
        "scope": "CACHED_LOCAL_SSD_RUN_NOT_A_SPEED_BENCHMARK",
    }
    report["raw_arrays"] = [[123456789.0, 987654321.0]]
    report["measurement_indices"] = [0, 1, 2]
    report["author_source_code"] = "def private_formula(): return secret"
    report["views"][0].update(
        {
            "local_bundle_path": "/Volumes/private/view_00",
            "raw_deflections": [123456789.0],
            "ray_indices": [11, 12],
            "source_snippet": "import secret_module",
        }
    )
    report["metric_summary"]["full_box_zero_fraction"]["private_samples"] = [
        0.123456789
    ]
    report["prevalence"]["private_index_path"] = "/tmp/private/indices.npy"
    report["decision"]["local_source_note"] = "file:///Users/alice/source.py"


def test_public_summary_is_a_strict_aggregate_allowlist() -> None:
    private = _private_report()
    _add_private_fields(private)

    result = exporter.build_public_summary(private)
    payload = json.dumps(result, sort_keys=True)

    assert set(result) == {
        "schema_version",
        "source_schema_version",
        "execution_status",
        "scientific_verdict",
        "status",
        "evidence_scope",
        "view_count",
        "views",
        "metric_summary",
        "pooled_geometry",
        "prevalence",
        "decision",
        "limitations",
        "provenance",
        "claim_boundary",
        "public_export_policy",
    }
    assert result["status"] == private["status"]
    assert result["execution_status"] == "COMPLETE"
    assert result["scientific_verdict"] == "NO_GO"
    assert result["evidence_scope"] == private["evidence_scope"]
    assert result["views"][0]["setup_status"] == private["views"][0][
        "setup_status"
    ]
    assert result["views"][0]["inactive_unsafe_geometry_count"] == 2
    assert result["metric_summary"] == {
        name: {key: item[key] for key in exporter.SUMMARY_VALUE_FIELDS + exporter.SUMMARY_VIEW_FIELDS}
        for name, item in private["metric_summary"].items()
    }
    assert result["pooled_geometry"] == private["pooled_geometry"]
    assert result["prevalence"]["setup_no_go_view_count"] == 1
    assert "setup_no_go_view_ids" not in result["prevalence"]
    assert result["decision"] == {
        field: private["decision"][field] for field in exporter.DECISION_FIELDS
    }
    assert result["limitations"] == private["limitations"]
    assert result["provenance"]["mat_sha256"] == "1" * 64
    assert result["provenance"]["run_contract_sha256"] == "3" * 64
    assert result["claim_boundary"]["algorithm_superiority_established"] is False
    assert result["public_export_policy"]["aggregate_only"] is True

    for secret in (
        "/Users/",
        "/Volumes/",
        "/tmp/",
        "private_library",
        "file://",
        "314159.265",
        "123456789",
        "987654321",
        "private_formula",
        "secret_module",
        "raw_arrays",
        "measurement_indices",
        "raw_deflections",
        "ray_indices",
        "source_snippet",
        "runtime_observation",
        "wall_seconds",
    ):
        assert secret not in payload


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
            "view ids must be unique",
        ),
        (
            lambda report: report["prevalence"].update(
                views_with_full_box_zero_rays=0
            ),
            "conflicts with per-view counts",
        ),
    ],
)
def test_public_summary_rejects_wrong_or_malformed_schema(mutation, message) -> None:
    private = _private_report()
    mutation(private)
    with pytest.raises(ValueError, match=message):
        exporter.build_public_summary(private)


def test_public_summary_rejects_path_in_preserved_text() -> None:
    private = _private_report()
    private["limitations"].append("debug artifact at /Users/alice/private.json")
    with pytest.raises(ValueError, match="private or local path"):
        exporter.build_public_summary(private)


def test_export_is_atomic_and_invalid_json_does_not_replace_output(
    tmp_path, monkeypatch
) -> None:
    input_path = tmp_path / "private" / "all_view_geometry_audit.json"
    output_path = tmp_path / "docs" / "psu_all_view_geometry_summary.json"
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
