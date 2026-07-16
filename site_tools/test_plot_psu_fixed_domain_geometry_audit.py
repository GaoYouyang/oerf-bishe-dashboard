from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
from PIL import Image, ImageStat

import site_tools.plot_psu_fixed_domain_geometry_audit as plotter
from site_tools.plot_psu_fixed_domain_geometry_audit import (
    CAPTION,
    DEFAULT_OUTPUT_STEM,
    FIGURE_SIZE_INCHES,
    PANEL_METRICS,
    PNG_DPI,
    build_fixed_domain_geometry_audit_figure,
    load_plot_records,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decision() -> dict[str, object]:
    return {
        "analytic_domain_invariants_pass": True,
        "declared_computational_domain_mechanically_enforced": True,
        "physical_spatial_domain_validated": False,
        "cone_parameter_physical_semantics_confirmed": False,
        "finite_aperture_sample_support_audited": False,
        "training_ready": "NO",
        "algorithm_superiority_claim": "LOCKED",
        "next_gate": ("B2_FINITE_APERTURE_DOMAIN_INDICATOR_AND_B3_GEOMETRY_SAFE_MASK"),
    }


def _mask_record(
    *,
    count: int,
    b0_hit: int,
    b1_hit: int,
    author_length: float,
    b0_length: float,
    b1_length: float,
) -> dict[str, int | float]:
    removed = b0_hit - b1_hit
    return {
        "count": count,
        "author_nonzero_count": count,
        "author_nonzero_fraction": 1.0,
        "b0_hit_count": b0_hit,
        "b0_hit_fraction": b0_hit / count,
        "b1_hit_count": b1_hit,
        "b1_hit_fraction": b1_hit / count,
        "b1_removed_from_b0_count": removed,
        "b1_removed_from_b0_fraction": removed / count,
        "author_length_sum_m": author_length,
        "b0_length_sum_m": b0_length,
        "b1_length_sum_m": b1_length,
        "b1_path_fraction_of_b0": b1_length / b0_length,
    }


def _view_record(view_id: int) -> dict[str, object]:
    ray_count = 1000
    b0_miss = 20 if view_id in {0, 3} else 0
    b0_hit = ray_count - b0_miss
    removed = 500 + 5 * view_id
    b1_hit = b0_hit - removed
    author_length = 800.0 + 10.0 * view_id
    b0_length = 1200.0 + 15.0 * view_id
    b1_length = 180.0 + 3.0 * view_id
    active_b1_hit = 299 if view_id == 0 else 300
    counts = {
        "author_full_box_zero_flag_count": b0_miss,
        "author_selected_nonzero_count": ray_count - b0_miss,
        "b0_endpoint_box_violation_count": 0,
        "b0_hit_count": b0_hit,
        "b0_point_touch_count": 0,
        "b0_zero_length_count": b0_miss,
        "b1_endpoint_box_violation_count": 0,
        "b1_endpoint_nappe_violation_count": 0,
        "b1_endpoint_cone_radial_violation_count": 0,
        "b1_midpoint_cone_violation_count": 0,
        "b1_hit_count": b1_hit,
        "b1_hit_without_b0_hit_count": 0,
        "b1_length_exceeds_b0_count": 0,
        "b1_nappe_rejected_double_cone_count": 10 + view_id,
        "b1_nappe_rejected_with_b0_hit_count": 10 + view_id,
        "b1_point_touch_count": 0,
        "b1_removed_from_b0_count": removed,
        "b1_zero_length_count": ray_count - b1_hit,
        "nonfinite_output_count": 0,
        "ray_count": ray_count,
    }
    return {
        "schema_version": "psu-bost-fixed-domain-geometry-audit-1.0",
        "status": "B0_B1_FIXED_DOMAIN_ANALYTIC_CONTRACT_PASS_B2_REQUIRED",
        "evidence_scope": (
            "REAL_ONE_VIEW_B0_FORWARD_BOX_AND_B1_ONE_NAPPE_CONE_BOX_RAY_CENSUS_"
            "NO_TENSORFLOW_NO_RECONSTRUCTION"
        ),
        "view_id_zero_based": view_id,
        "source": {
            "audit_implementation_sha256": f"{view_id + 1:064x}",
            "bundle_manifest_sha256": f"{view_id + 11:064x}",
            "corrected_mask_manifest_sha256": f"{view_id + 21:064x}",
            "geometry_implementation_filename": "psu_bost_forward_geometry.py",
            "geometry_implementation_sha256": "a" * 64,
            "setup_manifest_sha256": f"{view_id + 31:064x}",
        },
        "configuration": {
            "rows": ray_count,
            "chunk_rows": 256,
            "outer_minimum_m": [-0.11, -0.11, -0.11],
            "outer_maximum_m": [0.11, 0.11, 0.11],
            "cone_vertex_m": [0.06, 0.015, 0.0],
            "cone_axis_normalized": [-1.0, 0.0, 0.0],
            "cone_angle_degrees": 25.0,
            "geometry_contract_version": "psu-bost-forward-geometry-1.0",
            "b0_policy": ("FORWARD_NORMALIZED_RAY_INTERSECT_CLOSED_AXIS_ALIGNED_BOX"),
            "b1_policy": "B0_INTERSECT_NORMALIZED_ONE_NAPPE_CONE_NO_FALLBACK",
        },
        "counts": counts,
        "path_length": {
            "author_selected_length_sum_m": author_length,
            "b0_length_sum_m": b0_length,
            "b1_length_sum_m": b1_length,
            "b1_fraction_of_b0": b1_length / b0_length,
            "b1_fraction_of_author_selected": b1_length / author_length,
            "b0_fraction_of_author_selected": b0_length / author_length,
        },
        "mask_conditioned": {
            "amask_all": _mask_record(
                count=300,
                b0_hit=300,
                b1_hit=active_b1_hit,
                author_length=100.0,
                b0_length=280.0,
                b1_length=100.0 - 0.1 * (view_id == 0),
            ),
            "imask_all": _mask_record(
                count=500,
                b0_hit=490,
                b1_hit=55 + 8 * view_id,
                author_length=300.0,
                b0_length=360.0,
                b1_length=7.0 + view_id,
            ),
        },
        "decision": _decision(),
        "limitations": [
            (
                "B1 reuses the released cone vertex, axis, and angle only as a "
                "computational-domain hypothesis; their physical meaning is not "
                "independently confirmed"
            ),
            (
                "centerline domain validity does not prove that finite-aperture "
                "samples stay inside the same domain"
            ),
            (
                "the active and inactive masks are diagnostic labels rather than "
                "density or refractive-index ground truth"
            ),
            (
                "no held-out camera, TensorFlow NIRT, neural field, inverse "
                "reconstruction, or superiority comparison is run"
            ),
        ],
        "upstream_view_contract": {
            "bundle_status": (
                "VIEW_SHARD_BUNDLE_TRANSCODED_AND_SOURCE_STREAMS_VERIFIED"
            ),
            "setup_status": "STREAMED_SETUP_MECHANICAL_CONTRACT_PASS",
            "mask_status": (
                "CORRECTED_VIEW_MASK_SHARDS_BUILT_MECHANICAL_CONTRACT_PASS"
            ),
        },
    }


def _synthetic_report(root: Path) -> tuple[Path, dict[str, object]]:
    views = [_view_record(view_id) for view_id in range(5)]
    count_fields = views[0]["counts"].keys()
    aggregate_counts = {
        key: sum(int(view["counts"][key]) for view in views) for key in count_fields
    }
    length_fields = (
        "author_selected_length_sum_m",
        "b0_length_sum_m",
        "b1_length_sum_m",
    )
    aggregate_lengths = {
        key: sum(float(view["path_length"][key]) for view in views)
        for key in length_fields
    }
    report = {
        "schema_version": "psu-bost-fixed-domain-all-view-audit-1.0",
        "execution_status": "COMPLETE",
        "scientific_verdict": (
            "MECHANICAL_PASS_PHYSICAL_CONE_SEMANTICS_AND_FINITE_APERTURE_UNCONFIRMED"
        ),
        "status": "B0_B1_ALL_VIEW_ANALYTIC_CONTRACT_PASS_B2_REQUIRED",
        "view_count": len(views),
        "views": views,
        "aggregate": {
            "invalid_view_ids": [],
            "active_mask_b1_zero_view_ids": [0],
            "upstream_author_setup_no_go_view_ids": [0, 3],
            "counts": aggregate_counts,
            "path_length": {
                **aggregate_lengths,
                "b1_fraction_of_b0": (
                    aggregate_lengths["b1_length_sum_m"]
                    / aggregate_lengths["b0_length_sum_m"]
                ),
                "b1_fraction_of_author_selected": (
                    aggregate_lengths["b1_length_sum_m"]
                    / aggregate_lengths["author_selected_length_sum_m"]
                ),
                "b0_fraction_of_author_selected": (
                    aggregate_lengths["b0_length_sum_m"]
                    / aggregate_lengths["author_selected_length_sum_m"]
                ),
            },
        },
        "decision": {
            **_decision(),
            "author_mixed_domain_length_comparison_is_context_only": True,
        },
        "limitations": [
            (
                "B0/B1 are deterministic geometry baselines and do not measure "
                "reconstruction quality"
            ),
            (
                "the released 25 degree cone is treated as a computational "
                "sampling hull, not a shock or Mach angle"
            ),
            (
                "B2 finite-aperture support and held-out reprojection are required "
                "before inverse or neural-operator comparison"
            ),
            (
                "no statistical uncertainty interval applies to this exhaustive "
                "ray census"
            ),
        ],
    }
    report_path = root / "fixed_domain_geometry_audit.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_path, report


def test_plot_values_and_markers_are_derived_from_validated_report(
    tmp_path: Path,
) -> None:
    report_path, report = _synthetic_report(tmp_path)
    records, provenance = load_plot_records(report_path)
    source = report["views"][3]

    assert records[3]["b0_hit_fraction"] == pytest.approx(
        source["counts"]["b0_hit_count"] / source["counts"]["ray_count"]
    )
    assert records[3]["b1_hit_fraction"] == pytest.approx(
        source["counts"]["b1_hit_count"] / source["counts"]["ray_count"]
    )
    assert records[3]["b1_path_fraction_of_b0"] == pytest.approx(
        source["path_length"]["b1_fraction_of_b0"]
    )
    assert records[0]["active_b1_hit_fraction"] == pytest.approx(299 / 300)
    assert provenance["b0_miss_view_ids_zero_based"] == [0, 3]
    assert provenance["active_b1_miss_view_ids_zero_based"] == [0]
    assert provenance["cone_angle_degrees"] == 25.0
    assert (
        provenance["source_sha256"]
        == hashlib.sha256(report_path.read_bytes()).hexdigest()
    )


def test_builds_deterministic_publication_outputs_and_strict_manifest(
    tmp_path: Path,
) -> None:
    report_path, _report = _synthetic_report(tmp_path)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first = build_fixed_domain_geometry_audit_figure(report_path, first_dir)
    second = build_fixed_domain_geometry_audit_figure(report_path, second_dir)

    manifest_name = f"{DEFAULT_OUTPUT_STEM}_manifest.json"
    assert first == second
    assert first["status"] == "FIGURE_BUILD_COMPLETE"
    assert first["hash_algorithm"] == "sha256"
    assert first["caption"] == CAPTION
    assert "no error bars" in first["caption"]
    assert "Not a reconstruction result" in first["caption"]
    assert "not physical validation" in first["caption"]
    assert first["b0_miss_view_ids_zero_based"] == [0, 3]
    assert first["active_b1_miss_view_ids_zero_based"] == [0]
    assert first["panels"] == {
        key: list(metrics) for key, metrics in PANEL_METRICS.items()
    }
    assert first["claim_boundary"]["physical_validation_of_cone_angle"] is False
    assert first["claim_boundary"]["algorithm_superiority_claim"] == "LOCKED"
    assert (
        first["data_contract"]["numeric_source"]
        == "all plotted values are derived from the validated report JSON; "
        "no dataset values or view ids are embedded in the plotting source"
    )

    for output_format in ("png", "svg", "pdf"):
        details = first["outputs"][output_format]
        first_path = first_dir / details["filename"]
        second_path = second_dir / details["filename"]
        assert first_path.is_file()
        assert second_path.is_file()
        assert first_path.stat().st_size > 1000
        assert details["sha256"] == _sha256(first_path)
        assert details["sha256"] == _sha256(second_path)

    expected_size = (
        round(FIGURE_SIZE_INCHES[0] * PNG_DPI),
        round(FIGURE_SIZE_INCHES[1] * PNG_DPI),
    )
    assert expected_size == (3300, 2220)
    png_path = first_dir / first["outputs"]["png"]["filename"]
    with Image.open(png_path) as image:
        image.load()
        assert image.format == "PNG"
        assert image.size == expected_size
        assert image.mode in {"RGB", "RGBA"}
        statistics = ImageStat.Stat(image.convert("L"))
        assert statistics.extrema[0][0] < 80
        assert statistics.var[0] > 25.0
    assert first["outputs"]["png"]["width_pixels"] == 3300
    assert first["outputs"]["png"]["height_pixels"] == 2220

    stored_manifest = json.loads(
        (first_dir / manifest_name).read_text(encoding="utf-8")
    )
    assert stored_manifest == first
    svg_text = (first_dir / first["outputs"]["svg"]["filename"]).read_text(
        encoding="utf-8"
    )
    assert "Deterministic centerline geometry census" in svg_text
    assert "Gray bands: views with B0 centerline misses (0, 3)" in svg_text
    assert "Star marker: active-mask B1 misses (0)" in svg_text
    assert not re.search(r"letter-spacing\s*:\s*-", svg_text)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda report: report.__setitem__("schema_version", "wrong-schema"),
            "report.schema_version",
        ),
        (
            lambda report: report.__setitem__(
                "status", "B0_B1_ALL_VIEW_ANALYTIC_CONTRACT_INVALID"
            ),
            "report.status",
        ),
        (
            lambda report: report["decision"].__setitem__(
                "cone_parameter_physical_semantics_confirmed", True
            ),
            "cone_parameter_physical_semantics_confirmed",
        ),
        (
            lambda report: report["limitations"].pop(),
            "claim boundary",
        ),
    ],
)
def test_rejects_schema_status_and_claim_boundary_before_rendering(
    tmp_path: Path,
    mutator,
    message: str,
) -> None:
    report_path, report = _synthetic_report(tmp_path)
    mutator(report)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    output_dir = tmp_path / "outputs"

    with pytest.raises(ValueError, match=message):
        build_fixed_domain_geometry_audit_figure(report_path, output_dir)

    assert not output_dir.exists()


def test_rejects_cross_field_and_aggregate_inconsistency(tmp_path: Path) -> None:
    report_path, report = _synthetic_report(tmp_path)
    report["views"][2]["counts"]["b1_removed_from_b0_count"] += 1
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="conflicts with B0/B1 hit counts"):
        build_fixed_domain_geometry_audit_figure(report_path, tmp_path / "outputs")

    report_path, report = _synthetic_report(tmp_path)
    report["aggregate"]["active_mask_b1_zero_view_ids"] = [0, 4]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="active_mask_b1_zero_view_ids conflicts"):
        build_fixed_domain_geometry_audit_figure(report_path, tmp_path / "outputs")


def test_staging_failure_preserves_existing_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_path, _report = _synthetic_report(tmp_path)
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    sentinel = output_dir / f"{DEFAULT_OUTPUT_STEM}.png"
    sentinel.write_bytes(b"existing-publication")

    def fail_save(*_args, **_kwargs):
        raise RuntimeError("synthetic render failure")

    monkeypatch.setattr(plotter, "_save_staged_outputs", fail_save)
    with pytest.raises(RuntimeError, match="synthetic render failure"):
        build_fixed_domain_geometry_audit_figure(report_path, output_dir)

    assert sentinel.read_bytes() == b"existing-publication"
    assert sorted(path.name for path in output_dir.iterdir()) == [sentinel.name]
