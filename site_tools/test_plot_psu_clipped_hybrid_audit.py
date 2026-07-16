from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
from PIL import Image, ImageStat

from site_tools.plot_psu_clipped_hybrid_audit import (
    CAPTION,
    DEFAULT_OUTPUT_STEM,
    FIGURE_SIZE_INCHES,
    PANEL_METRICS,
    PNG_DPI,
    build_clipped_hybrid_audit_figure,
    load_plot_records,
)


VIEW_SCHEMA = "psu-bost-author-compatible-clipped-hybrid-audit-1.0"
ROOT_SCHEMA = "psu-bost-author-compatible-clipped-hybrid-all-view-audit-1.0"
VIEW_PASS_STATUS = "AUTHOR_COMPATIBLE_CLIPPED_HYBRID_CONTRACT_PASS"
VIEW_FILTER_STATUS = (
    "AUTHOR_COMPATIBLE_CLIPPED_HYBRID_CONTRACT_PASS_MASK_FILTER_REQUIRED"
)
ROOT_FILTER_STATUS = (
    "AUTHOR_COMPATIBLE_CLIPPED_HYBRID_ALL_VIEW_CONTRACT_PASS_MASK_FILTER_REQUIRED"
)
AUTHOR_POLICY = (
    "AUTHOR_CONE_INTERVAL_INTERSECT_FORWARD_BOX_"
    "AUTHOR_ZERO_CONE_FALLBACK_TO_FORWARD_BOX"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mask_record(
    *,
    count: int,
    changed: int,
    shortened: int,
    zeroed: int,
    author_length: float,
    clipped_length: float,
) -> dict[str, int | float]:
    return {
        "count": count,
        "changed_count": changed,
        "changed_fraction": changed / count,
        "shortened_count": shortened,
        "shortened_fraction": shortened / count,
        "zeroed_count": zeroed,
        "zeroed_fraction": zeroed / count,
        "author_length_sum_m": author_length,
        "clipped_length_sum_m": clipped_length,
        "path_length_retained_fraction": clipped_length / author_length,
    }


def _view_record(view_id: int, changed: int) -> dict:
    ray_count = 1000
    newly_zeroed = 0 if changed == 0 else 20 + view_id
    clipped_nonzero = ray_count - newly_zeroed
    cone_shortened = 0 if changed == 0 else max(1, changed // 2)
    forward_shortened = 0 if changed == 0 else max(1, changed // 10)
    author_length = 800.0 + 25.0 * view_id
    removed_length = 0.0 if changed == 0 else 12.0 + 2.0 * view_id
    clipped_length = author_length - removed_length
    filter_required = newly_zeroed > 0
    active_changed = 0 if changed == 0 else 4 + view_id
    inactive_changed = 0 if changed == 0 else 11 + view_id
    status = VIEW_FILTER_STATUS if filter_required else VIEW_PASS_STATUS
    return {
        "schema_version": VIEW_SCHEMA,
        "status": status,
        "view_id_zero_based": view_id,
        "source": {
            "view_bundle_manifest_sha256": f"{view_id + 1:064x}",
            "corrected_mask_manifest_sha256": f"{view_id + 11:064x}",
            "geometry_source_filename": "meas.py",
            "geometry_source_sha256": "a" * 64,
            "author_source_modified": False,
        },
        "configuration": {
            "rows": ray_count,
            "policy": AUTHOR_POLICY,
        },
        "counts": {
            "ray_count": ray_count,
            "author_nonzero_count": ray_count,
            "clipped_nonzero_count": clipped_nonzero,
            "changed_from_author_count": changed,
            "cone_shortened_count": cone_shortened,
            "cone_zeroed_for_no_box_overlap_count": newly_zeroed,
            "forward_box_shortened_count": forward_shortened,
            "clipped_zero_length_count": newly_zeroed,
            "clipped_endpoint_box_violation_count": 0,
            "clipped_length_exceeds_author_count": 0,
            "nonfinite_output_count": 0,
            "negative_clipped_inner_aperture_radius_count": 0,
            "negative_clipped_outer_aperture_radius_count": 0,
        },
        "path_length": {
            "author_length_sum_m": author_length,
            "clipped_length_sum_m": clipped_length,
            "removed_length_sum_m": removed_length,
            "retained_fraction": clipped_length / author_length,
            "removed_fraction": removed_length / author_length,
        },
        "mask_conditioned": {
            "amask_all": _mask_record(
                count=400,
                changed=active_changed,
                shortened=0 if changed == 0 else 2,
                zeroed=0 if changed == 0 else 1,
                author_length=120.0 + view_id,
                clipped_length=(
                    120.0 + view_id if changed == 0 else 119.5 + 0.9 * view_id
                ),
            ),
            "imask_all": _mask_record(
                count=600,
                changed=inactive_changed,
                shortened=0 if changed == 0 else 4,
                zeroed=0 if changed == 0 else 2,
                author_length=300.0 + 2.0 * view_id,
                clipped_length=(
                    300.0 + 2.0 * view_id if changed == 0 else 297.0 + 1.7 * view_id
                ),
            ),
        },
        "decision": {
            "positive_segments_inside_forward_box": True,
            "geometry_safe_zero_row_filter_required": filter_required,
            "fixed_spatial_domain_established": False,
            "training_ready": "NO",
            "algorithm_superiority_claim": "LOCKED",
        },
        "upstream_view_contract": {
            "bundle_status": (
                "VIEW_SHARD_BUNDLE_TRANSCODED_AND_SOURCE_STREAMS_VERIFIED"
            ),
            "mask_status": (
                "CORRECTED_VIEW_MASK_SHARDS_BUILT_MECHANICAL_CONTRACT_PASS"
            ),
        },
    }


def _synthetic_report(path: Path) -> tuple[Path, dict]:
    changes = [0, 120, 75, 0, 95]
    views = [_view_record(view_id, changed) for view_id, changed in enumerate(changes)]
    total_author = sum(view["path_length"]["author_length_sum_m"] for view in views)
    total_clipped = sum(view["path_length"]["clipped_length_sum_m"] for view in views)
    total_removed = total_author - total_clipped
    filter_ids = [
        view["view_id_zero_based"]
        for view in views
        if view["counts"]["clipped_zero_length_count"] > 0
    ]
    report = {
        "schema_version": ROOT_SCHEMA,
        "execution_status": "COMPLETE",
        "scientific_verdict": "AUTHOR_COMPATIBILITY_ABLATION_ONLY",
        "status": ROOT_FILTER_STATUS,
        "view_count": len(views),
        "views": views,
        "aggregate": {
            "invalid_view_ids": [],
            "zero_row_filter_required_view_ids": filter_ids,
            "author_length_sum_m": total_author,
            "clipped_length_sum_m": total_clipped,
            "removed_length_sum_m": total_removed,
            "retained_fraction": total_clipped / total_author,
            "removed_fraction": total_removed / total_author,
            "changed_ray_count": sum(
                view["counts"]["changed_from_author_count"] for view in views
            ),
            "clipped_zero_length_count": sum(
                view["counts"]["clipped_zero_length_count"] for view in views
            ),
        },
        "decision": {
            "domain_clipping_mechanically_valid": True,
            "fixed_spatial_domain_established": False,
            "training_ready": "NO",
            "algorithm_superiority_claim": "LOCKED",
        },
    }
    report_path = path / "domain_clipped_author_compatibility_audit.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_path, report


def test_plot_values_are_derived_from_validated_report(tmp_path: Path) -> None:
    report_path, report = _synthetic_report(tmp_path)
    records, provenance = load_plot_records(report_path)
    source = report["views"][1]
    plotted = records[1]

    assert plotted["path_removed_fraction"] == pytest.approx(
        source["path_length"]["removed_fraction"]
    )
    assert plotted["changed_ray_fraction"] == pytest.approx(
        source["counts"]["changed_from_author_count"] / source["counts"]["ray_count"]
    )
    assert plotted["shortened_ray_fraction"] == pytest.approx(
        (
            source["counts"]["cone_shortened_count"]
            + source["counts"]["forward_box_shortened_count"]
        )
        / source["counts"]["ray_count"]
    )
    assert plotted["zeroed_ray_fraction"] == pytest.approx(
        source["counts"]["clipped_zero_length_count"] / source["counts"]["ray_count"]
    )
    assert plotted["active_mask_changed_fraction"] == pytest.approx(
        source["mask_conditioned"]["amask_all"]["changed_fraction"]
    )
    assert plotted["inactive_path_retained_fraction"] == pytest.approx(
        source["mask_conditioned"]["imask_all"]["path_length_retained_fraction"]
    )
    assert provenance["changed_view_ids_zero_based"] == [1, 2, 4]
    assert (
        provenance["source_sha256"]
        == hashlib.sha256(report_path.read_bytes()).hexdigest()
    )


def test_builds_deterministic_publication_outputs_and_manifest(
    tmp_path: Path,
) -> None:
    report_path, _report = _synthetic_report(tmp_path)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first = build_clipped_hybrid_audit_figure(report_path, first_dir)
    second = build_clipped_hybrid_audit_figure(report_path, second_dir)

    manifest_name = f"{DEFAULT_OUTPUT_STEM}_manifest.json"
    assert first == second
    assert first["status"] == "FIGURE_BUILD_COMPLETE"
    assert first["caption"] == CAPTION
    assert "preserves the author's double-cone primitive" in first["caption"]
    assert "not a fixed-domain or reconstruction result" in first["caption"]
    assert first["changed_view_ids_zero_based"] == [1, 2, 4]
    assert first["panels"] == {
        key: list(metrics) for key, metrics in PANEL_METRICS.items()
    }
    assert (
        first["data_contract"]["numeric_source"]
        == "all plotted values are derived from the validated report JSON; "
        "no dataset values are embedded in the plotting source"
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
        luminance = image.convert("L")
        statistics = ImageStat.Stat(luminance)
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
    assert "A1 is an author-compatibility clipping ablation" in svg_text
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
                "status", "AUTHOR_COMPATIBLE_CLIPPED_HYBRID_ALL_VIEW_INVALID"
            ),
            "unreviewed status",
        ),
        (
            lambda report: report["views"][1]["path_length"].__setitem__(
                "removed_fraction", 1.1
            ),
            "removed_fraction must be <= 1.0",
        ),
        (
            lambda report: report["views"][1]["mask_conditioned"][
                "imask_all"
            ].__setitem__("changed_fraction", 0.5),
            "changed_fraction is inconsistent",
        ),
    ],
)
def test_rejects_malformed_schema_status_and_fractions_before_rendering(
    tmp_path: Path,
    mutator,
    message: str,
) -> None:
    report_path, report = _synthetic_report(tmp_path)
    mutator(report)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    output_dir = tmp_path / "outputs"

    with pytest.raises(ValueError, match=message):
        build_clipped_hybrid_audit_figure(report_path, output_dir)

    assert not output_dir.exists()


def test_rejects_count_and_aggregate_inconsistency(tmp_path: Path) -> None:
    report_path, report = _synthetic_report(tmp_path)
    report["views"][2]["counts"]["clipped_zero_length_count"] += 1
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="clipped_zero_length_count conflicts"):
        build_clipped_hybrid_audit_figure(report_path, tmp_path / "outputs")
