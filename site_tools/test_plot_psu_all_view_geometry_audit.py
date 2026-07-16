from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import pytest
from PIL import Image, ImageStat

from site_tools.plot_psu_all_view_geometry_audit import (
    DEFAULT_OUTPUT_STEM,
    FIGURE_SIZE_INCHES,
    PNG_DPI,
    PLOT_METRICS,
    build_all_view_geometry_figure,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_inputs(root: Path) -> tuple[Path, Path]:
    records = []
    for view_id in range(5):
        scale = view_id + 1
        records.append(
            {
                "view_id_zero_based": view_id,
                "measurement_count": 1000,
                "cone_length_weighted_outside_box_fraction": 0.025 * scale,
                "full_box_zero_fraction": 0.002 * scale,
                "box_miss_but_cone_nonzero_fraction": 0.003 * scale,
                "final_zero_length_fraction": 0.001 * scale,
                "active_rms_magnitude_pixels": 0.18 + 0.035 * view_id,
                "inactive_rms_magnitude_pixels": 0.12 + 0.018 * view_id,
                "active_unsafe_geometry_fraction": 0.0005 * view_id,
                "inactive_unsafe_geometry_fraction": 0.0015 * scale,
            }
        )

    report = {
        "schema_version": "psu-bost-all-view-geometry-audit-1.0",
        "status": "ALL_VIEW_GEOMETRY_AUDIT_NO_GO",
        "view_count": len(records),
        "views": records,
    }
    report_path = root / "all_view_geometry_audit.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    csv_path = root / "all_view_geometry_metrics.csv"
    fieldnames = ["view_id_zero_based", "measurement_count", *PLOT_METRICS]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return report_path, csv_path


def test_builds_deterministic_publication_outputs_and_manifest(tmp_path: Path) -> None:
    report_path, csv_path = _synthetic_inputs(tmp_path)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first = build_all_view_geometry_figure(report_path, csv_path, first_dir)
    second = build_all_view_geometry_figure(report_path, csv_path, second_dir)

    manifest_name = f"{DEFAULT_OUTPUT_STEM}_manifest.json"
    assert (first_dir / manifest_name).is_file()
    assert (second_dir / manifest_name).is_file()
    assert first == second
    assert first["status"] == "FIGURE_BUILD_COMPLETE"
    assert first["source_sha256"] == second["source_sha256"]
    assert (
        first["source_sha256"]
        == hashlib.sha256(
            report_path.read_bytes() + b"\0" + csv_path.read_bytes()
        ).hexdigest()
    )
    assert first["data_contract"]["view_ids_zero_based"] == [0, 1, 2, 3, 4]
    assert set(first["panels"]) == {"A", "B", "C", "D"}

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
    assert first["outputs"]["png"]["width_pixels"] == expected_size[0]
    assert first["outputs"]["png"]["height_pixels"] == expected_size[1]

    stored_manifest = json.loads((first_dir / manifest_name).read_text("utf-8"))
    assert stored_manifest == first
    svg_text = (first_dir / first["outputs"]["svg"]["filename"]).read_text("utf-8")
    assert not re.search(r"letter-spacing\s*:\s*-", svg_text)


def test_rejects_malformed_report_schema_before_rendering(tmp_path: Path) -> None:
    report_path, csv_path = _synthetic_inputs(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    del report["views"][2]["inactive_unsafe_geometry_fraction"]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    output_dir = tmp_path / "outputs"

    with pytest.raises(ValueError, match="missing required field.*inactive_unsafe"):
        build_all_view_geometry_figure(report_path, csv_path, output_dir)

    assert not output_dir.exists()


def test_rejects_malformed_csv_schema_before_rendering(tmp_path: Path) -> None:
    report_path, csv_path = _synthetic_inputs(tmp_path)
    rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))
    fieldnames = [field for field in rows[0] if field != "active_rms_magnitude_pixels"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="missing required column.*active_rms"):
        build_all_view_geometry_figure(report_path, csv_path, tmp_path / "outputs")
