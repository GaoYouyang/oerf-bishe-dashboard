from __future__ import annotations

import json
from pathlib import Path

import pytest

from site_tools.plot_psu_b1_parameter_sensitivity import (
    DEFAULT_STEM,
    LABELS,
    MANIFEST_SCHEMA,
    MANIFEST_STATUS,
    plot_public_summary,
)


def _scope(index: int) -> dict:
    return {
        "candidate_hit_fraction": max(0.0, 1.0 - index * 0.05),
        "ray_support_length_iou": max(0.0, 1.0 - index * 0.07),
        "candidate_path_fraction_of_b0": max(0.0, 0.32 - index * 0.02),
    }


def _summary() -> dict:
    ids = list(LABELS)
    families = [
        "reference",
        "axis_semantics",
        *(["angle"] * 4),
        *(["vertex"] * 6),
    ]
    aggregate = [
        {
            "id": variant_id,
            "family": families[index],
            "scopes": {
                "active": _scope(index),
                "all": _scope(index),
            },
        }
        for index, variant_id in enumerate(ids)
    ]
    return {
        "schema_version": "psu-b1-parameter-sensitivity-public-summary-1.0",
        "status": "B1_PARAMETER_DEPENDENCE_QUANTIFIED_PHYSICAL_SELECTION_REQUIRED",
        "aggregate_variants": aggregate,
        "per_view": [
            {
                "view_id_zero_based": view_id,
                "variants": [
                    {
                        "id": variant_id,
                        "active": _scope(index),
                        "all": _scope(index),
                    }
                    for index, variant_id in enumerate(ids)
                ],
            }
            for view_id in range(9)
        ],
        "headline_metrics": {"fixture": True},
    }


def test_generates_png_pdf_svg_and_manifest(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(_summary()), encoding="utf-8")
    output_dir = tmp_path / "figure"

    manifest = plot_public_summary(summary_path, output_dir)

    assert manifest["schema_version"] == MANIFEST_SCHEMA
    assert manifest["status"] == MANIFEST_STATUS
    assert manifest["claim_boundary"]["algorithm_superiority"] == "LOCKED"
    for extension in ("png", "pdf", "svg"):
        output = output_dir / f"{DEFAULT_STEM}.{extension}"
        assert output.is_file()
        assert output.stat().st_size > 100
    assert (output_dir / f"{DEFAULT_STEM}_manifest.json").is_file()


def test_rejects_missing_variant_or_invalid_fraction(tmp_path: Path) -> None:
    value = _summary()
    value["aggregate_variants"].pop()
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="12 frozen"):
        plot_public_summary(summary_path, tmp_path / "figure")

    value = _summary()
    value["aggregate_variants"][0]["scopes"]["active"][
        "candidate_hit_fraction"
    ] = 1.1
    summary_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        plot_public_summary(summary_path, tmp_path / "figure")
