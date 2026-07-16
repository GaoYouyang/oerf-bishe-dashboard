from __future__ import annotations

import json
from pathlib import Path

import pytest

from site_tools.plot_psu_b3_policy_audit import (
    DEFAULT_STEM,
    MANIFEST_SCHEMA,
    plot_public_summary,
)


def _policy(kept: int, hits: int) -> dict:
    return {
        "kept_count": kept,
        "kept_fraction_of_centerline_hits": kept / hits,
    }


def _category(hits: int, retained: float, shift: int = 0) -> dict:
    return {
        "centerline_hit_count": hits,
        "b2_fixed_denominator_retained_sample_fraction": retained,
        "policies": {
            "indicator_keep": _policy(hits, hits),
            "support_floor_0.875": _policy(hits - 1 - shift, hits),
            "support_floor_0.9375": _policy(hits - 2 - shift, hits),
            "drop_any_out": _policy(hits - 3 - shift, hits),
        },
    }


def _summary() -> dict:
    sensitivity = []
    for index, sample_count in enumerate((8, 16, 32)):
        views = []
        for view_id in range(3):
            views.append(
                {
                    "view_id_zero_based": view_id,
                    **_category(100, 0.999, shift=view_id + index),
                }
            )
        sensitivity.append(
            {
                "sample_count_per_centerline_hit": sample_count,
                "domains": {
                    "B0": {
                        "active": _category(1000, 1.0),
                    },
                    "B1": {
                        "active": _category(1000, 0.999 - index * 0.0001, index),
                        "inactive": _category(1000, 0.98 - index * 0.01, index),
                    },
                },
                "active_b1_per_view": views,
            }
        )
    return {
        "schema_version": "psu-bost-b3-policy-public-summary-1.0",
        "status": "B3_POLICY_SENSITIVITY_COMPLETE_HELD_OUT_SELECTION_REQUIRED",
        "sensitivity": sensitivity,
        "headline_metrics": {"synthetic_fixture": True},
    }


def test_generates_publication_formats_and_manifest(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(_summary()), encoding="utf-8")
    output_dir = tmp_path / "figure"

    manifest = plot_public_summary(summary_path, output_dir)

    assert manifest["schema_version"] == MANIFEST_SCHEMA
    assert manifest["claim_boundary"]["algorithm_superiority"] == "LOCKED"
    for extension in ("png", "pdf", "svg"):
        output = output_dir / f"{DEFAULT_STEM}.{extension}"
        assert output.is_file()
        assert output.stat().st_size > 100
        assert manifest["outputs"][extension]["bytes"] == output.stat().st_size
    manifest_path = output_dir / f"{DEFAULT_STEM}_manifest.json"
    assert json.loads(manifest_path.read_text())["caption"] == manifest["caption"]


def test_rejects_wrong_sample_counts(tmp_path: Path) -> None:
    value = _summary()
    value["sensitivity"][1]["sample_count_per_centerline_hit"] = 12
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly"):
        plot_public_summary(summary_path, tmp_path / "figure")


def test_rejects_kept_count_above_hits(tmp_path: Path) -> None:
    value = _summary()
    value["sensitivity"][0]["domains"]["B1"]["active"]["policies"]["drop_any_out"][
        "kept_count"
    ] = 1001
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="exceeds"):
        plot_public_summary(summary_path, tmp_path / "figure")
