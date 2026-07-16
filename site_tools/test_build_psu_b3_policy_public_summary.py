from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from site_tools.build_psu_b3_policy_public_summary import (
    EXPECTED_SAMPLE_COUNTS,
    PUBLIC_SCHEMA,
    build_public_summary,
)


def _support(sample_count: int, histogram: list[int], misses: int = 2) -> dict:
    hits = sum(histogram)
    retained = sum(index * count for index, count in enumerate(histogram))
    return {
        "ray_count": hits + misses,
        "centerline_hit_count": hits,
        "centerline_miss_count": misses,
        "fixed_denominator_sample_retained_fraction": retained / (hits * sample_count),
        "retained_sample_count_histogram": histogram,
    }


def _report(sample_count: int) -> dict:
    histogram = [0] * (sample_count + 1)
    histogram[0] = 1
    histogram[max(1, sample_count - 2)] = 2
    histogram[sample_count - 1] = 3
    histogram[sample_count] = 94
    b0_histogram = [0] * (sample_count + 1)
    b0_histogram[sample_count] = 100

    def view(view_id: int) -> dict:
        return {
            "schema_version": "psu-bost-aperture-domain-audit-1.0",
            "status": "B2_DETERMINISTIC_APERTURE_DOMAIN_AUDIT_PASS_B3_REQUIRED",
            "view_id_zero_based": view_id,
            "domains": {
                "B0": _support(sample_count, b0_histogram),
                "B1": _support(sample_count, histogram),
            },
            "mask_conditioned": {
                "amask_all": {
                    "B0": _support(sample_count, b0_histogram),
                    "B1": _support(sample_count, histogram),
                },
                "imask_all": {
                    "B0": _support(sample_count, b0_histogram),
                    "B1": _support(sample_count, histogram),
                },
            },
        }

    views = [view(0), view(1)]
    aggregate = copy.deepcopy(views[0])
    aggregate.pop("schema_version")
    aggregate.pop("status")
    aggregate.pop("view_id_zero_based")
    for domain in ("B0", "B1"):
        for key in (
            "ray_count",
            "centerline_hit_count",
            "centerline_miss_count",
        ):
            aggregate["domains"][domain][key] *= 2
        aggregate["domains"][domain]["retained_sample_count_histogram"] = [
            value * 2
            for value in aggregate["domains"][domain]["retained_sample_count_histogram"]
        ]
        for mask in ("amask_all", "imask_all"):
            for key in (
                "ray_count",
                "centerline_hit_count",
                "centerline_miss_count",
            ):
                aggregate["mask_conditioned"][mask][domain][key] *= 2
            aggregate["mask_conditioned"][mask][domain][
                "retained_sample_count_histogram"
            ] = [
                value * 2
                for value in aggregate["mask_conditioned"][mask][domain][
                    "retained_sample_count_histogram"
                ]
            ]
    return {
        "schema_version": "psu-bost-aperture-domain-all-view-audit-1.0",
        "status": "B2_ALL_VIEW_APERTURE_DOMAIN_AUDIT_PASS_B3_REQUIRED",
        "sample_count_per_centerline_hit": sample_count,
        "views": views,
        "aggregate": {
            "domains": aggregate["domains"],
            "mask_conditioned": aggregate["mask_conditioned"],
        },
    }


def _write_reports(tmp_path: Path) -> list[Path]:
    paths = []
    for sample_count in EXPECTED_SAMPLE_COUNTS:
        path = tmp_path / f"qmc{sample_count}.json"
        path.write_text(json.dumps(_report(sample_count)), encoding="utf-8")
        paths.append(path)
    return paths


def test_builds_strict_policy_summary_and_keeps_thresholds_predeclared(
    tmp_path: Path,
) -> None:
    summary = build_public_summary(_write_reports(tmp_path))

    assert summary["schema_version"] == PUBLIC_SCHEMA
    assert summary["view_count"] == 2
    assert summary["decision"]["default_b3_policy_selected"] is False
    assert summary["decision"]["training_ready"] == "NO"
    assert summary["headline_metrics"]["qmc_designs_are_nested"] is False

    qmc16 = summary["sensitivity"][1]["domains"]["B1"]["active"]
    assert qmc16["policies"]["indicator_keep"]["kept_count"] == 200
    assert qmc16["policies"]["drop_empty"]["kept_count"] == 198
    assert (
        qmc16["policies"]["support_floor_0.875"]["minimum_retained_sample_count"] == 14
    )
    assert (
        qmc16["policies"]["support_floor_0.9375"]["minimum_retained_sample_count"] == 15
    )
    assert qmc16["policies"]["drop_any_out"]["kept_count"] == 188
    assert qmc16["policies"]["indicator_keep"]["excluded_centerline_miss_count"] == 4


def test_output_contains_no_private_provenance_keys(tmp_path: Path) -> None:
    summary = build_public_summary(_write_reports(tmp_path))
    serialized = json.dumps(summary, sort_keys=True).lower()
    for forbidden in (
        "private_library",
        "source_path",
        "sha256",
        "manifest",
        "runtime_observation",
        "bundle",
    ):
        assert forbidden not in serialized


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda reports: reports[0].update({"status": "INVALID"}),
            "not a passing B2 report",
        ),
        (
            lambda reports: reports[1].update({"sample_count_per_centerline_hit": 8}),
            "sample count must be 16",
        ),
        (
            lambda reports: reports[2]["views"][1].update({"view_id_zero_based": 2}),
            "ordered and contiguous",
        ),
        (
            lambda reports: reports[0]["aggregate"]["domains"]["B1"].update(
                {"centerline_hit_count": 999}
            ),
            "hit/miss counts do not reconcile",
        ),
    ],
)
def test_rejects_incompatible_or_inconsistent_reports(
    tmp_path: Path,
    mutation,
    match: str,
) -> None:
    reports = [_report(sample_count) for sample_count in EXPECTED_SAMPLE_COUNTS]
    mutation(reports)
    paths = []
    for sample_count, report in zip(EXPECTED_SAMPLE_COUNTS, reports):
        path = tmp_path / f"qmc{sample_count}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        paths.append(path)
    with pytest.raises(ValueError, match=match):
        build_public_summary(paths)


def test_requires_exactly_three_reports(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly three"):
        build_public_summary(_write_reports(tmp_path)[:2])
