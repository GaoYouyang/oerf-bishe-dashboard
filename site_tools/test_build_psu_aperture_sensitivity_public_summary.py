from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from site_tools import build_psu_aperture_sensitivity_public_summary as exporter


REPO_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = (
    REPO_ROOT / "private_library" / "external_datasets" / "psu_bost_flight_body"
)
REAL_REPORT_PATHS = {
    sample_count: PRIVATE_ROOT / f"aperture_domain_qmc{sample_count}_audit.json"
    for sample_count in exporter.EXPECTED_SAMPLE_COUNTS
}


@pytest.fixture(scope="module")
def real_reports() -> tuple[dict, dict, dict]:
    missing = [path for path in REAL_REPORT_PATHS.values() if not path.exists()]
    if missing:
        pytest.skip(f"private aperture reports are missing: {missing}")
    return tuple(
        copy.deepcopy(exporter.load_private_report(REAL_REPORT_PATHS[count]))
        for count in exporter.EXPECTED_SAMPLE_COUNTS
    )


def test_real_reports_export_strict_public_sensitivity(
    real_reports: tuple[dict, dict, dict],
) -> None:
    private_hashes = {
        value
        for report in real_reports
        for view in report["views"]
        for key, value in view["source"].items()
        if key.endswith("_sha256")
    }
    reports = tuple(copy.deepcopy(report) for report in real_reports)
    reports[0]["local_report_path"] = (
        "/Users/alice/private_library/aperture_domain_qmc8_audit.json"
    )
    reports[1]["views"][0]["runtime_observation"]["private_log"] = (
        "file:///Users/alice/qmc16.log"
    )
    reports[2]["views"][0]["configuration"]["private_vector"] = [987654321.25]
    reports[2]["aggregate"]["private_ray_indices"] = [3, 5, 8]

    result = exporter.build_public_summary(*reports)
    payload = json.dumps(result, sort_keys=True)

    assert set(result) == {
        "schema_version",
        "source_schema_version",
        "source_view_schema_version",
        "execution_status",
        "scientific_verdict",
        "status",
        "view_count",
        "sample_counts_per_centerline_hit",
        "fixed_denominator_policy",
        "support_sensitivity",
        "claim_boundary",
        "public_export_policy",
    }
    assert result["sample_counts_per_centerline_hit"] == [8, 16, 32]
    assert result["view_count"] == 9
    assert [
        item["sample_count_per_centerline_hit"]
        for item in result["support_sensitivity"]
    ] == [8, 16, 32]
    assert all(len(item["views"]) == 9 for item in result["support_sensitivity"])

    qmc8 = result["support_sensitivity"][0]
    assert qmc8["aggregate"]["B0"]["all"] == {
        "retained_sample_fraction": pytest.approx(0.9991690001472863),
        "any_ood_ray_count": 151_246,
        "any_ood_ray_fraction_of_centerline_hits": pytest.approx(0.003054499590767012),
        "empty_support_ray_count": 0,
    }
    assert qmc8["aggregate"]["B1"]["active"]["any_ood_ray_count"] == 2_660
    assert qmc8["aggregate"]["B1"]["inactive"][
        "retained_sample_fraction"
    ] == pytest.approx(0.9665245623061449)
    assert qmc8["views"][0]["view_id_zero_based"] == 0
    assert qmc8["views"][0]["support"]["B1"]["all"]["empty_support_ray_count"] == 181

    assert result["claim_boundary"] == {
        "supported_claim": (
            "DISCRETE_DETERMINISTIC_APERTURE_SUPPORT_SAMPLE_COUNT_SENSITIVITY_ONLY"
        ),
        "continuous_aperture_containment": "UNCONFIRMED",
        "zero_extension": "UNCONFIRMED",
        "b1_physical_cone_semantics": "UNCONFIRMED",
        "any_ood_ray_count_is_sample_resolution_sensitive": True,
        "b3_geometry_safe_mask": "LOCKED",
        "held_out_reprojection": "LOCKED",
        "training": "LOCKED",
        "algorithm_superiority": "LOCKED",
    }
    for secret in (
        "/Users/",
        "file://",
        "private_library",
        "987654321.25",
        "runtime_observation",
        "configuration",
        "unit_disk_offsets",
        "longitudinal_fractions",
        "private_ray_indices",
        "sha256",
    ):
        assert secret not in payload
    for private_hash in private_hashes:
        assert private_hash not in payload


@pytest.mark.parametrize(
    ("report_index", "mutation", "message"),
    [
        (
            1,
            lambda report: report["views"][0]["source"].update(
                bundle_manifest_sha256="f" * 64
            ),
            "source is incompatible across sample counts",
        ),
        (
            2,
            lambda report: report["views"][0]["configuration"].update(
                cone_angle_degrees=24.0
            ),
            "configuration is incompatible across sample counts",
        ),
    ],
)
def test_rejects_inconsistent_source_or_configuration(
    real_reports: tuple[dict, dict, dict],
    report_index: int,
    mutation,
    message: str,
) -> None:
    reports = [copy.deepcopy(report) for report in real_reports]
    mutation(reports[report_index])
    with pytest.raises(ValueError, match=message):
        exporter.build_public_summary(*reports)


@pytest.mark.parametrize(
    ("report_index", "mutation", "message"),
    [
        (
            0,
            lambda report: report.update(schema_version="wrong-schema"),
            "schema_version has an unreviewed value",
        ),
        (
            1,
            lambda report: report["views"][0].update(status="INVALID"),
            "status has an unreviewed value",
        ),
        (
            2,
            lambda report: report.update(sample_count_per_centerline_hit=31),
            "must equal 32",
        ),
        (
            0,
            lambda report: report["views"][1].update(view_id_zero_based=7),
            "exact expected view id 1",
        ),
        (
            1,
            lambda report: report["decision"].update(
                fixed_denominator_indicator_implemented=False
            ),
            "conflicts with the claim boundary",
        ),
    ],
)
def test_rejects_schema_status_ids_counts_and_denominator_policy(
    real_reports: tuple[dict, dict, dict],
    report_index: int,
    mutation,
    message: str,
) -> None:
    reports = [copy.deepcopy(report) for report in real_reports]
    mutation(reports[report_index])
    with pytest.raises(ValueError, match=message):
        exporter.build_public_summary(*reports)


@pytest.mark.parametrize(
    ("report_index", "mutation", "message"),
    [
        (
            0,
            lambda report: report["views"][0]["diagnostics"]["B1"].update(
                nonfinite_radius_count=1
            ),
            "nonfinite_radius_count must be zero",
        ),
        (
            1,
            lambda report: report["aggregate"]["domains"]["B0"][
                "retained_sample_count_histogram"
            ].__setitem__(0, 1),
            "histogram sum does not match centerline hits",
        ),
        (
            2,
            lambda report: report["views"][0]["domains"]["B1"].update(
                fixed_denominator_sample_retained_fraction=float("nan")
            ),
            "non-finite",
        ),
        (
            0,
            lambda report: report["views"][0]["runtime_observation"].update(
                wall_seconds=float("inf")
            ),
            "non-finite",
        ),
    ],
)
def test_rejects_diagnostics_histogram_and_nonfinite_values(
    real_reports: tuple[dict, dict, dict],
    report_index: int,
    mutation,
    message: str,
) -> None:
    reports = [copy.deepcopy(report) for report in real_reports]
    mutation(reports[report_index])
    with pytest.raises(ValueError, match=message):
        exporter.build_public_summary(*reports)


def test_rejects_aggregate_and_per_view_mismatch(
    real_reports: tuple[dict, dict, dict],
) -> None:
    reports = [copy.deepcopy(report) for report in real_reports]
    reports[0]["aggregate"]["mask_conditioned"]["imask_all"]["B1"][
        "any_sample_out_of_domain_ray_count"
    ] += 1
    with pytest.raises(ValueError, match="support ray counts do not sum"):
        exporter.build_public_summary(*reports)


def test_public_payload_validator_rejects_path_hash_and_raw_design_leaks(
    real_reports: tuple[dict, dict, dict],
) -> None:
    result = exporter.build_public_summary(*real_reports)
    mutations = (
        ("debug_path", "/Users/alice/private.json"),
        ("private_sha256", "a" * 64),
        ("unit_disk_offsets", [[0.0, 0.0]]),
    )
    for key, value in mutations:
        leaked = copy.deepcopy(result)
        leaked[key] = value
        with pytest.raises(ValueError, match="private|forbidden"):
            exporter._validate_public_payload(leaked)


def test_loader_rejects_nonstandard_json_constants(tmp_path: Path) -> None:
    input_path = tmp_path / "private.json"
    input_path.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-standard JSON constant"):
        exporter.load_private_report(input_path)


def test_missing_qmc32_does_not_replace_existing_output(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "public.json"
    output_path.write_text('{"old": true}\n', encoding="utf-8")
    original = output_path.read_bytes()

    with pytest.raises(FileNotFoundError):
        exporter.export_public_summary(
            REAL_REPORT_PATHS[8],
            REAL_REPORT_PATHS[16],
            tmp_path / "missing-qmc32.json",
            output_path,
        )

    assert output_path.read_bytes() == original
    assert list(tmp_path.glob(f".{output_path.name}.*.tmp")) == []


def test_export_is_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "nested" / "public.json"
    replace_calls = []
    real_replace = exporter.os.replace

    def record_replace(source: Path, destination: Path) -> None:
        replace_calls.append((source, destination))
        real_replace(source, destination)

    monkeypatch.setattr(exporter.os, "replace", record_replace)
    result = exporter.export_public_summary(
        REAL_REPORT_PATHS[8],
        REAL_REPORT_PATHS[16],
        REAL_REPORT_PATHS[32],
        output_path,
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == result
    assert len(replace_calls) == 1
    assert replace_calls[0][0].parent == output_path.parent
    assert replace_calls[0][1] == output_path
    assert list(output_path.parent.glob(f".{output_path.name}.*.tmp")) == []


def test_atomic_write_cleans_up_after_replace_failure(
    real_reports: tuple[dict, dict, dict],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "public.json"
    output_path.write_text('{"old": true}\n', encoding="utf-8")
    original = output_path.read_bytes()
    result = exporter.build_public_summary(*real_reports)

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(exporter.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        exporter.write_json_atomic(result, output_path)

    assert output_path.read_bytes() == original
    assert list(tmp_path.glob(f".{output_path.name}.*.tmp")) == []
