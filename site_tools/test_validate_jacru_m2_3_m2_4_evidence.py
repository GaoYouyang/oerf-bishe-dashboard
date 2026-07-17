import csv
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from site_tools.validate_jacru_m2_3_m2_4_evidence import (
    ValidationError,
    validate_packet,
)


ROOT = Path(__file__).resolve().parents[1]
PACKETS = (
    {
        "stage": "M2.3",
        "config": (
            ROOT
            / "demo_t16_operator/configs/"
            "jacru_m2_3_matrix_free_projection_postopen_v1.json"
        ),
        "output": (
            ROOT
            / "demo_t16_operator/results/"
            "jacru_m2_3_matrix_free_projection_postopen_public"
        ),
        "status": "VALIDATED_M2_3_MATRIX_FREE_PROJECTION_NO_GO",
        "report_status": "M2_3_POSTOPEN_MATRIX_FREE_PROJECTION_NO_GO",
        "metric_rows": 3240,
        "reference_rows": 390,
        "baseline_rows": 540,
        "aggregates": 216,
        "baseline_aggregates": 36,
    },
    {
        "stage": "M2.4",
        "config": (
            ROOT
            / "demo_t16_operator/configs/"
            "jacru_m2_4_affine_observation_projection_postopen_v1.json"
        ),
        "output": (
            ROOT
            / "demo_t16_operator/results/"
            "jacru_m2_4_affine_observation_projection_postopen_public"
        ),
        "status": "VALIDATED_M2_4_AFFINE_OBSERVATION_PROJECTION_NO_GO",
        "report_status": "M2_4_POSTOPEN_AFFINE_CG_NO_GO",
        "metric_rows": 4320,
        "reference_rows": 570,
        "baseline_rows": 720,
        "aggregates": 288,
        "baseline_aggregates": 48,
    },
)


def _copy_bundle(tmp_path: Path, packet: dict[str, object]) -> Path:
    output = tmp_path / str(packet["stage"]).replace(".", "_")
    shutil.copytree(packet["output"], output)
    return output


def _validate(packet: dict[str, object], output: Path | None = None) -> dict[str, object]:
    return validate_packet(
        config_path=Path(packet["config"]),
        output_dir=output or Path(packet["output"]),
    )


def _refresh_checksum(output: Path, filename: str) -> None:
    manifest = output / "checksums.sha256"
    digest = hashlib.sha256((output / filename).read_bytes()).hexdigest()
    lines = manifest.read_text(encoding="ascii").splitlines()
    rewritten = [
        f"{digest}  {filename}" if line.endswith(f"  {filename}") else line
        for line in lines
    ]
    manifest.write_text("\n".join(rewritten) + "\n", encoding="ascii")


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), list(reader)


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize("packet", PACKETS, ids=("m2-3", "m2-4"))
def test_current_packets_independently_validate_as_no_go(
    packet: dict[str, object],
) -> None:
    report = _validate(packet)
    assert report["status"] == packet["status"]
    assert report["report_status"] == packet["report_status"]
    assert report["metric_row_count"] == packet["metric_rows"]
    assert report["reference_row_count"] == packet["reference_rows"]
    assert report["matched_baseline_row_count"] == packet["baseline_rows"]
    assert report["aggregate_count"] == packet["aggregates"]
    assert report["matched_baseline_aggregate_count"] == packet["baseline_aggregates"]
    assert report["decision_count"] == 2
    assert report["eligible_development_candidate_count"] == 0
    assert report["development_selection_count"] == 0
    authorization = report["authorization"]
    assert authorization["continue_matrix_free_preconditioner_research"] is True
    assert not any(
        value
        for key, value in authorization.items()
        if key != "continue_matrix_free_preconditioner_research"
    )


@pytest.mark.parametrize("packet", PACKETS, ids=("m2-3", "m2-4"))
def test_stale_checksum_rejects_byte_tampering(
    tmp_path: Path,
    packet: dict[str, object],
) -> None:
    output = _copy_bundle(tmp_path, packet)
    metric_path = output / "metric_rows.csv"
    metric_path.write_bytes(metric_path.read_bytes() + b"\n")
    with pytest.raises(ValidationError, match=r"checksum mismatch: metric_rows\.csv"):
        _validate(packet, output)


@pytest.mark.parametrize("packet", PACKETS, ids=("m2-3", "m2-4"))
def test_checksummed_forward_budget_tampering_is_rejected(
    tmp_path: Path,
    packet: dict[str, object],
) -> None:
    output = _copy_bundle(tmp_path, packet)
    metric_path = output / "metric_rows.csv"
    fields, rows = _read_csv(metric_path)
    rows[0]["projection_forward_calls"] = "2"
    _write_csv(metric_path, fields, rows)
    _refresh_checksum(output, "metric_rows.csv")
    with pytest.raises(
        ValidationError,
        match="metric projection_forward_calls budget formula drift",
    ):
        _validate(packet, output)


@pytest.mark.parametrize("packet", PACKETS, ids=("m2-3", "m2-4"))
def test_deleted_metric_row_is_rejected_after_checksum_refresh(
    tmp_path: Path,
    packet: dict[str, object],
) -> None:
    output = _copy_bundle(tmp_path, packet)
    metric_path = output / "metric_rows.csv"
    fields, rows = _read_csv(metric_path)
    _write_csv(metric_path, fields, rows[:-1])
    _refresh_checksum(output, "metric_rows.csv")
    expected = int(packet["metric_rows"])
    with pytest.raises(ValidationError, match=rf"expected {expected} metric rows"):
        _validate(packet, output)


@pytest.mark.parametrize("packet", PACKETS, ids=("m2-3", "m2-4"))
def test_forged_success_status_is_rejected_after_checksum_refresh(
    tmp_path: Path,
    packet: dict[str, object],
) -> None:
    output = _copy_bundle(tmp_path, packet)
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["status"] = f"{str(packet['stage']).replace('.', '_')}_FORGED_PASS"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _refresh_checksum(output, "summary.json")
    stage = re_escape(str(packet["stage"]))
    with pytest.raises(
        ValidationError,
        match=rf"summary status must remain {stage} NO-GO",
    ):
        _validate(packet, output)


@pytest.mark.parametrize("packet", PACKETS, ids=("m2-3", "m2-4"))
def test_forged_development_selection_and_pass_are_rejected(
    tmp_path: Path,
    packet: dict[str, object],
) -> None:
    output = _copy_bundle(tmp_path, packet)
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    decision = summary["decisions"]["jacru_m2"]
    decision["selection"] = {
        "projection_variant": decision["screened_candidates"][0]["projection_variant"],
        "projection_iterations": 0,
    }
    decision["checks"]["development_selection_exists"] = True
    decision["passed_m2_3_mechanism_gate"] = True
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _refresh_checksum(output, "summary.json")
    with pytest.raises(
        ValidationError,
        match=r"summary\.decisions\.jacru_m2\.selection: expected null",
    ):
        _validate(packet, output)


@pytest.mark.parametrize("packet", PACKETS, ids=("m2-3", "m2-4"))
def test_forged_fresh_authorization_is_rejected(
    tmp_path: Path,
    packet: dict[str, object],
) -> None:
    output = _copy_bundle(tmp_path, packet)
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["authorization"]["open_fresh_or_final"] = True
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _refresh_checksum(output, "summary.json")
    with pytest.raises(
        ValidationError,
        match=r"summary\.authorization\.open_fresh_or_final: boolean mismatch",
    ):
        _validate(packet, output)


@pytest.mark.parametrize("packet", PACKETS, ids=("m2-3", "m2-4"))
def test_checksummed_aggregate_tampering_is_rejected(
    tmp_path: Path,
    packet: dict[str, object],
) -> None:
    output = _copy_bundle(tmp_path, packet)
    aggregate_path = output / "aggregate_rows.csv"
    fields, rows = _read_csv(aggregate_path)
    rows[0]["field_relative_l2_mean"] = str(
        float(rows[0]["field_relative_l2_mean"]) + 0.01
    )
    _write_csv(aggregate_path, fields, rows)
    _refresh_checksum(output, "aggregate_rows.csv")
    with pytest.raises(
        ValidationError,
        match=r"aggregate_rows\.csv.*numeric mismatch",
    ):
        _validate(packet, output)


@pytest.mark.parametrize("packet", PACKETS, ids=("m2-3", "m2-4"))
def test_checksummed_decision_tampering_is_rejected(
    tmp_path: Path,
    packet: dict[str, object],
) -> None:
    output = _copy_bundle(tmp_path, packet)
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    development = summary["decisions"]["jacru_m2"]["screened_candidates"][0][
        "development"
    ]
    development["field_gain_mean"] += 0.01
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _refresh_checksum(output, "summary.json")
    with pytest.raises(
        ValidationError,
        match=r"summary\.decisions.*field_gain_mean: numeric mismatch",
    ):
        _validate(packet, output)


def test_m2_4_target_mode_tampering_is_rejected(tmp_path: Path) -> None:
    packet = PACKETS[1]
    output = _copy_bundle(tmp_path, packet)
    metric_path = output / "metric_rows.csv"
    fields, rows = _read_csv(metric_path)
    rows[0]["projection_target_mode"] = "base_anchor_correction"
    _write_csv(metric_path, fields, rows)
    _refresh_checksum(output, "metric_rows.csv")
    with pytest.raises(
        ValidationError,
        match=r"metric_rows\[0\]\.projection_target_mode: target mode drift",
    ):
        _validate(packet, output)


def test_csv_schema_tampering_is_rejected(tmp_path: Path) -> None:
    packet = PACKETS[0]
    output = _copy_bundle(tmp_path, packet)
    metric_path = output / "metric_rows.csv"
    fields, rows = _read_csv(metric_path)
    fields[0] = "forged_case_id"
    for row in rows:
        row["forged_case_id"] = row.pop("case_id")
    _write_csv(metric_path, fields, rows)
    _refresh_checksum(output, "metric_rows.csv")
    with pytest.raises(
        ValidationError,
        match=r"metric_rows\.csv: columns differ from the frozen schema",
    ):
        _validate(packet, output)


def re_escape(value: str) -> str:
    return value.replace(".", r"\.")
