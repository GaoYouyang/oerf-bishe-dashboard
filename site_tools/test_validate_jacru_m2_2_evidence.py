import csv
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from site_tools.validate_jacru_m2_2_evidence import ValidationError, validate_packet


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "demo_t16_operator/configs/"
    "jacru_m2_2_exact_nullspace_oracle_postopen_v1.json"
)
SOURCE_OUTPUT = (
    ROOT
    / "demo_t16_operator/results/"
    "jacru_m2_2_exact_nullspace_oracle_postopen_public"
)


def _copy_bundle(tmp_path: Path) -> Path:
    output = tmp_path / "evidence"
    shutil.copytree(SOURCE_OUTPUT, output)
    return output


def _validate(output: Path = SOURCE_OUTPUT) -> dict[str, object]:
    return validate_packet(config_path=CONFIG, output_dir=output)


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


def test_current_packet_independently_validates_as_oracle_only() -> None:
    report = _validate()
    assert report["status"] == "VALIDATED_M2_2_EXACT_NULLSPACE_HEADROOM_ORACLE_ONLY"
    assert report["oracle_row_count"] == 180
    assert report["reference_row_count"] == 30
    assert report["zero_step_exact_row_count"] == 180
    assert report["geometry_ledger_count"] == 12
    assert report["aggregate_count"] == 12
    assert report["decision_count"] == 2
    assert report["passed_headroom_decision_count"] == 2
    assert report["authorization"] == {
        "claim_deployable_algorithm": False,
        "claim_method_superiority": False,
        "claim_real_bost_generalization": False,
        "open_fresh_or_final": False,
        "continue_matrix_free_projection_research": True,
    }


def test_stale_checksum_rejects_byte_tampering(tmp_path: Path) -> None:
    output = _copy_bundle(tmp_path)
    metric_path = output / "metric_rows.csv"
    metric_path.write_bytes(metric_path.read_bytes() + b"\n")
    with pytest.raises(ValidationError, match=r"checksum mismatch: metric_rows\.csv"):
        _validate(output)


def test_recomputed_metric_rejects_checksummed_row_tampering(tmp_path: Path) -> None:
    output = _copy_bundle(tmp_path)
    metric_path = output / "metric_rows.csv"
    fields, rows = _read_csv(metric_path)
    rows[0]["field_gain_to_reference"] = str(
        float(rows[0]["field_gain_to_reference"]) + 0.01
    )
    _write_csv(metric_path, fields, rows)
    _refresh_checksum(output, "metric_rows.csv")
    with pytest.raises(
        ValidationError,
        match=r"metric_rows\[0\]\.field_gain_to_reference: numeric mismatch",
    ):
        _validate(output)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("rank", 149, "geometry ledger rank drift"),
        ("nullity_lower_bound", 849, "geometry ledger nullity drift"),
    ),
)
def test_geometry_rank_nullity_tampering_is_rejected(
    tmp_path: Path,
    field: str,
    value: int,
    message: str,
) -> None:
    output = _copy_bundle(tmp_path)
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    first = next(iter(summary["dense_setup_ledger"].values()))
    first[field] = value
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _refresh_checksum(output, "summary.json")
    with pytest.raises(ValidationError, match=message):
        _validate(output)


def test_forged_deployable_success_is_rejected_after_checksum_refresh(
    tmp_path: Path,
) -> None:
    output = _copy_bundle(tmp_path)
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["status"] = "M2_2_DEPLOYABLE_ALGORITHM_SUCCESS"
    summary["authorization"]["claim_deployable_algorithm"] = True
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _refresh_checksum(output, "summary.json")
    with pytest.raises(
        ValidationError,
        match="summary status does not match recomputed oracle result",
    ):
        _validate(output)


def test_forged_deployment_authorization_is_independently_rejected(
    tmp_path: Path,
) -> None:
    output = _copy_bundle(tmp_path)
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["authorization"]["claim_deployable_algorithm"] = True
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _refresh_checksum(output, "summary.json")
    with pytest.raises(
        ValidationError,
        match=r"summary\.authorization\.claim_deployable_algorithm: boolean mismatch",
    ):
        _validate(output)


def test_deleted_reference_row_is_rejected_after_checksum_refresh(
    tmp_path: Path,
) -> None:
    output = _copy_bundle(tmp_path)
    reference_path = output / "reference_rows.csv"
    fields, rows = _read_csv(reference_path)
    _write_csv(reference_path, fields, rows[:-1])
    _refresh_checksum(output, "reference_rows.csv")
    with pytest.raises(ValidationError, match="expected 30 reference rows"):
        _validate(output)


def test_tampered_aggregate_is_rejected_after_checksum_refresh(tmp_path: Path) -> None:
    output = _copy_bundle(tmp_path)
    aggregate_path = output / "aggregate_rows.csv"
    fields, rows = _read_csv(aggregate_path)
    rows[0]["oracle_field_relative_l2_mean"] = str(
        float(rows[0]["oracle_field_relative_l2_mean"]) + 0.01
    )
    _write_csv(aggregate_path, fields, rows)
    _refresh_checksum(output, "aggregate_rows.csv")
    with pytest.raises(
        ValidationError,
        match=r"aggregate_rows\.csv.*numeric mismatch",
    ):
        _validate(output)
