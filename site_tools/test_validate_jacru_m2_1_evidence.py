import csv
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from site_tools.validate_jacru_m2_1_evidence import ValidationError, validate_packet


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "demo_t16_operator/configs/"
    "jacru_m2_1_matched_data_consistency_postopen_v1_1.json"
)
SOURCE_OUTPUT = (
    ROOT
    / "demo_t16_operator/results/"
    "jacru_m2_1_matched_data_consistency_postopen_public"
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


def test_current_packet_independently_validates_as_no_go() -> None:
    report = _validate()
    assert report["status"] == "VALIDATED_M2_1_MATCHED_BUDGET_NO_GO"
    assert report["learned_row_count"] == 1620
    assert report["matched_baseline_row_count"] == 450
    assert report["zero_step_exact_row_count"] == 180
    assert report["learned_aggregate_count"] == 108
    assert report["matched_baseline_aggregate_count"] == 30
    assert report["decision_count"] == 18
    assert report["failed_reprojection_check_count"] == 36
    assert not any(report["authorization"].values())


def test_stale_checksum_rejects_byte_tampering(tmp_path: Path) -> None:
    output = _copy_bundle(tmp_path)
    metric_path = output / "metric_rows.csv"
    metric_path.write_bytes(metric_path.read_bytes() + b"\n")
    with pytest.raises(ValidationError, match=r"checksum mismatch: metric_rows\.csv"):
        _validate(output)


def test_recomputed_budget_rejects_checksummed_tampering(tmp_path: Path) -> None:
    output = _copy_bundle(tmp_path)
    metric_path = output / "metric_rows.csv"
    fields, rows = _read_csv(metric_path)
    rows[0]["optimization_forward_calls"] = "12"
    _write_csv(metric_path, fields, rows)
    _refresh_checksum(output, "metric_rows.csv")
    with pytest.raises(ValidationError, match="learned forward budget drift"):
        _validate(output)


def test_summary_pass_claim_is_rejected_after_checksum_refresh(tmp_path: Path) -> None:
    output = _copy_bundle(tmp_path)
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["status"] = "M2_1_POSTOPEN_HEADROOM_FOUND_NOT_CONFIRMATORY"
    summary["authorization"]["draft_new_preregistered_data_consistency_gate"] = True
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _refresh_checksum(output, "summary.json")
    with pytest.raises(ValidationError, match="summary status must remain NO-GO"):
        _validate(output)


def test_deleted_matched_baseline_row_is_rejected_after_checksum_refresh(
    tmp_path: Path,
) -> None:
    output = _copy_bundle(tmp_path)
    baseline_path = output / "matched_baseline_rows.csv"
    fields, rows = _read_csv(baseline_path)
    _write_csv(baseline_path, fields, rows[:-1])
    _refresh_checksum(output, "matched_baseline_rows.csv")
    with pytest.raises(ValidationError, match="expected 450 matched baseline rows"):
        _validate(output)


def test_tampered_aggregate_is_rejected_after_checksum_refresh(tmp_path: Path) -> None:
    output = _copy_bundle(tmp_path)
    aggregate_path = output / "aggregate_rows.csv"
    fields, rows = _read_csv(aggregate_path)
    rows[0]["field_relative_l2_mean"] = str(
        float(rows[0]["field_relative_l2_mean"]) + 0.01
    )
    _write_csv(aggregate_path, fields, rows)
    _refresh_checksum(output, "aggregate_rows.csv")
    with pytest.raises(ValidationError, match=r"aggregate_rows\.csv.*numeric mismatch"):
        _validate(output)
