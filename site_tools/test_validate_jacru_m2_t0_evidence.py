import csv
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from site_tools.validate_jacru_m2_t0_evidence import ValidationError, validate_packet


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "demo_t16_operator/configs/jacru_m2_learned_residual_t0_v1.json"
SOURCE_OUTPUT = ROOT / "demo_t16_operator/results/jacru_m2_learned_residual_t0_public"


def _paths(output: Path = SOURCE_OUTPUT) -> dict[str, Path]:
    return {
        "config_path": CONFIG,
        "summary_path": output / "summary.json",
        "metric_rows_path": output / "metric_rows.csv",
        "aggregate_rows_path": output / "aggregate_rows.csv",
        "history_path": output / "training_history.csv",
        "checksums_path": output / "checksums.sha256",
    }


def _copy_bundle(tmp_path: Path) -> Path:
    output = tmp_path / "evidence"
    shutil.copytree(SOURCE_OUTPUT, output)
    return output


def _refresh_checksum(output: Path, filename: str) -> None:
    manifest = output / "checksums.sha256"
    digest = hashlib.sha256((output / filename).read_bytes()).hexdigest()
    lines = manifest.read_text(encoding="ascii").splitlines()
    rewritten = [
        f"{digest}  {filename}" if line.endswith(f"  {filename}") else line
        for line in lines
    ]
    manifest.write_text("\n".join(rewritten) + "\n", encoding="ascii")


def test_current_m2_t0_packet_independently_validates_as_no_go() -> None:
    report = validate_packet(**_paths())
    assert report["status"] == "VALIDATED_M2_T0_NO_GO"
    assert report["split_case_counts"] == {"train": 32, "development": 12, "ood": 18}
    assert report["metric_row_count"] == 420
    assert report["aggregate_row_count"] == 28
    assert report["training_run_count"] == 12
    assert report["paired_comparison_count"] == 360
    assert report["primary_method"] == "jacru_m2"
    assert report["primary_passed"] is False
    assert report["primary_failed_checks"] == [
        "development_reprojection",
        "ood_reprojection",
    ]
    assert not any(report["authorization"].values())


def test_stale_checksum_rejects_metric_byte_tampering(tmp_path: Path) -> None:
    output = _copy_bundle(tmp_path)
    metric_path = output / "metric_rows.csv"
    metric_path.write_bytes(metric_path.read_bytes() + b"\n")
    with pytest.raises(ValidationError, match=r"checksum mismatch: metric_rows\.csv"):
        validate_packet(**_paths(output))


def test_recomputed_budget_rejects_tampering_after_checksum_refresh(tmp_path: Path) -> None:
    output = _copy_bundle(tmp_path)
    metric_path = output / "metric_rows.csv"
    with metric_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    learned = next(row for row in rows if row["method"] == "jacru_m2")
    learned["optimization_forward_calls"] = "12"
    with metric_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _refresh_checksum(output, "metric_rows.csv")
    with pytest.raises(ValidationError, match="learned forward budget drift"):
        validate_packet(**_paths(output))


def test_recomputed_aggregate_rejects_coherently_checksummed_tampering(tmp_path: Path) -> None:
    output = _copy_bundle(tmp_path)
    aggregate_path = output / "aggregate_rows.csv"
    with aggregate_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    rows[0]["field_relative_l2_mean"] = str(float(rows[0]["field_relative_l2_mean"]) + 0.01)
    with aggregate_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _refresh_checksum(output, "aggregate_rows.csv")
    with pytest.raises(ValidationError, match=r"aggregate_rows\.csv.*numeric mismatch"):
        validate_packet(**_paths(output))


def test_summary_pass_claim_is_rejected_after_checksum_refresh(tmp_path: Path) -> None:
    output = _copy_bundle(tmp_path)
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["primary_passed"] = True
    summary["status"] = "M2_T0_DEVELOPMENT_OOD_PASS_FOR_LARGER_PREREGISTERED_GATE"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _refresh_checksum(output, "summary.json")
    with pytest.raises(ValidationError, match=r"summary\.primary_passed.*NO-GO"):
        validate_packet(**_paths(output))
