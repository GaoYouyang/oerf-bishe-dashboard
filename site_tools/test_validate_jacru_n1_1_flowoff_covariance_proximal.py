from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from site_tools.validate_jacru_n1_1_flowoff_covariance_proximal import (
    OUTPUT,
    VALIDATED_STATUS,
    ValidationError,
    validate_packet,
)


ROOT = Path(__file__).resolve().parents[1]


def _copy_packet(tmp_path: Path) -> Path:
    target = tmp_path / "n1_1"
    shutil.copytree(OUTPUT, target)
    return target


def _refresh_manifest(output: Path, name: str) -> None:
    digest = hashlib.sha256((output / name).read_bytes()).hexdigest()
    manifest = output / "checksums.sha256"
    lines = manifest.read_text(encoding="ascii").splitlines()
    manifest.write_text(
        "\n".join(
            f"{digest}  {name}" if line.endswith(f"  {name}") else line
            for line in lines
        )
        + "\n",
        encoding="ascii",
    )


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), list(reader)


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_current_n1_1_packet_validates() -> None:
    report = validate_packet()
    assert report["status"] == VALIDATED_STATUS
    assert report["packet_status"] == "N1_1_FLOWOFF_COVARIANCE_PROXIMAL_NO_GO"
    assert report["candidate_count"] == 7
    assert report["calibration_row_count"] == 60
    assert report["metric_row_count"] == 1260
    assert report["aggregate_row_count"] == 84
    assert report["reference_row_count"] == 180
    assert report["dense_setup_row_count"] == 12
    assert report["decision_count"] == 14
    assert report["deployable_input_pass_count"] == 0
    assert report["oracle_pass_count"] == 0
    assert report["authorization"]["open_fresh_or_final"] is False


def test_checksum_tamper_is_rejected(tmp_path: Path) -> None:
    output = _copy_packet(tmp_path)
    path = output / "metric_rows.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValidationError, match=r"checksum mismatch: metric_rows\.csv"):
        validate_packet(output_dir=output)


def test_summary_tamper_is_rejected_after_manifest_refresh(tmp_path: Path) -> None:
    output = _copy_packet(tmp_path)
    path = output / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["authorization"]["claim_method_superiority"] = True
    summary["status"] = "N1_1_FALSE_SUCCESS"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _refresh_manifest(output, path.name)
    with pytest.raises(ValidationError, match="summary status drift|authorization"):
        validate_packet(output_dir=output)


def test_metric_tamper_is_rejected_after_manifest_refresh(tmp_path: Path) -> None:
    output = _copy_packet(tmp_path)
    path = output / "metric_rows.csv"
    fields, rows = _read_csv(path)
    rows[0]["field_gain_to_best_matched"] = str(
        float(rows[0]["field_gain_to_best_matched"]) + 0.5
    )
    _write_csv(path, fields, rows)
    _refresh_manifest(output, path.name)
    with pytest.raises(ValidationError, match="field_harm drift|numeric mismatch|decisions"):
        validate_packet(output_dir=output)


def test_aggregate_tamper_is_rejected_after_manifest_refresh(tmp_path: Path) -> None:
    output = _copy_packet(tmp_path)
    path = output / "aggregate_rows.csv"
    fields, rows = _read_csv(path)
    rows[0]["field_gain_mean"] = str(float(rows[0]["field_gain_mean"]) + 0.25)
    _write_csv(path, fields, rows)
    _refresh_manifest(output, path.name)
    with pytest.raises(ValidationError, match="field_gain_mean: numeric mismatch"):
        validate_packet(output_dir=output)


def test_truth_flag_tamper_is_rejected_after_manifest_refresh(tmp_path: Path) -> None:
    output = _copy_packet(tmp_path)
    path = output / "metric_rows.csv"
    fields, rows = _read_csv(path)
    target = next(row for row in rows if row["uses_truth"] == "True")
    target["uses_truth"] = "False"
    _write_csv(path, fields, rows)
    _refresh_manifest(output, path.name)
    with pytest.raises(ValidationError, match="truth flag drift"):
        validate_packet(output_dir=output)


def test_dense_budget_tamper_is_rejected_after_manifest_refresh(tmp_path: Path) -> None:
    output = _copy_packet(tmp_path)
    path = output / "metric_rows.csv"
    fields, rows = _read_csv(path)
    rows[0]["dense_setup_in_budget"] = "True"
    _write_csv(path, fields, rows)
    _refresh_manifest(output, path.name)
    with pytest.raises(ValidationError, match="dense setup entered budget"):
        validate_packet(output_dir=output)


def test_obvious_leakage_column_is_rejected_after_manifest_refresh(tmp_path: Path) -> None:
    output = _copy_packet(tmp_path)
    path = output / "calibration_rows.csv"
    fields, rows = _read_csv(path)
    fields.append("persistent_bias_uv")
    for row in rows:
        row["persistent_bias_uv"] = "private-vector"
    _write_csv(path, fields, rows)
    _refresh_manifest(output, path.name)
    with pytest.raises(ValidationError, match="columns differ from frozen schema|leakage"):
        validate_packet(output_dir=output)


def test_validator_does_not_import_runner_model_or_operator() -> None:
    source = (
        ROOT / "site_tools/validate_jacru_n1_1_flowoff_covariance_proximal.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "import site_tools.run_",
        "from site_tools import run_",
        "demo_t16_operator.jacru",
        "torch",
        "numpy",
    )
    assert all(fragment not in source for fragment in forbidden)
