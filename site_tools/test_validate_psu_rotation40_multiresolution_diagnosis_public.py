from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from site_tools.validate_psu_rotation40_multiresolution_diagnosis_public import (
    DEFAULT_RESULT,
    EXPECTED_FILES,
    validate_public_result,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _refresh_checksums(result_dir: Path) -> None:
    lines = [
        f"{_sha256(result_dir / filename)}  {filename}"
        for filename in sorted(EXPECTED_FILES - {"checksums.sha256"})
    ]
    (result_dir / "checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="ascii"
    )


def _copy(tmp_path: Path) -> Path:
    target = tmp_path / "result"
    shutil.copytree(DEFAULT_RESULT, target)
    return target


def _summary(result_dir: Path) -> dict:
    return json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))


def _write_summary(result_dir: Path, summary: dict) -> None:
    (result_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _refresh_checksums(result_dir)


def test_actual_public_package_is_valid() -> None:
    report = validate_public_result(DEFAULT_RESULT)
    assert report["status"] == "PUBLIC_AGGREGATE_PACKAGE_VALID"
    assert report["private_payload_detected"] is False
    assert report["file_count"] == 6


def test_checksum_tamper_fails(tmp_path: Path) -> None:
    result = _copy(tmp_path)
    with (result / "README.md").open("a", encoding="utf-8") as stream:
        stream.write("tamper\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_public_result(result)


def test_private_path_fails_even_with_fresh_checksums(tmp_path: Path) -> None:
    result = _copy(tmp_path)
    summary = _summary(result)
    summary["note"] = "/Users/example/private.npy"
    _write_summary(result, summary)
    with pytest.raises(ValueError, match="private path|top-level schema"):
        validate_public_result(result)


def test_large_numeric_payload_fails_even_with_fresh_checksums(tmp_path: Path) -> None:
    result = _copy(tmp_path)
    summary = _summary(result)
    summary["field_diagnostics"]["hidden"] = list(range(1000))
    _write_summary(result, summary)
    with pytest.raises(ValueError, match="list exceeds|scalar budget"):
        validate_public_result(result)


def test_line_curve_tamper_fails_even_with_fresh_checksums(tmp_path: Path) -> None:
    result = _copy(tmp_path)
    summary = _summary(result)
    target = next(
        row for row in summary["candidates"] if row["candidate_id"] == "line_alpha_0.50"
    )
    target["aggregate"]["vector_relative_l2"] += 0.01
    _write_summary(result, summary)
    with pytest.raises(ValueError, match="line_alpha_0.50 relative L2"):
        validate_public_result(result)


def test_machine_diagnosis_tamper_fails_even_with_fresh_checksums(tmp_path: Path) -> None:
    result = _copy(tmp_path)
    summary = _summary(result)
    summary["status"] = "SUCCESS"
    _write_summary(result, summary)
    with pytest.raises(ValueError, match="status changed"):
        validate_public_result(result)


def test_claim_firewall_tamper_fails_even_with_fresh_checksums(tmp_path: Path) -> None:
    result = _copy(tmp_path)
    summary = _summary(result)
    summary["claim_boundary"]["algorithm_superiority"] = True
    _write_summary(result, summary)
    with pytest.raises(ValueError, match="firewall opened"):
        validate_public_result(result)


def test_csv_tamper_fails_even_with_fresh_checksums(tmp_path: Path) -> None:
    result = _copy(tmp_path)
    csv_path = result / "comparison_rows.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
        fieldnames = list(rows[0])
    rows[0]["vector_relative_l2"] = "0.1"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _refresh_checksums(result)
    with pytest.raises(ValueError, match="CSV"):
        validate_public_result(result)
