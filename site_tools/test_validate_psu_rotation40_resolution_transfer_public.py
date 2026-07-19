from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from site_tools.validate_psu_rotation40_resolution_transfer_public import (
    DEFAULT_RESULT,
    validate_public_result,
)


def _copy(tmp_path: Path) -> Path:
    destination = tmp_path / "result"
    shutil.copytree(DEFAULT_RESULT, destination)
    return destination


def _rewrite_summary(result: Path, mutate) -> None:
    path = result / "summary.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = []
    import hashlib

    for item in sorted(result.iterdir()):
        if item.is_file() and item.name != "checksums.sha256":
            lines.append(f"{hashlib.sha256(item.read_bytes()).hexdigest()}  {item.name}")
    (result / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="ascii")


def test_frozen_public_result_validates() -> None:
    report = validate_public_result(DEFAULT_RESULT)
    assert report["status"] == "PUBLIC_AGGREGATE_PACKAGE_VALID"
    assert not report["private_payload_detected"]


def test_arbitrary_large_numeric_list_is_rejected(tmp_path: Path) -> None:
    result = _copy(tmp_path)
    _rewrite_summary(result, lambda value: value.update({"data": [0.0] * 100}))
    with pytest.raises(ValueError, match="top-level schema|payload key|list exceeds"):
        validate_public_result(result)


def test_short_arbitrary_data_key_is_rejected(tmp_path: Path) -> None:
    result = _copy(tmp_path)
    _rewrite_summary(result, lambda value: value.update({"data": [0.0]}))
    with pytest.raises(ValueError, match="top-level schema|payload key"):
        validate_public_result(result)


def test_private_path_is_rejected(tmp_path: Path) -> None:
    result = _copy(tmp_path)
    _rewrite_summary(result, lambda value: value.update({"note": "/Users/test/private.npy"}))
    with pytest.raises(ValueError, match="top-level schema|private path"):
        validate_public_result(result)


def test_machine_decision_tamper_is_rejected(tmp_path: Path) -> None:
    result = _copy(tmp_path)
    _rewrite_summary(result, lambda value: value.update({"status": "PASS"}))
    with pytest.raises(ValueError, match="decision changed"):
        validate_public_result(result)


def test_csv_metric_tamper_is_rejected_even_with_new_checksum(tmp_path: Path) -> None:
    result = _copy(tmp_path)
    csv_path = result / "comparison_rows.csv"
    raw = csv_path.read_text(encoding="utf-8")
    csv_path.write_text(raw.replace("0.8432631430215097", "0.1", 1), encoding="utf-8")
    import hashlib

    lines = []
    for item in sorted(result.iterdir()):
        if item.is_file() and item.name != "checksums.sha256":
            lines.append(f"{hashlib.sha256(item.read_bytes()).hexdigest()}  {item.name}")
    (result / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="CSV"):
        validate_public_result(result)


def test_checksum_tamper_is_rejected(tmp_path: Path) -> None:
    result = _copy(tmp_path)
    (result / "README.md").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_public_result(result)
