from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil

import pytest

from site_tools.validate_n2_pvgr_n5_d4c_msra_semantic_v2 import (
    DEFAULT_CONFIG,
    DEFAULT_RESULT,
    validate,
)


def _copy_with_private_artifacts(
    tmp_path: Path, *private_names: str
) -> Path:
    destination = tmp_path / "result"
    destination.mkdir()
    private = set(private_names) | {"manifest.json"}
    for source in DEFAULT_RESULT.iterdir():
        if source.name == "validation_report.json":
            continue
        target = destination / source.name
        if source.name in private:
            shutil.copy2(source, target)
        else:
            os.symlink(source, target)
    return destination


def _mutate_first_csv_row(path: Path, field: str, value: str) -> None:
    temporary = path.with_suffix(".tampered.csv")
    with path.open(newline="", encoding="utf-8") as source, temporary.open(
        "w", newline="", encoding="utf-8"
    ) as target:
        reader = csv.DictReader(source)
        assert reader.fieldnames is not None
        writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
        writer.writeheader()
        first = next(reader)
        first[field] = value
        writer.writerow(first)
        writer.writerows(reader)
    temporary.replace(path)


def _refresh_artifact_hash(result_dir: Path, name: str) -> None:
    manifest_path = result_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_hashes"][name] = hashlib.sha256(
        (result_dir / name).read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def test_formal_semantic_v2_bundle_independently_validates(tmp_path: Path) -> None:
    report = validate(
        DEFAULT_CONFIG,
        DEFAULT_RESULT,
        tmp_path / "formal-validation-report.json",
    )

    assert report["valid"] is True
    assert report["errors"] == []
    assert report["details"]["counts"]["case_spec_rows"] == 720
    assert report["details"]["counts"]["fd_rows"] == 34560
    assert report["details"]["counts"]["decision_rows"] == 36000
    assert report["details"]["all_forward_calls_replayed"] is True
    assert report["details"]["all_worst_prefix_decisions_rebuilt"] is True


def test_tampered_plus_input_sha256_fails_after_manifest_refresh(
    tmp_path: Path,
) -> None:
    copied = _copy_with_private_artifacts(tmp_path, "fd_rows.csv")
    _mutate_first_csv_row(copied / "fd_rows.csv", "plus_input_sha256", "0" * 64)
    _refresh_artifact_hash(copied, "fd_rows.csv")

    report = validate(
        DEFAULT_CONFIG,
        copied,
        tmp_path / "plus-input-tamper-report.json",
    )

    assert report["valid"] is False
    assert any("plus_input_sha256" in error for error in report["errors"])


def test_tampered_forward_branch_state_fails_after_manifest_refresh(
    tmp_path: Path,
) -> None:
    copied = _copy_with_private_artifacts(tmp_path, "fd_rows.csv")
    _mutate_first_csv_row(
        copied / "fd_rows.csv", "plus_branch_state", "linear:tampered"
    )
    _refresh_artifact_hash(copied, "fd_rows.csv")

    report = validate(
        DEFAULT_CONFIG,
        copied,
        tmp_path / "branch-tamper-report.json",
    )

    assert report["valid"] is False
    assert any("branch" in error.lower() for error in report["errors"])


def test_tampered_reported_metric_fails_after_manifest_refresh(
    tmp_path: Path,
) -> None:
    copied = _copy_with_private_artifacts(tmp_path, "decision_rows.csv")
    _mutate_first_csv_row(
        copied / "decision_rows.csv", "maximum_fd_relative_error", "0.125"
    )
    _refresh_artifact_hash(copied, "decision_rows.csv")

    report = validate(
        DEFAULT_CONFIG,
        copied,
        tmp_path / "metric-tamper-report.json",
    )

    assert report["valid"] is False
    assert any("reported metric" in error for error in report["errors"])


def test_tampered_reported_decision_fails_after_manifest_refresh(
    tmp_path: Path,
) -> None:
    copied = _copy_with_private_artifacts(tmp_path, "decision_rows.csv")
    _mutate_first_csv_row(copied / "decision_rows.csv", "status", "FAIL_FD")
    _refresh_artifact_hash(copied, "decision_rows.csv")

    report = validate(
        DEFAULT_CONFIG,
        copied,
        tmp_path / "decision-tamper-report.json",
    )

    assert report["valid"] is False
    assert any("reported decision" in error for error in report["errors"])


def test_default_report_refuses_overwrite(tmp_path: Path) -> None:
    existing = tmp_path / "existing-report.json"
    existing.write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to replace"):
        validate(DEFAULT_CONFIG, DEFAULT_RESULT, existing)


def test_validator_has_no_forbidden_experiment_imports() -> None:
    source_path = Path(__file__).with_name(
        "validate_n2_pvgr_n5_d4c_msra_semantic_v2.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    assert not any(
        name.endswith("run_n2_pvgr_n5_d4c_msra_semantic_v2")
        or name.endswith("side_weighted_adjoint_certificate")
        for name in imported
    )
