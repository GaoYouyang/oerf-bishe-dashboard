from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from site_tools.validate_jacru_n1_0_evidence import (
    CONFIG,
    OUTPUT,
    PACKET_STATUS,
    VALIDATED_STATUS,
    ValidationError,
    _build_trajectory_groups,
    observable_selection_signature,
    validate_packet,
)


ROOT = Path(__file__).resolve().parents[1]


def _copy_packet(tmp_path: Path) -> Path:
    target = tmp_path / "n1_0"
    shutil.copytree(OUTPUT, target)
    return target


def _refresh_manifest(output: Path, name: str) -> None:
    digest = hashlib.sha256((output / name).read_bytes()).hexdigest()
    manifest = output / "checksums.sha256"
    manifest.write_text(
        "\n".join(
            f"{digest}  {name}" if line.endswith(f"  {name}") else line
            for line in manifest.read_text(encoding="ascii").splitlines()
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


def _source_rows() -> tuple[dict, list[dict[str, str]], list[dict[str, str]], float]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    t0 = json.loads((ROOT / config["source_t0_config"]).read_text(encoding="utf-8"))
    fixture = t0["fixture"]
    noise_floor = (
        float(fixture["noise_relative_std"]) ** 2
        + float(fixture["camera_bias_relative_std"]) ** 2
    ) ** 0.5
    _, trajectory = _read_csv(ROOT / config["source_m2_7_results"] / "metric_rows.csv")
    _, reference = _read_csv(ROOT / config["source_m2_7_results"] / "reference_rows.csv")
    return config, trajectory, reference, noise_floor


def test_current_n1_0_packet_validates() -> None:
    report = validate_packet()
    assert report["status"] == VALIDATED_STATUS
    assert report["selected_row_count"] == 6660
    assert report["aggregate_row_count"] == 444
    assert report["pareto_joint_safe_count"] == {"jacru_m2": 0, "pooled_cnn": 0}
    assert report["authorization"]["claim_method_superiority"] is False
    assert report["authorization"]["open_fresh_or_final"] is False


def test_checksum_tampering_is_rejected(tmp_path: Path) -> None:
    output = _copy_packet(tmp_path)
    path = output / "selected_rows.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValidationError, match=r"checksum mismatch: selected_rows\.csv"):
        validate_packet(output_dir=output)


def test_selected_iteration_tampering_is_rejected_after_manifest_refresh(
    tmp_path: Path,
) -> None:
    output = _copy_packet(tmp_path)
    path = output / "selected_rows.csv"
    fields, rows = _read_csv(path)
    target = next(row for row in rows if row["comparator_only"] == "False")
    target["selected_iteration"] = str(int(target["selected_iteration"]) + 1)
    _write_csv(path, fields, rows)
    _refresh_manifest(output, path.name)
    with pytest.raises(ValidationError, match="first-crossing selected iteration drift"):
        validate_packet(output_dir=output)


def test_field_and_clean_columns_do_not_change_observable_selection() -> None:
    config, trajectory, reference, noise_floor = _source_rows()
    original = observable_selection_signature(
        config=config,
        trajectory_rows=trajectory,
        reference_rows=reference,
        noise_floor=noise_floor,
    )
    changed_trajectory = copy.deepcopy(trajectory)
    changed_reference = copy.deepcopy(reference)
    for row in changed_trajectory:
        row["field_relative_l2"] = "truth-must-not-be-read"
        row["h1_seminorm_relative_error"] = "truth-must-not-be-read"
        row["clean_reprojection_relative_l2"] = "clean-must-not-be-read"
        row["field_gain_to_best_matched_classical"] = "truth-must-not-be-read"
        row["h1_gain_to_best_matched_classical"] = "truth-must-not-be-read"
        row["field_harm_to_best_matched_classical"] = "truth-must-not-be-read"
    for row in changed_reference:
        row["field_relative_l2"] = "truth-must-not-be-read"
        row["h1_seminorm_relative_error"] = "truth-must-not-be-read"
        row["clean_reprojection_relative_l2"] = "clean-must-not-be-read"
    changed = observable_selection_signature(
        config=config,
        trajectory_rows=changed_trajectory,
        reference_rows=changed_reference,
        noise_floor=noise_floor,
    )
    assert changed == original


def test_m27_nonzero_damping_is_rejected() -> None:
    config, trajectory, _, _ = _source_rows()
    changed = copy.deepcopy(trajectory)
    changed[0]["damping_absolute"] = "0.01"
    with pytest.raises(ValidationError, match="damping_absolute"):
        _build_trajectory_groups(changed, config)


def test_aggregate_tampering_is_rejected_after_manifest_refresh(tmp_path: Path) -> None:
    output = _copy_packet(tmp_path)
    path = output / "aggregate_rows.csv"
    fields, rows = _read_csv(path)
    rows[0]["field_gain_mean"] = str(float(rows[0]["field_gain_mean"]) + 0.5)
    _write_csv(path, fields, rows)
    _refresh_manifest(output, path.name)
    with pytest.raises(ValidationError, match="field_gain_mean: numeric mismatch"):
        validate_packet(output_dir=output)


def test_authorization_tampering_is_rejected_after_manifest_refresh(
    tmp_path: Path,
) -> None:
    output = _copy_packet(tmp_path)
    path = output / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["authorization"]["claim_deployable_algorithm"] = True
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _refresh_manifest(output, path.name)
    with pytest.raises(ValidationError, match="N1.0 authorization"):
        validate_packet(output_dir=output)


def test_status_tampering_is_rejected_after_manifest_refresh(tmp_path: Path) -> None:
    output = _copy_packet(tmp_path)
    path = output / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["status"] == PACKET_STATUS
    summary["status"] = "N1_0_FALSE_SUCCESS"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _refresh_manifest(output, path.name)
    with pytest.raises(ValidationError, match="status must remain NO-GO"):
        validate_packet(output_dir=output)


def test_exact_camera_block_oracle_cannot_enter_budget(tmp_path: Path) -> None:
    output = _copy_packet(tmp_path)
    path = output / "selected_rows.csv"
    fields, rows = _read_csv(path)
    rows[0]["exact_camera_block_setup_in_budget"] = "True"
    _write_csv(path, fields, rows)
    _refresh_manifest(output, path.name)
    with pytest.raises(ValidationError, match="exact camera-block oracle entered budget"):
        validate_packet(output_dir=output)


def test_validator_does_not_import_an_experiment_runner() -> None:
    source = (ROOT / "site_tools/validate_jacru_n1_0_evidence.py").read_text(
        encoding="utf-8"
    )
    assert "import site_tools.run_" not in source
    assert "from site_tools import run_" not in source
