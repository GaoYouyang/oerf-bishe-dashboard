import csv
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from site_tools.validate_jacru_m2_5_m2_6_evidence import ValidationError, validate_packet


ROOT = Path(__file__).resolve().parents[1]
PACKETS = (
    {
        "stage": "M2.5",
        "config": ROOT / "demo_t16_operator/configs/jacru_m2_5_exact_jacobi_preconditioner_oracle_postopen_v1.json",
        "output": ROOT / "demo_t16_operator/results/jacru_m2_5_exact_jacobi_preconditioner_oracle_postopen_public",
        "status": "VALIDATED_M2_5_ORACLE_NO_GO_CLOSURE_AND_APPLICATION_LEDGER_UNAVAILABLE",
        "rows": 2880,
    },
    {
        "stage": "M2.6",
        "config": ROOT / "demo_t16_operator/configs/jacru_m2_6_camera_block_preconditioner_oracle_postopen_v1.json",
        "output": ROOT / "demo_t16_operator/results/jacru_m2_6_camera_block_preconditioner_oracle_postopen_public",
        "status": "VALIDATED_M2_6_CAMERA_BLOCK_ORACLE_NO_GO_HARM_BLOCKS_AUTHORIZATION",
        "rows": 4320,
    },
)


def _copy(tmp_path: Path, packet: dict[str, object]) -> Path:
    target = tmp_path / str(packet["stage"]).replace(".", "_")
    if target.exists():
        target = tmp_path / (target.name + "_second_copy")
    shutil.copytree(packet["output"], target)
    return target


def _refresh(output: Path, name: str) -> None:
    digest = hashlib.sha256((output / name).read_bytes()).hexdigest()
    manifest = output / "checksums.sha256"
    manifest.write_text("\n".join(f"{digest}  {name}" if line.endswith(f"  {name}") else line for line in manifest.read_text(encoding="ascii").splitlines()) + "\n", encoding="ascii")


def _csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), list(reader)


def _write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _copy_config(tmp_path: Path, packet: dict[str, object]) -> Path:
    path = tmp_path / (str(packet["stage"]).replace(".", "_") + "_config.json")
    shutil.copyfile(packet["config"], path)
    return path


@pytest.mark.parametrize("packet", PACKETS, ids=("m2-5", "m2-6"))
def test_current_packets_validate_with_their_actual_no_go_boundaries(packet: dict[str, object]) -> None:
    report = validate_packet(config_path=Path(packet["config"]), output_dir=Path(packet["output"]))
    assert report["status"] == packet["status"]
    assert report["metric_row_count"] == packet["rows"]
    assert not report["authorization"]["claim_method_superiority"]
    assert not report["authorization"]["open_fresh_or_final"]
    if packet["stage"] == "M2.5":
        assert not report["closure_ledger_available"]
        assert not report["preconditioner_application_ledger_available"]
    else:
        assert report["closure_ledger_available"]
        assert report["preconditioner_application_ledger_available"]
        assert report["development_selection_count"] == 2


@pytest.mark.parametrize("packet", PACKETS, ids=("m2-5", "m2-6"))
def test_checksum_tampering_is_rejected(tmp_path: Path, packet: dict[str, object]) -> None:
    output = _copy(tmp_path, packet)
    (output / "metric_rows.csv").write_bytes((output / "metric_rows.csv").read_bytes() + b"\n")
    with pytest.raises(ValidationError, match=r"checksum mismatch: metric_rows\.csv"):
        validate_packet(config_path=Path(packet["config"]), output_dir=output)


@pytest.mark.parametrize("packet", PACKETS, ids=("m2-5", "m2-6"))
def test_budget_tampering_is_rejected_after_checksum_refresh(tmp_path: Path, packet: dict[str, object]) -> None:
    output = _copy(tmp_path, packet)
    path = output / "metric_rows.csv"
    fields, rows = _csv(path)
    rows[0]["projection_forward_calls"] = "2"
    _write(path, fields, rows)
    _refresh(output, "metric_rows.csv")
    with pytest.raises(ValidationError, match=r"metric projection_forward_calls budget formula drift"):
        validate_packet(config_path=Path(packet["config"]), output_dir=output)


def test_m2_6_closure_tampering_is_rejected(tmp_path: Path) -> None:
    packet = PACKETS[1]
    output = _copy(tmp_path, packet)
    path = output / "metric_rows.csv"
    fields, rows = _csv(path)
    rows[0]["projection_closure_relative_error"] = "1e-4"
    _write(path, fields, rows)
    _refresh(output, "metric_rows.csv")
    with pytest.raises(ValidationError, match=r"projection closure exceeds 1e-10"):
        validate_packet(config_path=Path(packet["config"]), output_dir=output)


def test_m2_6_preconditioner_application_tampering_is_rejected(tmp_path: Path) -> None:
    packet = PACKETS[1]
    output = _copy(tmp_path, packet)
    path = output / "metric_rows.csv"
    fields, rows = _csv(path)
    rows[0]["preconditioner_applications"] = "999"
    _write(path, fields, rows)
    _refresh(output, "metric_rows.csv")
    with pytest.raises(ValidationError, match=r"preconditioner application ledger drift"):
        validate_packet(config_path=Path(packet["config"]), output_dir=output)


def test_exact_oracle_setup_cost_tampering_is_rejected(tmp_path: Path) -> None:
    packet = PACKETS[1]
    output = _copy(tmp_path, packet)
    path = output / "metric_rows.csv"
    fields, rows = _csv(path)
    target = next(row for row in rows if row["preconditioner_kind"] == "dense_exact_camera_block_jacobi_oracle")
    target["preconditioner_setup_forward_equivalents"] = "0"
    _write(path, fields, rows)
    _refresh(output, "metric_rows.csv")
    with pytest.raises(ValidationError, match=r"oracle setup ledger drift"):
        validate_packet(config_path=Path(packet["config"]), output_dir=output)


def test_csv_schema_and_row_count_tampering_are_rejected(tmp_path: Path) -> None:
    packet = PACKETS[0]
    output = _copy(tmp_path, packet)
    path = output / "metric_rows.csv"
    fields, rows = _csv(path)
    fields[0] = "forged_case_id"
    for row in rows:
        row["forged_case_id"] = row.pop("case_id")
    _write(path, fields, rows)
    _refresh(output, "metric_rows.csv")
    with pytest.raises(ValidationError, match=r"columns differ from frozen schema"):
        validate_packet(config_path=Path(packet["config"]), output_dir=output)

    output = _copy(tmp_path, packet)
    path = output / "metric_rows.csv"
    fields, rows = _csv(path)
    _write(path, fields, rows[:-1])
    _refresh(output, "metric_rows.csv")
    with pytest.raises(ValidationError, match=r"expected 2880 metric_rows rows"):
        validate_packet(config_path=Path(packet["config"]), output_dir=output)


def test_frozen_source_hash_tampering_is_rejected(tmp_path: Path) -> None:
    packet = PACKETS[1]
    config_path = _copy_config(tmp_path, packet)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["source_m2_5_summary_sha256"] = "0" * 64
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValidationError, match=r"source_m2_5 summary hash drift"):
        validate_packet(config_path=config_path, output_dir=Path(packet["output"]))


def test_m2_6_harm_cannot_be_hidden_by_a_forged_pass(tmp_path: Path) -> None:
    packet = PACKETS[1]
    output = _copy(tmp_path, packet)
    path = output / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["decisions"]["jacru_m2"]["checks"]["development_harm_rate"] = True
    summary["decisions"]["jacru_m2"]["checks"]["development_worst_case"] = True
    summary["decisions"]["jacru_m2"]["passed_m2_3_mechanism_gate"] = True
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _refresh(output, "summary.json")
    with pytest.raises(ValidationError, match=r"summary\.decisions\.jacru_m2"):
        validate_packet(config_path=Path(packet["config"]), output_dir=output)


def test_m2_6_authorization_escalation_is_rejected(tmp_path: Path) -> None:
    packet = PACKETS[1]
    output = _copy(tmp_path, packet)
    path = output / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["authorization"]["claim_method_superiority"] = True
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _refresh(output, "summary.json")
    with pytest.raises(ValidationError, match=r"summary\.authorization"):
        validate_packet(config_path=Path(packet["config"]), output_dir=output)


def test_m2_5_cannot_gain_missing_ledger_columns_by_implication(tmp_path: Path) -> None:
    packet = PACKETS[0]
    output = _copy(tmp_path, packet)
    report = validate_packet(config_path=Path(packet["config"]), output_dir=output)
    assert report["status"].endswith("LEDGER_UNAVAILABLE")
    assert report["closure_ledger_available"] is False


def test_auditor_does_not_import_the_experiment_runner() -> None:
    source = (ROOT / "site_tools/validate_jacru_m2_5_m2_6_evidence.py").read_text(encoding="utf-8")
    assert "run_jacru_m2_3_matrix_free_projection" not in source
