import csv
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from site_tools.validate_jacru_m2_7_m2_8_evidence import ValidationError, validate_packet


ROOT = Path(__file__).resolve().parents[1]
PACKETS = (
    {
        "stage": "m2-7",
        "config": ROOT / "demo_t16_operator/configs/jacru_m2_7_target_no_harm_pareto_ceiling_postopen_v1.json",
        "output": ROOT / "demo_t16_operator/results/jacru_m2_7_target_no_harm_pareto_ceiling_postopen_public",
        "status": "VALIDATED_M2_7_TARGET_NO_HARM_PARETO_ORACLE_NO_GO",
    },
    {
        "stage": "m2-8",
        "config": ROOT / "demo_t16_operator/configs/jacru_m2_8_interpolation_calibration_ceiling_postopen_v1.json",
        "output": ROOT / "demo_t16_operator/results/jacru_m2_8_interpolation_calibration_ceiling_postopen_public",
        "status": "VALIDATED_M2_8_INTERPOLATION_CALIBRATION_ENVELOPE_NO_GO",
    },
)


def _copy(tmp_path: Path, packet: dict[str, object]) -> Path:
    target = tmp_path / str(packet["stage"])
    shutil.copytree(packet["output"], target)
    return target


def _refresh(output: Path, name: str) -> None:
    digest = hashlib.sha256((output / name).read_bytes()).hexdigest()
    manifest = output / "checksums.sha256"
    manifest.write_text(
        "\n".join(
            f"{digest}  {name}" if line.endswith(f"  {name}") else line
            for line in manifest.read_text(encoding="ascii").splitlines()
        ) + "\n",
        encoding="ascii",
    )


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
    path = tmp_path / (str(packet["stage"]) + "_config.json")
    shutil.copyfile(packet["config"], path)
    return path


@pytest.mark.parametrize("packet", PACKETS, ids=("m2-7", "m2-8"))
def test_current_packets_validate(packet: dict[str, object]) -> None:
    report = validate_packet(config_path=Path(packet["config"]), output_dir=Path(packet["output"]))
    assert report["status"] == packet["status"]
    assert not report["authorization"]["claim_method_superiority"]
    assert not report["authorization"]["open_fresh_or_final"]


@pytest.mark.parametrize("packet", PACKETS, ids=("m2-7", "m2-8"))
def test_checksum_tampering_is_rejected(tmp_path: Path, packet: dict[str, object]) -> None:
    output = _copy(tmp_path, packet)
    (output / "summary.json").write_bytes((output / "summary.json").read_bytes() + b"\n")
    with pytest.raises(ValidationError, match=r"checksum mismatch: summary\.json"):
        validate_packet(config_path=Path(packet["config"]), output_dir=output)


def test_m27_rejects_budget_tampering_after_manifest_refresh(tmp_path: Path) -> None:
    packet = PACKETS[0]
    output = _copy(tmp_path, packet)
    path = output / "metric_rows.csv"
    fields, rows = _csv(path)
    rows[0]["optimization_forward_calls"] = "99"
    _write(path, fields, rows)
    _refresh(output, path.name)
    with pytest.raises(ValidationError, match="F/A budget formula drift"):
        validate_packet(config_path=Path(packet["config"]), output_dir=output)


def test_m27_rejects_k_above_the_frozen_cap(tmp_path: Path) -> None:
    packet = PACKETS[0]
    output = _copy(tmp_path, packet)
    path = output / "metric_rows.csv"
    fields, rows = _csv(path)
    rows[0]["projection_iterations"] = "11"
    _write(path, fields, rows)
    _refresh(output, path.name)
    with pytest.raises(ValidationError, match="K>10"):
        validate_packet(config_path=Path(packet["config"]), output_dir=output)


def test_m27_rejects_camera_block_oracle_setup_tampering(tmp_path: Path) -> None:
    packet = PACKETS[0]
    output = _copy(tmp_path, packet)
    path = output / "metric_rows.csv"
    fields, rows = _csv(path)
    rows[0]["preconditioner_setup_forward_equivalents"] = "0"
    _write(path, fields, rows)
    _refresh(output, path.name)
    with pytest.raises(ValidationError, match="camera-block oracle setup ledger drift"):
        validate_packet(config_path=Path(packet["config"]), output_dir=output)


def test_m27_rejects_forged_tail_risk_pass(tmp_path: Path) -> None:
    packet = PACKETS[0]
    output = _copy(tmp_path, packet)
    path = output / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    for decision in summary["decisions"].values():
        decision["checks"]["development_harm_rate"] = True
        decision["checks"]["development_worst_case"] = True
        decision["passed_m2_3_mechanism_gate"] = True
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _refresh(output, path.name)
    with pytest.raises(ValidationError, match="harm/worst NO-GO checks"):
        validate_packet(config_path=Path(packet["config"]), output_dir=output)


def test_m27_rejects_frozen_source_hash_tampering(tmp_path: Path) -> None:
    packet = PACKETS[0]
    config_path = _copy_config(tmp_path, packet)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["source_m2_6_summary_sha256"] = "0" * 64
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="source_m2_6 summary hash drift"):
        validate_packet(config_path=config_path, output_dir=Path(packet["output"]))


def test_m28_rejects_incomplete_fixed_alpha_grid(tmp_path: Path) -> None:
    packet = PACKETS[1]
    output = _copy(tmp_path, packet)
    path = output / "fixed_interpolation_rows.csv"
    fields, rows = _csv(path)
    rows[0]["interpolation_fraction"] = "0.7"
    _write(path, fields, rows)
    _refresh(output, path.name)
    with pytest.raises(ValidationError, match="fixed alpha identity grid drift"):
        validate_packet(config_path=Path(packet["config"]), output_dir=output)


def test_m28_rejects_truth_interval_tampering(tmp_path: Path) -> None:
    packet = PACKETS[1]
    output = _copy(tmp_path, packet)
    path = output / "truth_oracle_ceiling_rows.csv"
    fields, rows = _csv(path)
    target = next(row for row in rows if row["reprojection_feasible"] == "1")
    target["feasible_alpha_lower"] = "0.0"
    _write(path, fields, rows)
    _refresh(output, path.name)
    with pytest.raises(ValidationError, match="feasible_alpha_lower"):
        validate_packet(config_path=Path(packet["config"]), output_dir=output)


def test_m28_rejects_truth_alpha_minimizer_tampering(tmp_path: Path) -> None:
    packet = PACKETS[1]
    output = _copy(tmp_path, packet)
    path = output / "truth_oracle_ceiling_rows.csv"
    fields, rows = _csv(path)
    target = next(row for row in rows if row["reprojection_feasible"] == "1")
    target["truth_oracle_alpha"] = target["feasible_alpha_upper"]
    _write(path, fields, rows)
    _refresh(output, path.name)
    with pytest.raises(ValidationError, match="truth_oracle_alpha"):
        validate_packet(config_path=Path(packet["config"]), output_dir=output)


def test_m28_rejects_truth_leak_into_candidate(tmp_path: Path) -> None:
    packet = PACKETS[1]
    output = _copy(tmp_path, packet)
    path = output / "fixed_interpolation_rows.csv"
    fields, rows = _csv(path)
    rows[0]["truth_used_by_candidate"] = "True"
    _write(path, fields, rows)
    _refresh(output, path.name)
    with pytest.raises(ValidationError, match="truth leaked into candidate"):
        validate_packet(config_path=Path(packet["config"]), output_dir=output)


def test_m28_rejects_truth_oracle_deployability_tampering(tmp_path: Path) -> None:
    packet = PACKETS[1]
    output = _copy(tmp_path, packet)
    path = output / "truth_oracle_ceiling_rows.csv"
    fields, rows = _csv(path)
    rows[0]["candidate_deployable"] = "True"
    _write(path, fields, rows)
    _refresh(output, path.name)
    with pytest.raises(ValidationError, match="truth oracle was not evaluator-only"):
        validate_packet(config_path=Path(packet["config"]), output_dir=output)


def test_m28_rejects_authorization_escalation(tmp_path: Path) -> None:
    packet = PACKETS[1]
    output = _copy(tmp_path, packet)
    path = output / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["authorization"]["continue_observable_calibration_research"] = True
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _refresh(output, path.name)
    with pytest.raises(ValidationError, match="M2.8 authorization"):
        validate_packet(config_path=Path(packet["config"]), output_dir=output)


def test_auditor_does_not_import_an_experiment_runner() -> None:
    source = (ROOT / "site_tools/validate_jacru_m2_7_m2_8_evidence.py").read_text(encoding="utf-8")
    assert "import site_tools.run_" not in source
    assert "from site_tools import run_" not in source
