from pathlib import Path

import pytest

from site_tools.validate_jacru_m0_evidence import ValidationError, validate_packet


ROOT = Path(__file__).resolve().parents[1]


def _paths() -> dict[str, Path]:
    return {
        "m0_config_path": ROOT / "demo_t16_operator/configs/jacru_m0_synthetic_gate_v1.json",
        "m0_report_path": ROOT / "demo_t16_operator/results/jacru_m0_synthetic_gate_public/summary.json",
        "m0_rows_path": ROOT / "demo_t16_operator/results/jacru_m0_synthetic_gate_public/metric_rows.csv",
        "initialization_report_path": ROOT / "demo_t16_operator/results/jacru_m0_initialization_audit_public/summary.json",
        "m0_1_config_path": ROOT / "demo_t16_operator/configs/jacru_m0_synthetic_gate_v1_1_diagnostic.json",
        "m0_1_report_path": ROOT / "demo_t16_operator/results/jacru_m0_1_diagnostic_public/summary.json",
        "m0_1_rows_path": ROOT / "demo_t16_operator/results/jacru_m0_1_diagnostic_public/metric_rows.csv",
        "m1_config_path": ROOT / "demo_t16_operator/configs/jacru_m1_cgls_residual_diagnostic_v1.json",
        "m1_report_path": ROOT / "demo_t16_operator/results/jacru_m1_cgls_residual_diagnostic_public/summary.json",
        "m1_rows_path": ROOT / "demo_t16_operator/results/jacru_m1_cgls_residual_diagnostic_public/metric_rows.csv",
    }


def test_current_negative_evidence_packet_validates() -> None:
    report = validate_packet(**_paths())
    assert report["status"] == "VALIDATED_NEGATIVE_EVIDENCE_PACKET"
    assert report["authorization"]["continue_learned_residual_operator"] is True
    assert report["authorization"]["claim_jacru_superiority"] is False
    assert report["diagnostics"]["m1_vs_cgls_field_gain_fraction"] > 0.0
    assert report["diagnostics"]["m1_vs_huber_field_gain_fraction"] < 0.0


def test_hash_tampering_is_rejected(tmp_path: Path) -> None:
    paths = _paths()
    copied = tmp_path / "m0.json"
    copied.write_bytes(paths["m0_report_path"].read_bytes() + b"\n")
    paths["m0_report_path"] = copied
    with pytest.raises(ValidationError, match="anchor M0 report"):
        validate_packet(**paths)
