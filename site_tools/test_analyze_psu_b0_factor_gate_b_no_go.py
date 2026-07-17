from __future__ import annotations

import json
from pathlib import Path

from site_tools.analyze_psu_b0_factor_gate_b_no_go import (
    EXPECTED_STATUS,
    build_public_bundle,
    verify_input_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "demo_t16_operator/results/psu_b0_factor_pdhg_gate_b"


def test_formal_bundle_checksums_are_complete() -> None:
    hashes = verify_input_bundle(INPUT)
    assert set(hashes) == {
        "audit.json",
        "metric_rows.csv",
        "report.json",
        "validation_report.json",
    }


def test_public_bundle_is_sanitized_and_recomputed(tmp_path: Path) -> None:
    output = tmp_path / "public"
    summary = build_public_bundle(INPUT, output)
    assert summary["status"] == EXPECTED_STATUS
    assert summary["gate_counts"] == {"passed": 5, "failed": 3}
    assert summary["neural_training_authorized"] is False
    assert summary["algorithm_superiority_claim_authorized"] is False
    assert summary["connectivity"] == {
        "support_active_voxels": 2744,
        "data_coupled_voxels": 2322,
        "data_null_support_voxels": 422,
    }
    assert len((output / "decision_gates.csv").read_text().splitlines()) == 9
    assert len((output / "method_frontier.csv").read_text().splitlines()) == 17
    assert len((output / "paired_k32_gains.csv").read_text().splitlines()) == 17
    assert (output / "factor_gate_b_no_go.png").stat().st_size > 100_000
    assert (output / "factor_gate_b_no_go.pdf").stat().st_size > 10_000
    reloaded = json.loads((output / "summary.json").read_text())
    assert reloaded == summary
    assert "/Users/" not in (output / "summary.json").read_text()
