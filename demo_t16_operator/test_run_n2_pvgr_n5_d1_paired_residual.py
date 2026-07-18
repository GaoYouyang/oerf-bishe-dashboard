from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import demo_t16_operator.run_n2_pvgr_n5_d1_paired_residual as d1


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "demo_t16_operator/configs/n2_pvgr_n5_d1_paired_residual_preregistered_v1.json"
)


def _config() -> dict[str, object]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_contract_freezes_four_post_n4_cells_and_no_broad_claims() -> None:
    config = _config()
    d1._validate_contract(config)
    _, scientific, parent, _ = d1._load_parent_contract(config)
    cells = d1._selected_cells(config, scientific, parent)
    assert len(cells) == 4
    assert {cell["pair_id"] for cell in cells} == {"p04", "p05"}
    assert sum(cell["role"] == "matched_control" for cell in cells) == 2
    assert sum(cell["role"] == "n3_failure" for cell in cells) == 2
    assert all(value is False for value in config["claim_authorizations"].values())


def test_contract_rejects_cell_threshold_or_toy_drift() -> None:
    config = _config()
    config["selected_cells"][0] = "smooth-s1729-orientation_58-narrow__stress_1"
    with pytest.raises(ValueError, match="selected cells"):
        d1._validate_contract(config)

    config = _config()
    config["gates"][
        "maximum_accumulation_fraction_of_refinement_for_too_small"
    ] = 0.02
    with pytest.raises(ValueError, match="gates drifted"):
        d1._validate_contract(config)

    config = _config()
    config["toy_contract"]["weak_step_count"] = 24
    with pytest.raises(ValueError, match="toy contract drifted"):
        d1._validate_contract(config)


def test_decision_boundaries_are_fail_closed_and_mixed_is_inconclusive() -> None:
    gates = _config()["gates"]
    assert d1._decision(
        contract_gates_pass=False,
        failed_cell_maximum_fractions=[0.0, 0.0],
        gates=gates,
    ) == "D1_CONTRACT_OR_REPRODUCTION_FAILED_CLOSED"
    assert d1._decision(
        contract_gates_pass=True,
        failed_cell_maximum_fractions=[0.009, 0.01],
        gates=gates,
    ) == "D1_ACCUMULATION_ORDER_TOO_SMALL_TO_EXPLAIN_N4_FLOOR"
    assert d1._decision(
        contract_gates_pass=True,
        failed_cell_maximum_fractions=[0.1, 0.3],
        gates=gates,
    ) == "D1_ACCUMULATION_ORDER_PLAUSIBLY_EXPLAINS_N4_FLOOR"
    assert d1._decision(
        contract_gates_pass=True,
        failed_cell_maximum_fractions=[0.005, 0.2],
        gates=gates,
    ) == "D1_ACCUMULATION_ORDER_INCONCLUSIVE"


def test_toy_contract_passes_without_selected_cell_results() -> None:
    checks = d1._toy_checks(_config())
    assert checks["constant_gate_met"]
    assert checks["weak_gate_met"]
    assert checks["rotation_gate_met"]


def test_manifest_hashes_staging_artifact_but_points_to_final_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(d1, "ROOT", tmp_path)
    staging = tmp_path / "formal.staging"
    output = tmp_path / "formal"
    staging.mkdir()
    source = tmp_path / "source.txt"
    source.write_text("frozen source\n", encoding="utf-8")
    artifact = staging / "result.json"
    artifact.write_text('{"valid": true}\n', encoding="utf-8")
    manifest = d1._manifest(
        staging,
        output,
        {"source": source},
        ["result.json"],
    )
    assert manifest["files"]["result.json"]["path"] == "formal/result.json"
    assert manifest["files"]["result.json"]["sha256"] == d1._sha256(artifact)


def test_committed_attestation_validates_when_present() -> None:
    config = _config()
    attestation = ROOT / str(config["pre_registration_attestation"])
    if not attestation.exists():
        assert not (ROOT / str(config["formal_output"])).exists()
        pytest.skip("N5-D1 attestation is created after the protocol commit")
    result = d1._validate_preregistration(config, CONFIG)
    assert result["formal_results_absent_at_creation"] is True
