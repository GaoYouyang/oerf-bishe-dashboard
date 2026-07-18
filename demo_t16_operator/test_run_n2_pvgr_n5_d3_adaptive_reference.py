from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import demo_t16_operator.run_n2_pvgr_n5_d3_adaptive_reference as d3
import site_tools.validate_n2_pvgr_n5_d3_adaptive_reference as d3_validator


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "demo_t16_operator/configs/n2_pvgr_n5_d3_adaptive_reference_preregistered_v1.json"
)


def _config() -> dict[str, object]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_contract_freezes_mixed_semantics_and_exact_order() -> None:
    config = _config()
    d3._validate_contract(config)
    assert len(config["expected_cell_order"]) == 32
    assert config["reference_policy"]["mixed_reference_pack"] is True
    assert config["reference_policy"]["uniform_paired_reference"] is False
    assert not any(config["claim_authorizations"].values())


def test_contract_rejects_allocation_order_and_uniformity_drift() -> None:
    config = _config()
    config["expected_allocation"]["H1024_raw_separate_subtraction"] = 22
    with pytest.raises(ValueError, match="allocation drifted"):
        d3._validate_contract(config)

    config = _config()
    config["expected_cell_order"][0], config["expected_cell_order"][1] = (
        config["expected_cell_order"][1],
        config["expected_cell_order"][0],
    )
    with pytest.raises(ValueError, match="cell order drifted"):
        d3._validate_contract(config)

    config = _config()
    config["reference_policy"]["uniform_paired_reference"] = True
    with pytest.raises(ValueError, match="reference policy drifted"):
        d3._validate_contract(config)


def test_parent_evidence_derives_exact_23_7_2_mapping() -> None:
    config = _config()
    parents = d3._validate_parents(config)
    allocation = d3._allocation(parents["n4_result"], config)
    counts = {
        step: sum(row["step_count"] == step for row in allocation)
        for step in (1024, 2048, 8192)
    }
    assert counts == {1024: 23, 2048: 7, 8192: 2}
    tail = [row for row in allocation if row["step_count"] == 8192]
    assert tuple(row["cell"]["cell_id"] for row in tail) == d3.TAIL_CELLS
    assert all(row["reference_method"] == "paired_neumaier" for row in tail)


def test_pack_arrays_are_finite_hash_stable_and_mixed() -> None:
    config = _config()
    parents = d3._validate_parents(config)
    allocation = d3._allocation(parents["n4_result"], config)
    cells = d3._pack_cells(allocation, config, parents["d2_levels"])
    stacked = np.stack([row["reference_values"] for row in cells])
    assert stacked.shape == (32, 256, 2)
    assert np.all(np.isfinite(stacked))
    assert d3._array_sha256(stacked) == (
        "8d2bba156028e4b14385f5a563d4d7c18817bb17a70dc0856bfeb240e8e765ed"
    )
    assert {row["reference_method"] for row in cells} == {
        "raw_separate_subtraction",
        "paired_neumaier",
    }
    for parent, packed in zip(parents["n4_result"]["cells"], cells, strict=True):
        identity = d3._identity_record(parent)
        assert packed["identity_sha256"] == d3._canonical_json_sha256(identity)
        assert packed["identity_sha256"] == d3_validator._canonical_json_sha256(
            d3_validator._identity_record(parent)
        )
    ledger_row = d3_validator._expected_ledger_row(cells[0], 0)
    d3_validator._verify_ledger_row(cells[0], ledger_row, 0)
    altered_ledger = dict(ledger_row)
    altered_ledger["source_logical_point_queries"] = "1"
    with pytest.raises(ValueError, match="source_logical_point_queries"):
        d3_validator._verify_ledger_row(cells[0], altered_ledger, 0)
    gates = d3._gate_results(cells, config, parents)
    assert all(gates.values())


def test_canonical_array_hash_is_little_endian_and_shape_sensitive() -> None:
    base = np.arange(12, dtype=np.float64).reshape(3, 4)
    big_endian = base.astype(">f8")
    assert d3._array_sha256(base) == d3._array_sha256(big_endian)
    assert d3._array_sha256(base) != d3._array_sha256(base.reshape(2, 6))


def test_independent_validator_rejects_missing_authorization_and_observable_drift() -> None:
    config = _config()
    d3_validator._validate_config_contract(config)
    result = {"authorizations": dict(config["claim_authorizations"])}
    pack = {
        "reference_policy": dict(config["reference_policy"]),
        "observable_contract": dict(config["observable_contract"]),
    }
    d3_validator._verify_claim_structures(config, result, pack)

    missing = {"authorizations": dict(config["claim_authorizations"])}
    missing["authorizations"].pop("field_jvp_vjp")
    with pytest.raises(ValueError, match="authorization structure"):
        d3_validator._verify_claim_structures(config, missing, pack)

    altered_pack = {
        "reference_policy": dict(config["reference_policy"]),
        "observable_contract": dict(config["observable_contract"]),
    }
    altered_pack["observable_contract"]["units"] = "pixel"
    with pytest.raises(ValueError, match="observable contract"):
        d3_validator._verify_claim_structures(config, result, altered_pack)

    altered_config = _config()
    altered_config["dtype"] = "float32"
    with pytest.raises(ValueError, match="numerical encoding"):
        d3_validator._validate_config_contract(altered_config)


def test_manifest_hashes_staging_but_points_to_final_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(d3, "ROOT", tmp_path)
    staging = tmp_path / "formal.staging"
    output = tmp_path / "formal"
    staging.mkdir()
    source = tmp_path / "source.txt"
    source.write_text("source\n", encoding="utf-8")
    artifact = staging / "result.json"
    artifact.write_text('{"valid": true}\n', encoding="utf-8")
    manifest = d3._manifest(staging, output, {"source": source}, ["result.json"])
    assert manifest["files"]["result.json"]["path"] == "formal/result.json"
    assert manifest["files"]["result.json"]["sha256"] == d3._sha256(artifact)


def test_committed_attestation_validates_when_present() -> None:
    config = _config()
    attestation = ROOT / str(config["pre_registration_attestation"])
    if not attestation.exists():
        assert not (ROOT / str(config["formal_output"])).exists()
        pytest.skip("N5-D3 attestation is created after the protocol commit")
    result = d3._validate_preregistration(config, CONFIG)
    assert result["formal_results_absent_at_creation"] is True
