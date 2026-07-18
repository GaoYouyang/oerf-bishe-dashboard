from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

import demo_t16_operator.run_n2_pvgr_n4_evaluator_convergence as n4


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "demo_t16_operator/configs/"
    "n2_pvgr_n4_evaluator_convergence_preregistered_v1.json"
)


def _config() -> dict[str, object]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _parents(config: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    parent = json.loads((ROOT / str(config["parent_n3_config"])).read_text())
    source = json.loads((ROOT / str(config["source_config"])).read_text())
    return parent, source


def test_contract_contains_all_and_only_16_n3_failures_plus_16_controls() -> None:
    config = _config()
    parent, source = _parents(config)
    n4._validate_contract(config, parent, source)
    cells = n4.expand_audit_cells(config, parent)
    assert len(cells) == 32
    assert len({cell["cell_id"] for cell in cells}) == 32
    assert sum(cell["role"] == "n3_failure" for cell in cells) == 16
    assert sum(cell["role"] == "matched_control" for cell in cells) == 16
    assert {cell["contrast_factor"] for cell in cells} == {"orientation", "aperture"}


def test_pair_contract_rejects_cross_field_or_two_factor_control() -> None:
    config = _config()
    parent, source = _parents(config)
    config["audit_pairs"][0]["control_cell"][
        "case_id"
    ] = "smooth-s1871-orientation_22-narrow"
    with pytest.raises(ValueError, match="field unit"):
        n4._validate_contract(config, parent, source)

    config = _config()
    config["audit_pairs"][0]["control_cell"][
        "case_id"
    ] = "smooth-s1729-orientation_22-narrow"
    with pytest.raises(ValueError, match="exactly one geometry"):
        n4._validate_contract(config, parent, source)


def test_contraction_gate_is_fail_closed_at_machine_zero() -> None:
    gates = _config()["convergence_gates"]
    scoreable = n4._contraction(1e-3, 4e-4, gates)
    assert scoreable["scoreable"]
    assert scoreable["contraction_gate_met"]
    assert scoreable["contraction_ratio"] == pytest.approx(0.4)

    unscoreable_pass = n4._contraction(1e-14, 5e-13, gates)
    assert not unscoreable_pass["scoreable"]
    assert unscoreable_pass["contraction_gate_met"]
    unscoreable_fail = n4._contraction(1e-14, 2e-12, gates)
    assert not unscoreable_fail["scoreable"]
    assert not unscoreable_fail["contraction_gate_met"]


def _diagnostics() -> dict[str, object]:
    return {
        "finite_ray_fraction": 1.0,
        "minimum_domain_margin": 0.2,
        "minimum_stencil_margin": 0.1,
        "maximum_direction_norm_error": 1e-15,
        "support_crossings_per_ray": [2] * 256,
        "frustum_violations_per_ray": [False] * 256,
        "frustum_violation_count": 0,
        "minimum_frustum_margin": 0.001,
    }


def test_endpoint_integrity_gate_detects_topology_and_frustum_drift() -> None:
    gates = _config()["convergence_gates"]
    lower = {"diagnostics": _diagnostics()}
    upper = {"diagnostics": _diagnostics()}
    assert all(n4._endpoint_integrity_gates(lower, upper, gates).values())
    upper = copy.deepcopy(upper)
    upper["diagnostics"]["support_crossings_per_ray"][17] = 3
    result = n4._endpoint_integrity_gates(lower, upper, gates)
    assert not result["support_topology_gate_met"]
    upper = {"diagnostics": _diagnostics()}
    upper["diagnostics"]["frustum_violations_per_ray"][5] = True
    upper["diagnostics"]["frustum_violation_count"] = 1
    result = n4._endpoint_integrity_gates(lower, upper, gates)
    assert not result["frustum_topology_gate_met"]
    assert not result["frustum_violation_gate_met"]


def test_checkpoint_validation_rejects_tampered_numerical_array() -> None:
    cell = {
        "cell_id": "x__stress_1",
        "case_id": "x",
        "field_unit_id": "f",
        "family_id": "smooth",
        "phantom_family": "smooth_plume",
        "phantom_seed": 1,
        "orientation_id": "o",
        "aperture_id": "a",
        "pair_id": "p01",
        "role": "n3_failure",
        "contrast_factor": "aperture",
        "dimensionless_stress_multiplier": 1.0,
        "n3_failed_gate": "matched_residual_convergence_gate_met",
    }
    metadata = n4._level_metadata(cell, 256)
    array = np.zeros((256, 2), dtype=np.float64)
    payload = {
        "schema": "n2-pvgr-n4-level-checkpoint-1.0",
        "preregistration_sha256": "abc",
        "metadata": metadata,
        "high_output_uv": array.tolist(),
        "high_output_uv_sha256": n4._array_sha256(array),
        "matched_residual_uv": array.tolist(),
        "matched_residual_uv_sha256": n4._array_sha256(array),
    }
    n4._validate_level_payload(payload, metadata, "abc")
    payload["high_output_uv"][0][0] = 1.0
    with pytest.raises(ValueError, match="hash drifted"):
        n4._validate_level_payload(payload, metadata, "abc")


def test_committed_attestation_validates_when_present() -> None:
    config = _config()
    parent, source = _parents(config)
    n4._validate_contract(config, parent, source)
    attestation = ROOT / str(config["pre_registration_attestation"])
    if not attestation.exists():
        assert not (ROOT / str(config["formal_output"])).exists()
        assert not (ROOT / str(config["formal_work_output"])).exists()
        pytest.skip("N4 attestation is intentionally created after protocol commit")
    validated = n4._validate_preregistration(config, CONFIG)
    assert validated["formal_results_absent_at_creation"] is True
    assert validated["formal_work_output_absent_at_creation"] is True
