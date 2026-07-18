from __future__ import annotations

from pathlib import Path

import pytest

import demo_t16_operator.run_n2_pvgr_n4_1_execution_amendment as amendment


base = amendment.base
ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "demo_t16_operator/configs/"
    "n2_pvgr_n4_1_execution_amendment_preregistered_v1.json"
)


def _contracts() -> tuple[dict, dict, dict, dict]:
    return amendment._load_contract(CONFIG)


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


def _parent_row(output: float, residual: float) -> dict[str, str]:
    return {
        "all_gates_pass": "False",
        "high256_to_high512_output_relative_l2": str(output),
        "matched_residual_256_to_512_relative_l2": str(residual),
    }


def test_amendment_inherits_frozen_scientific_contract_without_redeclaring_it() -> None:
    config, scientific, parent, source = _contracts()
    base_attestation = amendment._validate_contract(config, scientific, parent, source)
    assert base_attestation["config_sha256"] == amendment._sha256(
        ROOT / config["base_protocol_config"]
    )
    change = config["implementation_amendment"]
    assert not change["scientific_sample_change"]
    assert not change["threshold_change"]
    assert not change["numerical_route_change"]
    assert not change["reuse_aborted_v1_checkpoint"]


def test_predecision_requests_h2048_without_requiring_its_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, scientific, parent, _ = _contracts()
    cell = base.expand_audit_cells(scientific, parent)[0]
    levels = {
        256: {"diagnostics": _diagnostics()},
        512: {"diagnostics": _diagnostics()},
        1024: {"diagnostics": _diagnostics()},
    }
    adjacent = iter(
        (
            {"output_relative_l2": 1e-4, "matched_residual_relative_l2": 1e-2},
            {"output_relative_l2": 7e-5, "matched_residual_relative_l2": 7e-3},
        )
    )
    monkeypatch.setattr(base, "_adjacent_metrics", lambda *_: next(adjacent))
    preliminary = amendment._h1024_predecision(
        cell,
        levels,
        _parent_row(1e-4, 1e-2),
        scientific["convergence_gates"],
    )
    assert preliminary["requires_h2048_escalation"]
    assert not preliminary["h1024_all_gates_pass"]
    assert 2048 not in levels


def test_frozen_v1_bug_is_reproduced_but_amendment_predecision_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, scientific, parent, _ = _contracts()
    cell = base.expand_audit_cells(scientific, parent)[0]
    levels = {
        256: {"diagnostics": _diagnostics()},
        512: {"diagnostics": _diagnostics()},
        1024: {"diagnostics": _diagnostics()},
    }

    def install_adjacent() -> None:
        adjacent = iter(
            (
                {
                    "output_relative_l2": 1e-4,
                    "matched_residual_relative_l2": 1e-2,
                },
                {
                    "output_relative_l2": 7e-5,
                    "matched_residual_relative_l2": 7e-3,
                },
            )
        )
        monkeypatch.setattr(base, "_adjacent_metrics", lambda *_: next(adjacent))

    install_adjacent()
    with pytest.raises(ValueError, match="H2048 escalation payload is missing"):
        base._cell_decision(
            cell,
            levels,
            _parent_row(1e-4, 1e-2),
            scientific["convergence_gates"],
        )
    install_adjacent()
    assert amendment._h1024_predecision(
        cell,
        levels,
        _parent_row(1e-4, 1e-2),
        scientific["convergence_gates"],
    )["requires_h2048_escalation"]


def test_amendment_hash_cannot_reuse_a_v1_checkpoint() -> None:
    config, _, _, _ = _contracts()
    base_config = ROOT / config["base_protocol_config"]
    assert amendment._sha256(CONFIG) != amendment._sha256(base_config)


def test_committed_amendment_attestation_validates_when_present() -> None:
    config, scientific, parent, source = _contracts()
    amendment._validate_contract(config, scientific, parent, source)
    attestation = ROOT / config["pre_registration_attestation"]
    if not attestation.exists():
        assert not (ROOT / config["formal_output"]).exists()
        assert not (ROOT / config["formal_work_output"]).exists()
        pytest.skip("N4.1 attestation is created after the amendment protocol commit")
    validated = amendment._validate_preregistration(config, CONFIG)
    assert validated["formal_results_absent_at_creation"] is True
    assert validated["formal_work_output_absent_at_creation"] is True
