"""Serial-only MPS checks for the PSU-B0 Gate A attestation."""

from __future__ import annotations

import pytest
import torch

from demo_t16_operator.psu_b0_gate_a_fixture import load_gate_a_config
from site_tools.run_psu_b0_gate_a_attestation import collect_mps_parity_evidence
from site_tools.validate_psu_b0_gate_a_attestation import (
    Validator,
    _independent_mps_check,
)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="formal Gate A requires Apple MPS and rejects this skip",
)
def test_mps_factor_recurrence_matches_cpu_reference() -> None:
    config = load_gate_a_config()
    evidence = collect_mps_parity_evidence(config)
    assert evidence["available"] is True
    assert (
        evidence["field_relative_difference"]
        <= config["thresholds"]["mps_float32_field_relative_difference_max"]
    )
    assert evidence["maximum_state_relative_difference"] <= 5e-4


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="independent MPS replay requires Apple MPS",
)
def test_independent_mps_validator_recomputes_parity() -> None:
    config = load_gate_a_config()
    mps = collect_mps_parity_evidence(config)
    validator = Validator()
    result = _independent_mps_check(
        validator,
        {"mps_float32_parity": mps},
        config,
    )
    assert validator.checks >= 3
    assert result["field_relative_difference"] <= result["threshold"]
