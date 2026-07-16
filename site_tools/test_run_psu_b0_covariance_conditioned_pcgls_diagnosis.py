from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from site_tools.run_psu_b0_covariance_conditioned_pcgls_diagnosis import (
    candidate_grid,
    candidate_is_eligible,
    select_candidate,
    validate_execution_plan,
    validate_partition,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "demo_t16_operator"
    / "configs"
    / "psu_b0_covariance_conditioned_pcgls_diagnosis_v1.json"
)


def _config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _passing_summary(
    candidate_id: str,
    *,
    mean: float,
    p10: float = 0.0,
    harm: float = 0.0,
    front: float = 0.01,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "covariance_mode": "spatial_tempered",
        "spatial_exponent": 0.5,
        "sobolev_strength": 5.0,
        "sobolev_epsilon": 0.05,
        "stages": 4,
        "mean_field_gain_percent": mean,
        "field_gain_p10_percent": p10,
        "field_harm_over_one_percent_rate": harm,
        "mean_gradient_gain_percent": 0.1,
        "mean_front_f1_gain": front,
    }


def test_frozen_grid_and_shared_call_budget_are_self_consistent() -> None:
    config = _config()
    candidates = candidate_grid(config)
    assert len(candidates) == 120
    assert len({row["candidate_id"] for row in candidates}) == 120
    budget = validate_execution_plan(
        config,
        candidates=candidates,
        replicate_count=16,
    )
    assert budget["logical_calls_total"] == 6784
    assert budget["physical_calls_total"] == 2464
    assert budget["physical_call_reduction_percent"] == pytest.approx(
        63.67924528301887
    )


def test_partition_is_complete_and_rejects_overlap() -> None:
    config = _config()
    mapping = validate_partition(config, replicate_count=16)
    assert [mapping[index] for index in range(8)] == ["selection"] * 8
    assert [mapping[index] for index in range(8, 16)] == [
        "opened_diagnostic_check"
    ] * 8
    invalid = copy.deepcopy(config)
    invalid["replicate_partition"]["opened_diagnostic_check"][0] = 7
    with pytest.raises(ValueError, match="overlap"):
        validate_partition(invalid, replicate_count=16)


def test_candidate_eligibility_separates_controls_from_selection() -> None:
    config = _config()
    candidates = {
        row["candidate_id"]: row for row in candidate_grid(config)
    }
    eligibility = config["candidate_eligibility"]
    assert not candidate_is_eligible(
        candidates["spatial_a0_s5_k4"],
        eligibility=eligibility,
    )
    assert not candidate_is_eligible(
        candidates["spatial_a0p25_s5_k5"],
        eligibility=eligibility,
    )
    assert candidate_is_eligible(
        candidates["spatial_a0p25_s5_k4"],
        eligibility=eligibility,
    )
    assert candidate_is_eligible(
        candidates["full_graph_s5_k4"],
        eligibility=eligibility,
    )


def test_selection_applies_gates_then_declared_tie_break() -> None:
    config = _config()
    summaries = [
        _passing_summary("lower_mean", mean=0.8, p10=0.2),
        _passing_summary("winner", mean=1.2, p10=0.1),
        _passing_summary("fails_tail", mean=2.0, p10=-0.7),
    ]
    selected = select_candidate(
        summaries,
        gates=config["selection_gates"],
        eligibility=config["candidate_eligibility"],
    )
    assert selected is not None
    assert selected["candidate_id"] == "winner"
    assert all(selected["gate_checks"].values())
