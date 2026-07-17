from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from site_tools import run_jacru_n1_4_adaptive_edge_warm_robust as n14


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "demo_t16_operator/configs/"
    "jacru_n1_4_adaptive_edge_warm_robust_development_v1.json"
)
CONFIG_V1_1 = (
    ROOT
    / "demo_t16_operator/configs/"
    "jacru_n1_4_adaptive_edge_warm_robust_development_v1_1.json"
)


def _config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _config_v1_1():
    return json.loads(CONFIG_V1_1.read_text(encoding="utf-8"))


def test_candidate_grid_has_matched_controls_and_unique_ids() -> None:
    assert n14.DIAGNOSTIC_TITLE.startswith("N1.4 ")
    candidates = n14._candidate_specs(_config())
    identifiers = [row["candidate_id"] for row in candidates]
    assert len(candidates) == 31
    assert len(identifiers) == len(set(identifiers))
    assert sum(row["solver_kind"].startswith("adaptive_edge") for row in candidates) == 27
    assert sum(row["solver_kind"].startswith("uniform_edge") for row in candidates) == 3
    assert sum(row["solver_kind"].startswith("zero_start_24") for row in candidates) == 1
    assert {row["mean_policy"] for row in candidates} == {"estimated"}


def test_v1_1_adds_matched_zero_start_controls_for_every_edge_weight() -> None:
    config = _config_v1_1()
    n14._validate_config(config, seed_limit=None)
    candidates = n14._candidate_specs(config)
    assert len(candidates) == 33
    zero_start = [
        row for row in candidates if row["solver_kind"].startswith("zero_start_24")
    ]
    assert len(zero_start) == 3
    assert {
        row["standardized_edge_regularization_weight"] for row in zero_start
    } == {0.05, 0.1, 0.2}


def test_v1_1_rejects_an_unmatched_zero_start_factorial() -> None:
    broken = copy.deepcopy(_config_v1_1())
    broken["candidate_grid"]["zero_start_control_regularization_weights"] = [0.1]
    with pytest.raises(ValueError, match="match every screened edge weight"):
        n14._validate_config(broken, seed_limit=None)


def test_full_development_contract_rejects_missing_seed_family_pair() -> None:
    source = {
        "splits": {
            "development": {
                "base_seeds": [1, 2],
                "families": ["smooth", "interface"],
            }
        }
    }
    complete = [
        SimpleNamespace(base_seed=seed, family=family)
        for seed in (1, 2)
        for family in ("smooth", "interface")
    ]
    n14._assert_full_development_contract(complete, source)
    with pytest.raises(RuntimeError, match="missing"):
        n14._assert_full_development_contract(complete[:-1], source)


def test_adaptive_edge_map_downweights_a_detected_step() -> None:
    field = torch.zeros((6, 6, 6), dtype=torch.float64)
    field[:, :, 3:] = 1.0
    support = torch.ones_like(field, dtype=torch.bool)
    weights, diagnostics = n14._adaptive_edge_weight_map(
        field,
        support=support,
        spacing_xyz=(1.0, 1.0, 1.0),
        quantile=0.85,
        minimum_weight=0.1,
        power=2.0,
    )
    assert weights.shape == field.shape
    assert torch.all((weights >= 0.1) & (weights <= 1.0))
    assert diagnostics["edge_indicator_threshold"] > 0.0
    assert torch.min(weights[:, :, 2:4]) < torch.mean(weights[:, :, :2])


def test_n1_4_budget_and_ood_firewall_are_fail_closed() -> None:
    n14._validate_config(_config(), seed_limit=None)
    broken = copy.deepcopy(_config())
    broken["solve_budget"]["total_forward_calls"] = 25
    with pytest.raises(ValueError, match="do not sum"):
        n14._validate_config(broken, seed_limit=None)
    broken = copy.deepcopy(_config())
    broken["may_construct_or_evaluate_ood"] = True
    with pytest.raises(RuntimeError, match="OOD"):
        n14._validate_config(broken, seed_limit=None)
    broken = copy.deepcopy(_config())
    broken["solve_budget"]["warm_start_forward_calls"] = 11
    with pytest.raises(ValueError, match="warm-start iteration and forward-call"):
        n14._validate_config(broken, seed_limit=None)
    broken = copy.deepcopy(_config())
    broken["registered_references"]["total_adjoint_calls"] = 23
    with pytest.raises(ValueError, match="registered reference and candidate adjoint"):
        n14._validate_config(broken, seed_limit=None)
