"""Tests for the post-open TV/Huber scale-screen runner."""

from __future__ import annotations

import copy

import pytest

from site_tools.run_psu_b0_edge_superiorization_screen import (
    add_comparative_gains,
    baseline_candidates,
    rank_superiorized,
    screen_decision,
    summarize_candidates,
    superiorized_candidate_grid,
    validate_replicates,
)


def _config() -> dict:
    return {
        "replicate_indices": [0, 8],
        "baselines": {
            "component_stages": 4,
            "graph_stages": [3, 4, 5, 6],
        },
        "screen": {
            "penalties": ["tv", "huber"],
            "stages": [3, 4],
            "perturbation_inner_steps": [1, 2],
            "perturbation_initial_steps": [0.005, 0.02, 0.08, 0.32],
            "perturbation_decays": [0.7],
            "superiorized_candidate_count_expected": 32,
        },
    }


def test_candidate_grid_and_call_budgets() -> None:
    candidates = superiorized_candidate_grid(_config())
    assert len(candidates) == 32
    assert len({row["candidate_id"] for row in candidates}) == 32
    for row in candidates:
        stages = int(row["stages"])
        assert row["forward_calls"] == 2 * stages - 1
        assert row["adjoint_calls"] == stages
        assert row["total_operator_calls"] == 3 * stages - 1
    baselines = baseline_candidates(_config())
    assert [row["total_operator_calls"] for row in baselines] == [
        8,
        6,
        8,
        10,
        12,
    ]


def test_replicate_validation_rejects_overlap_and_out_of_range() -> None:
    assert validate_replicates(_config(), replicate_count=16) == [0, 8]
    duplicate = copy.deepcopy(_config())
    duplicate["replicate_indices"] = [0, 0]
    with pytest.raises(ValueError, match="unique"):
        validate_replicates(duplicate, replicate_count=16)
    outside = copy.deepcopy(_config())
    outside["replicate_indices"] = [0, 16]
    with pytest.raises(ValueError, match="outside"):
        validate_replicates(outside, replicate_count=16)


def test_comparative_gain_uses_total_call_floor_and_ceiling() -> None:
    candidates = baseline_candidates(_config())
    stage_four = next(
        row
        for row in superiorized_candidate_grid(_config())
        if int(row["stages"]) == 4
    )
    sup = {**stage_four, "candidate_id": "sup"}
    rows = []
    for replicate in (0, 8):
        for sample in range(2):
            for candidate in [*candidates, sup]:
                error = {
                    "component_s3_k4": 1.0,
                    "graph_s3_k3": 0.98,
                    "graph_s3_k4": 0.95,
                    "graph_s3_k5": 0.90,
                    "graph_s3_k6": 0.88,
                    "sup": 0.87,
                }[candidate["candidate_id"]]
                rows.append(
                    {
                        "replicate": replicate,
                        "sample_index": sample,
                        "reaction_family": f"f{sample}",
                        **candidate,
                        "field_relative_l2": error,
                        "gradient_relative_l2": error,
                        "front_top10_f1": 1.0 - error / 2.0,
                        "solver_elapsed_seconds": 0.1,
                        "mean_perturbation_norm": 0.0,
                    }
                )
    comparative = add_comparative_gains(
        rows,
        component_baseline_id="component_s3_k4",
    )
    sup_rows = [row for row in comparative if row["candidate_id"] == "sup"]
    assert {row["graph_budget_floor_id"] for row in sup_rows} == {
        "graph_s3_k5"
    }
    assert {row["graph_budget_ceiling_id"] for row in sup_rows} == {
        "graph_s3_k6"
    }
    assert {row["graph_same_stage_id"] for row in sup_rows} == {
        "graph_s3_k4"
    }
    summaries = summarize_candidates(comparative)
    ranked = rank_superiorized(summaries)
    assert ranked[0]["candidate_id"] == "sup"
    assert ranked[0]["mean_field_gain_vs_graph_budget_floor_percent"] > 0.0
    decision = screen_decision(ranked)
    assert decision["status"] == "POSTOPEN_SUPPCG_BUDGET_SIGNAL_PRESENT"
    assert decision["fresh_authorized"] is False
