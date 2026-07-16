from __future__ import annotations

import json
from pathlib import Path

from site_tools.run_psu_b0_pcgls_conditional_headroom import (
    annotate_noise_bins,
    candidate_grid,
    materialize_group_strategy,
    materialize_sample_oracle,
    relative_noise_bin,
    select_candidates_by_group,
)


ROOT = Path(__file__).resolve().parents[1]


def _row(
    sample: str,
    candidate: str,
    *,
    field: float,
    combined: float,
    views: int,
    noise: float,
    family: str = "plume",
) -> dict:
    return {
        "sample_id": sample,
        "split": "risk_train",
        "candidate_id": candidate,
        "method": f"candidate_{candidate}",
        "field_relative_l2": field,
        "gradient_relative_l2": field + 0.1,
        "front_top10_f1": 1.0 - field,
        "measurement_relative_l2": field + 0.2,
        "combined_loss": combined,
        "active_view_count": views,
        "relative_noise": noise,
        "family": family,
    }


def test_candidate_grid_contains_frozen_baseline() -> None:
    config = json.loads(
        (
            ROOT
            / "demo_t16_operator/configs/"
            "psu_b0_pcgls_conditional_headroom_v1.json"
        ).read_text(encoding="utf-8")
    )
    candidates = candidate_grid(config)
    identifiers = {row["candidate_id"] for row in candidates}
    assert len(candidates) == 105
    assert "pcgls4_s4_e0.05_isotropic" in identifiers


def test_relative_noise_bins_are_closed_on_upper_edge() -> None:
    edges = [0.04, 0.07]
    labels = ["low", "medium", "high"]
    assert relative_noise_bin(0.04, edges=edges, labels=labels) == "low"
    assert relative_noise_bin(0.05, edges=edges, labels=labels) == "medium"
    assert relative_noise_bin(0.08, edges=edges, labels=labels) == "high"


def test_group_selection_and_materialization_use_only_declared_strata() -> None:
    rows = [
        _row("a", "c0", field=1.0, combined=1.0, views=6, noise=0.02),
        _row("a", "c1", field=0.8, combined=0.8, views=6, noise=0.02),
        _row("b", "c0", field=0.7, combined=0.7, views=6, noise=0.08),
        _row("b", "c1", field=0.9, combined=0.9, views=6, noise=0.08),
        _row("c", "c0", field=0.8, combined=0.8, views=9, noise=0.02),
        _row("c", "c1", field=0.6, combined=0.6, views=9, noise=0.02),
    ]
    annotated = annotate_noise_bins(
        rows,
        edges=[0.04, 0.07],
        labels=["low", "medium", "high"],
    )
    mapping = select_candidates_by_group(
        annotated,
        group_keys=("active_view_count", "relative_noise_bin"),
    )
    assert mapping[("6", "low")] == "c1"
    assert mapping[("6", "high")] == "c0"
    assert mapping[("9", "low")] == "c1"
    selected, usage = materialize_group_strategy(
        annotated,
        method="stratified",
        mapping=mapping,
        group_keys=("active_view_count", "relative_noise_bin"),
        fallback_candidate_id="c0",
    )
    assert {row["sample_id"]: row["selected_candidate_id"] for row in selected} == {
        "a": "c1",
        "b": "c0",
        "c": "c1",
    }
    assert usage == {"c1": 2, "c0": 1}


def test_sample_oracle_selects_requested_metric() -> None:
    rows = [
        _row("a", "c0", field=0.5, combined=0.7, views=6, noise=0.02),
        _row("a", "c1", field=0.6, combined=0.4, views=6, noise=0.02),
        _row("b", "c0", field=0.7, combined=0.6, views=7, noise=0.05),
        _row("b", "c1", field=0.4, combined=0.8, views=7, noise=0.05),
    ]
    field_rows, _ = materialize_sample_oracle(
        rows,
        method="field_oracle",
        metric="field_relative_l2",
    )
    combined_rows, _ = materialize_sample_oracle(
        rows,
        method="combined_oracle",
        metric="combined_loss",
    )
    assert [row["selected_candidate_id"] for row in field_rows] == ["c0", "c1"]
    assert [row["selected_candidate_id"] for row in combined_rows] == ["c1", "c0"]
