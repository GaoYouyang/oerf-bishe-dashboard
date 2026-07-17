from __future__ import annotations

import copy

import pytest

from demo_t16_operator.jacru_n1_2_dual_reference import (
    add_dual_reference_metrics,
    aggregate_dual_reference_rows,
    dual_reference_decisions,
)


def _base_row(**updates):
    row = {
        "candidate_id": "joint_band",
        "method": "jacru_m2",
        "model_seed": 17,
        "split": "development",
        "session_id": "session-a",
        "family": "single_interface",
        "candidate_field_relative_l2": 0.80,
        "candidate_h1_relative_error": 0.85,
        "raw_field_relative_l2": 1.00,
        "raw_h1_relative_error": 1.00,
        "registered_classical_field_relative_l2": 0.90,
        "registered_classical_h1_relative_error": 0.95,
        "clean_reprojection_ratio_to_base": 0.90,
        "selector_valid": True,
        "residual_closure_relative_error": 1e-12,
    }
    row.update(updates)
    return row


def _gates():
    return {
        "development_classical_field_gain_mean_minimum": 0.05,
        "development_classical_h1_gain_mean_minimum": 0.03,
        "ood_classical_field_gain_mean_minimum": 0.02,
        "ood_classical_h1_gain_mean_minimum": 0.0,
        "raw_field_gain_mean_minimum": 0.0,
        "field_harm_rate_maximum": 0.05,
        "worst_field_gain_minimum": -0.05,
        "minimum_session_mean_field_gain": -0.05,
        "minimum_family_mean_field_gain": 0.0,
        "development_clean_ratio_mean_maximum": 1.10,
        "development_clean_ratio_worst_maximum": 1.50,
        "ood_clean_ratio_mean_maximum": 1.15,
        "ood_clean_ratio_worst_maximum": 1.75,
        "selector_valid_rate_minimum": 0.95,
        "residual_closure_relative_error_maximum": 1e-10,
    }


def _metric_rows():
    rows = []
    for split in ("development", "ood"):
        for seed in (17, 29):
            for session, family in (
                ("session-a", "single_interface"),
                ("session-b", "smooth_no_interface"),
            ):
                rows.append(
                    add_dual_reference_metrics(
                        _base_row(
                            split=split,
                            model_seed=seed,
                            session_id=session,
                            family=family,
                        ),
                        harm_threshold_fraction=0.01,
                    )
                )
    return rows


def test_dual_reference_metrics_use_their_own_denominators() -> None:
    row = add_dual_reference_metrics(_base_row(), harm_threshold_fraction=0.01)
    assert row["field_gain_to_raw"] == pytest.approx(0.20)
    assert row["field_gain_to_registered_classical"] == pytest.approx(1.0 / 9.0)
    assert row["h1_gain_to_raw"] == pytest.approx(0.15)
    assert row["h1_gain_to_registered_classical"] == pytest.approx(0.10 / 0.95)
    assert not row["field_harm_vs_raw"]
    assert not row["field_harm_vs_registered_classical"]


def test_aggregate_preserves_seed_session_and_family_tails() -> None:
    rows = _metric_rows()
    rows[0] = add_dual_reference_metrics(
        _base_row(
            split="development",
            model_seed=17,
            session_id="session-a",
            family="single_interface",
            candidate_field_relative_l2=1.20,
        ),
        harm_threshold_fraction=0.01,
    )
    aggregate = aggregate_dual_reference_rows(rows)[0]
    assert aggregate["row_count"] == 4
    assert aggregate["worst_field_gain_to_raw"] == pytest.approx(-0.20)
    assert set(aggregate["session_mean_field_gains_to_raw"]) == {"session-a", "session-b"}
    assert set(aggregate["family_mean_field_gains_to_raw"]) == {
        "single_interface",
        "smooth_no_interface",
    }
    assert set(aggregate["per_model_seed_field_gain_means_to_raw"]) == {"17", "29"}


def test_joint_decision_passes_only_when_both_references_and_calibration_pass() -> None:
    aggregates = aggregate_dual_reference_rows(_metric_rows())
    decisions = dual_reference_decisions(
        aggregates,
        candidate_metadata={
            "joint_band": {
                "uses_truth": False,
                "uses_exact_nuisance": False,
                "selector_family": "session_joint_conformal_band",
            }
        },
        candidate_calibration_sanity_passed={"joint_band": True},
        gates=_gates(),
    )
    assert len(decisions) == 1
    assert decisions[0]["passed"]
    assert all(decisions[0]["checks"].values())


def test_raw_harm_cannot_be_hidden_by_a_classical_win() -> None:
    rows = _metric_rows()
    for row in rows:
        row["field_gain_to_raw"] = -0.06
        row["field_harm_vs_raw"] = True
    decision = dual_reference_decisions(
        aggregate_dual_reference_rows(rows),
        candidate_metadata={"joint_band": {}},
        candidate_calibration_sanity_passed={"joint_band": True},
        gates=_gates(),
    )[0]
    assert not decision["checks"]["development_raw_field_mean"]
    assert not decision["checks"]["development_raw_harm"]
    assert not decision["passed"]


def test_one_bad_session_survives_positive_global_mean() -> None:
    rows = _metric_rows()
    for row in rows:
        if row["session_id"] == "session-a":
            row["field_gain_to_raw"] = -0.08
            row["field_harm_vs_raw"] = True
        else:
            row["field_gain_to_raw"] = 0.30
            row["field_harm_vs_raw"] = False
    aggregate = aggregate_dual_reference_rows(rows)
    development = next(row for row in aggregate if row["split"] == "development")
    assert development["field_gain_to_raw_mean"] > 0.0
    assert development["minimum_session_mean_field_gain_to_raw"] == pytest.approx(-0.08)
    decision = dual_reference_decisions(
        aggregate,
        candidate_metadata={"joint_band": {}},
        candidate_calibration_sanity_passed={"joint_band": True},
        gates=_gates(),
    )[0]
    assert not decision["checks"]["session_tail_raw"]
    assert not decision["passed"]


def test_candidate_specific_calibration_fails_closed() -> None:
    decision = dual_reference_decisions(
        aggregate_dual_reference_rows(_metric_rows()),
        candidate_metadata={"joint_band": {}},
        candidate_calibration_sanity_passed={},
        gates=_gates(),
    )[0]
    assert not decision["checks"]["candidate_specific_calibration_sanity"]
    assert not decision["passed"]


def test_mutating_only_classical_denominator_changes_only_classical_gain() -> None:
    original = add_dual_reference_metrics(_base_row(), harm_threshold_fraction=0.01)
    mutated_input = copy.deepcopy(_base_row())
    mutated_input["registered_classical_field_relative_l2"] = 1.20
    mutated = add_dual_reference_metrics(mutated_input, harm_threshold_fraction=0.01)
    assert mutated["field_gain_to_raw"] == original["field_gain_to_raw"]
    assert mutated["field_gain_to_registered_classical"] != original[
        "field_gain_to_registered_classical"
    ]


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"candidate_field_relative_l2": 0.0}, "must be positive"),
        ({"raw_field_relative_l2": float("nan")}, "must be finite"),
    ],
)
def test_invalid_metric_rows_fail_closed(updates, message) -> None:
    with pytest.raises(ValueError, match=message):
        add_dual_reference_metrics(
            _base_row(**updates), harm_threshold_fraction=0.01
        )
