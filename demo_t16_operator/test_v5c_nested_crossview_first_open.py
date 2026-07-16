from __future__ import annotations

from dataclasses import replace

import numpy as np

import demo_t16_operator.run_v5c_nested_crossview_first_open as v5c
from demo_t16_operator.run_v5b_rig_shared_profile_pilot import DevelopmentBlock
from demo_t16_operator.run_v5c_nested_crossview_first_open import (
    block_gate_reasons,
    evaluate,
    outer_view_block_metrics,
    sample_outer_gate,
    write_csv,
)


def gate_config() -> dict[str, object]:
    return {
        "minimum_camera_deletion_radius_stability_fraction": 0.75,
        "minimum_relative_radius_margin": 0.001,
        "minimum_block_outer_view_median_improvement_percent": 2.0,
        "minimum_block_outer_view_positive_fraction": 0.75,
        "minimum_block_outer_view_worst_improvement_percent": -1.0,
        "minimum_sample_outer_improvement_percent": 2.0,
        "maximum_metadata_z": 2.0,
        "require_radius_change": True,
        "require_all_outer_views_improve": True,
    }


def passing_block() -> dict[str, object]:
    return {
        "camera_deletion_radius_stability_fraction": 1.0,
        "camera_deletion_kappa_stability_fraction": 0.75,
        "relative_score_margin": 0.01,
        "relative_radius_margin": 0.02,
        "outer_view_median_improvements_percent": (2.0, 3.0),
        "outer_view_positive_fractions": (0.75, 1.0),
        "outer_view_worst_improvements_percent": (-1.0, 0.5),
        "metadata_z": 1.0,
        "boundary": False,
        "radius_changed": True,
    }


def test_block_gate_passes_without_any_audit_field() -> None:
    passed, reasons = block_gate_reasons(passing_block(), gate_config())
    assert passed
    assert reasons == ()


def test_each_block_failure_has_an_explicit_reason() -> None:
    metrics = passing_block()
    metrics.update(
        {
            "camera_deletion_radius_stability_fraction": 0.5,
            "relative_radius_margin": 0.0,
            "outer_view_median_improvements_percent": (1.9, 3.0),
            "outer_view_positive_fractions": (0.5, 1.0),
            "outer_view_worst_improvements_percent": (-1.1, 0.5),
            "metadata_z": 3.0,
            "boundary": True,
            "radius_changed": False,
        }
    )
    passed, reasons = block_gate_reasons(metrics, gate_config())
    assert not passed
    assert set(reasons) == {
        "radius_unstable",
        "radius_margin_small",
        "outer_view_0_median_below_threshold",
        "outer_view_0_sign_inconsistent",
        "outer_view_0_tail_harm",
        "metadata_conflict",
        "radius_boundary",
        "no_optical_change",
    }


def test_unanimous_outer_gate_rejects_one_bad_camera() -> None:
    passed, reasons = sample_outer_gate([4.0, 1.99], gate_config())
    assert not passed
    assert reasons == ("outer_camera_not_improved",)
    passed, reasons = sample_outer_gate([4.0, 2.0], gate_config())
    assert passed
    assert reasons == ()


def test_outer_block_metrics_never_pool_cameras() -> None:
    metrics = outer_view_block_metrics(
        [[4.0, -3.0], [5.0, -2.0], [6.0, 1.0], [7.0, 2.0]]
    )
    assert metrics["outer_view_median_improvements_percent"] == (5.5, -0.5)
    assert metrics["outer_view_positive_fractions"] == (1.0, 0.5)
    assert metrics["outer_view_worst_improvements_percent"] == (4.0, -3.0)


def _toy_config() -> dict[str, object]:
    return {
        "candidate_aperture_radii": [0.0, 0.5, 1.0],
        "crossview_kappas": [1e-8, 1e-3],
        "ridge_lambda": 1e-3,
        "metadata_sigma": 1.0,
        "support_threshold": 0.5,
        "acceptance": {
            "minimum_camera_deletion_radius_stability_fraction": 0.0,
            "minimum_relative_radius_margin": 0.0,
            "minimum_block_outer_view_median_improvement_percent": -1e9,
            "minimum_block_outer_view_positive_fraction": 0.0,
            "minimum_block_outer_view_worst_improvement_percent": -1e9,
            "minimum_sample_outer_improvement_percent": -1e9,
            "maximum_metadata_z": 100.0,
            "require_radius_change": False,
            "require_all_outer_views_improve": True,
        },
        "pilot_evaluation": {
            "minimum_nearest_bank_match_rate": 0.0,
            "minimum_coverage": 0.0,
            "minimum_accepted_mean_field_gain_percent": -1e9,
            "minimum_accepted_p10_field_gain_percent": -1e9,
            "maximum_accepted_field_harm_rate_over_1_percent": 1.0,
            "maximum_accepted_audit_increase_rate": 1.0,
            "maximum_mean_selected_audit_change_percent": 1e9,
        },
    }


def _toy_block() -> DevelopmentBlock:
    patterns = np.asarray(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5],
            [1.0, 1.5, 2.5, 4.0, 6.0, 8.0, 10.0],
        ],
        dtype=float,
    )
    bank = patterns[:, None, :, None, None]
    fields = (np.asarray([0.7]), np.asarray([1.2]))
    observations = tuple(bank[1, ..., 0] * field[0] for field in fields)
    sigma = tuple(np.ones(7) * 0.05 for _ in fields)
    return DevelopmentBlock(
        rig_id="toy",
        block_id="toy:radius=0.5",
        true_radius=0.5,
        metadata_radius=0.0,
        families=("toy_a", "toy_b"),
        fields=fields,
        clean_observations=observations,
        observations=observations,
        noise_std=sigma,
        reconstruction_bank=bank,
        truth_operator=bank[1],
        inner_views=(0, 1, 2, 3),
        outer_views=(4, 5),
        audit_views=(6,),
    )


def test_end_to_end_evaluate_serializes_and_audit_cannot_change_routing(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        v5c, "support_mask_from_config", lambda _config: np.ones(1, dtype=bool)
    )
    block = _toy_block()
    samples, blocks, candidates, summary = evaluate(_toy_config(), [block])
    write_csv(tmp_path / "samples.csv", samples)
    write_csv(tmp_path / "blocks.csv", blocks)
    write_csv(tmp_path / "candidates.csv", candidates)
    assert len(samples) == 2
    assert summary["decision_construction"][
        "all_decisions_constructed_before_truth_and_audit_evaluation"
    ]

    changed_observations = tuple(value.copy() for value in block.observations)
    for value in changed_observations:
        value[:, 6] += 10_000.0
    changed_block = replace(block, observations=changed_observations)
    changed_samples, _, _, changed_summary = evaluate(_toy_config(), [changed_block])
    assert [row["outcome_code"] for row in changed_samples] == [
        row["outcome_code"] for row in samples
    ]
    assert changed_summary["decision_construction"][
        "in_memory_routing_sha256_before_truth_field_and_audit_evaluation"
    ] == summary["decision_construction"][
        "in_memory_routing_sha256_before_truth_field_and_audit_evaluation"
    ]

    relabelled = replace(
        block,
        true_radius=0.83,
        families=("report_only_x", "report_only_y"),
        block_id="report-only-true-radius-label",
    )
    relabelled_samples, _, _, relabelled_summary = evaluate(
        _toy_config(), [relabelled]
    )
    assert [row["outcome_code"] for row in relabelled_samples] == [
        row["outcome_code"] for row in samples
    ]
    assert relabelled_summary["decision_construction"][
        "in_memory_routing_sha256_before_truth_field_and_audit_evaluation"
    ] == summary["decision_construction"][
        "in_memory_routing_sha256_before_truth_field_and_audit_evaluation"
    ]
