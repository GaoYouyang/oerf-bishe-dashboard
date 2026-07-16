"""Tests for the conditioned-PCGLS development runner."""

from __future__ import annotations

from site_tools.run_psu_b0_conditioned_pcgls_development import (
    _seed_gates,
    build_public_summary,
    paired_gain_summary,
)


def _rows() -> list[dict]:
    output = []
    for split in ("risk_validation", "risk_calibration"):
        for index, baseline in enumerate((1.0, 0.8, 0.6, 0.5)):
            output.append(
                {
                    "split": split,
                    "sample_id": f"{split}-{index}",
                    "method": "static_pcgls4",
                    "field_relative_l2": baseline,
                }
            )
            output.append(
                {
                    "split": split,
                    "sample_id": f"{split}-{index}",
                    "method": "conditioned_pcgls_seed_7",
                    "field_relative_l2": 0.95 * baseline,
                }
            )
    return output


def test_paired_gain_summary_is_strictly_paired() -> None:
    summary = paired_gain_summary(
        _rows(),
        split="risk_validation",
        candidate_method="conditioned_pcgls_seed_7",
        bootstrap_seed=1,
    )
    assert abs(summary["mean_field_gain_percent"] - 5.0) < 1e-10
    assert summary["win_count"] == 4
    assert summary["harm_over_one_percent_count"] == 0


def test_seed_gate_requires_validation_and_calibration() -> None:
    summaries = [
        paired_gain_summary(
            _rows(),
            split=split,
            candidate_method="conditioned_pcgls_seed_7",
            bootstrap_seed=index + 1,
        )
        for index, split in enumerate(
            ("risk_validation", "risk_calibration")
        )
    ]
    config = {
        "development_gates": {
            "validation_mean_field_gain_percent_minimum": 2.0,
            "calibration_mean_field_gain_percent_minimum": 2.0,
            "validation_bootstrap_lower_percent_minimum": 0.0,
            "calibration_bootstrap_lower_percent_minimum": 0.0,
            "validation_harm_over_one_percent_rate_maximum": 0.05,
            "calibration_harm_over_one_percent_rate_maximum": 0.05,
            "seeds_required_to_pass": 1,
        }
    }
    gates = _seed_gates(summaries, seeds=[7], config=config)
    assert gates["development_gate_pass"]
    summaries[1]["mean_field_gain_percent"] = -1.0
    gates = _seed_gates(summaries, seeds=[7], config=config)
    assert not gates["development_gate_pass"]


def test_public_summary_removes_checkpoint_material() -> None:
    private = {
        "status": "test",
        "evidence_scope": "development",
        "configuration_public": {"data_firewall": {"opened_fresh": "forbidden"}},
        "regeneration_checks": {"opened_fresh_not_loaded": True},
        "training": [
            {
                "seed": 1,
                "parameter_count": 10,
                "checkpoint_sha256_private": "secret",
                "state_dict": {"weight": "secret"},
            }
        ],
        "paired_gain_summary": [],
        "seed_gates": {"development_gate_pass": False},
        "execution": [],
        "runtime": {"wall_seconds": 1.0},
        "claim_boundary": {"algorithm_superiority": False},
    }
    public = build_public_summary(private)
    assert "checkpoint_sha256_private" not in public["training"][0]
    assert "state_dict" not in public["training"][0]
