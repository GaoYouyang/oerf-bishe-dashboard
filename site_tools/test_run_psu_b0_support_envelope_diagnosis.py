"""Tests for the post-open PSU support-envelope diagnosis."""

from __future__ import annotations

from site_tools.run_psu_b0_support_envelope_diagnosis import (
    STATUS,
    _paired_equivalence,
    build_public_summary,
)


def _row(method: str, value: float) -> dict[str, object]:
    return {
        "sample_id": "a",
        "split": "test_joint_ood",
        "method": method,
        "field_relative_l2": value,
        "gradient_relative_l2": value,
        "front_top10_f1": value,
        "combined_loss": value,
        "measurement_relative_l2": value,
    }


def test_paired_equivalence_reports_maximum_delta() -> None:
    result = _paired_equivalence(
        [_row("left", 1.0), _row("right", 1.25)],
        left_method="left",
        right_method="right",
        split="test_joint_ood",
    )
    assert result["sample_count"] == 1
    assert result["maximum_over_metrics"] == 0.25


def test_public_summary_strips_private_material() -> None:
    private = {
        "status": STATUS,
        "evidence_scope": "postopen",
        "configuration_public": {"seed_count": 3},
        "dataset_public": {"audit_values_previously_opened": True},
        "aggregates": [],
        "equivalence_diagnostics": [],
        "execution_public": {"wall_seconds": 1.0},
        "gates": {"fresh": False},
        "claim_boundary": {"algorithm_superiority": False},
        "configuration_private": {"view_root": "/private"},
        "dataset_private": {"checkpoint_records": ["secret"]},
    }
    public = build_public_summary(private)
    assert public["status"] == STATUS
    assert "configuration_private" not in public
    assert "dataset_private" not in public
    assert public["public_export_policy"]["fresh_candidate_gate"] is False
