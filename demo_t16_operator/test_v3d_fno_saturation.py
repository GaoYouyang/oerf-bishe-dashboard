from __future__ import annotations

import pytest

from demo_t16_operator.run_v3d_fno_saturation_audit import (
    read_csv,
    relative_improvement_pct,
    select_validation_plateau,
    summarize_validation,
)


RULE = {
    "metric": "validation_relative_l2",
    "maximum_mean_relative_improvement_pct_per_block": 0.5,
    "maximum_seed_relative_improvement_pct_per_block": 1.0,
    "required_consecutive_final_plateau_blocks": 2,
    "selection_uses_validation_only": True,
    "test_and_q_audit_locked_out_of_selection": True,
}


def make_rows(improvements: dict[int, tuple[float, float, float]]) -> list[dict]:
    rows = []
    for epoch, values in improvements.items():
        for seed, value in enumerate(values, start=1):
            rows.append(
                {
                    "model_seed": seed,
                    "cumulative_epochs": epoch,
                    "selected_validation_rel_l2": 0.2 - epoch / 10_000,
                    "relative_validation_improvement_pct": value,
                    "retained_previous_checkpoint": value == 0.0,
                    "cumulative_train_seconds": float(epoch),
                }
            )
    return rows


def test_relative_improvement_uses_previous_checkpoint() -> None:
    assert relative_improvement_pct(0.2, 0.18) == pytest.approx(10.0)
    assert relative_improvement_pct(0.2, 0.2) == 0.0


def test_plateau_requires_final_consecutive_blocks() -> None:
    rows = make_rows(
        {
            24: (0.0, 0.0, 0.0),
            36: (3.0, 2.0, 1.5),
            48: (0.2, 0.3, 0.4),
            60: (0.1, 0.2, 0.3),
        }
    )
    summary = summarize_validation(rows, RULE, 24)
    decision = select_validation_plateau(summary, RULE, 24)
    assert decision["plateau_reached"] is True
    assert decision["plateau_onset_endpoint_epoch"] == 60
    assert decision["reference_checkpoint_epoch"] == 60


def test_late_improvement_breaks_plateau() -> None:
    rows = make_rows(
        {
            24: (0.0, 0.0, 0.0),
            36: (0.2, 0.3, 0.4),
            48: (0.1, 0.2, 0.3),
            60: (1.2, 0.2, 0.1),
        }
    )
    summary = summarize_validation(rows, RULE, 24)
    decision = select_validation_plateau(summary, RULE, 24)
    assert decision["plateau_reached"] is False
    assert decision["plateau_onset_endpoint_epoch"] is None


def test_analysis_csv_reader_restores_boolean_and_numeric_types(tmp_path) -> None:
    path = tmp_path / "checkpoints.csv"
    path.write_text(
        "cumulative_epochs,selected_validation_rel_l2,retained_previous_checkpoint,phase\n"
        "36,0.125,False,continuation\n",
        encoding="utf-8",
    )
    row = read_csv(path)[0]
    assert row == {
        "cumulative_epochs": 36,
        "selected_validation_rel_l2": 0.125,
        "retained_previous_checkpoint": False,
        "phase": "continuation",
    }
