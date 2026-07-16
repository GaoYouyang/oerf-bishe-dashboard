from __future__ import annotations

import pytest
import torch

from demo_t16_operator.psu_b0_spectral_preconditioner import (
    IterativeReconstruction,
)
from site_tools.run_psu_b0_covariance_stopping_feature_audit import (
    _trajectory_feature_rows,
)


def _snapshot(
    *,
    stage: int,
    volume_value: float,
    residual_value: float,
) -> IterativeReconstruction:
    history = []
    for index in range(stage):
        row = {
            "stage": torch.full((2,), index + 1, dtype=torch.int64),
            "alpha": torch.full((2,), 0.1 * (index + 1)),
            "relative_objective_before": torch.ones(2),
            "relative_objective_after": torch.full((2,), 0.5),
        }
        if index + 1 < stage:
            row["beta"] = torch.full((2,), 0.2 * (index + 1))
        history.append(row)
    return IterativeReconstruction(
        volume=torch.full((2, 1, 4, 4, 4), volume_value),
        residual_uv=torch.full((2, 6, 2), residual_value),
        history=history,
        forward_calls=stage,
        adjoint_calls=stage,
    )


def test_trajectory_rows_use_only_declared_observable_features() -> None:
    rows = _trajectory_feature_rows(
        replicate=3,
        split="selection",
        families=["plume", "thin_front"],
        prepared_observation=torch.ones((2, 6, 2)),
        trajectory={
            2: _snapshot(
                stage=2,
                volume_value=1.0,
                residual_value=0.5,
            ),
            3: _snapshot(
                stage=3,
                volume_value=1.25,
                residual_value=0.4,
            ),
        },
        baseline_metrics={
            "field_relative_l2": torch.tensor([0.5, 0.6]),
            "gradient_relative_l2": torch.tensor([0.7, 0.8]),
            "front_top10_f1": torch.tensor([0.3, 0.4]),
        },
        spacing_xyz=(1.0, 1.0, 1.0),
    )
    assert len(rows) == 4
    assert {row["stage"] for row in rows} == {2, 3}
    assert {row["reaction_family"] for row in rows} == {
        "plume",
        "thin_front",
    }
    assert all(row["replicate"] == 3 for row in rows)
    assert all(row["split"] == "selection" for row in rows)
    assert rows[0]["relative_whitened_residual_objective"] == 0.25
    assert rows[2]["relative_whitened_residual_objective"] == pytest.approx(
        0.16
    )
