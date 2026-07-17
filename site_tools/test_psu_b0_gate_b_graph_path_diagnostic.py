from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from demo_t16_operator.psu_b0_classical_baselines import (
    preconditioned_cgls_trajectory,
)
from site_tools.run_psu_b0_factor_gate_b import REPOSITORY_ROOT


RESULTS = (
    REPOSITORY_ROOT
    / "demo_t16_operator/results/psu_b0_gate_b_graph_path_diagnostic"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _TinyOperator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.grid_shape = (1, 1, 2)
        self.register_buffer("support", torch.ones(self.grid_shape, dtype=torch.float64))
        self.register_buffer(
            "matrix",
            torch.tensor([[1.0, 0.2], [0.3, 1.4]], dtype=torch.float64),
        )

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        values = volume[:, 0].reshape(len(volume), 2)
        return (values @ self.matrix.T).reshape(len(volume), 1, 2)

    def adjoint(self, detector: torch.Tensor) -> torch.Tensor:
        values = detector.reshape(len(detector), 2)
        return (values @ self.matrix).reshape(len(detector), 1, *self.grid_shape)


class _IdentityDirection:
    def __call__(self, normal: torch.Tensor, **_: object) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return normal, {}


def test_graph_path_diagnostic_is_graph_only_complete_and_checksummed() -> None:
    report = json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))
    with (RESULTS / "metric_differences.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    declared = {}
    for line in (RESULTS / "checksums.sha256").read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        declared[name] = digest
    assert declared == {
        "report.json": _sha256(RESULTS / "report.json"),
        "metric_differences.csv": _sha256(RESULTS / "metric_differences.csv"),
    }
    assert report["status"] == "POSTDIAGNOSTIC_GRAPH_PATH_NONBINDING_ONLY"
    assert report["factor_solver_calls"] == 0
    assert report["factor_metric_rows_observed"] == 0
    assert report["diagnostic_repeat_count"] == 3
    assert report["graph_row_count"] == len(rows) == 192
    assert report["parent_batch_rows_authorized_as_binding_control"] is False
    assert report["algorithm_superiority_claim_authorized"] is False
    assert {int(row["diagnostic_repeat"]) for row in rows} == {0, 1, 2}
    maximum = max(
        float(row[f"absolute_difference_{metric}"])
        for row in rows
        for metric in ("field_relative_l2", "gradient_relative_l2", "front_top10_f1")
    )
    assert report["maximum_absolute_metric_difference"] == pytest.approx(maximum)


def test_pcgls_batch_and_singleton_are_sample_separable_on_cpu_float64() -> None:
    operator = _TinyOperator()
    observation = torch.tensor(
        [[[0.7, -0.2]], [[-0.4, 1.1]]],
        dtype=torch.float64,
    )
    sigma = torch.ones((2, 1), dtype=torch.float64)
    mask = torch.ones((2, 1), dtype=torch.float64)
    batch = preconditioned_cgls_trajectory(
        operator,
        observation,
        sigma_by_view=sigma,
        view_mask=mask,
        rays_per_view=1,
        checkpoint_stages=(1, 2, 3),
        preconditioner=_IdentityDirection(),
    )
    single = [
        preconditioned_cgls_trajectory(
            operator,
            observation[index : index + 1],
            sigma_by_view=sigma[index : index + 1],
            view_mask=mask[index : index + 1],
            rays_per_view=1,
            checkpoint_stages=(1, 2, 3),
            preconditioner=_IdentityDirection(),
        )
        for index in range(2)
    ]
    for iteration in (1, 2, 3):
        expected_volume = torch.cat(
            [trajectory[iteration].volume for trajectory in single],
            dim=0,
        )
        expected_residual = torch.cat(
            [trajectory[iteration].residual_uv for trajectory in single],
            dim=0,
        )
        assert torch.allclose(batch[iteration].volume, expected_volume, atol=1e-12, rtol=1e-12)
        assert torch.allclose(batch[iteration].residual_uv, expected_residual, atol=1e-12, rtol=1e-12)
