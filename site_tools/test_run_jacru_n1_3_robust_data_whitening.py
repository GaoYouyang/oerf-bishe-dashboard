from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from site_tools import run_jacru_n1_3_robust_data_whitening as n13


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "demo_t16_operator/configs/"
    "jacru_n1_3_robust_data_whitening_development_v1.json"
)
SOURCE = ROOT / "demo_t16_operator/configs/jacru_m2_learned_residual_t0_v1.json"


def _config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_candidate_grid_is_unique_and_contains_required_controls() -> None:
    candidates = n13._candidate_specs(_config())
    identifiers = [row["candidate_id"] for row in candidates]
    assert len(candidates) == 128
    assert len(identifiers) == len(set(identifiers))
    assert {row["solver_kind"] for row in candidates} == {
        "huber_measurement_pdhg",
        "quadratic_measurement_pdhg_control",
    }
    assert {row["whitening_policy"] for row in candidates} == {
        "unwhitened",
        "diagonal_flowoff",
        "isotropic_flowoff",
        "structured_flowoff",
    }
    assert {row["mean_policy"] for row in candidates} == {"zero", "estimated"}


def test_development_source_config_never_constructs_train_or_ood() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    limited = n13._development_source_config(source, seed_limit=1)
    assert limited["splits"]["train"]["base_seeds"] == []
    assert limited["splits"]["ood"]["base_seeds"] == []
    assert len(limited["splits"]["development"]["base_seeds"]) == 1
    assert source["splits"]["train"]["base_seeds"]
    assert source["splits"]["ood"]["base_seeds"]


def test_whitening_and_transpose_are_an_exact_adjoint_pair() -> None:
    matrix = torch.tensor(
        [[2.0, 0.3, 0.0], [0.3, 1.5, 0.2], [0.0, 0.2, 0.8]],
        dtype=torch.float64,
    )
    factor = torch.linalg.cholesky(matrix)
    left = torch.tensor([[0.4, -0.2, 0.7]], dtype=torch.float64)
    right = torch.tensor([[1.1, 0.3, -0.8]], dtype=torch.float64)
    whitened_left = n13._whiten_vector(left, factor)
    transposed_right = n13._whiten_transpose_vector(right, factor)
    assert torch.sum(whitened_left * right) == pytest.approx(
        float(torch.sum(left * transposed_right)),
        abs=1e-12,
    )


def test_diagonal_whitening_discards_off_diagonal_covariance_only() -> None:
    class Selector:
        proximal_covariance = torch.tensor(
            [[4.0, 1.5], [1.5, 9.0]], dtype=torch.float64
        )

    factor = n13._whitening_factor_for_policy(Selector(), "diagonal_flowoff")
    torch.testing.assert_close(
        factor,
        torch.diag(torch.tensor([2.0, 3.0], dtype=torch.float64)),
    )
    assert n13._whitening_factor_for_policy(Selector(), "unwhitened") is None


def test_sparse_outlier_stress_is_deterministic_and_camera_balanced() -> None:
    observation = torch.zeros((6, 2), dtype=torch.float64)
    cameras = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int64)
    covariance = torch.diag(torch.linspace(0.01, 0.12, 12, dtype=torch.float64))
    first = n13._sparse_outlier_delta(
        observation,
        camera_index=cameras,
        covariance=covariance,
        fraction_per_camera=0.01,
        minimum_components_per_camera=1,
        standardized_amplitude=8.0,
        seed=41,
    )
    second = n13._sparse_outlier_delta(
        observation,
        camera_index=cameras,
        covariance=covariance,
        fraction_per_camera=0.01,
        minimum_components_per_camera=1,
        standardized_amplitude=8.0,
        seed=41,
    )
    assert torch.equal(first, second)
    assert int(torch.count_nonzero(first)) == 2
    assert torch.count_nonzero(first[cameras == 0]) == 1
    assert torch.count_nonzero(first[cameras == 1]) == 1


def test_development_decision_rejects_ood_rows() -> None:
    with pytest.raises(ValueError, match="non-development"):
        n13._decisions(
            [{"split": "ood"}],
            [],
            gates=_config()["development_gates"],
            full_screen_complete=True,
        )


def test_config_firewall_rejects_ood_authorization() -> None:
    config = copy.deepcopy(_config())
    config["may_construct_or_evaluate_ood"] = True
    with pytest.raises(RuntimeError, match="OOD"):
        n13._validate_config(config, seed_limit=None)

    with pytest.raises(ValueError, match="positive"):
        n13._validate_config(_config(), seed_limit=0)
