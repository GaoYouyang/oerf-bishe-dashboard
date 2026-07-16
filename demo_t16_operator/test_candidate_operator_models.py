"""Mechanism tests for the research candidate operator scaffolds."""

from __future__ import annotations

import torch

from demo_t16_operator.candidate_operator_models import (
    AdaptiveRankInnovation4DOperator,
    CovarianceGeometryUnrolledOperator,
    DataProximalNullspaceSampler,
    apply_linear_adjoint,
    apply_linear_operator,
)


def test_dense_forward_and_adjoint_match_inner_products() -> None:
    torch.manual_seed(1)
    operator = torch.randn(2, 5, 7)
    field = torch.randn(2, 7)
    residual = torch.randn(2, 5)
    lhs = torch.sum(apply_linear_operator(field, operator) * residual)
    rhs = torch.sum(field * apply_linear_adjoint(residual, operator))
    torch.testing.assert_close(lhs, rhs)


def make_unrolled_case() -> tuple[CovarianceGeometryUnrolledOperator, dict[str, torch.Tensor]]:
    torch.manual_seed(2)
    model = CovarianceGeometryUnrolledOperator(
        voxel_count=6,
        geometry_features=4,
        iterations=2,
        initial_step=0.03,
        correction_scale=0.1,
    )
    case = {
        "observation": torch.randn(3, 5),
        "operator": torch.randn(3, 5, 6),
        "geometry": torch.randn(3, 5, 4),
        "active_mask": torch.tensor([[1, 1, 1, 0, 0]] * 3, dtype=torch.float32),
        "support": torch.tensor([1, 1, 1, 1, 0, 0], dtype=torch.float32),
        "initial": torch.full((3, 6), 0.2),
        "noise_std": torch.tensor([[0.8, 1.1, 1.4, 4.0, 5.0]] * 3),
    }
    return model, case


def test_unrolled_ray_order_is_permutation_invariant() -> None:
    model, case = make_unrolled_case()
    model.eval()
    original = model(**case)
    permutation = torch.tensor([2, 4, 0, 3, 1])
    permuted = dict(case)
    permuted["observation"] = case["observation"][:, permutation]
    permuted["operator"] = case["operator"][:, permutation]
    permuted["geometry"] = case["geometry"][:, permutation]
    permuted["active_mask"] = case["active_mask"][:, permutation]
    permuted["noise_std"] = case["noise_std"][:, permutation]
    torch.testing.assert_close(original, model(**permuted), atol=1e-6, rtol=1e-6)


def test_unrolled_ignores_inactive_ray_values() -> None:
    model, case = make_unrolled_case()
    model.eval()
    original = model(**case)
    changed = dict(case)
    changed["observation"] = case["observation"].clone()
    changed["geometry"] = case["geometry"].clone()
    changed["noise_std"] = case["noise_std"].clone()
    changed["observation"][:, 3:] = 1e5
    changed["geometry"][:, 3:] = -1e5
    changed["noise_std"][:, 3:] = 1e-5
    torch.testing.assert_close(original, model(**changed), atol=1e-6, rtol=1e-6)


def test_zero_initialized_unrolled_block_is_prewhitened_projected_gradient() -> None:
    model = CovarianceGeometryUnrolledOperator(
        voxel_count=3,
        geometry_features=2,
        iterations=1,
        initial_step=0.05,
        nonnegative=False,
    )
    observation = torch.tensor([[1.0, -0.5]])
    operator = torch.tensor([[[1.0, 0.5, 0.0], [0.0, 1.0, 1.0]]])
    geometry = torch.zeros(1, 2, 2)
    mask = torch.ones(1, 2)
    support = torch.tensor([1.0, 1.0, 0.0])
    initial = torch.tensor([[0.2, 0.3, 0.4]])
    std = torch.tensor([[0.5, 2.0]])
    prediction = model(
        observation,
        operator,
        geometry,
        mask,
        support,
        initial=initial,
        noise_std=std,
    )
    residual = apply_linear_operator(initial, operator) - observation
    gradient = apply_linear_adjoint(residual / std.square(), operator)
    expected = support * (initial - 0.05 * gradient)
    torch.testing.assert_close(prediction, expected, atol=1e-6, rtol=1e-6)


def test_unrolled_hard_support_and_nonnegativity() -> None:
    model, case = make_unrolled_case()
    result, history = model(**case, return_history=True)
    assert torch.all(result[:, 4:] == 0)
    assert torch.all(result >= 0)
    assert int(history["forward_calls"]) == 2
    assert int(history["adjoint_calls"]) == 2


def test_unrolled_accepts_shared_voxel_coordinates() -> None:
    torch.manual_seed(3)
    model = CovarianceGeometryUnrolledOperator(
        voxel_count=4,
        geometry_features=2,
        voxel_features=3,
        iterations=1,
    )
    result = model(
        observation=torch.randn(2, 3),
        operator=torch.randn(3, 4),
        geometry=torch.randn(2, 3, 2),
        active_mask=torch.ones(2, 3),
        support=torch.ones(4),
        voxel_features=torch.randn(4, 3),
    )
    assert result.shape == (2, 4)


def test_zero_correction_budget_recovers_deterministic_iteration_after_training() -> None:
    model, case = make_unrolled_case()
    with torch.no_grad():
        model.proximal[-1].weight.fill_(0.4)
        model.proximal[-1].bias.fill_(0.2)
    guarded = model(**case, correction_budget=torch.zeros(3))
    control, _ = make_unrolled_case()
    deterministic = control(**case)
    torch.testing.assert_close(guarded, deterministic, atol=1e-6, rtol=1e-6)


def test_nullspace_samples_preserve_measurements_for_full_row_rank_operator() -> None:
    torch.manual_seed(4)
    operator = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, -1.0]])
    truth = torch.tensor([[0.4, -0.2, 0.7], [0.1, 0.3, -0.4]])
    observation = apply_linear_operator(truth, operator)
    visible = torch.zeros_like(truth)
    latent = torch.randn(2, 7, 5)
    sampler = DataProximalNullspaceSampler(
        voxel_count=3, latent_features=5, damping=1e-8
    )
    output = sampler(visible, observation, operator, latent)
    repeated_operator = operator.unsqueeze(0).expand(2 * 7, -1, -1)
    measured = apply_linear_operator(output["samples"].reshape(14, 3), repeated_operator)
    expected = observation[:, None].expand(-1, 7, -1).reshape(14, 2)
    torch.testing.assert_close(measured, expected, atol=2e-5, rtol=2e-5)


def test_4d_operator_is_causal_and_starts_without_innovation() -> None:
    torch.manual_seed(6)
    model = AdaptiveRankInnovation4DOperator(
        measurement_features=4,
        voxel_count=9,
        rank=3,
        hidden_features=10,
    )
    coordinates = torch.randn(9, 3)
    sequence = torch.randn(2, 6, 4)
    output = model(sequence, coordinates)
    assert output["field"].shape == (2, 6, 9)
    torch.testing.assert_close(output["innovation"], torch.zeros_like(output["innovation"]))
    changed_future = sequence.clone()
    changed_future[:, 4:] += 100.0
    future_output = model(changed_future, coordinates)
    torch.testing.assert_close(output["field"][:, :4], future_output["field"][:, :4])
