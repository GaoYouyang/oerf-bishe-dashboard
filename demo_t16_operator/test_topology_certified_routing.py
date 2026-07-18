from __future__ import annotations

import itertools
import math

import pytest
import torch

from demo_t16_operator.topology_certified_routing import (
    RoutingValidationError,
    allocate_inclusion_probabilities,
    conditional_trace_variance,
    horvitz_thompson_mean,
    horvitz_thompson_sparse_mean,
    two_replica_quadratic_loss,
    validate_inclusion_probabilities,
)


DTYPE = torch.float64


def test_probability_validation_detaches_and_enforces_unsafe_exactly() -> None:
    probabilities = torch.tensor([0.2, 1.0, 0.75], dtype=DTYPE, requires_grad=True)
    unsafe = torch.tensor([False, True, False])
    validated = validate_inclusion_probabilities(
        probabilities,
        pi_min=0.2,
        unsafe_mask=unsafe,
    )
    assert torch.equal(validated, probabilities.detach())
    assert not validated.requires_grad

    with pytest.raises(RoutingValidationError, match="unsafe replays"):
        validate_inclusion_probabilities(
            torch.tensor([0.2, 1.0 - 1e-12, 0.75], dtype=DTYPE),
            pi_min=0.2,
            unsafe_mask=unsafe,
        )


@pytest.mark.parametrize("pi_min", [0.0, -0.1, 1.01, float("nan"), float("inf")])
def test_probability_validation_rejects_invalid_floor(pi_min: float) -> None:
    with pytest.raises(RoutingValidationError, match="pi_min"):
        validate_inclusion_probabilities(
            torch.tensor([0.5], dtype=DTYPE),
            pi_min=pi_min,
        )


@pytest.mark.parametrize(
    "probabilities, message",
    [
        (torch.tensor([0.1, 0.5], dtype=DTYPE), ">= pi_min"),
        (torch.tensor([0.5, 1.1], dtype=DTYPE), "<= 1"),
        (torch.tensor([0.5, float("nan")], dtype=DTYPE), "finite"),
        (torch.tensor([], dtype=DTYPE), "non-empty"),
        (torch.tensor([[0.5]], dtype=DTYPE), "one-dimensional"),
        (torch.tensor([1], dtype=torch.int64), "floating-point"),
    ],
)
def test_probability_validation_rejects_bad_probability_vectors(
    probabilities: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(RoutingValidationError, match=message):
        validate_inclusion_probabilities(probabilities, pi_min=0.2)


def test_probability_validation_rejects_bad_unsafe_masks() -> None:
    probabilities = torch.tensor([0.5, 1.0], dtype=DTYPE)
    with pytest.raises(RoutingValidationError, match="boolean"):
        validate_inclusion_probabilities(
            probabilities,
            pi_min=0.2,
            unsafe_mask=torch.tensor([0, 1]),
        )
    with pytest.raises(RoutingValidationError, match="shape"):
        validate_inclusion_probabilities(
            probabilities,
            pi_min=0.2,
            unsafe_mask=torch.tensor([True]),
        )


def test_ht_mean_matches_manual_formula_and_supports_replica_batches() -> None:
    low = torch.tensor([[1.0, -1.0], [2.0, 0.5], [-0.5, 3.0]], dtype=DTYPE)
    high = low + torch.tensor([[0.3, -0.2], [1.0, 0.5], [-0.4, 0.8]], dtype=DTYPE)
    probabilities = torch.tensor([0.25, 1.0, 0.5], dtype=DTYPE, requires_grad=True)
    unsafe = torch.tensor([False, True, False])
    uniforms = torch.tensor(
        [[0.1, 0.999, 0.6], [0.4, 0.2, 0.2]],
        dtype=DTYPE,
    )

    estimate = horvitz_thompson_mean(
        low,
        high,
        probabilities,
        uniforms,
        pi_min=0.2,
        unsafe_mask=unsafe,
    )
    indicators = (uniforms < probabilities.detach()).to(DTYPE)
    expected = (
        low.mean(dim=0)
        + ((indicators / probabilities.detach()) @ (high - low)) / low.shape[0]
    )
    assert estimate.shape == (2, 2)
    assert torch.allclose(estimate, expected)
    assert not estimate.requires_grad


def test_ht_mean_preserves_replay_gradients_but_not_probability_gradients() -> None:
    low = torch.tensor([[0.2], [0.4]], dtype=DTYPE, requires_grad=True)
    high = torch.tensor([[0.8], [1.4]], dtype=DTYPE, requires_grad=True)
    probabilities = torch.tensor([0.5, 1.0], dtype=DTYPE, requires_grad=True)
    estimate = horvitz_thompson_mean(
        low,
        high,
        probabilities,
        torch.tensor([0.1, 0.7], dtype=DTYPE),
        pi_min=0.25,
        unsafe_mask=torch.tensor([False, True]),
    )
    estimate.sum().backward()
    assert low.grad is not None
    assert high.grad is not None
    assert probabilities.grad is None


def test_sparse_execution_matches_full_replay_for_the_same_mask() -> None:
    low = torch.tensor(
        [[0.1, -0.2], [0.4, 0.3], [-0.5, 0.7], [0.8, -0.1]],
        dtype=DTYPE,
    )
    high = low + torch.tensor(
        [[0.02, 0.01], [-0.03, 0.05], [0.04, -0.02], [0.01, 0.03]],
        dtype=DTYPE,
    )
    probabilities = torch.tensor([0.25, 1.0, 0.5, 0.75], dtype=DTYPE)
    unsafe = torch.tensor([False, True, False, False])
    uniforms = torch.tensor([0.1, 0.9, 0.7, 0.2], dtype=DTYPE)
    selected = torch.nonzero(uniforms < probabilities, as_tuple=False).flatten()
    replay = horvitz_thompson_mean(
        low,
        high,
        probabilities,
        uniforms,
        pi_min=0.2,
        unsafe_mask=unsafe,
    )
    sparse = horvitz_thompson_sparse_mean(
        low,
        high.index_select(0, selected),
        selected,
        probabilities,
        pi_min=0.2,
        unsafe_mask=unsafe,
    )
    assert torch.equal(sparse, replay)


def test_sparse_execution_rejects_missing_unsafe_and_bad_indices() -> None:
    low = torch.zeros((3, 2), dtype=DTYPE)
    probabilities = torch.tensor([0.3, 1.0, 0.4], dtype=DTYPE)
    unsafe = torch.tensor([False, True, False])
    with pytest.raises(RoutingValidationError, match="unsafe"):
        horvitz_thompson_sparse_mean(
            low,
            torch.zeros((0, 2), dtype=DTYPE),
            torch.empty(0, dtype=torch.long),
            probabilities,
            pi_min=0.2,
            unsafe_mask=unsafe,
        )
    with pytest.raises(RoutingValidationError, match="duplicates"):
        horvitz_thompson_sparse_mean(
            low,
            torch.zeros((2, 2), dtype=DTYPE),
            torch.tensor([1, 1], dtype=torch.long),
            probabilities,
            pi_min=0.2,
            unsafe_mask=unsafe,
        )


def test_ht_unsafe_entries_are_always_included_for_valid_uniforms() -> None:
    low = torch.zeros((2, 1), dtype=DTYPE)
    high = torch.tensor([[4.0], [7.0]], dtype=DTYPE)
    probabilities = torch.tensor([0.2, 1.0], dtype=DTYPE)
    uniforms = torch.tensor([[0.99, 1.0 - 1e-15]], dtype=DTYPE)
    estimate = horvitz_thompson_mean(
        low,
        high,
        probabilities,
        uniforms,
        pi_min=0.2,
        unsafe_mask=torch.tensor([False, True]),
    )
    assert torch.allclose(estimate, torch.tensor([[3.5]], dtype=DTYPE))


def test_ht_monte_carlo_mean_is_unbiased() -> None:
    generator = torch.Generator().manual_seed(712)
    low = torch.randn((17, 4), generator=generator, dtype=DTYPE)
    high = low + 0.7 * torch.randn((17, 4), generator=generator, dtype=DTYPE)
    probabilities = torch.linspace(0.15, 0.95, 17, dtype=DTYPE)
    uniforms = torch.rand((120_000, 17), generator=generator, dtype=DTYPE)
    estimates = horvitz_thompson_mean(
        low,
        high,
        probabilities,
        uniforms,
        pi_min=0.15,
    )
    exact = high.mean(dim=0)
    standard_error = estimates.std(dim=0, unbiased=True) / math.sqrt(estimates.shape[0])
    assert torch.all(
        torch.abs(estimates.mean(dim=0) - exact) < 4.5 * standard_error + 2e-4
    )


@pytest.mark.parametrize(
    "uniforms, message",
    [
        (torch.tensor([0.1], dtype=DTYPE), "sample_count"),
        (torch.tensor([0.1, 1.0], dtype=DTYPE), "half-open"),
        (torch.tensor([-0.1, 0.2], dtype=DTYPE), "half-open"),
        (torch.tensor([0.1, float("nan")], dtype=DTYPE), "finite"),
        (torch.tensor([0, 0]), "floating-point"),
    ],
)
def test_ht_rejects_bad_uniforms(uniforms: torch.Tensor, message: str) -> None:
    with pytest.raises(RoutingValidationError, match=message):
        horvitz_thompson_mean(
            torch.zeros((2, 1), dtype=DTYPE),
            torch.ones((2, 1), dtype=DTYPE),
            torch.tensor([0.5, 0.5], dtype=DTYPE),
            uniforms,
            pi_min=0.1,
        )


def test_ht_rejects_malformed_replays_and_probability_count() -> None:
    with pytest.raises(RoutingValidationError, match="identical shapes"):
        horvitz_thompson_mean(
            torch.zeros((2, 1), dtype=DTYPE),
            torch.zeros((3, 1), dtype=DTYPE),
            torch.tensor([0.5, 0.5], dtype=DTYPE),
            torch.tensor([0.1, 0.2], dtype=DTYPE),
            pi_min=0.1,
        )
    with pytest.raises(RoutingValidationError, match="expected 2"):
        horvitz_thompson_mean(
            torch.zeros((2, 1), dtype=DTYPE),
            torch.ones((2, 1), dtype=DTYPE),
            torch.tensor([0.5, 0.5, 0.5], dtype=DTYPE),
            torch.tensor([0.1, 0.2], dtype=DTYPE),
            pi_min=0.1,
        )


def test_ht_supports_nonzero_sample_dimension() -> None:
    low = torch.tensor([[1.0, 2.0, 3.0], [0.0, -1.0, 4.0]], dtype=DTYPE)
    high = low + torch.tensor([[0.2, 0.4, 0.6], [-0.3, 0.5, 0.1]], dtype=DTYPE)
    probabilities = torch.ones(3, dtype=DTYPE)
    estimate = horvitz_thompson_mean(
        low,
        high,
        probabilities,
        torch.tensor([0.2, 0.4, 0.8], dtype=DTYPE),
        pi_min=0.2,
        sample_dim=1,
    )
    assert torch.allclose(estimate, high.mean(dim=1))


def test_ht_supports_scalar_outputs_with_and_without_replica_batch() -> None:
    low = torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE)
    high = torch.tensor([2.0, 4.0, 6.0], dtype=DTYPE)
    probabilities = torch.ones(3, dtype=DTYPE)
    one = horvitz_thompson_mean(
        low,
        high,
        probabilities,
        torch.tensor([0.1, 0.2, 0.3], dtype=DTYPE),
        pi_min=0.2,
    )
    many = horvitz_thompson_mean(
        low,
        high,
        probabilities,
        torch.tensor([[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]], dtype=DTYPE),
        pi_min=0.2,
    )
    assert one.ndim == 0
    assert one == high.mean()
    assert torch.equal(many, high.mean().expand(2))


def test_conditional_trace_variance_matches_complete_bernoulli_enumeration() -> None:
    low = torch.tensor([[0.0, 1.0], [2.0, -1.0], [0.5, 0.25]], dtype=DTYPE)
    high = low + torch.tensor([[1.0, -0.5], [-0.2, 0.8], [0.7, 0.4]], dtype=DTYPE)
    probabilities = torch.tensor([0.25, 0.6, 1.0], dtype=DTYPE)
    unsafe = torch.tensor([False, False, True])
    exact_trace = conditional_trace_variance(
        low,
        high,
        probabilities,
        pi_min=0.2,
        unsafe_mask=unsafe,
    )

    target = high.mean(dim=0)
    enumerated_trace = torch.zeros((), dtype=DTYPE)
    for bits in itertools.product((0, 1), repeat=2):
        full_bits = (*bits, 1)
        probability = math.prod(
            float(probabilities[index] if bit else 1.0 - probabilities[index])
            for index, bit in enumerate(full_bits)
        )
        uniforms = torch.tensor(
            [
                0.0 if bit else float((probabilities[index] + 1.0) / 2.0)
                for index, bit in enumerate(full_bits)
            ],
            dtype=DTYPE,
        )
        estimate = horvitz_thompson_mean(
            low,
            high,
            probabilities,
            uniforms,
            pi_min=0.2,
            unsafe_mask=unsafe,
        )
        enumerated_trace += probability * torch.sum((estimate - target).square())
    assert torch.allclose(exact_trace, enumerated_trace, atol=2e-15, rtol=2e-15)


def test_conditional_variance_is_zero_at_full_high_fidelity() -> None:
    low = torch.randn((5, 2, 3), dtype=DTYPE)
    high = torch.randn((5, 2, 3), dtype=DTYPE)
    variance = conditional_trace_variance(
        low,
        high,
        torch.ones(5, dtype=DTYPE),
        pi_min=0.1,
    )
    assert variance == 0.0


def test_allocation_respects_budget_floor_unsafe_and_risk_order() -> None:
    risk = torch.tensor([0.0, 1.0, 4.0, 9.0, 2.0, 0.5], dtype=DTYPE, requires_grad=True)
    unsafe = torch.tensor([True, False, False, False, True, False])
    probabilities = allocate_inclusion_probabilities(
        risk,
        average_high_fidelity_budget=0.62,
        pi_floor=0.1,
        unsafe_mask=unsafe,
    )
    assert not probabilities.requires_grad
    assert torch.all(probabilities[unsafe] == 1.0)
    assert torch.all(probabilities[~unsafe] >= 0.1)
    assert torch.all(probabilities <= 1.0)
    assert float(probabilities.mean()) == pytest.approx(0.62, abs=2e-15)
    safe_indices = torch.tensor([1, 2, 3, 5])
    ordered = safe_indices[torch.argsort(risk.detach()[safe_indices])]
    assert torch.all(probabilities[ordered][1:] >= probabilities[ordered][:-1])


def test_allocation_saturates_high_risk_then_uses_remaining_headroom() -> None:
    risk = torch.tensor([1000.0, 10.0, 1.0, 0.0], dtype=DTYPE)
    probabilities = allocate_inclusion_probabilities(
        risk,
        average_high_fidelity_budget=0.7,
        pi_floor=0.1,
    )
    assert probabilities[0] == 1.0
    assert probabilities[1] >= probabilities[2] >= probabilities[3]
    assert float(probabilities.mean()) == pytest.approx(0.7, abs=2e-15)


def test_allocation_matches_clipped_kkt_not_floor_plus_risk() -> None:
    probabilities = allocate_inclusion_probabilities(
        torch.tensor([1.0, 2.0], dtype=DTYPE),
        average_high_fidelity_budget=0.5,
        pi_floor=0.2,
    )
    expected = torch.tensor([1.0 / 3.0, 2.0 / 3.0], dtype=DTYPE)
    assert torch.allclose(probabilities, expected, atol=2e-15, rtol=0.0)
    interior_ratio = probabilities / torch.tensor([1.0, 2.0], dtype=DTYPE)
    assert float(torch.max(interior_ratio) - torch.min(interior_ratio)) <= 2e-15


def test_allocation_zero_risk_fallback_is_uniform_and_budget_exact() -> None:
    probabilities = allocate_inclusion_probabilities(
        torch.zeros(5, dtype=DTYPE),
        average_high_fidelity_budget=0.4,
        pi_floor=0.05,
    )
    assert torch.allclose(probabilities, torch.full((5,), 0.4, dtype=DTYPE))


def test_allocation_minimum_budget_and_all_unsafe_boundary() -> None:
    risk = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=DTYPE)
    unsafe = torch.tensor([True, False, True, False])
    minimum = (2.0 + 2.0 * 0.2) / 4.0
    probabilities = allocate_inclusion_probabilities(
        risk,
        average_high_fidelity_budget=minimum,
        pi_floor=0.2,
        unsafe_mask=unsafe,
    )
    assert torch.equal(probabilities, torch.tensor([1.0, 0.2, 1.0, 0.2], dtype=DTYPE))

    all_unsafe = torch.ones(4, dtype=torch.bool)
    assert torch.equal(
        allocate_inclusion_probabilities(
            risk,
            average_high_fidelity_budget=1.0,
            pi_floor=0.2,
            unsafe_mask=all_unsafe,
        ),
        torch.ones(4, dtype=DTYPE),
    )
    with pytest.raises(
        RoutingValidationError, match="minimum average budget|all entries"
    ):
        allocate_inclusion_probabilities(
            risk,
            average_high_fidelity_budget=0.9,
            pi_floor=0.2,
            unsafe_mask=all_unsafe,
        )


@pytest.mark.parametrize(
    "risk, budget, floor, message",
    [
        (torch.tensor([1.0, -0.1], dtype=DTYPE), 0.5, 0.1, "non-negative"),
        (torch.tensor([1.0, float("nan")], dtype=DTYPE), 0.5, 0.1, "finite"),
        (torch.tensor([1.0, 2.0], dtype=DTYPE), 0.0, 0.1, "budget"),
        (torch.tensor([1.0, 2.0], dtype=DTYPE), 1.1, 0.1, "budget"),
        (torch.tensor([1.0, 2.0], dtype=DTYPE), 0.5, 0.0, "pi_min"),
    ],
)
def test_allocation_rejects_invalid_inputs(
    risk: torch.Tensor,
    budget: float,
    floor: float,
    message: str,
) -> None:
    with pytest.raises(RoutingValidationError, match=message):
        allocate_inclusion_probabilities(
            risk,
            average_high_fidelity_budget=budget,
            pi_floor=floor,
        )


def test_allocation_rejects_infeasible_unsafe_budget() -> None:
    with pytest.raises(RoutingValidationError, match="minimum average budget"):
        allocate_inclusion_probabilities(
            torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE),
            average_high_fidelity_budget=0.4,
            pi_floor=0.2,
            unsafe_mask=torch.tensor([True, True, False]),
        )


def test_two_replica_quadratic_loss_matches_cross_product_reductions() -> None:
    replica_a = torch.tensor([1.5, -0.5, 2.0], dtype=DTYPE)
    replica_b = torch.tensor([0.5, 1.0, 3.5], dtype=DTYPE)
    target = torch.tensor([1.0, 0.0, 2.5], dtype=DTYPE)
    elementwise = (replica_a - target) * (replica_b - target)
    assert torch.equal(
        two_replica_quadratic_loss(replica_a, replica_b, target, reduction="none"),
        elementwise,
    )
    assert (
        two_replica_quadratic_loss(replica_a, replica_b, target) == elementwise.mean()
    )
    assert (
        two_replica_quadratic_loss(replica_a, replica_b, target, reduction="sum")
        == elementwise.sum()
    )


def test_two_independent_replica_loss_is_unbiased_for_squared_error() -> None:
    generator = torch.Generator().manual_seed(991)
    repetitions = 100_000
    prediction_mean = torch.tensor([0.5, -1.2, 2.0], dtype=DTYPE)
    target = torch.tensor([0.1, -0.7, 1.4], dtype=DTYPE)
    noise_a = 0.8 * torch.randn((repetitions, 3), generator=generator, dtype=DTYPE)
    noise_b = 0.8 * torch.randn((repetitions, 3), generator=generator, dtype=DTYPE)
    replica_a = prediction_mean + noise_a
    replica_b = prediction_mean + noise_b
    repeated_target = target.expand_as(replica_a)
    losses = two_replica_quadratic_loss(
        replica_a,
        replica_b,
        repeated_target,
        reduction="none",
    ).mean(dim=1)
    exact = torch.mean((prediction_mean - target).square())
    standard_error = losses.std(unbiased=True) / math.sqrt(repetitions)
    assert torch.abs(losses.mean() - exact) < 4.5 * standard_error


def test_two_replica_loss_rejects_bad_contracts() -> None:
    vector = torch.ones(2, dtype=DTYPE)
    with pytest.raises(RoutingValidationError, match="identical shapes"):
        two_replica_quadratic_loss(vector, torch.ones(3, dtype=DTYPE), vector)
    with pytest.raises(RoutingValidationError, match="floating-point"):
        two_replica_quadratic_loss(
            torch.ones(2, dtype=torch.int64),
            torch.ones(2, dtype=torch.int64),
            torch.ones(2, dtype=torch.int64),
        )
    with pytest.raises(RoutingValidationError, match="reduction"):
        two_replica_quadratic_loss(vector, vector, vector, reduction="median")  # type: ignore[arg-type]
