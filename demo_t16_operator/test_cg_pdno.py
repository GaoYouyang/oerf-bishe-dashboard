from __future__ import annotations

import torch

try:
    from .cg_pdno import CGPDNO
    from .measurement_contract import (
        BOSTBatch,
        DepthSeparableLinearBOST,
        estimated_relative_noise,
        geometry_noise_features,
        inverse_variance_trust_budget,
        robust_noise_std_from_repeats,
    )
except ImportError:
    from cg_pdno import CGPDNO
    from measurement_contract import (
        BOSTBatch,
        DepthSeparableLinearBOST,
        estimated_relative_noise,
        geometry_noise_features,
        inverse_variance_trust_budget,
        robust_noise_std_from_repeats,
    )


def make_case(dtype: torch.dtype = torch.float64):
    generator = torch.Generator().manual_seed(124)
    batch_size, depth, height, width = 3, 2, 4, 4
    views, detector = 5, 7
    matrix = torch.rand((views, detector, height * width), generator=generator, dtype=dtype)
    matrix /= torch.linalg.vector_norm(matrix, dim=(1, 2), keepdim=True)
    operator = DepthSeparableLinearBOST(matrix, (depth, height, width))
    truth = torch.rand((batch_size, 1, depth, height, width), generator=generator, dtype=dtype)
    support = torch.ones((1, 1, depth, height, width), dtype=dtype)
    mask = torch.tensor(
        [[1, 1, 1, 1, 1], [1, 0, 1, 0, 1], [0, 1, 1, 1, 0]], dtype=dtype
    )
    angles = torch.tensor(
        [[0, 30, 65, 105, 145], [0, 30, 65, 105, 145], [0, 30, 65, 105, 145]],
        dtype=dtype,
    )
    placeholder = torch.zeros((batch_size, depth, views, detector), dtype=dtype)
    noise = torch.tensor([0.03, 0.08, 0.04, 0.10, 0.06], dtype=dtype)[None, None, :, None]
    shell = BOSTBatch(
        observation=placeholder,
        view_mask=mask,
        noise_std=noise,
        view_angles_degrees=angles,
        support=support,
        geometry_ids=("g0", "g1", "g2"),
        truth=truth,
    )
    observation = operator.forward(truth, shell)
    batch = BOSTBatch(
        observation=observation,
        view_mask=mask,
        noise_std=noise,
        view_angles_degrees=angles,
        support=support,
        geometry_ids=shell.geometry_ids,
        truth=truth,
    ).validate()
    return batch, operator


def test_contract_rejects_nonpositive_active_noise():
    batch, _ = make_case()
    bad_noise = batch.expanded_noise_std().clone()
    bad_noise[0, 0, 0, 0] = 0.0
    broken = BOSTBatch(
        observation=batch.observation,
        view_mask=batch.view_mask,
        noise_std=bad_noise,
        view_angles_degrees=batch.view_angles_degrees,
        support=batch.support,
        geometry_ids=batch.geometry_ids,
        truth=batch.truth,
    )
    try:
        broken.validate()
    except ValueError as exc:
        assert "positive noise_std" in str(exc)
    else:
        raise AssertionError("nonpositive active noise was accepted")


def test_geometry_features_ignore_camera_storage_order():
    batch, operator = make_case()
    permutation = torch.tensor([3, 0, 4, 1, 2])
    permuted = BOSTBatch(
        observation=batch.observation[:, :, permutation],
        view_mask=batch.view_mask[:, permutation],
        noise_std=batch.expanded_noise_std()[:, :, permutation],
        view_angles_degrees=batch.view_angles_degrees[:, permutation],
        support=batch.support,
        geometry_ids=batch.geometry_ids,
        truth=batch.truth,
    )
    assert torch.allclose(
        geometry_noise_features(batch), geometry_noise_features(permuted), atol=1e-12
    )
    assert operator.operator.shape[0] == len(permutation)


def test_linear_adapter_passes_adjoint_identity():
    batch, operator = make_case()
    assert operator.adjoint_relative_error(batch, seed=52) < 1e-12


def test_zero_initialized_model_matches_deterministic_fallback():
    batch, operator = make_case()
    model = CGPDNO(stages=3, proximal_width=4, controller_hidden=6).double()
    start = torch.zeros_like(batch.truth)
    lipschitz = operator.weighted_lipschitz(batch, power_iterations=24)
    output = model(batch, operator, start, lipschitz)
    fallback = model.deterministic_fallback(batch, operator, start, lipschitz)
    assert torch.allclose(output["prediction"], fallback, atol=1e-11, rtol=1e-10)
    assert torch.all(output["normalized_step"] > 0.0)
    assert torch.all(output["normalized_step"] < 2.0)
    assert int(output["forward_calls"]) == 3
    assert int(output["adjoint_calls"]) == 3


def test_training_signal_reaches_controller_and_proximal():
    batch, operator = make_case(dtype=torch.float32)
    model = CGPDNO(stages=2, proximal_width=4, controller_hidden=6)
    start = torch.zeros_like(batch.truth)
    lipschitz = operator.weighted_lipschitz(batch, power_iterations=12)
    output = model(batch, operator, start, lipschitz)
    loss = torch.mean((output["prediction"] - batch.truth) ** 2)
    loss.backward()
    controller_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.controller.parameters()
        if parameter.grad is not None
    )
    proximal_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.proximal.parameters()
        if parameter.grad is not None
    )
    assert controller_grad > 0.0
    assert proximal_grad > 0.0


def test_flow_off_repeat_noise_estimator_is_positive_and_reasonable():
    torch.manual_seed(17)
    true_std = torch.tensor([0.03, 0.08])[None, None, :, None]
    repeats = true_std * torch.randn(51, 1, 3, 2, 80)
    estimated = robust_noise_std_from_repeats(
        repeats, view_mask=torch.ones(1, 2), shrinkage=0.35
    )
    camera_rms = torch.sqrt(torch.mean(estimated.square(), dim=(1, 3))).squeeze(0)
    torch.testing.assert_close(camera_rms, true_std.flatten(), rtol=0.16, atol=0.004)
    assert torch.all(estimated > 0)


def test_relative_noise_and_budget_use_declared_measurement_contract():
    batch, _ = make_case()
    batch.observation.fill_(1.0)
    batch.noise_std.fill_(0.1 / (1.0 + 0.1**2) ** 0.5)
    q_hat = estimated_relative_noise(batch)
    torch.testing.assert_close(q_hat, torch.full_like(q_hat, 0.1), rtol=1e-4, atol=1e-5)
    budget = inverse_variance_trust_budget(batch, reference_noise=0.05, power=2.0)
    torch.testing.assert_close(budget, torch.full_like(budget, 0.25), rtol=1e-4, atol=1e-5)


def test_zero_trust_after_training_recovers_fixed_physics_fallback():
    batch, operator = make_case(dtype=torch.float32)
    model = CGPDNO(stages=2, proximal_width=4, controller_hidden=5)
    with torch.no_grad():
        model.controller.network[-1].weight.fill_(0.7)
        model.controller.network[-1].bias.fill_(1.2)
        model.proximal.network[-1].weight.fill_(0.4)
        model.proximal.network[-1].bias.fill_(0.3)
    lipschitz = operator.weighted_lipschitz(batch, power_iterations=5)
    warm = torch.zeros_like(batch.truth)
    guarded = model(
        batch,
        operator,
        warm,
        lipschitz,
        trust_budget=torch.zeros(len(batch.geometry_ids)),
    )["prediction"]
    fallback = model.deterministic_fallback(batch, operator, warm, lipschitz)
    torch.testing.assert_close(guarded, fallback, rtol=1e-6, atol=1e-6)
