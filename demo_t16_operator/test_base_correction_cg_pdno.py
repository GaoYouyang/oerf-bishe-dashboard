from __future__ import annotations

import numpy as np
import torch

try:
    from .cg_pdno import BaseCorrectionCGPDNO, PBBBaseCorrectionCGPDNO, descent_certificate
    from .independent_reaction_bost import (
        build_curved_cone_operator,
        correlated_camera_noise,
        make_reaction_field,
    )
    from .measurement_contract import BOSTBatch, DenseVolumeLinearBOST
    from .run_pbb_ensemble_selective_gate import (
        apply_uncertainty_gate,
        shared_ensemble_state,
    )
except ImportError:
    from cg_pdno import BaseCorrectionCGPDNO, PBBBaseCorrectionCGPDNO, descent_certificate
    from independent_reaction_bost import (
        build_curved_cone_operator,
        correlated_camera_noise,
        make_reaction_field,
    )
    from measurement_contract import BOSTBatch, DenseVolumeLinearBOST
    from run_pbb_ensemble_selective_gate import (
        apply_uncertainty_gate,
        shared_ensemble_state,
    )


def make_dense_case(dtype: torch.dtype = torch.float64):
    generator = torch.Generator().manual_seed(77)
    batch_size, depth, height, width = 3, 2, 3, 3
    views, detector = 4, 3
    matrix = torch.randn(
        (depth, views, detector, depth * height * width),
        generator=generator,
        dtype=dtype,
    )
    matrix /= torch.linalg.vector_norm(matrix.flatten(0, 2), dim=0).mean()
    operator = DenseVolumeLinearBOST(matrix, (depth, height, width))
    truth = torch.rand(
        (batch_size, 1, depth, height, width), generator=generator, dtype=dtype
    )
    mask = torch.tensor(
        [[1, 1, 1, 1], [1, 0, 1, 1], [0, 1, 1, 1]], dtype=dtype
    )
    angles = torch.tensor(
        [[3, 48, 93, 138], [3, 48, 93, 138], [3, 48, 93, 138]], dtype=dtype
    )
    support = torch.ones((1, 1, depth, height, width), dtype=dtype)
    noise = torch.tensor([0.04, 0.06, 0.08, 0.05], dtype=dtype)[None, None, :, None]
    empty = torch.zeros((batch_size, depth, views, detector), dtype=dtype)
    shell = BOSTBatch(
        observation=empty,
        view_mask=mask,
        noise_std=noise,
        view_angles_degrees=angles,
        support=support,
        geometry_ids=("d0", "d1", "d2"),
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


def test_dense_volume_operator_passes_adjoint_identity():
    batch, operator = make_dense_case()
    assert operator.adjoint_relative_error(batch, seed=19) < 1e-12


def test_exact_lipschitz_dominates_power_iteration_estimate():
    batch, operator = make_dense_case(dtype=torch.float64)
    exact = operator.weighted_lipschitz_exact(batch)
    estimate = operator.weighted_lipschitz(batch, power_iterations=30)
    assert torch.all(exact >= estimate * (1.0 - 1e-8))
    assert torch.all(torch.isfinite(exact))


def test_base_correction_zero_init_is_same_call_fallback():
    batch, operator = make_dense_case(dtype=torch.float32)
    model = BaseCorrectionCGPDNO(stages=3, correction_width=5)
    lipschitz = 1.08 * operator.weighted_lipschitz(batch, power_iterations=30)
    warm = torch.zeros_like(batch.truth)
    output = model(batch, operator, warm, lipschitz)
    fallback = model.deterministic_fallback(batch, operator, warm, lipschitz)
    torch.testing.assert_close(output["prediction"], fallback, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(output["candidate"], fallback, rtol=1e-6, atol=1e-6)
    assert int(output["forward_calls"]) == 3
    assert int(output["adjoint_calls"]) == 3
    assert torch.all(output["raw_correction_ratio"] == 0)


def test_base_correction_receives_training_signal():
    batch, operator = make_dense_case(dtype=torch.float32)
    model = BaseCorrectionCGPDNO(stages=2, correction_width=4)
    lipschitz = 1.08 * operator.weighted_lipschitz(batch, power_iterations=20)
    prediction = model(batch, operator, torch.zeros_like(batch.truth), lipschitz)[
        "prediction"
    ]
    torch.mean((prediction - batch.truth) ** 2).backward()
    gradient = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.correction.parameters()
        if parameter.grad is not None
    )
    assert gradient > 0


def test_pbb_base_zero_init_matches_its_deterministic_fallback():
    batch, operator = make_dense_case(dtype=torch.float32)
    model = PBBBaseCorrectionCGPDNO(
        stages=4,
        correction_width=5,
        certificate_safety=0.95,
        bb_normalized_step_max=1.8,
    )
    lipschitz = 1.10 * operator.weighted_lipschitz(batch, power_iterations=30)
    warm = torch.zeros_like(batch.truth)
    output = model(batch, operator, warm, lipschitz)
    fallback = model.deterministic_fallback(batch, operator, warm, lipschitz)
    torch.testing.assert_close(output["prediction"], fallback, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(output["candidate"], fallback, rtol=1e-6, atol=1e-6)
    assert torch.all(output["acceptance_gate"] == 1)
    assert int(output["forward_calls"]) == 4
    assert int(output["adjoint_calls"]) == 4


def test_pbb_saturation_gate_abstains_to_fallback():
    batch, operator = make_dense_case(dtype=torch.float32)
    model = PBBBaseCorrectionCGPDNO(
        stages=3,
        correction_width=4,
        maximum_correction_ratio=0.2,
        certificate_safety=0.95,
        bb_normalized_step_max=1.8,
        use_saturation_gate=True,
    )
    with torch.no_grad():
        model.correction.output.bias.fill_(5.0)
    model.eval()
    lipschitz = 1.10 * operator.weighted_lipschitz(batch, power_iterations=30)
    warm = torch.zeros_like(batch.truth)
    output = model(batch, operator, warm, lipschitz)
    fallback = model.deterministic_fallback(batch, operator, warm, lipschitz)
    assert torch.all(output["saturation_gate"] == 0)
    assert torch.all(output["acceptance_gate"] == 0)
    torch.testing.assert_close(output["prediction"], fallback, rtol=1e-6, atol=1e-6)


def test_pbb_ensemble_reuses_one_physical_trajectory():
    batch, operator = make_dense_case(dtype=torch.float32)

    class CountingOperator:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.gradient_calls = 0

        def weighted_gradient(self, values, current_batch):
            self.gradient_calls += 1
            return self.wrapped.weighted_gradient(values, current_batch)

    counting = CountingOperator(operator)
    models = [
        PBBBaseCorrectionCGPDNO(stages=4, correction_width=4),
        PBBBaseCorrectionCGPDNO(stages=4, correction_width=4),
        PBBBaseCorrectionCGPDNO(stages=4, correction_width=4),
    ]
    lipschitz = 1.01 * operator.weighted_lipschitz_exact(batch)
    state = shared_ensemble_state(models, batch, counting, lipschitz)
    assert counting.gradient_calls == 4
    prediction, accepted = apply_uncertainty_gate(state, -1.0)
    assert torch.all(~accepted)
    torch.testing.assert_close(prediction, state["fallback"], rtol=0, atol=0)


def test_certificate_rejects_direction_away_from_data():
    gradient = torch.tensor([[[[[1.0, -2.0]]]]])
    bad_direction = -gradient
    alpha, upper, descent = descent_certificate(
        gradient, bad_direction, torch.tensor([3.0]), safety=0.9
    )
    assert float(descent[0]) < 0
    assert float(alpha[0]) == 0
    assert float(upper[0]) == 0


def test_independent_generator_is_finite_and_nonseparable():
    angles = np.array([4.0, 51.0, 99.0, 146.0], dtype=np.float32)
    matrix = build_curved_cone_operator(
        4, 3, angles, path_samples=12, cone_u=0.09, cone_z=0.08, bend=0.04
    )
    assert matrix.shape == (3, 4, 4, 48)
    assert np.all(np.isfinite(matrix))
    first_z_support = np.linalg.norm(matrix[0, 0, 1].reshape(3, 4, 4), axis=(1, 2))
    assert np.count_nonzero(first_z_support > 1e-7) >= 2
    rng = np.random.default_rng(29)
    for family in [
        "expanding_kernel",
        "jet_shear",
        "interacting_fronts",
        "shock_cell",
        "helical_plume",
        "stratified_ignition",
        "wrinkled_flame_sheet",
        "vortex_ring_pair",
        "triple_jet_merger",
        "tilted_flame_brush",
        "pulsed_toroidal_plume",
    ]:
        field = make_reaction_field(family, 6, 4, rng)
        assert field.shape == (4, 6, 6)
        assert np.all(np.isfinite(field))
        assert float(field.min()) >= 0
        assert 0.99 <= float(field.max()) <= 1.01
    clean = np.ones((3, 4, 4), dtype=np.float32)
    noise = correlated_camera_noise(clean, np.full(4, 0.05), rng)
    assert noise.shape == clean.shape
    assert np.all(np.isfinite(noise))
