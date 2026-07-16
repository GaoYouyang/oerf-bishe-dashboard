import numpy as np
import pytest

from demo_t16_operator.limited_query_calibration import (
    BudgetedForwardOracle,
    HybridGateSecantOperator,
    LowRankSecantCorrection,
    NominalLinearOperator,
    QueryBudgetExceeded,
    RayKernelChannelOperator,
    build_gate_design,
    collect_forward_observations,
    fit_channel_gates,
    fit_residual_secant,
    positive_part_residual_signal_fraction,
    ridge_effective_degrees_of_freedom,
    ridge_residual_noise_energy_diagonal,
    ridge_residual_noise_degrees_of_freedom,
    sha256_arrays,
    shift_volume,
    voxel_kernel_offsets,
)


def _random_operator(seed: int = 7, shape: tuple[int, int, int] = (3, 3, 3)):
    rng = np.random.default_rng(seed)
    input_size = int(np.prod(shape))
    output_size = 48
    offsets = voxel_kernel_offsets(1)
    nominal_matrix = rng.normal(scale=0.2, size=(output_size, input_size))
    coefficients = rng.normal(scale=0.03, size=(output_size, len(offsets)))
    nominal = NominalLinearOperator.from_matrix(nominal_matrix)
    return rng, nominal, shape, offsets, coefficients


def test_shift_transpose_relation() -> None:
    rng = np.random.default_rng(13)
    x = rng.normal(size=(3, 4, 5))
    z = rng.normal(size=x.shape)
    offset = (1, -2, 1)
    left = float(np.vdot(shift_volume(x, offset), z))
    right = float(np.vdot(x, shift_volume(z, tuple(-v for v in offset))))
    assert abs(left - right) <= 1e-12


def test_ray_kernel_operator_passes_float64_dot_product() -> None:
    rng, nominal, shape, offsets, coefficients = _random_operator()
    gates = rng.normal(loc=1.0, scale=0.2, size=len(offsets))
    operator = RayKernelChannelOperator(
        nominal,
        input_shape=shape,
        ray_coefficients=coefficients,
        gates=gates,
        offsets=offsets,
    )
    for _ in range(20):
        x = rng.normal(size=operator.input_size)
        z = rng.normal(size=operator.output_size)
        left = float(np.vdot(operator.forward(x), z))
        right = float(np.vdot(x, operator.adjoint(z)))
        defect = abs(left - right) / max(abs(left), abs(right), 1e-15)
        assert defect <= 1e-12


def test_residual_gradient_matches_central_finite_difference() -> None:
    rng, nominal, shape, offsets, coefficients = _random_operator(seed=9)
    operator = RayKernelChannelOperator(
        nominal,
        input_shape=shape,
        ray_coefficients=coefficients,
        gates=rng.normal(loc=1.0, scale=0.15, size=len(offsets)),
        offsets=offsets,
    )
    x = rng.normal(size=operator.input_size)
    direction = rng.normal(size=operator.input_size)
    direction /= np.linalg.norm(direction)
    observation = rng.normal(size=operator.output_size)
    residual = operator.forward(x) - observation
    gradient = operator.adjoint(residual)

    def objective(values: np.ndarray) -> float:
        difference = operator.forward(values) - observation
        return 0.5 * float(np.vdot(difference, difference))

    step = 1e-6
    finite_difference = (
        objective(x + step * direction) - objective(x - step * direction)
    ) / (2.0 * step)
    analytic = float(np.vdot(gradient, direction))
    assert abs(finite_difference - analytic) / max(abs(analytic), 1.0) <= 2e-8


def test_matrix_free_and_materialized_actions_match() -> None:
    rng, nominal, shape, offsets, coefficients = _random_operator(seed=11)
    gates = rng.normal(loc=1.0, scale=0.1, size=len(offsets))
    operator = RayKernelChannelOperator(
        nominal,
        input_shape=shape,
        ray_coefficients=coefficients,
        gates=gates,
        offsets=offsets,
    )
    matrix = operator.materialize()
    x = rng.normal(size=operator.input_size)
    z = rng.normal(size=operator.output_size)
    np.testing.assert_allclose(operator.forward(x), matrix @ x, atol=2e-13)
    np.testing.assert_allclose(operator.adjoint(z), matrix.T @ z, atol=2e-13)


def test_hybrid_gate_secant_has_exact_adjoint_and_no_extra_queries() -> None:
    rng, nominal, shape, offsets, coefficients = _random_operator(seed=17)
    gate_truth = RayKernelChannelOperator(
        nominal,
        input_shape=shape,
        ray_coefficients=coefficients,
        gates=rng.uniform(0.7, 1.3, len(offsets)),
        offsets=offsets,
    )
    unmodelled = rng.normal(scale=0.01, size=(gate_truth.output_size, gate_truth.input_size))
    truth_matrix = gate_truth.materialize() + unmodelled
    probes = rng.normal(size=(gate_truth.input_size, 5))
    probes /= np.linalg.norm(probes, axis=0, keepdims=True)
    oracle = BudgetedForwardOracle(
        lambda x: truth_matrix @ x,
        input_size=gate_truth.input_size,
        output_size=gate_truth.output_size,
        budget=probes.shape[1],
    )
    observations = collect_forward_observations(oracle, probes)
    correction = fit_residual_secant(
        gate_truth,
        probes=probes,
        observations=observations,
        relative_ridge=0.0,
    )
    hybrid = HybridGateSecantOperator(gate_truth, correction)
    assert oracle.query_count == probes.shape[1]
    np.testing.assert_allclose(
        hybrid.materialize() @ probes, observations, rtol=2e-12, atol=2e-12
    )
    x = rng.normal(size=hybrid.input_size)
    z = rng.normal(size=hybrid.output_size)
    np.testing.assert_allclose(
        np.vdot(hybrid.forward(x), z),
        np.vdot(x, hybrid.adjoint(z)),
        rtol=1e-12,
        atol=1e-12,
    )


def test_regularized_secant_shrinks_ill_conditioned_update() -> None:
    probes = np.array([[1.0, 1.0], [0.0, 1e-8], [0.0, 0.0]])
    residuals = np.array([[1.0, -1.0], [0.5, -0.5]])
    unregularized = LowRankSecantCorrection.fit(
        probes, residuals, relative_ridge=0.0
    ).materialize()
    regularized = LowRankSecantCorrection.fit(
        probes, residuals, relative_ridge=1e-2
    ).materialize()
    assert np.linalg.norm(regularized) < np.linalg.norm(unregularized)


def test_positive_part_noise_fraction_switches_only_above_floor() -> None:
    residuals = np.ones((4, 5), dtype=np.float64)
    assert positive_part_residual_signal_fraction(
        residuals, expected_noise_energy=20.0
    ) == 0.0
    fraction = positive_part_residual_signal_fraction(
        residuals, expected_noise_energy=10.0
    )
    assert fraction == pytest.approx(0.5)
    assert positive_part_residual_signal_fraction(
        residuals, expected_noise_energy=0.0
    ) == 1.0
    assert positive_part_residual_signal_fraction(
        residuals,
        expected_noise_energy=20.0,
        residual_noise_degrees_of_freedom=10.0,
    ) == pytest.approx(0.5)


def test_ridge_effective_degrees_of_freedom_is_bounded() -> None:
    rng = np.random.default_rng(43)
    design = rng.normal(size=(80, 7))
    unregularized = ridge_effective_degrees_of_freedom(
        design, relative_ridge=0.0
    )
    regularized = ridge_effective_degrees_of_freedom(
        design, relative_ridge=0.1
    )
    assert unregularized == pytest.approx(7.0)
    assert 0.0 < regularized < unregularized


def test_ridge_residual_noise_df_matches_explicit_hat_matrix() -> None:
    rng = np.random.default_rng(47)
    design = rng.normal(size=(31, 5))
    relative_ridge = 0.07
    gram = design.T @ design
    penalty = relative_ridge * np.trace(gram) / design.shape[1]
    hat = design @ np.linalg.solve(
        gram + penalty * np.eye(design.shape[1]), design.T
    )
    residual = np.eye(design.shape[0]) - hat
    expected = np.trace(residual.T @ residual)
    actual = ridge_residual_noise_degrees_of_freedom(
        design, relative_ridge=relative_ridge
    )
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_unregularized_residual_noise_df_is_n_minus_rank() -> None:
    design = np.array(
        [[1.0, 2.0, 1.0], [0.0, 1.0, 0.0], [2.0, 0.0, 2.0], [1.0, 1.0, 1.0]]
    )
    actual = ridge_residual_noise_degrees_of_freedom(
        design, relative_ridge=0.0
    )
    assert actual == pytest.approx(design.shape[0] - np.linalg.matrix_rank(design))


def test_diagonal_heteroscedastic_residual_noise_matches_explicit_covariance() -> None:
    rng = np.random.default_rng(53)
    design = rng.normal(size=(29, 6))
    variances = np.geomspace(1e-4, 3e-2, design.shape[0])
    relative_ridge = 0.03
    gram = design.T @ design
    penalty = relative_ridge * np.trace(gram) / design.shape[1]
    hat = design @ np.linalg.solve(
        gram + penalty * np.eye(design.shape[1]), design.T
    )
    residual = np.eye(design.shape[0]) - hat
    expected = np.trace(residual @ np.diag(variances) @ residual.T)
    actual = ridge_residual_noise_energy_diagonal(
        design, variances, relative_ridge=relative_ridge
    )
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_diagonal_noise_energy_rejects_invalid_variances() -> None:
    design = np.eye(3)
    with pytest.raises(ValueError, match="one entry"):
        ridge_residual_noise_energy_diagonal(
            design, np.ones(2), relative_ridge=0.0
        )
    with pytest.raises(ValueError, match="non-negative"):
        ridge_residual_noise_energy_diagonal(
            design, np.array([1.0, -1.0, 1.0]), relative_ridge=0.0
        )


def test_ridge_diagnostics_reject_nonfinite_inputs() -> None:
    design = np.eye(3)
    with pytest.raises(ValueError, match="finite"):
        ridge_effective_degrees_of_freedom(
            design, relative_ridge=float("nan")
        )
    design[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        ridge_residual_noise_degrees_of_freedom(
            design, relative_ridge=0.0
        )


def test_budgeted_oracle_refuses_k_plus_one_and_hides_truth_api() -> None:
    matrix = np.arange(20, dtype=np.float64).reshape(4, 5)
    oracle = BudgetedForwardOracle(
        lambda x: matrix @ x, input_size=5, output_size=4, budget=2
    )
    probes = np.eye(5, 2, dtype=np.float64)
    observations = collect_forward_observations(oracle, probes)
    np.testing.assert_allclose(observations, matrix[:, :2])
    assert oracle.query_count == 2
    assert oracle.remaining_queries == 0
    for forbidden in ("operator", "matrix", "adjoint", "truth", "measure_fn"):
        assert not hasattr(oracle, forbidden)
    with pytest.raises(QueryBudgetExceeded):
        oracle.measure(np.ones(5))
    assert oracle.query_count == 2


def test_twenty_seven_channel_gates_recover_from_forward_queries() -> None:
    rng, nominal, shape, offsets, coefficients = _random_operator(seed=23)
    truth_gates = rng.uniform(0.65, 1.35, size=len(offsets))
    truth = RayKernelChannelOperator(
        nominal,
        input_shape=shape,
        ray_coefficients=coefficients,
        gates=truth_gates,
        offsets=offsets,
    )
    probes = rng.normal(size=(truth.input_size, 3))
    probes /= np.linalg.norm(probes, axis=0, keepdims=True)
    oracle = BudgetedForwardOracle(
        truth.forward,
        input_size=truth.input_size,
        output_size=truth.output_size,
        budget=probes.shape[1],
    )
    observations = collect_forward_observations(oracle, probes)
    design, target = build_gate_design(
        nominal,
        input_shape=shape,
        ray_coefficients=coefficients,
        offsets=offsets,
        probes=probes,
        observations=observations,
    )
    recovered = fit_channel_gates(design, target, relative_ridge=0.0)
    assert np.linalg.matrix_rank(design) == len(offsets)
    np.testing.assert_allclose(recovered, truth_gates, rtol=2e-10, atol=2e-10)
    assert oracle.query_count == 3


def test_gate_fit_uses_prior_when_queries_are_uninformative() -> None:
    design = np.zeros((8, 3), dtype=np.float64)
    target = np.zeros(8, dtype=np.float64)
    prior = np.array([0.8, 1.0, 1.2])
    fitted = fit_channel_gates(
        design, target, relative_ridge=1e-3, prior=prior
    )
    np.testing.assert_allclose(fitted, prior)


def test_large_materialization_is_refused() -> None:
    rng, nominal, shape, offsets, coefficients = _random_operator(seed=31)
    operator = RayKernelChannelOperator(
        nominal,
        input_shape=shape,
        ray_coefficients=coefficients,
        gates=np.ones(len(offsets)),
        offsets=offsets,
    )
    with pytest.raises(MemoryError):
        operator.materialize(max_elements=10)


def test_array_hash_changes_with_shape_or_content() -> None:
    values = np.arange(6, dtype=np.float64)
    assert sha256_arrays(values) == sha256_arrays(values.copy())
    assert sha256_arrays(values) != sha256_arrays(values.reshape(2, 3))
    changed = values.copy()
    changed[-1] += 1.0
    assert sha256_arrays(values) != sha256_arrays(changed)


def test_shape_validation_rejects_malformed_calibration_data() -> None:
    _, nominal, shape, offsets, coefficients = _random_operator(seed=37)
    with pytest.raises(ValueError, match="observations"):
        build_gate_design(
            nominal,
            input_shape=shape,
            ray_coefficients=coefficients,
            offsets=offsets,
            probes=np.ones((nominal.input_size, 2)),
            observations=np.ones((nominal.output_size, 3)),
        )
