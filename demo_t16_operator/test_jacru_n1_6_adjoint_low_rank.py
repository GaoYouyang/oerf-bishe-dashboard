from __future__ import annotations

import torch

from demo_t16_operator.jacru_n1_6_adjoint_low_rank import (
    adjoint_optimal_coefficients,
    coefficient_abs_limits,
    fail_closed_predict,
    fit_measurement_pca,
    fit_multioutput_ridge,
    measurement_optimal_coefficients,
    standardized_feature_limit,
    visible_case_feature_blocks,
)
from demo_t16_operator.jacru_synthetic_fixture import (
    JACRUSyntheticFixtureConfig,
    build_jacru_synthetic_case,
)


def test_multioutput_ridge_recovers_regularized_linear_map() -> None:
    generator = torch.Generator().manual_seed(12)
    features = torch.randn((40, 4), generator=generator, dtype=torch.float64)
    exact_weights = torch.tensor(
        [[1.0, -0.5], [0.2, 0.7], [-0.4, 0.1], [0.8, -0.2]],
        dtype=torch.float64,
    )
    targets = features @ exact_weights + torch.tensor([0.3, -0.8])
    model = fit_multioutput_ridge(
        features,
        targets,
        feature_names=("a", "b", "c", "d"),
        alpha=1e-10,
    )
    assert torch.allclose(model.predict(features), targets, atol=1e-8, rtol=1e-8)


def test_measurement_basis_is_centered_and_orthonormal() -> None:
    training = torch.tensor(
        [[1.0, 0.0, 2.0], [0.0, 1.0, 1.0], [2.0, 1.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=torch.float64,
    )
    basis = fit_measurement_pca(training, rank=2)
    assert basis.mean.shape == (3,)
    assert basis.vectors.shape == (2, 3)
    assert torch.allclose(
        basis.vectors @ basis.vectors.T,
        torch.eye(2, dtype=torch.float64),
        atol=1e-12,
    )


def test_adjoint_weighted_coefficients_improve_adjoint_residual() -> None:
    training = torch.tensor(
        [[-1.0, 0.0, -1.0], [1.0, 0.0, 1.0], [-0.5, 0.0, -0.5], [0.5, 0.0, 0.5]],
        dtype=torch.float64,
    )
    basis = fit_measurement_pca(training, rank=1)
    target = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)
    matrix = torch.tensor([[10.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)

    def adjoint(values: torch.Tensor) -> torch.Tensor:
        return matrix @ values.reshape(-1)

    measurement = measurement_optimal_coefficients(basis, target)
    weighted = adjoint_optimal_coefficients(
        basis,
        target,
        observation_shape=(3,),
        adjoint=adjoint,
        l2=1e-12,
    )
    measurement_residual = torch.linalg.vector_norm(
        adjoint(target - basis.synthesize(measurement))
    )
    weighted_residual = torch.linalg.vector_norm(
        adjoint(target - basis.synthesize(weighted.coefficients))
    )
    assert weighted_residual < measurement_residual
    assert weighted.evaluator_adjoint_calls == 2


def test_visible_features_are_finite_and_have_nested_contract() -> None:
    config = JACRUSyntheticFixtureConfig(
        detector_shape=(5, 5),
        samples_per_ray=16,
        enable_noise=False,
        enable_camera_bias=False,
    )
    case = build_jacru_synthetic_case(
        family="single_interface", split="train", base_seed=1101, config=config
    )
    observation = case.evaluation.clean_observations_uv[0]
    field = case.evaluation.truth_volume[0, 0]
    projection = case.inference.operator(field[None, None])[0]
    blocks = visible_case_feature_blocks(
        geometry=case.inference.geometry,
        observation_uv=observation,
        warm_projection_uv=projection,
        warm_field=field,
    )
    assert set(blocks) == {"summary", "camera", "camera_geometry"}
    summary_names, summary = blocks["summary"]
    camera_names, camera = blocks["camera"]
    full_names, full = blocks["camera_geometry"]
    assert tuple(camera_names[: len(summary_names)]) == summary_names
    assert tuple(full_names[: len(camera_names)]) == camera_names
    assert summary.ndim == camera.ndim == full.ndim == 1
    assert bool(torch.all(torch.isfinite(full)))


def test_fail_closed_prediction_clips_then_falls_back_outside_feature_envelope() -> None:
    features = torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float64)
    targets = torch.tensor([[-2.0], [0.0], [2.0]], dtype=torch.float64)
    model = fit_multioutput_ridge(
        features, targets, feature_names=("signal",), alpha=1e-8
    )
    basis = fit_measurement_pca(
        torch.tensor([[-1.0, -1.0], [1.0, 1.0], [-0.5, -0.5], [0.5, 0.5]]),
        rank=1,
    )
    limits = coefficient_abs_limits(targets, quantile=1.0, multiplier=1.0)
    feature_limit = standardized_feature_limit(
        model, features, quantile=1.0, multiplier=1.0
    )
    accepted = fail_closed_predict(
        model=model,
        features=torch.tensor([[0.25]], dtype=torch.float64),
        basis=basis,
        coefficient_limits=limits,
        feature_max_abs_z_limit=feature_limit,
        residual_rms_limit=10.0,
    )
    rejected = fail_closed_predict(
        model=model,
        features=torch.tensor([[5.0]], dtype=torch.float64),
        basis=basis,
        coefficient_limits=limits,
        feature_max_abs_z_limit=feature_limit,
        residual_rms_limit=10.0,
    )
    assert accepted.fallback is False
    assert rejected.fallback is True
    assert rejected.fallback_reason == "feature_envelope"
    assert torch.count_nonzero(rejected.residual) == 0
