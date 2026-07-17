from __future__ import annotations

import torch

from demo_t16_operator.jacru_n1_5_approximation_error import (
    fit_standardized_ridge,
    pca_oracle_prediction,
    visible_feature_blocks,
)
from demo_t16_operator.jacru_synthetic_fixture import (
    JACRUSyntheticFixtureConfig,
    build_jacru_synthetic_case,
)


def _case():
    config = JACRUSyntheticFixtureConfig(
        detector_shape=(5, 5),
        samples_per_ray=16,
        enable_noise=False,
        enable_camera_bias=False,
    )
    return build_jacru_synthetic_case(
        family="single_interface", split="train", base_seed=1101, config=config
    )


def test_standardized_ridge_recovers_linear_target() -> None:
    features = torch.tensor(
        [[-2.0, 0.0], [-1.0, 1.0], [0.0, 2.0], [1.0, 3.0], [2.0, 4.0]],
        dtype=torch.float64,
    )
    target = 1.5 + 2.0 * features[:, 0] - 0.25 * features[:, 1]
    model = fit_standardized_ridge(
        features, target, feature_names=("first", "second"), alpha=1e-12
    )
    assert torch.allclose(model.predict(features), target, atol=1e-8, rtol=1e-8)


def test_visible_features_have_expected_rows_and_are_finite() -> None:
    case = _case()
    observation = case.evaluation.clean_observations_uv[0]
    warm = 0.9 * observation
    blocks = visible_feature_blocks(
        geometry=case.inference.geometry,
        observation_uv=observation,
        warm_projection_uv=warm,
    )
    assert tuple(blocks) == (
        "geometry_only",
        "geometry_observation",
        "geometry_signal",
        "curvature_visible",
    )
    previous_columns = 0
    for names, values in blocks.values():
        assert values.shape[0] == observation.numel()
        assert values.shape[1] == len(names)
        assert values.shape[1] > previous_columns
        assert bool(torch.all(torch.isfinite(values)))
        previous_columns = values.shape[1]


def test_pca_oracle_residual_is_monotone_with_rank() -> None:
    generator = torch.Generator().manual_seed(19)
    training = torch.randn((8, 12), generator=generator, dtype=torch.float64)
    target = torch.randn((12,), generator=generator, dtype=torch.float64)
    residuals = []
    for rank in (0, 1, 2, 4, 7):
        prediction = pca_oracle_prediction(
            training_vectors=training, target_vector=target, rank=rank
        )
        residuals.append(float(torch.linalg.vector_norm(target - prediction)))
    assert all(
        later <= earlier + 1e-12
        for earlier, later in zip(residuals, residuals[1:])
    )


def test_pca_rank_zero_is_training_mean() -> None:
    training = torch.tensor([[0.0, 2.0], [2.0, 4.0], [4.0, 6.0]])
    prediction = pca_oracle_prediction(
        training_vectors=training, target_vector=torch.tensor([9.0, -3.0]), rank=0
    )
    assert torch.allclose(prediction, torch.tensor([2.0, 4.0], dtype=torch.float64))
