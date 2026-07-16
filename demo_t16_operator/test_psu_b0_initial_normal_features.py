from __future__ import annotations

import torch

from demo_t16_operator.psu_b0_initial_normal_features import (
    initial_normal_spectral_features,
    measurement_metadata_features,
)


def test_measurement_metadata_features_are_finite() -> None:
    observation = torch.randn(3, 4 * 9, 2)
    sigma = torch.full((3, 4), 0.2)
    mask = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 1.0],
        ]
    )
    features, names = measurement_metadata_features(
        observation,
        sigma_by_view=sigma,
        view_mask=mask,
        rays_per_view=9,
    )
    assert features.shape == (3, len(names))
    assert len(names) == 21
    assert torch.all(torch.isfinite(features))


def test_initial_normal_features_have_stable_schema() -> None:
    normal = torch.randn(4, 1, 12, 10, 8)
    features, names = initial_normal_spectral_features(normal)
    assert features.shape == (4, len(names))
    assert len(names) == 44
    assert torch.all(torch.isfinite(features))


def test_normalized_shape_features_survive_amplitude_scaling() -> None:
    normal = torch.randn(2, 1, 8, 8, 8)
    scaled = 7.0 * normal
    first, names = initial_normal_spectral_features(normal)
    second, _ = initial_normal_spectral_features(scaled)
    amplitude_index = names.index("log_initial_normal_rms")
    keep = [index for index in range(len(names)) if index != amplitude_index]
    assert torch.allclose(first[:, keep], second[:, keep], atol=2e-5, rtol=2e-5)
    assert torch.allclose(
        second[:, amplitude_index] - first[:, amplitude_index],
        torch.full((2,), torch.log(torch.tensor(7.0))),
        atol=2e-5,
        rtol=2e-5,
    )
