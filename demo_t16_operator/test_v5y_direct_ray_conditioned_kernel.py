from __future__ import annotations

import numpy as np
import torch

from demo_t16_operator.run_v5y_direct_ray_conditioned_kernel import (
    build_ray_basis_and_targets,
    predict_ray_residuals,
)
from demo_t16_operator.run_v5w_clean_aperture_kernel_screening import (
    voxel_kernel_offsets,
)


def test_ray_basis_alignment_and_zero_prediction() -> None:
    rng = np.random.default_rng(17)
    views, depth, detector = 2, 3, 4
    voxels = depth * detector * detector
    thin = rng.normal(size=(1, views * depth * detector, voxels))
    finite = thin + 0.1 * rng.normal(size=thin.shape)
    offsets = voxel_kernel_offsets(1)
    basis, targets = build_ray_basis_and_targets(
        thin, finite, views, depth, detector, offsets
    )
    assert basis.shape == (views * depth * detector, voxels, len(offsets))
    assert targets.shape == (views * depth * detector, voxels)
    features = rng.normal(size=(len(basis), 5))
    coefficients = np.zeros((5, len(offsets)))
    prediction = predict_ray_residuals(
        features, basis, coefficients, device=torch.device("cpu")
    )
    np.testing.assert_allclose(prediction, 0.0)
