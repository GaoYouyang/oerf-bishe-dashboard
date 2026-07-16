from __future__ import annotations

import numpy as np
import torch

from demo_t16_operator.run_v6a_ray_kernel_hypernetwork_development import (
    RayKernelHypernetwork,
    predict_hypernetwork_residuals,
)


def test_zero_initialized_hypernetwork_predicts_zero_residual() -> None:
    rng = np.random.default_rng(19)
    features = rng.normal(size=(8, 7)).astype(np.float32)
    basis = rng.normal(size=(8, 11, 5)).astype(np.float32)
    model = RayKernelHypernetwork(7, 12, 2, 5)
    prediction = predict_hypernetwork_residuals(
        model,
        features,
        basis,
        device=torch.device("cpu"),
        batch_rays=4,
    )
    np.testing.assert_allclose(prediction, 0.0, atol=1e-12)
