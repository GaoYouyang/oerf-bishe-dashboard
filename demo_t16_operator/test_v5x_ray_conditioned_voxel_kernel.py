from __future__ import annotations

import numpy as np

from demo_t16_operator.run_v5x_ray_conditioned_voxel_kernel import (
    apply_rowwise_voxel_kernels_to_operator,
    ray_feature_matrix,
)
from demo_t16_operator.run_v5w_clean_aperture_kernel_screening import (
    voxel_kernel_offsets,
)


def test_rowwise_identity_kernel_preserves_operator() -> None:
    rng = np.random.default_rng(13)
    views, depth, detector = 2, 3, 4
    operator = rng.normal(
        size=(views * depth * detector, depth * detector * detector)
    )
    offsets = voxel_kernel_offsets(1)
    kernels = np.zeros((views, depth, detector, len(offsets)))
    kernels[..., offsets.index((0, 0, 0))] = 1.0
    corrected = apply_rowwise_voxel_kernels_to_operator(
        operator, kernels, views, depth, detector, offsets
    )
    np.testing.assert_allclose(corrected, operator, rtol=1e-12, atol=1e-12)


def test_ray_feature_matrix_has_one_row_per_ray() -> None:
    vector = np.asarray(
        [5, 33, 61, 89, 117, 145, 173, 0.08, 0.07, 0.05, 0.03],
        dtype=np.float64,
    )
    design, center, scale = ray_feature_matrix([vector, vector + 0.001], 7, 4, 6)
    assert design.shape == (2 * 7 * 4 * 6, 33)
    assert center.shape == scale.shape == (16,)
    assert np.all(np.isfinite(design))
