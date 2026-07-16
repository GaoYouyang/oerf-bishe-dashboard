from __future__ import annotations

import numpy as np

from demo_t16_operator.run_v5w_clean_aperture_kernel_screening import (
    apply_voxel_kernel,
    apply_voxel_kernels_to_operator,
    voxel_kernel_offsets,
)


def test_right_voxel_kernel_composition_matches_operator_product() -> None:
    rng = np.random.default_rng(11)
    views, depth, detector = 2, 3, 4
    voxels = depth * detector * detector
    operator = rng.normal(size=(views * depth * detector, voxels))
    field = rng.normal(size=(depth, detector, detector))
    offsets = voxel_kernel_offsets(1)
    kernels = rng.normal(size=(views, len(offsets)))
    corrected = apply_voxel_kernels_to_operator(
        operator, kernels, views, depth, detector, offsets
    ).reshape(views, depth * detector, voxels)
    original = operator.reshape(views, depth * detector, voxels)
    for view_index in range(views):
        expected = original[view_index] @ apply_voxel_kernel(
            field, kernels[view_index], offsets
        ).reshape(-1)
        actual = corrected[view_index] @ field.reshape(-1)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_zero_radius_voxel_kernel_is_identity_shape() -> None:
    offsets = voxel_kernel_offsets(0)
    field = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    np.testing.assert_allclose(apply_voxel_kernel(field, np.ones(1), offsets), field)
