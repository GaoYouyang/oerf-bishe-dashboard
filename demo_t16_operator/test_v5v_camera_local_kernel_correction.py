from __future__ import annotations

import numpy as np

from demo_t16_operator.run_v5v_camera_local_kernel_correction import (
    apply_camera_kernels,
    apply_camera_kernels_transpose,
    identity_kernel,
    kernel_offsets,
    view_feature_matrix,
)


def test_identity_kernel_and_exact_measurement_adjoint() -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(3, 4, 6))
    test = rng.normal(size=(3, 4, 6))
    offsets = kernel_offsets(1)
    kernels = np.stack([identity_kernel(offsets)] * 3)
    np.testing.assert_allclose(apply_camera_kernels(values, kernels, offsets), values)
    random_kernels = rng.normal(size=(3, len(offsets)))
    left = np.vdot(apply_camera_kernels(values, random_kernels, offsets), test)
    right = np.vdot(
        values, apply_camera_kernels_transpose(test, random_kernels, offsets)
    )
    np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12)


def test_view_feature_matrix_groups_views_inside_each_rig() -> None:
    vector = np.asarray(
        [5, 33, 61, 89, 117, 145, 173, 0.08, 0.07, 0.05, 0.03],
        dtype=np.float64,
    )
    design, center, scale = view_feature_matrix(
        [vector, vector + 0.001, vector + 0.002], 7
    )
    assert design.shape == (21, 15)
    assert center.shape == scale.shape == (7,)
    assert np.all(np.isfinite(design))
