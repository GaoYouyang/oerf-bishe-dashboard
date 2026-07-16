from __future__ import annotations

import numpy as np

from demo_t16_operator.run_v5u_calibrated_renderer_residual_screening import (
    _feature_matrix,
    calibrated_geometry_features,
)


def test_calibrated_geometry_features_and_design_shapes() -> None:
    vector = np.asarray(
        [5, 33, 61, 89, 117, 145, 173, 0.08, 0.07, 0.05, 0.03],
        dtype=np.float64,
    )
    features = calibrated_geometry_features(vector, 7)
    assert features.shape == (20,)
    design, center, scale = _feature_matrix(
        [vector, vector + 0.001, vector + 0.002], 7
    )
    assert design.shape == (3, 41)
    assert center.shape == scale.shape == (20,)
    assert np.all(np.isfinite(design))
