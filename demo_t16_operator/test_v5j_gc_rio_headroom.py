from __future__ import annotations

import numpy as np

from .run_v5j_gc_rio_headroom_diagnostic import paired_target_projection


def test_paired_target_projection_fits_one_shared_supported_field() -> None:
    first = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    second = np.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    field = paired_target_projection(
        [first, second],
        [np.asarray([2.0, 3.0]), np.asarray([3.0, 2.0])],
        np.asarray([1.0, 1.0, 0.0]),
        relative_ridge=1e-10,
    )
    np.testing.assert_allclose(field, [2.0, 3.0, 0.0], atol=1e-6)
