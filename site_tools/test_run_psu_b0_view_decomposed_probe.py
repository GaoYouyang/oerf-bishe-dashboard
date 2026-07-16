from __future__ import annotations

import numpy as np
import pytest

from site_tools.run_psu_b0_view_decomposed_probe import (
    leave_group_out_scores,
)


def test_leave_group_out_scores_never_fit_on_held_group() -> None:
    features = np.asarray(
        [[-2.0], [-1.0], [1.0], [2.0], [4.0], [5.0]],
        dtype=np.float64,
    )
    targets = np.concatenate((features, -features), axis=1)
    groups = np.asarray(["a", "a", "b", "b", "c", "c"])

    scores = leave_group_out_scores(
        features,
        targets,
        groups,
        regularization=1.0,
    )

    assert scores.shape == targets.shape
    assert np.all(np.isfinite(scores))
    assert np.all(scores[:, 0] * scores[:, 1] <= 0.0)


def test_leave_group_out_scores_rejects_misaligned_rows() -> None:
    with pytest.raises(ValueError, match="align"):
        leave_group_out_scores(
            np.zeros((3, 2)),
            np.zeros((2, 1)),
            np.asarray([0, 1, 2]),
            regularization=1.0,
        )
