from __future__ import annotations

import numpy as np
import pytest

from demo_t16_operator.dual_regularization import (
    error_reduction_percent,
    outer_only_route,
    refit_fixed_radius_with_method,
)
from demo_t16_operator.test_decoupled_complexity import (
    observations_for,
    patterned_bank,
)


def test_refit_uses_frozen_radius_and_requested_method() -> None:
    bank, _ = patterned_bank()
    observations = observations_for(bank, 1)
    result = refit_fixed_radius_with_method(
        1,
        "gcv",
        bank,
        observations,
        [np.full(4, 0.01), np.full(4, 0.01)],
        [0, 1, 2, 3],
        np.ones(2, dtype=bool),
        [1e-8, 1e-4, 1e-2],
    )
    assert result.radius_index == 1
    assert result.method == "gcv"
    assert result.choice.method == "gcv"
    assert len(result.refit.fits) == 2


def test_error_reduction_positive_means_better() -> None:
    assert error_reduction_percent(0.8, 1.0) == pytest.approx(20.0)
    assert error_reduction_percent(1.2, 1.0) == pytest.approx(-20.0)


def test_outer_route_uses_only_changed_flag_and_outer_reductions() -> None:
    assert outer_only_route(
        True, [0.0, 2.0], minimum_per_view_reduction_percent=0.0
    )
    assert not outer_only_route(
        False, [100.0, 100.0], minimum_per_view_reduction_percent=0.0
    )
    assert not outer_only_route(
        True, [2.0, -0.1], minimum_per_view_reduction_percent=0.0
    )
