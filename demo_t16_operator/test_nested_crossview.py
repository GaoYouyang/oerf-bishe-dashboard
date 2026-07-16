from __future__ import annotations

import numpy as np
import pytest

from demo_t16_operator.nested_crossview import (
    refit_scaled_selection,
    select_radius_kappa_crossview,
    whitened_per_view_rms,
)


def patterned_bank(scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    radii = np.array([0.0, 0.5, 1.0])
    patterns = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.4, 2.0, 2.8],
            [1.0, 2.0, 4.0, 8.0],
        ],
        dtype=float,
    )
    # [radius, detector_z, view, detector_x, voxel]
    return (scale * patterns[:, None, :, None, None]), radii


def observations_for(bank: np.ndarray, index: int) -> list[np.ndarray]:
    fields = [0.4, 0.9, 1.3, 1.8]
    return [bank[index, ..., 0] * field for field in fields]


def test_crossview_recovers_shared_radius_without_outer_camera() -> None:
    bank, radii = patterned_bank()
    observations = observations_for(bank, 1)
    sigma = [np.ones(4) * 0.02 for _ in observations]
    selection = select_radius_kappa_crossview(
        bank,
        radii,
        observations,
        sigma,
        [0, 1, 2],
        np.ones(1, dtype=bool),
        [1e-8, 1e-3, 1.0],
    )
    assert selection.selected.radius == pytest.approx(0.5)
    assert selection.selected.kappa == pytest.approx(1e-8)
    assert selection.fold_score_deletion_radius_stability_fraction == pytest.approx(1.0)
    assert selection.relative_score_margin > 0.0
    assert selection.relative_radius_margin > 0.0


def test_candidate_specific_normal_scale_makes_selection_scale_invariant() -> None:
    bank, radii = patterned_bank()
    observations = observations_for(bank, 1)
    sigma = [np.ones(4) * 0.03 for _ in observations]
    original = select_radius_kappa_crossview(
        bank,
        radii,
        observations,
        sigma,
        [0, 1, 2],
        np.ones(1, dtype=bool),
        [1e-6, 1e-3],
    )
    scaled_bank = bank * 11.0
    scaled = select_radius_kappa_crossview(
        scaled_bank,
        radii,
        observations,
        sigma,
        [0, 1, 2],
        np.ones(1, dtype=bool),
        [1e-6, 1e-3],
    )
    assert scaled.selected.radius == pytest.approx(original.selected.radius)
    assert scaled.selected.kappa == pytest.approx(original.selected.kappa)
    assert scaled.selected.mean_validation_mse == pytest.approx(
        original.selected.mean_validation_mse
    )
    assert scaled.selected.median_effective_lambda == pytest.approx(
        original.selected.median_effective_lambda * 11.0**2
    )


def test_poisoned_noninner_camera_cannot_change_selection_or_refit() -> None:
    bank, radii = patterned_bank()
    observations = observations_for(bank, 1)
    poisoned = [value.copy() for value in observations]
    for value in poisoned:
        value[:, 3] += 1_000_000.0
    sigma = [np.ones(4) * 0.02 for _ in observations]
    clean_selection = select_radius_kappa_crossview(
        bank,
        radii,
        observations,
        sigma,
        [0, 1, 2],
        np.ones(1, dtype=bool),
        [1e-6, 1e-3],
    )
    poisoned_selection = select_radius_kappa_crossview(
        bank,
        radii,
        poisoned,
        sigma,
        [0, 1, 2],
        np.ones(1, dtype=bool),
        [1e-6, 1e-3],
    )
    assert poisoned_selection.selected_candidate_index == clean_selection.selected_candidate_index
    assert [item.mean_validation_mse for item in poisoned_selection.candidates] == pytest.approx(
        [item.mean_validation_mse for item in clean_selection.candidates]
    )
    clean_refit = refit_scaled_selection(
        clean_selection,
        bank,
        observations,
        sigma,
        [0, 1, 2],
        np.ones(1, dtype=bool),
    )
    poisoned_refit = refit_scaled_selection(
        poisoned_selection,
        bank,
        poisoned,
        sigma,
        [0, 1, 2],
        np.ones(1, dtype=bool),
    )
    for clean, changed in zip(clean_refit.fits, poisoned_refit.fits, strict=True):
        assert changed.field == pytest.approx(clean.field)


def test_per_view_rms_keeps_outer_cameras_separate() -> None:
    bank, _ = patterned_bank()
    field = np.array([0.7])
    observation = np.einsum("dvnp,p->dvn", bank[1], field)
    corrupted = observation.copy()
    corrupted[:, 3] += 10.0
    values = whitened_per_view_rms(
        bank[1], field, corrupted, np.ones(4), [2, 3]
    )
    assert values[0] == pytest.approx(0.0)
    assert values[1] == pytest.approx(10.0)


def test_noninner_operator_observation_and_sigma_are_physically_isolated() -> None:
    bank, radii = patterned_bank()
    observations = observations_for(bank, 1)
    sigma = [np.ones(4) * 0.02 for _ in observations]
    clean = select_radius_kappa_crossview(
        bank,
        radii,
        observations,
        sigma,
        [0, 1, 2],
        np.ones(1, dtype=bool),
        [1e-6, 1e-3],
    )
    poisoned_bank = bank.copy()
    poisoned_bank[:, :, 3] = np.nan
    poisoned_observations = [value.copy() for value in observations]
    poisoned_sigma = [value.copy() for value in sigma]
    for value, scale in zip(poisoned_observations, poisoned_sigma, strict=True):
        value[:, 3] = np.nan
        scale[3] = 0.0
    changed = select_radius_kappa_crossview(
        poisoned_bank,
        radii,
        poisoned_observations,
        poisoned_sigma,
        [0, 1, 2],
        np.ones(1, dtype=bool),
        [1e-3, 1e-6],
    )
    assert changed.selected.radius == pytest.approx(clean.selected.radius)
    assert changed.selected.kappa == pytest.approx(clean.selected.kappa)
    assert [item.mean_validation_mse for item in changed.candidates] == pytest.approx(
        [item.mean_validation_mse for item in clean.candidates]
    )


def test_float_support_is_rejected_instead_of_silently_cast() -> None:
    bank, radii = patterned_bank()
    observations = observations_for(bank, 1)
    sigma = [np.ones(4) for _ in observations]
    with pytest.raises(ValueError, match="explicit boolean"):
        select_radius_kappa_crossview(
            bank, radii, observations, sigma, [0, 1], np.ones(1), [1e-3]
        )


@pytest.mark.parametrize(
    ("views", "kappas", "message"),
    [
        ([0], [1e-3], "at least two"),
        ([0, 0], [1e-3], "unique"),
        ([0, 1], [0.0], "strictly positive"),
        ([0, 1], [1e-3, 1e-3], "unique"),
    ],
)
def test_invalid_nested_selection_contract_is_rejected(
    views: list[int], kappas: list[float], message: str
) -> None:
    bank, radii = patterned_bank()
    observations = observations_for(bank, 1)
    sigma = [np.ones(4) for _ in observations]
    with pytest.raises(ValueError, match=message):
        select_radius_kappa_crossview(
            bank, radii, observations, sigma, views, np.ones(1), kappas
        )
