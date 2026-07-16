import numpy as np
import pytest

from demo_t16_operator.decoupled_complexity import (
    build_decoupled_surface,
    choose_complexity,
    refit_decoupled_selection,
    ridge_diagnostics,
    select_radius_decoupled,
    select_radius_from_surface,
)


def patterned_bank(scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    radii = np.asarray([0.0, 0.5, 1.0], dtype=float)
    bank = np.zeros((3, 1, 4, 2, 2), dtype=float)
    patterns = (
        np.asarray([[1.0, 0.0], [0.0, 0.25]]),
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        np.asarray([[0.25, 0.0], [0.0, 1.0]]),
    )
    for radius_index, pattern in enumerate(patterns):
        for view in range(4):
            bank[radius_index, 0, view] = scale * (1.0 + 0.1 * view) * pattern
    return bank, radii


def observations_for(bank: np.ndarray, index: int) -> list[np.ndarray]:
    fields = [np.asarray([1.0, 0.4]), np.asarray([0.7, 1.1])]
    return [np.einsum("dvcp,p->dvc", bank[index], field) for field in fields]


def test_exact_ridge_diagnostics_match_hat_matrix_trace() -> None:
    operator = np.zeros((1, 2, 2, 2), dtype=float)
    operator[0, 0] = np.eye(2)
    operator[0, 1] = np.eye(2)
    observation = np.asarray([[[1.0, 2.0], [1.0, 2.0]]])
    support = np.ones(2, dtype=bool)
    result = ridge_diagnostics(
        operator,
        observation,
        np.ones(2),
        [0, 1],
        support,
        0.5,
    )

    assert result.effective_lambda == pytest.approx(1.0)
    assert result.effective_degrees_of_freedom == pytest.approx(4.0 / 3.0)
    assert result.squared_hat_trace == pytest.approx(8.0 / 9.0)
    assert result.residual_noise_degrees_of_freedom == pytest.approx(20.0 / 9.0)
    assert result.degrees_of_freedom_corrected_discrepancy == pytest.approx(0.5)
    assert result.effective_degrees_of_freedom_fraction == pytest.approx(2.0 / 3.0)
    assert result.generalized_cross_validation > result.whitened_discrepancy


def test_complexity_choice_cannot_read_an_excluded_camera() -> None:
    bank, _ = patterned_bank()
    observations = observations_for(bank, 1)
    sigma = [np.ones(4), np.ones(4)]
    support = np.ones(2, dtype=bool)
    clean = choose_complexity(
        "gcv", bank[1], observations, sigma, [1, 2, 3], support, [1e-4, 1e-2]
    )
    poisoned = [value.copy() for value in observations]
    for value in poisoned:
        value[:, 0, :] += 1e6
    changed = choose_complexity(
        "gcv", bank[1], poisoned, sigma, [1, 2, 3], support, [1e-4, 1e-2]
    )

    assert changed == clean


@pytest.mark.parametrize(
    "method",
    [
        "gcv",
        "upre",
        "morozov",
        "df_corrected_morozov",
        "equal_df",
        "nested_cv",
    ],
)
def test_decoupled_selector_and_refit_are_finite(method: str) -> None:
    bank, radii = patterned_bank()
    observations = observations_for(bank, 1)
    sigma = [np.full(4, 0.01), np.full(4, 0.01)]
    support = np.ones(2, dtype=bool)
    selection = select_radius_decoupled(
        method,
        bank,
        radii,
        observations,
        sigma,
        [0, 1, 2, 3],
        support,
        [1e-8, 1e-4, 1e-2],
        discrepancy_target=1.0,
        effective_degrees_of_freedom_target=0.75,
    )
    refit = refit_decoupled_selection(
        selection,
        bank,
        observations,
        sigma,
        [0, 1, 2, 3],
        support,
        [1e-8, 1e-4, 1e-2],
    )

    assert selection.selected.radius in radii
    assert np.isfinite(selection.relative_radius_margin)
    assert len(refit.refit.fits) == 2
    assert 0.0 < refit.choice.kappa <= 1e-2


def test_gcv_recovers_the_patterned_bank_radius() -> None:
    bank, radii = patterned_bank()
    observations = observations_for(bank, 1)
    sigma = [np.full(4, 0.01), np.full(4, 0.01)]
    selection = select_radius_decoupled(
        "gcv",
        bank,
        radii,
        observations,
        sigma,
        [0, 1, 2, 3],
        np.ones(2, dtype=bool),
        [1e-8, 1e-4, 1e-2],
    )

    assert selection.selected.radius == pytest.approx(0.5)


def test_one_surface_is_shared_without_changing_method_results() -> None:
    bank, radii = patterned_bank()
    observations = observations_for(bank, 1)
    sigma = [np.full(4, 0.01), np.full(4, 0.01)]
    support = np.ones(2, dtype=bool)
    kappas = [1e-8, 1e-4, 1e-2]
    surface = build_decoupled_surface(
        bank,
        radii,
        observations,
        sigma,
        [0, 1, 2, 3],
        support,
        kappas,
        include_nested_cross_validation=True,
    )

    for method in [
        "gcv",
        "upre",
        "morozov",
        "df_corrected_morozov",
        "equal_df",
        "nested_cv",
    ]:
        shared = select_radius_from_surface(method, surface)
        direct = select_radius_decoupled(
            method,
            bank,
            radii,
            observations,
            sigma,
            [0, 1, 2, 3],
            support,
            kappas,
        )
        assert shared == direct


def test_nested_cv_rejects_a_surface_without_nested_scores() -> None:
    bank, radii = patterned_bank()
    observations = observations_for(bank, 1)
    surface = build_decoupled_surface(
        bank,
        radii,
        observations,
        [np.ones(4), np.ones(4)],
        [0, 1, 2, 3],
        np.ones(2, dtype=bool),
        [1e-4, 1e-2],
        include_nested_cross_validation=False,
    )

    with pytest.raises(ValueError, match="does not contain nested"):
        select_radius_from_surface("nested_cv", surface)


def test_invalid_complexity_targets_are_rejected() -> None:
    bank, _ = patterned_bank()
    observations = observations_for(bank, 1)
    sigma = [np.ones(4), np.ones(4)]
    support = np.ones(2, dtype=bool)

    with pytest.raises(ValueError, match="discrepancy_target"):
        choose_complexity(
            "morozov",
            bank[1],
            observations,
            sigma,
            [0, 1, 2],
            support,
            [1e-4],
            discrepancy_target=0.0,
        )
    with pytest.raises(ValueError, match="degrees_of_freedom_target"):
        choose_complexity(
            "equal_df",
            bank[1],
            observations,
            sigma,
            [0, 1, 2],
            support,
            [1e-4],
            effective_degrees_of_freedom_target=1.0,
        )
