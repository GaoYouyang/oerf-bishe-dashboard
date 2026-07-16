from __future__ import annotations

import numpy as np
import pytest

from demo_t16_operator.rig_shared_profile import (
    apply_metadata_prior,
    fit_support_ridge,
    operator_radius_derivative,
    profile_fisher_scalar,
    profile_shared_radius,
    whitened_normal_mean_diagonal,
    whitened_view_rms,
)
from demo_t16_operator.run_v5b_rig_shared_profile_pilot import (
    support_mask_from_config,
    validate_forward_model_gate,
)


def small_bank() -> tuple[np.ndarray, np.ndarray]:
    radii = np.array([0.0, 0.5, 1.0])
    base = np.array(
        [
            [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [1.0, -1.0]]],
        ]
    )
    direction = np.array(
        [
            [[[0.0, 1.0], [1.0, 0.0]], [[0.4, -0.2], [-0.3, 0.5]]],
        ]
    )
    return np.stack([base + radius * direction for radius in radii]), radii


def test_shared_profile_recovers_common_radius() -> None:
    bank, radii = small_bank()
    fields = [np.array([1.0, 0.3]), np.array([0.2, 1.2]), np.array([0.8, -0.4])]
    observations = [np.einsum("dvnp,p->dvn", bank[1], field) for field in fields]
    sigma = [np.ones(2) * 0.05 for _ in fields]
    selection = profile_shared_radius(
        bank,
        radii,
        observations,
        sigma,
        [0, 1],
        np.ones(2, dtype=bool),
        1e-6,
    )
    assert selection.selected.radius == pytest.approx(0.5)
    assert selection.selected.data_score < selection.candidates[0].data_score
    assert selection.selected.data_score < selection.candidates[2].data_score


def test_metadata_penalty_is_explicit_and_can_change_selection() -> None:
    bank, radii = small_bank()
    observation = np.zeros(bank.shape[1:4])
    unanchored = profile_shared_radius(
        bank, radii, [observation], [np.ones(2)], [0, 1], np.ones(2), 0.1
    )
    anchored = profile_shared_radius(
        bank,
        radii,
        [observation],
        [np.ones(2)],
        [0, 1],
        np.ones(2),
        0.1,
        metadata_radius=1.0,
        metadata_sigma=0.1,
        metadata_weight=1.0,
    )
    assert unanchored.selected.radius == pytest.approx(0.0)
    assert anchored.selected.radius == pytest.approx(1.0)
    rescored = apply_metadata_prior(unanchored, 1.0, 0.1, 1.0)
    assert rescored.selected.radius == pytest.approx(1.0)
    assert [value.data_score for value in rescored.candidates] == pytest.approx(
        [value.data_score for value in unanchored.candidates]
    )


def test_profile_fisher_returns_bounded_information() -> None:
    bank, radii = small_bank()
    field = np.array([0.7, 0.2])
    observation = np.einsum("dvnp,p->dvn", bank[1], field)
    fit = fit_support_ridge(
        bank[1], observation, np.ones(2), [0, 1], np.ones(2), 1e-6
    )
    derivative = operator_radius_derivative(bank, radii, 1)
    result = profile_fisher_scalar(
        bank[1],
        derivative,
        fit,
        observation,
        np.ones(2),
        [0, 1],
        np.ones(2),
        1e-6,
    )
    assert result.raw_parameter_energy > 0.0
    assert 0.0 <= result.retained_fraction <= 1.0
    assert np.isfinite(result.approximate_standard_error)


def test_profile_fisher_rejects_direction_absorbed_by_field_nuisance() -> None:
    bank, _ = small_bank()
    operator = bank[0]
    mixing = np.array([[0.2, -0.1], [0.05, 0.3]])
    derivative = (operator.reshape(-1, 2) @ mixing).reshape(operator.shape)
    field = np.array([0.7, 0.2])
    observation = np.einsum("dvnp,p->dvn", operator, field)
    fit = fit_support_ridge(
        operator, observation, np.ones(2), [0, 1], np.ones(2), 1e-10
    )
    result = profile_fisher_scalar(
        operator,
        derivative,
        fit,
        observation,
        np.ones(2),
        [0, 1],
        np.ones(2),
        1e-10,
    )
    assert result.raw_parameter_energy > 0.0
    assert result.retained_fraction < 1e-8


def test_view_rms_uses_only_requested_camera() -> None:
    bank, _ = small_bank()
    field = np.array([1.0, 0.5])
    observation = np.einsum("dvnp,p->dvn", bank[0], field)
    corrupted = observation.copy()
    corrupted[:, 1] += 100.0
    assert whitened_view_rms(bank[0], field, corrupted, np.ones(2), [0]) == pytest.approx(0.0)
    assert whitened_view_rms(bank[0], field, corrupted, np.ones(2), [1]) > 10.0


def test_profile_fit_cannot_see_poisoned_nonfit_camera() -> None:
    bank, radii = small_bank()
    field = np.array([1.0, 0.5])
    observation = np.einsum("dvnp,p->dvn", bank[1], field)
    poisoned = observation.copy()
    poisoned[:, 1] += 1_000_000.0
    clean_selection = profile_shared_radius(
        bank, radii, [observation], [np.ones(2)], [0], np.ones(2), 0.1
    )
    poisoned_selection = profile_shared_radius(
        bank, radii, [poisoned], [np.ones(2)], [0], np.ones(2), 0.1
    )
    assert poisoned_selection.selected_index == clean_selection.selected_index
    assert [item.total_score for item in poisoned_selection.candidates] == pytest.approx(
        [item.total_score for item in clean_selection.candidates]
    )
    for clean, changed in zip(
        clean_selection.selected.fits,
        poisoned_selection.selected.fits,
        strict=True,
    ):
        assert changed.field == pytest.approx(clean.field)


def test_invalid_sigma_is_rejected() -> None:
    bank, _ = small_bank()
    observation = np.zeros(bank.shape[1:4])
    with pytest.raises(ValueError, match="strictly positive"):
        fit_support_ridge(
            bank[0], observation, np.array([1.0, 0.0]), [0], np.ones(2), 0.1
        )


def test_normal_scale_is_positive_and_noise_aware() -> None:
    bank, _ = small_bank()
    observation = np.zeros(bank.shape[1:4])
    unit = whitened_normal_mean_diagonal(
        bank[0], observation, np.ones(2), [0, 1], np.ones(2)
    )
    doubled_noise = whitened_normal_mean_diagonal(
        bank[0], observation, np.ones(2) * 2.0, [0, 1], np.ones(2)
    )
    assert unit > 0.0
    assert doubled_noise == pytest.approx(unit / 4.0)


def test_soft_support_requires_an_explicit_nonfull_threshold() -> None:
    config = {"grid_size": 8, "depth": 5, "support_threshold": 0.05}
    support = support_mask_from_config(config)
    assert support.shape == (5, 8, 8)
    assert int(np.sum(support)) == 96
    with pytest.raises(ValueError, match="nonempty, non-full"):
        support_mask_from_config({**config, "support_threshold": 1e-12})


def test_repository_pilot_uses_a_source_matched_safe_renderer() -> None:
    import json
    from pathlib import Path

    config_path = (
        Path(__file__).resolve().parent / "configs" / "v5b_rig_shared_profile_pilot.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    audit = validate_forward_model_gate(config)
    assert audit["safe_settings_used"] == [
        {"path_samples": 35, "aperture_samples": 41}
    ]
