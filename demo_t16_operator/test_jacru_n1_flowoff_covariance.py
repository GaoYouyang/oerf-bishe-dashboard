import math

import pytest
import torch

from demo_t16_operator.jacru_n1_flowoff_covariance import (
    build_flowoff_calibration_payload,
    calibrate_discrepancy_threshold,
    dense_covariance_proximal_discrepancy,
    estimate_camera_random_effect_covariance,
    exact_camera_random_effect_covariance,
    isotropic_covariance_like,
    lock_coverage,
    whitened_quadratic,
)


def _camera_index() -> torch.Tensor:
    return torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int64)


def _bias() -> torch.Tensor:
    return torch.tensor([[0.2, -0.1], [-0.15, 0.05]], dtype=torch.float64)


def _payload(mode: str, *, repeats: int = 96):
    return build_flowoff_calibration_payload(
        case_id="case-a",
        geometry_digest="geometry-a",
        camera_index=_camera_index(),
        persistent_camera_bias_uv=_bias(),
        iid_noise_std=0.02,
        camera_bias_std=0.08,
        mode=mode,
        fit_repeats=repeats,
        selection_repeats=repeats,
        lock_repeats=repeats,
        seed=71,
    )


def test_payload_is_deterministic_split_and_truth_free() -> None:
    first = _payload("paired_static")
    second = _payload("paired_static")
    assert first.payload_digest == second.payload_digest
    assert torch.equal(first.fit_samples_uv, second.fit_samples_uv)
    assert not torch.equal(first.fit_samples_uv, first.selection_samples_uv)
    assert not hasattr(first, "truth_volume")
    assert not hasattr(first, "clean_observations_uv")
    assert set(first.estimator_kwargs()) == {"fit_samples_uv", "camera_index"}


def test_paired_static_recovers_persistent_group_mean() -> None:
    payload = _payload("paired_static", repeats=512)
    estimate = estimate_camera_random_effect_covariance(**payload.estimator_kwargs())
    expected = _bias()[_camera_index()]
    assert torch.max(torch.abs(estimate.mean_uv - expected)) < 0.004
    assert float(torch.max(estimate.shared_variance_by_group)) < 8e-5


def test_unpaired_distribution_recovers_camera_random_effect() -> None:
    payload = _payload("unpaired_distribution", repeats=1024)
    estimate = estimate_camera_random_effect_covariance(
        **payload.estimator_kwargs(), shrinkage=0.0
    )
    expected = 0.08**2
    assert torch.allclose(
        estimate.shared_variance_by_group,
        torch.full_like(estimate.shared_variance_by_group, expected),
        rtol=0.22,
        atol=7e-4,
    )
    groups = estimate.camera_component_index
    same = torch.nonzero(groups == groups[0], as_tuple=False).reshape(-1)
    other = torch.nonzero(groups == groups[-1], as_tuple=False).reshape(-1)
    assert float(estimate.covariance[same[0], same[1]]) > 0.003
    assert estimate.covariance[same[0], other[0]] == 0.0


def test_structured_covariance_is_strictly_spd() -> None:
    estimate = estimate_camera_random_effect_covariance(
        **_payload("unpaired_distribution").estimator_kwargs()
    )
    assert estimate.minimum_eigenvalue > 0.0
    assert estimate.maximum_eigenvalue >= estimate.minimum_eigenvalue
    assert math.isfinite(estimate.condition_number)
    torch.linalg.cholesky(estimate.covariance)


def test_exact_covariance_has_expected_within_group_structure() -> None:
    covariance = exact_camera_random_effect_covariance(
        camera_index=_camera_index(), iid_noise_std=0.02, camera_bias_std=0.08
    )
    assert covariance.shape == (12, 12)
    assert covariance[0, 2] == pytest.approx(0.08**2)
    assert covariance[0, 1] == 0.0
    assert covariance[0, 0] == pytest.approx(0.08**2 + 0.02**2)


def test_exact_covariance_rejects_negative_bias_standard_deviation() -> None:
    with pytest.raises(ValueError, match="camera_bias_std must be nonnegative"):
        exact_camera_random_effect_covariance(
            camera_index=_camera_index(), iid_noise_std=0.02, camera_bias_std=-0.08
        )


def test_empirical_threshold_has_reasonable_independent_lock_coverage() -> None:
    payload = _payload("unpaired_distribution", repeats=512)
    estimate = estimate_camera_random_effect_covariance(**payload.estimator_kwargs())
    calibration = calibrate_discrepancy_threshold(
        samples_uv=payload.selection_samples_uv,
        estimate=estimate,
        quantile=0.95,
    )
    coverage, scores = lock_coverage(
        samples_uv=payload.lock_samples_uv,
        estimate=estimate,
        threshold=calibration.threshold,
    )
    assert 0.88 <= coverage <= 0.99
    assert scores.shape == (512,)


def test_unpaired_zero_mean_calibration_does_not_require_fit_mean() -> None:
    payload = _payload("unpaired_distribution", repeats=256)
    estimate = estimate_camera_random_effect_covariance(**payload.estimator_kwargs())
    zero = torch.zeros_like(estimate.mean_uv)
    calibration = calibrate_discrepancy_threshold(
        samples_uv=payload.selection_samples_uv,
        estimate=estimate,
        quantile=0.9,
        mean_uv=zero,
    )
    coverage, _ = lock_coverage(
        samples_uv=payload.lock_samples_uv,
        estimate=estimate,
        threshold=calibration.threshold,
        mean_uv=zero,
    )
    assert 0.80 <= coverage <= 0.97


def _toy_problem():
    support = torch.tensor(
        [
            [[1, 1], [0, 0]],
            [[1, 0], [0, 0]],
        ],
        dtype=torch.bool,
    )
    matrix = torch.tensor(
        [[1.0, 0.2, 0.0], [0.0, 0.4, 1.0]], dtype=torch.float64
    )
    field = torch.zeros((2, 2, 2), dtype=torch.float64)
    field.masked_scatter_(support, torch.tensor([0.8, -0.2, 0.1], dtype=torch.float64))
    target = torch.tensor([[0.15, -0.3]], dtype=torch.float64)
    covariance = torch.tensor([[0.04, 0.012], [0.012, 0.09]], dtype=torch.float64)
    return support, matrix, field, target, covariance


def test_proximal_returns_initial_field_when_already_inside_discrepancy() -> None:
    support, matrix, field, target, covariance = _toy_problem()
    raw = whitened_quadratic(matrix @ field.masked_select(support) - target.reshape(-1), covariance)
    result = dense_covariance_proximal_discrepancy(
        initial_field=field,
        target_observation_uv=target,
        dense_active_matrix=matrix,
        support_mask=support,
        proximal_covariance=covariance,
        selector_covariance=covariance,
        discrepancy_threshold=raw * 1.01,
    )
    assert math.isinf(result.alpha)
    assert result.correction_norm == 0.0
    assert torch.equal(result.field, field)


def test_proximal_hits_threshold_and_preserves_support() -> None:
    support, matrix, field, target, covariance = _toy_problem()
    raw = whitened_quadratic(matrix @ field.masked_select(support) - target.reshape(-1), covariance)
    result = dense_covariance_proximal_discrepancy(
        initial_field=field,
        target_observation_uv=target,
        dense_active_matrix=matrix,
        support_mask=support,
        proximal_covariance=covariance,
        selector_covariance=covariance,
        discrepancy_threshold=0.35 * raw,
    )
    assert result.target_crossed
    assert result.selected_discrepancy <= 0.35 * raw * (1.0 + 1e-8)
    assert result.selected_discrepancy > 0.34 * raw
    assert torch.count_nonzero(result.field.masked_select(~support)) == 0
    assert result.correction_norm > 0.0
    assert result.residual_closure_relative_error < 1e-12


def test_structured_and_isotropic_proximal_paths_are_not_identical() -> None:
    support, matrix, field, target, covariance = _toy_problem()
    raw = whitened_quadratic(matrix @ field.masked_select(support) - target.reshape(-1), covariance)
    structured = dense_covariance_proximal_discrepancy(
        initial_field=field,
        target_observation_uv=target,
        dense_active_matrix=matrix,
        support_mask=support,
        proximal_covariance=covariance,
        selector_covariance=covariance,
        discrepancy_threshold=0.5 * raw,
    )
    isotropic = dense_covariance_proximal_discrepancy(
        initial_field=field,
        target_observation_uv=target,
        dense_active_matrix=matrix,
        support_mask=support,
        proximal_covariance=isotropic_covariance_like(covariance),
        selector_covariance=covariance,
        discrepancy_threshold=0.5 * raw,
    )
    assert not torch.allclose(structured.field, isotropic.field, atol=1e-8, rtol=1e-8)


def test_covariance_and_threshold_scale_together_without_changing_field() -> None:
    support, matrix, field, target, covariance = _toy_problem()
    raw = whitened_quadratic(matrix @ field.masked_select(support) - target.reshape(-1), covariance)
    reference = dense_covariance_proximal_discrepancy(
        initial_field=field,
        target_observation_uv=target,
        dense_active_matrix=matrix,
        support_mask=support,
        proximal_covariance=covariance,
        selector_covariance=covariance,
        discrepancy_threshold=0.45 * raw,
    )
    factor = 7.5
    rescaled = dense_covariance_proximal_discrepancy(
        initial_field=field,
        target_observation_uv=target,
        dense_active_matrix=matrix,
        support_mask=support,
        proximal_covariance=factor * covariance,
        selector_covariance=factor * covariance,
        discrepancy_threshold=(0.45 * raw) / factor,
    )
    assert torch.allclose(reference.field, rescaled.field, atol=1e-11, rtol=1e-11)
    assert reference.alpha == pytest.approx(rescaled.alpha, rel=1e-10)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"mode": "bad"}, "unsupported flow-off mode"),
        ({"fit_repeats": 1}, "at least two repeats"),
        ({"iid_noise_std": 0.0}, "positive finite"),
    ],
)
def test_payload_rejects_invalid_contract(kwargs, message) -> None:
    values = {
        "case_id": "case-a",
        "geometry_digest": "geometry-a",
        "camera_index": _camera_index(),
        "persistent_camera_bias_uv": _bias(),
        "iid_noise_std": 0.02,
        "camera_bias_std": 0.08,
        "mode": "paired_static",
        "fit_repeats": 8,
        "selection_repeats": 8,
        "lock_repeats": 8,
        "seed": 3,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        build_flowoff_calibration_payload(**values)
