from __future__ import annotations

import numpy as np
import torch

try:
    from .measurement_contract import BOSTBatch, DenseVolumeLinearBOST
    from .run_v5a_blind_aperture_calibration import (
        SamplewiseDenseVolumeLinearBOST,
        cross_view_masks,
        one_hot_weights,
        reconstruction_signal_rms,
        samplewise_operator_from_weights,
        unique_view_masks,
    )
except ImportError:
    from measurement_contract import BOSTBatch, DenseVolumeLinearBOST
    from run_v5a_blind_aperture_calibration import (
        SamplewiseDenseVolumeLinearBOST,
        cross_view_masks,
        one_hot_weights,
        reconstruction_signal_rms,
        samplewise_operator_from_weights,
        unique_view_masks,
    )


def make_batch(batch_size: int = 3) -> BOSTBatch:
    return BOSTBatch(
        observation=torch.zeros((batch_size, 2, 4, 3)),
        view_mask=torch.tensor([[1, 1, 1, 0]] * batch_size, dtype=torch.float32),
        noise_std=torch.ones((batch_size, 1, 4, 1)),
        view_angles_degrees=torch.tensor([[0, 45, 90, 135]] * batch_size),
        support=torch.ones((1, 1, 2, 3, 3)),
        geometry_ids=tuple(f"sample-{index}" for index in range(batch_size)),
        truth=torch.ones((batch_size, 1, 2, 3, 3)),
    ).validate()


def test_unique_view_masks_exclude_audit_and_do_not_repeat():
    masks, identifiers = unique_view_masks(
        count=12,
        views=7,
        audit_camera=6,
        budgets=[3, 4, 5],
        rng=np.random.default_rng(9),
    )
    assert np.all(masks[:, 6] == 0)
    assert len(set(identifiers)) == len(identifiers)
    assert set(masks.sum(axis=1).astype(int)) <= {3, 4, 5}


def test_noise_scale_cannot_see_audit_camera_signal():
    clean = np.ones((2, 4, 3), dtype=np.float32)
    reconstruction_mask = np.array([1, 1, 0, 0], dtype=np.float32)
    reference = reconstruction_signal_rms(clean, reconstruction_mask)
    clean[:, 3] = 1_000_000.0
    assert reconstruction_signal_rms(clean, reconstruction_mask) == reference


def test_cross_view_fold_withholds_one_visible_camera_per_sample():
    batch = make_batch()
    first_fit, first_held = cross_view_masks(batch, fold=0, folds=2)
    second_fit, second_held = cross_view_masks(batch, fold=1, folds=2)
    assert torch.all(first_held.sum(dim=1) == 1)
    assert torch.all(second_held.sum(dim=1) == 1)
    assert torch.all(first_fit.sum(dim=1) == batch.view_mask.sum(dim=1) - 1)
    assert torch.all((first_fit + first_held) == batch.view_mask)
    assert torch.all((second_fit + second_held) == batch.view_mask)


def test_samplewise_operator_has_exact_adjoint():
    generator = torch.Generator().manual_seed(12)
    matrix = torch.randn((3, 2, 4, 3, 18), generator=generator)
    operator = SamplewiseDenseVolumeLinearBOST(matrix, (2, 3, 3))
    batch = make_batch()
    assert operator.adjoint_relative_error(batch, seed=14) < 1e-6


def test_one_hot_samplewise_operator_matches_dense_operator_per_sample():
    rng = np.random.default_rng(22)
    bank = rng.normal(size=(2, 2, 4, 3, 18)).astype(np.float32)
    indices = np.array([0, 1, 0])
    weights = one_hot_weights(indices, classes=2)
    samplewise = samplewise_operator_from_weights(bank, weights, (2, 3, 3))
    batch = make_batch()
    field = torch.randn((3, 1, 2, 3, 3), generator=torch.Generator().manual_seed(3))
    predicted = samplewise.forward(field, batch)
    expected = []
    for sample, operator_index in enumerate(indices):
        dense = DenseVolumeLinearBOST(torch.from_numpy(bank[operator_index]), (2, 3, 3))
        one = BOSTBatch(
            observation=batch.observation[sample : sample + 1],
            view_mask=batch.view_mask[sample : sample + 1],
            noise_std=batch.noise_std[sample : sample + 1],
            view_angles_degrees=batch.view_angles_degrees[sample : sample + 1],
            support=batch.support,
            geometry_ids=(f"dense-{sample}",),
            truth=batch.truth[sample : sample + 1],
        )
        expected.append(dense.forward(field[sample : sample + 1], one))
    torch.testing.assert_close(predicted, torch.cat(expected), rtol=1e-6, atol=1e-6)
