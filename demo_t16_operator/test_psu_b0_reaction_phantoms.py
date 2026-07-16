"""Tests for deterministic analytic reaction morphology proxies."""

from __future__ import annotations

import torch

from .psu_b0_reaction_phantoms import (
    SUPPORTED_FAMILIES,
    reaction_morphology_batch,
    reaction_morphology_field,
)


def test_reaction_fields_are_deterministic_normalized_and_gauge_fixed() -> None:
    for family in SUPPORTED_FAMILIES:
        first = reaction_morphology_field(
            grid_size=16,
            family=family,
            seed=17,
        )
        second = reaction_morphology_field(
            grid_size=16,
            family=family,
            seed=17,
        )
        assert first.shape == (1, 1, 16, 16, 16)
        assert torch.equal(first, second)
        assert torch.isclose(torch.sqrt(torch.mean(first.square())), torch.tensor(1.0))
        assert torch.count_nonzero(first[:, :, 0]) == 0
        assert torch.count_nonzero(first[:, :, -1]) == 0
        assert torch.count_nonzero(first[:, :, :, 0]) == 0
        assert torch.count_nonzero(first[:, :, :, -1]) == 0
        assert torch.count_nonzero(first[:, :, :, :, 0]) == 0
        assert torch.count_nonzero(first[:, :, :, :, -1]) == 0


def test_reaction_batch_aligns_families_and_seeds() -> None:
    values = reaction_morphology_batch(
        grid_size=16,
        families=("plume", "thin_front"),
        seeds=(2, 3),
    )
    assert values.shape == (2, 1, 16, 16, 16)
    assert not torch.equal(values[0], values[1])


def test_extended_reaction_families_are_distinct_and_deterministic() -> None:
    families = ("annular_kernel", "oblique_shock", "vortex_pair", "multi_plume")
    first = reaction_morphology_batch(
        grid_size=16,
        families=families,
        seeds=(41, 42, 43, 44),
    )
    second = reaction_morphology_batch(
        grid_size=16,
        families=families,
        seeds=(41, 42, 43, 44),
    )
    assert torch.equal(first, second)
    assert torch.all(torch.isfinite(first))
    rms = torch.sqrt(torch.mean(first.square(), dim=(1, 2, 3, 4)))
    assert torch.allclose(rms, torch.ones_like(rms), atol=2e-5)
    pairwise = torch.cdist(first.flatten(1), first.flatten(1))
    off_diagonal = ~torch.eye(4, dtype=torch.bool)
    assert torch.all(pairwise[off_diagonal] > 1.0)
