from __future__ import annotations

from dataclasses import fields, replace
import inspect

import pytest
import torch

import demo_t16_operator.jacru_synthetic_fixture as fixture_module
from demo_t16_operator.analytic_bost_phantoms import ANALYTIC_RENDERER_SCHEMA
from demo_t16_operator.jacru_synthetic_fixture import (
    JACRU_FAMILIES,
    JACRU_SPLITS,
    JACRUInferencePayload,
    JACRUSyntheticFixtureConfig,
    build_jacru_fixture_suite,
    build_jacru_geometry,
    build_jacru_synthetic_case,
    inference_field_names,
)
from demo_t16_operator.psu_b0_reconstruction_interface import INTERFACE_SCHEMA


@pytest.fixture(scope="module")
def compact_config() -> JACRUSyntheticFixtureConfig:
    return JACRUSyntheticFixtureConfig(
        grid_shape=(12, 12, 12),
        detector_shape=(4, 4),
        samples_per_ray=12,
    )


def test_case_generation_is_bitwise_deterministic(
    compact_config: JACRUSyntheticFixtureConfig,
) -> None:
    first = build_jacru_synthetic_case(
        family="single_interface",
        split="development",
        base_seed=41,
        config=compact_config,
    )
    second = build_jacru_synthetic_case(
        family="single_interface",
        split="development",
        base_seed=41,
        config=compact_config,
    )
    assert first.inference.case_id == second.inference.case_id
    assert first.inference.geometry.digest == second.inference.geometry.digest
    assert first.inference.observation_digest == second.inference.observation_digest
    assert torch.equal(first.inference.observations_uv, second.inference.observations_uv)
    assert torch.equal(first.evaluation.truth_volume, second.evaluation.truth_volume)
    assert torch.equal(first.evaluation.level_sets, second.evaluation.level_sets)

    changed = build_jacru_synthetic_case(
        family="single_interface",
        split="development",
        base_seed=42,
        config=compact_config,
    )
    assert changed.inference.case_id != first.inference.case_id
    assert changed.inference.geometry.digest != first.inference.geometry.digest
    assert not torch.equal(changed.evaluation.truth_volume, first.evaluation.truth_volume)


@pytest.mark.parametrize("split", JACRU_SPLITS)
def test_each_split_voxel_operator_has_an_exact_adjoint(
    split: str,
    compact_config: JACRUSyntheticFixtureConfig,
) -> None:
    case = build_jacru_synthetic_case(
        family="smooth_no_interface",
        split=split,
        base_seed=57,
        config=compact_config,
    )
    assert case.inference.operator.adjoint_relative_error(seed=731) < 2e-12
    assert case.inference.operator.call_report() == {
        "forward_calls": 0,
        "adjoint_calls": 0,
    }


def test_suite_has_mutually_exclusive_split_records_and_multiple_seeds(
    compact_config: JACRUSyntheticFixtureConfig,
) -> None:
    suite = build_jacru_fixture_suite(base_seeds=(11, 29), config=compact_config)
    assert len(suite.cases) == len(JACRU_SPLITS) * len(JACRU_FAMILIES) * 2
    assert len(suite.manifest_sha256) == 64
    assert suite.manifest_sha256 == build_jacru_fixture_suite(
        base_seeds=(11, 29), config=compact_config
    ).manifest_sha256
    for split in JACRU_SPLITS:
        split_cases = suite.for_split(split)
        assert len(split_cases) == len(JACRU_FAMILIES) * 2
        assert {case.evaluation.base_seed for case in split_cases} == {11, 29}
        assert {case.evaluation.family for case in split_cases} == set(JACRU_FAMILIES)

    for attribute in ("case_id",):
        split_sets = [
            {getattr(case.inference, attribute) for case in suite.for_split(split)}
            for split in JACRU_SPLITS
        ]
        assert all(
            split_sets[left].isdisjoint(split_sets[right])
            for left in range(len(split_sets))
            for right in range(left + 1, len(split_sets))
        )
    phantom_sets = [
        {case.evaluation.phantom_seed for case in suite.for_split(split)}
        for split in JACRU_SPLITS
    ]
    geometry_sets = [
        {case.inference.geometry.digest for case in suite.for_split(split)}
        for split in JACRU_SPLITS
    ]
    assert all(
        phantom_sets[left].isdisjoint(phantom_sets[right])
        and geometry_sets[left].isdisjoint(geometry_sets[right])
        for left in range(3)
        for right in range(left + 1, 3)
    )


def test_split_geometry_changes_camera_pose_and_sampled_rays(
    compact_config: JACRUSyntheticFixtureConfig,
) -> None:
    geometries = {
        split: build_jacru_geometry(split=split, base_seed=83, config=compact_config)
        for split in JACRU_SPLITS
    }
    assert len({geometry.digest for geometry in geometries.values()}) == 3
    assert len({geometry.camera_azimuth_degrees for geometry in geometries.values()}) == 3
    assert len({geometry.camera_elevation_degrees for geometry in geometries.values()}) == 3
    assert not torch.equal(
        geometries["train"].sample_points_xyz,
        geometries["development"].sample_points_xyz,
    )
    assert max(abs(value) for value in geometries["ood"].camera_elevation_degrees) > 10.0
    for geometry in geometries.values():
        assert geometry.camera_count == 3
        assert geometry.ray_count == 3 * 4 * 4
        assert torch.all(geometry.sample_valid)


@pytest.mark.parametrize(
    ("family", "interface_count"),
    (
        ("smooth_no_interface", 0),
        ("single_interface", 1),
        ("two_interface", 2),
    ),
)
def test_all_morphology_families_have_only_evaluation_labels(
    family: str,
    interface_count: int,
    compact_config: JACRUSyntheticFixtureConfig,
) -> None:
    case = build_jacru_synthetic_case(
        family=family,
        split="train",
        base_seed=97,
        config=compact_config,
    )
    assert case.evaluation.interface_count == interface_count
    assert case.evaluation.level_sets.shape == (
        1,
        interface_count,
        *compact_config.grid_shape,
    )
    assert case.evaluation.truth_volume.shape == (1, 1, *compact_config.grid_shape)
    assert case.inference.observations_uv.shape == (
        1,
        case.inference.geometry.ray_count,
        2,
    )


def test_inference_contract_cannot_expose_truth_or_level_sets(
    compact_config: JACRUSyntheticFixtureConfig,
) -> None:
    forbidden_fragments = ("truth", "level", "phantom", "clean", "noise", "bias", "evaluation")
    names = inference_field_names()
    assert names == tuple(field.name for field in fields(JACRUInferencePayload))
    assert not any(fragment in name.lower() for name in names for fragment in forbidden_fragments)

    case = build_jacru_synthetic_case(
        family="two_interface",
        split="ood",
        base_seed=113,
        config=compact_config,
    )
    model_kwargs = case.inference.model_kwargs()
    assert set(model_kwargs) == {"observations_uv", "geometry", "operator"}
    assert not any(
        fragment in key.lower()
        for key in model_kwargs
        for fragment in forbidden_fragments
    )
    assert not hasattr(case.inference, "truth_volume")
    assert not hasattr(case.inference, "level_sets")
    assert not hasattr(case.inference.geometry, "truth_volume")
    assert not hasattr(case.inference.operator, "truth_volume")


def test_noise_and_camera_bias_switches_are_exact_and_reproducible(
    compact_config: JACRUSyntheticFixtureConfig,
) -> None:
    clean_config = replace(
        compact_config,
        enable_noise=False,
        enable_camera_bias=False,
    )
    noisy_config = replace(
        compact_config,
        enable_noise=True,
        enable_camera_bias=True,
    )
    clean = build_jacru_synthetic_case(
        family="single_interface",
        split="development",
        base_seed=127,
        config=clean_config,
    )
    noisy = build_jacru_synthetic_case(
        family="single_interface",
        split="development",
        base_seed=127,
        config=noisy_config,
    )
    assert torch.count_nonzero(clean.evaluation.additive_noise_uv) == 0
    assert torch.count_nonzero(clean.evaluation.camera_bias_uv) == 0
    assert torch.equal(
        clean.inference.observations_uv,
        clean.evaluation.clean_observations_uv,
    )
    assert torch.count_nonzero(noisy.evaluation.additive_noise_uv) > 0
    assert torch.count_nonzero(noisy.evaluation.camera_bias_uv) > 0
    reconstructed_measurement = (
        noisy.evaluation.clean_observations_uv
        + noisy.evaluation.camera_bias_uv[noisy.inference.geometry.camera_index][None]
        + noisy.evaluation.additive_noise_uv
    )
    assert torch.equal(noisy.inference.observations_uv, reconstructed_measurement)


def test_observations_do_not_use_the_voxel_forward_chain(
    compact_config: JACRUSyntheticFixtureConfig,
) -> None:
    case = build_jacru_synthetic_case(
        family="single_interface",
        split="train",
        base_seed=139,
        config=replace(
            compact_config,
            enable_noise=False,
            enable_camera_bias=False,
        ),
    )
    voxel_render = case.inference.operator(case.evaluation.truth_volume)
    analytic_render = case.evaluation.clean_observations_uv
    relative_mismatch = (
        torch.linalg.vector_norm(voxel_render - analytic_render)
        / torch.linalg.vector_norm(analytic_render)
    )
    assert relative_mismatch > 1e-3
    assert case.inference.observation_generator_schema == ANALYTIC_RENDERER_SCHEMA
    assert case.inference.inverse_operator_schema == INTERFACE_SCHEMA
    source = inspect.getsource(fixture_module._case_from_geometry)
    observation_block = source.split("# Crucial boundary:", 1)[1].split(
        "operator = _build_voxel_operator", 1
    )[0]
    assert "render_analytic_bost" in observation_block
    assert ".operator(" not in observation_block
    assert "operator.forward" not in observation_block
