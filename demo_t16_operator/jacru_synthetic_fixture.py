"""Small deterministic JACRU fixture with an explicit inverse-crime barrier.

The measurement side integrates the continuous analytic gradients from
``analytic_bost_phantoms``.  The inference side receives a separately
discretized voxel finite-difference/trilinear operator from
``psu_b0_reconstruction_interface``.  Sharing ray geometry is intentional;
sharing the discrete forward chain used to create observations is not.

This fixture is a morphology and reconstruction benchmark, not CFD truth or a
claim that the normalized scalar has a calibrated refractive-index scale.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from typing import Any, Iterable

import numpy as np
import torch

from .analytic_bost_phantoms import (
    ANALYTIC_PHANTOM_SCHEMA,
    ANALYTIC_RENDERER_SCHEMA,
    AnalyticPhantomSpec,
    analytic_phantom_grid,
    make_analytic_phantom,
    render_analytic_bost,
)
from .psu_b0_reconstruction_interface import (
    INTERFACE_SCHEMA,
    PSUB0VoxelGradientOperator,
    build_trilinear_stencil,
)


JACRU_FIXTURE_SCHEMA = "jacru-synthetic-fixture-1.0"
JACRU_SPLITS = ("train", "development", "ood")
JACRU_FAMILIES = (
    "smooth_no_interface",
    "single_interface",
    "two_interface",
)

_ANALYTIC_FAMILY_BY_JACRU_FAMILY = {
    "smooth_no_interface": "smooth_plume",
    "single_interface": "wrinkled_density_interface",
    "two_interface": "shock_expansion_pair",
}
_INTERFACE_COUNT_BY_FAMILY = {
    "smooth_no_interface": 0,
    "single_interface": 1,
    "two_interface": 2,
}
_CAMERA_POSES_DEGREES = {
    "train": ((0.0, 0.0), (60.0, 0.0), (120.0, 0.0)),
    "development": ((20.0, 6.0), (80.0, -5.0), (140.0, 4.0)),
    "ood": ((37.0, 18.0), (97.0, -15.0), (157.0, 12.0)),
}


def _stable_seed(*tokens: Any) -> int:
    encoded = "\x1f".join(str(token) for token in tokens).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") % (2**63)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _tensor_digest(*tensors: torch.Tensor, metadata: Any = None) -> str:
    digest = hashlib.sha256()
    if metadata is not None:
        digest.update(_canonical_json(metadata).encode("ascii"))
    for tensor in tensors:
        values = torch.as_tensor(tensor).detach().cpu().contiguous()
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(_canonical_json(tuple(int(v) for v in values.shape)).encode("ascii"))
        digest.update(values.numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class JACRUSyntheticFixtureConfig:
    """Frozen generation policy for a laptop-sized benchmark."""

    grid_shape: tuple[int, int, int] = (12, 12, 12)
    domain_minimum_xyz: tuple[float, float, float] = (-1.0, -1.0, -1.0)
    domain_maximum_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0)
    detector_shape: tuple[int, int] = (6, 6)
    detector_half_width: float = 0.62
    samples_per_ray: int = 24
    geometry_azimuth_jitter_degrees: float = 1.25
    geometry_elevation_jitter_degrees: float = 0.75
    detector_shift_limit: float = 0.025
    noise_relative_std: float = 0.01
    camera_bias_relative_std: float = 0.02
    enable_noise: bool = True
    enable_camera_bias: bool = True

    def __post_init__(self) -> None:
        shape = tuple(int(value) for value in self.grid_shape)
        if shape not in {(12, 12, 12), (16, 16, 16)}:
            raise ValueError("grid_shape must be either 12^3 or 16^3")
        if len(self.detector_shape) != 2 or any(int(value) < 3 for value in self.detector_shape):
            raise ValueError("detector_shape must contain two dimensions of at least three")
        minimum = np.asarray(self.domain_minimum_xyz, dtype=np.float64)
        maximum = np.asarray(self.domain_maximum_xyz, dtype=np.float64)
        if minimum.shape != (3,) or maximum.shape != (3,) or np.any(maximum <= minimum):
            raise ValueError("domain bounds must contain increasing x, y, z values")
        if self.detector_half_width <= 0.0 or self.detector_half_width >= 0.8:
            raise ValueError("detector_half_width must lie in (0, 0.8)")
        if int(self.samples_per_ray) < 4:
            raise ValueError("samples_per_ray must be at least four")
        if self.geometry_azimuth_jitter_degrees < 0.0:
            raise ValueError("azimuth jitter must be nonnegative")
        if self.geometry_elevation_jitter_degrees < 0.0:
            raise ValueError("elevation jitter must be nonnegative")
        if self.detector_shift_limit < 0.0:
            raise ValueError("detector_shift_limit must be nonnegative")
        if self.noise_relative_std < 0.0 or self.camera_bias_relative_std < 0.0:
            raise ValueError("noise and camera-bias scales must be nonnegative")

    def manifest(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JACRUGeometry:
    """Deterministic parallel-ray geometry supplied to inference methods."""

    split: str
    base_seed: int
    geometry_seed: int
    camera_azimuth_degrees: tuple[float, ...]
    camera_elevation_degrees: tuple[float, ...]
    detector_shape: tuple[int, int]
    samples_per_ray: int
    sample_points_xyz: torch.Tensor
    sample_valid: torch.Tensor
    projection_u_xyz: torch.Tensor
    projection_v_xyz: torch.Tensor
    line_length: torch.Tensor
    system_constant: torch.Tensor
    camera_index: torch.Tensor
    digest: str

    @property
    def camera_count(self) -> int:
        return len(self.camera_azimuth_degrees)

    @property
    def ray_count(self) -> int:
        return int(self.sample_points_xyz.shape[0])


@dataclass(frozen=True)
class JACRUInferencePayload:
    """The complete, truth-free API presented to a reconstruction method."""

    case_id: str
    split: str
    observations_uv: torch.Tensor
    geometry: JACRUGeometry
    operator: PSUB0VoxelGradientOperator
    observation_digest: str
    observation_generator_schema: str = ANALYTIC_RENDERER_SCHEMA
    inverse_operator_schema: str = INTERFACE_SCHEMA

    def model_kwargs(self) -> dict[str, Any]:
        """Return only quantities a reconstruction method is allowed to see."""

        return {
            "observations_uv": self.observations_uv,
            "geometry": self.geometry,
            "operator": self.operator,
        }


@dataclass(frozen=True)
class JACRUEvaluationPayload:
    """Withheld labels and nuisance realizations used only for evaluation."""

    case_id: str
    family: str
    analytic_family: str
    base_seed: int
    phantom_seed: int
    phantom_spec: AnalyticPhantomSpec
    truth_volume: torch.Tensor
    level_sets: torch.Tensor
    interface_count: int
    clean_observations_uv: torch.Tensor
    additive_noise_uv: torch.Tensor
    camera_bias_uv: torch.Tensor
    scalar_contract: str


@dataclass(frozen=True)
class JACRUSyntheticCase:
    inference: JACRUInferencePayload
    evaluation: JACRUEvaluationPayload


@dataclass(frozen=True)
class JACRUSyntheticSuite:
    """A deterministic collection with disjoint train/development/OOD IDs."""

    config: JACRUSyntheticFixtureConfig
    base_seeds: tuple[int, ...]
    cases: tuple[JACRUSyntheticCase, ...]
    manifest_json: str
    manifest_sha256: str

    def for_split(self, split: str) -> tuple[JACRUSyntheticCase, ...]:
        _validate_split(split)
        return tuple(case for case in self.cases if case.inference.split == split)

    def inference_payloads(self, split: str) -> tuple[JACRUInferencePayload, ...]:
        return tuple(case.inference for case in self.for_split(split))

    def evaluation_payloads(self, split: str) -> tuple[JACRUEvaluationPayload, ...]:
        return tuple(case.evaluation for case in self.for_split(split))


def _validate_split(split: str) -> str:
    name = str(split)
    if name not in JACRU_SPLITS:
        raise ValueError(f"split must be one of {JACRU_SPLITS}")
    return name


def _validate_family(family: str) -> str:
    name = str(family)
    if name not in JACRU_FAMILIES:
        raise ValueError(f"family must be one of {JACRU_FAMILIES}")
    return name


def _camera_frame(azimuth_degrees: float, elevation_degrees: float) -> tuple[np.ndarray, ...]:
    azimuth = math.radians(float(azimuth_degrees))
    elevation = math.radians(float(elevation_degrees))
    direction = np.asarray(
        [
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        ],
        dtype=np.float64,
    )
    reference = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(direction, reference))) > 0.9:
        reference = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    projection_u = np.cross(reference, direction)
    projection_u /= np.linalg.norm(projection_u)
    projection_v = np.cross(direction, projection_u)
    projection_v /= np.linalg.norm(projection_v)
    return direction, projection_u, projection_v


def _ray_box_interval(
    origin_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    minimum_xyz: np.ndarray,
    maximum_xyz: np.ndarray,
) -> tuple[float, float]:
    entry = -math.inf
    exit_ = math.inf
    for axis in range(3):
        direction = float(direction_xyz[axis])
        origin = float(origin_xyz[axis])
        if abs(direction) < 1e-14:
            if origin < minimum_xyz[axis] or origin > maximum_xyz[axis]:
                raise RuntimeError("detector ray misses the reconstruction box")
            continue
        first = (float(minimum_xyz[axis]) - origin) / direction
        second = (float(maximum_xyz[axis]) - origin) / direction
        entry = max(entry, min(first, second))
        exit_ = min(exit_, max(first, second))
    if not math.isfinite(entry) or not math.isfinite(exit_) or exit_ <= entry:
        raise RuntimeError("detector ray has no positive box intersection")
    return entry, exit_


def build_jacru_geometry(
    *,
    split: str,
    base_seed: int,
    config: JACRUSyntheticFixtureConfig | None = None,
) -> JACRUGeometry:
    """Build one split-specific, multi-camera geometry deterministically."""

    name = _validate_split(split)
    policy = config or JACRUSyntheticFixtureConfig()
    geometry_seed = _stable_seed(JACRU_FIXTURE_SCHEMA, "geometry", name, int(base_seed))
    rng = np.random.default_rng(geometry_seed)
    nominal_poses = _CAMERA_POSES_DEGREES[name]
    azimuths = tuple(
        float(
            azimuth
            + rng.uniform(
                -policy.geometry_azimuth_jitter_degrees,
                policy.geometry_azimuth_jitter_degrees,
            )
        )
        for azimuth, _ in nominal_poses
    )
    elevations = tuple(
        float(
            elevation
            + rng.uniform(
                -policy.geometry_elevation_jitter_degrees,
                policy.geometry_elevation_jitter_degrees,
            )
        )
        for _, elevation in nominal_poses
    )

    detector_y = np.linspace(
        -policy.detector_half_width,
        policy.detector_half_width,
        int(policy.detector_shape[0]),
        dtype=np.float64,
    )
    detector_x = np.linspace(
        -policy.detector_half_width,
        policy.detector_half_width,
        int(policy.detector_shape[1]),
        dtype=np.float64,
    )
    minimum = np.asarray(policy.domain_minimum_xyz, dtype=np.float64)
    maximum = np.asarray(policy.domain_maximum_xyz, dtype=np.float64)
    ray_samples: list[np.ndarray] = []
    projection_u_rows: list[np.ndarray] = []
    projection_v_rows: list[np.ndarray] = []
    line_lengths: list[float] = []
    camera_indices: list[int] = []
    sample_fractions = (np.arange(policy.samples_per_ray, dtype=np.float64) + 0.5) / float(
        policy.samples_per_ray
    )

    for camera, (azimuth, elevation) in enumerate(zip(azimuths, elevations, strict=True)):
        direction, projection_u, projection_v = _camera_frame(azimuth, elevation)
        shift_u, shift_v = rng.uniform(
            -policy.detector_shift_limit,
            policy.detector_shift_limit,
            size=2,
        )
        for detector_v in detector_y:
            for detector_u in detector_x:
                origin = (
                    (detector_u + shift_u) * projection_u
                    + (detector_v + shift_v) * projection_v
                )
                entry, exit_ = _ray_box_interval(origin, direction, minimum, maximum)
                parameters = entry + sample_fractions * (exit_ - entry)
                ray_samples.append(origin[None, :] + parameters[:, None] * direction[None, :])
                projection_u_rows.append(projection_u)
                projection_v_rows.append(projection_v)
                line_lengths.append(exit_ - entry)
                camera_indices.append(camera)

    sample_points = torch.as_tensor(np.stack(ray_samples), dtype=torch.float64)
    sample_valid = torch.ones(sample_points.shape[:2], dtype=torch.bool)
    projection_u_tensor = torch.as_tensor(np.stack(projection_u_rows), dtype=torch.float64)
    projection_v_tensor = torch.as_tensor(np.stack(projection_v_rows), dtype=torch.float64)
    line_length_tensor = torch.as_tensor(line_lengths, dtype=torch.float64)
    system_constant = torch.ones(len(line_lengths), dtype=torch.float64)
    camera_index = torch.as_tensor(camera_indices, dtype=torch.int64)
    metadata = {
        "schema": JACRU_FIXTURE_SCHEMA,
        "split": name,
        "base_seed": int(base_seed),
        "geometry_seed": geometry_seed,
        "azimuths": azimuths,
        "elevations": elevations,
        "detector_shape": tuple(int(value) for value in policy.detector_shape),
        "samples_per_ray": int(policy.samples_per_ray),
        "domain_minimum_xyz": policy.domain_minimum_xyz,
        "domain_maximum_xyz": policy.domain_maximum_xyz,
    }
    digest = _tensor_digest(
        sample_points,
        sample_valid,
        projection_u_tensor,
        projection_v_tensor,
        line_length_tensor,
        system_constant,
        camera_index,
        metadata=metadata,
    )
    return JACRUGeometry(
        split=name,
        base_seed=int(base_seed),
        geometry_seed=geometry_seed,
        camera_azimuth_degrees=azimuths,
        camera_elevation_degrees=elevations,
        detector_shape=tuple(int(value) for value in policy.detector_shape),
        samples_per_ray=int(policy.samples_per_ray),
        sample_points_xyz=sample_points,
        sample_valid=sample_valid,
        projection_u_xyz=projection_u_tensor,
        projection_v_xyz=projection_v_tensor,
        line_length=line_length_tensor,
        system_constant=system_constant,
        camera_index=camera_index,
        digest=digest,
    )


def _build_voxel_operator(
    geometry: JACRUGeometry,
    config: JACRUSyntheticFixtureConfig,
) -> PSUB0VoxelGradientOperator:
    stencil = build_trilinear_stencil(
        geometry.sample_points_xyz,
        grid_shape=config.grid_shape,
        grid_minimum_xyz=config.domain_minimum_xyz,
        grid_maximum_xyz=config.domain_maximum_xyz,
        sample_valid=geometry.sample_valid,
        dtype=torch.float64,
    )
    return PSUB0VoxelGradientOperator(
        stencil=stencil,
        projection_u_xyz=geometry.projection_u_xyz,
        projection_v_xyz=geometry.projection_v_xyz,
        line_length=geometry.line_length,
        system_constant=geometry.system_constant,
        grid_minimum_xyz=config.domain_minimum_xyz,
        grid_maximum_xyz=config.domain_maximum_xyz,
        dtype=torch.float64,
    )


def _measurement_components(
    clean_observations_uv: torch.Tensor,
    geometry: JACRUGeometry,
    *,
    case_seed: int,
    config: JACRUSyntheticFixtureConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    signal_rms = torch.sqrt(torch.mean(clean_observations_uv.square())).clamp_min(1e-12)
    bias_generator = torch.Generator().manual_seed(
        _stable_seed(JACRU_FIXTURE_SCHEMA, "camera-bias", case_seed)
    )
    noise_generator = torch.Generator().manual_seed(
        _stable_seed(JACRU_FIXTURE_SCHEMA, "noise", case_seed)
    )
    if config.enable_camera_bias:
        camera_bias = torch.randn(
            (geometry.camera_count, 2),
            generator=bias_generator,
            dtype=torch.float64,
        ) * (float(config.camera_bias_relative_std) * signal_rms)
    else:
        camera_bias = torch.zeros((geometry.camera_count, 2), dtype=torch.float64)
    if config.enable_noise:
        additive_noise = torch.randn(
            clean_observations_uv.shape,
            generator=noise_generator,
            dtype=torch.float64,
        ) * (float(config.noise_relative_std) * signal_rms)
    else:
        additive_noise = torch.zeros_like(clean_observations_uv)
    return camera_bias, additive_noise


def _case_from_geometry(
    *,
    family: str,
    split: str,
    base_seed: int,
    geometry: JACRUGeometry,
    config: JACRUSyntheticFixtureConfig,
) -> JACRUSyntheticCase:
    family_name = _validate_family(family)
    split_name = _validate_split(split)
    if geometry.split != split_name or geometry.base_seed != int(base_seed):
        raise ValueError("geometry must match the requested split and base_seed")
    phantom_seed = _stable_seed(
        JACRU_FIXTURE_SCHEMA,
        "phantom",
        split_name,
        family_name,
        int(base_seed),
    )
    analytic_family = _ANALYTIC_FAMILY_BY_JACRU_FAMILY[family_name]
    phantom_spec = make_analytic_phantom(
        family=analytic_family,
        seed=phantom_seed,
        domain_minimum_xyz=config.domain_minimum_xyz,
        domain_maximum_xyz=config.domain_maximum_xyz,
    )

    # Crucial boundary: observations come from the continuous analytic gradient.
    clean = render_analytic_bost(
        phantom_spec,
        sample_points_xyz=geometry.sample_points_xyz,
        projection_u_xyz=geometry.projection_u_xyz,
        projection_v_xyz=geometry.projection_v_xyz,
        line_length=geometry.line_length,
        system_constant=geometry.system_constant,
        sample_valid=geometry.sample_valid,
    )[None, :, :]
    case_seed = _stable_seed(
        JACRU_FIXTURE_SCHEMA,
        "case",
        split_name,
        family_name,
        int(base_seed),
    )
    camera_bias, additive_noise = _measurement_components(
        clean,
        geometry,
        case_seed=case_seed,
        config=config,
    )
    observations = clean + camera_bias[geometry.camera_index][None, :, :] + additive_noise
    operator = _build_voxel_operator(geometry, config)

    evaluation = analytic_phantom_grid(
        phantom_spec,
        grid_shape=config.grid_shape,
        dtype=torch.float64,
    )
    level_sets = evaluation.level_sets.movedim(-1, 0)[None, :, :, :, :].contiguous()
    expected_interfaces = _INTERFACE_COUNT_BY_FAMILY[family_name]
    if level_sets.shape[1] != expected_interfaces:
        raise RuntimeError("analytic family violated the declared interface count")

    case_contract = {
        "schema": JACRU_FIXTURE_SCHEMA,
        "split": split_name,
        "family": family_name,
        "base_seed": int(base_seed),
        "phantom_seed": phantom_seed,
        "geometry_digest": geometry.digest,
        "config": config.manifest(),
        "observation_generator_schema": ANALYTIC_RENDERER_SCHEMA,
        "inverse_operator_schema": INTERFACE_SCHEMA,
    }
    case_id = hashlib.sha256(_canonical_json(case_contract).encode("ascii")).hexdigest()[:20]
    observation_digest = _tensor_digest(observations, metadata={"case_id": case_id})
    inference_payload = JACRUInferencePayload(
        case_id=case_id,
        split=split_name,
        observations_uv=observations.detach().clone(),
        geometry=geometry,
        operator=operator,
        observation_digest=observation_digest,
    )
    evaluation_payload = JACRUEvaluationPayload(
        case_id=case_id,
        family=family_name,
        analytic_family=analytic_family,
        base_seed=int(base_seed),
        phantom_seed=phantom_seed,
        phantom_spec=phantom_spec,
        truth_volume=evaluation.field[None, None, :, :, :].contiguous(),
        level_sets=level_sets,
        interface_count=expected_interfaces,
        clean_observations_uv=clean.detach().clone(),
        additive_noise_uv=additive_noise.detach().clone(),
        camera_bias_uv=camera_bias.detach().clone(),
        scalar_contract=phantom_spec.scalar_contract,
    )
    return JACRUSyntheticCase(
        inference=inference_payload,
        evaluation=evaluation_payload,
    )


def build_jacru_synthetic_case(
    *,
    family: str,
    split: str,
    base_seed: int,
    config: JACRUSyntheticFixtureConfig | None = None,
) -> JACRUSyntheticCase:
    """Build one deterministic case and keep labels outside its inference API."""

    policy = config or JACRUSyntheticFixtureConfig()
    geometry = build_jacru_geometry(split=split, base_seed=base_seed, config=policy)
    return _case_from_geometry(
        family=family,
        split=split,
        base_seed=base_seed,
        geometry=geometry,
        config=policy,
    )


def _suite_manifest(
    config: JACRUSyntheticFixtureConfig,
    base_seeds: tuple[int, ...],
    cases: Iterable[JACRUSyntheticCase],
) -> dict[str, Any]:
    return {
        "schema": JACRU_FIXTURE_SCHEMA,
        "analytic_phantom_schema": ANALYTIC_PHANTOM_SCHEMA,
        "observation_generator_schema": ANALYTIC_RENDERER_SCHEMA,
        "inverse_operator_schema": INTERFACE_SCHEMA,
        "inverse_crime_barrier": (
            "continuous_analytic_gradient_observations_vs_"
            "voxel_fd_trilinear_inverse"
        ),
        "config": config.manifest(),
        "base_seeds": base_seeds,
        "cases": [
            {
                "case_id": case.inference.case_id,
                "split": case.inference.split,
                "family": case.evaluation.family,
                "base_seed": case.evaluation.base_seed,
                "phantom_seed": case.evaluation.phantom_seed,
                "geometry_digest": case.inference.geometry.digest,
                "observation_digest": case.inference.observation_digest,
                "interface_count": case.evaluation.interface_count,
            }
            for case in cases
        ],
    }


def _assert_disjoint_split_records(cases: Iterable[JACRUSyntheticCase]) -> None:
    by_split: dict[str, dict[str, set[Any]]] = {
        split: {"case_id": set(), "phantom_seed": set(), "geometry_digest": set()}
        for split in JACRU_SPLITS
    }
    for case in cases:
        record = by_split[case.inference.split]
        record["case_id"].add(case.inference.case_id)
        record["phantom_seed"].add(case.evaluation.phantom_seed)
        record["geometry_digest"].add(case.inference.geometry.digest)
    for index, left in enumerate(JACRU_SPLITS):
        for right in JACRU_SPLITS[index + 1 :]:
            for key in ("case_id", "phantom_seed", "geometry_digest"):
                if by_split[left][key] & by_split[right][key]:
                    raise RuntimeError(f"{left} and {right} overlap in {key}")


def build_jacru_fixture_suite(
    *,
    base_seeds: Iterable[int] = (101, 211, 307),
    config: JACRUSyntheticFixtureConfig | None = None,
) -> JACRUSyntheticSuite:
    """Build all three families on disjoint split geometries for many seeds."""

    policy = config or JACRUSyntheticFixtureConfig()
    seeds = tuple(int(seed) for seed in base_seeds)
    if not seeds:
        raise ValueError("base_seeds must not be empty")
    if len(set(seeds)) != len(seeds):
        raise ValueError("base_seeds must be unique")
    cases: list[JACRUSyntheticCase] = []
    for split in JACRU_SPLITS:
        for base_seed in seeds:
            geometry = build_jacru_geometry(
                split=split,
                base_seed=base_seed,
                config=policy,
            )
            for family in JACRU_FAMILIES:
                cases.append(
                    _case_from_geometry(
                        family=family,
                        split=split,
                        base_seed=base_seed,
                        geometry=geometry,
                        config=policy,
                    )
                )
    frozen_cases = tuple(cases)
    _assert_disjoint_split_records(frozen_cases)
    manifest_json = _canonical_json(_suite_manifest(policy, seeds, frozen_cases))
    return JACRUSyntheticSuite(
        config=policy,
        base_seeds=seeds,
        cases=frozen_cases,
        manifest_json=manifest_json,
        manifest_sha256=hashlib.sha256(manifest_json.encode("ascii")).hexdigest(),
    )


def inference_field_names() -> tuple[str, ...]:
    """Expose the allowed payload surface for independent audit/tests."""

    return tuple(field.name for field in fields(JACRUInferencePayload))


__all__ = [
    "JACRU_FAMILIES",
    "JACRU_FIXTURE_SCHEMA",
    "JACRU_SPLITS",
    "JACRUEvaluationPayload",
    "JACRUGeometry",
    "JACRUInferencePayload",
    "JACRUSyntheticCase",
    "JACRUSyntheticFixtureConfig",
    "JACRUSyntheticSuite",
    "build_jacru_fixture_suite",
    "build_jacru_geometry",
    "build_jacru_synthetic_case",
    "inference_field_names",
]
