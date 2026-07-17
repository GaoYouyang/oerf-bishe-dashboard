"""Frozen tiny fixture for PSU-B0 factor-majorizer Gate A attestation.

The fixture is deliberately synthetic and contains no truth field.  It exists
only to attest implementation mechanics on one view-local, single-scale
instance.  Performance, deployment, and scientific conclusions belong to a
later gate.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any

import torch
from torch import nn

from .psu_b0_absolute_measurement_factor import ExactAbsoluteMeasurementFactor
from .psu_b0_active_coordinates import CoordinateSupportGauge
from .psu_b0_factor_majorizer_pipeline import (
    PSUB0FactorMajorizerSetup,
    build_psu_b0_factor_majorizer_pipeline,
)
from .psu_b0_primal_dual import ForwardNeumannRegularizationOperator
from .psu_b0_reconstruction_interface import (
    PSUB0VoxelGradientOperator,
    TrilinearStencil,
    build_trilinear_stencil,
)


GATE_A_FIXTURE_SCHEMA = "psu-b0-gate-a-fixture-1.0"
GATE_A_CONFIG_SCHEMA = "psu-b0-gate-a-attestation-config-1.0"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GATE_A_CONFIG_PATH = (
    Path(__file__).resolve().parent
    / "configs"
    / "psu_b0_gate_a_attestation_v1.json"
)
GATE_A_THRESHOLD_CEILINGS = {
    "float64_adjoint_relative_error_max": 1e-8,
    "float64_dominance_violation_max": 1e-12,
    "float64_sum_relative_error_max": 1e-10,
    "float64_svd_slack": 1e-12,
    "float64_recurrence_relative_error_max": 1e-10,
    "mps_float32_field_relative_difference_max": 5e-4,
}
EXPECTED_TEST_NODE_MANIFEST_SHA256 = (
    "b638e53bba305318dcbb4242d00d17446929bf4b8af6c6d978d5951301376a5e"
)
EXPECTED_E1_MAPPING_SHA256 = (
    "6b85214c7d1c63fb7ee63dee0a4d5834e23a2a3487b49c9ba1e468747cbfe874"
)


def _reject_json_constant(raw: str) -> None:
    raise ValueError(f"non-finite Gate A JSON constant: {raw}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate Gate A JSON key: {key}")
        result[key] = value
    return result


def canonical_json_bytes(value: Any) -> bytes:
    """Return the single canonical encoding used by all Gate A fingerprints."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_repository_path(raw_path: str) -> Path:
    token = PurePosixPath(raw_path)
    if token.is_absolute() or ".." in token.parts or not token.parts:
        raise ValueError(f"unsafe repository-relative path: {raw_path!r}")
    path = (REPOSITORY_ROOT / token).resolve()
    if REPOSITORY_ROOT not in path.parents:
        raise ValueError(f"path escapes repository: {raw_path!r}")
    return path


def load_gate_a_config(path: Path = DEFAULT_GATE_A_CONFIG_PATH) -> dict[str, Any]:
    """Load and strictly validate the public, truth-free attestation config."""

    values = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_json_constant,
        object_pairs_hook=_unique_json_object,
    )
    if not isinstance(values, dict):
        raise ValueError("Gate A config must be a JSON object")
    if values.get("schema_version") != GATE_A_CONFIG_SCHEMA:
        raise ValueError("Gate A config schema mismatch")
    if values.get("evidence_scope") != "VIEW_LOCAL_SINGLE_FROZEN_SCALE_MECHANICS_ONLY":
        raise ValueError("Gate A evidence scope must remain mechanics-only")
    if values.get("scientific_claim_boundary") != "NO_GATE_B_NO_FRESH_NO_REAL_NO_WIN_CLAIM":
        raise ValueError("Gate A scientific claim boundary is not fail-closed")

    fixture = values.get("fixture")
    provenance = values.get("calibration_provenance")
    thresholds = values.get("thresholds")
    test_nodes = values.get("test_nodes")
    source_files = values.get("source_files")
    mapping = values.get("e1_test_mapping")
    if not isinstance(fixture, dict) or not isinstance(provenance, dict):
        raise ValueError("Gate A fixture and provenance must be objects")
    if not isinstance(thresholds, dict) or not thresholds:
        raise ValueError("Gate A thresholds must be a non-empty object")
    if set(thresholds) != set(GATE_A_THRESHOLD_CEILINGS):
        raise ValueError("Gate A threshold key set drifted")
    for name, ceiling in GATE_A_THRESHOLD_CEILINGS.items():
        value = thresholds[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 < float(value) <= ceiling
        ):
            raise ValueError(f"Gate A threshold is invalid or relaxed: {name}")
    if not isinstance(test_nodes, list) or not test_nodes:
        raise ValueError("Gate A test_nodes must be a non-empty list")
    if not isinstance(source_files, list) or not source_files:
        raise ValueError("Gate A source_files must be a non-empty list")
    if not isinstance(mapping, dict) or set(mapping) != {
        f"E1-{index:02d}" for index in range(1, 14)
    }:
        raise ValueError("Gate A must map every E1-01 through E1-13")

    forbidden_fixture_keys = {
        "truth",
        "truth_field",
        "morphology",
        "ground_truth",
        "clean_projection_from_truth",
    }
    if forbidden_fixture_keys & set(fixture):
        raise ValueError("Gate A mechanics fixture must not contain truth or morphology")
    expected_provenance = {
        "kind": "FROZEN_SYNTHETIC_MECHANICS_FIXTURE",
        "independent_flow_off_calibration": False,
        "truth_available_to_setup": False,
        "morphology_available_to_setup": False,
        "deployment_authorized": False,
    }
    if provenance != expected_provenance:
        raise ValueError("Gate A calibration provenance must remain synthetic and non-deployable")

    if any(not isinstance(node, str) or "::test_" not in node for node in test_nodes):
        raise ValueError("Gate A test_nodes must be explicit pytest node ids")
    if len(set(test_nodes)) != len(test_nodes):
        raise ValueError("Gate A test_nodes must be unique")
    if canonical_json_sha256(test_nodes) != EXPECTED_TEST_NODE_MANIFEST_SHA256:
        raise ValueError("Gate A test-node manifest drifted")
    for raw_path in source_files:
        if not isinstance(raw_path, str):
            raise ValueError("Gate A source_files entries must be strings")
        _safe_repository_path(raw_path)
    if len(set(source_files)) != len(source_files):
        raise ValueError("Gate A source_files must be unique")
    declared_source_set = set(source_files)
    for node in test_nodes:
        test_path = node.split("::", 1)[0]
        if test_path not in declared_source_set:
            raise ValueError(
                f"Gate A test node is not source-fingerprinted: {test_path}"
            )
    for gate, indices in mapping.items():
        if not isinstance(indices, list) or not indices:
            raise ValueError(f"{gate} needs at least one declared test node")
        if any(
            not isinstance(index, int) or not 0 <= index < len(test_nodes)
            for index in indices
        ):
            raise ValueError(f"{gate} contains an invalid test-node index")
    if canonical_json_sha256(mapping) != EXPECTED_E1_MAPPING_SHA256:
        raise ValueError("Gate A E1 mapping manifest drifted")

    required_fixture = {
        "grid_shape_zyx",
        "grid_minimum_xyz",
        "grid_maximum_xyz",
        "support_zero_flat_indices",
        "sample_points_xyz",
        "projection_u_xyz",
        "projection_v_xyz",
        "line_length",
        "system_constant",
        "whitening_matrix_by_view",
        "scale_by_view",
        "measurement_scale",
        "eta",
        "target_view_ray_uv",
        "regularization_weight",
        "penalty",
        "huber_delta",
        "theta",
        "iterations",
        "seed",
    }
    if set(fixture) != required_fixture:
        missing = sorted(required_fixture - set(fixture))
        extra = sorted(set(fixture) - required_fixture)
        raise ValueError(f"Gate A fixture keys drifted; missing={missing}, extra={extra}")
    if fixture["iterations"] != 6:
        raise ValueError("Gate A recurrence must remain exactly six steps")
    if fixture["penalty"] != "huber":
        raise ValueError("Gate A fixture must exercise the Huber branch")
    return deepcopy(values)


def gate_a_input_payload(config: dict[str, Any]) -> dict[str, Any]:
    """Return the exact scientific input subset covered by the input hash."""

    return {
        "schema_version": GATE_A_FIXTURE_SCHEMA,
        "evidence_scope": config["evidence_scope"],
        "scientific_claim_boundary": config["scientific_claim_boundary"],
        "fixture": deepcopy(config["fixture"]),
        "calibration_provenance": deepcopy(config["calibration_provenance"]),
        "thresholds": deepcopy(config["thresholds"]),
    }


def gate_a_source_hashes(config: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_path in config["source_files"]:
        path = _safe_repository_path(raw_path)
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Gate A source file is missing or symlinked: {raw_path}")
        result[raw_path] = file_sha256(path)
    return result


class _ViewLocalWhitening(nn.Module):
    """Minimal explicit view-local whitening contract for the frozen fixture."""

    def __init__(self, matrix: torch.Tensor, scale_by_view: torch.Tensor) -> None:
        super().__init__()
        if matrix.ndim != 3 or matrix.shape[1] != matrix.shape[2]:
            raise ValueError("fixture whitening matrix must be [view,detector,detector]")
        self.view_count = int(matrix.shape[0])
        detector_count = int(matrix.shape[1])
        if detector_count % 2:
            raise ValueError("fixture detector count must contain u/v pairs")
        self.rays_per_view = detector_count // 2
        self.whitening_block_scope = "view_local"
        self.independent_whitening_blocks = True
        self.cross_view_covariance = False
        self.cross_view_coupling = False
        self.cross_view_covariance_supported = False
        self.cross_view_coupling_supported = False
        self.covariance_block_ids = tuple(range(self.view_count))
        self.register_buffer("matrix", matrix.contiguous())
        self.register_buffer("scale_by_view", scale_by_view.contiguous())


@dataclass(frozen=True)
class GateAFixture:
    config: dict[str, Any]
    setup: PSUB0FactorMajorizerSetup
    target: torch.Tensor


def build_gate_a_fixture(
    config: dict[str, Any],
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> GateAFixture:
    """Build the frozen factor pipeline without accepting truth-like inputs."""

    values = deepcopy(config)
    if values.get("schema_version") != GATE_A_CONFIG_SCHEMA:
        raise ValueError("Gate A config must be loaded and schema-validated")
    fixture = values["fixture"]
    target_device = torch.device(device)
    shape = tuple(int(value) for value in fixture["grid_shape_zyx"])
    support = torch.ones(shape, dtype=torch.float64, device="cpu")
    flat_support = support.reshape(-1)
    for raw_index in fixture["support_zero_flat_indices"]:
        index = int(raw_index)
        if not 0 <= index < flat_support.numel():
            raise ValueError("support_zero_flat_indices is outside the grid")
        flat_support[index] = 0.0
    gauge = CoordinateSupportGauge.from_support(support)

    cpu_stencil = build_trilinear_stencil(
        fixture["sample_points_xyz"],
        grid_shape=shape,
        grid_minimum_xyz=fixture["grid_minimum_xyz"],
        grid_maximum_xyz=fixture["grid_maximum_xyz"],
        dtype=dtype,
    )
    stencil = TrilinearStencil(
        indices=cpu_stencil.indices.to(device=target_device),
        weights=cpu_stencil.weights.to(device=target_device, dtype=dtype),
        valid=cpu_stencil.valid.to(device=target_device),
        grid_shape=shape,
    )
    projection_u = torch.as_tensor(
        fixture["projection_u_xyz"], dtype=dtype, device=target_device
    )
    projection_v = torch.as_tensor(
        fixture["projection_v_xyz"], dtype=dtype, device=target_device
    )
    support_on_device = support.to(device=target_device, dtype=dtype)
    voxel = PSUB0VoxelGradientOperator(
        stencil=stencil,
        projection_u_xyz=projection_u,
        projection_v_xyz=projection_v,
        line_length=torch.as_tensor(
            fixture["line_length"], dtype=dtype, device=target_device
        ),
        system_constant=torch.as_tensor(
            fixture["system_constant"], dtype=dtype, device=target_device
        ),
        grid_minimum_xyz=fixture["grid_minimum_xyz"],
        grid_maximum_xyz=fixture["grid_maximum_xyz"],
        support=support_on_device,
        dtype=dtype,
    )
    whitening = _ViewLocalWhitening(
        torch.as_tensor(
            fixture["whitening_matrix_by_view"],
            dtype=dtype,
            device=target_device,
        ),
        torch.as_tensor(
            fixture["scale_by_view"], dtype=dtype, device=target_device
        ),
    )
    factor = ExactAbsoluteMeasurementFactor(
        whitening,
        projection_u_xyz=projection_u,
        projection_v_xyz=projection_v,
        ray_scale=voxel.ray_scale,
        sample_count=stencil.sample_count,
        measurement_scale=float(fixture["measurement_scale"]),
    )
    regularization = ForwardNeumannRegularizationOperator(voxel.spacing_xyz)
    setup = build_psu_b0_factor_majorizer_pipeline(
        gauge=gauge,
        voxel_operator=voxel,
        measurement_factor=factor,
        regularization_operator=regularization,
        eta=float(fixture["eta"]),
    )
    target = torch.as_tensor(
        fixture["target_view_ray_uv"],
        dtype=dtype,
        device=target_device,
    )
    expected_target = (
        setup.pipeline.view_count,
        setup.pipeline.rays_per_view,
        2,
    )
    if tuple(target.shape) != expected_target:
        raise ValueError("target_view_ray_uv does not match the factor shape")
    return GateAFixture(values, setup, target)


__all__ = [
    "DEFAULT_GATE_A_CONFIG_PATH",
    "GATE_A_CONFIG_SCHEMA",
    "GATE_A_FIXTURE_SCHEMA",
    "GateAFixture",
    "REPOSITORY_ROOT",
    "build_gate_a_fixture",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "file_sha256",
    "gate_a_input_payload",
    "gate_a_source_hashes",
    "load_gate_a_config",
]
