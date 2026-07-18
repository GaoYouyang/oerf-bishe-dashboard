#!/usr/bin/env python3
"""Independently validate the N2-PVGR N5-D4b 32-cell derivative census."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from demo_t16_operator.analytic_bost_phantoms import (  # noqa: E402
    analytic_phantom_grid,
    make_analytic_phantom,
)
from demo_t16_operator.automatic_discrete_multifidelity import (  # noqa: E402
    SyntheticRayRig,
)
from demo_t16_operator.field_dependent_ray import sample_pupil_sobol  # noqa: E402
from site_tools.validate_n2_pvgr_n5_d4_tiny_field_derivative import (  # noqa: E402
    _closures as independent_closures,
)
from site_tools.validate_n2_pvgr_n5_d4_tiny_field_derivative import (  # noqa: E402
    _expand_audit_cells as independently_expand_audit_cells,
)
from site_tools.validate_n2_pvgr_n5_d4_tiny_field_derivative import (  # noqa: E402
    _independent_audit,
    _independent_cotangent,
    _independent_direction,
    _independent_program_signature,
    _metrics,
    _rig_from_case,
    _stable_seed,
)


DEFAULT_CONFIG = (
    ROOT / "demo_t16_operator/configs/"
    "n2_pvgr_n5_d4b_population_field_derivative_preregistered_v1.json"
)

DERIVATIVE_GATE_KEYS = (
    "maximum_dot_relative_defect",
    "minimum_dot_signal_absolute",
    "maximum_best_finite_difference_relative_l2",
    "maximum_required_h_finite_difference_relative_l2",
    "maximum_repeat_output_relative_l2",
    "maximum_raw_structural_output_relative_l2",
    "maximum_raw_structural_jvp_relative_l2",
    "maximum_raw_structural_vjp_relative_l2",
    "maximum_raw_paired_output_relative_l2",
    "maximum_raw_paired_jvp_relative_l2",
    "maximum_raw_paired_vjp_relative_l2",
    "structural_absolute_tolerance_floor",
    "required_finite_fraction",
    "minimum_domain_margin",
    "minimum_stencil_margin",
    "maximum_direction_norm_error",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _array(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(value, dtype="<f8"))


def _stored_float64_hash(value: np.ndarray) -> str:
    if value.dtype != np.dtype("<f8") or not value.flags.c_contiguous:
        raise ValueError(
            "N5-D4b stored array is not C-contiguous little-endian float64"
        )
    return _sha256_bytes(value.tobytes(order="C"))


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], cwd=ROOT, check=check, capture_output=True)


def _same(left: float, right: float, *, label: str, atol: float = 1e-18) -> None:
    if not np.isclose(float(left), float(right), rtol=2e-13, atol=atol):
        raise ValueError(f"N5-D4b {label} drifted: {left} != {right}")


def _path_occupied(path: Path) -> bool:
    return os.path.lexists(path)


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_exclusive(path: Path, value: object) -> None:
    temporary = _temporary_sibling(path)
    if _path_occupied(path) or _path_occupied(temporary):
        raise FileExistsError(f"refusing to replace N5-D4b validation artifact: {path}")
    payload = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, path)
    _fsync_directory(path.parent)
    temporary.unlink()
    _fsync_directory(path.parent)


def _recomputed_budget(config: dict[str, Any]) -> dict[str, int]:
    rays = int(config["ray_count"])
    steps = int(config["step_count"])
    cells = int(config["population_contract"]["expected_cell_count"])
    directions = int(config["direction_count_per_cell"])
    maps = len(config["map_order"])
    h_count = len(config["finite_difference"]["h_values"])
    closure_invocations = 2 + 2 * h_count
    contexts = cells * directions
    derivative_queries = (
        contexts * closure_invocations * (35 + 7 + 42 + 42) * rays * steps
    )
    curved_dispatches = 28 * steps + 7
    straight_dispatches = 7
    derivative_dispatches = (
        contexts
        * closure_invocations
        * (
            curved_dispatches
            + straight_dispatches
            + curved_dispatches
            + straight_dispatches
            + 28 * steps
            + 1
        )
    )
    signatures = contexts * (1 + 2 * h_count)
    topology_queries = signatures * 70 * rays * steps
    topology_dispatches = signatures * (28 * steps + 4 * steps + 2)
    return {
        "derivative_map_logical_queries": derivative_queries,
        "derivative_map_interpolation_dispatches": derivative_dispatches,
        "topology_logical_queries": topology_queries,
        "topology_interpolation_dispatches": topology_dispatches,
        "d4b_total_logical_queries": derivative_queries + topology_queries,
        "d4b_total_interpolation_dispatches": derivative_dispatches
        + topology_dispatches,
        "map_closure_invocations": contexts * maps * closure_invocations,
        "jvp_sweeps": contexts * maps,
        "vjp_sweeps": contexts * maps,
        "finite_difference_forward_calls": contexts * maps * 2 * h_count,
        "topology_signature_invocations": signatures,
    }


def _validate_static_config(config: dict[str, Any]) -> None:
    if config.get("schema") != (
        "n2-pvgr-n5-d4b-population-field-derivative-preregistered-1.0"
    ):
        raise ValueError("N5-D4b config schema drifted")
    if (config.get("device"), config.get("dtype")) != ("cpu", "float64"):
        raise ValueError("N5-D4b config must remain CPU float64")
    if [
        int(config[key])
        for key in (
            "ray_count",
            "source_population_count",
            "step_count",
            "direction_count_per_cell",
        )
    ] != [4, 256, 16, 2]:
        raise ValueError("N5-D4b shape contract drifted")
    population = config["population_contract"]
    if int(population["expected_cell_count"]) != 32:
        raise ValueError("N5-D4b population count drifted")
    if population["expected_pair_ids"] != [f"p{i:02d}" for i in range(1, 17)]:
        raise ValueError("N5-D4b pair order drifted")
    if population["expected_family_counts"] != {"smooth": 20, "wrinkled": 12}:
        raise ValueError("N5-D4b family census drifted")
    if population["expected_stress_counts"] != {"1": 12, "3": 10, "10": 10}:
        raise ValueError("N5-D4b stress census drifted")
    if population["expected_d3_reference_method_counts"] != {
        "paired_neumaier": 2,
        "raw_separate_subtraction": 30,
    }:
        raise ValueError("N5-D4b D3 reference-method census drifted")
    if population["required_reporting_levels"] != [
        "cell",
        "pair_cluster",
        "field_unit_cluster",
    ]:
        raise ValueError("N5-D4b cluster-reporting contract drifted")
    if config["map_order"] != [
        "curved_detector",
        "straight_detector",
        "raw_curved_minus_straight",
        "paired_neumaier_residual",
    ]:
        raise ValueError("N5-D4b map contract drifted")
    if config["direction_contract"]["field_direction_modes_in_order"] != [
        "smooth_avg_pool3d_kernel3_stride1_padding1",
        "raw_torch_randn",
    ]:
        raise ValueError("N5-D4b direction modes drifted")
    if [
        int(config["direction_contract"]["seed_base"]),
        int(config["direction_contract"]["cotangent_seed_base"]),
    ] != [66001, 66101]:
        raise ValueError("N5-D4b seed bases drifted")
    h_values = [0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]
    if config["finite_difference"]["h_values"] != h_values:
        raise ValueError("N5-D4b h-grid drifted")
    if config["finite_difference"]["h_values_float64_hex"] != [
        value.hex() for value in h_values
    ]:
        raise ValueError("N5-D4b h bytes drifted")
    if config["finite_difference"]["required_h_values"] != [0.01, 0.003, 0.001]:
        raise ValueError("N5-D4b required h-grid drifted")

    parent_d4 = _read_json(_resolve(str(config["parent_d4_config"])))
    for key in (
        "device",
        "dtype",
        "ray_count",
        "source_population_count",
        "step_count",
        "direction_count_per_cell",
        "map_order",
        "observable_contract",
        "finite_difference",
        "topology_contract",
    ):
        if config[key] != parent_d4[key]:
            raise ValueError(f"N5-D4b changed parent D4 contract: {key}")
    for key in DERIVATIVE_GATE_KEYS:
        if config["gates"].get(key) != parent_d4["gates"].get(key):
            raise ValueError(f"N5-D4b changed parent D4 threshold: {key}")
    if any(config["claim_authorizations"].values()):
        raise ValueError("N5-D4b preregistered authorizations drifted")
    bundle = _resolve(str(config["pre_registration_bundle"]))
    if {
        _resolve(str(config["pre_registration_attestation"])).parent,
        _resolve(str(config["frozen_input_archive"])).parent,
        _resolve(str(config["pre_registration_ready_marker"])).parent,
    } != {bundle}:
        raise ValueError("N5-D4b preregistration bundle path contract drifted")

    budget = _recomputed_budget(config)
    contract = config["budget_contract"]
    mapping = {
        "expected_derivative_map_logical_queries": "derivative_map_logical_queries",
        "expected_derivative_map_interpolation_dispatches": "derivative_map_interpolation_dispatches",
        "expected_topology_logical_queries": "topology_logical_queries",
        "expected_topology_interpolation_dispatches": "topology_interpolation_dispatches",
        "expected_total_logical_queries": "d4b_total_logical_queries",
        "expected_total_interpolation_dispatches": "d4b_total_interpolation_dispatches",
        "expected_map_closure_invocations": "map_closure_invocations",
        "expected_jvp_sweeps": "jvp_sweeps",
        "expected_vjp_sweeps": "vjp_sweeps",
        "expected_finite_difference_forward_calls": "finite_difference_forward_calls",
        "expected_topology_signature_invocations": "topology_signature_invocations",
    }
    for config_key, result_key in mapping.items():
        if int(contract[config_key]) != budget[result_key]:
            raise ValueError(
                f"N5-D4b independently recomputed budget drifted: {config_key}"
            )
    if (
        int(
            contract[
                "lineage_total_logical_queries_including_once_only_parent_toy_control"
            ]
        )
        != budget["d4b_total_logical_queries"] + 3360
    ):
        raise ValueError("N5-D4b lineage query budget drifted")
    if (
        int(
            contract[
                "lineage_total_interpolation_dispatches_including_once_only_parent_toy_control"
            ]
        )
        != budget["d4b_total_interpolation_dispatches"] + 686
    ):
        raise ValueError("N5-D4b lineage dispatch budget drifted")


def _verify_attestation(
    config: dict[str, Any], config_path: Path
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    bundle_path = _resolve(str(config["pre_registration_bundle"]))
    attestation_path = _resolve(str(config["pre_registration_attestation"]))
    archive_path = _resolve(str(config["frozen_input_archive"]))
    ready_path = _resolve(str(config["pre_registration_ready_marker"]))
    if (
        not bundle_path.is_dir()
        or {attestation_path.parent, archive_path.parent, ready_path.parent}
        != {bundle_path}
        or not attestation_path.is_file()
        or not archive_path.is_file()
        or not ready_path.is_file()
    ):
        raise FileNotFoundError(
            "N5-D4b complete atomic preregistration bundle is missing"
        )
    attestation = _read_json(attestation_path)
    ready = _read_json(ready_path)
    if attestation.get("schema") != (
        "n2-pvgr-n5-d4b-population-field-derivative-attestation-1.0"
    ):
        raise ValueError("N5-D4b attestation schema drifted")
    if attestation.get("formal_results_absent_at_creation") is not True:
        raise ValueError("N5-D4b output-absence claim drifted")
    if attestation.get("formal_work_output_absent_at_creation") is not True:
        raise ValueError("N5-D4b work-output-absence claim drifted")
    if attestation.get("pre_registration_bundle") != config["pre_registration_bundle"]:
        raise ValueError("N5-D4b attested bundle path drifted")
    if (
        attestation.get("pre_registration_ready_marker")
        != config["pre_registration_ready_marker"]
    ):
        raise ValueError("N5-D4b attested READY path drifted")
    if attestation.get("bundle_publication") != (
        "single_atomic_directory_rename_after_READY"
    ):
        raise ValueError("N5-D4b atomic publication contract drifted")
    if _sha256_bytes(config_path.read_bytes()) != attestation.get("config_sha256"):
        raise ValueError("N5-D4b attested config hash drifted")
    protocol_commit = str(attestation.get("protocol_commit", ""))
    if len(protocol_commit) != 40:
        raise ValueError("N5-D4b protocol commit drifted")
    if _git(
        "merge-base", "--is-ancestor", protocol_commit, "HEAD", check=False
    ).returncode:
        raise ValueError("N5-D4b protocol commit is not an ancestor of HEAD")
    if ready != {
        "schema": "n2-pvgr-n5-d4b-preregistration-ready-1.0",
        "protocol_commit": protocol_commit,
        "config_sha256": _sha256(config_path),
        "attestation_path": config["pre_registration_attestation"],
        "attestation_sha256": _sha256(attestation_path),
        "frozen_input_archive_path": config["frozen_input_archive"],
        "frozen_input_archive_sha256": _sha256(archive_path),
        "publication_rule": "atomically_rename_complete_staging_directory",
    }:
        raise ValueError("N5-D4b READY marker or atomic bundle binding drifted")
    if set(attestation["attested_files"]) != set(config["attested_files"]):
        raise ValueError("N5-D4b attested file set drifted")
    for key, relative in config["attested_files"].items():
        path = _resolve(str(relative))
        record = attestation["attested_files"][key]
        frozen = _git("show", f"{protocol_commit}:{relative}")
        if (
            record.get("path") != relative
            or _sha256(path) != record.get("sha256")
            or frozen.returncode != 0
            or _sha256_bytes(frozen.stdout) != record.get("sha256")
        ):
            raise ValueError(f"N5-D4b attested dependency drifted: {key}")
    if attestation.get("frozen_input_archive") != config["frozen_input_archive"]:
        raise ValueError("N5-D4b frozen archive path drifted")
    if _sha256(archive_path) != attestation.get("frozen_input_archive_sha256"):
        raise ValueError("N5-D4b frozen archive hash drifted")
    status_paths = [str(value) for value in config["attested_files"].values()]
    status_paths.extend((str(attestation_path), str(archive_path), str(ready_path)))
    if _git("status", "--porcelain", "--", *status_paths).stdout.decode().strip():
        raise ValueError("N5-D4b preregistration bundle has uncommitted changes")
    with np.load(archive_path, allow_pickle=False) as handle:
        archive = {key: np.array(handle[key], copy=True) for key in handle.files}
    inventory = attestation["frozen_input_inventory"]
    if set(archive) != set(inventory):
        raise ValueError("N5-D4b frozen archive inventory drifted")
    for key, value in archive.items():
        record = inventory[key]
        if (
            list(value.shape) != record["shape"]
            or str(value.dtype) != record["dtype"]
            or _stored_float64_hash(value) != record["sha256_float64_le_c_order"]
        ):
            raise ValueError(f"N5-D4b frozen array drifted: {key}")
    metadata = attestation["frozen_input_metadata"]
    if _sha256_bytes(_canonical_json(metadata)) != attestation.get(
        "frozen_input_metadata_sha256"
    ):
        raise ValueError("N5-D4b frozen metadata hash drifted")
    return attestation, archive


def _counter(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _reconstruct_contexts(
    config: dict[str, Any],
    attestation: dict[str, Any],
    archive: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    parent_d4 = _read_json(_resolve(str(config["parent_d4_result"])))
    parent_d4_validation = _read_json(_resolve(str(config["parent_d4_validation"])))
    if (
        parent_d4.get("machine_decision")
        != config["decision_contract"]["parent_d4_required_decision"]
    ):
        raise ValueError("N5-D4b parent D4 decision drifted")
    if parent_d4_validation.get("valid") is not True:
        raise ValueError("N5-D4b parent D4 validation drifted")
    d3_result = _read_json(_resolve(str(config["parent_d3_result"])))
    d3_pack = _read_json(_resolve(str(config["parent_d3_pack"])))
    d3_validation = _read_json(_resolve(str(config["parent_d3_validation"])))
    if d3_result.get("machine_decision") != "D3_VALID_MIXED_RESIDUAL_REFERENCE_ONLY":
        raise ValueError("N5-D4b parent D3 decision drifted")
    if d3_validation.get("valid") is not True:
        raise ValueError("N5-D4b parent D3 validation drifted")
    population = config["population_contract"]
    if d3_pack.get("cell_order_sha256") != population["expected_d3_cell_order_sha256"]:
        raise ValueError("N5-D4b D3 cell order drifted")
    packed_cells = list(d3_pack["cells"])
    scientific = _read_json(_resolve(str(config["scientific_n4_config"])))
    parent_n3 = _read_json(_resolve(str(config["parent_n3_config"])))
    expanded = independently_expand_audit_cells(scientific, parent_n3)
    if [row["cell_id"] for row in expanded] != [row["cell_id"] for row in packed_cells]:
        raise ValueError("N5-D4b D3/N4 cell order drifted")
    metadata = attestation["frozen_input_metadata"]
    metadata_contexts = list(metadata["contexts"])
    if len(metadata_contexts) != 32:
        raise ValueError("N5-D4b frozen metadata population drifted")
    if [row["cell_id"] for row in metadata_contexts] != [
        row["cell_id"] for row in packed_cells
    ]:
        raise ValueError("N5-D4b frozen metadata order drifted")

    family_counts = _counter([str(row["family_id"]) for row in packed_cells])
    field_unit_counts = _counter([str(row["field_unit_id"]) for row in packed_cells])
    stress_counts = _counter(
        [f"{float(row['dimensionless_stress_multiplier']):g}" for row in packed_cells]
    )
    reference_method_counts = _counter(
        [str(row["reference_method"]) for row in packed_cells]
    )
    if family_counts != {
        str(key): int(value)
        for key, value in population["expected_family_counts"].items()
    }:
        raise ValueError("N5-D4b independent family census drifted")
    if field_unit_counts != {
        str(key): int(value)
        for key, value in population["expected_field_unit_counts"].items()
    }:
        raise ValueError("N5-D4b independent field-unit census drifted")
    if stress_counts != {
        str(key): int(value)
        for key, value in population["expected_stress_counts"].items()
    }:
        raise ValueError("N5-D4b independent stress census drifted")
    if reference_method_counts != {
        str(key): int(value)
        for key, value in population["expected_d3_reference_method_counts"].items()
    }:
        raise ValueError("N5-D4b independent D3 reference-method census drifted")
    if metadata["family_counts"] != family_counts:
        raise ValueError("N5-D4b frozen family summary drifted")
    if metadata["field_unit_counts"] != field_unit_counts:
        raise ValueError("N5-D4b frozen field-unit summary drifted")
    if metadata["stress_counts"] != stress_counts:
        raise ValueError("N5-D4b frozen stress summary drifted")
    if metadata["d3_reference_method_counts"] != reference_method_counts:
        raise ValueError("N5-D4b frozen D3 reference-method summary drifted")

    source = _read_json(_resolve(str(config["source_config"])))
    source["population_count"] = int(config["source_population_count"])
    contract = config["direction_contract"]
    output: dict[str, dict[str, Any]] = {}
    for index, (packed, cell, frozen) in enumerate(
        zip(packed_cells, expanded, metadata_contexts, strict=True)
    ):
        cell_id = str(packed["cell_id"])
        if int(packed["pack_index"]) != index or int(frozen["cell_index"]) != index:
            raise ValueError("N5-D4b pack index drifted")
        if packed["identity_sha256"] != frozen["d3_identity_sha256"]:
            raise ValueError("N5-D4b D3 identity drifted")
        for key in ("pair_id", "role", "field_unit_id", "family_id"):
            if packed[key] != cell[key] or packed[key] != frozen[key]:
                raise ValueError(f"N5-D4b cell metadata drifted: {cell_id}/{key}")
        spec = make_analytic_phantom(
            family=str(cell["phantom_family"]), seed=int(cell["phantom_seed"])
        )
        field = analytic_phantom_grid(
            spec,
            grid_shape=tuple(int(value) for value in source["grid_shape_zyx"]),
            dtype=torch.float64,
            device="cpu",
        ).field
        pupil = sample_pupil_sobol(
            int(source["population_count"]),
            seed=_stable_seed(
                int(source["seed_roles"]["population_state_base"]), cell["id"]
            ),
        )
        rays = pupil[: int(config["ray_count"])]
        if not np.array_equal(_array(field), archive[frozen["field_key"]]):
            raise ValueError(f"N5-D4b independently rebuilt field drifted: {cell_id}")
        if not np.array_equal(_array(rays), archive[frozen["rays_key"]]):
            raise ValueError(f"N5-D4b independently rebuilt rays drifted: {cell_id}")
        rig = _rig_from_case(cell)
        if frozen["rig"] != asdict(rig):
            raise ValueError("N5-D4b frozen rig drifted")
        scale = float(source["base_refractivity_scale"]) * float(
            cell["dimensionless_stress_multiplier"]
        )
        support = float(
            source["certificate"]["support_threshold_fraction_of_grid_peak"]
        ) * float(torch.max(torch.abs(field)))
        _same(frozen["refractivity_scale"], scale, label="refractivity scale")
        _same(frozen["support_threshold"], support, label="support threshold")
        directions = []
        for direction_index, entry in enumerate(frozen["directions"]):
            seed = _stable_seed(int(contract["seed_base"]), cell_id, direction_index)
            cotangent_seed = _stable_seed(
                int(contract["cotangent_seed_base"]), cell_id, direction_index
            )
            direction = _independent_direction(
                tuple(int(v) for v in config["observable_contract"]["field_shape"]),
                seed=seed,
                boundary_layers=int(contract["field_direction_boundary_layers_zeroed"]),
                mode=str(contract["field_direction_modes_in_order"][direction_index]),
            )
            cotangent = _independent_cotangent(
                tuple(int(v) for v in config["observable_contract"]["output_shape"]),
                seed=cotangent_seed,
            )
            if (
                entry["direction_seed"] != seed
                or entry["cotangent_seed"] != cotangent_seed
            ):
                raise ValueError("N5-D4b frozen RNG provenance drifted")
            if not np.array_equal(direction, archive[entry["direction_key"]]):
                raise ValueError("N5-D4b independently rebuilt direction drifted")
            if not np.array_equal(cotangent, archive[entry["cotangent_key"]]):
                raise ValueError("N5-D4b independently rebuilt cotangent drifted")
            directions.append((direction, cotangent))
        output[cell_id] = {
            "cell_index": index,
            "cell": cell,
            "packed": packed,
            "metadata": frozen,
            "field": _array(field),
            "rays": _array(rays),
            "directions": directions,
        }
    return output


def _verify_array(
    stored: dict[str, np.ndarray],
    verified: set[str],
    key: str,
    expected: torch.Tensor,
) -> None:
    if key not in stored:
        raise ValueError(f"N5-D4b result array is missing: {key}")
    candidate = stored[key]
    reference = _array(expected)
    candidate_hash = _stored_float64_hash(candidate)
    reference_hash = _stored_float64_hash(reference)
    if (
        candidate.shape != reference.shape
        or candidate_hash != reference_hash
        or candidate.tobytes(order="C") != reference.tobytes(order="C")
    ):
        maximum = float(np.max(np.abs(candidate - reference)))
        raise ValueError(
            "N5-D4b independently recomputed array is not byte-exact: "
            f"{key}, max={maximum}, stored={candidate_hash}, "
            f"recomputed={reference_hash}"
        )
    verified.add(key)


def _cluster_summary(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    output = []
    for cluster_id, members in sorted(groups.items()):
        maps = [
            bool(payload["map_gate"])
            for row in members
            for payload in row["maps"].values()
        ]
        structures = [
            bool(payload["gate"])
            for row in members
            for payload in row["structure"].values()
        ]
        topology = [bool(row["topology_gate"]) for row in members]
        output.append(
            {
                "cluster_id": cluster_id,
                "direction_context_count": len(members),
                "map_context_count": len(maps),
                "passing_map_context_count": sum(maps),
                "structural_control_count": len(structures),
                "passing_structural_control_count": sum(structures),
                "topology_context_count": len(topology),
                "passing_topology_context_count": sum(topology),
                "all_gates_pass": bool(all(maps) and all(structures) and all(topology)),
            }
        )
    return output


def _extrema(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, float]:
    maps = [payload for row in rows for payload in row["maps"].values()]
    required = [
        level for payload in maps for level in payload["levels"] if level["required"]
    ]
    signatures = [row["topology"]["base"] for row in rows] + [
        item for row in rows for item in row["topology"]["perturbations"]
    ]
    floor = float(config["gates"]["minimum_dot_signal_absolute"])
    return {
        "worst_dot_relative_defect": max(
            float(item["dot_relative_defect"]) for item in maps
        ),
        "minimum_dot_signal_absolute": min(float(item["dot_signal"]) for item in maps),
        "minimum_dot_signal_over_floor": min(
            float(item["dot_signal"]) / floor for item in maps
        ),
        "worst_best_h_finite_difference_relative_l2": max(
            min(float(level["relative_error"]) for level in item["levels"])
            for item in maps
        ),
        "worst_required_h_finite_difference_relative_l2": max(
            float(item["relative_error"]) for item in required
        ),
        "minimum_domain_margin": min(
            float(item["minimum_domain_margin"]) for item in signatures
        ),
        "minimum_stencil_margin": min(
            float(item["minimum_stencil_margin"]) for item in signatures
        ),
        "minimum_frustum_margin": min(
            float(item["minimum_frustum_margin"]) for item in signatures
        ),
        "maximum_direction_norm_error": max(
            float(item["maximum_direction_norm_error"]) for item in signatures
        ),
    }


def _verify_manifest(output: Path, config_path: Path, config: dict[str, Any]) -> None:
    manifest = _read_json(output / "manifest.json")
    if manifest.get("schema") != (
        "n2-pvgr-n5-d4b-population-field-derivative-manifest-1.0"
    ):
        raise ValueError("N5-D4b manifest schema drifted")
    if manifest.get("post_validation_artifacts_excluded_from_manifest") != [
        "validation_report.json"
    ]:
        raise ValueError("N5-D4b post-validation manifest boundary drifted")
    expected = {
        "config": config_path,
        "attestation": _resolve(str(config["pre_registration_attestation"])),
        "frozen_input_archive": _resolve(str(config["frozen_input_archive"])),
        "pre_registration_ready_marker": _resolve(
            str(config["pre_registration_ready_marker"])
        ),
    }
    for key, path in expected.items():
        if manifest["files"].get(key) != {
            "path": _relative(path),
            "sha256": _sha256(path),
        }:
            raise ValueError(f"N5-D4b manifest dependency drifted: {key}")
    for name in (
        "array_inventory.json",
        "config_snapshot.json",
        "derivative_arrays.npz",
        "n2_pvgr_n5_d4b_population_field_derivative.png",
        "result.json",
        "rows.json",
        "summary.md",
    ):
        path = output / name
        if manifest["files"].get(name) != {
            "path": _relative(path),
            "sha256": _sha256(path),
        }:
            raise ValueError(f"N5-D4b output manifest drifted: {name}")


def _verify_figure(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
        extrema = image.convert("L").getextrema()
    if width < 1200 or height < 700 or extrema[0] == extrema[1]:
        raise ValueError("N5-D4b figure is blank or undersized")
    return {
        "path": _relative(path),
        "width": width,
        "height": height,
        "sha256": _sha256(path),
    }


def validate(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = _read_json(config_path)
    _validate_static_config(config)
    attestation, frozen_archive = _verify_attestation(config, config_path)
    contexts = _reconstruct_contexts(config, attestation, frozen_archive)
    output = _resolve(str(config["formal_output"]))
    if not output.is_dir():
        raise FileNotFoundError("N5-D4b formal output is missing")
    if _path_occupied(_resolve(str(config["formal_work_output"]))):
        raise ValueError("N5-D4b work output remains after publication")
    if _read_json(output / "config_snapshot.json") != config:
        raise ValueError("N5-D4b config snapshot drifted")
    result = _read_json(output / "result.json")
    rows = _read_json(output / "rows.json")
    inventory = _read_json(output / "array_inventory.json")
    with np.load(output / "derivative_arrays.npz", allow_pickle=False) as handle:
        stored_arrays = {key: np.array(handle[key], copy=True) for key in handle.files}
    if inventory.get("schema") != "n2-pvgr-n5-d4b-derivative-array-inventory-1.0":
        raise ValueError("N5-D4b derivative inventory schema drifted")
    if int(inventory.get("array_count", -1)) != len(stored_arrays):
        raise ValueError("N5-D4b derivative inventory count drifted")
    if set(stored_arrays) != set(inventory["arrays"]):
        raise ValueError("N5-D4b derivative inventory keys drifted")
    for key, value in stored_arrays.items():
        entry = inventory["arrays"][key]
        if (
            list(value.shape) != entry["shape"]
            or str(value.dtype) != entry["dtype"]
            or value.dtype.str != entry["dtype_str"]
            or bool(value.flags.c_contiguous) != entry["c_contiguous"]
            or _stored_float64_hash(value) != entry["sha256_float64_le_c_order"]
        ):
            raise ValueError(f"N5-D4b derivative-array bytes drifted: {key}")
    if len(rows) != 64:
        raise ValueError("N5-D4b direction-row count drifted")
    rows_by_key = {(row["cell_id"], int(row["direction_index"])): row for row in rows}
    if len(rows_by_key) != 64:
        raise ValueError("N5-D4b direction-row keys are not unique")

    verified_arrays: set[str] = set()
    map_gates: list[bool] = []
    structure_gates: list[bool] = []
    topology_gates: list[bool] = []
    h_values = [float(value) for value in config["finite_difference"]["h_values"]]
    required_h = {
        float(value) for value in config["finite_difference"]["required_h_values"]
    }
    gates = config["gates"]
    eps = torch.finfo(torch.float64).eps

    for cell_id, context in contexts.items():
        metadata = context["metadata"]
        field = torch.from_numpy(context["field"].copy()).to(torch.float64)
        rays = torch.from_numpy(context["rays"].copy()).to(torch.float64)
        rig = SyntheticRayRig(**metadata["rig"])
        closures = independent_closures(
            rays,
            rig,
            difference_step=float(metadata["difference_step"]),
            refractivity_scale=float(metadata["refractivity_scale"]),
            step_count=int(config["step_count"]),
        )
        signature_kwargs = {
            "difference_step": float(metadata["difference_step"]),
            "refractivity_scale": float(metadata["refractivity_scale"]),
            "step_count": int(config["step_count"]),
            "support_threshold": float(metadata["support_threshold"]),
            "frustum_half_width_u": float(metadata["frustum_half_width_u"]),
            "frustum_half_width_v": float(metadata["frustum_half_width_v"]),
        }
        for direction_index, (direction_array, cotangent_array) in enumerate(
            context["directions"]
        ):
            row = rows_by_key[(cell_id, direction_index)]
            for key in (
                "pair_id",
                "role",
                "field_unit_id",
                "family_id",
                "d3_identity_sha256",
            ):
                if row[key] != metadata[key]:
                    raise ValueError(f"N5-D4b result-row metadata drifted: {key}")
            direction = torch.from_numpy(direction_array.copy()).to(torch.float64)
            cotangent = torch.from_numpy(cotangent_array.copy()).to(torch.float64)
            base = _independent_program_signature(field, rays, rig, **signature_kwargs)
            if row["topology"]["base"] != base:
                raise ValueError("N5-D4b independently rebuilt base topology drifted")
            topology_rows = {
                (float(item["h"]), item["side"]): item
                for item in row["topology"]["perturbations"]
            }
            if len(topology_rows) != 14:
                raise ValueError("N5-D4b topology perturbation count drifted")
            topology_pass = True
            for h in h_values:
                for sign, side in ((1.0, "plus"), (-1.0, "minus")):
                    signature = _independent_program_signature(
                        field + sign * h * direction, rays, rig, **signature_kwargs
                    )
                    stored = topology_rows[(h, side)]
                    for key in (
                        "digest_sha256",
                        "interpolation_cells_sha256",
                        "support_bits_sha256",
                        "frustum_margin_signs_sha256",
                    ):
                        if stored[key] != signature[key]:
                            raise ValueError(
                                f"N5-D4b perturbed topology drifted: {key}"
                            )
                    same = signature["digest_sha256"] == base["digest_sha256"]
                    margins = bool(
                        signature["minimum_domain_margin"]
                        >= float(gates["minimum_domain_margin"])
                        and signature["minimum_stencil_margin"]
                        >= float(gates["minimum_stencil_margin"])
                        and signature["maximum_direction_norm_error"]
                        <= float(gates["maximum_direction_norm_error"])
                    )
                    if (
                        bool(stored["matches_base"]) != same
                        or bool(stored["margins_gate"]) != margins
                    ):
                        raise ValueError("N5-D4b topology gate serialization drifted")
                    topology_pass = topology_pass and same and margins
            if bool(row["topology_gate"]) != topology_pass:
                raise ValueError("N5-D4b topology decision drifted")
            topology_gates.append(topology_pass)

            audits: dict[str, dict[str, Any]] = {}
            for map_id in config["map_order"]:
                audit = _independent_audit(
                    closures[map_id], field, direction, cotangent, h_values
                )
                audits[map_id] = audit
                prefix = (
                    f"c{int(context['cell_index']):02d}_d{direction_index:02d}_{map_id}"
                )
                _verify_array(
                    stored_arrays, verified_arrays, f"{prefix}_primal", audit["primal"]
                )
                _verify_array(
                    stored_arrays, verified_arrays, f"{prefix}_jvp", audit["jvp"]
                )
                _verify_array(
                    stored_arrays, verified_arrays, f"{prefix}_vjp", audit["vjp"]
                )
                stored_map = row["maps"][map_id]
                for label, stored_key, audit_key in (
                    ("dot defect", "dot_relative_defect", "dot_relative_defect"),
                    ("dot signal", "dot_signal", "dot_signal"),
                    ("jvp norm", "jvp_norm", "jvp_norm"),
                    ("vjp norm", "vjp_norm", "vjp_norm"),
                    (
                        "repeat primal",
                        "repeated_primal_relative_defect",
                        "repeat_relative_defect",
                    ),
                ):
                    _same(stored_map[stored_key], audit[audit_key], label=label)
                best_gate = False
                required_gates: list[bool] = []
                all_signal = True
                all_finite = True
                for level_index, level in enumerate(audit["levels"]):
                    key = f"{prefix}_h{level_index:02d}"
                    _verify_array(
                        stored_arrays, verified_arrays, f"{key}_plus", level["plus"]
                    )
                    _verify_array(
                        stored_arrays, verified_arrays, f"{key}_minus", level["minus"]
                    )
                    _verify_array(
                        stored_arrays,
                        verified_arrays,
                        f"{key}_estimate",
                        level["estimate"],
                    )
                    stored_level = stored_map["levels"][level_index]
                    _same(
                        stored_level["relative_error"],
                        level["relative_error"],
                        label="FD relative",
                    )
                    _same(
                        stored_level["absolute_error"],
                        level["absolute_error"],
                        label="FD absolute",
                    )
                    finite = bool(
                        torch.all(torch.isfinite(level["plus"]))
                        and torch.all(torch.isfinite(level["minus"]))
                        and torch.all(torch.isfinite(level["estimate"]))
                    )
                    all_finite = all_finite and finite
                    roundoff = 128.0 * eps * level["output_scale"] / level["h"]
                    signal_min = 1024.0 * eps * level["output_scale"]
                    informative = level["symmetric_difference_norm"] > signal_min
                    all_signal = all_signal and informative
                    scale = max(level["estimate_norm"], audit["jvp_norm"], 1e-30)
                    best = bool(
                        finite
                        and informative
                        and level["relative_error"]
                        <= float(gates["maximum_best_finite_difference_relative_l2"])
                        and level["absolute_error"]
                        <= roundoff
                        + float(gates["maximum_best_finite_difference_relative_l2"])
                        * scale
                    )
                    best_gate = best_gate or best
                    required = level["h"] in required_h
                    required_gate = bool(
                        finite
                        and informative
                        and level["relative_error"]
                        <= float(
                            gates["maximum_required_h_finite_difference_relative_l2"]
                        )
                        and level["absolute_error"]
                        <= roundoff
                        + float(
                            gates["maximum_required_h_finite_difference_relative_l2"]
                        )
                        * scale
                    )
                    if required:
                        required_gates.append(required_gate)
                    if bool(stored_level["informative"]) != informative:
                        raise ValueError("N5-D4b FD signal gate drifted")
                dot_gate = bool(
                    audit["dot_signal"] >= float(gates["minimum_dot_signal_absolute"])
                    and audit["dot_relative_defect"]
                    <= float(gates["maximum_dot_relative_defect"])
                    and audit["jvp_norm"] > float(gates["minimum_dot_signal_absolute"])
                    and audit["vjp_norm"] > float(gates["minimum_dot_signal_absolute"])
                )
                repeat_gate = audit["repeat_relative_defect"] <= float(
                    gates["maximum_repeat_output_relative_l2"]
                )
                map_gate = bool(
                    all_finite
                    and dot_gate
                    and repeat_gate
                    and best_gate
                    and len(required_gates) == len(required_h)
                    and all(required_gates)
                    and all_signal
                )
                if bool(stored_map["map_gate"]) != map_gate:
                    raise ValueError("N5-D4b map gate drifted")
                map_gates.append(map_gate)

            curved = audits["curved_detector"]
            straight = audits["straight_detector"]
            raw = audits["raw_curved_minus_straight"]
            paired = audits["paired_neumaier_residual"]
            raw_metrics = {
                key: _metrics(
                    raw[key],
                    curved[key] - straight[key],
                    curved[key],
                    straight[key],
                )
                for key in ("primal", "jvp", "vjp")
            }
            paired_metrics = {
                key: _metrics(paired[key], raw[key], curved[key], straight[key])
                for key in ("primal", "jvp", "vjp")
            }
            stored_structure = row["structure"]
            for key in ("primal", "jvp", "vjp"):
                _same(
                    stored_structure["raw_equals_curved_minus_straight"][key][
                        "relative_l2"
                    ],
                    raw_metrics[key]["relative_l2"],
                    label=f"raw structure {key}",
                )
                _same(
                    stored_structure["paired_equals_raw_control"][key]["relative_l2"],
                    paired_metrics[key]["relative_l2"],
                    label=f"paired structure {key}",
                )
            floor = float(gates["structural_absolute_tolerance_floor"])

            def structural_gate(metric: dict[str, float], rtol: float) -> bool:
                return bool(metric["absolute_l2"] <= floor + rtol * metric["scale"])

            raw_gate = bool(
                structural_gate(
                    raw_metrics["primal"],
                    float(gates["maximum_raw_structural_output_relative_l2"]),
                )
                and structural_gate(
                    raw_metrics["jvp"],
                    float(gates["maximum_raw_structural_jvp_relative_l2"]),
                )
                and structural_gate(
                    raw_metrics["vjp"],
                    float(gates["maximum_raw_structural_vjp_relative_l2"]),
                )
            )
            paired_gate = bool(
                structural_gate(
                    paired_metrics["primal"],
                    float(gates["maximum_raw_paired_output_relative_l2"]),
                )
                and structural_gate(
                    paired_metrics["jvp"],
                    float(gates["maximum_raw_paired_jvp_relative_l2"]),
                )
                and structural_gate(
                    paired_metrics["vjp"],
                    float(gates["maximum_raw_paired_vjp_relative_l2"]),
                )
            )
            if (
                bool(stored_structure["raw_equals_curved_minus_straight"]["gate"])
                != raw_gate
            ):
                raise ValueError("N5-D4b raw structural gate drifted")
            if (
                bool(stored_structure["paired_equals_raw_control"]["gate"])
                != paired_gate
            ):
                raise ValueError("N5-D4b paired structural gate drifted")
            structure_gates.extend((raw_gate, paired_gate))

    if verified_arrays != set(stored_arrays):
        extra = sorted(set(stored_arrays) - verified_arrays)
        raise ValueError(f"N5-D4b unverified derivative arrays remain: {extra[:3]}")
    context_changed = not all(topology_gates)
    ordinary_pass = bool(
        len(map_gates) == 256
        and all(map_gates)
        and len(structure_gates) == 128
        and all(structure_gates)
        and len(topology_gates) == 64
        and all(topology_gates)
    )
    expected_decision = (
        config["decision_contract"]["context_change_decision"]
        if context_changed
        else config["decision_contract"]["pass_decision"]
        if ordinary_pass
        else config["decision_contract"]["failure_decision"]
    )
    if result.get("protocol_commit") != attestation["protocol_commit"]:
        raise ValueError("N5-D4b result protocol binding drifted")
    if result.get("machine_decision") != expected_decision:
        raise ValueError("N5-D4b machine decision drifted")
    expected_counts = {
        "cell_count": 32,
        "pair_count": 16,
        "field_unit_count": 5,
        "direction_context_count": 64,
        "map_context_count": 256,
        "passing_map_context_count": sum(map_gates),
        "structural_control_count": 128,
        "passing_structural_control_count": sum(structure_gates),
        "combined_structure_context_count": 64,
        "passing_combined_structure_context_count": sum(
            all(bool(payload["gate"]) for payload in row["structure"].values())
            for row in rows
        ),
        "topology_context_count": 64,
        "stable_topology_context_count": sum(topology_gates),
    }
    if result.get("counts") != expected_counts:
        raise ValueError("N5-D4b result counts drifted")
    expected_budget = {
        **_recomputed_budget(config),
        "lineage_total_logical_queries_including_once_only_parent_toy_control": 12_561_696,
        "lineage_total_interpolation_dispatches_including_once_only_parent_toy_control": 1_900_078,
        "failed_or_retried_calls": 0,
        "host_sync_count": "not_instrumented_not_claimed_zero",
        "saved_tensor_bytes": "not_instrumented_not_claimed_zero",
    }
    if result.get("budget") != expected_budget:
        raise ValueError("N5-D4b result budget drifted")
    expected_authorizations = dict(config["claim_authorizations"])
    expected_authorizations["closed_n4_32_cell_field_derivative_contract"] = bool(
        expected_decision == config["decision_contract"]["pass_decision"]
    )
    if result.get("authorizations") != expected_authorizations:
        raise ValueError("N5-D4b claim authorizations drifted")
    expected_clusters = {
        "statistical_independence_claimed": False,
        "cell": _cluster_summary(rows, "cell_id"),
        "pair_cluster": _cluster_summary(rows, "pair_id"),
        "field_unit_cluster": _cluster_summary(rows, "field_unit_id"),
        "family": _cluster_summary(rows, "family_id"),
    }
    if result.get("cluster_accounting") != expected_clusters:
        raise ValueError("N5-D4b cluster accounting drifted")
    expected_extrema = _extrema(rows, config)
    for key, value in expected_extrema.items():
        _same(result["extrema"][key], value, label=f"extrema {key}")
    parent_d4 = _read_json(_resolve(str(config["parent_d4_result"])))
    parent_validation = _read_json(_resolve(str(config["parent_d4_validation"])))
    expected_parent = {
        "result_sha256": _sha256(_resolve(str(config["parent_d4_result"]))),
        "validation_sha256": _sha256(_resolve(str(config["parent_d4_validation"]))),
        "machine_decision": parent_d4["machine_decision"],
        "validation_valid": parent_validation["valid"],
        "strong_toy_detach_control_gate": parent_d4["strong_toy_detach_control"][
            "gate"
        ],
    }
    if result.get("parent_d4_evidence_reused_not_recomputed") != expected_parent:
        raise ValueError("N5-D4b parent D4 evidence binding drifted")
    if result.get("closed_development_population_only") is not True:
        raise ValueError("N5-D4b closed-population boundary drifted")
    if result.get("iid_population_claim") is not False:
        raise ValueError("N5-D4b iid claim must remain false")
    if result.get("decoder_parameterization_tested") is not False:
        raise ValueError("N5-D4b decoder claim drifted")
    _verify_manifest(output, config_path, config)
    figure = _verify_figure(output / "n2_pvgr_n5_d4b_population_field_derivative.png")
    report = {
        "schema": "n2-pvgr-n5-d4b-population-field-derivative-validation-1.0",
        "valid": True,
        "machine_decision": expected_decision,
        "protocol_commit": attestation["protocol_commit"],
        "independent_validator_imports_d4b_runner": False,
        "independent_validator_imports_gate_helper": False,
        "independent_validator_imports_d4b_frozen_input_builder": False,
        "independent_validator_imports_program_signature_serializer": False,
        "parent_d4_result_and_validation_verified_without_repeating_toy_control": True,
        "parent_d3_identity_and_population_order_verified": True,
        "frozen_fields_rays_directions_and_cotangents_regenerated": True,
        "all_derivative_arrays_recomputed": True,
        "ordered_topology_signatures_recomputed": True,
        "cost_ledger_recomputed": True,
        "cluster_nonindependence_accounting_recomputed": True,
        "decision_and_claim_boundary_recomputed": True,
        "deterministic_post_validation_artifact_excluded_from_result_manifest": True,
        "counts": expected_counts,
        "figure": figure,
    }
    _write_json_exclusive(output / "validation_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> int:
    report = validate(parse_args().config)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
