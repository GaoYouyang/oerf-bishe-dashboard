#!/usr/bin/env python3
"""Independently validate the N2-PVGR N5-D4 formal result.

This validator deliberately does not import the D4 runner, derivative-gate
helper, frozen-input builder, or program-signature serializer.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable

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
    smoothstep_grid_field,
)
from demo_t16_operator.cancellation_aware_residual import (  # noqa: E402
    evaluate_paired_residual,
)
from demo_t16_operator.field_dependent_ray import (  # noqa: E402
    RayTraceResult,
    _ray_rhs,
    _validate_trace_parameters,
    initial_pupil_rays,
    path_integrated_deflection,
    sample_pupil_sobol,
    straight_ray_deflection,
    trace_field_dependent_rays,
)

DEFAULT_CONFIG = (
    ROOT
    / "demo_t16_operator/configs/n2_pvgr_n5_d4_tiny_field_derivative_preregistered_v1.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    payload = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.tmp")
    if os.path.lexists(temporary):
        raise FileExistsError(f"N5-D4 validation staging artifact exists: {temporary}")
    if os.path.lexists(path):
        if path.is_file() and not path.is_symlink() and path.read_bytes() == payload:
            return
        raise FileExistsError(f"refusing to replace N5-D4 validation artifact: {path}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
        temporary.unlink()
        os.fsync(directory)
    finally:
        os.close(directory)


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


def _array_hash(value: np.ndarray | torch.Tensor) -> str:
    return _sha256_bytes(_array(value).tobytes(order="C"))


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], cwd=ROOT, check=check, capture_output=True)


def _same(left: float, right: float, *, label: str, atol: float = 1e-18) -> None:
    if not math.isclose(float(left), float(right), rel_tol=2e-12, abs_tol=atol):
        raise ValueError(f"N5-D4 numeric drift for {label}: {left} != {right}")


def _stable_seed(base: int, *parts: object) -> int:
    payload = "|".join(str(part) for part in (base, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def _rig_from_case(case: dict[str, Any]) -> SyntheticRayRig:
    values = case["rig"]
    return SyntheticRayRig(
        rig_id=str(case["id"]),
        view_angle_degrees=float(values["view_angle_degrees"]),
        detector_u=float(values["detector_u"]),
        detector_z=float(values["detector_z"]),
        aperture_radius=float(values["aperture_radius"]),
        path_half_length=float(values["path_half_length"]),
        cone_u=float(values["cone_u"]),
        cone_z=float(values["cone_z"]),
        bend=float(values.get("bend", 0.0)),
    )


def _expand_factorial_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    factorial = config["factorial"]
    cases: list[dict[str, Any]] = []
    for family in factorial["field_families"]:
        for seed in family["field_seeds"]:
            field_unit_id = f"{family['id']}-s{int(seed)}"
            for orientation in factorial["rig_orientation_packages"]:
                for aperture in factorial["aperture_packages"]:
                    rig = {
                        key: value for key, value in orientation.items() if key != "id"
                    }
                    rig.update(
                        {key: value for key, value in aperture.items() if key != "id"}
                    )
                    cases.append(
                        {
                            "id": (
                                f"{field_unit_id}-{orientation['id']}-{aperture['id']}"
                            ),
                            "field_unit_id": field_unit_id,
                            "family_id": str(family["id"]),
                            "phantom_family": str(family["phantom_family"]),
                            "phantom_seed": int(seed),
                            "orientation_id": str(orientation["id"]),
                            "aperture_id": str(aperture["id"]),
                            "rig": rig,
                        }
                    )
    return cases


def _expand_audit_cells(
    config: dict[str, Any], parent_config: dict[str, Any]
) -> list[dict[str, Any]]:
    cases = {case["id"]: case for case in _expand_factorial_cases(parent_config)}
    rows: list[dict[str, Any]] = []
    pair_ids: list[str] = []
    for pair in config["audit_pairs"]:
        pair_ids.append(str(pair["id"]))
        failed_case = cases[pair["failed_cell"]["case_id"]]
        control_case = cases[pair["control_cell"]["case_id"]]
        if failed_case["field_unit_id"] != control_case["field_unit_id"]:
            raise ValueError("N5-D4 independent audit pair changed field unit")
        failed_stress = float(pair["failed_cell"]["stress"])
        control_stress = float(pair["control_cell"]["stress"])
        if failed_stress != control_stress:
            raise ValueError("N5-D4 independent audit pair changed stress")
        orientation_changed = (
            failed_case["orientation_id"] != control_case["orientation_id"]
        )
        aperture_changed = failed_case["aperture_id"] != control_case["aperture_id"]
        if orientation_changed == aperture_changed:
            raise ValueError("N5-D4 audit pair must change exactly one rig factor")
        contrast_factor = "orientation" if orientation_changed else "aperture"
        allowed_failed_gates = {
            "matched_residual_convergence_gate_met",
            "output_convergence_gate_met",
        }
        if pair["failed_cell"].get("n3_failed_gate") not in allowed_failed_gates:
            raise ValueError("N5-D4 audit failure cell has an invalid failed N3 gate")
        if "n3_failed_gate" in pair["control_cell"]:
            raise ValueError(
                "N5-D4 audit control cell unexpectedly carries failure gate"
            )
        for role, entry in (
            ("n3_failure", pair["failed_cell"]),
            ("matched_control", pair["control_cell"]),
        ):
            base = cases[entry["case_id"]]
            case = {**base, "rig": dict(base["rig"])}
            stress = float(entry["stress"])
            rows.append(
                {
                    **case,
                    "case_id": str(case["id"]),
                    "pair_id": str(pair["id"]),
                    "role": role,
                    "contrast_factor": contrast_factor,
                    "dimensionless_stress_multiplier": stress,
                    "n3_failed_gate": entry.get("n3_failed_gate", "none"),
                    "cell_id": f"{case['id']}__stress_{stress:g}",
                }
            )
    expected_pair_ids = [f"p{index:02d}" for index in range(1, 17)]
    if pair_ids != expected_pair_ids:
        raise ValueError("N5-D4 independent audit pair identifiers or order drifted")
    if len(rows) != 32 or len({row["cell_id"] for row in rows}) != 32:
        raise ValueError("N5-D4 independent N4 expansion must produce 32 cells")
    return rows


def _recomputed_budget(config: dict[str, Any]) -> dict[str, int]:
    rays = int(config["ray_count"])
    steps = int(config["step_count"])
    cells = len(config["selected_cells"])
    directions = int(config["direction_count_per_cell"])
    h_count = len(config["finite_difference"]["h_values"])
    closure_invocations_per_map_direction = 2 + 2 * h_count
    direction_contexts = cells * directions
    map_query_coefficient = 35 + 7 + 42 + 42
    derivative_queries = (
        direction_contexts
        * closure_invocations_per_map_direction
        * map_query_coefficient
        * rays
        * steps
    )
    curved_dispatches = 28 * steps + 7
    straight_dispatches = 7
    raw_dispatches = curved_dispatches + straight_dispatches
    paired_dispatches = 28 * steps + 1
    derivative_dispatches = (
        direction_contexts
        * closure_invocations_per_map_direction
        * (curved_dispatches + straight_dispatches + raw_dispatches + paired_dispatches)
    )
    signatures = direction_contexts * (1 + 2 * h_count)
    topology_queries = signatures * 70 * rays * steps
    topology_dispatches = signatures * (28 * steps + 4 * steps + 2)
    toy_queries = 2 * 35 * 4 * 12
    toy_dispatches = 2 * (28 * 12 + 7)
    return {
        "derivative_map_logical_queries": derivative_queries,
        "derivative_map_interpolation_dispatches": derivative_dispatches,
        "topology_logical_queries": topology_queries,
        "topology_interpolation_dispatches": topology_dispatches,
        "selected_context_logical_queries": derivative_queries + topology_queries,
        "selected_context_interpolation_dispatches": derivative_dispatches
        + topology_dispatches,
        "strong_toy_control_logical_queries": toy_queries,
        "strong_toy_control_interpolation_dispatches": toy_dispatches,
        "grand_total_logical_queries": derivative_queries
        + topology_queries
        + toy_queries,
        "grand_total_interpolation_dispatches": derivative_dispatches
        + topology_dispatches
        + toy_dispatches,
        "map_closure_invocations": direction_contexts
        * len(config["map_order"])
        * closure_invocations_per_map_direction,
        "jvp_sweeps": direction_contexts * len(config["map_order"]),
        "vjp_sweeps": direction_contexts * len(config["map_order"]),
        "finite_difference_forward_calls": direction_contexts
        * len(config["map_order"])
        * 2
        * h_count,
        "topology_signature_invocations": signatures,
    }


def _validate_static_config(config: dict[str, Any]) -> None:
    if config.get("schema") != "n2-pvgr-n5-d4-tiny-field-derivative-preregistered-1.0":
        raise ValueError("N5-D4 config schema drifted")
    if (config.get("device"), config.get("dtype")) != ("cpu", "float64"):
        raise ValueError("N5-D4 config must remain CPU float64")
    if [
        int(config[key])
        for key in ("ray_count", "step_count", "direction_count_per_cell")
    ] != [4, 16, 2]:
        raise ValueError("N5-D4 shape contract drifted")
    if len(config.get("selected_cells", [])) != 4:
        raise ValueError("N5-D4 selected-cell count drifted")
    if config["direction_contract"].get("field_direction_modes_in_order") != [
        "smooth_avg_pool3d_kernel3_stride1_padding1",
        "raw_torch_randn",
    ]:
        raise ValueError("N5-D4 field-direction modes drifted")
    if config.get("map_order") != [
        "curved_detector",
        "straight_detector",
        "raw_curved_minus_straight",
        "paired_neumaier_residual",
    ]:
        raise ValueError("N5-D4 map contract drifted")
    if config["finite_difference"]["h_values"] != [
        0.01,
        0.003,
        0.001,
        0.0003,
        0.0001,
        0.00003,
        0.00001,
    ]:
        raise ValueError("N5-D4 h-grid drifted")
    if config["finite_difference"]["required_h_values"] != [0.01, 0.003, 0.001]:
        raise ValueError("N5-D4 required h-grid drifted")
    if config["finite_difference"].get("h_values_float64_hex") != [
        value.hex() for value in [0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]
    ]:
        raise ValueError("N5-D4 float64 h bytes drifted")
    budget = config["budget_contract"]
    recomputed = _recomputed_budget(config)
    if [
        int(budget["expected_derivative_map_logical_queries"]),
        int(budget["expected_derivative_map_interpolation_dispatches"]),
        int(budget["expected_topology_logical_queries"]),
        int(budget["expected_topology_interpolation_dispatches"]),
        int(budget["expected_total_logical_queries"]),
        int(budget["expected_total_interpolation_dispatches"]),
        int(budget["expected_strong_toy_detach_control_logical_queries"]),
        int(budget["expected_strong_toy_detach_control_interpolation_dispatches"]),
        int(budget["expected_grand_total_logical_queries"]),
        int(budget["expected_grand_total_interpolation_dispatches"]),
    ] != [
        1_032_192,
        175_744,
        537_600,
        61_680,
        1_569_792,
        237_424,
        3_360,
        686,
        1_573_152,
        238_110,
    ]:
        raise ValueError("N5-D4 query budget drifted")
    config_budget_mapping = {
        "expected_derivative_map_logical_queries": "derivative_map_logical_queries",
        "expected_derivative_map_interpolation_dispatches": "derivative_map_interpolation_dispatches",
        "expected_topology_logical_queries": "topology_logical_queries",
        "expected_topology_interpolation_dispatches": "topology_interpolation_dispatches",
        "expected_total_logical_queries": "selected_context_logical_queries",
        "expected_total_interpolation_dispatches": "selected_context_interpolation_dispatches",
        "expected_strong_toy_detach_control_logical_queries": "strong_toy_control_logical_queries",
        "expected_strong_toy_detach_control_interpolation_dispatches": "strong_toy_control_interpolation_dispatches",
        "expected_grand_total_logical_queries": "grand_total_logical_queries",
        "expected_grand_total_interpolation_dispatches": "grand_total_interpolation_dispatches",
    }
    for config_key, recomputed_key in config_budget_mapping.items():
        if int(budget[config_key]) != recomputed[recomputed_key]:
            raise ValueError(
                f"N5-D4 independently recomputed budget drifted: {config_key}"
            )


def _verify_attestation(
    config: dict[str, Any], config_path: Path
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    attestation_path = _resolve(str(config["pre_registration_attestation"]))
    archive_path = _resolve(str(config["frozen_input_archive"]))
    attestation = _read_json(attestation_path)
    if (
        attestation.get("schema")
        != "n2-pvgr-n5-d4-tiny-field-derivative-attestation-1.0"
    ):
        raise ValueError("N5-D4 attestation schema drifted")
    if attestation.get("formal_results_absent_at_creation") is not True:
        raise ValueError("N5-D4 attestation output-absence claim drifted")
    if attestation.get("formal_work_output_absent_at_creation") is not True:
        raise ValueError("N5-D4 attestation work-output-absence claim drifted")
    if (
        attestation.get("formal_output") != config["formal_output"]
        or attestation.get("formal_work_output") != config["formal_work_output"]
    ):
        raise ValueError("N5-D4 attestation output paths drifted")
    if attestation.get("config_sha256") != _sha256(config_path):
        raise ValueError("N5-D4 attested config hash drifted")
    protocol_commit = str(attestation["protocol_commit"])
    if attestation.get("repository_head_at_creation") != protocol_commit:
        raise ValueError("N5-D4 attestation commit binding drifted")
    if _git(
        "merge-base", "--is-ancestor", protocol_commit, "HEAD", check=False
    ).returncode:
        raise ValueError("N5-D4 protocol commit is not an ancestor of HEAD")
    if set(attestation["attested_files"]) != set(config["attested_files"]):
        raise ValueError("N5-D4 attested file set drifted")
    for key, entry in attestation["attested_files"].items():
        relative = str(config["attested_files"][key])
        if entry["path"] != relative or _sha256(_resolve(relative)) != entry["sha256"]:
            raise ValueError(f"N5-D4 attested file drifted: {key}")
        frozen = _git("show", f"{protocol_commit}:{relative}").stdout
        if _sha256_bytes(frozen) != entry["sha256"]:
            raise ValueError(f"N5-D4 protocol bytes drifted: {key}")
    if attestation.get("frozen_input_archive") != config["frozen_input_archive"]:
        raise ValueError("N5-D4 frozen archive path drifted")
    if _sha256(archive_path) != attestation.get("frozen_input_archive_sha256"):
        raise ValueError("N5-D4 frozen archive file hash drifted")
    for path in (attestation_path, archive_path):
        relative = _relative(path)
        if _git("ls-files", "--error-unmatch", relative, check=False).returncode:
            raise ValueError(f"N5-D4 prerequisite is untracked: {relative}")
        if _git("status", "--porcelain", "--", relative).stdout.strip():
            raise ValueError(f"N5-D4 prerequisite is dirty: {relative}")
    with np.load(archive_path, allow_pickle=False) as handle:
        arrays = {key: _array(handle[key]) for key in handle.files}
    inventory = attestation["frozen_input_inventory"]
    if set(arrays) != set(inventory):
        raise ValueError("N5-D4 frozen archive inventory drifted")
    for key, value in arrays.items():
        entry = inventory[key]
        if list(value.shape) != entry["shape"] or str(value.dtype) != entry["dtype"]:
            raise ValueError(f"N5-D4 frozen array metadata drifted: {key}")
        if _array_hash(value) != entry["sha256_float64_le_c_order"]:
            raise ValueError(f"N5-D4 frozen array bytes drifted: {key}")
    metadata = attestation["frozen_input_metadata"]
    if (
        _sha256_bytes(_canonical_json(metadata))
        != attestation["frozen_input_metadata_sha256"]
    ):
        raise ValueError("N5-D4 frozen metadata hash drifted")
    return attestation, arrays


def _independent_direction(
    shape: tuple[int, int, int],
    *,
    seed: int,
    boundary_layers: int,
    mode: str,
) -> np.ndarray:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    value = torch.randn(shape, generator=generator, dtype=torch.float64)
    if mode == "smooth_avg_pool3d_kernel3_stride1_padding1":
        value = torch.nn.functional.avg_pool3d(
            value[None, None], kernel_size=3, stride=1, padding=1
        )[0, 0]
    elif mode != "raw_torch_randn":
        raise ValueError(f"N5-D4 independent direction mode drifted: {mode}")
    mask = torch.zeros(shape, dtype=torch.float64)
    layers = int(boundary_layers)
    mask[layers:-layers, layers:-layers, layers:-layers] = 1.0
    value = value * mask
    value = value / torch.linalg.vector_norm(value)
    return _array(value)


def _independent_cotangent(shape: tuple[int, int], *, seed: int) -> np.ndarray:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    value = torch.sign(torch.randn(shape, generator=generator, dtype=torch.float64))
    if bool(torch.any(value == 0.0)):
        raise ValueError("N5-D4 independent Rademacher cotangent contains zero")
    value = value / torch.linalg.vector_norm(value)
    return _array(value)


def _reconstruct_and_verify_frozen_inputs(
    config: dict[str, Any],
    attestation: dict[str, Any],
    archive: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    d3_result = _read_json(_resolve(str(config["parent_d3_result"])))
    d3_pack = _read_json(_resolve(str(config["parent_d3_pack"])))
    d3_validation = _read_json(_resolve(str(config["parent_d3_validation"])))
    if d3_result.get("machine_decision") != "D3_VALID_MIXED_RESIDUAL_REFERENCE_ONLY":
        raise ValueError("N5-D4 parent D3 decision drifted")
    if d3_validation.get("valid") is not True:
        raise ValueError("N5-D4 parent D3 validation drifted")
    pack_by_id = {cell["cell_id"]: cell for cell in d3_pack["cells"]}
    scientific = _read_json(_resolve(str(config["scientific_n4_config"])))
    parent_n3 = _read_json(_resolve(str(config["parent_n3_config"])))
    source = _read_json(_resolve(str(config["source_config"])))
    source["population_count"] = int(config["source_population_count"])
    cell_by_id = {
        cell["cell_id"]: cell for cell in _expand_audit_cells(scientific, parent_n3)
    }
    metadata_by_id = {
        row["cell_id"]: row for row in attestation["frozen_input_metadata"]["contexts"]
    }
    output: dict[str, dict[str, Any]] = {}
    contract = config["direction_contract"]
    for cell_index, selection in enumerate(config["selected_cells"]):
        cell_id = str(selection["cell_id"])
        cell = cell_by_id[cell_id]
        packed = pack_by_id[cell_id]
        if (
            packed["pack_index"] != selection["pack_index"]
            or packed["identity_sha256"] != selection["d3_identity_sha256"]
        ):
            raise ValueError(f"N5-D4 parent D3 identity drifted: {cell_id}")
        metadata = metadata_by_id[cell_id]
        if int(metadata["cell_index"]) != cell_index:
            raise ValueError("N5-D4 frozen cell order drifted")
        spec = make_analytic_phantom(
            family=str(cell["phantom_family"]), seed=int(cell["phantom_seed"])
        )
        field = analytic_phantom_grid(
            spec,
            grid_shape=tuple(int(value) for value in source["grid_shape_zyx"]),
            dtype=torch.float64,
            device="cpu",
        ).field
        population = sample_pupil_sobol(
            int(source["population_count"]),
            seed=_stable_seed(
                int(source["seed_roles"]["population_state_base"]), cell["id"]
            ),
        )
        rays = population[: int(config["ray_count"])]
        if not np.array_equal(_array(field), archive[metadata["field_key"]]):
            raise ValueError(
                f"N5-D4 independently regenerated field drifted: {cell_id}"
            )
        if not np.array_equal(_array(rays), archive[metadata["rays_key"]]):
            raise ValueError(f"N5-D4 independently regenerated rays drifted: {cell_id}")
        support_threshold = float(
            source["certificate"]["support_threshold_fraction_of_grid_peak"]
        ) * float(torch.max(torch.abs(field)))
        _same(
            metadata["support_threshold"], support_threshold, label="support threshold"
        )
        scale = float(source["base_refractivity_scale"]) * float(
            cell["dimensionless_stress_multiplier"]
        )
        _same(metadata["refractivity_scale"], scale, label="refractivity scale")
        if metadata["rig"] != asdict(_rig_from_case(cell)):
            raise ValueError("N5-D4 frozen rig drifted")
        directions: list[tuple[np.ndarray, np.ndarray]] = []
        for direction_index, entry in enumerate(metadata["directions"]):
            direction_mode = str(
                contract["field_direction_modes_in_order"][direction_index]
            )
            direction_seed = _stable_seed(
                int(contract["seed_base"]), cell_id, direction_index
            )
            cotangent_seed = _stable_seed(
                int(contract["cotangent_seed_base"]), cell_id, direction_index
            )
            direction = _independent_direction(
                tuple(
                    int(value) for value in config["observable_contract"]["field_shape"]
                ),
                seed=direction_seed,
                boundary_layers=int(contract["field_direction_boundary_layers_zeroed"]),
                mode=direction_mode,
            )
            cotangent = _independent_cotangent(
                tuple(
                    int(value)
                    for value in config["observable_contract"]["output_shape"]
                ),
                seed=cotangent_seed,
            )
            if (
                entry["direction_seed"] != direction_seed
                or entry["cotangent_seed"] != cotangent_seed
            ):
                raise ValueError("N5-D4 frozen RNG provenance drifted")
            if entry.get("direction_mode") != direction_mode:
                raise ValueError("N5-D4 frozen direction mode drifted")
            if not np.array_equal(direction, archive[entry["direction_key"]]):
                raise ValueError("N5-D4 independently regenerated direction drifted")
            if not np.array_equal(cotangent, archive[entry["cotangent_key"]]):
                raise ValueError("N5-D4 independently regenerated cotangent drifted")
            directions.append((direction, cotangent))
        output[cell_id] = {
            "cell_index": cell_index,
            "cell": cell,
            "metadata": metadata,
            "field": _array(field),
            "rays": _array(rays),
            "directions": directions,
        }
    return output


def _closures(
    rays: torch.Tensor,
    rig: SyntheticRayRig,
    *,
    difference_step: float,
    refractivity_scale: float,
    step_count: int,
) -> dict[str, Callable[[torch.Tensor], torch.Tensor]]:
    def curved(field: torch.Tensor) -> torch.Tensor:
        trace = trace_field_dependent_rays(
            field,
            rays,
            rig,
            gradient_mode="central",
            difference_step=difference_step,
            refractivity_scale=refractivity_scale,
            step_count=step_count,
            create_graph=True,
        )
        return path_integrated_deflection(
            field,
            trace,
            gradient_mode="central",
            difference_step=difference_step,
            refractivity_scale=refractivity_scale,
            create_graph=True,
            detach_path=False,
        )

    def straight(field: torch.Tensor) -> torch.Tensor:
        return straight_ray_deflection(
            field,
            rays,
            rig,
            gradient_mode="central",
            difference_step=difference_step,
            refractivity_scale=refractivity_scale,
            step_count=step_count,
            create_graph=True,
        )

    def raw(field: torch.Tensor) -> torch.Tensor:
        return curved(field) - straight(field)

    def paired(field: torch.Tensor) -> torch.Tensor:
        return evaluate_paired_residual(
            field,
            rays,
            rig,
            difference_step=difference_step,
            refractivity_scale=refractivity_scale,
            step_count=step_count,
        ).paired_neumaier

    return {
        "curved_detector": curved,
        "straight_detector": straight,
        "raw_curved_minus_straight": raw,
        "paired_neumaier_residual": paired,
    }


def _norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.detach().reshape(-1)))


def _independent_audit(
    closure: Callable[[torch.Tensor], torch.Tensor],
    field: torch.Tensor,
    direction: torch.Tensor,
    cotangent: torch.Tensor,
    h_values: list[float],
) -> dict[str, Any]:
    primal, jvp = torch.func.jvp(closure, (field,), (direction,))
    repeated, pullback = torch.func.vjp(closure, field)
    (vjp,) = pullback(cotangent)
    lhs = float(torch.sum(jvp * cotangent))
    rhs = float(torch.sum(direction * vjp))
    dot_signal = max(abs(lhs), abs(rhs))
    levels = []
    for h in h_values:
        plus = closure(field + h * direction)
        minus = closure(field - h * direction)
        estimate = (plus - minus) / (2.0 * h)
        absolute_error = _norm(estimate - jvp)
        estimate_norm = _norm(estimate)
        jvp_norm = _norm(jvp)
        levels.append(
            {
                "h": h,
                "plus": plus.detach(),
                "minus": minus.detach(),
                "estimate": estimate.detach(),
                "estimate_norm": estimate_norm,
                "absolute_error": absolute_error,
                "relative_error": absolute_error / max(estimate_norm, jvp_norm, 1e-30),
                "symmetric_difference_norm": _norm(plus - minus),
                "output_scale": max(_norm(plus), _norm(minus), 1e-30),
            }
        )
    return {
        "primal": primal.detach(),
        "repeated": repeated.detach(),
        "jvp": jvp.detach(),
        "vjp": vjp.detach(),
        "lhs": lhs,
        "rhs": rhs,
        "dot_signal": dot_signal,
        "dot_absolute_defect": abs(lhs - rhs),
        "dot_relative_defect": abs(lhs - rhs) / max(dot_signal, 1e-30),
        "jvp_norm": _norm(jvp),
        "vjp_norm": _norm(vjp),
        "repeat_relative_defect": _norm(primal - repeated)
        / max(_norm(primal), _norm(repeated), 1e-30),
        "levels": levels,
    }


def _lower_cells(queries: torch.Tensor, shape: tuple[int, int, int]) -> np.ndarray:
    nz, ny, nx = shape
    sizes = torch.tensor([nx - 1, ny - 1, nz - 1], dtype=torch.float64)
    scaled = 0.5 * (queries + 1.0) * sizes
    lower = torch.floor(scaled).to(torch.int64)
    maximum = torch.tensor([nx - 2, ny - 2, nz - 2], dtype=torch.int64)
    lower = torch.minimum(torch.maximum(lower, torch.zeros_like(lower)), maximum)
    return lower.cpu().numpy().astype("<i2", copy=False)


def _independent_diagnostic_trace(
    field: torch.Tensor,
    rays: torch.Tensor,
    rig: SyntheticRayRig,
    *,
    difference_step: float,
    refractivity_scale: float,
    step_count: int,
) -> tuple[RayTraceResult, list[tuple[str, torch.Tensor]]]:
    mode, delta, scale, steps = _validate_trace_parameters(
        gradient_mode="central",
        difference_step=difference_step,
        refractivity_scale=refractivity_scale,
        step_count=step_count,
    )
    position, direction, projection_u, projection_v = initial_pupil_rays(rays, rig)
    position = position.to(dtype=field.dtype)
    direction = direction.to(dtype=field.dtype)
    projection_u = projection_u.to(dtype=field.dtype)
    projection_v = projection_v.to(dtype=field.dtype)
    step_size = 2.0 * float(rig.path_half_length) / float(steps)
    positions = [position]
    directions = [direction]
    stages: list[tuple[str, torch.Tensor]] = []
    domain_margins: list[float] = []
    stencil_margins: list[float] = []

    def rhs(
        p: torch.Tensor, d: torch.Tensor, *, step_index: int, stage: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stages.append((f"step={step_index},stage={stage}", p.detach().clone()))
        dp, dd, domain_margin, stencil_margin = _ray_rhs(
            field,
            p,
            d,
            gradient_mode=mode,
            difference_step=delta,
            refractivity_scale=scale,
            create_graph=False,
            stage_label=f"step={step_index},stage={stage}",
        )
        domain_margins.append(domain_margin)
        stencil_margins.append(stencil_margin)
        return dp, dd

    for index in range(steps):
        k1p, k1d = rhs(position, direction, step_index=index, stage="k1")
        k2p, k2d = rhs(
            position + 0.5 * step_size * k1p,
            direction + 0.5 * step_size * k1d,
            step_index=index,
            stage="k2",
        )
        k3p, k3d = rhs(
            position + 0.5 * step_size * k2p,
            direction + 0.5 * step_size * k2d,
            step_index=index,
            stage="k3",
        )
        k4p, k4d = rhs(
            position + step_size * k3p,
            direction + step_size * k3d,
            step_index=index,
            stage="k4",
        )
        position = position + (step_size / 6.0) * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)
        direction = direction + (step_size / 6.0) * (k1d + 2.0 * k2d + 2.0 * k3d + k4d)
        direction = direction / torch.linalg.vector_norm(
            direction, dim=-1, keepdim=True
        ).clamp_min(1e-30)
        positions.append(position)
        directions.append(direction)
    stacked_positions = torch.stack(positions, dim=1)
    stacked_directions = torch.stack(directions, dim=1)
    final_domain_margin = 1.0 - float(torch.max(torch.abs(stacked_positions.detach())))
    domain_margins.append(final_domain_margin)
    stencil_margins.append(final_domain_margin - delta)
    trace = RayTraceResult(
        positions=stacked_positions,
        directions=stacked_directions,
        projection_u=projection_u,
        projection_v=projection_v,
        step_size=step_size,
        gradient_mode=mode,
        minimum_domain_margin=float(min(domain_margins)),
        minimum_stencil_margin=float(min(stencil_margins)),
        maximum_direction_norm_error=float(
            torch.max(
                torch.abs(
                    torch.linalg.vector_norm(stacked_directions.detach(), dim=-1) - 1.0
                )
            )
        ),
    )
    return trace, stages


def _independent_program_signature(
    field: torch.Tensor,
    rays: torch.Tensor,
    rig: SyntheticRayRig,
    *,
    difference_step: float,
    refractivity_scale: float,
    step_count: int,
    support_threshold: float,
    frustum_half_width_u: float,
    frustum_half_width_v: float,
) -> dict[str, Any]:
    trace, stages = _independent_diagnostic_trace(
        field,
        rays,
        rig,
        difference_step=difference_step,
        refractivity_scale=refractivity_scale,
        step_count=step_count,
    )
    groups = list(stages)
    groups.append(
        (
            "curved_path_midpoints",
            (0.5 * (trace.positions[:, :-1] + trace.positions[:, 1:])).reshape(-1, 3),
        )
    )
    start, direction, _, _ = initial_pupil_rays(rays, rig)
    distance = (torch.arange(step_count, dtype=torch.float64) + 0.5) * float(
        trace.step_size
    )
    straight = start[:, None, :] + distance[None, :, None] * direction[:, None, :]
    groups.append(("straight_path_midpoints", straight.reshape(-1, 3)))
    eye = torch.eye(3, dtype=torch.float64)
    offsets = torch.cat(
        (
            torch.zeros((1, 3), dtype=torch.float64),
            difference_step * eye,
            -difference_step * eye,
        )
    )
    labels = []
    cells = []
    bits = []
    minimum_query_margin = math.inf
    for index, (label, points) in enumerate(groups):
        flat = points.reshape(-1, 3)
        queries = flat[:, None, :] + offsets[None, :, :]
        minimum_query_margin = min(
            minimum_query_margin, 1.0 - float(torch.max(torch.abs(queries)))
        )
        flattened = queries.reshape(-1, 3)
        cells.append(_lower_cells(flattened, tuple(field.shape)))
        sampled = smoothstep_grid_field(field, flattened)
        bits.append(
            (torch.abs(sampled) >= support_threshold).cpu().numpy().astype(np.uint8)
        )
        labels.append(
            {
                "group_index": index,
                "label": label,
                "point_count": len(flat),
                "offset_order": ["base", "+x", "+y", "+z", "-x", "-y", "-z"],
            }
        )
    positions = trace.positions.detach()
    progress = torch.arange(positions.shape[1], dtype=torch.float64) * float(
        trace.step_size
    )
    baseline = (
        positions[:, :1, :]
        + progress[None, :, None] * trace.directions[:, :1, :].detach()
    )
    deviation = positions - baseline
    offset_u = torch.sum(deviation * trace.projection_u[:, None, :].detach(), dim=-1)
    offset_v = torch.sum(deviation * trace.projection_v[:, None, :].detach(), dim=-1)
    margins = torch.stack(
        (
            frustum_half_width_u - torch.abs(offset_u),
            frustum_half_width_v - torch.abs(offset_v),
        )
    )
    labels_hash = _sha256_bytes(_canonical_json(labels))
    cell_array = np.concatenate(cells)
    bit_array = np.concatenate(bits)
    frustum = (margins >= 0.0).cpu().numpy().astype(np.uint8)
    components = {
        "schema": "field-program-signature-1.0",
        "labels_sha256": labels_hash,
        "interpolation_cells_sha256": _sha256_bytes(cell_array.tobytes(order="C")),
        "support_bits_sha256": _sha256_bytes(bit_array.tobytes(order="C")),
        "frustum_margin_signs_sha256": _sha256_bytes(frustum.tobytes(order="C")),
    }
    return {
        "digest_sha256": _sha256_bytes(_canonical_json(components)),
        **{key: value for key, value in components.items() if key != "schema"},
        "group_count": len(groups),
        "query_record_count": int(len(cell_array)),
        "support_true_count": int(np.sum(bit_array)),
        "minimum_domain_margin": float(trace.minimum_domain_margin),
        "minimum_stencil_margin": min(
            float(trace.minimum_stencil_margin), minimum_query_margin
        ),
        "minimum_frustum_margin": float(torch.min(margins)),
        "maximum_direction_norm_error": float(trace.maximum_direction_norm_error),
    }


def _verify_array(
    stored: dict[str, np.ndarray], key: str, expected: torch.Tensor
) -> None:
    if key not in stored:
        raise ValueError(f"N5-D4 result array is missing: {key}")
    candidate = stored[key]
    reference = _array(expected)
    if not np.array_equal(candidate, reference):
        maximum = float(np.max(np.abs(candidate - reference)))
        if not np.allclose(candidate, reference, rtol=2e-13, atol=2e-18):
            raise ValueError(
                f"N5-D4 independently recomputed array drifted: {key}, max={maximum}"
            )


def _metrics(
    candidate: torch.Tensor, reference: torch.Tensor, *scale_values: torch.Tensor
) -> dict[str, float]:
    delta = candidate - reference
    absolute = _norm(delta)
    scale = max(
        _norm(candidate),
        _norm(reference),
        *(_norm(value) for value in scale_values),
        1e-30,
    )
    return {
        "absolute_l2": absolute,
        "relative_l2": absolute / scale,
        "maximum_absolute": float(torch.max(torch.abs(delta))),
        "scale": scale,
    }


def _verify_result_manifest(
    output: Path, config_path: Path, config: dict[str, Any]
) -> None:
    manifest = _read_json(output / "manifest.json")
    if manifest.get("schema") != "n2-pvgr-n5-d4-tiny-field-derivative-manifest-1.0":
        raise ValueError("N5-D4 result manifest schema drifted")
    if manifest.get("post_validation_artifacts_excluded_from_manifest") != [
        "validation_report.json"
    ]:
        raise ValueError("N5-D4 post-validation manifest boundary drifted")
    expected_special = {
        "config": config_path,
        "attestation": _resolve(str(config["pre_registration_attestation"])),
        "frozen_input_archive": _resolve(str(config["frozen_input_archive"])),
    }
    for key, path in expected_special.items():
        entry = manifest["files"].get(key)
        if entry != {"path": _relative(path), "sha256": _sha256(path)}:
            raise ValueError(f"N5-D4 result manifest drifted: {key}")
    for name in (
        "array_inventory.json",
        "config_snapshot.json",
        "derivative_arrays.npz",
        "n2_pvgr_n5_d4_tiny_field_derivative.png",
        "result.json",
        "rows.json",
        "summary.md",
    ):
        path = output / name
        entry = manifest["files"].get(name)
        if entry != {"path": _relative(path), "sha256": _sha256(path)}:
            raise ValueError(f"N5-D4 output manifest drifted: {name}")


def _verify_figure(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
        extrema = image.convert("L").getextrema()
    if width < 1200 or height < 700 or extrema[0] == extrema[1]:
        raise ValueError("N5-D4 figure is blank or undersized")
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
    contexts = _reconstruct_and_verify_frozen_inputs(
        config, attestation, frozen_archive
    )
    output = _resolve(str(config["formal_output"]))
    if not output.is_dir():
        raise FileNotFoundError("N5-D4 formal output is missing")
    if os.path.lexists(_resolve(str(config["formal_work_output"]))):
        raise ValueError("N5-D4 work output still exists after formal publication")
    if _read_json(output / "config_snapshot.json") != config:
        raise ValueError("N5-D4 config snapshot drifted")
    result = _read_json(output / "result.json")
    rows = _read_json(output / "rows.json")
    inventory = _read_json(output / "array_inventory.json")
    with np.load(output / "derivative_arrays.npz", allow_pickle=False) as handle:
        stored_arrays = {key: _array(handle[key]) for key in handle.files}
    if set(stored_arrays) != set(inventory["arrays"]):
        raise ValueError("N5-D4 result-array inventory drifted")
    for key, value in stored_arrays.items():
        entry = inventory["arrays"][key]
        if (
            list(value.shape) != entry["shape"]
            or _array_hash(value) != entry["sha256_float64_le_c_order"]
        ):
            raise ValueError(f"N5-D4 result-array bytes drifted: {key}")
    if len(rows) != 8:
        raise ValueError("N5-D4 direction-row count drifted")
    rows_by_key = {(row["cell_id"], int(row["direction_index"])): row for row in rows}
    map_gates: list[bool] = []
    topology_gates: list[bool] = []
    structure_gates: list[bool] = []
    h_values = [float(value) for value in config["finite_difference"]["h_values"]]
    required_h = set(
        float(value) for value in config["finite_difference"]["required_h_values"]
    )
    gates = config["gates"]
    eps = torch.finfo(torch.float64).eps

    for cell_id, context in contexts.items():
        metadata = context["metadata"]
        field = torch.from_numpy(context["field"].copy()).to(torch.float64)
        rays = torch.from_numpy(context["rays"].copy()).to(torch.float64)
        rig = SyntheticRayRig(**metadata["rig"])
        closures = _closures(
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
            direction = torch.from_numpy(direction_array.copy()).to(torch.float64)
            cotangent = torch.from_numpy(cotangent_array.copy()).to(torch.float64)
            base_signature = _independent_program_signature(
                field, rays, rig, **signature_kwargs
            )
            if row["topology"]["base"] != base_signature:
                raise ValueError("N5-D4 independently rebuilt base topology drifted")
            topology_pass = True
            topology_rows = {
                (float(item["h"]), item["side"]): item
                for item in row["topology"]["perturbations"]
            }
            for h in h_values:
                for sign, side in ((1.0, "plus"), (-1.0, "minus")):
                    signature = _independent_program_signature(
                        field + sign * h * direction, rays, rig, **signature_kwargs
                    )
                    stored = topology_rows[(h, side)]
                    if stored["digest_sha256"] != signature["digest_sha256"]:
                        raise ValueError(
                            "N5-D4 independently rebuilt perturbed topology drifted"
                        )
                    same = signature["digest_sha256"] == base_signature["digest_sha256"]
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
                        raise ValueError("N5-D4 topology gate serialization drifted")
                    topology_pass = topology_pass and same and margins
            if bool(row["topology_gate"]) != topology_pass:
                raise ValueError("N5-D4 topology decision drifted")
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
                _verify_array(stored_arrays, f"{prefix}_primal", audit["primal"])
                _verify_array(stored_arrays, f"{prefix}_jvp", audit["jvp"])
                _verify_array(stored_arrays, f"{prefix}_vjp", audit["vjp"])
                stored_map = row["maps"][map_id]
                _same(
                    stored_map["dot_relative_defect"],
                    audit["dot_relative_defect"],
                    label="dot defect",
                )
                _same(stored_map["dot_signal"], audit["dot_signal"], label="dot signal")
                _same(stored_map["jvp_norm"], audit["jvp_norm"], label="jvp norm")
                _same(stored_map["vjp_norm"], audit["vjp_norm"], label="vjp norm")
                _same(
                    stored_map["repeated_primal_relative_defect"],
                    audit["repeat_relative_defect"],
                    label="repeat primal",
                )
                best_gate = False
                required_gates = []
                all_signal = True
                for level_index, level in enumerate(audit["levels"]):
                    key = f"{prefix}_h{level_index:02d}"
                    _verify_array(stored_arrays, f"{key}_plus", level["plus"])
                    _verify_array(stored_arrays, f"{key}_minus", level["minus"])
                    _verify_array(stored_arrays, f"{key}_estimate", level["estimate"])
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
                    roundoff = 128.0 * eps * level["output_scale"] / level["h"]
                    signal_min = 1024.0 * eps * level["output_scale"]
                    informative = level["symmetric_difference_norm"] > signal_min
                    all_signal = all_signal and informative
                    scale = max(level["estimate_norm"], audit["jvp_norm"], 1e-30)
                    best = bool(
                        informative
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
                        informative
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
                        raise ValueError("N5-D4 FD signal gate drifted")
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
                    dot_gate
                    and repeat_gate
                    and best_gate
                    and all(required_gates)
                    and all_signal
                )
                if bool(stored_map["map_gate"]) != map_gate:
                    raise ValueError("N5-D4 map gate drifted")
                map_gates.append(map_gate)

            curved = audits["curved_detector"]
            straight = audits["straight_detector"]
            raw = audits["raw_curved_minus_straight"]
            paired = audits["paired_neumaier_residual"]
            raw_metrics = {
                "primal": _metrics(
                    raw["primal"],
                    curved["primal"] - straight["primal"],
                    curved["primal"],
                    straight["primal"],
                ),
                "jvp": _metrics(
                    raw["jvp"],
                    curved["jvp"] - straight["jvp"],
                    curved["jvp"],
                    straight["jvp"],
                ),
                "vjp": _metrics(
                    raw["vjp"],
                    curved["vjp"] - straight["vjp"],
                    curved["vjp"],
                    straight["vjp"],
                ),
            }
            paired_metrics = {
                "primal": _metrics(
                    paired["primal"],
                    raw["primal"],
                    curved["primal"],
                    straight["primal"],
                ),
                "jvp": _metrics(
                    paired["jvp"], raw["jvp"], curved["jvp"], straight["jvp"]
                ),
                "vjp": _metrics(
                    paired["vjp"], raw["vjp"], curved["vjp"], straight["vjp"]
                ),
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
            absolute_floor = float(gates["structural_absolute_tolerance_floor"])

            def structural_gate(metric: dict[str, float], rtol: float) -> bool:
                return bool(
                    metric["absolute_l2"]
                    <= absolute_floor + float(rtol) * metric["scale"]
                )

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
                raise ValueError("N5-D4 raw structural gate drifted")
            if (
                bool(stored_structure["paired_equals_raw_control"]["gate"])
                != paired_gate
            ):
                raise ValueError("N5-D4 paired structural gate drifted")
            structure_gates.extend((raw_gate, paired_gate))

    # Independently replay the fixed high-curvature control.
    axis = torch.linspace(-1.0, 1.0, 9, dtype=torch.float64)
    zz, yy, xx = torch.meshgrid(axis, axis, axis, indexing="ij")
    toy_field = -torch.exp(-((xx / 0.42) ** 2 + (yy / 0.34) ** 2 + (zz / 0.38) ** 2))
    toy_rays = sample_pupil_sobol(4, seed=9107)
    toy_rig = SyntheticRayRig(
        rig_id="n5-d4-strong-detach-control",
        view_angle_degrees=27.0,
        detector_u=0.04,
        detector_z=-0.03,
        aperture_radius=0.025,
        path_half_length=0.62,
        cone_u=0.02,
        cone_z=0.015,
        bend=0.0,
    )
    generator = torch.Generator(device="cpu").manual_seed(9109)
    toy_direction = torch.randn(
        toy_field.shape, generator=generator, dtype=torch.float64
    )
    toy_direction = toy_direction / torch.linalg.vector_norm(toy_direction)

    def toy(detach: bool):
        def closure(value: torch.Tensor) -> torch.Tensor:
            trace = trace_field_dependent_rays(
                value,
                toy_rays,
                toy_rig,
                gradient_mode="central",
                difference_step=0.002,
                refractivity_scale=0.03,
                step_count=12,
                create_graph=True,
            )
            return path_integrated_deflection(
                value,
                trace,
                gradient_mode="central",
                difference_step=0.002,
                refractivity_scale=0.03,
                create_graph=True,
                detach_path=detach,
            )

        return closure

    _, toy_full = torch.func.jvp(toy(False), (toy_field,), (toy_direction,))
    _, toy_frozen = torch.func.jvp(toy(True), (toy_field,), (toy_direction,))
    toy_relative = _norm(toy_full - toy_frozen) / max(
        _norm(toy_full), _norm(toy_frozen), 1e-30
    )
    toy_gate = toy_relative >= float(
        gates[
            "minimum_full_vs_frozen_path_jvp_relative_difference_in_strong_toy_control"
        ]
    )
    stored_toy = result["strong_toy_detach_control"]
    _same(stored_toy["comparison"]["relative_l2"], toy_relative, label="strong toy")
    if bool(stored_toy["gate"]) != toy_gate:
        raise ValueError("N5-D4 strong toy gate drifted")

    context_changed = not all(topology_gates)
    ordinary_pass = bool(
        len(map_gates) == 32
        and all(map_gates)
        and len(structure_gates) == 16
        and all(structure_gates)
        and toy_gate
    )
    expected_decision = (
        config["decision_contract"]["context_change_decision"]
        if context_changed
        else config["decision_contract"]["pass_decision"]
        if ordinary_pass
        else config["decision_contract"]["failure_decision"]
    )
    if result.get("protocol_commit") != attestation["protocol_commit"]:
        raise ValueError("N5-D4 result protocol binding drifted")
    if result.get("machine_decision") != expected_decision:
        raise ValueError("N5-D4 machine decision drifted")
    expected_counts = {
        "cell_count": 4,
        "direction_context_count": 8,
        "map_context_count": 32,
        "passing_map_context_count": sum(map_gates),
        "structure_gate_count": 16,
        "passing_structure_gate_count": sum(structure_gates),
        "stable_topology_context_count": sum(topology_gates),
    }
    if result.get("counts") != expected_counts:
        raise ValueError("N5-D4 result counts drifted")
    expected_budget = {
        **_recomputed_budget(config),
        "failed_or_retried_calls": 0,
        "host_sync_count": "not_instrumented_not_claimed_zero",
        "saved_tensor_bytes": "not_instrumented_not_claimed_zero",
    }
    if result["budget"] != expected_budget:
        raise ValueError("N5-D4 result budget drifted")
    expected_authorizations = dict(config["claim_authorizations"])
    expected_authorizations["tiny_selected_field_derivative_contract"] = bool(
        expected_decision == config["decision_contract"]["pass_decision"]
    )
    if result.get("authorizations") != expected_authorizations:
        raise ValueError("N5-D4 claim authorization drifted")
    _verify_result_manifest(output, config_path, config)
    figure = _verify_figure(output / "n2_pvgr_n5_d4_tiny_field_derivative.png")
    report = {
        "schema": "n2-pvgr-n5-d4-tiny-field-derivative-validation-1.0",
        "valid": True,
        "machine_decision": expected_decision,
        "protocol_commit": attestation["protocol_commit"],
        "independent_validator_imports_runner": False,
        "independent_validator_imports_gate_helper": False,
        "independent_validator_imports_frozen_input_builder": False,
        "independent_validator_imports_program_signature_serializer": False,
        "parent_d3_identity_and_validation_verified": True,
        "frozen_fields_rays_directions_and_cotangents_regenerated": True,
        "all_derivative_arrays_recomputed": True,
        "ordered_topology_signatures_recomputed": True,
        "cost_ledger_recomputed": True,
        "decision_and_claim_boundary_recomputed": True,
        "deterministic_post_validation_artifact_excluded_from_result_manifest": True,
        "counts": expected_counts,
        "figure": figure,
    }
    _write_json(output / "validation_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> int:
    result = validate(parse_args().config)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
