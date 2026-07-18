"""Deterministic frozen-input builder for the N2-PVGR N5-D4b census gate."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from . import run_n2_pvgr_n4_evaluator_convergence as n4
    from .analytic_bost_phantoms import analytic_phantom_grid, make_analytic_phantom
    from .field_dependent_ray import sample_pupil_sobol
    from .run_n2_pvgr_n0_trifidelity_development import _rig_from_case, stable_seed
except ImportError:
    import run_n2_pvgr_n4_evaluator_convergence as n4
    from analytic_bost_phantoms import analytic_phantom_grid, make_analytic_phantom
    from field_dependent_ray import sample_pupil_sobol
    from run_n2_pvgr_n0_trifidelity_development import _rig_from_case, stable_seed


ROOT = Path(__file__).resolve().parents[1]
FROZEN_INPUT_SCHEMA = "n2-pvgr-n5-d4b-population-frozen-inputs-1.0"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def canonical_float64_array(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(value, dtype="<f8"))


def array_sha256(value: np.ndarray | torch.Tensor) -> str:
    return hashlib.sha256(canonical_float64_array(value).tobytes(order="C")).hexdigest()


def _direction(
    shape: tuple[int, int, int],
    *,
    seed: int,
    boundary_layers: int,
    mode: str,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    value = torch.randn(shape, generator=generator, dtype=torch.float64)
    if mode == "smooth_avg_pool3d_kernel3_stride1_padding1":
        value = torch.nn.functional.avg_pool3d(
            value[None, None], kernel_size=3, stride=1, padding=1
        )[0, 0]
    elif mode != "raw_torch_randn":
        raise ValueError(f"unknown D4b field-direction mode: {mode}")
    layers = int(boundary_layers)
    if layers < 1 or 2 * layers >= min(shape):
        raise ValueError("invalid D4b field-direction boundary width")
    mask = torch.zeros(shape, dtype=torch.float64)
    mask[layers:-layers, layers:-layers, layers:-layers] = 1.0
    value = value * mask
    norm = torch.linalg.vector_norm(value)
    if not bool(torch.isfinite(norm)) or float(norm) <= 0.0:
        raise ValueError("D4b field direction is degenerate")
    return value / norm


def _cotangent(shape: tuple[int, int], *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    value = torch.sign(torch.randn(shape, generator=generator, dtype=torch.float64))
    if bool(torch.any(value == 0.0)):
        raise ValueError("D4b Rademacher cotangent contains a zero component")
    norm = torch.linalg.vector_norm(value)
    if not bool(torch.isfinite(norm)) or float(norm) <= 0.0:
        raise ValueError("D4b cotangent is degenerate")
    return value / norm


def _counter(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def build_frozen_inputs(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Build all 32 preregistered contexts in frozen D3 pack order."""

    parent_d4 = _read_json(_resolve(str(config["parent_d4_result"])))
    parent_d4_validation = _read_json(_resolve(str(config["parent_d4_validation"])))
    required_parent = str(config["decision_contract"]["parent_d4_required_decision"])
    if parent_d4.get("machine_decision") != required_parent:
        raise ValueError("D4b parent D4 decision is not authorized")
    if parent_d4_validation.get("valid") is not True:
        raise ValueError("D4b parent D4 validation is not valid")

    d3_result = _read_json(_resolve(str(config["parent_d3_result"])))
    d3_pack = _read_json(_resolve(str(config["parent_d3_pack"])))
    d3_validation = _read_json(_resolve(str(config["parent_d3_validation"])))
    if d3_result.get("machine_decision") != "D3_VALID_MIXED_RESIDUAL_REFERENCE_ONLY":
        raise ValueError("D4b parent D3 decision drifted")
    if d3_pack.get("machine_decision") != d3_result["machine_decision"]:
        raise ValueError("D4b parent D3 pack decision drifted")
    if d3_validation.get("valid") is not True:
        raise ValueError("D4b parent D3 validation is not valid")

    population = config["population_contract"]
    if d3_pack.get("cell_order_sha256") != population["expected_d3_cell_order_sha256"]:
        raise ValueError("D4b D3 cell-order hash drifted")
    packed_cells = list(d3_pack["cells"])
    expected_count = int(population["expected_cell_count"])
    if len(packed_cells) != expected_count:
        raise ValueError("D4b D3 population size drifted")

    scientific = _read_json(_resolve(str(config["scientific_n4_config"])))
    parent_n3 = _read_json(_resolve(str(config["parent_n3_config"])))
    source = _read_json(_resolve(str(config["source_config"])))
    source = n4._source_for_run(
        source, {"population_count": config["source_population_count"]}
    )
    expanded_cells = n4.expand_audit_cells(scientific, parent_n3)
    if len(expanded_cells) != expected_count:
        raise ValueError("D4b N4 expansion size drifted")
    expanded_by_id = {str(cell["cell_id"]): cell for cell in expanded_cells}
    packed_ids = [str(cell["cell_id"]) for cell in packed_cells]
    if packed_ids != [str(cell["cell_id"]) for cell in expanded_cells]:
        raise ValueError("D4b D3 pack and N4 expansion order differ")
    if len(set(packed_ids)) != expected_count:
        raise ValueError("D4b population contains duplicate cell identifiers")

    expected_pairs = [str(value) for value in population["expected_pair_ids"]]
    observed_pairs: list[str] = []
    expected_roles = [
        str(value) for value in population["expected_role_order_within_each_pair"]
    ]
    for pair_index in range(0, expected_count, 2):
        pair = packed_cells[pair_index : pair_index + 2]
        if [str(row["role"]) for row in pair] != expected_roles:
            raise ValueError("D4b within-pair role order drifted")
        if len({str(row["pair_id"]) for row in pair}) != 1:
            raise ValueError("D4b pair membership drifted")
        observed_pairs.append(str(pair[0]["pair_id"]))
    if observed_pairs != expected_pairs:
        raise ValueError("D4b pair identifiers or order drifted")

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
        raise ValueError("D4b family census drifted")
    if field_unit_counts != {
        str(key): int(value)
        for key, value in population["expected_field_unit_counts"].items()
    }:
        raise ValueError("D4b field-unit census drifted")
    if stress_counts != {
        str(key): int(value)
        for key, value in population["expected_stress_counts"].items()
    }:
        raise ValueError("D4b stress census drifted")
    if reference_method_counts != {
        str(key): int(value)
        for key, value in population["expected_d3_reference_method_counts"].items()
    }:
        raise ValueError("D4b D3 reference-method census drifted")

    direction_contract = config["direction_contract"]
    boundary_layers = int(direction_contract["field_direction_boundary_layers_zeroed"])
    direction_modes = [
        str(value) for value in direction_contract["field_direction_modes_in_order"]
    ]
    direction_count = int(config["direction_count_per_cell"])
    if len(direction_modes) != direction_count:
        raise ValueError("D4b field-direction mode count drifted")
    ray_count = int(config["ray_count"])
    expected_field_shape = tuple(
        int(value) for value in config["observable_contract"]["field_shape"]
    )
    expected_output_shape = tuple(
        int(value) for value in config["observable_contract"]["output_shape"]
    )
    if expected_output_shape != (ray_count, 2):
        raise ValueError("D4b output/ray shape drifted")

    arrays: dict[str, np.ndarray] = {}
    contexts: list[dict[str, Any]] = []
    for cell_index, packed in enumerate(packed_cells):
        cell_id = str(packed["cell_id"])
        cell = expanded_by_id[cell_id]
        for packed_key, cell_key in (
            ("pair_id", "pair_id"),
            ("role", "role"),
            ("field_unit_id", "field_unit_id"),
            ("family_id", "family_id"),
            ("phantom_seed", "phantom_seed"),
        ):
            if packed[packed_key] != cell[cell_key]:
                raise ValueError(f"D4b D3/N4 metadata drifted: {cell_id}/{packed_key}")
        if int(packed["pack_index"]) != cell_index:
            raise ValueError("D4b D3 pack index drifted")

        spec = make_analytic_phantom(
            family=str(cell["phantom_family"]), seed=int(cell["phantom_seed"])
        )
        field = analytic_phantom_grid(
            spec,
            grid_shape=tuple(int(value) for value in source["grid_shape_zyx"]),
            dtype=torch.float64,
            device="cpu",
        ).field
        if tuple(field.shape) != expected_field_shape:
            raise ValueError("D4b frozen field shape drifted")
        pupil = sample_pupil_sobol(
            int(source["population_count"]),
            seed=stable_seed(
                int(source["seed_roles"]["population_state_base"]), cell["id"]
            ),
        )
        rays = pupil[:ray_count].clone()
        rig = _rig_from_case(cell)
        scale = float(source["base_refractivity_scale"]) * float(
            cell["dimensionless_stress_multiplier"]
        )
        support_threshold = float(
            source["certificate"]["support_threshold_fraction_of_grid_peak"]
        ) * float(torch.max(torch.abs(field)))

        prefix = f"cell_{cell_index:02d}"
        arrays[f"{prefix}_field"] = canonical_float64_array(field)
        arrays[f"{prefix}_rays"] = canonical_float64_array(rays)
        direction_rows: list[dict[str, Any]] = []
        for direction_index in range(direction_count):
            direction_seed = stable_seed(
                int(direction_contract["seed_base"]), cell_id, direction_index
            )
            cotangent_seed = stable_seed(
                int(direction_contract["cotangent_seed_base"]),
                cell_id,
                direction_index,
            )
            direction = _direction(
                expected_field_shape,
                seed=direction_seed,
                boundary_layers=boundary_layers,
                mode=direction_modes[direction_index],
            )
            cotangent = _cotangent(expected_output_shape, seed=cotangent_seed)
            direction_key = f"{prefix}_direction_{direction_index:02d}"
            cotangent_key = f"{prefix}_cotangent_{direction_index:02d}"
            arrays[direction_key] = canonical_float64_array(direction)
            arrays[cotangent_key] = canonical_float64_array(cotangent)
            direction_rows.append(
                {
                    "direction_index": direction_index,
                    "direction_seed": direction_seed,
                    "direction_mode": direction_modes[direction_index],
                    "cotangent_seed": cotangent_seed,
                    "direction_key": direction_key,
                    "cotangent_key": cotangent_key,
                    "direction_sha256": array_sha256(direction),
                    "cotangent_sha256": array_sha256(cotangent),
                }
            )

        row_ids = [
            f"{cell_id}:ray_{index:03d}:{axis}"
            for index in range(ray_count)
            for axis in ("u", "v")
        ]
        contexts.append(
            {
                "cell_index": cell_index,
                "cell_id": cell_id,
                "pair_id": str(cell["pair_id"]),
                "role": str(cell["role"]),
                "field_unit_id": str(cell["field_unit_id"]),
                "family_id": str(cell["family_id"]),
                "phantom_family": str(cell["phantom_family"]),
                "phantom_seed": int(cell["phantom_seed"]),
                "dimensionless_stress_multiplier": float(
                    cell["dimensionless_stress_multiplier"]
                ),
                "d3_identity_sha256": str(packed["identity_sha256"]),
                "d3_reference_method": str(packed["reference_method"]),
                "d3_source_kind": str(packed["source_kind"]),
                "d3_selection_basis": str(packed["selection_basis"]),
                "refractivity_scale": scale,
                "difference_step": float(source["difference_step"]),
                "support_threshold": support_threshold,
                "frustum_half_width_u": float(
                    source["certificate"]["frustum_half_width_u"]
                ),
                "frustum_half_width_v": float(
                    source["certificate"]["frustum_half_width_v"]
                ),
                "rig": asdict(rig),
                "field_key": f"{prefix}_field",
                "rays_key": f"{prefix}_rays",
                "field_sha256": array_sha256(field),
                "rays_sha256": array_sha256(rays),
                "row_ids": row_ids,
                "row_ids_sha256": hashlib.sha256(
                    json.dumps(row_ids, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
                "directions": direction_rows,
            }
        )

    metadata = {
        "schema": FROZEN_INPUT_SCHEMA,
        "candidate_id": str(config["candidate_id"]),
        "array_encoding": "float64_little_endian_c_order",
        "selection_rule": str(population["selection_rule"]),
        "d3_cell_order_sha256": str(d3_pack["cell_order_sha256"]),
        "cell_count": len(contexts),
        "pair_count": len(expected_pairs),
        "field_unit_count": len(field_unit_counts),
        "family_counts": family_counts,
        "field_unit_counts": field_unit_counts,
        "stress_counts": stress_counts,
        "d3_reference_method_counts": reference_method_counts,
        "direction_count_per_cell": direction_count,
        "ray_count": ray_count,
        "step_count": int(config["step_count"]),
        "contexts": contexts,
    }
    return metadata, arrays


__all__ = [
    "FROZEN_INPUT_SCHEMA",
    "array_sha256",
    "build_frozen_inputs",
    "canonical_float64_array",
]
