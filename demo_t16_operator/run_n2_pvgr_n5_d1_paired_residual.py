#!/usr/bin/env python3
"""Run the preregistered N5-D1 paired-residual accumulation audit."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from . import run_n2_pvgr_n4_evaluator_convergence as n4
    from .automatic_discrete_multifidelity import SyntheticRayRig
    from .cancellation_aware_residual import evaluate_paired_residual
    from .field_dependent_ray import (
        path_integrated_deflection,
        sample_pupil_sobol,
    )
    from .shared_straight_state import build_straight_path_state
except ImportError:
    import run_n2_pvgr_n4_evaluator_convergence as n4
    from automatic_discrete_multifidelity import SyntheticRayRig
    from cancellation_aware_residual import evaluate_paired_residual
    from field_dependent_ray import path_integrated_deflection, sample_pupil_sobol
    from shared_straight_state import build_straight_path_state


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT
    / "demo_t16_operator/configs/n2_pvgr_n5_d1_paired_residual_preregistered_v1.json"
)
NONRAW_METHODS = (
    "paired_naive",
    "paired_pairwise",
    "paired_neumaier",
    "separate_neumaier_subtraction",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], cwd=ROOT, check=check, capture_output=True)


def _relative_l2(left: np.ndarray, right: np.ndarray) -> float:
    numerator = np.linalg.norm(np.asarray(left, dtype=np.float64) - right)
    denominator = max(float(np.linalg.norm(right)), 1e-30)
    return float(numerator / denominator)


def _validate_contract(config: dict[str, Any]) -> None:
    if config.get("schema") != "n2-pvgr-n5-d1-paired-residual-preregistered-1.0":
        raise ValueError("N5-D1 schema drifted")
    if config.get("candidate_id") != "N2-PVGR-N5-D1-PAIR4":
        raise ValueError("N5-D1 identifier drifted")
    if config.get("status") != (
        "preregistered_post_n4_mechanism_audit_no_algorithm_authorization"
    ):
        raise ValueError("N5-D1 status drifted")
    if (config.get("device"), config.get("dtype")) != ("cpu", "float64"):
        raise ValueError("N5-D1 requires CPU float64")
    if int(config.get("population_count", 0)) != 256:
        raise ValueError("N5-D1 requires 256 common rays")
    if tuple(int(value) for value in config["step_counts"]) != (1024, 2048):
        raise ValueError("N5-D1 levels drifted")
    expected_cells = (
        "smooth-s1871-orientation_58-narrow__stress_1",
        "smooth-s1871-orientation_58-wide__stress_1",
        "smooth-s1871-orientation_58-narrow__stress_3",
        "smooth-s1871-orientation_58-wide__stress_3",
    )
    if tuple(config["selected_cells"]) != expected_cells:
        raise ValueError("N5-D1 selected cells drifted")
    expected_methods = (
        "raw_separate_subtraction",
        *NONRAW_METHODS,
    )
    if tuple(config["accumulation_methods"]) != expected_methods:
        raise ValueError("N5-D1 accumulation methods drifted")
    expected_gates = {
        "maximum_constant_absolute_output": 0.0,
        "maximum_weak_quadratic_finite_difference_relative_l2": 0.03,
        "maximum_rotation_equivariance_relative_l2": 2e-10,
        "maximum_frozen_route_equivalence_relative_l2": 5e-12,
        "maximum_parent_n4_adjacent_metric_absolute_difference": 1e-12,
        "maximum_accumulation_fraction_of_refinement_for_too_small": 0.01,
        "minimum_accumulation_fraction_of_refinement_for_plausible": 0.1,
    }
    if {key: float(value) for key, value in config["gates"].items()} != expected_gates:
        raise ValueError("N5-D1 gates drifted")
    if not all(value is False for value in config["claim_authorizations"].values()):
        raise ValueError("N5-D1 broad claim authorizations must remain false")
    expected_paths = {
        "parent_n4_config": "demo_t16_operator/configs/n2_pvgr_n4_1_execution_amendment_preregistered_v1.json",
        "parent_n4_result": "demo_t16_operator/results/n2_pvgr_n4_1_evaluator_convergence_v1/result.json",
        "scientific_n4_config": "demo_t16_operator/configs/n2_pvgr_n4_evaluator_convergence_preregistered_v1.json",
        "source_config": "demo_t16_operator/configs/n2_pvgr_n0_trifidelity_development_v1.json",
        "pre_registration_attestation": "demo_t16_operator/configs/n2_pvgr_n5_d1_paired_residual_preregistration_attestation_v1.json",
        "formal_output": "demo_t16_operator/results/n2_pvgr_n5_d1_paired_residual_v1",
    }
    if any(config.get(key) != value for key, value in expected_paths.items()):
        raise ValueError("N5-D1 evidence path drifted")
    expected_selection = {
        "selection_is_post_n4_and_not_an_independent_test": True,
        "selected_pair_ids": ["p04", "p05"],
        "selected_physical_cell_count": 4,
        "rule": "the_two_N4_1_final_failures_and_their_same_field_same_stress_wide_controls",
    }
    if config.get("selection_contract") != expected_selection:
        raise ValueError("N5-D1 selection contract drifted")
    expected_toy = {
        "constant_grid_size": 9,
        "constant_ray_count": 8,
        "constant_step_count": 16,
        "weak_grid_size": 17,
        "weak_ray_count": 6,
        "weak_step_count": 48,
        "weak_center_scale": 0.0004,
        "weak_scale_difference": 0.0001,
        "rotation_grid_size": 17,
        "rotation_ray_count": 10,
        "rotation_step_count": 36,
        "rotation_angles_degrees": [0.0, 90.0],
        "toy_difference_step": 0.002,
    }
    if config.get("toy_contract") != expected_toy:
        raise ValueError("N5-D1 toy contract drifted")
    expected_decision = {
        "too_small_requires_every_nonraw_method_on_both_failed_cells_below_one_percent_of_the_raw_H1024_H2048_absolute_difference": True,
        "plausible_requires_at_least_one_nonraw_method_on_both_failed_cells_at_or_above_ten_percent": True,
        "intermediate_or_mixed_results_are_inconclusive": True,
        "D1_never_changes_the_N4_1_threshold_or_authorizes_a_reference": True,
        "D1_never_authorizes_field_JVP_VJP_or_neural_training": True,
    }
    if config.get("decision_contract") != expected_decision:
        raise ValueError("N5-D1 decision contract drifted")


def _load_parent_contract(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    amendment = _read_json(_resolve(str(config["parent_n4_config"])))
    scientific = _read_json(_resolve(str(config["scientific_n4_config"])))
    parent_n3 = _read_json(_resolve(str(scientific["parent_n3_config"])))
    source = _read_json(_resolve(str(config["source_config"])))
    if amendment["base_protocol_config"] != config["scientific_n4_config"]:
        raise ValueError("N5-D1 parent N4.1 does not bind the scientific N4 config")
    n4._validate_contract(scientific, parent_n3, source)
    parent_result = _read_json(_resolve(str(config["parent_n4_result"])))
    if parent_result.get("machine_decision") != (
        "FAIL_CLOSED_EVALUATOR_REMAINS_UNAUTHORIZED"
    ):
        raise ValueError("N5-D1 parent N4.1 decision drifted")
    if parent_result.get("counts", {}).get("final_reference_authorized_count") != 30:
        raise ValueError("N5-D1 parent N4.1 count drifted")
    return amendment, scientific, parent_n3, source


def _validate_preregistration(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    attestation_path = _resolve(str(config["pre_registration_attestation"]))
    if not attestation_path.is_file():
        raise FileNotFoundError("committed N5-D1 attestation is missing")
    attestation = _read_json(attestation_path)
    if attestation.get("schema") != "n2-pvgr-n5-d1-paired-residual-attestation-1.0":
        raise ValueError("N5-D1 attestation schema drifted")
    if attestation.get("formal_results_absent_at_creation") is not True:
        raise ValueError("N5-D1 attestation does not prove output absence")
    if attestation.get("formal_output") != config["formal_output"]:
        raise ValueError("N5-D1 attestation output path drifted")
    if attestation.get("config_sha256") != _sha256(config_path):
        raise ValueError("N5-D1 config does not match its attestation")
    protocol_commit = str(attestation["protocol_commit"])
    if attestation.get("repository_head_at_creation") != protocol_commit:
        raise ValueError("N5-D1 attestation was not created at the protocol commit")
    if _git(
        "merge-base", "--is-ancestor", protocol_commit, "HEAD", check=False
    ).returncode:
        raise ValueError("N5-D1 protocol commit is not an ancestor of HEAD")
    if set(attestation["attested_files"]) != set(config["attested_files"]):
        raise ValueError("N5-D1 attested file set drifted")
    for key, entry in attestation["attested_files"].items():
        expected = str(config["attested_files"][key])
        if entry["path"] != expected or _sha256(_resolve(expected)) != entry["sha256"]:
            raise ValueError(f"N5-D1 attested file drifted: {key}")
        frozen = _git("show", f"{protocol_commit}:{expected}").stdout
        if hashlib.sha256(frozen).hexdigest() != entry["sha256"]:
            raise ValueError(f"N5-D1 protocol file hash drifted: {key}")
    if _git(
        "ls-files", "--error-unmatch", _relative(attestation_path), check=False
    ).returncode:
        raise ValueError("N5-D1 attestation is not committed")
    paths = [
        _relative(attestation_path),
        *(str(value) for value in config["attested_files"].values()),
    ]
    if _git("status", "--porcelain", "--", *paths).stdout.strip():
        raise ValueError("N5-D1 preregistered files have uncommitted changes")
    return attestation


def _selected_cells(
    config: dict[str, Any],
    scientific: dict[str, Any],
    parent_n3: dict[str, Any],
) -> list[dict[str, Any]]:
    all_cells = {
        cell["cell_id"]: cell
        for cell in n4.expand_audit_cells(scientific, parent_n3)
    }
    cells = [copy.deepcopy(all_cells[cell_id]) for cell_id in config["selected_cells"]]
    if {cell["pair_id"] for cell in cells} != {"p04", "p05"}:
        raise ValueError("N5-D1 pair selection drifted")
    failed = [cell for cell in cells if cell["cell_id"].endswith("narrow__stress_1")]
    failed += [cell for cell in cells if cell["cell_id"].endswith("narrow__stress_3")]
    if len({cell["cell_id"] for cell in failed}) != 2:
        raise ValueError("N5-D1 must contain exactly the two N4.1 failures")
    return cells


def _toy_rig(*, angle: float = 27.0, detector_z: float = -0.03) -> SyntheticRayRig:
    return SyntheticRayRig(
        rig_id="n5-d1-toy",
        view_angle_degrees=float(angle),
        detector_u=0.04,
        detector_z=float(detector_z),
        aperture_radius=0.025,
        path_half_length=0.62,
        cone_u=0.02,
        cone_z=0.015,
        bend=0.0,
    )


def _toy_grid(size: int, *, radial: bool) -> torch.Tensor:
    axis = torch.linspace(-1.0, 1.0, int(size), dtype=torch.float64)
    zz, yy, xx = torch.meshgrid(axis, axis, axis, indexing="ij")
    if radial:
        return -torch.exp(
            -((xx / 0.39) ** 2 + (yy / 0.39) ** 2 + (zz / 0.39) ** 2)
        )
    return (
        -0.8
        * torch.exp(
            -(
                ((xx - 0.08) / 0.42) ** 2
                + ((yy + 0.05) / 0.31) ** 2
                + (zz / 0.36) ** 2
            )
        )
        + 0.03 * xx
        - 0.02 * yy
    )


def _toy_checks(config: dict[str, Any]) -> dict[str, Any]:
    contract = config["toy_contract"]
    delta = float(contract["toy_difference_step"])
    constant = evaluate_paired_residual(
        torch.zeros(
            (int(contract["constant_grid_size"]),) * 3,
            dtype=torch.float64,
        ),
        sample_pupil_sobol(int(contract["constant_ray_count"]), seed=1701),
        _toy_rig(),
        difference_step=delta,
        refractivity_scale=3e-3,
        step_count=int(contract["constant_step_count"]),
    )
    constant_max = max(
        float(torch.max(torch.abs(value)))
        for value in (
            constant.raw_separate_subtraction,
            constant.paired_naive,
            constant.paired_pairwise,
            constant.paired_neumaier,
            constant.separate_neumaier_subtraction,
        )
    )

    weak_grid = _toy_grid(int(contract["weak_grid_size"]), radial=False)
    weak_states = sample_pupil_sobol(int(contract["weak_ray_count"]), seed=1741)
    alpha = float(contract["weak_center_scale"])
    epsilon = float(contract["weak_scale_difference"])
    weak_outputs = []
    for scale in (alpha - epsilon, alpha, alpha + epsilon):
        weak_outputs.append(
            evaluate_paired_residual(
                weak_grid,
                weak_states,
                _toy_rig(),
                difference_step=delta,
                refractivity_scale=scale,
                step_count=int(contract["weak_step_count"]),
            ).paired_neumaier.detach().cpu().numpy()
        )
    finite_difference = (weak_outputs[2] - weak_outputs[0]) / (2.0 * epsilon)
    quadratic_prediction = 2.0 * weak_outputs[1] / alpha
    weak_metric = _relative_l2(finite_difference, quadratic_prediction)

    radial_grid = _toy_grid(int(contract["rotation_grid_size"]), radial=True)
    radial_states = sample_pupil_sobol(int(contract["rotation_ray_count"]), seed=1753)
    rotation_outputs = []
    for angle in contract["rotation_angles_degrees"]:
        rotation_outputs.append(
            evaluate_paired_residual(
                radial_grid,
                radial_states,
                _toy_rig(angle=float(angle), detector_z=0.0),
                difference_step=delta,
                refractivity_scale=3e-3,
                step_count=int(contract["rotation_step_count"]),
            ).paired_neumaier.detach().cpu().numpy()
        )
    rotation_metric = _relative_l2(rotation_outputs[0], rotation_outputs[1])
    gates = config["gates"]
    return {
        "constant_maximum_absolute_output": constant_max,
        "constant_gate_met": constant_max
        <= float(gates["maximum_constant_absolute_output"]),
        "weak_quadratic_finite_difference_relative_l2": weak_metric,
        "weak_gate_met": weak_metric
        <= float(gates["maximum_weak_quadratic_finite_difference_relative_l2"]),
        "rotation_equivariance_relative_l2": rotation_metric,
        "rotation_gate_met": rotation_metric
        <= float(gates["maximum_rotation_equivariance_relative_l2"]),
    }


def _method_arrays(evaluation: Any) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(getattr(evaluation, name).detach().cpu(), dtype=np.float64)
        for name in (
            "raw_separate_subtraction",
            *NONRAW_METHODS,
        )
    }


def _run_level(
    cell: dict[str, Any],
    source: dict[str, Any],
    *,
    step_count: int,
) -> dict[str, Any]:
    execution_case = n4.n3._execution_case(cell)
    values, states, rig = n4.bridge._build_case_context(execution_case, source)
    scale = float(source["base_refractivity_scale"]) * float(
        cell["dimensionless_stress_multiplier"]
    )
    delta = float(source["difference_step"])
    started = time.perf_counter()
    evaluation = evaluate_paired_residual(
        values,
        states,
        rig,
        difference_step=delta,
        refractivity_scale=scale,
        step_count=int(step_count),
    )
    paired_elapsed = time.perf_counter() - started

    certificate = source["certificate"]
    verification_started = time.perf_counter()
    frozen_curved = path_integrated_deflection(
        values,
        evaluation.trace,
        gradient_mode="central",
        difference_step=delta,
        refractivity_scale=scale,
        create_graph=False,
        detach_path=False,
    )
    frozen_straight = build_straight_path_state(
        values,
        states,
        rig,
        difference_step=delta,
        refractivity_scale=scale,
        step_count=int(step_count),
        frustum_half_width_u=float(certificate["frustum_half_width_u"]),
        frustum_half_width_v=float(certificate["frustum_half_width_v"]),
    ).projected_outputs
    verification_elapsed = time.perf_counter() - verification_started
    frozen_residual = frozen_curved.detach() - frozen_straight.detach()
    methods = _method_arrays(evaluation)
    frozen_arrays = {
        "curved": np.asarray(frozen_curved.detach().cpu(), dtype=np.float64),
        "straight": np.asarray(frozen_straight.detach().cpu(), dtype=np.float64),
        "residual": np.asarray(frozen_residual.detach().cpu(), dtype=np.float64),
    }
    equivalence_arrays = {
        "paired_curved_naive": np.asarray(
            evaluation.curved_output_naive.detach().cpu(), dtype=np.float64
        ),
        "paired_straight_naive": np.asarray(
            evaluation.straight_output_naive.detach().cpu(), dtype=np.float64
        ),
        "frozen_curved": frozen_arrays["curved"],
        "frozen_straight": frozen_arrays["straight"],
        "frozen_residual": frozen_arrays["residual"],
    }
    route_equivalence = {
        "curved_relative_l2": _relative_l2(
            np.asarray(evaluation.curved_output_naive.detach().cpu()),
            frozen_arrays["curved"],
        ),
        "straight_relative_l2": _relative_l2(
            np.asarray(evaluation.straight_output_naive.detach().cpu()),
            frozen_arrays["straight"],
        ),
        "raw_residual_relative_l2": _relative_l2(
            methods["raw_separate_subtraction"],
            frozen_arrays["residual"],
        ),
    }
    return {
        "cell_id": cell["cell_id"],
        "pair_id": cell["pair_id"],
        "role": cell["role"],
        "step_count": int(step_count),
        "methods": {
            name: {
                "values": array.tolist(),
                "sha256": _array_sha256(array),
                "l2_norm": float(np.linalg.norm(array)),
            }
            for name, array in methods.items()
        },
        "equivalence_arrays": {
            name: {
                "values": array.tolist(),
                "sha256": _array_sha256(array),
                "l2_norm": float(np.linalg.norm(array)),
            }
            for name, array in equivalence_arrays.items()
        },
        "route_equivalence": route_equivalence,
        "diagnostics": {
            "minimum_domain_margin": evaluation.minimum_paired_domain_margin,
            "minimum_stencil_margin": evaluation.minimum_paired_stencil_margin,
            "maximum_direction_norm_error": (
                evaluation.maximum_paired_direction_norm_error
            ),
        },
        "cost": {
            "paired_wall_seconds": float(paired_elapsed),
            "frozen_route_verification_wall_seconds": float(verification_elapsed),
            "paired_logical_point_queries": int(
                evaluation.query_accounting.total_field_point_queries
            ),
            "verification_logical_point_queries": 14 * len(states) * int(step_count),
            "total_audit_logical_point_queries": 56 * len(states) * int(step_count),
        },
    }


def _cell_summary(
    cell: dict[str, Any],
    levels: dict[int, dict[str, Any]],
    parent_result: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    arrays = {
        step: {
            method: np.asarray(level["methods"][method]["values"], dtype=np.float64)
            for method in config["accumulation_methods"]
        }
        for step, level in levels.items()
    }
    adjacent = {
        method: _relative_l2(arrays[1024][method], arrays[2048][method])
        for method in config["accumulation_methods"]
    }
    raw_difference = arrays[1024]["raw_separate_subtraction"] - arrays[2048][
        "raw_separate_subtraction"
    ]
    raw_difference_l2 = float(np.linalg.norm(raw_difference))
    if raw_difference_l2 <= 0.0:
        raise ValueError("N5-D1 raw refinement difference is not scoreable")
    accumulation: dict[str, dict[str, float]] = {}
    for method in NONRAW_METHODS:
        difference = arrays[2048][method] - arrays[2048]["raw_separate_subtraction"]
        difference_l2 = float(np.linalg.norm(difference))
        accumulation[method] = {
            "absolute_l2": difference_l2,
            "relative_to_raw_residual_l2": difference_l2
            / max(float(np.linalg.norm(arrays[2048]["raw_separate_subtraction"])), 1e-30),
            "fraction_of_raw_h1024_h2048_absolute_difference": (
                difference_l2 / raw_difference_l2
            ),
        }
    parent_cell = next(
        row for row in parent_result["cells"] if row["cell_id"] == cell["cell_id"]
    )
    parent_metric = float(
        parent_cell["d1024_to_d2048"]["matched_residual_relative_l2"]
    )
    return {
        "cell_id": cell["cell_id"],
        "pair_id": cell["pair_id"],
        "role": cell["role"],
        "parent_n4_final_authorized": bool(
            parent_cell["final_cellwise_reference_authorized"]
        ),
        "parent_n4_raw_adjacent_relative_l2": parent_metric,
        "d1_raw_adjacent_relative_l2": adjacent["raw_separate_subtraction"],
        "parent_n4_metric_absolute_difference": abs(
            adjacent["raw_separate_subtraction"] - parent_metric
        ),
        "adjacent_relative_l2_by_method": adjacent,
        "raw_h1024_h2048_absolute_difference_l2": raw_difference_l2,
        "accumulation_at_h2048": accumulation,
        "maximum_nonraw_accumulation_fraction_of_refinement": max(
            row["fraction_of_raw_h1024_h2048_absolute_difference"]
            for row in accumulation.values()
        ),
        "maximum_route_equivalence_relative_l2": max(
            value
            for level in levels.values()
            for value in level["route_equivalence"].values()
        ),
    }


def _decision(
    *,
    contract_gates_pass: bool,
    failed_cell_maximum_fractions: list[float],
    gates: dict[str, Any],
) -> str:
    if not contract_gates_pass:
        return "D1_CONTRACT_OR_REPRODUCTION_FAILED_CLOSED"
    if len(failed_cell_maximum_fractions) != 2:
        raise ValueError("N5-D1 decision requires exactly two failed cells")
    if all(
        value
        <= float(gates["maximum_accumulation_fraction_of_refinement_for_too_small"])
        for value in failed_cell_maximum_fractions
    ):
        return "D1_ACCUMULATION_ORDER_TOO_SMALL_TO_EXPLAIN_N4_FLOOR"
    if all(
        value
        >= float(gates["minimum_accumulation_fraction_of_refinement_for_plausible"])
        for value in failed_cell_maximum_fractions
    ):
        return "D1_ACCUMULATION_ORDER_PLAUSIBLY_EXPLAINS_N4_FLOOR"
    return "D1_ACCUMULATION_ORDER_INCONCLUSIVE"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "cell_id",
        "pair_id",
        "role",
        "parent_n4_final_authorized",
        "parent_n4_raw_adjacent_relative_l2",
        "d1_raw_adjacent_relative_l2",
        "parent_n4_metric_absolute_difference",
        "raw_h1024_h2048_absolute_difference_l2",
        "maximum_nonraw_accumulation_fraction_of_refinement",
        "maximum_route_equivalence_relative_l2",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def _plot(result: dict[str, Any], path: Path) -> None:
    cells = result["cells"]
    methods = result["accumulation_methods"]
    labels = [
        row["cell_id"].replace("smooth-s1871-orientation_58-", "").replace(
            "__stress_", " s"
        )
        for row in cells
    ]
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    x = np.arange(len(cells))
    for method in methods:
        axes[0].plot(
            x,
            [row["adjacent_relative_l2_by_method"][method] for row in cells],
            marker="o",
            label=method.replace("_subtraction", ""),
        )
    axes[0].axhline(1.25e-3, color="#a44a3f", linestyle="--", label="N4.1 gate")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, labels, rotation=24, ha="right")
    axes[0].set_ylabel("H1024-H2048 residual relative-L2")
    axes[0].set_title("Same discrete observable")
    axes[0].grid(alpha=0.22)
    axes[0].legend(fontsize=7)

    failed = [row for row in cells if not row["parent_n4_final_authorized"]]
    display_floor = 1e-18
    width = 0.18
    failed_x = np.arange(len(failed))
    for index, method in enumerate(NONRAW_METHODS):
        axes[1].bar(
            failed_x + (index - 1.5) * width,
            [
                max(
                    float(
                        row["accumulation_at_h2048"][method][
                            "fraction_of_raw_h1024_h2048_absolute_difference"
                        ]
                    ),
                    display_floor,
                )
                for row in failed
            ],
            width=width,
            label=method.replace("_subtraction", ""),
        )
    axes[1].axhline(0.01, color="#2f766d", linestyle="--", label="1% too-small")
    axes[1].axhline(0.1, color="#a44a3f", linestyle=":", label="10% plausible")
    axes[1].set_yscale("log")
    axes[1].set_xticks(
        failed_x,
        [row["cell_id"].split("__")[-1] for row in failed],
    )
    axes[1].set_ylabel("accumulation difference / H-refinement difference")
    axes[1].set_title("Can ordering explain the floor?")
    axes[1].grid(alpha=0.22)
    axes[1].legend(fontsize=7)

    gates = result["gates"]
    toy = result["toy_checks"]
    ratios = [
        toy["weak_quadratic_finite_difference_relative_l2"]
        / gates["maximum_weak_quadratic_finite_difference_relative_l2"],
        toy["rotation_equivariance_relative_l2"]
        / gates["maximum_rotation_equivariance_relative_l2"],
        result["maximum_route_equivalence_relative_l2"]
        / gates["maximum_frozen_route_equivalence_relative_l2"],
        result["maximum_parent_n4_metric_absolute_difference"]
        / gates["maximum_parent_n4_adjacent_metric_absolute_difference"],
    ]
    axes[2].bar(
        ["weak FD", "rotation", "route equiv.", "N4 replay"],
        [max(float(value), display_floor) for value in ratios],
        color=["#2f766d", "#4c78a8", "#c49a42", "#a44a3f"],
    )
    axes[2].axhline(1.0, color="#222222", linestyle="--", label="gate")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("metric / preregistered maximum")
    axes[2].set_title("Contract checks")
    axes[2].tick_params(axis="x", rotation=22)
    axes[2].grid(alpha=0.22)
    axes[2].legend(fontsize=8)
    axes[2].text(
        0.01,
        0.02,
        "exact zeros shown at 1e-18",
        transform=axes[2].transAxes,
        fontsize=7,
        color="#555555",
    )

    figure.suptitle(
        "N5-D1 paired residual accumulation audit (post-N4 selected; not algorithm performance)",
        fontsize=12,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _summary(result: dict[str, Any]) -> str:
    return "\n".join(
        (
            "# N2-PVGR N5-D1 paired residual accumulation audit",
            "",
            f"Machine decision: `{result['machine_decision']}`",
            "",
            "This is a post-N4 selected numerical-mechanism audit. It does not authorize a reference, field adjoint, reconstruction, neural training, real-data or paper claim.",
            "",
            f"- Selected cells: {result['selected_cell_count']} (two N4.1 failures plus matched wide controls)",
            f"- Numerical levels: {result['level_evaluation_count']} (H1024/H2048)",
            f"- Maximum frozen-route equivalence relative-L2: {result['maximum_route_equivalence_relative_l2']:.6e}",
            f"- Maximum parent-N4 adjacent-metric replay difference: {result['maximum_parent_n4_metric_absolute_difference']:.6e}",
            f"- Failed-cell accumulation/refinement fractions: {result['failed_cell_maximum_accumulation_fractions']}",
            f"- Paired logical queries: {result['paired_logical_point_queries']:,}",
            f"- Extra equivalence-audit queries: {result['verification_logical_point_queries']:,}",
            "",
            "The next allowed action is only the separately preregistered D2 H4096/H8192 truncation probe when the D1 contract passes.",
            "",
        )
    )


def _manifest(
    staging: Path,
    output: Path,
    sources: dict[str, Path],
    files: list[str],
) -> dict[str, Any]:
    return {
        "schema": "n2-pvgr-n5-d1-paired-residual-manifest-1.0",
        "files": {
            key: {"path": _relative(path), "sha256": _sha256(path)}
            for key, path in sources.items()
        }
        | {
            name: {
                "path": _relative(output / name),
                "sha256": _sha256(staging / name),
            }
            for name in files
        },
    }


def run(config_path: Path, output: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = _read_json(config_path)
    _validate_contract(config)
    _, scientific, parent_n3, source = _load_parent_contract(config)
    attestation = _validate_preregistration(config, config_path)
    cells = _selected_cells(config, scientific, parent_n3)
    parent_result = _read_json(_resolve(str(config["parent_n4_result"])))
    output = output.resolve()
    staging = output.with_name(output.name + ".staging")
    if output.exists() or staging.exists():
        raise FileExistsError("N5-D1 output or staging directory already exists")
    staging.mkdir(parents=True)

    adapted_source = n4._source_for_run(source, scientific)
    level_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    for cell in cells:
        levels: dict[int, dict[str, Any]] = {}
        for step_count in config["step_counts"]:
            level = _run_level(cell, adapted_source, step_count=int(step_count))
            levels[int(step_count)] = level
            level_rows.append(level)
            print(
                f"[N5-D1] {cell['cell_id']} H{int(step_count)} "
                f"route={max(level['route_equivalence'].values()):.3e}",
                flush=True,
            )
        cell_rows.append(_cell_summary(cell, levels, parent_result, config))

    toy = _toy_checks(config)
    gates = config["gates"]
    maximum_route = max(
        row["maximum_route_equivalence_relative_l2"] for row in cell_rows
    )
    maximum_parent_replay = max(
        row["parent_n4_metric_absolute_difference"] for row in cell_rows
    )
    contract_gates_pass = (
        bool(toy["constant_gate_met"])
        and bool(toy["weak_gate_met"])
        and bool(toy["rotation_gate_met"])
        and maximum_route
        <= float(gates["maximum_frozen_route_equivalence_relative_l2"])
        and maximum_parent_replay
        <= float(gates["maximum_parent_n4_adjacent_metric_absolute_difference"])
    )
    failed_rows = [row for row in cell_rows if not row["parent_n4_final_authorized"]]
    failed_fractions = [
        float(row["maximum_nonraw_accumulation_fraction_of_refinement"])
        for row in failed_rows
    ]
    machine_decision = _decision(
        contract_gates_pass=contract_gates_pass,
        failed_cell_maximum_fractions=failed_fractions,
        gates=gates,
    )
    result = {
        "schema": "n2-pvgr-n5-d1-paired-residual-result-1.0",
        "candidate_id": config["candidate_id"],
        "protocol_commit": attestation["protocol_commit"],
        "machine_decision": machine_decision,
        "contract_gates_pass": contract_gates_pass,
        "selection_is_post_n4_and_not_an_independent_test": True,
        "selected_cell_count": len(cell_rows),
        "level_evaluation_count": len(level_rows),
        "accumulation_methods": config["accumulation_methods"],
        "gates": gates,
        "toy_checks": toy,
        "maximum_route_equivalence_relative_l2": maximum_route,
        "maximum_parent_n4_metric_absolute_difference": maximum_parent_replay,
        "failed_cell_maximum_accumulation_fractions": failed_fractions,
        "paired_logical_point_queries": sum(
            row["cost"]["paired_logical_point_queries"] for row in level_rows
        ),
        "verification_logical_point_queries": sum(
            row["cost"]["verification_logical_point_queries"] for row in level_rows
        ),
        "total_audit_logical_point_queries": sum(
            row["cost"]["total_audit_logical_point_queries"] for row in level_rows
        ),
        "paired_wall_seconds": sum(
            row["cost"]["paired_wall_seconds"] for row in level_rows
        ),
        "verification_wall_seconds": sum(
            row["cost"]["frozen_route_verification_wall_seconds"] for row in level_rows
        ),
        "cells": cell_rows,
        "authorizations": {
            **config["claim_authorizations"],
            "d2_refinement_probe_preregistration_allowed": contract_gates_pass,
            "replace_n4_metric_or_threshold": False,
        },
        "limitations": config["known_limitations"],
    }
    (staging / "config_snapshot.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (staging / "levels.json").write_text(
        json.dumps(level_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (staging / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (staging / "toy_checks.json").write_text(
        json.dumps(toy, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_csv(staging / "cell_summary.csv", cell_rows)
    (staging / "summary.md").write_text(_summary(result), encoding="utf-8")
    _plot(result, staging / "n2_pvgr_n5_d1_paired_residual.png")
    files = [
        "cell_summary.csv",
        "config_snapshot.json",
        "levels.json",
        "n2_pvgr_n5_d1_paired_residual.png",
        "result.json",
        "summary.md",
        "toy_checks.json",
    ]
    sources = {
        "config": config_path,
        "attestation": _resolve(str(config["pre_registration_attestation"])),
        "runner": Path(__file__).resolve(),
        "paired_kernel": _resolve(str(config["attested_files"]["paired_kernel"])),
        "parent_n4_result": _resolve(str(config["parent_n4_result"])),
    }
    (staging / "manifest.json").write_text(
        json.dumps(
            _manifest(staging, output, sources, files),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    staging.replace(output)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = _read_json(args.config.resolve())
    output = args.output or _resolve(str(config["formal_output"]))
    result = run(args.config.resolve(), output.resolve())
    print(json.dumps({
        "machine_decision": result["machine_decision"],
        "contract_gates_pass": result["contract_gates_pass"],
        "failed_cell_maximum_accumulation_fractions": result[
            "failed_cell_maximum_accumulation_fractions"
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
