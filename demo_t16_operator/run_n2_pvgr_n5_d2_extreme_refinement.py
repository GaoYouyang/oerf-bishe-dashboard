#!/usr/bin/env python3
"""Run the preregistered N5-D2 H4096/H8192 residual-tail probe."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from . import run_n2_pvgr_n5_d1_paired_residual as d1
    from .cancellation_aware_residual import evaluate_paired_residual
except ImportError:
    import run_n2_pvgr_n5_d1_paired_residual as d1
    from cancellation_aware_residual import evaluate_paired_residual


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT
    / "demo_t16_operator/configs/n2_pvgr_n5_d2_extreme_refinement_preregistered_v1.json"
)
METHODS = ("raw_separate_subtraction", *d1.NONRAW_METHODS)


def _read_json(path: Path) -> dict[str, Any] | list[Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], cwd=ROOT, check=check, capture_output=True)


def _validate_contract(config: dict[str, Any]) -> None:
    if config.get("schema") != "n2-pvgr-n5-d2-extreme-refinement-preregistered-1.0":
        raise ValueError("N5-D2 schema drifted")
    if config.get("candidate_id") != "N2-PVGR-N5-D2-H8192-TAIL4":
        raise ValueError("N5-D2 identifier drifted")
    if config.get("status") != (
        "preregistered_post_d1_selected_tail_probe_no_algorithm_authorization"
    ):
        raise ValueError("N5-D2 status drifted")
    if (config.get("device"), config.get("dtype")) != ("cpu", "float64"):
        raise ValueError("N5-D2 requires CPU float64")
    if int(config.get("population_count", 0)) != 256:
        raise ValueError("N5-D2 requires 256 common rays")
    if int(config.get("parent_step_count", 0)) != 2048:
        raise ValueError("N5-D2 parent level drifted")
    if tuple(int(value) for value in config.get("new_step_counts", ())) != (4096, 8192):
        raise ValueError("N5-D2 new levels drifted")
    expected_cells = (
        "smooth-s1871-orientation_58-narrow__stress_1",
        "smooth-s1871-orientation_58-wide__stress_1",
        "smooth-s1871-orientation_58-narrow__stress_3",
        "smooth-s1871-orientation_58-wide__stress_3",
    )
    if tuple(config.get("selected_cells", ())) != expected_cells:
        raise ValueError("N5-D2 selected cells drifted")
    if config.get("reference_method") != "paired_neumaier":
        raise ValueError("N5-D2 reference method drifted")
    if tuple(config.get("monitor_methods", ())) != METHODS:
        raise ValueError("N5-D2 monitor methods drifted")
    expected_gates = {
        "maximum_h4096_to_h8192_reference_relative_l2": 0.000625,
        "maximum_adjacent_contraction_ratio": 0.5,
        "maximum_h8192_raw_paired_fraction_of_final_refinement": 0.01,
        "minimum_finite_ray_fraction": 1.0,
        "minimum_domain_margin": 0.0,
        "minimum_stencil_margin": 0.0,
        "maximum_direction_norm_error": 1e-12,
    }
    if {key: float(value) for key, value in config.get("gates", {}).items()} != expected_gates:
        raise ValueError("N5-D2 gates drifted")
    expected_selection = {
        "selection_is_post_n4_and_post_d1_not_an_independent_test": True,
        "selected_pair_ids": ["p04", "p05"],
        "selected_physical_cell_count": 4,
        "rule": "reuse_exactly_the_four_D1_cells_without_reselection",
    }
    if config.get("selection_contract") != expected_selection:
        raise ValueError("N5-D2 selection contract drifted")
    expected_decision = {
        "resolved_requires_every_gate_on_all_four_selected_cells": True,
        "final_adjacent_gate_is_half_the_original_N4_1_0_00125_gate": True,
        "contraction_gate_reuses_the_original_N4_maximum_0_5": True,
        "richardson_and_geometric_tail_values_are_indicators_not_proofs": True,
        "D2_does_not_reselect_cells_or_change_D1": True,
        "D2_does_not_authorize_field_JVP_VJP_or_neural_training": True,
    }
    if config.get("decision_contract") != expected_decision:
        raise ValueError("N5-D2 decision contract drifted")
    expected_paths = {
        "parent_d1_config": "demo_t16_operator/configs/n2_pvgr_n5_d1_paired_residual_preregistered_v1.json",
        "parent_d1_attestation": "demo_t16_operator/configs/n2_pvgr_n5_d1_paired_residual_preregistration_attestation_v1.json",
        "parent_d1_result": "demo_t16_operator/results/n2_pvgr_n5_d1_paired_residual_v1/result.json",
        "parent_d1_levels": "demo_t16_operator/results/n2_pvgr_n5_d1_paired_residual_v1/levels.json",
        "parent_d1_manifest": "demo_t16_operator/results/n2_pvgr_n5_d1_paired_residual_v1/manifest.json",
        "parent_d1_validation": "demo_t16_operator/results/n2_pvgr_n5_d1_paired_residual_v1/validation_report.json",
        "source_config": "demo_t16_operator/configs/n2_pvgr_n0_trifidelity_development_v1.json",
        "pre_registration_attestation": "demo_t16_operator/configs/n2_pvgr_n5_d2_extreme_refinement_attestation_v1.json",
        "formal_output": "demo_t16_operator/results/n2_pvgr_n5_d2_extreme_refinement_v1",
    }
    if any(config.get(key) != value for key, value in expected_paths.items()):
        raise ValueError("N5-D2 evidence path drifted")
    if not all(value is False for value in config.get("claim_authorizations", {}).values()):
        raise ValueError("N5-D2 broad claim authorizations must remain false")


def _validate_parent_manifest(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = _resolve(str(config["parent_d1_manifest"]))
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("schema") != (
        "n2-pvgr-n5-d1-paired-residual-manifest-1.0"
    ):
        raise ValueError("N5-D2 parent D1 manifest drifted")
    required = {
        "result.json": config["parent_d1_result"],
        "levels.json": config["parent_d1_levels"],
    }
    for key, expected_path in required.items():
        entry = manifest["files"].get(key)
        if entry is None or entry["path"] != expected_path:
            raise ValueError(f"N5-D2 parent D1 manifest path drifted: {key}")
        path = _resolve(str(expected_path))
        if not path.is_file() or _sha256(path) != entry["sha256"]:
            raise ValueError(f"N5-D2 parent D1 manifest hash drifted: {key}")
    return manifest


def _load_parent_contract(
    config: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
]:
    d1_config_path = _resolve(str(config["parent_d1_config"]))
    parent_config = _read_json(d1_config_path)
    if not isinstance(parent_config, dict):
        raise ValueError("N5-D2 parent D1 config is invalid")
    d1._validate_contract(parent_config)
    d1_attestation = d1._validate_preregistration(parent_config, d1_config_path)
    _, scientific, parent_n3, source = d1._load_parent_contract(parent_config)
    cells = d1._selected_cells(parent_config, scientific, parent_n3)
    if [cell["cell_id"] for cell in cells] != config["selected_cells"]:
        raise ValueError("N5-D2 does not exactly reuse D1 cells")
    parent_result = _read_json(_resolve(str(config["parent_d1_result"])))
    parent_validation = _read_json(_resolve(str(config["parent_d1_validation"])))
    parent_levels = _read_json(_resolve(str(config["parent_d1_levels"])))
    if not isinstance(parent_result, dict) or not isinstance(parent_validation, dict):
        raise ValueError("N5-D2 parent D1 result bundle is invalid")
    if not isinstance(parent_levels, list):
        raise ValueError("N5-D2 parent D1 levels are invalid")
    if parent_result.get("protocol_commit") != d1_attestation["protocol_commit"]:
        raise ValueError("N5-D2 parent D1 protocol binding drifted")
    if parent_result.get("machine_decision") != (
        "D1_ACCUMULATION_ORDER_TOO_SMALL_TO_EXPLAIN_N4_FLOOR"
    ) or parent_result.get("contract_gates_pass") is not True:
        raise ValueError("N5-D2 parent D1 decision does not allow D2")
    if parent_result.get("authorizations", {}).get(
        "d2_refinement_probe_preregistration_allowed"
    ) is not True:
        raise ValueError("N5-D2 was not opened by D1")
    if parent_validation.get("valid") is not True or parent_validation.get(
        "machine_decision"
    ) != parent_result["machine_decision"]:
        raise ValueError("N5-D2 parent D1 validation drifted")
    _validate_parent_manifest(config)
    return parent_config, parent_result, parent_levels, source, cells


def _validate_preregistration(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    attestation_path = _resolve(str(config["pre_registration_attestation"]))
    if not attestation_path.is_file():
        raise FileNotFoundError("committed N5-D2 attestation is missing")
    attestation = _read_json(attestation_path)
    if not isinstance(attestation, dict) or attestation.get("schema") != (
        "n2-pvgr-n5-d2-extreme-refinement-attestation-1.0"
    ):
        raise ValueError("N5-D2 attestation schema drifted")
    if attestation.get("formal_results_absent_at_creation") is not True:
        raise ValueError("N5-D2 attestation does not prove output absence")
    if attestation.get("formal_output") != config["formal_output"]:
        raise ValueError("N5-D2 attestation output path drifted")
    if attestation.get("config_sha256") != _sha256(config_path):
        raise ValueError("N5-D2 config does not match its attestation")
    protocol_commit = str(attestation["protocol_commit"])
    if attestation.get("repository_head_at_creation") != protocol_commit:
        raise ValueError("N5-D2 attestation was not created at the protocol commit")
    if _git("merge-base", "--is-ancestor", protocol_commit, "HEAD", check=False).returncode:
        raise ValueError("N5-D2 protocol commit is not an ancestor of HEAD")
    if set(attestation["attested_files"]) != set(config["attested_files"]):
        raise ValueError("N5-D2 attested file set drifted")
    for key, entry in attestation["attested_files"].items():
        expected = str(config["attested_files"][key])
        if entry["path"] != expected or _sha256(_resolve(expected)) != entry["sha256"]:
            raise ValueError(f"N5-D2 attested file drifted: {key}")
        frozen = _git("show", f"{protocol_commit}:{expected}").stdout
        if hashlib.sha256(frozen).hexdigest() != entry["sha256"]:
            raise ValueError(f"N5-D2 protocol file hash drifted: {key}")
    if _git(
        "ls-files", "--error-unmatch", _relative(attestation_path), check=False
    ).returncode:
        raise ValueError("N5-D2 attestation is not committed")
    paths = [
        _relative(attestation_path),
        *(str(value) for value in config["attested_files"].values()),
    ]
    if _git("status", "--porcelain", "--", *paths).stdout.strip():
        raise ValueError("N5-D2 preregistered files have uncommitted changes")
    return attestation


def _array(entry: dict[str, Any], *, label: str) -> np.ndarray:
    array = np.asarray(entry["values"], dtype=np.float64)
    if array.shape != (256, 2) or not np.all(np.isfinite(array)):
        raise ValueError(f"N5-D2 invalid parent array: {label}")
    if entry["sha256"] != d1._array_sha256(array):
        raise ValueError(f"N5-D2 parent array hash drifted: {label}")
    return array


def _parent_h2048_levels(
    config: dict[str, Any],
    levels: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    selected = set(config["selected_cells"])
    rows = {
        row["cell_id"]: row
        for row in levels
        if row["cell_id"] in selected
        and int(row["step_count"]) == int(config["parent_step_count"])
    }
    if set(rows) != selected:
        raise ValueError("N5-D2 parent H2048 level set drifted")
    for cell_id, row in rows.items():
        if set(row["methods"]) != set(METHODS):
            raise ValueError(f"N5-D2 parent methods drifted: {cell_id}")
        for method in METHODS:
            _array(row["methods"][method], label=f"{cell_id} H2048 {method}")
    return rows


def _run_level(
    cell: dict[str, Any],
    source: dict[str, Any],
    *,
    step_count: int,
) -> dict[str, Any]:
    execution_case = d1.n4.n3._execution_case(cell)
    values, states, rig = d1.n4.bridge._build_case_context(execution_case, source)
    scale = float(source["base_refractivity_scale"]) * float(
        cell["dimensionless_stress_multiplier"]
    )
    started = time.perf_counter()
    evaluation = evaluate_paired_residual(
        values,
        states,
        rig,
        difference_step=float(source["difference_step"]),
        refractivity_scale=scale,
        step_count=int(step_count),
    )
    elapsed = time.perf_counter() - started
    methods = d1._method_arrays(evaluation)
    finite_per_ray = np.ones(len(states), dtype=bool)
    for array in methods.values():
        finite_per_ray &= np.all(np.isfinite(array), axis=1)
    finite_per_ray &= np.all(
        np.isfinite(np.asarray(evaluation.trace.positions.detach().cpu())),
        axis=(1, 2),
    )
    finite_per_ray &= np.all(
        np.isfinite(np.asarray(evaluation.trace.directions.detach().cpu())),
        axis=(1, 2),
    )
    return {
        "cell_id": cell["cell_id"],
        "pair_id": cell["pair_id"],
        "role": cell["role"],
        "step_count": int(step_count),
        "methods": {
            name: {
                "values": array.tolist(),
                "sha256": d1._array_sha256(array),
                "l2_norm": float(np.linalg.norm(array)),
            }
            for name, array in methods.items()
        },
        "diagnostics": {
            "finite_ray_fraction": float(np.mean(finite_per_ray)),
            "minimum_domain_margin": evaluation.minimum_paired_domain_margin,
            "minimum_stencil_margin": evaluation.minimum_paired_stencil_margin,
            "maximum_direction_norm_error": evaluation.maximum_paired_direction_norm_error,
        },
        "cost": {
            "wall_seconds": float(elapsed),
            "logical_point_queries": int(
                evaluation.query_accounting.total_field_point_queries
            ),
        },
    }


def _cell_metrics(
    cell: dict[str, Any],
    parent: dict[str, Any],
    new_levels: dict[int, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    reference = str(config["reference_method"])
    arrays: dict[int, dict[str, np.ndarray]] = {
        2048: {
            method: _array(
                parent["methods"][method],
                label=f"{cell['cell_id']} H2048 {method}",
            )
            for method in METHODS
        }
    }
    for step, level in new_levels.items():
        arrays[step] = {
            method: np.asarray(level["methods"][method]["values"], dtype=np.float64)
            for method in METHODS
        }
    d2048_4096 = d1._relative_l2(arrays[2048][reference], arrays[4096][reference])
    d4096_8192 = d1._relative_l2(arrays[4096][reference], arrays[8192][reference])
    contraction = d4096_8192 / max(d2048_4096, 1e-30)
    observed_order = -math.log2(contraction) if contraction > 0.0 else None
    final_difference_l2 = float(
        np.linalg.norm(arrays[4096][reference] - arrays[8192][reference])
    )
    raw_paired_l2 = float(
        np.linalg.norm(
            arrays[8192]["raw_separate_subtraction"] - arrays[8192][reference]
        )
    )
    raw_paired_fraction = raw_paired_l2 / max(final_difference_l2, 1e-30)
    diagnostics = {
        "minimum_finite_ray_fraction": min(
            float(level["diagnostics"]["finite_ray_fraction"])
            for level in new_levels.values()
        ),
        "minimum_domain_margin": min(
            float(level["diagnostics"]["minimum_domain_margin"])
            for level in new_levels.values()
        ),
        "minimum_stencil_margin": min(
            float(level["diagnostics"]["minimum_stencil_margin"])
            for level in new_levels.values()
        ),
        "maximum_direction_norm_error": max(
            float(level["diagnostics"]["maximum_direction_norm_error"])
            for level in new_levels.values()
        ),
    }
    gates = config["gates"]
    gate_results = {
        "final_adjacent_relative_l2_gate_met": d4096_8192
        <= float(gates["maximum_h4096_to_h8192_reference_relative_l2"]),
        "adjacent_contraction_gate_met": contraction
        <= float(gates["maximum_adjacent_contraction_ratio"]),
        "raw_paired_fraction_gate_met": raw_paired_fraction
        <= float(gates["maximum_h8192_raw_paired_fraction_of_final_refinement"]),
        "finite_ray_gate_met": diagnostics["minimum_finite_ray_fraction"]
        >= float(gates["minimum_finite_ray_fraction"]),
        "domain_margin_gate_met": diagnostics["minimum_domain_margin"]
        >= float(gates["minimum_domain_margin"]),
        "stencil_margin_gate_met": diagnostics["minimum_stencil_margin"]
        >= float(gates["minimum_stencil_margin"]),
        "direction_norm_gate_met": diagnostics["maximum_direction_norm_error"]
        <= float(gates["maximum_direction_norm_error"]),
    }
    geometric_tail = (
        d4096_8192 * contraction / (1.0 - contraction)
        if 0.0 <= contraction < 1.0
        else None
    )
    return {
        "cell_id": cell["cell_id"],
        "pair_id": cell["pair_id"],
        "role": cell["role"],
        "d2048_to_d4096_reference_relative_l2": d2048_4096,
        "d4096_to_d8192_reference_relative_l2": d4096_8192,
        "adjacent_contraction_ratio": contraction,
        "observed_order": observed_order,
        "fixed_second_order_richardson_correction_relative_l2": d4096_8192 / 3.0,
        "observed_geometric_tail_indicator_relative_l2": geometric_tail,
        "final_refinement_absolute_l2": final_difference_l2,
        "h8192_raw_paired_absolute_l2": raw_paired_l2,
        "h8192_raw_paired_fraction_of_final_refinement": raw_paired_fraction,
        "diagnostics": diagnostics,
        "gates": gate_results,
        "all_gates_pass": all(gate_results.values()),
    }


def _decision(rows: list[dict[str, Any]]) -> str:
    if len(rows) != 4 or len({row["cell_id"] for row in rows}) != 4:
        raise ValueError("N5-D2 decision requires four unique cells")
    return (
        "D2_SELECTED_TAIL_RESOLVED_AT_H8192"
        if all(bool(row["all_gates_pass"]) for row in rows)
        else "D2_SELECTED_TAIL_NOT_RESOLVED_AT_H8192"
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "cell_id",
        "pair_id",
        "role",
        "d2048_to_d4096_reference_relative_l2",
        "d4096_to_d8192_reference_relative_l2",
        "adjacent_contraction_ratio",
        "observed_order",
        "fixed_second_order_richardson_correction_relative_l2",
        "observed_geometric_tail_indicator_relative_l2",
        "h8192_raw_paired_fraction_of_final_refinement",
        "all_gates_pass",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def _plot(result: dict[str, Any], path: Path) -> None:
    rows = result["cells"]
    labels = [
        row["cell_id"].replace("smooth-s1871-orientation_58-", "").replace(
            "__stress_", " s"
        )
        for row in rows
    ]
    x = np.arange(len(rows))
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    axes[0].plot(
        x,
        [row["d2048_to_d4096_reference_relative_l2"] for row in rows],
        marker="o",
        label="H2048-H4096",
    )
    axes[0].plot(
        x,
        [row["d4096_to_d8192_reference_relative_l2"] for row in rows],
        marker="s",
        label="H4096-H8192",
    )
    axes[0].axhline(6.25e-4, color="#a44a3f", linestyle="--", label="final gate")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, labels, rotation=24, ha="right")
    axes[0].set_ylabel("paired-Neumaier residual relative-L2")
    axes[0].set_title("True step-refinement tail")
    axes[0].grid(alpha=0.22)
    axes[0].legend(fontsize=8)

    ratios = [row["adjacent_contraction_ratio"] for row in rows]
    bars = axes[1].bar(x, ratios, color="#397f76")
    axes[1].axhline(0.5, color="#a44a3f", linestyle="--", label="contraction gate")
    axes[1].set_xticks(x, labels, rotation=24, ha="right")
    axes[1].set_ylabel("d4096-8192 / d2048-4096")
    axes[1].set_title("Observed contraction")
    axes[1].grid(alpha=0.22)
    axes[1].legend(fontsize=8)
    for bar, row in zip(bars, rows, strict=True):
        order = row["observed_order"]
        label = "p=NA" if order is None else f"p={order:.2f}"
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )

    width = 0.36
    axes[2].bar(
        x - width / 2,
        [
            row["fixed_second_order_richardson_correction_relative_l2"]
            for row in rows
        ],
        width=width,
        label="p=2 Richardson correction",
        color="#c49a42",
    )
    axes[2].bar(
        x + width / 2,
        [
            max(float(row["h8192_raw_paired_fraction_of_final_refinement"]), 1e-18)
            for row in rows
        ],
        width=width,
        label="raw/paired fraction",
        color="#4c78a8",
    )
    axes[2].axhline(0.01, color="#a44a3f", linestyle="--", label="1% float gate")
    axes[2].set_yscale("log")
    axes[2].set_xticks(x, labels, rotation=24, ha="right")
    axes[2].set_ylabel("diagnostic magnitude (different units by bar)")
    axes[2].set_title("Tail indicator and float monitor")
    axes[2].grid(alpha=0.22)
    axes[2].legend(fontsize=7)

    figure.suptitle(
        "N5-D2 H4096/H8192 selected residual-tail probe (not algorithm performance)",
        fontsize=12,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _summary(result: dict[str, Any]) -> str:
    return "\n".join(
        (
            "# N2-PVGR N5-D2 H4096/H8192 selected tail probe",
            "",
            f"Machine decision: `{result['machine_decision']}`",
            "",
            "This is a post-N4/post-D1 four-cell numerical-tail probe. It does not authorize a fresh reference, field adjoint, reconstruction, neural training, real data, generalization or paper claim.",
            "",
            f"- Selected cells: {result['selected_cell_count']}",
            f"- New level evaluations: {result['new_level_evaluation_count']} (H4096/H8192)",
            f"- Maximum final adjacent relative-L2: {result['maximum_final_adjacent_relative_l2']:.6e}",
            f"- Maximum adjacent contraction ratio: {result['maximum_adjacent_contraction_ratio']:.6e}",
            f"- Maximum H8192 raw/paired refinement fraction: {result['maximum_raw_paired_fraction_of_final_refinement']:.6e}",
            f"- Logical point queries: {result['logical_point_queries']:,}",
            "",
            "A resolved result only opens a separately preregistered 32-cell adaptive-reference reconciliation.",
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
        "schema": "n2-pvgr-n5-d2-extreme-refinement-manifest-1.0",
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
    if not isinstance(config, dict):
        raise ValueError("N5-D2 config is invalid")
    _validate_contract(config)
    parent_config, parent_result, parent_levels, source, cells = _load_parent_contract(
        config
    )
    attestation = _validate_preregistration(config, config_path)
    parent_h2048 = _parent_h2048_levels(config, parent_levels)
    output = output.resolve()
    staging = output.with_name(output.name + ".staging")
    if output.exists() or staging.exists():
        raise FileExistsError("N5-D2 output or staging directory already exists")
    staging.mkdir(parents=True)

    scientific = _read_json(_resolve(str(parent_config["scientific_n4_config"])))
    if not isinstance(scientific, dict):
        raise ValueError("N5-D2 scientific N4 config is invalid")
    adapted_source = d1.n4._source_for_run(source, scientific)
    level_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    for cell in cells:
        levels: dict[int, dict[str, Any]] = {}
        for step_count in config["new_step_counts"]:
            level = _run_level(cell, adapted_source, step_count=int(step_count))
            levels[int(step_count)] = level
            level_rows.append(level)
            print(
                f"[N5-D2] {cell['cell_id']} H{int(step_count)} "
                f"margin={level['diagnostics']['minimum_stencil_margin']:.3e}",
                flush=True,
            )
        cell_rows.append(
            _cell_metrics(cell, parent_h2048[cell["cell_id"]], levels, config)
        )

    machine_decision = _decision(cell_rows)
    resolved = machine_decision == "D2_SELECTED_TAIL_RESOLVED_AT_H8192"
    result = {
        "schema": "n2-pvgr-n5-d2-extreme-refinement-result-1.0",
        "candidate_id": config["candidate_id"],
        "protocol_commit": attestation["protocol_commit"],
        "parent_d1_protocol_commit": parent_result["protocol_commit"],
        "machine_decision": machine_decision,
        "selection_is_post_n4_and_post_d1_not_an_independent_test": True,
        "parent_h2048_arrays_reused_without_recomputation": True,
        "selected_cell_count": len(cell_rows),
        "new_level_evaluation_count": len(level_rows),
        "reference_method": config["reference_method"],
        "monitor_methods": config["monitor_methods"],
        "gates": config["gates"],
        "maximum_final_adjacent_relative_l2": max(
            row["d4096_to_d8192_reference_relative_l2"] for row in cell_rows
        ),
        "maximum_adjacent_contraction_ratio": max(
            row["adjacent_contraction_ratio"] for row in cell_rows
        ),
        "maximum_raw_paired_fraction_of_final_refinement": max(
            row["h8192_raw_paired_fraction_of_final_refinement"] for row in cell_rows
        ),
        "logical_point_queries": sum(
            row["cost"]["logical_point_queries"] for row in level_rows
        ),
        "wall_seconds": sum(row["cost"]["wall_seconds"] for row in level_rows),
        "cells": cell_rows,
        "authorizations": {
            **config["claim_authorizations"],
            "adaptive_reference_reconciliation_preregistration_allowed": resolved,
            "selected_four_cell_tail_resolved": resolved,
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
    _write_csv(staging / "cell_summary.csv", cell_rows)
    (staging / "summary.md").write_text(_summary(result), encoding="utf-8")
    _plot(result, staging / "n2_pvgr_n5_d2_extreme_refinement.png")
    files = [
        "cell_summary.csv",
        "config_snapshot.json",
        "levels.json",
        "n2_pvgr_n5_d2_extreme_refinement.png",
        "result.json",
        "summary.md",
    ]
    sources = {
        "config": config_path,
        "attestation": _resolve(str(config["pre_registration_attestation"])),
        "runner": Path(__file__).resolve(),
        "paired_kernel": _resolve(str(config["attested_files"]["paired_kernel"])),
        "parent_d1_result": _resolve(str(config["parent_d1_result"])),
        "parent_d1_levels": _resolve(str(config["parent_d1_levels"])),
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
    if not isinstance(config, dict):
        raise ValueError("N5-D2 config is invalid")
    output = args.output or _resolve(str(config["formal_output"]))
    result = run(args.config.resolve(), output.resolve())
    print(
        json.dumps(
            {
                "machine_decision": result["machine_decision"],
                "maximum_final_adjacent_relative_l2": result[
                    "maximum_final_adjacent_relative_l2"
                ],
                "maximum_adjacent_contraction_ratio": result[
                    "maximum_adjacent_contraction_ratio"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
