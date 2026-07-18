#!/usr/bin/env python3
"""Independently validate the formal N5-D2 H4096/H8192 result bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import demo_t16_operator.run_n2_pvgr_n5_d2_extreme_refinement as d2  # noqa: E402


DEFAULT_CONFIG = (
    ROOT
    / "demo_t16_operator/configs/n2_pvgr_n5_d2_extreme_refinement_preregistered_v1.json"
)
RESULT_FILES = (
    "cell_summary.csv",
    "config_snapshot.json",
    "levels.json",
    "n2_pvgr_n5_d2_extreme_refinement.png",
    "result.json",
    "summary.md",
)
SOURCE_KEYS = (
    "config",
    "attestation",
    "runner",
    "paired_kernel",
    "parent_d1_result",
    "parent_d1_levels",
)


def _read_json(path: Path) -> dict[str, Any] | list[Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same(left: float, right: float, *, label: str) -> None:
    if not math.isclose(float(left), float(right), rel_tol=2e-13, abs_tol=2e-15):
        raise ValueError(f"N5-D2 numeric mismatch for {label}: {left} != {right}")


def _array(entry: dict[str, Any], *, label: str) -> np.ndarray:
    value = np.asarray(entry["values"], dtype=np.float64)
    if value.shape != (256, 2) or not np.all(np.isfinite(value)):
        raise ValueError(f"N5-D2 invalid array for {label}")
    if entry["sha256"] != d2.d1._array_sha256(value):
        raise ValueError(f"N5-D2 array hash mismatch for {label}")
    _same(entry["l2_norm"], np.linalg.norm(value), label=f"{label} L2")
    return value


def _verify_manifest(
    manifest: dict[str, Any],
    output: Path,
    config_path: Path,
    config: dict[str, Any],
) -> None:
    if manifest.get("schema") != "n2-pvgr-n5-d2-extreme-refinement-manifest-1.0":
        raise ValueError("N5-D2 manifest schema drifted")
    expected_keys = set(SOURCE_KEYS) | set(RESULT_FILES)
    if set(manifest.get("files", {})) != expected_keys:
        raise ValueError("N5-D2 manifest file set drifted")
    expected_sources = {
        "config": config_path.resolve().relative_to(ROOT).as_posix(),
        "attestation": str(config["pre_registration_attestation"]),
        "runner": Path(d2.__file__).resolve().relative_to(ROOT).as_posix(),
        "paired_kernel": str(config["attested_files"]["paired_kernel"]),
        "parent_d1_result": str(config["parent_d1_result"]),
        "parent_d1_levels": str(config["parent_d1_levels"]),
    }
    for key, entry in manifest["files"].items():
        path = ROOT / str(entry["path"])
        if not path.is_file() or _sha256(path) != entry["sha256"]:
            raise ValueError(f"N5-D2 manifest hash mismatch: {key}")
        if key in RESULT_FILES and path.resolve() != (output / key).resolve():
            raise ValueError(f"N5-D2 manifest output path drifted: {key}")
        if key in SOURCE_KEYS and entry["path"] != expected_sources[key]:
            raise ValueError(f"N5-D2 manifest source path drifted: {key}")


def _verify_figure(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float64)
        width, height = image.size
    sampled = rgb[::8, ::8]
    if width < 1200 or height < 400 or float(np.std(rgb)) < 3.0:
        raise ValueError("N5-D2 figure is too small or blank")
    if np.unique(sampled.reshape(-1, 3), axis=0).shape[0] < 64:
        raise ValueError("N5-D2 figure has insufficient visual content")
    return {"width": int(width), "height": int(height), "pixel_std": float(np.std(rgb))}


def _recompute_cell(
    cell: dict[str, Any],
    parent_level: dict[str, Any],
    levels: dict[int, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    reference = str(config["reference_method"])
    arrays = {
        2048: {
            method: _array(
                parent_level["methods"][method],
                label=f"{cell['cell_id']} H2048 {method}",
            )
            for method in d2.METHODS
        }
    }
    for step, level in levels.items():
        arrays[step] = {
            method: _array(
                level["methods"][method],
                label=f"{cell['cell_id']} H{step} {method}",
            )
            for method in d2.METHODS
        }
    first = d2.d1._relative_l2(arrays[2048][reference], arrays[4096][reference])
    final = d2.d1._relative_l2(arrays[4096][reference], arrays[8192][reference])
    contraction = final / max(first, 1e-30)
    order = -math.log2(contraction) if contraction > 0.0 else None
    final_absolute = float(
        np.linalg.norm(arrays[4096][reference] - arrays[8192][reference])
    )
    raw_paired_absolute = float(
        np.linalg.norm(
            arrays[8192]["raw_separate_subtraction"] - arrays[8192][reference]
        )
    )
    raw_fraction = raw_paired_absolute / max(final_absolute, 1e-30)
    diagnostics = {
        "minimum_finite_ray_fraction": min(
            float(level["diagnostics"]["finite_ray_fraction"])
            for level in levels.values()
        ),
        "minimum_domain_margin": min(
            float(level["diagnostics"]["minimum_domain_margin"])
            for level in levels.values()
        ),
        "minimum_stencil_margin": min(
            float(level["diagnostics"]["minimum_stencil_margin"])
            for level in levels.values()
        ),
        "maximum_direction_norm_error": max(
            float(level["diagnostics"]["maximum_direction_norm_error"])
            for level in levels.values()
        ),
    }
    gates = config["gates"]
    gate_results = {
        "final_adjacent_relative_l2_gate_met": final
        <= float(gates["maximum_h4096_to_h8192_reference_relative_l2"]),
        "adjacent_contraction_gate_met": contraction
        <= float(gates["maximum_adjacent_contraction_ratio"]),
        "raw_paired_fraction_gate_met": raw_fraction
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
    geometric = final * contraction / (1.0 - contraction) if 0.0 <= contraction < 1.0 else None
    return {
        "cell_id": cell["cell_id"],
        "pair_id": cell["pair_id"],
        "role": cell["role"],
        "d2048_to_d4096_reference_relative_l2": first,
        "d4096_to_d8192_reference_relative_l2": final,
        "adjacent_contraction_ratio": contraction,
        "observed_order": order,
        "fixed_second_order_richardson_correction_relative_l2": final / 3.0,
        "observed_geometric_tail_indicator_relative_l2": geometric,
        "final_refinement_absolute_l2": final_absolute,
        "h8192_raw_paired_absolute_l2": raw_paired_absolute,
        "h8192_raw_paired_fraction_of_final_refinement": raw_fraction,
        "diagnostics": diagnostics,
        "gates": gate_results,
        "all_gates_pass": all(gate_results.values()),
    }


def _compare_cell(stored: dict[str, Any], expected: dict[str, Any]) -> None:
    identity = ("cell_id", "pair_id", "role", "all_gates_pass")
    for key in identity:
        if stored[key] != expected[key]:
            raise ValueError(f"N5-D2 cell identity mismatch: {stored['cell_id']} {key}")
    numeric = (
        "d2048_to_d4096_reference_relative_l2",
        "d4096_to_d8192_reference_relative_l2",
        "adjacent_contraction_ratio",
        "fixed_second_order_richardson_correction_relative_l2",
        "final_refinement_absolute_l2",
        "h8192_raw_paired_absolute_l2",
        "h8192_raw_paired_fraction_of_final_refinement",
    )
    for key in numeric:
        _same(stored[key], expected[key], label=f"{stored['cell_id']} {key}")
    for key in ("observed_order", "observed_geometric_tail_indicator_relative_l2"):
        if expected[key] is None:
            if stored[key] is not None:
                raise ValueError(f"N5-D2 optional metric mismatch: {stored['cell_id']} {key}")
        else:
            _same(stored[key], expected[key], label=f"{stored['cell_id']} {key}")
    if stored["gates"] != expected["gates"]:
        raise ValueError(f"N5-D2 cell gate mismatch: {stored['cell_id']}")
    for key, value in expected["diagnostics"].items():
        _same(stored["diagnostics"][key], value, label=f"{stored['cell_id']} {key}")


def validate(config_path: Path, output: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = _read_json(config_path)
    if not isinstance(config, dict):
        raise ValueError("N5-D2 config is invalid")
    d2._validate_contract(config)
    _, parent_result, parent_levels, _, cells = d2._load_parent_contract(config)
    attestation = d2._validate_preregistration(config, config_path)
    expected_output = (ROOT / str(config["formal_output"])).resolve()
    if output.resolve() != expected_output:
        raise ValueError("N5-D2 validator output path drifted")
    if _read_json(output / "config_snapshot.json") != config:
        raise ValueError("N5-D2 config snapshot drifted")
    result = _read_json(output / "result.json")
    manifest = _read_json(output / "manifest.json")
    levels = _read_json(output / "levels.json")
    if not isinstance(result, dict) or not isinstance(manifest, dict):
        raise ValueError("N5-D2 result bundle is invalid")
    if not isinstance(levels, list):
        raise ValueError("N5-D2 levels payload is invalid")
    _verify_manifest(manifest, output, config_path, config)
    if result.get("schema") != "n2-pvgr-n5-d2-extreme-refinement-result-1.0":
        raise ValueError("N5-D2 result schema drifted")
    if result.get("candidate_id") != config["candidate_id"]:
        raise ValueError("N5-D2 candidate ID drifted")
    if result.get("protocol_commit") != attestation["protocol_commit"]:
        raise ValueError("N5-D2 protocol commit drifted")
    if result.get("parent_d1_protocol_commit") != parent_result["protocol_commit"]:
        raise ValueError("N5-D2 parent protocol commit drifted")
    if result.get("reference_method") != config["reference_method"]:
        raise ValueError("N5-D2 result reference method drifted")
    if result.get("monitor_methods") != config["monitor_methods"]:
        raise ValueError("N5-D2 result monitor methods drifted")
    if result.get("gates") != config["gates"]:
        raise ValueError("N5-D2 result gates drifted")

    selected_ids = [cell["cell_id"] for cell in cells]
    expected_keys = {
        (cell_id, step) for cell_id in selected_ids for step in config["new_step_counts"]
    }
    actual_keys = {(row["cell_id"], int(row["step_count"])) for row in levels}
    if len(levels) != 8 or actual_keys != expected_keys:
        raise ValueError("N5-D2 level population drifted")
    grouped: dict[str, dict[int, dict[str, Any]]] = {cell_id: {} for cell_id in selected_ids}
    for level in levels:
        cell_id = str(level["cell_id"])
        step = int(level["step_count"])
        grouped[cell_id][step] = level
        if set(level["methods"]) != set(d2.METHODS):
            raise ValueError(f"N5-D2 method payload drifted: {cell_id} H{step}")
        for method, entry in level["methods"].items():
            _array(entry, label=f"{cell_id} H{step} {method}")
        diagnostics = level["diagnostics"]
        if not all(math.isfinite(float(value)) for value in diagnostics.values()):
            raise ValueError(f"N5-D2 non-finite diagnostic: {cell_id} H{step}")
        expected_queries = 42 * 256 * step
        if int(level["cost"]["logical_point_queries"]) != expected_queries:
            raise ValueError(f"N5-D2 query accounting drifted: {cell_id} H{step}")
        if not math.isfinite(float(level["cost"]["wall_seconds"])) or float(
            level["cost"]["wall_seconds"]
        ) < 0.0:
            raise ValueError(f"N5-D2 wall time drifted: {cell_id} H{step}")

    parent_h2048 = d2._parent_h2048_levels(config, parent_levels)
    expected_cells = [
        _recompute_cell(cell, parent_h2048[cell["cell_id"]], grouped[cell["cell_id"]], config)
        for cell in cells
    ]
    stored_cells = result["cells"]
    if [row["cell_id"] for row in stored_cells] != selected_ids:
        raise ValueError("N5-D2 result cell order drifted")
    for stored, expected in zip(stored_cells, expected_cells, strict=True):
        _compare_cell(stored, expected)

    decision = d2._decision(expected_cells)
    if result["machine_decision"] != decision:
        raise ValueError("N5-D2 machine decision drifted")
    resolved = decision == "D2_SELECTED_TAIL_RESOLVED_AT_H8192"
    maxima = {
        "maximum_final_adjacent_relative_l2": max(
            row["d4096_to_d8192_reference_relative_l2"] for row in expected_cells
        ),
        "maximum_adjacent_contraction_ratio": max(
            row["adjacent_contraction_ratio"] for row in expected_cells
        ),
        "maximum_raw_paired_fraction_of_final_refinement": max(
            row["h8192_raw_paired_fraction_of_final_refinement"]
            for row in expected_cells
        ),
    }
    for key, value in maxima.items():
        _same(result[key], value, label=key)
    if int(result["selected_cell_count"]) != 4 or int(
        result["new_level_evaluation_count"]
    ) != 8:
        raise ValueError("N5-D2 count ledger drifted")
    expected_queries = sum(int(row["cost"]["logical_point_queries"]) for row in levels)
    if int(result["logical_point_queries"]) != expected_queries:
        raise ValueError("N5-D2 total query ledger drifted")
    _same(
        result["wall_seconds"],
        sum(float(row["cost"]["wall_seconds"]) for row in levels),
        label="wall seconds",
    )
    expected_authorizations = {
        **config["claim_authorizations"],
        "adaptive_reference_reconciliation_preregistration_allowed": resolved,
        "selected_four_cell_tail_resolved": resolved,
    }
    if result["authorizations"] != expected_authorizations:
        raise ValueError("N5-D2 authorization ledger drifted")
    if result["limitations"] != config["known_limitations"]:
        raise ValueError("N5-D2 limitations drifted")
    if result["selection_is_post_n4_and_post_d1_not_an_independent_test"] is not True:
        raise ValueError("N5-D2 post-selection disclosure is missing")
    if result["parent_h2048_arrays_reused_without_recomputation"] is not True:
        raise ValueError("N5-D2 parent-array reuse disclosure is missing")

    csv_rows = _read_csv(output / "cell_summary.csv")
    if len(csv_rows) != 4 or [row["cell_id"] for row in csv_rows] != selected_ids:
        raise ValueError("N5-D2 CSV row population drifted")
    for csv_row, expected in zip(csv_rows, expected_cells, strict=True):
        for key in (
            "d2048_to_d4096_reference_relative_l2",
            "d4096_to_d8192_reference_relative_l2",
            "adjacent_contraction_ratio",
            "fixed_second_order_richardson_correction_relative_l2",
            "h8192_raw_paired_fraction_of_final_refinement",
        ):
            _same(float(csv_row[key]), expected[key], label=f"CSV {csv_row['cell_id']} {key}")
        if csv_row["all_gates_pass"] != str(expected["all_gates_pass"]):
            raise ValueError(f"N5-D2 CSV gate drifted: {csv_row['cell_id']}")

    summary = (output / "summary.md").read_text(encoding="utf-8")
    if decision not in summary or "does not authorize a fresh reference" not in summary:
        raise ValueError("N5-D2 summary claim boundary drifted")
    figure = _verify_figure(output / "n2_pvgr_n5_d2_extreme_refinement.png")
    report = {
        "schema": "n2-pvgr-n5-d2-extreme-refinement-validation-1.0",
        "valid": True,
        "machine_decision": decision,
        "selected_cell_count": 4,
        "new_level_evaluation_count": 8,
        **maxima,
        "array_hashes_and_metrics_recomputed": True,
        "parent_d1_manifest_reverified": True,
        "query_accounting_verified": True,
        "figure_verified": figure,
        "claim_boundary_verified": True,
    }
    (output / "validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


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
    output = (args.output or ROOT / str(config["formal_output"])).resolve()
    print(json.dumps(validate(args.config.resolve(), output), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
