#!/usr/bin/env python3
"""Independently validate the formal N5-D1 paired-residual result bundle."""

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

import demo_t16_operator.run_n2_pvgr_n5_d1_paired_residual as d1  # noqa: E402


DEFAULT_CONFIG = (
    ROOT
    / "demo_t16_operator/configs/n2_pvgr_n5_d1_paired_residual_preregistered_v1.json"
)
METHODS = ("raw_separate_subtraction", *d1.NONRAW_METHODS)
EQUIVALENCE_ARRAYS = (
    "paired_curved_naive",
    "paired_straight_naive",
    "frozen_curved",
    "frozen_straight",
    "frozen_residual",
)
RESULT_FILES = (
    "cell_summary.csv",
    "config_snapshot.json",
    "levels.json",
    "n2_pvgr_n5_d1_paired_residual.png",
    "result.json",
    "summary.md",
    "toy_checks.json",
)
SOURCE_KEYS = (
    "config",
    "attestation",
    "runner",
    "paired_kernel",
    "parent_n4_result",
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
        raise ValueError(f"N5-D1 numeric mismatch for {label}: {left} != {right}")


def _array(entry: dict[str, Any], *, label: str) -> np.ndarray:
    value = np.asarray(entry["values"], dtype=np.float64)
    if value.shape != (256, 2) or not np.all(np.isfinite(value)):
        raise ValueError(f"N5-D1 invalid array for {label}")
    if entry["sha256"] != d1._array_sha256(value):
        raise ValueError(f"N5-D1 array hash mismatch for {label}")
    _same(entry["l2_norm"], np.linalg.norm(value), label=f"{label} L2")
    return value


def _verify_manifest(
    manifest: dict[str, Any],
    output_dir: Path,
    config_path: Path,
    config: dict[str, Any],
) -> None:
    if manifest.get("schema") != "n2-pvgr-n5-d1-paired-residual-manifest-1.0":
        raise ValueError("N5-D1 manifest schema drifted")
    expected = set(SOURCE_KEYS) | set(RESULT_FILES)
    if set(manifest.get("files", {})) != expected:
        raise ValueError("N5-D1 manifest file set drifted")
    expected_sources = {
        "config": config_path.resolve().relative_to(ROOT).as_posix(),
        "attestation": str(config["pre_registration_attestation"]),
        "runner": Path(d1.__file__).resolve().relative_to(ROOT).as_posix(),
        "paired_kernel": str(config["attested_files"]["paired_kernel"]),
        "parent_n4_result": str(config["parent_n4_result"]),
    }
    for key, entry in manifest["files"].items():
        path = ROOT / str(entry["path"])
        if not path.is_file() or _sha256(path) != entry["sha256"]:
            raise ValueError(f"N5-D1 manifest hash mismatch: {key}")
        if key in RESULT_FILES and path.resolve() != (output_dir / key).resolve():
            raise ValueError(f"N5-D1 manifest output path drifted: {key}")
        if key in SOURCE_KEYS and entry["path"] != expected_sources[key]:
            raise ValueError(f"N5-D1 manifest source path drifted: {key}")


def _verify_figure(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float64)
        width, height = image.size
    if width < 1200 or height < 400:
        raise ValueError("N5-D1 figure dimensions are unexpectedly small")
    sampled = rgb[::8, ::8]
    if float(np.std(rgb)) < 3.0 or np.unique(sampled.reshape(-1, 3), axis=0).shape[0] < 64:
        raise ValueError("N5-D1 figure appears blank")
    return {"width": int(width), "height": int(height), "pixel_std": float(np.std(rgb))}


def _recompute_route(level: dict[str, Any], arrays: dict[str, np.ndarray]) -> float:
    route = {
        "curved_relative_l2": d1._relative_l2(
            arrays["paired_curved_naive"], arrays["frozen_curved"]
        ),
        "straight_relative_l2": d1._relative_l2(
            arrays["paired_straight_naive"], arrays["frozen_straight"]
        ),
        "raw_residual_relative_l2": d1._relative_l2(
            arrays["raw_separate_subtraction"], arrays["frozen_residual"]
        ),
    }
    for key, value in route.items():
        _same(level["route_equivalence"][key], value, label=f"route {key}")
    return max(route.values())


def _recompute_cell(
    cell_id: str,
    levels: dict[int, dict[str, Any]],
    arrays: dict[int, dict[str, np.ndarray]],
    parent_result: dict[str, Any],
    config: dict[str, Any],
    route_maximum: float,
) -> dict[str, Any]:
    adjacent = {
        method: d1._relative_l2(arrays[1024][method], arrays[2048][method])
        for method in METHODS
    }
    raw_difference = arrays[1024]["raw_separate_subtraction"] - arrays[2048][
        "raw_separate_subtraction"
    ]
    raw_difference_l2 = float(np.linalg.norm(raw_difference))
    if raw_difference_l2 <= 0.0:
        raise ValueError(f"N5-D1 unscoreable raw refinement: {cell_id}")
    accumulation: dict[str, dict[str, float]] = {}
    for method in d1.NONRAW_METHODS:
        difference_l2 = float(
            np.linalg.norm(
                arrays[2048][method] - arrays[2048]["raw_separate_subtraction"]
            )
        )
        accumulation[method] = {
            "absolute_l2": difference_l2,
            "relative_to_raw_residual_l2": difference_l2
            / max(float(np.linalg.norm(arrays[2048]["raw_separate_subtraction"])), 1e-30),
            "fraction_of_raw_h1024_h2048_absolute_difference": difference_l2
            / raw_difference_l2,
        }
    parent_cell = next(row for row in parent_result["cells"] if row["cell_id"] == cell_id)
    parent_metric = float(parent_cell["d1024_to_d2048"]["matched_residual_relative_l2"])
    return {
        "cell_id": cell_id,
        "pair_id": levels[1024]["pair_id"],
        "role": levels[1024]["role"],
        "parent_n4_final_authorized": bool(parent_cell["final_cellwise_reference_authorized"]),
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
        "maximum_route_equivalence_relative_l2": route_maximum,
    }


def _compare_cell(stored: dict[str, Any], expected: dict[str, Any]) -> None:
    scalar_keys = (
        "parent_n4_raw_adjacent_relative_l2",
        "d1_raw_adjacent_relative_l2",
        "parent_n4_metric_absolute_difference",
        "raw_h1024_h2048_absolute_difference_l2",
        "maximum_nonraw_accumulation_fraction_of_refinement",
        "maximum_route_equivalence_relative_l2",
    )
    identity_keys = ("cell_id", "pair_id", "role", "parent_n4_final_authorized")
    for key in identity_keys:
        if stored[key] != expected[key]:
            raise ValueError(f"N5-D1 cell identity mismatch: {stored['cell_id']} {key}")
    for key in scalar_keys:
        _same(stored[key], expected[key], label=f"{stored['cell_id']} {key}")
    for method in METHODS:
        _same(
            stored["adjacent_relative_l2_by_method"][method],
            expected["adjacent_relative_l2_by_method"][method],
            label=f"{stored['cell_id']} adjacent {method}",
        )
    for method in d1.NONRAW_METHODS:
        for key, value in expected["accumulation_at_h2048"][method].items():
            _same(
                stored["accumulation_at_h2048"][method][key],
                value,
                label=f"{stored['cell_id']} accumulation {method} {key}",
            )


def validate(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = _read_json(config_path)
    assert isinstance(config, dict)
    d1._validate_contract(config)
    _, scientific, parent_n3, _ = d1._load_parent_contract(config)
    attestation = d1._validate_preregistration(config, config_path)
    selected = d1._selected_cells(config, scientific, parent_n3)
    selected_ids = [row["cell_id"] for row in selected]
    expected_output = (ROOT / str(config["formal_output"])).resolve()
    if output_dir.resolve() != expected_output:
        raise ValueError("N5-D1 validator output path drifted")
    if _read_json(output_dir / "config_snapshot.json") != config:
        raise ValueError("N5-D1 config snapshot drifted")

    result = _read_json(output_dir / "result.json")
    manifest = _read_json(output_dir / "manifest.json")
    levels = _read_json(output_dir / "levels.json")
    toy_stored = _read_json(output_dir / "toy_checks.json")
    assert isinstance(result, dict) and isinstance(manifest, dict)
    assert isinstance(levels, list) and isinstance(toy_stored, dict)
    _verify_manifest(manifest, output_dir, config_path, config)
    if result.get("schema") != "n2-pvgr-n5-d1-paired-residual-result-1.0":
        raise ValueError("N5-D1 result schema drifted")
    if result.get("candidate_id") != config["candidate_id"]:
        raise ValueError("N5-D1 candidate ID drifted")
    if result.get("protocol_commit") != attestation["protocol_commit"]:
        raise ValueError("N5-D1 protocol commit drifted")
    if result.get("accumulation_methods") != list(METHODS):
        raise ValueError("N5-D1 method list drifted")
    if result.get("gates") != config["gates"]:
        raise ValueError("N5-D1 gate snapshot drifted")

    expected_level_keys = {
        (cell_id, step) for cell_id in selected_ids for step in config["step_counts"]
    }
    actual_level_keys = {(row["cell_id"], int(row["step_count"])) for row in levels}
    if len(levels) != 8 or actual_level_keys != expected_level_keys:
        raise ValueError("N5-D1 level population drifted")

    grouped: dict[str, dict[int, dict[str, Any]]] = {cell_id: {} for cell_id in selected_ids}
    method_arrays: dict[str, dict[int, dict[str, np.ndarray]]] = {
        cell_id: {} for cell_id in selected_ids
    }
    route_by_cell: dict[str, float] = {cell_id: 0.0 for cell_id in selected_ids}
    for level in levels:
        cell_id = str(level["cell_id"])
        step = int(level["step_count"])
        grouped[cell_id][step] = level
        if set(level["methods"]) != set(METHODS):
            raise ValueError(f"N5-D1 method payload drifted: {cell_id} H{step}")
        if set(level["equivalence_arrays"]) != set(EQUIVALENCE_ARRAYS):
            raise ValueError(f"N5-D1 equivalence payload drifted: {cell_id} H{step}")
        arrays = {
            name: _array(entry, label=f"{cell_id} H{step} {name}")
            for name, entry in level["methods"].items()
        }
        arrays.update(
            {
                name: _array(entry, label=f"{cell_id} H{step} {name}")
                for name, entry in level["equivalence_arrays"].items()
            }
        )
        method_arrays[cell_id][step] = arrays
        route_by_cell[cell_id] = max(route_by_cell[cell_id], _recompute_route(level, arrays))
        expected_paired = 42 * 256 * step
        expected_verification = 14 * 256 * step
        expected_total = 56 * 256 * step
        cost = level["cost"]
        if (
            int(cost["paired_logical_point_queries"]) != expected_paired
            or int(cost["verification_logical_point_queries"]) != expected_verification
            or int(cost["total_audit_logical_point_queries"]) != expected_total
        ):
            raise ValueError(f"N5-D1 query accounting drifted: {cell_id} H{step}")
        for key in ("paired_wall_seconds", "frozen_route_verification_wall_seconds"):
            if not math.isfinite(float(cost[key])) or float(cost[key]) < 0.0:
                raise ValueError(f"N5-D1 invalid wall time: {cell_id} H{step}")
        if not all(math.isfinite(float(value)) for value in level["diagnostics"].values()):
            raise ValueError(f"N5-D1 non-finite diagnostic: {cell_id} H{step}")

    parent_result = _read_json(ROOT / str(config["parent_n4_result"]))
    assert isinstance(parent_result, dict)
    expected_cells = [
        _recompute_cell(
            cell_id,
            grouped[cell_id],
            method_arrays[cell_id],
            parent_result,
            config,
            route_by_cell[cell_id],
        )
        for cell_id in selected_ids
    ]
    stored_cells = result["cells"]
    if [row["cell_id"] for row in stored_cells] != selected_ids:
        raise ValueError("N5-D1 result cell order drifted")
    for stored, expected in zip(stored_cells, expected_cells, strict=True):
        _compare_cell(stored, expected)

    toy_fresh = d1._toy_checks(config)
    for key, value in toy_fresh.items():
        if isinstance(value, bool):
            if toy_stored[key] is not value:
                raise ValueError(f"N5-D1 toy gate mismatch: {key}")
        else:
            _same(toy_stored[key], value, label=f"toy {key}")
    if result["toy_checks"] != toy_stored:
        raise ValueError("N5-D1 embedded toy checks drifted")

    maximum_route = max(row["maximum_route_equivalence_relative_l2"] for row in expected_cells)
    maximum_replay = max(row["parent_n4_metric_absolute_difference"] for row in expected_cells)
    gates = config["gates"]
    contract_pass = (
        all(bool(toy_stored[key]) for key in ("constant_gate_met", "weak_gate_met", "rotation_gate_met"))
        and maximum_route <= float(gates["maximum_frozen_route_equivalence_relative_l2"])
        and maximum_replay <= float(gates["maximum_parent_n4_adjacent_metric_absolute_difference"])
    )
    failed_fractions = [
        float(row["maximum_nonraw_accumulation_fraction_of_refinement"])
        for row in expected_cells
        if not row["parent_n4_final_authorized"]
    ]
    decision = d1._decision(
        contract_gates_pass=contract_pass,
        failed_cell_maximum_fractions=failed_fractions,
        gates=gates,
    )
    if bool(result["contract_gates_pass"]) is not contract_pass:
        raise ValueError("N5-D1 aggregate contract gate drifted")
    if result["machine_decision"] != decision:
        raise ValueError("N5-D1 machine decision drifted")
    _same(result["maximum_route_equivalence_relative_l2"], maximum_route, label="maximum route")
    _same(result["maximum_parent_n4_metric_absolute_difference"], maximum_replay, label="maximum replay")
    if len(result["failed_cell_maximum_accumulation_fractions"]) != len(failed_fractions):
        raise ValueError("N5-D1 failed-cell fraction count drifted")
    for stored, expected in zip(
        result["failed_cell_maximum_accumulation_fractions"], failed_fractions, strict=True
    ):
        _same(stored, expected, label="failed-cell maximum fraction")

    expected_counts = {"selected_cell_count": 4, "level_evaluation_count": 8}
    for key, value in expected_counts.items():
        if int(result[key]) != value:
            raise ValueError(f"N5-D1 count drifted: {key}")
    for result_key, cost_key in (
        ("paired_logical_point_queries", "paired_logical_point_queries"),
        ("verification_logical_point_queries", "verification_logical_point_queries"),
        ("total_audit_logical_point_queries", "total_audit_logical_point_queries"),
    ):
        expected = sum(int(level["cost"][cost_key]) for level in levels)
        if int(result[result_key]) != expected:
            raise ValueError(f"N5-D1 total query count drifted: {result_key}")
    for result_key, cost_key in (
        ("paired_wall_seconds", "paired_wall_seconds"),
        ("verification_wall_seconds", "frozen_route_verification_wall_seconds"),
    ):
        _same(
            result[result_key],
            sum(float(level["cost"][cost_key]) for level in levels),
            label=result_key,
        )

    expected_authorizations = {
        **config["claim_authorizations"],
        "d2_refinement_probe_preregistration_allowed": contract_pass,
        "replace_n4_metric_or_threshold": False,
    }
    if result["authorizations"] != expected_authorizations:
        raise ValueError("N5-D1 authorization ledger drifted")
    if result["limitations"] != config["known_limitations"]:
        raise ValueError("N5-D1 limitations drifted")
    if result["selection_is_post_n4_and_not_an_independent_test"] is not True:
        raise ValueError("N5-D1 post-selection disclosure is missing")

    csv_rows = _read_csv(output_dir / "cell_summary.csv")
    if len(csv_rows) != 4 or [row["cell_id"] for row in csv_rows] != selected_ids:
        raise ValueError("N5-D1 cell summary CSV drifted")
    for csv_row, expected in zip(csv_rows, expected_cells, strict=True):
        for key in (
            "parent_n4_raw_adjacent_relative_l2",
            "d1_raw_adjacent_relative_l2",
            "parent_n4_metric_absolute_difference",
            "raw_h1024_h2048_absolute_difference_l2",
            "maximum_nonraw_accumulation_fraction_of_refinement",
            "maximum_route_equivalence_relative_l2",
        ):
            _same(float(csv_row[key]), expected[key], label=f"CSV {csv_row['cell_id']} {key}")
        if csv_row["parent_n4_final_authorized"] != str(expected["parent_n4_final_authorized"]):
            raise ValueError(f"N5-D1 CSV authorization drifted: {csv_row['cell_id']}")

    summary = (output_dir / "summary.md").read_text(encoding="utf-8")
    if decision not in summary or "does not authorize a reference" not in summary:
        raise ValueError("N5-D1 summary claim boundary drifted")
    figure = _verify_figure(output_dir / "n2_pvgr_n5_d1_paired_residual.png")
    report = {
        "schema": "n2-pvgr-n5-d1-paired-residual-validation-1.0",
        "valid": True,
        "machine_decision": decision,
        "contract_gates_pass": contract_pass,
        "selected_cell_count": 4,
        "level_evaluation_count": 8,
        "maximum_route_equivalence_relative_l2": maximum_route,
        "maximum_parent_n4_metric_absolute_difference": maximum_replay,
        "failed_cell_maximum_accumulation_fractions": failed_fractions,
        "query_accounting_verified": True,
        "array_hashes_and_metrics_recomputed": True,
        "toy_contract_reexecuted": True,
        "figure_verified": figure,
        "claim_boundary_verified": True,
    }
    (output_dir / "validation_report.json").write_text(
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
    assert isinstance(config, dict)
    output = (args.output or ROOT / str(config["formal_output"])).resolve()
    print(json.dumps(validate(args.config.resolve(), output), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
