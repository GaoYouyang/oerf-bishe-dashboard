#!/usr/bin/env python3
"""Characterize the official PSU NIRT geometry primitives without TensorFlow."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import warnings
from pathlib import Path
from types import FunctionType
from typing import Any

import numpy as np


FUNCTION_NAMES = ("rayBoxIntersection", "rayConeIntersection")


def load_numpy_geometry_functions(source_path: Path) -> dict[str, FunctionType]:
    """Compile only the two NumPy geometry functions from the author module."""

    source = source_path.read_text(encoding="utf-8", errors="strict")
    tree = ast.parse(source, filename=source_path.name)
    selected = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in FUNCTION_NAMES
    }
    missing = sorted(set(FUNCTION_NAMES) - set(selected))
    if missing:
        raise ValueError(f"missing geometry functions: {', '.join(missing)}")
    if any(isinstance(selected[name], ast.AsyncFunctionDef) for name in FUNCTION_NAMES):
        raise ValueError("geometry functions must be synchronous")

    module = ast.Module(
        body=[ast.Import(names=[ast.alias(name="numpy", asname="np")])]
        + [selected[name] for name in FUNCTION_NAMES],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {}
    exec(compile(module, source_path.name, "exec"), namespace)
    return {name: namespace[name] for name in FUNCTION_NAMES}


def _number(value: Any) -> float | str:
    number = float(value)
    if np.isfinite(number):
        return number
    if np.isnan(number):
        return "nan"
    return "inf" if number > 0 else "-inf"


def _values(array: Any) -> list[float | str]:
    return [_number(value) for value in np.asarray(array).reshape(-1)]


def _check(name: str, passed: bool, meaning: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "meaning": meaning}


def run_geometry_fixture(source_path: Path) -> dict[str, Any]:
    functions = load_numpy_geometry_functions(source_path)
    box = functions["rayBoxIntersection"]
    cone = functions["rayConeIntersection"]

    lower = np.array([[-1.0], [-1.0], [-1.0]])
    upper = np.array([[1.0], [1.0], [1.0]])
    box_origins = np.array(
        [
            [-2.0, 2.0, -2.0, 0.0, 2.0],
            [0.0, 0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    box_directions = np.array(
        [
            [1.0, -1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    with warnings.catch_warnings(record=True) as box_warnings:
        warnings.simplefilter("always")
        box_enter, box_exit, box_length = box(
            box_origins, box_directions, lower, upper
        )

    vertex = np.zeros((3, 1))
    axis = np.array([[1.0], [0.0], [0.0]])
    angle = np.pi / 4.0
    cone_origins = np.array(
        [
            [2.0, 2.0, 2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0, 0.0, 3.0],
            [0.0, 2.0, 3.0, 0.0, 0.0],
        ]
    )
    cone_directions = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, -1.0, -1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    with warnings.catch_warnings(record=True) as cone_warnings:
        warnings.simplefilter("always")
        cone_first, cone_second, cone_length = cone(
            cone_origins, cone_directions, vertex, axis, angle
        )

    checks = [
        _check(
            "box_forward_hit",
            np.allclose(
                [box_enter[0], box_exit[0], box_length[0]], [1.0, 3.0, 2.0]
            ),
            "a forward ray crosses the unit box with the analytic segment length",
        ),
        _check(
            "box_reverse_hit",
            np.allclose(
                [box_enter[1], box_exit[1], box_length[1]], [1.0, 3.0, 2.0]
            ),
            "reversing origin and direction preserves the forward segment",
        ),
        _check(
            "box_parallel_miss_zeroed",
            np.isclose(box_length[2], 0.0),
            "a parallel ray outside the y slab is rejected despite divide-by-zero warnings",
        ),
        _check(
            "box_inside_line_segment",
            np.allclose(
                [box_enter[3], box_exit[3], box_length[3]], [-1.0, 1.0, 2.0]
            ),
            "an origin inside the box retains the segment behind the ray origin",
        ),
        _check(
            "box_behind_camera_not_rejected",
            np.allclose(
                [box_enter[4], box_exit[4], box_length[4]], [-3.0, -1.0, 2.0]
            ),
            "the author primitive computes a line-box intersection and does not enforce t >= 0",
        ),
        _check(
            "cone_two_positive_roots",
            np.allclose(
                sorted([cone_first[0], cone_second[0]]), [1.0, 5.0]
            )
            and np.isclose(cone_length[0], 4.0),
            "a transverse secant through the x=2 cone section has analytic roots 1 and 5",
        ),
        _check(
            "cone_root_labels_not_ordered",
            cone_first[0] > cone_second[0],
            "the returned t_min/t_max names do not guarantee numerical ordering",
        ),
        _check(
            "cone_tangent_collapses_to_zero",
            np.isclose(cone_length[1], 0.0),
            "a tangent and a miss share the downstream L == 0 signal",
        ),
        _check(
            "cone_miss_zeroed",
            np.isclose(cone_length[2], 0.0),
            "negative discriminant produces zero reported length",
        ),
        _check(
            "cone_single_positive_root_discarded",
            np.isclose(cone_length[3], 0.0)
            and np.count_nonzero(
                np.isfinite([cone_first[3], cone_second[3]])
            )
            == 1,
            "an origin inside the cone has one forward exit but the pairwise length is discarded",
        ),
        _check(
            "cone_roots_behind_zeroed",
            np.isclose(cone_length[4], 0.0),
            "two intersections behind the origin are filtered from the ray",
        ),
    ]
    passed = all(item["passed"] for item in checks)
    source_bytes = source_path.read_bytes()
    return {
        "schema_version": "official-psu-nirt-geometry-fixture-1.0",
        "status": (
            "GEOMETRY_PRIMITIVE_CONTRACT_CHARACTERIZED_WITH_LIMITATIONS"
            if passed
            else "GEOMETRY_FIXTURE_UNEXPECTED_RESULT"
        ),
        "evidence_scope": "EXACT_AUTHOR_NUMPY_FUNCTIONS_TINY_ANALYTIC_FIXTURE_NO_TENSORFLOW_NO_NIRT_RECONSTRUCTION",
        "source": {
            "filename": source_path.name,
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "functions": list(FUNCTION_NAMES),
            "author_source_modified": False,
        },
        "fixture": {
            "box_bounds": {"minimum": [-1.0] * 3, "maximum": [1.0] * 3},
            "cone": {
                "vertex": [0.0, 0.0, 0.0],
                "axis": [1.0, 0.0, 0.0],
                "half_angle_degrees": 45.0,
            },
            "box_ray_count": int(box_origins.shape[1]),
            "cone_ray_count": int(cone_origins.shape[1]),
        },
        "checks": checks,
        "observed": {
            "box": {
                "t_enter": _values(box_enter),
                "t_exit": _values(box_exit),
                "length": _values(box_length),
                "runtime_warning_count": len(box_warnings),
            },
            "cone": {
                "first_root": _values(cone_first),
                "second_root": _values(cone_second),
                "length": _values(cone_length),
                "runtime_warning_count": len(cone_warnings),
            },
        },
        "known_limitations": [
            "box primitive accepts line segments entirely behind the ray origin",
            "parallel box components rely on NumPy inf/nan behavior",
            "cone primitive represents an unbounded double cone without a forward-nappe test",
            "cone axis normalization is a caller-side requirement",
            "one-positive-root cone cases are converted to zero length",
            "tangent, miss, and numerical degeneracy all share the L == 0 fallback signal",
            "this fixture does not exercise TensorFlow sampling, LoS autodiff, masks, training, or reconstruction",
        ],
        "decision": {
            "geometry_primitive_execution": "PASS_WITH_CHARACTERIZED_LIMITATIONS"
            if passed
            else "NO_GO",
            "full_nirt_reconstruction": "NOT_UNLOCKED",
            "next_gate": "STREAMED_ONE_VIEW_SETUP_ASSEMBLY_AND_MASK_INDEX_BASE_AUDIT",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meas-source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_geometry_fixture(args.meas_source)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["status"].startswith("GEOMETRY_PRIMITIVE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
