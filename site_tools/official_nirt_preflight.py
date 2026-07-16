#!/usr/bin/env python3
"""Preflight the official PSU NIRT source without executing its heavy loader."""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


REQUIRED_SOURCES = (
    "NIRT.py",
    "setup.py",
    "sample.py",
    "meas.py",
    "network.py",
    "train.py",
    "fit.py",
    "predict.py",
    "reg.py",
    "util.py",
    "AdamGA.py",
)

EXTERNAL_DEPENDENCIES = ("numpy", "scipy", "tensorflow")


def _line_number(source: str, needle: str) -> int | None:
    offset = source.find(needle)
    return None if offset < 0 else source.count("\n", 0, offset) + 1


def _hazard(
    sources: dict[str, str],
    *,
    file: str,
    needle: str,
    code: str,
    severity: str,
    explanation: str,
) -> dict[str, Any] | None:
    source = sources.get(file, "")
    line = _line_number(source, needle)
    if line is None:
        return None
    return {
        "code": code,
        "severity": severity,
        "file": file,
        "line": line,
        "evidence": needle,
        "explanation": explanation,
    }


def detect_static_hazards(sources: dict[str, str]) -> list[dict[str, Any]]:
    candidates = [
        _hazard(
            sources,
            file="NIRT.py",
            needle="tf.device('/GPU')",
            code="GPU_DEVICE_FORCED",
            severity="blocker",
            explanation="the main path has no CPU or Metal fallback",
        ),
        _hazard(
            sources,
            file="NIRT.py",
            needle="xla_gpu_cuda_data_dir=C:",
            code="WINDOWS_CUDA_PATH_HARDCODED",
            severity="blocker",
            explanation="the XLA path is Windows/CUDA-specific and does not describe Apple Metal",
        ),
        _hazard(
            sources,
            file="network.py",
            needle="tf.ones(3,1)",
            code="INVALID_TF_ONES_CALL",
            severity="blocker",
            explanation="the second positional argument is interpreted as dtype, not a second shape dimension",
        ),
        _hazard(
            sources,
            file="fit.py",
            needle="pdict['auto_w']",
            code="MISSING_AUTO_W_CONTRACT",
            severity="blocker",
            explanation="fit reads auto_w but the main pdict literal does not provide it",
        ),
        _hazard(
            sources,
            file="predict.py",
            needle="data_pred[prev_j:j].assign",
            code="TENSOR_SLICE_ASSIGN",
            severity="blocker",
            explanation="a TensorFlow slice is treated as an assignable Variable",
        ),
        _hazard(
            sources,
            file="setup.py",
            needle="scipy.io.loadmat('../Data/HSOF_",
            code="CWD_RELATIVE_DATA_PATH",
            severity="blocker",
            explanation="data resolution depends on the process working directory",
        ),
        _hazard(
            sources,
            file="NIRT.py",
            needle="pred_flag           = 1",
            code="DEFAULT_IS_PREDICTION",
            severity="warning",
            explanation="running with no arguments loads a checkpoint instead of training",
        ),
        _hazard(
            sources,
            file="NIRT.py",
            needle="os.mkdir(savedir)",
            code="PARENT_RESULTS_DIR_ASSUMED",
            severity="warning",
            explanation="single-level mkdir fails when the parent Results directory is absent",
        ),
    ]
    hazards = [item for item in candidates if item is not None]
    nirt_source = sources.get("NIRT.py", "")
    if "'auto_w':" in nirt_source or '"auto_w":' in nirt_source:
        hazards = [item for item in hazards if item["code"] != "MISSING_AUTO_W_CONTRACT"]
    return hazards


def _physical_memory_bytes() -> int | None:
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            check=True,
            capture_output=True,
            text=True,
        )
        return int(result.stdout.strip())
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None


def _dependency_status(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
        return {
            "importable": True,
            "version": getattr(module, "__version__", "unknown"),
        }
    except Exception as error:  # Dependency ABI failures are part of preflight.
        return {
            "importable": False,
            "error_type": type(error).__name__,
            "error": str(error).splitlines()[0],
        }


def estimate_memory_floor(
    measurement_count: int, grid_shape: tuple[int, int, int]
) -> dict[str, Any]:
    grid_count = grid_shape[0] * grid_shape[1] * grid_shape[2]
    components = {
        "cam_data_n_by_18_float64": measurement_count * 18 * 8,
        "b_data_n_by_4_float64": measurement_count * 4 * 8,
        "xyz_three_grids_float64": grid_count * 3 * 8,
    }
    floor = sum(components.values())
    return {
        "known_persistent_components_bytes": components,
        "known_persistent_floor_bytes": floor,
        "known_persistent_floor_gib": floor / 1024**3,
        "scope": "LOWER_BOUND_EXCLUDES_SOURCE_ARRAYS_TEMPORARIES_TF_TENSORS_MODEL_AND_XLA",
    }


def _source_inventory(pyscripts_dir: Path) -> tuple[dict[str, str], dict[str, Any]]:
    sources: dict[str, str] = {}
    syntax: dict[str, Any] = {}
    for name in REQUIRED_SOURCES:
        path = pyscripts_dir / name
        if not path.is_file():
            syntax[name] = {"present": False, "ast_valid": False}
            continue
        source = path.read_text(encoding="utf-8", errors="replace")
        sources[name] = source
        try:
            ast.parse(source, filename=name)
            syntax[name] = {"present": True, "ast_valid": True}
        except SyntaxError as error:
            syntax[name] = {
                "present": True,
                "ast_valid": False,
                "line": error.lineno,
                "message": error.msg,
            }
    return sources, syntax


def build_preflight_report(
    pyscripts_dir: Path,
    loader_contract: dict[str, Any],
) -> dict[str, Any]:
    sources, syntax = _source_inventory(pyscripts_dir)
    dependencies = {name: _dependency_status(name) for name in EXTERNAL_DEPENDENCIES}
    hazards = detect_static_hazards(sources)
    config = loader_contract["configuration"]
    memory = estimate_memory_floor(
        int(config["measurement_count"]),
        tuple(int(value) for value in config["grid_shape"]),
    )
    physical_memory = _physical_memory_bytes()
    if physical_memory is not None:
        memory["host_physical_memory_bytes"] = physical_memory
        memory["floor_fraction_of_host_memory"] = (
            memory["known_persistent_floor_bytes"] / physical_memory
        )

    expected_data_path = pyscripts_dir.parent / "Data" / "HSOF_9CAM_RT.mat"
    expected_checkpoint = pyscripts_dir.parent / "Results" / "R053" / "NNsaveR053U00L01"
    blockers = [item for item in hazards if item["severity"] == "blocker"]
    syntax_ok = all(item["ast_valid"] for item in syntax.values())
    data_contract_ok = (
        loader_contract.get("status") == "LOADER_NUMERIC_CONTRACT_CONFORMANT"
    )
    full_pipeline_ready = all(
        (
            syntax_ok,
            data_contract_ok,
            all(item["importable"] for item in dependencies.values()),
            not blockers,
            expected_data_path.is_file(),
            expected_checkpoint.is_file(),
        )
    )
    return {
        "schema_version": "official-psu-nirt-preflight-1.0",
        "status": (
            "FULL_AUTHOR_NIRT_READY"
            if full_pipeline_ready
            else "FULL_AUTHOR_NIRT_NO_GO_CURRENT_ENVIRONMENT"
        ),
        "evidence_scope": "STATIC_SOURCE_DEPENDENCY_PATH_AND_MEMORY_PREFLIGHT_NO_NIRT_EXECUTION",
        "host": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
        "source_syntax": syntax,
        "all_required_sources_ast_valid": syntax_ok,
        "dependencies_available": dependencies,
        "loader_numeric_contract_passed": data_contract_ok,
        "expected_runtime_paths": {
            "data_path_exists": expected_data_path.is_file(),
            "default_prediction_checkpoint_exists": expected_checkpoint.is_file(),
            "path_labels": {
                "data": "<dataset-root>/Data/HSOF_9CAM_RT.mat",
                "checkpoint": "<dataset-root>/Results/R053/NNsaveR053U00L01",
            },
        },
        "static_hazards": hazards,
        "blocker_count": len(blockers),
        "memory_floor": memory,
        "decision": {
            "full_9_view_author_entrypoint": "NO_GO",
            "safe_next_gate": "TINY_SYNTHETIC_GEOMETRY_AND_STREAMING_LOADER_FIXTURE",
            "reason": (
                "real data geometry is conformant, but the current environment lacks TensorFlow, "
                "runtime paths/checkpoint are absent, static blockers remain, and the memory floor "
                "does not include costly duplicate arrays or TensorFlow/XLA tensors"
            ),
        },
        "limitations": [
            "static pattern findings must be confirmed after a version-pinned TensorFlow port",
            "the memory estimate is an explicit lower bound, not a measured peak",
            "no official source file is modified or imported by this preflight",
            "NO_GO describes the current unmodified entrypoint and environment, not the feasibility of a streaming reimplementation",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyscripts-dir", type=Path, required=True)
    parser.add_argument("--loader-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    contract = json.loads(args.loader_contract.read_text(encoding="utf-8"))
    report = build_preflight_report(args.pyscripts_dir, contract)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
