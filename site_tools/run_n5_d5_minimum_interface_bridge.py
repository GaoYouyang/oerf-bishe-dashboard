#!/usr/bin/env python3
"""Run the N5-D5 minimum callable-interface replay protocol.

The runner launches a JSONL adapter without ``shell=True``.  It generates the
two tangents and one cotangent itself, performs every central finite-difference
forward call, and derives all metrics from raw adapter responses.  A successful
synthetic protocol run is only transport/semantic evidence; every scientific
claim remains closed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo_t16_operator.n5_d5_adapter_protocol import (
    DESCRIPTOR_SCHEMA,
    PATH_ROLES,
    REQUEST_SCHEMA,
    RESPONSE_SCHEMA,
    array_sha256,
    canonical_json,
    finite_vector,
    generated_vectors,
    normalize,
    sha256_bytes,
    sha256_file,
    state_digest,
)


DEFAULT_SCHEMA = ROOT / "data_templates/n5_d5_minimum_bost_interface.schema.json"
RESULT_SCHEMA = "n5-d5-minimum-interface-result-1.0"
PUBLIC_SUMMARY_SCHEMA = "n5-d5-minimum-interface-public-summary-1.0"
MANIFEST_SCHEMA = "n5-d5-minimum-interface-manifest-1.0"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _assert_relative_private_path(path: str) -> Path:
    target = _resolve(path)
    private_root = (ROOT / "private_library").resolve()
    if private_root not in target.parents:
        raise ValueError("lab vector paths must remain under private_library")
    return target


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=check,
        capture_output=True,
    )


def _assert_committed(paths: Iterable[Path]) -> str:
    unique = sorted({_relative(path) for path in paths})
    for relative in unique:
        tracked = _git("ls-files", "--error-unmatch", "--", relative, check=False)
        if tracked.returncode != 0:
            raise RuntimeError(f"formal source is not tracked: {relative}")
        dirty = _git("diff", "--quiet", "HEAD", "--", relative, check=False)
        if dirty.returncode != 0:
            raise RuntimeError(f"formal source has uncommitted changes: {relative}")
    return _git("rev-parse", "HEAD").stdout.decode("utf-8").strip()


def _validate_schema(config: dict[str, Any], schema_path: Path) -> None:
    schema = _read_json(schema_path)
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(config), key=lambda item: list(item.path))
    if errors:
        details = "; ".join(
            f"{'/'.join(str(part) for part in error.path) or '<root>'}: {error.message}"
            for error in errors[:8]
        )
        raise ValueError(f"N5-D5 schema validation failed: {details}")


def _validate_semantics(config: dict[str, Any]) -> None:
    roles = [str(path["role"]) for path in config["paths"]]
    if tuple(roles) != PATH_ROLES:
        raise ValueError("paths must appear in curved/straight/direct_residual order")
    for key in ("path_id", "callable_id", "semantic_digest_sha256"):
        values = [str(path[key]) for path in config["paths"]]
        if len(set(values)) != len(values):
            raise ValueError(f"path {key} values must be distinct")
    observation = config["observation"]
    expected_output = int(observation["ray_count"]) * int(
        observation["components_per_ray"]
    )
    if int(observation["output_dimension"]) != expected_output:
        raise ValueError("output dimension must equal rays times components")
    h_values = [float(value) for value in config["probe_plan"]["h_values"]]
    if h_values != sorted(set(h_values)):
        raise ValueError("h_values must be sorted and unique")
    if any(not math.isfinite(value) or value <= 0.0 for value in h_values):
        raise ValueError("h_values must be finite and positive")
    base = config["field"]["base_input"]
    if base["source"] == "seeded_gaussian":
        if base["seed"] is None or base["relative_path"] is not None:
            raise ValueError("seeded base input requires only a seed")
        if base["sha256"] is not None:
            raise ValueError("seeded base input hash is derived, not supplied")
    elif base["source"] == "npy_relative":
        if base["seed"] is not None or not base["relative_path"] or not base["sha256"]:
            raise ValueError("npy base input requires a relative path and SHA-256")
        _assert_relative_private_path(str(base["relative_path"]))
    else:  # pragma: no cover - guarded by JSON Schema
        raise ValueError("unsupported base input source")
    identity = config["identity"]
    if config["record_kind"] == "SYNTHETIC_REFERENCE_FIXTURE":
        if identity["real_lab_data_present"] is not False:
            raise ValueError("synthetic fixture cannot contain real lab data")
        if config["privacy"]["raw_trace_publication_permitted"] is not True:
            raise ValueError("synthetic fixture trace should remain inspectable")
        if config["adapter"]["source_review_status"] != "committed_source_reviewed":
            raise ValueError("synthetic fixture requires committed reviewed source")
    else:
        if identity["real_lab_data_present"] is not True:
            raise ValueError("lab record must explicitly acknowledge real lab data")
        if config["privacy"]["raw_trace_publication_permitted"] is not False:
            raise ValueError("lab raw traces must not be publicly authorized")
    if identity["anonymized"] is not True or identity["contains_private_identifiers"]:
        raise ValueError("the D5 bridge accepts only anonymized identifiers")
    if any(bool(value) for value in config["claim_authorizations"].values()):
        raise ValueError("N5-D5 cannot pre-authorize scientific claims")


def validate_config(config: dict[str, Any], schema_path: Path = DEFAULT_SCHEMA) -> None:
    _validate_schema(config, schema_path)
    _validate_semantics(config)


def _load_vector_bank(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    field = config["field"]
    observation = config["observation"]
    plan = config["probe_plan"]
    base_spec = field["base_input"]
    input_dimension = int(field["input_dimension"])
    output_dimension = int(observation["output_dimension"])
    generated_base, tangents, cotangents = generated_vectors(
        input_dimension=input_dimension,
        output_dimension=output_dimension,
        base_seed=int(base_spec["seed"] or 0),
        tangent_seed=int(plan["tangent_seed"]),
        cotangent_seed=int(plan["cotangent_seed"]),
    )
    if base_spec["source"] == "seeded_gaussian":
        base = generated_base
    else:
        path = _assert_relative_private_path(str(base_spec["relative_path"]))
        if not path.is_file():
            raise FileNotFoundError(f"private base input is missing: {path}")
        if sha256_file(path) != base_spec["sha256"]:
            raise ValueError("private base input SHA-256 mismatch")
        base = finite_vector(
            "base_input",
            np.load(path, allow_pickle=False),
            expected_size=input_dimension,
        )
    cosine = abs(float(tangents[0] @ tangents[1]))
    if cosine > float(plan["maximum_tangent_absolute_cosine"]):
        raise ValueError("generated tangents are too nearly collinear")
    return base, tangents, cotangents


def _nonce(protocol_commit: str, request_id: str) -> str:
    return sha256_bytes(f"runner|{protocol_commit}|{request_id}".encode("utf-8"))


def _request(
    *,
    request_id: str,
    operation: str,
    protocol_commit: str,
    context_id: str | None = None,
    path_role: str | None = None,
    x: np.ndarray | None = None,
    tangent: np.ndarray | None = None,
    cotangent: np.ndarray | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema": REQUEST_SCHEMA,
        "request_id": request_id,
        "audit_nonce": _nonce(protocol_commit, request_id),
        "operation": operation,
    }
    if operation != "describe":
        value.update(
            {
                "context_id": context_id,
                "path_role": path_role,
                "x": np.asarray(x, dtype=np.float64).tolist(),
            }
        )
    if tangent is not None:
        value["tangent"] = np.asarray(tangent, dtype=np.float64).tolist()
    if cotangent is not None:
        value["cotangent"] = np.asarray(cotangent, dtype=np.float64).tolist()
    return value


def build_requests(
    config: dict[str, Any],
    *,
    protocol_commit: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base, tangents, cotangents = _load_vector_bank(config)
    context_id = str(config["identity"]["context_id"])
    requests = [
        _request(
            request_id=f"describe-{index}",
            operation="describe",
            protocol_commit=protocol_commit,
        )
        for index in range(2)
    ]
    for role in PATH_ROLES:
        for repeat in range(2):
            requests.append(
                _request(
                    request_id=f"forward-base-{role}-repeat-{repeat}",
                    operation="forward",
                    protocol_commit=protocol_commit,
                    context_id=context_id,
                    path_role=role,
                    x=base,
                )
            )
    for role in PATH_ROLES:
        for probe_index, tangent in enumerate(tangents):
            requests.append(
                _request(
                    request_id=f"jvp-{role}-v{probe_index}",
                    operation="jvp",
                    protocol_commit=protocol_commit,
                    context_id=context_id,
                    path_role=role,
                    x=base,
                    tangent=tangent,
                )
            )
    for role in PATH_ROLES:
        requests.append(
            _request(
                request_id=f"vjp-{role}-q0",
                operation="vjp",
                protocol_commit=protocol_commit,
                context_id=context_id,
                path_role=role,
                x=base,
                cotangent=cotangents[0],
            )
        )
    for role in PATH_ROLES:
        for probe_index, tangent in enumerate(tangents):
            for h_index, h in enumerate(config["probe_plan"]["h_values"]):
                step = float(h)
                for side, sign in (("plus", 1.0), ("minus", -1.0)):
                    requests.append(
                        _request(
                            request_id=(
                                f"forward-fd-{role}-v{probe_index}-"
                                f"h{h_index}-{side}"
                            ),
                            operation="forward",
                            protocol_commit=protocol_commit,
                            context_id=context_id,
                            path_role=role,
                            x=base + sign * step * tangent,
                        )
                    )
    vector_summary = {
        "base_input_sha256": array_sha256(base),
        "tangent_sha256": [array_sha256(value) for value in tangents],
        "cotangent_sha256": [array_sha256(value) for value in cotangents],
        "tangent_absolute_cosine": abs(float(tangents[0] @ tangents[1])),
    }
    return requests, {
        "base": base,
        "tangents": tangents,
        "cotangents": cotangents,
        "summary": vector_summary,
    }


def _expand_command(command: list[str]) -> list[str]:
    expanded = [sys.executable if token == "{python}" else token for token in command]
    if any("\n" in token or "\x00" in token for token in expanded):
        raise ValueError("adapter command contains forbidden control characters")
    return expanded


def run_adapter(
    config: dict[str, Any], requests: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    declared_command = [str(value) for value in config["adapter"]["command"]]
    command = _expand_command(declared_command)
    payload = "".join(canonical_json(value) + "\n" for value in requests)
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        input=payload,
        text=True,
        capture_output=True,
        timeout=float(config["adapter"]["timeout_seconds"]),
        check=False,
        env={**os.environ, "PYTHONHASHSEED": "0"},
    )
    wall_time = float(time.perf_counter() - started)
    stderr_bytes = len(completed.stderr.encode("utf-8"))
    if completed.returncode != 0:
        raise RuntimeError(
            f"adapter exited with {completed.returncode}: {completed.stderr[:500]}"
        )
    if stderr_bytes > int(config["cost_contract"]["maximum_stderr_bytes"]):
        raise RuntimeError("adapter wrote more stderr than the frozen contract permits")
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != len(requests):
        raise RuntimeError(
            f"adapter returned {len(lines)} responses for {len(requests)} requests"
        )
    responses: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"response {index} is not JSON") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"response {index} is not an object")
        responses.append(value)
    return responses, {
        # Keep the public trace reproducible without leaking a local executable path.
        "command": declared_command,
        "returncode": completed.returncode,
        "external_wall_time_s": wall_time,
        "stderr_bytes": stderr_bytes,
        "stderr_sha256": sha256_bytes(completed.stderr.encode("utf-8")),
    }


def _path_map(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(path["role"]): path for path in config["paths"]}


def _validate_descriptor(config: dict[str, Any], descriptor: object) -> dict[str, Any]:
    if not isinstance(descriptor, dict):
        raise ValueError("adapter descriptor is not an object")
    if descriptor.get("schema") != DESCRIPTOR_SCHEMA:
        raise ValueError("adapter descriptor schema mismatch")
    adapter = config["adapter"]
    expected_pairs = {
        "adapter_id": adapter["expected_adapter_id"],
        "adapter_version": adapter["expected_adapter_version"],
        "implementation_sha256": adapter["expected_implementation_sha256"],
        "context_id": config["identity"]["context_id"],
        "deterministic": adapter["deterministic_required"],
    }
    for key, expected in expected_pairs.items():
        if descriptor.get(key) != expected:
            raise ValueError(f"adapter descriptor {key} mismatch")
    if descriptor.get("field") != {
        key: config["field"][key]
        for key in ("input_dimension", "parameterization", "units", "axis_order", "dtype")
    }:
        raise ValueError("adapter field descriptor mismatch")
    if descriptor.get("observation") != config["observation"]:
        raise ValueError("adapter observation descriptor mismatch")
    if descriptor.get("paths") != config["paths"]:
        raise ValueError("adapter path descriptor mismatch")
    expected_state = {
        key: config["state_contract"][key]
        for key in (
            "branch_state_source",
            "diagnostic_state_source",
            "branch_fields",
            "diagnostic_fields",
        )
    }
    if descriptor.get("state_contract") != expected_state:
        raise ValueError("adapter state descriptor mismatch")
    cost = descriptor.get("cost_contract")
    if not isinstance(cost, dict):
        raise ValueError("adapter cost descriptor is missing")
    if cost.get("cost_attestation") != config["cost_contract"]["cost_attestation"]:
        raise ValueError("adapter cost attestation mismatch")
    if cost.get("ray_sample_evaluations_are_physical") is not False and (
        config["record_kind"] == "SYNTHETIC_REFERENCE_FIXTURE"
    ):
        raise ValueError("synthetic adapter cannot report physical ray samples")
    primitive = cost.get("primitive_evaluations_per_call")
    if not isinstance(primitive, dict):
        raise ValueError("primitive cost table is missing")
    for operation in ("forward", "jvp", "vjp"):
        if set(primitive.get(operation, {})) != set(PATH_ROLES):
            raise ValueError("primitive cost table path roles mismatch")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in primitive[operation].values()
        ):
            raise ValueError("primitive costs must be nonnegative integers")
    return descriptor


def _allowed_response_keys(operation: str) -> set[str]:
    common = {
        "schema",
        "request_id",
        "audit_nonce",
        "operation",
        "status",
        "runtime_ledger",
        "adapter_wall_time_s",
    }
    if operation == "describe":
        return common | {"descriptor"}
    common |= {
        "path_role",
        "path_id",
        "callable_id",
        "linearization_branch_state",
        "output",
    }
    if operation == "forward":
        common |= {"branch_state", "diagnostic_state"}
    return common


def validate_responses(
    config: dict[str, Any],
    requests: list[dict[str, Any]],
    responses: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if len(requests) != len(responses):
        raise ValueError("request/response count mismatch")
    path_map = _path_map(config)
    descriptors: list[dict[str, Any]] = []
    response_map: dict[str, dict[str, Any]] = {}
    expected_ledger = {
        "describe_calls": 0,
        "forward_api_calls": 0,
        "jvp_api_calls": 0,
        "vjp_api_calls": 0,
        "primitive_evaluations": 0,
        "ray_sample_evaluations": 0,
    }
    primitive_cost: dict[str, dict[str, int]] | None = None
    for request, response in zip(requests, responses, strict=True):
        operation = str(request["operation"])
        if set(response) != _allowed_response_keys(operation):
            extra = sorted(set(response) - _allowed_response_keys(operation))
            missing = sorted(_allowed_response_keys(operation) - set(response))
            raise ValueError(f"response fields drifted; extra={extra}, missing={missing}")
        for key in ("request_id", "audit_nonce", "operation"):
            if response.get(key) != request.get(key):
                raise ValueError(f"response {key} does not echo request")
        if response.get("schema") != RESPONSE_SCHEMA or response.get("status") != "ok":
            raise ValueError("adapter response status/schema mismatch")
        wall = response.get("adapter_wall_time_s")
        if isinstance(wall, bool) or not isinstance(wall, (int, float)):
            raise ValueError("adapter wall time must be numeric")
        if not math.isfinite(float(wall)) or float(wall) < 0.0:
            raise ValueError("adapter wall time is invalid")
        if operation == "describe":
            expected_ledger["describe_calls"] += 1
            descriptor = _validate_descriptor(config, response.get("descriptor"))
            descriptors.append(descriptor)
            primitive_cost = descriptor["cost_contract"][
                "primitive_evaluations_per_call"
            ]
        else:
            role = str(request["path_role"])
            expected = path_map[role]
            for key in ("path_role", "path_id", "callable_id"):
                expected_value = role if key == "path_role" else expected[key]
                if response.get(key) != expected_value:
                    raise ValueError(f"response {key} mismatch")
            if response.get("linearization_branch_state") is None:
                raise ValueError("derivative linearization state is missing")
            expected_ledger[f"{operation}_api_calls"] += 1
            if primitive_cost is None:
                raise ValueError("describe must precede callable operations")
            expected_ledger["primitive_evaluations"] += int(
                primitive_cost[operation][role]
            )
            output_size = (
                int(config["field"]["input_dimension"])
                if operation == "vjp"
                else int(config["observation"]["output_dimension"])
            )
            finite_vector("response output", response.get("output"), expected_size=output_size)
            if operation == "forward":
                branch = response.get("branch_state")
                diagnostic = response.get("diagnostic_state")
                if not isinstance(branch, dict) or not isinstance(diagnostic, dict):
                    raise ValueError("forward states must be objects")
                if set(branch) != set(config["state_contract"]["branch_fields"]):
                    raise ValueError("branch_state fields mismatch")
                if set(diagnostic) != set(
                    config["state_contract"]["diagnostic_fields"]
                ):
                    raise ValueError("diagnostic_state fields mismatch")
        if response.get("runtime_ledger") != expected_ledger:
            raise ValueError("adapter cumulative call ledger mismatch")
        request_id = str(request["request_id"])
        if request_id in response_map:
            raise ValueError("duplicate request_id")
        response_map[request_id] = response
    if len(descriptors) != 2 or descriptors[0] != descriptors[1]:
        raise ValueError("repeated adapter descriptors differ")
    expected_counts = config["cost_contract"]
    for key in ("describe_calls", "forward_api_calls", "jvp_api_calls", "vjp_api_calls"):
        if expected_ledger[key] != int(expected_counts[key]):
            raise ValueError(f"frozen {key} count mismatch")
    return response_map, {
        "descriptor": descriptors[0],
        "final_ledger": expected_ledger,
    }


def _relative_residual(
    residual: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    denominator = max(
        float(np.linalg.norm(first)),
        float(np.linalg.norm(second)),
        1e-300,
    )
    return float(np.linalg.norm(residual)) / denominator


def _relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    denominator = max(
        float(np.linalg.norm(candidate)),
        float(np.linalg.norm(reference)),
        1e-300,
    )
    return float(np.linalg.norm(candidate - reference)) / denominator


def _output(
    response_map: dict[str, dict[str, Any]], request_id: str, size: int
) -> np.ndarray:
    return finite_vector(
        request_id,
        response_map[request_id]["output"],
        expected_size=size,
    )


def derive_metrics(
    config: dict[str, Any],
    response_map: dict[str, dict[str, Any]],
    vectors: dict[str, Any],
) -> dict[str, Any]:
    output_dimension = int(config["observation"]["output_dimension"])
    input_dimension = int(config["field"]["input_dimension"])
    tangents = np.asarray(vectors["tangents"], dtype=np.float64)
    cotangent = np.asarray(vectors["cotangents"][0], dtype=np.float64)
    repeat_rows: list[dict[str, Any]] = []
    base_outputs: dict[str, np.ndarray] = {}
    for role in PATH_ROLES:
        first_id = f"forward-base-{role}-repeat-0"
        second_id = f"forward-base-{role}-repeat-1"
        first = _output(response_map, first_id, output_dimension)
        second = _output(response_map, second_id, output_dimension)
        first_response = response_map[first_id]
        second_response = response_map[second_id]
        base_outputs[role] = first
        repeat_rows.append(
            {
                "path_role": role,
                "maximum_absolute_output_difference": float(
                    np.max(np.abs(first - second))
                ),
                "output_sha256_first": array_sha256(first),
                "output_sha256_second": array_sha256(second),
                "branch_state_equal": (
                    first_response["branch_state"] == second_response["branch_state"]
                ),
                "diagnostic_state_equal": (
                    first_response["diagnostic_state"]
                    == second_response["diagnostic_state"]
                ),
            }
        )
    jvps: dict[tuple[str, int], np.ndarray] = {}
    for role in PATH_ROLES:
        for probe_index in range(2):
            jvps[(role, probe_index)] = _output(
                response_map,
                f"jvp-{role}-v{probe_index}",
                output_dimension,
            )
    vjps = {
        role: _output(response_map, f"vjp-{role}-q0", input_dimension)
        for role in PATH_ROLES
    }
    structure_rows = [
        {
            "obligation": "output",
            "probe_index": None,
            "relative_error": _relative_residual(
                base_outputs["direct_residual"]
                - (base_outputs["curved"] - base_outputs["straight"]),
                base_outputs["direct_residual"],
                base_outputs["curved"] - base_outputs["straight"],
            ),
        }
    ]
    for probe_index in range(2):
        structure_rows.append(
            {
                "obligation": "jvp",
                "probe_index": probe_index,
                "relative_error": _relative_residual(
                    jvps[("direct_residual", probe_index)]
                    - (
                        jvps[("curved", probe_index)]
                        - jvps[("straight", probe_index)]
                    ),
                    jvps[("direct_residual", probe_index)],
                    jvps[("curved", probe_index)]
                    - jvps[("straight", probe_index)],
                ),
            }
        )
    structure_rows.append(
        {
            "obligation": "vjp",
            "probe_index": 0,
            "relative_error": _relative_residual(
                vjps["direct_residual"]
                - (vjps["curved"] - vjps["straight"]),
                vjps["direct_residual"],
                vjps["curved"] - vjps["straight"],
            ),
        }
    )
    fd_rows: list[dict[str, Any]] = []
    any_branch_change = False
    any_diagnostic_change = False
    h_values = [float(value) for value in config["probe_plan"]["h_values"]]
    for role in PATH_ROLES:
        base_response = response_map[f"forward-base-{role}-repeat-0"]
        for probe_index in range(2):
            candidate = jvps[(role, probe_index)]
            for h_index, h in enumerate(h_values):
                plus_id = f"forward-fd-{role}-v{probe_index}-h{h_index}-plus"
                minus_id = f"forward-fd-{role}-v{probe_index}-h{h_index}-minus"
                plus = _output(response_map, plus_id, output_dimension)
                minus = _output(response_map, minus_id, output_dimension)
                estimate = (plus - minus) / (2.0 * h)
                plus_response = response_map[plus_id]
                minus_response = response_map[minus_id]
                branch_changed = bool(
                    plus_response["branch_state"] != minus_response["branch_state"]
                    or plus_response["branch_state"] != base_response["branch_state"]
                    or minus_response["branch_state"] != base_response["branch_state"]
                )
                diagnostic_changed = bool(
                    plus_response["diagnostic_state"]
                    != minus_response["diagnostic_state"]
                    or plus_response["diagnostic_state"]
                    != base_response["diagnostic_state"]
                    or minus_response["diagnostic_state"]
                    != base_response["diagnostic_state"]
                )
                any_branch_change = any_branch_change or branch_changed
                any_diagnostic_change = any_diagnostic_change or diagnostic_changed
                fd_rows.append(
                    {
                        "path_role": role,
                        "probe_index": probe_index,
                        "h": h,
                        "h_float_hex": h.hex(),
                        "plus_input_sha256": array_sha256(
                            vectors["base"] + h * tangents[probe_index]
                        ),
                        "minus_input_sha256": array_sha256(
                            vectors["base"] - h * tangents[probe_index]
                        ),
                        "plus_output_sha256": array_sha256(plus),
                        "minus_output_sha256": array_sha256(minus),
                        "fd_estimate_sha256": array_sha256(estimate),
                        "jvp_sha256": array_sha256(candidate),
                        "fd_relative_error": _relative_l2(estimate, candidate),
                        "branch_changed": branch_changed,
                        "diagnostic_changed": diagnostic_changed,
                        "plus_branch_state_sha256": state_digest(
                            plus_response["branch_state"]
                        ),
                        "minus_branch_state_sha256": state_digest(
                            minus_response["branch_state"]
                        ),
                    }
                )
    adjoint_rows: list[dict[str, Any]] = []
    for role in PATH_ROLES:
        for probe_index, tangent in enumerate(tangents):
            jvp = jvps[(role, probe_index)]
            vjp = vjps[role]
            lhs = float(cotangent @ jvp)
            rhs = float(tangent @ vjp)
            absolute = abs(lhs - rhs)
            action = max(
                float(np.linalg.norm(cotangent)) * float(np.linalg.norm(jvp)),
                float(np.linalg.norm(tangent)) * float(np.linalg.norm(vjp)),
            )
            adjoint_rows.append(
                {
                    "path_role": role,
                    "probe_index": probe_index,
                    "lhs": lhs,
                    "rhs": rhs,
                    "absolute_defect": absolute,
                    "normwise_defect": absolute / max(action, 1e-300),
                    "bilinear_signal_to_action": max(abs(lhs), abs(rhs))
                    / max(action, 1e-300),
                    "jvp_sha256": array_sha256(jvp),
                    "vjp_sha256": array_sha256(vjp),
                }
            )
    aliases = {
        "path_ids_unique": len(
            {path["path_id"] for path in config["paths"]}
        )
        == 3,
        "callable_ids_unique": len(
            {path["callable_id"] for path in config["paths"]}
        )
        == 3,
        "semantic_digests_unique": len(
            {path["semantic_digest_sha256"] for path in config["paths"]}
        )
        == 3,
        "base_output_hashes_unique": len(
            {array_sha256(base_outputs[role]) for role in PATH_ROLES}
        )
        == 3,
    }
    summary = {
        "maximum_repeatability_absolute_difference": max(
            float(row["maximum_absolute_output_difference"])
            for row in repeat_rows
        ),
        "all_repeat_branch_states_equal": all(
            bool(row["branch_state_equal"]) for row in repeat_rows
        ),
        "maximum_structure_relative_error": max(
            float(row["relative_error"]) for row in structure_rows
        ),
        "maximum_fd_relative_error_all_h": max(
            float(row["fd_relative_error"]) for row in fd_rows
        ),
        "maximum_adjoint_normwise_defect": max(
            float(row["normwise_defect"]) for row in adjoint_rows
        ),
        "minimum_bilinear_signal_to_action": min(
            float(row["bilinear_signal_to_action"]) for row in adjoint_rows
        ),
        "any_actual_forward_branch_change": any_branch_change,
        "any_diagnostic_state_change": any_diagnostic_change,
        "all_path_alias_checks_passed": all(aliases.values()),
    }
    return {
        "summary": summary,
        "path_alias_checks": aliases,
        "repeatability_rows": repeat_rows,
        "structure_rows": structure_rows,
        "finite_difference_rows": fd_rows,
        "adjoint_rows": adjoint_rows,
    }


def decide(
    config: dict[str, Any],
    metrics: dict[str, Any],
    descriptor: dict[str, Any],
    final_ledger: dict[str, int],
) -> dict[str, Any]:
    summary = metrics["summary"]
    tolerances = config["engineering_tolerances"]
    gates = {
        "finite_gate": all(
            math.isfinite(float(value))
            for key, value in summary.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ),
        "contract_gate": True,
        "path_alias_gate": bool(summary["all_path_alias_checks_passed"]),
        "cost_gate": all(
            final_ledger[key] == int(config["cost_contract"][key])
            for key in (
                "describe_calls",
                "forward_api_calls",
                "jvp_api_calls",
                "vjp_api_calls",
            )
        ),
        "repeatability_gate": bool(
            float(summary["maximum_repeatability_absolute_difference"])
            <= float(tolerances["repeatability_absolute_max"])
            and summary["all_repeat_branch_states_equal"]
        ),
        "branch_gate": not bool(summary["any_actual_forward_branch_change"]),
        "structure_gate": bool(
            float(summary["maximum_structure_relative_error"])
            <= float(tolerances["structure_relative_max"])
        ),
        "finite_difference_gate": bool(
            float(summary["maximum_fd_relative_error_all_h"])
            <= float(tolerances["finite_difference_relative_max"])
        ),
        "adjoint_gate": bool(
            float(summary["maximum_adjoint_normwise_defect"])
            <= float(tolerances["adjoint_normwise_max"])
        ),
        "strong_signal_gate": bool(
            float(summary["minimum_bilinear_signal_to_action"])
            >= float(tolerances["minimum_bilinear_signal_to_action"])
        ),
        "state_provenance_gate": bool(
            config["state_contract"]["branch_state_source"]
            in {
                "source_instrumented_control_flow",
                "private_source_reviewed_control_flow",
            }
            and config["adapter"]["source_review_status"]
            != "black_box_unverified"
        ),
        "cost_provenance_gate": bool(
            descriptor["cost_contract"]["cost_attestation"]
            != "black_box_self_reported"
        ),
    }
    if not gates["finite_gate"]:
        status = "FAIL_NONFINITE"
    elif not gates["contract_gate"]:
        status = "FAIL_CONTRACT"
    elif not gates["path_alias_gate"]:
        status = "FAIL_PATH_ALIAS"
    elif not gates["cost_gate"]:
        status = "FAIL_COST"
    elif not gates["repeatability_gate"]:
        status = "FAIL_REPEATABILITY"
    elif not gates["branch_gate"]:
        status = "FAIL_BRANCH"
    elif not gates["structure_gate"]:
        status = "FAIL_STRUCTURE"
    elif not gates["finite_difference_gate"]:
        status = "FAIL_FD"
    elif not gates["adjoint_gate"]:
        status = "FAIL_ADJOINT"
    elif not gates["strong_signal_gate"]:
        status = "LOW_SIGNAL_UNRESOLVED"
    elif not gates["state_provenance_gate"]:
        status = "STATE_PROVENANCE_UNRESOLVED"
    elif not gates["cost_provenance_gate"]:
        status = "COST_PROVENANCE_UNRESOLVED"
    elif config["record_kind"] == "SYNTHETIC_REFERENCE_FIXTURE":
        status = "SYNTHETIC_PROTOCOL_PASS_NO_LAB_AUTHORIZATION"
    else:
        status = "LAB_INTERFACE_REPLAYED_NO_PHYSICS_AUTHORIZATION"
    return {
        "status": status,
        "gates": gates,
        "threshold_selection_performed": False,
        "all_h_values_consumed": True,
        "claim_authorized": False,
    }


def _write_fd_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("finite-difference rows are empty")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(metrics: dict[str, Any], decision: dict[str, Any], output: Path) -> None:
    plt.rcParams.update({"font.size": 9})
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.2), constrained_layout=True)
    colors = {
        "curved": "#16756f",
        "straight": "#315f93",
        "direct_residual": "#a34b3f",
    }
    fd_rows = metrics["finite_difference_rows"]
    for role in PATH_ROLES:
        for probe_index, marker in ((0, "o"), (1, "s")):
            rows = [
                row
                for row in fd_rows
                if row["path_role"] == role and row["probe_index"] == probe_index
            ]
            axes[0, 0].loglog(
                [row["h"] for row in rows],
                [max(row["fd_relative_error"], 1e-18) for row in rows],
                marker=marker,
                color=colors[role],
                label=f"{role} / v{probe_index}",
            )
    axes[0, 0].set_title("Actual F(x +/- h v) versus adapter Jv")
    axes[0, 0].set_xlabel("h")
    axes[0, 0].set_ylabel("relative L2")
    axes[0, 0].grid(True, which="both", alpha=0.25)
    axes[0, 0].legend(fontsize=7, ncol=2)

    structure = metrics["structure_rows"]
    labels = [
        f"{row['obligation']}"
        + ("" if row["probe_index"] is None else f"-v{row['probe_index']}")
        for row in structure
    ]
    axes[0, 1].bar(
        labels,
        [max(row["relative_error"], 1e-18) for row in structure],
        color=["#16756f", "#315f93", "#315f93", "#a34b3f"],
    )
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("direct - (curved - straight)")
    axes[0, 1].set_ylabel("relative structure residual")
    axes[0, 1].grid(True, axis="y", which="both", alpha=0.25)

    adjoint = metrics["adjoint_rows"]
    labels = [f"{row['path_role']}-v{row['probe_index']}" for row in adjoint]
    x = np.arange(len(labels))
    width = 0.38
    axes[1, 0].bar(
        x - width / 2,
        [max(row["normwise_defect"], 1e-18) for row in adjoint],
        width,
        label="adjoint normwise defect",
        color="#315f93",
    )
    axes[1, 0].bar(
        x + width / 2,
        [max(row["bilinear_signal_to_action"], 1e-18) for row in adjoint],
        width,
        label="bilinear signal / action",
        color="#d09a35",
    )
    axes[1, 0].set_xticks(x, labels, rotation=25, ha="right")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Adjoint defect and low-signal floor")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, axis="y", which="both", alpha=0.25)

    gates = decision["gates"]
    gate_labels = [
        "contract",
        "path alias",
        "cost",
        "repeat",
        "branch",
        "structure",
        "FD",
        "adjoint",
        "signal",
        "state src",
        "cost src",
    ]
    gate_keys = [
        "contract_gate",
        "path_alias_gate",
        "cost_gate",
        "repeatability_gate",
        "branch_gate",
        "structure_gate",
        "finite_difference_gate",
        "adjoint_gate",
        "strong_signal_gate",
        "state_provenance_gate",
        "cost_provenance_gate",
    ]
    values = [1 if gates[key] else 0 for key in gate_keys]
    axes[1, 1].barh(
        gate_labels,
        values,
        color=["#16756f" if value else "#a34b3f" for value in values],
    )
    axes[1, 1].set_xlim(0, 1.05)
    axes[1, 1].set_xticks([0, 1], ["closed", "met"])
    axes[1, 1].set_title(decision["status"].replace("_", " "))
    figure.suptitle(
        "N5-D5 minimum callable-interface replay\n"
        "Synthetic protocol evidence only; no real BOST authorization",
        fontsize=14,
        fontweight="bold",
    )
    figure.savefig(output, dpi=220)
    plt.close(figure)


def _summary_markdown(result: dict[str, Any]) -> str:
    metrics = result["metrics_summary"]
    counts = result["counts"]
    return "\n".join(
        [
            "# N5-D5 minimum callable-interface bridge",
            "",
            f"- Machine decision: `{result['machine_decision']}`",
            f"- Evidence class: `{result['evidence_class']}`",
            f"- Adapter responses: `{counts['response_count']}`",
            f"- Actual forward API calls: `{counts['forward_api_calls']}`",
            f"- JVP/VJP API calls: `{counts['jvp_api_calls']}/{counts['vjp_api_calls']}`",
            f"- Maximum all-h FD relative error: `{metrics['maximum_fd_relative_error_all_h']:.6e}`",
            f"- Maximum three-path relative error: `{metrics['maximum_structure_relative_error']:.6e}`",
            f"- Maximum adjoint normwise defect: `{metrics['maximum_adjoint_normwise_defect']:.6e}`",
            f"- Actual branch change: `{metrics['any_actual_forward_branch_change']}`",
            f"- Claim authorized: `{result['decision']['claim_authorized']}`",
            "",
            "This is a committed synthetic transport fixture. It does not establish",
            "a real BOST interface, physical correctness, reconstruction quality,",
            "generalization, algorithm superiority, or publication readiness.",
            "",
        ]
    )


def _manifest(directory: Path, names: Iterable[str]) -> dict[str, Any]:
    files = {
        name: {
            "sha256": sha256_file(directory / name),
            "bytes": (directory / name).stat().st_size,
        }
        for name in names
    }
    return {"schema": MANIFEST_SCHEMA, "files": files}


def run(
    *,
    config_path: Path,
    output_dir: Path,
    schema_path: Path = DEFAULT_SCHEMA,
    enforce_committed: bool = True,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    schema_path = schema_path.resolve()
    output_dir = output_dir.resolve()
    config = _read_json(config_path)
    if not isinstance(config, dict):
        raise ValueError("N5-D5 config must be a JSON object")
    validate_config(config, schema_path)
    source_paths = [schema_path, config_path] + [
        _resolve(path) for path in config["adapter"]["source_files"]
    ]
    protocol_commit = (
        _assert_committed(source_paths)
        if enforce_committed
        else _git("rev-parse", "HEAD").stdout.decode("utf-8").strip()
    )
    if config["record_kind"] == "LAB_ANONYMOUS_INTERFACE":
        private_root = (ROOT / "private_library").resolve()
        if private_root not in output_dir.parents:
            raise ValueError("lab traces must be written under private_library")
    requests, vectors = build_requests(config, protocol_commit=protocol_commit)
    responses, process = run_adapter(config, requests)
    response_map, replay = validate_responses(config, requests, responses)
    metrics = derive_metrics(config, response_map, vectors)
    decision = decide(
        config,
        metrics,
        replay["descriptor"],
        replay["final_ledger"],
    )
    source_hashes = {_relative(path): sha256_file(path) for path in source_paths}
    counts = {
        "request_count": len(requests),
        "response_count": len(responses),
        **{
            key: int(replay["final_ledger"][key])
            for key in (
                "describe_calls",
                "forward_api_calls",
                "jvp_api_calls",
                "vjp_api_calls",
                "primitive_evaluations",
                "ray_sample_evaluations",
            )
        },
        "finite_difference_pair_count": len(metrics["finite_difference_rows"]),
        "structure_obligation_count": len(metrics["structure_rows"]),
        "adjoint_probe_count": len(metrics["adjoint_rows"]),
    }
    result = {
        "schema": RESULT_SCHEMA,
        "candidate_id": "N5-D5-MINIMUM-BOST-INTERFACE-BRIDGE",
        "machine_decision": decision["status"],
        "evidence_class": config["identity"]["evidence_class"],
        "record_kind": config["record_kind"],
        "protocol_commit": protocol_commit,
        "config": _relative(config_path),
        "config_sha256": sha256_file(config_path),
        "schema_path": _relative(schema_path),
        "schema_sha256": sha256_file(schema_path),
        "source_hashes": source_hashes,
        "descriptor_sha256": sha256_bytes(
            canonical_json(replay["descriptor"]).encode("utf-8")
        ),
        "vector_summary": vectors["summary"],
        "process": process,
        "counts": counts,
        "metrics_summary": metrics["summary"],
        "path_alias_checks": metrics["path_alias_checks"],
        "decision": decision,
        "claim_authorizations": config["claim_authorizations"],
        "claim_boundary": [
            "The committed fixture validates the JSONL protocol and replay obligations only.",
            "Two Jv directions and one cotangent do not prove a high-dimensional derivative.",
            "Passing FD and adjoint checks does not prove the physical renderer is correct.",
            "No real BOST data, reconstruction, neural operator, or generalization result is present.",
        ],
    }
    public_summary = {
        "schema": PUBLIC_SUMMARY_SCHEMA,
        "machine_decision": decision["status"],
        "evidence_scope": "SYNTHETIC_CALLABLE_INTERFACE_PROTOCOL_ONLY",
        "real_lab_data_present": bool(config["identity"]["real_lab_data_present"]),
        "counts": counts,
        "metrics_summary": metrics["summary"],
        "gates": decision["gates"],
        "claim_authorized": False,
        "claim_authorizations": config["claim_authorizations"],
        "private_identifiers_emitted": False,
        "raw_lab_arrays_emitted": False,
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        _write_jsonl(temporary / "requests.jsonl", requests)
        _write_jsonl(temporary / "responses.jsonl", responses)
        _write_json(temporary / "metrics.json", metrics)
        _write_fd_csv(
            temporary / "finite_difference_rows.csv",
            metrics["finite_difference_rows"],
        )
        _write_json(temporary / "result.json", result)
        _write_json(temporary / "public_summary.json", public_summary)
        (temporary / "summary.md").write_text(
            _summary_markdown(result), encoding="utf-8"
        )
        _plot(metrics, decision, temporary / "interface_bridge.png")
        names = (
            "requests.jsonl",
            "responses.jsonl",
            "metrics.json",
            "finite_difference_rows.csv",
            "result.json",
            "public_summary.json",
            "summary.md",
            "interface_bridge.png",
        )
        _write_json(temporary / "manifest.json", _manifest(temporary, names))
        if output_dir.exists():
            shutil.rmtree(output_dir)
        temporary.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument(
        "--allow-uncommitted",
        action="store_true",
        help="Tests only; formal evidence must use committed protocol sources.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(
        config_path=_resolve(args.config),
        output_dir=_resolve(args.output),
        schema_path=_resolve(args.schema),
        enforce_committed=not args.allow_uncommitted,
    )
    print(
        json.dumps(
            {
                "machine_decision": result["machine_decision"],
                "output": str(_resolve(args.output)),
                "claim_authorized": result["decision"]["claim_authorized"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
