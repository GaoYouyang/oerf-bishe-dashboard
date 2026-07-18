#!/usr/bin/env python3
"""Independent replay validator for the N5-D5 callable-interface bridge.

This file deliberately imports neither the runner, the shared wire helpers,
nor the synthetic adapter.  It regenerates vectors and requests, invokes the
declared adapter with fresh nonces, and recomputes every reported metric from
the replayed outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REQUEST_SCHEMA = "n5-d5-adapter-request-1.0"
RESPONSE_SCHEMA = "n5-d5-adapter-response-1.0"
DESCRIPTOR_SCHEMA = "n5-d5-adapter-descriptor-1.0"
RESULT_SCHEMA = "n5-d5-minimum-interface-result-1.0"
MANIFEST_SCHEMA = "n5-d5-minimum-interface-manifest-1.0"
VALIDATION_SCHEMA = "n5-d5-minimum-interface-validation-1.0"
PATH_ROLES = ("curved", "straight", "direct_residual")
MANIFEST_FILES = (
    "requests.jsonl",
    "responses.jsonl",
    "metrics.json",
    "finite_difference_rows.csv",
    "result.json",
    "public_summary.json",
    "summary.md",
    "interface_bridge.png",
)


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray | Iterable[float]) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64).reshape(-1))
    return sha256_bytes(array.tobytes(order="C"))


def state_digest(value: object) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def finite_vector(name: str, value: object, size: int) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64).reshape(-1))
    if array.size != size:
        raise ValueError(f"{name} has size {array.size}, expected {size}")
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{name} contains non-finite values")
    return array


def normalize(value: np.ndarray) -> np.ndarray:
    vector = np.ascontiguousarray(np.asarray(value, dtype=np.float64).reshape(-1))
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("invalid probe direction")
    return vector / norm


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path.name} row {index} is not an object")
        rows.append(value)
    return rows


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


class Checks:
    def __init__(self) -> None:
        self.count = 0

    def require(self, condition: bool, message: str) -> None:
        self.count += 1
        if not condition:
            raise ValueError(message)

    def close(self, first: float, second: float, message: str) -> None:
        self.require(
            math.isclose(
                float(first),
                float(second),
                rel_tol=2e-12,
                abs_tol=2e-15,
            ),
            message,
        )


def _git_show(commit: str, relative: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise ValueError(f"protocol commit does not contain {relative}")
    return completed.stdout


def verify_manifest(result_dir: Path, checks: Checks) -> dict[str, Any]:
    manifest = read_json(result_dir / "manifest.json")
    checks.require(isinstance(manifest, dict), "manifest is not an object")
    checks.require(manifest.get("schema") == MANIFEST_SCHEMA, "manifest schema drifted")
    files = manifest.get("files")
    checks.require(isinstance(files, dict), "manifest files are missing")
    checks.require(set(files) == set(MANIFEST_FILES), "manifest inventory drifted")
    for name in MANIFEST_FILES:
        path = result_dir / name
        checks.require(path.is_file(), f"manifest file is missing: {name}")
        entry = files[name]
        checks.require(isinstance(entry, dict), f"manifest entry is invalid: {name}")
        checks.require(entry.get("sha256") == sha256_file(path), f"hash mismatch: {name}")
        checks.require(entry.get("bytes") == path.stat().st_size, f"size mismatch: {name}")
    return manifest


def generated_vectors(config: dict[str, Any]) -> dict[str, Any]:
    input_dimension = int(config["field"]["input_dimension"])
    output_dimension = int(config["observation"]["output_dimension"])
    base_spec = config["field"]["base_input"]
    if base_spec["source"] == "seeded_gaussian":
        base_rng = np.random.default_rng(int(base_spec["seed"]))
        base = base_rng.normal(scale=0.2, size=input_dimension).astype(np.float64)
    elif base_spec["source"] == "npy_relative":
        path = resolve(base_spec["relative_path"])
        private = (ROOT / "private_library").resolve()
        if private not in path.parents:
            raise ValueError("lab base input escaped private_library")
        if sha256_file(path) != base_spec["sha256"]:
            raise ValueError("private base input hash mismatch")
        base = finite_vector("base input", np.load(path, allow_pickle=False), input_dimension)
    else:
        raise ValueError("unsupported base input source")
    tangent_rng = np.random.default_rng(int(config["probe_plan"]["tangent_seed"]))
    tangents = np.stack(
        [normalize(tangent_rng.normal(size=input_dimension)) for _ in range(2)], axis=0
    )
    cotangent_rng = np.random.default_rng(int(config["probe_plan"]["cotangent_seed"]))
    cotangents = np.stack(
        [normalize(cotangent_rng.normal(size=output_dimension))], axis=0
    )
    return {"base": base, "tangents": tangents, "cotangents": cotangents}


def runner_nonce(commit: str, request_id: str) -> str:
    return sha256_bytes(f"runner|{commit}|{request_id}".encode("utf-8"))


def validator_nonce(manifest_hash: str, request_id: str) -> str:
    return sha256_bytes(f"validator|{manifest_hash}|{request_id}".encode("utf-8"))


def make_request(
    request_id: str,
    operation: str,
    nonce: str,
    *,
    context_id: str | None = None,
    role: str | None = None,
    x: np.ndarray | None = None,
    tangent: np.ndarray | None = None,
    cotangent: np.ndarray | None = None,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "schema": REQUEST_SCHEMA,
        "request_id": request_id,
        "audit_nonce": nonce,
        "operation": operation,
    }
    if operation != "describe":
        request.update(
            {
                "context_id": context_id,
                "path_role": role,
                "x": np.asarray(x, dtype=np.float64).tolist(),
            }
        )
    if tangent is not None:
        request["tangent"] = np.asarray(tangent, dtype=np.float64).tolist()
    if cotangent is not None:
        request["cotangent"] = np.asarray(cotangent, dtype=np.float64).tolist()
    return request


def build_requests(
    config: dict[str, Any],
    vectors: dict[str, Any],
    nonce_for: Any,
) -> list[dict[str, Any]]:
    context = str(config["identity"]["context_id"])
    base = vectors["base"]
    tangents = vectors["tangents"]
    cotangent = vectors["cotangents"][0]
    requests: list[dict[str, Any]] = []
    for index in range(2):
        request_id = f"describe-{index}"
        requests.append(make_request(request_id, "describe", nonce_for(request_id)))
    for role in PATH_ROLES:
        for repeat in range(2):
            request_id = f"forward-base-{role}-repeat-{repeat}"
            requests.append(
                make_request(
                    request_id,
                    "forward",
                    nonce_for(request_id),
                    context_id=context,
                    role=role,
                    x=base,
                )
            )
    for role in PATH_ROLES:
        for probe_index, tangent in enumerate(tangents):
            request_id = f"jvp-{role}-v{probe_index}"
            requests.append(
                make_request(
                    request_id,
                    "jvp",
                    nonce_for(request_id),
                    context_id=context,
                    role=role,
                    x=base,
                    tangent=tangent,
                )
            )
    for role in PATH_ROLES:
        request_id = f"vjp-{role}-q0"
        requests.append(
            make_request(
                request_id,
                "vjp",
                nonce_for(request_id),
                context_id=context,
                role=role,
                x=base,
                cotangent=cotangent,
            )
        )
    for role in PATH_ROLES:
        for probe_index, tangent in enumerate(tangents):
            for h_index, raw_h in enumerate(config["probe_plan"]["h_values"]):
                h = float(raw_h)
                for side, sign in (("plus", 1.0), ("minus", -1.0)):
                    request_id = f"forward-fd-{role}-v{probe_index}-h{h_index}-{side}"
                    requests.append(
                        make_request(
                            request_id,
                            "forward",
                            nonce_for(request_id),
                            context_id=context,
                            role=role,
                            x=base + sign * h * tangent,
                        )
                    )
    return requests


def run_adapter(config: dict[str, Any], requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    command = [
        sys.executable if str(token) == "{python}" else str(token)
        for token in config["adapter"]["command"]
    ]
    payload = "".join(canonical_json(value) + "\n" for value in requests)
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
    if completed.returncode != 0:
        raise ValueError("independent adapter replay failed")
    if len(completed.stderr.encode("utf-8")) > int(
        config["cost_contract"]["maximum_stderr_bytes"]
    ):
        raise ValueError("independent adapter replay wrote forbidden stderr")
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != len(requests):
        raise ValueError("independent response count mismatch")
    responses = [json.loads(line) for line in lines]
    if not all(isinstance(value, dict) for value in responses):
        raise ValueError("independent response is not an object")
    return responses


def response_map(
    requests: list[dict[str, Any]],
    responses: list[dict[str, Any]],
    checks: Checks,
) -> dict[str, dict[str, Any]]:
    checks.require(len(requests) == len(responses), "request/response length mismatch")
    mapped: dict[str, dict[str, Any]] = {}
    expected = {
        "describe_calls": 0,
        "forward_api_calls": 0,
        "jvp_api_calls": 0,
        "vjp_api_calls": 0,
        "primitive_evaluations": 0,
        "ray_sample_evaluations": 0,
    }
    primitive: dict[str, dict[str, int]] | None = None
    for request, response in zip(requests, responses, strict=True):
        for key in ("request_id", "audit_nonce", "operation"):
            checks.require(response.get(key) == request.get(key), f"response {key} mismatch")
        checks.require(response.get("schema") == RESPONSE_SCHEMA, "response schema mismatch")
        checks.require(response.get("status") == "ok", "adapter returned an error")
        operation = str(request["operation"])
        if operation == "describe":
            expected["describe_calls"] += 1
            descriptor = response.get("descriptor")
            checks.require(isinstance(descriptor, dict), "descriptor missing")
            checks.require(descriptor.get("schema") == DESCRIPTOR_SCHEMA, "descriptor schema")
            primitive = descriptor["cost_contract"]["primitive_evaluations_per_call"]
        else:
            role = str(request["path_role"])
            expected[f"{operation}_api_calls"] += 1
            checks.require(primitive is not None, "call preceded descriptor")
            expected["primitive_evaluations"] += int(primitive[operation][role])
        checks.require(response.get("runtime_ledger") == expected, "cumulative ledger mismatch")
        request_id = str(request["request_id"])
        checks.require(request_id not in mapped, "duplicate response id")
        mapped[request_id] = response
    return mapped


def compare_replays(
    stored: dict[str, dict[str, Any]],
    fresh: dict[str, dict[str, Any]],
    checks: Checks,
) -> None:
    checks.require(set(stored) == set(fresh), "stored/fresh request ids differ")
    ignored = {"audit_nonce", "adapter_wall_time_s"}
    for request_id in sorted(stored):
        first = {key: value for key, value in stored[request_id].items() if key not in ignored}
        second = {key: value for key, value in fresh[request_id].items() if key not in ignored}
        checks.require(first == second, f"independent replay mismatch: {request_id}")


def output(mapped: dict[str, dict[str, Any]], request_id: str, size: int) -> np.ndarray:
    return finite_vector(request_id, mapped[request_id]["output"], size)


def relative_residual(residual: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(first)), float(np.linalg.norm(second)), 1e-300)
    return float(np.linalg.norm(residual)) / denominator


def relative_l2(first: np.ndarray, second: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(first)), float(np.linalg.norm(second)), 1e-300)
    return float(np.linalg.norm(first - second)) / denominator


def recompute_metrics(
    config: dict[str, Any],
    mapped: dict[str, dict[str, Any]],
    vectors: dict[str, Any],
) -> dict[str, Any]:
    output_dimension = int(config["observation"]["output_dimension"])
    input_dimension = int(config["field"]["input_dimension"])
    tangents = vectors["tangents"]
    cotangent = vectors["cotangents"][0]
    repeat_rows: list[dict[str, Any]] = []
    base_outputs: dict[str, np.ndarray] = {}
    for role in PATH_ROLES:
        first_id = f"forward-base-{role}-repeat-0"
        second_id = f"forward-base-{role}-repeat-1"
        first = output(mapped, first_id, output_dimension)
        second = output(mapped, second_id, output_dimension)
        base_outputs[role] = first
        repeat_rows.append(
            {
                "path_role": role,
                "maximum_absolute_output_difference": float(np.max(np.abs(first - second))),
                "output_sha256_first": array_sha256(first),
                "output_sha256_second": array_sha256(second),
                "branch_state_equal": mapped[first_id]["branch_state"]
                == mapped[second_id]["branch_state"],
                "diagnostic_state_equal": mapped[first_id]["diagnostic_state"]
                == mapped[second_id]["diagnostic_state"],
            }
        )
    jvps = {
        (role, probe): output(mapped, f"jvp-{role}-v{probe}", output_dimension)
        for role in PATH_ROLES
        for probe in range(2)
    }
    vjps = {
        role: output(mapped, f"vjp-{role}-q0", input_dimension)
        for role in PATH_ROLES
    }
    structure_rows = [
        {
            "obligation": "output",
            "probe_index": None,
            "relative_error": relative_residual(
                base_outputs["direct_residual"]
                - (base_outputs["curved"] - base_outputs["straight"]),
                base_outputs["direct_residual"],
                base_outputs["curved"] - base_outputs["straight"],
            ),
        }
    ]
    for probe in range(2):
        structure_rows.append(
            {
                "obligation": "jvp",
                "probe_index": probe,
                "relative_error": relative_residual(
                    jvps[("direct_residual", probe)]
                    - (jvps[("curved", probe)] - jvps[("straight", probe)]),
                    jvps[("direct_residual", probe)],
                    jvps[("curved", probe)] - jvps[("straight", probe)],
                ),
            }
        )
    structure_rows.append(
        {
            "obligation": "vjp",
            "probe_index": 0,
            "relative_error": relative_residual(
                vjps["direct_residual"] - (vjps["curved"] - vjps["straight"]),
                vjps["direct_residual"],
                vjps["curved"] - vjps["straight"],
            ),
        }
    )
    fd_rows: list[dict[str, Any]] = []
    branch_changed_any = False
    diagnostic_changed_any = False
    for role in PATH_ROLES:
        base_response = mapped[f"forward-base-{role}-repeat-0"]
        for probe, tangent in enumerate(tangents):
            candidate = jvps[(role, probe)]
            for h_index, raw_h in enumerate(config["probe_plan"]["h_values"]):
                h = float(raw_h)
                plus_id = f"forward-fd-{role}-v{probe}-h{h_index}-plus"
                minus_id = f"forward-fd-{role}-v{probe}-h{h_index}-minus"
                plus = output(mapped, plus_id, output_dimension)
                minus = output(mapped, minus_id, output_dimension)
                estimate = (plus - minus) / (2.0 * h)
                plus_response = mapped[plus_id]
                minus_response = mapped[minus_id]
                branch_changed = bool(
                    plus_response["branch_state"] != minus_response["branch_state"]
                    or plus_response["branch_state"] != base_response["branch_state"]
                    or minus_response["branch_state"] != base_response["branch_state"]
                )
                diagnostic_changed = bool(
                    plus_response["diagnostic_state"] != minus_response["diagnostic_state"]
                    or plus_response["diagnostic_state"] != base_response["diagnostic_state"]
                    or minus_response["diagnostic_state"] != base_response["diagnostic_state"]
                )
                branch_changed_any = branch_changed_any or branch_changed
                diagnostic_changed_any = diagnostic_changed_any or diagnostic_changed
                fd_rows.append(
                    {
                        "path_role": role,
                        "probe_index": probe,
                        "h": h,
                        "h_float_hex": h.hex(),
                        "plus_input_sha256": array_sha256(vectors["base"] + h * tangent),
                        "minus_input_sha256": array_sha256(vectors["base"] - h * tangent),
                        "plus_output_sha256": array_sha256(plus),
                        "minus_output_sha256": array_sha256(minus),
                        "fd_estimate_sha256": array_sha256(estimate),
                        "jvp_sha256": array_sha256(candidate),
                        "fd_relative_error": relative_l2(estimate, candidate),
                        "branch_changed": branch_changed,
                        "diagnostic_changed": diagnostic_changed,
                        "plus_branch_state_sha256": state_digest(plus_response["branch_state"]),
                        "minus_branch_state_sha256": state_digest(minus_response["branch_state"]),
                    }
                )
    adjoint_rows: list[dict[str, Any]] = []
    for role in PATH_ROLES:
        for probe, tangent in enumerate(tangents):
            jvp = jvps[(role, probe)]
            vjp = vjps[role]
            lhs = float(cotangent @ jvp)
            rhs = float(tangent @ vjp)
            defect = abs(lhs - rhs)
            action = max(
                float(np.linalg.norm(cotangent)) * float(np.linalg.norm(jvp)),
                float(np.linalg.norm(tangent)) * float(np.linalg.norm(vjp)),
            )
            adjoint_rows.append(
                {
                    "path_role": role,
                    "probe_index": probe,
                    "lhs": lhs,
                    "rhs": rhs,
                    "absolute_defect": defect,
                    "normwise_defect": defect / max(action, 1e-300),
                    "bilinear_signal_to_action": max(abs(lhs), abs(rhs))
                    / max(action, 1e-300),
                    "jvp_sha256": array_sha256(jvp),
                    "vjp_sha256": array_sha256(vjp),
                }
            )
    paths = config["paths"]
    aliases = {
        "path_ids_unique": len({path["path_id"] for path in paths}) == 3,
        "callable_ids_unique": len({path["callable_id"] for path in paths}) == 3,
        "semantic_digests_unique": len(
            {path["semantic_digest_sha256"] for path in paths}
        )
        == 3,
        "base_output_hashes_unique": len(
            {array_sha256(base_outputs[role]) for role in PATH_ROLES}
        )
        == 3,
    }
    summary = {
        "maximum_repeatability_absolute_difference": max(
            row["maximum_absolute_output_difference"] for row in repeat_rows
        ),
        "all_repeat_branch_states_equal": all(
            row["branch_state_equal"] for row in repeat_rows
        ),
        "maximum_structure_relative_error": max(
            row["relative_error"] for row in structure_rows
        ),
        "maximum_fd_relative_error_all_h": max(
            row["fd_relative_error"] for row in fd_rows
        ),
        "maximum_adjoint_normwise_defect": max(
            row["normwise_defect"] for row in adjoint_rows
        ),
        "minimum_bilinear_signal_to_action": min(
            row["bilinear_signal_to_action"] for row in adjoint_rows
        ),
        "any_actual_forward_branch_change": branch_changed_any,
        "any_diagnostic_state_change": diagnostic_changed_any,
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


def compare_metrics(stored: dict[str, Any], fresh: dict[str, Any], checks: Checks) -> None:
    checks.require(set(stored) == set(fresh), "metric sections differ")
    for section in ("path_alias_checks", "repeatability_rows"):
        checks.require(stored[section] == fresh[section], f"{section} mismatch")
    for key, value in fresh["summary"].items():
        if isinstance(value, bool):
            checks.require(stored["summary"].get(key) is value, f"summary {key} mismatch")
        else:
            checks.close(stored["summary"][key], value, f"summary {key} mismatch")
    for section in ("structure_rows", "finite_difference_rows", "adjoint_rows"):
        first_rows = stored[section]
        second_rows = fresh[section]
        checks.require(len(first_rows) == len(second_rows), f"{section} length mismatch")
        for index, (first, second) in enumerate(zip(first_rows, second_rows, strict=True)):
            checks.require(set(first) == set(second), f"{section} row keys mismatch")
            for key, value in second.items():
                if isinstance(value, float):
                    checks.close(first[key], value, f"{section}[{index}].{key} mismatch")
                else:
                    checks.require(first[key] == value, f"{section}[{index}].{key} mismatch")


def expected_decision(config: dict[str, Any], metrics: dict[str, Any]) -> str:
    summary = metrics["summary"]
    tolerance = config["engineering_tolerances"]
    if not bool(summary["all_path_alias_checks_passed"]):
        return "FAIL_PATH_ALIAS"
    if float(summary["maximum_repeatability_absolute_difference"]) > float(
        tolerance["repeatability_absolute_max"]
    ) or not summary["all_repeat_branch_states_equal"]:
        return "FAIL_REPEATABILITY"
    if summary["any_actual_forward_branch_change"]:
        return "FAIL_BRANCH"
    if float(summary["maximum_structure_relative_error"]) > float(
        tolerance["structure_relative_max"]
    ):
        return "FAIL_STRUCTURE"
    if float(summary["maximum_fd_relative_error_all_h"]) > float(
        tolerance["finite_difference_relative_max"]
    ):
        return "FAIL_FD"
    if float(summary["maximum_adjoint_normwise_defect"]) > float(
        tolerance["adjoint_normwise_max"]
    ):
        return "FAIL_ADJOINT"
    if float(summary["minimum_bilinear_signal_to_action"]) < float(
        tolerance["minimum_bilinear_signal_to_action"]
    ):
        return "LOW_SIGNAL_UNRESOLVED"
    if config["adapter"]["source_review_status"] == "black_box_unverified" or (
        config["state_contract"]["branch_state_source"] == "wrapper_label_unverified"
    ):
        return "STATE_PROVENANCE_UNRESOLVED"
    if config["cost_contract"]["cost_attestation"] == "black_box_self_reported":
        return "COST_PROVENANCE_UNRESOLVED"
    if config["record_kind"] == "SYNTHETIC_REFERENCE_FIXTURE":
        return "SYNTHETIC_PROTOCOL_PASS_NO_LAB_AUTHORIZATION"
    return "LAB_INTERFACE_REPLAYED_NO_PHYSICS_AUTHORIZATION"


def validate(result_dir: Path, *, write_report: bool = True) -> dict[str, Any]:
    result_dir = result_dir.resolve()
    checks = Checks()
    manifest = verify_manifest(result_dir, checks)
    result = read_json(result_dir / "result.json")
    metrics = read_json(result_dir / "metrics.json")
    public = read_json(result_dir / "public_summary.json")
    stored_requests = read_jsonl(result_dir / "requests.jsonl")
    stored_responses = read_jsonl(result_dir / "responses.jsonl")
    checks.require(result.get("schema") == RESULT_SCHEMA, "result schema drifted")
    commit = str(result["protocol_commit"])
    config_path = resolve(result["config"])
    config = read_json(config_path)
    checks.require(isinstance(config, dict), "config is not an object")
    checks.require(sha256_file(config_path) == result["config_sha256"], "config hash mismatch")
    process = result.get("process")
    checks.require(isinstance(process, dict), "process record missing")
    checks.require(
        process.get("command") == config["adapter"]["command"],
        "public process command drifted",
    )
    checks.require(
        not any(Path(str(token)).is_absolute() for token in process["command"]),
        "public process command leaked an absolute path",
    )
    checks.require(process.get("returncode") == 0, "adapter return code drifted")
    checks.require(
        int(process.get("stderr_bytes", -1))
        <= int(config["cost_contract"]["maximum_stderr_bytes"]),
        "adapter stderr budget drifted",
    )
    checks.require(
        sha256_file(resolve(result["schema_path"])) == result["schema_sha256"],
        "schema hash mismatch",
    )
    for relative, expected_hash in result["source_hashes"].items():
        current = resolve(relative)
        checks.require(current.is_file(), f"source missing: {relative}")
        checks.require(sha256_file(current) == expected_hash, f"source hash drift: {relative}")
        committed = _git_show(commit, relative)
        checks.require(sha256_bytes(committed) == expected_hash, f"commit source drift: {relative}")
    vectors = generated_vectors(config)
    vector_summary = {
        "base_input_sha256": array_sha256(vectors["base"]),
        "tangent_sha256": [array_sha256(value) for value in vectors["tangents"]],
        "cotangent_sha256": [array_sha256(value) for value in vectors["cotangents"]],
        "tangent_absolute_cosine": abs(
            float(vectors["tangents"][0] @ vectors["tangents"][1])
        ),
    }
    checks.require(result["vector_summary"] == vector_summary, "vector summary mismatch")
    expected_stored = build_requests(
        config,
        vectors,
        lambda request_id: runner_nonce(commit, request_id),
    )
    checks.require(stored_requests == expected_stored, "stored request trace mismatch")
    stored_map = response_map(stored_requests, stored_responses, checks)
    manifest_hash = sha256_file(result_dir / "manifest.json")
    fresh_requests = build_requests(
        config,
        vectors,
        lambda request_id: validator_nonce(manifest_hash, request_id),
    )
    fresh_responses = run_adapter(config, fresh_requests)
    fresh_map = response_map(fresh_requests, fresh_responses, checks)
    compare_replays(stored_map, fresh_map, checks)
    fresh_metrics = recompute_metrics(config, fresh_map, vectors)
    compare_metrics(metrics, fresh_metrics, checks)
    expected_status = expected_decision(config, fresh_metrics)
    checks.require(result["machine_decision"] == expected_status, "machine decision mismatch")
    checks.require(result["decision"]["status"] == expected_status, "decision status mismatch")
    checks.require(result["decision"]["claim_authorized"] is False, "claim opened")
    checks.require(not any(result["claim_authorizations"].values()), "claim authorization opened")
    checks.require(public["machine_decision"] == expected_status, "public decision mismatch")
    checks.require(public["claim_authorized"] is False, "public claim opened")
    checks.require(public["private_identifiers_emitted"] is False, "identifier leak flag")
    checks.require(public["raw_lab_arrays_emitted"] is False, "raw lab leak flag")
    checks.require(
        result["descriptor_sha256"]
        == sha256_bytes(
            canonical_json(fresh_map["describe-0"]["descriptor"]).encode("utf-8")
        ),
        "descriptor digest mismatch",
    )
    report = {
        "schema": VALIDATION_SCHEMA,
        "valid": True,
        "check_count": checks.count,
        "machine_decision_recomputed": expected_status,
        "manifest_sha256": sha256_file(result_dir / "manifest.json"),
        "independence_contract": {
            "runner_imported": False,
            "shared_protocol_helper_imported": False,
            "adapter_imported": False,
            "vectors_regenerated": True,
            "fresh_nonces_used": True,
            "adapter_relaunched": True,
            "all_forward_calls_replayed": True,
            "all_metrics_recomputed": True,
        },
        "claim_boundary": {
            "real_bost_authorized": False,
            "physical_forward_correctness_authorized": False,
            "derivative_proof_authorized": False,
            "reconstruction_authorized": False,
            "algorithm_superiority_authorized": False,
            "generalization_authorized": False,
            "publication_authorized": False,
        },
        "residual_risks": [
            "The validator relaunches the same adapter implementation; it is not a second physical renderer.",
            "A reviewed source hash cannot prove the optical model matches the laboratory apparatus.",
            "Two tangents and one cotangent leave a large high-dimensional derivative null space.",
            "The manifest is content-addressed but not cryptographically signed by an external authority.",
        ],
    }
    if write_report:
        write_json(result_dir / "validation_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate(resolve(args.result_dir), write_report=not args.no_write)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
