"""Committed synthetic JSONL adapter for the N5-D5 interface protocol.

This module is a transport and replay fixture, not a BOST renderer.  Its three
paths are deliberately simple and exactly related:

``direct_residual(x) = curved(x) - straight(x)``.

The direct path evaluates the nonlinear residual natively, so the runner can
check output/JVP/VJP structure without reading a hidden clean matrix.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from demo_t16_operator.n5_d5_adapter_protocol import (
    DESCRIPTOR_SCHEMA,
    PATH_ROLES,
    REQUEST_SCHEMA,
    RESPONSE_SCHEMA,
    array_sha256,
    canonical_json,
    finite_vector,
    response_line,
    sha256_bytes,
    sha256_file,
)


ADAPTER_ID = "n5-d5-synthetic-smooth-residual"
ADAPTER_VERSION = "1.0"
CONTEXT_ID = "synthetic_context_8r23x1"
INPUT_DIMENSION = 23
RAY_COUNT = 8
COMPONENTS_PER_RAY = 1
OUTPUT_DIMENSION = RAY_COUNT * COMPONENTS_PER_RAY
MODEL_SEED = 55119
NONLINEAR_SCALE = 0.05


def _matrices() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(MODEL_SEED)
    straight = rng.normal(size=(OUTPUT_DIMENSION, INPUT_DIMENSION)).astype(
        np.float64
    ) / math.sqrt(INPUT_DIMENSION)
    bending = rng.normal(size=(OUTPUT_DIMENSION, INPUT_DIMENSION)).astype(
        np.float64
    ) / math.sqrt(INPUT_DIMENSION)
    return straight, bending


STRAIGHT_MATRIX, BENDING_MATRIX = _matrices()


def _semantic_digest(role: str) -> str:
    payload: dict[str, Any] = {
        "role": role,
        "model_seed": MODEL_SEED,
        "input_dimension": INPUT_DIMENSION,
        "output_dimension": OUTPUT_DIMENSION,
    }
    if role in {"curved", "straight"}:
        payload["straight_matrix_sha256"] = array_sha256(STRAIGHT_MATRIX)
    if role in {"curved", "direct_residual"}:
        payload["bending_matrix_sha256"] = array_sha256(BENDING_MATRIX)
        payload["nonlinear_scale"] = NONLINEAR_SCALE
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


PATHS = {
    "curved": {
        "path_id": "synthetic_curved_smooth_v1",
        "callable_id": "curved_forward_jvp_vjp_v1",
        "semantic_digest_sha256": _semantic_digest("curved"),
    },
    "straight": {
        "path_id": "synthetic_straight_linear_v1",
        "callable_id": "straight_forward_jvp_vjp_v1",
        "semantic_digest_sha256": _semantic_digest("straight"),
    },
    "direct_residual": {
        "path_id": "synthetic_direct_residual_native_v1",
        "callable_id": "direct_residual_forward_jvp_vjp_v1",
        "semantic_digest_sha256": _semantic_digest("direct_residual"),
    },
}


EXPECTED_COST = {
    "forward": {
        "curved": 2,
        "straight": 1,
        "direct_residual": 1,
    },
    "jvp": {
        "curved": 2,
        "straight": 1,
        "direct_residual": 1,
    },
    "vjp": {
        "curved": 2,
        "straight": 1,
        "direct_residual": 1,
    },
}


def descriptor() -> dict[str, Any]:
    source = Path(__file__).resolve()
    return {
        "schema": DESCRIPTOR_SCHEMA,
        "adapter_id": ADAPTER_ID,
        "adapter_version": ADAPTER_VERSION,
        "implementation_sha256": sha256_file(source),
        "context_id": CONTEXT_ID,
        "deterministic": True,
        "field": {
            "input_dimension": INPUT_DIMENSION,
            "parameterization": "anonymous_field_vector",
            "units": "dimensionless_synthetic",
            "axis_order": "flat_anonymous",
            "dtype": "float64",
        },
        "observation": {
            "ray_count": RAY_COUNT,
            "components_per_ray": COMPONENTS_PER_RAY,
            "output_dimension": OUTPUT_DIMENSION,
            "units": "normalized_deflection_synthetic",
            "component_order": "scalar_per_ray",
            "dtype": "float64",
        },
        "paths": [
            {
                "role": role,
                **PATHS[role],
                "operation_support": {
                    "forward": True,
                    "jvp": True,
                    "vjp": True,
                },
            }
            for role in PATH_ROLES
        ],
        "state_contract": {
            "branch_state_source": "source_instrumented_control_flow",
            "diagnostic_state_source": "post_forward_diagnostic",
            "branch_fields": [
                "control_flow_id",
                "active_ray_count",
                "sample_count",
                "termination_digest",
            ],
            "diagnostic_fields": [
                "maximum_absolute_output_bin",
                "positive_output_count",
            ],
        },
        "cost_contract": {
            "cost_attestation": "source_instrumented_committed",
            "primitive_evaluations_per_call": EXPECTED_COST,
            "ray_sample_evaluations_are_physical": False,
        },
        "claim_boundary": {
            "synthetic_transport_fixture_only": True,
            "real_bost_authorized": False,
            "physical_forward_correctness_authorized": False,
        },
    }


def _nonlinear_argument(x: np.ndarray) -> np.ndarray:
    return BENDING_MATRIX @ x


def _forward(role: str, x: np.ndarray) -> np.ndarray:
    straight = STRAIGHT_MATRIX @ x
    if role == "straight":
        return straight
    residual = NONLINEAR_SCALE * np.sin(_nonlinear_argument(x))
    if role == "direct_residual":
        return residual
    if role == "curved":
        return straight + residual
    raise ValueError(f"unknown path role: {role}")


def _jvp(role: str, x: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    straight = STRAIGHT_MATRIX @ tangent
    if role == "straight":
        return straight
    residual = (
        NONLINEAR_SCALE
        * np.cos(_nonlinear_argument(x))
        * (BENDING_MATRIX @ tangent)
    )
    if role == "direct_residual":
        return residual
    if role == "curved":
        return straight + residual
    raise ValueError(f"unknown path role: {role}")


def _vjp(role: str, x: np.ndarray, cotangent: np.ndarray) -> np.ndarray:
    straight = STRAIGHT_MATRIX.T @ cotangent
    if role == "straight":
        return straight
    weighted = NONLINEAR_SCALE * np.cos(_nonlinear_argument(x)) * cotangent
    residual = BENDING_MATRIX.T @ weighted
    if role == "direct_residual":
        return residual
    if role == "curved":
        return straight + residual
    raise ValueError(f"unknown path role: {role}")


def _branch_state(role: str) -> dict[str, Any]:
    primitive_count = EXPECTED_COST["forward"][role]
    payload = {
        "role": role,
        "sampling": "fixed_smooth_synthetic",
        "ray_count": RAY_COUNT,
        "primitive_count": primitive_count,
    }
    return {
        "control_flow_id": f"smooth_fixed_sampling:{role}",
        "active_ray_count": RAY_COUNT,
        "sample_count": RAY_COUNT * primitive_count,
        "termination_digest": sha256_bytes(
            canonical_json(payload).encode("utf-8")
        ),
    }


def _diagnostic_state(output: np.ndarray) -> dict[str, Any]:
    maximum = float(np.max(np.abs(output)))
    return {
        "maximum_absolute_output_bin": int(math.floor(maximum * 100.0)),
        "positive_output_count": int(np.count_nonzero(output > 0.0)),
    }


class Ledger:
    def __init__(self) -> None:
        self.describe_calls = 0
        self.forward_api_calls = 0
        self.jvp_api_calls = 0
        self.vjp_api_calls = 0
        self.primitive_evaluations = 0
        self.ray_sample_evaluations = 0

    def update(self, operation: str, role: str | None) -> None:
        if operation == "describe":
            self.describe_calls += 1
            return
        if role is None:
            raise ValueError("non-describe operation requires a path role")
        if operation == "forward":
            self.forward_api_calls += 1
        elif operation == "jvp":
            self.jvp_api_calls += 1
        elif operation == "vjp":
            self.vjp_api_calls += 1
        else:
            raise ValueError(f"unknown operation: {operation}")
        self.primitive_evaluations += EXPECTED_COST[operation][role]

    def to_dict(self) -> dict[str, int]:
        return {
            "describe_calls": self.describe_calls,
            "forward_api_calls": self.forward_api_calls,
            "jvp_api_calls": self.jvp_api_calls,
            "vjp_api_calls": self.vjp_api_calls,
            "primitive_evaluations": self.primitive_evaluations,
            "ray_sample_evaluations": self.ray_sample_evaluations,
        }


def _validate_request(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("request must be a JSON object")
    if value.get("schema") != REQUEST_SCHEMA:
        raise ValueError("unexpected request schema")
    operation = value.get("operation")
    if operation not in {"describe", "forward", "jvp", "vjp"}:
        raise ValueError("unexpected operation")
    if not isinstance(value.get("request_id"), str) or not value["request_id"]:
        raise ValueError("request_id must be nonempty")
    nonce = value.get("audit_nonce")
    if not isinstance(nonce, str) or len(nonce) != 64:
        raise ValueError("audit_nonce must contain 64 characters")
    if operation == "describe":
        return value
    if value.get("context_id") != CONTEXT_ID:
        raise ValueError("context_id mismatch")
    if value.get("path_role") not in PATH_ROLES:
        raise ValueError("path_role mismatch")
    return value


def handle_request(value: object, ledger: Ledger) -> dict[str, Any]:
    request = _validate_request(value)
    operation = str(request["operation"])
    role = None if operation == "describe" else str(request["path_role"])
    started = time.perf_counter()
    ledger.update(operation, role)
    response: dict[str, Any] = {
        "schema": RESPONSE_SCHEMA,
        "request_id": request["request_id"],
        "audit_nonce": request["audit_nonce"],
        "operation": operation,
        "status": "ok",
    }
    if operation == "describe":
        response["descriptor"] = descriptor()
    else:
        assert role is not None
        x = finite_vector(
            "x", request.get("x"), expected_size=INPUT_DIMENSION
        )
        response.update(
            {
                "path_role": role,
                "path_id": PATHS[role]["path_id"],
                "callable_id": PATHS[role]["callable_id"],
                "linearization_branch_state": _branch_state(role),
            }
        )
        if operation == "forward":
            output = _forward(role, x)
            response["output"] = output.tolist()
            response["branch_state"] = _branch_state(role)
            response["diagnostic_state"] = _diagnostic_state(output)
        elif operation == "jvp":
            tangent = finite_vector(
                "tangent",
                request.get("tangent"),
                expected_size=INPUT_DIMENSION,
            )
            response["output"] = _jvp(role, x, tangent).tolist()
        elif operation == "vjp":
            cotangent = finite_vector(
                "cotangent",
                request.get("cotangent"),
                expected_size=OUTPUT_DIMENSION,
            )
            response["output"] = _vjp(role, x, cotangent).tolist()
    response["runtime_ledger"] = ledger.to_dict()
    response["adapter_wall_time_s"] = float(time.perf_counter() - started)
    return response


def main() -> int:
    ledger = Ledger()
    for raw in sys.stdin:
        if not raw.strip():
            continue
        try:
            request = json.loads(raw)
            response = handle_request(request, ledger)
        except Exception as exc:  # pragma: no cover - CLI fail-closed path
            response = {
                "schema": RESPONSE_SCHEMA,
                "request_id": (
                    request.get("request_id", "unknown")
                    if isinstance(locals().get("request"), dict)
                    else "unknown"
                ),
                "audit_nonce": (
                    request.get("audit_nonce", "")
                    if isinstance(locals().get("request"), dict)
                    else ""
                ),
                "operation": (
                    request.get("operation", "unknown")
                    if isinstance(locals().get("request"), dict)
                    else "unknown"
                ),
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "runtime_ledger": ledger.to_dict(),
            }
        sys.stdout.write(response_line(response))
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
