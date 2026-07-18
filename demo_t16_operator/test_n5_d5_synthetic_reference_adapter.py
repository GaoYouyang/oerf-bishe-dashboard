from __future__ import annotations

import math

import numpy as np
import pytest

from demo_t16_operator.n5_d5_adapter_protocol import DESCRIPTOR_SCHEMA
from demo_t16_operator.n5_d5_synthetic_reference_adapter import (
    CONTEXT_ID,
    INPUT_DIMENSION,
    OUTPUT_DIMENSION,
    PATHS,
    Ledger,
    descriptor,
    handle_request,
)


NONCE = "0" * 64


def _request(
    request_id: str,
    operation: str,
    *,
    role: str | None = None,
    x: np.ndarray | None = None,
    tangent: np.ndarray | None = None,
    cotangent: np.ndarray | None = None,
) -> dict[str, object]:
    request: dict[str, object] = {
        "schema": "n5-d5-adapter-request-1.0",
        "request_id": request_id,
        "audit_nonce": NONCE,
        "operation": operation,
    }
    if operation != "describe":
        request.update(
            {
                "context_id": CONTEXT_ID,
                "path_role": role,
                "x": np.asarray(x, dtype=np.float64).tolist(),
            }
        )
    if tangent is not None:
        request["tangent"] = tangent.tolist()
    if cotangent is not None:
        request["cotangent"] = cotangent.tolist()
    return request


def _output(response: dict[str, object]) -> np.ndarray:
    return np.asarray(response["output"], dtype=np.float64)


def test_descriptor_exposes_three_distinct_callable_paths() -> None:
    value = descriptor()

    assert value["schema"] == DESCRIPTOR_SCHEMA
    assert value["deterministic"] is True
    paths = value["paths"]
    assert [path["role"] for path in paths] == [
        "curved",
        "straight",
        "direct_residual",
    ]
    for key in ("path_id", "callable_id", "semantic_digest_sha256"):
        assert len({path[key] for path in paths}) == 3
    assert value["claim_boundary"] == {
        "synthetic_transport_fixture_only": True,
        "real_bost_authorized": False,
        "physical_forward_correctness_authorized": False,
    }


def test_forward_jvp_and_vjp_obey_the_three_path_identity() -> None:
    rng = np.random.default_rng(20260719)
    x = rng.normal(scale=0.2, size=INPUT_DIMENSION)
    tangent = rng.normal(size=INPUT_DIMENSION)
    cotangent = rng.normal(size=OUTPUT_DIMENSION)
    ledger = Ledger()

    outputs: dict[str, np.ndarray] = {}
    jvps: dict[str, np.ndarray] = {}
    vjps: dict[str, np.ndarray] = {}
    for role in PATHS:
        outputs[role] = _output(
            handle_request(
                _request(f"f-{role}", "forward", role=role, x=x), ledger
            )
        )
        jvps[role] = _output(
            handle_request(
                _request(
                    f"j-{role}",
                    "jvp",
                    role=role,
                    x=x,
                    tangent=tangent,
                ),
                ledger,
            )
        )
        vjps[role] = _output(
            handle_request(
                _request(
                    f"v-{role}",
                    "vjp",
                    role=role,
                    x=x,
                    cotangent=cotangent,
                ),
                ledger,
            )
        )

    for values in (outputs, jvps, vjps):
        np.testing.assert_allclose(
            values["curved"] - values["straight"],
            values["direct_residual"],
            rtol=2e-14,
            atol=2e-14,
        )
    for role in PATHS:
        lhs = float(cotangent @ jvps[role])
        rhs = float(tangent @ vjps[role])
        assert math.isclose(lhs, rhs, rel_tol=2e-14, abs_tol=2e-14)


def test_forward_returns_actual_branch_and_separate_diagnostic_state() -> None:
    x = np.linspace(-0.2, 0.2, INPUT_DIMENSION)
    response = handle_request(
        _request("forward", "forward", role="curved", x=x), Ledger()
    )

    assert response["branch_state"] == response["linearization_branch_state"]
    assert set(response["branch_state"]) == {
        "control_flow_id",
        "active_ray_count",
        "sample_count",
        "termination_digest",
    }
    assert set(response["diagnostic_state"]) == {
        "maximum_absolute_output_bin",
        "positive_output_count",
    }
    assert response["branch_state"] != response["diagnostic_state"]


def test_ledger_is_cumulative_and_counts_primitive_calls() -> None:
    ledger = Ledger()
    x = np.zeros(INPUT_DIMENSION)
    described = handle_request(_request("d", "describe"), ledger)
    forwarded = handle_request(
        _request("f", "forward", role="curved", x=x), ledger
    )

    assert described["runtime_ledger"]["describe_calls"] == 1
    assert forwarded["runtime_ledger"] == {
        "describe_calls": 1,
        "forward_api_calls": 1,
        "jvp_api_calls": 0,
        "vjp_api_calls": 0,
        "primitive_evaluations": 2,
        "ray_sample_evaluations": 0,
    }


def test_nonfinite_input_is_rejected_before_evaluation() -> None:
    x = np.zeros(INPUT_DIMENSION)
    x[3] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        handle_request(
            _request("bad", "forward", role="curved", x=x), Ledger()
        )
