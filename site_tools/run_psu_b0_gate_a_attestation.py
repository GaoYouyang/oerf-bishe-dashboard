#!/usr/bin/env python3
"""Generate a clean-commit Gate A mechanics attestation for PSU-B0.

This runner proves only the frozen tiny-fixture implementation contract.  It
does not run a reconstruction benchmark and cannot authorize a performance,
fresh-data, real-data, or method-superiority claim.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from importlib import metadata
import os
from pathlib import Path, PurePosixPath
import platform
import subprocess
import sys
import time
from typing import Any, Sequence
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch import nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from demo_t16_operator.psu_b0_factor_majorizer_pipeline import (  # noqa: E402
    FactorPipelineCallLedger,
    PSUB0FactorMajorizerSetup,
    build_deleted_data_ledger,
    factor_pdhg_objective,
    run_factor_pdhg,
    build_psu_b0_factor_majorizer_pipeline,
)
from demo_t16_operator.psu_b0_absolute_measurement_factor import (  # noqa: E402
    ExactAbsoluteMeasurementFactor,
)
from demo_t16_operator.psu_b0_gate_a_fixture import (  # noqa: E402
    DEFAULT_GATE_A_CONFIG_PATH,
    build_gate_a_fixture,
    canonical_json_sha256,
    file_sha256,
    gate_a_input_payload,
    gate_a_source_hashes,
    load_gate_a_config,
)
from demo_t16_operator.psu_b0_reconstruction_interface import (  # noqa: E402
    finite_difference_gradient,
    finite_difference_gradient_adjoint,
)
from demo_t16_operator.psu_b0_signed_factor_majorizer import (  # noqa: E402
    SignedFactorSystem,
    build_majorizer_setup,
)


ATTESTATION_SCHEMA = "psu-b0-gate-a-attestation-1.0"
FORMAL_STATUS = "FORMAL_GATE_A_ATTESTED_MECHANICS_ONLY"
CLAIM_BOUNDARY = "GATE_B_NOT_RUN_NO_FRESH_REAL_OR_WIN_CLAIM"
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
FORBIDDEN_ENVIRONMENT = (
    "PYTEST_ADDOPTS",
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "PYTORCH_ENABLE_MPS_FALLBACK",
)


def _run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def worktree_is_clean() -> bool:
    return not bool(_run_git("status", "--porcelain", "--untracked-files=all"))


def git_index_is_plain() -> bool:
    return all(
        line.startswith("H ")
        for line in _run_git("ls-files", "-v").splitlines()
    )


def _relative_error(left: torch.Tensor, right: torch.Tensor) -> float:
    a = left.detach().to(device="cpu").to(dtype=torch.float64)
    b = right.detach().to(device="cpu").to(dtype=torch.float64)
    numerator = torch.linalg.vector_norm((a - b).reshape(-1))
    denominator = torch.maximum(
        torch.maximum(
            torch.linalg.vector_norm(a.reshape(-1)),
            torch.linalg.vector_norm(b.reshape(-1)),
        ),
        torch.as_tensor(1e-30, dtype=torch.float64),
    )
    return float(numerator / denominator)


def _dot_relative_error(left: torch.Tensor, right: torch.Tensor) -> float:
    a = float(left.detach().to(device="cpu").to(dtype=torch.float64))
    b = float(right.detach().to(device="cpu").to(dtype=torch.float64))
    return abs(a - b) / max(abs(a), abs(b), 1e-30)


def _maximum_relative_error(left: torch.Tensor, right: torch.Tensor) -> float:
    a = left.detach().to(device="cpu").to(dtype=torch.float64)
    b = right.detach().to(device="cpu").to(dtype=torch.float64)
    scale = torch.maximum(torch.maximum(a.abs(), b.abs()), torch.ones_like(a))
    return float(torch.max((a - b).abs() / scale)) if a.numel() else 0.0


def _basis_matrix(
    forward: Any,
    columns: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    basis = torch.eye(columns, dtype=dtype, device=device)
    values = forward(basis)
    return values.reshape(columns, -1).T.contiguous()


def _dense_factor_matrices(
    setup: PSUB0FactorMajorizerSetup,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pipeline = setup.pipeline
    kwargs = {
        "columns": pipeline.n_active,
        "dtype": pipeline.dtype,
        "device": pipeline.device,
    }
    return (
        _basis_matrix(pipeline.signed_data_forward, **kwargs),
        _basis_matrix(pipeline.signed_tv_forward, **kwargs),
        _basis_matrix(pipeline.absolute_data_forward, **kwargs),
        _basis_matrix(pipeline.absolute_tv_forward, **kwargs),
    )


def _deterministic_values(
    count: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    phase: float,
) -> torch.Tensor:
    indices = torch.arange(count, dtype=dtype, device=device)
    return torch.sin(indices * 0.37 + phase) + 0.2 * torch.cos(indices * 0.19)


def _adjoint_evidence(setup: PSUB0FactorMajorizerSetup) -> dict[str, float]:
    pipeline = setup.pipeline
    dtype = pipeline.dtype
    device = pipeline.device
    x = _deterministic_values(
        pipeline.n_active, dtype=dtype, device=device, phase=0.13
    )[None]
    data = _deterministic_values(
        pipeline.ray_count * 2, dtype=dtype, device=device, phase=0.41
    ).reshape(1, pipeline.ray_count, 2)
    tv = _deterministic_values(
        3 * math.prod(pipeline.grid_shape),
        dtype=dtype,
        device=device,
        phase=0.77,
    ).reshape(1, 3, *pipeline.grid_shape)

    data_error = _dot_relative_error(
        torch.sum(pipeline.signed_data_forward(x) * data),
        torch.sum(x * pipeline.signed_data_transpose(data)),
    )
    tv_error = _dot_relative_error(
        torch.sum(pipeline.signed_tv_forward(x) * tv),
        torch.sum(x * pipeline.signed_tv_transpose(tv)),
    )

    components = _deterministic_values(
        2 * 3 * math.prod(pipeline.grid_shape),
        dtype=dtype,
        device=device,
        phase=0.29,
    ).reshape(2, 3, *pipeline.grid_shape)
    sampled = _deterministic_values(
        2 * 3 * pipeline.ray_count * pipeline.sample_count,
        dtype=dtype,
        device=device,
        phase=0.53,
    ).reshape(2, 3, pipeline.ray_count, pipeline.sample_count)
    p_error = _dot_relative_error(
        torch.sum(pipeline.voxel_operator.trilinear_interpolation(components) * sampled),
        torch.sum(
            components
            * pipeline.voxel_operator.trilinear_interpolation_adjoint(sampled)
        ),
    )

    volume = _deterministic_values(
        2 * math.prod(pipeline.grid_shape),
        dtype=dtype,
        device=device,
        phase=0.17,
    ).reshape(2, *pipeline.grid_shape)
    gradient_dual = _deterministic_values(
        2 * 3 * math.prod(pipeline.grid_shape),
        dtype=dtype,
        device=device,
        phase=0.83,
    ).reshape(2, 3, *pipeline.grid_shape)
    gradient = finite_difference_gradient(
        volume,
        spacing_xyz=pipeline.voxel_operator.spacing_xyz,
    )
    gradient_error = _dot_relative_error(
        torch.sum(gradient * gradient_dual),
        torch.sum(
            volume
            * finite_difference_gradient_adjoint(
                gradient_dual,
                spacing_xyz=pipeline.voxel_operator.spacing_xyz,
            )
        ),
    )
    return {
        "A_AT_relative_error": data_error,
        "P_PT_relative_error": p_error,
        "G_GT_relative_error": gradient_error,
        "D_DT_relative_error": tv_error,
        "maximum_relative_error": max(
            data_error, p_error, gradient_error, tv_error
        ),
    }


def _metric_evidence(
    setup: PSUB0FactorMajorizerSetup,
    matrices: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, Any]:
    A, D, M, N = matrices
    data_row_error = _maximum_relative_error(
        setup.data_row_sums.reshape(-1), M.sum(dim=1)
    )
    data_column_error = _maximum_relative_error(
        setup.data_column_sums, M.sum(dim=0)
    )
    tv_row_error = _maximum_relative_error(
        setup.tv_row_sums.reshape(-1), N.sum(dim=1)
    )
    tv_column_error = _maximum_relative_error(
        setup.tv_column_sums, N.sum(dim=0)
    )
    data_rows = M.sum(dim=1).reshape(
        setup.pipeline.view_count,
        setup.pipeline.rays_per_view,
        2,
    )
    expected_rho_data = data_rows.amax(dim=(1, 2))
    tv_rows_site_major = torch.movedim(
        N.sum(dim=1).reshape(3, *setup.pipeline.grid_shape), 0, -1
    )
    expected_rho_tv = tv_rows_site_major.amax(dim=-1)
    maximum_sum_error = max(
        data_row_error,
        data_column_error,
        tv_row_error,
        tv_column_error,
        _maximum_relative_error(setup.rho_data_by_view, expected_rho_data),
        _maximum_relative_error(setup.rho_tv_by_site, expected_rho_tv),
    )

    dominance_data = torch.clamp(A.abs() - M, min=0.0)
    dominance_tv = torch.clamp(D.abs() - N, min=0.0)
    data_mask = setup.data_row_mask.reshape(-1)
    tv_mask = setup.tv_site_mask.unsqueeze(0).expand(
        3, *setup.pipeline.grid_shape
    ).reshape(-1)
    primal_mask = setup.active_primal_mask
    K = torch.cat(
        (A[data_mask][:, primal_mask], D[tv_mask][:, primal_mask]), dim=0
    )
    sigma_data = setup.data_sigma_rows.reshape(-1)[data_mask]
    sigma_tv = setup.tv_sigma_rows.reshape(-1)[tv_mask]
    sigma = torch.cat((sigma_data, sigma_tv))
    scaled = torch.sqrt(sigma)[:, None] * K * torch.sqrt(setup.tau)[None]
    singular_values = torch.linalg.svdvals(scaled)
    norm_squared = (
        float(singular_values[0].square().detach().cpu())
        if singular_values.numel()
        else 0.0
    )
    finite_metric = all(
        bool(torch.all(torch.isfinite(value)))
        for value in (
            setup.data_row_sums,
            setup.data_column_sums,
            setup.tv_row_sums,
            setup.tv_column_sums,
            setup.rho_data_by_view,
            setup.rho_tv_by_site,
            setup.tau,
        )
    )
    return {
        "embedding_minimum": float(setup.pipeline.gauge.E.min()),
        "trilinear_weight_minimum": float(
            setup.pipeline.voxel_operator.sample_weights.min().detach().cpu()
        ),
        "data_dominance_violation_max": float(dominance_data.max().detach().cpu()),
        "tv_dominance_violation_max": float(dominance_tv.max().detach().cpu()),
        "maximum_sum_relative_error": maximum_sum_error,
        "scaled_operator_norm_squared": norm_squared,
        "eta_squared": float(setup.eta**2),
        "svd_margin": float(setup.eta**2 - norm_squared),
        "finite_metric": finite_metric,
        "active_data_rows": int(torch.count_nonzero(data_mask)),
        "deleted_data_rows": int(torch.count_nonzero(~data_mask)),
        "active_tv_sites": int(torch.count_nonzero(setup.tv_site_mask)),
        "active_primal_columns": int(setup.active_primal_count),
        "deleted_primal_columns": int(
            setup.pipeline.n_active - setup.active_primal_count
        ),
    }


def _ledger_delta(
    after: FactorPipelineCallLedger,
    before: FactorPipelineCallLedger,
) -> dict[str, int]:
    return {
        name: int(getattr(after, name) - getattr(before, name))
        for name in FactorPipelineCallLedger.__dataclass_fields__
    }


def _dense_six_step_trace(
    setup: PSUB0FactorMajorizerSetup,
    target: torch.Tensor,
    A: torch.Tensor,
    D: torch.Tensor,
    *,
    regularization_weight: float,
    huber_delta: float,
    theta: float,
    iterations: int,
) -> list[dict[str, torch.Tensor]]:
    dtype = A.dtype
    device = A.device
    active_primal = setup.active_primal_indices
    data_mask = setup.data_row_mask.reshape(-1)
    target_flat = target.reshape(-1)
    sigma_data = setup.data_sigma_rows.reshape(-1)
    sigma_tv = setup.tv_sigma_rows
    x = torch.zeros(setup.active_primal_count, dtype=dtype, device=device)
    x_bar = torch.zeros_like(x)
    p = torch.zeros_like(target_flat)
    q = torch.zeros((3, *setup.pipeline.grid_shape), dtype=dtype, device=device)
    trace: list[dict[str, torch.Tensor]] = []
    for _ in range(iterations):
        full_x_bar = torch.zeros(
            setup.pipeline.n_active, dtype=dtype, device=device
        )
        full_x_bar.index_copy_(0, active_primal, x_bar)
        data_value = A @ full_x_bar
        candidate_p = (p + sigma_data * (data_value - target_flat)) / (
            1.0 + sigma_data
        )
        p = torch.where(data_mask, candidate_p, torch.zeros_like(candidate_p))

        tv_value = (D @ full_x_bar).reshape(3, *setup.pipeline.grid_shape)
        candidate_q = q + sigma_tv * tv_value
        candidate_q = candidate_q / (
            1.0 + sigma_tv * huber_delta / regularization_weight
        )
        norm = torch.linalg.vector_norm(candidate_q, dim=0)
        candidate_q = candidate_q / torch.clamp(
            norm / regularization_weight, min=1.0
        )[None]
        q = torch.where(
            setup.tv_site_mask[None], candidate_q, torch.zeros_like(candidate_q)
        )

        gradient_full = A.T @ p + D.T @ q.reshape(-1)
        gradient = gradient_full.index_select(0, active_primal)
        next_x = x - setup.tau * gradient
        next_x_bar = next_x + theta * (next_x - x)
        x, x_bar = next_x, next_x_bar
        trace.append(
            {
                "x": x.clone(),
                "x_bar": x_bar.clone(),
                "data_dual": p.reshape_as(target).clone(),
                "tv_dual": q.clone(),
            }
        )
    return trace


def collect_cpu_numeric_evidence(config: dict[str, Any]) -> dict[str, Any]:
    """Recompute all quantitative float64 mechanics checks on CPU."""

    fixture = build_gate_a_fixture(config, device="cpu", dtype=torch.float64)
    setup = fixture.setup
    audit_logical_before = setup.pipeline.call_ledger()
    audit_physical_before = setup.pipeline.physical_call_ledger()
    matrices = _dense_factor_matrices(setup)
    A, D, _, _ = matrices
    adjoint = _adjoint_evidence(setup)
    metric = _metric_evidence(setup, matrices)
    audit_logical_after = setup.pipeline.call_ledger()
    audit_physical_after = setup.pipeline.physical_call_ledger()
    audit_logical_delta = _ledger_delta(
        audit_logical_after, audit_logical_before
    )
    audit_physical_delta = _ledger_delta(
        audit_physical_after, audit_physical_before
    )
    parameters = config["fixture"]

    logical_before = setup.pipeline.call_ledger()
    physical_before = setup.pipeline.physical_call_ledger()
    states = run_factor_pdhg(
        setup,
        fixture.target,
        iterations=int(parameters["iterations"]),
        regularization_weight=float(parameters["regularization_weight"]),
        penalty=str(parameters["penalty"]),
        huber_delta=float(parameters["huber_delta"]),
        theta=float(parameters["theta"]),
    )
    logical_after = setup.pipeline.call_ledger()
    physical_after = setup.pipeline.physical_call_ledger()
    logical_delta = _ledger_delta(logical_after, logical_before)
    physical_delta = _ledger_delta(physical_after, physical_before)
    expected_solve_ledger = {
        "signed_data_forward_calls": 6,
        "signed_data_transpose_calls": 6,
        "absolute_data_forward_calls": 0,
        "absolute_data_transpose_calls": 0,
        "signed_tv_forward_calls": 6,
        "signed_tv_transpose_calls": 6,
        "absolute_tv_forward_calls": 0,
        "absolute_tv_transpose_calls": 0,
    }
    expected_audit_ledger = {
        "signed_data_forward_calls": 2,
        "signed_data_transpose_calls": 1,
        "absolute_data_forward_calls": 1,
        "absolute_data_transpose_calls": 0,
        "signed_tv_forward_calls": 2,
        "signed_tv_transpose_calls": 1,
        "absolute_tv_forward_calls": 1,
        "absolute_tv_transpose_calls": 0,
    }
    dense_trace = _dense_six_step_trace(
        setup,
        fixture.target,
        A,
        D,
        regularization_weight=float(parameters["regularization_weight"]),
        huber_delta=float(parameters["huber_delta"]),
        theta=float(parameters["theta"]),
        iterations=int(parameters["iterations"]),
    )
    state_errors: list[dict[str, float]] = []
    objective_errors: list[float] = []
    state_trace: list[dict[str, Any]] = []
    scorer_logical_before = setup.pipeline.call_ledger()
    scorer_physical_before = setup.pipeline.physical_call_ledger()
    for production, dense in zip(states, dense_trace):
        errors = {
            name: _relative_error(getattr(production, name), dense[name])
            for name in ("x", "x_bar", "data_dual", "tv_dual")
        }
        state_errors.append(errors)
        state_trace.append(
            {
                "x": production.x.detach().cpu().tolist(),
                "x_bar": production.x_bar.detach().cpu().tolist(),
                "data_dual": production.data_dual.detach().cpu().tolist(),
                "tv_dual": production.tv_dual.detach().cpu().tolist(),
            }
        )

        full_x = torch.zeros(setup.pipeline.n_active, dtype=torch.float64)
        full_x.index_copy_(0, setup.active_primal_indices, dense["x"])
        residual = A @ full_x - fixture.target.reshape(-1)
        magnitudes = torch.linalg.vector_norm(
            (D @ full_x).reshape(3, *setup.pipeline.grid_shape), dim=0
        )
        delta = float(parameters["huber_delta"])
        huber = torch.where(
            magnitudes <= delta,
            0.5 * magnitudes.square() / delta,
            magnitudes - 0.5 * delta,
        )
        dense_objective = 0.5 * torch.sum(residual.square()) + float(
            parameters["regularization_weight"]
        ) * torch.sum(huber)
        production_objective = factor_pdhg_objective(
            setup,
            production,
            fixture.target,
            regularization_weight=float(parameters["regularization_weight"]),
            penalty=str(parameters["penalty"]),
            huber_delta=delta,
        )
        objective_errors.append(
            _relative_error(production_objective, dense_objective)
        )
        state_trace[-1]["objective"] = float(
            production_objective.detach().cpu()
        )

    scorer_logical_after = setup.pipeline.call_ledger()
    scorer_physical_after = setup.pipeline.physical_call_ledger()
    scorer_logical_delta = _ledger_delta(
        scorer_logical_after, scorer_logical_before
    )
    scorer_physical_delta = _ledger_delta(
        scorer_physical_after, scorer_physical_before
    )
    expected_scorer_ledger = {
        "signed_data_forward_calls": 6,
        "signed_data_transpose_calls": 0,
        "absolute_data_forward_calls": 0,
        "absolute_data_transpose_calls": 0,
        "signed_tv_forward_calls": 6,
        "signed_tv_transpose_calls": 0,
        "absolute_tv_forward_calls": 0,
        "absolute_tv_transpose_calls": 0,
    }

    deleted = build_deleted_data_ledger(setup, fixture.target)
    direct_deleted_constant = 0.5 * torch.sum(
        fixture.target.reshape(-1)
        .index_select(0, deleted.deleted_flat_indices)
        .square()
    )
    deleted_constant_error = _relative_error(
        deleted.objective_constant, direct_deleted_constant
    )
    maximum_state_error = max(
        value for row in state_errors for value in row.values()
    )
    maximum_objective_error = max(objective_errors)
    setup_ledger = asdict(setup.setup_call_ledger)
    setup_physical_ledger = asdict(setup.setup_physical_call_ledger)
    expected_setup_ledger = {
        "signed_data_forward_calls": 0,
        "signed_data_transpose_calls": 0,
        "absolute_data_forward_calls": 1,
        "absolute_data_transpose_calls": 1,
        "signed_tv_forward_calls": 0,
        "signed_tv_transpose_calls": 0,
        "absolute_tv_forward_calls": 1,
        "absolute_tv_transpose_calls": 1,
    }
    return {
        "device": "cpu",
        "dtype": "torch.float64",
        "factor_content_sha256": setup.factor_freeze_token.content_sha256,
        "setup_content_sha256": setup.setup_freeze_token.content_sha256,
        "whitening_contract": setup.pipeline.measurement_factor.contract_metadata,
        "adjoint": adjoint,
        "metric": metric,
        "recurrence": {
            "iterations": int(parameters["iterations"]),
            "state_relative_errors_by_step": state_errors,
            "objective_relative_errors_by_step": objective_errors,
            "state_trace": state_trace,
            "maximum_state_relative_error": maximum_state_error,
            "maximum_objective_relative_error": maximum_objective_error,
            "final_reduced_x": states[-1].x.detach().cpu().tolist(),
            "final_reduced_x_bar": states[-1].x_bar.detach().cpu().tolist(),
        },
        "zero_coupling": {
            "active_data_indices": deleted.active_flat_indices.detach().cpu().tolist(),
            "deleted_data_indices": deleted.deleted_flat_indices.detach().cpu().tolist(),
            "deleted_target_values": deleted.deleted_target_values.detach().cpu().tolist(),
            "objective_constant": float(deleted.objective_constant.detach().cpu()),
            "direct_constant_relative_error": deleted_constant_error,
        },
        "ledgers": {
            "setup_logical": setup_ledger,
            "setup_physical": setup_physical_ledger,
            "setup_expected": expected_setup_ledger,
            "oracle_audit_logical_delta": audit_logical_delta,
            "oracle_audit_physical_delta": audit_physical_delta,
            "oracle_audit_expected": expected_audit_ledger,
            "solve_logical_delta": logical_delta,
            "solve_physical_delta": physical_delta,
            "solve_expected": expected_solve_ledger,
            "scorer_logical_delta": scorer_logical_delta,
            "scorer_physical_delta": scorer_physical_delta,
            "scorer_expected": expected_scorer_ledger,
        },
    }


def collect_mps_parity_evidence(config: dict[str, Any]) -> dict[str, Any]:
    """Compare the frozen six-step CPU/float64 and MPS/float32 recurrences."""

    if not torch.backends.mps.is_available():
        raise RuntimeError("formal Gate A requires an available MPS device")
    parameters = config["fixture"]
    cpu = build_gate_a_fixture(config, device="cpu", dtype=torch.float64)
    mps = build_gate_a_fixture(config, device="mps", dtype=torch.float32)
    kwargs = {
        "iterations": int(parameters["iterations"]),
        "regularization_weight": float(parameters["regularization_weight"]),
        "penalty": str(parameters["penalty"]),
        "huber_delta": float(parameters["huber_delta"]),
        "theta": float(parameters["theta"]),
    }
    cpu_states = run_factor_pdhg(cpu.setup, cpu.target, **kwargs)
    mps_states = run_factor_pdhg(mps.setup, mps.target, **kwargs)
    torch.mps.synchronize()
    by_step = [
        {
            name: _relative_error(
                getattr(cpu_state, name), getattr(mps_state, name)
            )
            for name in ("x", "x_bar", "data_dual", "tv_dual")
        }
        for cpu_state, mps_state in zip(cpu_states, mps_states)
    ]
    by_state = by_step[-1]
    field_error = by_state["x"]
    def state_payload(state: Any) -> dict[str, Any]:
        return {
            name: getattr(state, name).detach().cpu().tolist()
            for name in ("x", "x_bar", "data_dual", "tv_dual")
        }
    return {
        "available": True,
        "cpu_dtype": "torch.float64",
        "mps_dtype": "torch.float32",
        "iterations": int(parameters["iterations"]),
        "final_state_relative_errors": by_state,
        "state_relative_errors_by_step": by_step,
        "cpu_state_trace": [state_payload(state) for state in cpu_states],
        "mps_state_trace": [state_payload(state) for state in mps_states],
        "field_relative_difference": field_error,
        "maximum_state_relative_difference": max(
            value for row in by_step for value in row.values()
        ),
        "cpu_factor_content_sha256": cpu.setup.factor_freeze_token.content_sha256,
        "mps_factor_content_sha256": mps.setup.factor_freeze_token.content_sha256,
    }


def _capture_rejection(operation: Any) -> dict[str, Any]:
    try:
        operation()
    except (AssertionError, RuntimeError, TypeError, ValueError) as error:
        return {
            "rejected": True,
            "exception_type": type(error).__name__,
            "message": str(error),
        }
    return {"rejected": False, "exception_type": None, "message": None}


def collect_negative_control_evidence(config: dict[str, Any]) -> dict[str, Any]:
    """Run fail-closed controls instead of treating E1-09/E1-13 as constants."""

    controls: dict[str, dict[str, Any]] = {}

    def stale_control(name: str, mutation: Any) -> None:
        fixture = build_gate_a_fixture(config, device="cpu", dtype=torch.float64)
        mutation(fixture.setup)
        controls[name] = _capture_rejection(
            lambda: run_factor_pdhg(fixture.setup, fixture.target, iterations=1)
        )

    stale_control(
        "stale_signed_kernel",
        lambda setup: setup.pipeline.measurement_factor.signed_kernel.data.__setitem__(
            (0, 0, 0, 0),
            setup.pipeline.measurement_factor.signed_kernel[0, 0, 0, 0] + 0.1,
        ),
    )
    stale_control(
        "stale_scale_by_view",
        lambda setup: setup.pipeline.measurement_factor.scale_by_view.data.__setitem__(
            (0, 0), setup.pipeline.measurement_factor.scale_by_view[0, 0] * 1.1
        ),
    )
    stale_control(
        "stale_geometry_weight",
        lambda setup: setup.pipeline.voxel_operator.sample_weights.data.__setitem__(
            (0, 0, 0), setup.pipeline.voxel_operator.sample_weights[0, 0, 0] + 0.1
        ),
    )
    stale_control(
        "stale_support",
        lambda setup: setup.pipeline.gauge.support.data.reshape(-1).__setitem__(
            1, 0.0
        ),
    )
    stale_control(
        "stale_tau",
        lambda setup: setup.tau.data.__setitem__(0, setup.tau[0] * 1.1),
    )
    stale_control(
        "stale_sigma_data",
        lambda setup: setup.sigma_data_by_view.data.__setitem__(
            0, setup.sigma_data_by_view[0] * 1.1
        ),
    )
    stale_control(
        "stale_data_mask",
        lambda setup: setup.data_row_mask.data.reshape(-1).__setitem__(0, False),
    )
    stale_control(
        "stale_active_primal_indices",
        lambda setup: setup.active_primal_indices.data.__setitem__(
            0, setup.active_primal_indices[0] + 1
        ),
    )
    stale_control(
        "stale_pipeline_device_indices",
        lambda setup: setup.pipeline._active_indices.data.__setitem__(
            0, setup.pipeline._active_indices[0] + 1
        ),
    )
    stale_control(
        "stale_voxel_spacing",
        lambda setup: setattr(
            setup.pipeline.voxel_operator, "spacing_xyz", (1.1, 1.0, 1.0)
        ),
    )

    base = build_gate_a_fixture(config, device="cpu", dtype=torch.float64)
    factor = base.setup.pipeline.measurement_factor
    factor.scale_by_view = factor.scale_by_view.expand(2, -1).clone()
    controls["multiple_scale_instances"] = _capture_rejection(
        lambda: build_psu_b0_factor_majorizer_pipeline(
            gauge=base.setup.pipeline.gauge,
            voxel_operator=base.setup.pipeline.voxel_operator,
            measurement_factor=factor,
            regularization_operator=base.setup.pipeline.regularization_operator,
            eta=float(config["fixture"]["eta"]),
        )
    )

    tiny = 1e-12
    tiny_system = SignedFactorSystem(
        np.eye(2),
        (np.eye(2),),
        np.eye(2),
        (np.diag([1.0, tiny]),),
        np.zeros((3, 2)),
    )
    tiny_setup = build_majorizer_setup(tiny_system)
    controls["tiny_nonzero_retained"] = {
        "rejected": False,
        "retained": bool(
            tiny_setup.active_data_rows[0][1]
            and tiny_setup.active_primal[1] == 1
            and tiny_setup.omega_data_rows[0][1] == tiny
        ),
        "coupling": tiny,
    }
    controls["nonzero_zero_tolerance"] = _capture_rejection(
        lambda: build_majorizer_setup(tiny_system, zero_tolerance=1e-9)
    )

    class CrossViewWhitening(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.view_count = 2
            self.rays_per_view = 1
            self.cross_view_coupling = True
            self.register_buffer(
                "matrix",
                torch.as_tensor(
                    config["fixture"]["whitening_matrix_by_view"],
                    dtype=torch.float64,
                ),
            )
            self.register_buffer(
                "scale_by_view",
                torch.as_tensor(
                    config["fixture"]["scale_by_view"], dtype=torch.float64
                ),
            )

    controls["cross_view_metadata"] = _capture_rejection(
        lambda: ExactAbsoluteMeasurementFactor(
            CrossViewWhitening(),
            projection_u_xyz=torch.as_tensor(
                config["fixture"]["projection_u_xyz"], dtype=torch.float64
            ),
            projection_v_xyz=torch.as_tensor(
                config["fixture"]["projection_v_xyz"], dtype=torch.float64
            ),
            ray_scale=base.setup.pipeline.voxel_operator.ray_scale,
            sample_count=base.setup.pipeline.sample_count,
            measurement_scale=float(config["fixture"]["measurement_scale"]),
        )
    )
    return controls


def evaluate_numeric_gates(
    config: dict[str, Any],
    cpu: dict[str, Any],
    mps: dict[str, Any],
    negative_controls: dict[str, Any],
) -> dict[str, bool]:
    thresholds = config["thresholds"]
    metric = cpu["metric"]
    recurrence = cpu["recurrence"]
    ledgers = cpu["ledgers"]
    return {
        "E1-01": (
            metric["embedding_minimum"] >= 0.0
            and metric["trilinear_weight_minimum"] >= 0.0
            and cpu["whitening_contract"]["whitening_block_scope"] == "view_local"
            and cpu["whitening_contract"]["independent_whitening_blocks"] is True
            and cpu["whitening_contract"]["cross_view_covariance_supported"] is False
        ),
        "E1-02": cpu["adjoint"]["maximum_relative_error"]
        <= thresholds["float64_adjoint_relative_error_max"],
        "E1-03": max(
            metric["data_dominance_violation_max"],
            metric["tv_dominance_violation_max"],
        )
        <= thresholds["float64_dominance_violation_max"],
        "E1-04": metric["maximum_sum_relative_error"]
        <= thresholds["float64_sum_relative_error_max"],
        "E1-05": metric["scaled_operator_norm_squared"]
        <= metric["eta_squared"] + thresholds["float64_svd_slack"],
        "E1-06": recurrence["maximum_objective_relative_error"]
        <= thresholds["float64_recurrence_relative_error_max"],
        "E1-07": recurrence["maximum_state_relative_error"]
        <= thresholds["float64_recurrence_relative_error_max"],
        "E1-08": (
            metric["finite_metric"]
            and metric["deleted_data_rows"] > 0
            and cpu["zero_coupling"]["direct_constant_relative_error"]
            <= thresholds["float64_recurrence_relative_error_max"]
        ),
        "E1-09": all(
            negative_controls[name]["rejected"]
            for name in (
                "stale_signed_kernel",
                "stale_scale_by_view",
                "stale_geometry_weight",
                "stale_support",
                "stale_tau",
                "stale_sigma_data",
                "stale_data_mask",
                "stale_active_primal_indices",
                "stale_pipeline_device_indices",
                "stale_voxel_spacing",
            )
        ),
        "E1-10": (
            ledgers["setup_logical"] == ledgers["setup_expected"]
            and ledgers["setup_physical"] == ledgers["setup_expected"]
            and ledgers["oracle_audit_logical_delta"]
            == ledgers["oracle_audit_expected"]
            and ledgers["oracle_audit_physical_delta"]
            == ledgers["oracle_audit_expected"]
            and ledgers["solve_logical_delta"] == ledgers["solve_expected"]
            and ledgers["solve_physical_delta"] == ledgers["solve_expected"]
            and ledgers["scorer_logical_delta"] == ledgers["scorer_expected"]
            and ledgers["scorer_physical_delta"] == ledgers["scorer_expected"]
        ),
        "E1-11": (
            config["calibration_provenance"]["truth_available_to_setup"] is False
            and config["calibration_provenance"]["morphology_available_to_setup"]
            is False
            and config["calibration_provenance"]["deployment_authorized"] is False
        ),
        "E1-12": mps["available"]
        and mps["field_relative_difference"]
        <= thresholds["mps_float32_field_relative_difference_max"],
        "E1-13": (
            negative_controls["multiple_scale_instances"]["rejected"]
            and negative_controls["tiny_nonzero_retained"]["retained"]
            and negative_controls["nonzero_zero_tolerance"]["rejected"]
            and negative_controls["cross_view_metadata"]["rejected"]
        ),
    }


def _parse_junit(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    fields = ("tests", "failures", "errors", "skipped")
    totals = {
        field: sum(int(suite.attrib.get(field, "0")) for suite in suites)
        for field in fields
    }
    totals["passed"] = (
        totals["tests"]
        - totals["failures"]
        - totals["errors"]
        - totals["skipped"]
    )
    totals["all_passed_without_skip"] = (
        totals["tests"] > 0
        and totals["failures"] == 0
        and totals["errors"] == 0
        and totals["skipped"] == 0
    )
    return totals


def _pytest_environment() -> dict[str, str]:
    for name in FORBIDDEN_ENVIRONMENT:
        if os.environ.get(name):
            raise RuntimeError(f"formal Gate A forbids inherited {name}")
    if "sitecustomize" in sys.modules or "usercustomize" in sys.modules:
        raise RuntimeError("formal Gate A forbids sitecustomize/usercustomize")
    environment = os.environ.copy()
    environment.update(THREAD_ENVIRONMENT)
    environment["PYTEST_ADDOPTS"] = ""
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONSTARTUP", None)
    environment.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
    return environment


def _collect_declared_nodes(
    config: dict[str, Any],
    output: Path,
    *,
    environment: dict[str, str],
) -> list[str]:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "--color=no",
            "--override-ini=addopts=",
            "--noconftest",
            *config["test_nodes"],
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0 or result.stderr.strip():
        raise RuntimeError(
            "Gate A test collection failed or emitted stderr: "
            f"{result.stderr.strip()}"
        )
    nodes = [
        line.strip()
        for line in result.stdout.splitlines()
        if "::" in line and not line.lstrip().startswith("<")
    ]
    if not nodes or len(nodes) != len(set(nodes)):
        raise RuntimeError("Gate A collected test nodes are empty or duplicated")
    (output / "collected_nodes.txt").write_text(
        "".join(f"{node}\n" for node in nodes),
        encoding="utf-8",
    )
    return nodes


def _run_declared_tests(
    config: dict[str, Any],
    output: Path,
) -> tuple[dict[str, Any], float]:
    junit_path = output / "pytest.xml"
    output_path = output / "pytest_output.txt"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--color=no",
        "--override-ini=addopts=",
        "--noconftest",
        f"--junitxml={junit_path}",
        *config["test_nodes"],
    ]
    environment = _pytest_environment()
    collected_nodes = _collect_declared_nodes(
        config,
        output,
        environment=environment,
    )
    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
    )
    elapsed = time.perf_counter() - started
    output_path.write_text(
        result.stdout + ("\nSTDERR\n" + result.stderr if result.stderr else ""),
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"declared Gate A tests failed with exit code {result.returncode}; "
            f"see {output_path}"
        )
    summary = _parse_junit(junit_path)
    if not summary["all_passed_without_skip"]:
        raise RuntimeError("formal Gate A forbids failed, errored, or skipped tests")
    if summary["tests"] != len(collected_nodes):
        raise RuntimeError("executed test count does not match exact collection")
    summary["declared_node_count"] = len(config["test_nodes"])
    summary["collected_node_count"] = len(collected_nodes)
    summary["collected_node_ids"] = collected_nodes
    summary["collected_node_manifest_sha256"] = canonical_json_sha256(
        collected_nodes
    )
    summary["command"] = [
        "python",
        "-m",
        "pytest",
        "-q",
        "--color=no",
        "--override-ini=addopts=",
        "--noconftest",
        "--junitxml=pytest.xml",
        *config["test_nodes"],
    ]
    return summary, elapsed


def _environment_report() -> dict[str, Any]:
    pip = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        cwd=REPOSITORY_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    packages = sorted(
        line.strip() for line in pip.stdout.splitlines() if line.strip()
    )
    python_binary = Path(sys.executable).resolve()
    module_files = {
        "torch": Path(torch.__file__).resolve(),
        "numpy": Path(np.__file__).resolve(),
    }
    return {
        "python": platform.python_version(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "cpu_count": os.cpu_count(),
        "python_executable_sha256": file_sha256(python_binary),
        "pip_freeze_entry_count": len(packages),
        "pip_freeze_sha256": canonical_json_sha256(packages),
        "module_entry_sha256": {
            name: file_sha256(path) for name, path in module_files.items()
        },
        "distribution_tree_fingerprints": {
            name: _distribution_tree_fingerprint(name)
            for name in ("torch", "numpy", "pytest")
        },
        "environment_isolation": "LOCAL_HOST_NOT_ISOLATED_FINGERPRINT_ONLY",
        "pytest_plugin_autoload_disabled": True,
        "python_user_site_disabled": True,
        "mps_fallback_disabled": True,
    }


def _distribution_tree_fingerprint(project: str) -> dict[str, Any]:
    distribution = metadata.distribution(project)
    roots: dict[str, Path] = {}
    declared = distribution.files or ()
    for raw in declared:
        token = PurePosixPath(str(raw))
        if not token.parts or token.parts[0] in {".", ".."}:
            continue
        root_name = token.parts[0]
        root = Path(distribution.locate_file(root_name)).resolve()
        if root.exists():
            roots[root_name] = root
    manifest: dict[str, str] = {}
    symlink_count = 0
    for root_name, root in sorted(roots.items()):
        candidates = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in candidates:
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            if path.is_symlink():
                symlink_count += 1
            if not path.is_file():
                continue
            relative = (
                root_name
                if path == root
                else f"{root_name}/{path.relative_to(root).as_posix()}"
            )
            manifest[relative] = file_sha256(path)
    if not manifest:
        raise RuntimeError(f"empty installed distribution tree: {project}")
    return {
        "version": distribution.version,
        "declared_file_count": len(declared),
        "scanned_file_count": len(manifest),
        "symlink_count": symlink_count,
        "tree_sha256": canonical_json_sha256(manifest),
    }


def _gate_records(
    config: dict[str, Any],
    numeric: dict[str, bool],
    tests: dict[str, Any],
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    all_tests = bool(tests["all_passed_without_skip"])
    for gate, indices in config["e1_test_mapping"].items():
        nodes = [config["test_nodes"][index] for index in indices]
        passed = bool(numeric[gate] and all_tests)
        records[gate] = {
            "status": "PASS" if passed else "FAIL",
            "numeric_contract_passed": bool(numeric[gate]),
            "declared_tests_passed_without_skip": all_tests,
            "test_nodes": nodes,
        }
    return records


def generate_attestation(
    *,
    config_path: Path,
    output: Path,
) -> Path:
    """Generate the formal report; the repository must remain clean throughout."""

    config_path = config_path.resolve()
    output = output.resolve()
    build_root = (REPOSITORY_ROOT / "build").resolve()
    if build_root not in output.parents:
        raise ValueError("formal Gate A output must live under repository build/")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("formal Gate A output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    if not worktree_is_clean():
        raise RuntimeError("formal Gate A requires a clean git worktree before execution")
    if not git_index_is_plain():
        raise RuntimeError("formal Gate A forbids skip-worktree/assume-unchanged flags")
    clean_before = True
    source_commit = _run_git("rev-parse", "HEAD")
    source_branch = _run_git("branch", "--show-current")
    config = load_gate_a_config(config_path)
    source_hashes = gate_a_source_hashes(config)
    input_payload = gate_a_input_payload(config)
    fingerprints = {
        "source_commit": source_commit,
        "config_file_sha256": file_sha256(config_path),
        "input_payload_sha256": canonical_json_sha256(input_payload),
        "test_node_manifest_sha256": canonical_json_sha256(config["test_nodes"]),
        "source_files_sha256": source_hashes,
        "source_bundle_sha256": canonical_json_sha256(source_hashes),
    }
    environment_before = _environment_report()
    started = time.perf_counter()
    cpu = collect_cpu_numeric_evidence(config)
    mps = collect_mps_parity_evidence(config)
    negative_controls = collect_negative_control_evidence(config)
    tests, test_elapsed = _run_declared_tests(config, output)
    fingerprints["collected_node_manifest_sha256"] = tests[
        "collected_node_manifest_sha256"
    ]
    fingerprints["environment_lock_sha256"] = canonical_json_sha256(
        environment_before
    )
    fingerprints["attestation_identity_sha256"] = canonical_json_sha256(
        {
            "source_commit": source_commit,
            "config_file_sha256": fingerprints["config_file_sha256"],
            "input_payload_sha256": fingerprints["input_payload_sha256"],
            "test_node_manifest_sha256": fingerprints["test_node_manifest_sha256"],
            "collected_node_manifest_sha256": fingerprints[
                "collected_node_manifest_sha256"
            ],
            "source_bundle_sha256": fingerprints["source_bundle_sha256"],
            "environment_lock_sha256": fingerprints["environment_lock_sha256"],
        }
    )
    numeric = evaluate_numeric_gates(config, cpu, mps, negative_controls)
    gates = _gate_records(config, numeric, tests)
    if not all(record["status"] == "PASS" for record in gates.values()):
        raise RuntimeError("one or more Gate A E1 records failed")
    environment_after = _environment_report()
    if environment_after != environment_before:
        raise RuntimeError("formal Gate A environment changed during execution")
    if _run_git("rev-parse", "HEAD") != source_commit:
        raise RuntimeError("formal Gate A HEAD changed during execution")
    if file_sha256(config_path) != fingerprints["config_file_sha256"]:
        raise RuntimeError("formal Gate A config changed during execution")
    if gate_a_source_hashes(config) != source_hashes:
        raise RuntimeError("formal Gate A source files changed during execution")
    if not worktree_is_clean():
        raise RuntimeError("formal Gate A changed the git worktree")
    if not git_index_is_plain():
        raise RuntimeError("formal Gate A git index flags changed during execution")
    clean_after = True
    elapsed = time.perf_counter() - started
    report = {
        "schema_version": ATTESTATION_SCHEMA,
        "status": FORMAL_STATUS,
        "evidence_scope": config["evidence_scope"],
        "scientific_claim_boundary": CLAIM_BOUNDARY,
        "gate_b_status": "NOT_RUN",
        "fresh_data_status": "NOT_RUN",
        "real_data_status": "NOT_RUN",
        "method_superiority_status": "NOT_ESTABLISHED",
        "environment_independence_status": (
            "NOT_ESTABLISHED_LOCAL_HOST_FINGERPRINT_ONLY"
        ),
        "source": {
            "commit": source_commit,
            "branch": source_branch,
            "worktree_clean_before": clean_before,
            "worktree_clean_after": clean_after,
        },
        "fingerprints": fingerprints,
        "calibration_provenance": config["calibration_provenance"],
        "environment": environment_before,
        "test_attestation": tests,
        "e1_records": gates,
        "cpu_float64": cpu,
        "mps_float32_parity": mps,
        "negative_controls": negative_controls,
        "runtime": {
            "total_wall_seconds": elapsed,
            "pytest_wall_seconds": test_elapsed,
            "timing_claim_authorized": False,
        },
    }
    report_path = output / "attestation.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    payload_marker = output / "attestation_payload.sha256"
    payload_marker.write_text(
        (
            f"{canonical_json_sha256(report)}  attestation.json.canonical-json-v1\n"
            f"{file_sha256(report_path)}  attestation.json.raw-file\n"
        ),
        encoding="ascii",
    )
    targets = (
        report_path,
        output / "pytest.xml",
        output / "pytest_output.txt",
        output / "collected_nodes.txt",
        payload_marker,
    )
    (output / "checksums.sha256").write_text(
        "".join(f"{file_sha256(path)}  {path.name}\n" for path in targets),
        encoding="ascii",
    )
    if not worktree_is_clean():
        raise RuntimeError("checksum generation changed the git worktree")
    return report_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_GATE_A_CONFIG_PATH,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report_path = generate_attestation(
        config_path=args.config,
        output=args.output,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "status": report["status"],
                "source_commit": report["source"]["commit"],
                "attestation_identity_sha256": report["fingerprints"][
                    "attestation_identity_sha256"
                ],
                "e1_passed": sum(
                    record["status"] == "PASS"
                    for record in report["e1_records"].values()
                ),
                "gate_b_status": report["gate_b_status"],
                "output": str(report_path.parent),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
