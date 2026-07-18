#!/usr/bin/env python3
"""Run the preregistered N5-D4c semantic-v2 derivative screen.

This is a synthetic explicit-matrix protocol audit, not a BOST reconstruction
benchmark.  It corrects three semantic defects found after D4c-v1 was frozen:

* finite differences call the scenario forward at ``x +/- h v``;
* branch and diagnostic states come back from those forward calls; and
* the structural identity is evaluated through three explicit paths.

No threshold is selected from this development screen.  A low dot-product
signal is reported as unresolved rather than silently converted into a pass.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from demo_t16_operator.side_weighted_adjoint_certificate import (  # noqa: E402
    evaluate_side_weighted_adjoint,
    summarize_side_weighted_probes,
)


DEFAULT_CONFIG = (
    ROOT
    / "demo_t16_operator/configs/"
    "n2_pvgr_n5_d4c_msra_semantic_v2_preregistered.json"
)
DEFAULT_OUTPUT = (
    ROOT / "demo_t16_operator/results/n2_pvgr_n5_d4c_msra_semantic_v2"
)
RESULT_SCHEMA = "n2-pvgr-n5-d4c-msra-semantic-v2-result-1.0"
EXPECTED_SCENARIOS = (
    "clean_general",
    "clean_low_bilinear_signal",
    "separate_cancellation_residual",
    "paired_cancellation_residual",
    "vjp_aligned_fault",
    "vjp_first_probe_blind_fault",
    "jvp_aligned_fault",
    "self_consistent_wrong_derivative",
    "three_path_structure_mismatch",
    "diagnostic_only_support_flip",
    "actual_piecewise_branch_crossing",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return f"external-test-input/{resolved.name}"


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ("git", *args), cwd=ROOT, check=check, capture_output=True
    )


def _assert_committed(paths: Iterable[Path]) -> str:
    commit = _git("rev-parse", "HEAD").stdout.decode().strip()
    for path in paths:
        relative = _relative(path)
        frozen = _git("show", f"{commit}:{relative}", check=False)
        if frozen.returncode != 0:
            raise ValueError(f"D4c-v2 source is not committed: {relative}")
        if hashlib.sha256(frozen.stdout).hexdigest() != _sha256(path):
            raise ValueError(f"D4c-v2 committed source drifted: {relative}")
    return commit


def _normalize(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("cannot normalize a degenerate vector")
    return array / norm


def _relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate_array = np.asarray(candidate, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    return float(
        np.linalg.norm(candidate_array - reference_array)
        / max(
            float(np.linalg.norm(candidate_array)),
            float(np.linalg.norm(reference_array)),
            1e-300,
        )
    )


def _relative_residual(residual: np.ndarray, *references: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(residual, dtype=np.float64))
        / max(
            *(float(np.linalg.norm(np.asarray(item, dtype=np.float64))) for item in references),
            1e-300,
        )
    )


def _vector_fields(prefix: str, value: np.ndarray) -> dict[str, float]:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    return {f"{prefix}_{index}": float(item) for index, item in enumerate(vector)}


def _positive_sorted_unique(config: dict[str, Any], key: str) -> list[float]:
    values = [float(value) for value in config[key]]
    if values != sorted(set(values)) or any(value <= 0.0 for value in values):
        raise ValueError(f"{key} must be sorted, unique, and positive")
    return values


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != (
        "n2-pvgr-n5-d4c-msra-semantic-v2-preregistered-1.0"
    ):
        raise ValueError("unexpected N5-D4c semantic-v2 config schema")
    if config.get("status") != (
        "preregistered_amendment_after_v1_semantic_red_team_no_bost_authorization"
    ):
        raise ValueError("N5-D4c semantic-v2 amendment status drifted")
    if config.get("device") != "cpu" or config.get("dtype") != "float64":
        raise ValueError("N5-D4c semantic-v2 must use CPU float64")
    if int(config["input_dimension"]) != 17**3:
        raise ValueError("input dimension drifted from the frozen D4b field shape")
    if int(config["output_dimension"]) != 8:
        raise ValueError("output dimension drifted from the frozen D4b detector shape")
    if int(config["trial_count"]) < 4:
        raise ValueError("at least four trials are required")
    probes = [int(value) for value in config["probe_counts"]]
    if probes != sorted(set(probes)) or probes[0] != 1:
        raise ValueError("probe_counts must be sorted, unique, and start at one")
    _positive_sorted_unique(config, "h_values")
    _positive_sorted_unique(config, "side_weighted_gamma_threshold_grid")
    _positive_sorted_unique(config, "fault_relative_magnitudes")
    _positive_sorted_unique(config, "cancellation_deltas")
    if tuple(config["scenario_order"]) != EXPECTED_SCENARIOS:
        raise ValueError("scenario order drifted")
    for key in (
        "traditional_signal_relative_threshold",
        "finite_difference_relative_threshold",
        "structure_relative_threshold",
        "diagnostic_support_threshold",
    ):
        if not math.isfinite(float(config[key])) or float(config[key]) <= 0.0:
            raise ValueError(f"{key} must be finite and positive")
    requirements = config["semantic_requirements"]
    for key in (
        "finite_difference",
        "branch_state",
        "structure",
        "gamma",
        "aggregation",
        "status",
    ):
        if not str(requirements.get(key, "")).strip():
            raise ValueError(f"missing semantic requirement: {key}")
    fd_contract = config["finite_difference_contract"]
    if fd_contract.get("formula") != "central_two_sided":
        raise ValueError("finite-difference formula drifted")
    if fd_contract.get("denominator") != "2*h":
        raise ValueError("finite-difference denominator drifted")
    if fd_contract.get("best_h_selection_is_forbidden") is not True:
        raise ValueError("best-h selection must remain forbidden")
    if config.get("status_precedence") != [
        "NONFINITE",
        "BRANCH",
        "STRUCTURE",
        "FD",
        "ADJOINT",
        "PASS_STRONG_SIGNAL_OR_LOW_SIGNAL_UNRESOLVED",
    ]:
        raise ValueError("status precedence drifted")
    if config["raw_evidence_contract"].get(
        "reported_metrics_are_not_validator_truth"
    ) is not True:
        raise ValueError("raw evidence contract must distrust reported metrics")
    contract = config["decision_contract"]
    if contract.get("threshold_selection_is_forbidden") is not True:
        raise ValueError("threshold selection must remain forbidden")
    if contract.get("all_h_values_must_be_consumed") is not True:
        raise ValueError("all h values must be consumed")
    if contract.get("all_threshold_probe_pairs_must_be_reported") is not True:
        raise ValueError("all threshold/probe pairs must be reported")
    if any(bool(value) for value in config["claim_authorizations"].values()):
        raise ValueError("semantic-v2 cannot pre-authorize a research claim")


@dataclass(frozen=True, slots=True)
class ForwardObservation:
    output: np.ndarray
    branch_state: str
    diagnostic_state: str


@dataclass(frozen=True, slots=True)
class LinearPath:
    name: str
    matrix: np.ndarray
    diagnostic_threshold: float | None = None

    def forward(self, value: np.ndarray) -> ForwardObservation:
        vector = np.asarray(value, dtype=np.float64)
        diagnostic = "not_evaluated"
        if self.diagnostic_threshold is not None:
            diagnostic = (
                "support_high"
                if float(vector[0]) >= self.diagnostic_threshold
                else "support_low"
            )
        return ForwardObservation(
            output=self.matrix @ vector,
            branch_state=f"linear:{self.name}",
            diagnostic_state=diagnostic,
        )

    def jvp(self, tangent: np.ndarray) -> np.ndarray:
        return self.matrix @ np.asarray(tangent, dtype=np.float64)

    def vjp(self, cotangent: np.ndarray) -> np.ndarray:
        return self.matrix.T @ np.asarray(cotangent, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class DifferencePath:
    name: str
    curved: LinearPath
    straight: LinearPath

    def forward(self, value: np.ndarray) -> ForwardObservation:
        curved = self.curved.forward(value)
        straight = self.straight.forward(value)
        return ForwardObservation(
            output=curved.output - straight.output,
            branch_state=f"linear_difference:{self.name}",
            diagnostic_state="not_evaluated",
        )

    def jvp(self, tangent: np.ndarray) -> np.ndarray:
        return self.curved.jvp(tangent) - self.straight.jvp(tangent)

    def vjp(self, cotangent: np.ndarray) -> np.ndarray:
        return self.curved.vjp(cotangent) - self.straight.vjp(cotangent)


@dataclass(frozen=True, slots=True)
class PiecewisePath:
    name: str
    positive: np.ndarray
    negative: np.ndarray

    def forward(self, value: np.ndarray) -> ForwardObservation:
        vector = np.asarray(value, dtype=np.float64)
        if float(vector[0]) >= 0.0:
            matrix = self.positive
            state = "positive"
        else:
            matrix = self.negative
            state = "negative"
        return ForwardObservation(
            output=matrix @ vector,
            branch_state=f"piecewise:{self.name}:{state}",
            diagnostic_state="not_evaluated",
        )


PathLike = LinearPath | DifferencePath | PiecewisePath


@dataclass(frozen=True, slots=True)
class StructurePaths:
    curved: LinearPath
    straight: LinearPath
    direct: LinearPath


@dataclass(frozen=True, slots=True)
class Variant:
    scenario: str
    expected_role: str
    trial: int
    parameter_name: str
    parameter_value: float
    base_x: np.ndarray
    cotangent: np.ndarray
    tangents: np.ndarray
    candidate_jvps: np.ndarray
    candidate_vjp: np.ndarray
    forward_path_id: str
    forward: Callable[[np.ndarray], ForwardObservation]
    structure_paths: StructurePaths | None = None
    ideal_jvps: np.ndarray | None = None
    ideal_vjp: np.ndarray | None = None


def _matrix(
    rng: np.random.Generator, output_dim: int, input_dim: int
) -> np.ndarray:
    return rng.normal(size=(output_dim, input_dim)).astype(np.float64) / math.sqrt(
        input_dim
    )


def _probe_bank(
    rng: np.random.Generator, count: int, input_dim: int
) -> np.ndarray:
    return np.stack(
        [_normalize(rng.normal(size=input_dim)) for _ in range(count)], axis=0
    )


def _rank_one_like(
    rng: np.random.Generator, reference: np.ndarray
) -> np.ndarray:
    output_direction = _normalize(rng.normal(size=reference.shape[0]))
    input_direction = _normalize(rng.normal(size=reference.shape[1]))
    return (
        np.outer(output_direction, input_direction)
        * float(np.linalg.norm(reference))
    )


def _variant_key(variant: Variant) -> dict[str, Any]:
    parameter_hex = float(variant.parameter_value).hex()
    return {
        "case_id": (
            f"trial-{variant.trial:02d}__{variant.scenario}__"
            f"{variant.parameter_name}__{parameter_hex}"
        ),
        "scenario": variant.scenario,
        "expected_role": variant.expected_role,
        "trial": int(variant.trial),
        "parameter_name": variant.parameter_name,
        "parameter_value": float(variant.parameter_value),
        "parameter_value_float_hex": parameter_hex,
        "forward_path_id": variant.forward_path_id,
    }


def _build_trial_variants(
    config: dict[str, Any], trial: int
) -> list[Variant]:
    input_dim = int(config["input_dimension"])
    output_dim = int(config["output_dimension"])
    maximum_probes = max(int(value) for value in config["probe_counts"])
    rng = np.random.default_rng(int(config["seed_base"]) + trial)

    operator = _matrix(rng, output_dim, input_dim)
    alternative = _matrix(rng, output_dim, input_dim)
    tangents = _probe_bank(rng, maximum_probes, input_dim)
    cotangent = _normalize(rng.normal(size=output_dim))
    base_x = _normalize(rng.normal(size=input_dim))
    base_path = LinearPath("base", operator)
    jvps = np.stack([base_path.jvp(value) for value in tangents])
    vjp = base_path.vjp(cotangent)
    variants: list[Variant] = []

    def add(
        scenario: str,
        role: str,
        *,
        path: PathLike = base_path,
        x: np.ndarray = base_x,
        candidate_jvps: np.ndarray = jvps,
        candidate_vjp: np.ndarray = vjp,
        parameter_name: str = "none",
        parameter_value: float = 0.0,
        structure_paths: StructurePaths | None = None,
        ideal_jvps: np.ndarray | None = None,
        ideal_vjp: np.ndarray | None = None,
    ) -> None:
        variants.append(
            Variant(
                scenario=scenario,
                expected_role=role,
                trial=trial,
                parameter_name=parameter_name,
                parameter_value=float(parameter_value),
                base_x=np.asarray(x, dtype=np.float64).copy(),
                cotangent=cotangent.copy(),
                tangents=tangents.copy(),
                candidate_jvps=np.asarray(candidate_jvps, dtype=np.float64).copy(),
                candidate_vjp=np.asarray(candidate_vjp, dtype=np.float64).copy(),
                forward_path_id=path.name,
                forward=path.forward,
                structure_paths=structure_paths,
                ideal_jvps=(
                    None
                    if ideal_jvps is None
                    else np.asarray(ideal_jvps, dtype=np.float64).copy()
                ),
                ideal_vjp=(
                    None
                    if ideal_vjp is None
                    else np.asarray(ideal_vjp, dtype=np.float64).copy()
                ),
            )
        )

    add("clean_general", "clean_derivative_strong_signal_should_pass")

    first_jvp = jvps[0]
    low_cotangent = rng.normal(size=output_dim)
    low_cotangent -= first_jvp * (
        float(low_cotangent @ first_jvp) / float(first_jvp @ first_jvp)
    )
    low_cotangent = _normalize(low_cotangent)
    variants.append(
        Variant(
            scenario="clean_low_bilinear_signal",
            expected_role="clean_but_signal_relative_gate_is_unresolved",
            trial=trial,
            parameter_name="none",
            parameter_value=0.0,
            base_x=base_x.copy(),
            cotangent=low_cotangent,
            tangents=tangents.copy(),
            candidate_jvps=jvps.copy(),
            candidate_vjp=base_path.vjp(low_cotangent),
            forward_path_id=base_path.name,
            forward=base_path.forward,
        )
    )

    for delta in [float(value) for value in config["cancellation_deltas"]]:
        curved = LinearPath("cancellation_curved", operator + 0.5 * delta * alternative)
        straight = LinearPath(
            "cancellation_straight", operator - 0.5 * delta * alternative
        )
        separate = DifferencePath("separate", curved, straight)
        paired = LinearPath("paired", delta * alternative)
        ideal_jvps = np.stack([paired.jvp(value) for value in tangents])
        ideal_vjp = paired.vjp(cotangent)
        add(
            "separate_cancellation_residual",
            "correct_formula_with_separate_arithmetic_requires_stability_report",
            path=separate,
            candidate_jvps=np.stack([separate.jvp(value) for value in tangents]),
            candidate_vjp=separate.vjp(cotangent),
            parameter_name="component_difference_scale",
            parameter_value=delta,
            ideal_jvps=ideal_jvps,
            ideal_vjp=ideal_vjp,
        )
        add(
            "paired_cancellation_residual",
            "algebraically_paired_reference_path",
            path=paired,
            candidate_jvps=ideal_jvps,
            candidate_vjp=ideal_vjp,
            parameter_name="component_difference_scale",
            parameter_value=delta,
            ideal_jvps=ideal_jvps,
            ideal_vjp=ideal_vjp,
        )

    blind = rng.normal(size=input_dim)
    blind -= tangents[0] * float(blind @ tangents[0])
    blind = _normalize(blind)
    aligned_vjp_scale = max(float(np.linalg.norm(vjp)), 1e-300)
    for magnitude in [float(value) for value in config["fault_relative_magnitudes"]]:
        add(
            "vjp_aligned_fault",
            "vjp_fault_should_be_exposed_by_adjoint_probe",
            candidate_vjp=vjp + magnitude * aligned_vjp_scale * tangents[0],
            parameter_name="relative_fault_magnitude",
            parameter_value=magnitude,
        )
        add(
            "vjp_first_probe_blind_fault",
            "first_probe_is_constructed_blind_and_more_probes_are_required",
            candidate_vjp=vjp + magnitude * aligned_vjp_scale * blind,
            parameter_name="relative_fault_magnitude",
            parameter_value=magnitude,
        )

        aligned_fault = np.outer(cotangent, tangents[0]) * float(
            np.linalg.norm(operator)
        )
        jvp_wrong = operator + magnitude * aligned_fault
        add(
            "jvp_aligned_fault",
            "jvp_fault_requires_actual_fd_and_may_also_break_adjoint",
            candidate_jvps=tangents @ jvp_wrong.T,
            parameter_name="relative_fault_magnitude",
            parameter_value=magnitude,
        )

        self_consistent_fault = _rank_one_like(rng, operator)
        self_consistent = operator + magnitude * self_consistent_fault
        add(
            "self_consistent_wrong_derivative",
            "adjoint_consistent_wrong_derivative_requires_actual_fd",
            candidate_jvps=tangents @ self_consistent.T,
            candidate_vjp=self_consistent.T @ cotangent,
            parameter_name="relative_fault_magnitude",
            parameter_value=magnitude,
        )

        structure_curved = LinearPath("structure_curved", operator + 0.5 * alternative)
        structure_straight = LinearPath(
            "structure_straight", operator - 0.5 * alternative
        )
        structure_fault = _rank_one_like(rng, alternative)
        structure_direct = LinearPath(
            "structure_direct", alternative + magnitude * structure_fault
        )
        paths = StructurePaths(
            curved=structure_curved,
            straight=structure_straight,
            direct=structure_direct,
        )
        add(
            "three_path_structure_mismatch",
            "direct_path_is_internally_consistent_but_breaks_curved_minus_straight",
            path=structure_direct,
            candidate_jvps=np.stack(
                [structure_direct.jvp(value) for value in tangents]
            ),
            candidate_vjp=structure_direct.vjp(cotangent),
            parameter_name="relative_fault_magnitude",
            parameter_value=magnitude,
            structure_paths=paths,
        )

    diagnostic_x = base_x.copy()
    diagnostic_x[0] = float(config["diagnostic_support_threshold"])
    diagnostic_path = LinearPath(
        "diagnostic_only",
        operator,
        diagnostic_threshold=float(config["diagnostic_support_threshold"]),
    )
    add(
        "diagnostic_only_support_flip",
        "diagnostic_state_changes_but_current_forward_remains_linear",
        path=diagnostic_path,
        x=diagnostic_x,
    )

    piecewise_x = np.zeros(input_dim, dtype=np.float64)
    piecewise = PiecewisePath("actual_branch", operator, alternative)
    add(
        "actual_piecewise_branch_crossing",
        "actual_forward_states_cross_and_must_fail_closed",
        path=piecewise,
        x=piecewise_x,
        candidate_jvps=jvps,
        candidate_vjp=vjp,
    )

    order = {name: index for index, name in enumerate(EXPECTED_SCENARIOS)}
    return sorted(
        variants,
        key=lambda item: (
            order[item.scenario],
            item.parameter_value,
            item.parameter_name,
        ),
    )


def _case_spec(config: dict[str, Any], variant: Variant) -> dict[str, Any]:
    structure: dict[str, Any] | None = None
    if variant.structure_paths is not None:
        paths = variant.structure_paths
        structure = {
            "curved_path_id": paths.curved.name,
            "straight_path_id": paths.straight.name,
            "direct_path_id": paths.direct.name,
            "curved_matrix_sha256": _array_sha256(paths.curved.matrix),
            "straight_matrix_sha256": _array_sha256(paths.straight.matrix),
            "direct_matrix_sha256": _array_sha256(paths.direct.matrix),
        }
    return _variant_key(variant) | {
        "seed": int(config["seed_base"]) + int(variant.trial),
        "input_dimension": int(config["input_dimension"]),
        "output_dimension": int(config["output_dimension"]),
        "base_x_sha256": _array_sha256(variant.base_x),
        "cotangent_sha256": _array_sha256(variant.cotangent),
        "tangent_sha256": [
            _array_sha256(value) for value in variant.tangents
        ],
        "candidate_jvp_sha256": [
            _array_sha256(value) for value in variant.candidate_jvps
        ],
        "candidate_vjp_sha256": _array_sha256(variant.candidate_vjp),
        "structure_paths": structure,
        "ideal_jvp_sha256": (
            None
            if variant.ideal_jvps is None
            else [_array_sha256(value) for value in variant.ideal_jvps]
        ),
        "ideal_vjp_sha256": (
            None
            if variant.ideal_vjp is None
            else _array_sha256(variant.ideal_vjp)
        ),
    }


def _structure_evidence(
    variant: Variant,
) -> tuple[list[dict[str, Any]], list[float]]:
    if variant.structure_paths is None:
        return [], [0.0] * len(variant.tangents)
    paths = variant.structure_paths
    curved_output = paths.curved.forward(variant.base_x).output
    straight_output = paths.straight.forward(variant.base_x).output
    direct_output = paths.direct.forward(variant.base_x).output
    output_residual = direct_output - (curved_output - straight_output)
    output_relative = _relative_residual(
        output_residual, direct_output, curved_output - straight_output
    )
    curved_vjp = paths.curved.vjp(variant.cotangent)
    straight_vjp = paths.straight.vjp(variant.cotangent)
    direct_vjp = paths.direct.vjp(variant.cotangent)
    vjp_residual = direct_vjp - (curved_vjp - straight_vjp)
    vjp_relative = _relative_residual(
        vjp_residual, direct_vjp, curved_vjp - straight_vjp
    )
    rows: list[dict[str, Any]] = []
    errors: list[float] = []
    for probe_index, tangent in enumerate(variant.tangents):
        curved_jvp = paths.curved.jvp(tangent)
        straight_jvp = paths.straight.jvp(tangent)
        direct_jvp = paths.direct.jvp(tangent)
        jvp_residual = direct_jvp - (curved_jvp - straight_jvp)
        jvp_relative = _relative_residual(
            jvp_residual, direct_jvp, curved_jvp - straight_jvp
        )
        maximum = max(output_relative, jvp_relative, vjp_relative)
        errors.append(maximum)
        rows.append(
            _variant_key(variant)
            | {
                "probe_index": probe_index,
                "output_relative_error": output_relative,
                "jvp_relative_error": jvp_relative,
                "vjp_relative_error": vjp_relative,
                "maximum_structure_relative_error": maximum,
                "curved_vjp_sha256": _array_sha256(curved_vjp),
                "straight_vjp_sha256": _array_sha256(straight_vjp),
                "direct_vjp_sha256": _array_sha256(direct_vjp),
                "vjp_residual_norm": float(np.linalg.norm(vjp_residual)),
                "curved_path_id": paths.curved.name,
                "straight_path_id": paths.straight.name,
                "direct_path_id": paths.direct.name,
                "curved_matrix_sha256": _array_sha256(paths.curved.matrix),
                "straight_matrix_sha256": _array_sha256(paths.straight.matrix),
                "direct_matrix_sha256": _array_sha256(paths.direct.matrix),
            }
            | _vector_fields("curved_output", curved_output)
            | _vector_fields("straight_output", straight_output)
            | _vector_fields("direct_output", direct_output)
            | _vector_fields("curved_jvp", curved_jvp)
            | _vector_fields("straight_jvp", straight_jvp)
            | _vector_fields("direct_jvp", direct_jvp)
        )
    return rows, errors


def _evaluate_variant(
    config: dict[str, Any], variant: Variant
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    probe_rows: list[dict[str, Any]] = []
    fd_rows: list[dict[str, Any]] = []
    structure_rows, structure_errors = _structure_evidence(variant)
    evidence_objects = []
    fd_by_probe: list[list[float]] = []
    branch_by_probe: list[bool] = []
    diagnostic_by_probe: list[bool] = []
    ideal_by_probe: list[float] = []
    base = variant.forward(variant.base_x)

    for probe_index, (tangent, candidate_jvp) in enumerate(
        zip(variant.tangents, variant.candidate_jvps, strict=True)
    ):
        adjoint = evaluate_side_weighted_adjoint(
            candidate_jvp,
            variant.cotangent,
            tangent,
            variant.candidate_vjp,
        )
        evidence_objects.append(adjoint)
        probe_rows.append(
            _variant_key(variant)
            | {
                "probe_index": probe_index,
                **adjoint.to_dict(),
                "tangent_sha256": _array_sha256(tangent),
                "candidate_jvp_sha256": _array_sha256(candidate_jvp),
                "candidate_vjp_sha256": _array_sha256(variant.candidate_vjp),
            }
            | _vector_fields("candidate_jvp", candidate_jvp)
        )
        probe_fd_errors: list[float] = []
        probe_branch_changed = False
        probe_diagnostic_changed = False
        for h in [float(value) for value in config["h_values"]]:
            plus_input = variant.base_x + h * tangent
            minus_input = variant.base_x - h * tangent
            plus = variant.forward(plus_input)
            minus = variant.forward(minus_input)
            estimate = (plus.output - minus.output) / (2.0 * h)
            fd_error = _relative_l2(estimate, candidate_jvp)
            branch_changed = bool(
                plus.branch_state != minus.branch_state
                or plus.branch_state != base.branch_state
                or minus.branch_state != base.branch_state
            )
            diagnostic_changed = bool(
                plus.diagnostic_state != minus.diagnostic_state
                or plus.diagnostic_state != base.diagnostic_state
                or minus.diagnostic_state != base.diagnostic_state
            )
            probe_fd_errors.append(fd_error)
            probe_branch_changed = probe_branch_changed or branch_changed
            probe_diagnostic_changed = (
                probe_diagnostic_changed or diagnostic_changed
            )
            fd_rows.append(
                _variant_key(variant)
                | {
                    "call_pair_id": (
                        f"{_variant_key(variant)['case_id']}__probe-{probe_index:02d}__"
                        f"h-{h.hex()}"
                    ),
                    "probe_index": probe_index,
                    "h": h,
                    "h_float_hex": h.hex(),
                    "base_input_sha256": _array_sha256(variant.base_x),
                    "plus_input_sha256": _array_sha256(plus_input),
                    "minus_input_sha256": _array_sha256(minus_input),
                    "base_branch_state": base.branch_state,
                    "plus_branch_state": plus.branch_state,
                    "minus_branch_state": minus.branch_state,
                    "actual_forward_branch_changed": branch_changed,
                    "base_diagnostic_state": base.diagnostic_state,
                    "plus_diagnostic_state": plus.diagnostic_state,
                    "minus_diagnostic_state": minus.diagnostic_state,
                    "diagnostic_state_changed": diagnostic_changed,
                    "fd_relative_error": fd_error,
                }
                | _vector_fields("base_output", base.output)
                | _vector_fields("plus_output", plus.output)
                | _vector_fields("minus_output", minus.output)
                | _vector_fields("candidate_jvp", candidate_jvp)
                | _vector_fields("fd_estimate", estimate)
            )
        fd_by_probe.append(probe_fd_errors)
        branch_by_probe.append(probe_branch_changed)
        diagnostic_by_probe.append(probe_diagnostic_changed)
        if variant.ideal_jvps is None:
            ideal_by_probe.append(0.0)
        else:
            ideal_by_probe.append(
                _relative_l2(candidate_jvp, variant.ideal_jvps[probe_index])
            )

    evidence_rows: list[dict[str, Any]] = []
    for probe_count in [int(value) for value in config["probe_counts"]]:
        summary = summarize_side_weighted_probes(
            evidence_objects[:probe_count]
        )
        maximum_fd = max(
            value for values in fd_by_probe[:probe_count] for value in values
        )
        maximum_structure = max(structure_errors[:probe_count], default=0.0)
        ideal_vjp_error = (
            0.0
            if variant.ideal_vjp is None
            else _relative_l2(variant.candidate_vjp, variant.ideal_vjp)
        )
        evidence_rows.append(
            _variant_key(variant)
            | summary.to_dict()
            | {
                "probe_count": probe_count,
                "traditional_signal_gate": bool(
                    summary.all_finite
                    and summary.maximum_signal_relative_defect
                    <= float(config["traditional_signal_relative_threshold"])
                ),
                "maximum_fd_relative_error": maximum_fd,
                "maximum_structure_relative_error": maximum_structure,
                "actual_forward_branch_changed": any(
                    branch_by_probe[:probe_count]
                ),
                "diagnostic_state_changed": any(
                    diagnostic_by_probe[:probe_count]
                ),
                "ideal_reference_jvp_relative_error": max(
                    ideal_by_probe[:probe_count], default=0.0
                ),
                "ideal_reference_vjp_relative_error": ideal_vjp_error,
                "ideal_reference_is_gate": False,
            }
        )
    return probe_rows, fd_rows, structure_rows, evidence_rows


def _decision_rows(
    config: dict[str, Any], evidence_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for evidence in evidence_rows:
        for threshold in config["side_weighted_gamma_threshold_grid"]:
            threshold_value = float(threshold)
            gates = {
                "finite_gate": bool(evidence["all_finite"]),
                "branch_gate": not bool(
                    evidence["actual_forward_branch_changed"]
                ),
                "structure_gate": bool(
                    float(evidence["maximum_structure_relative_error"])
                    <= float(config["structure_relative_threshold"])
                ),
                "fd_gate": bool(
                    float(evidence["maximum_fd_relative_error"])
                    <= float(config["finite_difference_relative_threshold"])
                ),
                "side_weighted_adjoint_gate": bool(
                    float(evidence["maximum_side_weighted_gamma_score"])
                    <= threshold_value
                ),
            }
            if not gates["finite_gate"]:
                status = "FAIL_NONFINITE"
                obligation = "finite_values"
            elif not gates["branch_gate"]:
                status = "FAIL_BRANCH"
                obligation = "actual_forward_branch_state"
            elif not gates["structure_gate"]:
                status = "FAIL_STRUCTURE"
                obligation = "three_path_structure"
            elif not gates["fd_gate"]:
                status = "FAIL_FD"
                obligation = "actual_central_finite_difference"
            elif not gates["side_weighted_adjoint_gate"]:
                status = "FAIL_ADJOINT"
                obligation = "side_weighted_adjoint"
            elif bool(evidence["traditional_signal_gate"]):
                status = "PASS_STRONG_SIGNAL"
                obligation = "none"
            else:
                status = "LOW_SIGNAL_UNRESOLVED"
                obligation = "traditional_signal_strength"
            rows.append(
                evidence
                | {
                    "side_weighted_gamma_threshold": threshold_value,
                    **gates,
                    "status": status,
                    "status_family": "FAIL" if status.startswith("FAIL_") else status,
                    "first_failing_obligation": obligation,
                    "threshold_selected": False,
                }
            )
    return rows


def _aggregate(
    config: dict[str, Any], decision_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    status_values = (
        "PASS_STRONG_SIGNAL",
        "LOW_SIGNAL_UNRESOLVED",
        "FAIL_NONFINITE",
        "FAIL_BRANCH",
        "FAIL_STRUCTURE",
        "FAIL_FD",
        "FAIL_ADJOINT",
    )
    keys = sorted(
        {
            (
                str(row["scenario"]),
                str(row["parameter_name"]),
                float(row["parameter_value"]),
                int(row["probe_count"]),
                float(row["side_weighted_gamma_threshold"]),
            )
            for row in decision_rows
        }
    )
    for scenario, parameter_name, parameter_value, probe_count, threshold in keys:
        members = [
            row
            for row in decision_rows
            if row["scenario"] == scenario
            and row["parameter_name"] == parameter_name
            and float(row["parameter_value"]) == parameter_value
            and int(row["probe_count"]) == probe_count
            and float(row["side_weighted_gamma_threshold"]) == threshold
        ]
        payload: dict[str, Any] = {
            "scenario": scenario,
            "parameter_name": parameter_name,
            "parameter_value": parameter_value,
            "probe_count": probe_count,
            "side_weighted_gamma_threshold": threshold,
            "count": len(members),
        }
        for status in status_values:
            count = sum(row["status"] == status for row in members)
            payload[f"{status.lower()}_count"] = count
            payload[f"{status.lower()}_rate"] = count / len(members)
        rows.append(payload)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path.name}")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty JSONL: {path.name}")
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _environment() -> dict[str, Any]:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        np.show_config()
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy_show_config": buffer.getvalue(),
    }


def _headline(
    config: dict[str, Any],
    evidence_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    first_probe = [row for row in evidence_rows if int(row["probe_count"]) == 1]
    low = [row for row in first_probe if row["scenario"] == "clean_low_bilinear_signal"]
    diagnostic = [
        row for row in first_probe if row["scenario"] == "diagnostic_only_support_flip"
    ]
    branch = [
        row
        for row in first_probe
        if row["scenario"] == "actual_piecewise_branch_crossing"
    ]
    structures = [
        row
        for row in first_probe
        if row["scenario"] == "three_path_structure_mismatch"
    ]
    by_fault: list[dict[str, Any]] = []
    for magnitude in [float(value) for value in config["fault_relative_magnitudes"]]:
        members = [
            row
            for row in structures
            if float(row["parameter_value"]) == magnitude
        ]
        by_fault.append(
            {
                "relative_fault_magnitude": magnitude,
                "count": len(members),
                "structure_gate_reject_count": sum(
                    float(row["maximum_structure_relative_error"])
                    > float(config["structure_relative_threshold"])
                    for row in members
                ),
                "fd_gate_reject_count": sum(
                    float(row["maximum_fd_relative_error"])
                    > float(config["finite_difference_relative_threshold"])
                    for row in members
                ),
            }
        )
    statuses = sorted({str(row["status"]) for row in decision_rows})
    return {
        "low_signal_count": len(low),
        "low_signal_traditional_reject_count": sum(
            not bool(row["traditional_signal_gate"]) for row in low
        ),
        "low_signal_maximum_side_weighted_gamma_score": max(
            (float(row["maximum_side_weighted_gamma_score"]) for row in low),
            default=0.0,
        ),
        "diagnostic_only_count": len(diagnostic),
        "diagnostic_state_change_count": sum(
            bool(row["diagnostic_state_changed"]) for row in diagnostic
        ),
        "diagnostic_actual_branch_change_count": sum(
            bool(row["actual_forward_branch_changed"]) for row in diagnostic
        ),
        "actual_branch_count": len(branch),
        "actual_branch_change_count": sum(
            bool(row["actual_forward_branch_changed"]) for row in branch
        ),
        "actual_branch_fd_reject_count": sum(
            float(row["maximum_fd_relative_error"])
            > float(config["finite_difference_relative_threshold"])
            for row in branch
        ),
        "three_path_by_fault": by_fault,
        "reported_status_values": statuses,
    }


def _plot(
    config: dict[str, Any],
    evidence_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    plot_probe_count = max(int(value) for value in config["probe_counts"])

    low = [
        row
        for row in evidence_rows
        if row["scenario"] == "clean_low_bilinear_signal"
        and int(row["probe_count"]) == 1
    ]
    axes[0, 0].scatter(
        [row["maximum_condition_proxy"] for row in low],
        [max(float(row["maximum_signal_relative_defect"]), 1e-300) for row in low],
        color="#c44536",
        label="signal-relative defect",
    )
    axes[0, 0].scatter(
        [row["maximum_condition_proxy"] for row in low],
        [max(float(row["maximum_side_weighted_gamma_score"]), 1e-300) for row in low],
        color="#197278",
        label="side-weighted score",
    )
    axes[0, 0].axhline(
        float(config["traditional_signal_relative_threshold"]),
        color="#c44536",
        linestyle="--",
        alpha=0.7,
    )
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("dot condition proxy")
    axes[0, 0].set_ylabel("diagnostic score; scales differ")
    axes[0, 0].set_title("Low bilinear signal remains explicitly unresolved")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(alpha=0.2)

    target_fault = float(config["fault_relative_magnitudes"][-2])
    thresholds = [float(value) for value in config["side_weighted_gamma_threshold_grid"]]
    probes = [int(value) for value in config["probe_counts"]]
    matrix = np.zeros((len(thresholds), len(probes)), dtype=np.float64)
    for i, threshold in enumerate(thresholds):
        for j, probe_count in enumerate(probes):
            members = [
                row
                for row in decision_rows
                if row["scenario"] == "vjp_first_probe_blind_fault"
                and float(row["parameter_value"]) == target_fault
                and int(row["probe_count"]) == probe_count
                and float(row["side_weighted_gamma_threshold"]) == threshold
            ]
            matrix[i, j] = sum(row["status"] == "FAIL_ADJOINT" for row in members) / len(
                members
            )
    image = axes[0, 1].imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
    axes[0, 1].set_xticks(range(len(probes)), probes)
    axes[0, 1].set_yticks(range(len(thresholds)), [f"{value:g}" for value in thresholds])
    axes[0, 1].set_xlabel("random tangent probes")
    axes[0, 1].set_ylabel("descriptive threshold grid")
    axes[0, 1].set_title(f"Blind VJP fault rejection, magnitude={target_fault:g}")
    fig.colorbar(image, ax=axes[0, 1], fraction=0.046)

    for scenario, color, marker in (
        ("jvp_aligned_fault", "#283d3b", "o"),
        ("self_consistent_wrong_derivative", "#edddd4", "s"),
    ):
        members = [
            row
            for row in evidence_rows
            if row["scenario"] == scenario
            and int(row["probe_count"]) == plot_probe_count
        ]
        grouped: dict[float, list[float]] = {}
        for row in members:
            grouped.setdefault(float(row["parameter_value"]), []).append(
                max(float(row["maximum_fd_relative_error"]), 1e-300)
            )
        axes[1, 0].plot(
            sorted(grouped),
            [float(np.median(grouped[key])) for key in sorted(grouped)],
            marker=marker,
            color=color,
            label=scenario.replace("_", " "),
        )
    axes[1, 0].axhline(
        float(config["finite_difference_relative_threshold"]),
        color="#c44536",
        linestyle="--",
        label="frozen FD gate",
    )
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("relative fault magnitude")
    axes[1, 0].set_ylabel("maximum actual FD relative error")
    axes[1, 0].set_title("Actual forward calls expose wrong JVPs")
    axes[1, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].grid(alpha=0.2)

    structure = [
        row
        for row in evidence_rows
        if row["scenario"] == "three_path_structure_mismatch"
        and int(row["probe_count"]) == plot_probe_count
    ]
    grouped_structure: dict[float, list[float]] = {}
    grouped_fd: dict[float, list[float]] = {}
    for row in structure:
        magnitude = float(row["parameter_value"])
        grouped_structure.setdefault(magnitude, []).append(
            max(float(row["maximum_structure_relative_error"]), 1e-300)
        )
        grouped_fd.setdefault(magnitude, []).append(
            max(float(row["maximum_fd_relative_error"]), 1e-300)
        )
    values = sorted(grouped_structure)
    axes[1, 1].plot(
        values,
        [float(np.median(grouped_structure[value])) for value in values],
        marker="o",
        color="#772e25",
        label="three-path residual",
    )
    axes[1, 1].plot(
        values,
        [float(np.median(grouped_fd[value])) for value in values],
        marker="s",
        color="#197278",
        label="direct-path actual FD",
    )
    axes[1, 1].axhline(
        float(config["structure_relative_threshold"]),
        color="#c44536",
        linestyle="--",
        label="frozen structure gate",
    )
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("relative structure mismatch")
    axes[1, 1].set_ylabel("relative error")
    axes[1, 1].set_title("Direct FD can pass while path identity fails")
    axes[1, 1].legend(frameon=False, fontsize=8)
    axes[1, 1].grid(alpha=0.2)

    fig.suptitle(
        "N5-D4c semantic v2: actual-forward derivative obligations",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _summary(result: dict[str, Any]) -> str:
    headline = result["headline_diagnostics"]
    return "\n".join(
        (
            "# N5-D4c semantic-v2 development screen",
            "",
            f"- Decision: `{result['machine_decision']}`",
            f"- Explicit-matrix synthetic trials: `{result['counts']['trial_count']}`",
            f"- Actual forward FD calls: `{result['counts']['forward_plus_minus_call_count']}` plus/minus pairs",
            f"- FD step sizes consumed: `{', '.join(str(value) for value in result['h_values_consumed'])}`",
            f"- Correct low-signal cases unresolved by the traditional signal gate: `{headline['low_signal_traditional_reject_count']}/{headline['low_signal_count']}`",
            f"- Diagnostic-only state changes with actual branch changes: `{headline['diagnostic_state_change_count']}/{headline['diagnostic_actual_branch_change_count']}` (the second number must remain zero)",
            f"- Actual piecewise crossings observed: `{headline['actual_branch_change_count']}/{headline['actual_branch_count']}`",
            "- D4c-v1 remains preserved as integrity-valid but semantic-NO-GO negative evidence.",
            "- No threshold was selected. This artifact does not authorize BOST, reconstruction, neural-operator, real-data, generalization, superiority, or paper claims.",
            "",
        )
    )


def run(
    config_path: Path,
    output: Path,
    *,
    require_committed_source: bool = True,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    output = output.resolve()
    config = _read_json(config_path)
    _validate_config(config)
    expected_output = _resolve(config["output"])
    if output != expected_output and require_committed_source:
        raise ValueError("formal semantic-v2 output path drifted")
    if output.exists() or os.path.lexists(output):
        raise FileExistsError(f"refusing to replace semantic-v2 output: {output}")
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists() or os.path.lexists(temporary):
        raise FileExistsError(f"semantic-v2 temporary path exists: {temporary}")

    source_paths = [
        config_path,
        Path(__file__).resolve(),
        ROOT / "demo_t16_operator/side_weighted_adjoint_certificate.py",
        ROOT / "demo_t16_operator/test_side_weighted_adjoint_certificate.py",
        ROOT / "demo_t16_operator/test_run_n2_pvgr_n5_d4c_msra_semantic_v2.py",
    ]
    protocol_commit = (
        _assert_committed(source_paths)
        if require_committed_source
        else "TEST_UNCOMMITTED_SOURCE_ALLOWED"
    )
    source_hashes_before = {_relative(path): _sha256(path) for path in source_paths}

    probe_rows: list[dict[str, Any]] = []
    fd_rows: list[dict[str, Any]] = []
    structure_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    case_specs: list[dict[str, Any]] = []
    variant_count = 0
    for trial in range(int(config["trial_count"])):
        variants = _build_trial_variants(config, trial)
        variant_count += len(variants)
        for variant in variants:
            case_specs.append(_case_spec(config, variant))
            probes, fd, structure, evidence = _evaluate_variant(config, variant)
            probe_rows.extend(probes)
            fd_rows.extend(fd)
            structure_rows.extend(structure)
            evidence_rows.extend(evidence)
    decision_rows = _decision_rows(config, evidence_rows)
    scenario_summary = _aggregate(config, decision_rows)

    expected_h = sorted(float(value) for value in config["h_values"])
    consumed_h = sorted({float(row["h"]) for row in fd_rows})
    if consumed_h != expected_h:
        raise RuntimeError("not every preregistered h value was consumed")
    if {str(row["scenario"]) for row in evidence_rows} != set(EXPECTED_SCENARIOS):
        raise RuntimeError("scenario coverage drifted")
    if len({str(row["case_id"]) for row in case_specs}) != len(case_specs):
        raise RuntimeError("case identifiers are not unique")
    if any("overall" in key or "accuracy" in key for row in scenario_summary for key in row):
        raise RuntimeError("pooled overall accuracy is forbidden")

    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "candidate_id": config["candidate_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "machine_decision": config["decision_contract"]["decision"],
        "threshold_selected": None,
        "threshold_selection_forbidden": True,
        "protocol_commit": protocol_commit,
        "config": _relative(config_path),
        "config_sha256": _sha256(config_path),
        "source_hashes": source_hashes_before,
        "environment": _environment(),
        "counts": {
            "trial_count": int(config["trial_count"]),
            "scenario_count": len(EXPECTED_SCENARIOS),
            "variant_count": variant_count,
            "case_spec_rows": len(case_specs),
            "probe_rows": len(probe_rows),
            "fd_rows": len(fd_rows),
            "forward_plus_minus_call_count": len(fd_rows),
            "actual_primary_forward_call_count": variant_count + 2 * len(fd_rows),
            "structure_rows": len(structure_rows),
            "evidence_rows": len(evidence_rows),
            "decision_rows": len(decision_rows),
            "scenario_summary_rows": len(scenario_summary),
        },
        "h_values_consumed": consumed_h,
        "semantic_corrections_from_v1": {
            "actual_forward_finite_difference": True,
            "forward_returned_branch_state": True,
            "diagnostic_state_separated_from_branch_state": True,
            "independent_curved_straight_direct_paths": True,
            "side_specific_gamma_weights": True,
            "low_signal_is_unresolved_not_pass": True,
            "pooled_overall_accuracy_removed": True,
        },
        "headline_diagnostics": _headline(config, evidence_rows, decision_rows),
        "scenario_summary": scenario_summary,
        "claim_authorizations": config["claim_authorizations"],
        "limitations": [
            "All operators are synthetic explicit float64 matrices, not the OERF ray renderer.",
            "The protocol tests certificate semantics, not reconstruction quality or speed.",
            "The threshold grid is descriptive and no value is selected here.",
            "Actual BOST branch states and three-path contracts remain to be wired after a real renderer/data contract is available.",
        ],
    }

    temporary.mkdir(parents=True)
    _write_jsonl(temporary / "case_specs.jsonl", case_specs)
    _write_csv(temporary / "probe_rows.csv", probe_rows)
    _write_csv(temporary / "fd_rows.csv", fd_rows)
    _write_csv(temporary / "structure_rows.csv", structure_rows)
    _write_csv(temporary / "evidence_rows.csv", evidence_rows)
    _write_csv(temporary / "decision_rows.csv", decision_rows)
    _write_csv(temporary / "scenario_summary.csv", scenario_summary)
    _write_json(temporary / "result.json", result)
    (temporary / "summary.md").write_text(_summary(result), encoding="utf-8")
    _plot(config, evidence_rows, decision_rows, temporary / "semantic_v2.png")

    source_hashes_after = {_relative(path): _sha256(path) for path in source_paths}
    if source_hashes_after != source_hashes_before:
        raise RuntimeError("semantic-v2 source changed during execution")
    artifact_names = (
        "case_specs.jsonl",
        "probe_rows.csv",
        "fd_rows.csv",
        "structure_rows.csv",
        "evidence_rows.csv",
        "decision_rows.csv",
        "scenario_summary.csv",
        "result.json",
        "summary.md",
        "semantic_v2.png",
    )
    manifest = {
        "schema": "n2-pvgr-n5-d4c-msra-semantic-v2-manifest-1.0",
        "protocol_commit": protocol_commit,
        "source_hashes": source_hashes_before,
        "artifact_hashes": {
            name: _sha256(temporary / name) for name in artifact_names
        },
    }
    _write_json(temporary / "manifest.json", manifest)
    temporary.rename(output)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-uncommitted-source",
        action="store_true",
        help="test-only escape hatch; formal runs must use committed source",
    )
    args = parser.parse_args()
    result = run(
        args.config,
        args.output,
        require_committed_source=not args.allow_uncommitted_source,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
