#!/usr/bin/env python3
"""Independently validate the frozen N5-D4c semantic-v2 evidence bundle.

The validator deliberately does not import the experiment runner or its
side-weighted certificate helper.  It reconstructs the deterministic random
stream, all explicit operators, every case, every derivative candidate, and
every forward call from the committed protocol.  Reported metrics are treated
as claims to check, never as validator inputs.

This remains a synthetic explicit-matrix protocol audit.  A valid report does
not authorize BOST, reconstruction, real-data, generalization, superiority, or
paper claims.
"""

from __future__ import annotations

import argparse
import csv
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT
    / "demo_t16_operator/configs/"
    "n2_pvgr_n5_d4c_msra_semantic_v2_preregistered.json"
)
DEFAULT_RESULT = (
    ROOT / "demo_t16_operator/results/n2_pvgr_n5_d4c_msra_semantic_v2"
)
DEFAULT_REPORT_NAME = "validation_report.json"
REPORT_SCHEMA = "n2-pvgr-n5-d4c-msra-semantic-v2-validation-1.0"
RESULT_SCHEMA = "n2-pvgr-n5-d4c-msra-semantic-v2-result-1.0"
MANIFEST_SCHEMA = "n2-pvgr-n5-d4c-msra-semantic-v2-manifest-1.0"
CONFIG_SCHEMA = "n2-pvgr-n5-d4c-msra-semantic-v2-preregistered-1.0"

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

STATUS_VALUES = (
    "PASS_STRONG_SIGNAL",
    "LOW_SIGNAL_UNRESOLVED",
    "FAIL_NONFINITE",
    "FAIL_BRANCH",
    "FAIL_STRUCTURE",
    "FAIL_FD",
    "FAIL_ADJOINT",
)

ARTIFACT_NAMES = (
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

IDENTITY_FIELDS = [
    "case_id",
    "scenario",
    "expected_role",
    "trial",
    "parameter_name",
    "parameter_value",
    "parameter_value_float_hex",
    "forward_path_id",
]

ADJOINT_FIELDS = [
    "input_dimension",
    "output_dimension",
    "lhs",
    "rhs",
    "absolute_defect",
    "signal_relative_defect",
    "left_action_scale",
    "right_action_scale",
    "maximum_action_scale",
    "normwise_defect",
    "dot_condition_proxy",
    "output_gamma",
    "input_gamma",
    "side_weighted_gamma_scale",
    "side_weighted_gamma_score",
    "contraction_roundoff_envelope",
    "contraction_envelope_ratio",
    "finite",
]

PROBE_HEADER = (
    IDENTITY_FIELDS
    + ["probe_index"]
    + ADJOINT_FIELDS
    + ["tangent_sha256", "candidate_jvp_sha256", "candidate_vjp_sha256"]
    + [f"candidate_jvp_{index}" for index in range(8)]
)

FD_HEADER = (
    IDENTITY_FIELDS
    + [
        "call_pair_id",
        "probe_index",
        "h",
        "h_float_hex",
        "base_input_sha256",
        "plus_input_sha256",
        "minus_input_sha256",
        "base_branch_state",
        "plus_branch_state",
        "minus_branch_state",
        "actual_forward_branch_changed",
        "base_diagnostic_state",
        "plus_diagnostic_state",
        "minus_diagnostic_state",
        "diagnostic_state_changed",
        "fd_relative_error",
    ]
    + [f"{prefix}_{index}" for prefix in ("base_output", "plus_output", "minus_output", "candidate_jvp", "fd_estimate") for index in range(8)]
)

STRUCTURE_HEADER = (
    IDENTITY_FIELDS
    + [
        "probe_index",
        "output_relative_error",
        "jvp_relative_error",
        "vjp_relative_error",
        "maximum_structure_relative_error",
        "curved_vjp_sha256",
        "straight_vjp_sha256",
        "direct_vjp_sha256",
        "vjp_residual_norm",
        "curved_path_id",
        "straight_path_id",
        "direct_path_id",
        "curved_matrix_sha256",
        "straight_matrix_sha256",
        "direct_matrix_sha256",
    ]
    + [f"{prefix}_{index}" for prefix in ("curved_output", "straight_output", "direct_output", "curved_jvp", "straight_jvp", "direct_jvp") for index in range(8)]
)

EVIDENCE_HEADER = IDENTITY_FIELDS + [
    "probe_count",
    "all_finite",
    "maximum_signal_relative_defect",
    "maximum_side_weighted_gamma_score",
    "maximum_contraction_envelope_ratio",
    "minimum_signal",
    "maximum_condition_proxy",
    "traditional_signal_gate",
    "maximum_fd_relative_error",
    "maximum_structure_relative_error",
    "actual_forward_branch_changed",
    "diagnostic_state_changed",
    "ideal_reference_jvp_relative_error",
    "ideal_reference_vjp_relative_error",
    "ideal_reference_is_gate",
]

DECISION_HEADER = EVIDENCE_HEADER + [
    "side_weighted_gamma_threshold",
    "finite_gate",
    "branch_gate",
    "structure_gate",
    "fd_gate",
    "side_weighted_adjoint_gate",
    "status",
    "status_family",
    "first_failing_obligation",
    "threshold_selected",
]

SUMMARY_HEADER = [
    "scenario",
    "parameter_name",
    "parameter_value",
    "probe_count",
    "side_weighted_gamma_threshold",
    "count",
] + [f"{status.lower()}_{suffix}" for status in STATUS_VALUES for suffix in ("count", "rate")]


class ValidationError(RuntimeError):
    """Raised when the evidence bundle violates a frozen obligation."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read valid JSON from {path}: {exc}") from exc


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists() or os.path.lexists(temporary):
        raise FileExistsError(f"temporary report path already exists: {temporary}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValidationError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise ValidationError(f"path must remain inside repository: {path}") from exc


def _git_blob(commit: str, relative_path: str) -> bytes:
    if Path(relative_path).is_absolute() or ".." in Path(relative_path).parts:
        raise ValidationError(f"unsafe committed source path: {relative_path}")
    completed = subprocess.run(
        ("git", "show", f"{commit}:{relative_path}"),
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode(errors="replace").strip()
        raise ValidationError(
            f"source is not present at protocol commit: {relative_path}: {message}"
        )
    return completed.stdout


def _normalize(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValidationError("deterministic RNG produced a degenerate vector")
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
            *(
                float(np.linalg.norm(np.asarray(item, dtype=np.float64)))
                for item in references
            ),
            1e-300,
        )
    )


def _vector_fields(prefix: str, value: np.ndarray) -> dict[str, float]:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    return {f"{prefix}_{index}": float(item) for index, item in enumerate(vector)}


def _parse_bool(value: str, context: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValidationError(f"{context}: expected canonical True/False, got {value!r}")


def _parse_float(value: str, context: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValidationError(f"{context}: invalid float {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValidationError(f"{context}: non-finite float {value!r}")
    return parsed


def _assert_float_close(actual: float, expected: float, context: str) -> None:
    if not math.isfinite(actual) or not math.isfinite(expected):
        raise ValidationError(f"{context}: non-finite comparison")
    if not math.isclose(actual, expected, rel_tol=5e-13, abs_tol=5e-15):
        raise ValidationError(
            f"{context}: reported {actual!r}, independently rebuilt {expected!r}"
        )


def _compare_csv_row(
    actual: Mapping[str, str], expected: Mapping[str, object], context: str
) -> None:
    if list(actual) != list(expected):
        raise ValidationError(
            f"{context}: field order/set drifted; actual={list(actual)!r}, "
            f"expected={list(expected)!r}"
        )
    for field, expected_value in expected.items():
        actual_value = actual[field]
        field_context = f"{context}.{field}"
        if isinstance(expected_value, (bool, np.bool_)):
            if _parse_bool(actual_value, field_context) is not bool(expected_value):
                raise ValidationError(
                    f"{field_context}: reported {actual_value}, "
                    f"independently rebuilt {bool(expected_value)}"
                )
        elif isinstance(expected_value, (int, np.integer)):
            try:
                parsed_int = int(actual_value)
            except ValueError as exc:
                raise ValidationError(
                    f"{field_context}: invalid integer {actual_value!r}"
                ) from exc
            if parsed_int != int(expected_value) or actual_value != str(parsed_int):
                raise ValidationError(
                    f"{field_context}: reported {actual_value!r}, "
                    f"independently rebuilt {int(expected_value)!r}"
                )
        elif isinstance(expected_value, (float, np.floating)):
            _assert_float_close(
                _parse_float(actual_value, field_context),
                float(expected_value),
                field_context,
            )
        elif actual_value != str(expected_value):
            raise ValidationError(
                f"{field_context}: reported {actual_value!r}, "
                f"independently rebuilt {expected_value!r}"
            )


def _compare_json(actual: object, expected: object, context: str) -> None:
    if expected is None:
        if actual is not None:
            raise ValidationError(f"{context}: expected null, got {actual!r}")
        return
    if isinstance(expected, bool):
        if not isinstance(actual, bool) or actual is not expected:
            raise ValidationError(f"{context}: expected {expected}, got {actual!r}")
        return
    if isinstance(expected, int):
        if isinstance(actual, bool) or not isinstance(actual, int) or actual != expected:
            raise ValidationError(f"{context}: expected {expected}, got {actual!r}")
        return
    if isinstance(expected, float):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            raise ValidationError(f"{context}: expected float, got {actual!r}")
        _assert_float_close(float(actual), expected, context)
        return
    if isinstance(expected, str):
        if actual != expected:
            raise ValidationError(f"{context}: expected {expected!r}, got {actual!r}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValidationError(
                f"{context}: list shape drifted, expected {len(expected)}, "
                f"got {len(actual) if isinstance(actual, list) else type(actual).__name__}"
            )
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected, strict=True)
        ):
            _compare_json(actual_item, expected_item, f"{context}[{index}]")
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValidationError(
                f"{context}: object keys drifted; actual={sorted(actual) if isinstance(actual, dict) else type(actual).__name__}, "
                f"expected={sorted(expected)}"
            )
        for key, expected_value in expected.items():
            _compare_json(actual[key], expected_value, f"{context}.{key}")
        return
    raise TypeError(f"unsupported expected JSON value at {context}: {type(expected)}")


class CsvCursor:
    """Strict, ordered CSV cursor with exact header and row-count checks."""

    def __init__(self, path: Path, expected_header: Sequence[str]) -> None:
        self.path = path
        self.expected_header = list(expected_header)
        self.handle: Any = None
        self.reader: csv.DictReader[str] | None = None
        self.rows_seen = 0

    def __enter__(self) -> CsvCursor:
        self.handle = self.path.open(newline="", encoding="utf-8")
        self.reader = csv.DictReader(self.handle)
        if self.reader.fieldnames != self.expected_header:
            raise ValidationError(
                f"{self.path.name}: header drifted; actual={self.reader.fieldnames!r}, "
                f"expected={self.expected_header!r}"
            )
        return self

    def __exit__(self, *_args: object) -> None:
        if self.handle is not None:
            self.handle.close()

    def check(self, expected: Mapping[str, object], label: str) -> None:
        assert self.reader is not None
        try:
            actual = next(self.reader)
        except StopIteration as exc:
            raise ValidationError(
                f"{self.path.name}: ended before expected row {self.rows_seen + 1} ({label})"
            ) from exc
        self.rows_seen += 1
        _compare_csv_row(
            actual,
            expected,
            f"{self.path.name}[{self.rows_seen}]/{label}",
        )

    def finish(self, expected_count: int) -> None:
        assert self.reader is not None
        try:
            extra = next(self.reader)
        except StopIteration:
            extra = None
        if extra is not None:
            raise ValidationError(
                f"{self.path.name}: contains extra row after {self.rows_seen} rows"
            )
        if self.rows_seen != expected_count:
            raise ValidationError(
                f"{self.path.name}: checked {self.rows_seen}, expected {expected_count}"
            )


class JsonlCursor:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: Any = None
        self.rows_seen = 0

    def __enter__(self) -> JsonlCursor:
        self.handle = self.path.open(encoding="utf-8")
        return self

    def __exit__(self, *_args: object) -> None:
        if self.handle is not None:
            self.handle.close()

    def check(self, expected: Mapping[str, object], label: str) -> None:
        line = self.handle.readline()
        if not line:
            raise ValidationError(
                f"{self.path.name}: ended before expected row {self.rows_seen + 1} ({label})"
            )
        self.rows_seen += 1
        try:
            actual = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValidationError(
                f"{self.path.name}[{self.rows_seen}]: invalid JSON"
            ) from exc
        _compare_json(
            actual,
            dict(expected),
            f"{self.path.name}[{self.rows_seen}]/{label}",
        )

    def finish(self, expected_count: int) -> None:
        if self.handle.readline():
            raise ValidationError(
                f"{self.path.name}: contains extra row after {self.rows_seen} rows"
            )
        if self.rows_seen != expected_count:
            raise ValidationError(
                f"{self.path.name}: checked {self.rows_seen}, expected {expected_count}"
            )


@dataclass(frozen=True, slots=True)
class Observation:
    output: np.ndarray
    branch_state: str
    diagnostic_state: str


@dataclass(frozen=True, slots=True)
class LinearForward:
    name: str
    matrix: np.ndarray
    diagnostic_threshold: float | None = None

    def call(self, value: np.ndarray) -> Observation:
        vector = np.asarray(value, dtype=np.float64)
        diagnostic = "not_evaluated"
        if self.diagnostic_threshold is not None:
            diagnostic = (
                "support_high"
                if float(vector[0]) >= self.diagnostic_threshold
                else "support_low"
            )
        return Observation(
            output=self.matrix @ vector,
            branch_state=f"linear:{self.name}",
            diagnostic_state=diagnostic,
        )

    def jvp(self, tangent: np.ndarray) -> np.ndarray:
        return self.matrix @ np.asarray(tangent, dtype=np.float64)

    def vjp(self, cotangent: np.ndarray) -> np.ndarray:
        return self.matrix.T @ np.asarray(cotangent, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class DifferenceForward:
    name: str
    curved: LinearForward
    straight: LinearForward

    def call(self, value: np.ndarray) -> Observation:
        curved = self.curved.call(value)
        straight = self.straight.call(value)
        return Observation(
            output=curved.output - straight.output,
            branch_state=f"linear_difference:{self.name}",
            diagnostic_state="not_evaluated",
        )

    def jvp(self, tangent: np.ndarray) -> np.ndarray:
        return self.curved.jvp(tangent) - self.straight.jvp(tangent)

    def vjp(self, cotangent: np.ndarray) -> np.ndarray:
        return self.curved.vjp(cotangent) - self.straight.vjp(cotangent)


@dataclass(frozen=True, slots=True)
class PiecewiseForward:
    name: str
    positive: np.ndarray
    negative: np.ndarray

    def call(self, value: np.ndarray) -> Observation:
        vector = np.asarray(value, dtype=np.float64)
        if float(vector[0]) >= 0.0:
            matrix = self.positive
            state = "positive"
        else:
            matrix = self.negative
            state = "negative"
        return Observation(
            output=matrix @ vector,
            branch_state=f"piecewise:{self.name}:{state}",
            diagnostic_state="not_evaluated",
        )


Forward = LinearForward | DifferenceForward | PiecewiseForward


@dataclass(frozen=True, slots=True)
class StructureTriplet:
    curved: LinearForward
    straight: LinearForward
    direct: LinearForward


@dataclass(frozen=True, slots=True)
class Case:
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
    forward: Forward
    structure: StructureTriplet | None = None
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
    return np.outer(output_direction, input_direction) * float(
        np.linalg.norm(reference)
    )


def _case_identity(case: Case) -> dict[str, object]:
    parameter_hex = float(case.parameter_value).hex()
    return {
        "case_id": (
            f"trial-{case.trial:02d}__{case.scenario}__"
            f"{case.parameter_name}__{parameter_hex}"
        ),
        "scenario": case.scenario,
        "expected_role": case.expected_role,
        "trial": int(case.trial),
        "parameter_name": case.parameter_name,
        "parameter_value": float(case.parameter_value),
        "parameter_value_float_hex": parameter_hex,
        "forward_path_id": case.forward.name,
    }


def _build_trial_cases(config: Mapping[str, object], trial: int) -> list[Case]:
    input_dim = int(config["input_dimension"])
    output_dim = int(config["output_dimension"])
    probe_count = max(int(value) for value in config["probe_counts"])
    rng = np.random.default_rng(int(config["seed_base"]) + trial)

    operator = _matrix(rng, output_dim, input_dim)
    alternative = _matrix(rng, output_dim, input_dim)
    tangents = _probe_bank(rng, probe_count, input_dim)
    cotangent = _normalize(rng.normal(size=output_dim))
    base_x = _normalize(rng.normal(size=input_dim))
    base = LinearForward("base", operator)
    base_jvps = np.stack([base.jvp(tangent) for tangent in tangents])
    base_vjp = base.vjp(cotangent)
    cases: list[Case] = []

    def add(
        scenario: str,
        role: str,
        *,
        forward: Forward = base,
        x: np.ndarray = base_x,
        candidate_jvps: np.ndarray = base_jvps,
        candidate_vjp: np.ndarray = base_vjp,
        parameter_name: str = "none",
        parameter_value: float = 0.0,
        structure: StructureTriplet | None = None,
        ideal_jvps: np.ndarray | None = None,
        ideal_vjp: np.ndarray | None = None,
    ) -> None:
        cases.append(
            Case(
                scenario=scenario,
                expected_role=role,
                trial=trial,
                parameter_name=parameter_name,
                parameter_value=float(parameter_value),
                base_x=np.asarray(x, dtype=np.float64),
                cotangent=cotangent,
                tangents=tangents,
                candidate_jvps=np.asarray(candidate_jvps, dtype=np.float64),
                candidate_vjp=np.asarray(candidate_vjp, dtype=np.float64),
                forward=forward,
                structure=structure,
                ideal_jvps=ideal_jvps,
                ideal_vjp=ideal_vjp,
            )
        )

    add("clean_general", "clean_derivative_strong_signal_should_pass")

    first_jvp = base_jvps[0]
    low_cotangent = rng.normal(size=output_dim)
    low_cotangent -= first_jvp * (
        float(low_cotangent @ first_jvp) / float(first_jvp @ first_jvp)
    )
    low_cotangent = _normalize(low_cotangent)
    cases.append(
        Case(
            scenario="clean_low_bilinear_signal",
            expected_role="clean_but_signal_relative_gate_is_unresolved",
            trial=trial,
            parameter_name="none",
            parameter_value=0.0,
            base_x=base_x,
            cotangent=low_cotangent,
            tangents=tangents,
            candidate_jvps=base_jvps,
            candidate_vjp=base.vjp(low_cotangent),
            forward=base,
        )
    )

    for raw_delta in config["cancellation_deltas"]:
        delta = float(raw_delta)
        curved = LinearForward(
            "cancellation_curved", operator + 0.5 * delta * alternative
        )
        straight = LinearForward(
            "cancellation_straight", operator - 0.5 * delta * alternative
        )
        separate = DifferenceForward("separate", curved, straight)
        paired = LinearForward("paired", delta * alternative)
        ideal_jvps = np.stack([paired.jvp(tangent) for tangent in tangents])
        ideal_vjp = paired.vjp(cotangent)
        add(
            "separate_cancellation_residual",
            "correct_formula_with_separate_arithmetic_requires_stability_report",
            forward=separate,
            candidate_jvps=np.stack(
                [separate.jvp(tangent) for tangent in tangents]
            ),
            candidate_vjp=separate.vjp(cotangent),
            parameter_name="component_difference_scale",
            parameter_value=delta,
            ideal_jvps=ideal_jvps,
            ideal_vjp=ideal_vjp,
        )
        add(
            "paired_cancellation_residual",
            "algebraically_paired_reference_path",
            forward=paired,
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
    aligned_vjp_scale = max(float(np.linalg.norm(base_vjp)), 1e-300)
    for raw_magnitude in config["fault_relative_magnitudes"]:
        magnitude = float(raw_magnitude)
        add(
            "vjp_aligned_fault",
            "vjp_fault_should_be_exposed_by_adjoint_probe",
            candidate_vjp=(
                base_vjp + magnitude * aligned_vjp_scale * tangents[0]
            ),
            parameter_name="relative_fault_magnitude",
            parameter_value=magnitude,
        )
        add(
            "vjp_first_probe_blind_fault",
            "first_probe_is_constructed_blind_and_more_probes_are_required",
            candidate_vjp=base_vjp + magnitude * aligned_vjp_scale * blind,
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

        structure_curved = LinearForward(
            "structure_curved", operator + 0.5 * alternative
        )
        structure_straight = LinearForward(
            "structure_straight", operator - 0.5 * alternative
        )
        structure_fault = _rank_one_like(rng, alternative)
        structure_direct = LinearForward(
            "structure_direct", alternative + magnitude * structure_fault
        )
        structure = StructureTriplet(
            curved=structure_curved,
            straight=structure_straight,
            direct=structure_direct,
        )
        add(
            "three_path_structure_mismatch",
            "direct_path_is_internally_consistent_but_breaks_curved_minus_straight",
            forward=structure_direct,
            candidate_jvps=np.stack(
                [structure_direct.jvp(tangent) for tangent in tangents]
            ),
            candidate_vjp=structure_direct.vjp(cotangent),
            parameter_name="relative_fault_magnitude",
            parameter_value=magnitude,
            structure=structure,
        )

    diagnostic_x = base_x.copy()
    diagnostic_x[0] = float(config["diagnostic_support_threshold"])
    diagnostic = LinearForward(
        "diagnostic_only",
        operator,
        diagnostic_threshold=float(config["diagnostic_support_threshold"]),
    )
    add(
        "diagnostic_only_support_flip",
        "diagnostic_state_changes_but_current_forward_remains_linear",
        forward=diagnostic,
        x=diagnostic_x,
    )

    piecewise = PiecewiseForward("actual_branch", operator, alternative)
    add(
        "actual_piecewise_branch_crossing",
        "actual_forward_states_cross_and_must_fail_closed",
        forward=piecewise,
        x=np.zeros(input_dim, dtype=np.float64),
        candidate_jvps=base_jvps,
        candidate_vjp=base_vjp,
    )

    scenario_order = {name: index for index, name in enumerate(EXPECTED_SCENARIOS)}
    return sorted(
        cases,
        key=lambda case: (
            scenario_order[case.scenario],
            case.parameter_value,
            case.parameter_name,
        ),
    )


def _case_spec(config: Mapping[str, object], case: Case) -> dict[str, object]:
    structure: dict[str, object] | None = None
    if case.structure is not None:
        structure = {
            "curved_path_id": case.structure.curved.name,
            "straight_path_id": case.structure.straight.name,
            "direct_path_id": case.structure.direct.name,
            "curved_matrix_sha256": _array_sha256(case.structure.curved.matrix),
            "straight_matrix_sha256": _array_sha256(case.structure.straight.matrix),
            "direct_matrix_sha256": _array_sha256(case.structure.direct.matrix),
        }
    return _case_identity(case) | {
        "seed": int(config["seed_base"]) + case.trial,
        "input_dimension": int(config["input_dimension"]),
        "output_dimension": int(config["output_dimension"]),
        "base_x_sha256": _array_sha256(case.base_x),
        "cotangent_sha256": _array_sha256(case.cotangent),
        "tangent_sha256": [_array_sha256(value) for value in case.tangents],
        "candidate_jvp_sha256": [
            _array_sha256(value) for value in case.candidate_jvps
        ],
        "candidate_vjp_sha256": _array_sha256(case.candidate_vjp),
        "structure_paths": structure,
        "ideal_jvp_sha256": (
            None
            if case.ideal_jvps is None
            else [_array_sha256(value) for value in case.ideal_jvps]
        ),
        "ideal_vjp_sha256": (
            None if case.ideal_vjp is None else _array_sha256(case.ideal_vjp)
        ),
    }


def _gamma_n(term_count: int) -> float:
    unit_roundoff = float(np.finfo(np.float64).eps / 2.0)
    product = term_count * unit_roundoff
    if term_count < 1 or product >= 1.0:
        raise ValidationError(f"invalid gamma_n term count: {term_count}")
    return product / (1.0 - product)


def _adjoint_evidence(
    jvp: np.ndarray,
    cotangent: np.ndarray,
    tangent: np.ndarray,
    vjp: np.ndarray,
) -> dict[str, object]:
    jvp_array = np.ascontiguousarray(np.asarray(jvp, dtype=np.float64).reshape(-1))
    cotangent_array = np.ascontiguousarray(
        np.asarray(cotangent, dtype=np.float64).reshape(-1)
    )
    tangent_array = np.ascontiguousarray(
        np.asarray(tangent, dtype=np.float64).reshape(-1)
    )
    vjp_array = np.ascontiguousarray(np.asarray(vjp, dtype=np.float64).reshape(-1))
    if jvp_array.shape != cotangent_array.shape:
        raise ValidationError("independent JVP/cotangent shape mismatch")
    if tangent_array.shape != vjp_array.shape:
        raise ValidationError("independent tangent/VJP shape mismatch")
    if not all(
        bool(np.all(np.isfinite(value)))
        for value in (jvp_array, cotangent_array, tangent_array, vjp_array)
    ):
        raise ValidationError("independent derivative vectors must be finite")

    left_products = jvp_array * cotangent_array
    right_products = tangent_array * vjp_array
    lhs = float(np.dot(jvp_array, cotangent_array))
    rhs = float(np.dot(tangent_array, vjp_array))
    absolute = abs(lhs - rhs)
    signal = max(abs(lhs), abs(rhs))
    left_action = float(np.linalg.norm(jvp_array)) * float(
        np.linalg.norm(cotangent_array)
    )
    right_action = float(np.linalg.norm(tangent_array)) * float(
        np.linalg.norm(vjp_array)
    )
    maximum_action = max(left_action, right_action)
    output_gamma = _gamma_n(jvp_array.size)
    input_gamma = _gamma_n(tangent_array.size)
    side_scale = output_gamma * left_action + input_gamma * right_action
    contraction_envelope = output_gamma * float(np.sum(np.abs(left_products))) + (
        input_gamma * float(np.sum(np.abs(right_products)))
    )
    normwise = absolute / max(maximum_action, 1e-300)
    values = (
        lhs,
        rhs,
        absolute,
        left_action,
        right_action,
        normwise,
        side_scale,
        contraction_envelope,
    )
    return {
        "input_dimension": int(tangent_array.size),
        "output_dimension": int(jvp_array.size),
        "lhs": lhs,
        "rhs": rhs,
        "absolute_defect": absolute,
        "signal_relative_defect": absolute / max(signal, 1e-300),
        "left_action_scale": left_action,
        "right_action_scale": right_action,
        "maximum_action_scale": maximum_action,
        "normwise_defect": normwise,
        "dot_condition_proxy": maximum_action / max(signal, 1e-300),
        "output_gamma": output_gamma,
        "input_gamma": input_gamma,
        "side_weighted_gamma_scale": side_scale,
        "side_weighted_gamma_score": absolute / max(side_scale, 1e-300),
        "contraction_roundoff_envelope": contraction_envelope,
        "contraction_envelope_ratio": absolute
        / max(contraction_envelope, 1e-300),
        "finite": all(math.isfinite(value) for value in values),
    }


def _probe_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValidationError("probe prefix cannot be empty")
    return {
        "probe_count": len(rows),
        "all_finite": all(bool(row["finite"]) for row in rows),
        "maximum_signal_relative_defect": max(
            float(row["signal_relative_defect"]) for row in rows
        ),
        "maximum_side_weighted_gamma_score": max(
            float(row["side_weighted_gamma_score"]) for row in rows
        ),
        "maximum_contraction_envelope_ratio": max(
            float(row["contraction_envelope_ratio"]) for row in rows
        ),
        "minimum_signal": min(
            max(abs(float(row["lhs"])), abs(float(row["rhs"]))) for row in rows
        ),
        "maximum_condition_proxy": max(
            float(row["dot_condition_proxy"]) for row in rows
        ),
    }


def _structure_rows(case: Case) -> tuple[list[dict[str, object]], list[float]]:
    if case.structure is None:
        return [], [0.0] * len(case.tangents)
    paths = case.structure
    curved_output = paths.curved.call(case.base_x).output
    straight_output = paths.straight.call(case.base_x).output
    direct_output = paths.direct.call(case.base_x).output
    output_residual = direct_output - (curved_output - straight_output)
    output_relative = _relative_residual(
        output_residual, direct_output, curved_output - straight_output
    )
    curved_vjp = paths.curved.vjp(case.cotangent)
    straight_vjp = paths.straight.vjp(case.cotangent)
    direct_vjp = paths.direct.vjp(case.cotangent)
    vjp_residual = direct_vjp - (curved_vjp - straight_vjp)
    vjp_relative = _relative_residual(
        vjp_residual, direct_vjp, curved_vjp - straight_vjp
    )
    rows: list[dict[str, object]] = []
    errors: list[float] = []
    for probe_index, tangent in enumerate(case.tangents):
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
            _case_identity(case)
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


def _decision(
    config: Mapping[str, object], evidence: Mapping[str, object], threshold: float
) -> dict[str, object]:
    gates = {
        "finite_gate": bool(evidence["all_finite"]),
        "branch_gate": not bool(evidence["actual_forward_branch_changed"]),
        "structure_gate": float(evidence["maximum_structure_relative_error"])
        <= float(config["structure_relative_threshold"]),
        "fd_gate": float(evidence["maximum_fd_relative_error"])
        <= float(config["finite_difference_relative_threshold"]),
        "side_weighted_adjoint_gate": float(
            evidence["maximum_side_weighted_gamma_score"]
        )
        <= threshold,
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
    return dict(evidence) | {
        "side_weighted_gamma_threshold": threshold,
        **gates,
        "status": status,
        "status_family": "FAIL" if status.startswith("FAIL_") else status,
        "first_failing_obligation": obligation,
        "threshold_selected": False,
    }


def _validate_config(config: Mapping[str, object]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValidationError("unexpected committed config schema")
    if config.get("candidate_id") != "N2-PVGR-N5-D4C-MSRA-SEMANTIC-V2":
        raise ValidationError("candidate ID drifted")
    if config.get("device") != "cpu" or config.get("dtype") != "float64":
        raise ValidationError("protocol must remain CPU float64")
    if int(config["input_dimension"]) != 4913:
        raise ValidationError("input dimension must remain 4913")
    if int(config["output_dimension"]) != 8:
        raise ValidationError("output dimension must remain 8")
    if int(config["trial_count"]) != 24 or int(config["seed_base"]) != 71401:
        raise ValidationError("trial count or seed base drifted")
    if [int(value) for value in config["probe_counts"]] != [1, 2, 4, 8, 16]:
        raise ValidationError("probe-count grid drifted")
    if [float(value) for value in config["h_values"]] != [0.0001, 0.001, 0.01]:
        raise ValidationError("finite-difference h grid drifted")
    if [float(value) for value in config["side_weighted_gamma_threshold_grid"]] != [
        0.25,
        0.5,
        1.0,
        2.0,
        4.0,
        8.0,
        16.0,
        32.0,
        64.0,
        128.0,
    ]:
        raise ValidationError("side-weighted threshold grid drifted")
    if [float(value) for value in config["fault_relative_magnitudes"]] != [
        1e-12,
        1e-10,
        1e-8,
        1e-6,
    ]:
        raise ValidationError("fault grid drifted")
    if [float(value) for value in config["cancellation_deltas"]] != [
        1e-8,
        1e-6,
        1e-4,
    ]:
        raise ValidationError("cancellation grid drifted")
    if tuple(config["scenario_order"]) != EXPECTED_SCENARIOS:
        raise ValidationError("scenario order drifted")
    if config.get("status_precedence") != [
        "NONFINITE",
        "BRANCH",
        "STRUCTURE",
        "FD",
        "ADJOINT",
        "PASS_STRONG_SIGNAL_OR_LOW_SIGNAL_UNRESOLVED",
    ]:
        raise ValidationError("status precedence drifted")
    if config["finite_difference_contract"].get("denominator") != "2*h":
        raise ValidationError("FD denominator drifted")
    if config["finite_difference_contract"].get("aggregation") != (
        "worst_value_over_every_h_and_every_probe_in_the_frozen_probe_prefix"
    ):
        raise ValidationError("FD aggregation drifted")
    if config["finite_difference_contract"].get(
        "best_h_selection_is_forbidden"
    ) is not True:
        raise ValidationError("best-h selection is no longer forbidden")
    if any(bool(value) for value in config["claim_authorizations"].values()):
        raise ValidationError("config pre-authorized a forbidden research claim")
    if config["decision_contract"].get("threshold_selection_is_forbidden") is not True:
        raise ValidationError("threshold selection must remain forbidden")
    if config["decision_contract"].get("all_h_values_must_be_consumed") is not True:
        raise ValidationError("all h values must remain mandatory")


def _forbidden_key(key: str) -> bool:
    lowered = key.lower()
    if lowered == "pooled_overall_accuracy_removed":
        return False
    return any(
        token in lowered
        for token in (
            "overall_accuracy",
            "pooled_accuracy",
            "pooled_score",
            "best_threshold",
            "selected_threshold",
            "area_under_curve",
            "auc",
        )
    )


def _scan_json_keys(value: object, context: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if _forbidden_key(str(key)):
                raise ValidationError(f"{context}: forbidden pooled/selected key {key!r}")
            _scan_json_keys(child, f"{context}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _scan_json_keys(child, f"{context}[{index}]")


def _verify_manifest_and_sources(
    config_path: Path, result_dir: Path
) -> tuple[dict[str, object], dict[str, object]]:
    manifest = _read_json(result_dir / "manifest.json")
    result = _read_json(result_dir / "result.json")
    if not isinstance(manifest, dict) or not isinstance(result, dict):
        raise ValidationError("manifest and result must be JSON objects")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValidationError("manifest schema drifted")
    if result.get("schema") != RESULT_SCHEMA:
        raise ValidationError("result schema drifted")
    protocol_commit = str(result.get("protocol_commit", ""))
    if len(protocol_commit) != 40 or any(
        character not in "0123456789abcdef" for character in protocol_commit
    ):
        raise ValidationError("result protocol_commit is not a full lowercase SHA")
    if manifest.get("protocol_commit") != protocol_commit:
        raise ValidationError("manifest/result protocol_commit mismatch")

    config_relative = _relative(config_path)
    committed_config = _git_blob(protocol_commit, config_relative)
    committed_config_hash = hashlib.sha256(committed_config).hexdigest()
    current_config_hash = _sha256(config_path)
    if committed_config_hash != current_config_hash:
        raise ValidationError("current config differs from protocol commit")
    if result.get("config") != config_relative:
        raise ValidationError("result config path drifted")
    if result.get("config_sha256") != current_config_hash:
        raise ValidationError("result config SHA256 drifted")

    source_hashes = manifest.get("source_hashes")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValidationError("manifest source_hashes is missing")
    if result.get("source_hashes") != source_hashes:
        raise ValidationError("manifest/result source_hashes mismatch")
    for relative_path, reported_hash in source_hashes.items():
        committed = _git_blob(protocol_commit, str(relative_path))
        rebuilt_hash = hashlib.sha256(committed).hexdigest()
        if reported_hash != rebuilt_hash:
            raise ValidationError(
                f"committed source hash drifted for {relative_path}: "
                f"reported {reported_hash}, rebuilt {rebuilt_hash}"
            )
    if source_hashes.get(config_relative) != current_config_hash:
        raise ValidationError("config is not attested in source_hashes")

    artifact_hashes = manifest.get("artifact_hashes")
    if not isinstance(artifact_hashes, dict) or set(artifact_hashes) != set(
        ARTIFACT_NAMES
    ):
        raise ValidationError("manifest artifact set drifted")
    for name in ARTIFACT_NAMES:
        path = result_dir / name
        if not path.is_file():
            raise ValidationError(f"required result artifact is missing: {name}")
        actual_hash = _sha256(path)
        if artifact_hashes[name] != actual_hash:
            raise ValidationError(
                f"manifest hash mismatch for {name}: reported "
                f"{artifact_hashes[name]}, actual {actual_hash}"
            )

    allowed = set(ARTIFACT_NAMES) | {"manifest.json", DEFAULT_REPORT_NAME}
    actual_names = {entry.name for entry in result_dir.iterdir()}
    extra = actual_names - allowed
    if extra:
        raise ValidationError(
            f"result directory has uncontracted artifacts: {sorted(extra)!r}"
        )
    _scan_json_keys(result, "result.json")
    if result.get("threshold_selected") is not None:
        raise ValidationError("result threshold_selected must be null")
    if result.get("threshold_selection_forbidden") is not True:
        raise ValidationError("result no longer forbids threshold selection")
    return manifest, result


def _preflight_cross_file_consistency(
    config: Mapping[str, object], result_dir: Path
) -> None:
    """Catch report tampering before the expensive independent reconstruction."""
    evidence_by_key: dict[tuple[str, int], dict[str, str]] = {}
    with (result_dir / "evidence_rows.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != EVIDENCE_HEADER:
            raise ValidationError("evidence_rows.csv header drifted in preflight")
        for row_number, row in enumerate(reader, start=2):
            key = (row["case_id"], int(row["probe_count"]))
            if key in evidence_by_key:
                raise ValidationError(f"duplicate evidence key at row {row_number}: {key}")
            evidence_by_key[key] = row
    if len(evidence_by_key) != 3600:
        raise ValidationError(
            f"evidence preflight count {len(evidence_by_key)} != 3600"
        )

    fd_count = 0
    fd_keys: set[tuple[str, int, str]] = set()
    with (result_dir / "fd_rows.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FD_HEADER:
            raise ValidationError("fd_rows.csv header drifted in preflight")
        for row_number, row in enumerate(reader, start=2):
            fd_count += 1
            context = f"fd_rows.csv[{row_number}]"
            h_value = _parse_float(row["h"], f"{context}.h")
            if row["h_float_hex"] != h_value.hex():
                raise ValidationError(f"{context}: h_float_hex is not canonical")
            key = (row["case_id"], int(row["probe_index"]), row["h_float_hex"])
            if key in fd_keys:
                raise ValidationError(f"{context}: duplicate case/probe/h key {key}")
            fd_keys.add(key)
            branch_changed = len(
                {
                    row["base_branch_state"],
                    row["plus_branch_state"],
                    row["minus_branch_state"],
                }
            ) > 1
            if _parse_bool(
                row["actual_forward_branch_changed"],
                f"{context}.actual_forward_branch_changed",
            ) != branch_changed:
                raise ValidationError(
                    f"{context}: reported branch-change boolean disagrees with raw states"
                )
            diagnostic_changed = len(
                {
                    row["base_diagnostic_state"],
                    row["plus_diagnostic_state"],
                    row["minus_diagnostic_state"],
                }
            ) > 1
            if _parse_bool(
                row["diagnostic_state_changed"],
                f"{context}.diagnostic_state_changed",
            ) != diagnostic_changed:
                raise ValidationError(
                    f"{context}: reported diagnostic-change boolean disagrees with raw states"
                )
    if fd_count != 34560 or len(fd_keys) != 34560:
        raise ValidationError(
            f"FD preflight count/uniqueness drifted: rows={fd_count}, keys={len(fd_keys)}"
        )

    decision_count = 0
    decision_keys: set[tuple[str, int, str]] = set()
    with (result_dir / "decision_rows.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != DECISION_HEADER:
            raise ValidationError("decision_rows.csv header drifted in preflight")
        for row_number, row in enumerate(reader, start=2):
            decision_count += 1
            context = f"decision_rows.csv[{row_number}]"
            evidence_key = (row["case_id"], int(row["probe_count"]))
            evidence = evidence_by_key.get(evidence_key)
            if evidence is None:
                raise ValidationError(f"{context}: no matching evidence row")
            for field in EVIDENCE_HEADER:
                if row[field] != evidence[field]:
                    raise ValidationError(
                        f"{context}.{field}: reported metric/evidence copy drifted"
                    )
            threshold = _parse_float(
                row["side_weighted_gamma_threshold"],
                f"{context}.side_weighted_gamma_threshold",
            )
            key = (row["case_id"], int(row["probe_count"]), threshold.hex())
            if key in decision_keys:
                raise ValidationError(f"{context}: duplicate decision key {key}")
            decision_keys.add(key)
            typed_evidence: dict[str, object] = {
                field: row[field] for field in IDENTITY_FIELDS
            }
            typed_evidence.update(
                {
                    "probe_count": int(row["probe_count"]),
                    "all_finite": _parse_bool(row["all_finite"], context),
                    "maximum_signal_relative_defect": _parse_float(
                        row["maximum_signal_relative_defect"], context
                    ),
                    "maximum_side_weighted_gamma_score": _parse_float(
                        row["maximum_side_weighted_gamma_score"], context
                    ),
                    "maximum_contraction_envelope_ratio": _parse_float(
                        row["maximum_contraction_envelope_ratio"], context
                    ),
                    "minimum_signal": _parse_float(row["minimum_signal"], context),
                    "maximum_condition_proxy": _parse_float(
                        row["maximum_condition_proxy"], context
                    ),
                    "traditional_signal_gate": _parse_bool(
                        row["traditional_signal_gate"], context
                    ),
                    "maximum_fd_relative_error": _parse_float(
                        row["maximum_fd_relative_error"], context
                    ),
                    "maximum_structure_relative_error": _parse_float(
                        row["maximum_structure_relative_error"], context
                    ),
                    "actual_forward_branch_changed": _parse_bool(
                        row["actual_forward_branch_changed"], context
                    ),
                    "diagnostic_state_changed": _parse_bool(
                        row["diagnostic_state_changed"], context
                    ),
                    "ideal_reference_jvp_relative_error": _parse_float(
                        row["ideal_reference_jvp_relative_error"], context
                    ),
                    "ideal_reference_vjp_relative_error": _parse_float(
                        row["ideal_reference_vjp_relative_error"], context
                    ),
                    "ideal_reference_is_gate": _parse_bool(
                        row["ideal_reference_is_gate"], context
                    ),
                }
            )
            expected = _decision(config, typed_evidence, threshold)
            for field in DECISION_HEADER[len(EVIDENCE_HEADER) :]:
                expected_value = expected[field]
                if isinstance(expected_value, bool):
                    actual_value: object = _parse_bool(row[field], f"{context}.{field}")
                elif isinstance(expected_value, float):
                    actual_value = _parse_float(row[field], f"{context}.{field}")
                else:
                    actual_value = row[field]
                if isinstance(expected_value, float):
                    _assert_float_close(
                        float(actual_value), expected_value, f"{context}.{field}"
                    )
                elif actual_value != expected_value:
                    raise ValidationError(
                        f"{context}.{field}: reported decision {actual_value!r}, "
                        f"recomputed {expected_value!r}"
                    )
    if decision_count != 36000 or len(decision_keys) != 36000:
        raise ValidationError(
            "decision preflight count/uniqueness drifted: "
            f"rows={decision_count}, keys={len(decision_keys)}"
        )


def _expected_summary_rows(
    aggregates: Mapping[tuple[str, str, float, int, float], Mapping[str, int]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key in sorted(aggregates):
        scenario, parameter_name, parameter_value, probe_count, threshold = key
        counts = aggregates[key]
        total = sum(int(counts.get(status, 0)) for status in STATUS_VALUES)
        row: dict[str, object] = {
            "scenario": scenario,
            "parameter_name": parameter_name,
            "parameter_value": parameter_value,
            "probe_count": probe_count,
            "side_weighted_gamma_threshold": threshold,
            "count": total,
        }
        for status in STATUS_VALUES:
            count = int(counts.get(status, 0))
            row[f"{status.lower()}_count"] = count
            row[f"{status.lower()}_rate"] = count / total
        rows.append(row)
    return rows


def _headline(
    config: Mapping[str, object],
    first_probe_rows: Sequence[Mapping[str, object]],
    statuses: set[str],
) -> dict[str, object]:
    low = [
        row
        for row in first_probe_rows
        if row["scenario"] == "clean_low_bilinear_signal"
    ]
    diagnostic = [
        row
        for row in first_probe_rows
        if row["scenario"] == "diagnostic_only_support_flip"
    ]
    branch = [
        row
        for row in first_probe_rows
        if row["scenario"] == "actual_piecewise_branch_crossing"
    ]
    structures = [
        row
        for row in first_probe_rows
        if row["scenario"] == "three_path_structure_mismatch"
    ]
    by_fault: list[dict[str, object]] = []
    for raw_magnitude in config["fault_relative_magnitudes"]:
        magnitude = float(raw_magnitude)
        members = [
            row for row in structures if float(row["parameter_value"]) == magnitude
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
        "reported_status_values": sorted(statuses),
    }


def _summary_markdown(result: Mapping[str, object]) -> str:
    headline = result["headline_diagnostics"]
    counts = result["counts"]
    return "\n".join(
        (
            "# N5-D4c semantic-v2 development screen",
            "",
            f"- Decision: `{result['machine_decision']}`",
            f"- Explicit-matrix synthetic trials: `{counts['trial_count']}`",
            f"- Actual forward FD calls: `{counts['forward_plus_minus_call_count']}` plus/minus pairs",
            "- FD step sizes consumed: `"
            + ", ".join(str(value) for value in result["h_values_consumed"])
            + "`",
            "- Correct low-signal cases unresolved by the traditional signal gate: `"
            f"{headline['low_signal_traditional_reject_count']}/{headline['low_signal_count']}`",
            "- Diagnostic-only state changes with actual branch changes: `"
            f"{headline['diagnostic_state_change_count']}/{headline['diagnostic_actual_branch_change_count']}` (the second number must remain zero)",
            "- Actual piecewise crossings observed: `"
            f"{headline['actual_branch_change_count']}/{headline['actual_branch_count']}`",
            "- D4c-v1 remains preserved as integrity-valid but semantic-NO-GO negative evidence.",
            "- No threshold was selected. This artifact does not authorize BOST, reconstruction, neural-operator, real-data, generalization, superiority, or paper claims.",
            "",
        )
    )


def _validate_result_object(
    config: Mapping[str, object],
    result: Mapping[str, object],
    protocol_commit: str,
    expected_counts: Mapping[str, int],
    expected_headline: Mapping[str, object],
    expected_summary: Sequence[Mapping[str, object]],
) -> None:
    expected_keys = {
        "schema",
        "candidate_id",
        "created_at_utc",
        "machine_decision",
        "threshold_selected",
        "threshold_selection_forbidden",
        "protocol_commit",
        "config",
        "config_sha256",
        "source_hashes",
        "environment",
        "counts",
        "h_values_consumed",
        "semantic_corrections_from_v1",
        "headline_diagnostics",
        "scenario_summary",
        "claim_authorizations",
        "limitations",
    }
    if set(result) != expected_keys:
        raise ValidationError("result.json top-level fields drifted")
    if result["candidate_id"] != config["candidate_id"]:
        raise ValidationError("result candidate ID drifted")
    if result["machine_decision"] != config["decision_contract"]["decision"]:
        raise ValidationError("result machine decision drifted")
    if result["protocol_commit"] != protocol_commit:
        raise ValidationError("result protocol commit drifted")
    try:
        created = datetime.fromisoformat(str(result["created_at_utc"]))
    except ValueError as exc:
        raise ValidationError("result created_at_utc is not ISO-8601") from exc
    if created.tzinfo is None:
        raise ValidationError("result created_at_utc lacks timezone")
    _compare_json(result["counts"], dict(expected_counts), "result.json.counts")
    _compare_json(
        result["h_values_consumed"],
        [float(value) for value in config["h_values"]],
        "result.json.h_values_consumed",
    )
    expected_corrections = {
        "actual_forward_finite_difference": True,
        "forward_returned_branch_state": True,
        "diagnostic_state_separated_from_branch_state": True,
        "independent_curved_straight_direct_paths": True,
        "side_specific_gamma_weights": True,
        "low_signal_is_unresolved_not_pass": True,
        "pooled_overall_accuracy_removed": True,
    }
    _compare_json(
        result["semantic_corrections_from_v1"],
        expected_corrections,
        "result.json.semantic_corrections_from_v1",
    )
    _compare_json(
        result["headline_diagnostics"],
        dict(expected_headline),
        "result.json.headline_diagnostics",
    )
    _compare_json(
        result["scenario_summary"],
        [dict(row) for row in expected_summary],
        "result.json.scenario_summary",
    )
    _compare_json(
        result["claim_authorizations"],
        config["claim_authorizations"],
        "result.json.claim_authorizations",
    )
    if any(bool(value) for value in result["claim_authorizations"].values()):
        raise ValidationError("result authorizes a forbidden research claim")
    environment = result["environment"]
    expected_environment_keys = {
        "python",
        "python_executable",
        "numpy",
        "platform",
        "machine",
        "processor",
        "numpy_show_config",
    }
    if not isinstance(environment, dict) or set(environment) != expected_environment_keys:
        raise ValidationError("result environment attestation fields drifted")
    if not all(isinstance(value, str) and value for value in environment.values()):
        raise ValidationError("result environment attestation is incomplete")
    if environment["numpy"] != np.__version__:
        raise ValidationError(
            "validator NumPy differs from recorded run; exact deterministic replay is unsafe"
        )
    expected_limitations = [
        "All operators are synthetic explicit float64 matrices, not the OERF ray renderer.",
        "The protocol tests certificate semantics, not reconstruction quality or speed.",
        "The threshold grid is descriptive and no value is selected here.",
        "Actual BOST branch states and three-path contracts remain to be wired after a real renderer/data contract is available.",
    ]
    _compare_json(
        result["limitations"], expected_limitations, "result.json.limitations"
    )


def _semantic_reconstruction(
    config: Mapping[str, object], result_dir: Path, result: Mapping[str, object]
) -> dict[str, object]:
    trial_count = int(config["trial_count"])
    probe_counts = [int(value) for value in config["probe_counts"]]
    h_values = [float(value) for value in config["h_values"]]
    thresholds = [
        float(value) for value in config["side_weighted_gamma_threshold_grid"]
    ]

    counts = {
        "trial_count": trial_count,
        "scenario_count": len(EXPECTED_SCENARIOS),
        "variant_count": 0,
        "case_spec_rows": 0,
        "probe_rows": 0,
        "fd_rows": 0,
        "forward_plus_minus_call_count": 0,
        "actual_primary_forward_call_count": 0,
        "structure_rows": 0,
        "evidence_rows": 0,
        "decision_rows": 0,
        "scenario_summary_rows": 0,
    }
    first_probe_rows: list[dict[str, object]] = []
    statuses: set[str] = set()
    aggregates: dict[tuple[str, str, float, int, float], dict[str, int]] = {}
    consumed_h: set[float] = set()
    case_ids: set[str] = set()

    with ExitStack() as stack:
        case_cursor = stack.enter_context(
            JsonlCursor(result_dir / "case_specs.jsonl")
        )
        probe_cursor = stack.enter_context(
            CsvCursor(result_dir / "probe_rows.csv", PROBE_HEADER)
        )
        fd_cursor = stack.enter_context(CsvCursor(result_dir / "fd_rows.csv", FD_HEADER))
        structure_cursor = stack.enter_context(
            CsvCursor(result_dir / "structure_rows.csv", STRUCTURE_HEADER)
        )
        evidence_cursor = stack.enter_context(
            CsvCursor(result_dir / "evidence_rows.csv", EVIDENCE_HEADER)
        )
        decision_cursor = stack.enter_context(
            CsvCursor(result_dir / "decision_rows.csv", DECISION_HEADER)
        )

        for trial in range(trial_count):
            cases = _build_trial_cases(config, trial)
            if len(cases) != 30:
                raise ValidationError(
                    f"trial {trial}: independently rebuilt {len(cases)} cases, expected 30"
                )
            counts["variant_count"] += len(cases)
            for case in cases:
                identity = _case_identity(case)
                case_id = str(identity["case_id"])
                if case_id in case_ids:
                    raise ValidationError(f"duplicate independently rebuilt case: {case_id}")
                case_ids.add(case_id)
                case_cursor.check(_case_spec(config, case), case_id)
                counts["case_spec_rows"] += 1

                expected_structure_rows, structure_errors = _structure_rows(case)
                for row in expected_structure_rows:
                    structure_cursor.check(
                        row, f"{case_id}/probe-{row['probe_index']}"
                    )
                    counts["structure_rows"] += 1

                base = case.forward.call(case.base_x)
                adjoint_rows: list[dict[str, object]] = []
                fd_errors: list[list[float]] = []
                branch_changes: list[bool] = []
                diagnostic_changes: list[bool] = []
                ideal_jvp_errors: list[float] = []

                for probe_index, (tangent, candidate_jvp) in enumerate(
                    zip(case.tangents, case.candidate_jvps, strict=True)
                ):
                    adjoint = _adjoint_evidence(
                        candidate_jvp,
                        case.cotangent,
                        tangent,
                        case.candidate_vjp,
                    )
                    adjoint_rows.append(adjoint)
                    probe_row = (
                        identity
                        | {"probe_index": probe_index}
                        | adjoint
                        | {
                            "tangent_sha256": _array_sha256(tangent),
                            "candidate_jvp_sha256": _array_sha256(candidate_jvp),
                            "candidate_vjp_sha256": _array_sha256(
                                case.candidate_vjp
                            ),
                        }
                        | _vector_fields("candidate_jvp", candidate_jvp)
                    )
                    probe_cursor.check(probe_row, f"{case_id}/probe-{probe_index}")
                    counts["probe_rows"] += 1

                    probe_fd_errors: list[float] = []
                    probe_branch_changed = False
                    probe_diagnostic_changed = False
                    for h in h_values:
                        plus_input = case.base_x + h * tangent
                        minus_input = case.base_x - h * tangent
                        plus = case.forward.call(plus_input)
                        minus = case.forward.call(minus_input)
                        estimate = (plus.output - minus.output) / (2.0 * h)
                        fd_error = _relative_l2(estimate, candidate_jvp)
                        branch_changed = len(
                            {
                                base.branch_state,
                                plus.branch_state,
                                minus.branch_state,
                            }
                        ) > 1
                        diagnostic_changed = len(
                            {
                                base.diagnostic_state,
                                plus.diagnostic_state,
                                minus.diagnostic_state,
                            }
                        ) > 1
                        expected_fd = (
                            identity
                            | {
                                "call_pair_id": (
                                    f"{case_id}__probe-{probe_index:02d}__h-{h.hex()}"
                                ),
                                "probe_index": probe_index,
                                "h": h,
                                "h_float_hex": h.hex(),
                                "base_input_sha256": _array_sha256(case.base_x),
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
                        fd_cursor.check(
                            expected_fd,
                            f"{case_id}/probe-{probe_index}/h-{h.hex()}",
                        )
                        counts["fd_rows"] += 1
                        consumed_h.add(h)
                        probe_fd_errors.append(fd_error)
                        probe_branch_changed = (
                            probe_branch_changed or branch_changed
                        )
                        probe_diagnostic_changed = (
                            probe_diagnostic_changed or diagnostic_changed
                        )
                    fd_errors.append(probe_fd_errors)
                    branch_changes.append(probe_branch_changed)
                    diagnostic_changes.append(probe_diagnostic_changed)
                    ideal_jvp_errors.append(
                        0.0
                        if case.ideal_jvps is None
                        else _relative_l2(
                            candidate_jvp, case.ideal_jvps[probe_index]
                        )
                    )

                for probe_count in probe_counts:
                    summary = _probe_summary(adjoint_rows[:probe_count])
                    maximum_fd = max(
                        value
                        for values in fd_errors[:probe_count]
                        for value in values
                    )
                    maximum_structure = max(
                        structure_errors[:probe_count], default=0.0
                    )
                    ideal_vjp_error = (
                        0.0
                        if case.ideal_vjp is None
                        else _relative_l2(case.candidate_vjp, case.ideal_vjp)
                    )
                    evidence = (
                        identity
                        | summary
                        | {
                            "probe_count": probe_count,
                            "traditional_signal_gate": bool(
                                summary["all_finite"]
                                and float(summary["maximum_signal_relative_defect"])
                                <= float(
                                    config["traditional_signal_relative_threshold"]
                                )
                            ),
                            "maximum_fd_relative_error": maximum_fd,
                            "maximum_structure_relative_error": maximum_structure,
                            "actual_forward_branch_changed": any(
                                branch_changes[:probe_count]
                            ),
                            "diagnostic_state_changed": any(
                                diagnostic_changes[:probe_count]
                            ),
                            "ideal_reference_jvp_relative_error": max(
                                ideal_jvp_errors[:probe_count], default=0.0
                            ),
                            "ideal_reference_vjp_relative_error": ideal_vjp_error,
                            "ideal_reference_is_gate": False,
                        }
                    )
                    evidence_cursor.check(
                        evidence, f"{case_id}/prefix-{probe_count}"
                    )
                    counts["evidence_rows"] += 1
                    if probe_count == 1:
                        first_probe_rows.append(evidence)

                    for threshold in thresholds:
                        decision = _decision(config, evidence, threshold)
                        decision_cursor.check(
                            decision,
                            f"{case_id}/prefix-{probe_count}/gamma-{threshold.hex()}",
                        )
                        counts["decision_rows"] += 1
                        status = str(decision["status"])
                        statuses.add(status)
                        aggregate_key = (
                            case.scenario,
                            case.parameter_name,
                            float(case.parameter_value),
                            probe_count,
                            threshold,
                        )
                        status_counts = aggregates.setdefault(aggregate_key, {})
                        status_counts[status] = status_counts.get(status, 0) + 1

        counts["forward_plus_minus_call_count"] = counts["fd_rows"]
        counts["actual_primary_forward_call_count"] = (
            counts["variant_count"] + 2 * counts["fd_rows"]
        )
        expected_summary = _expected_summary_rows(aggregates)
        counts["scenario_summary_rows"] = len(expected_summary)

        case_cursor.finish(720)
        probe_cursor.finish(11520)
        fd_cursor.finish(34560)
        structure_cursor.finish(1536)
        evidence_cursor.finish(3600)
        decision_cursor.finish(36000)

    if counts["variant_count"] != 720 or len(case_ids) != 720:
        raise ValidationError("independent case grid did not produce 720 unique cases")
    if consumed_h != set(h_values):
        raise ValidationError(
            f"not every h was consumed: actual={sorted(consumed_h)}, expected={h_values}"
        )
    if len(first_probe_rows) != 720:
        raise ValidationError("first-probe evidence coverage drifted")
    if len(expected_summary) != 1500:
        raise ValidationError(
            f"independent scenario summary has {len(expected_summary)} rows, expected 1500"
        )

    with CsvCursor(result_dir / "scenario_summary.csv", SUMMARY_HEADER) as cursor:
        for row in expected_summary:
            label = (
                f"{row['scenario']}/{row['parameter_value']}/"
                f"{row['probe_count']}/{row['side_weighted_gamma_threshold']}"
            )
            cursor.check(row, label)
        cursor.finish(1500)

    expected_headline = _headline(config, first_probe_rows, statuses)
    _validate_result_object(
        config,
        result,
        str(result["protocol_commit"]),
        counts,
        expected_headline,
        expected_summary,
    )
    expected_summary_text = _summary_markdown(result)
    actual_summary_text = (result_dir / "summary.md").read_text(encoding="utf-8")
    if actual_summary_text != expected_summary_text:
        raise ValidationError("summary.md disagrees with independently validated result")

    return {
        "counts": counts,
        "h_values_consumed": sorted(consumed_h),
        "status_values": sorted(statuses),
        "scenario_summary_rows": len(expected_summary),
        "low_signal_unresolved_is_not_pass": True,
        "all_case_specs_rebuilt": True,
        "all_probe_metrics_rebuilt": True,
        "all_forward_calls_replayed": True,
        "all_input_hashes_rebuilt": True,
        "all_branch_and_diagnostic_states_rebuilt": True,
        "all_three_path_obligations_rebuilt": True,
        "all_worst_prefix_decisions_rebuilt": True,
    }


def _validate_bundle(config_path: Path, result_dir: Path) -> dict[str, object]:
    config = _read_json(config_path)
    if not isinstance(config, dict):
        raise ValidationError("config must be a JSON object")
    _validate_config(config)
    manifest, result = _verify_manifest_and_sources(config_path, result_dir)
    _preflight_cross_file_consistency(config, result_dir)
    semantic = _semantic_reconstruction(config, result_dir, result)
    return {
        "protocol_commit": result["protocol_commit"],
        "config_sha256": result["config_sha256"],
        "manifest_sha256": _sha256(result_dir / "manifest.json"),
        "manifest_source_count": len(manifest["source_hashes"]),
        "manifest_artifact_count": len(manifest["artifact_hashes"]),
        **semantic,
    }


def validate(
    config_path: Path = DEFAULT_CONFIG,
    result_dir: Path = DEFAULT_RESULT,
    report_path: Path | None = None,
) -> dict[str, object]:
    """Validate a bundle and write a non-overwriting machine-readable report."""
    config_path = Path(config_path).resolve()
    result_dir = Path(result_dir).resolve()
    output = (
        result_dir / DEFAULT_REPORT_NAME
        if report_path is None
        else Path(report_path).resolve()
    )
    if output.exists() or os.path.lexists(output):
        raise FileExistsError(f"refusing to replace validation report: {output}")

    started = time.monotonic()
    validated_at = datetime.now(timezone.utc).isoformat()
    errors: list[str] = []
    details: dict[str, object] = {}
    try:
        details = _validate_bundle(config_path, result_dir)
    except (ValidationError, OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")

    report: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "valid": not errors,
        "validated_at_utc": validated_at,
        "duration_seconds": time.monotonic() - started,
        "validator": _relative(Path(__file__)),
        "validator_sha256": _sha256(Path(__file__)),
        "config": _relative(config_path),
        "result_directory": (
            _relative(result_dir)
            if result_dir.is_relative_to(ROOT)
            else f"external-test-input/{result_dir.name}"
        ),
        "independence_contract": {
            "runner_imported": False,
            "certificate_helper_imported": False,
            "rng_cases_and_forward_calls_rebuilt": True,
            "reported_metrics_used_as_truth": False,
        },
        "errors": errors,
        "details": details,
        "claim_boundary": {
            "synthetic_explicit_matrix_only": True,
            "bost_authorized": False,
            "reconstruction_authorized": False,
            "real_data_authorized": False,
            "generalization_authorized": False,
            "superiority_authorized": False,
            "paper_claim_authorized": False,
        },
        "residual_risks": [
            "The validator independently reimplements the frozen synthetic semantics, but it is not an external implementation in another language or numerical stack.",
            "The manifest is not cryptographically signed; semantic replay protects contracted numerical evidence, while the PNG receives only hash/inventory checking.",
            "Exact array hashes assume the recorded NumPy/BLAS numerical stream remains reproducible on this machine.",
            "A valid report does not test the OERF ray renderer, BOST data, a neural operator, reconstruction quality, or generalization.",
        ],
    }
    _write_json_atomic(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Report path. Defaults to RESULT_DIR/validation_report.json and "
            "never overwrites an existing file."
        ),
    )
    args = parser.parse_args()
    report = validate(args.config, args.result_dir, args.output)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
