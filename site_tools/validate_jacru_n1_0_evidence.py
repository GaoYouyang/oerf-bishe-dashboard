#!/usr/bin/env python3
"""Independently audit the frozen JACRU N1.0 evidence packet.

The validator uses only the Python standard library and frozen public scalar
artifacts.  It never imports an experiment runner, model, or optical operator.
Selections are reconstructed from the opened M2.7 trajectory through a narrow
observable-only record; field and clean-renderer columns are consulted only
after the selected iteration has been fixed.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "demo_t16_operator/configs/"
    "jacru_n1_0_observable_discrepancy_stopping_postopen_v1.json"
)
OUTPUT = (
    ROOT
    / "demo_t16_operator/results/"
    "jacru_n1_0_observable_discrepancy_stopping_postopen_public"
)

METHODS = ("jacru_m2", "pooled_cnn")
MODEL_SEEDS = (17, 29, 43)
SPLITS = ("development", "ood")
SPLIT_CASE_COUNTS = {"development": 12, "ood": 18}
ITERATIONS = tuple(range(11))
TRAJECTORY_VARIANT = "affine_pcg_dense_exact_camera_block_oracle"
CAMERA_BLOCK_ORACLE = "dense_exact_camera_block_jacobi_oracle"

VALIDATED_STATUS = "VALIDATED_N1_0_OBSERVABLE_DISCREPANCY_STOPPING_NO_GO"
PACKET_STATUS = "N1_0_OBSERVABLE_DISCREPANCY_STOPPING_NO_GO"
M27_STATUS = "M2_7_TARGET_NO_HARM_PARETO_ORACLE_NO_GO"
M28_STATUS = "M2_8_INTERPOLATION_CALIBRATION_ENVELOPE_NO_GO"

AUTHORIZATION = {
    "claim_deployable_algorithm": False,
    "claim_method_superiority": False,
    "claim_real_bost_generalization": False,
    "continue_flow_off_covariance_research": True,
    "continue_heldout_fail_closed_research": True,
    "open_fresh_or_final": False,
}

N1_PAYLOADS = {
    "README.md",
    "aggregate_rows.csv",
    "diagnostic.pdf",
    "diagnostic.png",
    "selected_rows.csv",
    "summary.json",
}
M27_PAYLOADS = {
    "README.md",
    "aggregate_rows.csv",
    "diagnostic.pdf",
    "diagnostic.png",
    "matched_baseline_aggregate_rows.csv",
    "matched_baseline_rows.csv",
    "metric_rows.csv",
    "reference_rows.csv",
    "summary.json",
}

SELECTOR_COLUMNS = (
    "measured_reprojection_relative_l2",
    "prepared_cgls_base_12_measured_reprojection_relative_l2",
    "system_residual_fraction",
)
FORBIDDEN_SELECTOR_FRAGMENTS = (
    "field",
    "clean",
    "truth",
    "h1",
    "harm",
    "gain",
)

SELECTED_FIELDS = (
    "candidate_id",
    "stopping_family",
    "stopping_parameter",
    "stopping_threshold",
    "comparator_only",
    "uses_simulator_nuisance_scale",
    "selection_uses_truth",
    "selection_uses_clean_renderer",
    "target_crossed",
    "selected_observable_value",
    "selected_iteration",
    "attempted_iteration",
    "returned_field_kind",
    "case_id",
    "split",
    "family",
    "base_seed",
    "method",
    "model_seed",
    "field_relative_l2",
    "h1_seminorm_relative_error",
    "measured_reprojection_relative_l2",
    "clean_reprojection_relative_l2",
    "clean_reprojection_ratio_to_base",
    "field_gain_to_best_matched_classical",
    "h1_gain_to_best_matched_classical",
    "reprojection_ratio_to_matched_cgls",
    "field_harm",
    "projection_closure_relative_error",
    "forward_calls",
    "adjoint_calls",
    "exact_camera_block_setup_forward_equivalents",
    "exact_camera_block_setup_in_budget",
)

AGGREGATE_FIELDS = (
    "candidate_id",
    "stopping_family",
    "stopping_parameter",
    "comparator_only",
    "method",
    "model_seed",
    "split",
    "case_count",
    "target_crossing_rate",
    "selected_iteration_mean",
    "selected_iteration_maximum",
    "field_gain_mean",
    "h1_gain_mean",
    "clean_reprojection_ratio_to_base_mean",
    "clean_reprojection_ratio_to_base_maximum",
    "reprojection_ratio_to_matched_cgls_mean",
    "field_harm_rate",
    "worst_field_gain",
    "projection_closure_relative_error_maximum",
    "forward_calls_mean",
    "forward_calls_maximum",
    "adjoint_calls_mean",
    "adjoint_calls_maximum",
)

M27_REQUIRED_FIELDS = {
    "case_id",
    "split",
    "family",
    "base_seed",
    "method",
    "model_seed",
    "field_relative_l2",
    "h1_seminorm_relative_error",
    "measured_reprojection_relative_l2",
    "clean_reprojection_relative_l2",
    "optimization_forward_calls",
    "optimization_adjoint_calls",
    "grouped_adjoint_calls",
    "evaluation_forward_calls",
    "projection_variant",
    "projection_iterations",
    "damping_fraction",
    "damping_absolute",
    "preconditioner",
    "projection_forward_calls",
    "projection_adjoint_calls",
    "paired_call_budget",
    "matched_cgls_field_relative_l2",
    "matched_huber_field_relative_l2",
    "field_gain_to_best_matched_classical",
    "h1_gain_to_best_matched_classical",
    "reprojection_ratio_to_matched_cgls",
    "system_residual_fraction",
    "projection_closure_relative_error",
    "field_harm_to_best_matched_classical",
    "breakdown",
    "projection_diagnostic_forward_calls",
    "dense_oracle_used_by_algorithm",
    "projection_target_mode",
    "preconditioner_kind",
    "preconditioner_is_oracle",
    "preconditioner_setup_forward_equivalents",
    "preconditioner_setup_adjoint_equivalents",
    "preconditioner_applications",
    "preconditioner_block_count",
    "preconditioner_largest_block_size",
    "preconditioner_minimum_block_eigenvalue",
    "preconditioner_maximum_block_condition_number",
}

REFERENCE_REQUIRED_FIELDS = {
    "case_id",
    "split",
    "family",
    "base_seed",
    "method",
    "model_seed",
    "field_relative_l2",
    "h1_seminorm_relative_error",
    "measured_reprojection_relative_l2",
    "clean_reprojection_relative_l2",
    "optimization_forward_calls",
    "optimization_adjoint_calls",
    "grouped_adjoint_calls",
    "evaluation_forward_calls",
    "reference_kind",
}


class ValidationError(RuntimeError):
    """Raised when a frozen evidence packet violates its declared contract."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ValidationError(f"cannot hash {path}: {error}") from error


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot read JSON {path}: {error}") from error
    _need(isinstance(value, dict), f"expected one JSON object: {path}")
    return value


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValidationError(f"{label}: expected finite number") from error
    _need(math.isfinite(result), f"{label}: expected finite number")
    return result


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValidationError(f"{label}: expected integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValidationError(f"{label}: expected integer") from error
    if isinstance(value, str):
        _need(value == str(result), f"{label}: non-canonical integer")
    else:
        _need(value == result, f"{label}: non-integral value")
    return result


def _boolean(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValidationError(f"{label}: expected canonical boolean")


def _close(
    actual: Any,
    expected: float,
    label: str,
    *,
    tolerance: float = 5e-10,
) -> None:
    observed = _finite_float(actual, label)
    _need(
        math.isclose(observed, expected, rel_tol=tolerance, abs_tol=tolerance),
        f"{label}: numeric mismatch ({observed!r} != {expected!r})",
    )


def _mean(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    _need(bool(materialized), "cannot average empty values")
    return math.fsum(materialized) / len(materialized)


def _read_csv_exact(
    path: Path,
    fields: Sequence[str],
    expected_count: int,
) -> list[dict[str, str]]:
    try:
        physical = path.read_text(encoding="utf-8").splitlines()
        _need(all(physical), f"{path.name}: blank physical line")
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            _need(
                tuple(reader.fieldnames or ()) == tuple(fields),
                f"{path.name}: columns differ from frozen schema",
            )
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValidationError(f"cannot read CSV {path}: {error}") from error
    _need(len(rows) == expected_count, f"{path.name}: row count drift")
    _need(len(physical) == expected_count + 1, f"{path.name}: physical row count drift")
    _need(
        all(None not in row and all(value is not None for value in row.values()) for row in rows),
        f"{path.name}: malformed row",
    )
    return rows


def _read_csv_required(
    path: Path,
    required_fields: set[str],
    expected_count: int,
) -> list[dict[str, str]]:
    try:
        physical = path.read_text(encoding="utf-8").splitlines()
        _need(all(physical), f"{path.name}: blank physical line")
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            header = tuple(reader.fieldnames or ())
            _need(len(header) == len(set(header)), f"{path.name}: duplicate column")
            _need(required_fields.issubset(header), f"{path.name}: required columns missing")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValidationError(f"cannot read CSV {path}: {error}") from error
    _need(len(rows) == expected_count, f"{path.name}: row count drift")
    _need(len(physical) == expected_count + 1, f"{path.name}: physical row count drift")
    _need(
        all(None not in row and all(value is not None for value in row.values()) for row in rows),
        f"{path.name}: malformed row",
    )
    return rows


def _validate_manifest(directory: Path, payloads: set[str]) -> None:
    manifest = directory / "checksums.sha256"
    try:
        lines = manifest.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValidationError(f"cannot read checksum manifest: {error}") from error
    _need(bool(lines), "checksums.sha256: empty manifest")
    entries: dict[str, str] = {}
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_.-]+)", line)
        _need(match is not None, "checksums.sha256: malformed line")
        assert match is not None
        digest, name = match.groups()
        _need(name not in entries, f"checksums.sha256: duplicate entry {name}")
        entries[name] = digest
    _need(set(entries) == payloads, "checksums.sha256: payload set mismatch")
    try:
        actual_files = {path.name for path in directory.iterdir()}
    except OSError as error:
        raise ValidationError(f"cannot inspect evidence directory: {error}") from error
    _need(
        actual_files == payloads | {"checksums.sha256"},
        "public packet contains unmanifested or missing files",
    )
    for name, digest in entries.items():
        path = directory / name
        _need(path.is_file() and not path.is_symlink(), f"invalid payload: {name}")
        _need(_sha256(path) == digest, f"checksum mismatch: {name}")


def _deep_match(actual: Any, expected: Any, label: str) -> None:
    if isinstance(expected, Mapping):
        _need(isinstance(actual, Mapping), f"{label}: object drift")
        _need(set(actual) == set(expected), f"{label}: keys drift")
        for key, value in expected.items():
            _deep_match(actual[key], value, f"{label}.{key}")
        return
    if isinstance(expected, list):
        _need(isinstance(actual, list), f"{label}: list drift")
        _need(len(actual) == len(expected), f"{label}: list length drift")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _deep_match(left, right, f"{label}[{index}]")
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        _close(actual, float(expected), label)
        return
    _need(actual == expected, f"{label}: value drift ({actual!r} != {expected!r})")


@dataclass(frozen=True)
class ObservablePoint:
    """The complete selector-visible part of one M2.7 trajectory row."""

    iteration: int
    measured_residual: float
    system_residual_fraction: float


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    family: str
    parameter: float | int
    threshold: float | None
    comparator_only: bool
    uses_simulator_nuisance_scale: bool


@dataclass(frozen=True)
class Selection:
    selected_iteration: int | None
    attempted_iteration: int
    crossed: bool
    observable_value: float | None
    threshold: float | None


GroupKey = tuple[str, int, str, str]
SelectedKey = tuple[str, str, int, str, str]


def _candidate_specs(config: Mapping[str, Any], noise_floor: float) -> tuple[CandidateSpec, ...]:
    families = config["observable_stopping_families"]
    specs: list[CandidateSpec] = []
    for raw in families["simulator_noise_floor_multiple"]["multipliers"]:
        value = float(raw)
        specs.append(
            CandidateSpec(
                f"noise_floor_x{value:g}",
                "simulator_noise_floor_multiple",
                value,
                value * noise_floor,
                False,
                True,
            )
        )
    for raw in families["base_anchor_residual_multiple"]["multipliers"]:
        value = float(raw)
        specs.append(
            CandidateSpec(
                f"base_residual_x{value:g}",
                "base_anchor_residual_multiple",
                value,
                value,
                False,
                False,
            )
        )
    for raw in families["initial_system_residual_fraction"]["maximum_fractions"]:
        value = float(raw)
        specs.append(
            CandidateSpec(
                f"system_fraction_{value:g}",
                "initial_system_residual_fraction",
                value,
                value,
                False,
                False,
            )
        )
    for raw in config["fixed_iteration_comparators"]:
        value = int(raw)
        specs.append(
            CandidateSpec(
                f"fixed_k{value}",
                "fixed_iteration_comparator",
                value,
                None,
                True,
                False,
            )
        )
    _need(len({spec.candidate_id for spec in specs}) == len(specs), "candidate IDs overlap")
    return tuple(specs)


def _build_trajectory_groups(
    rows: Sequence[Mapping[str, str]],
    config: Mapping[str, Any],
) -> tuple[
    dict[GroupKey, list[Mapping[str, str]]],
    dict[GroupKey, tuple[ObservablePoint, ...]],
]:
    expected_iterations = tuple(int(value) for value in config["trajectory"]["source_iterations"])
    maximum_iteration = int(config["trajectory"]["maximum_iteration"])
    groups: dict[GroupKey, list[Mapping[str, str]]] = {}
    for index, row in enumerate(rows):
        label = f"M2.7 trajectory[{index}]"
        method = row["method"]
        seed = _integer(row["model_seed"], f"{label}.model_seed")
        split = row["split"]
        case_id = row["case_id"]
        iteration = _integer(row["projection_iterations"], f"{label}.K")
        _need(method in METHODS and seed in MODEL_SEEDS, f"{label}: method/seed drift")
        _need(split in SPLITS and bool(case_id), f"{label}: split/case drift")
        _need(iteration in expected_iterations, f"{label}: K grid drift")
        _need(row["projection_variant"] == TRAJECTORY_VARIANT, f"{label}: variant drift")
        _need(row["projection_target_mode"] == "affine_observation", f"{label}: target drift")
        _close(row["damping_absolute"], 0.0, f"{label}.damping_absolute")
        _close(row["damping_fraction"], 0.0, f"{label}.damping_fraction")
        _need(
            row["preconditioner"] == CAMERA_BLOCK_ORACLE
            and row["preconditioner_kind"] == CAMERA_BLOCK_ORACLE,
            f"{label}: exact camera-block oracle drift",
        )
        _need(_integer(row["preconditioner_is_oracle"], label) == 1, f"{label}: oracle flag drift")
        _need(
            _integer(row["preconditioner_setup_forward_equivalents"], label) == 1001
            and _integer(row["preconditioner_setup_adjoint_equivalents"], label) == 0,
            f"{label}: exact camera-block setup ledger drift",
        )
        _need(
            _integer(row["preconditioner_applications"], label) == maximum_iteration + 1,
            f"{label}: fixed preconditioner application ledger drift",
        )
        _need(
            (_integer(row["preconditioner_block_count"], label),
             _integer(row["preconditioner_largest_block_size"], label)) == (3, 50),
            f"{label}: camera-block partition drift",
        )
        _need(
            _finite_float(row["preconditioner_minimum_block_eigenvalue"], label) > 0.0
            and _finite_float(row["preconditioner_maximum_block_condition_number"], label) >= 1.0,
            f"{label}: camera-block SPD drift",
        )
        expected_forward = 14 + iteration
        expected_adjoint = 13 + iteration
        for field, expected in (
            ("projection_forward_calls", iteration + 1),
            ("projection_adjoint_calls", iteration),
            ("optimization_forward_calls", expected_forward),
            ("optimization_adjoint_calls", expected_adjoint),
            ("paired_call_budget", expected_forward),
            ("projection_diagnostic_forward_calls", 1),
            ("grouped_adjoint_calls", 1),
            ("evaluation_forward_calls", 1),
        ):
            _need(_integer(row[field], f"{label}.{field}") == expected, f"{label}: F/A call ledger drift")
        _need(expected_forward <= 24 and expected_adjoint <= 23, f"{label}: N1 call cap drift")
        _need(1001 > expected_forward, f"{label}: exact oracle setup entered call budget")
        _need(row["dense_oracle_used_by_algorithm"] == "False", f"{label}: dense SVD oracle entered algorithm")
        _need(_integer(row["breakdown"], label) == 0, f"{label}: source trajectory breakdown")
        _need(
            _finite_float(row["projection_closure_relative_error"], label)
            <= float(config["decision_gates"]["maximum_projection_closure_relative_error"]),
            f"{label}: closure gate violated",
        )
        measured = _finite_float(row["measured_reprojection_relative_l2"], f"{label}.measured")
        system = _finite_float(row["system_residual_fraction"], f"{label}.system")
        _need(measured >= 0.0 and system >= 0.0, f"{label}: negative selector observable")
        key = (method, seed, split, case_id)
        groups.setdefault(key, []).append(row)

    observable: dict[GroupKey, tuple[ObservablePoint, ...]] = {}
    for key, values in groups.items():
        values.sort(key=lambda row: _integer(row["projection_iterations"], "trajectory K"))
        observed_iterations = tuple(_integer(row["projection_iterations"], "trajectory K") for row in values)
        _need(observed_iterations == expected_iterations, f"incomplete M2.7 trajectory for {key}")
        first = values[0]
        _need(
            all(
                row["family"] == first["family"]
                and row["base_seed"] == first["base_seed"]
                for row in values
            ),
            f"M2.7 trajectory identity drift for {key}",
        )
        observable[key] = tuple(
            ObservablePoint(
                iteration=_integer(row["projection_iterations"], "trajectory K"),
                measured_residual=_finite_float(row["measured_reprojection_relative_l2"], "measured residual"),
                system_residual_fraction=_finite_float(row["system_residual_fraction"], "system residual"),
            )
            for row in values
        )

    _need(len(groups) == 180, "M2.7 trajectory group count drift")
    for method in METHODS:
        for seed in MODEL_SEEDS:
            for split in SPLITS:
                count = sum(key[:3] == (method, seed, split) for key in groups)
                _need(count == SPLIT_CASE_COUNTS[split], "M2.7 trajectory split catalog drift")
    return groups, observable


def _base_anchor_lookup(
    rows: Sequence[Mapping[str, str]],
    cases: set[str],
) -> dict[str, Mapping[str, str]]:
    anchors: dict[str, Mapping[str, str]] = {}
    for index, row in enumerate(rows):
        if row["reference_kind"] != "base_anchor":
            continue
        label = f"M2.7 base anchor[{index}]"
        case_id = row["case_id"]
        _need(case_id in cases and case_id not in anchors, f"{label}: identity drift")
        _need(row["method"] == "prepared_cgls_base_12", f"{label}: method drift")
        _need(_integer(row["model_seed"], label) == -1, f"{label}: model seed drift")
        _need(
            (_integer(row["optimization_forward_calls"], label),
             _integer(row["optimization_adjoint_calls"], label),
             _integer(row["grouped_adjoint_calls"], label),
             _integer(row["evaluation_forward_calls"], label)) == (12, 12, 0, 1),
            f"{label}: base-anchor call ledger drift",
        )
        for field in (
            "field_relative_l2",
            "h1_seminorm_relative_error",
            "measured_reprojection_relative_l2",
            "clean_reprojection_relative_l2",
        ):
            _need(_finite_float(row[field], f"{label}.{field}") >= 0.0, f"{label}: negative metric")
        anchors[case_id] = row
    _need(set(anchors) == cases and len(anchors) == 30, "base-anchor case catalog drift")
    return anchors


def _base_measured_anchor_lookup(
    rows: Sequence[Mapping[str, str]],
    cases: set[str],
) -> dict[str, float]:
    """Build the selector anchor without touching field or clean-renderer data."""

    anchors: dict[str, float] = {}
    for index, row in enumerate(rows):
        if row["reference_kind"] != "base_anchor":
            continue
        label = f"M2.7 observable base anchor[{index}]"
        case_id = row["case_id"]
        _need(case_id in cases and case_id not in anchors, f"{label}: identity drift")
        _need(row["method"] == "prepared_cgls_base_12", f"{label}: method drift")
        _need(_integer(row["model_seed"], label) == -1, f"{label}: model seed drift")
        measured = _finite_float(
            row["measured_reprojection_relative_l2"],
            f"{label}.measured_reprojection_relative_l2",
        )
        _need(measured >= 0.0, f"{label}: negative measured residual")
        anchors[case_id] = measured
    _need(set(anchors) == cases and len(anchors) == 30, "observable base-anchor catalog drift")
    return anchors


def _select_observable(
    points: Sequence[ObservablePoint],
    spec: CandidateSpec,
    base_measured_residual: float,
) -> Selection:
    if spec.comparator_only:
        iteration = int(spec.parameter)
        _need(0 <= iteration < len(points), "fixed comparator K outside trajectory")
        return Selection(iteration, iteration, True, None, None)

    threshold = float(spec.threshold)
    _need(math.isfinite(threshold) and threshold >= 0.0, "invalid stopping threshold")
    selected: ObservablePoint | None = None
    observed: float | None = None
    for point in points:
        if spec.family == "simulator_noise_floor_multiple":
            value = point.measured_residual
        elif spec.family == "base_anchor_residual_multiple":
            value = point.measured_residual / max(base_measured_residual, 1e-30)
        elif spec.family == "initial_system_residual_fraction":
            value = point.system_residual_fraction
        else:
            raise ValidationError(f"unsupported stopping family: {spec.family}")
        _need(math.isfinite(value) and value >= 0.0, "invalid stopping observable")
        if value <= threshold:
            selected = point
            observed = value
            break
    return Selection(
        selected_iteration=None if selected is None else selected.iteration,
        attempted_iteration=points[-1].iteration if selected is None else selected.iteration,
        crossed=selected is not None,
        observable_value=observed,
        threshold=threshold,
    )


def observable_selection_signature(
    *,
    config: Mapping[str, Any],
    trajectory_rows: Sequence[Mapping[str, str]],
    reference_rows: Sequence[Mapping[str, str]],
    noise_floor: float,
) -> dict[SelectedKey, tuple[int | None, bool]]:
    """Return selections while intentionally ignoring all field/clean columns."""

    groups, observable = _build_trajectory_groups(trajectory_rows, config)
    cases = {key[3] for key in groups}
    anchors = _base_measured_anchor_lookup(reference_rows, cases)
    specs = _candidate_specs(config, noise_floor)
    output: dict[SelectedKey, tuple[int | None, bool]] = {}
    for key, points in observable.items():
        method, seed, split, case_id = key
        base_residual = anchors[case_id]
        for spec in specs:
            selected = _select_observable(points, spec, base_residual)
            output[(spec.candidate_id, method, seed, split, case_id)] = (
                selected.selected_iteration,
                selected.crossed,
            )
    return output


def _best_h1_from_row(row: Mapping[str, str]) -> float:
    gain = _finite_float(row["h1_gain_to_best_matched_classical"], "H1 gain")
    denominator = 1.0 - gain
    _need(denominator > 1e-12, "cannot recover matched-classical H1 reference")
    return _finite_float(row["h1_seminorm_relative_error"], "H1 error") / denominator


def _validate_selected_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    groups: Mapping[GroupKey, Sequence[Mapping[str, str]]],
    observable: Mapping[GroupKey, Sequence[ObservablePoint]],
    anchors: Mapping[str, Mapping[str, str]],
    specs: Sequence[CandidateSpec],
    config: Mapping[str, Any],
) -> dict[SelectedKey, Mapping[str, str]]:
    spec_lookup = {spec.candidate_id: spec for spec in specs}
    selected_lookup: dict[SelectedKey, Mapping[str, str]] = {}
    maximum_iteration = int(config["trajectory"]["maximum_iteration"])
    harm_threshold = float(config["decision_gates"]["field_harm_threshold_fraction"])

    for index, row in enumerate(rows):
        label = f"selected_rows[{index}]"
        candidate_id = row["candidate_id"]
        _need(candidate_id in spec_lookup, f"{label}: candidate ID drift")
        spec = spec_lookup[candidate_id]
        method = row["method"]
        seed = _integer(row["model_seed"], f"{label}.model_seed")
        split = row["split"]
        case_id = row["case_id"]
        group_key = (method, seed, split, case_id)
        _need(group_key in groups, f"{label}: source trajectory identity drift")
        key = (candidate_id, method, seed, split, case_id)
        _need(key not in selected_lookup, f"duplicate selected row: {key}")
        selected_lookup[key] = row

        source_rows = groups[group_key]
        source_points = observable[group_key]
        base = anchors[case_id]
        base_residual = _finite_float(base["measured_reprojection_relative_l2"], f"{label}.base residual")
        expected_selection = _select_observable(source_points, spec, base_residual)

        _need(row["stopping_family"] == spec.family, f"{label}: stopping family drift")
        _close(row["stopping_parameter"], float(spec.parameter), f"{label}.stopping_parameter")
        _need(_boolean(row["comparator_only"], label) == spec.comparator_only, f"{label}: comparator flag drift")
        _need(
            _boolean(row["uses_simulator_nuisance_scale"], label)
            == spec.uses_simulator_nuisance_scale,
            f"{label}: simulator-scale flag drift",
        )
        _need(not _boolean(row["selection_uses_truth"], label), f"{label}: selector used truth")
        _need(
            not _boolean(row["selection_uses_clean_renderer"], label),
            f"{label}: selector used clean renderer",
        )
        _need(
            _boolean(row["target_crossed"], label) == expected_selection.crossed,
            f"{label}: first-crossing flag drift",
        )
        _need(
            _integer(row["selected_iteration"], f"{label}.selected_iteration")
            == (-1 if expected_selection.selected_iteration is None else expected_selection.selected_iteration),
            f"{label}: first-crossing selected iteration drift",
        )
        _need(
            _integer(row["attempted_iteration"], f"{label}.attempted_iteration")
            == expected_selection.attempted_iteration,
            f"{label}: attempted iteration drift",
        )
        if expected_selection.threshold is None:
            _need(row["stopping_threshold"] == "", f"{label}: fixed comparator threshold must be blank")
            _need(
                math.isnan(float(row["selected_observable_value"])),
                f"{label}: fixed comparator observable must be nan",
            )
        elif expected_selection.observable_value is None:
            _need(
                math.isnan(float(row["selected_observable_value"])),
                f"{label}: uncrossed observable must be nan",
            )
        else:
            _close(row["stopping_threshold"], expected_selection.threshold, f"{label}.threshold")
            _close(
                row["selected_observable_value"],
                expected_selection.observable_value,
                f"{label}.selected_observable_value",
            )

        if expected_selection.selected_iteration is None:
            source = source_rows[-1]
            expected_kind = "prepared_cgls_base_fallback"
            expected_field = _finite_float(base["field_relative_l2"], f"{label}.base field")
            expected_h1 = _finite_float(base["h1_seminorm_relative_error"], f"{label}.base H1")
            expected_measured = base_residual
            expected_clean = _finite_float(base["clean_reprojection_relative_l2"], f"{label}.base clean")
            best_field = min(
                _finite_float(source["matched_cgls_field_relative_l2"], label),
                _finite_float(source["matched_huber_field_relative_l2"], label),
            )
            best_h1 = _best_h1_from_row(source)
            matched_cgls_residual = _finite_float(source["measured_reprojection_relative_l2"], label) / max(
                _finite_float(source["reprojection_ratio_to_matched_cgls"], label), 1e-30
            )
            expected_field_gain = (best_field - expected_field) / max(best_field, 1e-30)
            expected_h1_gain = (best_h1 - expected_h1) / max(best_h1, 1e-30)
            expected_reprojection_ratio = expected_measured / max(matched_cgls_residual, 1e-30)
            expected_forward, expected_adjoint = 14 + maximum_iteration, 13 + maximum_iteration
        else:
            source = source_rows[expected_selection.selected_iteration]
            expected_kind = "selected_affine_pcg_iterate"
            expected_field = _finite_float(source["field_relative_l2"], label)
            expected_h1 = _finite_float(source["h1_seminorm_relative_error"], label)
            expected_measured = _finite_float(source["measured_reprojection_relative_l2"], label)
            expected_clean = _finite_float(source["clean_reprojection_relative_l2"], label)
            expected_field_gain = _finite_float(source["field_gain_to_best_matched_classical"], label)
            expected_h1_gain = _finite_float(source["h1_gain_to_best_matched_classical"], label)
            expected_reprojection_ratio = _finite_float(source["reprojection_ratio_to_matched_cgls"], label)
            expected_forward = 14 + expected_selection.selected_iteration
            expected_adjoint = 13 + expected_selection.selected_iteration

        _need(row["returned_field_kind"] == expected_kind, f"{label}: returned field kind drift")
        _need(
            row["family"] == source_rows[0]["family"]
            and _integer(row["base_seed"], label) == _integer(source_rows[0]["base_seed"], label),
            f"{label}: case metadata drift",
        )
        for field, expected in (
            ("field_relative_l2", expected_field),
            ("h1_seminorm_relative_error", expected_h1),
            ("measured_reprojection_relative_l2", expected_measured),
            ("clean_reprojection_relative_l2", expected_clean),
            ("field_gain_to_best_matched_classical", expected_field_gain),
            ("h1_gain_to_best_matched_classical", expected_h1_gain),
            ("reprojection_ratio_to_matched_cgls", expected_reprojection_ratio),
            (
                "projection_closure_relative_error",
                _finite_float(source["projection_closure_relative_error"], label),
            ),
        ):
            _close(row[field], expected, f"{label}.{field}")
        base_clean = max(_finite_float(base["clean_reprojection_relative_l2"], label), 1e-30)
        _close(
            row["clean_reprojection_ratio_to_base"],
            expected_clean / base_clean,
            f"{label}.clean_reprojection_ratio_to_base",
        )
        _need(
            _boolean(row["field_harm"], label) == (expected_field_gain < -harm_threshold),
            f"{label}: field-harm flag drift",
        )
        _need(
            (_integer(row["forward_calls"], label), _integer(row["adjoint_calls"], label))
            == (expected_forward, expected_adjoint),
            f"{label}: selected F/A call ledger drift",
        )
        _need(expected_forward <= 24 and expected_adjoint <= 23, f"{label}: selected call cap drift")
        _need(
            _integer(row["exact_camera_block_setup_forward_equivalents"], label) == 1001
            and not _boolean(row["exact_camera_block_setup_in_budget"], label),
            f"{label}: exact camera-block oracle entered budget",
        )

    expected_keys = {
        (spec.candidate_id, method, seed, split, case_id)
        for spec in specs
        for method, seed, split, case_id in groups
    }
    _need(set(selected_lookup) == expected_keys, "selected row identity grid drift")
    return selected_lookup


def _aggregate_selected(
    rows: Sequence[Mapping[str, str]],
) -> dict[tuple[str, str, int, str], dict[str, Any]]:
    groups: dict[tuple[str, str, int, str], list[Mapping[str, str]]] = {}
    for row in rows:
        key = (
            row["candidate_id"],
            row["method"],
            _integer(row["model_seed"], "aggregate seed"),
            row["split"],
        )
        groups.setdefault(key, []).append(row)
    output: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for key, values in groups.items():
        effective_iterations = [
            max(_integer(row["selected_iteration"], "selected K"), _integer(row["attempted_iteration"], "attempted K"))
            for row in values
        ]
        output[key] = {
            "candidate_id": key[0],
            "stopping_family": values[0]["stopping_family"],
            "stopping_parameter": _finite_float(values[0]["stopping_parameter"], "stopping parameter"),
            "comparator_only": _boolean(values[0]["comparator_only"], "comparator flag"),
            "method": key[1],
            "model_seed": key[2],
            "split": key[3],
            "case_count": len(values),
            "target_crossing_rate": _mean(float(_boolean(row["target_crossed"], "crossing")) for row in values),
            "selected_iteration_mean": _mean(effective_iterations),
            "selected_iteration_maximum": max(effective_iterations),
            "field_gain_mean": _mean(_finite_float(row["field_gain_to_best_matched_classical"], "field gain") for row in values),
            "h1_gain_mean": _mean(_finite_float(row["h1_gain_to_best_matched_classical"], "H1 gain") for row in values),
            "clean_reprojection_ratio_to_base_mean": _mean(_finite_float(row["clean_reprojection_ratio_to_base"], "clean ratio") for row in values),
            "clean_reprojection_ratio_to_base_maximum": max(_finite_float(row["clean_reprojection_ratio_to_base"], "clean ratio") for row in values),
            "reprojection_ratio_to_matched_cgls_mean": _mean(_finite_float(row["reprojection_ratio_to_matched_cgls"], "reprojection ratio") for row in values),
            "field_harm_rate": _mean(float(_boolean(row["field_harm"], "field harm")) for row in values),
            "worst_field_gain": min(_finite_float(row["field_gain_to_best_matched_classical"], "field gain") for row in values),
            "projection_closure_relative_error_maximum": max(_finite_float(row["projection_closure_relative_error"], "closure") for row in values),
            "forward_calls_mean": _mean(_integer(row["forward_calls"], "forward calls") for row in values),
            "forward_calls_maximum": max(_integer(row["forward_calls"], "forward calls") for row in values),
            "adjoint_calls_mean": _mean(_integer(row["adjoint_calls"], "adjoint calls") for row in values),
            "adjoint_calls_maximum": max(_integer(row["adjoint_calls"], "adjoint calls") for row in values),
        }
    return output


def _validate_aggregate_rows(
    actual_rows: Sequence[Mapping[str, str]],
    computed: Mapping[tuple[str, str, int, str], Mapping[str, Any]],
) -> None:
    actual: dict[tuple[str, str, int, str], Mapping[str, str]] = {}
    for row in actual_rows:
        key = (
            row["candidate_id"],
            row["method"],
            _integer(row["model_seed"], "aggregate model seed"),
            row["split"],
        )
        _need(key not in actual, f"duplicate aggregate row: {key}")
        actual[key] = row
    _need(set(actual) == set(computed), "aggregate identity grid drift")
    for key, expected in computed.items():
        row = actual[key]
        for field, value in expected.items():
            label = f"aggregate[{key!r}].{field}"
            if isinstance(value, bool):
                _need(_boolean(row[field], label) == value, f"{label}: boolean mismatch")
            elif isinstance(value, str):
                _need(row[field] == value, f"{label}: value mismatch")
            elif field in {"model_seed", "case_count", "selected_iteration_maximum", "forward_calls_maximum", "adjoint_calls_maximum"}:
                _need(_integer(row[field], label) == int(value), f"{label}: integer mismatch")
            else:
                _close(row[field], float(value), label)


def _pooled_metrics(
    rows: Sequence[Mapping[str, str]],
    *,
    candidate_id: str,
    method: str,
    split: str,
) -> dict[str, Any]:
    values = [
        row
        for row in rows
        if row["candidate_id"] == candidate_id
        and row["method"] == method
        and row["split"] == split
    ]
    _need(bool(values), f"missing pooled rows for {candidate_id}/{method}/{split}")
    seed_means = [
        _mean(
            _finite_float(row["field_gain_to_best_matched_classical"], "field gain")
            for row in values
            if _integer(row["model_seed"], "model seed") == seed
        )
        for seed in MODEL_SEEDS
    ]
    effective_iterations = [
        max(_integer(row["selected_iteration"], "selected K"), _integer(row["attempted_iteration"], "attempted K"))
        for row in values
    ]
    return {
        "row_count": len(values),
        "target_crossing_rate": _mean(float(_boolean(row["target_crossed"], "crossing")) for row in values),
        "selected_iteration_mean": _mean(effective_iterations),
        "field_gain_mean": _mean(_finite_float(row["field_gain_to_best_matched_classical"], "field gain") for row in values),
        "h1_gain_mean": _mean(_finite_float(row["h1_gain_to_best_matched_classical"], "H1 gain") for row in values),
        "clean_reprojection_ratio_to_base_mean": _mean(_finite_float(row["clean_reprojection_ratio_to_base"], "clean ratio") for row in values),
        "clean_reprojection_ratio_to_base_maximum": max(_finite_float(row["clean_reprojection_ratio_to_base"], "clean ratio") for row in values),
        "reprojection_ratio_to_matched_cgls_mean": _mean(_finite_float(row["reprojection_ratio_to_matched_cgls"], "reprojection ratio") for row in values),
        "field_harm_rate": _mean(float(_boolean(row["field_harm"], "harm")) for row in values),
        "worst_field_gain": min(_finite_float(row["field_gain_to_best_matched_classical"], "field gain") for row in values),
        "projection_closure_relative_error_maximum": max(_finite_float(row["projection_closure_relative_error"], "closure") for row in values),
        "forward_calls_maximum": max(_integer(row["forward_calls"], "forward calls") for row in values),
        "adjoint_calls_maximum": max(_integer(row["adjoint_calls"], "adjoint calls") for row in values),
        "per_model_seed_field_gain_means": seed_means,
    }


def _expected_decisions(
    rows: Sequence[Mapping[str, str]],
    specs: Sequence[CandidateSpec],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gates = config["decision_gates"]
    decisions: dict[str, Any] = {}
    for method in METHODS:
        screened: list[dict[str, Any]] = []
        for spec in specs:
            if spec.comparator_only:
                continue
            metrics = _pooled_metrics(rows, candidate_id=spec.candidate_id, method=method, split="development")
            checks = {
                "field_gain": metrics["field_gain_mean"] >= float(gates["development_field_gain_minimum"]),
                "h1_gain": metrics["h1_gain_mean"] >= float(gates["development_h1_gain_minimum"]),
                "clean_reprojection_mean": metrics["clean_reprojection_ratio_to_base_mean"] <= float(gates["development_clean_reprojection_ratio_to_base_mean_maximum"]),
                "clean_reprojection_worst": metrics["clean_reprojection_ratio_to_base_maximum"] <= float(gates["development_clean_reprojection_ratio_to_base_worst_maximum"]),
                "harm_rate": metrics["field_harm_rate"] <= float(gates["field_harm_rate_maximum"]),
                "worst_field_gain": metrics["worst_field_gain"] >= float(gates["worst_field_gain_minimum"]),
                "target_crossing": metrics["target_crossing_rate"] >= float(gates["minimum_target_crossing_rate"]),
                "closure": metrics["projection_closure_relative_error_maximum"] <= float(gates["maximum_projection_closure_relative_error"]),
                "forward_budget": metrics["forward_calls_maximum"] <= int(gates["maximum_forward_calls"]),
                "adjoint_budget": metrics["adjoint_calls_maximum"] <= int(gates["maximum_adjoint_calls"]),
                "all_seed_means_positive": all(value > 0.0 for value in metrics["per_model_seed_field_gain_means"]),
            }
            screened.append(
                {
                    "candidate_id": spec.candidate_id,
                    "stopping_family": spec.family,
                    "stopping_parameter": spec.parameter,
                    "development": metrics,
                    "development_checks": checks,
                    "development_eligible": all(checks.values()),
                }
            )
        eligible = [item for item in screened if item["development_eligible"]]
        eligible.sort(
            key=lambda item: (
                -float(item["development"]["field_gain_mean"]),
                float(item["development"]["selected_iteration_mean"]),
                str(item["candidate_id"]),
            )
        )
        if not eligible:
            decisions[method] = {
                "screened_candidates": screened,
                "selection": None,
                "passed_opened_n1_0_gate": False,
            }
            continue
        chosen = eligible[0]
        ood = _pooled_metrics(rows, candidate_id=chosen["candidate_id"], method=method, split="ood")
        ood_checks = {
            "field_gain": ood["field_gain_mean"] >= float(gates["ood_field_gain_minimum"]),
            "h1_gain": ood["h1_gain_mean"] >= float(gates["ood_h1_gain_minimum"]),
            "clean_reprojection_mean": ood["clean_reprojection_ratio_to_base_mean"] <= float(gates["ood_clean_reprojection_ratio_to_base_mean_maximum"]),
            "clean_reprojection_worst": ood["clean_reprojection_ratio_to_base_maximum"] <= float(gates["ood_clean_reprojection_ratio_to_base_worst_maximum"]),
            "harm_rate": ood["field_harm_rate"] <= float(gates["field_harm_rate_maximum"]),
            "worst_field_gain": ood["worst_field_gain"] >= float(gates["worst_field_gain_minimum"]),
            "target_crossing": ood["target_crossing_rate"] >= float(gates["minimum_target_crossing_rate"]),
            "closure": ood["projection_closure_relative_error_maximum"] <= float(gates["maximum_projection_closure_relative_error"]),
            "forward_budget": ood["forward_calls_maximum"] <= int(gates["maximum_forward_calls"]),
            "adjoint_budget": ood["adjoint_calls_maximum"] <= int(gates["maximum_adjoint_calls"]),
            "all_seed_means_positive": all(value > 0.0 for value in ood["per_model_seed_field_gain_means"]),
        }
        decisions[method] = {
            "screened_candidates": screened,
            "selection": {
                **chosen,
                "ood": ood,
                "ood_checks": ood_checks,
                "passed_ood_gate": all(ood_checks.values()),
            },
            "passed_opened_n1_0_gate": all(ood_checks.values()),
        }
    return decisions


def _expected_pareto(
    rows: Sequence[Mapping[str, str]],
    specs: Sequence[CandidateSpec],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gates = config["decision_gates"]
    output: dict[str, Any] = {}
    for method in METHODS:
        candidates: list[dict[str, Any]] = []
        for spec in specs:
            if spec.comparator_only:
                continue
            metrics = _pooled_metrics(rows, candidate_id=spec.candidate_id, method=method, split="development")
            tail_safe = (
                metrics["field_harm_rate"] <= float(gates["field_harm_rate_maximum"])
                and metrics["worst_field_gain"] >= float(gates["worst_field_gain_minimum"])
            )
            renderer_safe = (
                metrics["clean_reprojection_ratio_to_base_mean"]
                <= float(gates["development_clean_reprojection_ratio_to_base_mean_maximum"])
                and metrics["clean_reprojection_ratio_to_base_maximum"]
                <= float(gates["development_clean_reprojection_ratio_to_base_worst_maximum"])
            )
            candidates.append(
                {
                    "candidate_id": spec.candidate_id,
                    "stopping_family": spec.family,
                    "tail_safe": tail_safe,
                    "renderer_safe": renderer_safe,
                    "joint_safe": tail_safe and renderer_safe,
                    "metrics": metrics,
                }
            )
        tail = sorted(
            (item for item in candidates if item["tail_safe"]),
            key=lambda item: (
                float(item["metrics"]["clean_reprojection_ratio_to_base_mean"]),
                str(item["candidate_id"]),
            ),
        )
        renderer = sorted(
            (item for item in candidates if item["renderer_safe"]),
            key=lambda item: (
                -float(item["metrics"]["worst_field_gain"]),
                float(item["metrics"]["clean_reprojection_ratio_to_base_mean"]),
                str(item["candidate_id"]),
            ),
        )
        joint = [item for item in candidates if item["joint_safe"]]
        output[method] = {
            "candidate_count": len(candidates),
            "tail_safe_count": len(tail),
            "renderer_safe_count": len(renderer),
            "joint_safe_count": len(joint),
            "best_renderer_consistency_among_tail_safe": tail[0] if tail else None,
            "best_field_tail_among_renderer_safe": renderer[0] if renderer else None,
        }
    return output


def _validate_config_and_sources(
    config_path: Path,
    config: Mapping[str, Any],
) -> tuple[dict[str, Path], float]:
    _need(
        config["schema_version"]
        == "jacru-n1-0-observable-discrepancy-stopping-postopen-config-1.0",
        "N1.0 config schema drift",
    )
    _need(
        config["report_schema_version"]
        == "jacru-n1-0-observable-discrepancy-stopping-postopen-report-1.0",
        "N1.0 report schema drift",
    )
    _need(str(config["status"]).startswith("FROZEN_BEFORE_FIRST_"), "N1.0 config is not frozen")
    _need(config["methods"] == list(METHODS), "N1.0 method grid drift")
    trajectory = config["trajectory"]
    _need(trajectory["source_iterations"] == list(ITERATIONS), "N1.0 trajectory K grid drift")
    _need(trajectory["maximum_iteration"] == 10, "N1.0 maximum K drift")
    _need(trajectory["projection_variant"] == TRAJECTORY_VARIANT, "N1.0 variant drift")
    _need(trajectory["fallback"] == "prepared_cgls_base_12", "N1.0 fallback drift")
    _need(
        trajectory["fallback_returns_base_after_full_attempted_budget"] is True,
        "N1.0 fallback budget boundary drift",
    )
    _need(
        trajectory["selection_uses_truth"] is False
        and trajectory["selection_uses_clean_renderer"] is False
        and trajectory["selection_uses_ood"] is False,
        "N1.0 selector boundary drift",
    )
    budget = config["matched_budget"]
    _need(
        budget["learned_feature_preparation_forward_calls"] == 13
        and budget["learned_feature_preparation_adjoint_calls"] == 13,
        "N1.0 feature-call budget drift",
    )
    _need(
        budget["maximum_forward_calls"] == 24
        and budget["maximum_adjoint_calls"] == 23,
        "N1.0 maximum F/A budget drift",
    )
    _need(
        budget["exact_camera_block_setup_forward_equivalents"] == 1001
        and budget["exact_camera_block_setup_is_excluded_oracle_and_reported"] is True,
        "N1.0 exact camera-block budget boundary drift",
    )
    _need(budget["no_unobserved_final_adjoint_call"] is True, "N1.0 adjoint-call boundary drift")
    claim = config["claim_boundary"]
    _need(
        claim["exact_camera_block_is_deployable"] is False
        and claim["exact_camera_block_setup_is_in_matched_budget"] is False
        and claim["may_claim_runtime_or_efficiency"] is False
        and claim["may_claim_method_superiority"] is False
        and claim["may_claim_real_bost_generalization"] is False
        and claim["may_open_fresh_or_final"] is False,
        "N1.0 claim boundary drift",
    )

    sources = {
        "source_t0_config": ROOT / str(config["source_t0_config"]),
        "source_m2_7_config": ROOT / str(config["source_m2_7_config"]),
        "source_m2_7_summary": ROOT / str(config["source_m2_7_results"]) / "summary.json",
        "source_m2_8_config": ROOT / str(config["source_m2_8_config"]),
        "source_m2_8_summary": ROOT / str(config["source_m2_8_results"]) / "summary.json",
    }
    for name, path in sources.items():
        _need(path.is_file(), f"{name}: source missing")
        _need(_sha256(path) == config[f"{name}_sha256"], f"{name}: source hash drift")

    m27_config = _read_json(sources["source_m2_7_config"])
    m27_summary = _read_json(sources["source_m2_7_summary"])
    _need(m27_config["methods"] == list(METHODS), "M2.7 source method grid drift")
    _need(m27_config["preconditioner_oracle_only"] is True, "M2.7 source oracle boundary drift")
    variant = m27_config["projection"]["variants"]
    _need(len(variant) == 1 and variant[0]["name"] == TRAJECTORY_VARIANT, "M2.7 source variant drift")
    _need(
        float(variant[0]["damping_fraction_of_operator_norm_squared_bound"]) == 0.0,
        "M2.7 source damping must be zero",
    )
    _need(variant[0]["preconditioner"] == CAMERA_BLOCK_ORACLE, "M2.7 source preconditioner drift")
    _need(
        m27_config["matched_budget"]["dense_exact_camera_block_setup_is_excluded_oracle_and_reported"] is True
        and m27_config["claim_boundary"]["exact_camera_block_setup_is_in_matched_budget"] is False,
        "M2.7 exact camera-block oracle entered budget",
    )
    _need(m27_summary["status"] == M27_STATUS, "M2.7 source summary is not NO-GO")
    _need(
        all(
            m27_summary["authorization"][name] is False
            for name in (
                "claim_deployable_algorithm",
                "claim_method_superiority",
                "claim_real_bost_generalization",
                "open_fresh_or_final",
                "draft_new_preregistered_fresh_gate",
                "continue_deployable_preconditioner_estimation",
            )
        ),
        "M2.7 source authorization drift",
    )
    m28_summary = _read_json(sources["source_m2_8_summary"])
    _need(m28_summary["status"] == M28_STATUS, "M2.8 source summary is not NO-GO")
    _validate_manifest(ROOT / str(config["source_m2_7_results"]), M27_PAYLOADS)

    t0 = _read_json(sources["source_t0_config"])
    fixture = t0["fixture"]
    noise_floor = math.sqrt(
        float(fixture["noise_relative_std"]) ** 2
        + float(fixture["camera_bias_relative_std"]) ** 2
    )
    _need(math.isfinite(noise_floor) and noise_floor > 0.0, "simulator noise floor invalid")
    return sources, noise_floor


def validate_packet(*, config_path: Path = CONFIG, output_dir: Path = OUTPUT) -> dict[str, Any]:
    """Validate the N1.0 packet and independently reconstruct every selection."""

    config_path = Path(config_path)
    output_dir = Path(output_dir)
    _need(config_path.is_file() and output_dir.is_dir(), "N1.0 packet path missing")
    config = _read_json(config_path)
    sources, noise_floor = _validate_config_and_sources(config_path, config)
    _validate_manifest(output_dir, N1_PAYLOADS)
    try:
        readme = (output_dir / "README.md").read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValidationError(f"cannot read N1.0 README: {error}") from error
    _need(f"Status: `{PACKET_STATUS}`" in readme, "N1.0 README does not preserve NO-GO")

    m27_dir = ROOT / str(config["source_m2_7_results"])
    trajectory_rows = _read_csv_required(
        m27_dir / "metric_rows.csv", M27_REQUIRED_FIELDS, 1980
    )
    reference_rows = _read_csv_required(
        m27_dir / "reference_rows.csv", REFERENCE_REQUIRED_FIELDS, 570
    )
    groups, observable = _build_trajectory_groups(trajectory_rows, config)
    cases = {key[3] for key in groups}
    anchors = _base_anchor_lookup(reference_rows, cases)
    specs = _candidate_specs(config, noise_floor)
    _need(len(specs) == 37, "N1.0 candidate spec count drift")
    _need(sum(not spec.comparator_only for spec in specs) == 26, "N1.0 Pareto candidate count drift")

    selected_rows = _read_csv_exact(output_dir / "selected_rows.csv", SELECTED_FIELDS, 6660)
    selected_lookup = _validate_selected_rows(
        selected_rows,
        groups=groups,
        observable=observable,
        anchors=anchors,
        specs=specs,
        config=config,
    )
    _need(len(selected_lookup) == len(groups) * len(specs), "selected row count formula drift")

    aggregate_rows = _read_csv_exact(output_dir / "aggregate_rows.csv", AGGREGATE_FIELDS, 444)
    computed_aggregate = _aggregate_selected(selected_rows)
    _validate_aggregate_rows(aggregate_rows, computed_aggregate)

    decisions = _expected_decisions(selected_rows, specs, config)
    pareto = _expected_pareto(selected_rows, specs, config)
    _need(
        all(decision["selection"] is None for decision in decisions.values()),
        "N1.0 development selection must remain absent",
    )
    _need(
        all(decision["passed_opened_n1_0_gate"] is False for decision in decisions.values()),
        "N1.0 gate must remain NO-GO",
    )
    _need(
        all(pareto[method]["joint_safe_count"] == 0 for method in METHODS),
        "N1.0 Pareto joint-safe count must remain zero",
    )

    summary = _read_json(output_dir / "summary.json")
    _need(summary["schema_version"] == config["report_schema_version"], "N1.0 summary schema drift")
    _need(summary["status"] == PACKET_STATUS, "N1.0 summary status must remain NO-GO")
    _need(summary["evidence_level"] == config["evidence_level"], "N1.0 evidence level drift")
    _need(summary["source_config"] == str(config_path.relative_to(ROOT)), "N1.0 source config path drift")
    _need(summary["source_config_sha256"] == _sha256(config_path), "N1.0 source config hash drift")
    _deep_match(
        summary["source_hashes"],
        {name: _sha256(path) for name, path in sources.items()},
        "N1.0 source hashes",
    )
    _close(summary["simulator_relative_noise_floor"], noise_floor, "N1.0 simulator noise floor")
    _need(
        summary["trajectory_group_count"] == len(groups)
        and summary["candidate_spec_count"] == len(specs)
        and summary["row_count"] == len(selected_rows)
        and summary["aggregate_row_count"] == len(aggregate_rows),
        "N1.0 summary row/count ledger drift",
    )
    _need(summary["selector_observable_columns"] == list(SELECTOR_COLUMNS), "N1.0 selector columns drift")
    _need(
        not any(
            fragment in column.lower()
            for column in summary["selector_observable_columns"]
            for fragment in FORBIDDEN_SELECTOR_FRAGMENTS
        ),
        "N1.0 selector exposes field/clean/truth columns",
    )
    _deep_match(summary["decisions"], decisions, "N1.0 decisions")
    _deep_match(summary["pareto_audit"], pareto, "N1.0 Pareto audit")
    _deep_match(summary["authorization"], AUTHORIZATION, "N1.0 authorization")
    _deep_match(summary["claim_boundary"], config["claim_boundary"], "N1.0 claim boundary")
    _need(
        _finite_float(summary["runtime_seconds"], "N1.0 runtime") >= 0.0,
        "N1.0 runtime must be nonnegative",
    )

    return {
        "status": VALIDATED_STATUS,
        "trajectory_group_count": len(groups),
        "candidate_spec_count": len(specs),
        "selected_row_count": len(selected_rows),
        "aggregate_row_count": len(aggregate_rows),
        "pareto_joint_safe_count": {
            method: int(pareto[method]["joint_safe_count"]) for method in METHODS
        },
        "authorization": dict(AUTHORIZATION),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = validate_packet(config_path=args.config, output_dir=args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
