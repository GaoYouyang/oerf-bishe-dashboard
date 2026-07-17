#!/usr/bin/env python3
"""Independent validator for the PSU-B0 post-Gate-B exact-|A| diagnostic.

This module intentionally does not import the D0 diagnostic module, its runner,
or its classification helper.  It treats the emitted release as untrusted data:
checksums, frozen-source hashes, CSV coverage, call ledgers, summaries, and the
descriptive label are recomputed locally from primitive values.
"""

from __future__ import annotations

import argparse
import ast
import csv
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_SCHEMA = "psu-b0-exact-absolute-root-cause-config-1.0"
REPORT_SCHEMA = "psu-b0-exact-absolute-root-cause-report-1.0"
VALIDATOR_SCHEMA = "psu-b0-exact-absolute-independent-validator-1.0"
VALIDATOR_STATUS = "PASS_INDEPENDENT_EXACT_ABSOLUTE_DIAGNOSTIC_VALIDATION"
EXACT_ABSOLUTE_SCHEMA = "psu-b0-exact-absolute-diagnostic-1.0"
REPLICATES = (0, 8)
FAMILIES = (
    "plume", "wavy_front", "thin_front", "double_front", "annular_kernel",
    "oblique_shock", "vortex_pair", "multi_plume",
)
CHECKPOINTS = (4, 8, 16, 32, 64, 128)
METHODS = (
    "scalar_a_only_pdhg", "formal_factor_view_a_only_pdhg",
    "factor_row_hybrid_a_only_pdhg", "exact_abs_view_a_only_pdhg",
    "exact_abs_row_a_only_pdhg", "graph_pcgls",
)
PDHG_METHODS = METHODS[:-1]
RELEASE_FILES = {
    "report.json", "trajectory_rows.csv", "tightness_rows.csv", "audit_rows.json",
}
EXPECTED_RUNTIME_SHAPE = {
    "grid_size": 16,
    "view_count": 9,
    "rays_per_view": 256,
    "finite_aperture_sample_count": 8,
    "measurement_count": 4608,
    "support_active_voxel_count": 2744,
    "factor_majorizer_active_coordinate_count": 2322,
    "factor_majorizer_zero_coordinate_count": 422,
    "signed_operator_nullspace_dimension": "UNKNOWN_NOT_EQUAL_TO_ZERO_COORDINATE_COUNT",
}
OPERATOR_CONTRACT = {
    "solver_recurrence_operator": "SIGNED_A",
    "absolute_operator_role": "DIAGONAL_METRIC_ONLY",
    "factor_majorizer_relation": "ENTRYWISE_M_GREATER_OR_EQUAL_ABS_A",
    "factor_active_coordinates_are_nullspace_dimension": False,
    "power_iteration_role": "NONBINDING_STRESS_ESTIMATE_NOT_BOUND",
    "schur_certificate_role": "THEOREM_BACKED_SAFETY_UPPER_BOUND",
    "graph_pcgls_binding": False,
    "graph_full_support_matches_reduced_pdhg_support": False,
}
SHA256 = re.compile(r"[0-9a-f]{64}")
COMMIT = re.compile(r"[0-9a-f]{40}")


class ValidationError(AssertionError):
    """The evidence release does not meet its frozen diagnostic contract."""


class Validator:
    def __init__(self) -> None:
        self.check_count = 0

    def require(self, condition: bool, message: str) -> None:
        self.check_count += 1
        if not condition:
            raise ValidationError(message)

    def close(
        self, actual: Any, expected: Any, message: str, *,
        rel_tol: float = 1e-9, abs_tol: float = 1e-11,
    ) -> None:
        left, right = float(actual), float(expected)
        self.require(
            math.isfinite(left) and math.isfinite(right)
            and math.isclose(left, right, rel_tol=rel_tol, abs_tol=abs_tol),
            f"{message}: {left} != {right}",
        )


def _reject_constant(raw: str) -> None:
    raise ValidationError(f"non-finite JSON constant: {raw}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValidationError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def load_strict_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=_reject_constant,
        object_pairs_hook=_unique_object,
    )
    if not isinstance(value, dict):
        raise ValidationError(f"JSON root must be an object: {path}")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_bytes(root: Path, commit: str, raw_path: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "show", f"{commit}:{raw_path}"], cwd=root, check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        raise ValidationError(f"recorded commit lacks frozen source: {raw_path}") from error


def _safe_relative(root: Path, raw: str) -> Path:
    token = PurePosixPath(raw)
    if token.is_absolute() or not token.parts or ".." in token.parts:
        raise ValidationError(f"unsafe relative path: {raw}")
    target = (root / token).resolve()
    if root.resolve() not in target.parents:
        raise ValidationError(f"path escaped repository root: {raw}")
    return target


def parse_checksum_manifest(path: Path) -> dict[str, str]:
    recorded: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        pieces = line.split("  ", 1)
        if len(pieces) != 2 or not SHA256.fullmatch(pieces[0]):
            raise ValidationError("malformed checksum manifest")
        name = pieces[1]
        token = PurePosixPath(name)
        if token.is_absolute() or len(token.parts) != 1 or ".." in token.parts:
            raise ValidationError(f"unsafe checksum entry: {name}")
        if name in recorded:
            raise ValidationError(f"duplicate checksum entry: {name}")
        recorded[name] = pieces[0]
    if set(recorded) != RELEASE_FILES:
        raise ValidationError("checksum manifest file set drift")
    return recorded


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValidationError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValidationError(f"{name} must be numeric") from error
    if not math.isfinite(result):
        raise ValidationError(f"{name} must be finite")
    return result


def _integer(value: Any, name: str) -> int:
    number = _finite_number(value, name)
    if not number.is_integer():
        raise ValidationError(f"{name} must be an integer")
    return int(number)


def _boolean(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValidationError(f"{name} must be a boolean")


def _mapping_cell(value: Any, name: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{name} must contain a mapping")
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as error:
        raise ValidationError(f"{name} is not a safe mapping literal") from error
    if not isinstance(parsed, dict) or any(not isinstance(key, str) for key in parsed):
        raise ValidationError(f"{name} must contain a string-keyed mapping")
    return parsed


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(None in row for row in rows):
        raise ValidationError(f"empty or malformed CSV: {path.name}")
    return rows


def validate_config(config_path: Path, root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    config = load_strict_json(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValidationError("config schema drift")
    if tuple(config.get("replicate_indices", ())) != REPLICATES:
        raise ValidationError("replicate set drift")
    if tuple(config.get("reaction_families", ())) != FAMILIES:
        raise ValidationError("reaction family set drift")
    if tuple(config.get("checkpoints", ())) != CHECKPOINTS:
        raise ValidationError("checkpoint set drift")
    if tuple(config.get("methods", ())) != METHODS:
        raise ValidationError("method set drift")
    if config.get("expected_runtime_shape") != EXPECTED_RUNTIME_SHAPE:
        raise ValidationError("runtime shape or nullspace boundary drift")
    if config.get("exact_absolute", {}).get("full_dense_production_matrix_materialized") is not False:
        raise ValidationError("production dense-matrix materialization is not forbidden")
    if config.get("classification_policy", {}).get("causal_krylov_proof_claimed") is not False:
        raise ValidationError("causal graph claim is not fail-closed")
    if config.get("classification_policy", {}).get("graph_used_in_primary_same_operator_decision") is not False:
        raise ValidationError("graph comparison is treated as binding")
    boundary = config.get("claim_boundary")
    required_boundary = {
        "formal_gate_b_reopened": False, "new_algorithm_claimed": False,
        "algorithm_superiority_claimed": False, "generalization_claimed": False,
    }
    if not isinstance(boundary, dict) or any(boundary.get(k) != v for k, v in required_boundary.items()):
        raise ValidationError("claim boundary drift")
    paths, expected = config.get("source_paths"), config.get("source_sha256")
    if not isinstance(paths, dict) or not isinstance(expected, dict) or set(paths) != set(expected):
        raise ValidationError("source path/hash map drift")
    for key, raw in paths.items():
        target = _safe_relative(root, str(raw))
        if not target.is_file() or target.is_symlink():
            raise ValidationError(f"missing or symlinked source: {key}")
        if file_sha256(target) != expected[key]:
            raise ValidationError(f"source hash mismatch: {key}")
    return config


def _expected_metric_keys() -> set[str]:
    return {
        "replicate", "sample_index", "reaction_family", "method", "iterations",
        "forward_calls", "adjoint_calls", "field_relative_l2", "gradient_relative_l2",
        "front_top10_f1", "data_coupled_relative_l2", "data_null_support_relative_l2",
        "data_coupled_error_energy", "data_null_support_error_energy",
        "data_null_support_reconstruction_energy", "normalized_data_residual_l2",
        "trajectory_elapsed_seconds",
    }


def validate_metric_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    expected = {(r, s, family, method, k)
                for r in REPLICATES for s, family in enumerate(FAMILIES)
                for method in METHODS for k in CHECKPOINTS}
    seen: set[tuple[int, int, str, str, int]] = set()
    for row in rows:
        if set(row) != _expected_metric_keys():
            raise ValidationError("metric CSV schema drift")
        replicate = _integer(row["replicate"], "replicate")
        sample = _integer(row["sample_index"], "sample_index")
        method = str(row["method"])
        iteration = _integer(row["iterations"], "iterations")
        family = str(row["reaction_family"])
        key = (replicate, sample, family, method, iteration)
        if key in seen:
            raise ValidationError(f"duplicate metric row: {key}")
        seen.add(key)
        if key not in expected:
            raise ValidationError(f"unexpected metric row: {key}")
        if _integer(row["forward_calls"], "forward_calls") != iteration:
            raise ValidationError("metric forward calls are not exact-K")
        if _integer(row["adjoint_calls"], "adjoint_calls") != iteration:
            raise ValidationError("metric adjoint calls are not exact-K")
        numeric_names = _expected_metric_keys() - {"replicate", "sample_index", "reaction_family", "method", "iterations", "forward_calls", "adjoint_calls"}
        if method == "graph_pcgls":
            numeric_names.remove("normalized_data_residual_l2")
            if row["normalized_data_residual_l2"] not in {None, ""}:
                raise ValidationError("graph residual entered primary PDHG arithmetic")
        for name in numeric_names:
            value = _finite_number(row[name], name)
            if name.endswith("relative_l2") or name.endswith("energy") or name == "trajectory_elapsed_seconds":
                if value < 0.0:
                    raise ValidationError(f"negative metric: {name}")
            if name == "front_top10_f1" and not 0.0 <= value <= 1.0:
                raise ValidationError("front_top10_f1 escaped [0, 1]")
    if seen != expected:
        raise ValidationError(f"metric row coverage drift: got {len(seen)}, expected {len(expected)}")


def recompute_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), _integer(row["iterations"], "iterations"))].append(row)
    result: list[dict[str, Any]] = []
    for method in sorted(METHODS):
        for iteration in CHECKPOINTS:
            values = grouped[(method, iteration)]
            if len(values) != len(REPLICATES) * len(FAMILIES):
                raise ValidationError("summary coverage is incomplete")
            field = np.asarray([_finite_number(x["field_relative_l2"], "field_relative_l2") for x in values], dtype=np.float64)
            gradient = [_finite_number(x["gradient_relative_l2"], "gradient_relative_l2") for x in values]
            f1 = [_finite_number(x["front_top10_f1"], "front_top10_f1") for x in values]
            residual_values = [x["normalized_data_residual_l2"] for x in values]
            result.append({
                "method": method, "iterations": iteration, "sample_count": len(values),
                "mean_field_relative_l2": statistics.fmean(field),
                "p90_field_relative_l2": float(np.quantile(field, 0.90)),
                "mean_gradient_relative_l2": statistics.fmean(gradient),
                "mean_front_top10_f1": statistics.fmean(f1),
                "mean_normalized_data_residual_l2": (
                    None if any(value in {None, ""} for value in residual_values)
                    else statistics.fmean(_finite_number(value, "normalized_data_residual_l2") for value in residual_values)
                ),
            })
    return result


def recompute_decision(
    rows: Sequence[Mapping[str, Any]], tightness_rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute descriptive labels; graph is a comparator, never a cause."""
    by_key = {( _integer(row["replicate"], "replicate"), _integer(row["sample_index"], "sample_index"),
                str(row["method"]), _integer(row["iterations"], "iterations")): row for row in rows}
    if len(by_key) != len(rows):
        raise ValidationError("duplicate decision row")
    def mean(method: str, iteration: int) -> float:
        values = [_finite_number(by_key[(r, s, method, iteration)]["field_relative_l2"], "field_relative_l2")
                  for r in REPLICATES for s in range(len(FAMILIES))]
        return statistics.fmean(values)
    endpoint = _integer(thresholds["descriptive_endpoint_k"], "descriptive endpoint")
    formal = "formal_factor_view_a_only_pdhg"
    variants = {
        "factor_row_hybrid": "factor_row_hybrid_a_only_pdhg",
        "exact_abs_view": "exact_abs_view_a_only_pdhg",
        "exact_abs_row": "exact_abs_row_a_only_pdhg",
    }
    def residual(method: str, replicate: int | None = None, sample: int | None = None) -> float:
        if replicate is None:
            return statistics.fmean(_finite_number(by_key[(r, s, method, endpoint)]["normalized_data_residual_l2"], "normalized_data_residual_l2") for r in REPLICATES for s in range(len(FAMILIES)))
        return _finite_number(by_key[(replicate, sample, method, endpoint)]["normalized_data_residual_l2"], "normalized_data_residual_l2")
    formal_mean = residual(formal)
    gains = {name: 100.0 * (formal_mean - residual(method)) / formal_mean for name, method in variants.items()}
    counts = {name: sum(100.0 * (residual(formal, r, s) - residual(method, r, s)) / max(residual(formal, r, s), 1e-30) >= float(thresholds["material_residual_gain_percent_min"]) for r in REPLICATES for s in range(len(FAMILIES))) for name, method in variants.items()}
    high_slack = statistics.median(max(1.0 - _finite_number(row["row_ratio_p05"], "row_ratio_p05"), 1.0 - _finite_number(row["column_ratio_p05"], "column_ratio_p05")) for row in tightness_rows)
    material_gain, material_count = float(thresholds["material_residual_gain_percent_min"]), int(thresholds["material_paired_sample_count_min"])
    exact_graph_ratio = mean("exact_abs_row_a_only_pdhg", endpoint) / mean("graph_pcgls", endpoint)
    exact_view_material = gains["exact_abs_view"] >= material_gain and counts["exact_abs_view"] >= material_count and high_slack >= float(thresholds["material_high_quantile_slack_min"])
    factor_row_material = gains["factor_row_hybrid"] >= material_gain and counts["factor_row_hybrid"] >= material_count
    exact_row_material = gains["exact_abs_row"] >= material_gain and counts["exact_abs_row"] >= material_count
    if exact_view_material:
        status = "FACTOR_MAJORIZER_CANCELLATION_MATERIAL_DESCRIPTIVE"
    elif factor_row_material:
        status = "VIEW_AGGREGATION_MATERIAL_DESCRIPTIVE"
    elif exact_row_material:
        status = "COMBINED_STATIC_DIAGONAL_MATERIAL_DESCRIPTIVE"
    elif gains["exact_abs_row"] < material_gain and exact_graph_ratio >= float(thresholds["nonbinding_exact_to_graph_field_ratio_min"]):
        status = "STATIC_DIAGONAL_GAIN_SMALL_GRAPH_HEADROOM_NONBINDING"
    else:
        status = "INCONCLUSIVE_MIXED_MECHANISM"
    return {
        "status": status, "formal_gate_b_reopened": False, "descriptive_endpoint_k": endpoint,
        "mean_normalized_residual_gain_percent": gains, "paired_material_gain_count": counts,
        "material_gain_threshold_percent": material_gain, "material_paired_sample_count_min": material_count,
        "median_high_quantile_factor_slack": high_slack,
        "material_factor_slack": high_slack >= float(thresholds["material_high_quantile_slack_min"]),
        "exact_abs_row_to_graph_field_error_ratio_nonbinding": exact_graph_ratio,
        "graph_comparison_binding": False, "graph_support_contract_matches_pdhg": False,
        "causal_krylov_explanation_claimed": False,
        "claim": "OPENED_SYNTHETIC_SAME_SIGNED_A_DIAGONAL_DIAGNOSTIC_ONLY_NO_NEW_ALGORITHM_NO_EXPERIMENTAL_OR_GENERALIZATION_CLAIM",
    }


def validate_tightness_rows(rows: Sequence[Mapping[str, Any]], thresholds: Mapping[str, Any]) -> None:
    if len(rows) != len(REPLICATES) * len(FAMILIES):
        raise ValidationError("tightness row coverage drift")
    seen: set[tuple[int, int]] = set()
    expected = {(r, s) for r in REPLICATES for s in range(len(FAMILIES))}
    required = {
        "replicate", "sample_index", "reaction_family", "row_ratio_minimum",
        "row_ratio_p05", "row_ratio_median", "row_ratio_mean",
        "column_ratio_minimum", "column_ratio_p05", "column_ratio_median",
        "column_ratio_mean", "global_exact_to_factor_mass_ratio",
        "global_slack_mass", "exact_zero_row_count", "exact_zero_column_count",
        "factor_only_nonzero_count", "exact_only_nonzero_count",
        "factor_majorizer_active_coordinate_count", "signed_A_nonzero_coordinate_count",
        "M_active_A_zero_coordinate_count", "nullspace_dimension_claimed",
        "dominance_violation_maximum", "dominance_relative_violation_maximum",
        "setup_factor_row_relative_error", "setup_factor_column_relative_error",
        "audit_content_sha256", "mps_repeat_content_sha256",
        "mps_repeat_required_for_this_row", "solver_mps_setup_call_ledger",
        "audit_cpu64_setup_call_ledger", "exact_streaming_replay_call_ledger",
    }
    nonnumeric = {
        "replicate", "sample_index", "reaction_family", "audit_content_sha256",
        "mps_repeat_content_sha256", "mps_repeat_required_for_this_row",
        "nullspace_dimension_claimed", "factor_majorizer_active_coordinate_count",
        "signed_A_nonzero_coordinate_count", "M_active_A_zero_coordinate_count",
        "solver_mps_setup_call_ledger", "audit_cpu64_setup_call_ledger",
        "exact_streaming_replay_call_ledger",
    }
    for row in rows:
        if set(row) != required:
            raise ValidationError("tightness CSV schema drift")
        key = (
            _integer(row["replicate"], "replicate"),
            _integer(row["sample_index"], "sample_index"),
        )
        if key in seen or key not in expected or row["reaction_family"] != FAMILIES[key[1]]:
            raise ValidationError("tightness identity drift")
        seen.add(key)
        if not SHA256.fullmatch(str(row["audit_content_sha256"])):
            raise ValidationError("invalid audit content hash")
        if _boolean(row["nullspace_dimension_claimed"], "nullspace_dimension_claimed"):
            raise ValidationError("A coordinate activity was treated as nullspace dimension")
        expected_active = EXPECTED_RUNTIME_SHAPE["factor_majorizer_active_coordinate_count"]
        m_active = _integer(
            row["factor_majorizer_active_coordinate_count"],
            "M-active column count",
        )
        a_nonzero = _integer(
            row["signed_A_nonzero_coordinate_count"],
            "A-nonzero column count",
        )
        m_only = _integer(
            row["M_active_A_zero_coordinate_count"],
            "M-active A-zero column count",
        )
        if m_active != expected_active or a_nonzero != expected_active or m_only != 0:
            raise ValidationError("M-active and A-nonzero coordinate labels drift")
        for name in required - nonnumeric:
            if _finite_number(row[name], name) < 0.0:
                raise ValidationError(f"negative tightness value: {name}")
        if (
            _finite_number(
                row["dominance_violation_maximum"],
                "dominance_violation_maximum",
            )
            > float(thresholds["dominance_absolute_tolerance"])
            and _finite_number(
                row["dominance_relative_violation_maximum"],
                "dominance_relative_violation_maximum",
            )
            > float(thresholds["dominance_relative_tolerance"])
        ):
            raise ValidationError("M does not dominate |A|")
        if max(
            _finite_number(
                row["setup_factor_row_relative_error"],
                "setup_factor_row_relative_error",
            ),
            _finite_number(
                row["setup_factor_column_relative_error"],
                "setup_factor_column_relative_error",
            ),
        ) > float(thresholds["factor_replay_relative_error_maximum"]):
            raise ValidationError("factor row/column replay drift")
        if _integer(row["exact_only_nonzero_count"], "exact_only_nonzero_count") != 0:
            raise ValidationError("|A| has nonzero entries outside M")
        replay = _mapping_cell(
            row["exact_streaming_replay_call_ledger"],
            "exact_streaming_replay_call_ledger",
        )
        expected_batches = math.ceil(m_active / 128)
        expected_replay = {
            "signed_data_forward_calls": expected_batches,
            "signed_data_transpose_calls": 0,
            "absolute_data_forward_calls": expected_batches,
            "absolute_data_transpose_calls": 0,
            "signed_tv_forward_calls": 0,
            "signed_tv_transpose_calls": 0,
            "absolute_tv_forward_calls": 0,
            "absolute_tv_transpose_calls": 0,
        }
        if set(replay) != set(expected_replay) or any(
            _integer(replay[name], name) != value
            for name, value in expected_replay.items()
        ):
            raise ValidationError("exact-|A| streaming replay ledger drift")
        for ledger_name in (
            "solver_mps_setup_call_ledger",
            "audit_cpu64_setup_call_ledger",
        ):
            setup_ledger = _mapping_cell(row[ledger_name], ledger_name)
            if not setup_ledger or any(
                _integer(value, f"{ledger_name} {name}") < 0
                for name, value in setup_ledger.items()
            ):
                raise ValidationError(f"invalid factor setup ledger: {ledger_name}")
        repeat_required = _boolean(
            row["mps_repeat_required_for_this_row"],
            "mps_repeat_required_for_this_row",
        )
        expected_repeat = key == (REPLICATES[0], 0)
        if repeat_required != expected_repeat:
            raise ValidationError("CPU64 repeated-audit row identity drift")
        repeat_hash = str(row["mps_repeat_content_sha256"])
        if expected_repeat and repeat_hash != row["audit_content_sha256"]:
            raise ValidationError("CPU64 repeated-audit content hash drift")
        if not expected_repeat and repeat_hash:
            raise ValidationError("unexpected repeated-audit hash on non-probe row")
    if seen != expected:
        raise ValidationError("tightness rows are incomplete")


def validate_audit_rows(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> None:
    expected_method = {(r, s, method) for r in REPLICATES for s in range(len(FAMILIES)) for method in METHODS}
    expected_power = {(r, s, mode) for r in REPLICATES for s in range(len(FAMILIES)) for mode in ("factor_row", "exact_view", "exact_row")}
    methods: set[tuple[int, int, str]] = set()
    powers: set[tuple[int, int, str]] = set()
    for row in rows:
        replicate, sample = _integer(row.get("replicate"), "replicate"), _integer(row.get("sample_index"), "sample_index")
        if "mode" in row:
            required = {
                "replicate", "sample_index", "mode",
                "normalized_norm_squared_power_estimate", "power_value_is_upper_bound",
                "schur_certificate_squared_upper_bound",
                "schur_certificate_is_theorem_backed", "eta_squared",
                "power_call_ledger", "power_estimate_device",
                "power_estimate_dtype", "solver_metric_device",
                "solver_metric_dtype",
            }
            if set(row) != required:
                raise ValidationError("power audit schema drift")
            key = (replicate, sample, str(row["mode"]))
            if key in powers or key not in expected_power:
                raise ValidationError("power audit coverage drift")
            powers.add(key)
            estimate = _finite_number(row["normalized_norm_squared_power_estimate"], "power estimate")
            if estimate < 0.0:
                raise ValidationError("negative power estimate")
            if row["power_value_is_upper_bound"] is not False:
                raise ValidationError("power iteration was presented as an upper bound")
            if row["schur_certificate_is_theorem_backed"] is not True:
                raise ValidationError("Schur certificate was not marked theorem-backed")
            if (
                row["power_estimate_device"] != "cpu"
                or row["power_estimate_dtype"] != "torch.float64"
                or row["solver_metric_device"] != "mps"
                or row["solver_metric_dtype"] != "torch.float32"
            ):
                raise ValidationError("CPU64 audit and MPS solver device contract drift")
            certificate = _finite_number(row["schur_certificate_squared_upper_bound"], "Schur certificate")
            if certificate > _finite_number(row["eta_squared"], "eta squared") + 1e-10:
                raise ValidationError("Schur certificate violates eta safety bound")
            # It is explicitly an estimate, never a certificate or a causal result.
            ledger = row["power_call_ledger"]
            if not isinstance(ledger, dict):
                raise ValidationError("missing power call ledger")
            count = int(config["exact_absolute"]["power_iterations"])
            expected_power_ledger = {
                "signed_data_forward_calls": count, "signed_data_transpose_calls": count,
                "absolute_data_forward_calls": 0, "absolute_data_transpose_calls": 0,
                "signed_tv_forward_calls": 0, "signed_tv_transpose_calls": 0,
                "absolute_tv_forward_calls": 0, "absolute_tv_transpose_calls": 0,
            }
            if set(ledger) != set(expected_power_ledger) or any(_integer(ledger[name], name) != value for name, value in expected_power_ledger.items()):
                raise ValidationError("power iteration call ledger drift")
            if estimate > float(config["thresholds"]["power_estimate_sanity_maximum"]):
                raise ValidationError("reported power estimate exceeds diagnostic stress threshold")
        else:
            graph = str(row.get("method")) == "graph_pcgls"
            required = ({"replicate", "sample_index", "method", "ledger", "elapsed_seconds"}
                        if graph else {"replicate", "sample_index", "method", "solver_ledger", "scorer_ledger", "elapsed_seconds"})
            if set(row) != required:
                raise ValidationError("method call ledger schema drift")
            key = (replicate, sample, str(row["method"]))
            if key in methods or key not in expected_method:
                raise ValidationError("method call ledger coverage drift")
            methods.add(key)
            ledger = row["ledger"] if graph else row["solver_ledger"]
            scorer = {} if graph else row["scorer_ledger"]
            if not isinstance(ledger, dict) or not isinstance(scorer, dict):
                raise ValidationError("missing method call ledger")
            if graph:
                if set(ledger) != {"forward_calls", "adjoint_calls"}:
                    raise ValidationError("graph call ledger schema drift")
                forward, adjoint = ledger["forward_calls"], ledger["adjoint_calls"]
            else:
                expected_ledger = {
                    "signed_data_forward_calls", "signed_data_transpose_calls",
                    "absolute_data_forward_calls", "absolute_data_transpose_calls",
                    "signed_tv_forward_calls", "signed_tv_transpose_calls",
                    "absolute_tv_forward_calls", "absolute_tv_transpose_calls",
                }
                if set(ledger) != expected_ledger:
                    raise ValidationError("PDHG call ledger schema drift")
                forward, adjoint = ledger["signed_data_forward_calls"], ledger["signed_data_transpose_calls"]
                for name in expected_ledger - {"signed_data_forward_calls", "signed_data_transpose_calls"}:
                    if _integer(ledger[name], name) != 0:
                        raise ValidationError("unexpected non-data PDHG operator call")
            if _integer(forward, "forward ledger") != max(CHECKPOINTS):
                raise ValidationError("method forward ledger is not exact-K")
            if _integer(adjoint, "adjoint ledger") != max(CHECKPOINTS):
                raise ValidationError("method adjoint ledger is not exact-K")
            scorer_expected = {
                "signed_data_forward_calls": len(CHECKPOINTS), "signed_data_transpose_calls": 0,
                "absolute_data_forward_calls": 0, "absolute_data_transpose_calls": 0,
                "signed_tv_forward_calls": 0, "signed_tv_transpose_calls": 0,
                "absolute_tv_forward_calls": 0, "absolute_tv_transpose_calls": 0,
            }
            if not graph and (set(scorer) != set(scorer_expected) or any(_integer(scorer[name], name) != value for name, value in scorer_expected.items())):
                raise ValidationError("PDHG scorer ledger drift")
            if _finite_number(row["elapsed_seconds"], "elapsed_seconds") < 0.0:
                raise ValidationError("negative ledger elapsed time")
    if methods != expected_method or powers != expected_power:
        raise ValidationError("audit ledger rows are incomplete")


def _compare_nested(validator: Validator, observed: Any, expected: Any, label: str) -> None:
    if isinstance(expected, dict):
        validator.require(isinstance(observed, dict) and set(observed) == set(expected), f"{label} key drift")
        for key in expected:
            _compare_nested(validator, observed[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, float):
        validator.close(observed, expected, label, rel_tol=1e-8, abs_tol=1e-10)
    else:
        validator.require(observed == expected, f"{label} drift")


def validate_release(
    evidence: Path, *, config_path: Path = REPOSITORY_ROOT / "demo_t16_operator/configs/psu_b0_exact_absolute_root_cause_v3_cpu64_audit_amendment.json",
    root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Validate a completed D0 release without executing its production solver."""
    validator = Validator()
    config = validate_config(config_path, root)
    evidence = evidence.resolve()
    manifest = parse_checksum_manifest(evidence / "checksums.sha256")
    for name, digest in manifest.items():
        target = evidence / name
        validator.require(target.is_file() and not target.is_symlink(), f"missing release file: {name}")
        validator.require(file_sha256(target) == digest, f"checksum mismatch: {name}")
    report = load_strict_json(evidence / "report.json")
    validator.require(report.get("schema_version") == REPORT_SCHEMA, "report schema drift")
    validator.require(report.get("exact_absolute_schema") == EXACT_ABSOLUTE_SCHEMA, "exact-|A| schema drift")
    validator.require(report.get("config_sha256") == file_sha256(config_path), "config checksum drift")
    validator.require(report.get("source_sha256") == config["source_sha256"], "release source hashes drift")
    validator.require(isinstance(report.get("source_commit"), str) and COMMIT.fullmatch(report["source_commit"]) is not None, "invalid source commit")
    config_relative = config_path.resolve().relative_to(root.resolve()).as_posix()
    validator.require(file_sha256(config_path) == hashlib.sha256(_git_bytes(root, report["source_commit"], config_relative)).hexdigest(), "recorded commit config hash drift")
    for key, raw_path in config["source_paths"].items():
        validator.require(hashlib.sha256(_git_bytes(root, report["source_commit"], raw_path)).hexdigest() == config["source_sha256"][key], f"recorded commit source hash drift: {key}")
    validator.require(report.get("claim_boundary") == config["claim_boundary"], "release claim boundary drift")
    validator.require(report.get("data_contract") == config["data_contract"], "release data contract drift")
    validator.require(report.get("operator_contract") == OPERATOR_CONTRACT, "release operator contract drift")
    validator.require(report.get("scientific_claim_boundary") == "POST_NO_GO_OPENED_SYNTHETIC_ROOT_CAUSE_DIAGNOSTIC_ONLY", "scientific boundary drift")
    environment = report.get("environment")
    validator.require(
        isinstance(environment, dict)
        and environment.get("device") == "mps"
        and environment.get("dtype") == "torch.float32"
        and environment.get("audit_device") == "cpu"
        and environment.get("audit_dtype") == "torch.float64",
        "CPU64 audit and MPS solver environment drift",
    )
    metric_rows, tightness_rows = load_csv_rows(evidence / "trajectory_rows.csv"), load_csv_rows(evidence / "tightness_rows.csv")
    audit_rows = json.loads((evidence / "audit_rows.json").read_text(encoding="utf-8"), parse_constant=_reject_constant, object_pairs_hook=_unique_object)
    validator.require(isinstance(audit_rows, list), "audit rows must be a list")
    validate_metric_rows(metric_rows)
    validate_tightness_rows(tightness_rows, config["thresholds"])
    validate_audit_rows(audit_rows, config)
    summaries = recompute_summaries(metric_rows)
    decision = recompute_decision(metric_rows, tightness_rows, config["thresholds"])
    _compare_nested(validator, report.get("summaries"), summaries, "summaries")
    _compare_nested(validator, report.get("decision"), decision, "decision")
    validator.require(report.get("status") == decision["status"], "report status drift")
    validator.require(decision["formal_gate_b_reopened"] is False, "D0 reopened formal Gate B")
    validator.require(report.get("metric_row_count") == len(metric_rows), "metric row count drift")
    validator.require(report.get("tightness_row_count") == len(tightness_rows), "tightness row count drift")
    validator.require(report.get("power_row_count") == len(REPLICATES) * len(FAMILIES) * 3, "power row count drift")
    validator.require(report.get("call_row_count") == len(REPLICATES) * len(FAMILIES) * len(METHODS), "call row count drift")
    return {"schema_version": VALIDATOR_SCHEMA, "status": VALIDATOR_STATUS, "checks": validator.check_count, "decision": decision}


def primitive_exact_absolute_certificate(
    signed_a: np.ndarray, factor_m: np.ndarray, *, eta: float = 0.7,
) -> dict[str, Any]:
    """Dense float64 fixture oracle for entrywise, adjoint, and Schur checks."""
    a, m = np.asarray(signed_a, dtype=np.float64), np.asarray(factor_m, dtype=np.float64)
    if a.ndim != 2 or a.shape != m.shape or not np.isfinite(a).all() or not np.isfinite(m).all():
        raise ValidationError("primitive matrices must be finite matching 2D arrays")
    if np.any(m < np.abs(a)):
        raise ValidationError("primitive M does not dominate |A|")
    m_rows, m_columns = m.sum(axis=1), m.sum(axis=0)
    a_rows, a_columns = np.abs(a).sum(axis=1), np.abs(a).sum(axis=0)
    m_active = np.flatnonzero(m_columns > 0.0)
    a_nonzero = np.flatnonzero(a_columns > 0.0)
    if not len(a_nonzero):
        raise ValidationError("primitive A must have nonzero columns")
    tau, sigma = eta / m_columns[m_active], eta / m_rows
    reduced = a[:, m_active]
    scaled = np.sqrt(sigma)[:, None] * reduced * np.sqrt(tau)[None, :]
    norm_squared = float(np.linalg.svd(scaled, compute_uv=False)[0] ** 2)
    return {
        "exact_row_sums": a_rows, "exact_column_sums": a_columns,
        "factor_row_sums": m_rows, "factor_column_sums": m_columns,
        "m_active_columns": m_active, "a_nonzero_columns": a_nonzero,
        "m_active_a_null_columns": np.setdiff1d(m_active, a_nonzero),
        "nullspace_dimension": int(reduced.shape[1] - np.linalg.matrix_rank(reduced)),
        "schur_norm_squared": norm_squared, "eta_squared": eta ** 2,
    }


def replay_primal_dual_recurrence(
    signed_a: np.ndarray, target: np.ndarray, tau: np.ndarray, sigma: np.ndarray, *, steps: int, theta: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent float64 replay of the D0 data-only primal-dual recurrence."""
    a = np.asarray(signed_a, dtype=np.float64)
    b, tau, sigma = (np.asarray(x, dtype=np.float64) for x in (target, tau, sigma))
    if a.ndim != 2 or b.shape != (a.shape[0],) or tau.shape != (a.shape[1],) or sigma.shape != (a.shape[0],):
        raise ValidationError("recurrence shape mismatch")
    x = np.zeros(a.shape[1], dtype=np.float64)
    x_bar, dual = x.copy(), np.zeros(a.shape[0], dtype=np.float64)
    for _ in range(steps):
        dual = (dual + sigma * (a @ x_bar - b)) / (1.0 + sigma)
        dual = np.where(sigma > 0.0, dual, 0.0)
        next_x = x - tau * (a.T @ dual)
        x_bar, x = next_x + theta * (next_x - x), next_x
    return x, x_bar, dual


def power_iteration_estimate(signed_a: np.ndarray, tau: np.ndarray, sigma: np.ndarray, *, iterations: int, seed: int) -> float:
    """Return an estimate only; this function is deliberately not a certificate."""
    a = np.asarray(signed_a, dtype=np.float64)
    scaled = np.sqrt(np.asarray(sigma))[:, None] * a * np.sqrt(np.asarray(tau))[None, :]
    rng, vector = np.random.default_rng(seed), np.random.default_rng(seed).standard_normal(a.shape[1])
    del rng
    vector /= np.linalg.norm(vector)
    for _ in range(iterations):
        vector = scaled.T @ (scaled @ vector)
        vector /= np.linalg.norm(vector)
    return float(np.linalg.norm(scaled @ vector) ** 2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "demo_t16_operator/configs/psu_b0_exact_absolute_root_cause_v3_cpu64_audit_amendment.json")
    args = parser.parse_args()
    print(json.dumps(validate_release(args.evidence, config_path=args.config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
