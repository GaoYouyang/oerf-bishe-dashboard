#!/usr/bin/env python3
"""Independently validate the frozen JACRU M2.2 exact-oracle packet.

The validator deliberately does not import the experiment runner.  Passing this
audit establishes only dense numerical-nullspace headroom on the opened 12^3
synthetic fixture; it cannot authorize deployment, method superiority, fresh
data, or real-BOST claims.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT
    / "demo_t16_operator/configs/"
    "jacru_m2_2_exact_nullspace_oracle_postopen_v1.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "demo_t16_operator/results/"
    "jacru_m2_2_exact_nullspace_oracle_postopen_public"
)

CONFIG_SCHEMA = "jacru-m2-2-exact-nullspace-oracle-postopen-config-1.0"
REPORT_SCHEMA = "jacru-m2-2-exact-nullspace-oracle-postopen-report-1.0"
EXPECTED_CONFIG_STATUS = "FROZEN_BEFORE_FIRST_EXACT_ORACLE_EXECUTION"
EXPECTED_REPORT_STATUS = "M2_2_EXACT_NULLSPACE_HEADROOM_FOUND_ORACLE_ONLY"
VALIDATED_STATUS = "VALIDATED_M2_2_EXACT_NULLSPACE_HEADROOM_ORACLE_ONLY"
EXPECTED_EVIDENCE_LEVEL = (
    "E1_OPENED_T0_DENSE_NULLSPACE_HEADROOM_ORACLE_NO_FRESH"
)

METHODS = ("jacru_m2", "pooled_cnn")
MODEL_SEEDS = (17, 29, 43)
SPLITS = ("development", "ood")
SPLIT_CASE_COUNTS = {"development": 12, "ood": 18}
EXPECTED_ORACLE_ROWS = 180
EXPECTED_REFERENCE_ROWS = 30
EXPECTED_ZERO_ROWS = 180
EXPECTED_AGGREGATES = 12
EXPECTED_DECISIONS = 2
EXPECTED_GEOMETRIES = 12
EXPECTED_MATRIX_SHAPE = (150, 1000)
EXPECTED_RANK = 150
EXPECTED_NULLITY = 850

CONFIG_FIELDS = {
    "schema_version",
    "status",
    "frozen_date",
    "evidence_level",
    "source_t0_config",
    "source_t0_config_sha256",
    "source_t0_results",
    "source_t0_summary_sha256",
    "methods",
    "reference",
    "dense_oracle",
    "decision_gates",
    "claim_boundary",
}
SUMMARY_FIELDS = {
    "schema_version",
    "status",
    "evidence_level",
    "source_config_sha256",
    "source_t0_config_sha256",
    "source_t0_summary_sha256",
    "device",
    "elapsed_seconds",
    "metric_row_count",
    "reference_row_count",
    "zero_step_source_reproduction",
    "dense_setup_ledger",
    "training_runs",
    "aggregate",
    "decisions",
    "authorization",
    "claim_boundary",
    "public_export_policy",
}
AUTHORIZATION = {
    "claim_deployable_algorithm": False,
    "claim_method_superiority": False,
    "claim_real_bost_generalization": False,
    "open_fresh_or_final": False,
    "continue_matrix_free_projection_research": True,
}
CLAIM_BOUNDARY = {
    "is_dense_headroom_oracle": True,
    "is_deployable_algorithm": False,
    "is_runtime_or_efficiency_evidence": False,
    "is_confirmatory_or_final": False,
    "is_experimental_reconstruction": False,
    "is_cfd_validation": False,
    "is_real_bost_generalization": False,
    "approximate_inverse_kernel_equals_true_optical_kernel": False,
    "opens_fresh_or_final": False,
    "may_only_authorize_matrix_free_approximation_research": True,
}
PUBLIC_EXPORT_POLICY = {
    "contains_model_checkpoints": False,
    "contains_restricted_papers": False,
    "contains_private_experimental_arrays": False,
}
CHECKSUM_PAYLOADS = {
    "README.md",
    "aggregate_rows.csv",
    "diagnostic.pdf",
    "diagnostic.png",
    "metric_rows.csv",
    "reference_rows.csv",
    "summary.json",
    "zero_step_reproduction.csv",
}

COMMON_METRIC_FIELDS = (
    "case_id",
    "split",
    "family",
    "base_seed",
    "method",
    "model_seed",
    "field_relative_l2",
    "field_rmse",
    "field_nrmse_dynamic_range",
    "field_mean_bias",
    "h1_seminorm_relative_error",
    "measured_reprojection_relative_l2",
    "clean_reprojection_relative_l2",
    "gate",
    "correction_rms",
    "optimization_forward_calls",
    "optimization_adjoint_calls",
    "grouped_adjoint_calls",
    "evaluation_forward_calls",
    "neural_inference_seconds",
)
ORACLE_FIELDS = COMMON_METRIC_FIELDS + (
    "reference_method",
    "reference_field_relative_l2",
    "reference_h1_relative_error",
    "reference_measured_reprojection_relative_l2",
    "reference_clean_reprojection_relative_l2",
    "original_field_relative_l2",
    "original_h1_relative_error",
    "original_measured_reprojection_ratio_to_reference",
    "field_gain_to_reference",
    "h1_gain_to_reference",
    "original_field_gain_to_reference",
    "original_gain_retention",
    "measured_reprojection_ratio_to_reference",
    "clean_reprojection_ratio_to_reference",
    "row_correction_energy_fraction",
    "null_correction_energy_fraction",
    "visible_null_correction_fraction",
    "numerical_rank",
    "active_voxel_count",
    "numerical_nullity_lower_bound",
    "internal_projection_residual",
    "nullspace_residual",
    "field_harm_to_reference",
    "oracle_setup_excluded_from_reconstruction_budget",
)
ZERO_FIELDS = (
    "method",
    "model_seed",
    "case_id",
    "field_absolute_delta",
    "reprojection_absolute_delta",
)
AGGREGATE_FIELDS = (
    "method",
    "model_seed",
    "split",
    "case_count",
    "reference_field_relative_l2_mean",
    "original_field_relative_l2_mean",
    "oracle_field_relative_l2_mean",
    "oracle_h1_relative_error_mean",
    "field_gain_to_reference_mean",
    "h1_gain_to_reference_mean",
    "original_gain_retention_mean",
    "measured_reprojection_ratio_to_reference_mean",
    "clean_reprojection_ratio_to_reference_mean",
    "null_correction_energy_fraction_mean",
    "row_correction_energy_fraction_mean",
    "visible_null_correction_fraction_maximum",
    "internal_projection_residual_maximum",
    "field_gain_to_reference_minimum",
    "field_harm_rate",
)


class ValidationError(RuntimeError):
    """Raised when the evidence packet violates its frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot read JSON object {path}: {error}") from error
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _read_csv(path: Path, fields: tuple[str, ...]) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            _require(
                tuple(reader.fieldnames or ()) == fields,
                f"{path.name}: columns differ from the frozen schema",
            )
            return list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValidationError(f"cannot read CSV {path}: {error}") from error


def _csv_int(value: str, *, path: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValidationError(f"{path}: expected integer") from error
    _require(str(parsed) == value, f"{path}: non-canonical integer")
    return parsed


def _csv_float(value: str, *, path: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValidationError(f"{path}: expected number") from error
    _require(math.isfinite(parsed), f"{path}: expected finite number")
    return parsed


def _json_float(value: Any, *, path: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{path}: expected number",
    )
    parsed = float(value)
    _require(math.isfinite(parsed), f"{path}: expected finite number")
    return parsed


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    _require(bool(materialized), "cannot average an empty collection")
    return math.fsum(materialized) / len(materialized)


def _compare(actual: Any, expected: Any, *, path: str) -> None:
    if isinstance(expected, bool):
        _require(actual is expected, f"{path}: boolean mismatch")
    elif isinstance(expected, int):
        _require(type(actual) is int and actual == expected, f"{path}: integer mismatch")
    elif isinstance(expected, float):
        observed = _json_float(actual, path=path)
        _require(
            math.isclose(observed, expected, rel_tol=5e-11, abs_tol=5e-12),
            f"{path}: numeric mismatch ({observed!r} != {expected!r})",
        )
    elif isinstance(expected, str):
        _require(actual == expected, f"{path}: string mismatch")
    elif isinstance(expected, list):
        _require(isinstance(actual, list), f"{path}: expected list")
        _require(len(actual) == len(expected), f"{path}: list length mismatch")
        for index, expected_value in enumerate(expected):
            _compare(actual[index], expected_value, path=f"{path}[{index}]")
    elif isinstance(expected, dict):
        _require(isinstance(actual, dict), f"{path}: expected object")
        _require(set(actual) == set(expected), f"{path}: object keys mismatch")
        for key, expected_value in expected.items():
            _compare(actual[key], expected_value, path=f"{path}.{key}")
    else:
        _require(actual == expected, f"{path}: value mismatch")


def _compare_csv_number(actual: str, expected: float | int, *, path: str) -> None:
    if isinstance(expected, int):
        _require(_csv_int(actual, path=path) == expected, f"{path}: integer mismatch")
        return
    observed = _csv_float(actual, path=path)
    _require(
        math.isclose(observed, expected, rel_tol=5e-11, abs_tol=5e-12),
        f"{path}: numeric mismatch ({observed!r} != {expected!r})",
    )


def _validate_checksum_manifest(output_dir: Path) -> None:
    manifest = output_dir / "checksums.sha256"
    try:
        lines = manifest.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValidationError(f"cannot read checksum manifest: {error}") from error
    entries: dict[str, str] = {}
    pattern = re.compile(r"^([0-9a-f]{64})  ([A-Za-z0-9_.-]+)$")
    for line in lines:
        match = pattern.fullmatch(line)
        _require(match is not None, "checksums.sha256: malformed entry")
        assert match is not None
        digest, filename = match.groups()
        _require(filename not in entries, f"checksums.sha256: duplicate {filename}")
        entries[filename] = digest
    _require(set(entries) == CHECKSUM_PAYLOADS, "checksums.sha256: payload set mismatch")
    for filename, expected in entries.items():
        path = output_dir / filename
        _require(path.is_file(), f"checksum payload missing: {filename}")
        _require(_sha256(path) == expected, f"checksum mismatch: {filename}")


def _validate_config(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    _require(set(config) == CONFIG_FIELDS, "config top-level schema drift")
    _require(config["schema_version"] == CONFIG_SCHEMA, "config schema drift")
    _require(config["status"] == EXPECTED_CONFIG_STATUS, "config is not frozen")
    _require(config["evidence_level"] == EXPECTED_EVIDENCE_LEVEL, "evidence level drift")
    _require(tuple(config["methods"]) == METHODS, "frozen method set drift")
    _compare(
        config["reference"],
        {"method": "cgls", "iterations": 24},
        path="config.reference",
    )
    _compare(
        config["dense_oracle"],
        {
            "maximum_grid_voxels": 1728,
            "assembly_batch_size": 256,
            "dtype": "float64",
            "rank_relative_tolerance": 1e-10,
            "rank_absolute_tolerance": 0.0,
            "factorize_once_per_geometry": True,
        },
        path="config.dense_oracle",
    )
    expected_gates = {
        "development_field_gain_to_reference_minimum_fraction": 0.05,
        "development_h1_gain_to_reference_minimum_fraction": 0.03,
        "development_original_gain_retention_minimum_fraction": 0.25,
        "development_reprojection_ratio_to_reference_maximum": 1.000001,
        "ood_field_gain_to_reference_minimum_fraction": 0.02,
        "ood_h1_gain_to_reference_minimum_fraction": 0.0,
        "ood_original_gain_retention_minimum_fraction": 0.25,
        "ood_reprojection_ratio_to_reference_maximum": 1.000001,
        "field_harm_threshold_fraction": 0.01,
        "field_harm_rate_maximum": 0.05,
        "worst_field_gain_minimum_fraction": -0.05,
        "maximum_internal_projection_residual": 1e-10,
        "maximum_visible_null_correction_fraction": 1e-10,
        "require_all_model_seed_mean_field_gains_positive": True,
    }
    _compare(config["decision_gates"], expected_gates, path="config.decision_gates")
    _compare(config["claim_boundary"], CLAIM_BOUNDARY, path="config.claim_boundary")

    source_config_path = (ROOT / str(config["source_t0_config"])).resolve()
    _require(source_config_path.is_relative_to(ROOT), "source T0 config escapes repository")
    _require(source_config_path.is_file(), "source T0 config is missing")
    _require(
        _sha256(source_config_path) == config["source_t0_config_sha256"],
        "source T0 config hash drift",
    )
    source_summary_path = (
        ROOT / str(config["source_t0_results"]) / "summary.json"
    ).resolve()
    _require(source_summary_path.is_relative_to(ROOT), "source T0 results escape repository")
    _require(source_summary_path.is_file(), "source T0 summary is missing")
    _require(
        _sha256(source_summary_path) == config["source_t0_summary_sha256"],
        "source T0 summary hash drift",
    )
    _require(config_path.is_file(), "oracle config is missing")
    source_config = _load_json(source_config_path)
    _require(tuple(source_config["training"]["model_seeds"]) == MODEL_SEEDS, "source model seed grid drift")
    _require(tuple(source_config["fixture"]["grid_shape"]) == (12, 12, 12), "source grid shape drift")
    _require(tuple(source_config["fixture"]["detector_shape"]) == (5, 5), "source detector shape drift")
    _require(set(METHODS).issubset(source_config["methods"]), "source methods are incomplete")
    return source_config


def _verify_source_metric_checksum(source_dir: Path) -> None:
    manifest = source_dir / "checksums.sha256"
    _require(manifest.is_file(), "source T0 checksum manifest is missing")
    expected: str | None = None
    try:
        lines = manifest.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValidationError(f"cannot read source checksum manifest: {error}") from error
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  metric_rows\.csv", line)
        if match:
            _require(expected is None, "source T0 metric checksum is duplicated")
            expected = match.group(1)
    _require(expected is not None, "source T0 metric checksum is missing")
    _require(
        _sha256(source_dir / "metric_rows.csv") == expected,
        "source T0 metric checksum mismatch",
    )


def _source_catalog(
    rows: list[dict[str, str]],
    source_config: dict[str, Any],
) -> tuple[
    dict[str, tuple[str, str, int]],
    dict[tuple[str, int, str], dict[str, str]],
]:
    catalog: dict[str, tuple[str, str, int]] = {}
    learned: dict[tuple[str, int, str], dict[str, str]] = {}
    for index, row in enumerate(rows):
        seed = _csv_int(row["model_seed"], path=f"source[{index}].model_seed")
        if row["method"] == "cgls_13" and seed == -1:
            _require(row["case_id"] not in catalog, f"duplicate source case: {row['case_id']}")
            catalog[row["case_id"]] = (
                row["split"],
                row["family"],
                _csv_int(row["base_seed"], path=f"source[{index}].base_seed"),
            )
        if row["method"] in METHODS and seed in MODEL_SEEDS:
            key = (row["method"], seed, row["case_id"])
            _require(key not in learned, f"duplicate source learned row: {key}")
            learned[key] = row

    _require(len(catalog) == EXPECTED_REFERENCE_ROWS, "source case count drift")
    for split, expected_count in SPLIT_CASE_COUNTS.items():
        actual = [value for value in catalog.values() if value[0] == split]
        _require(len(actual) == expected_count, f"source {split} case count drift")
        split_config = source_config["splits"][split]
        expected = {
            (split, family, int(seed))
            for seed in split_config["base_seeds"]
            for family in split_config["families"]
        }
        _require(set(actual) == expected, f"source {split} seed/family grid drift")
    expected_learned = {
        (method, seed, case_id)
        for method in METHODS
        for seed in MODEL_SEEDS
        for case_id in catalog
    }
    _require(set(learned) == expected_learned, "source learned row grid drift")
    return catalog, learned


def _validate_reference_rows(
    rows: list[dict[str, str]],
    catalog: dict[str, tuple[str, str, int]],
) -> dict[str, dict[str, str]]:
    _require(len(rows) == EXPECTED_REFERENCE_ROWS, "expected 30 reference rows")
    lookup: dict[str, dict[str, str]] = {}
    nonnegative = (
        "field_relative_l2",
        "field_rmse",
        "field_nrmse_dynamic_range",
        "h1_seminorm_relative_error",
        "measured_reprojection_relative_l2",
        "clean_reprojection_relative_l2",
        "neural_inference_seconds",
    )
    for index, row in enumerate(rows):
        case_id = row["case_id"]
        _require(case_id in catalog, f"reference_rows[{index}]: unknown case")
        _require(case_id not in lookup, f"duplicate reference row: {case_id}")
        split, family, base_seed = catalog[case_id]
        _require(row["split"] == split, f"reference_rows[{index}]: split mismatch")
        _require(row["family"] == family, f"reference_rows[{index}]: family mismatch")
        _require(
            _csv_int(row["base_seed"], path=f"reference_rows[{index}].base_seed") == base_seed,
            f"reference_rows[{index}]: base seed mismatch",
        )
        _require(row["method"] == "cgls_24_reference", "reference method drift")
        _require(_csv_int(row["model_seed"], path=f"reference_rows[{index}].model_seed") == -1, "reference model seed drift")
        for field in nonnegative:
            _require(_csv_float(row[field], path=f"reference_rows[{index}].{field}") >= 0.0, f"reference_rows[{index}].{field}: negative")
        _csv_float(row["field_mean_bias"], path=f"reference_rows[{index}].field_mean_bias")
        _require(row["gate"] == "", "reference gate must remain empty")
        _require(row["correction_rms"] == "", "reference correction RMS must remain empty")
        _require(_csv_int(row["optimization_forward_calls"], path="reference.optimization_forward_calls") == 24, "reference forward budget drift")
        _require(_csv_int(row["optimization_adjoint_calls"], path="reference.optimization_adjoint_calls") == 24, "reference adjoint budget drift")
        _require(_csv_int(row["grouped_adjoint_calls"], path="reference.grouped_adjoint_calls") == 0, "reference grouped-adjoint budget drift")
        _require(_csv_int(row["evaluation_forward_calls"], path="reference.evaluation_forward_calls") == 1, "reference evaluation budget drift")
        lookup[case_id] = row
    _require(set(lookup) == set(catalog), "reference case grid drift")
    return lookup


def _validate_oracle_rows(
    rows: list[dict[str, str]],
    catalog: dict[str, tuple[str, str, int]],
    references: dict[str, dict[str, str]],
    source_rows: dict[tuple[str, int, str], dict[str, str]],
    gates: dict[str, Any],
) -> dict[tuple[str, int, str], dict[str, str]]:
    _require(len(rows) == EXPECTED_ORACLE_ROWS, "expected 180 oracle rows")
    lookup: dict[tuple[str, int, str], dict[str, str]] = {}
    nonnegative = (
        "field_relative_l2",
        "field_rmse",
        "field_nrmse_dynamic_range",
        "h1_seminorm_relative_error",
        "measured_reprojection_relative_l2",
        "clean_reprojection_relative_l2",
        "correction_rms",
        "neural_inference_seconds",
        "reference_field_relative_l2",
        "reference_h1_relative_error",
        "reference_measured_reprojection_relative_l2",
        "reference_clean_reprojection_relative_l2",
        "original_field_relative_l2",
        "original_h1_relative_error",
        "original_measured_reprojection_ratio_to_reference",
        "row_correction_energy_fraction",
        "null_correction_energy_fraction",
        "visible_null_correction_fraction",
        "internal_projection_residual",
        "nullspace_residual",
    )
    harm_threshold = float(gates["field_harm_threshold_fraction"])
    projection_limit = float(gates["maximum_internal_projection_residual"])
    visible_limit = float(gates["maximum_visible_null_correction_fraction"])
    ratio_tolerance = max(
        float(gates["development_reprojection_ratio_to_reference_maximum"]) - 1.0,
        float(gates["ood_reprojection_ratio_to_reference_maximum"]) - 1.0,
    )
    for index, row in enumerate(rows):
        method = row["method"]
        seed = _csv_int(row["model_seed"], path=f"metric_rows[{index}].model_seed")
        case_id = row["case_id"]
        _require(method in METHODS, f"metric_rows[{index}]: unknown method")
        _require(seed in MODEL_SEEDS, f"metric_rows[{index}]: unknown model seed")
        _require(case_id in catalog, f"metric_rows[{index}]: unknown case")
        key = (method, seed, case_id)
        _require(key not in lookup, f"duplicate oracle row: {key}")
        split, family, base_seed = catalog[case_id]
        _require(row["split"] == split, f"metric_rows[{index}]: split mismatch")
        _require(row["family"] == family, f"metric_rows[{index}]: family mismatch")
        _require(_csv_int(row["base_seed"], path=f"metric_rows[{index}].base_seed") == base_seed, f"metric_rows[{index}]: base seed mismatch")
        for field in nonnegative:
            _require(_csv_float(row[field], path=f"metric_rows[{index}].{field}") >= 0.0, f"metric_rows[{index}].{field}: negative")
        _csv_float(row["field_mean_bias"], path=f"metric_rows[{index}].field_mean_bias")
        gate = _csv_float(row["gate"], path=f"metric_rows[{index}].gate")
        _require(0.0 <= gate <= 1.0, f"metric_rows[{index}].gate: outside [0, 1]")
        _require(_csv_int(row["optimization_forward_calls"], path="oracle.optimization_forward_calls") == 24, "oracle forward budget drift")
        _require(_csv_int(row["optimization_adjoint_calls"], path="oracle.optimization_adjoint_calls") == 24, "oracle adjoint budget drift")
        _require(_csv_int(row["grouped_adjoint_calls"], path="oracle.grouped_adjoint_calls") == 1, "oracle grouped-adjoint budget drift")
        _require(_csv_int(row["evaluation_forward_calls"], path="oracle.evaluation_forward_calls") == 1, "oracle evaluation budget drift")
        _require(row["reference_method"] == "cgls_24", "oracle reference label drift")
        _require(row["oracle_setup_excluded_from_reconstruction_budget"] == "True", "dense setup budget boundary drift")

        rank = _csv_int(row["numerical_rank"], path=f"metric_rows[{index}].numerical_rank")
        active = _csv_int(row["active_voxel_count"], path=f"metric_rows[{index}].active_voxel_count")
        nullity = _csv_int(row["numerical_nullity_lower_bound"], path=f"metric_rows[{index}].numerical_nullity_lower_bound")
        _require(rank == EXPECTED_RANK, "oracle row numerical rank drift")
        _require(active == EXPECTED_MATRIX_SHAPE[1], "oracle row active voxel count drift")
        _require(nullity == EXPECTED_NULLITY and nullity == active - rank, "oracle row rank/nullity drift")

        reference = references[case_id]
        source = source_rows[key]
        ref_field = float(reference["field_relative_l2"])
        ref_h1 = float(reference["h1_seminorm_relative_error"])
        ref_measured = float(reference["measured_reprojection_relative_l2"])
        ref_clean = float(reference["clean_reprojection_relative_l2"])
        field = float(row["field_relative_l2"])
        h1 = float(row["h1_seminorm_relative_error"])
        measured = float(row["measured_reprojection_relative_l2"])
        clean = float(row["clean_reprojection_relative_l2"])
        original_field = float(source["field_relative_l2"])
        original_h1 = float(source["h1_seminorm_relative_error"])
        original_measured = float(source["measured_reprojection_relative_l2"])
        original_gain = (ref_field - original_field) / ref_field
        oracle_gain = (ref_field - field) / ref_field
        expected_values = {
            "reference_field_relative_l2": ref_field,
            "reference_h1_relative_error": ref_h1,
            "reference_measured_reprojection_relative_l2": ref_measured,
            "reference_clean_reprojection_relative_l2": ref_clean,
            "original_field_relative_l2": original_field,
            "original_h1_relative_error": original_h1,
            "original_measured_reprojection_ratio_to_reference": original_measured / max(ref_measured, 1e-30),
            "field_gain_to_reference": oracle_gain,
            "h1_gain_to_reference": (ref_h1 - h1) / ref_h1,
            "original_field_gain_to_reference": original_gain,
            "original_gain_retention": oracle_gain / max(original_gain, 1e-30),
            "measured_reprojection_ratio_to_reference": measured / max(ref_measured, 1e-30),
            "clean_reprojection_ratio_to_reference": clean / max(ref_clean, 1e-30),
        }
        for field_name, expected in expected_values.items():
            _compare_csv_number(row[field_name], expected, path=f"metric_rows[{index}].{field_name}")

        row_fraction = float(row["row_correction_energy_fraction"])
        null_fraction = float(row["null_correction_energy_fraction"])
        _require(row_fraction <= 1.0 + 1e-10, "row correction energy fraction exceeds one")
        _require(null_fraction <= 1.0 + 1e-10, "null correction energy fraction exceeds one")
        _require(math.isclose(row_fraction**2 + null_fraction**2, 1.0, rel_tol=0.0, abs_tol=1e-10), "row/null correction decomposition is not orthogonal")
        visible = float(row["visible_null_correction_fraction"])
        internal = float(row["internal_projection_residual"])
        nullspace = float(row["nullspace_residual"])
        _require(visible <= visible_limit, "visible null correction residual exceeds frozen gate")
        _require(internal <= projection_limit, "internal projection residual exceeds frozen gate")
        _require(nullspace <= projection_limit, "nullspace residual exceeds frozen gate")
        measured_ratio = float(row["measured_reprojection_ratio_to_reference"])
        clean_ratio = float(row["clean_reprojection_ratio_to_reference"])
        _require(abs(measured_ratio - 1.0) <= ratio_tolerance + 1e-12, "oracle measured reprojection is not preserved")
        _require(abs(clean_ratio - 1.0) <= ratio_tolerance + 1e-12, "oracle clean reprojection is not preserved")
        expected_harm = int(field > ref_field * (1.0 + harm_threshold))
        _compare_csv_number(row["field_harm_to_reference"], expected_harm, path=f"metric_rows[{index}].field_harm_to_reference")
        lookup[key] = row

    expected = {
        (method, seed, case_id)
        for method in METHODS
        for seed in MODEL_SEEDS
        for case_id in catalog
    }
    _require(set(lookup) == expected, "oracle method/seed/case grid drift")
    return lookup


def _validate_zero_step(
    rows: list[dict[str, str]],
    oracle: dict[tuple[str, int, str], dict[str, str]],
    source: dict[tuple[str, int, str], dict[str, str]],
) -> dict[str, Any]:
    _require(len(rows) == EXPECTED_ZERO_ROWS, "expected 180 zero-step reproduction rows")
    lookup: dict[tuple[str, int, str], dict[str, str]] = {}
    for index, row in enumerate(rows):
        key = (
            row["method"],
            _csv_int(row["model_seed"], path=f"zero_step_reproduction[{index}].model_seed"),
            row["case_id"],
        )
        _require(key in oracle and key in source, f"zero-step source row missing: {key}")
        _require(key not in lookup, f"duplicate zero-step row: {key}")
        field_delta = _csv_float(row["field_absolute_delta"], path=f"zero_step_reproduction[{index}].field_absolute_delta")
        reprojection_delta = _csv_float(row["reprojection_absolute_delta"], path=f"zero_step_reproduction[{index}].reprojection_absolute_delta")
        _require(field_delta == 0.0, "zero-step field reproduction is not exact")
        _require(reprojection_delta == 0.0, "zero-step reprojection reproduction is not exact")
        _require(oracle[key]["original_field_relative_l2"] == source[key]["field_relative_l2"], "zero-step field does not exactly reproduce source T0")
        _require(oracle[key]["original_h1_relative_error"] == source[key]["h1_seminorm_relative_error"], "zero-step H1 does not exactly reproduce source T0")
        lookup[key] = row
    _require(set(lookup) == set(oracle), "zero-step method/seed/case grid drift")
    return {
        "row_count": len(rows),
        "maximum_field_absolute_delta": 0.0,
        "maximum_reprojection_absolute_delta": 0.0,
        "passed_1e_6": True,
    }


def _validate_geometry_ledger(ledger: Any, dense_config: dict[str, Any]) -> int:
    _require(isinstance(ledger, dict), "dense setup ledger must be an object")
    _require(len(ledger) == EXPECTED_GEOMETRIES, "expected 12 geometry ledger entries")
    fields = {
        "matrix_shape",
        "active_voxel_count",
        "measurement_count",
        "setup_forward_calls",
        "zero_forward_maximum_absolute",
        "status",
        "rank",
        "nullity_lower_bound",
        "largest_singular_value",
        "smallest_retained_singular_value",
        "rank_tolerance",
        "factorization_seconds",
    }
    expected_setup_calls = 1 + math.ceil(
        EXPECTED_MATRIX_SHAPE[1] / int(dense_config["assembly_batch_size"])
    )
    for digest, entry in ledger.items():
        _require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None, "geometry ledger digest is malformed")
        _require(isinstance(entry, dict), f"geometry ledger {digest}: expected object")
        _require(set(entry) == fields, f"geometry ledger {digest}: schema drift")
        _require(entry["matrix_shape"] == list(EXPECTED_MATRIX_SHAPE), "geometry ledger matrix shape drift")
        _require(type(entry["active_voxel_count"]) is int and entry["active_voxel_count"] == EXPECTED_MATRIX_SHAPE[1], "geometry ledger active voxel count drift")
        _require(type(entry["measurement_count"]) is int and entry["measurement_count"] == EXPECTED_MATRIX_SHAPE[0], "geometry ledger measurement count drift")
        _require(type(entry["rank"]) is int and entry["rank"] == EXPECTED_RANK, "geometry ledger rank drift")
        _require(type(entry["nullity_lower_bound"]) is int and entry["nullity_lower_bound"] == EXPECTED_NULLITY, "geometry ledger nullity drift")
        _require(entry["nullity_lower_bound"] == entry["active_voxel_count"] - entry["rank"], "geometry ledger rank/nullity identity drift")
        _require(type(entry["setup_forward_calls"]) is int and entry["setup_forward_calls"] == expected_setup_calls, "geometry ledger setup-call count drift")
        _require(entry["status"] == "DENSE_TOY_ORACLE_SETUP_NOT_RECONSTRUCTION_BUDGET", "geometry ledger status drift")
        zero = _json_float(entry["zero_forward_maximum_absolute"], path=f"ledger.{digest}.zero_forward_maximum_absolute")
        _require(zero == 0.0, "geometry ledger operator is not zero preserving")
        largest = _json_float(entry["largest_singular_value"], path=f"ledger.{digest}.largest_singular_value")
        smallest = _json_float(entry["smallest_retained_singular_value"], path=f"ledger.{digest}.smallest_retained_singular_value")
        tolerance = _json_float(entry["rank_tolerance"], path=f"ledger.{digest}.rank_tolerance")
        expected_tolerance = max(
            float(dense_config["rank_absolute_tolerance"]),
            largest * float(dense_config["rank_relative_tolerance"]),
        )
        _require(largest >= smallest > tolerance > 0.0, "geometry ledger singular-value ordering drift")
        _require(math.isclose(tolerance, expected_tolerance, rel_tol=5e-11, abs_tol=5e-15), "geometry ledger rank tolerance drift")
        _require(_json_float(entry["factorization_seconds"], path=f"ledger.{digest}.factorization_seconds") >= 0.0, "geometry ledger factorization time is negative")
    return len(ledger)


def _recompute_aggregates(rows: Iterable[dict[str, str]]) -> dict[tuple[str, int, str], dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["method"], int(row["model_seed"]), row["split"])
        groups.setdefault(key, []).append(row)
    output: dict[tuple[str, int, str], dict[str, Any]] = {}
    for key, values in groups.items():
        output[key] = {
            "method": key[0],
            "model_seed": key[1],
            "split": key[2],
            "case_count": len(values),
            "reference_field_relative_l2_mean": _mean(float(row["reference_field_relative_l2"]) for row in values),
            "original_field_relative_l2_mean": _mean(float(row["original_field_relative_l2"]) for row in values),
            "oracle_field_relative_l2_mean": _mean(float(row["field_relative_l2"]) for row in values),
            "oracle_h1_relative_error_mean": _mean(float(row["h1_seminorm_relative_error"]) for row in values),
            "field_gain_to_reference_mean": _mean(float(row["field_gain_to_reference"]) for row in values),
            "h1_gain_to_reference_mean": _mean(float(row["h1_gain_to_reference"]) for row in values),
            "original_gain_retention_mean": _mean(float(row["original_gain_retention"]) for row in values),
            "measured_reprojection_ratio_to_reference_mean": _mean(float(row["measured_reprojection_ratio_to_reference"]) for row in values),
            "clean_reprojection_ratio_to_reference_mean": _mean(float(row["clean_reprojection_ratio_to_reference"]) for row in values),
            "null_correction_energy_fraction_mean": _mean(float(row["null_correction_energy_fraction"]) for row in values),
            "row_correction_energy_fraction_mean": _mean(float(row["row_correction_energy_fraction"]) for row in values),
            "visible_null_correction_fraction_maximum": max(float(row["visible_null_correction_fraction"]) for row in values),
            "internal_projection_residual_maximum": max(float(row["internal_projection_residual"]) for row in values),
            "field_gain_to_reference_minimum": min(float(row["field_gain_to_reference"]) for row in values),
            "field_harm_rate": _mean(float(row["field_harm_to_reference"]) for row in values),
        }
    return output


def _csv_aggregate_map(rows: list[dict[str, str]]) -> dict[tuple[str, int, str], dict[str, Any]]:
    output: dict[tuple[str, int, str], dict[str, Any]] = {}
    for index, row in enumerate(rows):
        key = (
            row["method"],
            _csv_int(row["model_seed"], path=f"aggregate_rows[{index}].model_seed"),
            row["split"],
        )
        _require(key not in output, f"duplicate aggregate row: {key}")
        converted: dict[str, Any] = {
            "method": row["method"],
            "model_seed": key[1],
            "split": row["split"],
            "case_count": _csv_int(row["case_count"], path=f"aggregate_rows[{index}].case_count"),
        }
        for field in AGGREGATE_FIELDS:
            if field not in converted:
                converted[field] = _csv_float(row[field], path=f"aggregate_rows[{index}].{field}")
        output[key] = converted
    return output


def _summary_aggregate_map(rows: Any) -> dict[tuple[str, int, str], dict[str, Any]]:
    _require(isinstance(rows, list), "summary aggregate must be a list")
    output: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in rows:
        _require(isinstance(row, dict), "summary aggregate row must be an object")
        _require(set(row) == set(AGGREGATE_FIELDS), "summary aggregate schema drift")
        key = (row["method"], int(row["model_seed"]), row["split"])
        _require(key not in output, f"duplicate summary aggregate row: {key}")
        output[key] = row
    return output


def _compare_aggregate_maps(
    actual: dict[tuple[str, int, str], dict[str, Any]],
    expected: dict[tuple[str, int, str], dict[str, Any]],
    *,
    path: str,
) -> None:
    _require(set(actual) == set(expected), f"{path}: aggregate identity grid mismatch")
    for key, expected_row in expected.items():
        _compare(actual[key], expected_row, path=f"{path}[{key!r}]")


def _recompute_decisions(
    rows: Iterable[dict[str, str]],
    gates: dict[str, Any],
) -> dict[str, Any]:
    all_rows = list(rows)
    decisions: dict[str, Any] = {}
    for method in METHODS:
        selected = [row for row in all_rows if row["method"] == method]
        diagnostics: dict[str, Any] = {}
        checks: dict[str, bool] = {}
        for split in SPLITS:
            values = [row for row in selected if row["split"] == split]
            expected_count = SPLIT_CASE_COUNTS[split] * len(MODEL_SEEDS)
            _require(len(values) == expected_count, f"decision {method}/{split}: row count drift")
            gains = [float(row["field_gain_to_reference"]) for row in values]
            h1_gains = [float(row["h1_gain_to_reference"]) for row in values]
            retention = [float(row["original_gain_retention"]) for row in values]
            ratios = [float(row["measured_reprojection_ratio_to_reference"]) for row in values]
            harm = [gain < -float(gates["field_harm_threshold_fraction"]) for gain in gains]
            seed_means = [
                _mean(
                    float(row["field_gain_to_reference"])
                    for row in values
                    if int(row["model_seed"]) == seed
                )
                for seed in MODEL_SEEDS
            ]
            diagnostics[f"{split}_field_gain_mean"] = _mean(gains)
            diagnostics[f"{split}_h1_gain_mean"] = _mean(h1_gains)
            diagnostics[f"{split}_original_gain_retention_mean"] = _mean(retention)
            diagnostics[f"{split}_reprojection_ratio_mean"] = _mean(ratios)
            diagnostics[f"{split}_field_harm_rate"] = _mean(float(value) for value in harm)
            diagnostics[f"{split}_worst_field_gain"] = min(gains)
            diagnostics[f"{split}_visible_null_fraction_maximum"] = max(float(row["visible_null_correction_fraction"]) for row in values)
            diagnostics[f"{split}_projection_residual_maximum"] = max(float(row["internal_projection_residual"]) for row in values)
            diagnostics[f"{split}_per_seed_field_gain_means"] = seed_means
            checks[f"{split}_field_gain"] = _mean(gains) >= float(gates[f"{split}_field_gain_to_reference_minimum_fraction"])
            checks[f"{split}_h1_gain"] = _mean(h1_gains) >= float(gates[f"{split}_h1_gain_to_reference_minimum_fraction"])
            checks[f"{split}_retention"] = _mean(retention) >= float(gates[f"{split}_original_gain_retention_minimum_fraction"])
            checks[f"{split}_reprojection"] = _mean(ratios) <= float(gates[f"{split}_reprojection_ratio_to_reference_maximum"])
            checks[f"{split}_harm"] = _mean(float(value) for value in harm) <= float(gates["field_harm_rate_maximum"])
            checks[f"{split}_worst_case"] = min(gains) >= float(gates["worst_field_gain_minimum_fraction"])
            checks[f"{split}_all_seed_means_positive"] = (
                not bool(gates["require_all_model_seed_mean_field_gains_positive"])
                or all(value > 0.0 for value in seed_means)
            )
            checks[f"{split}_projection_residual"] = diagnostics[f"{split}_projection_residual_maximum"] <= float(gates["maximum_internal_projection_residual"])
            checks[f"{split}_visible_null_fraction"] = diagnostics[f"{split}_visible_null_fraction_maximum"] <= float(gates["maximum_visible_null_correction_fraction"])
        decisions[method] = {
            "checks": checks,
            "diagnostics": diagnostics,
            "passed_exact_oracle_headroom_gate": all(checks.values()),
        }
    return decisions


def validate_packet(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Validate the packet without importing or trusting the experiment runner."""

    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    _validate_checksum_manifest(output_dir)
    config = _load_json(config_path)
    source_config = _validate_config(config, config_path)

    source_dir = (ROOT / str(config["source_t0_results"])).resolve()
    _require(source_dir.is_relative_to(ROOT), "source T0 results escape repository")
    _verify_source_metric_checksum(source_dir)
    source_metric_rows = _read_csv(source_dir / "metric_rows.csv", COMMON_METRIC_FIELDS)
    catalog, source_learned = _source_catalog(source_metric_rows, source_config)

    summary = _load_json(output_dir / "summary.json")
    metric_rows = _read_csv(output_dir / "metric_rows.csv", ORACLE_FIELDS)
    reference_rows = _read_csv(output_dir / "reference_rows.csv", COMMON_METRIC_FIELDS)
    zero_rows = _read_csv(output_dir / "zero_step_reproduction.csv", ZERO_FIELDS)
    aggregate_rows = _read_csv(output_dir / "aggregate_rows.csv", AGGREGATE_FIELDS)

    references = _validate_reference_rows(reference_rows, catalog)
    oracle = _validate_oracle_rows(
        metric_rows,
        catalog,
        references,
        source_learned,
        config["decision_gates"],
    )
    zero_summary = _validate_zero_step(zero_rows, oracle, source_learned)
    geometry_count = _validate_geometry_ledger(
        summary.get("dense_setup_ledger"),
        config["dense_oracle"],
    )

    expected_aggregates = _recompute_aggregates(metric_rows)
    _require(len(expected_aggregates) == EXPECTED_AGGREGATES, "aggregate count drift")
    _compare_aggregate_maps(
        _csv_aggregate_map(aggregate_rows),
        expected_aggregates,
        path="aggregate_rows.csv",
    )
    decisions = _recompute_decisions(metric_rows, config["decision_gates"])
    _require(len(decisions) == EXPECTED_DECISIONS, "decision count drift")
    _require(
        all(value["passed_exact_oracle_headroom_gate"] for value in decisions.values()),
        "frozen packet no longer passes the exact-oracle headroom gate",
    )

    _require(set(summary) == SUMMARY_FIELDS, "summary top-level schema drift")
    _require(summary["schema_version"] == REPORT_SCHEMA, "summary schema drift")
    expected_status = (
        EXPECTED_REPORT_STATUS
        if any(value["passed_exact_oracle_headroom_gate"] for value in decisions.values())
        else "M2_2_EXACT_NULLSPACE_ORACLE_NO_GO"
    )
    _require(summary["status"] == expected_status, "summary status does not match recomputed oracle result")
    _require(summary["evidence_level"] == EXPECTED_EVIDENCE_LEVEL, "summary evidence level drift")
    _require(summary["source_config_sha256"] == _sha256(config_path), "summary config hash mismatch")
    _require(summary["source_t0_config_sha256"] == config["source_t0_config_sha256"], "summary source config hash mismatch")
    _require(summary["source_t0_summary_sha256"] == config["source_t0_summary_sha256"], "summary source result hash mismatch")
    _require(type(summary["metric_row_count"]) is int and summary["metric_row_count"] == EXPECTED_ORACLE_ROWS, "summary oracle row count mismatch")
    _require(type(summary["reference_row_count"]) is int and summary["reference_row_count"] == EXPECTED_REFERENCE_ROWS, "summary reference row count mismatch")
    _compare(summary["zero_step_source_reproduction"], zero_summary, path="summary.zero_step_source_reproduction")
    _compare_aggregate_maps(
        _summary_aggregate_map(summary["aggregate"]),
        expected_aggregates,
        path="summary.aggregate",
    )
    _compare(summary["decisions"], decisions, path="summary.decisions")
    _compare(summary["authorization"], AUTHORIZATION, path="summary.authorization")
    _compare(summary["claim_boundary"], CLAIM_BOUNDARY, path="summary.claim_boundary")
    _compare(summary["public_export_policy"], PUBLIC_EXPORT_POLICY, path="summary.public_export_policy")

    training_runs = summary["training_runs"]
    _require(isinstance(training_runs, list) and len(training_runs) == 6, "training run count drift")
    _require(
        {(run["method"], int(run["model_seed"])) for run in training_runs}
        == {(method, seed) for method in METHODS for seed in MODEL_SEEDS},
        "training method/seed grid drift",
    )
    _require(_json_float(summary["elapsed_seconds"], path="summary.elapsed_seconds") >= 0.0, "negative elapsed time")

    return {
        "schema_version": "jacru-m2-2-evidence-validation-1.0",
        "status": VALIDATED_STATUS,
        "oracle_row_count": len(metric_rows),
        "reference_row_count": len(reference_rows),
        "zero_step_exact_row_count": len(zero_rows),
        "geometry_ledger_count": geometry_count,
        "aggregate_count": len(expected_aggregates),
        "decision_count": len(decisions),
        "passed_headroom_decision_count": sum(
            int(value["passed_exact_oracle_headroom_gate"])
            for value in decisions.values()
        ),
        "maximum_visible_null_correction_fraction": max(
            float(row["visible_null_correction_fraction"]) for row in metric_rows
        ),
        "maximum_internal_projection_residual": max(
            float(row["internal_projection_residual"]) for row in metric_rows
        ),
        "authorization": dict(AUTHORIZATION),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = validate_packet(config_path=args.config, output_dir=args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
