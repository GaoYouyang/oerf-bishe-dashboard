#!/usr/bin/env python3
"""Independently validate the frozen JACRU M2.1 matched-budget packet."""

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
    "jacru_m2_1_matched_data_consistency_postopen_v1_1.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "demo_t16_operator/results/"
    "jacru_m2_1_matched_data_consistency_postopen_public"
)

CONFIG_SCHEMA = "jacru-m2-1-matched-data-consistency-postopen-config-1.1"
REPORT_SCHEMA = "jacru-m2-1-matched-data-consistency-postopen-report-1.1"
EXPECTED_CONFIG_STATUS = "FROZEN_BEFORE_FIRST_MATCHED_BUDGET_POSTOPEN_EXECUTION"
EXPECTED_REPORT_STATUS = "M2_1_POSTOPEN_DATA_CONSISTENCY_NO_GO"
VALIDATED_STATUS = "VALIDATED_M2_1_MATCHED_BUDGET_NO_GO"
EXPECTED_EVIDENCE_LEVEL = (
    "E1_OPENED_T0_MATCHED_BUDGET_DATA_CONSISTENCY_DIAGNOSTIC_NO_FRESH"
)

METHODS = ("jacru_m2", "pooled_cnn")
MODEL_SEEDS = (17, 29, 43)
MODES = ("measurement_pullback", "base_nullspace_filter")
STEPS = (0, 1, 3, 5, 11)
SPLITS = ("development", "ood")
SPLIT_CASE_COUNTS = {"development": 12, "ood": 18}
BASELINE_METHODS = (
    "cgls_matched",
    "huber_pdhg_matched",
    "base_landweber_matched",
)
EXPECTED_LEARNED_ROWS = 1620
EXPECTED_BASELINE_ROWS = 450
EXPECTED_LEARNED_AGGREGATES = 108
EXPECTED_BASELINE_AGGREGATES = 30
EXPECTED_DECISIONS = 18
EXPECTED_ZERO_ROWS = 180

CONFIG_FIELDS = {
    "schema_version",
    "status",
    "frozen_date",
    "evidence_level",
    "supersedes_unmatched_config",
    "supersedes_unmatched_config_sha256",
    "reason_for_revision",
    "source_t0_config",
    "source_t0_config_sha256",
    "source_t0_results",
    "methods",
    "modes",
    "snapshot_steps",
    "step_safety_factor",
    "matched_baselines",
    "decision_gates",
    "claim_boundary",
}
SUMMARY_FIELDS = {
    "schema_version",
    "status",
    "evidence_level",
    "source_config_sha256",
    "source_t0_config_sha256",
    "device",
    "elapsed_seconds",
    "metric_row_count",
    "matched_baseline_metric_row_count",
    "zero_step_source_reproduction",
    "training_runs",
    "operator_norm_setup",
    "decisions",
    "aggregate",
    "matched_baseline_aggregate",
    "authorization",
    "claim_boundary",
    "public_export_policy",
}
AUTHORIZATION_FIELDS = {
    "claim_method_superiority",
    "claim_real_bost_generalization",
    "claim_interface_detection",
    "open_fresh_or_final",
    "draft_new_preregistered_data_consistency_gate",
}
CHECKSUM_PAYLOADS = {
    "README.md",
    "aggregate_rows.csv",
    "diagnostic.pdf",
    "diagnostic.png",
    "matched_baseline_aggregate_rows.csv",
    "matched_baseline_rows.csv",
    "metric_rows.csv",
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
METRIC_FIELDS = COMMON_METRIC_FIELDS + (
    "dc_mode",
    "dc_steps",
    "dc_step_size",
    "operator_norm_squared_bound",
    "field_gain_to_source_cgls13",
    "reprojection_ratio_to_source_cgls13",
    "matched_cgls_field_relative_l2",
    "matched_huber_field_relative_l2",
    "matched_base_landweber_field_relative_l2",
    "field_gain_to_best_matched_classical",
    "h1_gain_to_best_matched_classical",
    "network_gain_to_base_landweber",
    "reprojection_ratio_to_matched_cgls",
    "observable_correction_to_base_residual_ratio",
    "field_harm_to_best_matched_classical",
)
BASELINE_FIELDS = COMMON_METRIC_FIELDS + (
    "matched_step",
    "total_calls",
    "baseline_kind",
    "dc_step_size",
    "operator_norm_squared_bound",
)
AGGREGATE_FIELDS = (
    "method",
    "model_seed",
    "split",
    "dc_mode",
    "dc_steps",
    "case_count",
    "field_relative_l2_mean",
    "h1_seminorm_relative_error_mean",
    "measured_reprojection_relative_l2_mean",
    "field_gain_to_best_matched_classical_mean",
    "h1_gain_to_best_matched_classical_mean",
    "network_gain_to_base_landweber_mean",
    "reprojection_ratio_to_matched_cgls_mean",
    "field_gain_to_best_matched_classical_minimum",
    "field_harm_rate",
    "optimization_forward_calls",
    "optimization_adjoint_calls",
)
BASELINE_AGGREGATE_FIELDS = (
    "method",
    "split",
    "matched_step",
    "total_calls",
    "case_count",
    "field_relative_l2_mean",
    "h1_seminorm_relative_error_mean",
    "measured_reprojection_relative_l2_mean",
)
ZERO_FIELDS = (
    "method",
    "model_seed",
    "case_id",
    "field_absolute_delta",
    "reprojection_absolute_delta",
)
SOURCE_T0_FIELDS = COMMON_METRIC_FIELDS


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


def _csv_float(value: str, *, path: str, nullable: bool = False) -> float | None:
    if nullable and value == "":
        return None
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
    if expected is None:
        _require(actual is None, f"{path}: expected null")
    elif isinstance(expected, bool):
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
    assert observed is not None
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


def _verify_source_metric_checksum(source_dir: Path) -> None:
    manifest = source_dir / "checksums.sha256"
    _require(manifest.is_file(), "source T0 checksum manifest is missing")
    expected: str | None = None
    for line in manifest.read_text(encoding="ascii").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  metric_rows\.csv", line)
        if match:
            _require(expected is None, "source T0 metric checksum is duplicated")
            expected = match.group(1)
    _require(expected is not None, "source T0 metric checksum is missing")
    _require(
        _sha256(source_dir / "metric_rows.csv") == expected,
        "source T0 metric checksum mismatch",
    )


def _validate_config(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    _require(set(config) == CONFIG_FIELDS, "config top-level schema drift")
    _require(config["schema_version"] == CONFIG_SCHEMA, "config schema drift")
    _require(config["status"] == EXPECTED_CONFIG_STATUS, "config is not frozen v1.1")
    _require(config["evidence_level"] == EXPECTED_EVIDENCE_LEVEL, "evidence level drift")
    _require(tuple(config["methods"]) == METHODS, "frozen method set drift")
    _require(tuple(config["modes"]) == MODES, "frozen correction modes drift")
    _require(tuple(config["snapshot_steps"]) == STEPS, "frozen snapshot steps drift")
    _require(
        tuple(config["matched_baselines"])
        == (
            "cgls_total_calls",
            "huber_pdhg_total_calls",
            "base_landweber_total_calls",
        ),
        "frozen matched baseline set drift",
    )
    _require(
        math.isclose(_json_float(config["step_safety_factor"], path="step_safety_factor"), 0.98),
        "step safety factor drift",
    )
    gates = config["decision_gates"]
    expected_gates = {
        "development_field_gain_to_best_matched_classical_minimum_fraction": 0.05,
        "development_h1_gain_to_best_matched_classical_minimum_fraction": 0.03,
        "development_network_gain_to_base_landweber_minimum_fraction": 0.05,
        "development_reprojection_ratio_to_matched_cgls_maximum": 1.1,
        "ood_field_gain_to_best_matched_classical_minimum_fraction": 0.02,
        "ood_h1_gain_to_best_matched_classical_minimum_fraction": 0.0,
        "ood_network_gain_to_base_landweber_minimum_fraction": 0.02,
        "ood_reprojection_ratio_to_matched_cgls_maximum": 1.15,
        "field_harm_threshold_fraction": 0.01,
        "field_harm_rate_maximum": 0.05,
        "worst_field_gain_minimum_fraction": -0.05,
        "require_all_model_seed_mean_field_gains_positive": True,
    }
    _compare(gates, expected_gates, path="config.decision_gates")
    boundary = config["claim_boundary"]
    _require(isinstance(boundary, dict), "config.claim_boundary must be an object")
    _require(boundary.get("is_postopen_diagnostic") is True, "post-open flag drift")
    for key in (
        "is_confirmatory_or_final",
        "is_experimental_reconstruction",
        "is_cfd_validation",
        "is_real_bost_generalization",
        "is_interface_detection_evidence",
        "finite_filter_is_exact_nullspace_projection",
        "opens_fresh_or_final",
    ):
        _require(boundary.get(key) is False, f"config claim boundary enabled: {key}")

    source_config_path = (ROOT / str(config["source_t0_config"])).resolve()
    _require(source_config_path.is_relative_to(ROOT), "source T0 config escapes repository")
    _require(source_config_path.is_file(), "source T0 config is missing")
    _require(
        _sha256(source_config_path) == config["source_t0_config_sha256"],
        "source T0 config hash drift",
    )
    source_config = _load_json(source_config_path)
    _require(
        tuple(source_config["training"]["model_seeds"]) == MODEL_SEEDS,
        "source model seed set drift",
    )
    _require(
        int(source_config["physical_budget"]["cgls_base_iterations"]) == 12,
        "source CGLS base budget drift",
    )
    _require(
        int(source_config["physical_budget"]["learned_feature_forward_calls"]) == 1,
        "source learned feature forward budget drift",
    )
    _require(
        int(source_config["physical_budget"]["learned_feature_grouped_adjoint_calls"])
        == 1,
        "source learned feature adjoint budget drift",
    )
    _require(config_path.is_file(), "diagnostic config is missing")
    return source_config


def _case_catalog(
    source_rows: list[dict[str, str]],
    source_config: dict[str, Any],
) -> tuple[
    dict[str, tuple[str, str, int]],
    dict[tuple[str, int, str], dict[str, str]],
    dict[str, dict[str, str]],
]:
    source_lookup: dict[tuple[str, int, str], dict[str, str]] = {}
    cgls_lookup: dict[str, dict[str, str]] = {}
    catalog: dict[str, tuple[str, str, int]] = {}
    for index, row in enumerate(source_rows):
        key = (row["method"], _csv_int(row["model_seed"], path=f"source[{index}].model_seed"), row["case_id"])
        _require(key not in source_lookup, f"duplicate source T0 row: {key}")
        source_lookup[key] = row
        if row["method"] == "cgls_13" and key[1] == -1:
            _require(row["case_id"] not in catalog, f"duplicate source case: {row['case_id']}")
            split = row["split"]
            _require(split in SPLITS, f"source case has unexpected split: {split}")
            base_seed = _csv_int(row["base_seed"], path=f"source[{index}].base_seed")
            catalog[row["case_id"]] = (split, row["family"], base_seed)
            cgls_lookup[row["case_id"]] = row

    _require(len(catalog) == sum(SPLIT_CASE_COUNTS.values()), "source case count drift")
    for split, expected_count in SPLIT_CASE_COUNTS.items():
        cases = [value for value in catalog.values() if value[0] == split]
        _require(len(cases) == expected_count, f"source {split} case count drift")
        source_split = source_config["splits"][split]
        expected = {
            (split, family, int(seed))
            for seed in source_split["base_seeds"]
            for family in source_split["families"]
        }
        _require(set(cases) == expected, f"source {split} seed/family grid drift")

    expected_learned_source = {
        (method, seed, case_id)
        for method in METHODS
        for seed in MODEL_SEEDS
        for case_id in catalog
    }
    _require(
        expected_learned_source.issubset(source_lookup),
        "source T0 learned rows are incomplete",
    )
    return catalog, source_lookup, cgls_lookup


def _metric_float(row: dict[str, str], key: str, index: int) -> float:
    value = _csv_float(row[key], path=f"metric_rows[{index}].{key}")
    assert value is not None
    return value


def _baseline_float(row: dict[str, str], key: str, index: int) -> float:
    value = _csv_float(row[key], path=f"matched_baseline_rows[{index}].{key}")
    assert value is not None
    return value


def _validate_metric_rows(
    rows: list[dict[str, str]],
    catalog: dict[str, tuple[str, str, int]],
) -> dict[tuple[str, int, str, str, int], dict[str, str]]:
    _require(len(rows) == EXPECTED_LEARNED_ROWS, "expected 1620 learned rows")
    allowed_mode_steps = {
        (MODES[0], step) for step in STEPS
    } | {
        (MODES[1], step) for step in STEPS if step > 0
    }
    lookup: dict[tuple[str, int, str, str, int], dict[str, str]] = {}
    nonnegative = (
        "field_relative_l2",
        "field_rmse",
        "field_nrmse_dynamic_range",
        "h1_seminorm_relative_error",
        "measured_reprojection_relative_l2",
        "clean_reprojection_relative_l2",
        "correction_rms",
        "observable_correction_to_base_residual_ratio",
    )
    for index, row in enumerate(rows):
        method = row["method"]
        seed = _csv_int(row["model_seed"], path=f"metric_rows[{index}].model_seed")
        case_id = row["case_id"]
        mode = row["dc_mode"]
        step = _csv_int(row["dc_steps"], path=f"metric_rows[{index}].dc_steps")
        _require(method in METHODS, f"metric_rows[{index}]: unknown method")
        _require(seed in MODEL_SEEDS, f"metric_rows[{index}]: unknown model seed")
        _require(case_id in catalog, f"metric_rows[{index}]: unknown case")
        _require((mode, step) in allowed_mode_steps, f"metric_rows[{index}]: invalid mode/step")
        split, family, base_seed = catalog[case_id]
        _require(row["split"] == split, f"metric_rows[{index}]: split mismatch")
        _require(row["family"] == family, f"metric_rows[{index}]: family mismatch")
        _require(
            _csv_int(row["base_seed"], path=f"metric_rows[{index}].base_seed") == base_seed,
            f"metric_rows[{index}]: base seed mismatch",
        )
        expected_calls = 13 + step
        _require(
            _csv_int(row["optimization_forward_calls"], path=f"metric_rows[{index}].optimization_forward_calls")
            == expected_calls,
            "learned forward budget drift",
        )
        _require(
            _csv_int(row["optimization_adjoint_calls"], path=f"metric_rows[{index}].optimization_adjoint_calls")
            == expected_calls,
            "learned adjoint budget drift",
        )
        _require(
            _csv_int(row["grouped_adjoint_calls"], path=f"metric_rows[{index}].grouped_adjoint_calls") == 1,
            "learned grouped-adjoint budget drift",
        )
        _require(
            _csv_int(row["evaluation_forward_calls"], path=f"metric_rows[{index}].evaluation_forward_calls") == 1,
            "learned evaluation budget drift",
        )
        for field in nonnegative:
            _require(_metric_float(row, field, index) >= 0.0, f"metric_rows[{index}].{field}: negative")
        gate = _metric_float(row, "gate", index)
        _require(0.0 <= gate <= 1.0, f"metric_rows[{index}].gate: outside [0, 1]")
        _metric_float(row, "field_mean_bias", index)
        _require(_metric_float(row, "dc_step_size", index) > 0.0, "invalid DC step size")
        _require(
            _metric_float(row, "operator_norm_squared_bound", index) > 0.0,
            "invalid operator norm bound",
        )
        _require(
            _csv_int(row["field_harm_to_best_matched_classical"], path=f"metric_rows[{index}].field_harm_to_best_matched_classical")
            in (0, 1),
            "invalid learned harm flag",
        )
        key = (method, seed, case_id, mode, step)
        _require(key not in lookup, f"duplicate learned row: {key}")
        lookup[key] = row

    expected = {
        (method, seed, case_id, mode, step)
        for method in METHODS
        for seed in MODEL_SEEDS
        for case_id in catalog
        for mode, step in allowed_mode_steps
    }
    _require(set(lookup) == expected, "learned method/seed/case/mode/step grid drift")
    return lookup


def _validate_baseline_rows(
    rows: list[dict[str, str]],
    catalog: dict[str, tuple[str, str, int]],
) -> dict[tuple[str, int, str], dict[str, str]]:
    _require(len(rows) == EXPECTED_BASELINE_ROWS, "expected 450 matched baseline rows")
    lookup: dict[tuple[str, int, str], dict[str, str]] = {}
    for index, row in enumerate(rows):
        case_id = row["case_id"]
        method = row["method"]
        step = _csv_int(row["matched_step"], path=f"matched_baseline_rows[{index}].matched_step")
        _require(case_id in catalog, f"matched_baseline_rows[{index}]: unknown case")
        _require(method in BASELINE_METHODS, f"matched_baseline_rows[{index}]: unknown method")
        _require(step in STEPS, f"matched_baseline_rows[{index}]: unknown step")
        split, family, base_seed = catalog[case_id]
        _require(row["split"] == split, f"matched_baseline_rows[{index}]: split mismatch")
        _require(row["family"] == family, f"matched_baseline_rows[{index}]: family mismatch")
        _require(
            _csv_int(row["base_seed"], path=f"matched_baseline_rows[{index}].base_seed") == base_seed,
            f"matched_baseline_rows[{index}]: base seed mismatch",
        )
        _require(_csv_int(row["model_seed"], path=f"matched_baseline_rows[{index}].model_seed") == -1, "baseline model seed drift")
        expected_calls = 13 + step
        for field in ("optimization_forward_calls", "optimization_adjoint_calls", "total_calls"):
            _require(
                _csv_int(row[field], path=f"matched_baseline_rows[{index}].{field}") == expected_calls,
                "matched baseline budget drift",
            )
        _require(_csv_int(row["grouped_adjoint_calls"], path=f"matched_baseline_rows[{index}].grouped_adjoint_calls") == 0, "baseline grouped-adjoint budget drift")
        _require(_csv_int(row["evaluation_forward_calls"], path=f"matched_baseline_rows[{index}].evaluation_forward_calls") == 1, "baseline evaluation budget drift")
        _require(row["baseline_kind"] == method, "baseline kind/method mismatch")
        for field in (
            "field_relative_l2",
            "field_rmse",
            "field_nrmse_dynamic_range",
            "h1_seminorm_relative_error",
            "measured_reprojection_relative_l2",
            "clean_reprojection_relative_l2",
        ):
            _require(_baseline_float(row, field, index) >= 0.0, f"matched_baseline_rows[{index}].{field}: negative")
        _baseline_float(row, "field_mean_bias", index)
        _require(
            _baseline_float(row, "operator_norm_squared_bound", index) > 0.0,
            "invalid baseline operator norm bound",
        )
        _require(row["gate"] == "", "baseline gate must remain empty")
        if method == "base_landweber_matched":
            _require(_baseline_float(row, "correction_rms", index) >= 0.0, "invalid Landweber correction")
            _require(_baseline_float(row, "dc_step_size", index) > 0.0, "invalid Landweber step size")
        else:
            _require(row["correction_rms"] == "", "classical correction RMS must remain empty")
            _require(row["dc_step_size"] == "", "classical DC step size must remain empty")
        key = (case_id, step, method)
        _require(key not in lookup, f"duplicate matched baseline row: {key}")
        lookup[key] = row

    expected = {
        (case_id, step, method)
        for case_id in catalog
        for step in STEPS
        for method in BASELINE_METHODS
    }
    _require(set(lookup) == expected, "matched baseline case/step/method grid drift")
    return lookup


def _validate_derived_rows(
    learned: dict[tuple[str, int, str, str, int], dict[str, str]],
    baselines: dict[tuple[str, int, str], dict[str, str]],
    source_cgls: dict[str, dict[str, str]],
    harm_threshold: float,
) -> None:
    for key, row in learned.items():
        _, _, case_id, _, step = key
        cgls = baselines[(case_id, step, "cgls_matched")]
        huber = baselines[(case_id, step, "huber_pdhg_matched")]
        landweber = baselines[(case_id, step, "base_landweber_matched")]
        expected_calls = 13 + step
        for baseline in (cgls, huber, landweber):
            _require(
                _csv_int(baseline["total_calls"], path="baseline.total_calls") == expected_calls,
                "learned/baseline total-call mismatch",
            )
        learned_bound = float(row["operator_norm_squared_bound"])
        for baseline in (cgls, huber, landweber):
            _require(
                math.isclose(float(baseline["operator_norm_squared_bound"]), learned_bound, rel_tol=0.0, abs_tol=0.0),
                "learned/baseline operator norm mismatch",
            )
        _require(
            math.isclose(float(row["dc_step_size"]), float(landweber["dc_step_size"]), rel_tol=0.0, abs_tol=0.0),
            "learned/baseline DC step-size mismatch",
        )

        field = float(row["field_relative_l2"])
        h1 = float(row["h1_seminorm_relative_error"])
        cgls_field = float(cgls["field_relative_l2"])
        huber_field = float(huber["field_relative_l2"])
        landweber_field = float(landweber["field_relative_l2"])
        cgls_h1 = float(cgls["h1_seminorm_relative_error"])
        huber_h1 = float(huber["h1_seminorm_relative_error"])
        best_field = min(cgls_field, huber_field)
        best_h1 = min(cgls_h1, huber_h1)
        source = source_cgls[case_id]
        source_field = float(source["field_relative_l2"])
        source_reprojection = max(float(source["measured_reprojection_relative_l2"]), 1e-30)
        cgls_reprojection = max(float(cgls["measured_reprojection_relative_l2"]), 1e-30)
        expected_values = {
            "matched_cgls_field_relative_l2": cgls_field,
            "matched_huber_field_relative_l2": huber_field,
            "matched_base_landweber_field_relative_l2": landweber_field,
            "field_gain_to_source_cgls13": (source_field - field) / source_field,
            "reprojection_ratio_to_source_cgls13": float(row["measured_reprojection_relative_l2"]) / source_reprojection,
            "field_gain_to_best_matched_classical": (best_field - field) / best_field,
            "h1_gain_to_best_matched_classical": (best_h1 - h1) / best_h1,
            "network_gain_to_base_landweber": (landweber_field - field) / landweber_field,
            "reprojection_ratio_to_matched_cgls": float(row["measured_reprojection_relative_l2"]) / cgls_reprojection,
        }
        for field_name, expected in expected_values.items():
            _compare_csv_number(row[field_name], expected, path=f"metric_rows.{field_name}")
        expected_harm = int(field > best_field * (1.0 + harm_threshold))
        _compare_csv_number(
            row["field_harm_to_best_matched_classical"],
            expected_harm,
            path="metric_rows.field_harm_to_best_matched_classical",
        )


def _learned_aggregates(rows: Iterable[dict[str, str]]) -> dict[tuple[str, int, str, str, int], dict[str, Any]]:
    groups: dict[tuple[str, int, str, str, int], list[dict[str, str]]] = {}
    for row in rows:
        key = (
            row["method"],
            int(row["model_seed"]),
            row["split"],
            row["dc_mode"],
            int(row["dc_steps"]),
        )
        groups.setdefault(key, []).append(row)
    output: dict[tuple[str, int, str, str, int], dict[str, Any]] = {}
    for key, values in groups.items():
        output[key] = {
            "method": key[0],
            "model_seed": key[1],
            "split": key[2],
            "dc_mode": key[3],
            "dc_steps": key[4],
            "case_count": len(values),
            "field_relative_l2_mean": _mean(float(row["field_relative_l2"]) for row in values),
            "h1_seminorm_relative_error_mean": _mean(float(row["h1_seminorm_relative_error"]) for row in values),
            "measured_reprojection_relative_l2_mean": _mean(float(row["measured_reprojection_relative_l2"]) for row in values),
            "field_gain_to_best_matched_classical_mean": _mean(float(row["field_gain_to_best_matched_classical"]) for row in values),
            "h1_gain_to_best_matched_classical_mean": _mean(float(row["h1_gain_to_best_matched_classical"]) for row in values),
            "network_gain_to_base_landweber_mean": _mean(float(row["network_gain_to_base_landweber"]) for row in values),
            "reprojection_ratio_to_matched_cgls_mean": _mean(float(row["reprojection_ratio_to_matched_cgls"]) for row in values),
            "field_gain_to_best_matched_classical_minimum": min(float(row["field_gain_to_best_matched_classical"]) for row in values),
            "field_harm_rate": _mean(float(row["field_harm_to_best_matched_classical"]) for row in values),
            "optimization_forward_calls": int(values[0]["optimization_forward_calls"]),
            "optimization_adjoint_calls": int(values[0]["optimization_adjoint_calls"]),
        }
    return output


def _baseline_aggregates(rows: Iterable[dict[str, str]]) -> dict[tuple[str, str, int], dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["method"], row["split"], int(row["matched_step"]))
        groups.setdefault(key, []).append(row)
    output: dict[tuple[str, str, int], dict[str, Any]] = {}
    for key, values in groups.items():
        output[key] = {
            "method": key[0],
            "split": key[1],
            "matched_step": key[2],
            "total_calls": int(values[0]["total_calls"]),
            "case_count": len(values),
            "field_relative_l2_mean": _mean(float(row["field_relative_l2"]) for row in values),
            "h1_seminorm_relative_error_mean": _mean(float(row["h1_seminorm_relative_error"]) for row in values),
            "measured_reprojection_relative_l2_mean": _mean(float(row["measured_reprojection_relative_l2"]) for row in values),
        }
    return output


def _csv_aggregate_map(
    rows: list[dict[str, str]],
    *,
    learned: bool,
) -> dict[Any, dict[str, Any]]:
    output: dict[Any, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if learned:
            key = (
                row["method"],
                _csv_int(row["model_seed"], path=f"aggregate_rows[{index}].model_seed"),
                row["split"],
                row["dc_mode"],
                _csv_int(row["dc_steps"], path=f"aggregate_rows[{index}].dc_steps"),
            )
            int_fields = ("model_seed", "dc_steps", "case_count", "optimization_forward_calls", "optimization_adjoint_calls")
            string_fields = ("method", "split", "dc_mode")
        else:
            key = (
                row["method"],
                row["split"],
                _csv_int(row["matched_step"], path=f"matched_baseline_aggregate_rows[{index}].matched_step"),
            )
            int_fields = ("matched_step", "total_calls", "case_count")
            string_fields = ("method", "split")
        _require(key not in output, f"duplicate aggregate row: {key}")
        converted: dict[str, Any] = {field: row[field] for field in string_fields}
        converted.update({field: int(row[field]) for field in int_fields})
        for field in row:
            if field not in converted:
                value = _csv_float(row[field], path=f"aggregate[{index}].{field}")
                assert value is not None
                converted[field] = value
        output[key] = converted
    return output


def _compare_aggregate_maps(actual: dict[Any, dict[str, Any]], expected: dict[Any, dict[str, Any]], *, path: str) -> None:
    _require(set(actual) == set(expected), f"{path}: aggregate identity grid mismatch")
    for key, expected_row in expected.items():
        _compare(actual[key], expected_row, path=f"{path}[{key!r}]")


def _summary_aggregate_map(rows: Any, *, learned: bool) -> dict[Any, dict[str, Any]]:
    _require(isinstance(rows, list), "summary aggregate must be a list")
    output: dict[Any, dict[str, Any]] = {}
    for row in rows:
        _require(isinstance(row, dict), "summary aggregate row must be an object")
        key = (
            (row["method"], int(row["model_seed"]), row["split"], row["dc_mode"], int(row["dc_steps"]))
            if learned
            else (row["method"], row["split"], int(row["matched_step"]))
        )
        _require(key not in output, f"duplicate summary aggregate row: {key}")
        output[key] = row
    return output


def _recompute_decisions(rows: Iterable[dict[str, str]], gates: dict[str, Any]) -> dict[str, Any]:
    all_rows = list(rows)
    output: dict[str, Any] = {}
    mode_steps = [(MODES[0], step) for step in STEPS] + [(MODES[1], step) for step in STEPS if step > 0]
    for method in METHODS:
        for mode, step in mode_steps:
            diagnostics: dict[str, Any] = {}
            checks: dict[str, bool] = {}
            for split in SPLITS:
                values = [
                    row
                    for row in all_rows
                    if row["method"] == method
                    and row["dc_mode"] == mode
                    and int(row["dc_steps"]) == step
                    and row["split"] == split
                ]
                expected_count = SPLIT_CASE_COUNTS[split] * len(MODEL_SEEDS)
                _require(len(values) == expected_count, f"decision {method}/{mode}/{step}/{split}: row count drift")
                gains = [float(row["field_gain_to_best_matched_classical"]) for row in values]
                h1_gains = [float(row["h1_gain_to_best_matched_classical"]) for row in values]
                network_gains = [float(row["network_gain_to_base_landweber"]) for row in values]
                ratios = [float(row["reprojection_ratio_to_matched_cgls"]) for row in values]
                harm = [value < -float(gates["field_harm_threshold_fraction"]) for value in gains]
                seed_means = [
                    _mean(
                        float(row["field_gain_to_best_matched_classical"])
                        for row in values
                        if int(row["model_seed"]) == seed
                    )
                    for seed in MODEL_SEEDS
                ]
                diagnostics[f"{split}_field_gain_mean"] = _mean(gains)
                diagnostics[f"{split}_h1_gain_mean"] = _mean(h1_gains)
                diagnostics[f"{split}_network_gain_to_base_landweber_mean"] = _mean(network_gains)
                diagnostics[f"{split}_reprojection_ratio_mean"] = _mean(ratios)
                diagnostics[f"{split}_field_harm_rate"] = _mean(float(value) for value in harm)
                diagnostics[f"{split}_worst_field_gain"] = min(gains)
                diagnostics[f"{split}_per_seed_field_gain_means"] = seed_means
                checks[f"{split}_field_gain"] = _mean(gains) >= float(gates[f"{split}_field_gain_to_best_matched_classical_minimum_fraction"])
                checks[f"{split}_h1_gain"] = _mean(h1_gains) >= float(gates[f"{split}_h1_gain_to_best_matched_classical_minimum_fraction"])
                checks[f"{split}_network_marginal_gain"] = _mean(network_gains) >= float(gates[f"{split}_network_gain_to_base_landweber_minimum_fraction"])
                checks[f"{split}_reprojection"] = _mean(ratios) <= float(gates[f"{split}_reprojection_ratio_to_matched_cgls_maximum"])
                checks[f"{split}_harm"] = _mean(float(value) for value in harm) <= float(gates["field_harm_rate_maximum"])
                checks[f"{split}_worst_case"] = min(gains) >= float(gates["worst_field_gain_minimum_fraction"])
                checks[f"{split}_all_seed_means_positive"] = (
                    not bool(gates["require_all_model_seed_mean_field_gains_positive"])
                    or all(value > 0.0 for value in seed_means)
                )
            key = f"{method}|{mode}|{step}"
            output[key] = {
                "method": method,
                "dc_mode": mode,
                "dc_steps": step,
                "checks": checks,
                "diagnostics": diagnostics,
                "passed_postopen_headroom_gate": all(checks.values()),
            }
    return output


def _validate_zero_step(
    rows: list[dict[str, str]],
    learned: dict[tuple[str, int, str, str, int], dict[str, str]],
    source_lookup: dict[tuple[str, int, str], dict[str, str]],
    catalog: dict[str, tuple[str, str, int]],
) -> dict[str, Any]:
    _require(len(rows) == EXPECTED_ZERO_ROWS, "expected 180 zero-step reproduction rows")
    lookup: dict[tuple[str, int, str], dict[str, str]] = {}
    for index, row in enumerate(rows):
        key = (
            row["method"],
            _csv_int(row["model_seed"], path=f"zero_step_reproduction[{index}].model_seed"),
            row["case_id"],
        )
        _require(key not in lookup, f"duplicate zero-step row: {key}")
        _require(key in source_lookup, f"zero-step source row missing: {key}")
        current = learned[(key[0], key[1], key[2], MODES[0], 0)]
        source = source_lookup[key]
        _require(
            current["field_relative_l2"] == source["field_relative_l2"],
            "zero-step field does not exactly reproduce source T0",
        )
        _require(
            current["measured_reprojection_relative_l2"]
            == source["measured_reprojection_relative_l2"],
            "zero-step reprojection does not exactly reproduce source T0",
        )
        field_delta = abs(float(current["field_relative_l2"]) - float(source["field_relative_l2"]))
        reprojection_delta = abs(float(current["measured_reprojection_relative_l2"]) - float(source["measured_reprojection_relative_l2"]))
        _compare_csv_number(row["field_absolute_delta"], field_delta, path="zero_step_reproduction.field_absolute_delta")
        _compare_csv_number(row["reprojection_absolute_delta"], reprojection_delta, path="zero_step_reproduction.reprojection_absolute_delta")
        lookup[key] = row
    expected = {
        (method, seed, case_id)
        for method in METHODS
        for seed in MODEL_SEEDS
        for case_id in catalog
    }
    _require(set(lookup) == expected, "zero-step method/seed/case grid drift")
    maximum_field = max(float(row["field_absolute_delta"]) for row in rows)
    maximum_reprojection = max(float(row["reprojection_absolute_delta"]) for row in rows)
    _require(maximum_field == 0.0, "zero-step field reproduction is not exact")
    _require(maximum_reprojection == 0.0, "zero-step reprojection reproduction is not exact")
    return {
        "row_count": len(rows),
        "maximum_field_absolute_delta": maximum_field,
        "maximum_reprojection_absolute_delta": maximum_reprojection,
        "passed_1e_6": True,
    }


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
    source_rows = _read_csv(source_dir / "metric_rows.csv", SOURCE_T0_FIELDS)
    catalog, source_lookup, source_cgls = _case_catalog(source_rows, source_config)

    summary = _load_json(output_dir / "summary.json")
    metric_rows = _read_csv(output_dir / "metric_rows.csv", METRIC_FIELDS)
    baseline_rows = _read_csv(output_dir / "matched_baseline_rows.csv", BASELINE_FIELDS)
    aggregate_rows = _read_csv(output_dir / "aggregate_rows.csv", AGGREGATE_FIELDS)
    baseline_aggregate_rows = _read_csv(
        output_dir / "matched_baseline_aggregate_rows.csv",
        BASELINE_AGGREGATE_FIELDS,
    )
    zero_rows = _read_csv(output_dir / "zero_step_reproduction.csv", ZERO_FIELDS)

    learned = _validate_metric_rows(metric_rows, catalog)
    baselines = _validate_baseline_rows(baseline_rows, catalog)
    _validate_derived_rows(
        learned,
        baselines,
        source_cgls,
        float(config["decision_gates"]["field_harm_threshold_fraction"]),
    )
    zero_summary = _validate_zero_step(zero_rows, learned, source_lookup, catalog)

    expected_aggregates = _learned_aggregates(metric_rows)
    expected_baseline_aggregates = _baseline_aggregates(baseline_rows)
    _require(len(expected_aggregates) == EXPECTED_LEARNED_AGGREGATES, "learned aggregate count drift")
    _require(len(expected_baseline_aggregates) == EXPECTED_BASELINE_AGGREGATES, "baseline aggregate count drift")
    csv_aggregates = _csv_aggregate_map(aggregate_rows, learned=True)
    csv_baseline_aggregates = _csv_aggregate_map(baseline_aggregate_rows, learned=False)
    _compare_aggregate_maps(csv_aggregates, expected_aggregates, path="aggregate_rows.csv")
    _compare_aggregate_maps(
        csv_baseline_aggregates,
        expected_baseline_aggregates,
        path="matched_baseline_aggregate_rows.csv",
    )

    decisions = _recompute_decisions(metric_rows, config["decision_gates"])
    _require(len(decisions) == EXPECTED_DECISIONS, "decision count drift")
    for key, decision in decisions.items():
        _require(
            decision["checks"]["development_reprojection"] is False,
            f"{key}: development reprojection gate unexpectedly passed",
        )
        _require(
            decision["checks"]["ood_reprojection"] is False,
            f"{key}: OOD reprojection gate unexpectedly passed",
        )
        _require(
            decision["passed_postopen_headroom_gate"] is False,
            f"{key}: post-open headroom gate unexpectedly passed",
        )

    _require(set(summary) == SUMMARY_FIELDS, "summary top-level schema drift")
    _require(summary["schema_version"] == REPORT_SCHEMA, "summary schema drift")
    _require(summary["status"] == EXPECTED_REPORT_STATUS, "summary status must remain NO-GO")
    _require(summary["evidence_level"] == EXPECTED_EVIDENCE_LEVEL, "summary evidence level drift")
    _require(summary["source_config_sha256"] == _sha256(config_path), "summary config hash mismatch")
    _require(summary["source_t0_config_sha256"] == config["source_t0_config_sha256"], "summary source config hash mismatch")
    _require(type(summary["metric_row_count"]) is int and summary["metric_row_count"] == EXPECTED_LEARNED_ROWS, "summary learned row count mismatch")
    _require(type(summary["matched_baseline_metric_row_count"]) is int and summary["matched_baseline_metric_row_count"] == EXPECTED_BASELINE_ROWS, "summary baseline row count mismatch")
    _compare(summary["zero_step_source_reproduction"], zero_summary, path="summary.zero_step_source_reproduction")
    _compare(summary["decisions"], decisions, path="summary.decisions")
    _compare_aggregate_maps(
        _summary_aggregate_map(summary["aggregate"], learned=True),
        expected_aggregates,
        path="summary.aggregate",
    )
    _compare_aggregate_maps(
        _summary_aggregate_map(summary["matched_baseline_aggregate"], learned=False),
        expected_baseline_aggregates,
        path="summary.matched_baseline_aggregate",
    )
    _require(summary["claim_boundary"] == config["claim_boundary"], "summary claim boundary drift")
    authorization = summary["authorization"]
    _require(isinstance(authorization, dict), "summary.authorization must be an object")
    _require(set(authorization) == AUTHORIZATION_FIELDS, "summary authorization schema drift")
    _require(
        all(value is False for value in authorization.values()),
        "summary authorization must remain entirely false",
    )
    public_policy = summary["public_export_policy"]
    _require(isinstance(public_policy, dict), "public export policy must be an object")
    _require(all(value is False for value in public_policy.values()), "public export policy contains forbidden material")

    training_runs = summary["training_runs"]
    _require(isinstance(training_runs, list) and len(training_runs) == 6, "training run count drift")
    _require(
        {(run["method"], int(run["model_seed"])) for run in training_runs}
        == {(method, seed) for method in METHODS for seed in MODEL_SEEDS},
        "training method/seed grid drift",
    )
    _require(_json_float(summary["elapsed_seconds"], path="summary.elapsed_seconds") >= 0.0, "negative elapsed time")

    return {
        "schema_version": "jacru-m2-1-evidence-validation-1.0",
        "status": VALIDATED_STATUS,
        "learned_row_count": len(metric_rows),
        "matched_baseline_row_count": len(baseline_rows),
        "learned_aggregate_count": len(expected_aggregates),
        "matched_baseline_aggregate_count": len(expected_baseline_aggregates),
        "zero_step_exact_row_count": len(zero_rows),
        "decision_count": len(decisions),
        "failed_reprojection_check_count": sum(
            1
            for decision in decisions.values()
            for split in SPLITS
            if decision["checks"][f"{split}_reprojection"] is False
        ),
        "authorization": dict(authorization),
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
