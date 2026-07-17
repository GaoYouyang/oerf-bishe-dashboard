#!/usr/bin/env python3
"""Independently validate the frozen JACRU M2-T0 evidence bundle."""

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
    ROOT / "demo_t16_operator/configs/jacru_m2_learned_residual_t0_v1.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "demo_t16_operator/results/jacru_m2_learned_residual_t0_public"
)
DEFAULT_SUMMARY = DEFAULT_OUTPUT_DIR / "summary.json"
DEFAULT_METRIC_ROWS = DEFAULT_OUTPUT_DIR / "metric_rows.csv"
DEFAULT_AGGREGATE_ROWS = DEFAULT_OUTPUT_DIR / "aggregate_rows.csv"
DEFAULT_HISTORY = DEFAULT_OUTPUT_DIR / "training_history.csv"
DEFAULT_CHECKSUMS = DEFAULT_OUTPUT_DIR / "checksums.sha256"

CONFIG_SCHEMA = "jacru-m2-learned-residual-t0-config-1.0"
REPORT_SCHEMA = "jacru-m2-learned-residual-t0-report-1.0"
VALIDATION_SCHEMA = "jacru-m2-t0-evidence-validation-1.0"
EXPECTED_CONFIG_STATUS = "FROZEN_BEFORE_FIRST_T0_EXECUTION_NO_FRESH_OR_FINAL"
EXPECTED_NO_GO_STATUS = "M2_T0_NO_GO_OR_REVISE"
SPLITS = ("train", "development", "ood")
EVALUATION_SPLITS = ("development", "ood")
EXPECTED_SPLIT_CASE_COUNTS = {"train": 32, "development": 12, "ood": 18}
LEARNED_METHODS = ("jacru_m2", "pooled_cnn", "grid_deeponet", "pooled_fno")
CLASSICAL_METHODS = ("cgls_13", "huber_pdhg_13")
MODEL_SEEDS = (17, 29, 43)
EXPECTED_METRIC_ROW_COUNT = 420
EXPECTED_PARAMETER_COUNTS = {
    "jacru_m2": 6440,
    "pooled_cnn": 3549,
    "grid_deeponet": 8162,
    "pooled_fno": 10211,
}
FROZEN_MODEL_SPECS = {
    "jacru_m2": {
        "set_channels": 6,
        "hidden_channels": 8,
        "gate_hidden": 8,
        "maximum_residual_magnitude": 0.25,
    },
    "pooled_cnn": {
        "hidden_channels": 8,
        "maximum_residual_magnitude": 0.25,
    },
    "grid_deeponet": {
        "branch_hidden": 48,
        "trunk_hidden": 48,
        "rank": 24,
        "maximum_residual_magnitude": 0.25,
    },
    "pooled_fno": {
        "hidden_channels": 8,
        "n_modes": [4, 4, 4],
        "n_layers": 3,
        "maximum_residual_magnitude": 0.25,
    },
}
EXPECTED_AUTHORIZATION_KEYS = {
    "continue_to_larger_preregistered_synthetic_gate",
    "claim_neural_operator_superiority",
    "claim_interface_detection",
    "claim_real_bost_generalization",
    "open_fresh_or_final",
}
CHECKSUM_PAYLOADS = {
    "README.md",
    "aggregate_rows.csv",
    "diagnostic.pdf",
    "diagnostic.png",
    "metric_rows.csv",
    "summary.json",
    "training_history.csv",
}
CONFIG_FIELDS = {
    "schema_version",
    "status",
    "frozen_date",
    "evidence_level",
    "fixture",
    "splits",
    "physical_budget",
    "training",
    "models",
    "methods",
    "decision_gates",
    "claim_boundary",
}
SUMMARY_FIELDS = {
    "schema_version",
    "status",
    "evidence_level",
    "source_config_sha256",
    "device",
    "fixture",
    "split_case_counts",
    "case_manifest",
    "physical_budget",
    "norm_setup",
    "training_runs",
    "metric_row_count",
    "aggregate",
    "method_decisions",
    "primary_method",
    "primary_passed",
    "authorization",
    "claim_boundary",
    "elapsed_seconds",
    "public_export_policy",
}

METRIC_FIELDS = (
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
AGGREGATE_FIELDS = (
    "method",
    "model_seed",
    "split",
    "case_count",
    "field_relative_l2_mean",
    "field_relative_l2_maximum",
    "h1_seminorm_relative_error_mean",
    "measured_reprojection_relative_l2_mean",
    "gate_mean",
    "correction_rms_mean",
    "neural_inference_seconds_mean",
)
HISTORY_FIELDS = (
    "method",
    "model_seed",
    "epoch",
    "learning_rate",
    "development_field_relative_l2",
    "train_total",
    "train_field",
    "train_h1",
    "train_correction",
    "train_gate",
)


class ValidationError(RuntimeError):
    """Raised when the evidence bundle violates the frozen T0 contract."""


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


def _strict_json_int(value: Any, *, path: str) -> int:
    _require(type(value) is int, f"{path}: expected integer")
    return value


def _strict_json_float(value: Any, *, path: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{path}: expected number",
    )
    parsed = float(value)
    _require(math.isfinite(parsed), f"{path}: expected finite number")
    return parsed


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


def _compare(actual: Any, expected: Any, *, path: str) -> None:
    if expected is None:
        _require(actual is None, f"{path}: expected null")
    elif isinstance(expected, bool):
        _require(actual is expected, f"{path}: boolean mismatch")
    elif isinstance(expected, int):
        _require(type(actual) is int and actual == expected, f"{path}: integer mismatch")
    elif isinstance(expected, float):
        observed = _strict_json_float(actual, path=path)
        _require(
            math.isclose(observed, expected, rel_tol=3e-11, abs_tol=3e-12),
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


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    _require(bool(materialized), "cannot average an empty collection")
    return math.fsum(materialized) / len(materialized)


def _validate_config(config: dict[str, Any]) -> dict[str, set[tuple[int, str]]]:
    _require(set(config) == CONFIG_FIELDS, "config top-level schema drift")
    _require(config.get("schema_version") == CONFIG_SCHEMA, "config schema drift")
    _require(config.get("status") == EXPECTED_CONFIG_STATUS, "config is not frozen T0")

    splits = config.get("splits")
    _require(isinstance(splits, dict), "config.splits must be an object")
    _require(set(splits) == set(SPLITS), "config contains a fresh/final or unknown split")
    expected_cases: dict[str, set[tuple[int, str]]] = {}
    split_seed_sets: dict[str, set[int]] = {}
    for split in SPLITS:
        spec = splits[split]
        _require(isinstance(spec, dict), f"config.splits.{split} must be an object")
        _require(set(spec) == {"base_seeds", "families"}, f"{split}: split schema drift")
        seeds = spec["base_seeds"]
        families = spec["families"]
        _require(isinstance(seeds, list) and seeds, f"{split}: missing base seeds")
        _require(isinstance(families, list) and families, f"{split}: missing families")
        parsed_seeds = [
            _strict_json_int(value, path=f"config.splits.{split}.base_seeds")
            for value in seeds
        ]
        _require(len(set(parsed_seeds)) == len(parsed_seeds), f"{split}: duplicate base seed")
        _require(
            all(isinstance(value, str) and value for value in families),
            f"{split}: invalid family",
        )
        _require(len(set(families)) == len(families), f"{split}: duplicate family")
        _require(
            not any("fresh" in value.lower() or "final" in value.lower() for value in families),
            f"{split}: fresh/final family is forbidden",
        )
        cases = {(seed, family) for seed in parsed_seeds for family in families}
        _require(
            len(cases) == EXPECTED_SPLIT_CASE_COUNTS[split],
            f"{split}: expected {EXPECTED_SPLIT_CASE_COUNTS[split]} cases",
        )
        expected_cases[split] = cases
        split_seed_sets[split] = set(parsed_seeds)

    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            _require(
                split_seed_sets[left].isdisjoint(split_seed_sets[right]),
                f"config split seeds overlap: {left}/{right}",
            )
            _require(
                expected_cases[left].isdisjoint(expected_cases[right]),
                f"config cases overlap: {left}/{right}",
            )

    _compare(config.get("methods"), list(LEARNED_METHODS), path="config.methods")
    training = config.get("training")
    _require(isinstance(training, dict), "config.training must be an object")
    _compare(training.get("model_seeds"), list(MODEL_SEEDS), path="config.training.model_seeds")
    _require(_strict_json_int(training.get("epochs"), path="config.training.epochs") == 80, "config epochs must remain 80")
    for key in ("batch_size", "early_stop_patience", "minimum_epoch"):
        _require(_strict_json_int(training.get(key), path=f"config.training.{key}") > 0, f"config.training.{key} must be positive")
    for key in (
        "learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "lambda_h1",
        "lambda_correction",
        "lambda_gate",
        "camera_dropout_probability",
    ):
        _require(_strict_json_float(training.get(key), path=f"config.training.{key}") >= 0.0, f"config.training.{key} must be nonnegative")
    _compare(config.get("models"), FROZEN_MODEL_SPECS, path="config.models")

    budget = config.get("physical_budget")
    _require(isinstance(budget, dict), "config.physical_budget must be an object")
    _require(budget.get("cgls_base_iterations") == 12, "CGLS base budget must remain 12")
    _require(budget.get("learned_feature_forward_calls") == 1, "learned feature forward budget must remain 1")
    _require(budget.get("learned_feature_grouped_adjoint_calls") == 1, "learned grouped adjoint budget must remain 1")
    _require(budget.get("classical_comparator_iterations") == 13, "classical budget must remain 13")
    _require(budget.get("grouped_adjoint_is_not_equal_flop_to_pooled_adjoint") is True, "grouped-adjoint accounting caveat is missing")

    gates = config.get("decision_gates")
    expected_gate_keys = {
        "development_field_gain_over_best_classical_minimum_fraction",
        "development_h1_gain_over_best_classical_minimum_fraction",
        "ood_field_gain_over_best_classical_minimum_fraction",
        "ood_h1_gain_over_best_classical_minimum_fraction",
        "development_reprojection_ratio_to_cgls_maximum",
        "ood_reprojection_ratio_to_cgls_maximum",
        "field_harm_threshold_fraction",
        "field_harm_rate_maximum",
        "worst_field_gain_minimum_fraction",
        "require_all_three_model_seed_mean_field_gains_positive",
    }
    _require(isinstance(gates, dict) and set(gates) == expected_gate_keys, "decision gate schema drift")
    for key in expected_gate_keys - {"require_all_three_model_seed_mean_field_gains_positive"}:
        _strict_json_float(gates[key], path=f"config.decision_gates.{key}")
    _require(gates["require_all_three_model_seed_mean_field_gains_positive"] is True, "three-seed positivity gate must remain enabled")

    boundary = config.get("claim_boundary")
    _require(isinstance(boundary, dict), "config.claim_boundary must be an object")
    for key in (
        "is_experimental_reconstruction",
        "is_cfd_validation",
        "is_real_bost_generalization",
        "is_interface_detection_evidence",
        "is_confirmatory_or_final",
        "opens_fresh_or_final",
    ):
        _require(boundary.get(key) is False, f"config forbidden claim is open: {key}")
    return expected_cases


def _verify_checksums(
    checksums_path: Path,
    *,
    summary_path: Path,
    metric_rows_path: Path,
    aggregate_rows_path: Path,
    history_path: Path,
) -> int:
    _require(not checksums_path.is_symlink(), "checksum manifest must not be a symlink")
    parent = checksums_path.resolve().parent
    supplied = {
        "summary.json": summary_path,
        "metric_rows.csv": metric_rows_path,
        "aggregate_rows.csv": aggregate_rows_path,
        "training_history.csv": history_path,
    }
    for name, path in supplied.items():
        _require(path.resolve() == (parent / name).resolve(), f"{name}: path is outside checksum bundle")
    try:
        lines = checksums_path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValidationError(f"cannot read checksum manifest: {error}") from error
    _require(len(lines) == len(CHECKSUM_PAYLOADS), "checksum manifest length mismatch")
    observed: dict[str, str] = {}
    pattern = re.compile(r"([0-9a-f]{64})  ([A-Za-z0-9_.-]+)")
    for line in lines:
        match = pattern.fullmatch(line)
        _require(match is not None, "checksum manifest syntax mismatch")
        digest, name = match.groups()
        _require(name not in observed, f"duplicate checksum entry: {name}")
        _require(name in CHECKSUM_PAYLOADS, f"unexpected checksum payload: {name}")
        observed[name] = digest
    _require(set(observed) == CHECKSUM_PAYLOADS, "checksum manifest coverage mismatch")
    _require(list(observed) == sorted(observed), "checksum manifest order drift")
    for name, digest in observed.items():
        path = parent / name
        _require(path.is_file() and not path.is_symlink(), f"checksum payload is missing or symbolic: {name}")
        _require(_sha256(path) == digest, f"checksum mismatch: {name}")
    return len(observed)


def _validate_summary_header(
    summary: dict[str, Any], config: dict[str, Any], *, config_path: Path
) -> None:
    _require(set(summary) == SUMMARY_FIELDS, "summary top-level schema drift")
    _require(summary.get("schema_version") == REPORT_SCHEMA, "summary schema drift")
    _require(summary.get("source_config_sha256") == _sha256(config_path), "source config hash mismatch")
    _require(summary.get("evidence_level") == config.get("evidence_level"), "evidence level drift")
    _compare(summary.get("physical_budget"), config.get("physical_budget"), path="summary.physical_budget")
    _compare(summary.get("claim_boundary"), config.get("claim_boundary"), path="summary.claim_boundary")
    fixture = summary.get("fixture")
    _require(isinstance(fixture, dict), "summary.fixture must be an object")
    for key, value in config["fixture"].items():
        _compare(fixture.get(key), value, path=f"summary.fixture.{key}")
    export = summary.get("public_export_policy")
    _require(isinstance(export, dict), "summary.public_export_policy must be an object")
    _require(export.get("contains_truth_observation_or_geometry_arrays") is False, "raw arrays must not be exported")
    _require(export.get("contains_model_checkpoints") is False, "model checkpoints must not be exported")
    _require(export.get("contains_aggregate_and_per_case_metrics") is True, "metric export declaration is missing")


def _validate_case_manifest(
    summary: dict[str, Any], expected_cases: dict[str, set[tuple[int, str]]]
) -> dict[str, dict[str, Any]]:
    manifest = summary.get("case_manifest")
    _require(isinstance(manifest, list), "summary.case_manifest must be a list")
    _require(len(manifest) == sum(EXPECTED_SPLIT_CASE_COUNTS.values()), "case manifest must contain 62 cases")
    expected_fields = {
        "case_id",
        "split",
        "family",
        "base_seed",
        "geometry_digest",
        "observation_digest",
    }
    by_id: dict[str, dict[str, Any]] = {}
    observed_cases = {split: set() for split in SPLITS}
    hex20 = re.compile(r"[0-9a-f]{20}")
    hex64 = re.compile(r"[0-9a-f]{64}")
    for index, item in enumerate(manifest):
        _require(isinstance(item, dict) and set(item) == expected_fields, f"case_manifest[{index}]: schema drift")
        split = item["split"]
        _require(split in SPLITS, f"case_manifest[{index}]: fresh/final or unknown split")
        case_id = item["case_id"]
        family = item["family"]
        seed = _strict_json_int(item["base_seed"], path=f"case_manifest[{index}].base_seed")
        _require(isinstance(case_id, str) and hex20.fullmatch(case_id) is not None, f"case_manifest[{index}]: invalid case ID")
        _require(case_id not in by_id, f"case ID overlaps splits or repeats: {case_id}")
        _require(isinstance(family, str), f"case_manifest[{index}]: invalid family")
        for digest_key in ("geometry_digest", "observation_digest"):
            digest = item[digest_key]
            _require(isinstance(digest, str) and hex64.fullmatch(digest) is not None, f"case_manifest[{index}]: invalid {digest_key}")
        observed_cases[split].add((seed, family))
        by_id[case_id] = item
    for split in SPLITS:
        _require(observed_cases[split] == expected_cases[split], f"{split}: manifest/config case mismatch")
    counts = {split: sum(item["split"] == split for item in manifest) for split in SPLITS}
    _compare(summary.get("split_case_counts"), counts, path="summary.split_case_counts")
    _require(counts == EXPECTED_SPLIT_CASE_COUNTS, "summary split counts are not 32/12/18")
    return by_id


def _parse_metric_rows(path: Path) -> list[dict[str, Any]]:
    raw_rows = _read_csv(path, METRIC_FIELDS)
    rows: list[dict[str, Any]] = []
    float_fields = (
        "field_relative_l2",
        "field_rmse",
        "field_nrmse_dynamic_range",
        "field_mean_bias",
        "h1_seminorm_relative_error",
        "measured_reprojection_relative_l2",
        "clean_reprojection_relative_l2",
        "neural_inference_seconds",
    )
    int_fields = (
        "base_seed",
        "model_seed",
        "optimization_forward_calls",
        "optimization_adjoint_calls",
        "grouped_adjoint_calls",
        "evaluation_forward_calls",
    )
    for index, raw in enumerate(raw_rows):
        row: dict[str, Any] = {
            "case_id": raw["case_id"],
            "split": raw["split"],
            "family": raw["family"],
            "method": raw["method"],
        }
        for key in int_fields:
            row[key] = _csv_int(raw[key], path=f"metric_rows[{index}].{key}")
        for key in float_fields:
            row[key] = _csv_float(raw[key], path=f"metric_rows[{index}].{key}")
        row["gate"] = _csv_float(raw["gate"], path=f"metric_rows[{index}].gate", nullable=True)
        row["correction_rms"] = _csv_float(raw["correction_rms"], path=f"metric_rows[{index}].correction_rms", nullable=True)
        rows.append(row)
    return rows


def _validate_metric_rows(
    rows: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    manifest_by_id: dict[str, dict[str, Any]],
) -> None:
    _require(len(rows) == EXPECTED_METRIC_ROW_COUNT, "metric_rows.csv: expected 420 rows")
    expected_method_seeds = {
        *((method, -1) for method in CLASSICAL_METHODS),
        *((method, seed) for method in LEARNED_METHODS for seed in MODEL_SEEDS),
    }
    by_case: dict[str, set[tuple[str, int]]] = {}
    keys: set[tuple[str, str, int]] = set()
    learned_forward = int(config["physical_budget"]["cgls_base_iterations"]) + int(config["physical_budget"]["learned_feature_forward_calls"])
    learned_adjoint = int(config["physical_budget"]["cgls_base_iterations"]) + int(config["physical_budget"]["learned_feature_grouped_adjoint_calls"])
    classical_calls = int(config["physical_budget"]["classical_comparator_iterations"])
    for index, row in enumerate(rows):
        case_id = row["case_id"]
        _require(case_id in manifest_by_id, f"metric_rows[{index}]: unknown case ID")
        manifest = manifest_by_id[case_id]
        _require(manifest["split"] in EVALUATION_SPLITS, f"metric_rows[{index}]: train/fresh/final row is forbidden")
        for key in ("split", "family", "base_seed"):
            _require(row[key] == manifest[key], f"metric_rows[{index}]: case metadata mismatch for {key}")
        key = (case_id, row["method"], row["model_seed"])
        _require(key not in keys, f"duplicate metric row key: {key}")
        keys.add(key)
        method_seed = (row["method"], row["model_seed"])
        _require(method_seed in expected_method_seeds, f"metric_rows[{index}]: method/seed drift")
        by_case.setdefault(case_id, set()).add(method_seed)
        for metric in (
            "field_relative_l2",
            "field_rmse",
            "field_nrmse_dynamic_range",
            "h1_seminorm_relative_error",
            "measured_reprojection_relative_l2",
            "clean_reprojection_relative_l2",
            "neural_inference_seconds",
        ):
            _require(float(row[metric]) >= 0.0, f"metric_rows[{index}]: negative {metric}")
        if row["method"] in LEARNED_METHODS:
            _require(row["optimization_forward_calls"] == learned_forward == 13, f"metric_rows[{index}]: learned forward budget drift")
            _require(row["optimization_adjoint_calls"] == learned_adjoint == 13, f"metric_rows[{index}]: learned adjoint budget drift")
            _require(row["grouped_adjoint_calls"] == 1, f"metric_rows[{index}]: learned grouped-adjoint budget drift")
            _require(row["gate"] is not None and 0.0 <= row["gate"] <= 1.0, f"metric_rows[{index}]: invalid learned gate")
            _require(row["correction_rms"] is not None and row["correction_rms"] >= 0.0, f"metric_rows[{index}]: invalid correction RMS")
        else:
            _require(row["optimization_forward_calls"] == classical_calls == 13, f"metric_rows[{index}]: classical forward budget drift")
            _require(row["optimization_adjoint_calls"] == classical_calls == 13, f"metric_rows[{index}]: classical adjoint budget drift")
            _require(row["grouped_adjoint_calls"] == 0, f"metric_rows[{index}]: classical grouped-adjoint call")
            _require(row["gate"] is None and row["correction_rms"] is None, f"metric_rows[{index}]: classical neural fields must be empty")
            _require(row["neural_inference_seconds"] == 0.0, f"metric_rows[{index}]: classical neural timing must be zero")
        _require(row["evaluation_forward_calls"] == 1, f"metric_rows[{index}]: evaluation forward budget drift")

    evaluation_ids = {case_id for case_id, item in manifest_by_id.items() if item["split"] in EVALUATION_SPLITS}
    _require(set(by_case) == evaluation_ids, "metric rows do not cover all 30 evaluation cases")
    for case_id, observed in by_case.items():
        _require(observed == expected_method_seeds, f"{case_id}: incomplete 4x3 learned plus 2 classical ledger")


def _recompute_aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["method"], row["model_seed"], row["split"])
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (method, model_seed, split), values in sorted(grouped.items()):
        gates = [row["gate"] for row in values]
        corrections = [row["correction_rms"] for row in values]
        _require(all(value is None for value in gates) or all(value is not None for value in gates), f"{method}/{model_seed}/{split}: mixed gate nullability")
        _require(all(value is None for value in corrections) or all(value is not None for value in corrections), f"{method}/{model_seed}/{split}: mixed correction nullability")
        output.append(
            {
                "method": method,
                "model_seed": model_seed,
                "split": split,
                "case_count": len(values),
                "field_relative_l2_mean": _mean(float(row["field_relative_l2"]) for row in values),
                "field_relative_l2_maximum": max(float(row["field_relative_l2"]) for row in values),
                "h1_seminorm_relative_error_mean": _mean(float(row["h1_seminorm_relative_error"]) for row in values),
                "measured_reprojection_relative_l2_mean": _mean(float(row["measured_reprojection_relative_l2"]) for row in values),
                "gate_mean": None if gates[0] is None else _mean(float(value) for value in gates if value is not None),
                "correction_rms_mean": None if corrections[0] is None else _mean(float(value) for value in corrections if value is not None),
                "neural_inference_seconds_mean": _mean(float(row["neural_inference_seconds"]) for row in values),
            }
        )
    return output


def _parse_aggregate_rows(path: Path) -> list[dict[str, Any]]:
    raw_rows = _read_csv(path, AGGREGATE_FIELDS)
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_rows):
        rows.append(
            {
                "method": raw["method"],
                "model_seed": _csv_int(raw["model_seed"], path=f"aggregate_rows[{index}].model_seed"),
                "split": raw["split"],
                "case_count": _csv_int(raw["case_count"], path=f"aggregate_rows[{index}].case_count"),
                "field_relative_l2_mean": _csv_float(raw["field_relative_l2_mean"], path=f"aggregate_rows[{index}].field_relative_l2_mean"),
                "field_relative_l2_maximum": _csv_float(raw["field_relative_l2_maximum"], path=f"aggregate_rows[{index}].field_relative_l2_maximum"),
                "h1_seminorm_relative_error_mean": _csv_float(raw["h1_seminorm_relative_error_mean"], path=f"aggregate_rows[{index}].h1_seminorm_relative_error_mean"),
                "measured_reprojection_relative_l2_mean": _csv_float(raw["measured_reprojection_relative_l2_mean"], path=f"aggregate_rows[{index}].measured_reprojection_relative_l2_mean"),
                "gate_mean": _csv_float(raw["gate_mean"], path=f"aggregate_rows[{index}].gate_mean", nullable=True),
                "correction_rms_mean": _csv_float(raw["correction_rms_mean"], path=f"aggregate_rows[{index}].correction_rms_mean", nullable=True),
                "neural_inference_seconds_mean": _csv_float(raw["neural_inference_seconds_mean"], path=f"aggregate_rows[{index}].neural_inference_seconds_mean"),
            }
        )
    return rows


def _compare_aggregate_table(actual: Any, expected: list[dict[str, Any]], *, path: str) -> None:
    _require(isinstance(actual, list), f"{path}: expected list")
    expected_by_key = {(row["method"], row["model_seed"], row["split"]): row for row in expected}
    actual_by_key: dict[tuple[Any, Any, Any], Any] = {}
    for index, row in enumerate(actual):
        _require(isinstance(row, dict) and set(row) == set(AGGREGATE_FIELDS), f"{path}[{index}]: schema drift")
        key = (row["method"], row["model_seed"], row["split"])
        _require(key not in actual_by_key, f"{path}: duplicate aggregate key {key}")
        actual_by_key[key] = row
    _require(set(actual_by_key) == set(expected_by_key), f"{path}: aggregate key set mismatch")
    for key, expected_row in expected_by_key.items():
        _compare(actual_by_key[key], expected_row, path=f"{path}[{key}]")


def _parse_history(path: Path) -> list[dict[str, Any]]:
    raw_rows = _read_csv(path, HISTORY_FIELDS)
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_rows):
        row: dict[str, Any] = {
            "method": raw["method"],
            "model_seed": _csv_int(raw["model_seed"], path=f"training_history[{index}].model_seed"),
            "epoch": _csv_int(raw["epoch"], path=f"training_history[{index}].epoch"),
        }
        for key in HISTORY_FIELDS[3:]:
            row[key] = _csv_float(raw[key], path=f"training_history[{index}].{key}")
        rows.append(row)
    return rows


def _validate_training_ledger(
    history: list[dict[str, Any]], summary: dict[str, Any], config: dict[str, Any]
) -> list[dict[str, Any]]:
    training = config["training"]
    expected_runs = [(method, seed) for method in LEARNED_METHODS for seed in MODEL_SEEDS]
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    seen_history_keys: set[tuple[str, int, int]] = set()
    for index, row in enumerate(history):
        run_key = (row["method"], row["model_seed"])
        _require(run_key in expected_runs, f"training_history[{index}]: method/seed drift")
        history_key = (*run_key, row["epoch"])
        _require(history_key not in seen_history_keys, f"duplicate history row: {history_key}")
        seen_history_keys.add(history_key)
        _require(row["epoch"] >= 1, f"training_history[{index}]: invalid epoch")
        for key in HISTORY_FIELDS[3:]:
            _require(float(row[key]) >= 0.0, f"training_history[{index}]: negative {key}")
        expected_total = (
            float(row["train_field"])
            + float(training["lambda_h1"]) * float(row["train_h1"])
            + float(training["lambda_correction"]) * float(row["train_correction"])
            + float(training["lambda_gate"]) * float(row["train_gate"])
        )
        _require(math.isclose(float(row["train_total"]), expected_total, rel_tol=2e-6, abs_tol=2e-7), f"training_history[{index}]: objective ledger mismatch")
        expected_lr = float(training["learning_rate"]) * (
            1.0 + math.cos(math.pi * row["epoch"] / int(training["epochs"]))
        ) / 2.0
        _require(math.isclose(float(row["learning_rate"]), expected_lr, rel_tol=3e-12, abs_tol=3e-14), f"training_history[{index}]: learning-rate/epoch ledger mismatch")
        groups.setdefault(run_key, []).append(row)
    _require(set(groups) == set(expected_runs), "training history does not cover all 4x3 runs")

    runs = summary.get("training_runs")
    _require(isinstance(runs, list) and len(runs) == len(expected_runs), "summary must contain 12 training runs")
    run_fields = {
        "method",
        "model_seed",
        "parameters",
        "best_epoch",
        "best_development_field_relative_l2",
        "epochs_ran",
        "train_seconds",
        "device",
    }
    observed_order: list[tuple[str, int]] = []
    ledger: list[dict[str, Any]] = []
    expected_history_order: list[tuple[str, int, int]] = []
    for index, run in enumerate(runs):
        _require(isinstance(run, dict) and set(run) == run_fields, f"training_runs[{index}]: schema drift")
        method = run["method"]
        seed = _strict_json_int(run["model_seed"], path=f"training_runs[{index}].model_seed")
        run_key = (method, seed)
        observed_order.append(run_key)
        _require(run_key in groups, f"training_runs[{index}]: unexpected method/seed")
        parameters = _strict_json_int(run["parameters"], path=f"training_runs[{index}].parameters")
        _require(parameters == EXPECTED_PARAMETER_COUNTS[method], f"training_runs[{index}]: parameter ledger drift for {method}")
        epochs_ran = _strict_json_int(run["epochs_ran"], path=f"training_runs[{index}].epochs_ran")
        best_epoch = _strict_json_int(run["best_epoch"], path=f"training_runs[{index}].best_epoch")
        selected = groups[run_key]
        _require([row["epoch"] for row in selected] == list(range(1, epochs_ran + 1)), f"training_runs[{index}]: epochs are not contiguous")
        _require(len(selected) == epochs_ran <= int(training["epochs"]), f"training_runs[{index}]: epochs_ran mismatch")
        expected_history_order.extend((method, seed, epoch) for epoch in range(1, epochs_ran + 1))
        recomputed_best = math.inf
        recomputed_epoch = 0
        for row in selected:
            value = float(row["development_field_relative_l2"])
            if value < recomputed_best - 1e-6:
                recomputed_best = value
                recomputed_epoch = int(row["epoch"])
        _require(best_epoch == recomputed_epoch, f"training_runs[{index}]: best_epoch ledger mismatch")
        reported_best = _strict_json_float(run["best_development_field_relative_l2"], path=f"training_runs[{index}].best_development_field_relative_l2")
        _require(math.isclose(reported_best, recomputed_best, rel_tol=3e-7, abs_tol=3e-8), f"training_runs[{index}]: best development value mismatch")
        _require(1 <= best_epoch <= epochs_ran, f"training_runs[{index}]: best epoch out of range")
        if epochs_ran < int(training["epochs"]):
            _require(epochs_ran >= int(training["minimum_epoch"]), f"training_runs[{index}]: early stop before minimum epoch")
            _require(epochs_ran - best_epoch >= int(training["early_stop_patience"]), f"training_runs[{index}]: early-stop patience ledger mismatch")
        seconds = _strict_json_float(run["train_seconds"], path=f"training_runs[{index}].train_seconds")
        _require(seconds >= 0.0, f"training_runs[{index}]: negative training time")
        _require(isinstance(run["device"], str) and run["device"] == summary.get("device"), f"training_runs[{index}]: device ledger mismatch")
        ledger.append(
            {
                "method": method,
                "model_seed": seed,
                "parameters": parameters,
                "best_epoch": best_epoch,
                "epochs_ran": epochs_ran,
            }
        )
    _require(observed_order == expected_runs, "training run method/seed order drift")
    actual_history_order = [(row["method"], row["model_seed"], row["epoch"]) for row in history]
    _require(actual_history_order == expected_history_order, "training history block order drift")
    return ledger


def _recompute_method_decisions(
    rows: list[dict[str, Any]], gates: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    classical = [row for row in rows if row["method"] in CLASSICAL_METHODS]
    learned = [row for row in rows if row["method"] in LEARNED_METHODS]
    classical_lookup = {(row["case_id"], row["method"]): row for row in classical}
    best_by_split: dict[str, str] = {}
    for split in EVALUATION_SPLITS:
        best_by_split[split] = min(
            CLASSICAL_METHODS,
            key=lambda method: _mean(
                float(row["field_relative_l2"])
                for row in classical
                if row["split"] == split and row["method"] == method
            ),
        )

    decisions: dict[str, Any] = {}
    paired_rows: list[dict[str, Any]] = []
    for method in LEARNED_METHODS:
        diagnostics: dict[str, float] = {}
        seed_gains: dict[str, list[float]] = {split: [] for split in EVALUATION_SPLITS}
        for split in EVALUATION_SPLITS:
            baseline_method = best_by_split[split]
            split_rows = sorted(
                (row for row in learned if row["method"] == method and row["split"] == split),
                key=lambda row: (row["model_seed"], row["case_id"]),
            )
            local_pairs: list[dict[str, Any]] = []
            for row in split_rows:
                baseline = classical_lookup[(row["case_id"], baseline_method)]
                cgls = classical_lookup[(row["case_id"], "cgls_13")]
                field_denominator = float(baseline["field_relative_l2"])
                h1_denominator = float(baseline["h1_seminorm_relative_error"])
                _require(field_denominator > 0.0 and h1_denominator > 0.0, f"{row['case_id']}: nonpositive paired baseline")
                field_gain = 1.0 - float(row["field_relative_l2"]) / field_denominator
                h1_gain = 1.0 - float(row["h1_seminorm_relative_error"]) / h1_denominator
                reprojection_ratio = float(row["measured_reprojection_relative_l2"]) / max(float(cgls["measured_reprojection_relative_l2"]), 1e-30)
                pair = {
                    "case_id": row["case_id"],
                    "split": split,
                    "method": method,
                    "model_seed": row["model_seed"],
                    "baseline_method": baseline_method,
                    "field_gain": field_gain,
                    "h1_gain": h1_gain,
                    "reprojection_ratio_to_cgls": reprojection_ratio,
                    "field_harmed": field_gain < -float(gates["field_harm_threshold_fraction"]),
                }
                local_pairs.append(pair)
                paired_rows.append(pair)
            _require(bool(local_pairs), f"{method}/{split}: missing paired comparisons")
            field_values = [float(pair["field_gain"]) for pair in local_pairs]
            h1_values = [float(pair["h1_gain"]) for pair in local_pairs]
            reprojection_values = [float(pair["reprojection_ratio_to_cgls"]) for pair in local_pairs]
            diagnostics[f"{split}_field_gain_mean"] = _mean(field_values)
            diagnostics[f"{split}_h1_gain_mean"] = _mean(h1_values)
            diagnostics[f"{split}_reprojection_ratio_mean"] = _mean(reprojection_values)
            diagnostics[f"{split}_field_harm_rate"] = _mean(1.0 if pair["field_harmed"] else 0.0 for pair in local_pairs)
            diagnostics[f"{split}_worst_field_gain"] = min(field_values)
            for seed in MODEL_SEEDS:
                seed_gains[split].append(
                    _mean(
                        float(pair["field_gain"])
                        for pair in local_pairs
                        if pair["model_seed"] == seed
                    )
                )
        checks = {
            "development_field_gain": diagnostics["development_field_gain_mean"] >= float(gates["development_field_gain_over_best_classical_minimum_fraction"]),
            "development_h1_gain": diagnostics["development_h1_gain_mean"] >= float(gates["development_h1_gain_over_best_classical_minimum_fraction"]),
            "ood_field_gain": diagnostics["ood_field_gain_mean"] >= float(gates["ood_field_gain_over_best_classical_minimum_fraction"]),
            "ood_h1_gain": diagnostics["ood_h1_gain_mean"] >= float(gates["ood_h1_gain_over_best_classical_minimum_fraction"]),
            "development_reprojection": diagnostics["development_reprojection_ratio_mean"] <= float(gates["development_reprojection_ratio_to_cgls_maximum"]),
            "ood_reprojection": diagnostics["ood_reprojection_ratio_mean"] <= float(gates["ood_reprojection_ratio_to_cgls_maximum"]),
            "development_harm": diagnostics["development_field_harm_rate"] <= float(gates["field_harm_rate_maximum"]),
            "ood_harm": diagnostics["ood_field_harm_rate"] <= float(gates["field_harm_rate_maximum"]),
            "worst_case": min(diagnostics["development_worst_field_gain"], diagnostics["ood_worst_field_gain"]) >= float(gates["worst_field_gain_minimum_fraction"]),
            "all_seed_means_positive": all(gain > 0.0 for values in seed_gains.values() for gain in values),
        }
        decisions[method] = {
            "passed": all(checks.values()),
            "best_classical_by_split": dict(best_by_split),
            "checks": checks,
            "diagnostics": diagnostics,
            "per_seed_field_gain_means": seed_gains,
        }
    _require(len(paired_rows) == len(LEARNED_METHODS) * len(MODEL_SEEDS) * 30, "paired comparison ledger must contain 360 rows")
    pair_keys = {(row["case_id"], row["method"], row["model_seed"]) for row in paired_rows}
    _require(len(pair_keys) == len(paired_rows), "duplicate paired comparison")
    return decisions, paired_rows


def _validate_no_go_and_authorization(
    summary: dict[str, Any], decisions: dict[str, Any], config: dict[str, Any]
) -> tuple[str, list[str]]:
    primary = LEARNED_METHODS[0]
    _require(config["methods"][0] == primary, "frozen primary method drift")
    _require(summary.get("primary_method") == primary, "summary primary method drift")
    recomputed_passed = decisions[primary]["passed"]
    _require(recomputed_passed is False, "primitive metrics no longer independently produce primary NO-GO")
    _require(summary.get("primary_passed") is False, "summary.primary_passed contradicts independently recomputed NO-GO")
    _require(summary.get("status") == EXPECTED_NO_GO_STATUS, "summary status must remain M2 T0 NO-GO")
    _compare(summary.get("method_decisions"), decisions, path="summary.method_decisions")
    authorization = summary.get("authorization")
    _require(isinstance(authorization, dict) and set(authorization) == EXPECTED_AUTHORIZATION_KEYS, "authorization schema drift")
    for key, value in authorization.items():
        _require(value is False, f"authorization must remain false: {key}")
    failed_checks = sorted(key for key, passed in decisions[primary]["checks"].items() if not passed)
    return primary, failed_checks


def _validate_packet_impl(
    *,
    config_path: Path,
    summary_path: Path,
    metric_rows_path: Path,
    aggregate_rows_path: Path,
    history_path: Path,
    checksums_path: Path,
) -> dict[str, Any]:
    checksum_count = _verify_checksums(
        checksums_path,
        summary_path=summary_path,
        metric_rows_path=metric_rows_path,
        aggregate_rows_path=aggregate_rows_path,
        history_path=history_path,
    )
    config = _load_json(config_path)
    summary = _load_json(summary_path)
    expected_cases = _validate_config(config)
    _validate_summary_header(summary, config, config_path=config_path)
    manifest_by_id = _validate_case_manifest(summary, expected_cases)

    metric_rows = _parse_metric_rows(metric_rows_path)
    _validate_metric_rows(metric_rows, config=config, manifest_by_id=manifest_by_id)
    _require(summary.get("metric_row_count") == EXPECTED_METRIC_ROW_COUNT, "summary metric row count drift")

    recomputed_aggregate = _recompute_aggregate(metric_rows)
    _require(len(recomputed_aggregate) == 28, "aggregate ledger must contain 28 rows")
    observed_aggregate = _parse_aggregate_rows(aggregate_rows_path)
    _compare_aggregate_table(observed_aggregate, recomputed_aggregate, path="aggregate_rows.csv")
    _compare_aggregate_table(summary.get("aggregate"), recomputed_aggregate, path="summary.aggregate")

    history = _parse_history(history_path)
    training_ledger = _validate_training_ledger(history, summary, config)
    decisions, paired_rows = _recompute_method_decisions(metric_rows, config["decision_gates"])
    primary, failed_checks = _validate_no_go_and_authorization(summary, decisions, config)
    paired_digest = hashlib.sha256(
        json.dumps(paired_rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()
    return {
        "schema_version": VALIDATION_SCHEMA,
        "status": "VALIDATED_M2_T0_NO_GO",
        "source_config_sha256": _sha256(config_path),
        "split_case_counts": dict(EXPECTED_SPLIT_CASE_COUNTS),
        "metric_row_count": len(metric_rows),
        "aggregate_row_count": len(recomputed_aggregate),
        "training_history_row_count": len(history),
        "training_run_count": len(training_ledger),
        "parameter_counts": dict(EXPECTED_PARAMETER_COUNTS),
        "primary_method": primary,
        "primary_passed": False,
        "primary_failed_checks": failed_checks,
        "method_passed": {method: decision["passed"] for method, decision in decisions.items()},
        "paired_comparison_count": len(paired_rows),
        "paired_comparison_sha256": paired_digest,
        "checksum_payload_count": checksum_count,
        "authorization": {key: False for key in sorted(EXPECTED_AUTHORIZATION_KEYS)},
        "training_ledger": training_ledger,
    }


def validate_packet(
    *,
    config_path: Path = DEFAULT_CONFIG,
    summary_path: Path = DEFAULT_SUMMARY,
    metric_rows_path: Path = DEFAULT_METRIC_ROWS,
    aggregate_rows_path: Path = DEFAULT_AGGREGATE_ROWS,
    history_path: Path = DEFAULT_HISTORY,
    checksums_path: Path = DEFAULT_CHECKSUMS,
) -> dict[str, Any]:
    """Validate the packet without importing or trusting runner computations."""

    try:
        return _validate_packet_impl(
            config_path=Path(config_path),
            summary_path=Path(summary_path),
            metric_rows_path=Path(metric_rows_path),
            aggregate_rows_path=Path(aggregate_rows_path),
            history_path=Path(history_path),
            checksums_path=Path(checksums_path),
        )
    except ValidationError:
        raise
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, OSError) as error:
        raise ValidationError(f"malformed M2-T0 evidence packet: {error}") from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--metric-rows", type=Path, default=DEFAULT_METRIC_ROWS)
    parser.add_argument("--aggregate-rows", type=Path, default=DEFAULT_AGGREGATE_ROWS)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--checksums", type=Path, default=DEFAULT_CHECKSUMS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_packet(
        config_path=args.config,
        summary_path=args.summary,
        metric_rows_path=args.metric_rows,
        aggregate_rows_path=args.aggregate_rows,
        history_path=args.history,
        checksums_path=args.checksums,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
