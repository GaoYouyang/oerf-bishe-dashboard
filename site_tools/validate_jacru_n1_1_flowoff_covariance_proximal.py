#!/usr/bin/env python3
"""Independently audit the frozen JACRU N1.1 public evidence packet.

This validator deliberately uses only the Python standard library plus the
frozen config and public scalar artifacts.  It does not import the N1.1
runner, a learned model, or an optical operator.  All CSV aggregates,
calibration gates, candidate/method decisions, status, and authorization are
reconstructed from the public rows before the summary is trusted.
"""

from __future__ import annotations

import argparse
import csv
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
    "jacru_n1_1_flowoff_covariance_proximal_postopen_v1.json"
)
OUTPUT = (
    ROOT
    / "demo_t16_operator/results/"
    "jacru_n1_1_flowoff_covariance_proximal_postopen_public"
)

CONFIG_SCHEMA = "jacru-n1-1-flowoff-covariance-proximal-postopen-config-1.0"
REPORT_SCHEMA = "jacru-n1-1-flowoff-covariance-proximal-postopen-report-1.0"
VALIDATED_STATUS = "VALIDATED_N1_1_FLOWOFF_COVARIANCE_PROXIMAL_NO_GO"

PACKET_PAYLOADS = {
    "README.md",
    "aggregate_rows.csv",
    "calibration_rows.csv",
    "dense_setup_rows.csv",
    "diagnostic.pdf",
    "diagnostic.png",
    "metric_rows.csv",
    "reference_rows.csv",
    "summary.json",
}

CALIBRATION_FIELDS = (
    "case_id",
    "split",
    "family",
    "base_seed",
    "geometry_digest",
    "mode",
    "payload_digest",
    "fit_repeats",
    "threshold_calibration_repeats",
    "audit_repeats",
    "discrepancy_quantile",
    "iid_noise_std",
    "camera_bias_std",
    "estimated_condition_number",
    "estimated_minimum_eigenvalue",
    "estimated_maximum_eigenvalue",
    "mean_error_relative_iid",
    "covariance_relative_frobenius_error",
    "empirical_threshold",
    "empirical_selection_score_mean",
    "empirical_selection_score_maximum",
    "empirical_audit_coverage",
    "empirical_audit_score_mean",
    "exact_threshold",
    "exact_selection_score_mean",
    "exact_audit_coverage",
    "exact_audit_score_mean",
)

METRIC_FIELDS = (
    "candidate_id",
    "calibration_mode",
    "mean_policy",
    "proximal_covariance_policy",
    "selector_covariance_policy",
    "threshold_policy",
    "uses_truth",
    "uses_exact_nuisance",
    "dense_ceiling_only",
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
    "field_gain_to_best_matched",
    "h1_gain_to_best_matched",
    "clean_reprojection_ratio_to_base",
    "measured_reprojection_ratio_to_cgls",
    "field_harm",
    "sensor_discrepancy_threshold",
    "selected_discrepancy_threshold",
    "truth_residual_oracle_threshold",
    "raw_discrepancy",
    "selected_discrepancy",
    "alpha",
    "target_crossed",
    "raw_no_correction",
    "correction_norm",
    "measurement_residual_norm",
    "proximal_covariance_scale",
    "residual_closure_relative_error",
    "bisection_iterations",
    "dense_matrix_rows",
    "dense_matrix_columns",
    "dense_setup_forward_equivalents",
    "dense_setup_in_budget",
    "learned_feature_forward_calls",
    "learned_feature_adjoint_calls",
)

AGGREGATE_FIELDS = (
    "candidate_id",
    "method",
    "model_seed",
    "split",
    "case_count",
    "field_gain_mean",
    "h1_gain_mean",
    "clean_reprojection_ratio_to_base_mean",
    "clean_reprojection_ratio_to_base_maximum",
    "measured_reprojection_ratio_to_cgls_mean",
    "field_harm_rate",
    "worst_field_gain",
    "target_crossing_rate",
    "raw_no_correction_rate",
    "log10_alpha_mean_finite",
    "residual_closure_relative_error_maximum",
    "correction_norm_mean",
)

DENSE_SETUP_FIELDS = (
    "geometry_digest",
    "matrix_rows",
    "matrix_columns",
    "rank",
    "rank_tolerance",
    "setup_forward_calls_batched",
    "setup_forward_equivalents_unbatched",
    "assembly_batch_size",
    "zero_forward_maximum_absolute",
    "factorization_seconds",
    "dense_setup_in_budget",
    "status",
)

REFERENCE_FIELDS = (
    "reference_kind",
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

SUMMARY_FIELDS = {
    "aggregate_row_count",
    "authorization",
    "calibration_decisions",
    "calibration_row_count",
    "candidate_count",
    "claim_boundary",
    "decisions",
    "dense_setup_row_count",
    "deployable_input_pass_count",
    "device",
    "evidence_level",
    "metric_row_count",
    "oracle_pass_count",
    "reference_row_count",
    "runtime_seconds",
    "schema_version",
    "source_config",
    "source_config_sha256",
    "source_hashes",
    "status",
}

SOURCE_PATHS = {
    "source_t0_config": ("source_t0_config", None),
    "source_t0_summary": ("source_t0_results", "summary.json"),
    "source_m2_7_config": ("source_m2_7_config", None),
    "source_m2_7_summary": ("source_m2_7_results", "summary.json"),
    "source_m2_8_config": ("source_m2_8_config", None),
    "source_m2_8_summary": ("source_m2_8_results", "summary.json"),
    "source_n1_0_config": ("source_n1_0_config", None),
    "source_n1_0_summary": ("source_n1_0_results", "summary.json"),
    "implementation_calibration_module": ("implementation_calibration_module", None),
    "implementation_dense_assembler": ("implementation_dense_assembler", None),
    "implementation_runner": ("implementation_runner", None),
}

FORBIDDEN_PUBLIC_KEYS = {
    "password",
    "vpn_password",
    "api_key",
    "access_token",
    "raw_observations",
    "raw_measurements",
    "measurement_array",
    "prediction_array",
    "field_truth_array",
    "clean_field_array",
    "fit_samples_uv",
    "selection_samples_uv",
    "audit_samples_uv",
    "persistent_bias_uv",
    "exact_persistent_bias_uv",
    "hidden_clean_scale_value",
}
FORBIDDEN_TEXT = (
    "/Users/",
    "file://",
    "webvpn.xmu.edu.cn",
    "BEGIN PRIVATE KEY",
)


class ValidationError(RuntimeError):
    """Raised when the public packet violates its frozen evidence contract."""


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


def _optional_float(value: Any, label: str) -> float | None:
    if value == "" or value is None:
        return None
    return _finite_float(value, label)


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


def _linear_quantile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    _need(bool(ordered), "cannot take quantile of empty values")
    _need(0.0 <= probability <= 1.0, "quantile probability outside [0, 1]")
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _read_csv_exact(
    path: Path,
    fields: Sequence[str],
) -> list[dict[str, str]]:
    try:
        physical = path.read_text(encoding="utf-8").splitlines()
        _need(bool(physical) and all(physical), f"{path.name}: blank physical line")
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            _need(
                tuple(reader.fieldnames or ()) == tuple(fields),
                f"{path.name}: columns differ from frozen schema",
            )
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValidationError(f"cannot read CSV {path}: {error}") from error
    _need(len(physical) == len(rows) + 1, f"{path.name}: physical row count drift")
    _need(
        all(None not in row and all(value is not None for value in row.values()) for row in rows),
        f"{path.name}: malformed row",
    )
    return rows


def _validate_manifest(directory: Path) -> None:
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
    _need(set(entries) == PACKET_PAYLOADS, "checksums.sha256: payload set mismatch")
    try:
        actual = {path.name for path in directory.iterdir()}
    except OSError as error:
        raise ValidationError(f"cannot inspect evidence directory: {error}") from error
    _need(
        actual == PACKET_PAYLOADS | {"checksums.sha256"},
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


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def _validate_no_obvious_leakage(
    output: Path,
    summary: Mapping[str, Any],
    csv_fields: Iterable[str],
) -> None:
    public_keys = set(_walk_keys(summary)) | set(csv_fields)
    leaked_keys = sorted(public_keys & FORBIDDEN_PUBLIC_KEYS)
    _need(not leaked_keys, f"obvious private/leakage fields present: {leaked_keys}")
    for name in ("README.md", "summary.json"):
        try:
            text = (output / name).read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise ValidationError(f"cannot scan public text {name}: {error}") from error
        for fragment in FORBIDDEN_TEXT:
            _need(fragment not in text, f"{name}: private path or credential endpoint leaked")


def _safe_source_path(config: Mapping[str, Any], field: str, child: str | None) -> Path:
    raw = ROOT / str(config[field])
    path = raw / child if child is not None else raw
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(ROOT.resolve())
    except (OSError, ValueError) as error:
        raise ValidationError(f"{field}: source path missing or escapes repository") from error
    _need(resolved.is_file() and not resolved.is_symlink(), f"{field}: invalid source file")
    return resolved


def _validate_sources(config: Mapping[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for summary_key, (field, child) in SOURCE_PATHS.items():
        path = _safe_source_path(config, field, child)
        digest = _sha256(path)
        expected_field = f"{summary_key}_sha256"
        _need(expected_field in config, f"config missing {expected_field}")
        _need(digest == config[expected_field], f"{summary_key} hash drift")
        observed[summary_key] = digest

    n10 = _read_json(_safe_source_path(config, "source_n1_0_results", "summary.json"))
    _need(
        n10.get("status") == "N1_0_OBSERVABLE_DISCREPANCY_STOPPING_NO_GO",
        "N1.0 source status drift",
    )
    _need(
        n10.get("authorization", {}).get("continue_flow_off_covariance_research") is True,
        "N1.0 source did not authorize flow-off covariance research",
    )
    return observed


def _validate_config(config: Mapping[str, Any]) -> None:
    _need(config.get("schema_version") == CONFIG_SCHEMA, "N1.1 config schema drift")
    _need(
        config.get("report_schema_version") == REPORT_SCHEMA,
        "N1.1 report schema drift",
    )
    _need(
        config.get("status") == "FROZEN_BEFORE_FIRST_FORMAL_N1_1_EXECUTION",
        "N1.1 config is not frozen",
    )
    methods = config.get("methods")
    _need(isinstance(methods, list) and len(methods) == len(set(methods)) > 0, "method grid drift")
    candidates = config.get("candidates")
    _need(isinstance(candidates, list) and len(candidates) > 0, "candidate grid missing")
    ids = [str(value.get("id")) for value in candidates if isinstance(value, Mapping)]
    _need(len(ids) == len(candidates) == len(set(ids)), "candidate IDs overlap or are malformed")
    for candidate in candidates:
        _need(isinstance(candidate, Mapping), "candidate must be an object")
        required = {
            "id",
            "calibration_mode",
            "mean_policy",
            "proximal_covariance_policy",
            "selector_covariance_policy",
            "threshold_policy",
            "discrepancy_quantile",
            "uses_truth",
            "uses_exact_nuisance",
        }
        _need(set(candidate) == required, f"candidate {candidate.get('id')}: schema drift")
        _need(
            candidate["calibration_mode"] in {"paired_static", "unpaired_distribution"},
            f"candidate {candidate['id']}: calibration mode drift",
        )
        truth_policy = candidate["threshold_policy"] == "truth_residual_oracle"
        _need(
            bool(candidate["uses_truth"]) == truth_policy,
            f"candidate {candidate['id']}: truth flag/policy mismatch",
        )
        exact_policy = "exact_generator_oracle" in {
            candidate["proximal_covariance_policy"],
            candidate["selector_covariance_policy"],
        } or candidate["mean_policy"] == "exact_persistent_bias_oracle"
        _need(
            bool(candidate["uses_exact_nuisance"]) == exact_policy,
            f"candidate {candidate['id']}: exact nuisance flag/policy mismatch",
        )

    limitations = config.get("current_selector_limitations", {})
    _need(limitations.get("uses_only_global_whitened_discrepancy") is True, "global selector boundary drift")
    _need(limitations.get("has_per_camera_upper_gate") is False, "unexpected per-camera gate claim")
    _need(limitations.get("has_lower_discrepancy_gate") is False, "unexpected lower gate claim")
    claim = config.get("claim_boundary", {})
    required_false = (
        "flowoff_payload_exposes_hidden_scale_or_nuisance",
        "clean_reprojection_is_independent_renderer",
        "dense_aa_t_ceiling_is_deployable",
        "may_claim_runtime_or_efficiency",
        "may_claim_method_superiority",
        "may_claim_real_bost_generalization",
        "may_open_fresh_or_final",
    )
    _need(all(claim.get(key) is False for key in required_false), "claim boundary overstates N1.1")
    _need(claim.get("uses_only_opened_synthetic_t0") is True, "opened-only boundary missing")
    _need(
        claim.get("clean_reprojection_is_same_voxel_operator_against_continuous_clean_target") is True,
        "clean-target operator boundary missing",
    )


def _case_metadata(
    rows: Sequence[Mapping[str, str]],
    config: Mapping[str, Any],
) -> tuple[dict[str, tuple[str, str, int, str]], dict[str, dict[str, Mapping[str, str]]]]:
    flowoff = config["flowoff_calibration"]
    expected_modes = {"paired_static", "unpaired_distribution"}
    cases: dict[str, tuple[str, str, int, str]] = {}
    grouped: dict[str, dict[str, Mapping[str, str]]] = {}
    payloads: set[str] = set()
    for index, row in enumerate(rows):
        label = f"calibration_rows[{index}]"
        case_id = row["case_id"]
        mode = row["mode"]
        _need(bool(case_id) and mode in expected_modes, f"{label}: identity drift")
        metadata = (
            row["split"],
            row["family"],
            _integer(row["base_seed"], f"{label}.base_seed"),
            row["geometry_digest"],
        )
        _need(metadata[0] in {"development", "ood"}, f"{label}: split drift")
        _need(bool(metadata[1]) and re.fullmatch(r"[0-9a-f]{64}", metadata[3]) is not None, f"{label}: metadata drift")
        if case_id in cases:
            _need(cases[case_id] == metadata, f"{label}: case metadata drift")
        else:
            cases[case_id] = metadata
        _need(mode not in grouped.setdefault(case_id, {}), f"{label}: duplicate case/mode")
        grouped[case_id][mode] = row
        _need(re.fullmatch(r"[0-9a-f]{64}", row["payload_digest"]) is not None, f"{label}: payload digest drift")
        _need(row["payload_digest"] not in payloads, f"{label}: payload stream reused")
        payloads.add(row["payload_digest"])
        for field in ("fit_repeats", "threshold_calibration_repeats", "audit_repeats"):
            _need(
                _integer(row[field], f"{label}.{field}") == int(flowoff[field]),
                f"{label}: {field} drift",
            )
        _close(row["discrepancy_quantile"], float(flowoff["discrepancy_quantile"]), f"{label}.quantile")
        minimum = _finite_float(row["estimated_minimum_eigenvalue"], f"{label}.minimum_eigenvalue")
        maximum = _finite_float(row["estimated_maximum_eigenvalue"], f"{label}.maximum_eigenvalue")
        condition = _finite_float(row["estimated_condition_number"], f"{label}.condition_number")
        _need(0.0 < minimum <= maximum, f"{label}: covariance is not SPD")
        _close(condition, maximum / minimum, f"{label}.condition_number", tolerance=2e-9)
        for field in (
            "iid_noise_std",
            "camera_bias_std",
            "mean_error_relative_iid",
            "covariance_relative_frobenius_error",
            "empirical_threshold",
            "empirical_selection_score_mean",
            "empirical_selection_score_maximum",
            "empirical_audit_score_mean",
            "exact_threshold",
            "exact_selection_score_mean",
            "exact_audit_score_mean",
        ):
            _need(_finite_float(row[field], f"{label}.{field}") >= 0.0, f"{label}: negative {field}")
        for field in ("empirical_audit_coverage", "exact_audit_coverage"):
            coverage = _finite_float(row[field], f"{label}.{field}")
            _need(0.0 <= coverage <= 1.0, f"{label}: coverage outside [0, 1]")
            repeats = int(flowoff["audit_repeats"])
            _close(coverage * repeats, round(coverage * repeats), f"{label}.{field}.count")
    _need(bool(cases), "calibration rows are empty")
    _need(all(set(value) == expected_modes for value in grouped.values()), "missing calibration mode")
    split_counts = {split: sum(meta[0] == split for meta in cases.values()) for split in ("development", "ood")}
    _need(split_counts == {"development": 12, "ood": 18}, "calibration split/case grid drift")
    return cases, grouped


def _calibration_decisions(
    rows: Sequence[Mapping[str, str]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gates = config["calibration_gates"]
    target = float(config["flowoff_calibration"]["discrepancy_quantile"])
    result: dict[str, Any] = {}
    for mode in ("paired_static", "unpaired_distribution"):
        selected = [row for row in rows if row["mode"] == mode]
        _need(bool(selected), f"missing calibration rows for {mode}")
        coverage = _mean(_finite_float(row["empirical_audit_coverage"], mode) for row in selected)
        p90_error = _linear_quantile(
            (abs(_finite_float(row["empirical_audit_coverage"], mode) - target) for row in selected),
            0.9,
        )
        max_condition = max(_finite_float(row["estimated_condition_number"], mode) for row in selected)
        checks = {
            "audit_coverage_mean_minimum": coverage >= float(gates["audit_coverage_mean_minimum"]),
            "audit_coverage_p90_error_maximum": p90_error <= float(gates["audit_coverage_p90_error_maximum"]),
            "condition_number_maximum": max_condition <= float(gates["condition_number_maximum"]),
            "covariance_spd": all(_finite_float(row["estimated_minimum_eigenvalue"], mode) > 0.0 for row in selected),
        }
        result[mode] = {
            "audit_coverage_mean": coverage,
            "audit_coverage_p90_absolute_error": p90_error,
            "condition_number_maximum": max_condition,
            "checks": checks,
            "passed": all(checks.values()),
        }
    return result


def _reference_lookup(
    rows: Sequence[Mapping[str, str]],
    cases: Mapping[str, tuple[str, str, int, str]],
    config: Mapping[str, Any],
) -> tuple[dict[tuple[str, str, int], Mapping[str, str]], tuple[int, ...]]:
    methods = tuple(str(value) for value in config["methods"])
    expected_forward = int(config["matched_budget"]["learned_feature_preparation_forward_calls"])
    expected_adjoint = int(config["matched_budget"]["learned_feature_preparation_adjoint_calls"])
    lookup: dict[tuple[str, str, int], Mapping[str, str]] = {}
    seeds: set[int] = set()
    for index, row in enumerate(rows):
        label = f"reference_rows[{index}]"
        case_id = row["case_id"]
        method = row["method"]
        seed = _integer(row["model_seed"], f"{label}.model_seed")
        _need(case_id in cases and method in methods, f"{label}: identity drift")
        split, family, base_seed, _ = cases[case_id]
        _need((row["split"], row["family"], _integer(row["base_seed"], label)) == (split, family, base_seed), f"{label}: case metadata drift")
        _need(row["reference_kind"] == "raw_learned", f"{label}: reference kind drift")
        key = (case_id, method, seed)
        _need(key not in lookup, f"{label}: duplicate reference row")
        lookup[key] = row
        seeds.add(seed)
        for field in (
            "field_relative_l2",
            "field_rmse",
            "field_nrmse_dynamic_range",
            "h1_seminorm_relative_error",
            "measured_reprojection_relative_l2",
            "clean_reprojection_relative_l2",
            "gate",
            "correction_rms",
            "neural_inference_seconds",
        ):
            _need(_finite_float(row[field], f"{label}.{field}") >= 0.0, f"{label}: negative {field}")
        _finite_float(row["field_mean_bias"], f"{label}.field_mean_bias")
        _need(_integer(row["optimization_forward_calls"], label) == expected_forward, f"{label}: forward ledger drift")
        _need(_integer(row["optimization_adjoint_calls"], label) == expected_adjoint, f"{label}: adjoint ledger drift")
        _need(_integer(row["grouped_adjoint_calls"], label) == 1, f"{label}: grouped adjoint drift")
        _need(_integer(row["evaluation_forward_calls"], label) == 1, f"{label}: evaluation ledger drift")
    ordered_seeds = tuple(sorted(seeds))
    _need(len(ordered_seeds) == 3, "model seed grid drift")
    expected = {(case, method, seed) for case in cases for method in methods for seed in ordered_seeds}
    _need(set(lookup) == expected, "reference case/method/seed grid drift")
    return lookup, ordered_seeds


def _metric_rows(
    rows: Sequence[Mapping[str, str]],
    cases: Mapping[str, tuple[str, str, int, str]],
    calibration: Mapping[str, Mapping[str, Mapping[str, str]]],
    references: Mapping[tuple[str, str, int], Mapping[str, str]],
    seeds: Sequence[int],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    candidates = {str(value["id"]): value for value in config["candidates"]}
    methods = tuple(str(value) for value in config["methods"])
    harm_threshold = float(config["decision_gates"]["field_harm_threshold_fraction"])
    expected_keys = {
        (candidate, case, method, seed)
        for candidate in candidates
        for case in cases
        for method in methods
        for seed in seeds
    }
    observed: set[tuple[str, str, str, int]] = set()
    parsed: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        label = f"metric_rows[{index}]"
        candidate_id = row["candidate_id"]
        case_id = row["case_id"]
        method = row["method"]
        seed = _integer(row["model_seed"], f"{label}.model_seed")
        key = (candidate_id, case_id, method, seed)
        _need(key in expected_keys and key not in observed, f"{label}: identity grid drift")
        observed.add(key)
        candidate = candidates[candidate_id]
        split, family, base_seed, _ = cases[case_id]
        _need((row["split"], row["family"], _integer(row["base_seed"], label)) == (split, family, base_seed), f"{label}: case metadata drift")
        for field in (
            "calibration_mode",
            "mean_policy",
            "proximal_covariance_policy",
            "selector_covariance_policy",
            "threshold_policy",
        ):
            _need(row[field] == str(candidate[field]), f"{label}: candidate {field} drift")
        uses_truth = _boolean(row["uses_truth"], f"{label}.uses_truth")
        uses_exact = _boolean(row["uses_exact_nuisance"], f"{label}.uses_exact_nuisance")
        _need(uses_truth is bool(candidate["uses_truth"]), f"{label}: truth flag drift")
        _need(uses_exact is bool(candidate["uses_exact_nuisance"]), f"{label}: exact nuisance flag drift")
        _need(_boolean(row["dense_ceiling_only"], label), f"{label}: dense ceiling label missing")
        _need(not _boolean(row["dense_setup_in_budget"], label), f"{label}: dense setup entered budget")
        _need(_integer(row["dense_setup_forward_equivalents"], label) == int(config["matched_budget"]["dense_setup_forward_equivalents"]), f"{label}: dense setup ledger drift")
        _need(_integer(row["learned_feature_forward_calls"], label) == int(config["matched_budget"]["learned_feature_preparation_forward_calls"]), f"{label}: learned forward ledger drift")
        _need(_integer(row["learned_feature_adjoint_calls"], label) == int(config["matched_budget"]["learned_feature_preparation_adjoint_calls"]), f"{label}: learned adjoint ledger drift")

        numeric: dict[str, float] = {}
        for field in (
            "field_relative_l2",
            "h1_seminorm_relative_error",
            "measured_reprojection_relative_l2",
            "clean_reprojection_relative_l2",
            "field_gain_to_best_matched",
            "h1_gain_to_best_matched",
            "clean_reprojection_ratio_to_base",
            "measured_reprojection_ratio_to_cgls",
            "sensor_discrepancy_threshold",
            "selected_discrepancy_threshold",
            "raw_discrepancy",
            "selected_discrepancy",
            "correction_norm",
            "measurement_residual_norm",
            "proximal_covariance_scale",
            "residual_closure_relative_error",
        ):
            numeric[field] = _finite_float(row[field], f"{label}.{field}")
        nonnegative = set(numeric) - {"field_gain_to_best_matched", "h1_gain_to_best_matched"}
        _need(all(numeric[field] >= 0.0 for field in nonnegative), f"{label}: negative norm/ratio/discrepancy")
        harm = _boolean(row["field_harm"], f"{label}.field_harm")
        _need(harm == (numeric["field_gain_to_best_matched"] < -harm_threshold), f"{label}: field_harm drift")
        _need(
            (case_id, method, seed) in references,
            f"{label}: matching raw learned reference missing",
        )

        calibration_row = calibration[case_id][str(candidate["calibration_mode"])]
        calibration_threshold_field = (
            "exact_threshold"
            if candidate["selector_covariance_policy"] == "exact_generator_oracle"
            else "empirical_threshold"
        )
        _close(
            numeric["sensor_discrepancy_threshold"],
            float(calibration_row[calibration_threshold_field]),
            f"{label}.sensor_threshold",
        )
        oracle_threshold = _optional_float(row["truth_residual_oracle_threshold"], f"{label}.truth_threshold")
        if candidate["threshold_policy"] == "empirical_sensor":
            _need(oracle_threshold is None, f"{label}: truth threshold leaked into sensor candidate")
            _close(numeric["selected_discrepancy_threshold"], numeric["sensor_discrepancy_threshold"], f"{label}.selected_threshold")
        else:
            _need(oracle_threshold is not None and uses_truth, f"{label}: unlabeled truth threshold")
            _close(
                numeric["selected_discrepancy_threshold"],
                max(numeric["sensor_discrepancy_threshold"], oracle_threshold),
                f"{label}.selected_threshold",
            )

        crossed = _boolean(row["target_crossed"], f"{label}.target_crossed")
        _need(
            crossed == (numeric["selected_discrepancy"] <= numeric["selected_discrepancy_threshold"] * (1.0 + 1e-10) + 1e-10),
            f"{label}: target crossing flag drift",
        )
        raw_no_correction = _boolean(row["raw_no_correction"], f"{label}.raw_no_correction")
        bisections = _integer(row["bisection_iterations"], f"{label}.bisection_iterations")
        if raw_no_correction:
            _need(row["alpha"] == "inf", f"{label}: no-correction alpha must be inf")
            _need(bisections == 0, f"{label}: no-correction bisection ledger drift")
            _close(numeric["correction_norm"], 0.0, f"{label}.correction_norm")
            _close(numeric["selected_discrepancy"], numeric["raw_discrepancy"], f"{label}.raw_selected")
            alpha = math.inf
        else:
            alpha = _finite_float(row["alpha"], f"{label}.alpha")
            _need(alpha > 0.0, f"{label}: corrected alpha must be positive")
            _need(bisections == int(config["dense_ceiling"]["bisection_iterations"]), f"{label}: bisection ledger drift")
        parsed.append(
            {
                **numeric,
                "candidate_id": candidate_id,
                "method": method,
                "model_seed": seed,
                "split": split,
                "field_harm": harm,
                "target_crossed": crossed,
                "raw_no_correction": raw_no_correction,
                "alpha": alpha,
            }
        )
    _need(observed == expected_keys, "metric candidate/case/method/seed grid incomplete")
    return parsed


def _recompute_aggregates(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["candidate_id"]),
            str(row["method"]),
            int(row["model_seed"]),
            str(row["split"]),
        )
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = groups[key]
        finite_logs = [math.log10(float(row["alpha"])) for row in group if math.isfinite(float(row["alpha"])) and float(row["alpha"]) > 0.0]
        output.append(
            {
                "candidate_id": key[0],
                "method": key[1],
                "model_seed": key[2],
                "split": key[3],
                "case_count": len(group),
                "field_gain_mean": _mean(row["field_gain_to_best_matched"] for row in group),
                "h1_gain_mean": _mean(row["h1_gain_to_best_matched"] for row in group),
                "clean_reprojection_ratio_to_base_mean": _mean(row["clean_reprojection_ratio_to_base"] for row in group),
                "clean_reprojection_ratio_to_base_maximum": max(float(row["clean_reprojection_ratio_to_base"]) for row in group),
                "measured_reprojection_ratio_to_cgls_mean": _mean(row["measured_reprojection_ratio_to_cgls"] for row in group),
                "field_harm_rate": _mean(row["field_harm"] for row in group),
                "worst_field_gain": min(float(row["field_gain_to_best_matched"]) for row in group),
                "target_crossing_rate": _mean(row["target_crossed"] for row in group),
                "raw_no_correction_rate": _mean(row["raw_no_correction"] for row in group),
                "log10_alpha_mean_finite": _mean(finite_logs) if finite_logs else "",
                "residual_closure_relative_error_maximum": max(float(row["residual_closure_relative_error"]) for row in group),
                "correction_norm_mean": _mean(row["correction_norm"] for row in group),
            }
        )
    return output


def _validate_aggregate_csv(
    published: Sequence[Mapping[str, str]],
    expected: Sequence[Mapping[str, Any]],
) -> None:
    published_map: dict[tuple[str, str, int, str], Mapping[str, str]] = {}
    for index, row in enumerate(published):
        key = (row["candidate_id"], row["method"], _integer(row["model_seed"], f"aggregate_rows[{index}].model_seed"), row["split"])
        _need(key not in published_map, f"aggregate_rows[{index}]: duplicate group")
        published_map[key] = row
    expected_map = {(str(row["candidate_id"]), str(row["method"]), int(row["model_seed"]), str(row["split"])): row for row in expected}
    _need(set(published_map) == set(expected_map), "aggregate group grid drift")
    for key, expected_row in expected_map.items():
        observed = published_map[key]
        label = f"aggregate_rows{key}"
        for field, value in expected_row.items():
            if isinstance(value, str):
                _need(observed[field] == value, f"{label}.{field}: value drift")
            elif isinstance(value, int):
                _need(_integer(observed[field], f"{label}.{field}") == value, f"{label}.{field}: integer drift")
            else:
                _close(observed[field], float(value), f"{label}.{field}")


def _pooled_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_id: str,
    method: str,
    split: str,
) -> dict[str, Any]:
    selected = [row for row in rows if row["candidate_id"] == candidate_id and row["method"] == method and row["split"] == split]
    _need(bool(selected), f"missing pooled rows for {candidate_id}/{method}/{split}")
    seed_means = []
    for seed in sorted({int(row["model_seed"]) for row in selected}):
        seed_rows = [row for row in selected if int(row["model_seed"]) == seed]
        seed_means.append(_mean(row["field_gain_to_best_matched"] for row in seed_rows))
    return {
        "row_count": len(selected),
        "field_gain_mean": _mean(row["field_gain_to_best_matched"] for row in selected),
        "h1_gain_mean": _mean(row["h1_gain_to_best_matched"] for row in selected),
        "clean_reprojection_ratio_to_base_mean": _mean(row["clean_reprojection_ratio_to_base"] for row in selected),
        "clean_reprojection_ratio_to_base_maximum": max(float(row["clean_reprojection_ratio_to_base"]) for row in selected),
        "field_harm_rate": _mean(row["field_harm"] for row in selected),
        "worst_field_gain": min(float(row["field_gain_to_best_matched"]) for row in selected),
        "target_crossing_rate": _mean(row["target_crossed"] for row in selected),
        "residual_closure_relative_error_maximum": max(float(row["residual_closure_relative_error"]) for row in selected),
        "all_model_seed_field_gain_means_positive": all(value > 0.0 for value in seed_means),
        "per_model_seed_field_gain_means": seed_means,
    }


def _decisions(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    calibration_decisions: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    gates = config["decision_gates"]
    output: list[dict[str, Any]] = []
    for candidate in config["candidates"]:
        candidate_id = str(candidate["id"])
        for method in config["methods"]:
            development = _pooled_metrics(rows, candidate_id=candidate_id, method=str(method), split="development")
            ood = _pooled_metrics(rows, candidate_id=candidate_id, method=str(method), split="ood")
            mode = str(candidate["calibration_mode"])
            checks = {
                "calibration_valid": bool(calibration_decisions[mode]["passed"]),
                "development_field_gain": development["field_gain_mean"] >= float(gates["development_field_gain_minimum"]),
                "development_h1_gain": development["h1_gain_mean"] >= float(gates["development_h1_gain_minimum"]),
                "development_clean_mean": development["clean_reprojection_ratio_to_base_mean"] <= float(gates["development_clean_reprojection_ratio_to_base_mean_maximum"]),
                "development_clean_worst": development["clean_reprojection_ratio_to_base_maximum"] <= float(gates["development_clean_reprojection_ratio_to_base_worst_maximum"]),
                "development_harm": development["field_harm_rate"] <= float(gates["field_harm_rate_maximum"]),
                "development_worst": development["worst_field_gain"] >= float(gates["worst_field_gain_minimum"]),
                "ood_field_gain": ood["field_gain_mean"] >= float(gates["ood_field_gain_minimum"]),
                "ood_h1_gain": ood["h1_gain_mean"] >= float(gates["ood_h1_gain_minimum"]),
                "ood_clean_mean": ood["clean_reprojection_ratio_to_base_mean"] <= float(gates["ood_clean_reprojection_ratio_to_base_mean_maximum"]),
                "ood_clean_worst": ood["clean_reprojection_ratio_to_base_maximum"] <= float(gates["ood_clean_reprojection_ratio_to_base_worst_maximum"]),
                "ood_harm": ood["field_harm_rate"] <= float(gates["field_harm_rate_maximum"]),
                "ood_worst": ood["worst_field_gain"] >= float(gates["worst_field_gain_minimum"]),
                "all_seed_means_positive": development["all_model_seed_field_gain_means_positive"] and ood["all_model_seed_field_gain_means_positive"],
                "target_crossing": min(development["target_crossing_rate"], ood["target_crossing_rate"]) >= float(gates["minimum_target_crossing_rate"]),
                "residual_closure": max(development["residual_closure_relative_error_maximum"], ood["residual_closure_relative_error_maximum"]) <= float(gates["maximum_residual_closure_relative_error"]),
            }
            output.append(
                {
                    "candidate_id": candidate_id,
                    "method": str(method),
                    "uses_truth": bool(candidate["uses_truth"]),
                    "uses_exact_nuisance": bool(candidate["uses_exact_nuisance"]),
                    "dense_ceiling_only": True,
                    "calibration_mode": mode,
                    "development": development,
                    "ood": ood,
                    "checks": checks,
                    "passed": all(checks.values()),
                }
            )
    return output


def _validate_dense_setup(
    rows: Sequence[Mapping[str, str]],
    cases: Mapping[str, tuple[str, str, int, str]],
    metric_rows: Sequence[Mapping[str, str]],
    config: Mapping[str, Any],
) -> None:
    geometry = {metadata[3] for metadata in cases.values()}
    _need(len(rows) == len(geometry), "dense setup row count/geometry drift")
    observed: set[str] = set()
    metric_dimensions = {(_integer(row["dense_matrix_rows"], "metric matrix rows"), _integer(row["dense_matrix_columns"], "metric matrix columns")) for row in metric_rows}
    _need(len(metric_dimensions) == 1, "metric dense dimensions drift")
    expected_dimensions = next(iter(metric_dimensions))
    for index, row in enumerate(rows):
        label = f"dense_setup_rows[{index}]"
        digest = row["geometry_digest"]
        _need(digest in geometry and digest not in observed, f"{label}: geometry grid drift")
        observed.add(digest)
        dimensions = (_integer(row["matrix_rows"], label), _integer(row["matrix_columns"], label))
        _need(dimensions == expected_dimensions, f"{label}: matrix dimensions drift")
        rank = _integer(row["rank"], f"{label}.rank")
        _need(0 < rank <= min(dimensions), f"{label}: impossible rank")
        _need(_finite_float(row["rank_tolerance"], label) > 0.0, f"{label}: rank tolerance drift")
        _need(_integer(row["setup_forward_calls_batched"], label) > 0, f"{label}: batched setup ledger drift")
        _need(_integer(row["setup_forward_equivalents_unbatched"], label) == int(config["matched_budget"]["dense_setup_forward_equivalents"]), f"{label}: unbatched setup ledger drift")
        _need(_integer(row["assembly_batch_size"], label) == int(config["dense_ceiling"]["assembly_batch_size"]), f"{label}: assembly batch drift")
        _need(_finite_float(row["zero_forward_maximum_absolute"], label) >= 0.0, f"{label}: negative zero check")
        _need(_finite_float(row["factorization_seconds"], label) >= 0.0, f"{label}: negative runtime")
        _need(not _boolean(row["dense_setup_in_budget"], label), f"{label}: dense setup entered budget")
        _need(row["status"] == "DENSE_TOY_ORACLE_SETUP_NOT_RECONSTRUCTION_BUDGET", f"{label}: dense status drift")
    _need(observed == geometry, "dense setup geometry grid incomplete")


def _status_and_authorization(decisions: Sequence[Mapping[str, Any]]) -> tuple[str, int, int, dict[str, bool]]:
    deployable = [value for value in decisions if value["passed"] and not value["uses_truth"] and not value["uses_exact_nuisance"]]
    passed = [value for value in decisions if value["passed"]]
    if deployable:
        status = "N1_1_FLOWOFF_COVARIANCE_PROXIMAL_MECHANISM_SIGNAL_ONLY"
    elif passed:
        status = "N1_1_ORACLE_ONLY_COVARIANCE_PROXIMAL_NO_GO"
    else:
        status = "N1_1_FLOWOFF_COVARIANCE_PROXIMAL_NO_GO"
    authorization = {
        "claim_deployable_algorithm": False,
        "claim_method_superiority": False,
        "claim_real_bost_generalization": False,
        "open_fresh_or_final": False,
        "continue_matrix_free_covariance_proximal_research": bool(deployable),
        "continue_model_mismatch_floor_research": True,
        "request_same_session_flowoff_from_lab": True,
    }
    return status, len(deployable), len(passed), authorization


def validate_packet(
    *,
    config_path: Path = CONFIG,
    output_dir: Path = OUTPUT,
) -> dict[str, Any]:
    """Validate one frozen packet and return a compact independent report."""

    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    _validate_manifest(output_dir)
    config = _read_json(config_path)
    summary = _read_json(output_dir / "summary.json")
    _validate_config(config)
    source_hashes = _validate_sources(config)

    calibration_rows = _read_csv_exact(output_dir / "calibration_rows.csv", CALIBRATION_FIELDS)
    metric_rows_raw = _read_csv_exact(output_dir / "metric_rows.csv", METRIC_FIELDS)
    aggregate_rows = _read_csv_exact(output_dir / "aggregate_rows.csv", AGGREGATE_FIELDS)
    reference_rows = _read_csv_exact(output_dir / "reference_rows.csv", REFERENCE_FIELDS)
    dense_rows = _read_csv_exact(output_dir / "dense_setup_rows.csv", DENSE_SETUP_FIELDS)
    _validate_no_obvious_leakage(
        output_dir,
        summary,
        (*CALIBRATION_FIELDS, *METRIC_FIELDS, *AGGREGATE_FIELDS, *REFERENCE_FIELDS, *DENSE_SETUP_FIELDS),
    )

    cases, calibration_grouped = _case_metadata(calibration_rows, config)
    calibration_decisions = _calibration_decisions(calibration_rows, config)
    references, seeds = _reference_lookup(reference_rows, cases, config)
    metric_rows = _metric_rows(
        metric_rows_raw,
        cases,
        calibration_grouped,
        references,
        seeds,
        config,
    )
    recomputed_aggregates = _recompute_aggregates(metric_rows)
    _validate_aggregate_csv(aggregate_rows, recomputed_aggregates)
    decisions = _decisions(metric_rows, config, calibration_decisions)
    _validate_dense_setup(dense_rows, cases, metric_rows_raw, config)
    status, deployable_count, pass_count, authorization = _status_and_authorization(decisions)

    _need(set(summary) == SUMMARY_FIELDS, "N1.1 summary fields drift")
    _need(summary["schema_version"] == config["report_schema_version"], "N1.1 summary schema drift")
    _need(summary["evidence_level"] == config["evidence_level"], "N1.1 evidence level drift")
    try:
        expected_config_path = str(config_path.relative_to(ROOT))
    except ValueError as error:
        raise ValidationError("config path must remain inside repository") from error
    _need(summary["source_config"] == expected_config_path, "N1.1 source config path drift")
    _need(summary["source_config_sha256"] == _sha256(config_path), "N1.1 source config hash drift")
    _deep_match(summary["source_hashes"], source_hashes, "N1.1 source hashes")
    expected_counts = {
        "candidate_count": len(config["candidates"]),
        "calibration_row_count": len(calibration_rows),
        "metric_row_count": len(metric_rows_raw),
        "aggregate_row_count": len(aggregate_rows),
        "reference_row_count": len(reference_rows),
        "dense_setup_row_count": len(dense_rows),
        "deployable_input_pass_count": deployable_count,
        "oracle_pass_count": pass_count,
    }
    for field, expected in expected_counts.items():
        _need(_integer(summary[field], f"summary.{field}") == expected, f"summary.{field}: row/count drift")
    _deep_match(summary["calibration_decisions"], calibration_decisions, "N1.1 calibration decisions")
    _deep_match(summary["decisions"], decisions, "N1.1 decisions")
    _need(summary["status"] == status, "N1.1 summary status drift")
    _deep_match(summary["authorization"], authorization, "N1.1 authorization")
    _deep_match(summary["claim_boundary"], config["claim_boundary"], "N1.1 claim boundary")
    _need(_finite_float(summary["runtime_seconds"], "summary.runtime_seconds") >= 0.0, "negative runtime")
    _need(isinstance(summary["device"], str) and bool(summary["device"]), "summary device missing")

    validated = f"VALIDATED_{status}"
    return {
        "status": validated,
        "packet_status": status,
        "candidate_count": len(config["candidates"]),
        "calibration_row_count": len(calibration_rows),
        "metric_row_count": len(metric_rows_raw),
        "aggregate_row_count": len(aggregate_rows),
        "reference_row_count": len(reference_rows),
        "dense_setup_row_count": len(dense_rows),
        "decision_count": len(decisions),
        "deployable_input_pass_count": deployable_count,
        "oracle_pass_count": pass_count,
        "authorization": authorization,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        report = validate_packet(config_path=args.config, output_dir=args.output_dir)
    except ValidationError as error:
        print(json.dumps({"status": "INVALID", "error": str(error)}, sort_keys=True))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
