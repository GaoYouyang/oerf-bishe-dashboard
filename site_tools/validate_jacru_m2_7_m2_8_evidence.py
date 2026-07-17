#!/usr/bin/env python3
"""Independently audit the frozen JACRU M2.7 and M2.8 evidence packets.

This auditor deliberately uses only the standard library and frozen public
artifacts.  It does not import a runner, a model, or an optical operator.
M2.8's public alpha=0/0.5/1 rows are sufficient to reconstruct the convex
quadratics used by the evaluator-only truth-oracle ceiling.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "demo_t16_operator/results"
CONFIGS = ROOT / "demo_t16_operator/configs"
METHODS = ("jacru_m2", "pooled_cnn")
SEEDS = (17, 29, 43)
SPLITS = ("development", "ood")
SPLIT_COUNTS = {"development": 12, "ood": 18}
BASELINES = ("cgls_matched", "huber_pdhg_matched", "base_landweber_matched")
M27_K = tuple(range(11))
M28_K = (9, 10)
M28_ALPHA = (0.0, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 0.995, 1.0)

M27_AUTHORIZATION = {
    "claim_deployable_algorithm": False,
    "claim_method_superiority": False,
    "claim_real_bost_generalization": False,
    "open_fresh_or_final": False,
    "draft_new_preregistered_fresh_gate": False,
    "continue_matrix_free_preconditioner_research": True,
    "continue_deployable_preconditioner_estimation": False,
}
M28_AUTHORIZATION = {
    "claim_deployable_algorithm": False,
    "claim_method_superiority": False,
    "claim_real_bost_generalization": False,
    "open_fresh_or_final": False,
    "continue_fixed_interpolation_research": False,
    "continue_observable_calibration_research": False,
    "continue_noise_aware_target_or_fail_closed_research": True,
}
M27_PAYLOADS = {
    "README.md", "aggregate_rows.csv", "diagnostic.pdf", "diagnostic.png",
    "matched_baseline_aggregate_rows.csv", "matched_baseline_rows.csv",
    "metric_rows.csv", "reference_rows.csv", "summary.json",
}
M28_PAYLOADS = {
    "README.md", "diagnostic.pdf", "diagnostic.png",
    "fixed_interpolation_aggregate.csv", "fixed_interpolation_rows.csv",
    "matched_baseline_rows.csv", "summary.json", "truth_oracle_ceiling_rows.csv",
}

COMMON = (
    "case_id", "split", "family", "base_seed", "method", "model_seed",
    "field_relative_l2", "field_rmse", "field_nrmse_dynamic_range",
    "field_mean_bias", "h1_seminorm_relative_error",
    "measured_reprojection_relative_l2", "clean_reprojection_relative_l2",
    "gate", "correction_rms", "optimization_forward_calls",
    "optimization_adjoint_calls", "grouped_adjoint_calls",
    "evaluation_forward_calls", "neural_inference_seconds",
)
REFERENCE = COMMON + ("reference_kind",)
BASELINE = COMMON + (
    "matched_step", "total_calls", "baseline_kind", "dc_step_size",
    "operator_norm_squared_bound", "projection_iterations", "paired_call_budget",
    "matched_step_internal_offset",
)
M27_METRIC = COMMON + (
    "projection_variant", "projection_iterations", "damping_fraction",
    "damping_absolute", "preconditioner", "projection_forward_calls",
    "projection_adjoint_calls", "paired_call_budget",
    "matched_cgls_field_relative_l2", "matched_huber_field_relative_l2",
    "matched_base_landweber_field_relative_l2",
    "field_gain_to_best_matched_classical",
    "h1_gain_to_best_matched_classical",
    "reprojection_ratio_to_matched_cgls", "visible_correction_fraction",
    "system_residual_fraction", "projection_closure_relative_error",
    "exact_projection_approximation_error",
    "exact_oracle_error_reduction_retention", "oracle_reduction_defined",
    "base_anchor_field_relative_l2", "exact_oracle_field_relative_l2",
    "raw_learned_field_relative_l2", "exact_oracle_rank",
    "exact_oracle_nullity_lower_bound", "field_harm_to_best_matched_classical",
    "converged", "breakdown", "projection_diagnostic_forward_calls",
    "dense_oracle_used_by_algorithm", "projection_target_mode",
    "exact_oracle_internal_projection_residual", "preconditioner_kind",
    "preconditioner_is_oracle", "preconditioner_setup_forward_equivalents",
    "preconditioner_setup_adjoint_equivalents", "preconditioner_applications",
    "preconditioner_block_count", "preconditioner_largest_block_size",
    "preconditioner_minimum_block_eigenvalue",
    "preconditioner_maximum_block_condition_number",
)
M27_AGGREGATE = (
    "method", "model_seed", "split", "projection_variant",
    "projection_iterations", "case_count", "paired_call_budget",
    "damping_fraction", "field_relative_l2_mean",
    "h1_seminorm_relative_error_mean",
    "field_gain_to_best_matched_classical_mean",
    "h1_gain_to_best_matched_classical_mean",
    "reprojection_ratio_to_matched_cgls_mean",
    "visible_correction_fraction_mean", "visible_correction_fraction_maximum",
    "system_residual_fraction_mean", "projection_closure_relative_error_mean",
    "projection_closure_relative_error_maximum",
    "exact_projection_approximation_error_mean",
    "exact_oracle_error_reduction_retention_mean",
    "oracle_reduction_defined_rate", "field_harm_rate",
    "worst_field_gain_to_best_matched_classical", "breakdown_rate",
)
BASELINE_AGGREGATE = (
    "method", "split", "matched_step", "total_calls", "case_count",
    "field_relative_l2_mean", "h1_seminorm_relative_error_mean",
    "measured_reprojection_relative_l2_mean",
)
M28_FIXED = COMMON + (
    "projection_iterations", "interpolation_fraction", "paired_call_budget",
    "field_gain", "h1_gain", "reprojection_ratio_to_matched_cgls",
    "field_harm", "truth_used_by_candidate",
    "exact_camera_block_setup_forward_equivalents", "preconditioner_block_count",
)
M28_FIXED_AGGREGATE = (
    "method", "model_seed", "split", "projection_iterations",
    "interpolation_fraction", "case_count", "paired_call_budget",
    "field_gain_mean", "h1_gain_mean", "reprojection_ratio_mean",
    "reprojection_ratio_maximum", "field_harm_rate", "worst_field_gain",
)
M28_TRUTH = (
    "case_id", "split", "family", "base_seed", "method", "model_seed",
    "projection_iterations", "paired_call_budget", "truth_used_by_candidate",
    "candidate_deployable", "per_case_reprojection_ratio_limit",
    "exact_camera_block_setup_forward_equivalents", "reprojection_feasible",
    "feasible_alpha_lower", "feasible_alpha_upper", "truth_oracle_alpha",
    "field_relative_l2", "h1_seminorm_relative_error",
    "measured_reprojection_relative_l2", "clean_reprojection_relative_l2",
    "field_gain", "h1_gain", "reprojection_ratio_to_matched_cgls", "field_harm",
)


class ValidationError(RuntimeError):
    """Raised when a frozen public evidence packet violates its contract."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ValidationError(f"cannot hash {path}: {error}") from error


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot read JSON {path}: {error}") from error
    _need(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _float(value: Any, path: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValidationError(f"{path}: expected finite number") from error
    _need(math.isfinite(result), f"{path}: expected finite number")
    return result


def _int(value: Any, path: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValidationError(f"{path}: expected integer") from error
    _need(str(result) == str(value), f"{path}: non-canonical integer")
    return result


def _close(actual: Any, expected: float, path: str, *, tolerance: float = 5e-10) -> None:
    observed = _float(actual, path)
    _need(math.isclose(observed, expected, rel_tol=tolerance, abs_tol=tolerance),
          f"{path}: numeric mismatch ({observed!r} != {expected!r})")


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    _need(bool(values), "cannot average empty values")
    return math.fsum(values) / len(values)


def _read_csv(path: Path, fields: tuple[str, ...], count: int) -> list[dict[str, str]]:
    try:
        physical = path.read_text(encoding="utf-8").splitlines()
        _need(all(physical), f"{path.name}: blank physical line")
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            _need(tuple(reader.fieldnames or ()) == fields,
                  f"{path.name}: columns differ from frozen schema")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValidationError(f"cannot read CSV {path}: {error}") from error
    _need(len(rows) == count, f"expected {count} {path.stem} rows")
    _need(len(physical) == count + 1, f"{path.name}: physical line count drift")
    _need(all(None not in row and all(value is not None for value in row.values()) for row in rows),
          f"{path.name}: malformed row")
    return rows


def _manifest(output: Path, payloads: set[str]) -> None:
    try:
        lines = (output / "checksums.sha256").read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValidationError(f"cannot read checksum manifest: {error}") from error
    entries: dict[str, str] = {}
    matcher = re.compile(r"^([0-9a-f]{64})  ([A-Za-z0-9_.-]+)$")
    for line in lines:
        match = matcher.fullmatch(line)
        _need(match is not None, "checksums.sha256: malformed entry")
        assert match is not None
        digest, name = match.groups()
        _need(name not in entries, f"checksums.sha256: duplicate {name}")
        entries[name] = digest
    _need(set(entries) == payloads, "checksums.sha256: payload set mismatch")
    _need({path.name for path in output.iterdir()} == payloads | {"checksums.sha256"},
          "public packet contains unmanifested or missing files")
    for name, digest in entries.items():
        path = output / name
        _need(path.is_file() and not path.is_symlink(), f"invalid payload: {name}")
        _need(_sha(path) == digest, f"checksum mismatch: {name}")


def _sources(config: dict[str, Any], prefixes: tuple[str, ...]) -> None:
    for prefix in prefixes:
        config_path = ROOT / str(config[f"{prefix}_config"])
        results = ROOT / str(config[f"{prefix}_results"])
        _need(config_path.is_file() and results.is_dir(), f"{prefix}: source path missing")
        _need(_sha(config_path) == config[f"{prefix}_config_sha256"],
              f"{prefix} config hash drift")
        _need(_sha(results / "summary.json") == config[f"{prefix}_summary_sha256"],
              f"{prefix} summary hash drift")


def _baseline_lookup(rows: list[dict[str, str]], ks: tuple[int, ...], cases: set[str]) -> dict[tuple[str, str, int], dict[str, str]]:
    lookup: dict[tuple[str, str, int], dict[str, str]] = {}
    for index, row in enumerate(rows):
        path = f"matched_baseline_rows[{index}]"
        k = _int(row["projection_iterations"], f"{path}.projection_iterations")
        _need(row["method"] in BASELINES and row["case_id"] in cases and k in ks,
              f"{path}: baseline identity grid drift")
        key = (row["method"], row["case_id"], k)
        _need(key not in lookup, f"duplicate baseline row: {key}")
        lookup[key] = row
        budget = 14 + k
        for field, expected in (("optimization_forward_calls", budget),
                                ("optimization_adjoint_calls", budget),
                                ("total_calls", budget), ("paired_call_budget", budget),
                                ("matched_step", 1 + k),
                                ("matched_step_internal_offset", 1 + k),
                                ("evaluation_forward_calls", 1)):
            _need(_int(row[field], f"{path}.{field}") == expected,
                  f"{path}: baseline {field} budget drift")
        _need(_int(row["model_seed"], f"{path}.model_seed") == -1,
              f"{path}: baseline model seed drift")
        _need(row["baseline_kind"] == row["method"], f"{path}: baseline kind drift")
    expected = {(method, case, k) for method in BASELINES for case in cases for k in ks}
    _need(set(lookup) == expected, "matched baseline identity grid drift")
    return lookup


def _reference_cases(rows: list[dict[str, str]]) -> set[str]:
    cases = {row["case_id"] for row in rows}
    _need(len(cases) == 30, "reference case catalog drift")
    counts = {split: sum(row["split"] == split and row["reference_kind"] == "base_anchor" for row in rows)
              for split in SPLITS}
    _need(counts == SPLIT_COUNTS, "base anchor split catalog drift")
    for index, row in enumerate(rows):
        path = f"reference_rows[{index}]"
        _need(row["split"] in SPLITS, f"{path}: split drift")
        if row["reference_kind"] == "base_anchor":
            _need((row["method"], _int(row["model_seed"], path)) == ("prepared_cgls_base_12", -1),
                  f"{path}: base anchor identity drift")
            _need((_int(row["optimization_forward_calls"], path), _int(row["optimization_adjoint_calls"], path)) == (12, 12),
                  f"{path}: base anchor F/A budget drift")
    return cases


def _m27_config(config: dict[str, Any]) -> None:
    _need(config["schema_version"] == "jacru-m2-7-target-no-harm-pareto-ceiling-postopen-config-1.0",
          "M2.7 config schema drift")
    _need(config["report_schema_version"] == "jacru-m2-7-target-no-harm-pareto-ceiling-postopen-report-1.0",
          "M2.7 report schema drift")
    _need(config["status"].startswith("FROZEN_BEFORE_FIRST_"), "M2.7 config is not frozen")
    _need(config["methods"] == list(METHODS), "M2.7 method grid drift")
    _need(config["preconditioner_oracle_only"] is True, "M2.7 oracle boundary drift")
    _need(config["projection"]["snapshot_iterations"] == list(M27_K), "M2.7 K grid drift")
    _need(max(config["projection"]["snapshot_iterations"]) <= 10, "M2.7 K cap drift")
    _need(config["matched_budget"]["learned_feature_preparation_forward_calls"] == 13,
          "M2.7 feature F budget drift")
    _need(config["matched_budget"]["learned_feature_preparation_adjoint_calls"] == 13,
          "M2.7 feature A budget drift")
    _need(config["matched_budget"]["maximum_forward_calls"] == 24,
          "M2.7 maximum F budget drift")
    _need(config["matched_budget"]["maximum_adjoint_calls"] == 24,
          "M2.7 maximum A budget drift")
    _need(config["claim_boundary"]["exact_camera_block_is_matrix_free_or_deployable"] is False,
          "M2.7 camera-block deployability boundary drift")
    _need(config["claim_boundary"]["opens_fresh_or_final"] is False,
          "M2.7 fresh/final boundary drift")
    _sources(config, ("source_t0", "source_m2_2", "source_m2_3", "source_m2_4", "source_m2_5", "source_m2_6"))


def _m27_aggregate(rows: list[dict[str, str]]) -> dict[tuple[Any, ...], dict[str, float | int]]:
    groups: dict[tuple[Any, ...], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["method"], _int(row["model_seed"], "seed"), row["split"],
               row["projection_variant"], _int(row["projection_iterations"], "K"))
        groups.setdefault(key, []).append(row)
    result: dict[tuple[Any, ...], dict[str, float | int]] = {}
    for key, values in groups.items():
        result[key] = {
            "case_count": len(values), "paired_call_budget": _int(values[0]["paired_call_budget"], "budget"),
            "damping_fraction": _float(values[0]["damping_fraction"], "damping"),
            "field_relative_l2_mean": _mean(_float(x["field_relative_l2"], "field") for x in values),
            "h1_seminorm_relative_error_mean": _mean(_float(x["h1_seminorm_relative_error"], "h1") for x in values),
            "field_gain_to_best_matched_classical_mean": _mean(_float(x["field_gain_to_best_matched_classical"], "gain") for x in values),
            "h1_gain_to_best_matched_classical_mean": _mean(_float(x["h1_gain_to_best_matched_classical"], "h1gain") for x in values),
            "reprojection_ratio_to_matched_cgls_mean": _mean(_float(x["reprojection_ratio_to_matched_cgls"], "ratio") for x in values),
            "visible_correction_fraction_mean": _mean(_float(x["visible_correction_fraction"], "visible") for x in values),
            "visible_correction_fraction_maximum": max(_float(x["visible_correction_fraction"], "visible") for x in values),
            "system_residual_fraction_mean": _mean(_float(x["system_residual_fraction"], "residual") for x in values),
            "projection_closure_relative_error_mean": _mean(_float(x["projection_closure_relative_error"], "closure") for x in values),
            "projection_closure_relative_error_maximum": max(_float(x["projection_closure_relative_error"], "closure") for x in values),
            "exact_projection_approximation_error_mean": _mean(_float(x["exact_projection_approximation_error"], "approx") for x in values),
            "exact_oracle_error_reduction_retention_mean": _mean(_float(x["exact_oracle_error_reduction_retention"], "retention") for x in values),
            "oracle_reduction_defined_rate": _mean(_float(x["oracle_reduction_defined"], "defined") for x in values),
            "field_harm_rate": _mean(_float(x["field_harm_to_best_matched_classical"], "harm") for x in values),
            "worst_field_gain_to_best_matched_classical": min(_float(x["field_gain_to_best_matched_classical"], "gain") for x in values),
            "breakdown_rate": _mean(_float(x["breakdown"], "breakdown") for x in values),
        }
    return result


def _m27_metrics(rows: list[dict[str, str]], cases: set[str], baselines: dict[tuple[str, str, int], dict[str, str]]) -> None:
    seen: set[tuple[str, int, str, int]] = set()
    for index, row in enumerate(rows):
        path = f"metric_rows[{index}]"
        method = row["method"]
        seed = _int(row["model_seed"], f"{path}.model_seed")
        k = _int(row["projection_iterations"], f"{path}.projection_iterations")
        _need(method in METHODS and seed in SEEDS and row["case_id"] in cases and row["split"] in SPLITS and k in M27_K,
              f"{path}: M2.7 metric identity grid drift or K>10")
        key = (method, seed, row["case_id"], k)
        _need(key not in seen, f"duplicate M2.7 metric row: {key}")
        seen.add(key)
        _need(row["projection_variant"] == "affine_pcg_dense_exact_camera_block_oracle", f"{path}: variant drift")
        _need(row["projection_target_mode"] == "affine_observation", f"{path}: target mode drift")
        _need(row["preconditioner"] == "dense_exact_camera_block_jacobi_oracle" and
              row["preconditioner_kind"] == "dense_exact_camera_block_jacobi_oracle",
              f"{path}: camera-block preconditioner drift")
        _need(_int(row["preconditioner_is_oracle"], path) == 1 and
              _int(row["preconditioner_setup_forward_equivalents"], path) == 1001 and
              _int(row["preconditioner_setup_adjoint_equivalents"], path) == 0,
              f"{path}: camera-block oracle setup ledger drift")
        _need((_int(row["preconditioner_block_count"], path), _int(row["preconditioner_largest_block_size"], path)) == (3, 50),
              f"{path}: camera-block partition drift")
        _need(_float(row["preconditioner_minimum_block_eigenvalue"], path) > 0 and
              _float(row["preconditioner_maximum_block_condition_number"], path) >= 1,
              f"{path}: camera-block SPD drift")
        _need(_int(row["preconditioner_applications"], path) == 11,
              f"{path}: fixed K=10 preconditioner application ledger drift")
        expected_f, expected_a = 14 + k, 13 + k
        for field, expected in (("projection_forward_calls", k + 1), ("projection_adjoint_calls", k),
                                ("optimization_forward_calls", expected_f), ("optimization_adjoint_calls", expected_a),
                                ("paired_call_budget", expected_f), ("projection_diagnostic_forward_calls", 1),
                                ("grouped_adjoint_calls", 1), ("evaluation_forward_calls", 1)):
            _need(_int(row[field], f"{path}.{field}") == expected,
                  f"{path}: F/A budget formula drift")
        _need(expected_f <= 24 and expected_a <= 24, f"{path}: M2.7 K cap/budget drift")
        _need(_float(row["projection_closure_relative_error"], path) <= 1e-10,
              f"{path}: projection closure exceeds 1e-10")
        _need(row["dense_oracle_used_by_algorithm"] == "False", f"{path}: dense oracle entered algorithm")
        _need((_int(row["exact_oracle_rank"], path), _int(row["exact_oracle_nullity_lower_bound"], path)) == (150, 850),
              f"{path}: dense audit geometry drift")
        cgls = baselines[("cgls_matched", row["case_id"], k)]
        huber = baselines[("huber_pdhg_matched", row["case_id"], k)]
        landweber = baselines[("base_landweber_matched", row["case_id"], k)]
        for field, source in (("matched_cgls_field_relative_l2", cgls),
                              ("matched_huber_field_relative_l2", huber),
                              ("matched_base_landweber_field_relative_l2", landweber)):
            _close(row[field], _float(source["field_relative_l2"], f"{path}.{field}"), f"{path}.{field}")
        best_field = min(_float(cgls["field_relative_l2"], path), _float(huber["field_relative_l2"], path))
        best_h1 = min(_float(cgls["h1_seminorm_relative_error"], path), _float(huber["h1_seminorm_relative_error"], path))
        gain = (best_field - _float(row["field_relative_l2"], path)) / best_field
        h1_gain = (best_h1 - _float(row["h1_seminorm_relative_error"], path)) / best_h1
        ratio = _float(row["measured_reprojection_relative_l2"], path) / _float(cgls["measured_reprojection_relative_l2"], path)
        _close(row["field_gain_to_best_matched_classical"], gain, f"{path}.field_gain")
        _close(row["h1_gain_to_best_matched_classical"], h1_gain, f"{path}.h1_gain")
        _close(row["reprojection_ratio_to_matched_cgls"], ratio, f"{path}.reprojection_ratio")
        _need(_int(row["field_harm_to_best_matched_classical"], path) == int(gain < -0.01),
              f"{path}: field harm threshold drift")
    expected = {(method, seed, case, k) for method in METHODS for seed in SEEDS for case in cases for k in M27_K}
    _need(seen == expected, "M2.7 metric identity grid drift")


def _m27_decision_rows(rows: list[dict[str, str]], method: str, k: int, split: str) -> dict[str, Any]:
    values = [row for row in rows if row["method"] == method and _int(row["projection_iterations"], "K") == k and row["split"] == split]
    _need(len(values) == len(SEEDS) * SPLIT_COUNTS[split], "M2.7 decision row count drift")
    return {
        "case_model_count": len(values), "paired_call_budget": _int(values[0]["paired_call_budget"], "budget"),
        "field_gain_mean": _mean(_float(x["field_gain_to_best_matched_classical"], "gain") for x in values),
        "h1_gain_mean": _mean(_float(x["h1_gain_to_best_matched_classical"], "h1") for x in values),
        "reprojection_ratio_mean": _mean(_float(x["reprojection_ratio_to_matched_cgls"], "ratio") for x in values),
        "visible_correction_fraction_mean": _mean(_float(x["visible_correction_fraction"], "visible") for x in values),
        "visible_correction_fraction_maximum": max(_float(x["visible_correction_fraction"], "visible") for x in values),
        "projection_closure_relative_error_mean": _mean(_float(x["projection_closure_relative_error"], "closure") for x in values),
        "projection_closure_relative_error_maximum": max(_float(x["projection_closure_relative_error"], "closure") for x in values),
        "exact_oracle_error_reduction_retention_mean": _mean(_float(x["exact_oracle_error_reduction_retention"], "retention") for x in values),
        "oracle_reduction_defined_rate": _mean(_float(x["oracle_reduction_defined"], "defined") for x in values),
        "field_harm_rate": _mean(_float(x["field_harm_to_best_matched_classical"], "harm") for x in values),
        "worst_field_gain": min(_float(x["field_gain_to_best_matched_classical"], "gain") for x in values),
        "breakdown_rate": _mean(_float(x["breakdown"], "breakdown") for x in values),
        "per_model_seed_field_gain_means": [_mean(_float(x["field_gain_to_best_matched_classical"], "gain") for x in values if _int(x["model_seed"], "seed") == seed) for seed in SEEDS],
    }


def _match_mapping(actual: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    _need(set(actual) == set(expected), f"{label}: keys drift")
    for key, value in expected.items():
        observed = actual[key]
        if isinstance(value, dict):
            _need(isinstance(observed, dict), f"{label}.{key}: object drift")
            _match_mapping(observed, value, f"{label}.{key}")
        elif isinstance(value, list):
            _need(isinstance(observed, list) and len(observed) == len(value), f"{label}.{key}: list drift")
            for i, (left, right) in enumerate(zip(observed, value)):
                if isinstance(right, (float, int)) and not isinstance(right, bool):
                    _close(left, float(right), f"{label}.{key}[{i}]")
                else:
                    _need(left == right, f"{label}.{key}[{i}]: value drift")
        elif isinstance(value, (float, int)) and not isinstance(value, bool):
            _close(observed, float(value), f"{label}.{key}")
        else:
            _need(observed == value, f"{label}.{key}: value drift")


def _validate_m27(config_path: Path, output: Path) -> dict[str, Any]:
    config = _json(config_path)
    _m27_config(config)
    _manifest(output, M27_PAYLOADS)
    _need(f"Status: `{config['report_status']['no_go']}`" in (output / "README.md").read_text(encoding="utf-8"),
          "M2.7 README does not preserve NO-GO")
    reference = _read_csv(output / "reference_rows.csv", REFERENCE, 570)
    cases = _reference_cases(reference)
    baseline = _read_csv(output / "matched_baseline_rows.csv", BASELINE, 990)
    baselines = _baseline_lookup(baseline, M27_K, cases)
    rows = _read_csv(output / "metric_rows.csv", M27_METRIC, 1980)
    _m27_metrics(rows, cases, baselines)
    aggregate = _read_csv(output / "aggregate_rows.csv", M27_AGGREGATE, 132)
    computed = _m27_aggregate(rows)
    actual = {(row["method"], _int(row["model_seed"], "aggregate seed"), row["split"], row["projection_variant"], _int(row["projection_iterations"], "aggregate K")): row for row in aggregate}
    _need(set(actual) == set(computed), "M2.7 aggregate identity grid drift")
    for key, values in computed.items():
        for field, value in values.items():
            _close(actual[key][field], float(value), f"M2.7 aggregate[{key!r}].{field}")
    summary = _json(output / "summary.json")
    _need(summary["schema_version"] == config["report_schema_version"], "M2.7 summary schema drift")
    _need(summary["status"] == config["report_status"]["no_go"], "M2.7 summary status must remain NO-GO")
    _need(summary["source_config_sha256"] == _sha(config_path), "M2.7 summary source config hash drift")
    for prefix in ("source_t0", "source_m2_2", "source_m2_3", "source_m2_4", "source_m2_5", "source_m2_6"):
        for suffix in ("config_sha256", "summary_sha256"):
            key = f"{prefix}_{suffix}"
            _need(summary[key] == config[key], f"M2.7 summary {key} drift")
    _match_mapping(summary["authorization"], M27_AUTHORIZATION, "M2.7 authorization")
    _need(summary["claim_boundary"] == config["claim_boundary"], "M2.7 claim boundary drift")
    _need(summary["metric_row_count"] == len(rows) and summary["reference_row_count"] == len(reference) and summary["matched_baseline_row_count"] == len(baseline),
          "M2.7 summary row count drift")
    selections = 0
    for method in METHODS:
        decision = summary["decisions"][method]
        candidates: list[tuple[int, dict[str, Any], bool]] = []
        for k in M27_K:
            development = _m27_decision_rows(rows, method, k, "development")
            eligible = (development["visible_correction_fraction_mean"] <= 0.1 and
                        development["visible_correction_fraction_maximum"] <= 0.25 and
                        development["reprojection_ratio_mean"] <= 1.1 and
                        development["projection_closure_relative_error_maximum"] <= 1e-10 and
                        development["paired_call_budget"] <= 24)
            candidates.append((k, development, eligible))
        eligible = [value for value in candidates if value[2]]
        expected_selection = None
        if eligible:
            k, _, _ = sorted(eligible, key=lambda item: (-item[1]["field_gain_mean"], item[0]))[0]
            expected_selection = {"projection_variant": "affine_pcg_dense_exact_camera_block_oracle", "projection_iterations": k, "damping_fraction": 0.0, "used_ood_for_selection": False}
            selections += 1
        _need(decision["selection"] == expected_selection, f"M2.7 {method}: development-only selection drift")
        _need(len(decision["screened_candidates"]) == len(M27_K), f"M2.7 {method}: screened candidate count drift")
        _need(expected_selection is not None and expected_selection["projection_iterations"] == 9,
              f"M2.7 {method}: expected K=9 development selection missing")
        diagnostics = {split: _m27_decision_rows(rows, method, 9, split) for split in SPLITS}
        _match_mapping(decision["diagnostics"], diagnostics, f"M2.7 {method}.diagnostics")
        _need(decision["checks"]["development_harm_rate"] is False and decision["checks"]["development_worst_case"] is False,
              f"M2.7 {method}: harm/worst NO-GO checks were not preserved")
        _need(diagnostics["development"]["field_harm_rate"] > 0.05 and diagnostics["development"]["worst_field_gain"] < -0.05,
              f"M2.7 {method}: required tail-risk NO-GO evidence missing")
        _need(decision["passed_m2_3_mechanism_gate"] is False,
              f"M2.7 {method}: NO-GO was incorrectly authorized")
    return {
        "status": "VALIDATED_M2_7_TARGET_NO_HARM_PARETO_ORACLE_NO_GO",
        "metric_row_count": len(rows), "development_selection_count": selections,
        "authorization": dict(M27_AUTHORIZATION),
    }


def _m28_config(config: dict[str, Any]) -> None:
    _need(config["schema_version"] == "jacru-m2-8-interpolation-calibration-ceiling-postopen-config-1.0",
          "M2.8 config schema drift")
    _need(config["report_schema_version"] == "jacru-m2-8-interpolation-calibration-ceiling-postopen-report-1.0",
          "M2.8 report schema drift")
    _need(config["status"].startswith("FROZEN_BEFORE_FIRST_"), "M2.8 config is not frozen")
    _need(config["methods"] == list(METHODS), "M2.8 method grid drift")
    _need(tuple(config["projection"]["iterations"]) == M28_K and max(config["projection"]["iterations"]) <= 10,
          "M2.8 K grid drift")
    _need(tuple(config["interpolation"]["fixed_fractions"]) == M28_ALPHA,
          "M2.8 fixed alpha grid drift")
    _need(config["interpolation"]["fixed_fraction_uses_truth"] is False and
          config["interpolation"]["truth_oracle_alpha_is_evaluator_only"] is True and
          config["interpolation"]["truth_oracle_alpha_is_not_a_candidate"] is True,
          "M2.8 truth-oracle boundary drift")
    budget = config["matched_budget"]
    _need((budget["learned_feature_preparation_forward_calls"], budget["learned_feature_preparation_adjoint_calls"], budget["maximum_forward_calls"], budget["maximum_adjoint_calls"]) == (13, 13, 24, 24),
          "M2.8 F/A budget drift")
    boundary = config["claim_boundary"]
    for key in ("fixed_alpha_with_exact_camera_block_is_deployable", "truth_oracle_alpha_is_deployable", "truth_oracle_alpha_may_authorize_a_method", "opens_fresh_or_final"):
        _need(boundary[key] is False, f"M2.8 claim boundary drift: {key}")
    _sources(config, ("source_t0", "source_m2_6", "source_m2_7"))


def _m28_fixed_aggregate(rows: list[dict[str, str]]) -> dict[tuple[Any, ...], dict[str, float | int]]:
    groups: dict[tuple[Any, ...], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["method"], _int(row["model_seed"], "seed"), row["split"],
               _int(row["projection_iterations"], "K"), _float(row["interpolation_fraction"], "alpha"))
        groups.setdefault(key, []).append(row)
    result: dict[tuple[Any, ...], dict[str, float | int]] = {}
    for key, values in groups.items():
        result[key] = {
            "case_count": len(values), "paired_call_budget": _int(values[0]["paired_call_budget"], "budget"),
            "field_gain_mean": _mean(_float(x["field_gain"], "gain") for x in values),
            "h1_gain_mean": _mean(_float(x["h1_gain"], "h1") for x in values),
            "reprojection_ratio_mean": _mean(_float(x["reprojection_ratio_to_matched_cgls"], "ratio") for x in values),
            "reprojection_ratio_maximum": max(_float(x["reprojection_ratio_to_matched_cgls"], "ratio") for x in values),
            "field_harm_rate": _mean(_float(x["field_harm"], "harm") for x in values),
            "worst_field_gain": min(_float(x["field_gain"], "gain") for x in values),
        }
    return result


def _quadratic_at_zero_half_one(zero: float, half: float, one: float) -> tuple[float, float, float]:
    """Recover a*x^2+b*x+c from values at x={0, .5, 1}."""
    c = zero
    a = 2.0 * zero + 2.0 * one - 4.0 * half
    b = one - c - a
    return a, b, c


def _feasible_interval(a: float, b: float, c: float, *, tolerance: float = 1e-12) -> tuple[float, float] | None:
    """Independent closed-interval solver for a*x^2+b*x+c <= 0 on [0, 1]."""
    if a < -tolerance:
        raise ValidationError("truth alpha quadratic is not convex")
    if abs(a) <= tolerance:
        if abs(b) <= tolerance:
            return (0.0, 1.0) if c <= tolerance else None
        boundary = -c / b
        result = (0.0, min(1.0, boundary)) if b > 0 else (max(0.0, boundary), 1.0)
        return result if result[0] <= result[1] + tolerance else None
    discriminant = b * b - 4.0 * a * c
    scale = max(b * b, abs(4.0 * a * c), 1.0)
    if discriminant < -tolerance * scale:
        return None
    root = math.sqrt(max(discriminant, 0.0))
    lo = max(0.0, (-b - root) / (2.0 * a))
    hi = min(1.0, (-b + root) / (2.0 * a))
    return (lo, hi) if lo <= hi + tolerance else None


def _minimizer(a: float, b: float, interval: tuple[float, float]) -> float:
    lo, hi = interval
    if a <= 1e-12:
        return lo if b >= 0 else hi
    return min(max(-b / (2.0 * a), lo), hi)


def _validate_m28(config_path: Path, output: Path) -> dict[str, Any]:
    config = _json(config_path)
    _m28_config(config)
    _manifest(output, M28_PAYLOADS)
    _need(f"Status: `{config['report_status']['no_go']}`" in (output / "README.md").read_text(encoding="utf-8"),
          "M2.8 README does not preserve NO-GO")
    fixed = _read_csv(output / "fixed_interpolation_rows.csv", M28_FIXED, 3240)
    fixed_aggregate = _read_csv(output / "fixed_interpolation_aggregate.csv", M28_FIXED_AGGREGATE, 216)
    truth = _read_csv(output / "truth_oracle_ceiling_rows.csv", M28_TRUTH, 360)
    baseline = _read_csv(output / "matched_baseline_rows.csv", BASELINE, 180)
    cases = {row["case_id"] for row in fixed}
    _need(len(cases) == 30, "M2.8 case catalog drift")
    baselines = _baseline_lookup(baseline, M28_K, cases)
    fixed_lookup: dict[tuple[str, int, str, int, float], dict[str, str]] = {}
    for index, row in enumerate(fixed):
        path = f"fixed_interpolation_rows[{index}]"
        method, seed = row["method"], _int(row["model_seed"], f"{path}.model_seed")
        k, alpha = _int(row["projection_iterations"], f"{path}.projection_iterations"), _float(row["interpolation_fraction"], f"{path}.alpha")
        _need(method in METHODS and seed in SEEDS and row["case_id"] in cases and row["split"] in SPLITS and k in M28_K and alpha in M28_ALPHA,
              f"{path}: fixed alpha identity grid drift")
        key = (method, seed, row["case_id"], k, alpha)
        _need(key not in fixed_lookup, f"duplicate fixed interpolation row: {key}")
        fixed_lookup[key] = row
        expected_f, expected_a = 14 + k, 13 + k
        _need((_int(row["optimization_forward_calls"], path), _int(row["optimization_adjoint_calls"], path), _int(row["paired_call_budget"], path)) == (expected_f, expected_a, expected_f),
              f"{path}: fixed interpolation F/A budget drift")
        _need(expected_f <= 24 and expected_a <= 24 and _int(row["grouped_adjoint_calls"], path) == 1 and _int(row["evaluation_forward_calls"], path) == 1,
              f"{path}: fixed interpolation call cap drift")
        _need(row["truth_used_by_candidate"] == "False", f"{path}: truth leaked into candidate")
        _need((_int(row["exact_camera_block_setup_forward_equivalents"], path), _int(row["preconditioner_block_count"], path)) == (1001, 3),
              f"{path}: camera-block oracle setup drift")
        cgls, huber = baselines[("cgls_matched", row["case_id"], k)], baselines[("huber_pdhg_matched", row["case_id"], k)]
        best_field = min(_float(cgls["field_relative_l2"], path), _float(huber["field_relative_l2"], path))
        best_h1 = min(_float(cgls["h1_seminorm_relative_error"], path), _float(huber["h1_seminorm_relative_error"], path))
        gain = (best_field - _float(row["field_relative_l2"], path)) / best_field
        h1_gain = (best_h1 - _float(row["h1_seminorm_relative_error"], path)) / best_h1
        ratio = _float(row["measured_reprojection_relative_l2"], path) / _float(cgls["measured_reprojection_relative_l2"], path)
        _close(row["field_gain"], gain, f"{path}.field_gain")
        _close(row["h1_gain"], h1_gain, f"{path}.h1_gain")
        _close(row["reprojection_ratio_to_matched_cgls"], ratio, f"{path}.reprojection_ratio")
        _need(_int(row["field_harm"], path) == int(gain < -0.01), f"{path}: harm threshold drift")
    expected_fixed = {(method, seed, case, k, alpha) for method in METHODS for seed in SEEDS for case in cases for k in M28_K for alpha in M28_ALPHA}
    _need(set(fixed_lookup) == expected_fixed, "M2.8 fixed alpha grid incomplete")
    computed_aggregate = _m28_fixed_aggregate(fixed)
    actual_aggregate = {(row["method"], _int(row["model_seed"], "aggregate seed"), row["split"], _int(row["projection_iterations"], "aggregate K"), _float(row["interpolation_fraction"], "aggregate alpha")): row for row in fixed_aggregate}
    _need(set(actual_aggregate) == set(computed_aggregate), "M2.8 fixed aggregate identity grid drift")
    for key, values in computed_aggregate.items():
        for field, value in values.items():
            _close(actual_aggregate[key][field], float(value), f"M2.8 aggregate[{key!r}].{field}")
    truth_lookup: set[tuple[str, int, str, int]] = set()
    for index, row in enumerate(truth):
        path = f"truth_oracle_ceiling_rows[{index}]"
        method, seed = row["method"], _int(row["model_seed"], f"{path}.model_seed")
        k = _int(row["projection_iterations"], f"{path}.projection_iterations")
        key = (method, seed, row["case_id"], k)
        _need(method in METHODS and seed in SEEDS and row["case_id"] in cases and k in M28_K and key not in truth_lookup,
              f"{path}: truth ceiling identity grid drift")
        truth_lookup.add(key)
        _need((_int(row["paired_call_budget"], path), _int(row["exact_camera_block_setup_forward_equivalents"], path)) == (14 + k, 1001),
              f"{path}: truth ceiling budget/setup drift")
        _need(row["truth_used_by_candidate"] == "True" and row["candidate_deployable"] == "False",
              f"{path}: truth oracle was not evaluator-only")
        _close(row["per_case_reprojection_ratio_limit"], 1.1, f"{path}.ratio_limit")
        anchor = {alpha: fixed_lookup[(method, seed, row["case_id"], k, alpha)] for alpha in (0.0, 0.5, 1.0)}
        cgls = baselines[("cgls_matched", row["case_id"], k)]
        threshold = 1.1 * _float(cgls["measured_reprojection_relative_l2"], path)
        a_r, b_r, c_r = _quadratic_at_zero_half_one(*(_float(anchor[alpha]["measured_reprojection_relative_l2"], path) ** 2 for alpha in (0.0, 0.5, 1.0)))
        interval = _feasible_interval(a_r, b_r, c_r - threshold * threshold)
        feasible = _int(row["reprojection_feasible"], f"{path}.feasible") == 1
        _need(feasible == (interval is not None), f"{path}: truth alpha feasibility drift")
        if not feasible:
            for field in ("feasible_alpha_lower", "feasible_alpha_upper", "truth_oracle_alpha", "field_relative_l2", "h1_seminorm_relative_error", "measured_reprojection_relative_l2", "clean_reprojection_relative_l2", "field_gain", "h1_gain", "reprojection_ratio_to_matched_cgls", "field_harm"):
                _need(row[field] == "", f"{path}: infeasible truth row must remain blank")
            continue
        assert interval is not None
        low, high = interval
        _close(row["feasible_alpha_lower"], low, f"{path}.feasible_alpha_lower", tolerance=2e-7)
        _close(row["feasible_alpha_upper"], high, f"{path}.feasible_alpha_upper", tolerance=2e-7)
        a_f, b_f, c_f = _quadratic_at_zero_half_one(*(_float(anchor[alpha]["field_relative_l2"], path) ** 2 for alpha in (0.0, 0.5, 1.0)))
        alpha = _minimizer(a_f, b_f, interval)
        _close(row["truth_oracle_alpha"], alpha, f"{path}.truth_oracle_alpha", tolerance=2e-7)
        expected_field = math.sqrt(max(0.0, a_f * alpha * alpha + b_f * alpha + c_f))
        _close(row["field_relative_l2"], expected_field, f"{path}.truth_oracle_field", tolerance=2e-7)
        ratio = _float(row["measured_reprojection_relative_l2"], path) / _float(cgls["measured_reprojection_relative_l2"], path)
        _close(row["reprojection_ratio_to_matched_cgls"], ratio, f"{path}.truth_oracle_ratio")
        _need(ratio <= 1.1 + 2e-10, f"{path}: truth alpha violates its reprojection constraint")
    expected_truth = {(method, seed, case, k) for method in METHODS for seed in SEEDS for case in cases for k in M28_K}
    _need(truth_lookup == expected_truth, "M2.8 truth ceiling grid drift")
    summary = _json(output / "summary.json")
    _need(summary["schema_version"] == config["report_schema_version"] and summary["status"] == config["report_status"]["no_go"],
          "M2.8 summary must remain NO-GO")
    _need(summary["source_config_sha256"] == _sha(config_path), "M2.8 source config hash drift")
    for prefix in ("source_t0", "source_m2_6", "source_m2_7"):
        for suffix in ("config_sha256", "summary_sha256"):
            key = f"{prefix}_{suffix}"
            _need(summary[key] == config[key], f"M2.8 summary {key} drift")
    _match_mapping(summary["authorization"], M28_AUTHORIZATION, "M2.8 authorization")
    _need(summary["claim_boundary"] == config["claim_boundary"], "M2.8 claim boundary drift")
    _need((summary["fixed_interpolation_row_count"], summary["truth_oracle_ceiling_row_count"], summary["matched_baseline_row_count"]) == (len(fixed), len(truth), len(baseline)),
          "M2.8 summary row count drift")
    for name in ("fixed_interpolation_decisions", "truth_oracle_ceiling_decisions"):
        for method in METHODS:
            decision = summary[name][method]
            _need(decision["selection"] is None, f"M2.8 {name}.{method}: selection must remain absent")
            passed_field = "passed_fixed_interpolation_gate" if name.startswith("fixed") else "passed_truth_oracle_ceiling"
            _need(decision[passed_field] is False, f"M2.8 {name}.{method}: NO-GO was incorrectly authorized")
    return {
        "status": "VALIDATED_M2_8_INTERPOLATION_CALIBRATION_ENVELOPE_NO_GO",
        "fixed_interpolation_row_count": len(fixed), "truth_oracle_ceiling_row_count": len(truth),
        "authorization": dict(M28_AUTHORIZATION),
    }


def validate_packet(*, config_path: Path, output_dir: Path) -> dict[str, Any]:
    """Validate one M2.7 or M2.8 frozen evidence packet."""
    config_path, output_dir = Path(config_path), Path(output_dir)
    _need(config_path.is_file() and output_dir.is_dir(), "packet path missing")
    schema = _json(config_path).get("schema_version", "")
    if schema.startswith("jacru-m2-7-"):
        return _validate_m27(config_path, output_dir)
    if schema.startswith("jacru-m2-8-"):
        return _validate_m28(config_path, output_dir)
    raise ValidationError("unsupported JACRU M2.7/M2.8 config schema")


def validate_all_packets() -> dict[str, dict[str, Any]]:
    return {
        "m2_7": validate_packet(
            config_path=CONFIGS / "jacru_m2_7_target_no_harm_pareto_ceiling_postopen_v1.json",
            output_dir=RESULTS / "jacru_m2_7_target_no_harm_pareto_ceiling_postopen_public",
        ),
        "m2_8": validate_packet(
            config_path=CONFIGS / "jacru_m2_8_interpolation_calibration_ceiling_postopen_v1.json",
            output_dir=RESULTS / "jacru_m2_8_interpolation_calibration_ceiling_postopen_public",
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", choices=("all", "m2-7", "m2-8"), default="all")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.packet == "all":
        _need(args.config is None and args.output_dir is None, "custom paths require one packet")
        report: Any = validate_all_packets()
    else:
        defaults = {
            "m2-7": (CONFIGS / "jacru_m2_7_target_no_harm_pareto_ceiling_postopen_v1.json", RESULTS / "jacru_m2_7_target_no_harm_pareto_ceiling_postopen_public"),
            "m2-8": (CONFIGS / "jacru_m2_8_interpolation_calibration_ceiling_postopen_v1.json", RESULTS / "jacru_m2_8_interpolation_calibration_ceiling_postopen_public"),
        }[args.packet]
        report = validate_packet(config_path=args.config or defaults[0], output_dir=args.output_dir or defaults[1])
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
