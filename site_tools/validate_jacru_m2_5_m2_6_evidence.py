#!/usr/bin/env python3
"""Independently audit the frozen JACRU M2.5 and M2.6 evidence packets.

The auditor intentionally uses only standard-library parsers and frozen public
files.  In particular, it does not import an experiment runner or reconstruct
the forward model.  M2.5 predates the closure/application ledger: that absence
is retained as a non-upgradeable audit finding, not silently treated as proof.
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
RESULTS = ROOT / "demo_t16_operator/results"
CONFIGS = ROOT / "demo_t16_operator/configs"
METHODS = ("jacru_m2", "pooled_cnn")
SEEDS = (17, 29, 43)
SPLITS = ("development", "ood")
SPLIT_COUNTS = {"development": 12, "ood": 18}
SNAPSHOTS = (0, 1, 2, 4, 8, 12, 20, 32)
BASELINES = ("cgls_matched", "huber_pdhg_matched", "base_landweber_matched")
PUBLIC_POLICY = {
    "contains_model_checkpoints": False,
    "contains_restricted_papers": False,
    "contains_private_experimental_arrays": False,
}
AUTHORIZATION = {
    "claim_deployable_algorithm": False,
    "claim_method_superiority": False,
    "claim_real_bost_generalization": False,
    "open_fresh_or_final": False,
    "draft_new_preregistered_fresh_gate": False,
    "continue_matrix_free_preconditioner_research": True,
    "continue_deployable_preconditioner_estimation": False,
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
PROJECTION = (
    "projection_variant", "projection_iterations", "damping_fraction",
    "damping_absolute", "preconditioner", "projection_forward_calls",
    "projection_adjoint_calls", "paired_call_budget",
    "matched_cgls_field_relative_l2", "matched_huber_field_relative_l2",
    "matched_base_landweber_field_relative_l2",
    "field_gain_to_best_matched_classical", "h1_gain_to_best_matched_classical",
    "reprojection_ratio_to_matched_cgls", "visible_correction_fraction",
    "system_residual_fraction", "exact_projection_approximation_error",
    "exact_oracle_error_reduction_retention", "oracle_reduction_defined",
    "base_anchor_field_relative_l2", "exact_oracle_field_relative_l2",
    "raw_learned_field_relative_l2", "exact_oracle_rank",
    "exact_oracle_nullity_lower_bound", "field_harm_to_best_matched_classical",
    "converged", "breakdown", "projection_diagnostic_forward_calls",
    "dense_oracle_used_by_algorithm", "projection_target_mode",
    "exact_oracle_internal_projection_residual", "preconditioner_kind",
    "preconditioner_is_oracle", "preconditioner_setup_forward_equivalents",
    "preconditioner_setup_adjoint_equivalents",
)
M2_5_METRIC = COMMON + PROJECTION
M2_6_METRIC = M2_5_METRIC[:36] + ("projection_closure_relative_error",) + M2_5_METRIC[36:] + (
    "preconditioner_applications", "preconditioner_block_count",
    "preconditioner_largest_block_size", "preconditioner_minimum_block_eigenvalue",
    "preconditioner_maximum_block_condition_number",
)
REFERENCE = COMMON + ("reference_kind",)
BASELINE = COMMON + (
    "matched_step", "total_calls", "baseline_kind", "dc_step_size",
    "operator_norm_squared_bound", "projection_iterations", "paired_call_budget",
    "matched_step_internal_offset",
)
AGGREGATE_PREFIX = (
    "method", "model_seed", "split", "projection_variant",
    "projection_iterations", "case_count", "paired_call_budget", "damping_fraction",
    "field_relative_l2_mean", "h1_seminorm_relative_error_mean",
    "field_gain_to_best_matched_classical_mean",
    "h1_gain_to_best_matched_classical_mean",
    "reprojection_ratio_to_matched_cgls_mean",
    "visible_correction_fraction_mean", "visible_correction_fraction_maximum",
    "system_residual_fraction_mean",
)
AGGREGATE_SUFFIX = (
    "exact_projection_approximation_error_mean",
    "exact_oracle_error_reduction_retention_mean", "oracle_reduction_defined_rate",
    "field_harm_rate", "worst_field_gain_to_best_matched_classical", "breakdown_rate",
)
M2_5_AGGREGATE = AGGREGATE_PREFIX + AGGREGATE_SUFFIX
M2_6_AGGREGATE = AGGREGATE_PREFIX + (
    "projection_closure_relative_error_mean", "projection_closure_relative_error_maximum",
) + AGGREGATE_SUFFIX
BASELINE_AGGREGATE = (
    "method", "split", "matched_step", "total_calls", "case_count",
    "field_relative_l2_mean", "h1_seminorm_relative_error_mean",
    "measured_reprojection_relative_l2_mean",
)
PAYLOADS = {
    "README.md", "aggregate_rows.csv", "diagnostic.pdf", "diagnostic.png",
    "matched_baseline_aggregate_rows.csv", "matched_baseline_rows.csv",
    "metric_rows.csv", "reference_rows.csv", "summary.json",
}


class ValidationError(RuntimeError):
    """Raised when an evidence packet violates its frozen, NO-GO contract."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ValidationError(f"cannot hash {path}: {error}") from error


def _load_json(path: Path) -> dict[str, Any]:
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


def _close(actual: Any, expected: float, path: str) -> None:
    observed = _float(actual, path)
    _need(math.isclose(observed, expected, rel_tol=5e-11, abs_tol=5e-12),
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
    for index, row in enumerate(rows):
        _need(None not in row and all(v is not None for v in row.values()),
              f"{path.name}[{index}]: malformed row")
    return rows


def _manifest(output: Path) -> None:
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
    _need(set(entries) == PAYLOADS, "checksums.sha256: payload set mismatch")
    _need({item.name for item in output.iterdir()} == PAYLOADS | {"checksums.sha256"},
          "public packet contains unmanifested or missing files")
    for name, digest in entries.items():
        path = output / name
        _need(path.is_file() and not path.is_symlink(), f"invalid payload: {name}")
        _need(_sha(path) == digest, f"checksum mismatch: {name}")


def _source_manifest_digest(directory: Path, name: str) -> str:
    lines = (directory / "checksums.sha256").read_text(encoding="ascii").splitlines()
    matches = [line.split("  ")[0] for line in lines if line.endswith(f"  {name}")]
    _need(len(matches) == 1, f"source manifest missing {name}")
    _need(_sha(directory / name) == matches[0], f"source manifest checksum mismatch: {name}")
    return matches[0]


def _variant_specs(stage: str) -> tuple[dict[str, Any], ...]:
    identity = {
        "name": "affine_cg_identity_control", "preconditioner": "identity",
        "row_preconditioner": "identity", "oracle": 0, "setup_f": 0,
        "block_count": 0, "block_size": 0,
    }
    jacobi = {
        "name": "affine_pcg_dense_exact_jacobi_oracle",
        "preconditioner": "dense_exact_jacobi_oracle",
        "row_preconditioner": "supplied_positive_diagonal", "oracle": 1,
        "setup_f": 1001, "block_count": 0, "block_size": 0,
    }
    if stage == "M2.5":
        return (identity, jacobi)
    block = {
        "name": "affine_pcg_dense_exact_camera_block_oracle",
        "preconditioner": "dense_exact_camera_block_jacobi_oracle",
        "row_preconditioner": "dense_exact_camera_block_jacobi_oracle",
        "oracle": 1, "setup_f": 1001, "block_count": 3, "block_size": 50,
    }
    return (identity, jacobi, block)


def _packet(stage: str) -> dict[str, Any]:
    if stage == "M2.5":
        stem = "jacru_m2_5_exact_jacobi_preconditioner_oracle_postopen"
        return {
            "stage": stage, "config": CONFIGS / f"{stem}_v1.json",
            "output": RESULTS / f"{stem}_public", "schema": "jacru-m2-5-exact-jacobi-preconditioner-oracle-postopen",
            "report_status": "M2_5_EXACT_JACOBI_PRECONDITIONER_ORACLE_NO_GO",
            "validated_status": "VALIDATED_M2_5_ORACLE_NO_GO_CLOSURE_AND_APPLICATION_LEDGER_UNAVAILABLE",
            "metric_fields": M2_5_METRIC, "aggregate_fields": M2_5_AGGREGATE,
            "variants": _variant_specs(stage), "requires_closure": False,
        }
    stem = "jacru_m2_6_camera_block_preconditioner_oracle_postopen"
    return {
        "stage": stage, "config": CONFIGS / f"{stem}_v1.json",
        "output": RESULTS / f"{stem}_public", "schema": "jacru-m2-6-camera-block-preconditioner-oracle-postopen",
        "report_status": "M2_6_CAMERA_BLOCK_PRECONDITIONER_ORACLE_NO_GO",
        "validated_status": "VALIDATED_M2_6_CAMERA_BLOCK_ORACLE_NO_GO_HARM_BLOCKS_AUTHORIZATION",
        "metric_fields": M2_6_METRIC, "aggregate_fields": M2_6_AGGREGATE,
        "variants": _variant_specs(stage), "requires_closure": True,
    }


def _expected_counts(spec: dict[str, Any]) -> dict[str, int]:
    cases = sum(SPLIT_COUNTS.values())
    return {
        "metric": len(METHODS) * len(SEEDS) * len(spec["variants"]) * len(SNAPSHOTS) * cases,
        "reference": cases * (1 + 3 * len(METHODS) * len(SEEDS)),
        "baseline": len(BASELINES) * len(SNAPSHOTS) * cases,
        "aggregate": len(METHODS) * len(SEEDS) * len(SPLITS) * len(spec["variants"]) * len(SNAPSHOTS),
        "baseline_aggregate": len(BASELINES) * len(SPLITS) * len(SNAPSHOTS),
    }


def _validate_config(config: dict[str, Any], config_path: Path, spec: dict[str, Any]) -> None:
    stage_number = spec["stage"].lower().replace(".", "_")
    _need(config["schema_version"] == f"{spec['schema']}-config-1.0", "config schema drift")
    _need(config["report_schema_version"] == f"{spec['schema']}-report-1.0", "config report schema drift")
    _need(config["status"].startswith("FROZEN_BEFORE_FIRST_"), "config is not frozen")
    _need(config["frozen_date"] == "2026-07-17", "config frozen date drift")
    _need(config["preconditioner_oracle_only"] is True, "config oracle boundary drift")
    _need(config["report_status"]["no_go"] == spec["report_status"], "config NO-GO status drift")
    _need(config["methods"] == list(METHODS), "config method grid drift")
    _need(config["projection"]["snapshot_iterations"] == list(SNAPSHOTS), "config snapshot grid drift")
    _need(config["projection"]["denominator_floor"] == 1e-30, "config denominator drift")
    frozen_variants = config["projection"]["variants"]
    _need(len(frozen_variants) == len(spec["variants"]), "config variant count drift")
    for actual, expected in zip(frozen_variants, spec["variants"]):
        _need(actual == {
            "name": expected["name"], "target_mode": "affine_observation",
            "preconditioner": expected["preconditioner"],
            "damping_fraction_of_operator_norm_squared_bound": 0.0,
        }, "config projection variant drift")
    budget = config["matched_budget"]
    _need(budget["learned_feature_preparation_forward_calls"] == 13, "config learned F budget drift")
    _need(budget["learned_feature_preparation_adjoint_calls"] == 13, "config learned A budget drift")
    _need(budget["projection_forward_calls_formula"] == "K+1", "config projection F formula drift")
    _need(budget["projection_adjoint_calls_formula"] == "K", "config projection A formula drift")
    _need(budget["matched_classical_pair_iterations_formula"] == "14+K", "config baseline formula drift")
    _need(budget["classical_methods"] == list(BASELINES), "config baseline methods drift")
    _need(budget["dense_norm_setup_excluded_and_reported"] is True, "config norm setup boundary drift")
    _need(budget["dense_exact_jacobi_setup_is_excluded_oracle_and_reported"] is True,
          "config Jacobi setup boundary drift")
    if spec["stage"] == "M2.6":
        _need(budget["dense_exact_camera_block_setup_is_excluded_oracle_and_reported"] is True,
              "config block setup boundary drift")
        _need(config["decision_gates"]["maximum_projection_closure_relative_error"] == 1e-10,
              "config closure gate drift")
        _need(config["development_selection_rule"]["maximum_projection_closure_relative_error"] == 1e-10,
              "config selection closure gate drift")
    boundary = config["claim_boundary"]
    for key in ("is_runtime_or_efficiency_evidence", "is_confirmatory_or_final",
                "is_experimental_reconstruction", "is_real_bost_generalization", "opens_fresh_or_final"):
        _need(boundary[key] is False, f"config claim boundary drift: {key}")
    _need(boundary["uses_only_opened_synthetic_t0_for_selection_and_scoring"] is True,
          "config opened-data boundary drift")
    if spec["stage"] == "M2.5":
        _need(boundary["exact_jacobi_is_matrix_free_or_deployable"] is False,
              "config deployability boundary drift")
        _need(boundary["exact_jacobi_setup_is_in_matched_budget"] is False,
              "config setup-budget boundary drift")
    else:
        _need(boundary["exact_camera_block_is_matrix_free_or_deployable"] is False,
              "config deployability boundary drift")
        _need(boundary["exact_camera_block_setup_is_in_matched_budget"] is False,
              "config setup-budget boundary drift")
    prefixes = ["source_t0", "source_m2_2", "source_m2_3", "source_m2_4"]
    if spec["stage"] == "M2.6":
        prefixes.append("source_m2_5")
    for prefix in prefixes:
        source_config = ROOT / config[f"{prefix}_config"]
        source_results = ROOT / config[f"{prefix}_results"]
        _need(source_config.is_file() and source_config.is_relative_to(ROOT), f"{prefix} config missing")
        _need(source_results.is_dir() and source_results.is_relative_to(ROOT), f"{prefix} result missing")
        _need(_sha(source_config) == config[f"{prefix}_config_sha256"], f"{prefix} config hash drift")
        _need(_sha(source_results / "summary.json") == config[f"{prefix}_summary_sha256"],
              f"{prefix} summary hash drift")
        _source_manifest_digest(source_results, "summary.json")
    _need(config_path.is_file() and _sha(config_path), f"{stage_number} frozen config missing")


def _validate_references(rows: list[dict[str, str]]) -> set[str]:
    cases: set[str] = set()
    grid: set[tuple[str, str, int, str]] = set()
    for i, row in enumerate(rows):
        path = f"reference_rows[{i}]"
        _need(row["split"] in SPLITS, f"{path}.split drift")
        case = row["case_id"]
        cases.add(case)
        kind = row["reference_kind"]
        seed = _int(row["model_seed"], f"{path}.model_seed")
        key = (kind, row["method"], seed, case)
        _need(key not in grid, f"duplicate reference row: {key}")
        grid.add(key)
        if kind == "base_anchor":
            _need((row["method"], seed) == ("prepared_cgls_base_12", -1),
                  f"{path}: base anchor identity drift")
            _need((_int(row["optimization_forward_calls"], path), _int(row["optimization_adjoint_calls"], path)) == (12, 12),
                  f"{path}: base anchor F/A budget drift")
        else:
            _need(kind in {"raw_learned", "retrospective_dense_oracle_base_anchor", "retrospective_dense_oracle_affine_observation"},
                  f"{path}: reference kind drift")
            _need(row["method"] in METHODS and seed in SEEDS, f"{path}: learned identity drift")
    _need(len(cases) == 30, "reference case catalog drift")
    return cases


def _validate_baselines(rows: list[dict[str, str]], cases: set[str]) -> dict[tuple[str, str, int], dict[str, str]]:
    lookup: dict[tuple[str, str, int], dict[str, str]] = {}
    for i, row in enumerate(rows):
        path = f"matched_baseline_rows[{i}]"
        k = _int(row["projection_iterations"], f"{path}.projection_iterations")
        _need(k in SNAPSHOTS and row["method"] in BASELINES and row["case_id"] in cases,
              f"{path}: baseline grid drift")
        key = (row["method"], row["case_id"], k)
        _need(key not in lookup, f"duplicate baseline row: {key}")
        lookup[key] = row
        paired = 14 + k
        for field, expected in (("optimization_forward_calls", paired), ("optimization_adjoint_calls", paired),
                                ("total_calls", paired), ("paired_call_budget", paired),
                                ("matched_step", 1 + k), ("matched_step_internal_offset", 1 + k),
                                ("evaluation_forward_calls", 1)):
            _need(_int(row[field], f"{path}.{field}") == expected,
                  f"matched baseline {field} budget drift")
        _need(_int(row["model_seed"], f"{path}.model_seed") == -1, f"{path}.seed drift")
        _need(row["baseline_kind"] == row["method"], f"{path}.kind drift")
        _need(_float(row["operator_norm_squared_bound"], path) > 0, f"{path}: invalid operator bound")
    expected = {(method, case, k) for method in BASELINES for case in cases for k in SNAPSHOTS}
    _need(set(lookup) == expected, "matched baseline identity grid drift")
    return lookup


def _validate_metrics(rows: list[dict[str, str]], cases: set[str], baseline: dict[tuple[str, str, int], dict[str, str]], spec: dict[str, Any]) -> None:
    variants = {entry["name"]: entry for entry in spec["variants"]}
    seen: set[tuple[str, int, str, str, int]] = set()
    for i, row in enumerate(rows):
        path = f"metric_rows[{i}]"
        seed = _int(row["model_seed"], f"{path}.model_seed")
        k = _int(row["projection_iterations"], f"{path}.projection_iterations")
        variant = variants.get(row["projection_variant"])
        _need(row["method"] in METHODS and seed in SEEDS and row["case_id"] in cases and k in SNAPSHOTS and variant is not None,
              f"{path}: metric identity grid drift")
        key = (row["method"], seed, row["case_id"], row["projection_variant"], k)
        _need(key not in seen, f"duplicate metric row: {key}")
        seen.add(key)
        _need(row["projection_target_mode"] == "affine_observation", f"{path}: target mode drift")
        _need(_float(row["exact_oracle_internal_projection_residual"], path) <= 1e-10,
              f"{path}: exact affine oracle residual drift")
        expected_f, expected_a = 14 + k, 13 + k
        for field, expected in (("projection_forward_calls", k + 1), ("projection_adjoint_calls", k),
                                ("optimization_forward_calls", expected_f), ("optimization_adjoint_calls", expected_a),
                                ("paired_call_budget", expected_f), ("projection_diagnostic_forward_calls", 1),
                                ("grouped_adjoint_calls", 1), ("evaluation_forward_calls", 1)):
            _need(_int(row[field], f"{path}.{field}") == expected,
                  f"metric {field} budget formula drift")
        _need(_float(row["damping_fraction"], path) == 0.0 and _float(row["damping_absolute"], path) == 0.0,
              f"{path}: damping drift")
        _need(row["preconditioner"] == variant["row_preconditioner"], f"{path}: preconditioner drift")
        _need(row["preconditioner_kind"] == variant["preconditioner"], f"{path}: preconditioner kind drift")
        _need(_int(row["preconditioner_is_oracle"], path) == variant["oracle"], f"{path}: oracle flag drift")
        _need(_int(row["preconditioner_setup_forward_equivalents"], path) == variant["setup_f"],
              f"{path}: oracle setup ledger drift")
        _need(_int(row["preconditioner_setup_adjoint_equivalents"], path) == 0,
              f"{path}: oracle setup adjoint ledger drift")
        _need(row["dense_oracle_used_by_algorithm"] == "False", f"{path}: dense oracle entered algorithm")
        _need((_int(row["exact_oracle_rank"], path), _int(row["exact_oracle_nullity_lower_bound"], path)) == (150, 850),
              f"{path}: dense oracle rank drift")
        if spec["requires_closure"]:
            _need(_float(row["projection_closure_relative_error"], path) <= 1e-10,
                  f"{path}: projection closure exceeds 1e-10")
            # The packet records the fixed full K=32 Krylov path once and
            # snapshots its iterates.  Thus every snapshot explicitly bills
            # 33 preconditioner applications rather than silently claiming a
            # cheaper K+1 recomputation for the earlier snapshots.
            _need(_int(row["preconditioner_applications"], path) == 33,
                  f"{path}: preconditioner application ledger drift")
            _need(_int(row["preconditioner_block_count"], path) == variant["block_count"], f"{path}: block count drift")
            _need(_int(row["preconditioner_largest_block_size"], path) == variant["block_size"], f"{path}: block size drift")
            eig = _float(row["preconditioner_minimum_block_eigenvalue"], path)
            cond = _float(row["preconditioner_maximum_block_condition_number"], path)
            if variant["block_count"]:
                _need(eig > 0.0 and cond >= 1.0, f"{path}: camera block is not SPD")
            else:
                _need(eig == 0.0 and cond == 0.0, f"{path}: unexpected block diagnostics")
        cgls = baseline[("cgls_matched", row["case_id"], k)]
        huber = baseline[("huber_pdhg_matched", row["case_id"], k)]
        landweber = baseline[("base_landweber_matched", row["case_id"], k)]
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
        _close(row["reprojection_ratio_to_matched_cgls"], ratio, f"{path}.reprojection ratio")
        _need(_int(row["field_harm_to_best_matched_classical"], path) == int(gain < -0.01), f"{path}: harm threshold drift")
    expected = {(method, seed, case, variant["name"], k) for method in METHODS for seed in SEEDS
                for case in cases for variant in spec["variants"] for k in SNAPSHOTS}
    _need(seen == expected, "metric identity grid drift")


def _aggregate(rows: list[dict[str, str]], closure: bool) -> dict[tuple[Any, ...], dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["method"], _int(row["model_seed"], "model seed"), row["split"], row["projection_variant"], _int(row["projection_iterations"], "iterations"))
        groups.setdefault(key, []).append(row)
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for key, values in groups.items():
        result[key] = {
            "case_count": len(values), "paired_call_budget": _int(values[0]["paired_call_budget"], "budget"),
            "damping_fraction": _float(values[0]["damping_fraction"], "damping"),
            "field_relative_l2_mean": _mean(_float(x["field_relative_l2"], "field") for x in values),
            "h1_seminorm_relative_error_mean": _mean(_float(x["h1_seminorm_relative_error"], "h1") for x in values),
            "field_gain_to_best_matched_classical_mean": _mean(_float(x["field_gain_to_best_matched_classical"], "gain") for x in values),
            "h1_gain_to_best_matched_classical_mean": _mean(_float(x["h1_gain_to_best_matched_classical"], "h1 gain") for x in values),
            "reprojection_ratio_to_matched_cgls_mean": _mean(_float(x["reprojection_ratio_to_matched_cgls"], "ratio") for x in values),
            "visible_correction_fraction_mean": _mean(_float(x["visible_correction_fraction"], "visible") for x in values),
            "visible_correction_fraction_maximum": max(_float(x["visible_correction_fraction"], "visible") for x in values),
            "system_residual_fraction_mean": _mean(_float(x["system_residual_fraction"], "residual") for x in values),
            "exact_projection_approximation_error_mean": _mean(_float(x["exact_projection_approximation_error"], "approx") for x in values),
            "exact_oracle_error_reduction_retention_mean": _mean(_float(x["exact_oracle_error_reduction_retention"], "retention") for x in values),
            "oracle_reduction_defined_rate": _mean(_float(x["oracle_reduction_defined"], "defined") for x in values),
            "field_harm_rate": _mean(_float(x["field_harm_to_best_matched_classical"], "harm") for x in values),
            "worst_field_gain_to_best_matched_classical": min(_float(x["field_gain_to_best_matched_classical"], "gain") for x in values),
            "breakdown_rate": _mean(_float(x["breakdown"], "breakdown") for x in values),
        }
        if closure:
            result[key]["projection_closure_relative_error_mean"] = _mean(_float(x["projection_closure_relative_error"], "closure") for x in values)
            result[key]["projection_closure_relative_error_maximum"] = max(_float(x["projection_closure_relative_error"], "closure") for x in values)
    return result


def _compare_aggregate(rows: list[dict[str, str]], computed: dict[tuple[Any, ...], dict[str, Any]], closure: bool, label: str) -> None:
    actual: dict[tuple[Any, ...], dict[str, str]] = {}
    for row in rows:
        key = (row["method"], _int(row["model_seed"], label), row["split"], row["projection_variant"], _int(row["projection_iterations"], label))
        _need(key not in actual, f"{label}: duplicate aggregate")
        actual[key] = row
    _need(set(actual) == set(computed), f"{label}: aggregate identity grid mismatch")
    for key, expected in computed.items():
        row = actual[key]
        for field, value in expected.items():
            _close(row[field], float(value), f"{label}[{key!r}].{field}")
        if closure:
            _need(_float(row["projection_closure_relative_error_maximum"], label) <= 1e-10,
                  f"{label}[{key!r}]: closure threshold exceeded")


def _diagnostics(rows: list[dict[str, str]], method: str, variant: str, k: int, split: str, closure: bool) -> dict[str, Any]:
    values = [row for row in rows if row["method"] == method and row["projection_variant"] == variant and _int(row["projection_iterations"], "K") == k and row["split"] == split]
    _need(len(values) == len(SEEDS) * SPLIT_COUNTS[split], "decision row count drift")
    out: dict[str, Any] = {
        "case_model_count": len(values),
        "field_gain_mean": _mean(_float(x["field_gain_to_best_matched_classical"], "gain") for x in values),
        "h1_gain_mean": _mean(_float(x["h1_gain_to_best_matched_classical"], "h1") for x in values),
        "reprojection_ratio_mean": _mean(_float(x["reprojection_ratio_to_matched_cgls"], "ratio") for x in values),
        "visible_correction_fraction_mean": _mean(_float(x["visible_correction_fraction"], "visible") for x in values),
        "visible_correction_fraction_maximum": max(_float(x["visible_correction_fraction"], "visible") for x in values),
        "exact_oracle_error_reduction_retention_mean": _mean(_float(x["exact_oracle_error_reduction_retention"], "retention") for x in values),
        "oracle_reduction_defined_rate": _mean(_float(x["oracle_reduction_defined"], "defined") for x in values),
        "field_harm_rate": _mean(_float(x["field_harm_to_best_matched_classical"], "harm") for x in values),
        "worst_field_gain": min(_float(x["field_gain_to_best_matched_classical"], "gain") for x in values),
        "breakdown_rate": _mean(_float(x["breakdown"], "breakdown") for x in values),
        "per_model_seed_field_gain_means": [_mean(_float(x["field_gain_to_best_matched_classical"], "gain") for x in values if _int(x["model_seed"], "seed") == seed) for seed in SEEDS],
    }
    if closure:
        out["projection_closure_relative_error_mean"] = _mean(_float(x["projection_closure_relative_error"], "closure") for x in values)
        out["projection_closure_relative_error_maximum"] = max(_float(x["projection_closure_relative_error"], "closure") for x in values)
    return out


def _compare_value(actual: Any, expected: Any, path: str) -> None:
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        _need(actual == expected, f"{path}: value mismatch")
    elif isinstance(expected, int):
        _need(type(actual) is int and actual == expected, f"{path}: integer mismatch")
    elif isinstance(expected, float):
        _close(actual, expected, path)
    elif isinstance(expected, list):
        _need(isinstance(actual, list) and len(actual) == len(expected), f"{path}: list mismatch")
        for index, value in enumerate(expected):
            _compare_value(actual[index], value, f"{path}[{index}]")
    elif isinstance(expected, dict):
        _need(isinstance(actual, dict) and set(actual) == set(expected), f"{path}: object keys mismatch")
        for key, value in expected.items():
            _compare_value(actual[key], value, f"{path}.{key}")
    else:
        _need(actual == expected, f"{path}: value mismatch")


def _decisions(rows: list[dict[str, str]], spec: dict[str, Any]) -> dict[str, Any]:
    closure = bool(spec["requires_closure"])
    result: dict[str, Any] = {}
    for method in METHODS:
        candidates: list[dict[str, Any]] = []
        for variant in spec["variants"]:
            for k in SNAPSHOTS:
                development = _diagnostics(rows, method, variant["name"], k, "development", closure)
                eligible = (development["visible_correction_fraction_mean"] <= 0.1 and
                            development["visible_correction_fraction_maximum"] <= 0.25 and
                            development["reprojection_ratio_mean"] <= 1.1 and
                            (not closure or development["projection_closure_relative_error_maximum"] <= 1e-10))
                candidates.append({"projection_variant": variant["name"], "projection_iterations": k,
                                   "damping_fraction": 0.0, "development": development,
                                   "development_eligible": eligible})
        eligible = [candidate for candidate in candidates if candidate["development_eligible"]]
        selection = None
        diagnostics = None
        if eligible:
            selection_candidate = sorted(eligible, key=lambda x: (-x["development"]["field_gain_mean"], x["projection_iterations"], x["damping_fraction"], x["projection_variant"]))[0]
            selection = {"projection_variant": selection_candidate["projection_variant"],
                         "projection_iterations": selection_candidate["projection_iterations"],
                         "damping_fraction": 0.0, "used_ood_for_selection": False}
            diagnostics = {split: _diagnostics(rows, method, selection["projection_variant"], selection["projection_iterations"], split, closure) for split in SPLITS}
        checks: dict[str, Any] = {"development_selection_exists": selection is not None}
        if diagnostics is not None:
            for split, field_gate, h1_gate, ratio_gate, visible_gate in (("development", .05, .03, 1.1, .1), ("ood", .02, 0.0, 1.15, .15)):
                diagnostic = diagnostics[split]
                checks.update({
                    f"{split}_field_gain": diagnostic["field_gain_mean"] >= field_gate,
                    f"{split}_h1_gain": diagnostic["h1_gain_mean"] >= h1_gate,
                    f"{split}_oracle_retention": diagnostic["exact_oracle_error_reduction_retention_mean"] >= .5,
                    f"{split}_reprojection": diagnostic["reprojection_ratio_mean"] <= ratio_gate,
                    f"{split}_visible_correction": diagnostic["visible_correction_fraction_mean"] <= visible_gate,
                    f"{split}_harm_rate": diagnostic["field_harm_rate"] <= .05,
                    f"{split}_worst_case": diagnostic["worst_field_gain"] >= -.05,
                    f"{split}_all_seed_means_positive": all(x > 0 for x in diagnostic["per_model_seed_field_gain_means"]),
                    f"{split}_no_breakdown": diagnostic["breakdown_rate"] == 0.0,
                })
                if closure:
                    checks[f"{split}_projection_closure"] = diagnostic["projection_closure_relative_error_maximum"] <= 1e-10
        decision = {"screened_candidates": candidates, "selection": selection,
                    "checks": checks,
                    "passed_m2_3_mechanism_gate": bool(selection is not None and all(checks.values()))}
        if diagnostics is not None:
            decision["diagnostics"] = diagnostics
        result[method] = decision
    return result


def _validate_summary(summary: dict[str, Any], config: dict[str, Any], config_path: Path, spec: dict[str, Any], counts: dict[str, int], rows: list[dict[str, str]], aggregate: list[dict[str, str]], baseline_aggregate: list[dict[str, str]]) -> dict[str, Any]:
    _need(summary["schema_version"] == f"{spec['schema']}-report-1.0", "summary schema drift")
    _need(summary["status"] == spec["report_status"], f"summary status must remain {spec['stage']} NO-GO")
    _need(summary["source_config_sha256"] == _sha(config_path), "summary source config hash drift")
    for prefix in ("source_t0", "source_m2_2", "source_m2_3", "source_m2_4") + (("source_m2_5",) if spec["stage"] == "M2.6" else ()):
        for suffix in ("config_sha256", "summary_sha256"):
            key = f"{prefix}_{suffix}"
            _need(summary[key] == config[key], f"summary {key} drift")
    _compare_value(summary["authorization"], AUTHORIZATION, "summary.authorization")
    _compare_value(summary["public_export_policy"], PUBLIC_POLICY, "summary.public_export_policy")
    boundary = summary["claim_boundary"]
    _need(boundary == config["claim_boundary"], "summary claim boundary drift")
    _need(all(value is False for key, value in AUTHORIZATION.items() if key != "continue_matrix_free_preconditioner_research"),
          "authorization exceeds NO-GO boundary")
    for summary_field, count_key in (("metric_row_count", "metric"), ("reference_row_count", "reference"), ("matched_baseline_row_count", "baseline")):
        _need(summary[summary_field] == counts[count_key], f"summary {summary_field} drift")
    computed_aggregate = _aggregate(rows, bool(spec["requires_closure"]))
    _compare_aggregate(aggregate, computed_aggregate, bool(spec["requires_closure"]), "aggregate_rows.csv")
    summary_aggregate = { (r["method"], _int(r["model_seed"], "summary aggregate"), r["split"], r["projection_variant"], _int(r["projection_iterations"], "summary aggregate")): r for r in summary["aggregate"] }
    _need(len(summary_aggregate) == len(computed_aggregate), "summary aggregate count drift")
    for key, expected in computed_aggregate.items():
        _need(key in summary_aggregate, "summary aggregate identity drift")
        for field, value in expected.items():
            _close(summary_aggregate[key][field], float(value), f"summary.aggregate[{key!r}].{field}")
    decisions = _decisions(rows, spec)
    _compare_value(summary["decisions"], decisions, "summary.decisions")
    if spec["stage"] == "M2.6":
        for method, decision in decisions.items():
            diagnostic = decision["diagnostics"]["development"]
            _need(math.isclose(diagnostic["field_harm_rate"], 1 / 12, rel_tol=0, abs_tol=1e-15),
                  f"{method}: expected 8.33% development harm evidence")
            _need(diagnostic["worst_field_gain"] < -0.05, f"{method}: worst negative gain evidence missing")
            _need(decision["passed_m2_3_mechanism_gate"] is False, f"{method}: harm was incorrectly authorized")
    _need(len(baseline_aggregate) == counts["baseline_aggregate"], "baseline aggregate count drift")
    return decisions


def validate_packet(*, config_path: Path, output_dir: Path) -> dict[str, Any]:
    """Validate an M2.5 or M2.6 packet and return the independent NO-GO verdict."""
    config_path, output_dir = Path(config_path), Path(output_dir)
    _need(config_path.is_file() and output_dir.is_dir(), "packet path missing")
    config = _load_json(config_path)
    schema = config.get("schema_version", "")
    if schema.startswith("jacru-m2-5-"):
        spec = _packet("M2.5")
    elif schema.startswith("jacru-m2-6-"):
        spec = _packet("M2.6")
    else:
        raise ValidationError("unsupported JACRU M2.5/M2.6 config schema")
    _manifest(output_dir)
    _validate_config(config, config_path, spec)
    counts = _expected_counts(spec)
    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    _need(f"Status: `{spec['report_status']}`" in readme, "README status does not preserve NO-GO")
    reference = _read_csv(output_dir / "reference_rows.csv", REFERENCE, counts["reference"])
    baseline = _read_csv(output_dir / "matched_baseline_rows.csv", BASELINE, counts["baseline"])
    rows = _read_csv(output_dir / "metric_rows.csv", spec["metric_fields"], counts["metric"])
    aggregate = _read_csv(output_dir / "aggregate_rows.csv", spec["aggregate_fields"], counts["aggregate"])
    baseline_aggregate = _read_csv(output_dir / "matched_baseline_aggregate_rows.csv", BASELINE_AGGREGATE, counts["baseline_aggregate"])
    cases = _validate_references(reference)
    baselines = _validate_baselines(baseline, cases)
    _validate_metrics(rows, cases, baselines, spec)
    summary = _load_json(output_dir / "summary.json")
    decisions = _validate_summary(summary, config, config_path, spec, counts, rows, aggregate, baseline_aggregate)
    return {
        "status": spec["validated_status"], "stage": spec["stage"], "report_status": spec["report_status"],
        "metric_row_count": len(rows), "reference_row_count": len(reference),
        "matched_baseline_row_count": len(baseline), "aggregate_count": len(aggregate),
        "development_selection_count": sum(d["selection"] is not None for d in decisions.values()),
        "closure_ledger_available": bool(spec["requires_closure"]),
        "preconditioner_application_ledger_available": bool(spec["requires_closure"]),
        "authorization": dict(AUTHORIZATION),
    }


def validate_all_packets() -> dict[str, dict[str, Any]]:
    return {"m2_5": validate_packet(config_path=_packet("M2.5")["config"], output_dir=_packet("M2.5")["output"]),
            "m2_6": validate_packet(config_path=_packet("M2.6")["config"], output_dir=_packet("M2.6")["output"])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", choices=("all", "m2-5", "m2-6"), default="all")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.packet == "all":
        _need(args.config is None and args.output_dir is None, "custom paths require one packet")
        report = validate_all_packets()
    else:
        spec = _packet("M2.5" if args.packet == "m2-5" else "M2.6")
        report = validate_packet(config_path=args.config or spec["config"], output_dir=args.output_dir or spec["output"])
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
