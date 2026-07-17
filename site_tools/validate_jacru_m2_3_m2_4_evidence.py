#!/usr/bin/env python3
"""Independently validate the frozen JACRU M2.3 and M2.4 evidence packets.

This module intentionally uses only the frozen configs, public packet files,
and standard-library parsers. It does not import either experiment runner.
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
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "demo_t16_operator/results"
CONFIG_ROOT = ROOT / "demo_t16_operator/configs"

METHODS = ("jacru_m2", "pooled_cnn")
MODEL_SEEDS = (17, 29, 43)
SPLITS = ("development", "ood")
SPLIT_CASE_COUNTS = {"development": 12, "ood": 18}
BASELINE_METHODS = (
    "cgls_matched",
    "huber_pdhg_matched",
    "base_landweber_matched",
)

SOURCE_T0_CONFIG = (
    "demo_t16_operator/configs/jacru_m2_learned_residual_t0_v1.json"
)
SOURCE_T0_RESULTS = (
    "demo_t16_operator/results/jacru_m2_learned_residual_t0_public"
)
SOURCE_M2_2_CONFIG = (
    "demo_t16_operator/configs/"
    "jacru_m2_2_exact_nullspace_oracle_postopen_v1.json"
)
SOURCE_M2_2_RESULTS = (
    "demo_t16_operator/results/"
    "jacru_m2_2_exact_nullspace_oracle_postopen_public"
)
SOURCE_M2_3_CONFIG = (
    "demo_t16_operator/configs/"
    "jacru_m2_3_matrix_free_projection_postopen_v1.json"
)
SOURCE_M2_3_RESULTS = (
    "demo_t16_operator/results/"
    "jacru_m2_3_matrix_free_projection_postopen_public"
)

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
PROJECTION_METRIC_FIELDS = COMMON_METRIC_FIELDS + (
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
    "matched_base_landweber_field_relative_l2",
    "field_gain_to_best_matched_classical",
    "h1_gain_to_best_matched_classical",
    "reprojection_ratio_to_matched_cgls",
    "visible_correction_fraction",
    "system_residual_fraction",
    "exact_projection_approximation_error",
    "exact_oracle_error_reduction_retention",
    "oracle_reduction_defined",
    "base_anchor_field_relative_l2",
    "exact_oracle_field_relative_l2",
    "raw_learned_field_relative_l2",
    "exact_oracle_rank",
    "exact_oracle_nullity_lower_bound",
    "field_harm_to_best_matched_classical",
    "converged",
    "breakdown",
    "projection_diagnostic_forward_calls",
    "dense_oracle_used_by_algorithm",
)
M2_4_EXTRA_METRIC_FIELDS = (
    "projection_target_mode",
    "exact_oracle_internal_projection_residual",
)
REFERENCE_FIELDS = COMMON_METRIC_FIELDS + ("reference_kind",)
BASELINE_FIELDS = COMMON_METRIC_FIELDS + (
    "matched_step",
    "total_calls",
    "baseline_kind",
    "dc_step_size",
    "operator_norm_squared_bound",
    "projection_iterations",
    "paired_call_budget",
    "matched_step_internal_offset",
)
AGGREGATE_FIELDS = (
    "method",
    "model_seed",
    "split",
    "projection_variant",
    "projection_iterations",
    "case_count",
    "paired_call_budget",
    "damping_fraction",
    "field_relative_l2_mean",
    "h1_seminorm_relative_error_mean",
    "field_gain_to_best_matched_classical_mean",
    "h1_gain_to_best_matched_classical_mean",
    "reprojection_ratio_to_matched_cgls_mean",
    "visible_correction_fraction_mean",
    "visible_correction_fraction_maximum",
    "system_residual_fraction_mean",
    "exact_projection_approximation_error_mean",
    "exact_oracle_error_reduction_retention_mean",
    "oracle_reduction_defined_rate",
    "field_harm_rate",
    "worst_field_gain_to_best_matched_classical",
    "breakdown_rate",
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

CHECKSUM_PAYLOADS = {
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
PUBLIC_EXPORT_POLICY = {
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
}
MATCHED_BUDGET = {
    "learned_feature_preparation_forward_calls": 13,
    "learned_feature_preparation_adjoint_calls": 13,
    "projection_forward_calls_formula": "K+1",
    "projection_adjoint_calls_formula": "K",
    "matched_classical_pair_iterations_formula": "14+K",
    "classical_methods": list(BASELINE_METHODS),
    "dense_norm_setup_excluded_and_reported": True,
    "grouped_adjoint_is_not_equal_flop_to_pooled_adjoint": True,
}
SELECTION_RULE = {
    "uses_ood": False,
    "maximum_mean_visible_correction_fraction": 0.1,
    "maximum_worst_case_visible_correction_fraction": 0.25,
    "maximum_mean_reprojection_ratio_to_matched_cgls": 1.1,
    "rank_by": "maximum_development_mean_field_gain_to_best_matched_classical",
    "tie_breakers": [
        "fewer_projection_iterations",
        "smaller_damping_fraction",
        "variant_name",
    ],
}
DECISION_GATES = {
    "development_field_gain_to_best_matched_classical_minimum_fraction": 0.05,
    "development_h1_gain_to_best_matched_classical_minimum_fraction": 0.03,
    "development_exact_oracle_error_reduction_retention_minimum_fraction": 0.5,
    "development_reprojection_ratio_to_matched_cgls_maximum": 1.1,
    "development_visible_correction_fraction_mean_maximum": 0.1,
    "ood_field_gain_to_best_matched_classical_minimum_fraction": 0.02,
    "ood_h1_gain_to_best_matched_classical_minimum_fraction": 0.0,
    "ood_exact_oracle_error_reduction_retention_minimum_fraction": 0.5,
    "ood_reprojection_ratio_to_matched_cgls_maximum": 1.15,
    "ood_visible_correction_fraction_mean_maximum": 0.15,
    "field_harm_threshold_fraction": 0.01,
    "field_harm_rate_maximum": 0.05,
    "worst_field_gain_minimum_fraction": -0.05,
    "require_all_model_seed_mean_field_gains_positive": True,
    "maximum_breakdown_rate": 0.0,
}
ORACLE_AUDIT = {
    "enabled": True,
    "maximum_grid_voxels": 1728,
    "assembly_batch_size": 256,
    "dtype": "float64",
    "rank_relative_tolerance": 1e-10,
    "rank_absolute_tolerance": 0.0,
    "excluded_from_algorithm_and_efficiency_claims": True,
}

M2_3_CONFIG_FIELDS = {
    "schema_version",
    "status",
    "frozen_date",
    "evidence_level",
    "source_t0_config",
    "source_t0_config_sha256",
    "source_t0_results",
    "source_t0_summary_sha256",
    "source_m2_2_config",
    "source_m2_2_config_sha256",
    "source_m2_2_results",
    "source_m2_2_summary_sha256",
    "methods",
    "reference_anchor",
    "projection",
    "retrospective_toy_oracle_audit",
    "matched_budget",
    "development_selection_rule",
    "decision_gates",
    "claim_boundary",
}
M2_4_CONFIG_FIELDS = M2_3_CONFIG_FIELDS | {
    "report_schema_version",
    "figure_title",
    "readme_title",
    "report_status",
    "source_m2_3_config",
    "source_m2_3_config_sha256",
    "source_m2_3_results",
    "source_m2_3_summary_sha256",
    "reason_for_new_target",
}
SUMMARY_FIELDS = {
    "schema_version",
    "status",
    "evidence_level",
    "source_config_sha256",
    "source_t0_config_sha256",
    "source_t0_summary_sha256",
    "source_m2_2_config_sha256",
    "source_m2_2_summary_sha256",
    "device",
    "elapsed_seconds",
    "metric_row_count",
    "reference_row_count",
    "matched_baseline_row_count",
    "training_runs",
    "operator_norm_setup",
    "retrospective_dense_oracle_setup",
    "decisions",
    "aggregate",
    "matched_baseline_aggregate",
    "authorization",
    "claim_boundary",
    "public_export_policy",
}


@dataclass(frozen=True)
class PacketSpec:
    stage: str
    config_schema: str
    report_schema: str
    config_status: str
    report_status: str
    validated_status: str
    evidence_level: str
    config_path: Path
    output_dir: Path
    snapshots: tuple[int, ...]
    variants: tuple[tuple[str, float], ...]
    target_mode: str | None
    exact_oracle_reference_kind: str
    reference_kinds: tuple[str, ...]
    config_fields: frozenset[str]
    claim_boundary: dict[str, Any]

    @property
    def metric_fields(self) -> tuple[str, ...]:
        if self.target_mode is None:
            return PROJECTION_METRIC_FIELDS
        return PROJECTION_METRIC_FIELDS + M2_4_EXTRA_METRIC_FIELDS

    @property
    def expected_metric_rows(self) -> int:
        return (
            len(METHODS)
            * len(MODEL_SEEDS)
            * len(self.variants)
            * len(self.snapshots)
            * sum(SPLIT_CASE_COUNTS.values())
        )

    @property
    def expected_reference_rows(self) -> int:
        learned_grids = len(self.reference_kinds) - 1
        return sum(SPLIT_CASE_COUNTS.values()) * (
            1 + learned_grids * len(METHODS) * len(MODEL_SEEDS)
        )

    @property
    def expected_baseline_rows(self) -> int:
        return (
            len(BASELINE_METHODS)
            * len(self.snapshots)
            * sum(SPLIT_CASE_COUNTS.values())
        )

    @property
    def expected_aggregates(self) -> int:
        return (
            len(METHODS)
            * len(MODEL_SEEDS)
            * len(SPLITS)
            * len(self.variants)
            * len(self.snapshots)
        )

    @property
    def expected_baseline_aggregates(self) -> int:
        return len(BASELINE_METHODS) * len(SPLITS) * len(self.snapshots)


M2_3_SPEC = PacketSpec(
    stage="M2.3",
    config_schema="jacru-m2-3-matrix-free-projection-postopen-config-1.0",
    report_schema="jacru-m2-3-matrix-free-projection-postopen-report-1.0",
    config_status="FROZEN_BEFORE_FIRST_MATRIX_FREE_PROJECTION_EXECUTION",
    report_status="M2_3_POSTOPEN_MATRIX_FREE_PROJECTION_NO_GO",
    validated_status="VALIDATED_M2_3_MATRIX_FREE_PROJECTION_NO_GO",
    evidence_level="E1_OPENED_T0_MATRIX_FREE_MECHANISM_DIAGNOSTIC_NO_FRESH",
    config_path=CONFIG_ROOT / "jacru_m2_3_matrix_free_projection_postopen_v1.json",
    output_dir=RESULTS_ROOT / "jacru_m2_3_matrix_free_projection_postopen_public",
    snapshots=(0, 1, 2, 4, 8, 12),
    variants=(
        ("cg_undamped", 0.0),
        ("cg_damped_1e-6", 1e-6),
        ("cg_damped_1e-4", 1e-4),
    ),
    target_mode=None,
    exact_oracle_reference_kind="retrospective_dense_oracle_base_anchor",
    reference_kinds=(
        "base_anchor",
        "raw_learned",
        "retrospective_dense_oracle_base_anchor",
    ),
    config_fields=frozenset(M2_3_CONFIG_FIELDS),
    claim_boundary={
        "is_postopen_mechanism_diagnostic": True,
        "uses_only_opened_synthetic_t0_for_selection_and_scoring": True,
        "is_confirmatory_or_final": False,
        "is_experimental_reconstruction": False,
        "is_cfd_validation": False,
        "is_real_bost_generalization": False,
        "finite_cg_is_exact_nullspace_projection": False,
        "approximate_inverse_kernel_equals_true_optical_kernel": False,
        "dense_toy_oracle_is_part_of_deployable_algorithm": False,
        "opens_fresh_or_final": False,
        "may_only_authorize_a_new_preregistered_fresh_gate": True,
    },
)
M2_4_SPEC = PacketSpec(
    stage="M2.4",
    config_schema="jacru-m2-4-affine-observation-projection-postopen-config-1.0",
    report_schema="jacru-m2-4-affine-observation-projection-postopen-report-1.0",
    config_status="FROZEN_BEFORE_FIRST_AFFINE_OBSERVATION_PROJECTION_EXECUTION",
    report_status="M2_4_POSTOPEN_AFFINE_CG_NO_GO",
    validated_status="VALIDATED_M2_4_AFFINE_OBSERVATION_PROJECTION_NO_GO",
    evidence_level="E1_OPENED_T0_AFFINE_OBSERVATION_CG_DIAGNOSTIC_NO_FRESH",
    config_path=CONFIG_ROOT / "jacru_m2_4_affine_observation_projection_postopen_v1.json",
    output_dir=(
        RESULTS_ROOT / "jacru_m2_4_affine_observation_projection_postopen_public"
    ),
    snapshots=(0, 1, 2, 4, 8, 12, 20, 32),
    variants=(
        ("affine_cg_undamped", 0.0),
        ("affine_cg_damped_1e-6", 1e-6),
        ("affine_cg_damped_1e-4", 1e-4),
    ),
    target_mode="affine_observation",
    exact_oracle_reference_kind=(
        "retrospective_dense_oracle_affine_observation"
    ),
    reference_kinds=(
        "base_anchor",
        "raw_learned",
        "retrospective_dense_oracle_base_anchor",
        "retrospective_dense_oracle_affine_observation",
    ),
    config_fields=frozenset(M2_4_CONFIG_FIELDS),
    claim_boundary={
        "is_postopen_mechanism_diagnostic": True,
        "uses_only_opened_synthetic_t0_for_selection_and_scoring": True,
        "is_confirmatory_or_final": False,
        "is_experimental_reconstruction": False,
        "is_cfd_validation": False,
        "is_real_bost_generalization": False,
        "finite_cg_is_exact_affine_projection": False,
        "measured_data_equals_clean_optical_forward": False,
        "dense_toy_oracle_is_part_of_deployable_algorithm": False,
        "opens_fresh_or_final": False,
        "may_only_authorize_a_new_preregistered_fresh_gate": True,
    },
)
SPECS = {M2_3_SPEC.config_schema: M2_3_SPEC, M2_4_SPEC.config_schema: M2_4_SPEC}


class ValidationError(RuntimeError):
    """Raised when a public evidence packet violates its frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ValidationError(f"cannot hash {path}: {error}") from error


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot read JSON object {path}: {error}") from error
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


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
    elif isinstance(expected, (list, tuple)):
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
        observed = _csv_int(actual, path=path)
        _require(observed == expected, f"{path}: integer mismatch")
        return
    observed = _csv_float(actual, path=path)
    assert observed is not None
    _require(
        math.isclose(observed, expected, rel_tol=5e-11, abs_tol=5e-12),
        f"{path}: numeric mismatch ({observed!r} != {expected!r})",
    )


def _read_csv(
    path: Path,
    fields: tuple[str, ...],
    *,
    expected_rows: int | None = None,
) -> list[dict[str, str]]:
    try:
        text = path.read_text(encoding="utf-8")
        physical_lines = text.splitlines()
        _require(all(physical_lines), f"{path.name}: blank physical line")
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            _require(
                tuple(reader.fieldnames or ()) == fields,
                f"{path.name}: columns differ from the frozen schema",
            )
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValidationError(f"cannot read CSV {path}: {error}") from error
    for index, row in enumerate(rows):
        _require(None not in row, f"{path.name}[{index}]: extra CSV value")
        _require(
            all(value is not None for value in row.values()),
            f"{path.name}[{index}]: missing CSV value",
        )
    if expected_rows is not None:
        row_label = path.stem.removesuffix("_rows").replace("_", " ") + " rows"
        _require(
            len(rows) == expected_rows,
            f"expected {expected_rows} {row_label}",
        )
        _require(
            len(physical_lines) == expected_rows + 1,
            f"{path.name}: physical line count drift",
        )
    return rows


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

    try:
        directory_entries = {path.name for path in output_dir.iterdir()}
    except OSError as error:
        raise ValidationError(f"cannot list evidence directory: {error}") from error
    _require(
        directory_entries == CHECKSUM_PAYLOADS | {"checksums.sha256"},
        "public packet contains an unmanifested or missing file",
    )
    for filename, expected in entries.items():
        path = output_dir / filename
        _require(path.is_file() and not path.is_symlink(), f"invalid payload: {filename}")
        _require(_sha256(path) == expected, f"checksum mismatch: {filename}")


def _manifest_digest(directory: Path, filename: str, *, label: str) -> str:
    manifest = directory / "checksums.sha256"
    _require(manifest.is_file(), f"{label} checksum manifest is missing")
    matches: list[str] = []
    try:
        lines = manifest.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValidationError(f"cannot read {label} checksum manifest: {error}") from error
    pattern = re.compile(rf"([0-9a-f]{{64}})  {re.escape(filename)}")
    for line in lines:
        match = pattern.fullmatch(line)
        if match:
            matches.append(match.group(1))
    _require(len(matches) == 1, f"{label} {filename} checksum is missing or duplicated")
    path = directory / filename
    _require(path.is_file(), f"{label} {filename} is missing")
    _require(_sha256(path) == matches[0], f"{label} {filename} checksum mismatch")
    return matches[0]


def _repo_path(relative_path: Any, *, label: str, directory: bool = False) -> Path:
    _require(isinstance(relative_path, str), f"{label}: expected repository-relative path")
    path = (ROOT / relative_path).resolve()
    _require(path.is_relative_to(ROOT), f"{label}: path escapes repository")
    if directory:
        _require(path.is_dir(), f"{label}: directory is missing")
    else:
        _require(path.is_file(), f"{label}: file is missing")
    return path


def _validate_source_pair(config: dict[str, Any], prefix: str) -> tuple[Path, Path]:
    config_path = _repo_path(config[f"{prefix}_config"], label=f"config.{prefix}_config")
    _require(
        _sha256(config_path) == config[f"{prefix}_config_sha256"],
        f"{prefix} config hash drift",
    )
    results_dir = _repo_path(
        config[f"{prefix}_results"],
        label=f"config.{prefix}_results",
        directory=True,
    )
    summary_path = results_dir / "summary.json"
    _require(summary_path.is_file(), f"{prefix} summary is missing")
    _require(
        _sha256(summary_path) == config[f"{prefix}_summary_sha256"],
        f"{prefix} summary hash drift",
    )
    _manifest_digest(results_dir, "summary.json", label=prefix)
    return config_path, results_dir


def _expected_projection(spec: PacketSpec) -> dict[str, Any]:
    variants: list[dict[str, Any]] = []
    for name, damping in spec.variants:
        variant: dict[str, Any] = {
            "name": name,
            "damping_fraction_of_operator_norm_squared_bound": damping,
        }
        if spec.target_mode is not None:
            variant["target_mode"] = spec.target_mode
        variants.append(variant)
    return {
        "snapshot_iterations": list(spec.snapshots),
        "denominator_floor": 1e-30,
        "preconditioner": "identity",
        "variants": variants,
    }


def _validate_config(
    config: dict[str, Any],
    config_path: Path,
    spec: PacketSpec,
) -> tuple[dict[str, Any], Path]:
    _require(set(config) == set(spec.config_fields), "config top-level schema drift")
    _require(config["schema_version"] == spec.config_schema, "config schema drift")
    _require(config["status"] == spec.config_status, "config is not frozen")
    _require(config["frozen_date"] == "2026-07-17", "config frozen date drift")
    _require(config["evidence_level"] == spec.evidence_level, "evidence level drift")
    _compare(config["methods"], list(METHODS), path="config.methods")

    if spec.stage == "M2.3":
        reference_anchor = {
            "method": "prepared_cgls_base",
            "iterations": 12,
            "reuse_feature_preparation_without_second_cgls": True,
        }
    else:
        reference_anchor = {
            "method": "prepared_cgls_base",
            "iterations": 12,
            "used_to_generate_network_features_but_not_as_projection_measurement_target": True,
        }
        _require(
            config["report_schema_version"] == spec.report_schema,
            "config report schema drift",
        )
        _require(
            config["figure_title"] == "M2.4 affine measured-data CG projection",
            "config figure title drift",
        )
        _require(
            config["readme_title"]
            == "JACRU-M2.4 affine measured-data projection diagnostic",
            "config README title drift",
        )
        _compare(
            config["report_status"],
            {
                "success": "M2_4_POSTOPEN_AFFINE_CG_MECHANISM_FOUND_NOT_CONFIRMATORY",
                "no_go": spec.report_status,
            },
            path="config.report_status",
        )
        _require(
            config["reason_for_new_target"]
            == (
                "M2.3 reduced A(x_net-x_base) but remained locked to the CGLS-12 "
                "anchor residual. M2.4 instead solves against A x_net-y so the "
                "finite-step output approaches the measured-data affine set."
            ),
            "config target rationale drift",
        )
    _compare(config["reference_anchor"], reference_anchor, path="config.reference_anchor")
    _compare(config["projection"], _expected_projection(spec), path="config.projection")
    _compare(
        config["retrospective_toy_oracle_audit"],
        ORACLE_AUDIT,
        path="config.retrospective_toy_oracle_audit",
    )
    _compare(config["matched_budget"], MATCHED_BUDGET, path="config.matched_budget")
    _compare(
        config["development_selection_rule"],
        SELECTION_RULE,
        path="config.development_selection_rule",
    )
    _compare(config["decision_gates"], DECISION_GATES, path="config.decision_gates")
    _compare(config["claim_boundary"], spec.claim_boundary, path="config.claim_boundary")

    _require(config["source_t0_config"] == SOURCE_T0_CONFIG, "source T0 config path drift")
    _require(config["source_t0_results"] == SOURCE_T0_RESULTS, "source T0 results path drift")
    _require(
        config["source_m2_2_config"] == SOURCE_M2_2_CONFIG,
        "source M2.2 config path drift",
    )
    _require(
        config["source_m2_2_results"] == SOURCE_M2_2_RESULTS,
        "source M2.2 results path drift",
    )
    source_t0_config_path, source_t0_results = _validate_source_pair(
        config, "source_t0"
    )
    _validate_source_pair(config, "source_m2_2")
    if spec.stage == "M2.4":
        _require(
            config["source_m2_3_config"] == SOURCE_M2_3_CONFIG,
            "source M2.3 config path drift",
        )
        _require(
            config["source_m2_3_results"] == SOURCE_M2_3_RESULTS,
            "source M2.3 results path drift",
        )
        _validate_source_pair(config, "source_m2_3")

    source_config = _load_json(source_t0_config_path)
    _compare(
        source_config["training"]["model_seeds"],
        list(MODEL_SEEDS),
        path="source_t0_config.training.model_seeds",
    )
    physical_budget = source_config["physical_budget"]
    expected_source_budget = {
        "cgls_base_iterations": 12,
        "learned_feature_forward_calls": 1,
        "learned_feature_grouped_adjoint_calls": 1,
        "classical_comparator_iterations": 13,
        "dense_norm_batch_size": 256,
        "dense_norm_safety_factor": 1.01,
        "grouped_adjoint_is_not_equal_flop_to_pooled_adjoint": True,
    }
    _compare(physical_budget, expected_source_budget, path="source_t0_config.physical_budget")
    _manifest_digest(source_t0_results, "metric_rows.csv", label="source_t0")
    _require(config_path.is_file(), "diagnostic config is missing")
    return source_config, source_t0_results


def _validate_readme(output_dir: Path, spec: PacketSpec) -> None:
    try:
        text = (output_dir / "README.md").read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValidationError(f"cannot read packet README: {error}") from error
    expected_title = (
        "# JACRU-M2.3 matrix-free projection diagnostic"
        if spec.stage == "M2.3"
        else "# JACRU-M2.4 affine measured-data projection diagnostic"
    )
    _require(text.startswith(expected_title + "\n"), "README title drift")
    _require(
        f"Status: `{spec.report_status}`" in text,
        "README status does not preserve NO-GO",
    )


def _source_case_catalog(
    source_rows: list[dict[str, str]],
    source_config: dict[str, Any],
) -> tuple[
    dict[str, tuple[str, str, int]],
    dict[tuple[str, int, str], dict[str, str]],
]:
    catalog: dict[str, tuple[str, str, int]] = {}
    learned: dict[tuple[str, int, str], dict[str, str]] = {}
    for index, row in enumerate(source_rows):
        model_seed = _csv_int(row["model_seed"], path=f"source_t0[{index}].model_seed")
        case_id = row["case_id"]
        if row["method"] == "cgls_13" and model_seed == -1:
            _require(case_id not in catalog, f"duplicate source T0 case: {case_id}")
            catalog[case_id] = (
                row["split"],
                row["family"],
                _csv_int(row["base_seed"], path=f"source_t0[{index}].base_seed"),
            )
        if row["method"] in METHODS and model_seed in MODEL_SEEDS:
            key = (row["method"], model_seed, case_id)
            _require(key not in learned, f"duplicate source T0 learned row: {key}")
            learned[key] = row

    _require(
        len(catalog) == sum(SPLIT_CASE_COUNTS.values()),
        "source T0 case count drift",
    )
    for split, expected_count in SPLIT_CASE_COUNTS.items():
        observed = [value for value in catalog.values() if value[0] == split]
        _require(len(observed) == expected_count, f"source T0 {split} case count drift")
        split_config = source_config["splits"][split]
        expected_grid = {
            (split, family, int(seed))
            for seed in split_config["base_seeds"]
            for family in split_config["families"]
        }
        _require(set(observed) == expected_grid, f"source T0 {split} grid drift")

    expected_learned = {
        (method, model_seed, case_id)
        for method in METHODS
        for model_seed in MODEL_SEEDS
        for case_id in catalog
    }
    _require(set(learned) == expected_learned, "source T0 learned row grid drift")
    return catalog, learned


ERROR_FIELDS = (
    "field_relative_l2",
    "field_rmse",
    "field_nrmse_dynamic_range",
    "field_mean_bias",
    "h1_seminorm_relative_error",
    "measured_reprojection_relative_l2",
    "clean_reprojection_relative_l2",
)


def _assert_case_metadata(
    row: dict[str, str],
    catalog: dict[str, tuple[str, str, int]],
    *,
    path: str,
) -> None:
    case_id = row["case_id"]
    _require(
        re.fullmatch(r"[0-9a-f]{20}", case_id) is not None,
        f"{path}.case_id: malformed identifier",
    )
    _require(case_id in catalog, f"{path}.case_id: unknown source case")
    split, family, base_seed = catalog[case_id]
    _require(row["split"] == split, f"{path}.split: source identity drift")
    _require(row["family"] == family, f"{path}.family: source identity drift")
    _require(
        _csv_int(row["base_seed"], path=f"{path}.base_seed") == base_seed,
        f"{path}.base_seed: source identity drift",
    )


def _validate_common_numbers(
    row: dict[str, str],
    *,
    path: str,
    empty_gate: bool,
    allow_nonempty_correction: bool = False,
) -> None:
    for field in ERROR_FIELDS:
        value = _csv_float(row[field], path=f"{path}.{field}")
        assert value is not None
        if field != "field_mean_bias":
            _require(value >= 0.0, f"{path}.{field}: expected nonnegative value")
    if empty_gate:
        _require(row["gate"] == "", f"{path}.gate: expected empty reference value")
        if row["correction_rms"] == "":
            pass
        elif allow_nonempty_correction:
            correction = _csv_float(
                row["correction_rms"], path=f"{path}.correction_rms"
            )
            assert correction is not None
            _require(
                correction >= 0.0,
                f"{path}.correction_rms: expected nonnegative value",
            )
        else:
            raise ValidationError(
                f"{path}.correction_rms: expected empty reference value"
            )
    else:
        _csv_float(row["gate"], path=f"{path}.gate")
        correction = _csv_float(row["correction_rms"], path=f"{path}.correction_rms")
        assert correction is not None
        _require(correction >= 0.0, f"{path}.correction_rms: expected nonnegative value")
    inference = _csv_float(
        row["neural_inference_seconds"], path=f"{path}.neural_inference_seconds"
    )
    assert inference is not None
    _require(inference >= 0.0, f"{path}.neural_inference_seconds: negative timing")


def _validate_reference_rows(
    rows: list[dict[str, str]],
    catalog: dict[str, tuple[str, str, int]],
    source_learned: dict[tuple[str, int, str], dict[str, str]],
    spec: PacketSpec,
) -> dict[tuple[str, str, int, str], dict[str, str]]:
    lookup: dict[tuple[str, str, int, str], dict[str, str]] = {}
    for index, row in enumerate(rows):
        path = f"reference_rows[{index}]"
        _assert_case_metadata(row, catalog, path=path)
        _validate_common_numbers(row, path=path, empty_gate=True)
        model_seed = _csv_int(row["model_seed"], path=f"{path}.model_seed")
        kind = row["reference_kind"]
        _require(kind in spec.reference_kinds, f"{path}.reference_kind: unexpected kind")
        key = (kind, row["method"], model_seed, row["case_id"])
        _require(key not in lookup, f"duplicate reference row: {key}")
        lookup[key] = row

        forward_calls = _csv_int(
            row["optimization_forward_calls"],
            path=f"{path}.optimization_forward_calls",
        )
        adjoint_calls = _csv_int(
            row["optimization_adjoint_calls"],
            path=f"{path}.optimization_adjoint_calls",
        )
        grouped_calls = _csv_int(
            row["grouped_adjoint_calls"], path=f"{path}.grouped_adjoint_calls"
        )
        evaluation_calls = _csv_int(
            row["evaluation_forward_calls"], path=f"{path}.evaluation_forward_calls"
        )
        _require(evaluation_calls == 1, f"{path}: evaluation budget drift")
        _require(
            math.isclose(
                float(row["neural_inference_seconds"]),
                0.0,
                abs_tol=0.0,
            ),
            f"{path}: reference timing must remain zero",
        )

        if kind == "base_anchor":
            _require(row["method"] == "prepared_cgls_base_12", "base anchor method drift")
            _require(model_seed == -1, "base anchor model seed drift")
            _require(
                (forward_calls, adjoint_calls, grouped_calls) == (12, 12, 0),
                "base anchor F/A budget drift",
            )
        else:
            _require(row["method"] in METHODS, f"{path}: learned method drift")
            _require(model_seed in MODEL_SEEDS, f"{path}: learned seed drift")
            _require(
                (forward_calls, adjoint_calls, grouped_calls) == (13, 13, 1),
                f"{path}: learned reference F/A budget drift",
            )
            if kind == "raw_learned":
                source = source_learned[(row["method"], model_seed, row["case_id"])]
                for field in ERROR_FIELDS:
                    expected = _csv_float(source[field], path=f"source_t0.{field}")
                    assert expected is not None
                    _compare_csv_number(row[field], expected, path=f"{path}.{field}")
                for field in (
                    "optimization_forward_calls",
                    "optimization_adjoint_calls",
                    "grouped_adjoint_calls",
                    "evaluation_forward_calls",
                ):
                    _require(
                        row[field] == source[field],
                        f"{path}.{field}: source T0 reproduction drift",
                    )

    expected_keys = {
        ("base_anchor", "prepared_cgls_base_12", -1, case_id)
        for case_id in catalog
    }
    for kind in spec.reference_kinds:
        if kind == "base_anchor":
            continue
        expected_keys.update(
            (kind, method, model_seed, case_id)
            for method in METHODS
            for model_seed in MODEL_SEEDS
            for case_id in catalog
        )
    _require(set(lookup) == expected_keys, "reference row identity grid drift")
    return lookup


def _validate_baseline_rows(
    rows: list[dict[str, str]],
    catalog: dict[str, tuple[str, str, int]],
    spec: PacketSpec,
) -> tuple[
    dict[tuple[str, str, int], dict[str, str]],
    dict[tuple[str, int], float],
]:
    lookup: dict[tuple[str, str, int], dict[str, str]] = {}
    bounds: dict[tuple[str, int], float] = {}
    for index, row in enumerate(rows):
        path = f"matched_baseline_rows[{index}]"
        _assert_case_metadata(row, catalog, path=path)
        _validate_common_numbers(
            row,
            path=path,
            empty_gate=True,
            allow_nonempty_correction=True,
        )
        method = row["method"]
        _require(method in BASELINE_METHODS, f"{path}.method: unexpected baseline")
        _require(row["baseline_kind"] == method, f"{path}.baseline_kind: method drift")
        _require(
            _csv_int(row["model_seed"], path=f"{path}.model_seed") == -1,
            f"{path}.model_seed: expected -1",
        )
        projection_iterations = _csv_int(
            row["projection_iterations"], path=f"{path}.projection_iterations"
        )
        _require(
            projection_iterations in spec.snapshots,
            f"{path}.projection_iterations: frozen grid drift",
        )
        key = (method, row["case_id"], projection_iterations)
        _require(key not in lookup, f"duplicate matched baseline row: {key}")
        lookup[key] = row

        paired_budget = 14 + projection_iterations
        matched_step = 1 + projection_iterations
        for field in ("optimization_forward_calls", "optimization_adjoint_calls"):
            _require(
                _csv_int(row[field], path=f"{path}.{field}") == paired_budget,
                f"matched baseline {field} budget drift",
            )
        _require(
            _csv_int(row["grouped_adjoint_calls"], path=f"{path}.grouped_adjoint_calls")
            == 0,
            "matched baseline grouped-adjoint budget drift",
        )
        _require(
            _csv_int(row["evaluation_forward_calls"], path=f"{path}.evaluation_forward_calls")
            == 1,
            "matched baseline evaluation budget drift",
        )
        for field, expected in (
            ("matched_step", matched_step),
            ("matched_step_internal_offset", matched_step),
            ("total_calls", paired_budget),
            ("paired_call_budget", paired_budget),
        ):
            _require(
                _csv_int(row[field], path=f"{path}.{field}") == expected,
                f"matched baseline {field} budget drift",
            )

        operator_bound = _csv_float(
            row["operator_norm_squared_bound"],
            path=f"{path}.operator_norm_squared_bound",
        )
        assert operator_bound is not None
        _require(operator_bound > 0.0, f"{path}: nonpositive operator bound")
        bound_key = (row["case_id"], projection_iterations)
        if bound_key in bounds:
            _require(
                math.isclose(bounds[bound_key], operator_bound, rel_tol=0.0, abs_tol=0.0),
                f"{path}: operator bound differs across matched methods",
            )
        else:
            bounds[bound_key] = operator_bound
        if method == "base_landweber_matched":
            step_size = _csv_float(row["dc_step_size"], path=f"{path}.dc_step_size")
            assert step_size is not None
            _require(step_size > 0.0, f"{path}: invalid Landweber step size")
        else:
            _require(row["dc_step_size"] == "", f"{path}: unexpected baseline step size")

    expected_keys = {
        (method, case_id, projection_iterations)
        for method in BASELINE_METHODS
        for case_id in catalog
        for projection_iterations in spec.snapshots
    }
    _require(set(lookup) == expected_keys, "matched baseline identity grid drift")
    expected_bounds = {
        (case_id, projection_iterations)
        for case_id in catalog
        for projection_iterations in spec.snapshots
    }
    _require(set(bounds) == expected_bounds, "matched baseline operator-bound grid drift")
    return lookup, bounds


def _row_float(row: dict[str, str], field: str, *, path: str) -> float:
    value = _csv_float(row[field], path=f"{path}.{field}")
    assert value is not None
    return value


def _validate_metric_rows(
    rows: list[dict[str, str]],
    catalog: dict[str, tuple[str, str, int]],
    references: dict[tuple[str, str, int, str], dict[str, str]],
    baselines: dict[tuple[str, str, int], dict[str, str]],
    operator_bounds: dict[tuple[str, int], float],
    spec: PacketSpec,
) -> dict[tuple[str, int, str, str, int], dict[str, str]]:
    lookup: dict[tuple[str, int, str, str, int], dict[str, str]] = {}
    variant_damping = dict(spec.variants)
    harm_threshold = float(DECISION_GATES["field_harm_threshold_fraction"])

    for index, row in enumerate(rows):
        path = f"metric_rows[{index}]"
        _assert_case_metadata(row, catalog, path=path)
        _validate_common_numbers(row, path=path, empty_gate=False)
        method = row["method"]
        model_seed = _csv_int(row["model_seed"], path=f"{path}.model_seed")
        variant = row["projection_variant"]
        projection_iterations = _csv_int(
            row["projection_iterations"], path=f"{path}.projection_iterations"
        )
        _require(method in METHODS, f"{path}.method: frozen method grid drift")
        _require(model_seed in MODEL_SEEDS, f"{path}.model_seed: frozen seed grid drift")
        _require(variant in variant_damping, f"{path}.projection_variant: frozen variant drift")
        _require(
            projection_iterations in spec.snapshots,
            f"{path}.projection_iterations: frozen snapshot drift",
        )
        if spec.target_mode is not None:
            _require(
                row["projection_target_mode"] == spec.target_mode,
                f"{path}.projection_target_mode: target mode drift",
            )
            residual = _row_float(
                row, "exact_oracle_internal_projection_residual", path=path
            )
            _require(
                residual <= 1e-10,
                f"{path}.exact_oracle_internal_projection_residual: oracle residual drift",
            )
        key = (method, model_seed, row["case_id"], variant, projection_iterations)
        _require(key not in lookup, f"duplicate metric row: {key}")
        lookup[key] = row

        expected_projection_forward = projection_iterations + 1
        expected_projection_adjoint = projection_iterations
        expected_paired_budget = projection_iterations + 14
        budget_expectations = {
            "projection_forward_calls": expected_projection_forward,
            "projection_adjoint_calls": expected_projection_adjoint,
            "optimization_forward_calls": 13 + expected_projection_forward,
            "optimization_adjoint_calls": 13 + expected_projection_adjoint,
            "paired_call_budget": expected_paired_budget,
            "projection_diagnostic_forward_calls": 1,
            "grouped_adjoint_calls": 1,
            "evaluation_forward_calls": 1,
        }
        for field, expected in budget_expectations.items():
            _require(
                _csv_int(row[field], path=f"{path}.{field}") == expected,
                f"metric {field} budget formula drift",
            )
        _require(row["preconditioner"] == "identity", f"{path}.preconditioner: drift")
        damping = _row_float(row, "damping_fraction", path=path)
        _require(
            math.isclose(damping, variant_damping[variant], rel_tol=0.0, abs_tol=0.0),
            f"{path}.damping_fraction: variant drift",
        )
        damping_absolute = _row_float(row, "damping_absolute", path=path)
        expected_damping_absolute = damping * operator_bounds[
            (row["case_id"], projection_iterations)
        ]
        _require(
            math.isclose(
                damping_absolute,
                expected_damping_absolute,
                rel_tol=5e-11,
                abs_tol=5e-12,
            ),
            f"{path}.damping_absolute: operator-bound formula drift",
        )
        for flag in (
            "oracle_reduction_defined",
            "field_harm_to_best_matched_classical",
            "converged",
            "breakdown",
        ):
            _require(
                _csv_int(row[flag], path=f"{path}.{flag}") in (0, 1),
                f"{path}.{flag}: expected binary integer",
            )
        _require(
            row["dense_oracle_used_by_algorithm"] == "False",
            f"{path}.dense_oracle_used_by_algorithm: budget boundary crossed",
        )
        _require(
            _csv_int(row["exact_oracle_rank"], path=f"{path}.exact_oracle_rank") == 150,
            f"{path}.exact_oracle_rank: drift",
        )
        _require(
            _csv_int(
                row["exact_oracle_nullity_lower_bound"],
                path=f"{path}.exact_oracle_nullity_lower_bound",
            )
            == 850,
            f"{path}.exact_oracle_nullity_lower_bound: drift",
        )

        for field in (
            "matched_cgls_field_relative_l2",
            "matched_huber_field_relative_l2",
            "matched_base_landweber_field_relative_l2",
            "field_gain_to_best_matched_classical",
            "h1_gain_to_best_matched_classical",
            "reprojection_ratio_to_matched_cgls",
            "visible_correction_fraction",
            "system_residual_fraction",
            "exact_projection_approximation_error",
            "exact_oracle_error_reduction_retention",
            "base_anchor_field_relative_l2",
            "exact_oracle_field_relative_l2",
            "raw_learned_field_relative_l2",
        ):
            _row_float(row, field, path=path)

        baseline_rows = {
            baseline_method: baselines[
                (baseline_method, row["case_id"], projection_iterations)
            ]
            for baseline_method in BASELINE_METHODS
        }
        baseline_field_errors = {
            baseline_method: _row_float(
                baseline_row,
                "field_relative_l2",
                path=f"baseline.{baseline_method}",
            )
            for baseline_method, baseline_row in baseline_rows.items()
        }
        baseline_h1_errors = {
            baseline_method: _row_float(
                baseline_row,
                "h1_seminorm_relative_error",
                path=f"baseline.{baseline_method}",
            )
            for baseline_method, baseline_row in baseline_rows.items()
        }
        for baseline_method, metric_field in (
            ("cgls_matched", "matched_cgls_field_relative_l2"),
            ("huber_pdhg_matched", "matched_huber_field_relative_l2"),
            ("base_landweber_matched", "matched_base_landweber_field_relative_l2"),
        ):
            _compare_csv_number(
                row[metric_field],
                baseline_field_errors[baseline_method],
                path=f"{path}.{metric_field}",
            )

        field_error = _row_float(row, "field_relative_l2", path=path)
        h1_error = _row_float(row, "h1_seminorm_relative_error", path=path)
        # The frozen gate defines "best matched classical" over CGLS and
        # Huber-PDHG; base Landweber is retained as a separately reported anchor.
        gate_baselines = ("cgls_matched", "huber_pdhg_matched")
        best_field_error = min(
            baseline_field_errors[baseline_method]
            for baseline_method in gate_baselines
        )
        best_h1_error = min(
            baseline_h1_errors[baseline_method]
            for baseline_method in gate_baselines
        )
        _require(best_field_error > 0.0 and best_h1_error > 0.0, "zero baseline error")
        expected_field_gain = (best_field_error - field_error) / best_field_error
        expected_h1_gain = (best_h1_error - h1_error) / best_h1_error
        _compare_csv_number(
            row["field_gain_to_best_matched_classical"],
            expected_field_gain,
            path=f"{path}.field_gain_to_best_matched_classical",
        )
        _compare_csv_number(
            row["h1_gain_to_best_matched_classical"],
            expected_h1_gain,
            path=f"{path}.h1_gain_to_best_matched_classical",
        )
        cgls_reprojection = _row_float(
            baseline_rows["cgls_matched"],
            "measured_reprojection_relative_l2",
            path="baseline.cgls_matched",
        )
        _require(cgls_reprojection > 0.0, "zero matched CGLS reprojection")
        expected_reprojection_ratio = (
            _row_float(row, "measured_reprojection_relative_l2", path=path)
            / cgls_reprojection
        )
        _compare_csv_number(
            row["reprojection_ratio_to_matched_cgls"],
            expected_reprojection_ratio,
            path=f"{path}.reprojection_ratio_to_matched_cgls",
        )
        expected_harm = int(expected_field_gain < -harm_threshold)
        _require(
            _csv_int(
                row["field_harm_to_best_matched_classical"],
                path=f"{path}.field_harm_to_best_matched_classical",
            )
            == expected_harm,
            f"{path}.field_harm_to_best_matched_classical: threshold drift",
        )

        base_reference = references[
            ("base_anchor", "prepared_cgls_base_12", -1, row["case_id"])
        ]
        raw_reference = references[
            ("raw_learned", method, model_seed, row["case_id"])
        ]
        exact_reference = references[
            (spec.exact_oracle_reference_kind, method, model_seed, row["case_id"])
        ]
        base_error = _row_float(base_reference, "field_relative_l2", path="base_reference")
        raw_error = _row_float(raw_reference, "field_relative_l2", path="raw_reference")
        exact_error = _row_float(exact_reference, "field_relative_l2", path="exact_reference")
        _compare_csv_number(
            row["base_anchor_field_relative_l2"],
            base_error,
            path=f"{path}.base_anchor_field_relative_l2",
        )
        _compare_csv_number(
            row["raw_learned_field_relative_l2"],
            raw_error,
            path=f"{path}.raw_learned_field_relative_l2",
        )
        _compare_csv_number(
            row["exact_oracle_field_relative_l2"],
            exact_error,
            path=f"{path}.exact_oracle_field_relative_l2",
        )
        oracle_reduction = base_error - exact_error
        _require(oracle_reduction > 0.0, f"{path}: exact oracle has no reduction")
        _require(
            _csv_int(
                row["oracle_reduction_defined"],
                path=f"{path}.oracle_reduction_defined",
            )
            == 1,
            f"{path}.oracle_reduction_defined: drift",
        )
        expected_retention = (base_error - field_error) / oracle_reduction
        _compare_csv_number(
            row["exact_oracle_error_reduction_retention"],
            expected_retention,
            path=f"{path}.exact_oracle_error_reduction_retention",
        )
        if projection_iterations == 0:
            for field in ERROR_FIELDS:
                expected = _row_float(raw_reference, field, path="raw_reference")
                _compare_csv_number(row[field], expected, path=f"{path}.{field}")

    expected_keys = {
        (method, model_seed, case_id, variant, projection_iterations)
        for method in METHODS
        for model_seed in MODEL_SEEDS
        for case_id in catalog
        for variant, _ in spec.variants
        for projection_iterations in spec.snapshots
    }
    _require(set(lookup) == expected_keys, "metric row identity grid drift")
    return lookup


def _recompute_aggregates(
    rows: Iterable[dict[str, str]],
) -> dict[tuple[str, int, str, str, int], dict[str, Any]]:
    groups: dict[tuple[str, int, str, str, int], list[dict[str, str]]] = {}
    for row in rows:
        key = (
            row["method"],
            int(row["model_seed"]),
            row["split"],
            row["projection_variant"],
            int(row["projection_iterations"]),
        )
        groups.setdefault(key, []).append(row)

    output: dict[tuple[str, int, str, str, int], dict[str, Any]] = {}
    for key, values in groups.items():
        method, model_seed, split, variant, projection_iterations = key
        expected_count = SPLIT_CASE_COUNTS[split]
        _require(
            len(values) == expected_count,
            f"aggregate {key}: expected {expected_count} metric rows",
        )
        paired_budgets = {int(row["paired_call_budget"]) for row in values}
        damping_fractions = {float(row["damping_fraction"]) for row in values}
        _require(len(paired_budgets) == 1, f"aggregate {key}: mixed paired budgets")
        _require(len(damping_fractions) == 1, f"aggregate {key}: mixed damping")
        output[key] = {
            "method": method,
            "model_seed": model_seed,
            "split": split,
            "projection_variant": variant,
            "projection_iterations": projection_iterations,
            "case_count": len(values),
            "paired_call_budget": next(iter(paired_budgets)),
            "damping_fraction": next(iter(damping_fractions)),
            "field_relative_l2_mean": _mean(
                float(row["field_relative_l2"]) for row in values
            ),
            "h1_seminorm_relative_error_mean": _mean(
                float(row["h1_seminorm_relative_error"]) for row in values
            ),
            "field_gain_to_best_matched_classical_mean": _mean(
                float(row["field_gain_to_best_matched_classical"])
                for row in values
            ),
            "h1_gain_to_best_matched_classical_mean": _mean(
                float(row["h1_gain_to_best_matched_classical"])
                for row in values
            ),
            "reprojection_ratio_to_matched_cgls_mean": _mean(
                float(row["reprojection_ratio_to_matched_cgls"]) for row in values
            ),
            "visible_correction_fraction_mean": _mean(
                float(row["visible_correction_fraction"]) for row in values
            ),
            "visible_correction_fraction_maximum": max(
                float(row["visible_correction_fraction"]) for row in values
            ),
            "system_residual_fraction_mean": _mean(
                float(row["system_residual_fraction"]) for row in values
            ),
            "exact_projection_approximation_error_mean": _mean(
                float(row["exact_projection_approximation_error"])
                for row in values
            ),
            "exact_oracle_error_reduction_retention_mean": _mean(
                float(row["exact_oracle_error_reduction_retention"])
                for row in values
                if int(row["oracle_reduction_defined"]) == 1
            ),
            "oracle_reduction_defined_rate": _mean(
                float(row["oracle_reduction_defined"]) for row in values
            ),
            "field_harm_rate": _mean(
                float(row["field_harm_to_best_matched_classical"])
                for row in values
            ),
            "worst_field_gain_to_best_matched_classical": min(
                float(row["field_gain_to_best_matched_classical"])
                for row in values
            ),
            "breakdown_rate": _mean(float(row["breakdown"]) for row in values),
        }
    return output


def _recompute_baseline_aggregates(
    rows: Iterable[dict[str, str]],
) -> dict[tuple[str, str, int], dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["method"], row["split"], int(row["matched_step"]))
        groups.setdefault(key, []).append(row)

    output: dict[tuple[str, str, int], dict[str, Any]] = {}
    for key, values in groups.items():
        method, split, matched_step = key
        expected_count = SPLIT_CASE_COUNTS[split]
        _require(
            len(values) == expected_count,
            f"baseline aggregate {key}: expected {expected_count} rows",
        )
        total_calls = {int(row["total_calls"]) for row in values}
        _require(len(total_calls) == 1, f"baseline aggregate {key}: mixed budgets")
        output[key] = {
            "method": method,
            "split": split,
            "matched_step": matched_step,
            "total_calls": next(iter(total_calls)),
            "case_count": len(values),
            "field_relative_l2_mean": _mean(
                float(row["field_relative_l2"]) for row in values
            ),
            "h1_seminorm_relative_error_mean": _mean(
                float(row["h1_seminorm_relative_error"]) for row in values
            ),
            "measured_reprojection_relative_l2_mean": _mean(
                float(row["measured_reprojection_relative_l2"]) for row in values
            ),
        }
    return output


def _csv_aggregate_map(
    rows: list[dict[str, str]],
    *,
    learned: bool,
) -> dict[Any, dict[str, str]]:
    output: dict[Any, dict[str, str]] = {}
    for index, row in enumerate(rows):
        if learned:
            key = (
                row["method"],
                _csv_int(row["model_seed"], path=f"aggregate_rows[{index}].model_seed"),
                row["split"],
                row["projection_variant"],
                _csv_int(
                    row["projection_iterations"],
                    path=f"aggregate_rows[{index}].projection_iterations",
                ),
            )
        else:
            key = (
                row["method"],
                row["split"],
                _csv_int(
                    row["matched_step"],
                    path=f"matched_baseline_aggregate_rows[{index}].matched_step",
                ),
            )
        _require(key not in output, f"duplicate aggregate row: {key}")
        output[key] = row
    return output


def _summary_aggregate_map(
    rows: Any,
    *,
    learned: bool,
) -> dict[Any, dict[str, Any]]:
    _require(isinstance(rows, list), "summary aggregate must be a list")
    fields = AGGREGATE_FIELDS if learned else BASELINE_AGGREGATE_FIELDS
    output: dict[Any, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        _require(isinstance(row, dict), f"summary aggregate[{index}] must be an object")
        _require(set(row) == set(fields), f"summary aggregate[{index}] schema drift")
        if learned:
            key = (
                row["method"],
                row["model_seed"],
                row["split"],
                row["projection_variant"],
                row["projection_iterations"],
            )
        else:
            key = (row["method"], row["split"], row["matched_step"])
        _require(key not in output, f"duplicate summary aggregate row: {key}")
        output[key] = row
    return output


def _compare_csv_aggregate_maps(
    actual: dict[Any, dict[str, str]],
    expected: dict[Any, dict[str, Any]],
    *,
    path: str,
) -> None:
    _require(set(actual) == set(expected), f"{path}: aggregate identity grid mismatch")
    for key, expected_row in expected.items():
        row = actual[key]
        for field, expected_value in expected_row.items():
            field_path = f"{path}[{key!r}].{field}"
            if isinstance(expected_value, str):
                _require(row[field] == expected_value, f"{field_path}: string mismatch")
            else:
                _compare_csv_number(row[field], expected_value, path=field_path)


def _compare_summary_aggregate_maps(
    actual: dict[Any, dict[str, Any]],
    expected: dict[Any, dict[str, Any]],
    *,
    path: str,
) -> None:
    _require(set(actual) == set(expected), f"{path}: aggregate identity grid mismatch")
    for key, expected_row in expected.items():
        _compare(actual[key], expected_row, path=f"{path}[{key!r}]")


def _recompute_decisions(
    rows: list[dict[str, str]],
    spec: PacketSpec,
) -> dict[str, Any]:
    decisions: dict[str, Any] = {}
    selection_rule = SELECTION_RULE
    for method in METHODS:
        candidates: list[dict[str, Any]] = []
        for variant, damping in spec.variants:
            for projection_iterations in spec.snapshots:
                values = [
                    row
                    for row in rows
                    if row["method"] == method
                    and row["split"] == "development"
                    and row["projection_variant"] == variant
                    and int(row["projection_iterations"]) == projection_iterations
                ]
                expected_count = len(MODEL_SEEDS) * SPLIT_CASE_COUNTS["development"]
                _require(
                    len(values) == expected_count,
                    f"decision {method}/{variant}/{projection_iterations}: row count drift",
                )
                development = {
                    "case_model_count": len(values),
                    "field_gain_mean": _mean(
                        float(row["field_gain_to_best_matched_classical"])
                        for row in values
                    ),
                    "h1_gain_mean": _mean(
                        float(row["h1_gain_to_best_matched_classical"])
                        for row in values
                    ),
                    "reprojection_ratio_mean": _mean(
                        float(row["reprojection_ratio_to_matched_cgls"])
                        for row in values
                    ),
                    "visible_correction_fraction_mean": _mean(
                        float(row["visible_correction_fraction"]) for row in values
                    ),
                    "visible_correction_fraction_maximum": max(
                        float(row["visible_correction_fraction"]) for row in values
                    ),
                    "exact_oracle_error_reduction_retention_mean": _mean(
                        float(row["exact_oracle_error_reduction_retention"])
                        for row in values
                        if int(row["oracle_reduction_defined"]) == 1
                    ),
                    "oracle_reduction_defined_rate": _mean(
                        float(row["oracle_reduction_defined"]) for row in values
                    ),
                    "field_harm_rate": _mean(
                        float(row["field_harm_to_best_matched_classical"])
                        for row in values
                    ),
                    "worst_field_gain": min(
                        float(row["field_gain_to_best_matched_classical"])
                        for row in values
                    ),
                    "breakdown_rate": _mean(
                        float(row["breakdown"]) for row in values
                    ),
                    "per_model_seed_field_gain_means": [
                        _mean(
                            float(row["field_gain_to_best_matched_classical"])
                            for row in values
                            if int(row["model_seed"]) == model_seed
                        )
                        for model_seed in MODEL_SEEDS
                    ],
                }
                eligible = (
                    development["visible_correction_fraction_mean"]
                    <= selection_rule["maximum_mean_visible_correction_fraction"]
                    and development["visible_correction_fraction_maximum"]
                    <= selection_rule["maximum_worst_case_visible_correction_fraction"]
                    and development["reprojection_ratio_mean"]
                    <= selection_rule[
                        "maximum_mean_reprojection_ratio_to_matched_cgls"
                    ]
                )
                candidates.append(
                    {
                        "projection_variant": variant,
                        "projection_iterations": projection_iterations,
                        "damping_fraction": damping,
                        "development": development,
                        "development_eligible": eligible,
                    }
                )

        eligible_candidates = [
            candidate for candidate in candidates if candidate["development_eligible"]
        ]
        _require(
            not eligible_candidates,
            f"{spec.stage} recomputation found an eligible development candidate",
        )
        decisions[method] = {
            "screened_candidates": candidates,
            "selection": None,
            "checks": {"development_selection_exists": False},
            "passed_m2_3_mechanism_gate": False,
        }
    return decisions


def _validate_training_runs(summary: dict[str, Any]) -> None:
    runs = summary["training_runs"]
    _require(isinstance(runs, list), "summary.training_runs must be a list")
    _require(len(runs) == len(METHODS) * len(MODEL_SEEDS), "training run count drift")
    fields = {
        "method",
        "model_seed",
        "parameters",
        "best_epoch",
        "best_development_field_relative_l2",
        "epochs_ran",
        "train_seconds",
        "device",
    }
    identities: set[tuple[str, int]] = set()
    for index, run in enumerate(runs):
        _require(isinstance(run, dict), f"training_runs[{index}] must be an object")
        _require(set(run) == fields, f"training_runs[{index}] schema drift")
        identity = (run["method"], run["model_seed"])
        _require(identity not in identities, f"duplicate training run: {identity}")
        identities.add(identity)
        _require(run["method"] in METHODS, f"training_runs[{index}].method drift")
        _require(run["model_seed"] in MODEL_SEEDS, f"training_runs[{index}].seed drift")
        for field in ("parameters", "best_epoch", "epochs_ran"):
            _require(
                type(run[field]) is int and run[field] > 0,
                f"training_runs[{index}].{field}: expected positive integer",
            )
        for field in ("best_development_field_relative_l2", "train_seconds"):
            _require(
                _json_float(run[field], path=f"training_runs[{index}].{field}") > 0.0,
                f"training_runs[{index}].{field}: expected positive value",
            )
        _require(
            run["device"] == summary["device"],
            f"training_runs[{index}].device: summary device drift",
        )
    expected = {(method, seed) for method in METHODS for seed in MODEL_SEEDS}
    _require(identities == expected, "training run identity grid drift")


def _validate_setup_ledgers(summary: dict[str, Any]) -> None:
    operator = summary["operator_norm_setup"]
    oracle = summary["retrospective_dense_oracle_setup"]
    _require(isinstance(operator, dict), "operator_norm_setup must be an object")
    _require(isinstance(oracle, dict), "retrospective_dense_oracle_setup must be an object")
    _require(len(operator) == 12, "operator norm setup geometry count drift")
    _require(len(oracle) == 12, "dense oracle setup geometry count drift")
    _require(set(operator) == set(oracle), "setup ledger geometry keys differ")
    operator_fields = {
        "matrix_shape",
        "spectral_norm_squared",
        "bound",
        "safety_factor",
        "setup_forward_calls",
        "status",
    }
    oracle_fields = {
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
        "used_by_algorithm",
    }
    for geometry_hash in operator:
        _require(
            re.fullmatch(r"[0-9a-f]{64}", geometry_hash) is not None,
            "setup ledger geometry hash malformed",
        )
        norm = operator[geometry_hash]
        _require(isinstance(norm, dict), "operator norm ledger entry must be an object")
        _require(set(norm) == operator_fields, "operator norm ledger schema drift")
        _compare(norm["matrix_shape"], [150, 1728], path="operator_norm.matrix_shape")
        spectral = _json_float(
            norm["spectral_norm_squared"], path="operator_norm.spectral_norm_squared"
        )
        bound = _json_float(norm["bound"], path="operator_norm.bound")
        _require(spectral > 0.0, "operator norm spectral value is nonpositive")
        _require(
            math.isclose(bound, spectral * 1.01, rel_tol=5e-11, abs_tol=5e-12),
            "operator norm safety-factor formula drift",
        )
        _compare(norm["safety_factor"], 1.01, path="operator_norm.safety_factor")
        _compare(norm["setup_forward_calls"], 7, path="operator_norm.setup_forward_calls")
        _require(
            norm["status"]
            == "DENSE_NUMERICAL_SVD_TIMES_SAFETY_FACTOR_NOT_INTERVAL_CERTIFIED",
            "operator norm setup status drift",
        )

        dense = oracle[geometry_hash]
        _require(isinstance(dense, dict), "dense oracle ledger entry must be an object")
        _require(set(dense) == oracle_fields, "dense oracle ledger schema drift")
        _compare(dense["matrix_shape"], [150, 1000], path="dense_oracle.matrix_shape")
        for field, expected in (
            ("active_voxel_count", 1000),
            ("measurement_count", 150),
            ("setup_forward_calls", 5),
            ("rank", 150),
            ("nullity_lower_bound", 850),
        ):
            _compare(dense[field], expected, path=f"dense_oracle.{field}")
        _compare(
            dense["zero_forward_maximum_absolute"],
            0.0,
            path="dense_oracle.zero_forward_maximum_absolute",
        )
        _require(
            dense["status"] == "RETROSPECTIVE_DENSE_TOY_ORACLE_NOT_ALGORITHM",
            "dense oracle setup status drift",
        )
        _require(dense["used_by_algorithm"] is False, "dense oracle entered algorithm")
        largest = _json_float(
            dense["largest_singular_value"], path="dense_oracle.largest_singular_value"
        )
        smallest = _json_float(
            dense["smallest_retained_singular_value"],
            path="dense_oracle.smallest_retained_singular_value",
        )
        tolerance = _json_float(
            dense["rank_tolerance"], path="dense_oracle.rank_tolerance"
        )
        _require(largest > 0.0 and smallest > 0.0, "dense oracle singular value drift")
        _require(
            math.isclose(tolerance, largest * 1e-10, rel_tol=5e-11, abs_tol=5e-12),
            "dense oracle rank tolerance formula drift",
        )
        _require(
            _json_float(
                dense["factorization_seconds"],
                path="dense_oracle.factorization_seconds",
            )
            > 0.0,
            "dense oracle factorization timing drift",
        )


def _validate_summary_header(
    summary: dict[str, Any],
    config: dict[str, Any],
    config_path: Path,
    spec: PacketSpec,
) -> None:
    expected_fields = set(SUMMARY_FIELDS)
    if spec.stage == "M2.4":
        expected_fields.update(
            {"source_m2_3_config_sha256", "source_m2_3_summary_sha256"}
        )
    _require(set(summary) == expected_fields, "summary top-level schema drift")
    _require(summary["schema_version"] == spec.report_schema, "summary schema drift")
    _require(
        summary["status"] == spec.report_status,
        f"summary status must remain {spec.stage} NO-GO",
    )
    _require(summary["evidence_level"] == spec.evidence_level, "summary evidence level drift")
    _require(
        summary["source_config_sha256"] == _sha256(config_path),
        "summary source config hash drift",
    )
    for prefix in ("source_t0", "source_m2_2"):
        for suffix in ("config_sha256", "summary_sha256"):
            field = f"{prefix}_{suffix}"
            _require(summary[field] == config[field], f"summary {field} drift")
    if spec.stage == "M2.4":
        for suffix in ("config_sha256", "summary_sha256"):
            field = f"source_m2_3_{suffix}"
            _require(summary[field] == config[field], f"summary {field} drift")
    _require(
        isinstance(summary["device"], str) and bool(summary["device"]),
        "summary device is missing",
    )
    _require(
        _json_float(summary["elapsed_seconds"], path="summary.elapsed_seconds") > 0.0,
        "summary elapsed time drift",
    )


def validate_packet(
    *,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Validate one M2.3 or M2.4 packet and return an independent verdict."""

    config_path = Path(config_path)
    output_dir = Path(output_dir)
    _require(config_path.is_file(), "diagnostic config is missing")
    _require(output_dir.is_dir(), "public evidence directory is missing")
    _validate_checksum_manifest(output_dir)

    config = _load_json(config_path)
    schema = config.get("schema_version")
    _require(schema in SPECS, "unsupported JACRU M2.3/M2.4 config schema")
    spec = SPECS[str(schema)]
    source_config, source_t0_results = _validate_config(config, config_path, spec)
    _validate_readme(output_dir, spec)

    summary = _load_json(output_dir / "summary.json")
    _validate_summary_header(summary, config, config_path, spec)
    _compare(summary["claim_boundary"], spec.claim_boundary, path="summary.claim_boundary")
    _compare(
        summary["public_export_policy"],
        PUBLIC_EXPORT_POLICY,
        path="summary.public_export_policy",
    )

    source_rows = _read_csv(
        source_t0_results / "metric_rows.csv",
        COMMON_METRIC_FIELDS,
    )
    catalog, source_learned = _source_case_catalog(source_rows, source_config)
    reference_rows = _read_csv(
        output_dir / "reference_rows.csv",
        REFERENCE_FIELDS,
        expected_rows=spec.expected_reference_rows,
    )
    baseline_rows = _read_csv(
        output_dir / "matched_baseline_rows.csv",
        BASELINE_FIELDS,
        expected_rows=spec.expected_baseline_rows,
    )
    metric_rows = _read_csv(
        output_dir / "metric_rows.csv",
        spec.metric_fields,
        expected_rows=spec.expected_metric_rows,
    )
    aggregate_rows = _read_csv(
        output_dir / "aggregate_rows.csv",
        AGGREGATE_FIELDS,
        expected_rows=spec.expected_aggregates,
    )
    baseline_aggregate_rows = _read_csv(
        output_dir / "matched_baseline_aggregate_rows.csv",
        BASELINE_AGGREGATE_FIELDS,
        expected_rows=spec.expected_baseline_aggregates,
    )

    references = _validate_reference_rows(
        reference_rows,
        catalog,
        source_learned,
        spec,
    )
    baselines, operator_bounds = _validate_baseline_rows(
        baseline_rows,
        catalog,
        spec,
    )
    _validate_metric_rows(
        metric_rows,
        catalog,
        references,
        baselines,
        operator_bounds,
        spec,
    )

    expected_aggregates = _recompute_aggregates(metric_rows)
    expected_baseline_aggregates = _recompute_baseline_aggregates(baseline_rows)
    _require(
        len(expected_aggregates) == spec.expected_aggregates,
        "learned aggregate count drift",
    )
    _require(
        len(expected_baseline_aggregates) == spec.expected_baseline_aggregates,
        "matched baseline aggregate count drift",
    )
    _compare_csv_aggregate_maps(
        _csv_aggregate_map(aggregate_rows, learned=True),
        expected_aggregates,
        path="aggregate_rows.csv",
    )
    _compare_csv_aggregate_maps(
        _csv_aggregate_map(baseline_aggregate_rows, learned=False),
        expected_baseline_aggregates,
        path="matched_baseline_aggregate_rows.csv",
    )
    _compare_summary_aggregate_maps(
        _summary_aggregate_map(summary["aggregate"], learned=True),
        expected_aggregates,
        path="summary.aggregate",
    )
    _compare_summary_aggregate_maps(
        _summary_aggregate_map(summary["matched_baseline_aggregate"], learned=False),
        expected_baseline_aggregates,
        path="summary.matched_baseline_aggregate",
    )

    decisions = _recompute_decisions(metric_rows, spec)
    _compare(summary["decisions"], decisions, path="summary.decisions")
    _compare(summary["authorization"], AUTHORIZATION, path="summary.authorization")
    _require(
        all(
            value is False
            for key, value in summary["authorization"].items()
            if key != "continue_matrix_free_preconditioner_research"
        ),
        "summary authorization exceeds the NO-GO boundary",
    )

    expected_counts = {
        "metric_row_count": spec.expected_metric_rows,
        "reference_row_count": spec.expected_reference_rows,
        "matched_baseline_row_count": spec.expected_baseline_rows,
    }
    for field, expected in expected_counts.items():
        _require(
            type(summary[field]) is int and summary[field] == expected,
            f"summary {field} drift",
        )
    _validate_training_runs(summary)
    _validate_setup_ledgers(summary)

    return {
        "status": spec.validated_status,
        "stage": spec.stage,
        "report_status": spec.report_status,
        "source_config_sha256": _sha256(config_path),
        "metric_row_count": len(metric_rows),
        "reference_row_count": len(reference_rows),
        "matched_baseline_row_count": len(baseline_rows),
        "aggregate_count": len(expected_aggregates),
        "matched_baseline_aggregate_count": len(expected_baseline_aggregates),
        "decision_count": len(decisions),
        "eligible_development_candidate_count": sum(
            1
            for decision in decisions.values()
            for candidate in decision["screened_candidates"]
            if candidate["development_eligible"]
        ),
        "development_selection_count": sum(
            decision["selection"] is not None for decision in decisions.values()
        ),
        "authorization": dict(AUTHORIZATION),
    }


def validate_all_packets() -> dict[str, dict[str, Any]]:
    """Validate the canonical M2.3 and M2.4 packets independently."""

    return {
        "m2_3": validate_packet(
            config_path=M2_3_SPEC.config_path,
            output_dir=M2_3_SPEC.output_dir,
        ),
        "m2_4": validate_packet(
            config_path=M2_4_SPEC.config_path,
            output_dir=M2_4_SPEC.output_dir,
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packet",
        choices=("all", "m2-3", "m2-4"),
        default="all",
        help="canonical packet to validate (default: both)",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.packet == "all":
        _require(
            args.config is None and args.output_dir is None,
            "custom paths require --packet m2-3 or --packet m2-4",
        )
        report: dict[str, Any] = validate_all_packets()
    else:
        spec = M2_3_SPEC if args.packet == "m2-3" else M2_4_SPEC
        report = validate_packet(
            config_path=args.config or spec.config_path,
            output_dir=args.output_dir or spec.output_dir,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
