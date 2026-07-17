#!/usr/bin/env python3
"""Independently validate the deterministic PSU-B0 Gate-B release.

The validator does not import the Gate-B runner.  It verifies the frozen
configuration and git-anchored source bytes, release checksums, exact row and
call-budget coverage, objective-unit conversions, graph replay, summaries,
paired timings, and the final gate decision from the emitted CSV and audit
payloads.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_SCHEMA = "psu-b0-factor-pdhg-gate-b-config-1.3-a-only-connectivity-amendment"
REPORT_SCHEMA = "psu-b0-factor-pdhg-gate-b-report-1.3-a-only-connectivity-amendment"
VALIDATOR_SCHEMA = "psu-b0-factor-pdhg-gate-b-validator-1.0"
FACTOR_SCHEMA = "psu-b0-gate-b-true-data-only-metric-1.0"
VALIDATOR_STATUS = "PASS_INDEPENDENT_GATE_B_RECOMPUTATION"
ALLOWED_GATE_STATUSES = {
    "GATE_B_E2_ORACLE_SCALE_CONDITIONING_SIGNAL_ONLY",
    "GATE_B_E2_MECHANISM_NO_GO",
}
REPLICATES = (0, 8)
FAMILIES = (
    "plume",
    "wavy_front",
    "thin_front",
    "double_front",
    "annular_kernel",
    "oblique_shock",
    "vortex_pair",
    "multi_plume",
)
CHECKPOINTS = (4, 8, 16, 32)
METHODS = (
    "scalar_a_only_pdhg",
    "view_block_a_only_pdhg",
    "voxel_factor_a_only_pdhg",
    "graph_pcgls",
)
MODES = ("scalar", "view_block", "voxel_factor")
TEST_SELECTORS = (
    "demo_t16_operator/test_psu_b0_gate_b_data_only.py",
    "site_tools/test_psu_b0_factor_interfaces.py::test_trilinear_p_and_pt_pass_dot_product_identity_without_full_calls",
    "demo_t16_operator/test_psu_b0_factor_majorizer_pipeline.py::test_contract_validation_and_setup_signature_fail_closed",
    "site_tools/test_psu_b0_gate_a_attestation_mps.py::test_mps_factor_recurrence_matches_cpu_reference",
    "site_tools/test_build_psu_b0_gate_b_parent_snapshot.py",
    "site_tools/test_psu_b0_gate_b_graph_path_diagnostic.py",
    "site_tools/test_psu_b0_gate_b_connectivity_diagnostic.py",
    "site_tools/test_run_psu_b0_factor_gate_b.py",
    "site_tools/test_validate_psu_b0_factor_gate_b.py",
)
TEST_CASE_COUNT = 41
TEST_NODE_MANIFEST_SHA256 = (
    "d2e2155e2053234bf445f3c41b38565976f9619581515cbb843aaf918b126826"
)
AMENDMENT = {
    "kind": "POST_SETUP_DIAGNOSTIC_PRE_FACTOR_A_ONLY_CONNECTIVITY_AMENDMENT",
    "parent_v3_commit": "81d9513eadb1f8ee6d3c0a33f9a913b9f58bd863",
    "parent_v3_config_sha256": "aedec9db6c8a81229cbe4c51f9c42538485f853cd24003564ba0e130d57239dd",
    "v3_failure": "a_only_data_coupled_voxel_count_differs_from_a_plus_d_count",
    "factor_trajectories_completed_before_v4_freeze": 0,
    "factor_solver_calls_before_v4_freeze": 0,
    "factor_metric_rows_observed_before_v4_freeze": 0,
    "emitted_metric_row_count": 0,
    "emitted_timing_pair_count": 0,
    "result_directory_created": False,
    "connectivity_diagnostic_report_sha256": "d3418148c4855419fdba15c2e0515b1dbbe4d81223352227fb9bee1b01582f19",
    "graph_diagnostic_report_sha256": "5499333734aca7e8327546a3db354ec84b930f057862e89b5563df89be02fda4",
    "parent_batch_replay_is_binding": False,
    "scored_graph_control": "same_run_single_sample_exact_k_graph_pcgls",
    "support_active_voxel_count": 2744,
    "data_coupled_voxel_count": 2322,
    "data_null_support_voxel_count": 422,
    "active_data_row_count": 4608,
    "active_primal_indices_sha256": "57cc5748864d0bb3bffe0f971b5625e3f40a1dd87fed2f10a6166514e406d0f5",
    "graph_full_support_sobolev_extrapolation_disclosed": True,
    "solver_math_changed": False,
    "factor_decision_thresholds_changed": False,
    "factor_decision_formula_changed": False,
    "sample_or_checkpoint_set_changed": False,
}
EXPECTED_RUNTIME_SHAPE = {
    "grid_size": 16,
    "view_count": 9,
    "rays_per_view": 256,
    "finite_aperture_sample_count": 8,
    "measurement_count": 4608,
    "support_active_voxel_count": 2744,
    "data_coupled_voxel_count": 2322,
}
DATA_ROLE = {
    "geometry": "REAL_PSU_DETECTOR_GEOMETRY",
    "fields": "ANALYTIC_REACTION_PHANTOMS_WITH_CLEAN_TRUTH",
    "noise": "SYNTHETIC_CORRELATED_GRAPH_HEAT",
    "scale_by_view": "CLEAN_TRUTH_DERIVED_ORACLE_DEVELOPMENT_ONLY",
    "statistical_unit": "TWO_REPLICATES_OF_EIGHT_PAIRED_MORPHOLOGIES_NOT_16_IID",
}
OPERATOR_BUDGET = {
    "zero_initialization": True,
    "exact_forward_calls_equal_checkpoint": True,
    "exact_adjoint_calls_equal_checkpoint": True,
    "tv_forward_calls_per_iteration": 0,
    "tv_adjoint_calls_per_iteration": 0,
    "graph_uses_exact_same_k": True,
    "best_less_than_or_equal_k_selection_forbidden": True,
    "early_stopping_forbidden": True,
}
OBJECTIVE_CONTRACT = {
    "factor_unit": "0.5*||A0*z-b0||^2",
    "graph_unit": "0.5*||A*x-b||^2",
    "conversion": "J_graph=amplitude_scale^2*4608*J_factor",
    "cross_method_raw_objective_ranking_forbidden": True,
}
TIMING_CONTRACT = {
    "primary_scope": "per_field_one_shot_setup_plus_k32_solver",
    "paired_execution": "serial_single_sample",
    "synchronization": "before_and_after_each_timed_call",
    "counterbalanced_by_replicate": True,
    "scoring_hashing_and_plotting_inside_timed_region": False,
    "process_cold_start_claimed": False,
    "multi_frame_amortized_speed_claimed": False,
}
INVALIDITY_POLICY = {
    "source_hash_mismatch": "INVALID_NO_RANKING",
    "test_preflight_failure": "INVALID_NO_RANKING",
    "operator_or_target_equivalence_failure": "INVALID_NO_RANKING",
    "missing_duplicate_nan_skip_or_oom": "INVALID_NO_RANKING",
    "postopen_protocol_mutation": "INVALID_NO_RANKING",
}
THRESHOLDS = {
    "operator_equivalence_relative_error_max": 5e-5,
    "factor_mean_reduction_vs_scalar_percent_min": 25.0,
    "factor_positive_sample_count_min": 12,
    "factor_worst_gain_vs_scalar_percent_min": -3.0,
    "factor_graph_mean_error_gap_percent_max": 20.0,
    "factor_mean_reduction_vs_view_block_percent_min": 3.0,
    "factor_wall_time_ratio_vs_single_sample_graph_max": 3.0,
    "target_normalization_relative_error_max": 1e-7,
    "positive_gain_tolerance_percent": 0.0,
    "mean_monotonic_tolerance": 1e-12,
}
CLAIM_BOUNDARY = {
    "real_psu_detector_geometry_used": True,
    "analytic_reaction_fields_used": True,
    "synthetic_correlated_noise_used": True,
    "scale_by_view_uses_clean_truth": True,
    "fresh_seed_set_opened": False,
    "experimental_flow_truth_used": False,
    "real_flowoff_repeats_used": False,
    "neural_operator_training_authorized": False,
    "algorithm_superiority_claimed": False,
}
SOURCE_KEYS = {
    "parent_gate_b_v3_config",
    "gate_b_v4_amendment_note",
    "gate_b_connectivity_diagnostic_runner_source",
    "gate_b_connectivity_diagnostic_report",
    "gate_b_connectivity_diagnostic_rows",
    "gate_b_connectivity_diagnostic_test_source",
    "parent_gate_b_v2_config",
    "gate_b_v3_amendment_note",
    "gate_b_graph_diagnostic_runner_source",
    "gate_b_graph_diagnostic_report",
    "gate_b_graph_diagnostic_rows",
    "gate_b_graph_diagnostic_test_source",
    "parent_gate_b_v1_config",
    "gate_b_amendment_note",
    "parent_pdhg_config",
    "parent_pdhg_report",
    "parent_metric_rows",
    "parent_public_summary",
    "parent_snapshot_builder_source",
    "parent_snapshot_test_source",
    "gate_a_attestation",
    "gate_a_validation_report",
    "factor_pipeline_source",
    "gate_b_metric_source",
    "gate_b_runner_source",
    "gate_b_validator_source",
    "gate_b_metric_test_source",
    "gate_b_runner_test_source",
    "gate_b_validator_test_source",
    "factor_interface_test_source",
    "factor_pipeline_test_source",
    "gate_a_mps_test_source",
}
LEDGER_KEYS = {
    "signed_data_forward_calls",
    "signed_data_transpose_calls",
    "absolute_data_forward_calls",
    "absolute_data_transpose_calls",
    "signed_tv_forward_calls",
    "signed_tv_transpose_calls",
    "absolute_tv_forward_calls",
    "absolute_tv_transpose_calls",
}
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA_PATTERN = re.compile(r"[0-9a-f]{64}")
RUNNER_FILES = {"report.json", "metric_rows.csv", "audit.json"}


class ValidationError(AssertionError):
    pass


class Validator:
    def __init__(self) -> None:
        self.check_count = 0

    def require(self, condition: bool, message: str) -> None:
        self.check_count += 1
        if not condition:
            raise ValidationError(message)

    def close(
        self,
        actual: Any,
        expected: Any,
        message: str,
        *,
        rel_tol: float = 1e-9,
        abs_tol: float = 1e-11,
    ) -> None:
        left = float(actual)
        right = float(expected)
        self.require(
            math.isfinite(left)
            and math.isfinite(right)
            and math.isclose(left, right, rel_tol=rel_tol, abs_tol=abs_tol),
            f"{message}: {left} != {right}",
        )


def _reject_constant(raw: str) -> None:
    raise ValidationError(f"non-finite JSON constant is forbidden: {raw}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValidationError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def load_strict_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
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


def _safe_repository_path(root: Path, raw: str) -> Path:
    token = PurePosixPath(raw)
    if token.is_absolute() or not token.parts or ".." in token.parts:
        raise ValidationError(f"unsafe repository path: {raw}")
    resolved = (root / token).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValidationError(f"repository path escaped root: {raw}")
    return resolved


def parse_checksum_manifest(path: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or not SHA_PATTERN.fullmatch(parts[0]):
            raise ValidationError("malformed Gate-B checksum line")
        name = parts[1]
        if name in output or Path(name).name != name:
            raise ValidationError("duplicate or unsafe Gate-B checksum name")
        output[name] = parts[0]
    if set(output) != RUNNER_FILES:
        raise ValidationError("Gate-B checksum manifest file set changed")
    return output


def _git_blob(root: Path, commit: str, raw_path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{raw_path}"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise ValidationError(f"source is absent from recorded commit: {raw_path}")
    return result.stdout


def _finite(value: Any, name: str, *, nonnegative: bool = False) -> float:
    number = float(value)
    if not math.isfinite(number) or (nonnegative and number < 0.0):
        raise ValidationError(f"{name} must be finite" + (" and nonnegative" if nonnegative else ""))
    return number


def _load_config(path: Path, validator: Validator) -> dict[str, Any]:
    config = load_strict_json(path)
    validator.require(config.get("schema_version") == CONFIG_SCHEMA, "config schema drift")
    validator.require(
        config.get("status") == "FROZEN_POST_SETUP_DIAGNOSTIC_PRE_FACTOR_PROTOCOL",
        "v4 A-only connectivity protocol was not frozen before factor performance",
    )
    validator.require(config.get("amendment") == AMENDMENT, "amendment provenance drift")
    validator.require(
        config.get("evidence_role") == "OPENED_DEVELOPMENT_MECHANISM_GATE",
        "config evidence role drift",
    )
    validator.require(tuple(config.get("replicate_indices", ())) == REPLICATES, "replicate drift")
    validator.require(tuple(config.get("reaction_families", ())) == FAMILIES, "family drift")
    validator.require(tuple(config.get("checkpoints", ())) == CHECKPOINTS, "checkpoint drift")
    validator.require(tuple(config.get("methods", ())) == METHODS, "method drift")
    validator.require(tuple(config.get("factor_modes", ())) == MODES, "mode drift")
    validator.require(config.get("expected_runtime_shape") == EXPECTED_RUNTIME_SHAPE, "shape drift")
    validator.require(config.get("data_role") == DATA_ROLE, "data role drift")
    validator.require(config.get("operator_budget") == OPERATOR_BUDGET, "operator budget drift")
    validator.require(config.get("objective_contract") == OBJECTIVE_CONTRACT, "objective contract drift")
    validator.require(config.get("timing_contract") == TIMING_CONTRACT, "timing contract drift")
    validator.require(config.get("invalidity_policy") == INVALIDITY_POLICY, "invalidity policy drift")
    validator.require(config.get("thresholds") == THRESHOLDS, "threshold drift")
    validator.require(
        config.get("solver") == {"eta": 0.7, "theta": 1.0, "regularization_weight": 0.0},
        "solver drift",
    )
    validator.require(config.get("operator_equivalence_seed") == 2026071702, "probe seed drift")
    validator.require(
        config.get("timing_order_by_replicate")
        == {
            "0": ["scalar", "view_block", "voxel_factor", "graph"],
            "8": ["graph", "voxel_factor", "view_block", "scalar"],
        },
        "timing order drift",
    )
    validator.require(config.get("claim_boundary") == CLAIM_BOUNDARY, "claim boundary drift")
    source_paths = config.get("source_paths")
    source_hashes = config.get("source_sha256")
    validator.require(isinstance(source_paths, dict), "source path map missing")
    validator.require(isinstance(source_hashes, dict), "source hash map missing")
    validator.require(set(source_paths) == SOURCE_KEYS, "source path key set drift")
    validator.require(set(source_hashes) == SOURCE_KEYS, "source hash key set drift")
    for key in sorted(SOURCE_KEYS):
        raw_path = source_paths[key]
        digest = source_hashes[key]
        validator.require(isinstance(raw_path, str), f"source path is not text: {key}")
        _safe_repository_path(REPOSITORY_ROOT, raw_path)
        validator.require(
            isinstance(digest, str) and SHA_PATTERN.fullmatch(digest) is not None,
            f"source hash is malformed: {key}",
        )
    preflight = config.get("test_preflight")
    validator.require(isinstance(preflight, dict), "test preflight missing")
    validator.require(
        tuple(preflight.get("selectors", ())) == TEST_SELECTORS,
        "test selectors changed",
    )
    validator.require(
        preflight.get("expected_case_count") == TEST_CASE_COUNT,
        "test count changed",
    )
    validator.require(
        preflight.get("expected_node_manifest_sha256") == TEST_NODE_MANIFEST_SHA256,
        "test manifest hash changed",
    )
    return config


def _read_metric_rows(path: Path, validator: Validator) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        validator.require(reader.fieldnames is not None, "metric CSV has no header")
        validator.require(len(reader.fieldnames) == len(set(reader.fieldnames)), "duplicate CSV column")
        source_rows = list(reader)
    expected_keys = {
        (replicate, sample, method, iteration)
        for replicate in REPLICATES
        for sample in range(len(FAMILIES))
        for method in METHODS
        for iteration in CHECKPOINTS
    }
    rows: list[dict[str, Any]] = []
    observed: set[tuple[int, int, str, int]] = set()
    for raw in source_rows:
        replicate = int(raw["replicate"])
        sample = int(raw["sample_index"])
        method = raw["method"]
        iteration = int(raw["iterations"])
        key = (replicate, sample, method, iteration)
        validator.require(key not in observed, f"duplicate metric row: {key}")
        observed.add(key)
        validator.require(key in expected_keys, f"unexpected metric row: {key}")
        validator.require(raw["reaction_family"] == FAMILIES[sample], "family label mismatch")
        validator.require(int(raw["forward_calls"]) == iteration, "forward budget mismatch")
        validator.require(int(raw["adjoint_calls"]) == iteration, "adjoint budget mismatch")
        row: dict[str, Any] = {
            "replicate": replicate,
            "sample_index": sample,
            "reaction_family": raw["reaction_family"],
            "method": method,
            "iterations": iteration,
            "field_relative_l2": _finite(raw["field_relative_l2"], "field error", nonnegative=True),
            "gradient_relative_l2": _finite(raw["gradient_relative_l2"], "gradient error", nonnegative=True),
            "front_top10_f1": _finite(raw["front_top10_f1"], "front F1", nonnegative=True),
            "whitened_data_objective": _finite(raw["whitened_data_objective"], "objective", nonnegative=True),
            "objective_scale_to_physical": _finite(raw["objective_scale_to_physical"], "objective scale", nonnegative=True),
            "physical_equivalent_whitened_objective": _finite(
                raw["physical_equivalent_whitened_objective"],
                "physical objective",
                nonnegative=True,
            ),
            "objective_unit": raw["objective_unit"],
            "source": raw["source"],
        }
        validator.require(row["front_top10_f1"] <= 1.0, "front F1 exceeds one")
        validator.require(row["objective_scale_to_physical"] > 0.0, "objective scale must be positive")
        validator.close(
            row["physical_equivalent_whitened_objective"],
            row["whitened_data_objective"] * row["objective_scale_to_physical"],
            "objective unit conversion mismatch",
            rel_tol=2e-7,
            abs_tol=1e-7,
        )
        if method == "graph_pcgls":
            validator.require(row["objective_unit"] == "graph_whitened_physical_field", "graph objective unit drift")
            validator.require(row["source"] == "single_sample_graph_replay", "graph source drift")
            validator.close(row["objective_scale_to_physical"], 1.0, "graph objective scale drift")
            validator.require(raw["trajectory_elapsed_seconds"] == "", "graph row timing must be separate")
        else:
            validator.require(row["objective_unit"] == "normalized_b0_per_sample", "factor objective unit drift")
            validator.require(row["source"] == "gate_b_production_factor_pipeline", "factor source drift")
            validator.require(
                _finite(raw["trajectory_elapsed_seconds"], "factor trajectory time") > 0.0,
                "factor trajectory time must be positive",
            )
        rows.append(row)
    validator.require(observed == expected_keys, "metric row coverage mismatch")
    validator.require(len(rows) == 256, "metric row count changed")
    return rows


def _summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in METHODS:
        for iteration in CHECKPOINTS:
            values = [
                row
                for row in rows
                if row["method"] == method and row["iterations"] == iteration
            ]
            if len(values) != 16:
                raise ValidationError("summary coverage changed")
            output.append(
                {
                    "method": method,
                    "iterations": iteration,
                    "sample_count": 16,
                    "mean_field_relative_l2": statistics.fmean(row["field_relative_l2"] for row in values),
                    "p90_field_relative_l2": float(np.quantile([row["field_relative_l2"] for row in values], 0.90)),
                    "mean_gradient_relative_l2": statistics.fmean(row["gradient_relative_l2"] for row in values),
                    "mean_front_top10_f1": statistics.fmean(row["front_top10_f1"] for row in values),
                }
            )
    return sorted(output, key=lambda row: (row["method"], row["iterations"]))


def recompute_decision(
    rows: Sequence[Mapping[str, Any]],
    timing_rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any] = THRESHOLDS,
) -> dict[str, Any]:
    lookup = {
        (int(row["replicate"]), int(row["sample_index"]), str(row["method"]), int(row["iterations"])): float(row["field_relative_l2"])
        for row in rows
    }
    if len(lookup) != 256:
        raise ValidationError("decision metric coverage changed")
    sample_keys = [(replicate, sample) for replicate in REPLICATES for sample in range(len(FAMILIES))]

    def values(method: str, iteration: int) -> list[float]:
        try:
            return [lookup[(replicate, sample, method, iteration)] for replicate, sample in sample_keys]
        except KeyError as error:
            raise ValidationError("decision metric row is missing") from error

    scalar = values("scalar_a_only_pdhg", 32)
    block = values("view_block_a_only_pdhg", 32)
    factor = values("voxel_factor_a_only_pdhg", 32)
    graph = values("graph_pcgls", 32)
    if min(scalar + block + graph) <= 0.0:
        raise ValidationError("decision percentage denominator is nonpositive")
    scalar_mean = statistics.fmean(scalar)
    block_mean = statistics.fmean(block)
    factor_mean = statistics.fmean(factor)
    graph_mean = statistics.fmean(graph)
    gains = [100.0 * (baseline - candidate) / baseline for baseline, candidate in zip(scalar, factor)]
    reduction_scalar = 100.0 * (scalar_mean - factor_mean) / scalar_mean
    reduction_block = 100.0 * (block_mean - factor_mean) / block_mean
    graph_gap = 100.0 * (factor_mean - graph_mean) / graph_mean
    replicate_reductions: dict[str, float] = {}
    for replicate in REPLICATES:
        selected = [index for index, key in enumerate(sample_keys) if key[0] == replicate]
        base = statistics.fmean(scalar[index] for index in selected)
        candidate = statistics.fmean(factor[index] for index in selected)
        replicate_reductions[str(replicate)] = 100.0 * (base - candidate) / base
    factor_means = [statistics.fmean(values("voxel_factor_a_only_pdhg", iteration)) for iteration in CHECKPOINTS]
    block_means = [statistics.fmean(values("view_block_a_only_pdhg", iteration)) for iteration in CHECKPOINTS]
    monotonic_tolerance = float(thresholds["mean_monotonic_tolerance"])
    factor_monotone = all(right <= left + monotonic_tolerance for left, right in zip(factor_means, factor_means[1:]))
    block_monotone = all(right <= left + monotonic_tolerance for left, right in zip(block_means, block_means[1:]))
    timing_lookup: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in timing_rows:
        key = (int(row["replicate"]), int(row["sample_index"]))
        if key in timing_lookup:
            raise ValidationError("duplicate decision timing row")
        timing_lookup[key] = row
    if set(timing_lookup) != set(sample_keys):
        raise ValidationError("decision timing coverage changed")
    ratios = []
    for key in sample_keys:
        factor_seconds = _finite(timing_lookup[key]["voxel_factor_seconds"], "factor timing")
        graph_seconds = _finite(timing_lookup[key]["graph_seconds"], "graph timing")
        if factor_seconds <= 0.0 or graph_seconds <= 0.0:
            raise ValidationError("decision timing must be positive")
        ratios.append(factor_seconds / graph_seconds)
    median_ratio = float(statistics.median(ratios))
    positive_tolerance = float(thresholds["positive_gain_tolerance_percent"])
    gates = {
        "mean_reduction_vs_scalar": reduction_scalar >= float(thresholds["factor_mean_reduction_vs_scalar_percent_min"]),
        "both_replicates_positive": all(value > positive_tolerance for value in replicate_reductions.values()),
        "positive_sample_count": sum(value > positive_tolerance for value in gains) >= int(thresholds["factor_positive_sample_count_min"]),
        "worst_gain_vs_scalar": min(gains) >= float(thresholds["factor_worst_gain_vs_scalar_percent_min"]),
        "factor_mean_monotone": factor_monotone,
        "same_k_graph_gap": graph_gap <= float(thresholds["factor_graph_mean_error_gap_percent_max"]),
        "voxel_attribution_vs_view_block": reduction_block >= float(thresholds["factor_mean_reduction_vs_view_block_percent_min"]),
        "single_sample_wall_time": median_ratio <= float(thresholds["factor_wall_time_ratio_vs_single_sample_graph_max"]),
    }
    passed = all(gates.values())
    return {
        "status": "GATE_B_E2_ORACLE_SCALE_CONDITIONING_SIGNAL_ONLY" if passed else "GATE_B_E2_MECHANISM_NO_GO",
        "all_gates_passed": passed,
        "gates": gates,
        "metrics": {
            "scalar_k32_mean_field_relative_l2": scalar_mean,
            "view_block_k32_mean_field_relative_l2": block_mean,
            "voxel_factor_k32_mean_field_relative_l2": factor_mean,
            "graph_k32_mean_field_relative_l2": graph_mean,
            "factor_mean_reduction_vs_scalar_percent": reduction_scalar,
            "factor_mean_reduction_vs_view_block_percent": reduction_block,
            "factor_graph_mean_error_gap_percent": graph_gap,
            "factor_positive_sample_count": sum(value > positive_tolerance for value in gains),
            "factor_worst_gain_vs_scalar_percent": min(gains),
            "paired_gain_vs_scalar_percent": {
                f"r{replicate}_s{sample}": gain
                for (replicate, sample), gain in zip(sample_keys, gains)
            },
            "replicate_reduction_vs_scalar_percent": replicate_reductions,
            "factor_mean_field_by_checkpoint": dict(zip(map(str, CHECKPOINTS), factor_means)),
            "view_block_mean_field_by_checkpoint": dict(zip(map(str, CHECKPOINTS), block_means)),
            "view_block_mean_monotone": block_monotone,
            "factor_median_wall_time_ratio_vs_single_sample_graph": median_ratio,
        },
        "thresholds": dict(thresholds),
        "fm_cg_pdno_zero_init_smoke_authorized": passed,
        "algorithm_superiority_claim_authorized": False,
    }


def _compare_nested(validator: Validator, actual: Any, expected: Any, path: str) -> None:
    if isinstance(expected, Mapping):
        validator.require(isinstance(actual, Mapping), f"{path} must be an object")
        validator.require(set(actual) == set(expected), f"{path} key set changed")
        for key in expected:
            _compare_nested(validator, actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        validator.require(isinstance(actual, list), f"{path} must be a list")
        validator.require(len(actual) == len(expected), f"{path} length changed")
        for index, value in enumerate(expected):
            _compare_nested(validator, actual[index], value, f"{path}[{index}]")
    elif isinstance(expected, float):
        validator.close(actual, expected, path, rel_tol=2e-8, abs_tol=2e-10)
    else:
        validator.require(actual == expected, f"{path} changed")


def _expected_ledger(*, signed_forward: int = 0, signed_transpose: int = 0, absolute_forward: int = 0, absolute_transpose: int = 0) -> dict[str, int]:
    return {
        "signed_data_forward_calls": signed_forward,
        "signed_data_transpose_calls": signed_transpose,
        "absolute_data_forward_calls": absolute_forward,
        "absolute_data_transpose_calls": absolute_transpose,
        "signed_tv_forward_calls": 0,
        "signed_tv_transpose_calls": 0,
        "absolute_tv_forward_calls": 0,
        "absolute_tv_transpose_calls": 0,
    }


def _validate_audit(
    audit: Mapping[str, Any],
    report: Mapping[str, Any],
    config: Mapping[str, Any],
    validator: Validator,
) -> list[dict[str, Any]]:
    validator.require(
        set(audit) == {"operator_equivalence_rows", "target_normalization_rows", "timing_rows", "factor_call_rows"},
        "audit section set changed",
    )
    equivalence = audit["operator_equivalence_rows"]
    target_rows = audit["target_normalization_rows"]
    timing_rows = audit["timing_rows"]
    call_rows = audit["factor_call_rows"]
    validator.require(all(isinstance(value, list) for value in (equivalence, target_rows, timing_rows, call_rows)), "audit rows must be lists")
    expected_samples = {(replicate, sample) for replicate in REPLICATES for sample in range(len(FAMILIES))}
    validator.require(len(equivalence) == 16, "operator-equivalence row count changed")
    observed_equivalence: set[tuple[int, int]] = set()
    maximum_forward = 0.0
    maximum_transpose = 0.0
    setup_ledger = _expected_ledger(absolute_forward=1, absolute_transpose=1)
    for row in equivalence:
        key = (int(row["replicate"]), int(row["sample_index"]))
        validator.require(key not in observed_equivalence, "duplicate operator-equivalence row")
        observed_equivalence.add(key)
        forward = _finite(row["forward_relative_error"], "forward equivalence", nonnegative=True)
        transpose = _finite(row["transpose_relative_error"], "transpose equivalence", nonnegative=True)
        maximum_forward = max(maximum_forward, forward)
        maximum_transpose = max(maximum_transpose, transpose)
        validator.require(int(row["active_support_count"]) == 2744, "active support count drift")
        validator.require(int(row["reduced_primal_count"]) == 2322, "reduced primal count drift")
        validator.require(
            row["active_primal_indices_sha256"]
            == AMENDMENT["active_primal_indices_sha256"],
            "A-only active-primal mask drift",
        )
        validator.require(_finite(row["tau_minimum"], "tau minimum") > 0.0, "tau minimum must be positive")
        validator.require(_finite(row["tau_maximum"], "tau maximum") >= float(row["tau_minimum"]), "tau range is invalid")
        validator.require(row["setup_logical_calls"] == setup_ledger, "equivalence setup logical ledger drift")
        validator.require(row["setup_physical_calls"] == setup_ledger, "equivalence setup physical ledger drift")
    validator.require(observed_equivalence == expected_samples, "operator-equivalence coverage drift")
    validator.require(max(maximum_forward, maximum_transpose) <= THRESHOLDS["operator_equivalence_relative_error_max"], "operator equivalence threshold failed")
    validator.close(report["operator_equivalence"]["maximum_forward_relative_error"], maximum_forward, "reported forward equivalence maximum")
    validator.close(report["operator_equivalence"]["maximum_transpose_relative_error"], maximum_transpose, "reported transpose equivalence maximum")

    validator.require(len(target_rows) == 16, "target-normalization row count changed")
    target_lookup: set[tuple[int, int]] = set()
    target_maximum = 0.0
    for row in target_rows:
        key = (int(row["replicate"]), int(row["sample_index"]))
        validator.require(key not in target_lookup, "duplicate target-normalization row")
        target_lookup.add(key)
        validator.require(int(row["measurement_count"]) == 4608, "target measurement count drift")
        target_maximum = max(target_maximum, _finite(row["relative_error"], "target normalization", nonnegative=True))
    validator.require(target_lookup == expected_samples, "target-normalization coverage drift")
    validator.require(target_maximum <= THRESHOLDS["target_normalization_relative_error_max"], "target normalization threshold failed")
    validator.close(report["target_normalization"]["maximum_relative_error"], target_maximum, "reported target-normalization maximum")

    validator.require(len(timing_rows) == 16, "timing row count changed")
    timing_lookup: set[tuple[int, int]] = set()
    for row in timing_rows:
        key = (int(row["replicate"]), int(row["sample_index"]))
        validator.require(key not in timing_lookup, "duplicate timing row")
        timing_lookup.add(key)
        validator.require(row["order"] == config["timing_order_by_replicate"][str(key[0])], "paired timing order drift")
        for name in (
            "scalar_seconds",
            "view_block_seconds",
            "voxel_factor_seconds",
            "graph_seconds",
            "scalar_setup_seconds",
            "view_block_setup_seconds",
            "voxel_factor_setup_seconds",
            "scalar_solver_only_seconds",
            "view_block_solver_only_seconds",
            "voxel_factor_solver_only_seconds",
        ):
            validator.require(_finite(row[name], name) > 0.0, f"{name} must be positive")
        for mode in MODES:
            validator.require(row[f"{mode}_setup_logical_calls"] == setup_ledger, f"{mode} timed setup logical ledger drift")
            validator.require(row[f"{mode}_setup_physical_calls"] == setup_ledger, f"{mode} timed setup physical ledger drift")
    validator.require(timing_lookup == expected_samples, "timing coverage drift")

    validator.require(len(call_rows) == 48, "factor call-ledger row count changed")
    call_lookup: set[tuple[int, int, str]] = set()
    solver_ledger = _expected_ledger(signed_forward=32, signed_transpose=32)
    scorer_ledger = _expected_ledger(signed_forward=4)
    for row in call_rows:
        key = (int(row["replicate"]), int(row["sample_index"]), str(row["mode"]))
        validator.require(key not in call_lookup, "duplicate factor call-ledger row")
        call_lookup.add(key)
        validator.require(row["solver_logical"] == solver_ledger, "solver logical ledger drift")
        validator.require(row["solver_physical"] == solver_ledger, "solver physical ledger drift")
        validator.require(row["scorer_physical"] == scorer_ledger, "scorer physical ledger drift")
    validator.require(
        call_lookup == {(replicate, sample, mode) for replicate, sample in expected_samples for mode in MODES},
        "factor call-ledger coverage drift",
    )
    return timing_rows


def _validate_graph_replay(
    rows: Sequence[Mapping[str, Any]],
    parent_path: Path,
    report: Mapping[str, Any],
    validator: Validator,
) -> None:
    expected: dict[tuple[int, int, int], dict[str, float]] = {}
    with parent_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            identifier = str(row["candidate_id"])
            if not identifier.startswith("graph_s3_k"):
                continue
            iteration = int(identifier.removeprefix("graph_s3_k"))
            if iteration not in CHECKPOINTS:
                continue
            key = (int(row["replicate"]), int(row["sample_index"]), iteration)
            if key in expected:
                raise ValidationError("duplicate parent graph row")
            expected[key] = {
                "field_relative_l2": _finite(row["field_relative_l2"], "parent field error"),
                "gradient_relative_l2": _finite(row["gradient_relative_l2"], "parent gradient error"),
                "front_top10_f1": _finite(row["front_top10_f1"], "parent front F1"),
            }
    validator.require(len(expected) == 64, "parent graph coverage changed")
    maximum = 0.0
    for row in rows:
        if row["method"] != "graph_pcgls":
            continue
        key = (row["replicate"], row["sample_index"], row["iterations"])
        validator.require(key in expected, "graph replay row has no frozen parent")
        for metric in ("field_relative_l2", "gradient_relative_l2", "front_top10_f1"):
            maximum = max(maximum, abs(float(row[metric]) - expected[key][metric]))
    graph_report = report["graph_replay"]
    validator.require(
        graph_report.get("status")
        == "NONBINDING_BATCH_SINGLETON_TRACEABILITY_DIAGNOSTIC",
        "graph replay diagnostic status drift",
    )
    validator.require(graph_report.get("binding") is False, "graph replay became binding")
    validator.require(
        graph_report.get("parent_batch_rows_used_for_scoring") is False,
        "parent batch rows entered scoring",
    )
    validator.require(
        graph_report.get("scored_control")
        == "same_run_single_sample_exact_k_graph_pcgls",
        "scored graph control changed",
    )
    validator.require(
        graph_report.get("diagnostic_report_sha256")
        == AMENDMENT["graph_diagnostic_report_sha256"],
        "graph diagnostic provenance drift",
    )
    validator.close(
        graph_report["maximum_absolute_metric_difference"],
        maximum,
        "reported graph replay maximum",
        abs_tol=1e-8,
    )


def validate_release(
    *,
    root: Path,
    config_path: Path,
    evidence_dir: Path,
) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    evidence_dir = evidence_dir.resolve()
    validator = Validator()
    config = _load_config(config_path, validator)
    checksums = parse_checksum_manifest(evidence_dir / "checksums.sha256")
    for name, digest in checksums.items():
        path = evidence_dir / name
        validator.require(path.is_file() and not path.is_symlink(), f"release file missing: {name}")
        validator.require(file_sha256(path) == digest, f"release checksum mismatch: {name}")
    report = load_strict_json(evidence_dir / "report.json")
    audit = load_strict_json(evidence_dir / "audit.json")
    validator.require(report.get("schema_version") == REPORT_SCHEMA, "report schema drift")
    validator.require(report.get("factor_metric_schema") == FACTOR_SCHEMA, "factor schema drift")
    validator.require(report.get("status") in ALLOWED_GATE_STATUSES, "invalid Gate-B status")
    validator.require(report.get("amendment") == AMENDMENT, "reported amendment drift")
    validator.require(report.get("evidence_role") == "OPENED_DEVELOPMENT_MECHANISM_GATE", "report evidence role drift")
    validator.require(report.get("claim_boundary") == CLAIM_BOUNDARY, "report claim boundary drift")
    validator.require(
        report.get("scientific_claim_boundary") == "OPENED_SYNTHETIC_MECHANISM_GATE_ONLY_NO_FRESH_REAL_FLOW_OR_WIN_CLAIM",
        "scientific claim boundary drift",
    )
    validator.require(file_sha256(config_path) == report.get("config_sha256"), "config file hash mismatch")
    commit = report.get("source_commit")
    validator.require(isinstance(commit, str) and COMMIT_PATTERN.fullmatch(commit) is not None, "source commit is malformed")
    validator.require(report.get("source_sha256") == config["source_sha256"], "reported source hash map drift")
    config_relative = config_path.relative_to(root).as_posix()
    validator.require(
        hashlib.sha256(_git_blob(root, commit, config_relative)).hexdigest() == report["config_sha256"],
        "recorded commit does not contain the executed config bytes",
    )
    for key in sorted(SOURCE_KEYS):
        raw_path = config["source_paths"][key]
        expected_hash = config["source_sha256"][key]
        validator.require(
            hashlib.sha256(_git_blob(root, commit, raw_path)).hexdigest() == expected_hash,
            f"recorded commit source hash mismatch: {key}",
        )
        current_path = _safe_repository_path(root, raw_path)
        validator.require(current_path.is_file(), f"current source missing: {key}")
        validator.require(file_sha256(current_path) == expected_hash, f"current source drift: {key}")
    preflight = report.get("test_preflight")
    validator.require(isinstance(preflight, dict), "reported preflight missing")
    validator.require(preflight.get("status") == "PASS", "reported preflight did not pass")
    validator.require(preflight.get("case_count") == config["test_preflight"]["expected_case_count"], "preflight test count drift")
    validator.require(preflight.get("node_manifest_sha256") == config["test_preflight"]["expected_node_manifest_sha256"], "preflight manifest drift")
    validator.require(
        preflight.get("counts")
        == {"tests": preflight["case_count"], "failures": 0, "errors": 0, "skipped": 0},
        "preflight JUnit counts drift",
    )

    rows = _read_metric_rows(evidence_dir / "metric_rows.csv", validator)
    validator.require(report.get("row_count") == len(rows), "reported metric row count drift")
    expected_summaries = _summaries(rows)
    actual_summaries = sorted(report.get("summaries", []), key=lambda row: (row["method"], row["iterations"]))
    _compare_nested(validator, actual_summaries, expected_summaries, "summaries")
    timing_rows = _validate_audit(audit, report, config, validator)
    validator.require(report.get("timing_pair_count") == len(timing_rows), "reported timing count drift")
    expected_decision = recompute_decision(rows, timing_rows, config["thresholds"])
    _compare_nested(validator, report.get("decision"), expected_decision, "decision")
    validator.require(report["status"] == expected_decision["status"], "top-level status differs from recomputation")
    _validate_graph_replay(
        rows,
        _safe_repository_path(root, config["source_paths"]["parent_metric_rows"]),
        report,
        validator,
    )
    validator.require(report.get("timing_contract") == TIMING_CONTRACT, "timing claim boundary drift")
    validator.require(report.get("operator_budget_contract") == OPERATOR_BUDGET, "reported operator budget drift")
    validator.require(report.get("objective_contract") == OBJECTIVE_CONTRACT, "reported objective contract drift")
    validator.require(report.get("invalidity_policy") == INVALIDITY_POLICY, "reported invalidity policy drift")
    return {
        "schema_version": VALIDATOR_SCHEMA,
        "status": VALIDATOR_STATUS,
        "gate_b_status": expected_decision["status"],
        "all_gate_b_gates_passed": expected_decision["all_gates_passed"],
        "source_commit": commit,
        "config_sha256": report["config_sha256"],
        "release_checksums_sha256": file_sha256(evidence_dir / "checksums.sha256"),
        "independent_check_count": validator.check_count,
        "metric_row_count": len(rows),
        "timing_pair_count": len(timing_rows),
        "claim_boundary": CLAIM_BOUNDARY,
        "algorithm_superiority_claim_authorized": False,
    }


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT
        / "demo_t16_operator/configs/psu_b0_factor_pdhg_gate_b_v4_a_only_connectivity_amendment.json",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=REPOSITORY_ROOT / "demo_t16_operator/results/psu_b0_factor_pdhg_gate_b",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    evidence = args.evidence.resolve()
    report = validate_release(
        root=REPOSITORY_ROOT,
        config_path=args.config,
        evidence_dir=evidence,
    )
    output = args.output.resolve() if args.output else evidence / "validation_report.json"
    output.write_bytes(_canonical_bytes(report) + b"\n")
    output.with_suffix(".sha256").write_text(
        f"{file_sha256(output)}  {output.name}\n",
        encoding="ascii",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
