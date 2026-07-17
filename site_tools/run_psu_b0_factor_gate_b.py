#!/usr/bin/env python3
"""Run the frozen deterministic PSU-B0 factor-metric Gate-B screen.

The screen reuses the already-opened two-replicate PSU geometry/noise data
from the scalar-PDHG study.  It does not open fresh seeds.  Graph-PCGLS is
independently replayed one sample at a time, and only then are three true
data-only PDHG trajectories executed from the same absolute-data majorizer:
global scalar, per-view dual block, and per-voxel primal factor.

The single-sample graph replay is intentional: the attested factor pipeline
currently accepts one whitening-scale instance, so a batch-eight graph timing
would be an implementation-confounded control.  Setup, solver, scorer, and
paired timing calls are recorded separately.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import torch

from demo_t16_operator.detector_covariance_whitening import (
    WhitenedMeasurementOperator,
)
from demo_t16_operator.detector_graph_covariance import (
    detector_graph_spectral_basis,
)
from demo_t16_operator.psu_b0_classical_baselines import (
    GeneralizedSobolevDirection,
    preconditioned_cgls_trajectory,
)
from demo_t16_operator.psu_b0_detector_graph_features import (
    build_detector_knn_graph,
)
from demo_t16_operator.psu_b0_factor_majorizer_pipeline import (
    FactorPipelineCallLedger,
)
from demo_t16_operator.psu_b0_gate_b_data_only import (
    GATE_B_DATA_ONLY_SCHEMA,
    SingleSampleViewLocalWhitening,
    build_single_sample_factor_setup,
    factor_state_to_volume,
    run_gate_b_data_only_trajectory,
)
from demo_t16_operator.psu_b0_streaming_operator import (
    zero_outer_boundary_support,
)
from site_tools.run_psu_b0_pdhg_screen import (
    OPENED_REPLICATES,
    REACTION_FAMILIES,
    _build_replicate_context,
)
from site_tools.run_psu_b0_real_interface_audit import (
    _make_operator,
    load_real_support_geometry,
)
from site_tools.run_psu_b0_spectral_preconditioner_pilot import (
    _field_metrics,
)


SCHEMA = "psu-b0-factor-pdhg-gate-b-report-1.3-a-only-connectivity-amendment"
CONFIG_SCHEMA = "psu-b0-factor-pdhg-gate-b-config-1.3-a-only-connectivity-amendment"
CHECKPOINTS = (4, 8, 16, 32)
METHODS = (
    "scalar_a_only_pdhg",
    "view_block_a_only_pdhg",
    "voxel_factor_a_only_pdhg",
    "graph_pcgls",
)
MODES = ("scalar", "view_block", "voxel_factor")

FROZEN_RUNTIME_SHAPE = {
    "grid_size": 16,
    "view_count": 9,
    "rays_per_view": 256,
    "finite_aperture_sample_count": 8,
    "measurement_count": 4608,
    "support_active_voxel_count": 2744,
    "data_coupled_voxel_count": 2322,
}

FROZEN_DATA_ROLE = {
    "geometry": "REAL_PSU_DETECTOR_GEOMETRY",
    "fields": "ANALYTIC_REACTION_PHANTOMS_WITH_CLEAN_TRUTH",
    "noise": "SYNTHETIC_CORRELATED_GRAPH_HEAT",
    "scale_by_view": "CLEAN_TRUTH_DERIVED_ORACLE_DEVELOPMENT_ONLY",
    "statistical_unit": "TWO_REPLICATES_OF_EIGHT_PAIRED_MORPHOLOGIES_NOT_16_IID",
}

FROZEN_OPERATOR_BUDGET = {
    "zero_initialization": True,
    "exact_forward_calls_equal_checkpoint": True,
    "exact_adjoint_calls_equal_checkpoint": True,
    "tv_forward_calls_per_iteration": 0,
    "tv_adjoint_calls_per_iteration": 0,
    "graph_uses_exact_same_k": True,
    "best_less_than_or_equal_k_selection_forbidden": True,
    "early_stopping_forbidden": True,
}

FROZEN_OBJECTIVE_CONTRACT = {
    "factor_unit": "0.5*||A0*z-b0||^2",
    "graph_unit": "0.5*||A*x-b||^2",
    "conversion": "J_graph=amplitude_scale^2*4608*J_factor",
    "cross_method_raw_objective_ranking_forbidden": True,
}

FROZEN_TIMING_CONTRACT = {
    "primary_scope": "per_field_one_shot_setup_plus_k32_solver",
    "paired_execution": "serial_single_sample",
    "synchronization": "before_and_after_each_timed_call",
    "counterbalanced_by_replicate": True,
    "scoring_hashing_and_plotting_inside_timed_region": False,
    "process_cold_start_claimed": False,
    "multi_frame_amortized_speed_claimed": False,
}

FROZEN_INVALIDITY_POLICY = {
    "source_hash_mismatch": "INVALID_NO_RANKING",
    "test_preflight_failure": "INVALID_NO_RANKING",
    "operator_or_target_equivalence_failure": "INVALID_NO_RANKING",
    "missing_duplicate_nan_skip_or_oom": "INVALID_NO_RANKING",
    "postopen_protocol_mutation": "INVALID_NO_RANKING",
}

FROZEN_TEST_SELECTORS = (
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
FROZEN_TEST_CASE_COUNT = 41
FROZEN_TEST_NODE_MANIFEST_SHA256 = (
    "d2e2155e2053234bf445f3c41b38565976f9619581515cbb843aaf918b126826"
)

FROZEN_AMENDMENT = {
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

FROZEN_THRESHOLDS = {
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

SOURCE_KEYS = (
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
)


def _reject_constant(raw: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {raw}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(
            handle,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().contiguous().cpu().numpy()
    digest = hashlib.sha256()
    digest.update(repr((tuple(array.shape), str(array.dtype))).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _ledger_delta(
    after: FactorPipelineCallLedger,
    before: FactorPipelineCallLedger,
) -> dict[str, int]:
    return {
        name: int(getattr(after, name) - getattr(before, name))
        for name in FactorPipelineCallLedger.__dataclass_fields__
    }


def load_gate_b_config(path: Path) -> dict[str, Any]:
    config = _load_json(path)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("Gate-B config schema mismatch")
    if config.get("status") != "FROZEN_POST_SETUP_DIAGNOSTIC_PRE_FACTOR_PROTOCOL":
        raise ValueError("Gate-B v4 protocol must be frozen before factor performance")
    if config.get("amendment") != FROZEN_AMENDMENT:
        raise ValueError("Gate-B infrastructure amendment changed")
    if config.get("evidence_role") != "OPENED_DEVELOPMENT_MECHANISM_GATE":
        raise ValueError("Gate-B evidence role must remain opened development")
    if tuple(config.get("replicate_indices", ())) != OPENED_REPLICATES:
        raise ValueError("Gate-B replicate indices are frozen to 0 and 8")
    if tuple(config.get("reaction_families", ())) != REACTION_FAMILIES:
        raise ValueError("Gate-B reaction family order changed")
    if tuple(config.get("checkpoints", ())) != CHECKPOINTS:
        raise ValueError("Gate-B checkpoints changed")
    if tuple(config.get("methods", ())) != METHODS:
        raise ValueError("Gate-B method set changed")
    if tuple(config.get("factor_modes", ())) != MODES:
        raise ValueError("Gate-B factor mode set changed")
    if config.get("expected_runtime_shape") != FROZEN_RUNTIME_SHAPE:
        raise ValueError("Gate-B runtime shape contract changed")
    if config.get("data_role") != FROZEN_DATA_ROLE:
        raise ValueError("Gate-B data-role contract changed")
    if config.get("operator_budget") != FROZEN_OPERATOR_BUDGET:
        raise ValueError("Gate-B operator-budget contract changed")
    if config.get("objective_contract") != FROZEN_OBJECTIVE_CONTRACT:
        raise ValueError("Gate-B objective-unit contract changed")
    if config.get("timing_contract") != FROZEN_TIMING_CONTRACT:
        raise ValueError("Gate-B timing contract changed")
    if config.get("invalidity_policy") != FROZEN_INVALIDITY_POLICY:
        raise ValueError("Gate-B invalidity policy changed")
    preflight = config.get("test_preflight")
    if not isinstance(preflight, Mapping):
        raise ValueError("Gate-B test_preflight must be an object")
    if tuple(preflight.get("selectors", ())) != FROZEN_TEST_SELECTORS:
        raise ValueError("Gate-B test selector manifest changed")
    if preflight.get("expected_case_count") != FROZEN_TEST_CASE_COUNT:
        raise ValueError("Gate-B expected test count changed")
    if (
        preflight.get("expected_node_manifest_sha256")
        != FROZEN_TEST_NODE_MANIFEST_SHA256
    ):
        raise ValueError("Gate-B test node manifest hash changed")
    if config.get("thresholds") != FROZEN_THRESHOLDS:
        raise ValueError("Gate-B thresholds differ from the frozen runner")
    solver = config.get("solver")
    if not isinstance(solver, Mapping):
        raise ValueError("solver must be an object")
    if float(solver.get("eta", math.nan)) != 0.7:
        raise ValueError("Gate-B eta is frozen to 0.7")
    if float(solver.get("theta", math.nan)) != 1.0:
        raise ValueError("Gate-B theta is frozen to 1.0")
    if float(solver.get("regularization_weight", math.nan)) != 0.0:
        raise ValueError("Gate-B must remain data-only")
    claim = config.get("claim_boundary")
    if not isinstance(claim, Mapping):
        raise ValueError("claim_boundary must be an object")
    required_false = {
        "fresh_seed_set_opened",
        "experimental_flow_truth_used",
        "real_flowoff_repeats_used",
        "neural_operator_training_authorized",
        "algorithm_superiority_claimed",
    }
    required_true = {
        "real_psu_detector_geometry_used",
        "analytic_reaction_fields_used",
        "synthetic_correlated_noise_used",
        "scale_by_view_uses_clean_truth",
    }
    if set(claim) != required_false | required_true:
        raise ValueError("Gate-B claim boundary key set changed")
    if any(claim.get(key) is not False for key in required_false):
        raise ValueError("Gate-B claim boundary was broadened")
    if claim.get("real_psu_detector_geometry_used") is not True:
        raise ValueError("Gate-B must retain the real PSU detector geometry label")
    if claim.get("analytic_reaction_fields_used") is not True:
        raise ValueError("Gate-B must disclose analytic reaction fields")
    if claim.get("synthetic_correlated_noise_used") is not True:
        raise ValueError("Gate-B must disclose synthetic correlated noise")
    if claim.get("scale_by_view_uses_clean_truth") is not True:
        raise ValueError("Gate-B must disclose truth-derived development scaling")
    timing_order = config.get("timing_order_by_replicate")
    if timing_order != {
        "0": ["scalar", "view_block", "voxel_factor", "graph"],
        "8": ["graph", "voxel_factor", "view_block", "scalar"],
    }:
        raise ValueError("Gate-B paired timing order changed")
    seed = config.get("operator_equivalence_seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed != 2026071702:
        raise ValueError("Gate-B operator-equivalence seed changed")

    paths = config.get("source_paths")
    hashes = config.get("source_sha256")
    if not isinstance(paths, Mapping) or set(paths) != set(SOURCE_KEYS):
        raise ValueError("Gate-B source_paths key set is incomplete")
    if not isinstance(hashes, Mapping) or set(hashes) != set(SOURCE_KEYS):
        raise ValueError("Gate-B source_sha256 key set is incomplete")
    for key in SOURCE_KEYS:
        raw_path = paths[key]
        raw_hash = hashes[key]
        if not isinstance(raw_path, str) or not raw_path or Path(raw_path).is_absolute():
            raise ValueError(f"source path {key} must be repository-relative")
        if not isinstance(raw_hash, str) or len(raw_hash) != 64:
            raise ValueError(f"source hash {key} must be SHA-256")
    return config


def validate_sources(root: Path, config: Mapping[str, Any]) -> dict[str, str]:
    paths = config["source_paths"]
    hashes = config["source_sha256"]
    observed: dict[str, str] = {}
    for key in SOURCE_KEYS:
        path = (root / str(paths[key])).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"source path {key} escapes repository") from error
        if not path.is_file():
            raise ValueError(f"source file {key} is missing")
        observed[key] = _sha256(path)
        if observed[key] != hashes[key]:
            raise ValueError(f"source hash mismatch for {key}")
    return observed


def validate_clean_repository(root: Path) -> str:
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise ValueError("formal Gate-B execution requires a clean worktree")
    return _git(root, "rev-parse", "HEAD")


def run_test_preflight(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Collect and replay the frozen mechanics tests before data are opened."""

    preflight = config["test_preflight"]
    selectors = [str(value) for value in preflight["selectors"]]
    python = root / ".venv/bin/python"
    if not python.is_file():
        raise ValueError("frozen Gate-B Python executable is missing")
    environment = os.environ.copy()
    environment.update(
        {
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTORCH_ENABLE_MPS_FALLBACK": "0",
        }
    )
    common = [
        str(python),
        "-m",
        "pytest",
        "--no-header",
        "-p",
        "no:cacheprovider",
        "-o",
        "addopts=",
        *selectors,
    ]
    collected = subprocess.run(
        [*common[:3], "--collect-only", "-q", *common[3:]],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if collected.returncode != 0:
        raise ValueError(f"Gate-B test collection failed:\n{collected.stdout}{collected.stderr}")
    nodes = [
        line.strip()
        for line in collected.stdout.splitlines()
        if "::" in line and not line.lstrip().startswith("<")
    ]
    if len(nodes) != len(set(nodes)):
        raise ValueError("Gate-B test collection contains duplicate nodes")
    manifest = "".join(f"{node}\n" for node in nodes)
    manifest_sha = hashlib.sha256(manifest.encode("utf-8")).hexdigest()
    if len(nodes) != int(preflight["expected_case_count"]):
        raise ValueError("Gate-B collected test count changed")
    if manifest_sha != str(preflight["expected_node_manifest_sha256"]):
        raise ValueError("Gate-B collected node manifest changed")

    with tempfile.TemporaryDirectory(prefix="psu-b0-gate-b-tests-") as directory:
        junit = Path(directory) / "pytest.xml"
        executed = subprocess.run(
            [*common, "-q", f"--junitxml={junit}"],
            cwd=root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        if executed.returncode != 0 or not junit.is_file():
            raise ValueError(f"Gate-B test replay failed:\n{executed.stdout}{executed.stderr}")
        root_node = ET.parse(junit).getroot()
        suite = root_node if root_node.tag == "testsuite" else root_node.find("testsuite")
        if suite is None:
            raise ValueError("Gate-B JUnit report has no testsuite")
        counts = {
            name: int(suite.attrib.get(name, 0))
            for name in ("tests", "failures", "errors", "skipped")
        }
    if counts != {
        "tests": len(nodes),
        "failures": 0,
        "errors": 0,
        "skipped": 0,
    }:
        raise ValueError("Gate-B preflight tests did not all pass without skip")
    return {
        "status": "PASS",
        "case_count": len(nodes),
        "node_manifest_sha256": manifest_sha,
        "counts": counts,
        "environment": {
            "plugin_autoload": False,
            "user_site": False,
            "mps_fallback": False,
        },
    }


def _float(row: Mapping[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"source metric {key} must be finite")
    return value


def load_parent_metric_rows(
    path: Path,
) -> tuple[dict[tuple[int, int, int], dict[str, Any]], dict[tuple[int, int, int], dict[str, Any]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))
    scalar: dict[tuple[int, int, int], dict[str, Any]] = {}
    graph: dict[tuple[int, int, int], dict[str, Any]] = {}
    for row in source:
        identifier = str(row["candidate_id"])
        target: dict[tuple[int, int, int], dict[str, Any]] | None = None
        prefix = ""
        if identifier.startswith("pdhg_data_only_k"):
            target = scalar
            prefix = "pdhg_data_only_k"
        elif identifier.startswith("graph_s3_k"):
            target = graph
            prefix = "graph_s3_k"
        if target is None:
            continue
        iteration = int(identifier.removeprefix(prefix))
        if iteration not in CHECKPOINTS:
            continue
        key = (int(row["replicate"]), int(row["sample_index"]), iteration)
        if key in target:
            raise ValueError(f"duplicate parent metric key {key}")
        target[key] = {
            "replicate": key[0],
            "sample_index": key[1],
            "reaction_family": str(row["reaction_family"]),
            "iterations": iteration,
            "field_relative_l2": _float(row, "field_relative_l2"),
            "gradient_relative_l2": _float(row, "gradient_relative_l2"),
            "front_top10_f1": _float(row, "front_top10_f1"),
        }
    expected = len(OPENED_REPLICATES) * len(REACTION_FAMILIES) * len(CHECKPOINTS)
    if len(scalar) != expected or len(graph) != expected:
        raise ValueError("parent scalar/graph metric rows are incomplete")
    expected_keys = {
        (replicate, sample, iteration)
        for replicate in OPENED_REPLICATES
        for sample in range(len(REACTION_FAMILIES))
        for iteration in CHECKPOINTS
    }
    if set(scalar) != expected_keys or set(graph) != expected_keys:
        raise ValueError("parent scalar/graph key coverage changed")
    return scalar, graph


def _metric_row(
    *,
    replicate: int,
    sample_index: int,
    family: str,
    method: str,
    iteration: int,
    metrics: Mapping[str, torch.Tensor],
    data_objective: float | None,
    objective_unit: str,
    objective_scale_to_physical: float | None,
    physical_equivalent_objective: float | None,
    trajectory_elapsed_seconds: float | None,
    source: str,
) -> dict[str, Any]:
    return {
        "replicate": int(replicate),
        "sample_index": int(sample_index),
        "reaction_family": str(family),
        "method": str(method),
        "iterations": int(iteration),
        "forward_calls": int(iteration),
        "adjoint_calls": int(iteration),
        "field_relative_l2": float(metrics["field_relative_l2"][0].detach().cpu()),
        "gradient_relative_l2": float(
            metrics["gradient_relative_l2"][0].detach().cpu()
        ),
        "front_top10_f1": float(metrics["front_top10_f1"][0].detach().cpu()),
        "whitened_data_objective": data_objective,
        "objective_unit": str(objective_unit),
        "objective_scale_to_physical": objective_scale_to_physical,
        "physical_equivalent_whitened_objective": physical_equivalent_objective,
        "trajectory_elapsed_seconds": trajectory_elapsed_seconds,
        "source": str(source),
    }


def _build_runtime(
    *,
    root: Path,
    config: Mapping[str, Any],
    view_root: Path,
    device: torch.device,
) -> tuple[Any, list[dict[str, Any]], GeneralizedSobolevDirection]:
    parent_config = _load_json(root / str(config["source_paths"]["parent_pdhg_config"]))
    smoke = _load_json(root / str(parent_config["source_smoke_config"]))
    multiseed = _load_json(root / str(parent_config["source_multiseed_config"]))
    anchors = _load_json(root / str(parent_config["source_runtime_anchor_manifest"]))
    geometry_config = smoke["geometry"]
    view_count = int(geometry_config["view_count"])
    rays_per_view = int(geometry_config["rays_per_view"])
    grid_size = int(geometry_config["grid_size"])
    geometry, provenance = load_real_support_geometry(
        view_root,
        rays_per_view=rays_per_view,
        sample_count=int(geometry_config["finite_aperture_sample_count"]),
    )
    if tuple(int(row["view_id_zero_based"]) for row in provenance) != tuple(
        range(view_count)
    ):
        raise ValueError("PSU view provenance order changed")
    operator = _make_operator(geometry, grid_size=grid_size, dtype=torch.float32)
    operator.support.copy_(zero_outer_boundary_support((grid_size,) * 3))
    operator = operator.to(device)

    detector_config = _load_json(root / str(smoke["source_detector_config"]))
    graph_config = detector_config["detector_graph"]
    graph = build_detector_knn_graph(
        geometry["detector_xy"],
        view_count=view_count,
        rays_per_view=rays_per_view,
        neighbor_count=int(graph_config["neighbor_count"]),
        least_squares_ridge=float(graph_config["least_squares_ridge"]),
    )
    eigenpairs = [
        detector_graph_spectral_basis(graph, view_index=view)
        for view in range(view_count)
    ]
    graph_family_index = [str(value) for value in smoke["data"]["noise_families"]].index(
        "graph_heat"
    )
    contexts = [
        _build_replicate_context(
            root=root,
            config=parent_config,
            runtime_anchor_manifest=anchors,
            smoke=smoke,
            multiseed=multiseed,
            replicate=replicate,
            operator=operator,
            geometry=geometry,
            eigenpairs=eigenpairs,
            graph=graph,
            graph_family_index=graph_family_index,
            device=device,
        )
        for replicate in OPENED_REPLICATES
    ]
    parent_solver = parent_config["solver"]
    direction = GeneralizedSobolevDirection(
        (grid_size,) * 3,
        strength=float(parent_solver["sobolev_strength"]),
        epsilon=float(parent_solver["sobolev_epsilon"]),
    ).to(device)
    return operator, contexts, direction


def _single_graph_operator(
    operator: Any,
    context: Mapping[str, Any],
    sample_index: int,
) -> WhitenedMeasurementOperator:
    whitening = SingleSampleViewLocalWhitening(
        context["graph_operator"].whitening,
        sample_index,
    )
    return WhitenedMeasurementOperator(operator, whitening).to(operator.support.device)


def _run_graph_replay(
    *,
    operator: Any,
    contexts: Sequence[Mapping[str, Any]],
    direction: GeneralizedSobolevDirection,
    parent_graph: Mapping[tuple[int, int, int], Mapping[str, Any]],
    threshold: float | None,
    device: torch.device,
) -> tuple[list[dict[str, Any]], float]:
    rows: list[dict[str, Any]] = []
    maximum_difference = 0.0
    for context in contexts:
        replicate = int(context["replicate"])
        for sample_index, family in enumerate(context["families"]):
            graph_operator = _single_graph_operator(operator, context, sample_index)
            graph_operator.reset_call_counts()
            trajectory = preconditioned_cgls_trajectory(
                graph_operator,
                context["graph_prepared"][sample_index : sample_index + 1],
                sigma_by_view=context["ones_sigma"][sample_index : sample_index + 1],
                view_mask=context["ones_mask"][sample_index : sample_index + 1],
                rays_per_view=int(context["rays_per_view"]),
                checkpoint_stages=CHECKPOINTS,
                preconditioner=direction,
            )
            if graph_operator.call_report() != {
                "forward_calls": max(CHECKPOINTS),
                "adjoint_calls": max(CHECKPOINTS),
            }:
                raise ValueError("single-sample graph replay violated call budget")
            truth = context["truth"][sample_index : sample_index + 1]
            for iteration in CHECKPOINTS:
                metrics = _field_metrics(trajectory[iteration].volume, truth)
                row = _metric_row(
                    replicate=replicate,
                    sample_index=sample_index,
                    family=str(family),
                    method="graph_pcgls",
                    iteration=iteration,
                    metrics=metrics,
                    data_objective=float(
                        0.5
                        * torch.sum(trajectory[iteration].residual_uv.square())
                        .detach()
                        .cpu()
                    ),
                    objective_unit="graph_whitened_physical_field",
                    objective_scale_to_physical=1.0,
                    physical_equivalent_objective=float(
                        0.5
                        * torch.sum(trajectory[iteration].residual_uv.square())
                        .detach()
                        .cpu()
                    ),
                    trajectory_elapsed_seconds=None,
                    source="single_sample_graph_replay",
                )
                expected = parent_graph[(replicate, sample_index, iteration)]
                if str(expected["reaction_family"]) != str(family):
                    raise ValueError("graph replay family differs from frozen parent")
                for metric_name in (
                    "field_relative_l2",
                    "gradient_relative_l2",
                    "front_top10_f1",
                ):
                    difference = abs(float(row[metric_name]) - float(expected[metric_name]))
                    maximum_difference = max(maximum_difference, difference)
                rows.append(row)
    if threshold is not None and maximum_difference > float(threshold):
        raise ValueError(
            "single-sample graph replay differs from the frozen batch-eight release"
        )
    return rows, maximum_difference


def _relative_error(observed: torch.Tensor, expected: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm((observed - expected).flatten())
    denominator = torch.linalg.vector_norm(expected.flatten()).clamp_min(1e-12)
    return float((numerator / denominator).detach().cpu())


def _operator_equivalence_probe(
    *,
    setup: Any,
    graph_operator: WhitenedMeasurementOperator,
    seed: int,
    measurement_scale: float,
) -> dict[str, float]:
    generator = torch.Generator().manual_seed(int(seed))
    volume = torch.randn(
        (1, 1, *setup.pipeline.grid_shape),
        generator=generator,
        dtype=setup.pipeline.dtype,
    ).to(setup.pipeline.device)
    volume = volume * setup.pipeline.voxel_operator.support[None, None]
    active = setup.pipeline.restrict_active(volume)
    factor_forward = setup.pipeline.signed_data_forward(active)
    graph_forward = float(measurement_scale) * graph_operator(volume)

    dual = torch.randn(
        (1, setup.pipeline.ray_count, 2),
        generator=generator,
        dtype=setup.pipeline.dtype,
    ).to(setup.pipeline.device)
    factor_transpose = setup.pipeline.signed_data_transpose(dual)
    graph_full = float(measurement_scale) * graph_operator.adjoint(dual)
    graph_transpose = setup.pipeline.restrict_active(graph_full)
    return {
        "forward_relative_error": _relative_error(factor_forward, graph_forward),
        "transpose_relative_error": _relative_error(
            factor_transpose,
            graph_transpose,
        ),
    }


def _paired_graph_timing(
    *,
    operator: Any,
    context: Mapping[str, Any],
    sample_index: int,
    direction: GeneralizedSobolevDirection,
    device: torch.device,
) -> float:
    _synchronize(device)
    started = time.perf_counter()
    graph_operator = _single_graph_operator(operator, context, sample_index)
    graph_operator.reset_call_counts()
    preconditioned_cgls_trajectory(
        graph_operator,
        context["graph_prepared"][sample_index : sample_index + 1],
        sigma_by_view=context["ones_sigma"][sample_index : sample_index + 1],
        view_mask=context["ones_mask"][sample_index : sample_index + 1],
        rays_per_view=int(context["rays_per_view"]),
        checkpoint_stages=(max(CHECKPOINTS),),
        preconditioner=direction,
    )
    _synchronize(device)
    elapsed = time.perf_counter() - started
    if graph_operator.call_report() != {
        "forward_calls": max(CHECKPOINTS),
        "adjoint_calls": max(CHECKPOINTS),
    }:
        raise ValueError("paired graph timing violated call budget")
    return float(elapsed)


def _factor_one_shot_timing(
    *,
    operator: Any,
    context: Mapping[str, Any],
    sample_index: int,
    target: torch.Tensor,
    mode: str,
    eta: float,
    theta: float,
    device: torch.device,
) -> dict[str, Any]:
    """Time per-field metric setup plus one independent K=32 solve."""

    _synchronize(device)
    started = time.perf_counter()
    setup = build_single_sample_factor_setup(
        voxel_operator=operator,
        source_whitening=context["graph_operator"].whitening,
        sample_index=sample_index,
        measurement_scale=float(context["measurement_scale"]),
        eta=float(eta),
    )
    _synchronize(device)
    setup_elapsed = time.perf_counter() - started
    solver_started = time.perf_counter()
    setup.pipeline.reset_call_ledger()
    run_gate_b_data_only_trajectory(
        setup,
        target,
        checkpoints=(max(CHECKPOINTS),),
        mode=mode,
        theta=float(theta),
    )
    _synchronize(device)
    solver_elapsed = time.perf_counter() - solver_started
    expected = FactorPipelineCallLedger(
        signed_data_forward_calls=max(CHECKPOINTS),
        signed_data_transpose_calls=max(CHECKPOINTS),
    )
    if setup.pipeline.call_ledger() != expected:
        raise ValueError(f"{mode} one-shot timing call ledger mismatch")
    return {
        "setup_seconds": float(setup_elapsed),
        "solver_seconds": float(solver_elapsed),
        "total_seconds": float(setup_elapsed + solver_elapsed),
        "setup_logical_calls": setup.setup_call_ledger.__dict__,
        "setup_physical_calls": setup.setup_physical_call_ledger.__dict__,
    }


def _run_factor_mode(
    *,
    setup: Any,
    target: torch.Tensor,
    truth: torch.Tensor,
    output_scale: torch.Tensor,
    replicate: int,
    sample_index: int,
    family: str,
    mode: str,
    theta: float,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
    setup.pipeline.reset_call_ledger()
    physical_before = setup.pipeline.physical_call_ledger()
    _synchronize(device)
    started = time.perf_counter()
    checkpoints = run_gate_b_data_only_trajectory(
        setup,
        target,
        checkpoints=CHECKPOINTS,
        mode=mode,
        theta=float(theta),
    )
    _synchronize(device)
    elapsed = time.perf_counter() - started
    solver_logical = setup.pipeline.call_ledger()
    solver_physical = _ledger_delta(
        setup.pipeline.physical_call_ledger(),
        physical_before,
    )
    expected_solver = {
        "signed_data_forward_calls": max(CHECKPOINTS),
        "signed_data_transpose_calls": max(CHECKPOINTS),
        "absolute_data_forward_calls": 0,
        "absolute_data_transpose_calls": 0,
        "signed_tv_forward_calls": 0,
        "signed_tv_transpose_calls": 0,
        "absolute_tv_forward_calls": 0,
        "absolute_tv_transpose_calls": 0,
    }
    if solver_logical.__dict__ != expected_solver or solver_physical != expected_solver:
        raise ValueError(f"{mode} solver call ledger mismatch")

    rows: list[dict[str, Any]] = []
    scorer_before = setup.pipeline.physical_call_ledger()
    method_by_mode = {
        "scalar": "scalar_a_only_pdhg",
        "view_block": "view_block_a_only_pdhg",
        "voxel_factor": "voxel_factor_a_only_pdhg",
    }
    if mode not in method_by_mode:
        raise ValueError("unsupported A-only Gate-B mode")
    method = method_by_mode[mode]
    for iteration in CHECKPOINTS:
        state = checkpoints[iteration]
        normalized = factor_state_to_volume(setup, state)
        prediction = normalized * output_scale
        metrics = _field_metrics(prediction, truth)
        active = setup.pipeline.restrict_active(normalized)
        projected = setup.pipeline.signed_data_forward(active).reshape_as(target)
        objective = 0.5 * torch.sum((projected - target).square())
        normalized_objective = float(objective.detach().cpu())
        amplitude = float(output_scale.reshape(()).detach().cpu())
        objective_scale = amplitude * amplitude * target.numel()
        physical_objective = normalized_objective * objective_scale
        rows.append(
            _metric_row(
                replicate=replicate,
                sample_index=sample_index,
                family=family,
                method=method,
                iteration=iteration,
                metrics=metrics,
                data_objective=normalized_objective,
                objective_unit="normalized_b0_per_sample",
                objective_scale_to_physical=objective_scale,
                physical_equivalent_objective=physical_objective,
                trajectory_elapsed_seconds=float(elapsed),
                source="gate_b_production_factor_pipeline",
            )
        )
    scorer_physical = _ledger_delta(
        setup.pipeline.physical_call_ledger(),
        scorer_before,
    )
    expected_scorer = {name: 0 for name in expected_solver}
    expected_scorer["signed_data_forward_calls"] = len(CHECKPOINTS)
    if scorer_physical != expected_scorer:
        raise ValueError(f"{mode} scorer call ledger mismatch")
    return (
        rows,
        {
            "mode": mode,
            "solver_logical": solver_logical.__dict__,
            "solver_physical": solver_physical,
            "scorer_physical": scorer_physical,
        },
        float(elapsed),
    )


def _summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), int(row["iterations"]))].append(row)
    output: list[dict[str, Any]] = []
    for (method, iteration), values in sorted(grouped.items()):
        if len(values) != len(OPENED_REPLICATES) * len(REACTION_FAMILIES):
            raise ValueError(f"metric coverage is incomplete for {method} K={iteration}")
        output.append(
            {
                "method": method,
                "iterations": iteration,
                "sample_count": len(values),
                "mean_field_relative_l2": statistics.fmean(
                    float(row["field_relative_l2"]) for row in values
                ),
                "p90_field_relative_l2": float(
                    np.quantile(
                        [float(row["field_relative_l2"]) for row in values],
                        0.90,
                    )
                ),
                "mean_gradient_relative_l2": statistics.fmean(
                    float(row["gradient_relative_l2"]) for row in values
                ),
                "mean_front_top10_f1": statistics.fmean(
                    float(row["front_top10_f1"]) for row in values
                ),
            }
        )
    return output


def _decision(
    rows: Sequence[Mapping[str, Any]],
    timings: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    expected_row_keys = {
        (replicate, sample, method, iteration)
        for replicate in OPENED_REPLICATES
        for sample in range(len(REACTION_FAMILIES))
        for method in METHODS
        for iteration in CHECKPOINTS
    }
    lookup: dict[tuple[int, int, str, int], float] = {}
    for row in rows:
        key = (
            int(row["replicate"]),
            int(row["sample_index"]),
            str(row["method"]),
            int(row["iterations"]),
        )
        if key in lookup:
            raise ValueError(f"duplicate Gate-B metric row {key}")
        value = float(row["field_relative_l2"])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("Gate-B field error must be finite and nonnegative")
        if str(row["reaction_family"]) != REACTION_FAMILIES[key[1]]:
            raise ValueError("Gate-B reaction-family label changed")
        if int(row["forward_calls"]) != key[3] or int(row["adjoint_calls"]) != key[3]:
            raise ValueError("Gate-B metric row violates the fixed operator budget")
        lookup[key] = value
    if set(lookup) != expected_row_keys:
        raise ValueError("Gate-B metric row coverage is incomplete or unexpected")

    sample_keys = [
        (replicate, sample)
        for replicate in OPENED_REPLICATES
        for sample in range(len(REACTION_FAMILIES))
    ]

    def values(method: str, iteration: int) -> list[float]:
        return [lookup[(rep, sample, method, iteration)] for rep, sample in sample_keys]

    scalar = values("scalar_a_only_pdhg", 32)
    factor = values("voxel_factor_a_only_pdhg", 32)
    block = values("view_block_a_only_pdhg", 32)
    graph = values("graph_pcgls", 32)
    if min(scalar) <= 0.0 or min(block) <= 0.0 or min(graph) <= 0.0:
        raise ValueError("Gate-B paired percentage denominators must be positive")
    gains = [100.0 * (base - candidate) / base for base, candidate in zip(scalar, factor)]
    scalar_mean = statistics.fmean(scalar)
    factor_mean = statistics.fmean(factor)
    block_mean = statistics.fmean(block)
    graph_mean = statistics.fmean(graph)
    reduction_vs_scalar = 100.0 * (scalar_mean - factor_mean) / scalar_mean
    reduction_vs_block = 100.0 * (block_mean - factor_mean) / block_mean
    graph_gap = 100.0 * (factor_mean - graph_mean) / graph_mean
    replicate_reductions: dict[str, float] = {}
    for replicate in OPENED_REPLICATES:
        indices = [index for index, key in enumerate(sample_keys) if key[0] == replicate]
        replicate_scalar = statistics.fmean(scalar[index] for index in indices)
        replicate_factor = statistics.fmean(factor[index] for index in indices)
        replicate_reductions[str(replicate)] = (
            100.0 * (replicate_scalar - replicate_factor) / replicate_scalar
        )

    factor_means_by_k = [
        statistics.fmean(values("voxel_factor_a_only_pdhg", k))
        for k in CHECKPOINTS
    ]
    block_means_by_k = [
        statistics.fmean(values("view_block_a_only_pdhg", k))
        for k in CHECKPOINTS
    ]
    monotonic_tolerance = float(thresholds["mean_monotonic_tolerance"])
    monotone_factor = all(
        right <= left + monotonic_tolerance
        for left, right in zip(factor_means_by_k, factor_means_by_k[1:])
    )
    monotone_block = all(
        right <= left + monotonic_tolerance
        for left, right in zip(block_means_by_k, block_means_by_k[1:])
    )
    expected_timing_keys = {
        (replicate, sample)
        for replicate in OPENED_REPLICATES
        for sample in range(len(REACTION_FAMILIES))
    }
    observed_timing_keys: set[tuple[int, int]] = set()
    factor_ratios = [
        float(row["voxel_factor_seconds"]) / float(row["graph_seconds"])
        for row in timings
    ]
    for row, ratio in zip(timings, factor_ratios):
        key = (int(row["replicate"]), int(row["sample_index"]))
        if key in observed_timing_keys:
            raise ValueError(f"duplicate Gate-B timing pair {key}")
        observed_timing_keys.add(key)
        values_to_check = (
            float(row["scalar_seconds"]),
            float(row["view_block_seconds"]),
            float(row["voxel_factor_seconds"]),
            float(row["graph_seconds"]),
            ratio,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in values_to_check):
            raise ValueError("Gate-B paired timing values must be finite and positive")
    if observed_timing_keys != expected_timing_keys:
        raise ValueError("Gate-B timing coverage is incomplete or unexpected")
    median_wall_ratio = float(statistics.median(factor_ratios))
    positive_tolerance = float(thresholds["positive_gain_tolerance_percent"])

    gates = {
        "mean_reduction_vs_scalar": reduction_vs_scalar
        >= float(thresholds["factor_mean_reduction_vs_scalar_percent_min"]),
        "both_replicates_positive": all(
            value > positive_tolerance for value in replicate_reductions.values()
        ),
        "positive_sample_count": sum(value > positive_tolerance for value in gains)
        >= int(thresholds["factor_positive_sample_count_min"]),
        "worst_gain_vs_scalar": min(gains)
        >= float(thresholds["factor_worst_gain_vs_scalar_percent_min"]),
        "factor_mean_monotone": monotone_factor,
        "same_k_graph_gap": graph_gap
        <= float(thresholds["factor_graph_mean_error_gap_percent_max"]),
        "voxel_attribution_vs_view_block": reduction_vs_block
        >= float(thresholds["factor_mean_reduction_vs_view_block_percent_min"]),
        "single_sample_wall_time": median_wall_ratio
        <= float(thresholds["factor_wall_time_ratio_vs_single_sample_graph_max"]),
    }
    return {
        "status": (
            "GATE_B_E2_ORACLE_SCALE_CONDITIONING_SIGNAL_ONLY"
            if all(gates.values())
            else "GATE_B_E2_MECHANISM_NO_GO"
        ),
        "all_gates_passed": all(gates.values()),
        "gates": gates,
        "metrics": {
            "scalar_k32_mean_field_relative_l2": scalar_mean,
            "view_block_k32_mean_field_relative_l2": block_mean,
            "voxel_factor_k32_mean_field_relative_l2": factor_mean,
            "graph_k32_mean_field_relative_l2": graph_mean,
            "factor_mean_reduction_vs_scalar_percent": reduction_vs_scalar,
            "factor_mean_reduction_vs_view_block_percent": reduction_vs_block,
            "factor_graph_mean_error_gap_percent": graph_gap,
            "factor_positive_sample_count": sum(
                value > positive_tolerance for value in gains
            ),
            "factor_worst_gain_vs_scalar_percent": min(gains),
            "paired_gain_vs_scalar_percent": {
                f"r{replicate}_s{sample}": gain
                for (replicate, sample), gain in zip(sample_keys, gains)
            },
            "replicate_reduction_vs_scalar_percent": replicate_reductions,
            "factor_mean_field_by_checkpoint": dict(zip(map(str, CHECKPOINTS), factor_means_by_k)),
            "view_block_mean_field_by_checkpoint": dict(zip(map(str, CHECKPOINTS), block_means_by_k)),
            "view_block_mean_monotone": monotone_block,
            "factor_median_wall_time_ratio_vs_single_sample_graph": median_wall_ratio,
        },
        "thresholds": dict(thresholds),
        "fm_cg_pdno_zero_init_smoke_authorized": all(gates.values()),
        "algorithm_superiority_claim_authorized": False,
    }


def run_gate_b(
    *,
    root: Path,
    config_path: Path,
    view_root: Path,
    device_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = load_gate_b_config(config_path)
    source_hashes = validate_sources(root, config)
    source_commit = validate_clean_repository(root)
    test_preflight = run_test_preflight(root, config)
    if device_name != "mps":
        raise ValueError("formal Gate-B device is frozen to mps")
    if not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    device = torch.device(device_name)
    torch.set_num_threads(8)

    legacy_scalar_parent, graph_parent = load_parent_metric_rows(
        root / str(config["source_paths"]["parent_metric_rows"])
    )
    if len(legacy_scalar_parent) != (
        len(OPENED_REPLICATES) * len(REACTION_FAMILIES) * len(CHECKPOINTS)
    ):
        raise ValueError("legacy scalar parent rows are incomplete")
    operator, contexts, direction = _build_runtime(
        root=root,
        config=config,
        view_root=view_root,
        device=device,
    )
    graph_rows, graph_replay_max_difference = _run_graph_replay(
        operator=operator,
        contexts=contexts,
        direction=direction,
        parent_graph=graph_parent,
        threshold=None,
        device=device,
    )

    setups: dict[tuple[int, int], Any] = {}
    equivalence_rows: list[dict[str, Any]] = []
    target_normalization_errors: list[float] = []
    target_normalization_rows: list[dict[str, Any]] = []
    for context in contexts:
        replicate = int(context["replicate"])
        for sample_index in range(len(REACTION_FAMILIES)):
            measurement_count = float(context["scalar_measurement_count"])
            expected_b0 = context["graph_prepared"][sample_index] / (
                context["amplitude_scale"][sample_index]
                * math.sqrt(measurement_count)
            )
            target_error = _relative_error(
                context["b0"][sample_index],
                expected_b0,
            )
            target_normalization_errors.append(target_error)
            target_normalization_rows.append(
                {
                    "replicate": replicate,
                    "sample_index": sample_index,
                    "relative_error": target_error,
                    "measurement_count": int(measurement_count),
                }
            )
            if target_error > float(
                config["thresholds"]["target_normalization_relative_error_max"]
            ):
                raise ValueError("normalized Gate-B target differs from parent b0")
            setup = build_single_sample_factor_setup(
                voxel_operator=operator,
                source_whitening=context["graph_operator"].whitening,
                sample_index=sample_index,
                measurement_scale=float(context["measurement_scale"]),
                eta=float(config["solver"]["eta"]),
            )
            runtime_shape = config["expected_runtime_shape"]
            if tuple(setup.pipeline.grid_shape) != (
                int(runtime_shape["grid_size"]),
                int(runtime_shape["grid_size"]),
                int(runtime_shape["grid_size"]),
            ):
                raise ValueError("Gate-B grid shape changed")
            if setup.pipeline.view_count != int(runtime_shape["view_count"]):
                raise ValueError("Gate-B view count changed")
            if setup.pipeline.rays_per_view != int(runtime_shape["rays_per_view"]):
                raise ValueError("Gate-B rays-per-view changed")
            if setup.pipeline.sample_count != int(
                runtime_shape["finite_aperture_sample_count"]
            ):
                raise ValueError("Gate-B aperture sample count changed")
            if setup.pipeline.n_active != int(
                runtime_shape["support_active_voxel_count"]
            ):
                raise ValueError("Gate-B support-active voxel count changed")
            if setup.active_primal_count != int(
                runtime_shape["data_coupled_voxel_count"]
            ):
                raise ValueError("Gate-B data-coupled voxel count changed")
            active_primal_sha = _tensor_sha256(setup.active_primal_indices)
            if active_primal_sha != str(
                config["amendment"]["active_primal_indices_sha256"]
            ):
                raise ValueError("Gate-B A-only active-primal mask changed")
            if int(torch.count_nonzero(setup.data_row_mask).detach().cpu()) != int(
                runtime_shape["measurement_count"]
            ):
                raise ValueError("Gate-B active data-row count changed")
            graph_operator = _single_graph_operator(operator, context, sample_index)
            probe = _operator_equivalence_probe(
                setup=setup,
                graph_operator=graph_operator,
                seed=int(config["operator_equivalence_seed"]) + 100 * replicate + sample_index,
                measurement_scale=float(context["measurement_scale"]),
            )
            if max(probe.values()) > float(
                config["thresholds"]["operator_equivalence_relative_error_max"]
            ):
                raise ValueError("factor and graph operators are not equivalent")
            equivalence_rows.append(
                {
                    "replicate": replicate,
                    "sample_index": sample_index,
                    **probe,
                    "active_support_count": int(setup.pipeline.n_active),
                    "reduced_primal_count": int(setup.active_primal_count),
                    "active_primal_indices_sha256": active_primal_sha,
                    "tau_minimum": float(
                        setup.tau_voxel_factor.detach().amin().cpu()
                    ),
                    "tau_maximum": float(
                        setup.tau_voxel_factor.detach().amax().cpu()
                    ),
                    "setup_logical_calls": setup.setup_call_ledger.__dict__,
                    "setup_physical_calls": setup.setup_physical_call_ledger.__dict__,
                }
            )
            setups[(replicate, sample_index)] = setup

    rows = list(graph_rows)
    timing_rows: list[dict[str, Any]] = []
    call_rows: list[dict[str, Any]] = []
    theta = float(config["solver"]["theta"])
    for context in contexts:
        replicate = int(context["replicate"])
        order = tuple(config["timing_order_by_replicate"][str(replicate)])
        if set(order) != {"scalar", "view_block", "voxel_factor", "graph"}:
            raise ValueError("timing order must contain the four paired methods")
        for sample_index, family in enumerate(context["families"]):
            setup = setups[(replicate, sample_index)]
            target = context["b0"][sample_index].reshape(
                setup.pipeline.view_count,
                setup.pipeline.rays_per_view,
                2,
            )
            truth = context["truth"][sample_index : sample_index + 1]
            output_scale = context["amplitude_scale"][sample_index].reshape(1, 1, 1, 1, 1)
            primary_timing: dict[str, Any] = {}
            for method in order:
                if method == "graph":
                    primary_timing[method] = {
                        "total_seconds": _paired_graph_timing(
                            operator=operator,
                            context=context,
                            sample_index=sample_index,
                            direction=direction,
                            device=device,
                        )
                    }
                else:
                    primary_timing[method] = _factor_one_shot_timing(
                        operator=operator,
                        context=context,
                        sample_index=sample_index,
                        target=target,
                        mode=method,
                        eta=float(config["solver"]["eta"]),
                        theta=theta,
                        device=device,
                    )

            solver_elapsed: dict[str, float] = {}
            staged_rows: dict[str, list[dict[str, Any]]] = {}
            for method in MODES:
                mode_rows, ledger, seconds = _run_factor_mode(
                    setup=setup,
                    target=target,
                    truth=truth,
                    output_scale=output_scale,
                    replicate=replicate,
                    sample_index=sample_index,
                    family=str(family),
                    mode=method,
                    theta=theta,
                    device=device,
                )
                staged_rows[method] = mode_rows
                solver_elapsed[method] = seconds
                call_rows.append(
                    {
                        "replicate": replicate,
                        "sample_index": sample_index,
                        **ledger,
                    }
                )
            rows.extend(staged_rows["scalar"])
            rows.extend(staged_rows["view_block"])
            rows.extend(staged_rows["voxel_factor"])
            timing_rows.append(
                {
                    "replicate": replicate,
                    "sample_index": sample_index,
                    "order": list(order),
                    "scalar_seconds": primary_timing["scalar"]["total_seconds"],
                    "view_block_seconds": primary_timing["view_block"]["total_seconds"],
                    "voxel_factor_seconds": primary_timing["voxel_factor"]["total_seconds"],
                    "graph_seconds": primary_timing["graph"]["total_seconds"],
                    "scalar_setup_seconds": primary_timing["scalar"]["setup_seconds"],
                    "view_block_setup_seconds": primary_timing["view_block"]["setup_seconds"],
                    "voxel_factor_setup_seconds": primary_timing["voxel_factor"]["setup_seconds"],
                    "scalar_setup_logical_calls": primary_timing["scalar"]["setup_logical_calls"],
                    "view_block_setup_logical_calls": primary_timing["view_block"]["setup_logical_calls"],
                    "voxel_factor_setup_logical_calls": primary_timing["voxel_factor"]["setup_logical_calls"],
                    "scalar_setup_physical_calls": primary_timing["scalar"]["setup_physical_calls"],
                    "view_block_setup_physical_calls": primary_timing["view_block"]["setup_physical_calls"],
                    "voxel_factor_setup_physical_calls": primary_timing["voxel_factor"]["setup_physical_calls"],
                    "scalar_solver_only_seconds": solver_elapsed["scalar"],
                    "view_block_solver_only_seconds": solver_elapsed["view_block"],
                    "voxel_factor_solver_only_seconds": solver_elapsed["voxel_factor"],
                }
            )

    summaries = _summaries(rows)
    decision = _decision(rows, timing_rows, config["thresholds"])
    report = {
        "schema_version": SCHEMA,
        "factor_metric_schema": GATE_B_DATA_ONLY_SCHEMA,
        "status": decision["status"],
        "amendment": config["amendment"],
        "source_commit": source_commit,
        "config_sha256": _sha256(config_path),
        "source_sha256": source_hashes,
        "test_preflight": test_preflight,
        "evidence_role": config["evidence_role"],
        "claim_boundary": config["claim_boundary"],
        "data_contract": {
            "replicates": list(OPENED_REPLICATES),
            "reaction_families": list(REACTION_FAMILIES),
            "sample_count": len(OPENED_REPLICATES) * len(REACTION_FAMILIES),
            "real_psu_detector_geometry": True,
            "analytic_field_truth": True,
            "synthetic_correlated_noise": True,
            "truth_derived_development_scale": True,
        },
        "graph_replay": {
            "status": "NONBINDING_BATCH_SINGLETON_TRACEABILITY_DIAGNOSTIC",
            "maximum_absolute_metric_difference": graph_replay_max_difference,
            "binding": False,
            "parent_batch_rows_used_for_scoring": False,
            "scored_control": "same_run_single_sample_exact_k_graph_pcgls",
            "diagnostic_report_sha256": config["amendment"][
                "graph_diagnostic_report_sha256"
            ],
            "batch_size": 1,
        },
        "operator_equivalence": {
            "status": "PASS",
            "maximum_forward_relative_error": max(
                row["forward_relative_error"] for row in equivalence_rows
            ),
            "maximum_transpose_relative_error": max(
                row["transpose_relative_error"] for row in equivalence_rows
            ),
            "threshold": config["thresholds"][
                "operator_equivalence_relative_error_max"
            ],
        },
        "target_normalization": {
            "status": "PASS",
            "maximum_relative_error": max(target_normalization_errors),
            "threshold": config["thresholds"][
                "target_normalization_relative_error_max"
            ],
            "formula": "b0=graph_prepared/(amplitude_scale*sqrt(measurement_count))",
        },
        "decision": decision,
        "summaries": summaries,
        "row_count": len(rows),
        "timing_pair_count": len(timing_rows),
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "device": device_name,
            "dtype": "torch.float32",
        },
        "timing_contract": config["timing_contract"],
        "operator_budget_contract": config["operator_budget"],
        "objective_contract": config["objective_contract"],
        "invalidity_policy": config["invalidity_policy"],
        "scientific_claim_boundary": (
            "OPENED_SYNTHETIC_MECHANISM_GATE_ONLY_NO_FRESH_REAL_FLOW_OR_WIN_CLAIM"
        ),
    }
    audit = {
        "operator_equivalence_rows": equivalence_rows,
        "target_normalization_rows": target_normalization_rows,
        "timing_rows": timing_rows,
        "factor_call_rows": call_rows,
    }
    return report, rows, audit


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty {path.name}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_release(
    output: Path,
    *,
    report: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_bytes(_canonical_bytes(report) + b"\n")
    _write_csv(output / "metric_rows.csv", rows)
    (output / "audit.json").write_bytes(_canonical_bytes(audit) + b"\n")
    files = ("report.json", "metric_rows.csv", "audit.json")
    checksums = "".join(f"{_sha256(output / name)}  {name}\n" for name in files)
    (output / "checksums.sha256").write_text(checksums, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT
        / "demo_t16_operator/configs/psu_b0_factor_pdhg_gate_b_v4_a_only_connectivity_amendment.json",
    )
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT
        / "demo_t16_operator/results/psu_b0_factor_pdhg_gate_b",
    )
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    report, rows, audit = run_gate_b(
        root=REPOSITORY_ROOT,
        config_path=args.config,
        view_root=args.view_root.resolve(),
        device_name=str(args.device),
    )
    write_release(args.output.resolve(), report=report, rows=rows, audit=audit)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
