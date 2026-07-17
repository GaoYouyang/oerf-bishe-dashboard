#!/usr/bin/env python3
"""Run the post-open N1.6 adjoint-weighted low-rank mechanism screen.

This runner may generate one hypothesis for a later frozen confirmation.  It
cannot establish real-BOST, OOD, or confirmed algorithm performance.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo_t16_operator.interface_baselines import cgls_baseline  # noqa: E402
from demo_t16_operator.jacru_n1_5_high_order_correction import (  # noqa: E402
    HighOrderTeacherMaps,
    warm_start_cgls,
)
from demo_t16_operator.jacru_n1_6_adjoint_low_rank import (  # noqa: E402
    MeasurementBasis,
    MultiOutputStandardizedRidge,
    adjoint_optimal_coefficients,
    coefficient_abs_limits,
    fail_closed_predict,
    fit_measurement_pca,
    fit_multioutput_ridge,
    measurement_optimal_coefficients,
    standardized_feature_limit,
    visible_case_feature_blocks,
)
from demo_t16_operator.psu_b0_streaming_operator import (  # noqa: E402
    zero_outer_boundary_support,
)
from site_tools import (  # noqa: E402
    run_jacru_n1_5_approximation_error_headroom as n15a,
)
from site_tools import (  # noqa: E402
    run_jacru_n1_5_reconstruction_aware_postopen as n15b,
)


DEFAULT_CONFIG = (
    ROOT
    / "demo_t16_operator/configs/"
    "jacru_n1_6_adjoint_low_rank_development_v1.json"
)
DEFAULT_OUTPUT = (
    ROOT / "demo_t16_operator/results/jacru_n1_6_adjoint_low_rank_scratch"
)
REPORT_SCHEMA = "jacru-n1-6-adjoint-low-rank-report-1.0"


@dataclass(frozen=True)
class PreparedCase:
    record: n15a.CaseRecord
    measured_observation: torch.Tensor
    signal_scale: float
    warm_field: torch.Tensor
    warm_projection: torch.Tensor
    feature_blocks: Mapping[str, tuple[tuple[str, ...], torch.Tensor]]
    damping_normalized: torch.Tensor
    shared_warm_seconds: float
    visible_feature_seconds: float

    @property
    def key(self) -> tuple[str, int, str]:
        return self.record.partition, self.record.base_seed, self.record.family


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    feature_set: str
    rank: int
    target_l2: float
    ridge_alpha: float
    shrinkage: float
    basis: MeasurementBasis
    model: MultiOutputStandardizedRidge
    coefficient_limits: torch.Tensor
    feature_limit: float
    residual_rms_limit: float
    fit_mean_adjoint_oracle_residual_ratio: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replace-output", action="store_true")
    parser.add_argument("--seed-limit", type=int)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _validate_split_integrity(
    manifest: list[dict[str, Any]],
    confirmation_manifest: list[dict[str, str]],
    *,
    families: list[str],
) -> dict[str, Any]:
    expected_families = set(families)
    seen: set[tuple[str, int, str]] = set()
    by_cluster: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in manifest:
        key = str(row["partition"]), int(row["base_seed"]), str(row["family"])
        if key in seen:
            raise RuntimeError(f"duplicate manifest case: {key}")
        seen.add(key)
        by_cluster.setdefault((key[0], key[1]), []).append(row)
    for key, rows in by_cluster.items():
        if {str(row["family"]) for row in rows} != expected_families:
            raise RuntimeError(f"incomplete family pairing: {key}")
        if len({str(row["geometry_digest"]) for row in rows}) != 1:
            raise RuntimeError(f"paired families do not share geometry: {key}")

    partitions = {str(row["partition"]) for row in manifest}
    seed_sets = {
        partition: {
            int(row["base_seed"])
            for row in manifest
            if str(row["partition"]) == partition
        }
        for partition in partitions
    }
    digest_sets = {
        partition: {
            str(row["geometry_digest"])
            for row in manifest
            if str(row["partition"]) == partition
        }
        for partition in partitions
    }
    for left in sorted(partitions):
        for right in sorted(partitions):
            if left >= right:
                continue
            if seed_sets[left] & seed_sets[right]:
                raise RuntimeError(f"seed leakage between {left} and {right}")
            if digest_sets[left] & digest_sets[right]:
                raise RuntimeError(f"geometry leakage between {left} and {right}")

    confirmation_development_seeds = {
        int(row["base_seed"])
        for row in confirmation_manifest
        if row["partition"] == "development"
    }
    confirmation_development_digests = {
        row["geometry_digest"]
        for row in confirmation_manifest
        if row["partition"] == "development"
    }
    if seed_sets.get("development", set()) & confirmation_development_seeds:
        raise RuntimeError("N1.6 development reuses N1.5 confirmation seeds")
    if digest_sets.get("development", set()) & confirmation_development_digests:
        raise RuntimeError("N1.6 development reuses N1.5 confirmation geometries")

    confirmation_fit_calibration = {
        (row["partition"], int(row["base_seed"]), row["family"], row["geometry_digest"])
        for row in confirmation_manifest
        if row["partition"] in {"fit", "calibration"}
    }
    current_fit_calibration = {
        (
            str(row["partition"]),
            int(row["base_seed"]),
            str(row["family"]),
            str(row["geometry_digest"]),
        )
        for row in manifest
        if str(row["partition"]) in {"fit", "calibration"}
    }
    if not current_fit_calibration <= confirmation_fit_calibration:
        raise RuntimeError("N1.6 fit/calibration source drifted from N1.5 confirmation")
    return {
        "partition_seed_disjoint": True,
        "partition_geometry_disjoint": True,
        "paired_family_contract": sorted(expected_families),
        "n1_5_confirmation_development_seed_overlap": 0,
        "n1_5_confirmation_development_geometry_overlap": 0,
        "fit_calibration_source_matches_n1_5_confirmation": True,
    }


def _write_checksums(output: Path) -> None:
    entries = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "checksums.sha256":
            entries.append(f"{_sha256(path)}  {path.name}")
    (output / "checksums.sha256").write_text("\n".join(entries) + "\n", encoding="utf-8")


def _validate_config(
    config: Mapping[str, Any],
    n15a_config: Mapping[str, Any],
    *,
    seed_limit: int | None,
) -> None:
    if config.get("schema_version") != "jacru-n1-6-adjoint-low-rank-development-1.0":
        raise ValueError("unexpected N1.6 config schema")
    if config.get("status") != "POSTOPEN_DEVELOPMENT_HYPOTHESIS_GENERATION_ONLY":
        raise ValueError("N1.6 may only run as post-open hypothesis generation")
    if config.get("may_construct_or_evaluate_ood") is not False:
        raise ValueError("N1.6 development must not construct or evaluate OOD")
    contract = config["deployment_contract"]
    if int(contract["high_order_forward_calls"]) != 0 or int(
        contract["high_order_adjoint_calls"]
    ) != 0:
        raise ValueError("deployable N1.6 must use zero high-order calls")
    if contract.get("independent_learned_adjoint") is not False:
        raise ValueError("N1.6 must route corrections through the current low adjoint")
    forbidden = set(contract["forbidden_inputs"])
    required_forbidden = {
        "truth_volume",
        "fresh_exact_mismatch",
        "phantom_family_label",
        "confirmation_target",
        "high_order_forward_output",
    }
    if forbidden != required_forbidden:
        raise ValueError("deployment forbidden-input contract drifted")

    budget = config["budget"]
    reference_forward = int(budget["low_reference_cgls_iterations"]) + int(
        budget["low_reference_final_projection_forward_calls"]
    )
    reference_adjoint = int(budget["low_reference_cgls_iterations"])
    corrected_forward = (
        int(budget["warm_cgls_iterations"])
        + int(budget["visible_low_projection_forward_calls"])
        + int(budget["corrected_warm_cgls_iterations"])
    )
    corrected_adjoint = int(budget["warm_cgls_iterations"]) + int(
        budget["corrected_warm_cgls_iterations"]
    )
    expected = (
        int(budget["deployable_total_low_forward_calls"]),
        int(budget["deployable_total_low_adjoint_calls"]),
    )
    if (reference_forward, reference_adjoint) != expected:
        raise ValueError("low-reference physical-call budget drifted")
    if (corrected_forward, corrected_adjoint) != expected:
        raise ValueError("corrected physical-call budget drifted")

    grid = config["model_grid"]
    if grid.get("anchor") != "component_damping":
        raise ValueError("component damping must remain the frozen low-cost anchor")
    if grid.get("basis_fit_split") != "fit" or grid.get("selection_split") != "calibration":
        raise ValueError("basis and selection splits drifted")
    for key in ("basis_ranks", "adjoint_target_l2", "ridge_alphas", "residual_shrinkages"):
        if not grid.get(key):
            raise ValueError(f"empty model grid: {key}")
    if any(int(value) < 1 for value in grid["basis_ranks"]):
        raise ValueError("basis ranks must be positive")
    if any(float(value) < 0.0 for value in grid["adjoint_target_l2"]):
        raise ValueError("adjoint target l2 values must be nonnegative")
    if any(float(value) < 0.0 for value in grid["ridge_alphas"]):
        raise ValueError("ridge alphas must be nonnegative")
    if any(not 0.0 < float(value) <= 1.0 for value in grid["residual_shrinkages"]):
        raise ValueError("residual shrinkages must lie in (0,1]")

    claim = config["claim_boundary"]
    if claim.get("may_claim_confirmed_algorithm_gain") is not False:
        raise ValueError("opened N1.6 development cannot claim confirmed gain")
    if claim.get("is_real_bost_evidence") is not False:
        raise ValueError("synthetic N1.6 cannot claim real-BOST evidence")
    if claim.get("opens_ood_fresh_or_final") is not False:
        raise ValueError("N1.6 development cannot open OOD/fresh/final")
    if n15a_config.get("status") != "DEVELOPMENT_ONLY_OPENED_NOT_CONFIRMATORY":
        raise ValueError("source N1.5-A contract drifted")
    if seed_limit is not None and seed_limit < 1:
        raise ValueError("seed-limit must be positive")


def _prepare_cases(
    config: Mapping[str, Any],
    n15a_config: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    seed_limit: int | None,
) -> tuple[list[PreparedCase], list[dict[str, Any]]]:
    records, manifest = n15a._prepare_records(
        n15a_config, source, seed_limit=seed_limit
    )
    fixed = n15a._fixed_predictors(records)
    warm_iterations = int(config["budget"]["warm_cgls_iterations"])
    states: list[PreparedCase] = []
    for record in records:
        operator = record.case.inference.operator
        observation = record.case.inference.observations_uv[0]
        clean_observation = record.case.evaluation.clean_observations_uv[0]
        if not torch.equal(observation, clean_observation):
            raise RuntimeError(
                "N1.6 representation-mismatch screen requires the preregistered "
                "noise-disabled measured observation"
            )
        signal_scale = float(
            torch.sqrt(torch.mean(observation.square())).clamp_min(1e-12)
        )
        support = zero_outer_boundary_support(operator.grid_shape, dtype=torch.float64)
        forward, adjoint = n15a._operator_maps(operator)
        operator.reset_call_counts()
        started = time.perf_counter()
        warm = cgls_baseline(
            observation,
            forward=forward,
            adjoint=adjoint,
            support=support,
            spacing_xyz=operator.spacing_xyz,
            iterations=warm_iterations,
        )
        warm_projection = operator(warm.field[None, None])[0]
        shared_seconds = time.perf_counter() - started
        if operator.call_report() != {
            "forward_calls": warm_iterations + 1,
            "adjoint_calls": warm_iterations,
        }:
            raise RuntimeError("shared warm/projection call budget drifted")
        feature_started = time.perf_counter()
        features = visible_case_feature_blocks(
            geometry=record.case.inference.geometry,
            observation_uv=observation,
            warm_projection_uv=warm_projection,
            warm_field=warm.field,
        )
        feature_seconds = time.perf_counter() - feature_started
        damping = (observation / signal_scale) * torch.as_tensor(
            fixed["component_damping"]["value"], dtype=torch.float64
        ).reshape(1, 2)
        states.append(
            PreparedCase(
                record=record,
                measured_observation=observation,
                signal_scale=signal_scale,
                warm_field=warm.field,
                warm_projection=warm_projection,
                feature_blocks=features,
                damping_normalized=damping,
                shared_warm_seconds=shared_seconds,
                visible_feature_seconds=feature_seconds,
            )
        )
    for row in manifest:
        row["n1_6_deployment_uses_truth"] = False
        row["n1_6_deployment_uses_high_order_output"] = False
        row["n1_6_observation_source"] = "case.inference.observations_uv"
        row["synthetic_observation_noise_disabled"] = True
    return states, manifest


def _fit_states(states: list[PreparedCase]) -> list[PreparedCase]:
    selected = [state for state in states if state.record.partition == "fit"]
    if len(selected) < 2:
        raise ValueError("at least two fit cases are required")
    return selected


def _target_residual(state: PreparedCase) -> torch.Tensor:
    return state.record.mismatch_normalized - state.damping_normalized


def _feature_matrix(
    states: list[PreparedCase], feature_set: str
) -> tuple[tuple[str, ...], torch.Tensor]:
    names = states[0].feature_blocks[feature_set][0]
    if any(state.feature_blocks[feature_set][0] != names for state in states):
        raise RuntimeError("visible case-feature contract drifted")
    return names, torch.stack(
        [state.feature_blocks[feature_set][1] for state in states]
    )


def _format_token(value: float) -> str:
    return f"{float(value):.0e}".replace("+", "").replace("-", "m")


def _fit_candidate_specs(
    states: list[PreparedCase], config: Mapping[str, Any]
) -> tuple[list[CandidateSpec], list[dict[str, Any]]]:
    fit = _fit_states(states)
    target_vectors = torch.stack([_target_residual(state).reshape(-1) for state in fit])
    grid = config["model_grid"]
    envelope = config["fail_closed_envelope"]
    maximum_rank = min(target_vectors.shape[0] - 1, target_vectors.shape[1])
    ranks = sorted(
        {int(value) for value in grid["basis_ranks"] if int(value) <= maximum_rank}
    )
    specs: list[CandidateSpec] = []
    fit_rows: list[dict[str, Any]] = []

    for rank in ranks:
        basis = fit_measurement_pca(target_vectors, rank=rank)
        for target_l2_value in grid["adjoint_target_l2"]:
            target_l2 = float(target_l2_value)
            coefficient_rows = []
            residual_ratios = []
            residual_rms_values = []
            for state in fit:
                operator = state.record.case.inference.operator
                _, adjoint = n15a._operator_maps(operator)
                operator.reset_call_counts()
                fitted = adjoint_optimal_coefficients(
                    basis,
                    _target_residual(state).reshape(-1),
                    observation_shape=state.record.mismatch_normalized.shape,
                    adjoint=adjoint,
                    l2=target_l2,
                )
                if operator.call_report() != {
                    "forward_calls": 0,
                    "adjoint_calls": rank + 1,
                }:
                    raise RuntimeError("offline adjoint-target call ledger drifted")
                coefficient_rows.append(fitted.coefficients)
                residual_ratios.append(fitted.residual_ratio)
                residual_rms_values.append(
                    float(torch.sqrt(torch.mean(basis.synthesize(fitted.coefficients).square())))
                )
                fit_rows.append(
                    {
                        "partition": "fit",
                        "base_seed": state.record.base_seed,
                        "family": state.record.family,
                        "geometry_digest": state.record.case.inference.geometry.digest,
                        "rank": rank,
                        "target_l2": target_l2,
                        "adjoint_oracle_residual_ratio": fitted.residual_ratio,
                        "target_adjoint_norm": fitted.target_adjoint_norm,
                        "offline_evaluator_adjoint_calls": fitted.evaluator_adjoint_calls,
                    }
                )
            coefficients = torch.stack(coefficient_rows)
            coefficient_limit = coefficient_abs_limits(
                coefficients,
                quantile=float(envelope["coefficient_abs_quantile"]),
                multiplier=float(envelope["coefficient_abs_multiplier"]),
            )
            rms_limit = float(
                torch.quantile(
                    torch.as_tensor(residual_rms_values, dtype=torch.float64),
                    float(envelope["residual_rms_quantile"]),
                )
                * float(envelope["residual_rms_multiplier"])
            )
            for feature_set in grid["feature_sets"]:
                names, features = _feature_matrix(fit, str(feature_set))
                for ridge_alpha_value in grid["ridge_alphas"]:
                    ridge_alpha = float(ridge_alpha_value)
                    model = fit_multioutput_ridge(
                        features,
                        coefficients,
                        feature_names=names,
                        alpha=ridge_alpha,
                    )
                    feature_limit = standardized_feature_limit(
                        model,
                        features,
                        quantile=float(envelope["feature_max_abs_z_quantile"]),
                        multiplier=float(envelope["feature_max_abs_z_multiplier"]),
                    )
                    for shrinkage_value in grid["residual_shrinkages"]:
                        shrinkage = float(shrinkage_value)
                        candidate_id = (
                            f"adjlr_{feature_set}_r{rank}_tl2{_format_token(target_l2)}_"
                            f"a{_format_token(ridge_alpha)}_s{str(shrinkage).replace('.', 'p')}"
                        )
                        specs.append(
                            CandidateSpec(
                                candidate_id=candidate_id,
                                feature_set=str(feature_set),
                                rank=rank,
                                target_l2=target_l2,
                                ridge_alpha=ridge_alpha,
                                shrinkage=shrinkage,
                                basis=basis,
                                model=model,
                                coefficient_limits=coefficient_limit,
                                feature_limit=feature_limit,
                                residual_rms_limit=rms_limit,
                                fit_mean_adjoint_oracle_residual_ratio=float(
                                    np.mean(residual_ratios)
                                ),
                            )
                        )
    if not specs:
        raise RuntimeError("no valid N1.6 model specs")
    return specs, fit_rows


def _field_score(field: torch.Tensor, state: PreparedCase) -> dict[str, float]:
    return n15b._field_metrics(field, state.record)


def _adjoint_diagnostic_row(
    state: PreparedCase,
    *,
    candidate_id: str,
    correction_normalized: torch.Tensor,
) -> dict[str, Any]:
    """Return a truth-derived evaluator row kept outside deployment metrics."""

    operator = state.record.case.inference.operator
    _, adjoint = n15a._operator_maps(operator)
    exact = state.record.mismatch_normalized
    operator.reset_call_counts()
    candidate_residual = adjoint(exact - correction_normalized)
    damping_residual = adjoint(exact - state.damping_normalized)
    calls = operator.call_report()
    if calls != {"forward_calls": 0, "adjoint_calls": 2}:
        raise RuntimeError("adjoint diagnostic call ledger drifted")
    ratio = float(torch.linalg.vector_norm(candidate_residual)) / max(
        float(torch.linalg.vector_norm(damping_residual)), 1e-30
    )
    return {
        "partition": state.record.partition,
        "base_seed": state.record.base_seed,
        "family": state.record.family,
        "case_id": state.record.case.inference.case_id,
        "geometry_digest": state.record.case.inference.geometry.digest,
        "candidate_id": candidate_id,
        "adjoint_residual_ratio_to_component_damping": ratio,
        "adjoint_residual_gain_over_component_damping": 1.0 - ratio,
        "fresh_exact_mismatch_access": True,
        "evaluator_only": True,
        "evaluator_forward_calls": 0,
        "evaluator_adjoint_calls": 2,
    }


def _correction_row(
    state: PreparedCase,
    *,
    candidate_id: str,
    correction_normalized: torch.Tensor,
    config: Mapping[str, Any],
    fallback: bool,
    fallback_reason: str | None,
    takeover: bool,
    evaluator_only: bool,
    high_order_forward_calls: int,
    high_order_adjoint_calls: int,
    deployment_setup_seconds: float = 0.0,
    evaluator_setup_seconds: float = 0.0,
    model_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    operator = state.record.case.inference.operator
    forward, adjoint = n15a._operator_maps(operator)
    observation = state.measured_observation
    refine_iterations = int(config["budget"]["corrected_warm_cgls_iterations"])
    correction = correction_normalized * state.signal_scale
    operator.reset_call_counts()
    started = time.perf_counter()
    result = warm_start_cgls(
        observation - correction,
        forward=forward,
        adjoint=adjoint,
        support=zero_outer_boundary_support(operator.grid_shape, dtype=torch.float64),
        initial_field=state.warm_field,
        initial_projection=state.warm_projection,
        iterations=refine_iterations,
    )
    refine_seconds = time.perf_counter() - started
    if operator.call_report() != {
        "forward_calls": refine_iterations,
        "adjoint_calls": refine_iterations,
    }:
        raise RuntimeError("N1.6 refinement call ledger drifted")
    field = _field_score(result.field, state)
    budget = config["budget"]
    row: dict[str, Any] = {
        "partition": state.record.partition,
        "base_seed": state.record.base_seed,
        "family": state.record.family,
        "case_id": state.record.case.inference.case_id,
        "geometry_digest": state.record.case.inference.geometry.digest,
        "candidate_id": candidate_id,
        "field_relative_l2": field["field_relative_l2"],
        "h1_seminorm_relative_l2": field["h1_seminorm_relative_error"],
        "field_mean_bias": field["field_mean_bias"],
        "data_residual_relative_l2": float(torch.linalg.vector_norm(result.residual))
        / max(float(torch.linalg.vector_norm(observation)), 1e-30),
        "fallback": fallback,
        "fallback_reason": fallback_reason,
        "takeover": takeover,
        "evaluator_only": evaluator_only,
        "low_forward_calls": int(budget["deployable_total_low_forward_calls"]),
        "low_adjoint_calls": int(budget["deployable_total_low_adjoint_calls"]),
        "high_order_forward_calls": high_order_forward_calls,
        "high_order_adjoint_calls": high_order_adjoint_calls,
        "shared_warm_seconds": state.shared_warm_seconds,
        "deployment_setup_seconds": deployment_setup_seconds,
        "evaluator_setup_seconds": evaluator_setup_seconds,
        "candidate_refine_seconds": refine_seconds,
        "end_to_end_seconds": state.shared_warm_seconds
        + deployment_setup_seconds
        + refine_seconds,
        "total_with_evaluator_seconds": state.shared_warm_seconds
        + deployment_setup_seconds
        + evaluator_setup_seconds
        + refine_seconds,
    }
    if model_metadata:
        row.update(model_metadata)
    return row


def _low_reference_row(
    state: PreparedCase, config: Mapping[str, Any]
) -> dict[str, Any]:
    operator = state.record.case.inference.operator
    forward, adjoint = n15a._operator_maps(operator)
    observation = state.measured_observation
    iterations = int(config["budget"]["low_reference_cgls_iterations"])
    operator.reset_call_counts()
    started = time.perf_counter()
    result = cgls_baseline(
        observation,
        forward=forward,
        adjoint=adjoint,
        support=zero_outer_boundary_support(operator.grid_shape, dtype=torch.float64),
        spacing_xyz=operator.spacing_xyz,
        iterations=iterations,
    )
    projection = operator(result.field[None, None])[0]
    elapsed = time.perf_counter() - started
    expected = {
        "forward_calls": int(config["budget"]["deployable_total_low_forward_calls"]),
        "adjoint_calls": int(config["budget"]["deployable_total_low_adjoint_calls"]),
    }
    if operator.call_report() != expected:
        raise RuntimeError("matched low-reference call ledger drifted")
    field = _field_score(result.field, state)
    return {
        "partition": state.record.partition,
        "base_seed": state.record.base_seed,
        "family": state.record.family,
        "case_id": state.record.case.inference.case_id,
        "geometry_digest": state.record.case.inference.geometry.digest,
        "candidate_id": "low_cgls24_matched",
        "field_relative_l2": field["field_relative_l2"],
        "h1_seminorm_relative_l2": field["h1_seminorm_relative_error"],
        "field_mean_bias": field["field_mean_bias"],
        "data_residual_relative_l2": float(torch.linalg.vector_norm(observation - projection))
        / max(float(torch.linalg.vector_norm(observation)), 1e-30),
        "fallback": False,
        "fallback_reason": None,
        "takeover": True,
        "evaluator_only": False,
        "low_forward_calls": expected["forward_calls"],
        "low_adjoint_calls": expected["adjoint_calls"],
        "high_order_forward_calls": 0,
        "high_order_adjoint_calls": 0,
        "shared_warm_seconds": 0.0,
        "deployment_setup_seconds": 0.0,
        "evaluator_setup_seconds": 0.0,
        "candidate_refine_seconds": elapsed,
        "end_to_end_seconds": elapsed,
        "total_with_evaluator_seconds": elapsed,
    }


def _candidate_prediction(
    state: PreparedCase, spec: CandidateSpec, *, fail_closed: bool
) -> tuple[torch.Tensor, dict[str, Any]]:
    started = time.perf_counter()
    names, features = state.feature_blocks[spec.feature_set]
    if names != spec.model.feature_names:
        raise RuntimeError("selected feature contract drifted")
    matrix = features[None]
    if fail_closed:
        predicted = fail_closed_predict(
            model=spec.model,
            features=matrix,
            basis=spec.basis,
            coefficient_limits=spec.coefficient_limits,
            feature_max_abs_z_limit=spec.feature_limit,
            residual_rms_limit=spec.residual_rms_limit,
        )
        residual = predicted.residual.reshape_as(state.damping_normalized) * spec.shrinkage
        metadata = {
            "route": "fail_closed",
            "feature_set": spec.feature_set,
            "basis_rank": spec.rank,
            "target_l2": spec.target_l2,
            "ridge_alpha": spec.ridge_alpha,
            "residual_shrinkage": spec.shrinkage,
            "feature_max_abs_z": predicted.feature_max_abs_z,
            "predicted_residual_rms": predicted.residual_rms,
            "fallback": predicted.fallback,
            "fallback_reason": predicted.fallback_reason,
            "takeover": not predicted.fallback,
            "model_inference_seconds": time.perf_counter() - started,
        }
        return state.damping_normalized + residual, metadata
    raw = spec.model.predict(matrix)[0]
    residual = spec.basis.synthesize(raw).reshape_as(state.damping_normalized) * spec.shrinkage
    return state.damping_normalized + residual, {
        "route": "raw_unbounded_diagnostic",
        "feature_set": spec.feature_set,
        "basis_rank": spec.rank,
        "target_l2": spec.target_l2,
        "ridge_alpha": spec.ridge_alpha,
        "residual_shrinkage": spec.shrinkage,
        "feature_max_abs_z": float(torch.max(torch.abs(spec.model.standardized(matrix)))),
        "predicted_residual_rms": float(torch.sqrt(torch.mean(residual.square()))),
        "fallback": False,
        "fallback_reason": None,
        "takeover": True,
        "model_inference_seconds": time.perf_counter() - started,
    }


def _baseline_rows(
    states: list[PreparedCase], config: Mapping[str, Any], *, partition: str
) -> list[dict[str, Any]]:
    rows = []
    for state in states:
        if state.record.partition != partition:
            continue
        rows.append(_low_reference_row(state, config))
        rows.append(
            _correction_row(
                state,
                candidate_id="component_damping",
                correction_normalized=state.damping_normalized,
                config=config,
                fallback=False,
                fallback_reason=None,
                takeover=True,
                evaluator_only=False,
                high_order_forward_calls=0,
                high_order_adjoint_calls=0,
            )
        )
    return rows


def _index_rows(rows: list[dict[str, Any]], candidate_id: str) -> dict[tuple[int, str], dict[str, Any]]:
    indexed: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        if row["candidate_id"] != candidate_id:
            continue
        key = int(row["base_seed"]), str(row["family"])
        if key in indexed:
            raise RuntimeError(f"duplicate candidate case row: {candidate_id} {key}")
        indexed[key] = row
    return indexed


def _aggregate_candidate(
    rows: list[dict[str, Any]],
    *,
    candidate_id: str,
    low_rows: Mapping[tuple[int, str], Mapping[str, Any]],
    damping_rows: Mapping[tuple[int, str], Mapping[str, Any]],
    teacher_rows: Mapping[tuple[int, str], Mapping[str, Any]] | None = None,
    adjoint_diagnostic_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    selected_index = _index_rows(rows, candidate_id)
    if not selected_index:
        raise ValueError(f"no rows for candidate: {candidate_id}")
    expected_keys = set(low_rows)
    if set(damping_rows) != expected_keys or set(selected_index) != expected_keys:
        raise RuntimeError(f"candidate/baseline case-set mismatch: {candidate_id}")
    if teacher_rows is not None and set(teacher_rows) != expected_keys:
        raise RuntimeError("teacher/baseline case-set mismatch")
    expected_families = {family for _, family in expected_keys}
    for seed in {seed for seed, _ in expected_keys}:
        if {family for case_seed, family in expected_keys if case_seed == seed} != expected_families:
            raise RuntimeError(f"incomplete paired-family geometry cluster: {seed}")
        digests = {
            str(selected_index[(seed, family)]["geometry_digest"])
            for family in expected_families
        }
        if len(digests) != 1:
            raise RuntimeError(f"paired families do not share one geometry digest: {seed}")
    diagnostic_index = (
        {}
        if adjoint_diagnostic_rows is None
        else _index_rows(adjoint_diagnostic_rows, candidate_id)
    )
    if diagnostic_index and set(diagnostic_index) != expected_keys:
        raise RuntimeError(f"adjoint diagnostic case-set mismatch: {candidate_id}")
    selected = list(selected_index.values())
    paired: list[dict[str, Any]] = []
    for row in selected:
        key = int(row["base_seed"]), str(row["family"])
        low = low_rows[key]
        damping = damping_rows[key]
        teacher = None if teacher_rows is None else teacher_rows.get(key)
        field = float(row["field_relative_l2"])
        h1 = float(row["h1_seminorm_relative_l2"])
        paired.append(
            {
                "base_seed": key[0],
                "family": key[1],
                "field_gain_over_low": 1.0 - field / max(float(low["field_relative_l2"]), 1e-30),
                "h1_gain_over_low": 1.0 - h1 / max(float(low["h1_seminorm_relative_l2"]), 1e-30),
                "field_gain_over_damping": 1.0
                - field / max(float(damping["field_relative_l2"]), 1e-30),
                "field_gain_over_teacher": None
                if teacher is None
                else 1.0 - field / max(float(teacher["field_relative_l2"]), 1e-30),
                "fallback": bool(row["fallback"]),
                "takeover": bool(row["takeover"]),
                "adjoint_gain_over_damping": None
                if key not in diagnostic_index
                else float(
                    diagnostic_index[key]["adjoint_residual_gain_over_component_damping"]
                ),
            }
        )
    cluster_rows = []
    for seed in sorted({int(row["base_seed"]) for row in paired}):
        cluster = [row for row in paired if int(row["base_seed"]) == seed]
        cluster_rows.append(
            {
                "base_seed": seed,
                "field_gain_over_low": float(np.mean([row["field_gain_over_low"] for row in cluster])),
                "h1_gain_over_low": float(np.mean([row["h1_gain_over_low"] for row in cluster])),
                "field_gain_over_damping": float(
                    np.mean([row["field_gain_over_damping"] for row in cluster])
                ),
                "field_gain_over_teacher": None
                if teacher_rows is None
                else float(np.mean([row["field_gain_over_teacher"] for row in cluster])),
                "takeover": float(np.mean([float(row["takeover"]) for row in cluster])),
            }
        )
    first = selected[0]
    teacher_values = [row["field_gain_over_teacher"] for row in paired if row["field_gain_over_teacher"] is not None]
    adjoint_values = [
        float(row["adjoint_gain_over_damping"])
        for row in paired
        if row["adjoint_gain_over_damping"] is not None
    ]
    return {
        "candidate_id": candidate_id,
        "partition": first["partition"],
        "feature_set": first.get("feature_set"),
        "basis_rank": first.get("basis_rank"),
        "target_l2": first.get("target_l2"),
        "ridge_alpha": first.get("ridge_alpha"),
        "residual_shrinkage": first.get("residual_shrinkage"),
        "geometry_cluster_count": len(cluster_rows),
        "case_count": len(paired),
        "mean_field_gain_over_low_cgls24": float(
            np.mean([row["field_gain_over_low"] for row in cluster_rows])
        ),
        "mean_h1_gain_over_low_cgls24": float(
            np.mean([row["h1_gain_over_low"] for row in cluster_rows])
        ),
        "mean_field_gain_over_component_damping": float(
            np.mean([row["field_gain_over_damping"] for row in cluster_rows])
        ),
        "mean_field_gain_over_high_order_teacher_b0p75": None
        if not teacher_values
        else float(np.mean(teacher_values)),
        "worst_case_field_gain_over_low_cgls24": float(
            min(row["field_gain_over_low"] for row in paired)
        ),
        "worst_geometry_field_gain_over_low_cgls24": float(
            min(row["field_gain_over_low"] for row in cluster_rows)
        ),
        "worst_case_field_gain_over_component_damping": float(
            min(row["field_gain_over_damping"] for row in paired)
        ),
        "case_harm_over_one_percent_rate_vs_low": float(
            np.mean([row["field_gain_over_low"] < -0.01 for row in paired])
        ),
        "case_harm_over_one_percent_rate_vs_component_damping": float(
            np.mean([row["field_gain_over_damping"] < -0.01 for row in paired])
        ),
        "takeover_rate": float(np.mean([row["takeover"] for row in paired])),
        "fallback_rate": float(np.mean([row["fallback"] for row in paired])),
        "evaluator_mean_adjoint_residual_gain_over_component_damping": None
        if not adjoint_values
        else float(np.mean(adjoint_values)),
        "low_forward_calls": int(first["low_forward_calls"]),
        "low_adjoint_calls": int(first["low_adjoint_calls"]),
        "high_order_forward_calls": int(first["high_order_forward_calls"]),
        "high_order_adjoint_calls": int(first["high_order_adjoint_calls"]),
        "evaluator_only": bool(first["evaluator_only"]),
    }


def _calibration_grid(
    states: list[PreparedCase],
    specs: list[CandidateSpec],
    config: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    baselines = _baseline_rows(states, config, partition="calibration")
    low = _index_rows(baselines, "low_cgls24_matched")
    damping = _index_rows(baselines, "component_damping")
    case_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    calibration_states = [state for state in states if state.record.partition == "calibration"]
    for spec in specs:
        for state in calibration_states:
            correction, metadata = _candidate_prediction(state, spec, fail_closed=True)
            case_rows.append(
                _correction_row(
                    state,
                    candidate_id=spec.candidate_id,
                    correction_normalized=correction,
                    config=config,
                    fallback=bool(metadata["fallback"]),
                    fallback_reason=metadata["fallback_reason"],
                    takeover=bool(metadata["takeover"]),
                    evaluator_only=False,
                    high_order_forward_calls=0,
                    high_order_adjoint_calls=0,
                    deployment_setup_seconds=state.visible_feature_seconds
                    + float(metadata["model_inference_seconds"]),
                    model_metadata=metadata,
                )
            )
            diagnostic_rows.append(
                _adjoint_diagnostic_row(
                    state,
                    candidate_id=spec.candidate_id,
                    correction_normalized=correction,
                )
            )
        aggregates.append(
            _aggregate_candidate(
                case_rows,
                candidate_id=spec.candidate_id,
                low_rows=low,
                damping_rows=damping,
                adjoint_diagnostic_rows=diagnostic_rows,
            )
        )
    return baselines, case_rows, diagnostic_rows, aggregates


def _select_candidate(
    specs: list[CandidateSpec], aggregates: list[dict[str, Any]], config: Mapping[str, Any]
) -> tuple[CandidateSpec, dict[str, Any], bool]:
    selection = config["calibration_selection"]

    def eligible(row: Mapping[str, Any]) -> bool:
        return (
            float(row["takeover_rate"]) >= float(selection["minimum_takeover_rate"])
            and float(row["mean_field_gain_over_component_damping"])
            >= float(selection["minimum_mean_field_gain_over_component_damping"])
            and float(row["worst_case_field_gain_over_component_damping"])
            >= float(selection["minimum_worst_case_field_gain_over_component_damping"])
            and float(row["case_harm_over_one_percent_rate_vs_component_damping"])
            <= float(selection["maximum_case_harm_over_one_percent_rate_vs_component_damping"])
        )

    safe = [row for row in aggregates if eligible(row)]
    pool = safe if safe else aggregates
    feature_counts = {
        spec.candidate_id: len(spec.model.feature_names) for spec in specs
    }
    chosen = max(
        pool,
        key=lambda row: (
            float(row["mean_field_gain_over_component_damping"]),
            float(row["evaluator_mean_adjoint_residual_gain_over_component_damping"]),
            -int(row["basis_rank"]),
            -feature_counts[str(row["candidate_id"])],
            float(row["ridge_alpha"]),
        ),
    )
    by_id = {spec.candidate_id: spec for spec in specs}
    return by_id[str(chosen["candidate_id"])], chosen, bool(safe)


def _selected_evaluation_rows(
    states: list[PreparedCase], spec: CandidateSpec, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    partitions = set(config["evaluated_partitions"])
    controls = config["strong_controls"]

    def append_correction(
        state: PreparedCase,
        *,
        candidate_id: str,
        correction_normalized: torch.Tensor,
        fallback: bool = False,
        fallback_reason: str | None = None,
        takeover: bool = True,
        evaluator_only: bool = False,
        high_order_forward_calls: int = 0,
        high_order_adjoint_calls: int = 0,
        deployment_setup_seconds: float = 0.0,
        evaluator_setup_seconds: float = 0.0,
        model_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        row = _correction_row(
            state,
            candidate_id=candidate_id,
            correction_normalized=correction_normalized,
            config=config,
            fallback=fallback,
            fallback_reason=fallback_reason,
            takeover=takeover,
            evaluator_only=evaluator_only,
            high_order_forward_calls=high_order_forward_calls,
            high_order_adjoint_calls=high_order_adjoint_calls,
            deployment_setup_seconds=deployment_setup_seconds,
            evaluator_setup_seconds=evaluator_setup_seconds,
            model_metadata=model_metadata,
        )
        rows.append(row)
        diagnostic_rows.append(
            _adjoint_diagnostic_row(
                state,
                candidate_id=candidate_id,
                correction_normalized=correction_normalized,
            )
        )
        return row

    for state in states:
        if state.record.partition not in partitions:
            continue
        rows.append(_low_reference_row(state, config))
        append_correction(
            state,
            candidate_id="component_damping",
            correction_normalized=state.damping_normalized,
        )
        correction, metadata = _candidate_prediction(state, spec, fail_closed=True)
        append_correction(
            state,
            candidate_id="selected_fail_closed",
            correction_normalized=correction,
            fallback=bool(metadata["fallback"]),
            fallback_reason=metadata["fallback_reason"],
            takeover=bool(metadata["takeover"]),
            deployment_setup_seconds=state.visible_feature_seconds
            + float(metadata["model_inference_seconds"]),
            model_metadata=metadata,
        )
        raw_correction, raw_metadata = _candidate_prediction(state, spec, fail_closed=False)
        append_correction(
            state,
            candidate_id="selected_raw_unbounded_diagnostic",
            correction_normalized=raw_correction,
            evaluator_only=True,
            deployment_setup_seconds=state.visible_feature_seconds
            + float(raw_metadata["model_inference_seconds"]),
            model_metadata=raw_metadata,
        )
        append_correction(
            state,
            candidate_id="damping_plus_train_basis_mean",
            correction_normalized=state.damping_normalized
            + spec.basis.mean.reshape_as(state.damping_normalized),
        )

        teacher = HighOrderTeacherMaps(state.record.case.inference.operator)
        teacher_started = time.perf_counter()
        teacher_correction = teacher.correction(
            state.warm_field, low_projection=state.warm_projection
        ) / state.signal_scale
        teacher_seconds = time.perf_counter() - teacher_started
        if teacher.call_report() != {"forward_calls": 1, "adjoint_calls": 0}:
            raise RuntimeError("high-order teacher control call ledger drifted")
        for beta_value in controls["include_high_order_teacher_interpolations"]:
            beta = float(beta_value)
            correction_teacher = state.damping_normalized + beta * (
                teacher_correction - state.damping_normalized
            )
            append_correction(
                state,
                candidate_id=f"high_order_teacher_b{str(beta).replace('.', 'p')}",
                correction_normalized=correction_teacher,
                evaluator_only=True,
                high_order_forward_calls=1,
                evaluator_setup_seconds=teacher_seconds,
            )

        exact_residual = _target_residual(state).reshape(-1)
        if controls["include_measurement_l2_low_rank_oracle"]:
            measurement_started = time.perf_counter()
            measurement_coefficients = measurement_optimal_coefficients(
                spec.basis, exact_residual
            )
            measurement_residual = spec.basis.synthesize(measurement_coefficients)
            measurement_seconds = time.perf_counter() - measurement_started
            append_correction(
                state,
                candidate_id="measurement_l2_low_rank_oracle",
                correction_normalized=state.damping_normalized
                + measurement_residual.reshape_as(state.damping_normalized),
                evaluator_only=True,
                evaluator_setup_seconds=measurement_seconds,
            )
        if controls["include_adjoint_low_rank_oracle"]:
            operator = state.record.case.inference.operator
            _, adjoint = n15a._operator_maps(operator)
            operator.reset_call_counts()
            adjoint_started = time.perf_counter()
            adjoint_target = adjoint_optimal_coefficients(
                spec.basis,
                exact_residual,
                observation_shape=state.record.mismatch_normalized.shape,
                adjoint=adjoint,
                l2=spec.target_l2,
            )
            adjoint_seconds = time.perf_counter() - adjoint_started
            if operator.call_report() != {
                "forward_calls": 0,
                "adjoint_calls": spec.rank + 1,
            }:
                raise RuntimeError("fresh adjoint oracle call ledger drifted")
            adjoint_residual = spec.basis.synthesize(adjoint_target.coefficients)
            row = append_correction(
                state,
                candidate_id="adjoint_low_rank_oracle",
                correction_normalized=state.damping_normalized
                + adjoint_residual.reshape_as(state.damping_normalized),
                evaluator_only=True,
                evaluator_setup_seconds=adjoint_seconds,
            )
            row["coefficient_target_evaluator_adjoint_calls"] = (
                adjoint_target.evaluator_adjoint_calls
            )
        if controls["include_exact_mismatch_oracle"]:
            append_correction(
                state,
                candidate_id="exact_mismatch_oracle",
                correction_normalized=state.record.mismatch_normalized,
                evaluator_only=True,
            )
    return rows, diagnostic_rows


def _aggregate_selected(
    rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
    partition: str,
) -> list[dict[str, Any]]:
    selected = [row for row in rows if row["partition"] == partition]
    low = _index_rows(selected, "low_cgls24_matched")
    damping = _index_rows(selected, "component_damping")
    teacher = _index_rows(selected, "high_order_teacher_b0p75")
    candidate_ids = sorted({str(row["candidate_id"]) for row in selected})
    return [
        _aggregate_candidate(
            selected,
            candidate_id=candidate_id,
            low_rows=low,
            damping_rows=damping,
            teacher_rows=teacher,
            adjoint_diagnostic_rows=[
                row for row in diagnostic_rows if row["partition"] == partition
            ],
        )
        for candidate_id in candidate_ids
    ]


def _future_gate(selected: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    gates = config["future_confirmation_gates"]
    checks = {
        "mean_field_gain_over_low": float(selected["mean_field_gain_over_low_cgls24"])
        >= float(gates["mean_field_gain_over_low_cgls24_minimum"]),
        "mean_h1_gain_over_low": float(selected["mean_h1_gain_over_low_cgls24"])
        >= float(gates["mean_h1_gain_over_low_cgls24_minimum"]),
        "mean_field_gain_over_damping": float(
            selected["mean_field_gain_over_component_damping"]
        )
        >= float(gates["mean_field_gain_over_component_damping_minimum"]),
        "mean_field_gain_over_high_order_teacher": float(
            selected["mean_field_gain_over_high_order_teacher_b0p75"]
        )
        >= float(gates["mean_field_gain_over_high_order_teacher_b0p75_minimum"]),
        "worst_case_field_gain": float(selected["worst_case_field_gain_over_low_cgls24"])
        >= float(gates["worst_case_field_gain_over_low_cgls24_minimum"]),
        "case_harm_rate": float(selected["case_harm_over_one_percent_rate_vs_low"])
        <= float(gates["case_field_harm_over_one_percent_rate_maximum"]),
        "case_harm_rate_vs_component_damping": float(
            selected["case_harm_over_one_percent_rate_vs_component_damping"]
        )
        <= float(
            gates[
                "case_field_harm_over_one_percent_rate_vs_component_damping_maximum"
            ]
        ),
        "takeover_rate": float(selected["takeover_rate"])
        >= float(gates["minimum_takeover_rate"]),
        "evaluator_adjoint_residual_gain": float(
            selected["evaluator_mean_adjoint_residual_gain_over_component_damping"]
        )
        >= float(
            gates[
                "evaluator_mean_adjoint_residual_gain_over_component_damping_minimum"
            ]
        ),
        "zero_high_order_forward_calls": int(selected["high_order_forward_calls"])
        == int(gates["required_high_order_forward_calls"]),
        "zero_high_order_adjoint_calls": int(selected["high_order_adjoint_calls"])
        == int(gates["required_high_order_adjoint_calls"]),
    }
    return {"checks": checks, "passed": all(checks.values())}


def _serialize_model(spec: CandidateSpec) -> dict[str, Any]:
    return {
        "schema": "jacru-n1-6-selected-model-1.0",
        "candidate_id": spec.candidate_id,
        "feature_set": spec.feature_set,
        "feature_names": list(spec.model.feature_names),
        "feature_mean": spec.model.feature_mean.tolist(),
        "feature_scale": spec.model.feature_scale.tolist(),
        "ridge_weights": spec.model.weights.tolist(),
        "ridge_alpha": spec.ridge_alpha,
        "basis_rank": spec.rank,
        "basis_mean": spec.basis.mean.tolist(),
        "basis_vectors": spec.basis.vectors.tolist(),
        "target_l2": spec.target_l2,
        "residual_shrinkage": spec.shrinkage,
        "coefficient_abs_limits": spec.coefficient_limits.tolist(),
        "feature_max_abs_z_limit": spec.feature_limit,
        "residual_rms_limit": spec.residual_rms_limit,
        "deployment_high_order_forward_calls": 0,
        "deployment_high_order_adjoint_calls": 0,
        "truth_or_exact_mismatch_required_at_inference": False,
    }


def _plot(
    calibration_rows: list[dict[str, Any]],
    development_rows: list[dict[str, Any]],
    case_rows: list[dict[str, Any]],
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    x = [
        row["evaluator_mean_adjoint_residual_gain_over_component_damping"]
        for row in calibration_rows
    ]
    y = [row["mean_field_gain_over_component_damping"] for row in calibration_rows]
    coverage = [row["takeover_rate"] for row in calibration_rows]
    scatter = axes[0].scatter(x, y, c=coverage, cmap="viridis", vmin=0.0, vmax=1.0, s=22)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].axvline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("Calibration model grid")
    axes[0].set_xlabel("adjoint-residual gain vs damping")
    axes[0].set_ylabel("field gain vs damping")
    fig.colorbar(scatter, ax=axes[0], label="takeover rate")

    preferred = [
        "component_damping",
        "damping_plus_train_basis_mean",
        "selected_fail_closed",
        "high_order_teacher_b0p75",
        "measurement_l2_low_rank_oracle",
        "adjoint_low_rank_oracle",
        "exact_mismatch_oracle",
    ]
    by_id = {row["candidate_id"]: row for row in development_rows}
    labels = [label for label in preferred if label in by_id]
    values = [by_id[label]["mean_field_gain_over_low_cgls24"] for label in labels]
    axes[1].barh(labels, values, color=["#477c72" if "oracle" not in label else "#a45b45" for label in labels])
    axes[1].axvline(0.05, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_title("Opened development field gate")
    axes[1].set_xlabel("mean field gain vs matched low CGLS-24")

    dev_selected = [
        row
        for row in case_rows
        if row["partition"] == "development" and row["candidate_id"] == "selected_fail_closed"
    ]
    low = _index_rows(case_rows, "low_cgls24_matched")
    cluster_values = []
    for seed in sorted({int(row["base_seed"]) for row in dev_selected}):
        chosen = [row for row in dev_selected if int(row["base_seed"]) == seed]
        gains = [
            1.0
            - float(row["field_relative_l2"])
            / max(float(low[(seed, str(row["family"]))]["field_relative_l2"]), 1e-30)
            for row in chosen
        ]
        cluster_values.append((seed, float(np.mean(gains))))
    axes[2].bar([str(seed) for seed, _ in cluster_values], [value for _, value in cluster_values], color="#6d7f42")
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].axhline(0.05, color="black", linestyle="--", linewidth=1.0)
    axes[2].set_title("Selected route by geometry cluster")
    axes[2].set_xlabel("development geometry seed")
    axes[2].set_ylabel("paired-family field gain vs low")
    fig.suptitle("N1.6 adjoint-weighted low-rank post-open screen", fontsize=15)
    fig.tight_layout()
    fig.savefig(output / "diagnostic.png", dpi=180)
    fig.savefig(output / "diagnostic.pdf")
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    config_path = args.config.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.replace_output:
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    config = _read_json(config_path)
    n15a_config_path = ROOT / config["source_n1_5_a_config"]
    n15a_config = _read_json(n15a_config_path)
    source_path = ROOT / config["source_t0_config"]
    source = _read_json(source_path)
    confirmation_manifest_path = (
        ROOT / config["source_n1_5_confirmation"] / "case_manifest.csv"
    )
    preregistration_path = ROOT / config["preregistration_document"]
    confirmation_manifest = _read_csv(confirmation_manifest_path)
    _validate_config(config, n15a_config, seed_limit=args.seed_limit)
    states, manifest = _prepare_cases(
        config, n15a_config, source, seed_limit=args.seed_limit
    )
    split_audit = _validate_split_integrity(
        manifest,
        confirmation_manifest,
        families=[str(value) for value in n15a_config["families"]],
    )
    specs, fit_rows = _fit_candidate_specs(states, config)
    _, calibration_cases, calibration_diagnostics, calibration_aggregates = _calibration_grid(
        states, specs, config
    )
    selected_spec, selected_calibration, calibration_had_safe_candidate = _select_candidate(
        specs, calibration_aggregates, config
    )
    selected_rows, selected_diagnostics = _selected_evaluation_rows(
        states, selected_spec, config
    )
    calibration_selected_aggregates = _aggregate_selected(
        selected_rows, selected_diagnostics, "calibration"
    )
    development_aggregates = _aggregate_selected(
        selected_rows, selected_diagnostics, "development"
    )
    selected_development = next(
        row
        for row in development_aggregates
        if row["candidate_id"] == "selected_fail_closed"
    )
    gate = _future_gate(selected_development, config)
    status = (
        "POSTOPEN_ROUTE_ELIGIBLE_REQUIRES_FROZEN_CONFIRMATION"
        if calibration_had_safe_candidate and gate["passed"]
        else "POSTOPEN_NO_GO_NO_CONFIRMATION_ROUTE"
    )

    summary = {
        "schema": REPORT_SCHEMA,
        "status": status,
        "evidence_level": config["evidence_level"],
        "runtime_seconds": time.perf_counter() - started,
        "seed_limit": args.seed_limit,
        "independent_unit": "base_seed_geometry_cluster",
        "paired_families_are_not_independent_rigs": True,
        "fit_geometry_cluster_count": len(
            {state.record.base_seed for state in states if state.record.partition == "fit"}
        ),
        "calibration_geometry_cluster_count": len(
            {state.record.base_seed for state in states if state.record.partition == "calibration"}
        ),
        "development_geometry_cluster_count": len(
            {state.record.base_seed for state in states if state.record.partition == "development"}
        ),
        "candidate_grid_count": len(specs),
        "split_integrity_audit": split_audit,
        "calibration_had_safe_candidate": calibration_had_safe_candidate,
        "selected_model": _serialize_model(selected_spec),
        "selected_calibration_aggregate": selected_calibration,
        "selected_development_aggregate": selected_development,
        "future_confirmation_gate": gate,
        "calibration_selected_controls": calibration_selected_aggregates,
        "development_controls": development_aggregates,
        "deployment_contract": config["deployment_contract"],
        "runtime_observation_source": "case.inference.observations_uv",
        "truth_derived_adjoint_diagnostics_are_separate_evaluator_rows": True,
        "budget": config["budget"],
        "claim_boundary": config["claim_boundary"],
        "opens_ood_fresh_or_final": False,
        "may_claim_confirmed_algorithm_gain": False,
        "real_bost_claim": False,
        "physics_scope": (
            "continuous analytic-gradient renderer versus voxel finite-difference/"
            "trilinear representation mismatch only"
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "selected_model.json").write_text(
        json.dumps(_serialize_model(selected_spec), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "config_snapshot.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output / "case_manifest.csv", manifest)
    _write_csv(output / "fit_adjoint_target_rows.csv", fit_rows)
    _write_csv(output / "calibration_model_rows.csv", calibration_aggregates)
    _write_csv(output / "calibration_case_rows.csv", calibration_cases)
    _write_csv(
        output / "calibration_adjoint_diagnostic_rows.csv", calibration_diagnostics
    )
    _write_csv(output / "selected_case_metrics.csv", selected_rows)
    _write_csv(
        output / "selected_adjoint_diagnostic_rows.csv", selected_diagnostics
    )
    _write_csv(
        output / "selected_aggregate_metrics.csv",
        calibration_selected_aggregates + development_aggregates,
    )
    _plot(calibration_aggregates, development_aggregates, selected_rows, output)
    provenance = {
        "schema": "jacru-n1-6-provenance-1.0",
        "git_commit": _git_commit(),
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "source_n1_5_a_config_sha256": _sha256(n15a_config_path),
        "source_n1_5_confirmation_manifest_sha256": _sha256(
            confirmation_manifest_path
        ),
        "preregistration_document_sha256": _sha256(preregistration_path),
        "source_t0_config_sha256": _sha256(source_path),
        "runner_sha256": _sha256(Path(__file__)),
        "model_module_sha256": _sha256(
            ROOT / "demo_t16_operator/jacru_n1_6_adjoint_low_rank.py"
        ),
        "development_was_already_opened_for_prior_n1_5_hypotheses": True,
        "confirmation_or_ood_opened": False,
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    readme = f"""# N1.6 adjoint-weighted low-rank post-open screen

Status: **{status}**.

This package is an opened synthetic development diagnostic.  It does not prove
an algorithm gain, real-BOST performance, OOD transfer, or novelty.

- Deployment candidate inputs: geometry, measured observation, low-CGLS warm
  field, and its low projection.
- Deployment candidate high-order calls: 0 forward / 0 adjoint.
- Matched production budget: 25 low forward / 24 low adjoint calls.
- PCA basis and coefficient targets use fit only; model selection uses
  calibration only.  Development was already open during N1.5 and can only
  generate a future hypothesis.
- Exact mismatch, measurement-L2 coefficient, and adjoint coefficient rows are
  evaluator-only ceilings.
- Selected route: `{selected_spec.candidate_id}`.
- Opened development field gain versus matched low CGLS-24:
  {selected_development['mean_field_gain_over_low_cgls24']:.6f}.
- Opened development field gain versus component damping:
  {selected_development['mean_field_gain_over_component_damping']:.6f}.
- Future confirmation gate passed: `{gate['passed']}`.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    _write_checksums(output)
    print(
        json.dumps(
            {
                "status": status,
                "selected": selected_spec.candidate_id,
                "development": selected_development,
                "future_gate": gate,
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
