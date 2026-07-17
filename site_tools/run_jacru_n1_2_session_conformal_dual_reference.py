#!/usr/bin/env python3
"""Run the opened-synthetic JACRU N1.2 session-conformal development screen.

N1.2 rebuilds every train/development/OOD observation with an absolute,
target-independent nuisance model.  Two or three fields sharing one geometry
form a session and consume exactly one flow-off fit/calibration/audit packet.
Candidate selectors see only their frozen calibration payload.  Dense
``A A^T`` solves remain an evaluator ceiling outside the reconstruction budget.

This development runner is not a preregistered or confirmatory experiment.
It exists to falsify the protocol before a formal configuration is frozen.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping
import uuid

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo_t16_operator.jacru_m2_learned_residual import prepare_jacru_m2_batch
from demo_t16_operator.jacru_n1_2_dual_reference import (
    add_dual_reference_metrics,
    aggregate_dual_reference_rows,
    dual_reference_decisions,
)
from demo_t16_operator.jacru_n1_2_session_conformal import (
    CandidateAuditCoverage,
    SelectorPayload,
    audit_candidate_coverage,
    build_session_packet,
    calibrate_candidate_selector,
    score_selector_residual,
    selector_gate_checks,
    stable_digest,
    stable_seed,
    verify_session_packet,
)
from demo_t16_operator.jacru_synthetic_fixture import build_jacru_synthetic_case
from demo_t16_operator.psu_b0_streaming_operator import zero_outer_boundary_support
from site_tools import run_jacru_m2_1_data_consistency_diagnostic as m21
from site_tools import run_jacru_m2_2_exact_nullspace_oracle as m22
from site_tools import run_jacru_m2_learned_residual_gate as m2


DEFAULT_CONFIG = (
    ROOT
    / "demo_t16_operator/configs/"
    "jacru_n1_2_session_conformal_dual_reference_development_v1.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "demo_t16_operator/results/"
    "jacru_n1_2_session_conformal_dual_reference_scratch"
)


@dataclass(frozen=True)
class DenseMultiGateResult:
    field: torch.Tensor
    alpha: float
    selector_valid: bool
    gate_checks: dict[str, bool]
    raw_global_score: float
    selected_global_score: float
    raw_camera_scores: tuple[float, ...]
    selected_camera_scores: tuple[float, ...]
    correction_norm: float
    residual_closure_relative_error: float
    evaluated_alpha_count: int
    fallback_reason: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-mode", choices=("scratch", "formal"), default="scratch")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed-limit", type=int)
    parser.add_argument("--replace-output", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _mean(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    if not materialized:
        raise ValueError("cannot average an empty sequence")
    return math.fsum(materialized) / len(materialized)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _source_manifest(config_path: Path, config: Mapping[str, Any]) -> dict[str, str]:
    paths = [
        config_path,
        ROOT / str(config["source_t0_config"]),
        ROOT / str(config["source_n1_1_results"]) / "summary.json",
        ROOT / "demo_t16_operator/jacru_synthetic_fixture.py",
        ROOT / "demo_t16_operator/jacru_m2_learned_residual.py",
        ROOT / "demo_t16_operator/jacru_m2_comparators.py",
        ROOT / "demo_t16_operator/jacru_n1_2_session_conformal.py",
        ROOT / "demo_t16_operator/jacru_n1_flowoff_covariance.py",
        ROOT / "demo_t16_operator/jacru_n1_2_dual_reference.py",
        ROOT / "demo_t16_operator/jacru_m2_exact_nullspace_oracle.py",
        ROOT / "demo_t16_operator/analytic_bost_phantoms.py",
        ROOT / "demo_t16_operator/spatial_reconstruction_metrics.py",
        ROOT / "demo_t16_operator/interface_baselines.py",
        ROOT / "demo_t16_operator/psu_b0_reconstruction_interface.py",
        ROOT / "site_tools/run_jacru_m2_learned_residual_gate.py",
        ROOT / "site_tools/run_jacru_m2_1_data_consistency_diagnostic.py",
        ROOT / "site_tools/run_jacru_m2_2_exact_nullspace_oracle.py",
        Path(__file__).resolve(),
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"source manifest is incomplete: {missing}")
    return {str(path.relative_to(ROOT)): _sha256(path) for path in paths}


def _validate_run_contract(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    if args.run_mode == "formal":
        if config.get("status") != "FROZEN_BEFORE_FIRST_FORMAL_N1_2_EXECUTION":
            raise RuntimeError("formal mode requires a separately frozen formal config")
        if args.epochs is not None or args.seed_limit is not None or args.replace_output:
            raise RuntimeError("formal mode forbids epochs, seed-limit, and replacement overrides")
    elif config.get("status") != "DEVELOPMENT_ONLY_NOT_PREREGISTERED_NOT_FORMAL":
        raise RuntimeError("scratch mode requires the explicit development-only config")
    if args.seed_limit is not None and args.seed_limit < 1:
        raise ValueError("seed-limit must be positive")
    if args.epochs is not None and args.epochs < 1:
        raise ValueError("epochs must be positive")


def _session_scale_uv(
    *,
    camera_index: torch.Tensor,
    session_id: str,
    geometry_digest: str,
    nuisance: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Create a target-independent synthetic background-confidence scale map."""

    cameras = torch.as_tensor(camera_index).detach().cpu().to(torch.int64)
    camera_count = int(torch.max(cameras)) + 1
    camera_multipliers = torch.tensor(
        nuisance["camera_scale_multipliers"], dtype=torch.float64
    )
    component_multipliers = torch.tensor(
        nuisance["component_scale_multipliers"], dtype=torch.float64
    )
    if camera_multipliers.shape != (camera_count,) or component_multipliers.shape != (2,):
        raise ValueError("session scale multipliers do not match camera/component count")
    nominal = float(nuisance["nominal_iid_std_benchmark_units"])
    log_std = float(nuisance["session_log_scale_std"])
    hetero = float(nuisance["heteroscedastic_fraction"])
    if nominal <= 0.0 or log_std < 0.0 or not 0.0 <= hetero <= 1.0:
        raise ValueError("invalid session nuisance scale policy")
    generator = torch.Generator().manual_seed(
        stable_seed(
            nuisance["manifest_id"],
            session_id,
            geometry_digest,
            "session-log-scale",
            int(nuisance["seed"]),
        )
    )
    session_multiplier = float(
        torch.exp(log_std * torch.randn((), generator=generator, dtype=torch.float64))
    )
    scale = torch.empty((cameras.numel(), 2), dtype=torch.float64)
    for camera in range(camera_count):
        indices = torch.nonzero(cameras == camera, as_tuple=False).reshape(-1)
        ranks = torch.arange(indices.numel(), dtype=torch.float64)
        phase = (ranks + 0.5) / max(float(indices.numel()), 1.0)
        texture_factor = 1.0 + hetero * (0.5 + 0.5 * torch.sin(2.0 * math.pi * phase))
        scale[indices] = (
            nominal
            * session_multiplier
            * camera_multipliers[camera]
            * texture_factor[:, None]
            * component_multipliers[None, :]
        )
    return scale, {
        "session_scale_multiplier": session_multiplier,
        "scale_minimum": float(torch.min(scale)),
        "scale_mean": float(torch.mean(scale)),
        "scale_maximum": float(torch.max(scale)),
    }


def _prepare_session_records(
    source_config: dict[str, Any],
    config: Mapping[str, Any],
) -> tuple[
    list[m2.PreparedRecord],
    dict[str, Any],
    dict[str, str],
    list[dict[str, Any]],
]:
    """Build clean analytic fields, then add one nuisance packet per session."""

    clean_config = copy.deepcopy(source_config)
    clean_config["fixture"]["enable_noise"] = False
    clean_config["fixture"]["enable_camera_bias"] = False
    fixture = m2._fixture_config(clean_config)
    base_iterations = int(source_config["physical_budget"]["cgls_base_iterations"])
    expected_calls = base_iterations + 1
    support = zero_outer_boundary_support(fixture.grid_shape, dtype=torch.float64)
    nuisance = config["session_nuisance"]
    records: list[m2.PreparedRecord] = []
    packets: dict[str, Any] = {}
    case_to_session: dict[str, str] = {}
    manifest_rows: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()

    for split in ("train", "development", "ood"):
        split_spec = source_config["splits"][split]
        for base_seed in split_spec["base_seeds"]:
            clean_cases = [
                build_jacru_synthetic_case(
                    family=str(family),
                    split=split,
                    base_seed=int(base_seed),
                    config=fixture,
                )
                for family in split_spec["families"]
            ]
            geometry = clean_cases[0].inference.geometry
            if any(case.inference.geometry.digest != geometry.digest for case in clean_cases):
                raise RuntimeError("one synthetic session must share one geometry")
            session_id = f"{split}:{int(base_seed)}:{geometry.digest[:16]}"
            field_ids = tuple(
                f"{case.evaluation.family}:{case.inference.case_id}" for case in clean_cases
            )
            clean_fields = torch.cat(
                [case.evaluation.clean_observations_uv for case in clean_cases], dim=0
            )
            scale, scale_row = _session_scale_uv(
                camera_index=geometry.camera_index,
                session_id=session_id,
                geometry_digest=geometry.digest,
                nuisance=nuisance,
            )
            packet = build_session_packet(
                manifest_id=str(nuisance["manifest_id"]),
                session_id=session_id,
                geometry_digest=geometry.digest,
                field_ids=field_ids,
                camera_index=geometry.camera_index,
                flow_on_fields_uv=clean_fields,
                session_scale_uv=scale,
                camera_bias_relative_std=float(
                    nuisance["persistent_camera_bias_relative_std"]
                ),
                frame_camera_jitter_relative_std=float(
                    nuisance["frame_camera_jitter_relative_std"]
                ),
                fit_repeats=int(nuisance["fit_repeats"]),
                calibration_repeats=int(nuisance["threshold_calibration_repeats"]),
                audit_repeats=int(nuisance["audit_repeats"]),
                seed=int(nuisance["seed"]),
            )
            verify_session_packet(packet)
            if session_id in packets:
                raise RuntimeError("session IDs must be unique")
            packets[session_id] = packet
            manifest_rows.append(
                {
                    "manifest_id": packet.manifest_id,
                    "manifest_digest": packet.manifest_digest,
                    "session_id": session_id,
                    "split": split,
                    "base_seed": int(base_seed),
                    "geometry_digest": geometry.digest,
                    "field_count": len(clean_cases),
                    "field_ids_json": _canonical_json(packet.field_ids),
                    "packet_digest": packet.digest,
                    "fit_digest": packet.fit.digest,
                    "calibration_digest": packet.calibration.digest,
                    "audit_digest": packet.audit.digest,
                    "fit_repeats": int(nuisance["fit_repeats"]),
                    "threshold_calibration_repeats": int(
                        nuisance["threshold_calibration_repeats"]
                    ),
                    "audit_repeats": int(nuisance["audit_repeats"]),
                    "target_amplitude_used_for_scale": False,
                    **scale_row,
                }
            )

            for index, clean_case in enumerate(clean_cases):
                observation = packet.flow_on_observations_uv[index : index + 1]
                new_case_id = hashlib.sha256(
                    _canonical_json(
                        {
                            "schema": config["schema_version"],
                            "clean_case_id": clean_case.inference.case_id,
                            "packet_digest": packet.digest,
                            "field_index": index,
                        }
                    ).encode("ascii")
                ).hexdigest()[:20]
                if new_case_id in seen_case_ids:
                    raise RuntimeError("N1.2 case IDs must be unique")
                seen_case_ids.add(new_case_id)
                inference = replace(
                    clean_case.inference,
                    case_id=new_case_id,
                    observations_uv=observation.clone(),
                    observation_digest=stable_digest(
                        observation,
                        metadata={"case_id": new_case_id, "packet_digest": packet.digest},
                    ),
                    observation_generator_schema="jacru-n1-2-session-nuisance-1.0",
                )
                total_nuisance = observation - clean_case.evaluation.clean_observations_uv
                evaluation = replace(
                    clean_case.evaluation,
                    case_id=new_case_id,
                    additive_noise_uv=total_nuisance.clone(),
                    camera_bias_uv=torch.zeros(
                        (geometry.camera_count, 2), dtype=torch.float64
                    ),
                )
                case = replace(clean_case, inference=inference, evaluation=evaluation)
                operator = case.inference.operator
                operator.reset_call_counts()
                batch = prepare_jacru_m2_batch(
                    case.inference,
                    support=support,
                    cgls_iterations=base_iterations,
                    model_dtype=torch.float32,
                    model_device="cpu",
                )
                calls = operator.call_report()
                if calls != {
                    "forward_calls": expected_calls,
                    "adjoint_calls": expected_calls,
                }:
                    raise RuntimeError("N1.2 feature preparation budget drifted")
                records.append(
                    m2.PreparedRecord(
                        split=split,
                        family=clean_case.evaluation.family,
                        base_seed=int(base_seed),
                        case=case,
                        batch=batch,
                        preparation_calls=calls,
                    )
                )
                case_to_session[new_case_id] = session_id

    expected_sessions = sum(
        len(source_config["splits"][split]["base_seeds"])
        for split in ("train", "development", "ood")
    )
    expected_records = sum(
        len(source_config["splits"][split]["base_seeds"])
        * len(source_config["splits"][split]["families"])
        for split in ("train", "development", "ood")
    )
    if len(packets) != expected_sessions or len(records) != expected_records:
        raise RuntimeError("N1.2 session/record cardinality drifted")
    return records, packets, case_to_session, manifest_rows


def _norm_cache(
    records: list[m2.PreparedRecord], source_config: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    budget = source_config["physical_budget"]
    for record in records:
        if record.split == "train":
            continue
        digest = record.case.inference.geometry.digest
        if digest not in output:
            output[digest] = m2._dense_norm_squared_bound(
                record.case.inference.operator,
                batch_size=int(budget["dense_norm_batch_size"]),
                safety_factor=float(budget["dense_norm_safety_factor"]),
            )
    return output


def _matched_baselines(
    records: list[m2.PreparedRecord],
    source_config: dict[str, Any],
    norm_cache: dict[str, dict[str, Any]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    rows = m21._matched_baseline_rows(
        records=records,
        source_config=source_config,
        diagnostic_config={"step_safety_factor": 0.9},
        norm_cache=norm_cache,
        steps=[int(config["registered_classical_reference"]["projection_step"])],
    )
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["case_id"]), str(row["baseline_kind"]))
        if key in lookup:
            raise RuntimeError(f"duplicate matched baseline: {key}")
        lookup[key] = row
    expected = 3 * sum(record.split != "train" for record in records)
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} matched baseline rows")
    return rows, lookup


def _coverage_event(
    audit: CandidateAuditCoverage, gate_policy: str
) -> Any:
    if gate_policy == "joint_two_sided":
        return audit.joint_band
    if gate_policy == "joint_upper_only":
        return audit.joint_upper
    if gate_policy == "global_upper_only":
        return audit.global_upper
    raise ValueError(f"unsupported gate policy: {gate_policy}")


def _calibrate_candidates(
    *,
    packets: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[
    dict[tuple[str, str], SelectorPayload],
    list[dict[str, Any]],
    dict[str, bool],
]:
    selectors: dict[tuple[str, str], SelectorPayload] = {}
    rows: list[dict[str, Any]] = []
    for session_id, packet in sorted(packets.items()):
        split = session_id.split(":", 1)[0]
        for candidate in config["candidates"]:
            selector = calibrate_candidate_selector(
                candidate_id=str(candidate["id"]),
                fit_payload=packet.fit,
                calibration_payload=packet.calibration,
                proximal_covariance_policy=str(
                    candidate["proximal_covariance_policy"]
                ),
                selector_covariance_policy=str(
                    candidate["selector_covariance_policy"]
                ),
                mean_policy=str(candidate["mean_policy"]),
                shrinkage=float(config["covariance_estimator"]["shrinkage"]),
                ridge_fraction=float(
                    config["covariance_estimator"]["ridge_fraction"]
                ),
                target_two_sided_coverage=float(
                    candidate["target_two_sided_coverage"]
                ),
            )
            audit = audit_candidate_coverage(
                audit_payload=packet.audit,
                selector_payload=selector,
            )
            candidate_id = str(candidate["id"])
            selectors[(session_id, candidate_id)] = selector
            event = _coverage_event(audit, str(candidate["gate_policy"]))
            condition_number = float(
                torch.linalg.cond(selector.selector_covariance)
            )
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "selector_digest": selector.digest,
                    "session_id": session_id,
                    "manifest_digest": selector.manifest_digest,
                    "split": split,
                    "gate_policy": candidate["gate_policy"],
                    "target_two_sided_coverage": candidate[
                        "target_two_sided_coverage"
                    ],
                    "joint_tail_quantile": selector.joint_tail_quantile,
                    "calibration_sample_count": selector.calibration_sample_count,
                    "joint_tail_order": selector.joint_tail_order,
                    "finite_sample_two_sided_lower_bound": audit.nominal_two_sided_coverage_lower_bound,
                    "same_session_repeat_event_successes": event.successes,
                    "same_session_repeat_event_sample_count": event.sample_count,
                    "same_session_repeat_event_coverage": event.coverage,
                    "same_session_repeat_event_ci_lower": event.confidence_interval[0],
                    "same_session_repeat_event_ci_upper": event.confidence_interval[1],
                    "same_session_repeat_event_ci_method": event.interval_method,
                    "global_band_coverage": audit.global_band.coverage,
                    "global_upper_coverage": audit.global_upper.coverage,
                    "joint_upper_coverage": audit.joint_upper.coverage,
                    "joint_lower_coverage": audit.joint_lower.coverage,
                    "joint_two_sided_coverage": audit.joint_band.coverage,
                    "per_camera_band_coverages_json": _canonical_json(
                        [value.coverage for value in audit.per_camera_bands]
                    ),
                    "selector_covariance_condition_number": condition_number,
                    "audit_digest": audit.digest,
                }
            )
    gates = config["calibration_gates"]
    decisions: dict[str, bool] = {}
    for candidate in config["candidates"]:
        candidate_id = str(candidate["id"])
        selected = [row for row in rows if row["candidate_id"] == candidate_id]
        selected = [row for row in selected if row["split"] != "train"]
        checks = [
            _mean(row["same_session_repeat_event_coverage"] for row in selected)
            >= float(gates["same_session_repeat_coverage_mean_minimum"]),
            min(float(row["same_session_repeat_event_coverage"]) for row in selected)
            >= float(gates["same_session_repeat_coverage_minimum_packet"]),
            min(float(row["same_session_repeat_event_ci_lower"]) for row in selected)
            >= float(
                gates["same_session_repeat_clopper_pearson_lower_minimum"]
            ),
            max(
                float(row["selector_covariance_condition_number"])
                for row in selected
            )
            <= float(gates["condition_number_maximum"]),
        ]
        decisions[candidate_id] = all(checks)
    return selectors, rows, decisions


def _gate_policy_pass(checks: Mapping[str, bool], policy: str) -> bool:
    if policy == "global_upper_only":
        return bool(checks["global_upper"])
    if policy == "joint_upper_only":
        return bool(checks["global_upper"] and checks["camera_upper"])
    if policy == "joint_two_sided":
        return bool(
            checks["global_upper"]
            and checks["camera_upper"]
            and checks["global_lower"]
            and checks["camera_lower"]
            and checks.get("global_lower_informative", True)
            and checks.get("camera_lower_informative", True)
        )
    raise ValueError(f"unsupported gate policy: {policy}")


def _gate_checks_with_optional_mismatch(
    *,
    scores: Any,
    selector: SelectorPayload,
    evaluator_mismatch_scores: Any | None,
) -> dict[str, bool]:
    if evaluator_mismatch_scores is None:
        return {
            **selector_gate_checks(scores, selector),
            "global_lower_informative": True,
            "camera_lower_informative": True,
        }
    mismatch_global = max(float(evaluator_mismatch_scores.global_score), 0.0)
    mismatch_camera = torch.clamp(
        evaluator_mismatch_scores.per_camera_scores.to(torch.float64), min=0.0
    )
    global_lower = max(
        0.0,
        math.sqrt(float(selector.global_bands[0])) - math.sqrt(mismatch_global),
    ) ** 2
    global_upper = (
        math.sqrt(float(selector.global_bands[1])) + math.sqrt(mismatch_global)
    ) ** 2
    camera_lower = torch.clamp(
        torch.sqrt(selector.per_camera_bands[:, 0]) - torch.sqrt(mismatch_camera),
        min=0.0,
    ).square()
    camera_upper = (
        torch.sqrt(selector.per_camera_bands[:, 1]) + torch.sqrt(mismatch_camera)
    ).square()
    global_lower_informative = mismatch_global < float(selector.global_bands[0])
    camera_lower_informative = bool(
        torch.all(mismatch_camera < selector.per_camera_bands[:, 0])
    )
    return {
        "global_upper": float(scores.global_score) <= global_upper,
        "global_lower": float(scores.global_score) >= global_lower,
        "camera_upper": bool(
            torch.all(scores.per_camera_scores <= camera_upper)
        ),
        "camera_lower": bool(
            torch.all(scores.per_camera_scores >= camera_lower)
        ),
        "global_lower_informative": global_lower_informative,
        "camera_lower_informative": camera_lower_informative,
    }


def _dense_multigate_ceiling(
    *,
    initial_field: torch.Tensor,
    target_observation_uv: torch.Tensor,
    dense_active_matrix: torch.Tensor,
    support_mask: torch.Tensor,
    selector: SelectorPayload,
    gate_policy: str,
    log10_alpha_bounds: tuple[float, float],
    alpha_grid_count: int,
    evaluator_mismatch_scores: Any | None = None,
) -> DenseMultiGateResult:
    """Evaluator-only dense path; audit payload and truth are structurally absent."""

    field = torch.as_tensor(initial_field).detach().cpu().to(torch.float64)
    support = torch.as_tensor(support_mask).detach().cpu().to(torch.bool)
    matrix = torch.as_tensor(dense_active_matrix).detach().cpu().to(torch.float64)
    target = torch.as_tensor(target_observation_uv).detach().cpu().to(torch.float64)
    if field.shape != support.shape or field.ndim != 3 or not bool(torch.any(support)):
        raise ValueError("field and support must share one nonempty 3D shape")
    if bool(torch.any(field.masked_select(~support) != 0.0)):
        raise ValueError("field must be zero outside support")
    if matrix.shape != (target.numel(), int(torch.count_nonzero(support))):
        raise ValueError("dense matrix shape does not match target/support")
    active = field.masked_select(support)
    residual0 = matrix @ active - target.reshape(-1)
    raw_scores = score_selector_residual(residual0.reshape_as(target), selector)
    raw_checks = _gate_checks_with_optional_mismatch(
        scores=raw_scores,
        selector=selector,
        evaluator_mismatch_scores=evaluator_mismatch_scores,
    )
    if _gate_policy_pass(raw_checks, gate_policy):
        return DenseMultiGateResult(
            field=field.clone(),
            alpha=math.inf,
            selector_valid=True,
            gate_checks=raw_checks,
            raw_global_score=raw_scores.global_score,
            selected_global_score=raw_scores.global_score,
            raw_camera_scores=tuple(float(v) for v in raw_scores.per_camera_scores),
            selected_camera_scores=tuple(
                float(v) for v in raw_scores.per_camera_scores
            ),
            correction_norm=0.0,
            residual_closure_relative_error=0.0,
            evaluated_alpha_count=0,
            fallback_reason="RAW_ALREADY_INSIDE_FROZEN_GATE",
        )

    lower_log, upper_log = (float(log10_alpha_bounds[0]), float(log10_alpha_bounds[1]))
    count = int(alpha_grid_count)
    if not math.isfinite(lower_log) or not math.isfinite(upper_log) or lower_log >= upper_log:
        raise ValueError("log10 alpha bounds must be finite and increasing")
    if count < 17:
        raise ValueError("alpha grid must contain at least 17 values")
    gram = matrix @ matrix.mT
    covariance = selector.proximal_covariance
    gram_scale = float(torch.mean(torch.diag(gram)))
    covariance_mean = float(torch.mean(torch.diag(covariance)))
    if gram_scale <= 0.0 or covariance_mean <= 0.0:
        raise ValueError("Gram and covariance scales must be positive")
    scaled_covariance = covariance * (gram_scale / covariance_mean)
    log_grid = torch.linspace(upper_log, lower_log, count, dtype=torch.float64)
    evaluated = 0
    for log_alpha in log_grid:
        evaluated += 1
        alpha = 10.0 ** float(log_alpha)
        dual = torch.linalg.solve(gram + alpha * scaled_covariance, residual0)
        candidate_active = active - matrix.mT @ dual
        residual = matrix @ candidate_active - target.reshape(-1)
        scores = score_selector_residual(residual.reshape_as(target), selector)
        checks = _gate_checks_with_optional_mismatch(
            scores=scores,
            selector=selector,
            evaluator_mismatch_scores=evaluator_mismatch_scores,
        )
        if not _gate_policy_pass(checks, gate_policy):
            continue
        candidate = torch.zeros_like(field)
        candidate.masked_scatter_(support, candidate_active)
        direct_residual = matrix @ candidate.masked_select(support) - target.reshape(-1)
        closure = float(torch.linalg.vector_norm(direct_residual - residual)) / max(
            float(torch.linalg.vector_norm(direct_residual)), 1e-30
        )
        return DenseMultiGateResult(
            field=candidate,
            alpha=alpha,
            selector_valid=True,
            gate_checks=checks,
            raw_global_score=raw_scores.global_score,
            selected_global_score=scores.global_score,
            raw_camera_scores=tuple(float(v) for v in raw_scores.per_camera_scores),
            selected_camera_scores=tuple(float(v) for v in scores.per_camera_scores),
            correction_norm=float(torch.linalg.vector_norm(candidate_active - active)),
            residual_closure_relative_error=closure,
            evaluated_alpha_count=evaluated,
            fallback_reason="NONE",
        )

    return DenseMultiGateResult(
        field=field.clone(),
        alpha=math.inf,
        selector_valid=False,
        gate_checks=raw_checks,
        raw_global_score=raw_scores.global_score,
        selected_global_score=raw_scores.global_score,
        raw_camera_scores=tuple(float(v) for v in raw_scores.per_camera_scores),
        selected_camera_scores=tuple(float(v) for v in raw_scores.per_camera_scores),
        correction_norm=0.0,
        residual_closure_relative_error=0.0,
        evaluated_alpha_count=evaluated,
        fallback_reason="NO_ALPHA_SATISFIED_FROZEN_GATE_RETURN_RAW",
    )


def _prepare_dense_evaluator(
    records: list[m2.PreparedRecord], config: Mapping[str, Any]
) -> tuple[
    dict[str, tuple[torch.Tensor, torch.Tensor]],
    list[dict[str, Any]],
]:
    matrices: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    rows: list[dict[str, Any]] = []
    dense = config["dense_evaluator_ceiling"]
    for record in records:
        if record.split == "train":
            continue
        operator = record.case.inference.operator
        digest = record.case.inference.geometry.digest
        if digest in matrices:
            continue
        started = time.perf_counter()
        matrix, setup = m22._assemble_active_matrix_batched(
            operator,
            support=operator.support,
            batch_size=int(dense["assembly_batch_size"]),
        )
        matrices[digest] = (matrix, operator.support.detach().cpu().to(torch.bool))
        rows.append(
            {
                "geometry_digest": digest,
                "matrix_rows": int(matrix.shape[0]),
                "matrix_columns": int(matrix.shape[1]),
                "matrix_rank": int(torch.linalg.matrix_rank(matrix)),
                "setup_forward_calls_batched": int(setup["setup_forward_calls"]),
                "setup_forward_equivalents_unbatched": int(matrix.shape[1] + 1),
                "assembly_batch_size": int(dense["assembly_batch_size"]),
                "setup_seconds": time.perf_counter() - started,
                "dense_setup_in_budget": False,
                "status": "EVALUATOR_DENSE_CEILING_NOT_PRIMARY_RUNTIME",
            }
        )
    return matrices, rows


def _base_and_mismatch_rows(
    *,
    records: list[m2.PreparedRecord],
    matrices: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    case_to_session: Mapping[str, str],
) -> tuple[dict[str, dict[str, Any]], dict[str, torch.Tensor], list[dict[str, Any]]]:
    base_scores: dict[str, dict[str, Any]] = {}
    mismatch_vectors: dict[str, torch.Tensor] = {}
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.split == "train":
            continue
        case = record.case
        case_id = case.inference.case_id
        matrix, support = matrices[case.inference.geometry.digest]
        base = record.batch.base_field[0, 0].to(case.inference.operator.support)
        base_scores[case_id] = m2._score_prediction(
            record=record,
            method="prepared_cgls_base_12",
            model_seed=-1,
            prediction=base,
            gate=None,
            correction_rms=0.0,
            optimization_forward_calls=12,
            optimization_adjoint_calls=12,
            grouped_adjoint_calls=0,
            neural_inference_seconds=0.0,
        )
        truth = case.evaluation.truth_volume[0, 0].detach().cpu().to(torch.float64)
        voxel_clean = matrix @ truth.masked_select(support)
        continuous_clean = case.evaluation.clean_observations_uv[0].reshape(-1)
        mismatch = (voxel_clean - continuous_clean).reshape_as(
            case.evaluation.clean_observations_uv[0]
        )
        mismatch_vectors[case_id] = mismatch
        rows.append(
            {
                "case_id": case_id,
                "session_id": case_to_session[case_id],
                "split": record.split,
                "family": record.family,
                "base_seed": record.base_seed,
                "geometry_digest": case.inference.geometry.digest,
                "voxel_vs_continuous_clean_relative_l2": float(
                    torch.linalg.vector_norm(mismatch)
                    / torch.linalg.vector_norm(continuous_clean).clamp_min(1e-30)
                ),
                "voxel_vs_continuous_clean_absolute_l2": float(
                    torch.linalg.vector_norm(mismatch)
                ),
                "is_sensor_noise": False,
                "available_to_selector": False,
            }
        )
    return base_scores, mismatch_vectors, rows


def _evaluate_candidates(
    *,
    trained: list[dict[str, Any]],
    records: list[m2.PreparedRecord],
    selectors: Mapping[tuple[str, str], SelectorPayload],
    case_to_session: Mapping[str, str],
    matrices: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    matched_lookup: Mapping[tuple[str, str], Mapping[str, Any]],
    base_scores: Mapping[str, Mapping[str, Any]],
    mismatch_vectors: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    harm = float(config["decision_gates"]["field_harm_threshold_fraction"])
    dense = config["dense_evaluator_ceiling"]
    registered_kind = str(
        config["registered_classical_reference"]["baseline_kind"]
    )

    for run in trained:
        model = run["model"]
        model_device = next(model.parameters()).device
        for record in records:
            if record.split == "train":
                continue
            kwargs = m2._to_device(record.batch.model_kwargs(), model_device)
            feature_f = int(record.preparation_calls["forward_calls"])
            feature_a = int(record.preparation_calls["adjoint_calls"])
            m2._synchronize(model_device)
            started = time.perf_counter()
            with torch.no_grad():
                prediction, gate = model(**kwargs, return_gate=True)
            m2._synchronize(model_device)
            inference_seconds = time.perf_counter() - started
            operator = record.case.inference.operator
            initial = prediction[0, 0].detach().cpu().to(operator.support)
            raw_score = m2._score_prediction(
                record=record,
                method=str(run["method"]),
                model_seed=int(run["model_seed"]),
                prediction=initial,
                gate=float(gate[0, 0, 0, 0, 0].detach().cpu()),
                correction_rms=float(
                    torch.sqrt(
                        torch.mean(
                            (initial - record.batch.base_field[0, 0]).square()
                        )
                    )
                ),
                optimization_forward_calls=feature_f,
                optimization_adjoint_calls=feature_a,
                grouped_adjoint_calls=1,
                neural_inference_seconds=inference_seconds,
            )
            references.append({"reference_kind": "raw_learned", **raw_score})
            case_id = record.case.inference.case_id
            session_id = case_to_session[case_id]
            matrix, support = matrices[record.case.inference.geometry.digest]
            observation = record.case.inference.observations_uv[0].detach().cpu().to(
                torch.float64
            )
            registered = matched_lookup[(case_id, registered_kind)]
            cgls = matched_lookup[(case_id, "cgls_matched")]
            huber = matched_lookup[(case_id, "huber_pdhg_matched")]
            envelope_field = min(
                float(cgls["field_relative_l2"]),
                float(huber["field_relative_l2"]),
            )
            envelope_h1 = min(
                float(cgls["h1_seminorm_relative_error"]),
                float(huber["h1_seminorm_relative_error"]),
            )
            base = base_scores[case_id]
            for candidate in config["candidates"]:
                candidate_id = str(candidate["id"])
                selector = selectors[(session_id, candidate_id)]
                target = observation - selector.mean_uv
                mismatch_scores = score_selector_residual(
                    mismatch_vectors[case_id], selector
                )
                mismatch_policy = str(candidate["model_mismatch_policy"])
                if mismatch_policy == "sensor_only":
                    evaluator_mismatch_scores = None
                elif mismatch_policy == "exact_target_oracle_triangle_bound":
                    mismatch_fraction = float(
                        candidate["model_mismatch_norm_fraction"]
                    )
                    if not 0.0 < mismatch_fraction <= 1.0:
                        raise ValueError(
                            "oracle model mismatch norm fraction must lie in (0, 1]"
                        )
                    evaluator_mismatch_scores = replace(
                        mismatch_scores,
                        global_score=(mismatch_fraction**2)
                        * mismatch_scores.global_score,
                        per_camera_scores=(mismatch_fraction**2)
                        * mismatch_scores.per_camera_scores,
                    )
                else:
                    raise ValueError(
                        f"unsupported model mismatch policy: {mismatch_policy}"
                    )
                result = _dense_multigate_ceiling(
                    initial_field=initial,
                    target_observation_uv=target,
                    dense_active_matrix=matrix,
                    support_mask=support,
                    selector=selector,
                    gate_policy=str(candidate["gate_policy"]),
                    log10_alpha_bounds=tuple(
                        float(value) for value in dense["log10_alpha_bounds"]
                    ),
                    alpha_grid_count=int(dense["alpha_grid_count"]),
                    evaluator_mismatch_scores=evaluator_mismatch_scores,
                )
                score = m2._score_prediction(
                    record=record,
                    method=str(run["method"]),
                    model_seed=int(run["model_seed"]),
                    prediction=result.field,
                    gate=float(gate[0, 0, 0, 0, 0].detach().cpu()),
                    correction_rms=float(
                        torch.sqrt(torch.mean((result.field - initial).square()))
                    ),
                    optimization_forward_calls=feature_f,
                    optimization_adjoint_calls=feature_a,
                    grouped_adjoint_calls=1,
                    neural_inference_seconds=inference_seconds,
                )
                row = add_dual_reference_metrics(
                    {
                        "candidate_id": candidate_id,
                        "method": str(run["method"]),
                        "model_seed": int(run["model_seed"]),
                        "case_id": case_id,
                        "session_id": session_id,
                        "split": record.split,
                        "family": record.family,
                        "base_seed": record.base_seed,
                        "candidate_field_relative_l2": float(
                            score["field_relative_l2"]
                        ),
                        "candidate_h1_relative_error": float(
                            score["h1_seminorm_relative_error"]
                        ),
                        "raw_field_relative_l2": float(raw_score["field_relative_l2"]),
                        "raw_h1_relative_error": float(
                            raw_score["h1_seminorm_relative_error"]
                        ),
                        "registered_classical_field_relative_l2": float(
                            registered["field_relative_l2"]
                        ),
                        "registered_classical_h1_relative_error": float(
                            registered["h1_seminorm_relative_error"]
                        ),
                        "clean_reprojection_ratio_to_base": float(
                            score["clean_reprojection_relative_l2"]
                        )
                        / max(float(base["clean_reprojection_relative_l2"]), 1e-30),
                        "selector_valid": result.selector_valid,
                        "residual_closure_relative_error": result.residual_closure_relative_error,
                    },
                    harm_threshold_fraction=harm,
                )
                row.update(
                    {
                        "gate_policy": candidate["gate_policy"],
                        "selector_family": candidate["selector_family"],
                        "model_mismatch_policy": mismatch_policy,
                        "model_mismatch_norm_fraction": candidate[
                            "model_mismatch_norm_fraction"
                        ],
                        "selector_digest": selector.digest,
                        "uses_truth": bool(candidate["uses_truth"]),
                        "uses_exact_nuisance": bool(candidate["uses_exact_nuisance"]),
                        "dense_evaluator_ceiling_only": True,
                        "field_gain_to_best_classical_envelope_diagnostic": (
                            envelope_field - float(score["field_relative_l2"])
                        )
                        / envelope_field,
                        "h1_gain_to_best_classical_envelope_diagnostic": (
                            envelope_h1 - float(score["h1_seminorm_relative_error"])
                        )
                        / envelope_h1,
                        "measured_reprojection_relative_l2": score[
                            "measured_reprojection_relative_l2"
                        ],
                        "clean_reprojection_relative_l2": score[
                            "clean_reprojection_relative_l2"
                        ],
                        "alpha": result.alpha,
                        "raw_no_correction": math.isinf(result.alpha),
                        "fallback_reason": result.fallback_reason,
                        "correction_norm": result.correction_norm,
                        "evaluated_alpha_count": result.evaluated_alpha_count,
                        "raw_global_score": result.raw_global_score,
                        "selected_global_score": result.selected_global_score,
                        "raw_camera_scores_json": _canonical_json(
                            result.raw_camera_scores
                        ),
                        "selected_camera_scores_json": _canonical_json(
                            result.selected_camera_scores
                        ),
                        "gate_checks_json": _canonical_json(result.gate_checks),
                        "model_mismatch_global_whitened_score": mismatch_scores.global_score,
                        "model_mismatch_camera_scores_json": _canonical_json(
                            [float(value) for value in mismatch_scores.per_camera_scores]
                        ),
                        "model_mismatch_available_to_selector": False,
                        "dense_matrix_rows": int(matrix.shape[0]),
                        "dense_matrix_columns": int(matrix.shape[1]),
                        "dense_setup_in_budget": False,
                        "candidate_budget_matched": False,
                        "dense_linear_solve_count": result.evaluated_alpha_count,
                        "candidate_cost_contract": (
                            f"{feature_f}F/{feature_a}AT learned preparation plus "
                            "unmatched evaluator-only dense assembly and alpha-grid solves"
                        ),
                        "learned_feature_forward_calls": feature_f,
                        "learned_feature_adjoint_calls": feature_a,
                    }
                )
                rows.append(row)
    return rows, references


def _plot(
    *,
    calibration_rows: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    output: Path,
) -> None:
    candidates = sorted({str(row["candidate_id"]) for row in calibration_rows})
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), constrained_layout=True)
    coverage = [
        _mean(
            row["same_session_repeat_event_coverage"]
            for row in calibration_rows
            if row["candidate_id"] == candidate and row["split"] != "train"
        )
        for candidate in candidates
    ]
    axes[0].barh(range(len(candidates)), coverage, color="#287271")
    axes[0].axvline(0.95, color="#c33c54", linestyle="--", linewidth=1.5)
    axes[0].set_yticks(range(len(candidates)), [value.replace("_", " ") for value in candidates])
    axes[0].set_xlim(0.7, 1.01)
    axes[0].set_title("Same-session flow-off repeat event")
    axes[0].set_xlabel("repeat coverage (not session/rig or field coverage)")

    selected_candidate = "structured_joint_two_sided_95"
    aggregate_lookup = {
        (row["method"], row["split"]): row
        for row in aggregates
        if row["candidate_id"] == selected_candidate
    }
    labels: list[str] = []
    raw_values: list[float] = []
    classical_values: list[float] = []
    for method in sorted({row["method"] for row in aggregates}):
        for split in ("development", "ood"):
            row = aggregate_lookup[(method, split)]
            labels.append(f"{method}\n{split}")
            raw_values.append(float(row["field_gain_to_raw_mean"]))
            classical_values.append(
                float(row["field_gain_to_registered_classical_mean"])
            )
    x = np.arange(len(labels))
    width = 0.36
    axes[1].bar(x - width / 2, raw_values, width, label="vs raw", color="#f4a261")
    axes[1].bar(
        x + width / 2,
        classical_values,
        width,
        label="vs registered Huber-24",
        color="#2a9d8f",
    )
    axes[1].axhline(0.0, color="#222222", linewidth=1.0)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("mean field gain")
    axes[1].set_title("Dual-reference mean gains")
    axes[1].legend(frameon=False)

    for decision in decisions:
        dev = decision["development"]
        axes[2].scatter(
            float(dev["worst_field_gain_to_raw"]),
            float(dev["worst_field_gain_to_registered_classical"]),
            s=70,
            alpha=0.8,
            label=f"{decision['candidate_id']} / {decision['method']}",
        )
    axes[2].axvline(-0.05, color="#c33c54", linestyle="--", linewidth=1.2)
    axes[2].axhline(-0.05, color="#c33c54", linestyle="--", linewidth=1.2)
    axes[2].set_xlabel("development worst gain vs raw")
    axes[2].set_ylabel("development worst gain vs registered Huber-24")
    axes[2].set_title("Tail gate: upper-right is safer")
    axes[2].legend(fontsize=6, frameon=False, loc="best")
    fig.suptitle(
        "JACRU N1.2 session-conformal dual-reference development screen",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(output / "diagnostic.png", dpi=200)
    fig.savefig(output / "diagnostic.pdf")
    plt.close(fig)


def _write_checksums(output: Path) -> None:
    files = sorted(
        path for path in output.iterdir() if path.is_file() and path.name != "checksums.sha256"
    )
    (output / "checksums.sha256").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )


def main() -> int:
    args = _parse_args()
    config_path = args.config.resolve()
    config = _read_json(config_path)
    _validate_run_contract(args, config)
    git_commit_at_start = _git_commit()
    source_hashes_at_start = _source_manifest(config_path, config)
    source_n1_1 = _read_json(
        ROOT / config["source_n1_1_results"] / "summary.json"
    )
    if source_n1_1.get("status") != "N1_1_FLOWOFF_COVARIANCE_PROXIMAL_NO_GO":
        raise RuntimeError("N1.2 development requires the frozen N1.1 NO-GO packet")
    source_config = _read_json(ROOT / config["source_t0_config"])
    if args.seed_limit is not None:
        source_config = copy.deepcopy(source_config)
        for split in source_config["splits"].values():
            split["base_seeds"] = split["base_seeds"][: args.seed_limit]
        source_config["training"]["model_seeds"] = source_config["training"][
            "model_seeds"
        ][: args.seed_limit]
    methods = [str(value) for value in config["methods"]]
    if not set(methods).issubset(set(source_config["methods"])):
        raise ValueError("N1.2 methods must come from the frozen T0 model family")
    output = args.output_dir.resolve()
    if output.exists():
        if not args.replace_output or args.run_mode != "scratch":
            raise FileExistsError(f"output already exists: {output}")
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    started = time.perf_counter()

    try:
        records, packets, case_to_session, session_rows = _prepare_session_records(
            source_config, config
        )
        selectors, calibration_rows, calibration_sanity_passed = _calibrate_candidates(
            packets=packets, config=config
        )
        device = m2._choose_device(
            args.device or source_config["training"]["device"]
        )
        trained: list[dict[str, Any]] = []
        for method in methods:
            for seed in source_config["training"]["model_seeds"]:
                trained.append(
                    m2._train_one(
                        method=method,
                        seed=int(seed),
                        config=source_config,
                        records=records,
                        device=device,
                        epoch_override=args.epochs,
                    )
                )
        norm_cache = _norm_cache(records, source_config)
        matched_rows, matched_lookup = _matched_baselines(
            records, source_config, norm_cache, config
        )
        matrices, dense_rows = _prepare_dense_evaluator(records, config)
        base_scores, mismatch_vectors, mismatch_rows = _base_and_mismatch_rows(
            records=records,
            matrices=matrices,
            case_to_session=case_to_session,
        )
        metric_rows, reference_rows = _evaluate_candidates(
            trained=trained,
            records=records,
            selectors=selectors,
            case_to_session=case_to_session,
            matrices=matrices,
            matched_lookup=matched_lookup,
            base_scores=base_scores,
            mismatch_vectors=mismatch_vectors,
            config=config,
        )
        aggregates = aggregate_dual_reference_rows(metric_rows)
        metadata = {
            str(candidate["id"]): candidate for candidate in config["candidates"]
        }
        decisions = dual_reference_decisions(
            aggregates,
            candidate_metadata=metadata,
            candidate_calibration_sanity_passed=calibration_sanity_passed,
            gates=config["decision_gates"],
        )
        passed = [decision for decision in decisions if decision["passed"]]
        observable_passed = [
            decision
            for decision in passed
            if not decision["uses_truth"] and not decision["uses_exact_nuisance"]
        ]
        status = (
            "N1_2_SESSION_CONFORMAL_DENSE_MECHANISM_SIGNAL_ONLY"
            if observable_passed
            else "N1_2_SESSION_CONFORMAL_DUAL_REFERENCE_NO_GO"
        )
        summary = {
            "schema_version": config["report_schema_version"],
            "status": status,
            "run_mode": args.run_mode,
            "config_status": config["status"],
            "evidence_level": config["evidence_level"],
            "development_only": True,
            "exact_cli": [sys.executable, *sys.argv],
            "effective_model_seeds": [
                int(value) for value in source_config["training"]["model_seeds"]
            ],
            "effective_epochs": int(
                source_config["training"]["epochs"]
                if args.epochs is None
                else args.epochs
            ),
            "device": str(device),
            "runtime_seconds": time.perf_counter() - started,
            "git_commit_at_start": git_commit_at_start,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "source_hashes_at_start": source_hashes_at_start,
            "session_count": len(session_rows),
            "evaluation_case_count": sum(record.split != "train" for record in records),
            "candidate_count": len(config["candidates"]),
            "calibration_row_count": len(calibration_rows),
            "metric_row_count": len(metric_rows),
            "reference_row_count": len(reference_rows),
            "matched_baseline_row_count": len(matched_rows),
            "model_mismatch_row_count": len(mismatch_rows),
            "dense_setup_row_count": len(dense_rows),
            "aggregate_row_count": len(aggregates),
            "decision_count": len(decisions),
            "candidate_calibration_sanity_passed": calibration_sanity_passed,
            "decisions": decisions,
            "dense_ceiling_pass_count": len(observable_passed),
            "oracle_dense_ceiling_pass_count": len(passed) - len(observable_passed),
            "authorization": {
                "claim_algorithm_superiority": False,
                "claim_real_bost_generalization": False,
                "claim_runtime_or_efficiency": False,
                "open_fresh_or_final": False,
                "freeze_formal_n1_2_without_red_team": False,
                "continue_protocol_development": True,
            },
            "claim_boundary": config["claim_boundary"],
        }
        _write_csv(temporary / "session_manifest_rows.csv", session_rows)
        _write_csv(temporary / "calibration_rows.csv", calibration_rows)
        _write_csv(temporary / "metric_rows.csv", metric_rows)
        _write_csv(temporary / "aggregate_rows.csv", aggregates)
        _write_csv(temporary / "reference_rows.csv", reference_rows)
        _write_csv(temporary / "matched_baseline_rows.csv", matched_rows)
        _write_csv(temporary / "model_mismatch_rows.csv", mismatch_rows)
        _write_csv(temporary / "dense_setup_rows.csv", dense_rows)
        (temporary / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "README.md").write_text(
            "# JACRU N1.2 session-conformal development evidence\n\n"
            f"- Status: `{status}`\n"
            "- This is opened-synthetic protocol development, not a formal run.\n"
            "- One flow-off packet is shared by every field in a session.\n"
            "- Coverage is same-synthetic-session repeat coverage only; it is not "
            "held-out-session, rig, or flow-on reconstruction coverage.\n"
            "- Dense measurement-space solves are evaluator ceilings outside the budget.\n"
            "- No method, deployment, fresh, real-data, or publication claim is authorized.\n",
            encoding="utf-8",
        )
        _plot(
            calibration_rows=calibration_rows,
            aggregates=aggregates,
            decisions=decisions,
            output=temporary,
        )
        _write_checksums(temporary)
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "status": status,
                "metric_rows": len(metric_rows),
                "dense_ceiling_passes": len(observable_passed),
                "oracle_dense_ceiling_passes": len(passed) - len(observable_passed),
                "output": str(output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
