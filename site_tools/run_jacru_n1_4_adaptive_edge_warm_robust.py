#!/usr/bin/env python3
"""Screen a matched-budget adaptive-edge warm robust reconstruction.

The candidate spends 12 physical pairs on a structured-whitened CGLS warm
start and 12 pairs on Huber-data PDHG. A fixed spatial edge-weight map is
computed from the warm-start gradient only. Truth, clean projections, family
labels, and OOD data are absent from reconstruction. Opened-development truth
and clean projections are used only by the diagnostic candidate gates; they are
not an inference-time selector or confirmation set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping
import uuid

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo_t16_operator.interface_baselines import (  # noqa: E402
    cgls_baseline,
    robust_data_pdhg_baseline,
)
from demo_t16_operator.psu_b0_primal_dual import (  # noqa: E402
    regularization_gradient,
)
from site_tools import run_jacru_m2_1_data_consistency_diagnostic as m21  # noqa: E402
from site_tools import run_jacru_m2_learned_residual_gate as m2  # noqa: E402
from site_tools import (  # noqa: E402
    run_jacru_n1_2_session_conformal_dual_reference as n12,
)
from site_tools import run_jacru_n1_3_robust_data_whitening as n13  # noqa: E402


DEFAULT_CONFIG = (
    ROOT
    / "demo_t16_operator/configs/"
    "jacru_n1_4_adaptive_edge_warm_robust_development_v1.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "demo_t16_operator/results/"
    "jacru_n1_4_adaptive_edge_warm_robust_development_scratch"
)
DIAGNOSTIC_TITLE = "N1.4 adaptive-edge warm robust development screen"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
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
        ROOT / str(config["source_n1_2_config"]),
        ROOT / str(config["source_n1_3_results"]) / "summary.json",
        ROOT / "demo_t16_operator/interface_baselines.py",
        ROOT / "demo_t16_operator/psu_b0_primal_dual.py",
        ROOT / "demo_t16_operator/jacru_n1_2_session_conformal.py",
        ROOT / "demo_t16_operator/jacru_n1_flowoff_covariance.py",
        ROOT / "site_tools/run_jacru_n1_2_session_conformal_dual_reference.py",
        ROOT / "site_tools/run_jacru_n1_3_robust_data_whitening.py",
        Path(__file__).resolve(),
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"source manifest is incomplete: {missing}")
    return {str(path.relative_to(ROOT)): _sha256(path) for path in paths}


def _validate_config(config: Mapping[str, Any], seed_limit: int | None) -> None:
    if config.get("status") != "DEVELOPMENT_ONLY_NOT_PREREGISTERED_NOT_FORMAL":
        raise RuntimeError("N1.4 runner accepts only the explicit development config")
    if config.get("evaluated_split") != "development":
        raise RuntimeError("this runner is development-only")
    if config.get("may_construct_or_evaluate_ood") is not False:
        raise RuntimeError("OOD construction must remain disabled")
    if seed_limit is not None and seed_limit < 1:
        raise ValueError("seed-limit must be positive")
    budget = config["solve_budget"]
    warm = int(budget["warm_start_cgls_iterations"])
    refinement = int(budget["robust_refinement_iterations"])
    if warm != int(budget["warm_start_forward_calls"]):
        raise ValueError("warm-start iteration and forward-call budgets differ")
    if warm != int(budget["warm_start_adjoint_calls"]):
        raise ValueError("warm-start iteration and adjoint-call budgets differ")
    if refinement != int(budget["robust_refinement_forward_calls"]):
        raise ValueError("refinement iteration and forward-call budgets differ")
    if refinement != int(budget["robust_refinement_adjoint_calls"]):
        raise ValueError("refinement iteration and adjoint-call budgets differ")
    if warm + refinement != int(budget["total_forward_calls"]):
        raise ValueError("warm and refinement forward budgets do not sum to total")
    if warm + refinement != int(budget["total_adjoint_calls"]):
        raise ValueError("warm and refinement adjoint budgets do not sum to total")
    references = config["registered_references"]
    if int(references["total_forward_calls"]) != int(budget["total_forward_calls"]):
        raise ValueError("registered reference and candidate forward budgets differ")
    if int(references["total_adjoint_calls"]) != int(budget["total_adjoint_calls"]):
        raise ValueError("registered reference and candidate adjoint budgets differ")
    grid = config["candidate_grid"]
    zero_start_weights = grid.get("zero_start_control_regularization_weights", [0.1])
    if not isinstance(zero_start_weights, list) or not zero_start_weights:
        raise ValueError("zero-start control weights must be a non-empty list")
    if any(float(weight) <= 0.0 for weight in zero_start_weights):
        raise ValueError("zero-start control weights must be positive")
    if grid.get("require_matched_zero_start_controls", False):
        screened = {float(weight) for weight in grid["standardized_edge_regularization_weights"]}
        controlled = {float(weight) for weight in zero_start_weights}
        if controlled != screened:
            raise ValueError("zero-start controls must match every screened edge weight")


def _assert_full_development_contract(
    records: list[m2.PreparedRecord], source_config: Mapping[str, Any]
) -> None:
    split = source_config["splits"]["development"]
    expected = {
        (int(seed), str(family))
        for seed in split["base_seeds"]
        for family in split["families"]
    }
    observed = {(int(record.base_seed), str(record.family)) for record in records}
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise RuntimeError(
            f"full development contract drifted: missing={missing}, extra={extra}"
        )


def _candidate_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    grid = config["candidate_grid"]
    output: list[dict[str, Any]] = []
    for weight in grid["standardized_edge_regularization_weights"]:
        for quantile in grid["edge_indicator_quantiles"]:
            for minimum in grid["minimum_edge_weights"]:
                output.append(
                    {
                        "candidate_id": (
                            "adaptive_edge_warm__"
                            f"l{n13._token(weight)}__q{n13._token(quantile)}__"
                            f"wmin{n13._token(minimum)}"
                        ),
                        "solver_kind": "adaptive_edge_warm_huber_measurement_pdhg",
                        "mean_policy": "estimated",
                        "whitening_policy": str(grid["whitening_policy"]),
                        "standardized_edge_regularization_weight": float(weight),
                        "edge_indicator_quantile": float(quantile),
                        "minimum_edge_weight": float(minimum),
                    }
                )
    if grid["include_uniform_warm_controls"]:
        for weight in grid["standardized_edge_regularization_weights"]:
            output.append(
                {
                    "candidate_id": f"uniform_edge_warm__l{n13._token(weight)}",
                    "solver_kind": "uniform_edge_warm_huber_measurement_pdhg_control",
                    "mean_policy": "estimated",
                    "whitening_policy": str(grid["whitening_policy"]),
                    "standardized_edge_regularization_weight": float(weight),
                    "edge_indicator_quantile": None,
                    "minimum_edge_weight": 1.0,
                }
            )
    if grid["include_zero_start_24_control"]:
        for weight in grid.get("zero_start_control_regularization_weights", [0.1]):
            output.append(
                {
                    "candidate_id": (
                        "zero_start_24__structured__d2__"
                        f"l{n13._token(weight)}_control"
                    ),
                    "solver_kind": "zero_start_24_huber_measurement_pdhg_control",
                    "mean_policy": "estimated",
                    "whitening_policy": str(grid["whitening_policy"]),
                    "standardized_edge_regularization_weight": float(weight),
                    "edge_indicator_quantile": None,
                    "minimum_edge_weight": 1.0,
                }
            )
    identifiers = [str(row["candidate_id"]) for row in output]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("candidate IDs must be unique")
    return output


def _adaptive_edge_weight_map(
    initial_field: torch.Tensor,
    *,
    support: torch.Tensor,
    spacing_xyz: tuple[float, float, float],
    quantile: float,
    minimum_weight: float,
    power: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    field = torch.as_tensor(initial_field).detach().cpu().to(torch.float64)
    mask = torch.as_tensor(support).detach().cpu().to(torch.bool)
    q = float(quantile)
    floor = float(minimum_weight)
    exponent = float(power)
    if field.shape != mask.shape:
        raise ValueError("initial field and support must share one 3D shape")
    if not 0.0 < q < 1.0 or not 0.0 < floor <= 1.0 or exponent <= 0.0:
        raise ValueError("adaptive edge parameters are outside their valid ranges")
    gradient = regularization_gradient(field[None], spacing_xyz=spacing_xyz)
    magnitude = torch.linalg.vector_norm(gradient, dim=1)[0]
    active = magnitude.masked_select(mask)
    threshold = float(torch.quantile(active, q).clamp_min(1e-12))
    ratio = magnitude / threshold
    weights = floor + (1.0 - floor) / (1.0 + ratio.pow(exponent))
    weights = torch.where(mask, weights, torch.ones_like(weights))
    return weights, {
        "edge_indicator_threshold": threshold,
        "edge_weight_minimum": float(torch.min(weights.masked_select(mask))),
        "edge_weight_mean": float(torch.mean(weights.masked_select(mask))),
        "edge_weight_maximum": float(torch.max(weights.masked_select(mask))),
    }


def _gain(reference: float, candidate: float) -> float:
    return (float(reference) - float(candidate)) / max(float(reference), 1e-30)


def _evaluate_candidates(
    *,
    records: list[m2.PreparedRecord],
    case_to_session: Mapping[str, str],
    stress_by_case: Mapping[str, str],
    selectors: Mapping[tuple[str, str], Any],
    matrices: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    references: Mapping[tuple[str, str], Mapping[str, Any]],
    candidates: list[dict[str, Any]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    warm_rows: list[dict[str, Any]] = []
    budget = config["solve_budget"]
    grid = config["candidate_grid"]
    context_cache: dict[
        tuple[str, str], tuple[torch.Tensor, float]
    ] = {}
    for record in records:
        case = record.case
        case_id = case.inference.case_id
        session_id = str(case_to_session[case_id])
        stress = str(stress_by_case[case_id])
        operator = case.inference.operator
        observation = case.inference.observations_uv[0].detach().cpu().to(torch.float64)
        support = operator.support.detach().cpu().to(torch.float64)
        selector = selectors[(session_id, str(grid["whitening_policy"]))]
        factor = n13._whitening_factor(selector)
        target = n13._whiten_vector(observation - selector.mean_uv, factor)
        physical_forward, physical_adjoint = m2._operator_maps(operator)
        forward, adjoint = n13._candidate_operator(
            forward=physical_forward,
            adjoint=physical_adjoint,
            factor=factor,
        )
        matrix, _ = matrices[case.inference.geometry.digest]
        context_key = (session_id, case.inference.geometry.digest)
        if context_key not in context_cache:
            transformed = torch.linalg.solve_triangular(factor, matrix, upper=False)
            largest = float(torch.linalg.svdvals(transformed).max())
            context_cache[context_key] = (
                factor,
                float(budget["norm_safety_factor"]) * largest**2,
            )
        _, norm_bound = context_cache[context_key]

        operator.reset_call_counts()
        warm_started = time.perf_counter()
        warm = cgls_baseline(
            target,
            forward=forward,
            adjoint=adjoint,
            support=support,
            spacing_xyz=operator.spacing_xyz,
            iterations=int(budget["warm_start_cgls_iterations"]),
        )
        warm_seconds = time.perf_counter() - warm_started
        if operator.call_report() != {
            "forward_calls": int(budget["warm_start_forward_calls"]),
            "adjoint_calls": int(budget["warm_start_adjoint_calls"]),
        }:
            raise RuntimeError("warm-start physical-call contract drifted")
        warm_rows.append(
            {
                "case_id": case_id,
                "session_id": session_id,
                "stress": stress,
                "warm_start_forward_calls": warm.forward_calls,
                "warm_start_adjoint_calls": warm.adjoint_calls,
                "warm_start_seconds": warm_seconds,
                "warm_start_reused_across_grid_for_development_runtime": True,
                "per_deployed_candidate_warm_start_cost": "12F/12AT",
            }
        )
        registered_huber = references[(case_id, "huber_pdhg_matched")]
        registered_cgls = references[(case_id, "cgls_matched")]
        for candidate in candidates:
            zero_start = candidate["solver_kind"].startswith("zero_start_24")
            if candidate["solver_kind"].startswith("adaptive_edge"):
                edge_weights, edge_diagnostics = _adaptive_edge_weight_map(
                    warm.field,
                    support=support,
                    spacing_xyz=operator.spacing_xyz,
                    quantile=float(candidate["edge_indicator_quantile"]),
                    minimum_weight=float(candidate["minimum_edge_weight"]),
                    power=float(grid["edge_weight_power"]),
                )
            elif zero_start:
                edge_weights = None
                edge_diagnostics = {
                    "edge_indicator_threshold": None,
                    "edge_weight_minimum": 1.0,
                    "edge_weight_mean": 1.0,
                    "edge_weight_maximum": 1.0,
                }
            else:
                edge_weights = torch.ones_like(support)
                edge_diagnostics = {
                    "edge_indicator_threshold": None,
                    "edge_weight_minimum": 1.0,
                    "edge_weight_mean": 1.0,
                    "edge_weight_maximum": 1.0,
                }
            refinement_iterations = (
                int(budget["total_forward_calls"])
                if zero_start
                else int(budget["robust_refinement_iterations"])
            )
            initial = None if zero_start else warm.field
            operator.reset_call_counts()
            refinement_started = time.perf_counter()
            result = robust_data_pdhg_baseline(
                target,
                forward=forward,
                adjoint=adjoint,
                support=support,
                spacing_xyz=operator.spacing_xyz,
                iterations=refinement_iterations,
                regularization_weight=float(
                    candidate["standardized_edge_regularization_weight"]
                ),
                data_norm_squared_bound=norm_bound,
                data_huber_delta=float(grid["data_huber_delta_sigma_multiplier"]),
                edge_penalty=str(grid["edge_penalty"]),
                edge_huber_delta=float(grid["edge_huber_delta"]),
                ridge_weight=float(grid["ridge_weight"]),
                initial_field=initial,
                edge_weight_map=edge_weights,
                step_safety=float(budget["step_safety"]),
            )
            refinement_seconds = time.perf_counter() - refinement_started
            expected_refinement_calls = {
                "forward_calls": refinement_iterations,
                "adjoint_calls": refinement_iterations,
            }
            if operator.call_report() != expected_refinement_calls:
                raise RuntimeError("refinement physical-call contract drifted")
            total_forward = result.forward_calls + (0 if zero_start else warm.forward_calls)
            total_adjoint = result.adjoint_calls + (0 if zero_start else warm.adjoint_calls)
            if total_forward != 24 or total_adjoint != 24:
                raise RuntimeError("candidate total physical-call contract drifted")
            score = m2._score_prediction(
                record=record,
                method=str(candidate["candidate_id"]),
                model_seed=-1,
                prediction=result.field,
                gate=None,
                correction_rms=None,
                optimization_forward_calls=total_forward,
                optimization_adjoint_calls=total_adjoint,
                grouped_adjoint_calls=0,
                neural_inference_seconds=0.0,
            )
            rows.append(
                {
                    **score,
                    **candidate,
                    **edge_diagnostics,
                    "session_id": session_id,
                    "stress": stress,
                    "selector_digest": selector.digest,
                    "reconstruction_uses_truth_or_clean_projection": False,
                    "development_gate_uses_truth_and_clean_projection": True,
                    "warm_start_used": not zero_start,
                    "warm_start_forward_calls": 0 if zero_start else warm.forward_calls,
                    "warm_start_adjoint_calls": 0 if zero_start else warm.adjoint_calls,
                    "refinement_forward_calls": result.forward_calls,
                    "refinement_adjoint_calls": result.adjoint_calls,
                    "warm_start_seconds_amortized_screening_only": 0.0 if zero_start else warm_seconds,
                    "refinement_seconds": refinement_seconds,
                    "candidate_budget_matched": True,
                    "operator_norm_squared_bound": norm_bound,
                    "dense_norm_setup_in_budget": False,
                    "field_gain_to_registered_huber": _gain(
                        registered_huber["field_relative_l2"], score["field_relative_l2"]
                    ),
                    "h1_gain_to_registered_huber": _gain(
                        registered_huber["h1_seminorm_relative_error"],
                        score["h1_seminorm_relative_error"],
                    ),
                    "clean_reprojection_ratio_to_cgls": float(
                        score["clean_reprojection_relative_l2"]
                    )
                    / max(
                        float(registered_cgls["clean_reprojection_relative_l2"]),
                        1e-30,
                    ),
                    "measured_reprojection_ratio_to_cgls": float(
                        score["measured_reprojection_relative_l2"]
                    )
                    / max(
                        float(registered_cgls["measured_reprojection_relative_l2"]),
                        1e-30,
                    ),
                }
            )
    return rows, warm_rows


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
    _validate_config(config, args.seed_limit)
    source_n1_3 = _read_json(ROOT / config["source_n1_3_results"] / "summary.json")
    if source_n1_3.get("status") != "N1_3_ROBUST_DATA_WHITENING_DEVELOPMENT_NO_GO":
        raise RuntimeError("N1.4 requires the frozen N1.3 development NO-GO packet")
    git_commit_at_start = _git_commit()
    source_hashes_at_start = _source_manifest(config_path, config)
    source_config = n13._development_source_config(
        _read_json(ROOT / config["source_t0_config"]), args.seed_limit
    )
    n12_config = _read_json(ROOT / config["source_n1_2_config"])
    output = args.output_dir.resolve()
    if output.exists():
        if not args.replace_output:
            raise FileExistsError(f"output already exists: {output}")
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    started = time.perf_counter()

    try:
        records, packets, case_to_session, session_rows = n12._prepare_session_records(
            source_config, n12_config
        )
        if any(record.split != "development" for record in records):
            raise RuntimeError("non-development record escaped the construction firewall")
        if args.seed_limit is None:
            _assert_full_development_contract(records, source_config)
        selectors, calibration_rows = n13._selector_maps(packets, n12_config)
        expanded, expanded_sessions, stress_by_case, stress_rows = n13._expanded_stress_records(
            records,
            case_to_session=case_to_session,
            selectors=selectors,
            stress_config=config["flowon_sparse_outlier_stress"],
        )
        matrices, dense_rows = n12._prepare_dense_evaluator(expanded, n12_config)
        norm_cache = n12._norm_cache(expanded, source_config)
        reference_rows = m21._matched_baseline_rows(
            records=expanded,
            source_config=source_config,
            diagnostic_config={"step_safety_factor": 0.9},
            norm_cache=norm_cache,
            steps=[int(config["registered_references"]["projection_step"])],
        )
        reference_lookup = {
            (str(row["case_id"]), str(row["baseline_kind"])): row
            for row in reference_rows
        }
        candidates = _candidate_specs(config)
        metric_rows, warm_rows = _evaluate_candidates(
            records=expanded,
            case_to_session=expanded_sessions,
            stress_by_case=stress_by_case,
            selectors=selectors,
            matrices=matrices,
            references=reference_lookup,
            candidates=candidates,
            config=config,
        )
        aggregates = n13._aggregate(
            metric_rows,
            float(config["development_gates"]["field_harm_threshold_fraction"]),
        )
        full_screen_complete = args.seed_limit is None
        decisions, selection = n13._decisions(
            metric_rows,
            aggregates,
            gates=config["development_gates"],
            full_screen_complete=full_screen_complete,
        )
        if not full_screen_complete:
            status = "N1_4_ADAPTIVE_EDGE_WARM_ROBUST_PILOT_ONLY"
        elif selection is None:
            status = "N1_4_ADAPTIVE_EDGE_WARM_ROBUST_DEVELOPMENT_NO_GO"
        else:
            status = "N1_4_ADAPTIVE_EDGE_WARM_ROBUST_DEVELOPMENT_SIGNAL_TO_FREEZE"
        summary = {
            "schema_version": config["report_schema_version"],
            "status": status,
            "evidence_level": config["evidence_level"],
            "development_only": True,
            "ood_constructed_or_evaluated": False,
            "full_screen_complete": full_screen_complete,
            "exact_cli": [sys.executable, *sys.argv],
            "runtime_seconds": time.perf_counter() - started,
            "git_commit_at_start": git_commit_at_start,
            "source_hashes_at_start": source_hashes_at_start,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "session_count": len(session_rows),
            "nominal_case_count": len(records),
            "stress_case_count": len(expanded),
            "candidate_count": len(candidates),
            "metric_row_count": len(metric_rows),
            "decision_count": len(decisions),
            "passed_candidate_count": sum(row["passed"] for row in decisions),
            "selection": selection,
            "decisions": decisions,
            "authorization": {
                "claim_algorithm_superiority": False,
                "claim_real_bost_generalization": False,
                "open_ood": selection is not None and full_screen_complete,
                "train_neural_correction": False,
                "continue_interface_mechanism_diagnosis": True,
            },
            "claim_boundary": config["claim_boundary"],
        }
        _write_csv(temporary / "session_rows.csv", session_rows)
        _write_csv(temporary / "calibration_rows.csv", calibration_rows)
        _write_csv(temporary / "stress_manifest_rows.csv", stress_rows)
        _write_csv(temporary / "dense_setup_rows.csv", dense_rows)
        _write_csv(temporary / "reference_rows.csv", reference_rows)
        _write_csv(temporary / "warm_start_rows.csv", warm_rows)
        _write_csv(temporary / "metric_rows.csv", metric_rows)
        _write_csv(temporary / "aggregate_rows.csv", aggregates)
        _write_csv(
            temporary / "decision_rows.csv",
            [
                {
                    "candidate_id": row["candidate_id"],
                    "passed": row["passed"],
                    "known_interface_seed_worst_gain": row[
                        "known_interface_seed_worst_gain"
                    ],
                    "failed_checks_json": _canonical_json(
                        [name for name, passed in row["checks"].items() if not passed]
                    ),
                }
                for row in decisions
            ],
        )
        (temporary / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temporary / "README.md").write_text(
            "# JACRU N1.4 adaptive-edge warm robust development screen\n\n"
            f"- Status: `{status}`\n"
            "- Every deployed candidate is costed as CGLS-12 plus robust PDHG-12, "
            "or a zero-start 24-pair control.\n"
            "- Edge weights read only the warm-start gradient; they never read truth.\n"
            "- Truth and clean projections are absent from reconstruction but are used by "
            "the opened-development diagnostic gates.\n"
            "- OOD was neither constructed nor evaluated by this runner.\n"
            "- Development-grid warm starts are cached for runtime only; their 12F/12AT "
            "cost remains in every candidate ledger.\n"
            "- No algorithm, real-data, generalization, efficiency, or publication claim is authorized.\n",
            encoding="utf-8",
        )
        n13._plot(aggregates, temporary, title=DIAGNOSTIC_TITLE)
        _write_checksums(temporary)
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "status": status,
                "candidate_count": len(candidates),
                "metric_rows": len(metric_rows),
                "passed_candidates": sum(row["passed"] for row in decisions),
                "output": str(output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
