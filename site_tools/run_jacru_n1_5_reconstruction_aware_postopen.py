#!/usr/bin/env python3
"""Post-open reconstruction screen for N1.5 approximation-error corrections.

The run diagnoses whether a forward-mismatch predictor improves the inverse
field under a fixed physical-call budget. It is explicitly post-open and may
only freeze a hypothesis for a later independent confirmation run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
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

from demo_t16_operator.analytic_bost_phantoms import analytic_phantom_grid  # noqa: E402
from demo_t16_operator.interface_baselines import cgls_baseline  # noqa: E402
from demo_t16_operator.jacru_n1_5_approximation_error import (  # noqa: E402
    StandardizedRidge,
)
from demo_t16_operator.jacru_n1_5_high_order_correction import (  # noqa: E402
    HighOrderTeacherMaps,
    warm_start_cgls,
)
from demo_t16_operator.psu_b0_streaming_operator import (  # noqa: E402
    zero_outer_boundary_support,
)
from demo_t16_operator.spatial_reconstruction_metrics import (  # noqa: E402
    synthetic_field_metrics,
)
from site_tools import (  # noqa: E402
    run_jacru_n1_5_approximation_error_headroom as n15a,
)


DEFAULT_CONFIG = (
    ROOT
    / "demo_t16_operator/configs/"
    "jacru_n1_5_reconstruction_aware_postopen_development_v1.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "demo_t16_operator/results/"
    "jacru_n1_5_reconstruction_aware_postopen_scratch"
)
REPORT_SCHEMA = "jacru-n1-5-reconstruction-aware-postopen-report-1.0"


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_checksums(output: Path) -> None:
    files = sorted(
        path for path in output.iterdir() if path.is_file() and path.name != "checksums.sha256"
    )
    (output / "checksums.sha256").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )


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
        ROOT / str(config["source_n1_5_a_config"]),
        ROOT / str(config["source_n1_5_a_results"]) / "summary.json",
        ROOT / str(config["source_n1_5_a_results"]) / "selected_ridge_models.json",
        ROOT / "demo_t16_operator/jacru_n1_5_high_order_correction.py",
        ROOT / "demo_t16_operator/interface_baselines.py",
        ROOT / "demo_t16_operator/jacru_synthetic_fixture.py",
        Path(__file__).resolve(),
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"source manifest is incomplete: {missing}")
    return {str(path.relative_to(ROOT)): _sha256(path) for path in paths}


def _validate_config(config: Mapping[str, Any], n15a_summary: Mapping[str, Any], seed_limit: int | None) -> None:
    if config.get("status") != "POSTOPEN_DEVELOPMENT_MECHANISM_DIAGNOSTIC_NOT_CONFIRMATORY":
        raise RuntimeError("runner requires the explicit post-open diagnostic config")
    if config.get("may_construct_or_evaluate_ood") is not False:
        raise RuntimeError("OOD construction must remain disabled")
    if seed_limit is not None and seed_limit < 1:
        raise ValueError("seed-limit must be positive")
    if n15a_summary.get("status") != "NO_GO_VISIBLE_FORWARD_PREDICTOR":
        raise RuntimeError("N1.5-A source status drifted")
    budget = config["budget"]
    warm = int(budget["warm_cgls_iterations"])
    refine = int(budget["corrected_warm_cgls_iterations"])
    projection = int(budget["visible_low_projection_forward_calls"])
    if warm + projection + refine != int(budget["corrected_total_low_forward_calls"]):
        raise ValueError("corrected forward-call budget drifted")
    if warm + refine != int(budget["corrected_total_low_adjoint_calls"]):
        raise ValueError("corrected adjoint-call budget drifted")
    if int(budget["low_cgls_reference_iterations"]) != int(
        budget["corrected_total_low_forward_calls"]
    ):
        raise ValueError("low CGLS reference must match corrected forward-call budget")
    betas = [float(value) for value in config["correction_candidates"]["damping_to_high_order_betas"]]
    if not betas or any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in betas):
        raise ValueError("high-order interpolation betas must lie in [0,1]")
    if config["claim_boundary"].get("may_claim_confirmed_algorithm_gain") is not False:
        raise ValueError("post-open run cannot claim confirmed gain")


def _load_curvature_model(path: Path) -> StandardizedRidge:
    payload = _read_json(path)["curvature_visible"]
    return StandardizedRidge(
        feature_names=tuple(str(value) for value in payload["feature_names"]),
        feature_mean=torch.as_tensor(payload["feature_mean"], dtype=torch.float64),
        feature_scale=torch.as_tensor(payload["feature_scale"], dtype=torch.float64),
        coefficients=torch.as_tensor(payload["coefficients"], dtype=torch.float64),
        intercept=torch.as_tensor(payload["intercept"], dtype=torch.float64),
        alpha=float(payload["alpha"]),
    )


def _candidate_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates = config["correction_candidates"]
    output = [
        {"candidate_id": "low_cgls25", "kind": "low_reference", "beta": None},
        {
            "candidate_id": "direct_high_order_cgls25",
            "kind": "direct_high_order_reference",
            "beta": None,
        },
    ]
    if candidates["include_component_damping"]:
        output.append(
            {"candidate_id": "component_damping_correction", "kind": "damping", "beta": 0.0}
        )
    if candidates["include_visible_curvature_ridge"]:
        output.append(
            {"candidate_id": "visible_curvature_correction", "kind": "curvature", "beta": 1.0}
        )
    if candidates["include_damping_curvature_half_interpolation"]:
        output.append(
            {
                "candidate_id": "damping_curvature_half",
                "kind": "damping_curvature",
                "beta": 0.5,
            }
        )
    for beta in candidates["damping_to_high_order_betas"]:
        output.append(
            {
                "candidate_id": f"damping_high_order_b{str(beta).replace('.', 'p')}",
                "kind": "damping_high_order",
                "beta": float(beta),
            }
        )
    if candidates["include_exact_mismatch_oracle"]:
        output.append(
            {"candidate_id": "exact_mismatch_oracle", "kind": "exact_oracle", "beta": 1.0}
        )
    return output


def _field_metrics(field: torch.Tensor, record: n15a.CaseRecord) -> dict[str, float]:
    evaluation = record.case.evaluation
    analytic = analytic_phantom_grid(
        evaluation.phantom_spec,
        grid_shape=tuple(int(value) for value in field.shape),
        dtype=torch.float64,
    )
    return synthetic_field_metrics(
        field.detach().cpu().numpy(),
        evaluation.truth_volume[0, 0].cpu().numpy(),
        analytic_truth_gradient_xyz=analytic.gradient_xyz.cpu().numpy(),
        spacing_xyz=record.case.inference.operator.spacing_xyz,
    )


def _relative_l2(value: torch.Tensor, reference: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(value - reference)
        / torch.linalg.vector_norm(reference).clamp_min(1e-30)
    )


def _score(
    *,
    record: n15a.CaseRecord,
    candidate_id: str,
    kind: str,
    beta: float | None,
    field: torch.Tensor,
    modeled_correction: torch.Tensor | None,
    modeled_projection: torch.Tensor,
    raw_low_projection: torch.Tensor,
    low_forward_calls: int,
    low_adjoint_calls: int,
    high_order_forward_calls: int,
    high_order_adjoint_calls: int,
    evaluator_only: bool,
    runtime_seconds: float,
) -> dict[str, Any]:
    metrics = _field_metrics(field, record)
    clean = record.case.evaluation.clean_observations_uv[0]
    correction_norm = 0.0 if modeled_correction is None else float(
        torch.linalg.vector_norm(modeled_correction)
        / torch.linalg.vector_norm(clean).clamp_min(1e-30)
    )
    return {
        "partition": record.partition,
        "base_seed": record.base_seed,
        "geometry_digest": record.case.inference.geometry.digest,
        "family": record.family,
        "case_id": record.case.inference.case_id,
        "candidate_id": candidate_id,
        "kind": kind,
        "beta": beta,
        **metrics,
        "modeled_continuous_reprojection_relative_l2": _relative_l2(modeled_projection, clean),
        "raw_low_reprojection_relative_l2": _relative_l2(raw_low_projection, clean),
        "modeled_correction_relative_norm": correction_norm,
        "low_forward_calls": low_forward_calls,
        "low_adjoint_calls": low_adjoint_calls,
        "high_order_forward_calls": high_order_forward_calls,
        "high_order_adjoint_calls": high_order_adjoint_calls,
        "evaluator_only": evaluator_only,
        "runtime_seconds": runtime_seconds,
    }


def _run_cases(
    records: list[n15a.CaseRecord],
    fixed: Mapping[str, Mapping[str, Any]],
    curvature_model: StandardizedRidge,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    budget = config["budget"]
    warm_iterations = int(budget["warm_cgls_iterations"])
    refine_iterations = int(budget["corrected_warm_cgls_iterations"])
    reference_iterations = int(budget["low_cgls_reference_iterations"])
    direct_iterations = int(budget["direct_high_order_reference_iterations"])
    support = zero_outer_boundary_support(
        records[0].case.inference.operator.grid_shape, dtype=torch.float64
    )
    rows: list[dict[str, Any]] = []
    specs = _candidate_specs(config)
    for record in records:
        if record.partition not in config["evaluated_partitions"]:
            continue
        case = record.case
        operator = case.inference.operator
        observation = case.evaluation.clean_observations_uv[0]
        forward, adjoint = n15a._operator_maps(operator)

        operator.reset_call_counts()
        warm = cgls_baseline(
            observation,
            forward=forward,
            adjoint=adjoint,
            support=support,
            spacing_xyz=operator.spacing_xyz,
            iterations=warm_iterations,
        )
        low_projection = operator(warm.field[None, None])[0]
        if operator.call_report() != {
            "forward_calls": warm_iterations + 1,
            "adjoint_calls": warm_iterations,
        }:
            raise RuntimeError("shared warm/visible projection budget drifted")
        teacher = HighOrderTeacherMaps(operator)
        high_order_correction = teacher.correction(
            warm.field, low_projection=low_projection
        )
        damping = n15a._predict_fixed(fixed["component_damping"], record) * record.signal_scale
        names, features = record.features["curvature_visible"]
        if names != curvature_model.feature_names:
            raise RuntimeError("curvature feature contract drifted")
        curvature = curvature_model.predict(features).reshape_as(observation) * record.signal_scale
        exact = record.mismatch_normalized * record.signal_scale

        for spec in specs:
            candidate_id = str(spec["candidate_id"])
            kind = str(spec["kind"])
            beta = spec["beta"]
            started = time.perf_counter()
            if kind == "low_reference":
                operator.reset_call_counts()
                result = cgls_baseline(
                    observation,
                    forward=forward,
                    adjoint=adjoint,
                    support=support,
                    spacing_xyz=operator.spacing_xyz,
                    iterations=reference_iterations,
                )
                field = result.field
                raw = operator._forward(field[None, None])[0]
                modeled = raw
                correction = None
                low_calls = operator.call_report()
                high_calls = {"forward_calls": 0, "adjoint_calls": 0}
                evaluator_only = False
            elif kind == "direct_high_order_reference":
                direct = HighOrderTeacherMaps(operator)
                direct.reset_call_counts()
                result = cgls_baseline(
                    observation,
                    forward=direct.forward,
                    adjoint=direct.adjoint,
                    support=support,
                    spacing_xyz=operator.spacing_xyz,
                    iterations=direct_iterations,
                )
                field = result.field
                modeled = direct.forward(field)
                high_calls = direct.call_report()
                high_calls["forward_calls"] -= 1
                raw = operator._forward(field[None, None])[0]
                correction = None
                low_calls = {"forward_calls": 0, "adjoint_calls": 0}
                evaluator_only = False
            else:
                if kind == "damping":
                    correction = damping
                elif kind == "curvature":
                    correction = curvature
                elif kind == "damping_curvature":
                    correction = damping + float(beta) * (curvature - damping)
                elif kind == "damping_high_order":
                    correction = damping + float(beta) * (high_order_correction - damping)
                elif kind == "exact_oracle":
                    correction = exact
                else:
                    raise ValueError(f"unknown candidate kind: {kind}")
                operator.reset_call_counts()
                refined = warm_start_cgls(
                    observation - correction,
                    forward=forward,
                    adjoint=adjoint,
                    support=support,
                    initial_field=warm.field,
                    initial_projection=low_projection,
                    iterations=refine_iterations,
                )
                field = refined.field
                raw = operator._forward(field[None, None])[0]
                modeled = raw + correction
                refinement_calls = operator.call_report()
                low_calls = {
                    "forward_calls": warm_iterations + 1 + refinement_calls["forward_calls"],
                    "adjoint_calls": warm_iterations + refinement_calls["adjoint_calls"],
                }
                high_calls = {
                    "forward_calls": 1 if kind == "damping_high_order" else 0,
                    "adjoint_calls": 0,
                }
                evaluator_only = kind == "exact_oracle"
            elapsed = time.perf_counter() - started
            rows.append(
                _score(
                    record=record,
                    candidate_id=candidate_id,
                    kind=kind,
                    beta=None if beta is None else float(beta),
                    field=field,
                    modeled_correction=correction,
                    modeled_projection=modeled,
                    raw_low_projection=raw,
                    low_forward_calls=int(low_calls["forward_calls"]),
                    low_adjoint_calls=int(low_calls["adjoint_calls"]),
                    high_order_forward_calls=int(high_calls["forward_calls"]),
                    high_order_adjoint_calls=int(high_calls["adjoint_calls"]),
                    evaluator_only=evaluator_only,
                    runtime_seconds=elapsed,
                )
            )
    return rows


def _aggregate(rows: list[dict[str, Any]], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    baselines = {
        (row["partition"], row["case_id"]): row
        for row in rows if row["candidate_id"] == "low_cgls25"
    }
    damping = {
        (row["partition"], row["case_id"]): row
        for row in rows if row["candidate_id"] == "component_damping_correction"
    }
    output: list[dict[str, Any]] = []
    for partition in config["evaluated_partitions"]:
        for candidate_id in sorted({str(row["candidate_id"]) for row in rows}):
            selected = [
                row for row in rows
                if row["partition"] == partition and row["candidate_id"] == candidate_id
            ]
            field_gains = []
            h1_gains = []
            damping_gains = []
            for row in selected:
                baseline = baselines[(partition, row["case_id"])]
                field_gains.append(
                    1.0 - float(row["field_relative_l2"]) / float(baseline["field_relative_l2"])
                )
                h1_gains.append(
                    1.0
                    - float(row["h1_seminorm_relative_error"])
                    / float(baseline["h1_seminorm_relative_error"])
                )
                damping_row = damping[(partition, row["case_id"])]
                damping_gains.append(
                    1.0
                    - float(row["field_relative_l2"])
                    / float(damping_row["field_relative_l2"])
                )
            output.append(
                {
                    "partition": partition,
                    "candidate_id": candidate_id,
                    "kind": selected[0]["kind"],
                    "beta": selected[0]["beta"],
                    "geometry_cluster_count": len({int(row["base_seed"]) for row in selected}),
                    "case_count": len(selected),
                    "mean_field_gain_over_low_cgls25": float(np.mean(field_gains)),
                    "median_field_gain_over_low_cgls25": float(np.median(field_gains)),
                    "worst_case_field_gain_over_low_cgls25": float(min(field_gains)),
                    "mean_h1_gain_over_low_cgls25": float(np.mean(h1_gains)),
                    "mean_field_gain_over_component_damping": float(np.mean(damping_gains)),
                    "case_field_harm_over_one_percent_rate": float(np.mean(np.asarray(field_gains) < -0.01)),
                    "mean_modeled_continuous_reprojection_relative_l2": float(np.mean([
                        row["modeled_continuous_reprojection_relative_l2"] for row in selected
                    ])),
                    "mean_runtime_seconds": float(np.mean([row["runtime_seconds"] for row in selected])),
                    "low_forward_calls": int(selected[0]["low_forward_calls"]),
                    "low_adjoint_calls": int(selected[0]["low_adjoint_calls"]),
                    "high_order_forward_calls": int(selected[0]["high_order_forward_calls"]),
                    "high_order_adjoint_calls": int(selected[0]["high_order_adjoint_calls"]),
                    "evaluator_only": bool(selected[0]["evaluator_only"]),
                }
            )
    return output


def _select_future_confirmation(aggregates: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        row for row in aggregates
        if row["partition"] == "calibration"
        and row["kind"] == "damping_high_order"
        and float(row["case_field_harm_over_one_percent_rate"]) == 0.0
    ]
    if not eligible:
        raise RuntimeError("no safe high-order correction candidate remains on calibration")
    selected = max(
        eligible,
        key=lambda row: (
            float(row["mean_field_gain_over_low_cgls25"]),
            float(row["worst_case_field_gain_over_low_cgls25"]),
            -float(row["beta"]),
        ),
    )
    return {
        "candidate_id": selected["candidate_id"],
        "beta": selected["beta"],
        "selection_partition": "calibration",
        "selection_metric": "mean_field_gain_over_low_cgls25",
        "selection_is_postopen_hypothesis_generation_only": True,
        "requires_independent_confirmation": True,
    }


def _plot(aggregates: list[dict[str, Any]], output: Path) -> None:
    development = [row for row in aggregates if row["partition"] == "development"]
    development = sorted(
        development,
        key=lambda row: float(row["mean_field_gain_over_low_cgls25"]),
        reverse=True,
    )
    labels = [str(row["candidate_id"]).replace("_correction", "") for row in development]
    y = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(19, 8), constrained_layout=True)
    axes[0].barh(
        y - 0.18,
        [row["mean_field_gain_over_low_cgls25"] for row in development],
        height=0.34,
        label="mean field",
        color="#1d6f78",
    )
    axes[0].barh(
        y + 0.18,
        [row["worst_case_field_gain_over_low_cgls25"] for row in development],
        height=0.34,
        label="worst field",
        color="#c05640",
    )
    axes[0].axvline(0.0, color="#202020", linewidth=1)
    axes[0].axvline(0.05, color="#202020", linewidth=1, linestyle="--")
    axes[0].set_yticks(y, labels, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("field gain vs low CGLS-25")
    axes[0].legend()
    axes[1].barh(
        y,
        [row["mean_h1_gain_over_low_cgls25"] for row in development],
        color="#517a3a",
    )
    axes[1].axvline(0.03, color="#202020", linewidth=1, linestyle="--")
    axes[1].set_yticks([])
    axes[1].set_xlabel("mean H1 gain vs low CGLS-25")
    axes[2].barh(
        y,
        [row["mean_field_gain_over_component_damping"] for row in development],
        color="#8a5a20",
    )
    axes[2].axvline(0.0, color="#202020", linewidth=1)
    axes[2].set_yticks([])
    axes[2].set_xlabel("field gain vs damping correction")
    fig.suptitle("N1.5-B post-open reconstruction-aware mechanism screen", fontsize=15)
    fig.savefig(output / "diagnostic.png", dpi=180)
    fig.savefig(output / "diagnostic.pdf")
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    config_path = args.config.resolve()
    config = _read_json(config_path)
    n15a_results = ROOT / str(config["source_n1_5_a_results"])
    n15a_summary = _read_json(n15a_results / "summary.json")
    _validate_config(config, n15a_summary, args.seed_limit)
    output = args.output_dir.resolve()
    if output.exists():
        if not args.replace_output:
            raise FileExistsError(f"output already exists: {output}")
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    started = time.perf_counter()
    source = _read_json(ROOT / str(config["source_t0_config"]))
    n15a_config = _read_json(ROOT / str(config["source_n1_5_a_config"]))
    records, manifest = n15a._prepare_records(
        n15a_config, source, seed_limit=args.seed_limit
    )
    fixed = n15a._fixed_predictors(records)
    curvature_model = _load_curvature_model(n15a_results / "selected_ridge_models.json")
    case_rows = _run_cases(records, fixed, curvature_model, config)
    aggregates = _aggregate(case_rows, config)
    selection = _select_future_confirmation(aggregates)
    _write_csv(output / "case_metrics.csv", case_rows)
    _write_csv(output / "aggregate_metrics.csv", aggregates)
    _write_csv(output / "case_manifest.csv", manifest)
    summary = {
        "schema": REPORT_SCHEMA,
        "status": "POSTOPEN_HYPOTHESIS_ONLY_REQUIRES_INDEPENDENT_CONFIRMATION",
        "is_postopen_exploratory": True,
        "may_claim_confirmed_algorithm_gain": False,
        "independent_unit": "base_seed_geometry_cluster",
        "calibration_geometry_cluster_count": len({
            int(row["base_seed"]) for row in case_rows if row["partition"] == "calibration"
        }),
        "development_geometry_cluster_count": len({
            int(row["base_seed"]) for row in case_rows if row["partition"] == "development"
        }),
        "future_confirmation_hypothesis": selection,
        "n1_5_a_status": n15a_summary["status"],
        "opens_ood_fresh_or_final": False,
        "runtime_seconds": time.perf_counter() - started,
        "aggregates": aggregates,
        "claim_boundary": config["claim_boundary"],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    provenance = {
        "schema": "jacru-n1-5-reconstruction-aware-postopen-provenance-1.0",
        "git_commit_at_start": _git_commit(),
        "source_sha256": _source_manifest(config_path, config),
        "config": config,
        "exact_cli": " ".join(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    _plot(aggregates, output)
    development_lookup = {
        row["candidate_id"]: row
        for row in aggregates if row["partition"] == "development"
    }
    chosen = development_lookup[selection["candidate_id"]]
    readme = f"""# N1.5-B post-open reconstruction-aware screen\n\nStatus: **POSTOPEN HYPOTHESIS ONLY**. This run cannot establish an algorithm gain.\n\nThe N1.5-A visible curvature ridge reduced forward mismatch but failed its no-harm gate. This follow-up asks whether any correction improves the reconstructed field under an explicit call budget.\n\n- Corrected candidates: CGLS-12 warm start + one visible low projection + warm-start CGLS-12 = 25 low forward and 24 low adjoint calls.\n- Strong reference: low-order CGLS-25 = 25 low forward and 25 low adjoint calls.\n- Direct fourth-order CGLS-25 is included as a strong physics comparator.\n- Exact mismatch is evaluator-only and excluded from selection.\n- The future confirmation hypothesis was chosen on calibration only: `{selection['candidate_id']}` (beta={selection['beta']}).\n- Its already-opened development mean field gain is {chosen['mean_field_gain_over_low_cgls25']:.6f}, H1 gain is {chosen['mean_h1_gain_over_low_cgls25']:.6f}, and worst-case field gain is {chosen['worst_case_field_gain_over_low_cgls25']:.6f}. These are exploratory numbers, not confirmation evidence.\n- OOD/fresh/final and real BOST claims remain closed.\n"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    _write_checksums(output)
    print(json.dumps({
        "status": summary["status"],
        "future_confirmation_hypothesis": selection,
        "output": str(output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
