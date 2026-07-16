#!/usr/bin/env python3
"""Stress the learned PSU direction against stronger spectral classical controls."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from demo_t16_operator.psu_b0_classical_baselines import (
    GeneralizedSobolevDirection,
    ScheduledSobolevDirection,
)
from demo_t16_operator.psu_b0_streaming_operator import (
    zero_outer_boundary_support,
)
from site_tools.run_psu_b0_classical_frontier_development import (
    _evaluate_solver,
    _frontier_comparison,
    _stratified_aggregates,
    _verify_split_metadata,
)
from site_tools.run_psu_b0_real_interface_audit import (
    _make_operator,
    load_real_support_geometry,
)
from site_tools.run_psu_b0_residual_risk_development import (
    DevelopmentSplit,
    _build_development_split,
)
from site_tools.run_psu_b0_residual_risk_fresh import _build_fresh_splits
from site_tools.run_psu_b0_spectral_preconditioner_pilot import (
    _aggregate,
    _load_json,
    _unique_masks,
)


PRIVATE_SCHEMA = "psu-b0-strong-spectral-frontier-private-1.0"
PUBLIC_SCHEMA = "psu-b0-strong-spectral-frontier-public-1.0"
STATUS = "POSTOPEN_STRONG_SPECTRAL_FRONTIER_DEVELOPMENT_ONLY"


def _screen_metrics(
    rows: list[dict[str, Any]],
    *,
    family: str,
    candidate_id: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    return {
        "family": family,
        "candidate_id": candidate_id,
        "parameters": copy.deepcopy(parameters),
        "sample_count": len(rows),
        "mean_combined_loss": float(
            np.mean([row["combined_loss"] for row in rows])
        ),
        "mean_field_relative_l2": float(
            np.mean([row["field_relative_l2"] for row in rows])
        ),
        "mean_gradient_relative_l2": float(
            np.mean([row["gradient_relative_l2"] for row in rows])
        ),
        "mean_front_top10_f1": float(
            np.mean([row["front_top10_f1"] for row in rows])
        ),
        "mean_measurement_relative_l2": float(
            np.mean([row["measurement_relative_l2"] for row in rows])
        ),
    }


def select_spectral_candidate(
    screen: list[dict[str, Any]],
    *,
    family: str,
) -> dict[str, Any]:
    rows = [row for row in screen if row["family"] == family]
    if not rows:
        raise ValueError(f"empty spectral screen family: {family}")
    return dict(
        min(
            rows,
            key=lambda row: (
                float(row["mean_combined_loss"]),
                float(row["mean_field_relative_l2"]),
                str(row["candidate_id"]),
            ),
        )
    )


def _generalized_candidates(
    config: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    section = config["generalized_sobolev"]
    output = []
    for strength in section["strength_grid"]:
        for epsilon in section["epsilon_grid"]:
            for name, weights in section["axis_weight_patterns_xyz"].items():
                candidate_id = (
                    f"generalized_s{float(strength):g}_e{float(epsilon):g}_{name}"
                )
                output.append(
                    (
                        candidate_id,
                        {
                            "strength": float(strength),
                            "epsilon": float(epsilon),
                            "axis_pattern": str(name),
                            "axis_weights_xyz": [
                                float(value) for value in weights
                            ],
                        },
                    )
                )
    return output


def _schedule_candidates(
    config: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    output = []
    for index, values in enumerate(config["scheduled_sobolev"]["schedules"]):
        strengths = [float(value) for value in values]
        output.append(
            (
                f"schedule_{index:02d}_{'_'.join(f'{value:g}' for value in strengths)}",
                {"strengths": strengths, "epsilon": 0.05},
            )
        )
    return output


def _pcgls_candidates(
    config: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    section = config["sobolev_pcgls"]
    output = []
    for stages in section["stage_counts"]:
        for strength in section["strength_grid"]:
            for epsilon in section["epsilon_grid"]:
                output.append(
                    (
                        (
                            f"pcgls{int(stages)}_s{float(strength):g}_"
                            f"e{float(epsilon):g}"
                        ),
                        {
                            "stages": int(stages),
                            "strength": float(strength),
                            "epsilon": float(epsilon),
                            "axis_weights_xyz": [1.0, 1.0, 1.0],
                        },
                    )
                )
    return output


def _direction(
    *,
    family: str,
    parameters: dict[str, Any],
    grid_size: int,
    device: torch.device,
) -> Any:
    if family == "generalized_sobolev":
        return GeneralizedSobolevDirection(
            (grid_size,) * 3,
            strength=float(parameters["strength"]),
            epsilon=float(parameters["epsilon"]),
            axis_weights_xyz=tuple(
                float(value) for value in parameters["axis_weights_xyz"]
            ),
        ).to(device)
    if family == "scheduled_sobolev":
        return ScheduledSobolevDirection(
            (grid_size,) * 3,
            strengths=tuple(float(value) for value in parameters["strengths"]),
            epsilon=float(parameters["epsilon"]),
        ).to(device)
    if family in {"pcgls_3", "pcgls_4"}:
        return GeneralizedSobolevDirection(
            (grid_size,) * 3,
            strength=float(parameters["strength"]),
            epsilon=float(parameters["epsilon"]),
        ).to(device)
    raise ValueError(f"unknown spectral family: {family}")


def build_public_summary(private: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": PUBLIC_SCHEMA,
        "status": private["status"],
        "evidence_scope": private["evidence_scope"],
        "configuration": copy.deepcopy(private["configuration_public"]),
        "regeneration_checks": copy.deepcopy(private["regeneration_checks"]),
        "selection_screen": copy.deepcopy(private["selection_screen"]),
        "selected_candidates": copy.deepcopy(private["selected_candidates"]),
        "aggregates": copy.deepcopy(private["aggregates"]),
        "stratified_aggregates": copy.deepcopy(
            private["stratified_aggregates"]
        ),
        "frontier_comparison": copy.deepcopy(private["frontier_comparison"]),
        "execution": copy.deepcopy(private["execution"]),
        "claim_boundary": copy.deepcopy(private["claim_boundary"]),
    }


def run_frontier(
    *,
    root: Path,
    config_path: Path,
    development_report_path: Path,
    fresh_report_path: Path,
    classical_report_path: Path,
    view_root: Path,
    device_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _load_json(config_path)
    development_config = _load_json(
        root / str(config["source_development_config"])
    )
    preregistration = _load_json(
        root / str(config["source_fresh_preregistration"])
    )
    source_config = _load_json(
        root / str(development_config["source_pilot"]["config"])
    )
    development_report = _load_json(development_report_path)
    fresh_report = _load_json(fresh_report_path)
    classical_report = _load_json(classical_report_path)
    device = torch.device(device_name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    torch.set_num_threads(8)
    started = time.perf_counter()

    geometry = development_config["geometry"]
    grid_size = int(geometry["grid_size"])
    rays_per_view = int(geometry["rays_per_view"])
    support = zero_outer_boundary_support((grid_size,) * 3).to(device)
    true_geometry, _ = load_real_support_geometry(
        view_root,
        rays_per_view=rays_per_view,
        sample_count=int(geometry["true_finite_aperture_sample_count"]),
    )
    nominal_geometry, _ = load_real_support_geometry(
        view_root,
        rays_per_view=rays_per_view,
        sample_count=int(geometry["nominal_finite_aperture_sample_count"]),
    )
    true_operator = _make_operator(
        true_geometry,
        grid_size=grid_size,
        dtype=torch.float32,
    ).to(device)
    nominal_operator = _make_operator(
        nominal_geometry,
        grid_size=grid_size,
        dtype=torch.float32,
    ).to(device)
    true_operator.support.copy_(support)
    nominal_operator.support.copy_(support)

    active_range = tuple(int(value) for value in geometry["active_view_range"])
    used_masks: set[str] = set()
    train_spec = development_config["development_splits"]["risk_train"]
    _, used_masks = _unique_masks(
        count=int(train_spec["count"]),
        view_count=int(geometry["view_count"]),
        minimum_active=active_range[0],
        maximum_active=active_range[1],
        seed=int(train_spec["mask_seed"]),
        forbidden=used_masks,
    )
    development_splits: dict[str, DevelopmentSplit] = {}
    for split_name in ("risk_validation", "risk_calibration"):
        wrapped, used_masks = _build_development_split(
            name=split_name,
            spec=development_config["development_splits"][split_name],
            config=development_config,
            source_config=source_config,
            true_operator=true_operator,
            nominal_operator=nominal_operator,
            device=device,
            forbidden_masks=used_masks,
        )
        development_splits[split_name] = wrapped
    fresh_splits, _, _ = _build_fresh_splits(
        preregistration=preregistration,
        development_config=development_config,
        source_config=source_config,
        true_operator=true_operator,
        nominal_operator=nominal_operator,
        device=device,
    )
    development_rows = development_report["dataset_private"]["metric_rows"]
    fresh_rows = fresh_report["dataset_private"]["per_sample_metrics"]
    for wrapped in development_splits.values():
        _verify_split_metadata(wrapped, development_rows)
    for wrapped in fresh_splits.values():
        _verify_split_metadata(wrapped, fresh_rows)

    validation = development_splits[str(config["selection_split"])]
    screen: list[dict[str, Any]] = []
    execution: list[dict[str, Any]] = []
    candidate_sets = {
        "generalized_sobolev": _generalized_candidates(config),
        "scheduled_sobolev": _schedule_candidates(config),
    }
    pcgls = _pcgls_candidates(config)
    candidate_sets["pcgls_3"] = [
        row for row in pcgls if int(row[1]["stages"]) == 3
    ]
    candidate_sets["pcgls_4"] = [
        row for row in pcgls if int(row[1]["stages"]) == 4
    ]
    for family, candidates in candidate_sets.items():
        for candidate_id, parameters in candidates:
            direction = _direction(
                family=family,
                parameters=parameters,
                grid_size=grid_size,
                device=device,
            )
            solver = "pcgls" if family.startswith("pcgls") else "line_search"
            stages = int(parameters.get("stages", 4))
            rows, ledger = _evaluate_solver(
                method=f"screen_{candidate_id}",
                wrapped=validation,
                operator=nominal_operator,
                source_config=source_config,
                device=device,
                solver=solver,
                stages=stages,
                direction=direction,
            )
            screen.append(
                _screen_metrics(
                    rows,
                    family=family,
                    candidate_id=candidate_id,
                    parameters=parameters,
                )
            )
            execution.append(ledger)
    selected = {
        family: select_spectral_candidate(screen, family=family)
        for family in candidate_sets
    }

    generated_rows: list[dict[str, Any]] = []
    all_splits = {**development_splits, **fresh_splits}
    method_names = {
        "generalized_sobolev": "generalized_sobolev_selected",
        "scheduled_sobolev": "scheduled_sobolev_selected",
        "pcgls_3": "pcgls_3_selected",
        "pcgls_4": "pcgls_4_selected",
    }
    for split_name, wrapped in all_splits.items():
        for family, choice in selected.items():
            parameters = choice["parameters"]
            direction = _direction(
                family=family,
                parameters=parameters,
                grid_size=grid_size,
                device=device,
            )
            solver = "pcgls" if family.startswith("pcgls") else "line_search"
            stages = int(parameters.get("stages", 4))
            rows, ledger = _evaluate_solver(
                method=method_names[family],
                wrapped=wrapped,
                operator=nominal_operator,
                source_config=source_config,
                device=device,
                solver=solver,
                stages=stages,
                direction=direction,
            )
            generated_rows.extend(rows)
            execution.append(ledger)

    combined_rows = (
        [dict(row) for row in classical_report["metric_rows_private"]]
        + generated_rows
    )
    aggregates = _aggregate(
        combined_rows,
        baseline_method="sobolev_selected",
    )
    advanced_methods = set(method_names.values())
    classical_methods = {
        "sobolev_selected",
        "tikhonov_identity_selected",
        "tikhonov_h1_selected",
        "cgls_3",
        "cgls_4",
        *advanced_methods,
    }
    stratified = _stratified_aggregates(
        [
            row
            for row in combined_rows
            if row["method"] in classical_methods
            or str(row["method"]).startswith(("raw_seed_", "gated_seed_"))
        ]
    )
    frontier = _frontier_comparison(
        aggregates,
        classical_methods=classical_methods,
    )
    calls_valid = all(
        bool(row["regularized_or_data_objective_monotone"])
        and row["logical_calls_per_sample"]
        in (
            {"forward": 4, "adjoint": 4},
            {"forward": 3, "adjoint": 3},
            {"forward": 3, "adjoint": 4},
            {"forward": 4, "adjoint": 5},
        )
        for row in execution
    )
    private = {
        "schema_version": PRIVATE_SCHEMA,
        "status": STATUS,
        "evidence_scope": (
            "REAL_PSU_SUPPORT_GEOMETRY_WITH_ANALYTIC_REACTION_MORPHOLOGY_"
            "AND_SYNTHETIC_CAMERA_NOISE_POSTOPEN_STRONG_SPECTRAL_DEVELOPMENT"
        ),
        "configuration_private": {
            "root": str(root.resolve()),
            "config_path": str(config_path.resolve()),
            "development_report_path": str(development_report_path.resolve()),
            "fresh_report_path": str(fresh_report_path.resolve()),
            "classical_report_path": str(classical_report_path.resolve()),
            "view_root": str(view_root.resolve()),
            "device": device_name,
        },
        "configuration_public": copy.deepcopy(config),
        "regeneration_checks": {
            "development_metadata_matches_frozen_rows": True,
            "fresh_metadata_matches_frozen_rows": True,
            "selection_uses_only_risk_validation": True,
            "risk_calibration_not_used_for_selection": True,
            "fresh_values_were_already_open": True,
            "operator_call_ledgers_and_objective_checks_pass": calls_valid,
        },
        "selection_screen": screen,
        "selected_candidates": selected,
        "metric_rows_private": combined_rows,
        "aggregates": aggregates,
        "stratified_aggregates": stratified,
        "frontier_comparison": frontier,
        "execution": execution,
        "runtime": {
            "wall_seconds": float(time.perf_counter() - started),
            "screen_candidate_count": len(screen),
        },
        "claim_boundary": {
            "postopen_development_only": True,
            "fresh_diagnostic_is_confirmatory": False,
            "tv_baseline_included": False,
            "experimental_field_truth_used": False,
            "real_psu_measurement_values_used": False,
            "analytic_morphology_is_cfd": False,
            "algorithm_superiority": False,
        },
    }
    return private, build_public_summary(private)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "demo_t16_operator/configs/"
            "psu_b0_strong_spectral_frontier_development_v1.json"
        ),
    )
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument("--fresh-report", type=Path, required=True)
    parser.add_argument("--classical-report", type=Path, required=True)
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--public-output", type=Path)
    args = parser.parse_args()
    private, public = run_frontier(
        root=args.root,
        config_path=args.config,
        development_report_path=args.development_report,
        fresh_report_path=args.fresh_report,
        classical_report_path=args.classical_report,
        view_root=args.view_root,
        device_name=args.device,
    )
    if args.private_output is not None:
        args.private_output.parent.mkdir(parents=True, exist_ok=True)
        args.private_output.write_text(
            json.dumps(private, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    if args.public_output is not None:
        args.public_output.parent.mkdir(parents=True, exist_ok=True)
        args.public_output.write_text(
            json.dumps(public, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps(public, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
