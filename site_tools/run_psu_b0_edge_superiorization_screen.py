#!/usr/bin/env python3
"""Screen TV/Huber superiorization scales on already opened PSU B0 cases."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import platform
import resource
import time
from typing import Any

import numpy as np
import torch

from demo_t16_operator.detector_covariance_whitening import (
    DetectorCovarianceWhitening,
    WhitenedMeasurementOperator,
)
from demo_t16_operator.detector_graph_covariance import (
    detector_graph_spectral_basis,
)
from demo_t16_operator.psu_b0_classical_baselines import (
    GeneralizedSobolevDirection,
    preconditioned_cgls_reconstruction,
    preconditioned_cgls_trajectory,
)
from demo_t16_operator.psu_b0_detector_graph_features import (
    build_detector_knn_graph,
)
from demo_t16_operator.psu_b0_edge_superiorization import (
    superiorized_pcgls_reconstruction,
)
from demo_t16_operator.psu_b0_reaction_phantoms import (
    reaction_morphology_batch,
)
from demo_t16_operator.psu_b0_streaming_operator import (
    zero_outer_boundary_support,
)
from site_tools.run_psu_b0_covariance_conditioned_pcgls_diagnosis import (
    _derive_smoke_config,
)
from site_tools.run_psu_b0_dg_wpcgls_smoke import (
    _calibrate,
    _flowon_unit_noise,
    _load_json,
    _synchronize,
)
from site_tools.run_psu_b0_real_interface_audit import (
    _make_operator,
    load_real_support_geometry,
)
from site_tools.run_psu_b0_spectral_preconditioner_pilot import (
    _field_metrics,
)


SCHEMA = "psu-b0-edge-superiorization-screen-report-1.0"
STATUS = "POSTOPEN_TV_HUBER_SCALE_SMOKE_COMPLETE"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _slug(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def superiorized_candidate_grid(
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    screen = config["screen"]
    candidates = []
    products = itertools.product(
        screen["penalties"],
        screen["stages"],
        screen["perturbation_inner_steps"],
        screen["perturbation_initial_steps"],
        screen["perturbation_decays"],
    )
    for penalty, stages, inner_steps, initial_step, decay in products:
        count = int(stages)
        candidate = {
            "candidate_id": (
                f"sup_{penalty}_k{count}_n{int(inner_steps)}_"
                f"g{_slug(float(initial_step))}_"
                f"a{_slug(float(decay))}"
            ),
            "method": "superiorized_pcgls",
            "covariance_mode": "full_graph",
            "penalty": str(penalty),
            "stages": count,
            "perturbation_inner_steps": int(inner_steps),
            "perturbation_initial_step": float(initial_step),
            "perturbation_decay": float(decay),
            "forward_calls": 2 * count - 1,
            "adjoint_calls": count,
            "total_operator_calls": 3 * count - 1,
        }
        candidates.append(candidate)
    expected = int(screen["superiorized_candidate_count_expected"])
    if len(candidates) != expected:
        raise ValueError(
            f"superiorized grid has {len(candidates)} rows, expected {expected}"
        )
    identifiers = [str(row["candidate_id"]) for row in candidates]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("superiorized candidate identifiers must be unique")
    return candidates


def baseline_candidates(config: dict[str, Any]) -> list[dict[str, Any]]:
    component_stages = int(config["baselines"]["component_stages"])
    candidates = [
        {
            "candidate_id": f"component_s3_k{component_stages}",
            "method": "pcgls",
            "covariance_mode": "component_iid",
            "penalty": "none",
            "stages": component_stages,
            "perturbation_inner_steps": 0,
            "perturbation_initial_step": 0.0,
            "perturbation_decay": 0.0,
            "forward_calls": component_stages,
            "adjoint_calls": component_stages,
            "total_operator_calls": 2 * component_stages,
        }
    ]
    for raw in config["baselines"]["graph_stages"]:
        stages = int(raw)
        candidates.append(
            {
                "candidate_id": f"graph_s3_k{stages}",
                "method": "pcgls",
                "covariance_mode": "full_graph",
                "penalty": "none",
                "stages": stages,
                "perturbation_inner_steps": 0,
                "perturbation_initial_step": 0.0,
                "perturbation_decay": 0.0,
                "forward_calls": stages,
                "adjoint_calls": stages,
                "total_operator_calls": 2 * stages,
            }
        )
    return candidates


def validate_replicates(
    config: dict[str, Any],
    *,
    replicate_count: int,
) -> list[int]:
    values = [int(value) for value in config["replicate_indices"]]
    if not values or len(values) != len(set(values)):
        raise ValueError("replicate_indices must be nonempty and unique")
    if any(not 0 <= value < int(replicate_count) for value in values):
        raise ValueError("replicate index is outside the opened seed range")
    return values


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _rows_for_metrics(
    *,
    replicate: int,
    candidate: dict[str, Any],
    families: list[str],
    metrics: dict[str, torch.Tensor],
    elapsed_seconds: float,
    perturbation_norm: float,
    perturbation_trials: float,
    final_exponent: float,
) -> list[dict[str, Any]]:
    return [
        {
            "replicate": int(replicate),
            "sample_index": int(index),
            "reaction_family": family,
            **candidate,
            "field_relative_l2": float(metrics["field_relative_l2"][index]),
            "gradient_relative_l2": float(
                metrics["gradient_relative_l2"][index]
            ),
            "front_top10_f1": float(metrics["front_top10_f1"][index]),
            "solver_elapsed_seconds": float(elapsed_seconds),
            "mean_perturbation_norm": float(perturbation_norm),
            "mean_perturbation_trials": float(perturbation_trials),
            "mean_final_exponent": float(final_exponent),
        }
        for index, family in enumerate(families)
    ]


def add_comparative_gains(
    rows: list[dict[str, Any]],
    *,
    component_baseline_id: str,
) -> list[dict[str, Any]]:
    by_key = {
        (
            int(row["replicate"]),
            int(row["sample_index"]),
            str(row["candidate_id"]),
        ): row
        for row in rows
    }
    graph_budgets = sorted(
        {
            (
                int(row["total_operator_calls"]),
                str(row["candidate_id"]),
            )
            for row in rows
            if row["covariance_mode"] == "full_graph"
            and row["method"] == "pcgls"
        }
    )
    output = []
    for row in rows:
        replicate = int(row["replicate"])
        sample_index = int(row["sample_index"])
        total_calls = int(row["total_operator_calls"])
        floor = max(
            (
                item
                for item in graph_budgets
                if item[0] <= total_calls
            ),
            default=graph_budgets[0],
        )
        ceiling = min(
            (
                item
                for item in graph_budgets
                if item[0] >= total_calls
            ),
            default=graph_budgets[-1],
        )
        references = {
            "component": by_key[
                (replicate, sample_index, component_baseline_id)
            ],
            "graph_same_stage": by_key[
                (
                    replicate,
                    sample_index,
                    f"graph_s3_k{int(row['stages'])}",
                )
            ],
            "graph_budget_floor": by_key[
                (replicate, sample_index, floor[1])
            ],
            "graph_budget_ceiling": by_key[
                (replicate, sample_index, ceiling[1])
            ],
        }
        additions: dict[str, Any] = {
            "graph_same_stage_id": f"graph_s3_k{int(row['stages'])}",
            "graph_budget_floor_id": floor[1],
            "graph_budget_ceiling_id": ceiling[1],
        }
        for label, reference in references.items():
            reference_field = float(reference["field_relative_l2"])
            reference_gradient = float(reference["gradient_relative_l2"])
            additions[f"field_gain_vs_{label}_percent"] = 100.0 * (
                reference_field - float(row["field_relative_l2"])
            ) / max(reference_field, 1e-12)
            additions[f"gradient_gain_vs_{label}_percent"] = 100.0 * (
                reference_gradient - float(row["gradient_relative_l2"])
            ) / max(reference_gradient, 1e-12)
            additions[f"front_gain_vs_{label}"] = (
                float(row["front_top10_f1"])
                - float(reference["front_top10_f1"])
            )
        output.append({**row, **additions})
    return output


def summarize_candidates(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["candidate_id"]), []).append(row)
    summaries = []
    for candidate_id, selected in sorted(grouped.items()):
        floor_gain = np.asarray(
            [
                float(row["field_gain_vs_graph_budget_floor_percent"])
                for row in selected
            ],
            dtype=np.float64,
        )
        ceiling_gain = np.asarray(
            [
                float(row["field_gain_vs_graph_budget_ceiling_percent"])
                for row in selected
            ],
            dtype=np.float64,
        )
        component_gain = np.asarray(
            [
                float(row["field_gain_vs_component_percent"])
                for row in selected
            ],
            dtype=np.float64,
        )
        summary = {
            key: selected[0][key]
            for key in (
                "candidate_id",
                "method",
                "covariance_mode",
                "penalty",
                "stages",
                "perturbation_inner_steps",
                "perturbation_initial_step",
                "perturbation_decay",
                "forward_calls",
                "adjoint_calls",
                "total_operator_calls",
                "graph_same_stage_id",
                "graph_budget_floor_id",
                "graph_budget_ceiling_id",
            )
        }
        summary.update(
            {
                "replicate_count": len(
                    {int(row["replicate"]) for row in selected}
                ),
                "field_count": len(selected),
                "mean_field_gain_vs_component_percent": float(
                    np.mean(component_gain)
                ),
                "mean_field_gain_vs_graph_same_stage_percent": float(
                    np.mean(
                        [
                            float(
                                row[
                                    "field_gain_vs_graph_same_stage_percent"
                                ]
                            )
                            for row in selected
                        ]
                    )
                ),
                "field_gain_vs_graph_same_stage_p10_percent": float(
                    np.quantile(
                        [
                            float(
                                row[
                                    "field_gain_vs_graph_same_stage_percent"
                                ]
                            )
                            for row in selected
                        ],
                        0.1,
                    )
                ),
                "mean_field_gain_vs_graph_budget_floor_percent": float(
                    np.mean(floor_gain)
                ),
                "field_gain_vs_graph_budget_floor_p10_percent": float(
                    np.quantile(floor_gain, 0.1)
                ),
                "field_harm_vs_graph_budget_floor_over_one_percent_rate": (
                    float(np.mean(floor_gain < -1.0))
                ),
                "worst_field_gain_vs_graph_budget_floor_percent": float(
                    np.min(floor_gain)
                ),
                "mean_field_gain_vs_graph_budget_ceiling_percent": float(
                    np.mean(ceiling_gain)
                ),
                "mean_gradient_gain_vs_graph_budget_floor_percent": float(
                    np.mean(
                        [
                            float(
                                row[
                                    "gradient_gain_vs_graph_budget_floor_percent"
                                ]
                            )
                            for row in selected
                        ]
                    )
                ),
                "mean_front_gain_vs_graph_budget_floor": float(
                    np.mean(
                        [
                            float(
                                row["front_gain_vs_graph_budget_floor"]
                            )
                            for row in selected
                        ]
                    )
                ),
                "mean_solver_elapsed_seconds": float(
                    np.mean(
                        [
                            float(row["solver_elapsed_seconds"])
                            for row in selected
                        ]
                    )
                ),
                "mean_perturbation_norm": float(
                    np.mean(
                        [
                            float(row["mean_perturbation_norm"])
                            for row in selected
                        ]
                    )
                ),
            }
        )
        summaries.append(summary)
    return summaries


def rank_superiorized(
    summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in summaries
        if row["method"] == "superiorized_pcgls"
    ]
    return sorted(
        selected,
        key=lambda row: (
            -float(
                row[
                    "mean_field_gain_vs_graph_budget_floor_percent"
                ]
            ),
            -float(
                row[
                    "field_gain_vs_graph_budget_floor_p10_percent"
                ]
            ),
            float(
                row[
                    "field_harm_vs_graph_budget_floor_over_one_percent_rate"
                ]
            ),
            -float(row["mean_front_gain_vs_graph_budget_floor"]),
            int(row["total_operator_calls"]),
            str(row["candidate_id"]),
        ),
    )


def screen_decision(
    ranking: list[dict[str, Any]],
) -> dict[str, Any]:
    if not ranking:
        raise ValueError("screen decision needs superiorized candidates")
    best = ranking[0]
    any_budget_positive = any(
        float(
            row["mean_field_gain_vs_graph_budget_floor_percent"]
        )
        > 0.0
        for row in ranking
    )
    any_same_stage_positive = any(
        float(row["mean_field_gain_vs_graph_same_stage_percent"]) > 0.0
        for row in ranking
    )
    return {
        "status": (
            "POSTOPEN_SUPPCG_BUDGET_SIGNAL_PRESENT"
            if any_budget_positive
            else "POSTOPEN_SUPPCG_BUDGET_EFFICIENCY_NO_GO"
        ),
        "any_candidate_positive_vs_graph_budget_floor_mean": (
            any_budget_positive
        ),
        "any_candidate_positive_vs_graph_same_stage_mean": (
            any_same_stage_positive
        ),
        "best_candidate_id": str(best["candidate_id"]),
        "best_mean_field_gain_vs_graph_budget_floor_percent": float(
            best["mean_field_gain_vs_graph_budget_floor_percent"]
        ),
        "best_field_gain_vs_graph_budget_floor_p10_percent": float(
            best["field_gain_vs_graph_budget_floor_p10_percent"]
        ),
        "best_field_harm_vs_graph_budget_floor_over_one_percent_rate": (
            float(
                best[
                    "field_harm_vs_graph_budget_floor_over_one_percent_rate"
                ]
            )
        ),
        "best_worst_field_gain_vs_graph_budget_floor_percent": float(
            best["worst_field_gain_vs_graph_budget_floor_percent"]
        ),
        "best_mean_field_gain_vs_graph_same_stage_percent": float(
            best["mean_field_gain_vs_graph_same_stage_percent"]
        ),
        "fresh_authorized": False,
        "full_opened_grid_authorized": False,
        "neural_operator_training_authorized": False,
    }


def run_screen(
    *,
    root: Path,
    config_path: Path,
    view_root: Path,
    device_name: str,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    config = _load_json(config_path)
    multiseed_path = root / str(config["source_multiseed_config"])
    smoke_path = root / str(config["source_smoke_config"])
    stopping_path = root / str(config["source_stopping_no_go"])
    multiseed = _load_json(multiseed_path)
    smoke = _load_json(smoke_path)
    replicates = validate_replicates(
        config,
        replicate_count=int(multiseed["replicates"]["count"]),
    )
    baselines = baseline_candidates(config)
    superiorized = superiorized_candidate_grid(config)
    device = torch.device(device_name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    torch.set_num_threads(8)
    started = time.perf_counter()

    geometry_config = smoke["geometry"]
    view_count = int(geometry_config["view_count"])
    rays_per_view = int(geometry_config["rays_per_view"])
    grid_size = int(geometry_config["grid_size"])
    geometry, _ = load_real_support_geometry(
        view_root,
        rays_per_view=rays_per_view,
        sample_count=int(
            geometry_config["finite_aperture_sample_count"]
        ),
    )
    operator = _make_operator(
        geometry,
        grid_size=grid_size,
        dtype=torch.float32,
    ).to(device)
    operator.support.copy_(
        zero_outer_boundary_support((grid_size,) * 3).to(device)
    )
    detector_config = _load_json(
        root / str(smoke["source_detector_config"])
    )
    graph_config = detector_config["detector_graph"]
    graph = build_detector_knn_graph(
        geometry["detector_xy"],
        view_count=view_count,
        rays_per_view=rays_per_view,
        neighbor_count=int(graph_config["neighbor_count"]),
        least_squares_ridge=float(
            graph_config["least_squares_ridge"]
        ),
    )
    eigenpairs = [
        detector_graph_spectral_basis(graph, view_index=view)
        for view in range(view_count)
    ]
    eigenvectors_by_view = [vectors for _, vectors in eigenpairs]
    graph_family_index = [
        str(value) for value in smoke["data"]["noise_families"]
    ].index("graph_heat")
    families = [str(value) for value in smoke["data"]["reaction_families"]]
    ones_sigma = torch.ones(
        (len(families), view_count),
        dtype=torch.float32,
        device=device,
    )
    ones_mask = torch.ones_like(ones_sigma)
    strength = float(config["solver"]["sobolev_strength"])
    epsilon = float(config["solver"]["sobolev_epsilon"])
    metric_rows: list[dict[str, Any]] = []
    ledger = []

    for replicate in replicates:
        derived = _derive_smoke_config(
            smoke,
            multiseed,
            replicate=replicate,
        )
        field_seeds = [
            int(derived["data"]["field_seed_start"]) + index
            for index in range(len(families))
        ]
        truth = reaction_morphology_batch(
            grid_size=grid_size,
            families=families,
            seeds=field_seeds,
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            clean = operator(truth).detach()
        clean_by_view = clean.reshape(
            len(truth),
            view_count,
            rays_per_view,
            2,
        )
        view_rms = torch.sqrt(
            torch.mean(
                clean_by_view.square(),
                dim=(2, 3),
            ).clamp_min(1e-20)
        )
        floor = (
            float(derived["data"]["minimum_view_signal_fraction"])
            * torch.mean(view_rms, dim=1, keepdim=True)
        )
        levels = tuple(
            float(value)
            for value in derived["data"]["relative_noise_levels"]
        )
        relative_noise = torch.as_tensor(
            [
                levels[index % len(levels)]
                for index in range(len(truth))
            ],
            dtype=torch.float32,
            device=device,
        )
        view_factors = torch.as_tensor(
            derived["data"]["view_noise_factors"],
            dtype=torch.float32,
            device=device,
        )
        scale_by_view = (
            relative_noise[:, None]
            * torch.maximum(view_rms, floor)
            * view_factors[None]
        ).clamp_min(1e-8)
        fits, gate_rows = _calibrate(
            family="graph_heat",
            family_index=graph_family_index,
            config=derived,
            graph=graph,
            eigenpairs=eigenpairs,
        )
        unit_noise = _flowon_unit_noise(
            family="graph_heat",
            family_index=graph_family_index,
            sample_count=len(truth),
            config=derived,
            graph=graph,
            eigenpairs=eigenpairs,
        ).to(device)
        observation = (
            clean
            + unit_noise
            * scale_by_view.repeat_interleave(
                rays_per_view,
                dim=1,
            )[:, :, None]
        )
        direction = GeneralizedSobolevDirection(
            (grid_size,) * 3,
            strength=strength,
            epsilon=epsilon,
        ).to(device)

        component_candidate = baselines[0]
        component_whitening = DetectorCovarianceWhitening(
            fits["component_iid"],
            eigenvectors_by_view=eigenvectors_by_view,
            scale_by_view=scale_by_view,
            predictive_mean_correction=True,
            dtype=torch.float32,
        ).to(device)
        component_operator = WhitenedMeasurementOperator(
            operator,
            component_whitening,
        ).to(device)
        component_prepared = component_operator.prepare_observation(
            observation
        )
        component_operator.reset_call_counts()
        _synchronize(device)
        solve_started = time.perf_counter()
        with torch.no_grad():
            component_result = preconditioned_cgls_reconstruction(
                component_operator,
                component_prepared,
                sigma_by_view=ones_sigma,
                view_mask=ones_mask,
                rays_per_view=rays_per_view,
                stages=int(component_candidate["stages"]),
                preconditioner=direction,
            )
        _synchronize(device)
        elapsed = time.perf_counter() - solve_started
        if component_operator.call_report() != {
            "forward_calls": int(component_candidate["forward_calls"]),
            "adjoint_calls": int(component_candidate["adjoint_calls"]),
        }:
            raise ValueError("component baseline violated its call budget")
        metric_rows.extend(
            _rows_for_metrics(
                replicate=replicate,
                candidate=component_candidate,
                families=families,
                metrics=_field_metrics(component_result.volume, truth),
                elapsed_seconds=elapsed,
                perturbation_norm=0.0,
                perturbation_trials=0.0,
                final_exponent=0.0,
            )
        )

        graph_whitening = DetectorCovarianceWhitening(
            fits["dg_covgate"],
            eigenvectors_by_view=eigenvectors_by_view,
            scale_by_view=scale_by_view,
            predictive_mean_correction=True,
            dtype=torch.float32,
        ).to(device)
        graph_operator = WhitenedMeasurementOperator(
            operator,
            graph_whitening,
        ).to(device)
        graph_prepared = graph_operator.prepare_observation(observation)
        graph_candidates = baselines[1:]
        graph_checkpoints = [
            int(row["stages"]) for row in graph_candidates
        ]
        graph_operator.reset_call_counts()
        _synchronize(device)
        solve_started = time.perf_counter()
        with torch.no_grad():
            graph_trajectory = preconditioned_cgls_trajectory(
                graph_operator,
                graph_prepared,
                sigma_by_view=ones_sigma,
                view_mask=ones_mask,
                rays_per_view=rays_per_view,
                checkpoint_stages=graph_checkpoints,
                preconditioner=direction,
            )
        _synchronize(device)
        graph_elapsed = time.perf_counter() - solve_started
        maximum_graph_stage = max(graph_checkpoints)
        if graph_operator.call_report() != {
            "forward_calls": maximum_graph_stage,
            "adjoint_calls": maximum_graph_stage,
        }:
            raise ValueError("graph trajectory violated its call budget")
        for candidate in graph_candidates:
            result = graph_trajectory[int(candidate["stages"])]
            metric_rows.extend(
                _rows_for_metrics(
                    replicate=replicate,
                    candidate=candidate,
                    families=families,
                    metrics=_field_metrics(result.volume, truth),
                    elapsed_seconds=graph_elapsed,
                    perturbation_norm=0.0,
                    perturbation_trials=0.0,
                    final_exponent=0.0,
                )
            )

        for candidate in superiorized:
            graph_operator.reset_call_counts()
            _synchronize(device)
            solve_started = time.perf_counter()
            with torch.no_grad():
                result = superiorized_pcgls_reconstruction(
                    graph_operator,
                    graph_prepared,
                    sigma_by_view=ones_sigma,
                    view_mask=ones_mask,
                    rays_per_view=rays_per_view,
                    stages=int(candidate["stages"]),
                    preconditioner=direction,
                    penalty=str(candidate["penalty"]),
                    perturbation_steps=int(
                        candidate["perturbation_inner_steps"]
                    ),
                    perturbation_initial_step=float(
                        candidate["perturbation_initial_step"]
                    ),
                    perturbation_decay=float(
                        candidate["perturbation_decay"]
                    ),
                    smoothing=float(config["screen"]["smoothing"]),
                    huber_delta=float(
                        config["screen"]["huber_delta"]
                    ),
                    maximum_backtracks=int(
                        config["screen"]["maximum_backtracks"]
                    ),
                )
            _synchronize(device)
            elapsed = time.perf_counter() - solve_started
            expected_calls = {
                "forward_calls": int(candidate["forward_calls"]),
                "adjoint_calls": int(candidate["adjoint_calls"]),
            }
            if graph_operator.call_report() != expected_calls:
                raise ValueError(
                    f"{candidate['candidate_id']} violated its call budget"
                )
            perturbed_history = result.history[1:]
            perturbation_norm = float(
                torch.mean(
                    torch.stack(
                        [
                            row["perturbation_norm"]
                            for row in perturbed_history
                        ]
                    )
                )
            )
            perturbation_trials = float(
                torch.mean(
                    torch.stack(
                        [
                            row["perturbation_trials"].to(torch.float32)
                            for row in perturbed_history
                        ]
                    )
                )
            )
            final_exponent = float(
                torch.mean(
                    result.history[-1]["perturbation_exponent"].to(
                        torch.float32
                    )
                )
            )
            metric_rows.extend(
                _rows_for_metrics(
                    replicate=replicate,
                    candidate=candidate,
                    families=families,
                    metrics=_field_metrics(result.volume, truth),
                    elapsed_seconds=elapsed,
                    perturbation_norm=perturbation_norm,
                    perturbation_trials=perturbation_trials,
                    final_exponent=final_exponent,
                )
            )
        ledger.append(
            {
                "replicate": int(replicate),
                "graph_activated_view_count": int(
                    sum(bool(row["graph_activated"]) for row in gate_rows)
                ),
                "candidate_count": len(baselines) + len(superiorized),
            }
        )

    comparative_rows = add_comparative_gains(
        metric_rows,
        component_baseline_id=str(baselines[0]["candidate_id"]),
    )
    summaries = summarize_candidates(comparative_rows)
    ranking = rank_superiorized(summaries)
    decision = screen_decision(ranking)
    report = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "evidence_scope": config["evidence_scope"],
        "configuration": config,
        "config_sha256": _sha256(config_path),
        "source_multiseed_config_sha256": _sha256(multiseed_path),
        "source_smoke_config_sha256": _sha256(smoke_path),
        "source_stopping_no_go_sha256": _sha256(stopping_path),
        "replicate_ledger": ledger,
        "candidate_count": len(baselines) + len(superiorized),
        "metric_row_count": len(comparative_rows),
        "top_superiorized_scale_smoke": ranking[:8],
        "decision": decision,
        "runtime": {
            "device": device_name,
            "torch_version": torch.__version__,
            "wall_seconds": float(time.perf_counter() - started),
            "maximum_rss_bytes": _max_rss_bytes(),
        },
        "claim_boundary": config["claim_boundary"],
    }
    return report, comparative_rows, summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = (
        args.config
        if args.config.is_absolute()
        else root / args.config
    )
    report, rows, summaries = run_screen(
        root=root,
        config_path=config_path,
        view_root=args.view_root.resolve(),
        device_name=str(args.device),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    row_path = args.output_dir / "metric_rows.csv"
    summary_path = args.output_dir / "candidate_summaries.csv"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(row_path, rows)
    _write_csv(summary_path, summaries)
    print(
        json.dumps(
            {
                "status": report["status"],
                "candidate_count": report["candidate_count"],
                "metric_row_count": report["metric_row_count"],
                "wall_seconds": report["runtime"]["wall_seconds"],
                "report": str(report_path),
                "rows": str(row_path),
                "summaries": str(summary_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
