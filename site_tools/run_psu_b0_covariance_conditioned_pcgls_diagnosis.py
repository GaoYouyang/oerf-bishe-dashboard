#!/usr/bin/env python3
"""Diagnose covariance, spectral prior, and early-stopping coupling."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import platform
import resource
import time
from typing import Any

import numpy as np
from scipy.stats import t as student_t
import torch

from demo_t16_operator.detector_covariance_whitening import (
    DetectorCovarianceWhitening,
    WhitenedMeasurementOperator,
    spatially_tempered_covariance_fit,
)
from demo_t16_operator.psu_b0_classical_baselines import (
    GeneralizedSobolevDirection,
    preconditioned_cgls_reconstruction,
    preconditioned_cgls_trajectory,
)
from demo_t16_operator.detector_graph_covariance import (
    detector_graph_spectral_basis,
)
from demo_t16_operator.psu_b0_detector_graph_features import (
    build_detector_knn_graph,
)
from demo_t16_operator.psu_b0_reaction_phantoms import (
    reaction_morphology_batch,
)
from demo_t16_operator.psu_b0_streaming_operator import (
    zero_outer_boundary_support,
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


SCHEMA = "psu-b0-covariance-conditioned-pcgls-diagnosis-report-1.0"
STATUS = "POSTOPEN_COVARIANCE_CONDITIONING_DIAGNOSIS_COMPLETE"


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


def candidate_grid(config: dict[str, Any]) -> list[dict[str, Any]]:
    screen = config["screen"]
    strengths = [float(value) for value in screen["sobolev_strengths"]]
    stages = [int(value) for value in screen["stage_counts"]]
    epsilon = float(screen["sobolev_epsilon"])
    candidates = []
    for exponent in screen["spatial_tempering_exponents"]:
        alpha = float(exponent)
        for strength in strengths:
            for count in stages:
                candidates.append(
                    {
                        "candidate_id": (
                            f"spatial_a{_slug(alpha)}_"
                            f"s{_slug(strength)}_k{count}"
                        ),
                        "covariance_mode": "spatial_tempered",
                        "spatial_exponent": alpha,
                        "sobolev_strength": strength,
                        "sobolev_epsilon": epsilon,
                        "stages": count,
                    }
                )
    if bool(screen["include_full_graph_fit_anchor"]):
        for strength in strengths:
            for count in stages:
                candidates.append(
                    {
                        "candidate_id": (
                            f"full_graph_s{_slug(strength)}_k{count}"
                        ),
                        "covariance_mode": "full_graph",
                        "spatial_exponent": 1.0,
                        "sobolev_strength": strength,
                        "sobolev_epsilon": epsilon,
                        "stages": count,
                    }
                )
    expected = int(screen["candidate_count_expected"])
    if len(candidates) != expected:
        raise ValueError(
            f"candidate grid has {len(candidates)} rows, expected {expected}"
        )
    identifiers = [str(row["candidate_id"]) for row in candidates]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("candidate identifiers must be unique")
    return candidates


def validate_partition(
    config: dict[str, Any],
    *,
    replicate_count: int,
) -> dict[int, str]:
    partition = config["replicate_partition"]
    mapping: dict[int, str] = {}
    for split in ("selection", "opened_diagnostic_check"):
        for raw in partition[split]:
            replicate = int(raw)
            if not 0 <= replicate < int(replicate_count):
                raise ValueError("replicate partition index is out of range")
            if replicate in mapping:
                raise ValueError("replicate partition contains overlap")
            mapping[replicate] = split
    if set(mapping) != set(range(int(replicate_count))):
        raise ValueError("replicate partition must cover every replicate")
    return mapping


def validate_execution_plan(
    config: dict[str, Any],
    *,
    candidates: list[dict[str, Any]],
    replicate_count: int,
) -> dict[str, int | float]:
    baseline_calls = int(config["baseline"]["stages"])
    logical_per_replicate = baseline_calls + sum(
        int(row["stages"]) for row in candidates
    )
    grouped: dict[tuple[str, float, float], list[int]] = {}
    for row in candidates:
        grouped.setdefault(
            (
                str(row["covariance_mode"]),
                float(row["spatial_exponent"]),
                float(row["sobolev_strength"]),
            ),
            [],
        ).append(int(row["stages"]))
    physical_per_replicate = baseline_calls + sum(
        max(values) for values in grouped.values()
    )
    logical_total = logical_per_replicate * int(replicate_count)
    physical_total = physical_per_replicate * int(replicate_count)
    reduction = 100.0 * (1.0 - physical_total / logical_total)
    declared = config["execution_plan"]
    if int(declared["logical_forward_and_adjoint_calls_total"]) != logical_total:
        raise ValueError("declared logical call total does not match grid")
    if int(declared["physical_forward_and_adjoint_calls_total"]) != physical_total:
        raise ValueError("declared physical call total does not match grid")
    if not np.isclose(
        float(declared["physical_call_reduction_percent"]),
        reduction,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("declared physical call reduction does not match grid")
    return {
        "logical_calls_per_replicate": logical_per_replicate,
        "physical_calls_per_replicate": physical_per_replicate,
        "logical_calls_total": logical_total,
        "physical_calls_total": physical_total,
        "physical_call_reduction_percent": reduction,
    }


def _derive_smoke_config(
    smoke: dict[str, Any],
    multiseed: dict[str, Any],
    *,
    replicate: int,
) -> dict[str, Any]:
    derived = copy.deepcopy(smoke)
    seeds = multiseed["replicates"]
    derived["status"] = "DERIVED_OPENED_DIAGNOSTIC_REPLICATE"
    derived["covariance_calibration"]["base_seed"] = (
        int(seeds["calibration_seed_start"])
        + int(replicate) * int(seeds["calibration_seed_stride"])
    )
    derived["data"]["field_seed_start"] = (
        int(seeds["field_seed_start"])
        + int(replicate) * int(seeds["field_seed_stride"])
    )
    derived["data"]["flowon_noise_seed"] = (
        int(seeds["flowon_noise_seed_start"])
        + int(replicate) * int(seeds["flowon_noise_seed_stride"])
    )
    return derived


def _mean_ci95(values: np.ndarray) -> tuple[float, float, float]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or len(vector) < 2:
        raise ValueError("confidence interval needs at least two replicates")
    mean = float(np.mean(vector))
    standard_error = float(
        np.std(vector, ddof=1) / np.sqrt(len(vector))
    )
    critical = float(student_t.ppf(0.975, len(vector) - 1))
    half_width = critical * standard_error
    return mean, mean - half_width, mean + half_width


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


def _gate_checks(
    summary: dict[str, Any],
    *,
    gates: dict[str, Any],
) -> dict[str, bool]:
    return {
        "mean_field_gain": float(summary["mean_field_gain_percent"])
        >= float(gates["mean_field_gain_percent_minimum"]),
        "field_p10": float(summary["field_gain_p10_percent"])
        >= float(gates["field_gain_p10_percent_minimum"]),
        "field_harm": float(
            summary["field_harm_over_one_percent_rate"]
        )
        <= float(
            gates["field_harm_over_one_percent_rate_maximum"]
        ),
        "mean_gradient_gain": float(
            summary["mean_gradient_gain_percent"]
        )
        >= float(gates["mean_gradient_gain_percent_minimum"]),
        "mean_front_gain": float(summary["mean_front_f1_gain"])
        >= float(gates["mean_front_f1_gain_minimum"]),
    }


def candidate_is_eligible(
    row: dict[str, Any],
    *,
    eligibility: dict[str, Any],
) -> bool:
    if int(row["stages"]) > int(eligibility["maximum_stages"]):
        return False
    if str(row["covariance_mode"]) == "full_graph":
        return bool(eligibility["full_graph_fit_is_eligible"])
    return float(row["spatial_exponent"]) >= float(
        eligibility["minimum_nonzero_spatial_tempering_exponent"]
    )


def select_candidate(
    summaries: list[dict[str, Any]],
    *,
    gates: dict[str, Any],
    eligibility: dict[str, Any],
) -> dict[str, Any] | None:
    passing = []
    for row in summaries:
        if not candidate_is_eligible(row, eligibility=eligibility):
            continue
        checks = _gate_checks(row, gates=gates)
        if all(checks.values()):
            passing.append({**row, "gate_checks": checks})
    if not passing:
        return None
    return dict(
        min(
            passing,
            key=lambda row: (
                -float(row["mean_field_gain_percent"]),
                -float(row["field_gain_p10_percent"]),
                float(row["field_harm_over_one_percent_rate"]),
                -float(row["mean_front_f1_gain"]),
                int(row["stages"]),
                str(row["candidate_id"]),
            ),
        )
    )


def _gain_rows(
    rows: list[dict[str, Any]],
    *,
    baseline_id: str,
) -> list[dict[str, Any]]:
    baseline = {
        (int(row["replicate"]), int(row["sample_index"])): row
        for row in rows
        if row["candidate_id"] == baseline_id
    }
    output = []
    for row in rows:
        reference = baseline[
            (int(row["replicate"]), int(row["sample_index"]))
        ]
        field_reference = float(reference["field_relative_l2"])
        gradient_reference = float(reference["gradient_relative_l2"])
        output.append(
            {
                **row,
                "field_gain_percent": float(
                    100.0
                    * (
                        field_reference
                        - float(row["field_relative_l2"])
                    )
                    / max(field_reference, 1e-12)
                ),
                "gradient_gain_percent": float(
                    100.0
                    * (
                        gradient_reference
                        - float(row["gradient_relative_l2"])
                    )
                    / max(gradient_reference, 1e-12)
                ),
                "front_f1_gain": float(
                    float(row["front_top10_f1"])
                    - float(reference["front_top10_f1"])
                ),
            }
        )
    return output


def aggregate_diagnosis(
    rows: list[dict[str, Any]],
    *,
    baseline_id: str,
    partition: dict[int, str],
    gates: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    gains = _gain_rows(rows, baseline_id=baseline_id)
    for row in gains:
        row["split"] = partition[int(row["replicate"])]

    replicate_groups: dict[
        tuple[str, str, int],
        list[dict[str, Any]],
    ] = {}
    for row in gains:
        replicate_groups.setdefault(
            (
                str(row["split"]),
                str(row["candidate_id"]),
                int(row["replicate"]),
            ),
            [],
        ).append(row)
    replicate_rows = []
    for (split, candidate_id, replicate), selected in sorted(
        replicate_groups.items()
    ):
        field = np.asarray(
            [float(row["field_gain_percent"]) for row in selected]
        )
        replicate_rows.append(
            {
                "split": split,
                "candidate_id": candidate_id,
                "replicate": replicate,
                "covariance_mode": selected[0]["covariance_mode"],
                "spatial_exponent": selected[0]["spatial_exponent"],
                "sobolev_strength": selected[0]["sobolev_strength"],
                "sobolev_epsilon": selected[0]["sobolev_epsilon"],
                "stages": selected[0]["stages"],
                "mean_field_gain_percent": float(np.mean(field)),
                "field_harm_over_one_percent_rate": float(
                    np.mean(field < -1.0)
                ),
                "mean_gradient_gain_percent": float(
                    np.mean(
                        [
                            float(row["gradient_gain_percent"])
                            for row in selected
                        ]
                    )
                ),
                "mean_front_f1_gain": float(
                    np.mean(
                        [
                            float(row["front_f1_gain"])
                            for row in selected
                        ]
                    )
                ),
            }
        )

    summary_groups: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = {}
    for row in gains:
        summary_groups.setdefault(
            (str(row["split"]), str(row["candidate_id"])),
            [],
        ).append(row)
    summaries = []
    for (split, candidate_id), selected in sorted(summary_groups.items()):
        replicate_selected = [
            row
            for row in replicate_rows
            if row["split"] == split
            and row["candidate_id"] == candidate_id
        ]
        replicate_means = np.asarray(
            [
                float(row["mean_field_gain_percent"])
                for row in replicate_selected
            ],
            dtype=np.float64,
        )
        mean, lower, upper = _mean_ci95(replicate_means)
        field = np.asarray(
            [float(row["field_gain_percent"]) for row in selected],
            dtype=np.float64,
        )
        summary = {
            "split": split,
            "candidate_id": candidate_id,
            "covariance_mode": selected[0]["covariance_mode"],
            "spatial_exponent": float(selected[0]["spatial_exponent"]),
            "sobolev_strength": float(selected[0]["sobolev_strength"]),
            "sobolev_epsilon": float(selected[0]["sobolev_epsilon"]),
            "stages": int(selected[0]["stages"]),
            "replicate_count": len(replicate_selected),
            "field_count": len(selected),
            "mean_field_gain_percent": mean,
            "mean_field_gain_ci95_lower": lower,
            "mean_field_gain_ci95_upper": upper,
            "field_gain_p10_percent": float(np.quantile(field, 0.1)),
            "field_gain_median_percent": float(np.median(field)),
            "field_harm_over_one_percent_rate": float(
                np.mean(field < -1.0)
            ),
            "mean_gradient_gain_percent": float(
                np.mean(
                    [
                        float(row["gradient_gain_percent"])
                        for row in selected
                    ]
                )
            ),
            "mean_front_f1_gain": float(
                np.mean(
                    [float(row["front_f1_gain"]) for row in selected]
                )
            ),
            "all_call_counts_match_candidate_stages": bool(
                all(
                    int(row["forward_calls"]) == int(row["stages"])
                    and int(row["adjoint_calls"]) == int(row["stages"])
                    for row in selected
                )
            ),
        }
        summary["gate_checks"] = _gate_checks(summary, gates=gates)
        summary["all_gates_pass"] = bool(
            all(summary["gate_checks"].values())
        )
        summaries.append(summary)

    family_groups: dict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ] = {}
    for row in gains:
        family_groups.setdefault(
            (
                str(row["split"]),
                str(row["candidate_id"]),
                str(row["reaction_family"]),
            ),
            [],
        ).append(row)
    family_summaries = []
    for (split, candidate_id, family), selected in sorted(
        family_groups.items()
    ):
        field = np.asarray(
            [float(row["field_gain_percent"]) for row in selected],
            dtype=np.float64,
        )
        family_summaries.append(
            {
                "split": split,
                "candidate_id": candidate_id,
                "reaction_family": family,
                "replicate_count": len(selected),
                "mean_field_gain_percent": float(np.mean(field)),
                "field_gain_p10_percent": float(
                    np.quantile(field, 0.1)
                ),
                "field_harm_over_one_percent_rate": float(
                    np.mean(field < -1.0)
                ),
            }
        )
    return replicate_rows, summaries, family_summaries


def _metric_rows(
    *,
    replicate: int,
    candidate: dict[str, Any],
    families: list[str],
    metrics: dict[str, torch.Tensor],
    elapsed_seconds: float,
    call_report: dict[str, int],
    shared_execution_id: str,
    shared_checkpoint_count: int,
) -> list[dict[str, Any]]:
    rows = []
    for index, family in enumerate(families):
        rows.append(
            {
                "replicate": int(replicate),
                "sample_index": index,
                "reaction_family": family,
                **candidate,
                "field_relative_l2": float(
                    metrics["field_relative_l2"][index]
                ),
                "gradient_relative_l2": float(
                    metrics["gradient_relative_l2"][index]
                ),
                "front_top10_f1": float(
                    metrics["front_top10_f1"][index]
                ),
                "solver_elapsed_seconds": float(elapsed_seconds),
                "solver_elapsed_is_shared_trajectory_total": bool(
                    shared_checkpoint_count > 1
                ),
                "shared_execution_id": shared_execution_id,
                "shared_checkpoint_count": int(shared_checkpoint_count),
                "forward_calls": int(call_report["forward_calls"]),
                "adjoint_calls": int(call_report["adjoint_calls"]),
            }
        )
    return rows


def _evaluate(
    *,
    base_operator: Any,
    whitening: DetectorCovarianceWhitening,
    observation: torch.Tensor,
    truth: torch.Tensor,
    candidate: dict[str, Any],
    view_count: int,
    rays_per_view: int,
    grid_size: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], float, dict[str, int]]:
    wrapped = WhitenedMeasurementOperator(
        base_operator,
        whitening,
    ).to(device)
    prepared = wrapped.prepare_observation(observation)
    direction = GeneralizedSobolevDirection(
        (grid_size,) * 3,
        strength=float(candidate["sobolev_strength"]),
        epsilon=float(candidate["sobolev_epsilon"]),
    ).to(device)
    wrapped.reset_call_counts()
    _synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        result = preconditioned_cgls_reconstruction(
            wrapped,
            prepared,
            sigma_by_view=torch.ones(
                (len(truth), view_count),
                dtype=torch.float32,
                device=device,
            ),
            view_mask=torch.ones(
                (len(truth), view_count),
                dtype=torch.float32,
                device=device,
            ),
            rays_per_view=rays_per_view,
            stages=int(candidate["stages"]),
            preconditioner=direction,
        )
    _synchronize(device)
    elapsed = time.perf_counter() - started
    report = wrapped.call_report()
    expected = int(candidate["stages"])
    if report != {
        "forward_calls": expected,
        "adjoint_calls": expected,
    }:
        raise ValueError("candidate violated its declared call budget")
    metrics = _field_metrics(result.volume, truth)
    return metrics, elapsed, report


def _evaluate_trajectory(
    *,
    base_operator: Any,
    whitening: DetectorCovarianceWhitening,
    observation: torch.Tensor,
    truth: torch.Tensor,
    candidates: list[dict[str, Any]],
    view_count: int,
    rays_per_view: int,
    grid_size: int,
    device: torch.device,
) -> tuple[
    dict[int, dict[str, torch.Tensor]],
    float,
    dict[str, int],
]:
    if not candidates:
        raise ValueError("trajectory candidates must be nonempty")
    strengths = {float(row["sobolev_strength"]) for row in candidates}
    epsilons = {float(row["sobolev_epsilon"]) for row in candidates}
    if len(strengths) != 1 or len(epsilons) != 1:
        raise ValueError(
            "one trajectory must share Sobolev strength and epsilon"
        )
    checkpoints = sorted({int(row["stages"]) for row in candidates})
    wrapped = WhitenedMeasurementOperator(
        base_operator,
        whitening,
    ).to(device)
    prepared = wrapped.prepare_observation(observation)
    direction = GeneralizedSobolevDirection(
        (grid_size,) * 3,
        strength=next(iter(strengths)),
        epsilon=next(iter(epsilons)),
    ).to(device)
    wrapped.reset_call_counts()
    _synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        trajectory = preconditioned_cgls_trajectory(
            wrapped,
            prepared,
            sigma_by_view=torch.ones(
                (len(truth), view_count),
                dtype=torch.float32,
                device=device,
            ),
            view_mask=torch.ones(
                (len(truth), view_count),
                dtype=torch.float32,
                device=device,
            ),
            rays_per_view=rays_per_view,
            checkpoint_stages=checkpoints,
            preconditioner=direction,
        )
    _synchronize(device)
    elapsed = time.perf_counter() - started
    report = wrapped.call_report()
    maximum = max(checkpoints)
    if report != {
        "forward_calls": maximum,
        "adjoint_calls": maximum,
    }:
        raise ValueError("shared trajectory violated its physical call budget")
    return {
        stage: _field_metrics(result.volume, truth)
        for stage, result in trajectory.items()
    }, elapsed, report


def run_diagnosis(
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
    multiseed = _load_json(multiseed_path)
    smoke = _load_json(smoke_path)
    replicate_count = int(multiseed["replicates"]["count"])
    partition = validate_partition(
        config,
        replicate_count=replicate_count,
    )
    candidates = candidate_grid(config)
    call_budget = validate_execution_plan(
        config,
        candidates=candidates,
        replicate_count=replicate_count,
    )
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
    baseline_spec = {
        "candidate_id": "baseline_component_s5_k4",
        "covariance_mode": "component_iid",
        "spatial_exponent": 0.0,
        "sobolev_strength": float(
            config["baseline"]["sobolev_strength"]
        ),
        "sobolev_epsilon": float(
            config["baseline"]["sobolev_epsilon"]
        ),
        "stages": int(config["baseline"]["stages"]),
    }
    families = [str(value) for value in smoke["data"]["reaction_families"]]
    rows: list[dict[str, Any]] = []
    replicate_ledger = []

    grouped_candidates: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for candidate in candidates:
        grouped_candidates.setdefault(
            (
                str(candidate["covariance_mode"]),
                float(candidate["spatial_exponent"]),
            ),
            [],
        ).append(candidate)

    for replicate in range(replicate_count):
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

        baseline_whitening = DetectorCovarianceWhitening(
            fits["component_iid"],
            eigenvectors_by_view=eigenvectors_by_view,
            scale_by_view=scale_by_view,
            predictive_mean_correction=True,
            dtype=torch.float32,
        ).to(device)
        metrics, elapsed, call_report = _evaluate(
            base_operator=operator,
            whitening=baseline_whitening,
            observation=observation,
            truth=truth,
            candidate=baseline_spec,
            view_count=view_count,
            rays_per_view=rays_per_view,
            grid_size=grid_size,
            device=device,
        )
        baseline_execution_id = f"rep{replicate:02d}_baseline"
        rows.extend(
            _metric_rows(
                replicate=replicate,
                candidate=baseline_spec,
                families=families,
                metrics=metrics,
                elapsed_seconds=elapsed,
                call_report=call_report,
                shared_execution_id=baseline_execution_id,
                shared_checkpoint_count=1,
            )
        )
        physical_calls_this_replicate = int(
            call_report["forward_calls"]
        )
        logical_calls_this_replicate = int(
            baseline_spec["stages"]
        )

        for (mode, exponent), selected_candidates in sorted(
            grouped_candidates.items()
        ):
            if mode == "full_graph":
                selected_fits = fits["dg_covgate"]
            elif mode == "spatial_tempered":
                selected_fits = [
                    spatially_tempered_covariance_fit(
                        component,
                        target,
                        spatial_exponent=exponent,
                    )
                    for component, target in zip(
                        fits["component_iid"],
                        fits["dg_covgate"],
                        strict=True,
                    )
                ]
            else:
                raise ValueError(f"unsupported covariance mode: {mode}")
            whitening = DetectorCovarianceWhitening(
                selected_fits,
                eigenvectors_by_view=eigenvectors_by_view,
                scale_by_view=scale_by_view,
                predictive_mean_correction=True,
                dtype=torch.float32,
            ).to(device)
            by_strength: dict[float, list[dict[str, Any]]] = {}
            for candidate in selected_candidates:
                by_strength.setdefault(
                    float(candidate["sobolev_strength"]),
                    [],
                ).append(candidate)
            for strength, trajectory_candidates in sorted(
                by_strength.items()
            ):
                metrics_by_stage, elapsed, physical_report = (
                    _evaluate_trajectory(
                        base_operator=operator,
                        whitening=whitening,
                        observation=observation,
                        truth=truth,
                        candidates=trajectory_candidates,
                        view_count=view_count,
                        rays_per_view=rays_per_view,
                        grid_size=grid_size,
                        device=device,
                    )
                )
                physical_calls_this_replicate += int(
                    physical_report["forward_calls"]
                )
                execution_id = (
                    f"rep{replicate:02d}_{mode}_a{_slug(exponent)}_"
                    f"s{_slug(strength)}"
                )
                for candidate in trajectory_candidates:
                    stage = int(candidate["stages"])
                    logical_report = {
                        "forward_calls": stage,
                        "adjoint_calls": stage,
                    }
                    logical_calls_this_replicate += stage
                    rows.extend(
                        _metric_rows(
                            replicate=replicate,
                            candidate=candidate,
                            families=families,
                            metrics=metrics_by_stage[stage],
                            elapsed_seconds=elapsed,
                            call_report=logical_report,
                            shared_execution_id=execution_id,
                            shared_checkpoint_count=len(
                                trajectory_candidates
                            ),
                        )
                    )
        replicate_ledger.append(
            {
                "replicate": replicate,
                "split": partition[replicate],
                "calibration_seed": derived[
                    "covariance_calibration"
                ]["base_seed"],
                "field_seed_start": derived["data"]["field_seed_start"],
                "flowon_noise_seed": derived["data"]["flowon_noise_seed"],
                "graph_activated_view_count": int(
                    sum(bool(row["graph_activated"]) for row in gate_rows)
                ),
                "candidate_count": len(candidates),
                "logical_forward_and_adjoint_calls": (
                    logical_calls_this_replicate
                ),
                "physical_forward_and_adjoint_calls": (
                    physical_calls_this_replicate
                ),
            }
        )

    replicate_rows, summaries, family_summaries = aggregate_diagnosis(
        rows,
        baseline_id=baseline_spec["candidate_id"],
        partition=partition,
        gates=config["selection_gates"],
    )
    selection_summaries = [
        row for row in summaries if row["split"] == "selection"
    ]
    selected = select_candidate(
        selection_summaries,
        gates=config["selection_gates"],
        eligibility=config["candidate_eligibility"],
    )
    diagnostic = None
    if selected is not None:
        diagnostic = next(
            row
            for row in summaries
            if row["split"] == "opened_diagnostic_check"
            and row["candidate_id"] == selected["candidate_id"]
        )
    selection_pass = selected is not None
    diagnostic_pass = bool(
        diagnostic is not None and diagnostic["all_gates_pass"]
    )
    authorize = bool(selection_pass and diagnostic_pass)
    if not selection_pass:
        verdict = "NO_ELIGIBLE_CANDIDATE_PASSED_SELECTION"
    elif not diagnostic_pass:
        verdict = "SELECTION_SIGNAL_FAILED_OPENED_DIAGNOSTIC_CHECK"
    else:
        verdict = (
            "POSTOPEN_DIAGNOSIS_SUPPORTS_FRESH_PREREGISTRATION_ONLY"
        )

    alpha_zero = [
        row
        for row in selection_summaries
        if row["covariance_mode"] == "spatial_tempered"
        and float(row["spatial_exponent"]) == 0.0
        and int(row["stages"])
        <= int(config["candidate_eligibility"]["maximum_stages"])
    ]
    original_full = next(
        row
        for row in summaries
        if row["split"] == "selection"
        and row["candidate_id"] == "full_graph_s5_k4"
    )
    best_alpha_zero = min(
        alpha_zero,
        key=lambda row: (
            -float(row["mean_field_gain_percent"]),
            -float(row["field_gain_p10_percent"]),
            str(row["candidate_id"]),
        ),
    )
    report = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "evidence_scope": config["evidence_scope"],
        "config_sha256": _sha256(config_path),
        "source_multiseed_config_sha256": _sha256(multiseed_path),
        "source_multiseed_report_sha256": _sha256(
            root / str(config["source_multiseed_report"])
        ),
        "source_smoke_config_sha256": _sha256(smoke_path),
        "configuration": config,
        "replicate_ledger": replicate_ledger,
        "baseline": baseline_spec,
        "candidate_count": len(candidates),
        "execution_optimization": {
            **config["execution_plan"],
            "validated_call_budget": call_budget,
            "observed_logical_forward_and_adjoint_calls_total": int(
                sum(
                    row["logical_forward_and_adjoint_calls"]
                    for row in replicate_ledger
                )
            ),
            "observed_physical_forward_and_adjoint_calls_total": int(
                sum(
                    row["physical_forward_and_adjoint_calls"]
                    for row in replicate_ledger
                )
            ),
        },
        "summaries": summaries,
        "family_summaries": family_summaries,
        "decision": {
            "selection_candidate": selected,
            "opened_diagnostic_check": diagnostic,
            "best_alpha_zero_solver_retuning_control": best_alpha_zero,
            "original_full_graph_anchor": original_full,
            "selection_pass": selection_pass,
            "opened_diagnostic_check_pass": diagnostic_pass,
            "fresh_preregistration_authorized": authorize,
            "verdict": verdict,
        },
        "runtime": {
            "device": device_name,
            "torch_version": torch.__version__,
            "wall_seconds": float(time.perf_counter() - started),
            "maximum_rss_bytes": _max_rss_bytes(),
        },
        "claim_boundary": config["claim_boundary"],
    }
    return report, _gain_rows(
        rows,
        baseline_id=baseline_spec["candidate_id"],
    ), replicate_rows


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
    report, rows, replicate_rows = run_diagnosis(
        root=root,
        config_path=config_path,
        view_root=args.view_root.resolve(),
        device_name=str(args.device),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    metric_path = args.output_dir / "metric_rows.csv"
    replicate_path = args.output_dir / "replicate_summaries.csv"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(metric_path, rows)
    _write_csv(replicate_path, replicate_rows)
    print(
        json.dumps(
            {
                "status": report["status"],
                "verdict": report["decision"]["verdict"],
                "selected_candidate": (
                    None
                    if report["decision"]["selection_candidate"] is None
                    else report["decision"]["selection_candidate"][
                        "candidate_id"
                    ]
                ),
                "fresh_preregistration_authorized": report["decision"][
                    "fresh_preregistration_authorized"
                ],
                "report": str(report_path),
                "metric_rows": str(metric_path),
                "replicate_summaries": str(replicate_path),
                "wall_seconds": report["runtime"]["wall_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
